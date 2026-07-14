"""
gmail_fetcher.py — Fetch Emails via Gmail API using OAuth Access Token

Uses the Gmail REST API directly (no google-api-python-client needed).
Requires an OAuth 2.0 access token with the 'gmail.readonly' scope.

Usage:
    from src.gmail_fetcher import fetch_gmail_emails

    emails = fetch_gmail_emails(access_token, count=25)
    # Returns list of dicts with 'subject', 'sender', 'date', 'body', 'snippet'
"""

import re
import base64
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_gmail_emails(access_token: str, count: int = 25) -> list[dict]:
    """Fetch the N most recent emails from the user's Gmail inbox.

    Args:
        access_token: OAuth 2.0 access token with gmail.readonly scope.
        count:        Number of recent emails to fetch (max 100).

    Returns:
        List of parsed email dictionaries, each with keys:
            - 'subject'  (str)
            - 'sender'   (str)
            - 'date'     (str, ISO format)
            - 'body'     (str, plain text)
            - 'snippet'  (str, first 200 chars of body)

    Raises:
        PermissionError: If the access token is invalid or lacks scope.
        ConnectionError: If the Gmail API is unreachable.
    """
    headers = _auth_headers(access_token)

    # Step 1: List message IDs
    params = {
        "maxResults": min(count, 100),
        "labelIds": "INBOX",
    }

    try:
        resp = requests.get(
            f"{GMAIL_API_BASE}/messages",
            headers=headers,
            params=params,
            timeout=30,
        )
    except requests.RequestException as e:
        raise ConnectionError(f"Cannot reach Gmail API: {e}") from e

    if resp.status_code == 401:
        raise PermissionError(
            "Access token is invalid or expired. Please sign in again."
        )
    if resp.status_code == 403:
        raise PermissionError(
            "Access token lacks Gmail read permissions. "
            "Please grant the gmail.readonly scope."
        )
    if resp.status_code != 200:
        raise ConnectionError(
            f"Gmail API error ({resp.status_code}): {resp.text[:200]}"
        )

    data = resp.json()
    message_ids = [m["id"] for m in data.get("messages", [])]

    if not message_ids:
        return []

    # Step 2: Fetch each message in full
    emails = []
    for msg_id in message_ids:
        try:
            msg = _fetch_message(msg_id, headers)
            if msg:
                emails.append(msg)
        except Exception:
            continue  # Skip messages that fail to parse

    return emails


def get_user_profile(access_token: str) -> dict:
    """Get the authenticated user's Gmail profile (email address).

    Args:
        access_token: OAuth 2.0 access token.

    Returns:
        Dict with 'email' key.
    """
    headers = _auth_headers(access_token)
    resp = requests.get(
        f"{GMAIL_API_BASE}/profile",
        headers=headers,
        timeout=10,
    )

    if resp.status_code != 200:
        return {"email": "unknown"}

    data = resp.json()
    return {"email": data.get("emailAddress", "unknown")}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _auth_headers(access_token: str) -> dict:
    """Build authorization headers for Gmail API requests."""
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


def _fetch_message(msg_id: str, headers: dict) -> dict | None:
    """Fetch and parse a single Gmail message by ID.

    Args:
        msg_id:  Gmail message ID.
        headers: Auth headers dict.

    Returns:
        Parsed email dict or None if parsing fails.
    """
    resp = requests.get(
        f"{GMAIL_API_BASE}/messages/{msg_id}",
        headers=headers,
        params={"format": "full"},
        timeout=15,
    )

    if resp.status_code != 200:
        return None

    msg = resp.json()
    return _parse_gmail_message(msg)


def _parse_gmail_message(msg: dict) -> dict:
    """Parse a Gmail API message resource into a structured dict.

    Args:
        msg: Gmail API message resource (full format).

    Returns:
        Dict with 'subject', 'sender', 'date', 'body', 'snippet'.
    """
    payload = msg.get("payload", {})
    headers = {
        h["name"].lower(): h["value"]
        for h in payload.get("headers", [])
    }

    subject = headers.get("subject", "(No Subject)")
    sender = headers.get("from", "Unknown")
    date_str = headers.get("date", "")

    # Parse date
    date_iso = date_str
    try:
        # Gmail dates can have various formats; try common ones
        from email.utils import parsedate_to_datetime
        date_obj = parsedate_to_datetime(date_str)
        date_iso = date_obj.isoformat()
    except Exception:
        pass

    # Extract body
    body = _extract_body_from_payload(payload)

    # Gmail also provides a snippet (already plain text, HTML entities decoded)
    gmail_snippet = msg.get("snippet", "")

    snippet = body[:200].strip() if body else gmail_snippet

    return {
        "subject": subject,
        "sender": sender,
        "date": date_iso,
        "body": body if body else gmail_snippet,
        "snippet": snippet,
    }


def _extract_body_from_payload(payload: dict) -> str:
    """Recursively extract the plain text body from a Gmail message payload.

    Prefers text/plain parts, falls back to text/html (stripped of tags).

    Args:
        payload: Gmail API payload object (can be nested with parts).

    Returns:
        Plain text body string.
    """
    mime_type = payload.get("mimeType", "")
    parts = payload.get("parts", [])

    # Single-part message
    if not parts:
        body_data = payload.get("body", {}).get("data", "")
        if body_data:
            decoded = _decode_base64url(body_data)
            if "text/plain" in mime_type:
                return decoded
            elif "text/html" in mime_type:
                return _html_to_text(decoded)
        return ""

    # Multipart message — collect text parts
    text_parts = []
    html_parts = []

    for part in parts:
        part_mime = part.get("mimeType", "")

        if part_mime.startswith("multipart/"):
            # Recurse into nested multipart
            nested = _extract_body_from_payload(part)
            if nested:
                text_parts.append(nested)
        elif part_mime == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                text_parts.append(_decode_base64url(data))
        elif part_mime == "text/html":
            data = part.get("body", {}).get("data", "")
            if data:
                html_parts.append(_decode_base64url(data))

    if text_parts:
        return "\n".join(text_parts)
    elif html_parts:
        return _html_to_text("\n".join(html_parts))

    return ""


def _decode_base64url(data: str) -> str:
    """Decode a base64url-encoded string (used by Gmail API).

    Args:
        data: Base64url-encoded string.

    Returns:
        Decoded UTF-8 string.
    """
    try:
        # Gmail uses URL-safe base64 without padding
        padded = data + "=" * (4 - len(data) % 4)
        decoded_bytes = base64.urlsafe_b64decode(padded)
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using BeautifulSoup.

    Args:
        html: HTML string.

    Returns:
        Plain text with tags removed.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
    except Exception:
        return re.sub(r"<[^>]+>", "", html)
