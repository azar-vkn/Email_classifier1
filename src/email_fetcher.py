"""
email_fetcher.py — IMAP Email Fetcher for Inbox Scanning

Connects to email providers (Gmail, Outlook, Yahoo, custom IMAP) via IMAP SSL,
fetches recent emails, and parses them into structured dictionaries for
classification by the ML model.

Security:
    - Credentials are NEVER stored on disk
    - Connections use IMAP over SSL (port 993)
    - Gmail users must use App Passwords (not regular passwords)

Usage:
    from src.email_fetcher import connect_imap, fetch_recent_emails, disconnect

    conn = connect_imap("user@gmail.com", "xxxx-xxxx-xxxx-xxxx", "gmail")
    emails = fetch_recent_emails(conn, count=25)
    disconnect(conn)
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
import re
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Provider Presets
# ---------------------------------------------------------------------------

IMAP_PROVIDERS = {
    "gmail": {
        "host": "imap.gmail.com",
        "port": 993,
        "name": "Gmail",
    },
    "outlook": {
        "host": "outlook.office365.com",
        "port": 993,
        "name": "Outlook / Hotmail",
    },
    "yahoo": {
        "host": "imap.mail.yahoo.com",
        "port": 993,
        "name": "Yahoo Mail",
    },
}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def connect_imap(email_address: str, password: str, provider: str,
                 custom_host: str = None, custom_port: int = 993) -> imaplib.IMAP4_SSL:
    """Establish an IMAP SSL connection to the email provider.

    Args:
        email_address: The user's email address.
        password:      App Password (Gmail) or regular password.
        provider:      One of 'gmail', 'outlook', 'yahoo', or 'custom'.
        custom_host:   IMAP server hostname (only used when provider='custom').
        custom_port:   IMAP server port (only used when provider='custom').

    Returns:
        An authenticated imaplib.IMAP4_SSL connection object.

    Raises:
        ValueError:  If provider is unknown and no custom_host is given.
        imaplib.IMAP4.error: If authentication fails.
        ConnectionError: If the server can't be reached.
    """
    if provider == "custom":
        if not custom_host:
            raise ValueError("Custom provider requires a 'custom_host' value.")
        host = custom_host
        port = custom_port
    elif provider in IMAP_PROVIDERS:
        host = IMAP_PROVIDERS[provider]["host"]
        port = IMAP_PROVIDERS[provider]["port"]
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Use one of: "
            f"{list(IMAP_PROVIDERS.keys())} or 'custom'."
        )

    try:
        conn = imaplib.IMAP4_SSL(host, port)
        conn.login(email_address, password)
        return conn
    except imaplib.IMAP4.error as e:
        error_msg = str(e)
        if "AUTHENTICATIONFAILED" in error_msg.upper() or "AUTHENTICATE" in error_msg.upper():
            raise imaplib.IMAP4.error(
                "Authentication failed. Check your email and password. "
                "Gmail users: use an App Password, not your regular password."
            ) from e
        raise
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to {host}:{port}. "
            f"Check your internet connection and provider settings. Error: {e}"
        ) from e


def disconnect(conn: imaplib.IMAP4_SSL):
    """Cleanly close the IMAP connection.

    Args:
        conn: An active IMAP4_SSL connection.
    """
    try:
        conn.close()
    except Exception:
        pass  # mailbox might not be selected
    try:
        conn.logout()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Email Fetching
# ---------------------------------------------------------------------------


def fetch_recent_emails(conn: imaplib.IMAP4_SSL, count: int = 25,
                        folder: str = "INBOX") -> list[dict]:
    """Fetch the N most recent emails from the specified folder.

    Args:
        conn:   Authenticated IMAP4_SSL connection.
        count:  Number of recent emails to fetch (default 25).
        folder: IMAP folder to read from (default "INBOX").

    Returns:
        List of parsed email dictionaries, newest first. Each dict has keys:
            - 'subject'  (str)
            - 'sender'   (str)
            - 'date'     (str, ISO format)
            - 'body'     (str, plain text)
            - 'snippet'  (str, first 200 chars of body)
    """
    conn.select(folder, readonly=True)

    # Search for all emails and take the last `count`
    status, data = conn.search(None, "ALL")
    if status != "OK" or not data[0]:
        return []

    email_ids = data[0].split()
    # Take the most recent `count` emails (end of the list = newest)
    recent_ids = email_ids[-count:] if len(email_ids) > count else email_ids
    # Reverse so newest is first
    recent_ids = list(reversed(recent_ids))

    emails = []
    for eid in recent_ids:
        try:
            status, msg_data = conn.fetch(eid, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            raw_email = msg_data[0][1]
            parsed = parse_email(raw_email)
            if parsed:
                emails.append(parsed)
        except Exception:
            # Skip emails that can't be parsed
            continue

    return emails


# ---------------------------------------------------------------------------
# Email Parsing
# ---------------------------------------------------------------------------


def parse_email(raw_email: bytes) -> dict | None:
    """Parse a raw email (RFC822 bytes) into a structured dictionary.

    Args:
        raw_email: Raw email bytes from IMAP FETCH.

    Returns:
        Dictionary with 'subject', 'sender', 'date', 'body', 'snippet',
        or None if parsing fails entirely.
    """
    try:
        msg = email.message_from_bytes(raw_email)
    except Exception:
        return None

    # --- Subject ---
    subject = _decode_header_value(msg.get("Subject", "(No Subject)"))

    # --- Sender ---
    sender = _decode_header_value(msg.get("From", "Unknown"))

    # --- Date ---
    date_str = msg.get("Date", "")
    try:
        date_obj = parsedate_to_datetime(date_str)
        date_iso = date_obj.isoformat()
    except Exception:
        date_iso = date_str  # fallback to raw string

    # --- Body ---
    body = _extract_body(msg)
    snippet = body[:200].strip() if body else ""

    return {
        "subject": subject,
        "sender": sender,
        "date": date_iso,
        "body": body,
        "snippet": snippet,
    }


def _decode_header_value(value: str) -> str:
    """Decode an email header value that may be MIME-encoded.

    Handles cases like:
        =?UTF-8?B?...?=  (base64-encoded UTF-8)
        =?iso-8859-1?Q?...?=  (quoted-printable)

    Args:
        value: Raw header string.

    Returns:
        Decoded plain-text string.
    """
    if not value:
        return ""

    decoded_parts = []
    try:
        parts = decode_header(value)
        for part, charset in parts:
            if isinstance(part, bytes):
                decoded_parts.append(
                    part.decode(charset or "utf-8", errors="replace")
                )
            else:
                decoded_parts.append(part)
    except Exception:
        return str(value)

    return " ".join(decoded_parts).strip()


def _extract_body(msg: email.message.Message) -> str:
    """Extract the plain-text body from an email message.

    Prefers text/plain parts. Falls back to text/html (stripped of tags).
    For multipart emails, concatenates all text parts.

    Args:
        msg: Parsed email.message.Message object.

    Returns:
        Plain-text email body string.
    """
    text_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                continue

            if content_type == "text/plain":
                text_parts.append(decoded)
            elif content_type == "text/html":
                html_parts.append(decoded)
    else:
        # Single-part message
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
                if content_type == "text/plain":
                    text_parts.append(decoded)
                elif content_type == "text/html":
                    html_parts.append(decoded)
        except Exception:
            pass

    # Prefer plain text; fall back to HTML stripped of tags
    if text_parts:
        body = "\n".join(text_parts)
    elif html_parts:
        body = _html_to_text("\n".join(html_parts))
    else:
        body = ""

    # Clean up excessive whitespace
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]{2,}", " ", body)

    return body.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML content to plain text using BeautifulSoup.

    Args:
        html: HTML string.

    Returns:
        Plain-text string with tags removed.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except Exception:
        # Fallback: strip tags with regex
        return re.sub(r"<[^>]+>", "", html)
