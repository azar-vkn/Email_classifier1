"""
app.py — Flask API for Email Spam & Phishing Classifier

Provides REST endpoints to classify emails in real-time and scan inboxes.

Endpoints:
    POST /predict
        Body:    {"email_text": "..."}
        Returns: {"label": "...", "confidence": 0.xx, "flagged_words": [...]}

    POST /scan-inbox
        Body:    {"email": "...", "password": "...", "provider": "gmail", "count": 25}
        Returns: {"total": N, "results": [...], "summary": {...}}

    POST /scan-inbox/test-connection
        Body:    {"email": "...", "password": "...", "provider": "gmail"}
        Returns: {"success": true/false, "message": "..."}

    GET /health
        Returns: {"status": "ok", "model_loaded": true/false}

Usage:
    python app/backend/app.py

The server runs on http://localhost:5000 by default.
"""

import os
import sys
import imaplib
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_email
from src.ai_classifier import is_model_loaded as ai_model_loaded, get_model_status
from src.email_fetcher import connect_imap, fetch_recent_emails, disconnect, IMAP_PROVIDERS
from src.gmail_fetcher import fetch_gmail_emails

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)  # Enable CORS for frontend requests


# ---------------------------------------------------------------------------
# Routes — Frontend Serving
# ---------------------------------------------------------------------------


@app.route("/")
def serve_frontend():
    """Serve the frontend index.html at the root URL."""
    return app.send_static_file("index.html")


# ---------------------------------------------------------------------------
# Routes — Core Classification
# ---------------------------------------------------------------------------



@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API and model are running.

    Returns:
        JSON with server status, AI model status, and sklearn model status.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
    sklearn_loaded = os.path.exists(model_path)
    ai_status = get_model_status()
    return jsonify({
        "status": "ok",
        "model_loaded": sklearn_loaded or ai_status["loaded"],
        "ai_model": ai_status,
        "sklearn_model_loaded": sklearn_loaded,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Classify an email as ham, spam, or phishing.

    Expects JSON body:
        {"email_text": "The full email text to classify..."}

    Returns JSON:
        {
            "label": "ham" | "spam" | "phishing",
            "confidence": 0.0 - 1.0,
            "flagged_words": ["word1", "word2", ...]
        }

    Error responses:
        400 — Missing or empty email_text field
        500 — Model not loaded or prediction error
    """
    data = request.get_json(force=True)

    if not data or "email_text" not in data or not data["email_text"].strip():
        return jsonify({"error": "Missing or empty 'email_text' field"}), 400

    try:
        result = predict_email(data["email_text"])
        if result is None:
            # AI model not available, fall back to rule-based
            result = _rule_based_classify(data["email_text"])
            result["model"] = "rule_based"

        return jsonify({
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "flagged_words": result.get("top_contributing_words", []),
            "model_used": result.get("model", "unknown"),
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Routes — Inbox Scanning
# ---------------------------------------------------------------------------


@app.route("/scan-inbox/test-connection", methods=["POST"])
def test_inbox_connection():
    """Test IMAP connection without fetching emails.

    Expects JSON body:
        {
            "email": "user@gmail.com",
            "password": "app-password",
            "provider": "gmail" | "outlook" | "yahoo" | "custom",
            "custom_host": "imap.example.com"  (optional, for custom provider)
        }

    Returns JSON:
        {"success": true/false, "message": "..."}
    """
    data = request.get_json(force=True)

    # Validate required fields
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    email_addr = data.get("email", "").strip()
    password = data.get("password", "").strip()
    provider = data.get("provider", "").strip().lower()
    custom_host = data.get("custom_host", "").strip()

    if not email_addr:
        return jsonify({"success": False, "message": "Email address is required"}), 400
    if not password:
        return jsonify({"success": False, "message": "Password is required"}), 400
    if not provider:
        return jsonify({"success": False, "message": "Provider is required"}), 400

    conn = None
    try:
        conn = connect_imap(email_addr, password, provider, custom_host=custom_host)
        provider_name = IMAP_PROVIDERS.get(provider, {}).get("name", provider.title())
        return jsonify({
            "success": True,
            "message": f"Successfully connected to {provider_name}!"
        })
    except imaplib.IMAP4.error as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 401
    except ConnectionError as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 502
    except ValueError as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }), 500
    finally:
        if conn:
            disconnect(conn)


@app.route("/scan-inbox", methods=["POST"])
def scan_inbox():
    """Connect to the user's inbox, fetch recent emails, and classify each one.

    Expects JSON body:
        {
            "email": "user@gmail.com",
            "password": "app-password",
            "provider": "gmail" | "outlook" | "yahoo" | "custom",
            "count": 25,
            "custom_host": "imap.example.com"  (optional)
        }

    Returns JSON:
        {
            "total": 25,
            "results": [
                {
                    "id": 0,
                    "subject": "...",
                    "sender": "...",
                    "date": "2026-07-13T10:00:00",
                    "label": "phishing",
                    "confidence": 0.94,
                    "flagged_words": ["verify", "suspended"],
                    "snippet": "First 200 chars...",
                    "body": "Full email body..."
                }
            ],
            "summary": {"ham": 18, "spam": 5, "phishing": 2}
        }
    """
    data = request.get_json(force=True)

    # Validate required fields
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email_addr = data.get("email", "").strip()
    password = data.get("password", "").strip()
    provider = data.get("provider", "").strip().lower()
    count = data.get("count", 25)
    custom_host = data.get("custom_host", "").strip()

    if not email_addr:
        return jsonify({"error": "Email address is required"}), 400
    if not password:
        return jsonify({"error": "Password is required"}), 400
    if not provider:
        return jsonify({"error": "Provider is required"}), 400

    # Clamp count to safe range
    try:
        count = int(count)
        count = max(1, min(count, 100))
    except (ValueError, TypeError):
        count = 25

    conn = None
    try:
        # Step 1: Connect to inbox
        conn = connect_imap(email_addr, password, provider, custom_host=custom_host)

        # Step 2: Fetch emails
        raw_emails = fetch_recent_emails(conn, count=count)

        # Step 3: Classify each email
        results = []
        summary = {"ham": 0, "spam": 0, "phishing": 0}

        for idx, email_data in enumerate(raw_emails):
            email_body = email_data.get("body", "")
            email_subject = email_data.get("subject", "")
            # Combine subject + body for better classification
            combined_text = f"{email_subject}\n\n{email_body}"

            # Attempt ML prediction; fall back to rule-based if model not trained
            try:
                prediction = predict_email(combined_text, sender=email_data.get("sender", ""))
                if prediction is None:
                    prediction = _rule_based_classify(combined_text)
            except Exception:
                prediction = _rule_based_classify(combined_text)

            label = prediction["label"]
            summary[label] = summary.get(label, 0) + 1

            results.append({
                "id": idx,
                "subject": email_subject,
                "sender": email_data.get("sender", "Unknown"),
                "date": email_data.get("date", ""),
                "label": label,
                "confidence": round(prediction.get("confidence", 0), 4),
                "flagged_words": prediction.get("top_contributing_words",
                                                prediction.get("flagged_words", [])),
                "snippet": email_data.get("snippet", ""),
                "body": email_body,
                "model_used": prediction.get("model", "rule_based"),
            })

        return jsonify({
            "total": len(results),
            "results": results,
            "summary": summary,
        })

    except imaplib.IMAP4.error as e:
        return jsonify({"error": str(e)}), 401
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Scan failed: {str(e)}"}), 500
    finally:
        if conn:
            disconnect(conn)


# ---------------------------------------------------------------------------
# Routes -- Gmail API Scanning (Google Sign-In)
# ---------------------------------------------------------------------------


@app.route("/scan-inbox/google", methods=["POST"])
def scan_inbox_google():
    """Scan inbox using Gmail API with an OAuth access token.

    Expects JSON body:
        {
            "access_token": "ya29.xxx...",
            "count": 25
        }

    Returns the same format as /scan-inbox.
    """
    data = request.get_json(force=True)

    if not data:
        return jsonify({"error": "No data provided"}), 400

    access_token = data.get("access_token", "").strip()
    count = data.get("count", 25)

    if not access_token:
        return jsonify({"error": "Access token is required"}), 400

    try:
        count = int(count)
        count = max(1, min(count, 100))
    except (ValueError, TypeError):
        count = 25

    try:
        # Fetch emails via Gmail API
        raw_emails = fetch_gmail_emails(access_token, count=count)

        # Classify each email
        results = []
        summary = {"ham": 0, "spam": 0, "phishing": 0}

        for idx, email_data in enumerate(raw_emails):
            email_body = email_data.get("body", "")
            email_subject = email_data.get("subject", "")
            combined_text = f"{email_subject}\n\n{email_body}"

            try:
                prediction = predict_email(combined_text, sender=email_data.get("sender", ""))
                if prediction is None:
                    prediction = _rule_based_classify(combined_text)
            except Exception:
                prediction = _rule_based_classify(combined_text)

            label = prediction["label"]
            summary[label] = summary.get(label, 0) + 1

            results.append({
                "id": idx,
                "subject": email_subject,
                "sender": email_data.get("sender", "Unknown"),
                "date": email_data.get("date", ""),
                "label": label,
                "confidence": round(prediction.get("confidence", 0), 4),
                "flagged_words": prediction.get("top_contributing_words",
                                                prediction.get("flagged_words", [])),
                "snippet": email_data.get("snippet", ""),
                "body": email_body,
                "model_used": prediction.get("model", "rule_based"),
            })

        return jsonify({
            "total": len(results),
            "results": results,
            "summary": summary,
        })

    except PermissionError as e:
        return jsonify({"error": str(e)}), 401
    except ConnectionError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gmail scan failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Rule-Based Fallback Classifier
# ---------------------------------------------------------------------------

# Used when the ML model hasn't been trained yet
_PHISHING_KEYWORDS = [
    "verify your account", "click here", "suspended", "unusual activity",
    "confirm your identity", "update your payment", "unauthorized",
    "act now", "limited time", "security alert", "reset your password",
    "deactivated", "validate your", "paypal", "bank of", "click below",
]

_SPAM_KEYWORDS = [
    "unsubscribe", "free", "winner", "congratulations", "buy now",
    "discount", "offer", "limited offer", "cash", "earn money",
    "no obligation", "click here", "subscribe", "deal", "prize",
    "lottery", "million dollars", "credit card", "weight loss",
]


def _rule_based_classify(text: str) -> dict:
    """Simple rule-based classifier used as a fallback when the ML model
    is not yet trained. Checks for keyword patterns common in phishing and spam.

    Args:
        text: Email text (subject + body combined).

    Returns:
        Dict with 'label', 'confidence', and 'top_contributing_words'.
    """
    text_lower = text.lower()

    phishing_hits = [kw for kw in _PHISHING_KEYWORDS if kw in text_lower]
    spam_hits = [kw for kw in _SPAM_KEYWORDS if kw in text_lower]

    phishing_score = len(phishing_hits)
    spam_score = len(spam_hits)

    if phishing_score >= 2:
        confidence = min(0.5 + phishing_score * 0.1, 0.95)
        return {
            "label": "phishing",
            "confidence": confidence,
            "top_contributing_words": phishing_hits[:10],
        }
    elif spam_score >= 2:
        confidence = min(0.5 + spam_score * 0.08, 0.90)
        return {
            "label": "spam",
            "confidence": confidence,
            "top_contributing_words": spam_hits[:10],
        }
    else:
        all_hits = phishing_hits + spam_hits
        if all_hits:
            return {
                "label": "spam" if spam_score > phishing_score else "ham",
                "confidence": 0.45,
                "top_contributing_words": all_hits[:5],
            }
        return {
            "label": "ham",
            "confidence": 0.75,
            "top_contributing_words": [],
        }


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Email Classifier API")
    print("  Powered by Hugging Face Transformers (BART-Large-MNLI)")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("Endpoints:")
    print("  POST /predict                    - classify an email")
    print("  POST /scan-inbox                 - scan inbox emails")
    print("  POST /scan-inbox/test-connection - test IMAP credentials")
    print("  POST /scan-inbox/google          - scan Gmail via OAuth")
    print("  GET  /health                     - check server status")
    print("")
    print("AI Model: facebook/bart-large-mnli (zero-shot classification)")
    print("Fallback: Rule-based keyword classifier")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True)

