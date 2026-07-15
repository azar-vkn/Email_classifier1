"""
ai_classifier.py — AI-Powered Email Classification using Hugging Face Transformers

Uses Facebook's BART-Large-MNLI model for zero-shot classification to classify
emails into three categories: ham (legitimate), spam, and phishing.

Zero-shot classification means the model works out of the box without any
training data — it uses natural language understanding to match email content
against candidate label descriptions.

Usage:
    from src.ai_classifier import predict_with_ai, is_model_loaded

    result = predict_with_ai("Dear user, your account has been suspended...")
    print(result)
    # {'label': 'phishing', 'confidence': 0.92, 'top_contributing_words': [...], 'model': 'ai'}

The model is loaded lazily on first use (~1.6 GB download on first run).
If the model fails to load, predict_with_ai() returns None so callers
can fall back to the rule-based classifier.
"""

import re
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    clean_text,
    extract_features,
    count_urls,
    has_ip_url,
    count_urgency_keywords,
    excessive_caps_ratio,
    URGENCY_KEYWORDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model name from Hugging Face Hub
MODEL_NAME = "facebook/bart-large-mnli"

# Candidate labels for zero-shot classification
# Using descriptive labels gives better results than single words
CANDIDATE_LABELS = [
    "legitimate personal or business email",
    "unsolicited spam or advertising email",
    "phishing attempt to steal credentials or personal information",
]

# Map model output labels back to project labels
LABEL_MAP = {
    "legitimate personal or business email": "ham",
    "unsolicited spam or advertising email": "spam",
    "phishing attempt to steal credentials or personal information": "phishing",
}

# ---------------------------------------------------------------------------
# Global Model State (lazy loading)
# ---------------------------------------------------------------------------

_classifier = None
_model_loaded = False
_model_load_error = None


def _load_model():
    """Load the zero-shot classification pipeline.

    This is called lazily on first prediction. The model is cached globally
    so subsequent predictions are fast.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _classifier, _model_loaded, _model_load_error

    if _model_loaded:
        return True

    if _model_load_error:
        # Already tried and failed — don't retry every request
        return False

    try:
        print(f"[AI] Loading model: {MODEL_NAME}")
        print("[AI] This may take a moment on first run (downloading ~1.6 GB)...")

        from transformers import pipeline

        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,  # CPU (use 0 for GPU if available)
        )

        _model_loaded = True
        print("[AI] Model loaded successfully!")
        return True

    except Exception as e:
        _model_load_error = str(e)
        print(f"[AI] Failed to load model: {e}")
        traceback.print_exc()
        return False


def is_model_loaded() -> bool:
    """Check if the AI model is currently loaded and ready.

    Returns:
        True if model is loaded, False otherwise.
    """
    return _model_loaded


def get_model_status() -> dict:
    """Get detailed model status information.

    Returns:
        Dict with model status, name, and any error messages.
    """
    return {
        "loaded": _model_loaded,
        "model_name": MODEL_NAME,
        "error": _model_load_error,
    }


# ---------------------------------------------------------------------------
# Key Phrase Extraction
# ---------------------------------------------------------------------------

# Phishing-specific indicator phrases to flag
_PHISHING_INDICATORS = [
    "verify your account", "click here", "suspended", "unusual activity",
    "confirm your identity", "update your payment", "unauthorized",
    "act now", "limited time", "security alert", "reset your password",
    "deactivated", "validate your", "click below", "urgent action",
    "account will be closed", "verify your identity", "login credentials",
    "bank account", "credit card", "social security", "wire transfer",
    "nigerian prince", "lottery winner",
]

_SPAM_INDICATORS = [
    "unsubscribe", "free", "winner", "congratulations", "buy now",
    "discount", "offer", "limited offer", "cash", "earn money",
    "no obligation", "subscribe", "deal", "prize", "weight loss",
    "make money", "work from home", "guaranteed", "risk free",
    "special promotion", "act now", "order now",
]


def _extract_key_phrases(text: str, label: str) -> list[str]:
    """Extract key phrases from the email that contribute to the classification.

    Uses pattern matching based on the predicted label to identify relevant
    words and phrases in the original text.

    Args:
        text:  Original (uncleaned) email text.
        label: Predicted label ('ham', 'spam', or 'phishing').

    Returns:
        List of flagged words/phrases found in the text.
    """
    text_lower = text.lower()
    flagged = []

    if label == "phishing":
        for phrase in _PHISHING_INDICATORS:
            if phrase in text_lower:
                flagged.append(phrase)

        # Also check for URL-based indicators
        if count_urls(text) > 0:
            flagged.append(f"{count_urls(text)} URL(s) found")
        if has_ip_url(text):
            flagged.append("IP-based URL detected")
        if excessive_caps_ratio(text) > 0.3:
            flagged.append("excessive CAPS")

    elif label == "spam":
        for phrase in _SPAM_INDICATORS:
            if phrase in text_lower:
                flagged.append(phrase)

        if text.count("!") > 3:
            flagged.append(f"{text.count('!')} exclamation marks")
        if excessive_caps_ratio(text) > 0.3:
            flagged.append("excessive CAPS")

    else:  # ham
        # For legitimate emails, note any mild signals that were outweighed
        urgency_count = count_urgency_keywords(text)
        if urgency_count > 0:
            flagged.append(f"{urgency_count} urgency keyword(s) (low weight)")

    return flagged[:10]  # Cap at 10 items


# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------


def predict_with_ai(text: str, sender: str = "") -> dict | None:
    """Classify an email using the AI zero-shot classification model.

    The model analyzes the email text and determines the most likely
    category among ham, spam, and phishing.

    Args:
        text:   Raw email body text.
        sender: Sender email address (optional, used for feature extraction).

    Returns:
        Dictionary with keys:
            - 'label':       'ham', 'spam', or 'phishing'
            - 'confidence':  Float 0–1
            - 'top_contributing_words': List of flagged phrases
            - 'model':       'ai' (indicates AI model was used)

        Returns None if the model is not available (triggers fallback).
    """
    # Try to load the model if not already loaded
    if not _load_model():
        return None

    if not text or not text.strip():
        return None

    try:
        # Prepare input: use first 1024 chars for efficiency
        # (BART has a 1024-token limit, but we truncate by chars as a rough proxy)
        input_text = text[:1024].strip()

        # If we have a subject line (first line), emphasize it
        lines = input_text.split("\n", 1)
        if len(lines) > 1 and len(lines[0]) < 200:
            # Format as "Subject: ... \n Body: ..."
            input_text = f"Subject: {lines[0]}\nBody: {lines[1]}"

        # Run zero-shot classification
        result = _classifier(
            input_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
        )

        # Extract the top prediction
        top_label_desc = result["labels"][0]
        top_score = result["scores"][0]

        # Map to project label
        label = LABEL_MAP.get(top_label_desc, "ham")

        # Boost confidence for phishing if sender domain is suspicious
        features = extract_features(text, sender)
        if features["suspicious_sender"] == 1 and label == "phishing":
            top_score = min(top_score + 0.05, 1.0)

        # If urgency keywords are very high and model says spam,
        # check if it might be phishing instead
        if (label == "spam" and features["urgency_keyword_count"] >= 3
                and features["num_urls"] > 0):
            # Check if phishing score is close to spam score
            phishing_idx = None
            for i, l in enumerate(result["labels"]):
                if LABEL_MAP.get(l) == "phishing":
                    phishing_idx = i
                    break
            if phishing_idx is not None and result["scores"][phishing_idx] > 0.25:
                label = "phishing"
                top_score = result["scores"][phishing_idx]

        # Extract key phrases
        key_phrases = _extract_key_phrases(text, label)

        return {
            "label": label,
            "confidence": round(top_score, 4),
            "top_contributing_words": key_phrases,
            "model": "ai",
        }

    except Exception as e:
        print(f"[AI] Prediction error: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test
    test_emails = [
        ("Legitimate", "Hi John, just wanted to confirm our meeting tomorrow at 3pm. "
                       "Let me know if the conference room works for you. Best, Sarah"),
        ("Spam", "CONGRATULATIONS!!! You've been selected as our LUCKY WINNER! "
                 "Click here to claim your FREE iPhone 15 and $10,000 cash prize! "
                 "Limited time offer - ACT NOW! Unsubscribe here."),
        ("Phishing", "Dear Customer, We noticed unusual activity on your account. "
                     "Your account has been temporarily suspended. Please click the link "
                     "below to verify your identity immediately: http://192.168.1.1/login "
                     "Failure to act within 24 hours will result in permanent account closure."),
    ]

    print("=" * 60)
    print("AI Email Classifier — Test")
    print("=" * 60)

    for name, email in test_emails:
        print(f"\n{'—' * 40}")
        print(f"Test: {name}")
        print(f"Input: {email[:80]}...")
        result = predict_with_ai(email)
        if result:
            print(f"🏷️  Label:      {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"🔍 Key words:  {', '.join(result['top_contributing_words'])}")
            print(f"🤖 Model:      {result['model']}")
        else:
            print("[ERROR] AI model not available.")
