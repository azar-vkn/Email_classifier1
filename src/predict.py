"""
predict.py — Load Saved Model & Predict on New Emails

Provides a predict_email() function that:
    1. Cleans the input text
    2. Extracts engineered features
    3. Vectorizes with the saved TF-IDF vectorizer
    4. Runs the saved model to produce a prediction
    5. Returns the label, confidence score, and top contributing words/features

The "top contributing words" feature provides explainability by surfacing
which TF-IDF terms or engineered features had the highest influence on the
model's decision.

Usage:
    from src.predict import predict_email

    result = predict_email("Dear user, your account has been suspended...")
    print(result)
    # {'label': 'phishing', 'confidence': 0.94, 'top_contributing_words': [...]}

Or run from the command line:
    python src/predict.py "Paste your email text here..."
"""

import os
import sys
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_text, extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# Engineered feature names (must match training order)
ENGINEERED_FEATURE_NAMES = [
    "num_urls",
    "has_ip_url",
    "urgency_keyword_count",
    "caps_ratio",
    "exclamation_count",
    "suspicious_sender",
]

# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
    """Load the trained model and TF-IDF vectorizer from disk.

    Args:
        model_path:      Path to the saved model .joblib file.
        vectorizer_path: Path to the saved TF-IDF vectorizer .joblib file.

    Returns:
        Tuple of (model, vectorizer).

    Raises:
        FileNotFoundError: If model or vectorizer files don't exist.
    """
    # TODO: load with joblib and return
    pass


# ---------------------------------------------------------------------------
# Feature Preparation
# ---------------------------------------------------------------------------


def prepare_features(text: str, vectorizer, sender: str = ""):
    """Prepare the combined feature vector for a single email.

    Steps:
        1. Clean the text
        2. Transform cleaned text with the fitted TF-IDF vectorizer
        3. Extract engineered features
        4. Horizontally stack TF-IDF + engineered features

    Args:
        text:       Raw email body text.
        vectorizer: Fitted TfidfVectorizer instance.
        sender:     Sender email address (optional).

    Returns:
        Sparse feature matrix (1 × n_features) ready for model.predict().
    """
    # TODO: clean → vectorize → extract features → hstack
    pass


# ---------------------------------------------------------------------------
# Explainability — Top Contributing Words
# ---------------------------------------------------------------------------


def get_top_contributing_words(
    text: str, vectorizer, model, predicted_class_idx: int, top_n: int = 10
) -> list[str]:
    """Identify the top TF-IDF terms that most influenced the prediction.

    For linear models (Logistic Regression, Naive Bayes): uses the model's
    feature coefficients/log-probabilities for the predicted class.

    For tree-based models (Random Forest): uses feature_importances_ combined
    with the email's non-zero TF-IDF entries.

    Args:
        text:                Raw email text.
        vectorizer:          Fitted TfidfVectorizer.
        model:               Trained classifier.
        predicted_class_idx: Index of the predicted class in model.classes_.
        top_n:               Number of top words to return.

    Returns:
        List of strings: the top_n most influential words/features.
    """
    # TODO: extract feature importance based on model type, return top words
    pass


# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------


def predict_email(text: str, sender: str = "") -> dict:
    """Classify a single email and return prediction with explainability info.

    Args:
        text:   Raw email body text.
        sender: Sender email address (optional).

    Returns:
        Dictionary with keys:
            - 'label':       Predicted class ('ham', 'spam', or 'phishing')
            - 'confidence':  Float (0–1) representing prediction confidence
            - 'top_contributing_words': List of most influential words/features
    """
    # TODO: load model, prepare features, predict, get contributing words
    pass


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<email text>\"")
        print("Example: python src/predict.py \"Your account has been suspended\"")
        sys.exit(1)

    email_text = " ".join(sys.argv[1:])
    print(f"\n📧 Input: {email_text[:100]}{'...' if len(email_text) > 100 else ''}\n")

    result = predict_email(email_text)

    if result:
        print(f"🏷️  Label:      {result['label']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"🔍 Key words:  {', '.join(result['top_contributing_words'])}")
    else:
        print("[ERROR] Prediction failed. Make sure the model is trained first.")
