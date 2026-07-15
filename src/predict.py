"""
predict.py — Load Saved Model & Predict on New Emails

Provides a predict_email() function that:
    1. Tries the AI zero-shot classifier first (Hugging Face Transformers)
    2. Falls back to the traditional ML model (if trained) 
    3. Returns the label, confidence score, and top contributing words/features

The "top contributing words" feature provides explainability by surfacing
which terms or engineered features had the highest influence on the
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
from src.ai_classifier import predict_with_ai, is_model_loaded

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
        Tuple of (model, vectorizer), or (None, None) if files don't exist.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        print(f"[WARN] Failed to load trained model: {e}")
        return None, None


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
    # Step 1: Clean the text
    cleaned = clean_text(text)

    # Step 2: Transform with TF-IDF vectorizer
    tfidf_vector = vectorizer.transform([cleaned])

    # Step 3: Extract engineered features
    features = extract_features(text, sender)
    feature_values = [features[name] for name in ENGINEERED_FEATURE_NAMES]
    feature_matrix = csr_matrix([feature_values])

    # Step 4: Horizontally stack
    combined = hstack([tfidf_vector, feature_matrix])
    return combined


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
    cleaned = clean_text(text)
    tfidf_vector = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()
    n_tfidf_features = len(feature_names)

    # Get feature importances based on model type
    if hasattr(model, "coef_"):
        # Linear model (Logistic Regression)
        coefficients = model.coef_[predicted_class_idx]
        tfidf_coefficients = coefficients[:n_tfidf_features]
    elif hasattr(model, "feature_log_prob_"):
        # Naive Bayes
        log_probs = model.feature_log_prob_[predicted_class_idx]
        tfidf_coefficients = log_probs[:n_tfidf_features]
    elif hasattr(model, "feature_importances_"):
        # Tree-based (Random Forest)
        importances = model.feature_importances_[:n_tfidf_features]
        tfidf_coefficients = importances
    else:
        return []

    # Get non-zero entries in the TF-IDF vector
    nonzero_indices = tfidf_vector.nonzero()[1]

    if len(nonzero_indices) == 0:
        return []

    # Score each non-zero feature by its coefficient × TF-IDF value
    scores = []
    for idx in nonzero_indices:
        if idx < n_tfidf_features:
            score = abs(tfidf_coefficients[idx]) * tfidf_vector[0, idx]
            scores.append((feature_names[idx], score))

    # Sort by score descending and return top_n
    scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in scores[:top_n]]


# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------


def predict_email(text: str, sender: str = "") -> dict:
    """Classify a single email and return prediction with explainability info.

    Prediction priority:
        1. AI model (Hugging Face zero-shot classification) — most accurate
        2. Trained sklearn model (if available in /models) — fallback
        3. Returns None if neither is available (app.py handles rule-based fallback)

    Args:
        text:   Raw email body text.
        sender: Sender email address (optional).

    Returns:
        Dictionary with keys:
            - 'label':       Predicted class ('ham', 'spam', or 'phishing')
            - 'confidence':  Float (0–1) representing prediction confidence
            - 'top_contributing_words': List of most influential words/features
            - 'model':       'ai' or 'sklearn' indicating which model was used

        Returns None if no model is available.
    """
    if not text or not text.strip():
        return None

    # Priority 1: Try AI model (zero-shot classification)
    ai_result = predict_with_ai(text, sender=sender)
    if ai_result is not None:
        return ai_result

    # Priority 2: Try trained sklearn model
    model, vectorizer = load_model()
    if model is not None and vectorizer is not None:
        try:
            X = prepare_features(text, vectorizer, sender=sender)
            predicted_label = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            class_idx = list(model.classes_).index(predicted_label)
            confidence = probabilities[class_idx]

            top_words = get_top_contributing_words(
                text, vectorizer, model, class_idx, top_n=10
            )

            return {
                "label": predicted_label,
                "confidence": float(confidence),
                "top_contributing_words": top_words,
                "model": "sklearn",
            }
        except Exception as e:
            print(f"[WARN] sklearn prediction failed: {e}")

    # No model available
    return None


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
        print(f"🤖 Model:      {result.get('model', 'unknown')}")
    else:
        print("[ERROR] Prediction failed. No model available.")
