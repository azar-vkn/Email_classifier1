"""
train.py — Model Training Pipeline for Email Spam & Phishing Classifier

Trains three classifiers on TF-IDF + engineered features:
    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Random Forest

After initial training, the best-performing model is further tuned using
GridSearchCV. The final model and fitted TF-IDF vectorizer are saved to
the /models directory using joblib.

Usage:
    python src/train.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import (
    build_tfidf,
    transform_tfidf,
    extract_features,
    preprocess_dataframe,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "processed_emails.csv"
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Engineered feature column names (must match extract_features output)
ENGINEERED_FEATURE_COLS = [
    "num_urls",
    "has_ip_url",
    "urgency_keyword_count",
    "caps_ratio",
    "exclamation_count",
    "suspicious_sender",
]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed email dataset from CSV.

    Args:
        path: Path to the processed CSV file.

    Returns:
        pandas DataFrame with cleaned text, engineered features, and labels.

    Raises:
        FileNotFoundError: If the processed data file doesn't exist.
    """
    # TODO: load CSV and validate required columns exist
    pass


# ---------------------------------------------------------------------------
# Feature Matrix Construction
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame, vectorizer=None, fit: bool = True):
    """Combine TF-IDF vectors with engineered features into a single matrix.

    Args:
        df:         Processed DataFrame with 'cleaned_text' and feature columns.
        vectorizer: An existing fitted TfidfVectorizer (used when fit=False).
        fit:        If True, fit a new vectorizer; if False, use the provided one.

    Returns:
        Tuple of (feature_matrix, fitted_vectorizer, label_array).
        feature_matrix is a sparse matrix: [TF-IDF | engineered_features].
    """
    # TODO: build TF-IDF, horizontally stack with engineered features
    pass


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------


def train_all_models(X_train, y_train) -> dict:
    """Train all three baseline models and return them in a dict.

    Models trained:
        - 'naive_bayes':         MultinomialNB
        - 'logistic_regression': LogisticRegression (max_iter=1000)
        - 'random_forest':       RandomForestClassifier (n_estimators=100)

    Args:
        X_train: Training feature matrix (sparse).
        y_train: Training labels array.

    Returns:
        Dict mapping model name (str) to fitted model object.
    """
    # TODO: instantiate, fit, and return each model
    pass


def evaluate_models(models: dict, X_test, y_test) -> dict:
    """Evaluate each trained model on the test set and return metrics.

    Args:
        models: Dict of {model_name: fitted_model}.
        X_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Dict of {model_name: {'accuracy': ..., 'f1_weighted': ..., 'report': ...}}
    """
    # TODO: compute accuracy, weighted F1, and classification_report for each
    pass


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------


def tune_best_model(model_name: str, model, X_train, y_train) -> object:
    """Run GridSearchCV on the best-performing model to optimize hyperparameters.

    Param grids:
        - naive_bayes:         alpha = [0.01, 0.1, 0.5, 1.0, 2.0]
        - logistic_regression: C = [0.01, 0.1, 1, 10], penalty = ['l1','l2']
        - random_forest:       n_estimators = [100,200,300], max_depth = [None,10,20,30]

    Args:
        model_name: Name of the model to tune.
        model:      The fitted model instance.
        X_train:    Training features.
        y_train:    Training labels.

    Returns:
        Best estimator from GridSearchCV.
    """
    # TODO: define param_grid based on model_name, run GridSearchCV, return best_estimator_
    pass


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------


def save_model(model, vectorizer, model_name: str = "best_model"):
    """Save the trained model and TF-IDF vectorizer to disk using joblib.

    Files saved:
        models/{model_name}.joblib
        models/tfidf_vectorizer.joblib

    Args:
        model:      Fitted model object.
        vectorizer: Fitted TfidfVectorizer.
        model_name: Base filename for the model artifact.
    """
    # TODO: joblib.dump the model and vectorizer
    pass


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    """End-to-end training pipeline.

    Steps:
        1. Load processed data
        2. Build combined feature matrix (TF-IDF + engineered)
        3. Stratified train/test split
        4. Train all three models
        5. Evaluate and print comparison
        6. Pick best model by weighted F1
        7. Tune best model with GridSearchCV
        8. Save tuned model + vectorizer to /models
    """
    print("=" * 60)
    print("Email Classifier — Training Pipeline")
    print("=" * 60)

    # TODO: implement the full pipeline
    print("[INFO] Training not yet implemented — stub only.")


if __name__ == "__main__":
    main()
