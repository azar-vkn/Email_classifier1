"""
evaluate.py — Model Evaluation & Visualization

Generates comprehensive evaluation metrics and plots for each trained model:
    • Accuracy, Precision, Recall, F1-Score, ROC-AUC
    • Confusion matrix heatmap (saved as PNG)
    • ROC curve per class using One-vs-Rest (saved as PNG)
    • Side-by-side comparison table of all models

All plots are saved to reports/figures/.

Usage:
    python src/evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["ham", "spam", "phishing"]

# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute a full suite of classification metrics.

    Args:
        y_true:  True labels (array-like).
        y_pred:  Predicted labels (array-like).
        y_proba: Predicted class probabilities (shape: n_samples × n_classes).
                 Required for ROC-AUC. Pass None if the model doesn't support it.

    Returns:
        Dict with keys: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.
        'roc_auc' will be None if y_proba is not provided.
    """
    # TODO: compute all metrics, handle multiclass ROC-AUC with OvR
    pass


# ---------------------------------------------------------------------------
# Confusion Matrix Plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    """Generate and optionally save a confusion matrix heatmap.

    Args:
        y_true:     True labels.
        y_pred:     Predicted labels.
        model_name: Name of the model (used in title and filename).
        save:       If True, save the figure to FIGURES_DIR.
    """
    # TODO: compute confusion_matrix, plot with seaborn heatmap, save PNG
    pass


# ---------------------------------------------------------------------------
# ROC Curve Plot
# ---------------------------------------------------------------------------


def plot_roc_curve(y_true, y_proba, model_name: str, save: bool = True):
    """Generate and optionally save a One-vs-Rest ROC curve for each class.

    Args:
        y_true:     True labels (will be binarized internally).
        y_proba:    Predicted probabilities (n_samples × n_classes).
        model_name: Name of the model (used in title and filename).
        save:       If True, save the figure to FIGURES_DIR.
    """
    # TODO: binarize labels, compute ROC per class, plot, save PNG
    pass


# ---------------------------------------------------------------------------
# Model Comparison Table
# ---------------------------------------------------------------------------


def print_comparison_table(results: dict):
    """Print a formatted table comparing metrics across all models.

    Args:
        results: Dict of {model_name: metrics_dict} where metrics_dict
                 has keys 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.
    """
    # TODO: format as a pandas DataFrame and print
    pass


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------


def main():
    """Run evaluation on all saved models.

    Steps:
        1. Load processed test data
        2. Load each saved model from /models
        3. Generate predictions and probabilities
        4. Compute metrics for each model
        5. Plot confusion matrix and ROC curve for each
        6. Print side-by-side comparison table
    """
    print("=" * 60)
    print("Email Classifier — Evaluation Pipeline")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # TODO: implement the full evaluation pipeline
    print("[INFO] Evaluation not yet implemented — stub only.")


if __name__ == "__main__":
    main()
