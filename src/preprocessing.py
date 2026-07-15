"""
preprocessing.py — Text Cleaning & Feature Engineering for Email Classification

This module handles all text preprocessing and feature extraction needed to
convert raw email text into ML-ready inputs. It includes:

  • General NLP cleaning (lowercasing, HTML removal, stopword removal)
  • Phishing-specific engineered features (URL counts, urgency keywords, etc.)
  • TF-IDF vectorization

Usage:
    from src.preprocessing import clean_text, extract_features, build_tfidf

Functions are designed to work both on individual strings (for prediction)
and on pandas DataFrames (for batch training).
"""

import re
import os
import sys
import string
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# NLTK imports — make sure to run:
#   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS = [
    "verify", "urgent", "suspended", "click here", "confirm", "immediately",
    "act now", "limited time", "expire", "unauthorized", "alert", "warning",
    "your account", "update your", "security notice", "unusual activity",
    "reset your password", "locked", "validate", "deactivated",
]

# Ensure NLTK data is available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------


def strip_html(text: str) -> str:
    """Remove HTML tags from email text, returning plain text content.

    Args:
        text: Raw email body potentially containing HTML markup.

    Returns:
        Plain-text version of the input with all HTML tags removed.
    """
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        # Fallback: regex-based HTML stripping
        return re.sub(r"<[^>]+>", " ", text)


def clean_text(text: str) -> str:
    """Apply full cleaning pipeline to a single email string.

    Pipeline steps:
        1. Strip HTML tags
        2. Lowercase
        3. Remove URLs (replace with token 'URL')
        4. Remove email addresses (replace with token 'EMAIL')
        5. Remove punctuation and special characters
        6. Tokenize
        7. Remove stopwords
        8. Rejoin tokens into a single string

    Args:
        text: Raw email body string.

    Returns:
        Cleaned, whitespace-joined string of tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Strip HTML tags
    text = strip_html(text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove URLs (replace with token 'URL')
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)

    # Step 4: Remove email addresses (replace with token 'EMAIL')
    text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)

    # Step 5: Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Step 6: Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Step 7: Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # Step 8: Rejoin tokens
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Phishing-Specific Feature Engineering
# ---------------------------------------------------------------------------


def count_urls(text: str) -> int:
    """Count the number of URLs (http/https/www) present in the raw text.

    Args:
        text: Raw (uncleaned) email text.

    Returns:
        Integer count of URLs found.
    """
    if not text:
        return 0
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    return len(urls)


def has_ip_url(text: str) -> int:
    """Check if any URL in the text uses a raw IP address instead of a domain.

    Example: http://192.168.1.1/login  →  1 (suspicious)

    Args:
        text: Raw email text.

    Returns:
        1 if an IP-based URL is found, 0 otherwise.
    """
    if not text:
        return 0
    # Match URLs containing IP addresses (e.g., http://192.168.1.1/...)
    ip_url_pattern = r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    return 1 if re.search(ip_url_pattern, text) else 0


def count_urgency_keywords(text: str) -> int:
    """Count occurrences of urgency/social-engineering keywords.

    Uses the URGENCY_KEYWORDS list defined at module level.

    Args:
        text: Lowercased email text.

    Returns:
        Total count of urgency keyword matches.
    """
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for keyword in URGENCY_KEYWORDS:
        count += text_lower.count(keyword)
    return count


def excessive_caps_ratio(text: str) -> float:
    """Compute the ratio of uppercase characters to total alphabetic characters.

    A high ratio suggests shouting/urgency often seen in spam/phishing.

    Args:
        text: Raw email text (before lowercasing).

    Returns:
        Float between 0 and 1 representing the uppercase ratio.
    """
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return upper_count / len(alpha_chars)


def count_exclamation_marks(text: str) -> int:
    """Count the number of exclamation marks in the text.

    Args:
        text: Raw email text.

    Returns:
        Integer count of '!' characters.
    """
    if not text:
        return 0
    return text.count("!")


def suspicious_sender_domain(sender: str) -> int:
    """Heuristic check for suspicious sender email domains.

    Flags domains that:
        - Use free email providers with unusual subdomains
        - Contain many hyphens or numbers
        - Are unusually long
        - Mimic well-known domains (e.g., "paypa1.com", "g00gle.com")

    Args:
        sender: The sender's email address string (or empty string if unavailable).

    Returns:
        1 if the domain looks suspicious, 0 otherwise.
    """
    if not sender or "@" not in sender:
        return 0

    try:
        domain = sender.split("@")[1].lower().strip()
    except (IndexError, AttributeError):
        return 0

    # Check for IP-address domains
    if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        return 1

    # Check for unusually long domains (> 30 chars)
    if len(domain) > 30:
        return 1

    # Check for many hyphens (> 2)
    if domain.count("-") > 2:
        return 1

    # Check for many numbers in domain name
    digit_count = sum(1 for c in domain.split(".")[0] if c.isdigit())
    if digit_count >= 3:
        return 1

    # Check for look-alike domains (common typosquatting patterns)
    lookalikes = [
        "paypa1", "g00gle", "micr0soft", "amaz0n", "app1e",
        "faceb00k", "netf1ix", "1inkedin", "twitt3r",
    ]
    domain_base = domain.split(".")[0]
    for lookalike in lookalikes:
        if lookalike in domain_base:
            return 1

    return 0


def extract_features(text: str, sender: str = "") -> dict:
    """Extract all engineered features from a single email.

    Combines all phishing-specific feature functions into a single dict
    that can be converted to a DataFrame row.

    Args:
        text:   Raw email body text (before cleaning).
        sender: Sender email address (optional; empty string if not available).

    Returns:
        Dictionary with keys:
            - 'num_urls'
            - 'has_ip_url'
            - 'urgency_keyword_count'
            - 'caps_ratio'
            - 'exclamation_count'
            - 'suspicious_sender'
    """
    return {
        "num_urls": count_urls(text),
        "has_ip_url": has_ip_url(text),
        "urgency_keyword_count": count_urgency_keywords(text),
        "caps_ratio": round(excessive_caps_ratio(text), 4),
        "exclamation_count": count_exclamation_marks(text),
        "suspicious_sender": suspicious_sender_domain(sender),
    }


# ---------------------------------------------------------------------------
# TF-IDF Vectorization
# ---------------------------------------------------------------------------


def build_tfidf(corpus: list[str], max_features: int = 5000) -> tuple:
    """Fit a TF-IDF vectorizer on the training corpus and return the matrix + vectorizer.

    Args:
        corpus:       List of cleaned email text strings.
        max_features: Maximum number of TF-IDF features to keep.

    Returns:
        Tuple of (tfidf_matrix, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer


def transform_tfidf(texts: list[str], vectorizer: TfidfVectorizer):
    """Transform new texts using an already-fitted TF-IDF vectorizer.

    Args:
        texts:      List of cleaned email text strings.
        vectorizer: A previously fitted TfidfVectorizer instance.

    Returns:
        Sparse TF-IDF matrix for the input texts.
    """
    return vectorizer.transform(texts)


# ---------------------------------------------------------------------------
# Batch Processing (DataFrame-level)
# ---------------------------------------------------------------------------


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text",
                         label_col: str = "label") -> pd.DataFrame:
    """Run the full preprocessing pipeline on a DataFrame of emails.

    Steps:
        1. Clean each email text
        2. Extract engineered features for each row
        3. Return a new DataFrame with cleaned text + feature columns + label

    Args:
        df:        Input DataFrame with raw email text.
        text_col:  Name of the column containing email body text.
        label_col: Name of the column containing the label (ham/spam/phishing).

    Returns:
        Processed DataFrame ready for model training.
    """
    result_df = df.copy()

    # Step 1: Clean text
    print("[INFO] Cleaning text...")
    result_df["cleaned_text"] = result_df[text_col].apply(clean_text)

    # Step 2: Extract engineered features
    print("[INFO] Extracting features...")
    features = result_df[text_col].apply(lambda t: extract_features(t, ""))
    feature_df = pd.DataFrame(features.tolist())

    # Combine cleaned text, features, and label
    result_df = pd.concat([result_df[["cleaned_text", label_col]], feature_df], axis=1)

    print(f"[INFO] Processed {len(result_df)} emails with {len(feature_df.columns)} engineered features.")
    return result_df


# ---------------------------------------------------------------------------
# CLI Entry Point — run preprocessing on the raw dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:  python src/preprocessing.py

    Expects a CSV in data/raw/ with columns 'text' and 'label'.
    Saves processed output to data/processed/processed_emails.csv
    """
    print("=" * 60)
    print("Email Preprocessing Pipeline")
    print("=" * 60)

    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    # Find CSV files in raw directory
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"[ERROR] No CSV files found in {RAW_DIR}")
        print("        Place a CSV with 'text' and 'label' columns there.")
        sys.exit(1)

    # Load and process the first CSV found
    input_path = os.path.join(RAW_DIR, csv_files[0])
    print(f"[INFO] Loading: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_files[0]}")

    processed_df = preprocess_dataframe(df)

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "processed_emails.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed data to: {output_path}")
