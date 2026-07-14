# 📧 Email Spam & Phishing Classifier

A machine-learning system that classifies emails into three categories:
- ✅ **Ham** — legitimate email
- ⚠️ **Spam** — unsolicited/junk email
- 🚨 **Phishing** — malicious email attempting to steal credentials or data

Built as a college project with a trained ML pipeline, a Flask REST API, and an interactive web frontend.

---

## 🗂️ Project Structure

```
email-spam-phishing-classifier/
├── data/
│   ├── raw/                 # Original downloaded dataset(s)
│   └── processed/           # Cleaned & feature-engineered data
├── notebooks/
│   └── exploration.ipynb    # Exploratory Data Analysis
├── src/
│   ├── preprocessing.py     # Text cleaning + feature engineering
│   ├── train.py             # Model training & hyperparameter tuning
│   ├── evaluate.py          # Metrics, confusion matrix, ROC curves
│   └── predict.py           # Load model & predict on new emails
├── app/
│   ├── backend/
│   │   ├── app.py           # Flask API with /predict endpoint
│   │   └── model/           # Saved model artifacts for the API
│   └── frontend/
│       ├── index.html       # Web UI
│       ├── style.css
│       └── script.js
├── models/                  # Trained model files (.joblib)
├── reports/
│   └── figures/             # Saved plots (confusion matrix, ROC curve)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone / Download the project

```bash
cd "Email Spam and Phishing Classifier"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Activate it:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## 📊 Training the Model

### Step 1 — Prepare your dataset

Place your raw dataset CSV in `data/raw/`. The expected format is a CSV with at least two columns:

| Column  | Description                                      |
|---------|--------------------------------------------------|
| `text`  | The full email body (string)                     |
| `label` | One of: `ham`, `spam`, or `phishing` (string)    |

### Step 2 — Preprocess the data

```bash
python src/preprocessing.py
```

This cleans the text, engineers phishing-specific features, and saves the processed data to `data/processed/`.

### Step 3 — Train models

```bash
python src/train.py
```

Trains Naive Bayes, Logistic Regression, and Random Forest models. The best model and TF-IDF vectorizer are saved to `models/`.

### Step 4 — Evaluate models

```bash
python src/evaluate.py
```

Prints a comparison table of all models and saves confusion matrix / ROC curve plots to `reports/figures/`.

---

## 🌐 Running the Web App

### Start the Flask API

```bash
python app/backend/app.py
```

The API starts on `http://localhost:5000`.

### Open the Frontend

Open `app/frontend/index.html` in your browser (or serve it with a simple HTTP server):

```bash
cd app/frontend
python -m http.server 8080
```

Then visit `http://localhost:8080`.

### API Usage

**POST** `/predict`

```json
{
  "email_text": "Dear user, your account has been suspended. Click here to verify..."
}
```

**Response:**

```json
{
  "label": "phishing",
  "confidence": 0.94,
  "flagged_words": ["suspended", "click here", "verify"]
}
```

---

## 📚 Dataset Suggestions

| Dataset | Source | Format | Notes |
|---------|--------|--------|-------|
| **Enron Spam** | [Kaggle](https://www.kaggle.com/datasets/wanderfj/enron-spam) | CSV with `text`, `label` (spam/ham) | Large corpus; no phishing class — you'd need to add phishing samples separately |
| **SpamAssassin Public Corpus** | [spamassassin.apache.org](https://spamassassin.apache.org/old/publiccorpus/) | Raw `.txt` files in folders | Needs parsing into CSV; ham + spam only |
| **Phishing Email Dataset** | [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) | CSV with email text and labels | Contains phishing-specific samples; can be merged with a spam/ham dataset |
| **Nazario Phishing Corpus** | [monkey.org](https://monkey.org/~jose/phishing/) | Raw email files (mbox) | Pure phishing samples; useful to supplement a spam/ham dataset |

**Recommended approach:** Combine the Enron Spam dataset (ham + spam) with a phishing-specific dataset from Kaggle. Relabel and merge into a single CSV with `text` and `label` columns.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — ML models (Naive Bayes, Logistic Regression, Random Forest)
- **NLTK** — Text preprocessing and tokenization
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Evaluation plots
- **Flask** — REST API backend
- **HTML / CSS / JS** — Frontend UI

---

## 📝 License

This project is for educational purposes (college coursework).
