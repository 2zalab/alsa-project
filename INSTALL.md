# Installation and Troubleshooting

## Quick Installation

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy scikit-learn nltk pandas tabulate matplotlib seaborn tqdm psutil
```

### Step 2: Download NLTK data

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Step 3: Verify the installation

```bash
python test_simple.py
```

If you see **"✓ ALL TESTS PASSED!"**, the installation was successful.

---

## Common Issues and Solutions

### 1. ImportError: Missing optional dependency 'tabulate'

**Symptom:**

```
ImportError: Missing optional dependency 'tabulate'. Use pip or conda to install tabulate.
```

**Solution:**

```bash
pip install tabulate
```

**Note:** This issue has been fixed in the latest version. The code now includes a fallback that manually generates Markdown tables if `tabulate` is not available.

---

### 2. ValueError: matmul dimension mismatch

**Symptom:**

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

**Solution:** This bug was fixed in commit `95fc4fbdb`. Make sure you are using the latest version:

```bash
git pull origin claude/implement-adaptive-lsa-Ndewy
```

See [BUGFIX.md](BUGFIX.md) for full details.

---

### 3. ModuleNotFoundError: No module named 'nltk'

**Solution:**

```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

---

### 4. Dataset Download Issues

If you experience problems downloading datasets (20 Newsgroups, etc.) due to network restrictions:

#### Option 1: Manual download

1. **SMS Spam Collection**

   * Download: [https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
   * Extract to: `data/sms_spam/SMSSpamCollection`

2. **IMDb Reviews**

   * Download: [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
   * Extract to: `data/imdb/train/` and `data/imdb/test/`

3. **20 Newsgroups**

   * Automatically downloaded by scikit-learn on first use

#### Option 2: Use synthetic test data

```bash
python test_simple.py  # Uses synthetic data
```

---

## Installation Verification

### Full Test Suite

```bash
# Unit tests
pytest tests/ -v

# Simple test with synthetic data
python test_simple.py

# Test on 20 Newsgroups (requires internet connection)
python experiments/run_newsgroups.py
```

### Minimal Installation for Testing

If you only want to test A-LSA without real datasets:

```bash
pip install numpy scipy scikit-learn pandas tabulate
python test_simple.py
```

---

## Dataset Structure

### SMS Spam Collection

```
data/sms_spam/
└── SMSSpamCollection
```

### IMDb Reviews

```
data/imdb/
├── train/
│   ├── pos/  (12,500 .txt files)
│   └── neg/  (12,500 .txt files)
└── test/
    ├── pos/  (12,500 .txt files)
    └── neg/  (12,500 .txt files)
```

### 20 Newsgroups

Automatically downloaded by scikit-learn into `~/scikit_learn_data/`

---

## Recommended Versions

* Python: 3.8 or higher
* NumPy: 1.21.0+
* scikit-learn: 1.0.0+
* pandas: 1.3.0+
* tabulate: 0.8.9+

---

## Support

If you encounter other issues:

1. Check that all dependencies are installed: `pip list`
2. Verify your Python version: `python --version`
3. Review the full error logs
4. Check [BUGFIX.md](BUGFIX.md) for known issues and fixes

---

## Installation with a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv alsa-env

# Activate the environment
# Windows:
alsa-env\Scripts\activate
# Linux/Mac:
source alsa-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Run tests
python test_simple.py
```

---

## Updating

To get the latest bug fixes:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---
