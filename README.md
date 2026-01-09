# A-LSA: Adaptive Latent Semantic Analysis

**Binary Text Classification via Dual Latent Spaces**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Isaac Touza
> **Institution:** Université de Maroua, Cameroun
> **Version:** 1.0
> **Date:** January 2026

---

## 📚 Overview

**Adaptive Latent Semantic Analysis (A-LSA)** is a novel approach to binary text classification that constructs **separate latent semantic spaces** for each class, rather than a single unified space as in classical LSA.

### Key Innovation

Instead of building one latent space for all classes (classical LSA), A-LSA:
1. **Partitions the corpus** by class (D+ and D-)
2. **Constructs class-specific latent spaces** using SVD
3. **Classifies documents** based on differential semantic distance to each space

This enables A-LSA to capture **class-specific semantic structures** and achieve competitive performance with significantly lower computational cost than deep learning methods.

---

## 🎯 Core Algorithm

### Training Phase

1. **Preprocess** all documents → TF-IDF representation
2. **Partition** corpus into D+ (positive) and D- (negative)
3. **Build matrices** X+ and X- (TF-IDF for each class)
4. **Apply SVD** to each matrix:
   - X+ ≈ U+ Σ+ V+ᵀ
   - X- ≈ U- Σ- V-ᵀ
5. **Compute threshold** θ as weighted midpoint of class mean distances:
   - θ = (μ+ × N+ + μ- × N-) / (N+ + N-)
   - where μ+ and μ- are mean differential distances for each class

### Classification Phase

For a new document d:

1. **Represent** as TF-IDF vector x_d
2. **Project** into both latent spaces:
   - z+ = Σ+⁻¹ U+ᵀ x_d
   - z- = Σ-⁻¹ U-ᵀ x_d
3. **Compute energies**:
   - E+ = ||z+||²
   - E- = ||z-||²
4. **Calculate differential distance**: Δ_sem = E- - E+
5. **Decide**:
   - If Δ_sem < θ → **positive class**
   - If Δ_sem ≥ θ → **negative class**

---

## 📊 Datasets

The implementation is evaluated on three benchmark datasets:

| Dataset | Size | Classes | Balance | Domain | Challenge |
|---------|------|---------|---------|--------|-----------|
| **SMS Spam** | 5,574 | spam/ham | 13.4% / 86.6% | Short messages | Imbalance |
| **IMDb Reviews** | 50,000 | pos/neg | 50% / 50% | Movie reviews | Long texts |
| **20 Newsgroups** | ~2,000 | comp.graphics / rec.sport.hockey | ~50% / 50% | Forum posts | Domain separation |

### Dataset Links

- **SMS Spam Collection**: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
- **IMDb Reviews**: https://ai.stanford.edu/~amaas/data/sentiment/
- **20 Newsgroups**: http://qwone.com/~jason/20Newsgroups/

---

## 🚀 Installation

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/alsa-project.git
cd alsa-project

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# (Optional) Install in development mode
pip install -e .
```

---

## 💻 Usage

### Quick Start

```python
from src.alsa import AdaptiveLSA

# Initialize model
model = AdaptiveLSA(n_components=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get probabilities
y_proba = model.predict_proba(X_test)
```

### Running Experiments

```bash
# SMS Spam experiment
python experiments/run_sms_spam.py

# IMDb experiment
python experiments/run_imdb.py

# 20 Newsgroups experiment
python experiments/run_newsgroups.py

# Sensitivity analysis
python experiments/sensitivity_analysis.py
```

### Command-Line Interface

If installed via `pip install -e .`:

```bash
# Run experiments
alsa-sms           # SMS Spam dataset
alsa-imdb          # IMDb dataset
alsa-newsgroups    # 20 Newsgroups dataset
alsa-sensitivity   # Sensitivity analysis
```

---

## 📁 Project Structure

```
alsa-project/
│
├── src/
│   ├── __init__.py
│   ├── alsa.py                  # Core A-LSA implementation
│   ├── preprocessing.py         # Text preprocessing pipeline
│   ├── baselines.py             # Baseline models
│   ├── evaluation.py            # Evaluation metrics
│   └── visualization.py         # Plotting functions
│
├── experiments/
│   ├── run_sms_spam.py          # SMS Spam experiment
│   ├── run_imdb.py              # IMDb experiment
│   ├── run_newsgroups.py        # 20 Newsgroups experiment
│   └── sensitivity_analysis.py  # Sensitivity analyses
│
├── data/
│   ├── sms_spam/                # SMS Spam dataset
│   ├── imdb/                    # IMDb dataset
│   └── 20newsgroups/            # 20 Newsgroups dataset
│
├── results/
│   ├── tables/                  # Result tables (CSV, Markdown)
│   └── figures/                 # Visualizations (PNG, PDF)
│
├── notebooks/
│   └── demo_alsa.ipynb          # Demonstration notebook
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📈 Performance Metrics

All models are evaluated using:

- **F1-score (macro)** - PRIMARY METRIC
- **Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **5-fold stratified cross-validation**

### Computational Efficiency

- **Inference time** (ms per document)
- **Memory footprint** (MB)
- **Training time** (seconds)

### Expected Performance

A-LSA aims to achieve:
- **Competitive F1-scores** (within 1-2% of best model)
- **~40× faster** than BERT
- **~9× smaller** memory footprint than BERT

---

## 📊 Visualizations

The implementation generates publication-ready figures:

1. **Sensitivity to k**: F1-score vs latent dimension (k = 10-500)
2. **Impact of Imbalance**: Performance across imbalance ratios (1:1 to 1:10)
3. **t-SNE Visualization**: 2D projection of latent spaces
4. **Performance Comparison**: Bar plots comparing all models
5. **Characteristic Terms**: Top terms for each class

All figures are saved at **300 DPI** in PNG and PDF formats.

---

## 🔬 Baseline Models

For comparison, the following baselines are implemented:

| Model | Description |
|-------|-------------|
| **Naive Bayes** | Multinomial NB with Laplace smoothing |
| **Logistic Regression** | L2-regularized (C=1.0) |
| **Linear SVM** | Linear kernel (C=1.0) |
| **LSA + LR** | Classical LSA (k=100) + Logistic Regression |
| **BERT** (optional) | Fine-tuned BERT-base-uncased |

---

## 📖 API Reference

### AdaptiveLSA

```python
class AdaptiveLSA(n_components=100, max_features=None,
                   min_df=2, max_df=0.95, random_state=None)
```

**Parameters:**
- `n_components`: Latent dimension k (default: 100)
- `max_features`: Maximum vocabulary size
- `min_df`: Minimum document frequency
- `max_df`: Maximum document frequency
- `random_state`: Random seed

**Methods:**
- `fit(X, y)`: Train the model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `decision_function(X)`: Get differential semantic distances
- `get_latent_projections(X)`: Get latent space projections
- `get_characteristic_terms(n_terms=10)`: Extract top terms per class

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

---

## 📄 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{touza2026alsa,
  title={Adaptive Latent Semantic Analysis for Binary Text Classification},
  author={Touza, Isaac},
  year={2026},
  institution={Université de Maroua, Cameroun}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Isaac Touza**
Université de Maroua, Cameroun
Email: isaac.touza@univ-maroua.cm

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for SMS Spam dataset
- Stanford University for IMDb dataset
- Carnegie Mellon University for 20 Newsgroups dataset
- scikit-learn developers for excellent machine learning tools

---

## 📚 References

### Datasets
1. Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). *Contributions to the study of SMS spam filtering*
2. Maas, A.L., et al. (2011). *Learning word vectors for sentiment analysis*. ACL-HLT
3. Lang, K. (1995). *NewsWeeder: Learning to filter netnews*. ICML

### Methods
- Deerwester, S., et al. (1990). *Indexing by latent semantic analysis*. JASIS
- Golub, G.H., Van Loan, C.F. (2013). *Matrix Computations*. Johns Hopkins University Press

---

**Made with ❤️ for the NLP research community**
