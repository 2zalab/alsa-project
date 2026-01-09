# Changelog - A-LSA Project

## [Version 1.0.1] - 2026-01-09

### 🐛 Bug Fixes

#### 1. SVD Dimension Mismatch Error (CRITICAL FIX)
**Commit:** `95fc4fbdb`

**Problem:**
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

**Root Cause:**
- Incorrectly applied SVD on transposed matrix `X.T`
- Caused dimension mismatch when projecting documents into latent spaces

**Solution:**
- Apply SVD directly on document-term matrix `X` (not transposed)
- Use `V^T` (components_) for term space projections
- Updated variable names from `U_pos_/U_neg_` to `V_pos_/V_neg_`

**Files Changed:**
- `src/alsa.py`: Fixed `fit()`, `_compute_semantic_distance()`, `get_latent_projections()`, `get_characteristic_terms()`
- `tests/test_alsa.py`: Updated test assertions

**Documentation:** See [BUGFIX.md](BUGFIX.md) for complete technical details

---

#### 2. Missing Tabulate Dependency
**Commit:** `74563fe8d`

**Problem:**
```
ImportError: Missing optional dependency 'tabulate'
```

**Solution:**
- Added `tabulate>=0.8.9` to `requirements.txt` and `setup.py`
- Implemented fallback in `save_results_table()` that creates markdown tables manually when tabulate is unavailable

**Files Changed:**
- `requirements.txt`: Added tabulate
- `setup.py`: Added tabulate to install_requires
- `src/visualization.py`: Added try-except fallback for markdown export

---

### ✨ New Features

#### Installation Tools
**Commit:** `2fbd7c2ea`

- **INSTALL.md**: Comprehensive installation and troubleshooting guide
  - Quick installation instructions
  - Common problems and solutions
  - Manual dataset download instructions
  - Environment setup recommendations

- **check_dependencies.py**: Automated dependency checker
  - Verifies all required packages are installed
  - Checks NLTK data availability
  - Lists optional dependencies
  - Provides clear installation instructions if checks fail

- **test_simple.py**: Quick validation test with synthetic data
  - Tests A-LSA without requiring large datasets
  - Validates all core functionality
  - Useful for verifying installation

---

### 📝 Documentation

#### New Files
- `BUGFIX.md`: Technical documentation of the SVD bug fix
- `INSTALL.md`: Installation and troubleshooting guide
- `CHANGELOG.md`: This file

#### Updated Files
- `README.md`: Already comprehensive
- `requirements.txt`: Added tabulate
- `setup.py`: Added tabulate dependency

---

## [Version 1.0.0] - 2026-01-09

### 🎉 Initial Release

#### Core Implementation
- **AdaptiveLSA class**: Full implementation of A-LSA algorithm
  - Dual latent space construction
  - Differential semantic distance classification
  - Class imbalance handling with threshold θ
  - TF-IDF preprocessing pipeline

#### Baseline Models
- Naive Bayes Multinomial
- Logistic Regression (TF-IDF)
- Linear SVM (TF-IDF)
- Classical LSA + Logistic Regression

#### Evaluation Framework
- Comprehensive metrics (F1-macro, Accuracy, Precision, Recall)
- 5-fold stratified cross-validation
- Confusion matrices
- Computational efficiency measurements

#### Experiments
- SMS Spam Collection dataset
- IMDb Movie Reviews dataset
- 20 Newsgroups dataset (binary)
- Sensitivity analysis (k dimension and class imbalance)

#### Visualizations
- Sensitivity to latent dimension k
- Impact of class imbalance
- t-SNE visualization of latent spaces
- Performance comparison plots
- Characteristic terms analysis

#### Documentation
- Comprehensive README with API reference
- Demonstration Jupyter notebook
- Unit tests
- Project structure documentation

---

## Summary of Commits

### Critical Bug Fixes
1. `95fc4fbdb` - Fix SVD dimension mismatch in A-LSA projection
2. `c04b408dc` - Add test and documentation for SVD bug fix
3. `74563fe8d` - Add tabulate dependency and fallback for markdown export

### Documentation and Tools
4. `2fbd7c2ea` - Add installation guide and dependency checker

### Initial Implementation
5. `2f79bbb08` - Implement Adaptive Latent Semantic Analysis (A-LSA)

---

## Upgrade Instructions

### From Version 1.0.0 to 1.0.1

```bash
# Pull latest changes
git pull origin claude/implement-adaptive-lsa-Ndewy

# Update dependencies
pip install -r requirements.txt --upgrade

# Verify installation
python check_dependencies.py

# Test the fixes
python test_simple.py
```

---

## Known Issues

None currently. All critical bugs have been fixed.

---

## Future Enhancements

Potential improvements for future versions:
- BERT baseline implementation (optional)
- Additional datasets support
- GPU acceleration for large datasets
- Interactive web interface for demonstrations
- Pre-trained models for common tasks

---

## Contributors

- **Isaac Touza** - Initial implementation and research
- Université de Maroua, Cameroun

---

## References

### Datasets
1. SMS Spam Collection: Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011)
2. IMDb Reviews: Maas, A.L., et al. (2011)
3. 20 Newsgroups: Lang, K. (1995)

### Methods
- Classical LSA: Deerwester, S., et al. (1990)
- SVD: Golub, G.H., Van Loan, C.F. (2013)

---

## License

MIT License - See [LICENSE](LICENSE) file for details
