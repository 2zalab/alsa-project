# A-LSA Article

This directory contains the LaTeX source for the paper "An Adaptive Latent Semantic Analysis Framework for Binary Text Classification".

## Contents

- `alsa_article.tex`: Main LaTeX source file with complete article
- `figures/`: Directory containing all experimental result figures
  - `sms_spam_comparison.png`: Performance comparison on SMS Spam dataset
  - `imdb_comparison.png`: Performance comparison on IMDb dataset
  - `sensitivity_to_k.png`: Sensitivity analysis for latent dimension k
  - `imbalance_impact.png`: Impact of class imbalance on performance
  - `sms_spam_terms.png`: Top characteristic terms for SMS Spam classes
  - `imdb_terms.png`: Top characteristic terms for IMDb classes

## Compilation

To compile the article to PDF:

```bash
pdflatex alsa_article.tex
bibtex alsa_article
pdflatex alsa_article.tex
pdflatex alsa_article.tex
```

Or use your preferred LaTeX editor (TeXstudio, Overleaf, etc.).

## Key Updates

This version includes:

1. **Real experimental results** from the `results/` directory
   - SMS Spam dataset: F1=0.946, Accuracy=0.975
   - IMDb dataset: F1=0.736, Accuracy=0.741

2. **SHADO citation** (Touza et al., 2025) added to Related Work section

3. **All result figures** integrated from experiments

4. **Updated analysis** reflecting actual performance characteristics:
   - Strong performance on short texts with distinct class semantics
   - Challenges with longer, more nuanced texts
   - Systematic superiority over classical LSA (6.9% average gain)
   - Excellent robustness to class imbalance with threshold adjustment

## Required LaTeX Packages

- amsmath, amsfonts, amssymb
- graphicx
- booktabs
- algorithm, algorithmic
- hyperref
- cite

All packages are standard and should be available in most LaTeX distributions (TeXLive, MikTeX, etc.).
