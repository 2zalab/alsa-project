| Model | Test F1 (macro) | Test Accuracy | Test Precision (macro) | Test Recall (macro) | CV F1 (mean) | CV F1 (std) | CV Accuracy (mean) | CV Accuracy (std) | Train time (s) | Inference (ms/doc) | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-LSA | 0.7538 | 0.7555 | 0.7649 | 0.7565 | 0.7048 | 0.0271 | 0.7150 | 0.0215 | 24.6277 | 2.6489 | IMDb |
| Naive Bayes | 0.8655 | 0.8655 | 0.8658 | 0.8657 | 0.8487 | 0.0039 | 0.8488 | 0.0039 | 1.9622 | 0.2498 | IMDb |
| Logistic Regression | 0.8749 | 0.8750 | 0.8754 | 0.8748 | 0.8647 | 0.0033 | 0.8649 | 0.0033 | 2.8586 | 0.2209 | IMDb |
| Linear SVM | 0.8730 | 0.8730 | 0.8731 | 0.8729 | 0.8619 | 0.0039 | 0.8620 | 0.0038 | 2.2442 | 0.2210 | IMDb |
| LSA + Logistic Regression | 0.8488 | 0.8490 | 0.8501 | 0.8487 | 0.8424 | 0.0070 | 0.8426 | 0.0069 | 4.0994 | 0.2503 | IMDb |
