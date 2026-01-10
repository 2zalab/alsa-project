| Model | Test F1 (macro) | Test Accuracy | Test Precision (macro) | Test Recall (macro) | CV F1 (mean) | CV F1 (std) | CV Accuracy (mean) | CV Accuracy (std) | Train time (s) | Inference (ms/doc) | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-LSA | 0.8749 | 0.8759 | 0.8860 | 0.8749 | 0.7800 | 0.1822 | 0.8060 | 0.1495 | 4.4839 | 1.6752 | 20 Newsgroups |
| Naive Bayes | 0.9569 | 0.9570 | 0.9581 | 0.9567 | 0.9651 | 0.0075 | 0.9651 | 0.0075 | 0.3910 | 0.3062 | 20 Newsgroups |
| Logistic Regression | 0.9544 | 0.9544 | 0.9545 | 0.9546 | 0.9638 | 0.0043 | 0.9639 | 0.0043 | 0.5289 | 0.3286 | 20 Newsgroups |
| Linear SVM | 0.9696 | 0.9696 | 0.9702 | 0.9694 | 0.9594 | 0.0062 | 0.9594 | 0.0062 | 0.3974 | 0.3466 | 20 Newsgroups |
| LSA + Logistic Regression | 0.9494 | 0.9494 | 0.9494 | 0.9494 | 0.9588 | 0.0063 | 0.9588 | 0.0063 | 1.0512 | 0.3159 | 20 Newsgroups |
