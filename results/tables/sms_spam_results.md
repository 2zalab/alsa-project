| Model | Test F1 (macro) | Test Accuracy | Test Precision (macro) | Test Recall (macro) | CV F1 (mean) | CV F1 (std) | CV Accuracy (mean) | CV Accuracy (std) | Train time (s) | Inference (ms/doc) | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-LSA | 0.9500 | 0.9776 | 0.9654 | 0.9360 | 0.9464 | 0.0086 | 0.9753 | 0.0042 | 6.9379 | 1.0498 | SMS Spam |
| Naive Bayes | 0.9437 | 0.9758 | 0.9828 | 0.9122 | 0.9486 | 0.0059 | 0.9778 | 0.0024 | 0.1009 | 0.0203 | SMS Spam |
| Logistic Regression | 0.9502 | 0.9767 | 0.9453 | 0.9553 | 0.9578 | 0.0055 | 0.9805 | 0.0025 | 0.1241 | 0.0264 | SMS Spam |
| Linear SVM | 0.9592 | 0.9812 | 0.9605 | 0.9579 | 0.9617 | 0.0086 | 0.9825 | 0.0040 | 0.1190 | 0.0191 | SMS Spam |
| LSA + Logistic Regression | 0.9133 | 0.9570 | 0.8884 | 0.9439 | 0.9342 | 0.0068 | 0.9686 | 0.0032 | 0.6600 | 0.0232 | SMS Spam |
