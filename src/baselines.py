"""
Baseline models for comparison with A-LSA

Implements standard text classification baselines:
- Naive Bayes
- Logistic Regression
- SVM
- Classical LSA + Logistic Regression
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict

from .preprocessing import TextPreprocessor


def get_baseline_models(
    n_components: int = 100,
    max_features: int = None,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = None
) -> Dict[str, Pipeline]:
    """
    Get dictionary of baseline models for comparison.

    All models use the same preprocessing pipeline for fair comparison.

    Parameters
    ----------
    n_components : int, optional (default=100)
        Number of components for LSA

    max_features : int, optional (default=None)
        Maximum vocabulary size

    min_df : int, optional (default=2)
        Minimum document frequency

    max_df : float, optional (default=0.95)
        Maximum document frequency

    random_state : int, optional (default=None)
        Random seed for reproducibility

    Returns
    -------
    models : dict
        Dictionary mapping model names to sklearn Pipeline objects
    """

    # Common TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        stop_words='english',
        norm='l2',
        dtype=np.float32
    )

    models = {}

    # 1. Naive Bayes Multinomial
    # Parameters: alpha=1.0 (Laplace smoothing)
    models['Naive Bayes'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            norm='l2',
            dtype=np.float32
        )),
        ('classifier', MultinomialNB(alpha=1.0))
    ])

    # 2. Logistic Regression (TF-IDF)
    # Parameters: C=1.0, max_iter=1000
    models['Logistic Regression'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            norm='l2',
            dtype=np.float32
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            class_weight='balanced'
        ))
    ])

    # 3. Linear SVM (TF-IDF)
    # Parameters: C=1.0, max_iter=10000
    models['Linear SVM'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            norm='l2',
            dtype=np.float32
        )),
        ('classifier', LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=random_state,
            class_weight='balanced',
            dual=False  # Recommended when n_samples > n_features
        ))
    ])

    # 4. Classical LSA + Logistic Regression
    # Parameters: n_components=100 (LSA), C=1.0 (LR)
    models['LSA + Logistic Regression'] = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            norm='l2',
            dtype=np.float32
        )),
        ('lsa', TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
            algorithm='randomized'
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            class_weight='balanced'
        ))
    ])

    return models


def get_model_info() -> Dict[str, str]:
    """
    Get information about each baseline model.

    Returns
    -------
    info : dict
        Dictionary mapping model names to descriptions
    """
    return {
        'Naive Bayes': 'Multinomial Naive Bayes with Laplace smoothing (alpha=1.0)',
        'Logistic Regression': 'L2-regularized logistic regression (C=1.0)',
        'Linear SVM': 'Linear Support Vector Machine (C=1.0)',
        'LSA + Logistic Regression': 'Classical LSA (k=100) + Logistic Regression (C=1.0)',
        'A-LSA': 'Adaptive Latent Semantic Analysis (k=100)'
    }


class BERTClassifier:
    """
    BERT-based text classifier (OPTIONAL).

    This is a placeholder for BERT fine-tuning.
    Requires transformers library and significant computational resources.

    Parameters
    ----------
    model_name : str, optional (default='bert-base-uncased')
        Pretrained BERT model to use

    learning_rate : float, optional (default=2e-5)
        Learning rate for fine-tuning

    batch_size : int, optional (default=16)
        Batch size for training

    epochs : int, optional (default=3)
        Number of training epochs

    max_length : int, optional (default=512)
        Maximum sequence length
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 3,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length

        self.model = None
        self.tokenizer = None

    def fit(self, X, y):
        """
        Fine-tune BERT on training data.

        Note: This is a placeholder. Full implementation requires
        transformers library and GPU resources.
        """
        raise NotImplementedError(
            "BERT fine-tuning is optional and requires transformers library. "
            "Install with: pip install transformers torch"
        )

    def predict(self, X):
        """Predict class labels."""
        raise NotImplementedError("BERT model not implemented")

    def predict_proba(self, X):
        """Predict class probabilities."""
        raise NotImplementedError("BERT model not implemented")


def create_imbalanced_dataset(X, y, ratio: float = 0.5, random_state: int = None):
    """
    Create an imbalanced version of a dataset.

    Parameters
    ----------
    X : array-like
        Features

    y : array-like
        Labels

    ratio : float, optional (default=0.5)
        Ratio of minority to majority class (e.g., 0.1 for 1:10 imbalance)

    random_state : int, optional (default=None)
        Random seed

    Returns
    -------
    X_imb, y_imb : array-like
        Imbalanced dataset
    """
    rng = np.random.RandomState(random_state)

    # Identify minority and majority classes
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]

    # Get indices for each class
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y == majority_class)[0]

    # Calculate target sizes
    n_minority = len(minority_idx)
    n_majority = int(n_minority / ratio)

    # Subsample majority class if needed
    if n_majority < len(majority_idx):
        majority_idx = rng.choice(majority_idx, size=n_majority, replace=False)

    # Combine indices
    selected_idx = np.concatenate([minority_idx, majority_idx])
    rng.shuffle(selected_idx)

    # Return imbalanced dataset
    if isinstance(X, list):
        X_imb = [X[i] for i in selected_idx]
    else:
        X_imb = X[selected_idx]

    y_imb = y[selected_idx]

    return X_imb, y_imb
