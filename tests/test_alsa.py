"""
Basic tests for A-LSA implementation

Run with: pytest tests/
"""

import pytest
import numpy as np
from sklearn.datasets import fetch_20newsgroups

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alsa import AdaptiveLSA
from src.preprocessing import TextPreprocessor
from src.baselines import get_baseline_models
from src.evaluation import evaluate_model


@pytest.fixture
def sample_data():
    """Load small sample dataset for testing."""
    # Load 20 Newsgroups (small subset)
    data = fetch_20newsgroups(
        subset='train',
        categories=('comp.graphics', 'rec.sport.hockey'),
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

    # Use only first 100 samples for speed
    X = data.data[:100]
    y = data.target[:100]

    return X, y


def test_text_preprocessor():
    """Test TextPreprocessor."""
    preprocessor = TextPreprocessor(min_df=1, max_df=1.0)

    texts = [
        "This is a test document.",
        "This is another test.",
        "Testing the preprocessor."
    ]

    # Fit and transform
    X = preprocessor.fit_transform(texts)

    # Check output
    assert X.shape[0] == 3
    assert X.shape[1] > 0
    assert preprocessor.is_fitted

    # Check vocabulary
    vocab_size = preprocessor.get_vocabulary_size()
    assert vocab_size > 0


def test_alsa_initialization():
    """Test A-LSA initialization."""
    model = AdaptiveLSA(n_components=10, random_state=42)

    assert model.n_components == 10
    assert model.random_state == 42
    assert model.svd_pos_ is None  # Not fitted yet


def test_alsa_fit(sample_data):
    """Test A-LSA fit method."""
    X, y = sample_data

    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X, y)

    # Check that model was fitted
    assert model.svd_pos_ is not None
    assert model.svd_neg_ is not None
    assert model.preprocessor_ is not None
    assert model.theta_ is not None

    # Check dimensions
    # V_pos_ should have shape (k, n_features)
    assert model.V_pos_.shape[0] == 10
    assert model.V_neg_.shape[0] == 10


def test_alsa_predict(sample_data):
    """Test A-LSA predict method."""
    X, y = sample_data

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train and predict
    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Check output
    assert len(y_pred) == len(X_test)
    assert set(y_pred).issubset({0, 1})


def test_alsa_predict_proba(sample_data):
    """Test A-LSA predict_proba method."""
    X, y = sample_data

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train and predict probabilities
    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    # Check output
    assert y_proba.shape == (len(X_test), 2)
    assert np.allclose(y_proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(y_proba >= 0) and np.all(y_proba <= 1)  # Valid probabilities


def test_alsa_decision_function(sample_data):
    """Test A-LSA decision_function method."""
    X, y = sample_data

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train and compute distances
    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X_train, y_train)
    distances = model.decision_function(X_test)

    # Check output
    assert len(distances) == len(X_test)
    assert isinstance(distances, np.ndarray)


def test_alsa_characteristic_terms(sample_data):
    """Test get_characteristic_terms method."""
    X, y = sample_data

    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X, y)

    terms = model.get_characteristic_terms(n_terms=5)

    # Check output
    assert 'positive' in terms
    assert 'negative' in terms
    assert len(terms['positive']) == 5
    assert len(terms['negative']) == 5

    # Check format
    for term, weight in terms['positive']:
        assert isinstance(term, str)
        assert isinstance(weight, (int, float, np.number))


def test_baseline_models():
    """Test baseline models initialization."""
    models = get_baseline_models(n_components=10, random_state=42)

    # Check that models are returned
    assert len(models) > 0
    assert 'Naive Bayes' in models
    assert 'Logistic Regression' in models
    assert 'Linear SVM' in models
    assert 'LSA + Logistic Regression' in models


def test_evaluate_model(sample_data):
    """Test evaluation function."""
    X, y = sample_data

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Check metrics
    assert 'F1-score (macro)' in metrics
    assert 'Accuracy' in metrics
    assert 'Precision (macro)' in metrics
    assert 'Recall (macro)' in metrics

    # Check values are valid
    for metric, value in metrics.items():
        assert 0 <= value <= 1


def test_imbalanced_classes():
    """Test A-LSA with imbalanced classes."""
    # Create imbalanced dataset
    X = [
        "hockey game goal player",
        "hockey team win championship",
        "graphics image rendering 3d",
    ]
    y = np.array([1, 1, 0])  # 2 positive, 1 negative

    model = AdaptiveLSA(n_components=2, random_state=42)
    model.fit(X, y)

    # Check that theta compensates for imbalance
    assert model.theta_ != 0  # Should be non-zero for imbalanced data

    # Check predictions work
    y_pred = model.predict(X)
    assert len(y_pred) == len(X)


def test_latent_projections(sample_data):
    """Test get_latent_projections method."""
    X, y = sample_data

    model = AdaptiveLSA(n_components=10, random_state=42)
    model.fit(X, y)

    z_pos, z_neg = model.get_latent_projections(X[:5])

    # Check output shapes
    assert z_pos.shape == (5, 10)
    assert z_neg.shape == (5, 10)
    assert isinstance(z_pos, np.ndarray)
    assert isinstance(z_neg, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
