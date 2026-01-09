"""
Adaptive Latent Semantic Analysis (A-LSA) for Binary Text Classification

Core implementation of the A-LSA algorithm that constructs separate latent
semantic spaces for each class and uses differential semantic distance for classification.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, List, Optional
from scipy import sparse
import warnings

from .preprocessing import TextPreprocessor


class AdaptiveLSA(BaseEstimator, ClassifierMixin):
    """
    Adaptive Latent Semantic Analysis (A-LSA) Classifier.

    A-LSA constructs separate latent semantic spaces for each class (positive and negative)
    and classifies documents based on their differential semantic distance to each space.

    The key innovation: Instead of building a single latent space for all classes (classical LSA),
    A-LSA builds two conditional latent spaces that capture class-specific semantic structures.

    Parameters
    ----------
    n_components : int, optional (default=100)
        Dimension of the latent semantic space (k).
        Recommended values: 50-200 depending on dataset size.

    max_features : int, optional (default=None)
        Maximum vocabulary size for TF-IDF vectorization.

    min_df : int or float, optional (default=2)
        Minimum document frequency for terms.
        Terms appearing in fewer documents are filtered out.

    max_df : float, optional (default=0.95)
        Maximum document frequency for terms.
        Terms appearing in more than this fraction of documents are filtered out.

    random_state : int, optional (default=None)
        Random seed for reproducibility.

    Attributes
    ----------
    svd_pos_ : TruncatedSVD
        SVD model for positive class space

    svd_neg_ : TruncatedSVD
        SVD model for negative class space

    V_pos_ : ndarray of shape (n_components, n_features)
        Right singular vectors (V^T) for positive class term space

    V_neg_ : ndarray of shape (n_components, n_features)
        Right singular vectors (V^T) for negative class term space

    Sigma_pos_ : ndarray of shape (n_components,)
        Singular values for positive class

    Sigma_neg_ : ndarray of shape (n_components,)
        Singular values for negative class

    theta_ : float
        Decision threshold computed as: θ = 0.5 * log(N+ / N-)
        Compensates for class imbalance

    preprocessor_ : TextPreprocessor
        Fitted text preprocessor

    classes_ : ndarray
        Unique class labels

    n_pos_ : int
        Number of positive class documents in training set

    n_neg_ : int
        Number of negative class documents in training set

    References
    ----------
    Based on the research by Isaac Touza, Université de Maroua, Cameroun (2026)
    """

    def __init__(
        self,
        n_components: int = 100,
        max_features: Optional[int] = None,
        min_df: int = 2,
        max_df: float = 0.95,
        random_state: Optional[int] = None
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

        # Will be set during fit
        self.svd_pos_ = None
        self.svd_neg_ = None
        self.V_pos_ = None
        self.V_neg_ = None
        self.Sigma_pos_ = None
        self.Sigma_neg_ = None
        self.theta_ = 0.0
        self.preprocessor_ = None
        self.classes_ = None
        self.n_pos_ = 0
        self.n_neg_ = 0

    def fit(self, X: Union[List[str], np.ndarray], y: np.ndarray) -> 'AdaptiveLSA':
        """
        Fit the A-LSA model.

        Training procedure:
        1. Preprocess all documents and create TF-IDF representation
        2. Partition corpus into D+ (positive class) and D- (negative class)
        3. Construct separate TF-IDF matrices X+ and X-
        4. Apply truncated SVD to each matrix: X ≈ U Σ V^T
        5. Compute decision threshold: θ = 0.5 * log(N+ / N-)

        Parameters
        ----------
        X : list of str or array-like
            Training documents

        y : array-like of shape (n_samples,)
            Binary labels (0 or 1)

        Returns
        -------
        self : AdaptiveLSA
            Fitted classifier
        """
        # Convert to numpy array
        y = np.asarray(y)

        # Get unique classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"A-LSA requires exactly 2 classes, got {len(self.classes_)}")

        # Assume binary labels are 0 and 1
        # Positive class = 1, Negative class = 0
        if not set(self.classes_).issubset({0, 1}):
            warnings.warn("Class labels are not {0, 1}. Mapping to binary labels.")
            # Map to 0 and 1
            label_map = {self.classes_[0]: 0, self.classes_[1]: 1}
            y = np.array([label_map[label] for label in y])
            self.classes_ = np.array([0, 1])

        # Step 1: Preprocess and vectorize all documents
        print(f"Preprocessing {len(X)} documents...")
        self.preprocessor_ = TextPreprocessor(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df
        )

        # Fit on all documents to get consistent vocabulary
        X_tfidf = self.preprocessor_.fit_transform(X)
        print(f"Vocabulary size: {self.preprocessor_.get_vocabulary_size()}")

        # Step 2: Partition corpus by class
        mask_pos = (y == 1)
        mask_neg = (y == 0)

        X_pos = X_tfidf[mask_pos]  # Documents of positive class
        X_neg = X_tfidf[mask_neg]  # Documents of negative class

        self.n_pos_ = X_pos.shape[0]
        self.n_neg_ = X_neg.shape[0]

        print(f"Positive class: {self.n_pos_} documents")
        print(f"Negative class: {self.n_neg_} documents")

        # Step 3: Compute decision threshold
        # θ = 0.5 * log(N+ / N-)
        # This compensates for class imbalance
        if self.n_pos_ > 0 and self.n_neg_ > 0:
            self.theta_ = 0.5 * np.log(self.n_pos_ / self.n_neg_)
        else:
            self.theta_ = 0.0

        print(f"Decision threshold θ = {self.theta_:.4f}")

        # Step 4: Apply truncated SVD to each class matrix
        # Ensure k is not larger than possible dimensions
        k = min(self.n_components, min(X_pos.shape) - 1, min(X_neg.shape) - 1)

        if k < self.n_components:
            warnings.warn(
                f"Reducing n_components from {self.n_components} to {k} "
                f"due to small dataset size"
            )

        print(f"Applying SVD with k={k} components...")

        # SVD for positive class: X+ ≈ U+ Σ+ V+^T
        # X_pos is (n_documents_pos, n_features)
        # After SVD, components_ gives V^T of shape (k, n_features)
        self.svd_pos_ = TruncatedSVD(
            n_components=k,
            random_state=self.random_state,
            algorithm='randomized'
        )
        self.svd_pos_.fit(X_pos)  # Fit on document-term matrix

        # SVD for negative class: X- ≈ U- Σ- V-^T
        self.svd_neg_ = TruncatedSVD(
            n_components=k,
            random_state=self.random_state,
            algorithm='randomized'
        )
        self.svd_neg_.fit(X_neg)  # Fit on document-term matrix

        # Store components (V^T) and singular values (Σ)
        # components_ has shape (k, n_features) - this is V^T for term space projection
        self.V_pos_ = self.svd_pos_.components_  # shape: (k, n_features)
        self.V_neg_ = self.svd_neg_.components_  # shape: (k, n_features)

        self.Sigma_pos_ = self.svd_pos_.singular_values_  # shape: (k,)
        self.Sigma_neg_ = self.svd_neg_.singular_values_  # shape: (k,)

        print(f"Training complete!")
        print(f"Explained variance ratio (positive): {self.svd_pos_.explained_variance_ratio_.sum():.4f}")
        print(f"Explained variance ratio (negative): {self.svd_neg_.explained_variance_ratio_.sum():.4f}")

        return self

    def _compute_semantic_distance(self, x_doc: np.ndarray) -> float:
        """
        Compute differential semantic distance for a single document.

        Δ_sem(d) = E- - E+ = ||z-||² - ||z+||²

        Where:
        - z+ = Σ+^-1 V+^T x_d (projection into positive latent space)
        - z- = Σ-^-1 V-^T x_d (projection into negative latent space)
        - E+ = ||z+||² (energy in positive space)
        - E- = ||z-||² (energy in negative space)

        Parameters
        ----------
        x_doc : sparse vector of shape (n_features,) or (1, n_features)
            TF-IDF representation of document

        Returns
        -------
        delta_sem : float
            Differential semantic distance
        """
        # Ensure x_doc is 1D
        if sparse.issparse(x_doc):
            x_doc = x_doc.toarray().ravel()
        else:
            x_doc = np.asarray(x_doc).ravel()

        # Project into positive latent space
        # z+ = Σ+^-1 V+^T x_d where V+^T has shape (k, n_features)
        z_pos = (self.V_pos_ @ x_doc) / self.Sigma_pos_

        # Project into negative latent space
        # z- = Σ-^-1 V-^T x_d where V-^T has shape (k, n_features)
        z_neg = (self.V_neg_ @ x_doc) / self.Sigma_neg_

        # Compute energies (squared L2 norms)
        E_pos = np.sum(z_pos ** 2)
        E_neg = np.sum(z_neg ** 2)

        # Differential semantic distance
        delta_sem = E_neg - E_pos

        return delta_sem

    def decision_function(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Compute differential semantic distances for documents.

        Parameters
        ----------
        X : list of str or array-like
            Documents to classify

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Differential semantic distances
            Negative values indicate positive class affinity
            Positive values indicate negative class affinity
        """
        # Preprocess documents
        X_tfidf = self.preprocessor_.transform(X)

        # Compute semantic distance for each document
        distances = np.array([
            self._compute_semantic_distance(X_tfidf[i])
            for i in range(X_tfidf.shape[0])
        ])

        return distances

    def predict(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predict class labels for documents.

        Decision rule:
        - If Δ_sem(d) < θ → positive class (1)
        - If Δ_sem(d) ≥ θ → negative class (0)

        Parameters
        ----------
        X : list of str or array-like
            Documents to classify

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        # Compute differential semantic distances
        distances = self.decision_function(X)

        # Apply decision rule
        y_pred = (distances < self.theta_).astype(int)

        return y_pred

    def predict_proba(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for documents.

        Uses sigmoid transformation of differential semantic distance:
        P(y=1|x) = sigmoid(-(Δ_sem - θ))

        Parameters
        ----------
        X : list of str or array-like
            Documents to classify

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities
        """
        # Compute differential semantic distances
        distances = self.decision_function(X)

        # Apply sigmoid transformation
        # Negative distance (closer to positive space) → higher P(y=1)
        scores = -(distances - self.theta_)
        proba_pos = 1 / (1 + np.exp(-scores))
        proba_neg = 1 - proba_pos

        # Stack probabilities
        proba = np.column_stack([proba_neg, proba_pos])

        return proba

    def get_latent_projections(self, X: Union[List[str], np.ndarray]) -> tuple:
        """
        Get latent space projections for documents.

        Useful for visualization and analysis.

        Parameters
        ----------
        X : list of str or array-like
            Documents to project

        Returns
        -------
        z_pos : ndarray of shape (n_samples, n_components)
            Projections in positive latent space

        z_neg : ndarray of shape (n_samples, n_components)
            Projections in negative latent space
        """
        # Preprocess documents
        X_tfidf = self.preprocessor_.transform(X)

        z_pos_list = []
        z_neg_list = []

        for i in range(X_tfidf.shape[0]):
            # Get document vector
            if sparse.issparse(X_tfidf):
                x_doc = X_tfidf[i].toarray().ravel()
            else:
                x_doc = X_tfidf[i].ravel()

            # Project into both spaces
            # V_pos_ has shape (k, n_features), x_doc has shape (n_features,)
            z_pos = (self.V_pos_ @ x_doc) / self.Sigma_pos_
            z_neg = (self.V_neg_ @ x_doc) / self.Sigma_neg_

            z_pos_list.append(z_pos)
            z_neg_list.append(z_neg)

        return np.array(z_pos_list), np.array(z_neg_list)

    def get_characteristic_terms(self, n_terms: int = 10) -> dict:
        """
        Get the most characteristic terms for each class.

        Extracts terms with highest weights on the first latent axis.

        Parameters
        ----------
        n_terms : int, optional (default=10)
            Number of top terms to return per class

        Returns
        -------
        terms : dict
            Dictionary with keys 'positive' and 'negative',
            each containing list of (term, weight) tuples
        """
        feature_names = self.preprocessor_.get_feature_names()

        # Get weights on first latent dimension
        # V_pos_ has shape (k, n_features), so first dimension is V_pos_[0, :]
        weights_pos = self.V_pos_[0, :]
        weights_neg = self.V_neg_[0, :]

        # Get top terms for positive class
        top_pos_idx = np.argsort(np.abs(weights_pos))[-n_terms:][::-1]
        top_pos = [(feature_names[i], weights_pos[i]) for i in top_pos_idx]

        # Get top terms for negative class
        top_neg_idx = np.argsort(np.abs(weights_neg))[-n_terms:][::-1]
        top_neg = [(feature_names[i], weights_neg[i]) for i in top_neg_idx]

        return {
            'positive': top_pos,
            'negative': top_neg
        }
