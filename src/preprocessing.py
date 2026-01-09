"""
Text preprocessing pipeline for A-LSA
Handles tokenization, stop word removal, and TF-IDF vectorization
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Complete text preprocessing pipeline for A-LSA.

    Performs:
    1. Tokenization
    2. Lowercase conversion
    3. Stop word removal
    4. Punctuation removal
    5. Rare/frequent term filtering
    6. TF-IDF vectorization with L2 normalization

    Parameters
    ----------
    max_features : int, optional (default=None)
        Maximum number of features (vocabulary size)
    min_df : int or float, optional (default=2)
        Minimum document frequency for terms
    max_df : float, optional (default=0.95)
        Maximum document frequency for terms
    language : str, optional (default='english')
        Language for stop words
    ngram_range : tuple, optional (default=(1, 1))
        N-gram range for tokenization
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: int = 2,
        max_df: float = 0.95,
        language: str = 'english',
        ngram_range: Tuple[int, int] = (1, 1)
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.language = language
        self.ngram_range = ngram_range

        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            print(f"Warning: Could not load stop words for {language}, using empty set")
            self.stop_words = set()

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=list(self.stop_words) if self.stop_words else None,
            tokenizer=self._tokenize,
            lowercase=True,
            norm='l2',  # L2 normalization as specified
            ngram_range=self.ngram_range,
            dtype=np.float32
        )

        self.vocabulary_ = None
        self.is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Custom tokenizer that:
        - Converts to lowercase
        - Removes punctuation and special characters
        - Splits on whitespace

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        tokens : list of str
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters, keep only alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Split on whitespace and filter empty strings
        tokens = [token for token in text.split() if token]

        return tokens

    def fit(self, documents: List[str]) -> 'TextPreprocessor':
        """
        Fit the preprocessor on training documents.

        Parameters
        ----------
        documents : list of str
            Training documents

        Returns
        -------
        self : TextPreprocessor
            Fitted preprocessor
        """
        self.vectorizer.fit(documents)
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.is_fitted = True
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix.

        Parameters
        ----------
        documents : list of str
            Documents to transform

        Returns
        -------
        X : sparse matrix of shape (n_documents, n_features)
            TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit and transform documents in one step.

        Parameters
        ----------
        documents : list of str
            Documents to fit and transform

        Returns
        -------
        X : sparse matrix of shape (n_documents, n_features)
            TF-IDF matrix
        """
        X = self.vectorizer.fit_transform(documents)
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.is_fitted = True
        return X

    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary).

        Returns
        -------
        feature_names : list of str
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")

        return self.vectorizer.get_feature_names_out().tolist()

    def get_vocabulary_size(self) -> int:
        """
        Get vocabulary size.

        Returns
        -------
        vocab_size : int
            Number of features in vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        return len(self.vocabulary_)


def preprocess_text(text: str, remove_stopwords: bool = True, language: str = 'english') -> str:
    """
    Simple text preprocessing function.

    Parameters
    ----------
    text : str
        Input text
    remove_stopwords : bool, optional (default=True)
        Whether to remove stop words
    language : str, optional (default='english')
        Language for stop words

    Returns
    -------
    cleaned_text : str
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words if requested
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words(language))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            pass

    # Join back
    return ' '.join(tokens)
