"""
A-LSA: Adaptive Latent Semantic Analysis for Binary Text Classification
Université de Maroua, Cameroun
Author: Isaac Touza
Version: 1.0
Date: January 2026
"""

from .alsa import AdaptiveLSA
from .preprocessing import TextPreprocessor
from .baselines import get_baseline_models
from .evaluation import evaluate_model, cross_validate_model
from .visualization import (
    plot_sensitivity_to_k,
    plot_imbalance_impact,
    plot_tsne_visualization,
    plot_performance_comparison,
    plot_characteristic_terms
)

__version__ = "1.0.0"
__author__ = "Isaac Touza"

__all__ = [
    'AdaptiveLSA',
    'TextPreprocessor',
    'get_baseline_models',
    'evaluate_model',
    'cross_validate_model',
    'plot_sensitivity_to_k',
    'plot_imbalance_impact',
    'plot_tsne_visualization',
    'plot_performance_comparison',
    'plot_characteristic_terms'
]
