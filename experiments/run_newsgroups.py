"""
Experiment script for 20 Newsgroups dataset (Binary)

Evaluates A-LSA and baseline models on topic classification task.
Dataset characteristics:
- ~2,000 documents
- Binary: comp.graphics vs rec.sport.hockey
- Variable length forum posts
- Well-separated semantic domains
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alsa import AdaptiveLSA
from src.baselines import get_baseline_models
from src.evaluation import compare_models, get_confusion_matrix
from src.visualization import (
    plot_performance_comparison,
    plot_characteristic_terms,
    plot_tsne_visualization,
    save_results_table
)


def load_newsgroups_binary(
    categories=('comp.graphics', 'rec.sport.hockey'),
    remove=('headers', 'footers', 'quotes')
):
    """
    Load 20 Newsgroups dataset as binary classification task.

    Parameters
    ----------
    categories : tuple of str
        Two newsgroup categories to use

    remove : tuple of str
        Parts of posts to remove

    Returns
    -------
    texts : list of str
        Documents

    labels : ndarray
        Binary labels (0 or 1)

    category_names : list of str
        Names of the two categories
    """
    print(f"   Loading categories: {categories[0]} vs {categories[1]}")

    # Fetch data
    data = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=remove,
        shuffle=True,
        random_state=42
    )

    texts = data.data
    labels = data.target

    return texts, labels, list(categories)


def main():
    """Run 20 Newsgroups experiment."""
    print("="*80)
    print("20 NEWSGROUPS BINARY CLASSIFICATION EXPERIMENT")
    print("="*80)

    # Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_CV_FOLDS = 5
    N_COMPONENTS = 100

    # Create results directory
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Load dataset
    print("\n1. Loading dataset...")
    texts, labels, category_names = load_newsgroups_binary()

    print(f"   Total documents: {len(texts)}")
    print(f"   {category_names[0]}: {np.sum(labels == 0)} documents")
    print(f"   {category_names[1]}: {np.sum(labels == 1)} documents")

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"   Training set: {len(X_train)} documents")
    print(f"   Test set: {len(X_test)} documents")

    # Initialize models
    print("\n3. Initializing models...")
    models = {}

    # A-LSA
    models['A-LSA'] = AdaptiveLSA(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )

    # Baselines
    baseline_models = get_baseline_models(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )
    models.update(baseline_models)

    print(f"   Models to evaluate: {list(models.keys())}")

    # Run comparison
    print("\n4. Running experiments...")
    results_df = compare_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_cv_folds=N_CV_FOLDS,
        random_state=RANDOM_STATE
    )

    # Add dataset name
    results_df['Dataset'] = '20 Newsgroups'

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df[['Model', 'Test F1 (macro)', 'Test Accuracy', 'CV F1 (mean)', 'CV F1 (std)']].to_string(index=False))

    # Save results
    print("\n5. Saving results...")
    save_results_table(
        results_df,
        'results/tables/newsgroups_results.csv',
        format='csv'
    )
    save_results_table(
        results_df,
        'results/tables/newsgroups_results.md',
        format='markdown'
    )
    print("   Saved to results/tables/")

    # Visualizations
    print("\n6. Generating visualizations...")

    # Performance comparison
    plot_performance_comparison(
        results_df,
        metric='Test F1 (macro)',
        save_path='results/figures/newsgroups_comparison.png',
        highlight_model='A-LSA'
    )

    # Characteristic terms (for A-LSA)
    print("\n7. Analyzing characteristic terms...")
    alsa_model = models['A-LSA']
    if hasattr(alsa_model, 'get_characteristic_terms'):
        char_terms = alsa_model.get_characteristic_terms(n_terms=10)

        plot_characteristic_terms(
            terms_pos=char_terms['positive'],
            terms_neg=char_terms['negative'],
            class_names=[category_names[0], category_names[1]],
            save_path='results/figures/newsgroups_terms.png'
        )

        print(f"   Top {category_names[1]} terms:", [term for term, _ in char_terms['positive'][:5]])
        print(f"   Top {category_names[0]} terms:", [term for term, _ in char_terms['negative'][:5]])

    # t-SNE visualization (unique to this dataset)
    print("\n8. Generating t-SNE visualization...")
    if hasattr(alsa_model, 'get_latent_projections'):
        z_pos, z_neg = alsa_model.get_latent_projections(X_test)

        plot_tsne_visualization(
            z_pos=z_pos,
            z_neg=z_neg,
            y_true=y_test,
            class_names=[category_names[0], category_names[1]],
            save_path='results/figures/newsgroups_tsne.png',
            random_state=RANDOM_STATE
        )

    # Confusion matrices
    print("\n9. Computing confusion matrices...")
    for model_name, model in models.items():
        cm, cm_df = get_confusion_matrix(
            model,
            X_test,
            y_test,
            class_names=[category_names[0], category_names[1]]
        )
        print(f"\n{model_name} Confusion Matrix:")
        print(cm_df)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/tables/newsgroups_results.csv")
    print("  - results/tables/newsgroups_results.md")
    print("  - results/figures/newsgroups_comparison.png")
    print("  - results/figures/newsgroups_terms.png")
    print("  - results/figures/newsgroups_tsne.png")


if __name__ == '__main__':
    main()
