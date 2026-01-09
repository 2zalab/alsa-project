"""
Experiment script for IMDb Movie Reviews dataset

Evaluates A-LSA and baseline models on sentiment analysis task.
Dataset characteristics:
- 50,000 movie reviews
- Binary: positive (50%) vs negative (50%)
- Long texts (~230 words)
- Perfectly balanced
"""

import sys
import os
import pandas as pd
import numpy as np
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
    save_results_table
)


def load_imdb_dataset(data_dir='data/imdb'):
    """
    Load IMDb Movie Reviews dataset.

    Expected structure:
    data_dir/
        train/
            pos/  (positive reviews)
            neg/  (negative reviews)
        test/
            pos/
            neg/

    Parameters
    ----------
    data_dir : str
        Path to IMDb dataset directory

    Returns
    -------
    texts : list of str
        Movie reviews

    labels : ndarray
        Binary labels (0=negative, 1=positive)
    """
    texts = []
    labels = []

    # Check if dataset exists
    train_pos_dir = os.path.join(data_dir, 'train', 'pos')
    if not os.path.exists(train_pos_dir):
        print(f"Dataset not found at {data_dir}")
        print("\nPlease download from:")
        print("https://ai.stanford.edu/~amaas/data/sentiment/")
        print(f"\nExpected directory structure:")
        print(f"  {data_dir}/train/pos/")
        print(f"  {data_dir}/train/neg/")
        print(f"  {data_dir}/test/pos/")
        print(f"  {data_dir}/test/neg/")
        return None, None

    # Load training data
    for split in ['train', 'test']:
        for label_name, label_value in [('pos', 1), ('neg', 0)]:
            dir_path = os.path.join(data_dir, split, label_name)

            if not os.path.exists(dir_path):
                continue

            for filename in os.listdir(dir_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(dir_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        texts.append(text)
                        labels.append(label_value)

    labels = np.array(labels)

    return texts, labels


def main():
    """Run IMDb experiment."""
    print("="*80)
    print("IMDB MOVIE REVIEWS EXPERIMENT")
    print("="*80)

    # Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_CV_FOLDS = 5
    N_COMPONENTS = 100
    MAX_SAMPLES = 10000  # Subsample for faster experimentation

    # Create results directory
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Load dataset
    print("\n1. Loading dataset...")
    texts, labels = load_imdb_dataset()

    if texts is None:
        return

    print(f"   Total reviews: {len(texts)}")
    print(f"   Positive reviews: {np.sum(labels == 1)} ({100*np.mean(labels == 1):.2f}%)")
    print(f"   Negative reviews: {np.sum(labels == 0)} ({100*np.mean(labels == 0):.2f}%)")

    # Subsample for efficiency
    if len(texts) > MAX_SAMPLES:
        print(f"\n   Subsampling to {MAX_SAMPLES} reviews for efficiency...")
        indices = np.random.RandomState(RANDOM_STATE).choice(
            len(texts),
            MAX_SAMPLES,
            replace=False
        )
        texts = [texts[i] for i in indices]
        labels = labels[indices]

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"   Training set: {len(X_train)} reviews")
    print(f"   Test set: {len(X_test)} reviews")

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
    results_df['Dataset'] = 'IMDb'

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df[['Model', 'Test F1 (macro)', 'Test Accuracy', 'CV F1 (mean)', 'CV F1 (std)']].to_string(index=False))

    # Save results
    print("\n5. Saving results...")
    save_results_table(
        results_df,
        'results/tables/imdb_results.csv',
        format='csv'
    )
    save_results_table(
        results_df,
        'results/tables/imdb_results.md',
        format='markdown'
    )
    print("   Saved to results/tables/")

    # Visualizations
    print("\n6. Generating visualizations...")

    # Performance comparison
    plot_performance_comparison(
        results_df,
        metric='Test F1 (macro)',
        save_path='results/figures/imdb_comparison.png',
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
            class_names=['Negative', 'Positive'],
            save_path='results/figures/imdb_terms.png'
        )

        print("   Top positive sentiment terms:", [term for term, _ in char_terms['positive'][:5]])
        print("   Top negative sentiment terms:", [term for term, _ in char_terms['negative'][:5]])

    # Confusion matrices
    print("\n8. Computing confusion matrices...")
    for model_name, model in models.items():
        cm, cm_df = get_confusion_matrix(
            model,
            X_test,
            y_test,
            class_names=['Negative', 'Positive']
        )
        print(f"\n{model_name} Confusion Matrix:")
        print(cm_df)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/tables/imdb_results.csv")
    print("  - results/tables/imdb_results.md")
    print("  - results/figures/imdb_comparison.png")
    print("  - results/figures/imdb_terms.png")


if __name__ == '__main__':
    main()
