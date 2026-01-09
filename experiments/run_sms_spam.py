"""
Experiment script for SMS Spam Collection dataset

Evaluates A-LSA and baseline models on spam detection task.
Dataset characteristics:
- 5,574 SMS messages
- Binary: spam (13.4%) vs ham (86.6%)
- Short texts (~80 characters)
- Highly imbalanced
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


def load_sms_spam_dataset(data_path='data/sms_spam/SMSSpamCollection'):
    """
    Load SMS Spam Collection dataset.

    Parameters
    ----------
    data_path : str
        Path to SMSSpamCollection file

    Returns
    -------
    texts : list of str
        SMS messages

    labels : ndarray
        Binary labels (0=ham, 1=spam)
    """
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("\nPlease download from:")
        print("https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        print(f"\nExpected file: {data_path}")
        return None, None

    # Load data (tab-separated: label \t message)
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label, text = parts
                data.append((label, text))

    # Convert to arrays
    texts = [text for _, text in data]
    labels = np.array([1 if label == 'spam' else 0 for label, _ in data])

    return texts, labels


def main():
    """Run SMS Spam experiment."""
    print("="*80)
    print("SMS SPAM COLLECTION EXPERIMENT")
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
    texts, labels = load_sms_spam_dataset()

    if texts is None:
        return

    print(f"   Total messages: {len(texts)}")
    print(f"   Spam messages: {np.sum(labels == 1)} ({100*np.mean(labels == 1):.2f}%)")
    print(f"   Ham messages: {np.sum(labels == 0)} ({100*np.mean(labels == 0):.2f}%)")

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"   Training set: {len(X_train)} messages")
    print(f"   Test set: {len(X_test)} messages")

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
    results_df['Dataset'] = 'SMS Spam'

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df[['Model', 'Test F1 (macro)', 'Test Accuracy', 'CV F1 (mean)', 'CV F1 (std)']].to_string(index=False))

    # Save results
    print("\n5. Saving results...")
    save_results_table(
        results_df,
        'results/tables/sms_spam_results.csv',
        format='csv'
    )
    save_results_table(
        results_df,
        'results/tables/sms_spam_results.md',
        format='markdown'
    )
    print("   Saved to results/tables/")

    # Visualizations
    print("\n6. Generating visualizations...")

    # Performance comparison
    plot_performance_comparison(
        results_df,
        metric='Test F1 (macro)',
        save_path='results/figures/sms_spam_comparison.png',
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
            class_names=['Ham', 'Spam'],
            save_path='results/figures/sms_spam_terms.png'
        )

        print("   Top spam terms:", [term for term, _ in char_terms['positive'][:5]])
        print("   Top ham terms:", [term for term, _ in char_terms['negative'][:5]])

    # Confusion matrices
    print("\n8. Computing confusion matrices...")
    for model_name, model in models.items():
        cm, cm_df = get_confusion_matrix(
            model,
            X_test,
            y_test,
            class_names=['Ham', 'Spam']
        )
        print(f"\n{model_name} Confusion Matrix:")
        print(cm_df)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/tables/sms_spam_results.csv")
    print("  - results/tables/sms_spam_results.md")
    print("  - results/figures/sms_spam_comparison.png")
    print("  - results/figures/sms_spam_terms.png")


if __name__ == '__main__':
    main()
