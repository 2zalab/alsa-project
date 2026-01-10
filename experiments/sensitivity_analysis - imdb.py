"""
Sensitivity analysis for A-LSA

Analyzes:
1. Sensitivity to latent dimension k
2. Impact of class imbalance
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
from src.baselines import get_baseline_models, create_imbalanced_dataset
from src.evaluation import evaluate_model
from src.visualization import plot_sensitivity_to_k, plot_imbalance_impact, save_results_table


def sensitivity_to_k(X_train, y_train, X_test, y_test, random_state=42):
    """
    Analyze sensitivity to latent dimension k.

    Tests k values: [10, 25, 50, 100, 150, 200, 300, 400, 500]
    """
    print("\n" + "="*80)
    print("SENSITIVITY TO LATENT DIMENSION k")
    print("="*80)

    k_values = [10, 25, 50, 100, 150, 200, 300, 400, 500]
    f1_scores = {
        'A-LSA': [],
        'LSA + Logistic Regression': [],
        'Logistic Regression': []
    }

    # Logistic Regression baseline (no dimensionality reduction)
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    lr_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            lowercase=True,
            stop_words='english',
            norm='l2'
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])

    print("\nTraining baseline Logistic Regression...")
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    lr_f1 = lr_metrics['F1-score (macro)']

    print(f"Baseline LR F1: {lr_f1:.4f}")

    # Test different k values
    for k in k_values:
        print(f"\nTesting k={k}...")

        # A-LSA
        try:
            alsa = AdaptiveLSA(n_components=k, random_state=random_state)
            alsa.fit(X_train, y_train)
            alsa_metrics = evaluate_model(alsa, X_test, y_test)
            f1_scores['A-LSA'].append(alsa_metrics['F1-score (macro)'])
            print(f"  A-LSA F1: {alsa_metrics['F1-score (macro)']:.4f}")
        except Exception as e:
            print(f"  A-LSA failed: {e}")
            f1_scores['A-LSA'].append(0)

        # Classical LSA + LR
        try:
            from sklearn.decomposition import TruncatedSVD

            lsa_lr = Pipeline([
                ('tfidf', TfidfVectorizer(
                    min_df=2,
                    max_df=0.95,
                    lowercase=True,
                    stop_words='english',
                    norm='l2'
                )),
                ('lsa', TruncatedSVD(n_components=k, random_state=random_state)),
                ('classifier', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state
                ))
            ])

            lsa_lr.fit(X_train, y_train)
            lsa_lr_metrics = evaluate_model(lsa_lr, X_test, y_test)
            f1_scores['LSA + Logistic Regression'].append(lsa_lr_metrics['F1-score (macro)'])
            print(f"  LSA+LR F1: {lsa_lr_metrics['F1-score (macro)']:.4f}")
        except Exception as e:
            print(f"  LSA+LR failed: {e}")
            f1_scores['LSA + Logistic Regression'].append(0)

        # Add LR baseline score (constant)
        f1_scores['Logistic Regression'].append(lr_f1)

    # Save results
    results_df = pd.DataFrame({
        'k': k_values,
        **f1_scores
    })

    save_results_table(
        results_df,
        'results/tables/sensitivity_to_k.csv',
        format='csv'
    )

    # Plot
    plot_sensitivity_to_k(
        k_values=k_values,
        f1_scores=f1_scores,
        save_path='results/figures/sensitivity_to_k.png',
        optimal_k=100
    )

    print("\nResults saved to:")
    print("  - results/tables/sensitivity_to_k.csv")
    print("  - results/figures/sensitivity_to_k.png")

    return results_df


def imbalance_analysis(X_train, y_train, X_test, y_test, random_state=42):
    """
    Analyze impact of class imbalance.

    Tests imbalance ratios: 1:1, 1:2, 1:3, 1:5, 1:10
    """
    print("\n" + "="*80)
    print("IMPACT OF CLASS IMBALANCE")
    print("="*80)

    imbalance_ratios = [1.0, 0.5, 0.33, 0.2, 0.1]  # 1:1, 1:2, 1:3, 1:5, 1:10
    f1_scores = {
        'A-LSA': [],
        'A-LSA (no θ adjustment)': [],
        'Logistic Regression': [],
        'LSA + Logistic Regression': []
    }

    for ratio in imbalance_ratios:
        print(f"\nTesting ratio {ratio:.2f} (1:{int(1/ratio)})...")

        # Create imbalanced training set
        if ratio < 1.0:
            X_train_imb, y_train_imb = create_imbalanced_dataset(
                X_train,
                y_train,
                ratio=ratio,
                random_state=random_state
            )
        else:
            X_train_imb = X_train
            y_train_imb = y_train

        print(f"  Training set: {len(X_train_imb)} samples")
        print(f"  Positive: {np.sum(y_train_imb == 1)}, Negative: {np.sum(y_train_imb == 0)}")

        # A-LSA (with θ adjustment)
        try:
            alsa = AdaptiveLSA(n_components=100, random_state=random_state)
            alsa.fit(X_train_imb, y_train_imb)
            alsa_metrics = evaluate_model(alsa, X_test, y_test)
            f1_scores['A-LSA'].append(alsa_metrics['F1-score (macro)'])
            print(f"  A-LSA F1: {alsa_metrics['F1-score (macro)']:.4f} (θ={alsa.theta_:.4f})")
        except Exception as e:
            print(f"  A-LSA failed: {e}")
            f1_scores['A-LSA'].append(0)

        # A-LSA without θ adjustment (θ = 0)
        try:
            alsa_no_theta = AdaptiveLSA(n_components=100, random_state=random_state)
            alsa_no_theta.fit(X_train_imb, y_train_imb)
            alsa_no_theta.theta_ = 0.0  # Force θ = 0
            alsa_no_theta_metrics = evaluate_model(alsa_no_theta, X_test, y_test)
            f1_scores['A-LSA (no θ adjustment)'].append(alsa_no_theta_metrics['F1-score (macro)'])
            print(f"  A-LSA (no θ) F1: {alsa_no_theta_metrics['F1-score (macro)']:.4f}")
        except Exception as e:
            print(f"  A-LSA (no θ) failed: {e}")
            f1_scores['A-LSA (no θ adjustment)'].append(0)

        # Logistic Regression
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline

            lr_model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    min_df=2,
                    max_df=0.95,
                    lowercase=True,
                    stop_words='english',
                    norm='l2'
                )),
                ('classifier', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state,
                    class_weight='balanced'
                ))
            ])

            lr_model.fit(X_train_imb, y_train_imb)
            lr_metrics = evaluate_model(lr_model, X_test, y_test)
            f1_scores['Logistic Regression'].append(lr_metrics['F1-score (macro)'])
            print(f"  LR F1: {lr_metrics['F1-score (macro)']:.4f}")
        except Exception as e:
            print(f"  LR failed: {e}")
            f1_scores['Logistic Regression'].append(0)

        # LSA + LR
        try:
            from sklearn.decomposition import TruncatedSVD

            lsa_lr = Pipeline([
                ('tfidf', TfidfVectorizer(
                    min_df=2,
                    max_df=0.95,
                    lowercase=True,
                    stop_words='english',
                    norm='l2'
                )),
                ('lsa', TruncatedSVD(n_components=100, random_state=random_state)),
                ('classifier', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=random_state,
                    class_weight='balanced'
                ))
            ])

            lsa_lr.fit(X_train_imb, y_train_imb)
            lsa_lr_metrics = evaluate_model(lsa_lr, X_test, y_test)
            f1_scores['LSA + Logistic Regression'].append(lsa_lr_metrics['F1-score (macro)'])
            print(f"  LSA+LR F1: {lsa_lr_metrics['F1-score (macro)']:.4f}")
        except Exception as e:
            print(f"  LSA+LR failed: {e}")
            f1_scores['LSA + Logistic Regression'].append(0)

    # Save results
    results_df = pd.DataFrame({
        'Ratio': imbalance_ratios,
        **f1_scores
    })

    save_results_table(
        results_df,
        'results/tables/imbalance_impact.csv',
        format='csv'
    )

    # Plot
    plot_imbalance_impact(
        imbalance_ratios=imbalance_ratios,
        f1_scores=f1_scores,
        save_path='results/figures/imbalance_impact.png'
    )

    print("\nResults saved to:")
    print("  - results/tables/imbalance_impact.csv")
    print("  - results/figures/imbalance_impact.png")

    return results_df


def main():
    """Run sensitivity analyses."""
    print("="*80)
    print("A-LSA SENSITIVITY ANALYSIS")
    print("="*80)

    # Configuration
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Create results directory
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Load IMDb dataset (best for these analyses)
    print("\n1. Loading IMDb dataset...")

    # Try to load IMDb, fall back to 20 Newsgroups
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from run_imdb import load_imdb_dataset

        texts, labels = load_imdb_dataset()

        if texts is not None:
            # Subsample for efficiency
            MAX_SAMPLES = 5000
            if len(texts) > MAX_SAMPLES:
                indices = np.random.RandomState(RANDOM_STATE).choice(
                    len(texts),
                    MAX_SAMPLES,
                    replace=False
                )
                texts = [texts[i] for i in indices]
                labels = labels[indices]
        else:
            raise Exception("IMDb not found")

    except:
        print("   IMDb not available, using 20 Newsgroups instead...")
        data = fetch_20newsgroups(
            subset='all',
            categories=('comp.graphics', 'rec.sport.hockey'),
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=RANDOM_STATE
        )
        texts = data.data
        labels = data.target

    print(f"   Total samples: {len(texts)}")

    # Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Run analyses
    print("\n3. Running sensitivity analyses...")

    # Sensitivity to k
    k_results = sensitivity_to_k(X_train, y_train, X_test, y_test, RANDOM_STATE)

    # Impact of imbalance
    imbalance_results = imbalance_analysis(X_train, y_train, X_test, y_test, RANDOM_STATE)

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
