"""
Evaluation metrics and utilities for A-LSA

Implements comprehensive evaluation pipeline including:
- Performance metrics (F1, Accuracy, Precision, Recall)
- Cross-validation
- Computational efficiency metrics
- Confusion matrices
"""

import numpy as np
import time
import psutil
import os
from typing import Dict, Tuple, Any, List
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    make_scorer
)
import pandas as pd


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a model on test data.

    Computes all required metrics:
    - F1-score macro (PRIMARY METRIC)
    - Accuracy
    - Precision (macro)
    - Recall (macro)

    Parameters
    ----------
    model : estimator
        Fitted model with predict() method

    X_test : array-like or list of str
        Test features

    y_test : array-like
        True labels

    model_name : str, optional (default="Model")
        Model name for display

    Returns
    -------
    metrics : dict
        Dictionary of metric name -> value
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        'F1-score (macro)': f1_score(y_test, y_pred, average='macro'),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-score (weighted)': f1_score(y_test, y_pred, average='weighted'),
    }

    # Per-class metrics
    f1_per_class = f1_score(y_test, y_pred, average=None)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)

    metrics['F1-score (class 0)'] = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    metrics['F1-score (class 1)'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    metrics['Precision (class 0)'] = precision_per_class[0] if len(precision_per_class) > 0 else 0.0
    metrics['Precision (class 1)'] = precision_per_class[1] if len(precision_per_class) > 1 else 0.0
    metrics['Recall (class 0)'] = recall_per_class[0] if len(recall_per_class) > 0 else 0.0
    metrics['Recall (class 1)'] = recall_per_class[1] if len(recall_per_class) > 1 else 0.0

    return metrics


def cross_validate_model(
    model,
    X,
    y,
    n_splits: int = 5,
    random_state: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.

    Parameters
    ----------
    model : estimator
        Model to evaluate (must have fit() and predict() methods)

    X : array-like or list of str
        Features

    y : array-like
        Labels

    n_splits : int, optional (default=5)
        Number of cross-validation folds

    random_state : int, optional (default=None)
        Random seed for reproducibility

    verbose : bool, optional (default=True)
        Whether to print progress

    Returns
    -------
    results : dict
        Dictionary containing:
        - Mean and std for each metric
        - Per-fold scores
        - Confusion matrices
    """
    if verbose:
        print(f"\nPerforming {n_splits}-fold stratified cross-validation...")

    # Define scoring metrics
    scoring = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0)
    }

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1  # Keep sequential for memory management
    )

    # Compute statistics
    results = {
        'F1-score (macro)': {
            'mean': np.mean(cv_results['test_f1_macro']),
            'std': np.std(cv_results['test_f1_macro']),
            'scores': cv_results['test_f1_macro']
        },
        'Accuracy': {
            'mean': np.mean(cv_results['test_accuracy']),
            'std': np.std(cv_results['test_accuracy']),
            'scores': cv_results['test_accuracy']
        },
        'Precision (macro)': {
            'mean': np.mean(cv_results['test_precision_macro']),
            'std': np.std(cv_results['test_precision_macro']),
            'scores': cv_results['test_precision_macro']
        },
        'Recall (macro)': {
            'mean': np.mean(cv_results['test_recall_macro']),
            'std': np.std(cv_results['test_recall_macro']),
            'scores': cv_results['test_recall_macro']
        }
    }

    if verbose:
        print(f"\nCross-validation results ({n_splits} folds):")
        print(f"  F1-score (macro):  {results['F1-score (macro)']['mean']:.4f} ± {results['F1-score (macro)']['std']:.4f}")
        print(f"  Accuracy:          {results['Accuracy']['mean']:.4f} ± {results['Accuracy']['std']:.4f}")
        print(f"  Precision (macro): {results['Precision (macro)']['mean']:.4f} ± {results['Precision (macro)']['std']:.4f}")
        print(f"  Recall (macro):    {results['Recall (macro)']['mean']:.4f} ± {results['Recall (macro)']['std']:.4f}")

    return results


def measure_efficiency(
    model,
    X_test,
    n_predictions: int = 1000
) -> Dict[str, float]:
    """
    Measure computational efficiency of a model.

    Metrics:
    - Average inference time (ms per document)
    - Memory footprint (MB)

    Parameters
    ----------
    model : estimator
        Fitted model

    X_test : array-like or list of str
        Test documents

    n_predictions : int, optional (default=1000)
        Number of predictions to time

    Returns
    -------
    efficiency : dict
        Dictionary of efficiency metrics
    """
    # Subsample if needed
    if len(X_test) > n_predictions:
        indices = np.random.choice(len(X_test), n_predictions, replace=False)
        if isinstance(X_test, list):
            X_sample = [X_test[i] for i in indices]
        else:
            X_sample = X_test[indices]
    else:
        X_sample = X_test
        n_predictions = len(X_test)

    # Measure inference time
    start_time = time.time()
    _ = model.predict(X_sample)
    end_time = time.time()

    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time_per_doc = total_time / n_predictions

    # Measure memory footprint
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    return {
        'Avg inference time (ms/doc)': avg_time_per_doc,
        'Memory footprint (MB)': memory_mb,
        'Total predictions': n_predictions
    }


def get_confusion_matrix(
    model,
    X_test,
    y_test,
    class_names: List[str] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute and format confusion matrix.

    Parameters
    ----------
    model : estimator
        Fitted model

    X_test : array-like or list of str
        Test features

    y_test : array-like
        True labels

    class_names : list of str, optional (default=None)
        Names of classes for display

    Returns
    -------
    cm : ndarray
        Confusion matrix

    cm_df : DataFrame
        Formatted confusion matrix as DataFrame
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in class_names],
        columns=[f"Pred {name}" for name in class_names]
    )

    return cm, cm_df


def print_classification_report(
    model,
    X_test,
    y_test,
    class_names: List[str] = None,
    model_name: str = "Model"
):
    """
    Print detailed classification report.

    Parameters
    ----------
    model : estimator
        Fitted model

    X_test : array-like or list of str
        Test features

    y_test : array-like
        True labels

    class_names : list of str, optional (default=None)
        Names of classes

    model_name : str, optional (default="Model")
        Model name for display
    """
    y_pred = model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*60}\n")

    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    cm, cm_df = get_confusion_matrix(model, X_test, y_test, class_names)
    print("\nConfusion Matrix:")
    print(cm_df)
    print()


def compare_models(
    models: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    n_cv_folds: int = 5,
    random_state: int = None,
    measure_efficiency_flag: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.

    Parameters
    ----------
    models : dict
        Dictionary of model_name -> model instance

    X_train : array-like or list of str
        Training features

    y_train : array-like
        Training labels

    X_test : array-like or list of str
        Test features

    y_test : array-like
        Test labels

    n_cv_folds : int, optional (default=5)
        Number of cross-validation folds

    random_state : int, optional (default=None)
        Random seed

    measure_efficiency_flag : bool, optional (default=True)
        Whether to measure computational efficiency

    Returns
    -------
    results_df : DataFrame
        Comparison table with all metrics
    """
    results = []

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # Train model
        print("Training...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        print(f"Training time: {train_time:.2f}s")

        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = evaluate_model(model, X_test, y_test, model_name)

        # Cross-validation
        cv_results = cross_validate_model(
            model,
            X_train,
            y_train,
            n_splits=n_cv_folds,
            random_state=random_state,
            verbose=True
        )

        # Efficiency metrics
        if measure_efficiency_flag:
            print("Measuring efficiency...")
            efficiency = measure_efficiency(model, X_test)
        else:
            efficiency = {}

        # Combine results
        result = {
            'Model': model_name,
            'Test F1 (macro)': test_metrics['F1-score (macro)'],
            'Test Accuracy': test_metrics['Accuracy'],
            'Test Precision (macro)': test_metrics['Precision (macro)'],
            'Test Recall (macro)': test_metrics['Recall (macro)'],
            'CV F1 (mean)': cv_results['F1-score (macro)']['mean'],
            'CV F1 (std)': cv_results['F1-score (macro)']['std'],
            'CV Accuracy (mean)': cv_results['Accuracy']['mean'],
            'CV Accuracy (std)': cv_results['Accuracy']['std'],
            'Train time (s)': train_time
        }

        if efficiency:
            result['Inference (ms/doc)'] = efficiency['Avg inference time (ms/doc)']

        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df
