"""
Visualization functions for A-LSA results

Generates all required plots:
1. Sensitivity to k (latent dimension)
2. Impact of class imbalance
3. t-SNE visualization of latent spaces
4. Performance comparison across models
5. Characteristic terms visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_sensitivity_to_k(
    k_values: List[int],
    f1_scores: Dict[str, List[float]],
    save_path: Optional[str] = None,
    optimal_k: int = 100,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot F1-score as a function of latent dimension k.

    Shows how model performance varies with the dimensionality of
    the latent semantic space.

    Parameters
    ----------
    k_values : list of int
        List of k values tested (e.g., [10, 25, 50, 100, 150, 200, 300, 400, 500])

    f1_scores : dict
        Dictionary mapping model names to lists of F1 scores
        Example: {'A-LSA': [0.85, 0.88, 0.90, ...], 'LSA + LR': [...]}

    save_path : str, optional (default=None)
        Path to save figure (if None, figure is displayed)

    optimal_k : int, optional (default=100)
        Optimal k value to highlight

    figsize : tuple, optional (default=(10, 6))
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(f1_scores)))

    for (model_name, scores), color in zip(f1_scores.items(), colors):
        ax.plot(
            k_values,
            scores,
            marker='o',
            linewidth=2,
            label=model_name,
            color=color,
            markersize=6
        )

    # Highlight optimal k
    if optimal_k in k_values:
        ax.axvline(
            optimal_k,
            color='red',
            linestyle='--',
            alpha=0.5,
            linewidth=1.5,
            label=f'Optimal k={optimal_k}'
        )

    ax.set_xlabel('Latent Dimension (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-score (macro)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity to Latent Dimension k', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.80, 0.95])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_imbalance_impact(
    imbalance_ratios: List[float],
    f1_scores: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot impact of class imbalance on model performance.

    Shows robustness of different models to varying degrees of class imbalance.

    Parameters
    ----------
    imbalance_ratios : list of float
        List of imbalance ratios (e.g., [1.0, 0.5, 0.33, 0.2, 0.1] for 1:1, 1:2, 1:3, 1:5, 1:10)

    f1_scores : dict
        Dictionary mapping model names to lists of F1 scores

    save_path : str, optional (default=None)
        Path to save figure

    figsize : tuple, optional (default=(10, 6))
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert ratios to readable labels
    ratio_labels = [f"1:{int(1/r)}" if r < 1 else "1:1" for r in imbalance_ratios]

    colors = plt.cm.tab10(np.linspace(0, 1, len(f1_scores)))

    for (model_name, scores), color in zip(f1_scores.items(), colors):
        ax.plot(
            ratio_labels,
            scores,
            marker='o',
            linewidth=2,
            label=model_name,
            color=color,
            markersize=6
        )

    ax.set_xlabel('Class Imbalance Ratio (Positive:Negative)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-score (macro)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Class Imbalance on Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.75, 0.95])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_tsne_visualization(
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    random_state: int = None
):
    """
    Visualize latent space separability using t-SNE.

    Concatenates positive and negative latent projections and reduces
    to 2D using t-SNE for visualization.

    Parameters
    ----------
    z_pos : ndarray of shape (n_samples, n_components)
        Projections in positive latent space

    z_neg : ndarray of shape (n_samples, n_components)
        Projections in negative latent space

    y_true : ndarray
        True class labels

    class_names : list of str, optional (default=None)
        Names of classes

    save_path : str, optional (default=None)
        Path to save figure

    figsize : tuple, optional (default=(10, 8))
        Figure size

    perplexity : int, optional (default=30)
        t-SNE perplexity parameter

    random_state : int, optional (default=None)
        Random seed
    """
    # Concatenate latent projections to form composite representation
    z_composite = np.concatenate([z_pos, z_neg], axis=1)

    print(f"Applying t-SNE to composite latent space (dim={z_composite.shape[1]})...")

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000
    )
    z_2d = tsne.fit_transform(z_composite)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if class_names is None:
        class_names = ['Negative', 'Positive']

    # Plot each class
    for class_idx, class_name in enumerate(class_names):
        mask = (y_true == class_idx)
        marker = 'o' if class_idx == 1 else 's'  # Circle for positive, square for negative
        color = 'blue' if class_idx == 1 else 'red'

        ax.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            c=color,
            marker=marker,
            s=50,
            alpha=0.6,
            label=class_name,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('t-SNE Visualization of Latent Space Separability', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_performance_comparison(
    results_df: pd.DataFrame,
    metric: str = 'Test F1 (macro)',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    highlight_model: str = 'A-LSA'
):
    """
    Compare performance of different models using grouped bar plot.

    Parameters
    ----------
    results_df : DataFrame
        Results DataFrame with columns: Model, dataset_name, metric_value

    metric : str, optional (default='Test F1 (macro)')
        Metric to plot

    save_path : str, optional (default=None)
        Path to save figure

    figsize : tuple, optional (default=(12, 6))
        Figure size

    highlight_model : str, optional (default='A-LSA')
        Model to highlight with different color
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    models = results_df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25

    # Get unique datasets (assuming results_df has a 'Dataset' column)
    if 'Dataset' in results_df.columns:
        datasets = results_df['Dataset'].unique()
    else:
        # If no dataset column, treat as single dataset
        datasets = ['Results']
        results_df['Dataset'] = 'Results'

    # Plot bars for each dataset
    for i, dataset in enumerate(datasets):
        dataset_data = results_df[results_df['Dataset'] == dataset]

        values = []
        errors = []

        for model in models:
            model_data = dataset_data[dataset_data['Model'] == model]
            if not model_data.empty:
                values.append(model_data[metric].values[0])

                # Get error if available (from CV std)
                std_col = metric.replace('Test', 'CV').replace('(macro)', '(std)').replace('(mean)', '(std)')
                if std_col in model_data.columns:
                    errors.append(model_data[std_col].values[0])
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)

        # Choose color
        colors = ['orange' if model == highlight_model else 'steelblue' for model in models]

        ax.bar(
            x + i * width,
            values,
            width,
            label=dataset,
            yerr=errors if any(errors) else None,
            capsize=3,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_characteristic_terms(
    terms_pos: List[Tuple[str, float]],
    terms_neg: List[Tuple[str, float]],
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    n_terms: int = 10
):
    """
    Visualize most characteristic terms for each class.

    Creates a horizontal bar plot showing top terms for each class.

    Parameters
    ----------
    terms_pos : list of (str, float)
        List of (term, weight) tuples for positive class

    terms_neg : list of (str, float)
        List of (term, weight) tuples for negative class

    class_names : list of str, optional (default=None)
        Names of classes

    save_path : str, optional (default=None)
        Path to save figure

    figsize : tuple, optional (default=(12, 8))
        Figure size

    n_terms : int, optional (default=10)
        Number of top terms to display
    """
    if class_names is None:
        class_names = ['Negative Class', 'Positive Class']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Positive class terms
    terms_pos = terms_pos[:n_terms]
    words_pos = [term for term, _ in terms_pos]
    weights_pos = [abs(weight) for _, weight in terms_pos]

    ax1.barh(
        range(len(words_pos)),
        weights_pos,
        color='steelblue',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_yticks(range(len(words_pos)))
    ax1.set_yticklabels(words_pos)
    ax1.invert_yaxis()
    ax1.set_xlabel('Weight (absolute)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {n_terms} Terms: {class_names[1]}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Negative class terms
    terms_neg = terms_neg[:n_terms]
    words_neg = [term for term, _ in terms_neg]
    weights_neg = [abs(weight) for _, weight in terms_neg]

    ax2.barh(
        range(len(words_neg)),
        weights_neg,
        color='coral',
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    ax2.set_yticks(range(len(words_neg)))
    ax2.set_yticklabels(words_neg)
    ax2.invert_yaxis()
    ax2.set_xlabel('Weight (absolute)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {n_terms} Terms: {class_names[0]}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def save_results_table(
    results_df: pd.DataFrame,
    save_path: str,
    format: str = 'csv'
):
    """
    Save results table to file.

    Parameters
    ----------
    results_df : DataFrame
        Results DataFrame

    save_path : str
        Path to save file

    format : str, optional (default='csv')
        Output format ('csv', 'latex', 'markdown')
    """
    if format == 'csv':
        results_df.to_csv(save_path, index=False)
    elif format == 'latex':
        latex_str = results_df.to_latex(index=False, float_format="%.4f")
        with open(save_path, 'w') as f:
            f.write(latex_str)
    elif format == 'markdown':
        try:
            # Try using pandas to_markdown (requires tabulate)
            md_str = results_df.to_markdown(index=False, floatfmt=".4f")
        except ImportError:
            # Fallback: create markdown manually if tabulate is not available
            print("Warning: 'tabulate' not installed. Creating basic markdown table.")

            # Header
            headers = results_df.columns.tolist()
            md_str = "| " + " | ".join(str(h) for h in headers) + " |\n"
            md_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            # Rows
            for _, row in results_df.iterrows():
                formatted_row = []
                for val in row:
                    if isinstance(val, float):
                        formatted_row.append(f"{val:.4f}")
                    else:
                        formatted_row.append(str(val))
                md_str += "| " + " | ".join(formatted_row) + " |\n"

        with open(save_path, 'w') as f:
            f.write(md_str)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved results table to {save_path}")
