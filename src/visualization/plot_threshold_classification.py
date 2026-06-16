# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

def plot_threshold_classification(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    accuracy: float,
    class_names: Tuple[str, str] = ('Class 0', 'Class 1'),
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plot of mean vs std MFBM with decision threshold line.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, 2): [mean_MFBM, std_MFBM].
    y : np.ndarray
        Binary label vector, shape (n_samples,).
    threshold : float
        Decision threshold to display as a horizontal line.
    accuracy : float
        Accuracy value to display in the plot title.
    class_names : tuple of str, optional
        Names for class 0 and class 1, by default ('Class 0', 'Class 1').
    save_path : str or None, optional
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label=class_names[0], alpha=0.7, marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label=class_names[1], alpha=0.7, marker='x')
    plt.axhline(y=threshold, linestyle='--', color='black', label=f'Threshold = {threshold:.2f}')
    plt.xlabel("Mean MFBM")
    plt.ylabel("Std MFBM (band 3)")
    plt.title(f"{class_names[0]} vs {class_names[1]} — Accuracy: {accuracy:.2%}")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()