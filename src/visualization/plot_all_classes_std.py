import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict

def plot_all_classes_std(
    df: pd.DataFrame,
    std_band: int = 2,
    thresholds: Optional[Dict[str, float]] = None,
    class_column: str = 'group',
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plot of std MFBM (single band) across all instances, 
    colored by class, with optional threshold lines.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'std_MFBM' (array per row) and the class column.
    std_band : int, optional
        Zero-based index of the mel band to plot, by default 2.
    thresholds : dict of str to float, or None, optional
        Mapping from threshold name to value, drawn as horizontal 
        reference lines.
    class_column : str, optional
        Column name containing class labels, by default 'group'.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    """
    std_values = df['std_MFBM'].apply(lambda x: x[std_band]).values
    classes = df[class_column].values
    x = np.arange(len(std_values))

    plt.figure(figsize=(8, 4))
    unique_classes = np.unique(classes)
    markers = ['^', 'x', 'o']

    for i, c in enumerate(unique_classes):
        mask = classes == c
        plt.scatter(x[mask], std_values[mask], label=c, alpha=0.7,
                    marker=markers[i % len(markers)], s=40)

    if thresholds:
        colors = ['tab:blue', 'tab:green', 'tab:green']
        linestyles = ['-.', '--', ':']
        for i, (name, t) in enumerate(thresholds.items()):
            plt.axhline(y=t, linestyle=linestyles[i % len(linestyles)],
                        color=colors[i % len(colors)], linewidth=1.9,
                        alpha=0.6, label=f'{name} threshold')

    plt.xlabel("Instance index (sorted)")
    plt.ylabel(f"Std MFBM (band {std_band + 1})")
    plt.title(f"Distribution of Std MFBM (Band {std_band + 1}) across all speakers by class")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()