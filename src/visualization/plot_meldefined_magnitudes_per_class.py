# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

def plot_meldefined_magnitudes_per_class(
    mean_dict: Dict[str, np.ndarray],
    std_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot mean and std of MFBM per class from precomputed aggregates.

    Parameters
    ----------
    mean_dict : dict
        Mapping from class label to mean MFBM per band, as produced by 
        aggregate_band_statistics_per_class.
    std_dict : dict
        Mapping from class label to std MFBM per band, as produced by 
        aggregate_band_statistics_per_class.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    """
    classes = list(mean_dict.keys())
    linestyles = {'control': '-', 'physio': '-.', 'neuro': '--'}
    banda_x = np.arange(len(mean_dict[classes[0]])) + 1

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title('Mean MFBM average per Class')
    for c in classes:
        plt.plot(banda_x, mean_dict[c], linestyle=linestyles.get(c, '-'), linewidth=2, label=c)
    plt.xlabel('Band')
    plt.ylabel('Mean magnitude')
    plt.grid(True)
    plt.xticks(range(len(banda_x) + 1))
    plt.legend()

    plt.subplot(122)
    plt.title('Std MFBM average per Class')
    for c in classes:
        plt.plot(banda_x, std_dict[c], linestyle=linestyles.get(c, '-'), linewidth=2, label=c)
    plt.xlabel('Band')
    plt.ylabel('Std magnitude')
    plt.grid(True)
    plt.xticks(range(len(banda_x) + 1))
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()