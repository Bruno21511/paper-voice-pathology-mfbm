# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

def plot_band_discriminability(
    mean_dict: Dict[str, np.ndarray],
    c1: str,
    c2: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot band-wise discriminability (absolute difference) between two 
    classes' mean MFBM.

    Parameters
    ----------
    mean_dict : dict
        Mapping from class label to mean MFBM per band, as produced by 
        aggregate_band_statistics_per_class.
    c1 : str
        First class name to compare.
    c2 : str
        Second class name to compare.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    """
    a = mean_dict[c1]
    b = mean_dict[c2]
    diff = np.abs(a - b)
    bandas = np.arange(len(diff)) + 1
    best = np.argmax(diff)

    plt.figure(figsize=(10, 5))
    plt.plot(bandas, diff, marker='o', label=f'Best band: {best + 1}')
    plt.title(f"{c1} vs {c2}")
    plt.xlabel("Band")
    plt.ylabel("Discriminability")
    plt.grid(True)
    plt.xticks(range(len(bandas) + 1))
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()