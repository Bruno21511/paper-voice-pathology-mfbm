# -*- coding: utf-8 -*-
import numpy as np
import logging
from itertools import combinations
from typing import Dict, List

logger = logging.getLogger(__name__)


def _discriminative_bands_to_config_format(results: List[dict]) -> dict:
    """
    Convert find_most_discriminative_bands() output into the same 
    nested-dict format previously used in config.yaml's 
    discriminative_bands section, keyed by 'class1_vs_class2'.

    Parameters
    ----------
    results : list of dict
        Output of find_most_discriminative_bands.

    Returns
    -------
    dict
        Mapping like {'control_vs_physio': {'mean_band': ..., 'std_band': ...}, ...}
    """
    db = {}
    for r in results:
        key = r['pair'].replace(' vs ', '_vs_')
        db[key] = {
            'mean_band': r['best_band_mean'],
            'std_band': r['best_band_std']
        }
    return db
    
    
    

def find_most_discriminative_bands(
    mean_dict: Dict[str, np.ndarray],
    std_dict: Dict[str, np.ndarray]
) -> List[dict]:
    """
    Identify the most discriminative mel band for each pair of classes, 
    based on absolute differences in mean and std MFBM.

    Parameters
    ----------
    mean_dict : dict
        Mapping from class label to mean MFBM per band, as produced by 
        aggregate_band_statistics_per_class.
    std_dict : dict
        Mapping from class label to std MFBM per band, as produced by 
        aggregate_band_statistics_per_class.

    Returns
    -------
    list of dict
        One entry per class pair, each containing:
        'pair', 'best_band_mean', 'diff_mean', 'best_band_std', 'diff_std'.
        Band indices are 0-based.
    """
    results = []
    class_pairs = list(combinations(mean_dict.keys(), 2))

    for c1, c2 in class_pairs:
        diff_mean = np.abs(mean_dict[c1] - mean_dict[c2])
        diff_std = np.abs(std_dict[c1] - std_dict[c2])

        best_band_mean = np.argmax(diff_mean)
        best_band_std = np.argmax(diff_std)

        results.append({
            'pair': f"{c1} vs {c2}",
            'best_band_mean': int(best_band_mean),
            'diff_mean': float(diff_mean[best_band_mean]),
            'best_band_std': int(best_band_std),
            'diff_std': float(diff_std[best_band_std])
        })

    for r in results:
        logger.info(f"{r['pair']}: Mean -> band {r['best_band_mean']+1}, Std -> band {r['best_band_std']+1}")

    return _discriminative_bands_to_config_format(results)