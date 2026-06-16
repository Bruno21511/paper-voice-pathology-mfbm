# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def build_X_y(
    df_subset: pd.DataFrame,
    label_map: Dict[str, int],
    mean_band: int,
    std_band: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and label vector from mean/std MFBM band values.

    Parameters
    ----------
    df_subset : pd.DataFrame
        DataFrame containing 'mean_MFBM', 'std_MFBM', and 'group' columns.
    label_map : dict
        Mapping from class label (in 'group') to integer class code.
    mean_band : int
        Zero-based index of the mel band to use from mean_MFBM.
    std_band : int
        Zero-based index of the mel band to use from std_MFBM.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_samples, 2): [mean_band_value, std_band_value].
    y : np.ndarray
        Label vector, shape (n_samples,).
    """
    X = []
    y = []
    for _, row in df_subset.iterrows():
        mean_vec = np.array(row['mean_MFBM'])
        std_vec = np.array(row['std_MFBM'])
        features = [
            mean_vec[mean_band],
            std_vec[std_band]
        ]
        X.append(features)
        y.append(label_map[row['group']])

    return np.array(X), np.array(y)