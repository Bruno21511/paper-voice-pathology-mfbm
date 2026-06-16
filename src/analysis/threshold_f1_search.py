import numpy as np
from typing import Tuple

def threshold_f1_search(
    X: np.ndarray, 
    y: np.ndarray, 
    step: float = 0.1, 
    tol: float = 1e-3
) -> float:
    """
    Find the decision threshold on the std-band feature that maximizes 
    F1-score, using the midpoint of the best F1 plateau.

    The threshold rule (>= t or <= t) is chosen automatically per 
    threshold value, whichever yields the higher F1.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, 2). Only the second column 
        (std-band feature, index 1) is used.
    y : np.ndarray
        Binary label vector, shape (n_samples,). Label 1 is treated as 
        the positive class.
    step : float, optional
        Step size for the threshold search grid, by default 0.1.

    Returns
    -------
    float
        Optimal threshold, rounded to 2 decimal places, taken as the 
        midpoint of the plateau of thresholds achieving maximum F1 
        (within a tolerance of 1e-3).
    """
    std_values = X[:, 1]
    t_min = np.min(std_values)
    t_max = np.max(std_values)
    thresholds = np.arange(t_min, t_max, step)

    def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    f1_values = []
    for t in thresholds:
        y_pred_1 = (std_values >= t).astype(int)
        f1_1 = compute_f1(y, y_pred_1)
        y_pred_2 = (std_values <= t).astype(int)
        f1_2 = compute_f1(y, y_pred_2)
        f1_values.append(max(f1_1, f1_2))

    f1_values = np.array(f1_values)
    best_f1 = np.max(f1_values)

    tol = 1e-3
    best_idxs = np.where(np.abs(f1_values - best_f1) < tol)[0]
    best_thresholds = thresholds[best_idxs]
    t_opt = (best_thresholds[0] + best_thresholds[-1]) / 2

    return round(float(t_opt), 2)