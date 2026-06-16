import numpy as np
from typing import Tuple

def compute_threshold_metrics(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, float, float]:
    """
    Evaluate a threshold classifier on the std-band feature (X[:,1]).

    The threshold rule (>= threshold or <= threshold) is chosen 
    automatically based on which yields F1 >= 0.5.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, 2). Only the second column 
        (std-band feature) is used.
    y : np.ndarray
        Binary label vector, shape (n_samples,).
    threshold : float
        Decision threshold on the std-band feature.

    Returns
    -------
    cm : np.ndarray
        2x2 confusion matrix, [[tn, fp], [fn, tp]].
    f1 : float
        F1-score, rounded to 4 decimal places.
    accuracy : float
        Accuracy, rounded to 4 decimal places.
    """
    std_values = X[:, 1]

    def compute_metrics(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
        cm = np.array([[tn, fp], [fn, tp]])
        return cm, f1, accuracy

    y_pred = (std_values >= threshold).astype(int)
    cm, f1, acc = compute_metrics(y, y_pred)

    if f1 < 0.5:
        y_pred = (std_values <= threshold).astype(int)
        cm, f1, acc = compute_metrics(y, y_pred)

    return cm, round(float(f1), 4), round(float(acc), 4)