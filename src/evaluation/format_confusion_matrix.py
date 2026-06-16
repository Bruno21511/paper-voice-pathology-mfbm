import numpy as np
from typing import Tuple

def format_confusion_matrix(
    cm: np.ndarray,
    f1: float,
    accuracy: float,
    class_names: Tuple[str, str] = ('Class 0', 'Class 1')
) -> str:
    """
    Format a confusion matrix and metrics as a human-readable string.

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix, [[tn, fp], [fn, tp]].
    f1 : float
        F1-score to display.
    accuracy : float
        Accuracy to display.
    class_names : tuple of str, optional
        Names for class 0 and class 1.

    Returns
    -------
    str
        Formatted summary, ready to print or log.
    """
    tn, fp = cm[0]
    fn, tp = cm[1]
    lines = [
        "Confusion Matrix:",
        f"{'':18s} Pred {class_names[0]:<8s}  Pred {class_names[1]:<8s}",
        f"True {class_names[0]:<8s} -> {tn:10d}   {fp:9d}",
        f"True {class_names[1]:<8s} -> {fn:10d}   {tp:9d}",
        f"\nF1-score: {f1:.4f}",
        f"Accuracy: {accuracy:.4f}"
    ]
    return "\n".join(lines)