import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def save_metrics_csv(
    results_dict: Dict[str, Tuple[np.ndarray, float, float]],
    output_path: str
) -> None:
    """
    Save classification metrics and confusion matrix components to CSV.

    Parameters
    ----------
    results_dict : dict
        Mapping from class-pair name to a tuple (confusion_matrix, f1, 
        accuracy), e.g.:
        {
            'control_vs_physio': (cm_12, f1_12, acc_12),
            'control_vs_neuro':  (cm_13, f1_13, acc_13),
            'physio_vs_neuro':   (cm_23, f1_23, acc_23)
        }
    output_path : str
        Full path (including filename) to save the CSV.
    """
    rows = []
    for name, (cm, f1, acc) in results_dict.items():
        tn, fp = cm[0]
        fn, tp = cm[1]
        rows.append({
            'class_pair': name,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        })

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=';', index=False)
    logger.info(f"Metrics saved to: {output_path}")