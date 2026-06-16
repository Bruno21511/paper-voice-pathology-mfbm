import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def plot_and_save_confusion_matrices(
    results_dict: Dict[str, Tuple[np.ndarray, float, float]],
    class_names_dict: Dict[str, Tuple[str, str]],
    output_dir: str
) -> None:
    """
    Plot and save confusion matrices as heatmap images.

    Parameters
    ----------
    results_dict : dict
        Mapping from class-pair name to a tuple (confusion_matrix, f1, 
        accuracy).
    class_names_dict : dict
        Mapping from class-pair name to a tuple of display names, e.g.:
        {'control_vs_physio': ('Control', 'Physio'), ...}
    output_dir : str
        Directory to save the figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (cm, f1, acc) in results_dict.items():
        class_names = class_names_dict[name]
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{class_names[0]} vs {class_names[1]}")

        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{value}", ha="center", va="center", color=color)

        fig.colorbar(im, ax=ax)

        filename = f"04_cm_{name}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        logger.info(f"Saved: {filepath}")