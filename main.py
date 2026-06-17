# main.py
# ---------------------------------------------------------
# Run full analysis pipeline on processed dataset (parquet)
# Reproduces all paper results: thresholds, metrics, figures
# ---------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, does NOT open windows

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path

from src.data.import_dataframe import import_dataframe
from src.data.merge_pathology_classes import merge_pathology_classes
from src.analysis.compute_band_statistics import compute_band_statistics
from src.analysis.aggregate_band_statistics_per_class import aggregate_band_statistics_per_class
from src.analysis.find_most_discriminative_bands import find_most_discriminative_bands
from src.analysis.build_X_y import build_X_y
from src.analysis.threshold_f1_search import threshold_f1_search
from src.evaluation.compute_threshold_metrics import compute_threshold_metrics
from src.evaluation.format_confusion_matrix import format_confusion_matrix
from src.evaluation.save_metrics_csv import save_metrics_csv
from src.visualization.plot_meldefined_magnitudes_per_class import plot_meldefined_magnitudes_per_class
from src.visualization.plot_threshold_classification import plot_threshold_classification
from src.visualization.plot_and_save_confusion_matrices import plot_and_save_confusion_matrices

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_classification_task(df, label_map, mean_band, std_band, class_names, figures_dir, fig_prefix):
    """
    Run threshold search, evaluation, and plotting for one binary 
    classification task (one pair of classes).
    """
    groups = list(label_map.keys())
    df_subset = df[df['group'].isin(groups)]

    X, y = build_X_y(df_subset, label_map=label_map, mean_band=mean_band, std_band=std_band)

    t_opt = threshold_f1_search(X, y)
    cm, f1, acc = compute_threshold_metrics(X, y, t_opt)

    plot_threshold_classification(
        X, y, t_opt, acc,
        class_names=class_names,
        save_path=str(figures_dir / f"{fig_prefix}.png")
    )

    logger.info(format_confusion_matrix(cm, f1, acc, class_names=class_names))

    return cm, f1, acc, t_opt


def main():
    parser = argparse.ArgumentParser(description="Run voice pathology analysis pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(str(PROJECT_ROOT / args.config))

    figures_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    dataset_name = config["data"]["corpus_name"]

    # --- 1. Load processed dataset
    logger.info("Loading processed dataset...")
    df = import_dataframe(dataset_name=dataset_name, input_root=str(processed_dir))

    # --- 2. Compute band statistics
    logger.info("Computing band statistics...")
    df = compute_band_statistics(df)

    # --- 3. Merge pathology classes (per paper methodology)
    cm_cfg = config["class_merging"]
    df = merge_pathology_classes(df, classes_to_merge=cm_cfg["groups_to_merge"], merged_label=cm_cfg["merged_label"])

    # --- 4. Aggregate statistics per class and plot
    logger.info("Aggregating and plotting per-class statistics...")
    mean_dict, std_dict = aggregate_band_statistics_per_class(df)
    plot_meldefined_magnitudes_per_class(
        mean_dict, std_dict,
        save_path=str(figures_dir / "02_mean_std_per_class.png")
    )

    # --- 5. Identify discriminative bands (informational, logged only)
    find_most_discriminative_bands(mean_dict, std_dict)

    # --- 6. Run the three pairwise classification tasks
    db = config["discriminative_bands"]
    logger.info("Running classification tasks...")

    cm_12, f1_12, acc_12, t_opt_12 = run_classification_task(
        df, label_map={'control': 0, 'physio': 1},
        mean_band=db['control_vs_physio']['mean_band'],
        std_band=db['control_vs_physio']['std_band'],
        class_names=('Control', 'Physio'),
        figures_dir=figures_dir, fig_prefix="03_control_vs_physio"
    )

    cm_13, f1_13, acc_13, t_opt_13 = run_classification_task(
        df, label_map={'control': 0, 'neuro': 1},
        mean_band=db['control_vs_neuro']['mean_band'],
        std_band=db['control_vs_neuro']['std_band'],
        class_names=('Control', 'Neuro'),
        figures_dir=figures_dir, fig_prefix="03_control_vs_neuro"
    )

    cm_23, f1_23, acc_23, t_opt_23 = run_classification_task(
        df, label_map={'physio': 0, 'neuro': 1},
        mean_band=db['physio_vs_neuro']['mean_band'],
        std_band=db['physio_vs_neuro']['std_band'],
        class_names=('Physio', 'Neuro'),
        figures_dir=figures_dir, fig_prefix="03_physio_vs_neuro"
    )

    # --- 7. Save metrics and confusion matrices
    logger.info("Saving metrics and confusion matrices...")
    results = {
        'control_vs_physio': (cm_12, f1_12, acc_12),
        'control_vs_neuro':  (cm_13, f1_13, acc_13),
        'physio_vs_neuro':   (cm_23, f1_23, acc_23)
    }
    class_names_dict = {
        'control_vs_physio': ('Control', 'Physio'),
        'control_vs_neuro':  ('Control', 'Neuro'),
        'physio_vs_neuro':   ('Physio', 'Neuro')
    }

    save_metrics_csv(results, output_path=str(metrics_dir / "metrics.csv"))
    plot_and_save_confusion_matrices(results, class_names_dict, output_dir=str(figures_dir))

    logger.info("Done.")


if __name__ == "__main__":
    main()