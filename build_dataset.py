# build_dataset.py
# ---------------------------------------------------------
# Build processed dataset from raw audio corpus
# ---------------------------------------------------------

import argparse
import yaml
import logging
from pathlib import Path

from src.data.data_loader import data_loader
from src.features.mel_filterbank import mel_filterbank
from src.features.get_MFBM import get_MFBM
from src.data.export_dataframe import export_dataframe

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Config loader
# ---------------------------------------------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# MFBM extraction pipeline
# ---------------------------------------------------------
def extract_mfbm(df, fs, config):

    features_cfg = config["features"]

    n_fft = features_cfg["n_fft"]
    fmax = features_cfg["fmax"]
    mel_bands = features_cfg["n_filters"]
    overlap = features_cfg["overlap"]

    ms = round(fs / 1000)
    frame_size = int(features_cfg["frame_size_ms"] * ms)
    hop_size = int(features_cfg["hop_size_ms"] * ms)

    # Build filterbank
    mel_filters = mel_filterbank(
        fs,
        mel_bands,
        fmax,
        overlap,
        n_fft,
        print_filters=False,
        save_path=str(
            PROJECT_ROOT /
            config["results"]["figures_dir"] /
            "01_mel_filterbank.png"            
        )
    )

    mfbm_list = []

    for signal in df["signal"]:
        mfbm = get_MFBM(signal, frame_size, hop_size, n_fft, mel_filters)
        mfbm_list.append(mfbm)

    df["MFBM"] = mfbm_list

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Build dataset from raw audio corpus using MFBM features"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--corpora-root",
        type=str,
        default=None,
        help="Override corpora root directory (optional)"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset_name = config["data"]["corpus_name"]

    corpora_root = args.corpora_root
    if corpora_root is None:
        corpora_root = str(Path(__file__).resolve().parents[0] / "../corpora")

    logger.info(f"Loading dataset: {dataset_name}")

    # Load corpus
    df, fs = data_loader(
        dataset_name=dataset_name,
        data_root=corpora_root
    )

    logger.info("Extracting MFBM features...")

    # Extract features
    df = extract_mfbm(df, fs, config)

    # Export processed dataset
    output_root = config["data"].get("processed_dir", "data/processed")

    logger.info("Exporting dataset...")

    export_dataframe(
        df,
        dataset_name=dataset_name,
        output_root=output_root,        
        expand_mfbm=True,
        drop_columns=['signal', 'MFBM', 'path', 'fs']
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()