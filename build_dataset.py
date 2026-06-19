# build_dataset.py
# ---------------------------------------------------------
# Build processed dataset from raw audio corpus
# ---------------------------------------------------------
import argparse
import yaml
import logging
from pathlib import Path

from src.data.data_loader import data_loader
from src.data.preprocessing import preprocessing
from src.data.export_dataframe import export_dataframe
from src.features.get_MFBM import get_MFBM

PROJECT_ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from raw audio corpus using MFBM features"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--corpora-root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(str(PROJECT_ROOT / args.config))
    dataset_name = config["data"]["corpus_name"]
    normalize = config["audio"]["normalize"]

    corpora_root = args.corpora_root
    if corpora_root is None:
        corpora_root = str(PROJECT_ROOT.parents[0] / "corpora")

    # --- Load corpus
    logger.info(f"Loading dataset: {dataset_name}")
    df = data_loader(dataset_name=dataset_name, data_root=corpora_root)

    # --- Preprocessing
    df = preprocessing(df, normalize=normalize)

    # --- MFBM extraction
    features_cfg = config["features"]
    logger.info("Extracting MFBM features...")
    df = get_MFBM(
        df,
        tamanho_in=features_cfg["frame_size_ms"],
        passo_in=features_cfg["hop_size_ms"],
        n_fft=features_cfg["n_fft"],
        n_filters=features_cfg["n_filters"],
        fmax=features_cfg["fmax"],
        sobrep=features_cfg["overlap"],
        edge_trim_frames=features_cfg["edge_trim_frames"],
        save_path=str(PROJECT_ROOT / config["results"]["figures_dir"] / "01_mel_filterbank.png"),
        print_filters=False
    )

    # --- Export
    output_root = PROJECT_ROOT / config["data"].get("processed_dir", "data/processed")
    logger.info("Exporting dataset...")
    export_dataframe(
        df,
        dataset_name=dataset_name,
        output_root=str(output_root),
        expand_mfbm=True,
        drop_columns=['signal', 'mfbm', 'path', 'fs']
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()