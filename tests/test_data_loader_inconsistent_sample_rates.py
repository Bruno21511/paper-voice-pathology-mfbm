# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from src.data.data_loader import data_loader

def test_data_loader_inconsistent_sample_rates(tmp_path):
    dataset_name = "toy_corpus"
    dataset_path = tmp_path / dataset_name
    dataset_path.mkdir()
    class_dir = dataset_path / "control"
    class_dir.mkdir()

    sf.write(class_dir / "sample1.wav", np.random.randn(22050), 22050)
    sf.write(class_dir / "sample2.wav", np.random.randn(16000), 16000)

    csv_path = dataset_path / f"{dataset_name}.csv"
    csv_path.write_text("sample1.wav;30;M;control\nsample2.wav;40;F;control\n")

    df, fs_global = data_loader(dataset_name=dataset_name, data_root=str(tmp_path))

    assert fs_global is None
    assert len(df) == 2