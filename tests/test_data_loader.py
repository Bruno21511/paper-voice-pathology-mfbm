# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
from pathlib import Path
from src.data.data_loader import data_loader

def test_data_loader_basic(tmp_path):
    # --- construir um corpus sintético mínimo
    dataset_name = "toy_corpus"
    dataset_path = tmp_path / dataset_name
    dataset_path.mkdir()

    # criar pasta de classe e um ficheiro de áudio sintético
    class_dir = dataset_path / "control"
    class_dir.mkdir()
    fs = 22050
    signal = np.random.randn(fs)  # 1 segundo de ruído
    sf.write(class_dir / "sample1.wav", signal, fs)

    # criar o CSV de metadados (sem header, ; como separador)
    csv_path = dataset_path / f"{dataset_name}.csv"
    csv_path.write_text("sample1.wav;30;M;control\n")

    # --- chamar a função
    df, fs_global = data_loader(dataset_name=dataset_name, data_root=str(tmp_path))

    # --- verificações
    assert len(df) == 1
    assert df['group'].iloc[0] == 'control'
    assert fs_global == fs
    assert df['signal'].iloc[0].shape[0] == fs
    assert np.max(np.abs(df['signal'].iloc[0])) <= 1.0  # normalizado