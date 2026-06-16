# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from src.analysis.build_X_y import build_X_y

def test_build_X_y_shape():
    df = pd.DataFrame({
        'mean_MFBM': [np.arange(20), np.arange(20) + 1],
        'std_MFBM': [np.arange(20), np.arange(20) + 1],
        'group': ['control', 'physio']
    })
    X, y = build_X_y(df, label_map={'control': 0, 'physio': 1}, mean_band=7, std_band=2)
    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert list(y) == [0, 1]