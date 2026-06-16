# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.analysis.compute_band_statistics import compute_band_statistics
from src.data.merge_pathology_classes import merge_pathology_classes

def test_compute_band_statistics_shape():
    df = pd.DataFrame({'MFBM': [np.random.rand(20, 50)]})
    df_out = compute_band_statistics(df)
    assert df_out['mean_MFBM'].iloc[0].shape == (20,)
    assert df_out['std_MFBM'].iloc[0].shape == (20,)

def test_merge_pathology_classes():
    df = pd.DataFrame({'group': ['control', 'edema', 'nodulo']})
    df_out = merge_pathology_classes(df, ['edema', 'nodulo'], 'physio')
    assert set(df_out['group']) == {'control', 'physio'}
    assert df_out['class'].nunique() == 2