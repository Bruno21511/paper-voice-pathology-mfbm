# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessing import preprocessing


def test_preprocessing_peak_normalization():

    df = pd.DataFrame({
        "signal": [
            np.array([0.5, -1.0, 0.2])
        ]
    })

    df_out = preprocessing(
        df,
        normalize="peak",
        dc_remove=False,
        trim_signal=False,
        equal_duration=False
    )

    signal = df_out["signal"].iloc[0]

    assert np.isclose(np.max(np.abs(signal)), 1.0)


def test_preprocessing_dc_removal():

    df = pd.DataFrame({
        "signal": [
            np.array([2.0, 3.0, 4.0])
        ]
    })

    df_out = preprocessing(
        df,
        normalize=None,
        dc_remove=True,
        trim_signal=False,
        equal_duration=False
    )

    signal = df_out["signal"].iloc[0]

    assert np.isclose(np.mean(signal), 0.0)

   
    
def test_preprocessing_empty_signal_handling():
    
    df = pd.DataFrame({"signal": [np.array([])], "fs": [16000]})
    
    
    with pytest.raises(ValueError) as exc_info:
        preprocessing(df, dc_remove=True, trim_signal=True, normalize="peak")
        
    assert "DataFrame has empty audio signals" in str(exc_info.value)