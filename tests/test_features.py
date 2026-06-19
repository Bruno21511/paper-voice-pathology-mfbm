# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from src.features.get_MFBM import _mel_filterbank, _obter_FFT, get_MFBM

def test_obter_FFT_shape():
    sinal = np.random.randn(22050)
    spec = _obter_FFT(sinal, tamanho=661, passo=220, n_fft=2048)
    assert spec.shape[0] == 1024

def test_get_MFBM_shape():
    df = pd.DataFrame({
        'signal': [np.random.randn(22050)],
        'fs': [22050]
    })
    df_out = get_MFBM(
        df,
        tamanho_in=30,
        passo_in=10,
        n_fft=2048,
        n_filters=20,
        fmax=4000,
        sobrep=0.5
    )
    assert df_out['mfbm'].iloc[0].shape[0] == 20
    assert not np.any(np.isnan(df_out['mfbm'].iloc[0]))
    
def test_mel_filterbank_shape():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert filt.shape[0] == 20

def test_mel_filterbank_no_nans():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert not np.any(np.isnan(filt))

def test_mel_filterbank_normalized():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    sums = filt.sum(axis=1)
    assert np.allclose(sums[1:], 1.0, atol=1e-6)