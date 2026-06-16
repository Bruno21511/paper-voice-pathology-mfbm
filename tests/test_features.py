# -*- coding: utf-8 -*-
import numpy as np
from src.features.mel_filterbank import mel_filterbank
from src.features.get_MFBM import obter_FFT, get_MFBM

def test_obter_FFT_shape():
    sinal = np.random.randn(22050)
    spec = obter_FFT(sinal, tamanho=661, passo=220, n_fft=2048)
    assert spec.shape[0] == 1024

def test_get_MFBM_shape():
    fs = 22050
    sinal = np.random.randn(fs)
    tamanho = 30 * fs // 1000
    passo = 10 * fs // 1000
    n_fft = 2048
    filt = mel_filterbank(fs, 20, 4000, 0.5, n_fft)
    mfbm = get_MFBM(sinal, tamanho, passo, n_fft, filt)
    assert mfbm.shape[0] == 20
    assert not np.any(np.isnan(mfbm))