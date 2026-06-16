# -*- coding: utf-8 -*-
import numpy as np
from src.features.mel_filterbank import mel_filterbank

def test_mel_filterbank_shape():
    filt = mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert filt.shape[0] == 20

def test_mel_filterbank_no_nans():
    filt = mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert not np.any(np.isnan(filt))

def test_mel_filterbank_normalized():
    filt = mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    sums = filt.sum(axis=1)
    assert np.allclose(sums[1:], 1.0, atol=1e-6)