# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft


def obter_FFT(
    sinal_in: np.ndarray,
    tamanho: int,
    passo: int,
    n_fft: int
) -> np.ndarray:
    """
    Compute frame-wise FFT of a signal (STFT-like representation).

    Parameters
    ----------
    sinal_in : np.ndarray
        Input signal.
    tamanho : int
        Frame size (samples).
    passo : int
        Hop size (samples).
    n_fft : int
        FFT size.

    Returns
    -------
    np.ndarray
        STFT-like matrix, shape (n_freq_bins, n_frames), using only 
        positive frequencies (Nyquist excluded).
    """

    frames = []

    for i in range(0, len(sinal_in) - tamanho, passo):
        frame = sinal_in[i:i + tamanho]
        spec = fft(frame, n_fft)
        frames.append(spec)

    sinal = np.vstack(frames)

    # keep only positive frequencies (NO Nyquist)
    return sinal[:, :n_fft // 2].T



def get_MFBM(
    sinal: np.ndarray,
    tamanho: int,
    passo: int,
    n_fft: int,
    filt: np.ndarray
) -> np.ndarray:
    """
    Compute Mel Filter Bank Magnitudes

    Parameters
    ----------
    sinal : ndarray
        Input signal
    tamanho : int
        Frame size (samples)
    passo : int
        Hop size (samples)
    n_fft : int
        FFT size
    filt : ndarray
        Precomputed filterbank (n_filters x n_fft/2)

    Returns
    -------
    mfbm : ndarray
    Note: the first and last 3 frames are discarded to avoid potential initial and final instabillities
        Shape (n_filters, n_frames)
    """

    # --- 1. STFT
    X = obter_FFT(sinal, tamanho, passo, n_fft)

    # --- 2. Magnitude spectrum
    X = np.abs(X)

    # --- 3. Apply filterbank
    mfbm = np.dot(filt, X[:, 3:-3])

    return mfbm