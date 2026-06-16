# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def mel_filterbank(
    fs: int,
    n_filters: int,
    fmax: float,
    sobrep: float,
    n_fft: int,
    print_filters: bool = False,
    save_path: Optional[str] = None
) -> np.ndarray:

    """
    Create Mel-scale triangular filterbank with controllable overlap.

    This implementation preserves the original behavior:
    - Uniform spacing in Mel scale
    - Overlap controlled by 'sobrep'
    - Explicit triangular construction
    - Normalization per filter

    Parameters
    ----------
    fs : int
        Sampling frequency
    n_filters : int
        Number of filters
    fmax : float
        Maximum frequency (Hz)
    sobrep : float
        Overlap factor (e.g., 0.5 = 50%)
    n_fft : int
        FFT size
    print_filters : boolean
        self explanatory

    Returns
    -------
    filt_mel : ndarray
        Shape (n_filters, n_fft/2)
    """

    # Frequency axis
    freqs = np.arange(0, n_fft // 2) * (fs / n_fft)

    # --- Mel conversion functions
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10**(m / 2595) - 1)

    # --- Step 1: Uniform spacing in Mel
    mel_max = hz_to_mel(fmax)
    banda = mel_max / n_filters
    banda_inicial = 0

    low = np.zeros(n_filters)
    center = np.zeros(n_filters)
    high = np.zeros(n_filters)

    for i in range(n_filters):
        center[i] = banda_inicial + banda / 2
        low[i] = banda_inicial - banda * sobrep
        high[i] = banda_inicial + banda + banda * sobrep

        banda_inicial += banda

    # --- Convert edges to Hz
    low = mel_to_hz(low)
    center = mel_to_hz(center)
    high = mel_to_hz(high)

   # --- Step 2: Build filters (vectorized safely)

    freqs2D = freqs[None, :]  # (1, N)

    low = low[:, None]
    center = center[:, None]
    high = high[:, None]

    # Rising slope
    rise = (freqs2D - low) / (center - low + 1e-12)

    # Falling slope
    fall = 1 - (freqs2D - center) / (high - center + 1e-12)

    # Combine using conditions
    filt = np.where(
        (freqs2D >= low) & (freqs2D <= center),
        rise,
        0
    )

    filt += np.where(
        (freqs2D >= center) & (freqs2D <= high),
        fall,
        0
    )

    # Normalize
    filt /= (np.sum(filt, axis=1, keepdims=True) + 1e-12)

    # Keep original behavior
    filt[0, 0] = 0
    

    # Plot
    plt.figure(figsize=(12, 6))

    for i in range(filt.shape[0]):
        plt.plot(freqs, filt[i])

    plt.title("Mel Filterbank")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, fmax*1.1)
    plt.grid()
        
    # Save figure (optional)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Display figure (optional)
    if print_filters:
        plt.show()

    plt.close()

    return filt