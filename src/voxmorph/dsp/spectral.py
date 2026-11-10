"""Short-time Fourier analysis/synthesis and the phase vocoder.

Everything here is written directly on top of NumPy so the maths stays visible:
framing, windowing, weighted overlap-add and phase-advance tracking.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def hann(n: int) -> np.ndarray:
    """Periodic Hann window (the right choice for STFT overlap-add)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256, center: bool = True) -> np.ndarray:
    """Short-time Fourier transform.

    Returns a complex matrix of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if center:
        x = np.pad(x, n_fft // 2)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop][:n_frames]
    return np.fft.rfft(frames * hann(n_fft), axis=1).T


def istft(
    X: np.ndarray,
    hop: int = 256,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT by weighted overlap-add (WOLA).

    The output is divided by the summed squared synthesis window, so any hop
    size reconstructs correctly, including hops that do not divide ``n_fft``.
    """
    n_fft = 2 * (X.shape[0] - 1)
    n_frames = X.shape[1]
    win = hann(n_fft)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1) * win

    out_len = n_fft + hop * (n_frames - 1)
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    win_sq = win**2
    for i in range(n_frames):
        start = i * hop
        y[start : start + n_fft] += frames[i]
        wsum[start : start + n_fft] += win_sq
    y /= np.where(wsum > 1e-3 * wsum.max(), wsum, 1.0)

    if center:
        y = y[n_fft // 2 :]
    if length is not None:
        y = fix_length(y, length)
    return y


def fix_length(x: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad ``x`` to exactly ``length`` samples."""
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x)))


def phase_vocoder(X: np.ndarray, rate: float, hop: int) -> np.ndarray:
    """Time-scale an STFT by ``rate`` (>1 = faster/shorter, <1 = slower/longer).

    Magnitudes are linearly interpolated between neighbouring frames, while
    each bin's phase is advanced by its *measured* instantaneous frequency so
    that partials stay continuous across the new frame grid.
    """
    n_bins, n_frames = X.shape
    n_fft = 2 * (n_bins - 1)
    steps = np.arange(0, n_frames, rate)

    expected_advance = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    Xp = np.pad(X, ((0, 0), (0, 2)))
    phase = np.angle(Xp[:, 0])
    out = np.empty((n_bins, len(steps)), dtype=np.complex128)

    for t, step in enumerate(steps):
        i = int(step)
        frac = step - i
        c0, c1 = Xp[:, i], Xp[:, i + 1]
        mag = (1.0 - frac) * np.abs(c0) + frac * np.abs(c1)
        out[:, t] = mag * np.exp(1j * phase)

        # Deviation from the bin-centre frequency, wrapped to [-pi, pi].
        dphi = np.angle(c1) - np.angle(c0) - expected_advance
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        phase = phase + expected_advance + dphi
    return out


def cepstral_envelope(mag: np.ndarray, n_lifter: int) -> np.ndarray:
    """Smooth spectral envelope of each frame via cepstral liftering.

    The real cepstrum separates the slowly varying vocal-tract response (low
    quefrency) from the harmonic excitation (high quefrency, at the pitch
    period). Keeping only the first ``n_lifter`` coefficients and transforming
    back gives the envelope, i.e. the formant structure.
    """
    n_fft = 2 * (mag.shape[0] - 1)
    cep = np.fft.irfft(np.log(mag + EPS), n=n_fft, axis=0)
    lifter = np.zeros(n_fft)
    lifter[:n_lifter] = 1.0
    lifter[n_fft - n_lifter + 1 :] = 1.0
    log_env = np.fft.rfft(cep * lifter[:, None], axis=0).real
    return np.exp(log_env)


def warp_frequency_axis(env: np.ndarray, factor: float) -> np.ndarray:
    """Stretch every column of ``env`` along frequency by ``factor``.

    ``factor > 1`` moves spectral features (formants) up, ``< 1`` moves them down.
    """
    n_bins = env.shape[0]
    src = np.arange(n_bins) / factor
    i0 = np.clip(np.floor(src).astype(int), 0, n_bins - 1)
    i1 = np.clip(i0 + 1, 0, n_bins - 1)
    frac = np.clip(src - i0, 0.0, 1.0)[:, None]
    warped = (1.0 - frac) * env[i0] + frac * env[i1]
    # Beyond the original Nyquist there is no information; fade it out.
    beyond = src > n_bins - 1
    warped[beyond] = env[-1] * 0.1
    return warped
