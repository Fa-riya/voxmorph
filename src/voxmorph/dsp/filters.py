"""Filter design: RBJ biquads written out by hand, plus Butterworth helpers.

Biquad formulas follow Robert Bristow-Johnson's "Audio EQ Cookbook".
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt


def _normalise(b: list[float], a: list[float]) -> tuple[np.ndarray, np.ndarray]:
    b, a = np.asarray(b, dtype=float), np.asarray(a, dtype=float)
    return b / a[0], a / a[0]


def peaking_eq(f0: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Bell filter: boost/cut ``gain_db`` around ``f0`` with bandwidth set by ``q``."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cos = np.cos(w0)
    b = [1 + alpha * A, -2 * cos, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos, 1 - alpha / A]
    return _normalise(b, a)


def shelf(f0: float, gain_db: float, sr: int, kind: str = "low", slope: float = 1.0):
    """Low or high shelving filter."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cos, sin = np.cos(w0), np.sin(w0)
    alpha = sin / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    sq = 2 * np.sqrt(A) * alpha
    if kind == "low":
        b = [A * ((A + 1) - (A - 1) * cos + sq), 2 * A * ((A - 1) - (A + 1) * cos), A * ((A + 1) - (A - 1) * cos - sq)]
        a = [(A + 1) + (A - 1) * cos + sq, -2 * ((A - 1) + (A + 1) * cos), (A + 1) + (A - 1) * cos - sq]
    else:
        b = [A * ((A + 1) + (A - 1) * cos + sq), -2 * A * ((A - 1) + (A + 1) * cos), A * ((A + 1) + (A - 1) * cos - sq)]
        a = [(A + 1) - (A - 1) * cos + sq, 2 * ((A - 1) - (A + 1) * cos), (A + 1) - (A - 1) * cos - sq]
    return _normalise(b, a)


def apply_biquad(x: np.ndarray, ba: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return lfilter(ba[0], ba[1], x)


def bandpass(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (forward-backward, so order is doubled)."""
    sos = butter(order, [low, min(high, 0.45 * sr)], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def highpass(x: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    sos = butter(order, cutoff, btype="high", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, min(cutoff, 0.45 * sr), btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def one_pole_lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """y[n] = (1-p) x[n] + p y[n-1]: the cheapest possible damping filter."""
    p = np.exp(-2 * np.pi * cutoff / sr)
    return lfilter([1 - p], [1, -p], x)
