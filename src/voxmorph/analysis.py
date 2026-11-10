"""Signal analysis for visualisation: log spectrogram, pitch contour, summary stats."""

from __future__ import annotations

import base64

import numpy as np

from .dsp.pitch import yin
from .dsp.spectral import stft

MIN_FREQ = 60.0
MAX_FREQ = 8000.0
ROWS = 180
MAX_COLS = 600
DYNAMIC_RANGE_DB = 80.0


def _pool(a: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Average-pool ``a`` down to ``n`` bins along ``axis`` (no-op if already smaller)."""
    size = a.shape[axis]
    if size <= n:
        return a
    edges = np.linspace(0, size, n + 1).astype(int)
    return np.add.reduceat(a, edges[:-1], axis=axis) / np.diff(edges).reshape(
        [-1 if i == axis else 1 for i in range(a.ndim)]
    )


def _log_frequency_rows(power: np.ndarray, sr: int, n_fft: int) -> tuple[np.ndarray, float]:
    """Resample linear STFT bins onto a log-frequency axis from MIN_FREQ to MAX_FREQ.

    Each output row averages the power of the bins inside its band (integrated
    via a cumulative sum with fractional edges), so wide high-frequency bands
    are not aliased and narrow low-frequency bands are interpolated smoothly.
    """
    top = min(MAX_FREQ, sr / 2)
    edges_hz = np.geomspace(MIN_FREQ, top, ROWS + 1)
    edges_bin = edges_hz * n_fft / sr  # fractional bin index of every band edge
    cum = np.concatenate([np.zeros((1, power.shape[1])), np.cumsum(power, axis=0)])
    idx = np.arange(cum.shape[0])

    def integral(b):  # integral of power from bin -0.5 up to fractional position b
        return np.stack([np.interp(b + 0.5, idx, col) for col in cum.T], axis=1)

    lo, hi = integral(edges_bin[:-1]), integral(edges_bin[1:])
    return (hi - lo) / np.diff(edges_bin)[:, None], top


def analyze(x: np.ndarray, sr: int) -> dict:
    """Return a compact, JSON-friendly description of ``x``.

    The spectrogram uses a log-frequency axis (row 0 = MIN_FREQ) and is encoded
    as base64 uint8, which is roughly 20x smaller than sending floats.
    """
    n_fft, hop = 2048, 256  # long window: resolves individual harmonics at low frequencies
    power = np.abs(stft(x, n_fft, hop)) ** 2
    power, top = _log_frequency_rows(_pool(power, MAX_COLS, 1), sr, n_fft)
    db = 10 * np.log10(power + 1e-12)
    db = np.clip(db - db.max() + DYNAMIC_RANGE_DB, 0, DYNAMIC_RANGE_DB)
    img = np.round(db / DYNAMIC_RANGE_DB * 255).astype(np.uint8)

    track = yin(x, sr, hop=hop)
    f0 = track.f0
    rms = np.sqrt(np.mean(np.square(x)) + 1e-12)

    return {
        "sample_rate": sr,
        "duration": len(x) / sr,
        "spectrogram": {
            "rows": int(img.shape[0]),
            "cols": int(img.shape[1]),
            "scale": "log",
            "min_freq": MIN_FREQ,
            "max_freq": float(top),
            "data": base64.b64encode(img.tobytes()).decode("ascii"),
        },
        "pitch": {
            "times": np.round(track.times, 4).tolist(),
            "f0": [None if np.isnan(v) else round(float(v), 2) for v in f0],
        },
        "summary": {
            # A handful of stray voiced frames (e.g. in a whisper) is not a pitch.
            "median_f0": None if track.voiced_ratio < 0.15 else round(track.median_f0, 1),
            "voiced_ratio": round(track.voiced_ratio, 3),
            "rms_db": round(20 * float(np.log10(rms)), 1),
            "peak_db": round(20 * float(np.log10(np.max(np.abs(x)) + 1e-12)), 1),
        },
    }
