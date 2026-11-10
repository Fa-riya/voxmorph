"""Level processing: compressor, soft clipping and normalisation."""

from __future__ import annotations

import numpy as np


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
    makeup_db: float = 0.0,
    block: int = 64,
) -> np.ndarray:
    """Feed-forward RMS compressor with separate attack and release.

    The level detector runs on short blocks (``block`` samples) so the
    attack/release recursion is cheap; the resulting gain curve is then
    interpolated back to sample rate so there is no zipper noise.
    """
    n_blocks = int(np.ceil(len(x) / block))
    padded = np.pad(x, (0, n_blocks * block - len(x)))
    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1) + 1e-12)
    level_db = 20 * np.log10(rms)

    over = np.maximum(level_db - threshold_db, 0.0)
    target_gain_db = -over * (1.0 - 1.0 / ratio)

    block_rate = sr / block
    att = np.exp(-1.0 / (attack_ms * 1e-3 * block_rate))
    rel = np.exp(-1.0 / (release_ms * 1e-3 * block_rate))
    gain_db = np.empty(n_blocks)
    g = 0.0
    for i, tgt in enumerate(target_gain_db):
        coeff = att if tgt < g else rel  # falling gain = attack phase
        g = coeff * g + (1.0 - coeff) * tgt
        gain_db[i] = g

    centres = (np.arange(n_blocks) + 0.5) * block
    gain = 10 ** ((np.interp(np.arange(len(x)), centres, gain_db) + makeup_db) / 20)
    return x * gain


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """tanh saturation, scaled so small signals pass at unity gain."""
    return np.tanh(drive * x) / np.tanh(drive)


def normalize(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Remove DC and scale to the given peak level in dBFS."""
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak < 1e-9:
        return x
    return x * (10 ** (peak_db / 20) / peak)
