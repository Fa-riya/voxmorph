"""A small source-filter speech synthesiser.

Used for tests, documentation figures and the "try a sample" button, so the
project works out of the box without shipping anyone's real voice recording.
A Rosenberg glottal pulse train (with a natural pitch contour, vibrato and
jitter) is passed through a cascade of formant resonators, one vowel at a time.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

# Peterson & Barney-style formant frequencies (Hz) for an adult male voice.
VOWELS = {
    "a": (730, 1090, 2440, 3400),
    "e": (530, 1840, 2480, 3500),
    "i": (270, 2290, 3010, 3600),
    "o": (570, 840, 2410, 3400),
    "u": (300, 870, 2240, 3300),
}
BANDWIDTHS = (80, 100, 140, 180)


def _rosenberg_source(f0: np.ndarray, sr: int) -> np.ndarray:
    phase = np.cumsum(f0 / sr) % 1.0
    open_q, close_q = 0.4, 0.16
    pulse = np.where(
        phase < open_q,
        0.5 * (1 - np.cos(np.pi * phase / open_q)),
        np.where(phase < open_q + close_q, np.cos(0.5 * np.pi * (phase - open_q) / close_q), 0.0),
    )
    return np.diff(pulse, prepend=0.0)  # lip radiation ~ first difference


def _resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    return lfilter([1.0 - r], a, x)


def synthetic_voice(
    sr: int = 22050,
    speaker: str = "male",
    vowels: str = "aiu-eo",
    vowel_ms: float = 380.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise a short vowel sequence; ``-`` in ``vowels`` inserts a pause."""
    rng = np.random.default_rng(seed)
    base_f0, scale = (115.0, 1.0) if speaker == "male" else (210.0, 1.17)
    seg = int(sr * vowel_ms / 1000)
    fade = int(sr * 0.04)
    pieces = []
    for i, v in enumerate(vowels):
        if v == "-":
            pieces.append(np.zeros(seg // 2))
            continue
        t = np.arange(seg) / sr
        glide = 1.0 + 0.12 * np.sin(np.pi * (i / max(len(vowels) - 1, 1)))  # phrase-level intonation
        f0 = base_f0 * glide * (1 + 0.04 * np.sin(np.pi * t / t[-1]))
        f0 *= 1 + 0.01 * np.sin(2 * np.pi * 5.5 * t)  # vibrato
        f0 *= 1 + 0.004 * rng.standard_normal(seg)  # jitter
        src = _rosenberg_source(f0, sr) + 0.002 * rng.standard_normal(seg)
        y = src
        for freq, bw in zip(VOWELS[v], BANDWIDTHS, strict=True):
            y = _resonator(y, freq * scale, bw, sr)
        env = np.ones(seg)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        pieces.append(y * env)
    out = np.concatenate([np.zeros(sr // 10), *pieces, np.zeros(sr // 10)])
    return (0.8 * out / np.max(np.abs(out))).astype(np.float32)
