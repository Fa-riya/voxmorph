"""Decoding, resampling and encoding audio."""

from __future__ import annotations

import io
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 22050  # plenty for speech (Nyquist 11 kHz) and keeps processing fast
MAX_SECONDS = 60.0


class AudioError(ValueError):
    pass


def load_audio(source: str | Path | bytes, sr: int = SAMPLE_RATE, max_seconds: float | None = MAX_SECONDS):
    """Read WAV/FLAC/OGG/MP3 (anything libsndfile supports) as mono float at ``sr``."""
    try:
        handle = io.BytesIO(source) if isinstance(source, bytes) else source
        data, file_sr = sf.read(handle, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several error types
        raise AudioError(f"Could not decode audio: {exc}") from exc
    if data.size == 0:
        raise AudioError("The audio file is empty.")

    x = data.mean(axis=1)
    if max_seconds is not None and len(x) / file_sr > max_seconds:
        raise AudioError(f"Audio is longer than {max_seconds:.0f} seconds.")
    if file_sr != sr:
        frac = Fraction(sr, file_sr).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)
    return x, sr


def to_wav_bytes(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_audio(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")
