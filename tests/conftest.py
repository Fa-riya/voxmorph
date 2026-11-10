import numpy as np
import pytest

from voxmorph.dsp.spectral import cepstral_envelope, stft
from voxmorph.dsp.transforms import default_lifter
from voxmorph.synth import synthetic_voice

SR = 22050


@pytest.fixture
def sr():
    return SR


@pytest.fixture(scope="session")
def male_voice():
    return synthetic_voice(SR, "male").astype(np.float64)


@pytest.fixture(scope="session")
def female_voice():
    return synthetic_voice(SR, "female").astype(np.float64)


@pytest.fixture(scope="session")
def vowel_i():
    """One second of a sustained /i/ (F1 ~ 270 Hz, F2 ~ 2290 Hz: well separated)."""
    return synthetic_voice(SR, "male", vowels="i", vowel_ms=1000).astype(np.float64)


def tone(freq, sr=SR, seconds=1.0, harmonics=5):
    t = np.arange(int(sr * seconds)) / sr
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1)) * 0.3


def formant_peak(x, sr, lo, hi):
    """Frequency of the strongest spectral-envelope peak between lo and hi Hz."""
    X = stft(x, 1024, 256)
    power = np.abs(X) ** 2
    loud = power.sum(axis=0) > 0.1 * power.sum(axis=0).max()
    env = cepstral_envelope(np.abs(X[:, loud]), default_lifter(sr)).mean(axis=1)
    freqs = np.arange(len(env)) * sr / 1024
    band = (freqs >= lo) & (freqs <= hi)
    return freqs[band][np.argmax(env[band])]
