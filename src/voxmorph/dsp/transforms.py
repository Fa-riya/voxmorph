"""Time-scale, pitch and formant transformations.

The key idea is a source-filter split done in the STFT domain:

    speech  =  excitation (glottal pulses -> pitch)  x  envelope (vocal tract -> formants)

The cepstral envelope is divided out to obtain a "whitened" excitation, the
excitation is pitch-shifted on its own, and the original envelope (optionally
warped along frequency) is multiplied back in. This gives *independent* control
of pitch and formants: a plain resampling pitch shift moves both together,
which is what produces the "chipmunk" artefact.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from .spectral import (
    cepstral_envelope,
    fix_length,
    istft,
    phase_vocoder,
    stft,
    warp_frequency_axis,
)

N_FFT = 1024
HOP = 256


def time_stretch(x: np.ndarray, rate: float, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Change duration without changing pitch (``rate > 1`` = faster)."""
    if np.isclose(rate, 1.0):
        return x.copy()
    X = stft(x, n_fft, hop)
    Y = phase_vocoder(X, rate, hop)
    return istft(Y, hop, length=int(round(len(x) / rate)))


def resample_by(x: np.ndarray, ratio: float) -> np.ndarray:
    """Resample so the output has ``len(x) * ratio`` samples (polyphase FIR)."""
    frac = Fraction(ratio).limit_denominator(128)
    return resample_poly(x, frac.numerator, frac.denominator)


def pitch_shift(x: np.ndarray, semitones: float, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Classic pitch shift: phase-vocoder stretch by r, then resample by 1/r.

    Duration is preserved, but formants move together with the pitch.
    """
    if np.isclose(semitones, 0.0):
        return x.copy()
    r = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(x, 1.0 / r, n_fft, hop)
    return fix_length(resample_by(stretched, 1.0 / r), len(x))


def default_lifter(sr: int) -> int:
    """Cepstral cut-off of ~1.8 ms: below the pitch period of voices up to ~550 Hz."""
    return max(12, int(round(0.0018 * sr)))


def pitch_formant_shift(
    x: np.ndarray,
    sr: int,
    semitones: float = 0.0,
    formant: float = 1.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    """Shift pitch by ``semitones`` and formants by the ratio ``formant``, independently.

    ``formant=1.0`` keeps the speaker's vocal-tract resonances where they were
    (a formant-preserving pitch shift). ``formant>1`` simulates a shorter vocal
    tract (child/female-like), ``<1`` a longer one.
    """
    if np.isclose(semitones, 0.0) and np.isclose(formant, 1.0):
        return x.copy()

    X = stft(x, n_fft, hop)
    env = cepstral_envelope(np.abs(X), default_lifter(sr))

    # 1. Whiten: remove the vocal-tract envelope, keep the excitation + phase.
    excitation = istft(X / env, hop, length=len(x))

    # 2. Shift the pitch of the flat excitation only.
    excitation = pitch_shift(excitation, semitones, n_fft, hop)

    # 3. Re-apply the (optionally warped) envelope, frame by frame.
    E = stft(excitation, n_fft, hop)
    target_env = warp_frequency_axis(env, formant)
    n = min(E.shape[1], target_env.shape[1])
    Y = E[:, :n] * target_env[:, :n]
    return istft(Y, hop, length=len(x))
