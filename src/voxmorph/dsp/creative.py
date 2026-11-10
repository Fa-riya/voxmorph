"""Creative effects built from STFT phase manipulation, delay lines and waveshaping."""

from __future__ import annotations

import numpy as np

from .filters import highpass, lowpass, one_pole_lowpass
from .spectral import cepstral_envelope, istft, stft
from .transforms import default_lifter, resample_by


def robotize(x: np.ndarray, sr: int, robot_hz: float = 110.0, n_fft: int = 1024) -> np.ndarray:
    """Phase-vocoder robotisation (Zölzer, *DAFX*, ch. 7).

    Every STFT frame is reduced to its smooth spectral envelope (cepstral
    liftering removes the original harmonics) and given zero phase, turning it
    into a single symmetric pulse that carries only the formants.
    Resynthesising those pulses every ``sr / robot_hz`` samples gives a
    perfectly monotone voice whose pitch *is* the hop rate.
    """
    hop = max(16, int(round(sr / robot_hz)))
    X = stft(x, n_fft, hop)
    env = cepstral_envelope(np.abs(X), default_lifter(sr))
    # Linear phase (-1)^k centres each zero-phase pulse inside the window.
    centred = env * (-1.0) ** np.arange(X.shape[0])[:, None]
    return istft(centred, hop, length=len(x))


def whisperize(x: np.ndarray, sr: int, n_fft: int = 512, seed: int = 0) -> np.ndarray:
    """Replace the excitation with noise: keep |X|, randomise the phase.

    Short frames (~23 ms) keep the envelope but destroy the harmonic fine
    structure, which is exactly what whispering does to the voice source.
    """
    rng = np.random.default_rng(seed)
    hop = n_fft // 4
    X = stft(x, n_fft, hop)
    Y = np.abs(X) * np.exp(2j * np.pi * rng.random(X.shape))
    return highpass(istft(Y, hop, length=len(x)), sr, 250.0)


def echo(
    x: np.ndarray,
    sr: int,
    delay_ms: float = 280.0,
    feedback: float = 0.45,
    damping_hz: float = 3500.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Feedback delay with high-frequency damping in the loop.

    Each repeat is quieter by ``feedback`` and darker because it passes through
    a one-pole low-pass again, like sound bouncing off soft surfaces. The output
    is extended so the tail is not cut off.
    """
    d = int(sr * delay_ms / 1000)
    n_repeats = int(np.ceil(np.log(1e-3) / np.log(max(feedback, 1e-3))))
    out = np.zeros(len(x) + d * n_repeats)
    out[: len(x)] += x
    tap = x.copy()
    wet = np.zeros_like(out)
    for k in range(1, n_repeats + 1):
        tap = one_pole_lowpass(tap, sr, damping_hz) * feedback
        wet[k * d : k * d + len(x)] += tap
    return out * (1.0 - mix * 0.5) + wet * mix


def distort(x: np.ndarray, sr: int, drive: float = 8.0, oversample: int = 4) -> np.ndarray:
    """Asymmetric tube-style waveshaper, run at 4x sample rate to limit aliasing.

    The asymmetry (a DC bias before ``tanh``) adds even harmonics. Running the
    nonlinearity oversampled pushes most aliasing above the original Nyquist,
    where the polyphase decimation filter removes it.
    """
    up = resample_by(x, oversample)
    bias = 0.2
    shaped = np.tanh(drive * up + bias) - np.tanh(bias)
    y = resample_by(shaped, 1.0 / oversample)[: len(x)]
    y = highpass(y, sr, 120.0)
    return lowpass(y, sr, 5500.0)  # "speaker cabinet" roll-off


def ring_modulate(x: np.ndarray, sr: int, carrier_hz: float, depth: float = 1.0) -> np.ndarray:
    t = np.arange(len(x)) / sr
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    return x * (1.0 - depth + depth * carrier)


def tremolo(x: np.ndarray, sr: int, rate_hz: float = 6.0, depth: float = 0.3) -> np.ndarray:
    t = np.arange(len(x)) / sr
    return x * (1.0 - depth * 0.5 * (1.0 - np.cos(2 * np.pi * rate_hz * t)))
