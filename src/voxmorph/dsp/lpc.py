"""Linear prediction and McAdams-coefficient speaker anonymisation.

The McAdams transform is the signal-processing baseline of the VoicePrivacy
Challenge (Patino et al., Interspeech 2021). Each frame is modelled as an
all-pole vocal-tract filter driven by a residual. The angles of the filter's
complex poles (i.e. the formant frequencies) are warped as phi -> phi**alpha,
which changes the perceived speaker while leaving the residual, and therefore
the linguistic content and prosody, untouched.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .spectral import hann


def autocorrelation(frame: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(frame)
    spec = np.fft.rfft(frame, n=2 * n)
    return np.fft.irfft(np.abs(spec) ** 2)[: max_lag + 1]


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Solve the Yule-Walker equations for LPC coefficients in O(order^2).

    Returns ``a`` with ``a[0] = 1`` (so the inverse filter is A(z) = sum a_k z^-k)
    and the final prediction error power.
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    if err <= 0:
        return a, 0.0
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])) / err
        a[1:i] = a[1:i] + k * a[i - 1 : 0 : -1]
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0:
            break
    return a, err


def lpc(frame: np.ndarray, order: int) -> np.ndarray:
    r = autocorrelation(frame, order)
    r[0] *= 1.0 + 1e-9  # white-noise correction for numerical stability
    return levinson_durbin(r, order)[0]


def warp_poles(a: np.ndarray, alpha: float) -> np.ndarray:
    """Apply phi -> phi**alpha to the angle of every complex pole of 1/A(z)."""
    poles = np.roots(a)
    complex_mask = np.abs(poles.imag) > 1e-8
    angles = np.angle(poles[complex_mask])
    new_angles = np.sign(angles) * np.abs(angles) ** alpha
    poles = poles.astype(np.complex128)
    poles[complex_mask] = np.abs(poles[complex_mask]) * np.exp(1j * new_angles)
    return np.real(np.poly(poles))


def mcadams_anonymize(
    x: np.ndarray,
    sr: int,
    alpha: float = 0.8,
    order: int = 20,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Frame-wise LPC analysis, pole-angle warping and resynthesis (50% overlap-add)."""
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    win = hann(n)
    out = np.zeros(len(x) + n)
    wsum = np.zeros(len(x) + n)
    padded = np.pad(x, (0, n))

    for start in range(0, len(x), hop):
        frame = padded[start : start + n] * win
        if not np.any(frame):
            continue
        a = lpc(frame, order)
        residual = lfilter(a, [1.0], frame)
        a_new = warp_poles(a, alpha)
        out[start : start + n] += lfilter([1.0], a_new, residual) * win
        wsum[start : start + n] += win**2

    out /= np.where(wsum > 1e-3, wsum, 1.0)
    return out[: len(x)]
