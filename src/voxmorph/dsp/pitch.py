"""Fundamental frequency (F0) estimation with the YIN algorithm.

de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for speech
and music", JASA 2002. Implemented here in vectorised NumPy: the difference
function is computed for all frames at once using FFT cross-correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PitchTrack:
    times: np.ndarray  # seconds, one per frame
    f0: np.ndarray  # Hz, NaN where unvoiced
    voiced: np.ndarray  # bool

    @property
    def median_f0(self) -> float | None:
        v = self.f0[self.voiced]
        return float(np.median(v)) if v.size else None

    @property
    def voiced_ratio(self) -> float:
        return float(self.voiced.mean()) if self.voiced.size else 0.0


def yin(
    x: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 600.0,
    hop: int = 256,
    threshold: float = 0.15,
    silence_db: float = -40.0,
) -> PitchTrack:
    """Estimate the F0 contour of ``x``.

    Steps (numbered as in the paper):
      2. difference function   d(tau) = sum_j (x_j - x_{j+tau})^2
      3. cumulative-mean-normalised difference  d'(tau)
      4. absolute threshold: first dip of d' below ``threshold``
      5. parabolic interpolation around the dip for sub-sample accuracy
    Frames whose energy is ``silence_db`` below the loudest frame are unvoiced.
    """
    x = np.asarray(x, dtype=np.float64)
    tau_min = max(2, int(sr / fmax))
    tau_max = int(np.ceil(sr / fmin))
    frame_len = 1 << int(np.ceil(np.log2(2.5 * tau_max)))
    win = frame_len - tau_max  # integration window W

    padded = np.pad(x, (frame_len // 2, frame_len))
    n_frames = 1 + len(x) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[::hop][:n_frames]

    # --- Step 2: difference function via FFT -------------------------------
    n_fft = 1 << int(np.ceil(np.log2(2 * frame_len)))
    F = np.fft.rfft(frames, n=n_fft, axis=1)
    A = np.fft.rfft(frames[:, :win], n=n_fft, axis=1)
    cross = np.fft.irfft(np.conj(A) * F, n=n_fft, axis=1)[:, : tau_max + 1]

    sq = frames**2
    csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(sq, axis=1)], axis=1)
    energy_ref = csum[:, win][:, None]  # sum_{j<W} x_j^2
    taus = np.arange(tau_max + 1)
    energy_lag = csum[:, taus + win] - csum[:, taus]  # sum_{j<W} x_{j+tau}^2
    d = np.maximum(energy_ref + energy_lag - 2.0 * cross, 0.0)

    # --- Step 3: cumulative mean normalised difference ----------------------
    cmnd = np.ones_like(d)
    running = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(running, 1e-12)

    # --- Step 4: absolute threshold + walk down to the local minimum --------
    region = cmnd[:, tau_min : tau_max + 1]
    below = region < threshold
    has_dip = below.any(axis=1)
    idx = np.argmax(below, axis=1)
    rows = np.arange(n_frames)
    for _ in range(tau_max):
        nxt = np.minimum(idx + 1, region.shape[1] - 1)
        step = (region[rows, nxt] < region[rows, idx]) & (nxt != idx)
        if not step.any():
            break
        idx = np.where(step, nxt, idx)
    tau = idx + tau_min

    # Missing-fundamental guard: a shallow dip whose double period is much
    # deeper means we caught the 2nd harmonic's period, not the true one.
    double = np.minimum(2 * tau, tau_max)
    lo = np.clip(double - 2, 1, tau_max)
    window = np.stack([cmnd[rows, np.clip(lo + k, 1, tau_max)] for k in range(5)], axis=1)
    best_double = lo + np.argmin(window, axis=1)
    use_double = (2 * tau <= tau_max) & (cmnd[rows, tau] > 0.05) & (window.min(axis=1) < 0.5 * cmnd[rows, tau])
    tau = np.where(use_double, best_double, tau)

    # --- Step 5: parabolic interpolation ------------------------------------
    t0 = np.clip(tau - 1, 1, tau_max)
    t2 = np.clip(tau + 1, 1, tau_max)
    a, b, c = cmnd[rows, t0], cmnd[rows, tau], cmnd[rows, t2]
    denom = a - 2.0 * b + c
    safe = np.abs(denom) > 1e-12
    shift = np.zeros_like(denom)
    shift[safe] = 0.5 * (a[safe] - c[safe]) / denom[safe]
    tau_refined = tau + np.clip(shift, -1.0, 1.0)

    # Energy gate (the centre of each frame is the analysis instant).
    centre = frames[:, frame_len // 2 - hop : frame_len // 2 + hop]
    rms = np.sqrt(np.mean(centre**2, axis=1) + 1e-20)
    loud = (20.0 * np.log10(rms / (rms.max() + 1e-20)) > silence_db) & (rms > 1e-4)

    voiced = has_dip & loud
    f0 = np.where(voiced, sr / tau_refined, np.nan)
    f0 = fix_octave_errors(f0)
    times = np.arange(n_frames) * hop / sr
    return PitchTrack(times=times, f0=f0, voiced=voiced)


def fix_octave_errors(f0: np.ndarray, radius: int = 12, tolerance: float = 0.18) -> np.ndarray:
    """Correct isolated octave jumps, YIN's most common failure.

    When the second harmonic sits on a strong formant, YIN can lock onto half
    the true period for a few frames. Each voiced frame is compared with the
    median of its voiced neighbours; values close to 2x or 0.5x that median
    are folded back by an octave.
    """
    out = f0.copy()
    voiced = np.flatnonzero(~np.isnan(f0))
    if voiced.size < 3:
        return out
    log_f0 = np.log2(f0[voiced])
    for k, i in enumerate(voiced):
        lo, hi = np.searchsorted(voiced, [i - radius, i + radius + 1])
        ref = np.median(log_f0[lo:hi])
        diff = log_f0[k] - ref
        if abs(abs(diff) - 1.0) < tolerance:
            out[i] = f0[i] * 2.0 ** (-np.sign(diff))
    return out
