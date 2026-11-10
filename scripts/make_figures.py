"""Regenerate the figures in docs/images from the library itself.

python scripts/make_figures.py                 # synthetic voice
python scripts/make_figures.py my_voice.wav    # your own recording
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from voxmorph import apply_effect, load_audio
from voxmorph.dsp import lpc
from voxmorph.dsp.pitch import yin
from voxmorph.dsp.spectral import cepstral_envelope, stft
from voxmorph.dsp.transforms import default_lifter, pitch_formant_shift, pitch_shift
from voxmorph.synth import synthetic_voice

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
SR = 22050
INK, MUTED, ACCENT = "#1C1F3E", "#5E6383", "#5A3FD0"
CMAP = LinearSegmentedColormap.from_list("voxmorph", ["#141334", "#3A1F78", "#B02E78", "#F0854A", "#FCECAA"], N=256)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#D9DBE6",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "figure.facecolor": "white",
    }
)


def spectrogram_panel(ax, x, title):
    n_fft, hop = 2048, 256
    power = np.abs(stft(x, n_fft, hop)) ** 2
    db = 10 * np.log10(power + 1e-12)
    db = np.clip(db - db.max(), -80, 0)
    freqs = np.arange(power.shape[0]) * SR / n_fft
    times = np.arange(power.shape[1]) * hop / SR
    ax.pcolormesh(times, np.maximum(freqs, 1), db, cmap=CMAP, shading="auto", rasterized=True)
    track = yin(x, SR, hop=hop)
    ax.plot(track.times, track.f0, color="#9EF0FF", lw=1.6)
    ax.set_yscale("log")
    ax.set_ylim(60, 8000)
    ax.set_yticks([100, 250, 500, 1000, 2000, 4000, 8000], ["100", "250", "500", "1k", "2k", "4k", "8k"])
    ax.minorticks_off()
    median = track.median_f0 if track.voiced_ratio >= 0.15 else None
    ax.set_title(f"{title}   (median F0 {median:.0f} Hz)" if median else f"{title}   (unvoiced)", loc="left")


def effects_grid(x):
    panels = [
        ("Original", x),
        ("Female", apply_effect(x, SR, "female")),
        ("Male", apply_effect(x, SR, "male")),
        ("Child", apply_effect(x, SR, "child")),
        ("Robot (110 Hz)", apply_effect(x, SR, "robot", {"metallic": 0})),
        ("Whisper", apply_effect(x, SR, "whisper")),
        ("Anonymize (McAdams 0.8)", apply_effect(x, SR, "anonymize")),
        ("Cartoon (naive +9 st)", apply_effect(x, SR, "cartoon")),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 10.5), sharey=True)
    for ax, (title, sig) in zip(axes.flat, panels, strict=True):
        spectrogram_panel(ax, np.asarray(sig, dtype=float), title)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(OUT / "effects_spectrograms.png", dpi=130)
    plt.close(fig)


def mean_envelope(x):
    X = stft(x, 1024, 256)
    power = (np.abs(X) ** 2).sum(axis=0)
    loud = power > 0.1 * power.max()
    env = cepstral_envelope(np.abs(X[:, loud]), default_lifter(SR)).mean(axis=1)
    return np.arange(len(env)) * SR / 1024, 20 * np.log10(env / env.max())


def formant_figure():
    vowel = synthetic_voice(SR, "male", vowels="i", vowel_ms=1000).astype(float)
    curves = [
        ("Original /i/", vowel, INK, "-"),
        ("Naive pitch shift +5 st", pitch_shift(vowel, 5), "#B02E78", "--"),
        ("Formant-preserving +5 st", pitch_formant_shift(vowel, SR, 5, 1.0), ACCENT, "-"),
        ("Formant warp x1.25, pitch kept", pitch_formant_shift(vowel, SR, 0, 1.25), "#E98A3F", "-."),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, sig, color, style in curves:
        f, env = mean_envelope(sig)
        ax.plot(f, env, label=label, color=color, ls=style, lw=2)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-45, 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spectral envelope (dB)")
    ax.set_title("Where the formants go: naive vs source-filter pitch shifting", loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "formant_shift.png", dpi=130)
    plt.close(fig)


def mcadams_figure():
    vowel = synthetic_voice(SR, "male", vowels="e", vowel_ms=600).astype(float)
    mid = len(vowel) // 2
    n = int(0.02 * SR)
    frame = vowel[mid : mid + n] * np.hanning(n)
    a = lpc.lpc(frame, 20)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="#D9DBE6", lw=1)
    for alpha, color, marker, label in [(1.0, INK, "x", "Original poles"), (0.8, ACCENT, "o", "Warped, alpha = 0.8")]:
        poles = np.roots(lpc.warp_poles(a, alpha) if alpha != 1 else a)
        face = "none" if marker == "o" else None
        ax.scatter(poles.real, poles.imag, color=color, marker=marker, s=46, label=label, facecolors=face)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("McAdams warp of LPC poles (z-plane)", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "mcadams_poles.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        x, _ = load_audio(sys.argv[1], SR, max_seconds=None)
    else:
        x = synthetic_voice(SR, "male")
    effects_grid(np.asarray(x, dtype=float))
    formant_figure()
    mcadams_figure()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
