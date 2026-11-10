"""The effect catalogue: each preset is a small chain of DSP building blocks.

Every effect is described by metadata (shown in the UI and the API) and a
function ``fn(x, sr, **params) -> y``. Parameters are declared with ranges so
the API can validate them and the UI can render sliders automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from .dsp import creative, dynamics, filters, lpc, transforms
from .dsp.pitch import yin


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    min: float
    max: float
    step: float
    default: float
    unit: str = ""


@dataclass(frozen=True)
class Effect:
    id: str
    name: str
    description: str
    techniques: list[str]
    fn: Callable[..., np.ndarray] = field(repr=False)
    params: list[Param] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("fn")
        return d


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def semitones_to_target(x: np.ndarray, sr: int, target_hz: float, fallback: float) -> float:
    """How far to shift so the speaker's median F0 lands on ``target_hz``.

    This is what makes the gender/age presets *adaptive*: a deep voice and a
    high voice are both moved to the same target register instead of by a
    fixed amount.
    """
    median = yin(x, sr).median_f0
    if median is None:
        return fallback
    return float(np.clip(12.0 * np.log2(target_hz / median), -18.0, 18.0))


# --------------------------------------------------------------------------
# Effect implementations
# --------------------------------------------------------------------------


def _male(x, sr, target_f0=110.0, formant=0.86):
    shift = semitones_to_target(x, sr, target_f0, fallback=-4.0)
    y = transforms.pitch_formant_shift(x, sr, shift, formant)
    return filters.apply_biquad(y, filters.shelf(180.0, 3.0, sr, "low"))


def _female(x, sr, target_f0=215.0, formant=1.17):
    shift = semitones_to_target(x, sr, target_f0, fallback=4.0)
    y = transforms.pitch_formant_shift(x, sr, shift, formant)
    y = filters.apply_biquad(y, filters.shelf(200.0, -2.0, sr, "low"))
    return filters.apply_biquad(y, filters.peaking_eq(3200.0, 3.0, 1.0, sr))


def _child(x, sr, target_f0=340.0, formant=1.3):
    shift = semitones_to_target(x, sr, target_f0, fallback=7.0)
    y = transforms.pitch_formant_shift(x, sr, shift, formant)
    return filters.apply_biquad(y, filters.shelf(250.0, -4.0, sr, "low"))


def _cartoon(x, sr, semitones=9.0, speed=1.15):
    # Deliberately the *naive* pitch shift: formants move up with the pitch,
    # which is the squeaky "chipmunk" sound.
    y = transforms.pitch_shift(x, semitones)
    y = transforms.time_stretch(y, speed)
    return creative.tremolo(y, sr, rate_hz=7.0, depth=0.15)


def _robot(x, sr, robot_hz=110.0, metallic=0.3):
    y = creative.robotize(x, sr, robot_hz)
    return creative.ring_modulate(y, sr, carrier_hz=robot_hz * 0.5, depth=metallic)


def _whisper(x, sr):
    return creative.whisperize(x, sr)


def _echo(x, sr, delay_ms=280.0, feedback=0.45):
    return creative.echo(x, sr, delay_ms=delay_ms, feedback=feedback)


def _distortion(x, sr, drive=8.0):
    return creative.distort(x, sr, drive=drive)


def _telephone(x, sr):
    y = filters.bandpass(x, sr, 300.0, 3400.0)
    y = dynamics.compress(dynamics.normalize(y), sr, threshold_db=-24.0, ratio=6.0, makeup_db=6.0)
    return dynamics.soft_clip(y, drive=1.5)


def _anonymize(x, sr, alpha=0.8):
    return lpc.mcadams_anonymize(x, sr, alpha=alpha)


def _custom(x, sr, semitones=0.0, formant=1.0, speed=1.0):
    y = transforms.pitch_formant_shift(x, sr, semitones, formant)
    return transforms.time_stretch(y, speed)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

EFFECTS: dict[str, Effect] = {
    e.id: e
    for e in [
        Effect(
            "male",
            "Male",
            "Moves the speaker's median pitch to a lower target and lengthens the virtual vocal tract.",
            ["YIN pitch tracking", "Formant-preserving pitch shift", "Formant warping", "Low-shelf EQ"],
            _male,
            [
                Param("target_f0", "Target pitch", 80, 160, 1, 110, "Hz"),
                Param("formant", "Formant ratio", 0.7, 1.0, 0.01, 0.86, "x"),
            ],
        ),
        Effect(
            "female",
            "Female",
            "Raises the median pitch to a higher target, shortens the vocal tract and adds presence.",
            ["YIN pitch tracking", "Formant-preserving pitch shift", "Formant warping", "Peaking EQ"],
            _female,
            [
                Param("target_f0", "Target pitch", 160, 300, 1, 215, "Hz"),
                Param("formant", "Formant ratio", 1.0, 1.35, 0.01, 1.17, "x"),
            ],
        ),
        Effect(
            "child",
            "Child",
            "High pitch with strongly raised formants, modelling a much shorter vocal tract.",
            ["YIN pitch tracking", "Formant-preserving pitch shift", "Formant warping", "Low-shelf EQ"],
            _child,
            [
                Param("target_f0", "Target pitch", 250, 450, 1, 340, "Hz"),
                Param("formant", "Formant ratio", 1.1, 1.5, 0.01, 1.3, "x"),
            ],
        ),
        Effect(
            "cartoon",
            "Cartoon",
            "Classic resampling pitch shift (formants move with pitch), sped up with a light tremolo.",
            ["Phase vocoder", "Polyphase resampling", "Time stretching", "Amplitude modulation"],
            _cartoon,
            [
                Param("semitones", "Pitch shift", 4, 12, 1, 9, "st"),
                Param("speed", "Speed", 1.0, 1.5, 0.01, 1.15, "x"),
            ],
        ),
        Effect(
            "robot",
            "Robot",
            "Turns each frame's spectral envelope into a pulse repeated at a fixed rate: a perfectly monotone voice.",
            ["Cepstral envelope", "Zero-phase resynthesis", "Fixed-hop overlap-add", "Ring modulation"],
            _robot,
            [
                Param("robot_hz", "Robot pitch", 60, 240, 1, 110, "Hz"),
                Param("metallic", "Metallic", 0.0, 1.0, 0.05, 0.3),
            ],
        ),
        Effect(
            "whisper",
            "Whisper",
            "Keeps the spectral envelope but randomises phase, replacing the voiced source with noise.",
            ["STFT phase randomisation", "Butterworth high-pass"],
            _whisper,
        ),
        Effect(
            "echo",
            "Echo",
            "Feedback delay line whose repeats get quieter and darker, like reflections off soft walls.",
            ["Feedback delay line", "One-pole low-pass damping"],
            _echo,
            [
                Param("delay_ms", "Delay", 80, 600, 10, 280, "ms"),
                Param("feedback", "Feedback", 0.1, 0.8, 0.05, 0.45),
            ],
        ),
        Effect(
            "distortion",
            "Distortion",
            "Asymmetric tanh waveshaper run at 4x oversampling to keep aliasing down.",
            ["Nonlinear waveshaping", "4x oversampling", "Band limiting"],
            _distortion,
            [Param("drive", "Drive", 2, 20, 0.5, 8)],
        ),
        Effect(
            "telephone",
            "Telephone",
            "Narrowband 300-3400 Hz channel with heavy compression and mild saturation.",
            ["Butterworth band-pass", "Dynamic range compression", "Soft clipping"],
            _telephone,
        ),
        Effect(
            "anonymize",
            "Anonymize",
            "McAdams-coefficient anonymisation: warps LPC pole angles to change speaker identity, keeping the words.",
            ["Linear prediction (Levinson-Durbin)", "Pole-angle warping", "Overlap-add resynthesis"],
            _anonymize,
            [Param("alpha", "McAdams alpha", 0.6, 1.0, 0.01, 0.8)],
        ),
        Effect(
            "custom",
            "Custom",
            "Independent control of pitch, formants and speed: the three building blocks behind every preset.",
            ["Formant-preserving pitch shift", "Formant warping", "Phase vocoder time stretch"],
            _custom,
            [
                Param("semitones", "Pitch shift", -12, 12, 0.5, 0, "st"),
                Param("formant", "Formant ratio", 0.6, 1.6, 0.01, 1.0, "x"),
                Param("speed", "Speed", 0.5, 2.0, 0.05, 1.0, "x"),
            ],
        ),
    ]
}


class EffectError(ValueError):
    pass


def resolve_params(effect: Effect, params: dict[str, float] | None) -> dict[str, float]:
    """Fill in defaults and reject unknown or out-of-range parameters."""
    params = dict(params or {})
    known = {p.name: p for p in effect.params}
    unknown = set(params) - set(known)
    if unknown:
        raise EffectError(f"Unknown parameter(s) for '{effect.id}': {', '.join(sorted(unknown))}")
    resolved = {}
    for name, spec in known.items():
        value = float(params.get(name, spec.default))
        if not spec.min <= value <= spec.max:
            raise EffectError(f"'{name}' must be between {spec.min} and {spec.max}, got {value}")
        resolved[name] = value
    return resolved


def apply_effect(x: np.ndarray, sr: int, effect_id: str, params: dict[str, float] | None = None) -> np.ndarray:
    """Run an effect by id and return a normalised, finite float32 signal."""
    if effect_id not in EFFECTS:
        raise EffectError(f"Unknown effect '{effect_id}'. Choose from: {', '.join(EFFECTS)}")
    effect = EFFECTS[effect_id]
    y = effect.fn(np.asarray(x, dtype=np.float64), sr, **resolve_params(effect, params))
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return dynamics.normalize(y, peak_db=-1.0).astype(np.float32)
