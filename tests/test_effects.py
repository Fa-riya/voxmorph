import numpy as np
import pytest

from voxmorph.dsp.pitch import yin
from voxmorph.effects import EFFECTS, EffectError, apply_effect


@pytest.mark.parametrize("effect_id", list(EFFECTS))
def test_every_effect_returns_clean_audio(sr, male_voice, effect_id):
    y = apply_effect(male_voice, sr, effect_id)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0
    assert len(y) > 0.8 * len(male_voice)


def test_gender_presets_move_pitch_towards_target(sr, male_voice, female_voice):
    assert yin(apply_effect(male_voice, sr, "female"), sr).median_f0 == pytest.approx(215, rel=0.05)
    assert yin(apply_effect(female_voice, sr, "male"), sr).median_f0 == pytest.approx(110, rel=0.05)
    assert yin(apply_effect(male_voice, sr, "child"), sr).median_f0 == pytest.approx(340, rel=0.05)


def test_robot_pitch_equals_hop_rate(sr, female_voice):
    y = apply_effect(female_voice, sr, "robot", {"robot_hz": 100, "metallic": 0})
    assert yin(y, sr).median_f0 == pytest.approx(100, rel=0.01)


def test_whisper_removes_voicing(sr, male_voice):
    assert yin(apply_effect(male_voice, sr, "whisper"), sr).voiced_ratio < 0.1


def test_echo_extends_the_tail(sr, male_voice):
    assert len(apply_effect(male_voice, sr, "echo")) > len(male_voice)


def test_custom_speed_changes_duration(sr, male_voice):
    y = apply_effect(male_voice, sr, "custom", {"speed": 2.0})
    assert len(y) == pytest.approx(len(male_voice) / 2, abs=2)


def test_parameter_validation(sr, male_voice):
    with pytest.raises(EffectError, match="between"):
        apply_effect(male_voice, sr, "custom", {"semitones": 40})
    with pytest.raises(EffectError, match="Unknown parameter"):
        apply_effect(male_voice, sr, "robot", {"volume": 1})
    with pytest.raises(EffectError, match="Unknown effect"):
        apply_effect(male_voice, sr, "underwater")
