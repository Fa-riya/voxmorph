import numpy as np
import pytest
from scipy.linalg import solve_toeplitz

from conftest import formant_peak, tone
from voxmorph.dsp import lpc
from voxmorph.dsp.pitch import yin
from voxmorph.dsp.spectral import istft, stft
from voxmorph.dsp.transforms import pitch_formant_shift, pitch_shift, time_stretch

# --- STFT -----------------------------------------------------------------


@pytest.mark.parametrize("n_fft,hop", [(1024, 256), (512, 128), (1024, 200), (2048, 512)])
def test_stft_istft_reconstructs_signal(n_fft, hop):
    x = np.random.default_rng(0).standard_normal(10_000)
    y = istft(stft(x, n_fft, hop), hop, length=len(x))
    assert np.allclose(x, y, atol=1e-9)


# --- Pitch tracking -------------------------------------------------------


@pytest.mark.parametrize("f0", [70, 110, 165, 220, 330, 500])
def test_yin_is_accurate_on_harmonic_tones(sr, f0):
    track = yin(tone(f0), sr)
    assert track.voiced_ratio > 0.9
    assert track.median_f0 == pytest.approx(f0, rel=0.005)


def test_yin_marks_noise_and_silence_unvoiced(sr):
    noise = np.random.default_rng(1).standard_normal(sr) * 0.1
    assert yin(noise, sr).voiced_ratio < 0.05
    assert yin(np.zeros(sr), sr).voiced_ratio == 0.0


def test_yin_on_synthetic_speech(sr, male_voice, female_voice):
    assert 100 < yin(male_voice, sr).median_f0 < 150
    assert 190 < yin(female_voice, sr).median_f0 < 270


# --- Time and pitch -------------------------------------------------------


@pytest.mark.parametrize("rate", [0.5, 0.8, 1.25, 2.0])
def test_time_stretch_changes_length_not_pitch(sr, rate):
    x = tone(200)
    y = time_stretch(x, rate)
    assert len(y) == round(len(x) / rate)
    assert yin(y, sr).median_f0 == pytest.approx(200, rel=0.01)


@pytest.mark.parametrize("semitones", [-12, -5, 3, 7, 12])
def test_pitch_shift_hits_target_and_keeps_length(sr, semitones):
    x = tone(200)
    y = pitch_shift(x, semitones)
    assert len(y) == len(x)
    assert yin(y, sr).median_f0 == pytest.approx(200 * 2 ** (semitones / 12), rel=0.01)


def test_formant_preserving_shift_keeps_vowel_formants(sr, vowel_i):
    f2_before = formant_peak(vowel_i, sr, 1800, 2700)
    naive = pitch_shift(vowel_i, 5)
    preserved = pitch_formant_shift(vowel_i, sr, semitones=5, formant=1.0)

    # Naive shifting drags F2 up by 2^(5/12) = 1.33x; the source-filter version keeps it put.
    assert formant_peak(naive, sr, 1800, 3600) == pytest.approx(f2_before * 2 ** (5 / 12), rel=0.05)
    assert formant_peak(preserved, sr, 1800, 2700) == pytest.approx(f2_before, rel=0.03)
    assert yin(preserved, sr).median_f0 == pytest.approx(yin(vowel_i, sr).median_f0 * 2 ** (5 / 12), rel=0.02)


def test_formant_warp_moves_formants_but_not_pitch(sr, vowel_i):
    f2_before = formant_peak(vowel_i, sr, 1800, 2700)
    y = pitch_formant_shift(vowel_i, sr, semitones=0, formant=1.25)
    assert formant_peak(y, sr, 2300, 3400) == pytest.approx(1.25 * f2_before, rel=0.03)
    assert yin(y, sr).median_f0 == pytest.approx(yin(vowel_i, sr).median_f0, rel=0.02)


# --- Linear prediction ----------------------------------------------------


def test_levinson_durbin_matches_toeplitz_solver():
    frame = np.random.default_rng(2).standard_normal(441) * np.hanning(441)
    r = lpc.autocorrelation(frame, 16)
    a, _ = lpc.levinson_durbin(r, 16)
    expected = solve_toeplitz(r[:16], -r[1:17])
    assert np.allclose(a[1:], expected, atol=1e-8)


def test_mcadams_alpha_one_is_near_identity(sr, male_voice):
    y = lpc.mcadams_anonymize(male_voice, sr, alpha=1.0)
    assert np.corrcoef(male_voice, y)[0, 1] > 0.99


def test_mcadams_changes_formants_keeps_pitch(sr, vowel_i):
    y = lpc.mcadams_anonymize(vowel_i, sr, alpha=0.8)
    assert yin(y, sr).median_f0 == pytest.approx(yin(vowel_i, sr).median_f0, rel=0.02)
    # F2 of /i/ sits at ~0.65 rad; phi -> phi^0.8 predicts ~0.71 rad, i.e. about +9%.
    f2_before = formant_peak(vowel_i, sr, 1800, 2700)
    predicted = (2 * np.pi * f2_before / sr) ** 0.8 * sr / (2 * np.pi)
    assert formant_peak(y, sr, 1800, 2900) == pytest.approx(predicted, rel=0.03)
