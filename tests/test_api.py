import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxmorph.__main__ import main as cli_main
from voxmorph.api import app
from voxmorph.audio_io import to_wav_bytes

client = TestClient(app)


@pytest.fixture
def wav_file(male_voice, sr):
    return {"file": ("voice.wav", to_wav_bytes(male_voice, sr), "audio/wav")}


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_effects_listing_includes_params():
    effects = {e["id"]: e for e in client.get("/api/effects").json()}
    assert {"male", "female", "robot", "anonymize", "custom"} <= set(effects)
    assert [p["name"] for p in effects["custom"]["params"]] == ["semitones", "formant", "speed"]


def test_process_returns_wav(wav_file, sr):
    r = client.post("/api/process", files=wav_file, data={"effect": "female", "params": json.dumps({"formant": 1.2})})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert float(r.headers["x-processing-time"]) >= 0
    y, out_sr = sf.read(io.BytesIO(r.content))
    assert out_sr == sr and len(y) > 0


def test_process_resamples_other_rates(male_voice):
    buf = io.BytesIO()
    sf.write(buf, np.repeat(male_voice, 2)[:44100], 44100, format="WAV")  # stereo-ish length, 44.1 kHz
    r = client.post("/api/process", files={"file": ("a.wav", buf.getvalue())}, data={"effect": "robot"})
    assert r.status_code == 200


def test_process_errors(wav_file):
    assert client.post("/api/process", files=wav_file, data={"effect": "nope"}).status_code == 404
    assert client.post("/api/process", files=wav_file, data={"effect": "custom", "params": "[1]"}).status_code == 422
    r = client.post("/api/process", files=wav_file, data={"effect": "custom", "params": '{"semitones": 99}'})
    assert r.status_code == 422
    bad = {"file": ("x.wav", b"definitely not audio")}
    assert client.post("/api/process", files=bad, data={"effect": "male"}).status_code == 400


def test_analyze(wav_file):
    a = client.post("/api/analyze", files=wav_file).json()
    spec = a["spectrogram"]
    assert len(base64.b64decode(spec["data"])) == spec["rows"] * spec["cols"]
    assert 100 < a["summary"]["median_f0"] < 150
    assert len(a["pitch"]["times"]) == len(a["pitch"]["f0"])


def test_sample_endpoint():
    r = client.get("/api/sample?speaker=female")
    assert r.status_code == 200 and r.content[:4] == b"RIFF"


def test_cli(tmp_path, male_voice, sr):
    src = tmp_path / "in.wav"
    sf.write(src, male_voice, sr)
    assert cli_main([str(src), "-e", "custom", "-p", "semitones=3"]) == 0
    assert (tmp_path / "in_custom.wav").exists()
    assert cli_main([str(src), "-e", "nope"]) == 1
