"""FastAPI service.

The API is stateless: audio goes in with the request and the processed WAV
comes straight back, so nothing is ever written to disk on the server.

Run with:  uvicorn voxmorph.api:app --reload
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .analysis import analyze
from .audio_io import SAMPLE_RATE, AudioError, load_audio, to_wav_bytes
from .effects import EFFECTS, EffectError, apply_effect
from .synth import synthetic_voice

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="VoxMorph API",
    version=__version__,
    description="Voice transformation with classic DSP: pitch, formants, LPC, phase vocoder and more.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("VOXMORPH_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time", "X-Effect"],
)


def _read_audio(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        return load_audio(data)
    except AudioError as exc:
        raise HTTPException(400, str(exc)) from exc


def _wav(content: bytes, **headers: str) -> Response:
    return Response(content, media_type="audio/wav", headers=headers)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "sample_rate": SAMPLE_RATE}


@app.get("/api/effects")
def list_effects() -> list[dict]:
    """All effects with their descriptions, DSP techniques and parameter ranges."""
    return [e.to_dict() for e in EFFECTS.values()]


@app.get("/api/sample")
def sample(speaker: str = Query("male", pattern="^(male|female)$")) -> Response:
    """A synthetic vowel sequence (source-filter model), handy for trying effects."""
    return _wav(to_wav_bytes(synthetic_voice(SAMPLE_RATE, speaker)))


@app.post("/api/analyze")
def analyze_audio(file: UploadFile = File(...)) -> dict:
    """Spectrogram, pitch contour and level statistics for an audio file."""
    x, sr = _read_audio(file)
    return analyze(x, sr)


@app.post("/api/process")
def process_audio(
    file: UploadFile = File(...),
    effect: str = Form(...),
    params: str = Form("{}", description='JSON object, e.g. {"semitones": 3}'),
) -> Response:
    """Apply an effect and return the result as a 16-bit WAV."""
    if effect not in EFFECTS:
        raise HTTPException(404, f"Unknown effect '{effect}'. Available: {', '.join(EFFECTS)}")
    try:
        parsed = json.loads(params)
        if not isinstance(parsed, dict):
            raise ValueError
    except ValueError as exc:
        raise HTTPException(422, "params must be a JSON object") from exc

    x, sr = _read_audio(file)
    start = time.perf_counter()
    try:
        y = apply_effect(x, sr, effect, parsed)
    except EffectError as exc:
        raise HTTPException(422, str(exc)) from exc
    elapsed = time.perf_counter() - start
    return _wav(to_wav_bytes(y, sr), **{"X-Processing-Time": f"{elapsed:.3f}", "X-Effect": effect})


# Serve the built React app (frontend/dist) when it exists, e.g. inside Docker.
_static = Path(os.getenv("VOXMORPH_STATIC_DIR", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="frontend")
