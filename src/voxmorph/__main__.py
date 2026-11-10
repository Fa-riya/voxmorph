"""Command-line interface.

voxmorph input.wav -e female -o output.wav
voxmorph input.wav -e custom -p semitones=-3 -p formant=0.9
voxmorph --list
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .audio_io import AudioError, load_audio, save_audio
from .effects import EFFECTS, EffectError, apply_effect


def _parse_param(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected name=value, got '{text}'")
    name, value = text.split("=", 1)
    return name.strip(), float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voxmorph", description="Transform a voice recording with DSP effects.")
    parser.add_argument("input", nargs="?", help="input audio file (wav, flac, ogg, mp3)")
    parser.add_argument("-e", "--effect", default="custom", help="effect id (see --list)")
    parser.add_argument(
        "-p", "--param", action="append", type=_parse_param, default=[], help="effect parameter, e.g. -p semitones=3"
    )
    parser.add_argument("-o", "--output", help="output wav path (default: <input>_<effect>.wav)")
    parser.add_argument("--list", action="store_true", help="list effects and their parameters")
    args = parser.parse_args(argv)

    if args.list:
        for e in EFFECTS.values():
            print(f"{e.id:<11} {e.description}")
            for p in e.params:
                print(f"{'':<11}   {p.name}: {p.min}..{p.max} (default {p.default}{p.unit and ' ' + p.unit})")
        return 0
    if not args.input:
        parser.error("an input file is required (or use --list)")

    try:
        x, sr = load_audio(args.input, max_seconds=None)
        start = time.perf_counter()
        y = apply_effect(x, sr, args.effect, dict(args.param))
        elapsed = time.perf_counter() - start
    except (AudioError, EffectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else Path(args.input).with_name(f"{Path(args.input).stem}_{args.effect}.wav")
    save_audio(out, y, sr)
    print(f"{args.effect}: {len(x) / sr:.2f}s of audio processed in {elapsed:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
