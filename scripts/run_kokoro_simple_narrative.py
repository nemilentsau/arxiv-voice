#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one Kokoro narrative WAV from a text file.")
    parser.add_argument("--input", required=True, help="Path to input text file.")
    parser.add_argument("--output", required=True, help="Path to output WAV file.")
    parser.add_argument("--segment-dir", default=None, help="Optional directory to write segment WAVs.")
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice name.")
    parser.add_argument("--lang-code", default="a", help="Kokoro language code, e.g. 'a' for American English.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--split-pattern", default=r"\n+", help="Regex used to split text into chunks.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of human-readable logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    segment_dir = Path(args.segment_dir) if args.segment_dir else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if segment_dir is not None:
        segment_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text()

    pipeline = KPipeline(lang_code=args.lang_code)
    generator = pipeline(
        text,
        voice=args.voice,
        speed=args.speed,
        split_pattern=args.split_pattern,
    )

    chunks: list[np.ndarray] = []
    segment_paths: list[str] = []
    sample_rate = 24000
    for idx, (_, _, audio) in enumerate(generator, start=1):
        chunk = np.asarray(audio, dtype=np.float32)
        chunks.append(chunk)
        if segment_dir is not None:
            segment_path = segment_dir / f"segment-{idx:03d}.wav"
            sf.write(segment_path, chunk, sample_rate)
            segment_paths.append(str(segment_path))
        if not args.json:
            print(f"segment={idx} samples={len(chunk)} seconds={len(chunk) / sample_rate:.2f}")

    if not chunks:
        raise RuntimeError("Kokoro returned no audio chunks")

    full_audio = np.concatenate(chunks)
    sf.write(output_path, full_audio, sample_rate)
    result = {
        "backend": "kokoro-local",
        "voice": args.voice,
        "lang_code": args.lang_code,
        "speed": args.speed,
        "split_pattern": args.split_pattern,
        "sample_rate": sample_rate,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "segment_count": len(chunks),
        "segments": segment_paths,
        "samples": int(len(full_audio)),
        "seconds": float(len(full_audio) / sample_rate),
        "output": str(output_path),
    }
    if args.json:
        print(json.dumps(result))
        return

    print("torch", torch.__version__)
    print("mps_available", torch.backends.mps.is_available())
    print("mps_built", torch.backends.mps.is_built())
    print(f"output={output_path}")
    print(f"segments={len(chunks)}")
    print(f"samples={len(full_audio)}")
    print(f"seconds={len(full_audio) / sample_rate:.2f}")


if __name__ == "__main__":
    main()
