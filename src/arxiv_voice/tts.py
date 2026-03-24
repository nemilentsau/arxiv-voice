from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .chunking import chunk_text
from .dialogue import parse_dialogue_script


class TTSError(RuntimeError):
    """Raised when local speech synthesis fails."""


@dataclass
class SaySynthesizer:
    voice: str = "Samantha"
    rate: int = 190
    chunk_chars: int = 1400

    def synthesize(self, script: str, chunk_dir: Path, output_path: Path) -> dict:
        say_path = shutil.which("say")
        if not say_path:
            raise TTSError("macOS 'say' is not available on this machine.")

        chunks = chunk_text(script, max_chars=self.chunk_chars, overlap_chars=0)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        generated_files: List[Path] = []

        for index, chunk in enumerate(chunks, start=1):
            chunk_path = chunk_dir / f"chunk-{index:03d}.wav"
            speak_to_file(say_path, chunk, self.voice, self.rate, chunk_path)
            generated_files.append(chunk_path)

        concatenate_wav_files(generated_files, output_path)
        return {
            "backend": "say",
            "voice": self.voice,
            "rate": self.rate,
            "chunks": [str(path) for path in generated_files],
            "output": str(output_path),
        }


@dataclass
class SayDialogueSynthesizer:
    host_voice: str = "Samantha"
    cohost_voice: str = "Daniel"
    host_rate: int = 190
    cohost_rate: int = 185
    gap_ms: int = 250

    def synthesize(self, script: str, chunk_dir: Path, output_path: Path) -> dict:
        say_path = shutil.which("say")
        if not say_path:
            raise TTSError("macOS 'say' is not available on this machine.")

        turns = parse_dialogue_script(script)
        if not turns:
            raise TTSError("Podcast script did not contain any HOST:/COHOST: turns.")

        chunk_dir.mkdir(parents=True, exist_ok=True)
        generated_files: List[Path] = []
        for index, turn in enumerate(turns, start=1):
            if turn.speaker == "HOST":
                voice = self.host_voice
                rate = self.host_rate
            elif turn.speaker == "COHOST":
                voice = self.cohost_voice
                rate = self.cohost_rate
            else:
                continue

            chunk_path = chunk_dir / f"turn-{index:03d}-{turn.speaker.lower()}.wav"
            speak_to_file(say_path, turn.text, voice, rate, chunk_path)
            generated_files.append(chunk_path)

        concatenate_wav_files(generated_files, output_path, gap_ms=self.gap_ms)
        return {
            "backend": "say-dialogue",
            "host_voice": self.host_voice,
            "cohost_voice": self.cohost_voice,
            "host_rate": self.host_rate,
            "cohost_rate": self.cohost_rate,
            "turn_count": len(turns),
            "chunks": [str(path) for path in generated_files],
            "output": str(output_path),
        }


def speak_to_file(say_path: str, text: str, voice: str, rate: int, output_path: Path) -> None:
    command = [
        say_path,
        "-v",
        voice,
        "-r",
        str(rate),
        "-o",
        str(output_path),
        "--data-format=LEI16@22050",
        text,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise TTSError(
            f"Speech synthesis failed for voice '{voice}'. stderr: {exc.stderr.strip() or 'no stderr'}"
        ) from exc


def concatenate_wav_files(files: List[Path], output_path: Path, gap_ms: int = 0) -> None:
    if not files:
        raise TTSError("No audio chunks were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(files[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for path in files[1:]:
        with wave.open(str(path), "rb") as current:
            current_params = current.getparams()
            if (
                current_params.nchannels != params.nchannels
                or current_params.sampwidth != params.sampwidth
                or current_params.framerate != params.framerate
                or current_params.comptype != params.comptype
                or current_params.compname != params.compname
            ):
                raise TTSError("Audio chunk formats do not match and cannot be concatenated.")
            frames.append(current.readframes(current.getnframes()))

    with wave.open(str(output_path), "wb") as merged:
        merged.setparams(params)
        silence = b""
        if gap_ms > 0:
            frame_count = int(params.framerate * (gap_ms / 1000.0))
            silence = b"\x00" * frame_count * params.nchannels * params.sampwidth
        for index, frame_block in enumerate(frames):
            merged.writeframes(frame_block)
            if silence and index != len(frames) - 1:
                merged.writeframes(silence)
