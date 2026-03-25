from __future__ import annotations

import importlib
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from .dialogue import chunk_dialogue_turns, coalesce_dialogue_turns, parse_dialogue_script, render_dia_script


DIA2_RUNTIME_REPO = "https://github.com/nari-labs/dia2.git"
DIA2_RUNTIME_COMMIT = "8687268f4ed3ed20704638fd353b51491de3b476"


class DiaError(RuntimeError):
    """Raised when local Dia generation fails."""


@dataclass
class DiaLocalSynthesizer:
    repo_id: str = "nari-labs/Dia2-2B"
    device: str = "mps"
    dtype: str = "auto"
    cfg_scale: float = 6.0
    text_temperature: float = 0.6
    audio_temperature: float = 0.8
    top_k: int = 50
    segment_turn_limit: int = 10
    prefix_speaker_1: Path | None = None
    prefix_speaker_2: Path | None = None
    include_prefix: bool = False
    runtime_root: Path = Path(".cache") / "dia2-runtime"

    def synthesize(self, script: str, segment_dir: Path, output_path: Path) -> dict:
        segments = split_dia_script(script, segment_turn_limit=self.segment_turn_limit)
        if not segments:
            raise DiaError("Podcast script did not contain valid dialogue content.")

        segment_dir.mkdir(parents=True, exist_ok=True)
        runtime = load_dia2_runtime(self.runtime_root)
        model = runtime["Dia2"].from_repo(self.repo_id, device=self.device, dtype=self.dtype)
        config = runtime["GenerationConfig"](
            text=runtime["SamplingConfig"](temperature=self.text_temperature, top_k=self.top_k),
            audio=runtime["SamplingConfig"](temperature=self.audio_temperature, top_k=self.top_k),
            cfg_scale=self.cfg_scale,
            prefix=self._build_prefix_config(),
            use_cuda_graph=False,
            use_torch_compile=False,
        )

        segment_paths: List[Path] = []
        total_timestamps = 0
        for index, dia_script in enumerate(segments, start=1):
            segment_path = segment_dir / f"segment-{index:03d}.wav"
            result = model.generate(
                dia_script,
                config=config,
                output_wav=segment_path,
                verbose=False,
            )
            total_timestamps += len(result.timestamps)
            segment_paths.append(segment_path)

        if not segment_paths:
            raise DiaError("Dia did not produce any audio segments.")

        concatenate_wav_files(segment_paths, output_path)
        return {
            "backend": "dia2-local",
            "repo_id": self.repo_id,
            "device": self.device,
            "dtype": self.dtype,
            "runtime_repo": str(self.runtime_root),
            "segment_count": len(segment_paths),
            "segments": [str(path) for path in segment_paths],
            "timestamp_count": total_timestamps,
            "output": str(output_path),
        }

    def _build_prefix_config(self) -> Any | None:
        if self.prefix_speaker_1 is None and self.prefix_speaker_2 is None:
            return None
        if self.prefix_speaker_1 is None or self.prefix_speaker_2 is None:
            raise DiaError("Provide both prefix speaker files or neither.")
        runtime = load_dia2_runtime(self.runtime_root)
        return runtime["PrefixConfig"](
            speaker_1=str(self.prefix_speaker_1),
            speaker_2=str(self.prefix_speaker_2),
            include_audio=self.include_prefix,
        )


def split_dia_script(script: str, segment_turn_limit: int) -> List[str]:
    dia_lines = [line.strip() for line in script.splitlines() if line.strip()]
    if dia_lines and all(line.startswith("[S1]") or line.startswith("[S2]") for line in dia_lines):
        return [
            "\n".join(dia_lines[index : index + segment_turn_limit]).strip()
            for index in range(0, len(dia_lines), segment_turn_limit)
        ]

    turns = coalesce_dialogue_turns(parse_dialogue_script(script))
    batches = chunk_dialogue_turns(turns, chunk_size=segment_turn_limit)
    return [render_dia_script(batch) for batch in batches if render_dia_script(batch)]


def load_dia2_runtime(runtime_root: Path) -> dict[str, Any]:
    repo_path = ensure_runtime_repo(runtime_root)
    repo_str = str(repo_path.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    module = importlib.import_module("dia2")
    return {
        "Dia2": getattr(module, "Dia2"),
        "GenerationConfig": getattr(module, "GenerationConfig"),
        "PrefixConfig": getattr(module, "PrefixConfig"),
        "SamplingConfig": getattr(module, "SamplingConfig"),
    }


def ensure_runtime_repo(runtime_root: Path) -> Path:
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_root.exists():
        run_git(["clone", "--depth", "1", DIA2_RUNTIME_REPO, str(runtime_root)])
        run_git(["-C", str(runtime_root), "fetch", "--depth", "1", "origin", DIA2_RUNTIME_COMMIT])
        run_git(["-C", str(runtime_root), "checkout", DIA2_RUNTIME_COMMIT])
        return runtime_root

    current = subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != DIA2_RUNTIME_COMMIT:
        run_git(["-C", str(runtime_root), "fetch", "--depth", "1", "origin", DIA2_RUNTIME_COMMIT])
        run_git(["-C", str(runtime_root), "checkout", DIA2_RUNTIME_COMMIT])
    return runtime_root


def run_git(args: List[str]) -> None:
    try:
        subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "git command failed"
        raise DiaError(stderr) from exc


def concatenate_wav_files(files: List[Path], output_path: Path) -> None:
    if not files:
        raise DiaError("No Dia audio segments were generated.")

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
                raise DiaError("Dia audio segments use incompatible WAV formats.")
            frames.append(current.readframes(current.getnframes()))

    with wave.open(str(output_path), "wb") as merged:
        merged.setparams(params)
        for frame_block in frames:
            merged.writeframes(frame_block)
