from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KokoroError(RuntimeError):
    """Raised when local Kokoro generation fails."""


@dataclass
class KokoroLocalNarrator:
    voice: str = "af_heart"
    lang_code: str = "a"
    speed: float = 1.0
    split_pattern: str = r"\n+"
    enable_mps_fallback: bool = True
    env_dir: Path = Path(".cache") / "kokoro-venv"
    runner_script: Path = Path("scripts") / "run_kokoro_simple_narrative.py"
    timeout_seconds: int = 1800

    def synthesize(self, script_path: Path, segment_dir: Path, output_path: Path) -> dict[str, Any]:
        python_executable = self.env_dir / "bin" / "python"
        if not python_executable.exists():
            raise KokoroError(
                "Kokoro environment is not installed. Run scripts/setup_kokoro_macos.sh first."
            )
        if not self.runner_script.exists():
            raise KokoroError(f"Kokoro runner script is missing: {self.runner_script}")
        if not script_path.exists():
            raise KokoroError(f"Narrative script does not exist: {script_path}")

        segment_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if self.enable_mps_fallback:
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        command = [
            str(python_executable),
            str(self.runner_script),
            "--input",
            str(script_path),
            "--output",
            str(output_path),
            "--segment-dir",
            str(segment_dir),
            "--voice",
            self.voice,
            "--lang-code",
            self.lang_code,
            "--speed",
            str(self.speed),
            "--split-pattern",
            self.split_pattern,
            "--json",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise KokoroError("Kokoro python executable is not available.") from exc
        except subprocess.TimeoutExpired as exc:
            raise KokoroError("Timed out while waiting for Kokoro narration to finish.") from exc

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Kokoro generation failed."
            raise KokoroError(message)

        try:
            payload = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise KokoroError("Kokoro runner returned invalid JSON output.") from exc

        return payload
