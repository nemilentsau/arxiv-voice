import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arxiv_voice.kokoro_local import KokoroError, KokoroLocalNarrator


class KokoroLocalNarratorTests(unittest.TestCase):
    def test_synthesize_invokes_runner_and_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_dir = root / "kokoro-venv"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").write_text("")

            runner_script = root / "run_kokoro.py"
            runner_script.write_text("")

            script_path = root / "narrative_script.txt"
            script_path.write_text("A short narrative.")

            narrator = KokoroLocalNarrator(
                env_dir=env_dir,
                runner_script=runner_script,
            )

            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"backend": "kokoro-local", "output": str(root / "narrative.wav")}),
                stderr="",
            )

            with patch("subprocess.run", return_value=completed) as run_mock:
                result = narrator.synthesize(
                    script_path=script_path,
                    segment_dir=root / "segments",
                    output_path=root / "narrative.wav",
                )

            self.assertEqual(result["backend"], "kokoro-local")
            command = run_mock.call_args.args[0]
            self.assertIn("--json", command)
            self.assertIn("--segment-dir", command)

    def test_synthesize_requires_kokoro_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner_script = root / "run_kokoro.py"
            runner_script.write_text("")
            script_path = root / "narrative_script.txt"
            script_path.write_text("A short narrative.")

            narrator = KokoroLocalNarrator(
                env_dir=root / "missing-env",
                runner_script=runner_script,
            )

            with self.assertRaises(KokoroError):
                narrator.synthesize(
                    script_path=script_path,
                    segment_dir=root / "segments",
                    output_path=root / "narrative.wav",
                )


if __name__ == "__main__":
    unittest.main()
