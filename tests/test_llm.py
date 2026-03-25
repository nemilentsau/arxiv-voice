import json
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_voice.llm import ClaudeCliClient, LLMError


class ClaudeCliClientTests(unittest.TestCase):
    def test_build_command_uses_headless_flags(self) -> None:
        client = ClaudeCliClient(model="sonnet", effort="medium", bare=False, max_budget_usd=2.5)
        command = client._build_command(
            system_prompt="System prompt",
            user_prompt="User prompt",
            tools=("Bash", "Read"),
        )
        self.assertIn("-p", command)
        self.assertIn("--append-system-prompt", command)
        self.assertIn("--output-format", command)
        self.assertIn("json", command)
        self.assertIn("--tools", command)
        self.assertIn("Bash,Read", command)
        self.assertIn("--allowedTools", command)
        self.assertIn("--max-budget-usd", command)

    @patch("subprocess.run")
    def test_generate_parses_json_result(self, run_mock) -> None:
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = json.dumps(
            {
                "is_error": False,
                "result": "Hello.",
                "session_id": "abc",
                "total_cost_usd": 0.12,
                "duration_ms": 100,
                "num_turns": 2,
                "stop_reason": "end_turn",
            }
        )
        run_mock.return_value.stderr = ""

        result = ClaudeCliClient().generate("sys", "user", cwd=Path("."))
        self.assertEqual(result.text, "Hello.")
        self.assertEqual(result.session_id, "abc")
        self.assertEqual(result.total_cost_usd, 0.12)

    @patch("subprocess.run")
    def test_generate_raises_on_cli_error(self, run_mock) -> None:
        run_mock.return_value.returncode = 1
        run_mock.return_value.stdout = json.dumps({"is_error": True, "result": "Not logged in"})
        run_mock.return_value.stderr = ""

        with self.assertRaises(LLMError) as exc:
            ClaudeCliClient().generate("sys", "user")
        self.assertIn("Not logged in", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
