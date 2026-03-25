from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


class LLMError(RuntimeError):
    """Raised when the configured LLM backend fails."""


@dataclass
class GenerationResult:
    text: str
    session_id: str | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeCliClient:
    model: str = "sonnet"
    effort: str = "medium"
    bare: bool = False
    max_budget_usd: float | None = None
    timeout_seconds: int = 1800
    executable: str = "claude"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        cwd: Path | None = None,
        tools: Sequence[str] = ("Bash", "Read"),
    ) -> GenerationResult:
        command = self._build_command(system_prompt=system_prompt, user_prompt=user_prompt, tools=tools)
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError("Claude CLI is not installed or not on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError("Timed out while waiting for Claude CLI to finish.") from exc

        payload = self._parse_payload(completed.stdout)
        if completed.returncode != 0 or payload.get("is_error"):
            message = (
                payload.get("result")
                or completed.stderr.strip()
                or completed.stdout.strip()
                or f"Claude CLI exited with code {completed.returncode}."
            )
            raise LLMError(message)

        text = (payload.get("result") or "").strip()
        if not text:
            raise LLMError("Claude CLI returned an empty response.")

        return GenerationResult(
            text=text,
            session_id=payload.get("session_id"),
            total_cost_usd=payload.get("total_cost_usd"),
            duration_ms=payload.get("duration_ms"),
            num_turns=payload.get("num_turns"),
            stop_reason=payload.get("stop_reason"),
            payload=payload,
        )

    def _build_command(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[str],
    ) -> list[str]:
        command = [self.executable]
        if self.bare:
            command.append("--bare")
        command.extend(
            [
                "-p",
                user_prompt,
                "--model",
                self.model,
                "--effort",
                self.effort,
                "--output-format",
                "json",
                "--no-session-persistence",
                "--append-system-prompt",
                system_prompt,
            ]
        )
        if tools:
            tool_list = ",".join(tools)
            command.extend(["--tools", tool_list, "--allowedTools", tool_list])
        if self.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(self.max_budget_usd)])
        return command

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any]:
        body = stdout.strip()
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Claude CLI returned invalid JSON output.") from exc


def build_llm_client(
    *,
    model: str | None = None,
    effort: str | None = None,
    bare: bool = False,
    max_budget_usd: float | None = None,
    timeout_seconds: int = 1800,
) -> ClaudeCliClient:
    if shutil.which("claude") is None:
        raise LLMError("Claude CLI is not installed or not on PATH.")

    chosen_model = model or os.getenv("ARXIV_VOICE_CLAUDE_MODEL") or "sonnet"
    chosen_effort = effort or os.getenv("ARXIV_VOICE_CLAUDE_EFFORT") or "medium"
    return ClaudeCliClient(
        model=chosen_model,
        effort=chosen_effort,
        bare=bare,
        max_budget_usd=max_budget_usd,
        timeout_seconds=timeout_seconds,
    )
