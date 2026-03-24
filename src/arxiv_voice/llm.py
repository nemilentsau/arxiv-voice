from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Raised when the configured LLM backend fails."""


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    timeout_seconds: int = 600

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Failed to reach Ollama at {self.base_url}. "
                "Ensure the Ollama app or server is running."
            ) from exc
        except TimeoutError as exc:
            raise LLMError("Timed out while waiting for Ollama to finish.") from exc

        text = (body.get("response") or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty response.")
        return text


def build_llm_client(backend: str, model: str | None = None):
    normalized = backend.strip().lower()
    if normalized != "ollama":
        raise LLMError(f"Unsupported LLM backend: {backend}")
    chosen_model = model or os.getenv("ARXIV_VOICE_OLLAMA_MODEL") or "qwen3.5:latest"
    base_url = os.getenv("ARXIV_VOICE_OLLAMA_URL") or "http://127.0.0.1:11434"
    return OllamaClient(model=chosen_model, base_url=base_url)
