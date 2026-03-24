from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    return files("arxiv_voice").joinpath("prompts", name).read_text(encoding="utf-8")
