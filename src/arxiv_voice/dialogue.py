from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str


def parse_dialogue_script(script: str) -> List[DialogueTurn]:
    turns: List[DialogueTurn] = []
    current_speaker: str | None = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker and current_lines:
            text = " ".join(line.strip() for line in current_lines if line.strip()).strip()
            if text:
                turns.append(DialogueTurn(speaker=current_speaker, text=text))
        current_speaker = None
        current_lines = []

    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HOST:") or line.startswith("COHOST:"):
            flush()
            speaker, text = line.split(":", 1)
            current_speaker = speaker.strip()
            current_lines = [text.strip()]
        elif current_speaker:
            current_lines.append(line)

    flush()
    return turns
