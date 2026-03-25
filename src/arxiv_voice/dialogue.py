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


def coalesce_dialogue_turns(turns: List[DialogueTurn]) -> List[DialogueTurn]:
    merged: List[DialogueTurn] = []
    for turn in turns:
        if not merged or merged[-1].speaker != turn.speaker:
            merged.append(turn)
            continue
        merged[-1] = DialogueTurn(
            speaker=turn.speaker,
            text=f"{merged[-1].text} {turn.text}".strip(),
        )
    return merged


def chunk_dialogue_turns(turns: List[DialogueTurn], chunk_size: int = 10) -> List[List[DialogueTurn]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [turns[index : index + chunk_size] for index in range(0, len(turns), chunk_size)]


def render_dia_script(turns: List[DialogueTurn]) -> str:
    speaker_map = {
        "HOST": "[S1]",
        "COHOST": "[S2]",
    }
    lines: List[str] = []
    for turn in turns:
        marker = speaker_map.get(turn.speaker)
        if not marker:
            continue
        text = turn.text.strip()
        if text:
            lines.append(f"{marker} {text}")
    return "\n".join(lines).strip()
