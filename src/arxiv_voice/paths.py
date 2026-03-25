from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    source_pdf: Path
    overview_md: Path
    podcast_script_txt: Path
    podcast_dia_txt: Path
    audio_dir: Path
    podcast_audio_segment_dir: Path
    podcast_audio: Path
    manifest_json: Path

    @classmethod
    def for_root(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            source_pdf=root / "source.pdf",
            overview_md=root / "overview.md",
            podcast_script_txt=root / "podcast_script.txt",
            podcast_dia_txt=root / "podcast_dia.txt",
            audio_dir=root / "audio",
            podcast_audio_segment_dir=root / "audio" / "segments",
            podcast_audio=root / "audio" / "podcast.wav",
            manifest_json=root / "manifest.json",
        )
