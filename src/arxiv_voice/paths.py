from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    source_pdf: Path
    paper_text: Path
    metadata_json: Path
    sections_json: Path
    chunk_text_dir: Path
    chunk_analysis_dir: Path
    overview_md: Path
    spoken_overview_txt: Path
    podcast_script_txt: Path
    audio_dir: Path
    audio_chunk_dir: Path
    final_audio: Path
    podcast_audio_turn_dir: Path
    podcast_audio: Path
    manifest_json: Path

    @classmethod
    def for_root(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            source_pdf=root / "source.pdf",
            paper_text=root / "paper.txt",
            metadata_json=root / "metadata.json",
            sections_json=root / "sections.json",
            chunk_text_dir=root / "chunks",
            chunk_analysis_dir=root / "analysis",
            overview_md=root / "overview.md",
            spoken_overview_txt=root / "spoken_overview.txt",
            podcast_script_txt=root / "podcast_script.txt",
            audio_dir=root / "audio",
            audio_chunk_dir=root / "audio" / "chunks",
            final_audio=root / "audio" / "research_brief.wav",
            podcast_audio_turn_dir=root / "audio" / "podcast_turns",
            podcast_audio=root / "audio" / "podcast.wav",
            manifest_json=root / "manifest.json",
        )
