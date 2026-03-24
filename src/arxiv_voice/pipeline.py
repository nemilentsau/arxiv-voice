from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .chunking import chunk_text
from .extract import PdfExtraction, extract_pdf
from .paths import RunPaths
from .prompts import load_prompt
from .utils import ensure_dir, read_json, slugify, utc_timestamp, write_json, write_text


@dataclass
class RunContext:
    run_id: str
    paths: RunPaths


def create_run(pdf_path: Path, runs_dir: Path, paper_title: str | None = None) -> RunContext:
    slug_source = paper_title or pdf_path.stem
    run_id = f"{utc_timestamp()}-{slugify(slug_source)}"
    run_root = ensure_dir(runs_dir / run_id)
    paths = RunPaths.for_root(run_root)
    shutil.copy2(pdf_path, paths.source_pdf)
    return RunContext(run_id=run_id, paths=paths)


def extract_into_run(pdf_path: Path, runs_dir: Path, page_limit: int | None = None) -> RunContext:
    initial_context = create_run(pdf_path=pdf_path, runs_dir=runs_dir)
    extraction = extract_pdf(pdf_path, page_limit=page_limit)

    if extraction.title and extraction.title != pdf_path.stem:
        desired_root = initial_context.paths.root.parent / f"{utc_timestamp()}-{slugify(extraction.title)}"
        if desired_root != initial_context.paths.root:
            shutil.rmtree(initial_context.paths.root)
            initial_context = create_run(pdf_path=pdf_path, runs_dir=runs_dir, paper_title=extraction.title)

    write_extraction(initial_context.paths, extraction)
    return initial_context


def write_extraction(paths: RunPaths, extraction: PdfExtraction) -> None:
    ensure_dir(paths.chunk_text_dir)
    ensure_dir(paths.chunk_analysis_dir)
    ensure_dir(paths.audio_chunk_dir)
    ensure_dir(paths.podcast_audio_turn_dir)
    write_text(paths.paper_text, extraction.full_text)
    write_json(
        paths.metadata_json,
        {
            "title": extraction.title,
            "authors": extraction.authors,
            "page_count": extraction.page_count,
        },
    )
    write_json(paths.sections_json, extraction.sections)


def load_run(run_dir: Path) -> RunContext:
    paths = RunPaths.for_root(run_dir)
    return RunContext(run_id=run_dir.name, paths=paths)


def generate_overview(
    context: RunContext,
    llm_client: Any,
    analysis_chunk_chars: int = 12000,
    analysis_overlap_chars: int = 800,
) -> Dict[str, Any]:
    metadata = read_json(context.paths.metadata_json)
    paper_text = context.paths.paper_text.read_text(encoding="utf-8")
    sections = read_json(context.paths.sections_json)

    chunks = chunk_text(
        paper_text,
        max_chars=analysis_chunk_chars,
        overlap_chars=analysis_overlap_chars,
    )
    analyses = []
    chunk_system = load_prompt("chunk_analysis_system.txt")

    for index, chunk in enumerate(chunks, start=1):
        write_text(context.paths.chunk_text_dir / f"chunk-{index:03d}.txt", chunk)
        user_prompt = (
            f"Paper title: {metadata['title']}\n"
            f"Authors: {', '.join(metadata['authors']) or 'Unknown'}\n"
            f"Chunk: {index} of {len(chunks)}\n"
            f"Detected section headings: {', '.join(item['heading'] for item in sections[:20]) or 'None'}\n\n"
            "Analyze this chunk of a scientific paper for a later synthesis pass.\n\n"
            f"{chunk}"
        )
        analysis = llm_client.generate(chunk_system, user_prompt)
        analyses.append(analysis)
        write_text(context.paths.chunk_analysis_dir / f"chunk-{index:03d}.md", analysis)

    final_system = load_prompt("final_overview_system.txt")
    combined_notes = "\n\n".join(
        f"### Chunk {index}\n{analysis}" for index, analysis in enumerate(analyses, start=1)
    )
    final_user_prompt = (
        f"Paper title: {metadata['title']}\n"
        f"Authors: {', '.join(metadata['authors']) or 'Unknown'}\n"
        f"Page count extracted: {metadata['page_count']}\n"
        f"Detected section headings: {', '.join(item['heading'] for item in sections[:30]) or 'None'}\n\n"
        "Synthesize the chunk analyses below into one deep overview.\n\n"
        f"{combined_notes}"
    )
    overview = llm_client.generate(final_system, final_user_prompt)
    write_text(context.paths.overview_md, overview)

    return {
        "chunk_count": len(chunks),
        "chunk_files": [str(path) for path in sorted(context.paths.chunk_text_dir.glob("chunk-*.txt"))],
        "analysis_files": [str(path) for path in sorted(context.paths.chunk_analysis_dir.glob("chunk-*.md"))],
        "overview_file": str(context.paths.overview_md),
    }


def generate_spoken_overview(
    context: RunContext,
    llm_client: Any,
    target_minutes: int = 10,
) -> Dict[str, Any]:
    metadata = read_json(context.paths.metadata_json)
    overview_text = context.paths.overview_md.read_text(encoding="utf-8")
    spoken_system = load_prompt("spoken_brief_system.txt").format(target_minutes=target_minutes)
    spoken_user_prompt = (
        f"Paper title: {metadata['title']}\n"
        f"Authors: {', '.join(metadata['authors']) or 'Unknown'}\n\n"
        "Rewrite this overview into a spoken research brief.\n\n"
        f"{overview_text}"
    )
    script = llm_client.generate(spoken_system, spoken_user_prompt)
    write_text(context.paths.spoken_overview_txt, script)
    return {
        "script_file": str(context.paths.spoken_overview_txt),
        "estimated_characters": len(script),
    }


def generate_podcast_script(
    context: RunContext,
    llm_client: Any,
    target_minutes: int = 12,
) -> Dict[str, Any]:
    metadata = read_json(context.paths.metadata_json)
    overview_text = context.paths.overview_md.read_text(encoding="utf-8")
    podcast_system = load_prompt("podcast_dialogue_system.txt").format(target_minutes=target_minutes)
    podcast_user_prompt = (
        f"Paper title: {metadata['title']}\n"
        f"Authors: {', '.join(metadata['authors']) or 'Unknown'}\n\n"
        "Turn this overview into a two-speaker podcast conversation.\n\n"
        f"{overview_text}"
    )
    script = llm_client.generate(podcast_system, podcast_user_prompt)
    write_text(context.paths.podcast_script_txt, script)
    return {
        "script_file": str(context.paths.podcast_script_txt),
        "estimated_characters": len(script),
    }


def synthesize_audio(context: RunContext, synthesizer: Any) -> Dict[str, Any]:
    script = context.paths.spoken_overview_txt.read_text(encoding="utf-8")
    return synthesizer.synthesize(
        script=script,
        chunk_dir=context.paths.audio_chunk_dir,
        output_path=context.paths.final_audio,
    )


def synthesize_podcast_audio(context: RunContext, synthesizer: Any) -> Dict[str, Any]:
    script = context.paths.podcast_script_txt.read_text(encoding="utf-8")
    return synthesizer.synthesize(
        script=script,
        chunk_dir=context.paths.podcast_audio_turn_dir,
        output_path=context.paths.podcast_audio,
    )


def write_manifest(context: RunContext, manifest: Dict[str, Any]) -> None:
    write_json(context.paths.manifest_json, manifest)
