from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .dialogue import coalesce_dialogue_turns, parse_dialogue_script, render_dia_script
from .llm import GenerationResult
from .paths import RunPaths
from .prompts import load_prompt
from .utils import ensure_dir, slugify, utc_timestamp, write_json, write_text


@dataclass
class RunContext:
    run_id: str
    paths: RunPaths


def create_run(pdf_path: Path, runs_dir: Path) -> RunContext:
    run_id = f"{utc_timestamp()}-{slugify(pdf_path.stem)}"
    run_root = ensure_dir(runs_dir / run_id)
    paths = RunPaths.for_root(run_root)
    shutil.copy2(pdf_path, paths.source_pdf)
    ensure_dir(paths.audio_dir)
    ensure_dir(paths.podcast_audio_segment_dir)
    return RunContext(run_id=run_id, paths=paths)


def load_run(run_dir: Path) -> RunContext:
    paths = RunPaths.for_root(run_dir)
    return RunContext(run_id=run_dir.name, paths=paths)


def stage_pdf_into_run(pdf_path: Path, runs_dir: Path) -> RunContext:
    return create_run(pdf_path=pdf_path, runs_dir=runs_dir)


def generate_overview(
    context: RunContext,
    llm_client: Any,
) -> Dict[str, Any]:
    system_prompt = load_prompt("overview_system.txt")
    user_prompt = (
        "Analyze the scientific paper in ./source.pdf directly.\n"
        "Use local tools if needed to inspect the PDF, but do not modify files.\n"
        "Return only the final Markdown overview."
    )
    result: GenerationResult = llm_client.generate(
        system_prompt,
        user_prompt,
        cwd=context.paths.root,
    )
    write_text(context.paths.overview_md, result.text)
    return {
        "overview_file": str(context.paths.overview_md),
        "session_id": result.session_id,
        "total_cost_usd": result.total_cost_usd,
        "duration_ms": result.duration_ms,
        "num_turns": result.num_turns,
        "stop_reason": result.stop_reason,
    }


def generate_podcast_script(
    context: RunContext,
    llm_client: Any,
    target_minutes: int = 12,
) -> Dict[str, Any]:
    podcast_system = load_prompt("podcast_dialogue_system.txt").format(target_minutes=target_minutes)
    podcast_user_prompt = (
        "Read ./overview.md in the current directory and turn it into a two-speaker podcast conversation.\n"
        "Return only the final plain-text script."
    )
    result: GenerationResult = llm_client.generate(
        podcast_system,
        podcast_user_prompt,
        cwd=context.paths.root,
        tools=("Read",),
    )
    write_text(context.paths.podcast_script_txt, result.text)
    turns = coalesce_dialogue_turns(parse_dialogue_script(result.text))
    dia_script = render_dia_script(turns)
    write_text(context.paths.podcast_dia_txt, dia_script)
    return {
        "script_file": str(context.paths.podcast_script_txt),
        "dia_script_file": str(context.paths.podcast_dia_txt),
        "turn_count": len(turns),
        "estimated_characters": len(result.text),
        "session_id": result.session_id,
        "total_cost_usd": result.total_cost_usd,
        "duration_ms": result.duration_ms,
        "num_turns": result.num_turns,
        "stop_reason": result.stop_reason,
    }


def synthesize_podcast_audio(context: RunContext, synthesizer: Any) -> Dict[str, Any]:
    script = context.paths.podcast_dia_txt.read_text(encoding="utf-8")
    return synthesizer.synthesize(
        script=script,
        segment_dir=context.paths.podcast_audio_segment_dir,
        output_path=context.paths.podcast_audio,
    )


def write_manifest(context: RunContext, manifest: Dict[str, Any]) -> None:
    write_json(context.paths.manifest_json, manifest)
