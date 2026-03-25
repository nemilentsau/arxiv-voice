from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dia_local import DiaError, DiaLocalSynthesizer
from .llm import LLMError, build_llm_client
from .pipeline import (
    generate_overview,
    generate_podcast_script,
    load_run,
    stage_pdf_into_run,
    synthesize_podcast_audio,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Copy a PDF into a run directory.")
    add_common_pdf_args(extract_parser)

    overview_parser = subparsers.add_parser("overview", help="Generate overview.md from source.pdf via Claude CLI.")
    overview_parser.add_argument("run_dir", type=Path)
    add_claude_args(overview_parser)

    podcast_parser = subparsers.add_parser("podcast", help="Generate a two-speaker podcast and Dia audio.")
    podcast_parser.add_argument("run_dir", type=Path)
    add_claude_args(podcast_parser)
    podcast_parser.add_argument("--target-minutes", type=int, default=12)
    add_dia_args(podcast_parser)
    podcast_parser.add_argument("--no-audio", action="store_true")

    run_parser = subparsers.add_parser("run", help="Stage a PDF, build an overview, and generate a podcast.")
    add_common_pdf_args(run_parser)
    add_claude_args(run_parser)
    run_parser.add_argument("--target-minutes", type=int, default=12)
    add_dia_args(run_parser)
    run_parser.add_argument("--no-audio", action="store_true")

    return parser


def add_common_pdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))


def add_claude_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high", "max"))
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=1800)


def add_dia_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dia-repo", default="nari-labs/Dia2-2B")
    parser.add_argument("--dia-device", default="mps")
    parser.add_argument("--dia-dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--dia-cfg-scale", type=float, default=6.0)
    parser.add_argument("--dia-text-temperature", type=float, default=0.6)
    parser.add_argument("--dia-audio-temperature", type=float, default=0.8)
    parser.add_argument("--dia-top-k", type=int, default=50)
    parser.add_argument("--dia-segment-turn-limit", type=int, default=10)
    parser.add_argument("--prefix-speaker-1", type=Path, default=None)
    parser.add_argument("--prefix-speaker-2", type=Path, default=None)
    parser.add_argument("--include-prefix", action="store_true")


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"artifacts": {}, "steps": {}}


def build_client(args: argparse.Namespace):
    return build_llm_client(
        model=args.model,
        effort=args.effort,
        bare=args.bare,
        max_budget_usd=args.max_budget_usd,
        timeout_seconds=args.timeout_seconds,
    )


def llm_manifest(client) -> dict:
    return {
        "backend": "claude-cli",
        "model": getattr(client, "model", None),
        "effort": getattr(client, "effort", None),
        "bare": getattr(client, "bare", None),
        "max_budget_usd": getattr(client, "max_budget_usd", None),
        "timeout_seconds": getattr(client, "timeout_seconds", None),
    }


def cmd_extract(args: argparse.Namespace) -> int:
    context = stage_pdf_into_run(args.pdf_path, args.runs_dir)
    manifest = load_manifest(context.paths.manifest_json)
    manifest.update(
        {
            "run_id": context.run_id,
            "source_pdf": str(args.pdf_path.resolve()),
        }
    )
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "source_pdf": str(context.paths.source_pdf),
        }
    )
    manifest["steps"]["extract"] = {"status": "completed"}
    write_manifest(context, manifest)
    print(context.paths.root)
    return 0


def cmd_overview(args: argparse.Namespace) -> int:
    context = load_run(args.run_dir)
    llm_client = build_client(args)
    overview_result = generate_overview(context=context, llm_client=llm_client)
    manifest = load_manifest(context.paths.manifest_json)
    manifest["run_id"] = context.run_id
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "source_pdf": str(context.paths.source_pdf),
            "overview_md": str(context.paths.overview_md),
        }
    )
    manifest["llm"] = llm_manifest(llm_client)
    manifest["steps"]["overview"] = {"status": "completed", **overview_result}
    write_manifest(context, manifest)
    print(context.paths.overview_md)
    return 0


def cmd_podcast(args: argparse.Namespace) -> int:
    context = load_run(args.run_dir)
    llm_client = build_client(args)
    manifest = load_manifest(context.paths.manifest_json)
    manifest["run_id"] = context.run_id
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "source_pdf": str(context.paths.source_pdf),
            "podcast_script_txt": str(context.paths.podcast_script_txt),
            "podcast_dia_txt": str(context.paths.podcast_dia_txt),
        }
    )
    manifest["llm"] = llm_manifest(llm_client)

    podcast_result = generate_podcast_script(
        context=context,
        llm_client=llm_client,
        target_minutes=args.target_minutes,
    )
    manifest["steps"]["podcast_script"] = {"status": "completed", **podcast_result}

    if not args.no_audio:
        synthesizer = DiaLocalSynthesizer(
            repo_id=args.dia_repo,
            device=args.dia_device,
            dtype=args.dia_dtype,
            cfg_scale=args.dia_cfg_scale,
            text_temperature=args.dia_text_temperature,
            audio_temperature=args.dia_audio_temperature,
            top_k=args.dia_top_k,
            segment_turn_limit=args.dia_segment_turn_limit,
            prefix_speaker_1=args.prefix_speaker_1,
            prefix_speaker_2=args.prefix_speaker_2,
            include_prefix=args.include_prefix,
        )
        audio_result = synthesize_podcast_audio(context=context, synthesizer=synthesizer)
        manifest["audio"] = {
            "backend": "dia2-local",
            "repo_id": args.dia_repo,
            "device": args.dia_device,
            "dtype": args.dia_dtype,
            "cfg_scale": args.dia_cfg_scale,
            "text_temperature": args.dia_text_temperature,
            "audio_temperature": args.dia_audio_temperature,
            "top_k": args.dia_top_k,
            "segment_turn_limit": args.dia_segment_turn_limit,
            "prefix_speaker_1": str(args.prefix_speaker_1) if args.prefix_speaker_1 else None,
            "prefix_speaker_2": str(args.prefix_speaker_2) if args.prefix_speaker_2 else None,
            "include_prefix": args.include_prefix,
        }
        manifest["artifacts"]["podcast_audio"] = str(context.paths.podcast_audio)
        manifest["steps"]["podcast_audio"] = {"status": "completed", **audio_result}

    write_manifest(context, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    context = stage_pdf_into_run(args.pdf_path, args.runs_dir)
    llm_client = build_client(args)
    manifest = {
        "run_id": context.run_id,
        "source_pdf": str(args.pdf_path.resolve()),
        "artifacts": {
            "run_dir": str(context.paths.root),
            "source_pdf": str(context.paths.source_pdf),
            "overview_md": str(context.paths.overview_md),
            "podcast_script_txt": str(context.paths.podcast_script_txt),
            "podcast_dia_txt": str(context.paths.podcast_dia_txt),
        },
        "llm": llm_manifest(llm_client),
        "steps": {"extract": {"status": "completed"}},
    }

    overview_result = generate_overview(context=context, llm_client=llm_client)
    manifest["steps"]["overview"] = {"status": "completed", **overview_result}

    podcast_result = generate_podcast_script(
        context=context,
        llm_client=llm_client,
        target_minutes=max(args.target_minutes, 4),
    )
    manifest["steps"]["podcast_script"] = {"status": "completed", **podcast_result}

    if not args.no_audio:
        podcast_synthesizer = DiaLocalSynthesizer(
            repo_id=args.dia_repo,
            device=args.dia_device,
            dtype=args.dia_dtype,
            cfg_scale=args.dia_cfg_scale,
            text_temperature=args.dia_text_temperature,
            audio_temperature=args.dia_audio_temperature,
            top_k=args.dia_top_k,
            segment_turn_limit=args.dia_segment_turn_limit,
            prefix_speaker_1=args.prefix_speaker_1,
            prefix_speaker_2=args.prefix_speaker_2,
            include_prefix=args.include_prefix,
        )
        podcast_audio_result = synthesize_podcast_audio(context=context, synthesizer=podcast_synthesizer)
        manifest["audio"] = {
            "backend": "dia2-local",
            "repo_id": args.dia_repo,
            "device": args.dia_device,
            "dtype": args.dia_dtype,
            "cfg_scale": args.dia_cfg_scale,
            "text_temperature": args.dia_text_temperature,
            "audio_temperature": args.dia_audio_temperature,
            "top_k": args.dia_top_k,
            "segment_turn_limit": args.dia_segment_turn_limit,
            "prefix_speaker_1": str(args.prefix_speaker_1) if args.prefix_speaker_1 else None,
            "prefix_speaker_2": str(args.prefix_speaker_2) if args.prefix_speaker_2 else None,
            "include_prefix": args.include_prefix,
        }
        manifest["artifacts"]["podcast_audio"] = str(context.paths.podcast_audio)
        manifest["steps"]["podcast_audio"] = {"status": "completed", **podcast_audio_result}

    write_manifest(context, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            return cmd_extract(args)
        if args.command == "overview":
            return cmd_overview(args)
        if args.command == "podcast":
            return cmd_podcast(args)
        if args.command == "run":
            return cmd_run(args)
    except (LLMError, DiaError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
