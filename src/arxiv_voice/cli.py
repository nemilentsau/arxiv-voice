from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import LLMError, build_llm_client
from .pipeline import (
    extract_into_run,
    generate_overview,
    generate_podcast_script,
    generate_spoken_overview,
    load_run,
    synthesize_audio,
    synthesize_podcast_audio,
    write_manifest,
)
from .tts import SayDialogueSynthesizer, SaySynthesizer, TTSError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract a PDF into a run directory.")
    add_common_pdf_args(extract_parser)

    overview_parser = subparsers.add_parser("overview", help="Generate overview.md for an extracted run.")
    overview_parser.add_argument("run_dir", type=Path)
    add_overview_args(overview_parser)

    narrate_parser = subparsers.add_parser("narrate", help="Generate spoken script and local audio for a run.")
    narrate_parser.add_argument("run_dir", type=Path)
    narrate_parser.add_argument("--backend", default="ollama")
    narrate_parser.add_argument("--model", default=None)
    narrate_parser.add_argument("--target-minutes", type=int, default=10)
    narrate_parser.add_argument("--voice", default="Samantha")
    narrate_parser.add_argument("--rate", type=int, default=190)
    narrate_parser.add_argument("--no-audio", action="store_true")

    podcast_parser = subparsers.add_parser("podcast", help="Generate a two-speaker podcast script and local audio.")
    podcast_parser.add_argument("run_dir", type=Path)
    podcast_parser.add_argument("--backend", default="ollama")
    podcast_parser.add_argument("--model", default=None)
    podcast_parser.add_argument("--target-minutes", type=int, default=12)
    podcast_parser.add_argument("--host-voice", default="Samantha")
    podcast_parser.add_argument("--cohost-voice", default="Daniel")
    podcast_parser.add_argument("--host-rate", type=int, default=190)
    podcast_parser.add_argument("--cohost-rate", type=int, default=185)
    podcast_parser.add_argument("--no-audio", action="store_true")

    run_parser = subparsers.add_parser("run", help="Extract a PDF, build an overview, and narrate it.")
    add_common_pdf_args(run_parser)
    add_overview_args(run_parser)
    run_parser.add_argument(
        "--mode",
        choices=("research-brief", "podcast", "both"),
        default="research-brief",
    )
    run_parser.add_argument("--target-minutes", type=int, default=10)
    run_parser.add_argument("--voice", default="Samantha")
    run_parser.add_argument("--rate", type=int, default=190)
    run_parser.add_argument("--host-voice", default="Samantha")
    run_parser.add_argument("--cohost-voice", default="Daniel")
    run_parser.add_argument("--host-rate", type=int, default=190)
    run_parser.add_argument("--cohost-rate", type=int, default=185)
    run_parser.add_argument("--no-audio", action="store_true")

    return parser


def add_common_pdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--page-limit", type=int, default=None)


def add_overview_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--analysis-chunk-chars", type=int, default=12000)
    parser.add_argument("--analysis-overlap-chars", type=int, default=800)


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"artifacts": {}, "steps": {}}


def cmd_extract(args: argparse.Namespace) -> int:
    context = extract_into_run(args.pdf_path, args.runs_dir, page_limit=args.page_limit)
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
            "paper_text": str(context.paths.paper_text),
            "metadata_json": str(context.paths.metadata_json),
            "sections_json": str(context.paths.sections_json),
        }
    )
    manifest["steps"]["extract"] = {
        "page_limit": args.page_limit,
        "status": "completed",
    }
    write_manifest(context, manifest)
    print(context.paths.root)
    return 0


def cmd_overview(args: argparse.Namespace) -> int:
    context = load_run(args.run_dir)
    llm_client = build_llm_client(args.backend, args.model)
    overview_result = generate_overview(
        context=context,
        llm_client=llm_client,
        analysis_chunk_chars=args.analysis_chunk_chars,
        analysis_overlap_chars=args.analysis_overlap_chars,
    )
    manifest = load_manifest(context.paths.manifest_json)
    manifest["run_id"] = context.run_id
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "overview_md": str(context.paths.overview_md),
        }
    )
    manifest["llm"] = {
        "backend": args.backend,
        "model": getattr(llm_client, "model", args.model),
    }
    manifest["steps"]["overview"] = {
        "status": "completed",
        **overview_result,
    }
    write_manifest(context, manifest)
    print(context.paths.overview_md)
    return 0


def cmd_narrate(args: argparse.Namespace) -> int:
    context = load_run(args.run_dir)
    llm_client = build_llm_client(args.backend, args.model)
    manifest = load_manifest(context.paths.manifest_json)
    manifest["run_id"] = context.run_id
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "spoken_overview_txt": str(context.paths.spoken_overview_txt),
        }
    )
    manifest["llm"] = {
        "backend": args.backend,
        "model": getattr(llm_client, "model", args.model),
    }

    spoken_result = generate_spoken_overview(
        context=context,
        llm_client=llm_client,
        target_minutes=args.target_minutes,
    )
    manifest["steps"]["spoken_overview"] = {
        "status": "completed",
        **spoken_result,
    }

    if not args.no_audio:
        synthesizer = SaySynthesizer(voice=args.voice, rate=args.rate)
        audio_result = synthesize_audio(context=context, synthesizer=synthesizer)
        manifest["tts"] = {
            "backend": "say",
            "voice": args.voice,
            "rate": args.rate,
        }
        manifest["artifacts"]["final_audio"] = str(context.paths.final_audio)
        manifest["steps"]["audio"] = {
            "status": "completed",
            **audio_result,
        }

    write_manifest(context, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_podcast(args: argparse.Namespace) -> int:
    context = load_run(args.run_dir)
    llm_client = build_llm_client(args.backend, args.model)
    manifest = load_manifest(context.paths.manifest_json)
    manifest["run_id"] = context.run_id
    manifest["artifacts"].update(
        {
            "run_dir": str(context.paths.root),
            "podcast_script_txt": str(context.paths.podcast_script_txt),
        }
    )
    manifest["llm"] = {
        "backend": args.backend,
        "model": getattr(llm_client, "model", args.model),
    }

    podcast_result = generate_podcast_script(
        context=context,
        llm_client=llm_client,
        target_minutes=args.target_minutes,
    )
    manifest["steps"]["podcast_script"] = {
        "status": "completed",
        **podcast_result,
    }

    if not args.no_audio:
        synthesizer = SayDialogueSynthesizer(
            host_voice=args.host_voice,
            cohost_voice=args.cohost_voice,
            host_rate=args.host_rate,
            cohost_rate=args.cohost_rate,
        )
        audio_result = synthesize_podcast_audio(context=context, synthesizer=synthesizer)
        manifest["tts"] = {
            "backend": "say-dialogue",
            "host_voice": args.host_voice,
            "cohost_voice": args.cohost_voice,
            "host_rate": args.host_rate,
            "cohost_rate": args.cohost_rate,
        }
        manifest["artifacts"]["podcast_audio"] = str(context.paths.podcast_audio)
        manifest["steps"]["podcast_audio"] = {
            "status": "completed",
            **audio_result,
        }

    write_manifest(context, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    context = extract_into_run(args.pdf_path, args.runs_dir, page_limit=args.page_limit)
    llm_client = build_llm_client(args.backend, args.model)
    manifest = {
        "run_id": context.run_id,
        "source_pdf": str(args.pdf_path.resolve()),
        "artifacts": {
            "run_dir": str(context.paths.root),
            "paper_text": str(context.paths.paper_text),
            "metadata_json": str(context.paths.metadata_json),
            "sections_json": str(context.paths.sections_json),
            "overview_md": str(context.paths.overview_md),
            "spoken_overview_txt": str(context.paths.spoken_overview_txt),
            "podcast_script_txt": str(context.paths.podcast_script_txt),
        },
        "llm": {
            "backend": args.backend,
            "model": getattr(llm_client, "model", args.model),
        },
        "steps": {
            "extract": {
                "page_limit": args.page_limit,
                "status": "completed",
            }
        },
    }

    overview_result = generate_overview(
        context=context,
        llm_client=llm_client,
        analysis_chunk_chars=args.analysis_chunk_chars,
        analysis_overlap_chars=args.analysis_overlap_chars,
    )
    manifest["steps"]["overview"] = {
        "status": "completed",
        **overview_result,
    }

    if args.mode in ("research-brief", "both"):
        spoken_result = generate_spoken_overview(
            context=context,
            llm_client=llm_client,
            target_minutes=args.target_minutes,
        )
        manifest["steps"]["spoken_overview"] = {
            "status": "completed",
            **spoken_result,
        }

        if not args.no_audio:
            synthesizer = SaySynthesizer(voice=args.voice, rate=args.rate)
            audio_result = synthesize_audio(context=context, synthesizer=synthesizer)
            manifest.setdefault("tts", {})
            manifest["tts"]["research_brief"] = {
                "backend": "say",
                "voice": args.voice,
                "rate": args.rate,
            }
            manifest["artifacts"]["final_audio"] = str(context.paths.final_audio)
            manifest["steps"]["audio"] = {
                "status": "completed",
                **audio_result,
            }

    if args.mode in ("podcast", "both"):
        podcast_result = generate_podcast_script(
            context=context,
            llm_client=llm_client,
            target_minutes=max(args.target_minutes, 4),
        )
        manifest["steps"]["podcast_script"] = {
            "status": "completed",
            **podcast_result,
        }

        if not args.no_audio:
            podcast_synthesizer = SayDialogueSynthesizer(
                host_voice=args.host_voice,
                cohost_voice=args.cohost_voice,
                host_rate=args.host_rate,
                cohost_rate=args.cohost_rate,
            )
            podcast_audio_result = synthesize_podcast_audio(context=context, synthesizer=podcast_synthesizer)
            manifest.setdefault("tts", {})
            manifest["tts"]["podcast"] = {
                "backend": "say-dialogue",
                "host_voice": args.host_voice,
                "cohost_voice": args.cohost_voice,
                "host_rate": args.host_rate,
                "cohost_rate": args.cohost_rate,
            }
            manifest["artifacts"]["podcast_audio"] = str(context.paths.podcast_audio)
            manifest["steps"]["podcast_audio"] = {
                "status": "completed",
                **podcast_audio_result,
            }

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
        if args.command == "narrate":
            return cmd_narrate(args)
        if args.command == "podcast":
            return cmd_podcast(args)
        if args.command == "run":
            return cmd_run(args)
    except (LLMError, TTSError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
