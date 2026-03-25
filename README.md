# arxiv-voice

Generate a rigorous paper overview and a local Dia podcast from a PDF.

## What It Does

- copies a source PDF into a run directory
- calls Claude Code headlessly to analyze `source.pdf` directly
- writes a high-signal `overview.md`
- calls Claude Code again to turn that overview into a two-speaker `podcast_script.txt`
- converts the script into Dia `[S1]` / `[S2]` dialogue
- generates local podcast audio with Dia2 on `mps`

There is no local text chunking pipeline and no raw extracted paper dump in the run artifacts.

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) or `pip`
- Claude Code CLI installed and logged in
- Apple Silicon Mac with MPS

Default Claude model: `sonnet`

## Quickstart

```bash
uv sync
uv run arxiv-voice run 2603.14473v1.pdf
```

Useful flags:

```bash
uv run arxiv-voice run 2603.14473v1.pdf --model sonnet --effort medium
uv run arxiv-voice run 2603.14473v1.pdf --no-audio
uv run arxiv-voice extract 2603.14473v1.pdf
uv run arxiv-voice overview runs/<run-id>
uv run arxiv-voice podcast runs/<run-id>
uv run arxiv-voice podcast runs/<run-id> --prefix-speaker-1 /path/to/s1.wav --prefix-speaker-2 /path/to/s2.wav
uv run arxiv-voice run 2603.14473v1.pdf --bare --model sonnet
```

## Claude Notes

- The app uses the Claude Code headless CLI pattern: `claude -p ... --output-format json`.
- It uses `--append-system-prompt` to keep Claude Code's default tool behavior while adding task instructions.
- It auto-approves only the tools needed for each step:
  - overview: `Bash,Read`
  - podcast script: `Read`
- `--bare` is supported, but on this machine it requires `ANTHROPIC_API_KEY` or an `apiKeyHelper`. A normal logged-in Claude CLI session works without `--bare`.

## Dia Notes

- Dia2 weights are downloaded locally from `nari-labs/Dia2-2B` on first audio generation.
- The official Dia2 repo is CUDA-first, but the runtime accepts `device="mps"`.
- On non-CUDA devices, Dia2 resolves `auto` precision to `float32`.
- Dia2 is capped at about 2 minutes per generation, so this project generates segment WAVs and concatenates them into one podcast file.

## Output Layout

Each run is written to `runs/<timestamp>-<paper-slug>/` and includes:

- `source.pdf`
- `overview.md`
- `podcast_script.txt`
- `podcast_dia.txt`
- `audio/segments/*.wav`
- `audio/podcast.wav`
- `manifest.json`
