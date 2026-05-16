#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${KOKORO_ENV_DIR:-$ROOT_DIR/.cache/kokoro-venv}"
INPUT_FILE="${1:-$ROOT_DIR/experiments/kokoro/simple_narrative.txt}"
OUTPUT_FILE="${2:-$ROOT_DIR/experiments/kokoro/simple_narrative.wav}"
VOICE="${VOICE:-af_heart}"

source "$ENV_DIR/bin/activate"
export PYTORCH_ENABLE_MPS_FALLBACK=1

python "$ROOT_DIR/scripts/run_kokoro_simple_narrative.py" \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --voice "$VOICE" \
  --lang-code a \
  --speed 1.0
