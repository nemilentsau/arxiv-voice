#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_FILE="${1:-$ROOT_DIR/experiments/dia2/reasonable_length_input.txt}"
OUTPUT_FILE="${2:-$ROOT_DIR/experiments/dia2/reasonable_length_output.wav}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# Use the pinned upstream Dia2 checkout directly because the published package
# currently imports incorrectly in this environment.
export PYTHONPATH="$ROOT_DIR/.cache/dia2-runtime${PYTHONPATH:+:$PYTHONPATH}"

uv run python -m dia2.cli \
  --hf nari-labs/Dia2-2B \
  --input "$INPUT_FILE" \
  --device mps \
  --dtype bfloat16 \
  --cfg 2.0 \
  --temperature 0.8 \
  --verbose \
  "$OUTPUT_FILE"
