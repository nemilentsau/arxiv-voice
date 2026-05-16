#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.cache/kokoro-venv}"

uv venv "$ENV_DIR" --python 3.12 --seed
source "$ENV_DIR/bin/activate"

if ! command -v espeak-ng >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install espeak-ng
  else
    echo "espeak-ng is required and Homebrew is not installed" >&2
    exit 1
  fi
fi

uv pip install \
  "torch>=2.8.0" \
  "kokoro==0.7.16" \
  "soundfile>=0.13.1" \
  "misaki[en]>=0.9.4"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps_available", torch.backends.mps.is_available())
print("mps_built", torch.backends.mps.is_built())
PY
