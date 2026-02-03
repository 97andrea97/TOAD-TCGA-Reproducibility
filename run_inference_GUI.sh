#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8765}"
python3 src_inference_GUI/local_GUI.py --port "$PORT"
