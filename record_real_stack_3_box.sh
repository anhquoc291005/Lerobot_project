#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LEROBOT_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_DIR/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$PROJECT_DIR/record_real_stack_3_box.py" "$@"
