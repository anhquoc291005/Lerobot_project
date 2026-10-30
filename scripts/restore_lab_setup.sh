#!/usr/bin/env bash
# Restore source only. Does not connect to cameras or move a robot.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$PROJECT_DIR/lerobot"
PATCH="$PROJECT_DIR/patches/lerobot-local.patch"
BASE="$(cat "$PROJECT_DIR/patches/lerobot-base-commit.txt")"

if [[ ! -e "$SOURCE" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$SOURCE"
fi
if [[ "$(git -C "$SOURCE" rev-parse --show-toplevel)" != "$SOURCE" ]]; then
  echo "Expected a separate LeRobot checkout at $SOURCE" >&2
  exit 1
fi

if [[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$BASE" && -s "$PATCH" ]] &&
   git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "This LeRobot setup is already restored."
  exit 0
fi
if [[ -n "$(git -C "$SOURCE" status --porcelain)" ]]; then
  echo "LeRobot has local changes. Save them before restoring; nothing was overwritten." >&2
  exit 1
fi
if ! git -C "$SOURCE" cat-file -e "$BASE^{commit}" 2>/dev/null; then
  git -C "$SOURCE" fetch origin "$BASE"
fi
git -C "$SOURCE" checkout --detach "$BASE"
if [[ -s "$PATCH" ]]; then
  git -C "$SOURCE" apply --check "$PATCH"
  git -C "$SOURCE" apply "$PATCH"
fi
echo "Restored LeRobot source. Follow HUONG_DAN_CHUYEN_SETUP_SANG_LAB.md to install dependencies."
