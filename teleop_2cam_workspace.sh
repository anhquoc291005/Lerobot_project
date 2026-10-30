#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LEROBOT_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_DIR/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.calibration_dir="$PROJECT_DIR/calibration" \
  --robot.cameras="{top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, left: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --teleop.calibration_dir="$PROJECT_DIR/calibration" \
  --fps=30 \
  --display_data=true \
  --display_workspace="$PROJECT_DIR/workspace_boundary.json" \
  "$@"
