#!/usr/bin/env bash
# SO-101 gesture teleop: webcam + MediaPipe hand tracking -> TCP to the receiver.
# Start the receiver (run_receiver.sh) BEFORE this.
#
# Usage:
#   ./run_controller.sh                          # uses CAMERA below
#   SO101_CAMERA=1 ./run_controller.sh           # one-off override (index or /dev/videoN)
#   SO101_HOST=192.168.1.50 ./run_controller.sh  # receiver on another machine
#   SO101_VENV=/path/to/venv ./run_controller.sh
#
# Works from any location: the repo is found relative to this script.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configure here ---------------------------------------------------
CAMERA=0                                  # camera index (0 -> /dev/video0) or a device path
RECEIVER_HOST=127.0.0.1                   # same machine as the receiver
VENV="${SO101_VENV:-$HOME/ros2_ws/.venv-controller}"   # venv with mediapipe + opencv
# -----------------------------------------------------------------------

export SO101_CAMERA="${SO101_CAMERA:-$CAMERA}"
export SO101_HOST="${SO101_HOST:-$RECEIVER_HOST}"

[ -x "$VENV/bin/python" ] || { echo "Missing venv python at $VENV/bin/python (set SO101_VENV)" >&2; exit 1; }

cd "$REPO/linux_controller" || exit 1
exec "$VENV/bin/python" combined_controller.py "$@"
