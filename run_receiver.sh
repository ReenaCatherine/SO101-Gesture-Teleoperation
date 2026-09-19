#!/usr/bin/env bash
# SO-101 gesture teleop: receiver (TCP :5000 -> IK -> MuJoCo, optionally real arm).
# Start this FIRST.
#
# Usage:
#   ./run_receiver.sh                  # simulation only
#   ./run_receiver.sh --real-hardware  # dry run on real arm (nothing moves)
#   ./run_receiver.sh --real-hardware --no-dry-run --serial-port /dev/ttyACM0   # LIVE
# Any arguments are passed straight to gesture_receiver_test.py.
#
# Works from any location: the repo is found relative to this script.
# Override the ROS workspace with SO101_WS (default: ~/ros2_ws).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${SO101_WS:-$HOME/ros2_ws}"

# ROS setup scripts reference unset vars, so don't use `set -u` before sourcing.
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash" || { echo "Cannot source $WS/install/setup.bash (set SO101_WS)" >&2; exit 1; }

cd "$REPO/lerobot_mujoco/scripts" || exit 1
LIBGL_ALWAYS_SOFTWARE=1 exec python3 gesture_receiver_test.py "$@"
