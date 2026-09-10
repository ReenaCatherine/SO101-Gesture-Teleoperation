#!/usr/bin/env python3
"""
MuJoCo counterpart to lerobot_moveit/scripts/pose_loop.py: cycles the SO101
arm through every named group_state in so101.srdf (home, pose_1, pose_2,
...), looping forever with no user input, but driven by MuJoCo's own
physics/position-servos instead of a MoveGroup action -- no ROS 2, no
|move_group, no controller stack required.

For each pose: sets the arm+gripper actuator targets, steps the simulation
until every joint settles within tolerance (or a timeout elapses), then
reports pass/fail exactly like the MoveIt version.

Usage:
    python3 pose_loop_mujoco.py                  # loop forever
    python3 pose_loop_mujoco.py --cycles 3        # stop after 3 full cycles
    python3 pose_loop_mujoco.py --pause 3.0       # seconds to hold each pose

Ctrl+C stops cleanly.
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
from ament_index_python.packages import get_package_share_directory

from build_scene import build_model

SRDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "so101.srdf")
POSITION_TOLERANCE = 0.05  # rad, matches pose_loop.py's real-hardware tolerance


def load_poses(srdf_path):
    """Returns [pose_name, ...] in file order and {(pose_name, group): {joint: value}}."""
    root = ET.parse(srdf_path).getroot()
    order = []
    poses = {}
    for gs in root.findall("group_state"):
        name, group = gs.get("name"), gs.get("group")
        if name not in order:
            order.append(name)
        poses[(name, group)] = {
            j.get("name"): float(j.get("value")) for j in gs.findall("joint")
        }
    return order, poses


def settle(model, data, viewer, targets, settle_timeout=5.0):
    """Steps the sim until every target joint is within tolerance or the
    timeout elapses. Returns {joint_name: final_position}."""
    for name, value in targets.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator_{name}")
        data.ctrl[aid] = value

    qadr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in targets
    }

    deadline = time.monotonic() + settle_timeout
    step_start = time.monotonic()
    while time.monotonic() < deadline:
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            # keep the viewer roughly real-time instead of running as fast as possible
            elapsed = time.monotonic() - step_start
            if model.opt.timestep - elapsed > 0:
                time.sleep(model.opt.timestep - elapsed)
            step_start = time.monotonic()

        errors = [abs(data.qpos[qadr[n]] - v) for n, v in targets.items()]
        if all(e <= POSITION_TOLERANCE for e in errors):
            break

    return {name: data.qpos[qadr[name]] for name in targets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5, help="0 = loop forever, default 5")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds to hold each pose")
    parser.add_argument("--no-viewer", action="store_true", help="run headless (no GUI window)")
    args = parser.parse_args()

    order, poses = load_poses(SRDF_PATH)
    print(f"Loaded {len(order)} poses from SRDF: {order}")

    model, _ = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    viewer = None if args.no_viewer else mujoco.viewer.launch_passive(model, data)

    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            for pose_name in order:
                print(f"\n=== cycle {cycle} -- pose '{pose_name}' ===")
                arm_targets = poses.get((pose_name, "arm"), {})
                gripper_targets = poses.get((pose_name, "gripper"), {})
                all_targets = {**arm_targets, **gripper_targets}

                print(f"targets -> {all_targets}")
                actuals = settle(model, data, viewer, all_targets)

                ok = True
                for name, target in all_targets.items():
                    actual = actuals[name]
                    error = abs(actual - target)
                    status = "OK" if error <= POSITION_TOLERANCE else "FAIL"
                    if status == "FAIL":
                        ok = False
                    print(f"  joint {name}: target={target:+.4f} actual={actual:+.4f} "
                          f"error={error:.4f} [{status}]")
                print(f"verify: {'PASS' if ok else 'FAIL'}")

                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
