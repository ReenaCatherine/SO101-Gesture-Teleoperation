#!/usr/bin/env python3
"""
Opens the SO101 MuJoCo model in an interactive viewer with no scripted
motion -- drag joints with the mouse (double-click a body, ctrl+right-drag
to apply a wrench), or pause the sim and set qpos directly from the
viewer's Watch/Joint panel. Useful for eyeballing the model built from the
lerobot_description URDF before running pose_loop_mujoco.py.

Usage:
    python3 view_so101.py
"""
import mujoco
import mujoco.viewer

from build_scene import build_model


def main():
    model, _ = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
