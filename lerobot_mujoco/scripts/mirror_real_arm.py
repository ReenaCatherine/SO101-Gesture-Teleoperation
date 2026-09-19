#!/usr/bin/env python3
"""
Live digital twin: mirrors the physical SO101 arm's joint positions into the
MuJoCo viewer in real time.

Subscribes to /joint_states -- published by lerobot_hardware's
so101_hardware_bridge.py at joint_state_rate (default 30 Hz) from the live
Feetech servos -- and writes each reported position straight into the
MuJoCo model's qpos, joint name-for-name ("1".."6", matching both this
package's URDF joints and the hardware bridge's JOINT_NAMES). No physics is
stepped and no actuator is driven: this is a pure kinematic mirror, so the
viewer always shows exactly the pose the real arm just reported, not a
PD-servo's chase of it.

This only reads from the real arm; it never writes anything back, so it's
safe to run alongside a live so101_hardware_bridge with dry_run:=False.

Usage:
    ros2 run lerobot_mujoco mirror_real_arm.py
    # bridge running under a namespace or remapped topic:
    ros2 run lerobot_mujoco mirror_real_arm.py --ros-args -r joint_states:=/so101/joint_states
"""
import threading
import time

import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from build_scene import build_model

JOINT_NAMES = ["1", "2", "3", "4", "5", "6"]
STALE_AFTER = 1.0  # seconds with no message before warning the mirror is frozen


class RealArmMirror(Node):

    def __init__(self):
        super().__init__("so101_mirror")
        self.lock = threading.Lock()
        self.latest = {}
        self.last_msg_time = None
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)

    def _on_joint_state(self, msg):
        with self.lock:
            for name, position in zip(msg.name, msg.position):
                if name in JOINT_NAMES:
                    self.latest[name] = position
            self.last_msg_time = time.monotonic()

    def snapshot(self):
        with self.lock:
            return dict(self.latest), self.last_msg_time


def main():
    rclpy.init()
    node = RealArmMirror()

    model, _ = build_model()
    data = mujoco.MjData(model)
    qadr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    }
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)
    print("waiting for /joint_states from the real arm "
          "(run lerobot_hardware's so101_hardware_bridge.py) ...")

    got_first_message = False
    was_stale = False
    try:
        while viewer.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            positions, last_msg_time = node.snapshot()

            if positions and not got_first_message:
                got_first_message = True
                print("receiving /joint_states -- mirroring the real arm.")

            if last_msg_time is not None:
                stale = (time.monotonic() - last_msg_time) > STALE_AFTER
                if stale and not was_stale:
                    print("warning: no /joint_states update in "
                          f"{STALE_AFTER}s -- mirror is frozen.")
                elif not stale and was_stale:
                    print("receiving /joint_states again.")
                was_stale = stale

            for name, position in positions.items():
                data.qpos[qadr[name]] = position

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
