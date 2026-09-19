import argparse
import os
import socket
import sys
import time
import threading

import mujoco
import mujoco.viewer
import numpy as np

from build_scene import build_model


# ============================================================
# CLI ARGS
#
# --real-hardware additionally writes each computed joint target
# to the physical arm's servos over serial (via lerobot_hardware's
# FeetechBus), on top of always driving the MuJoCo sim/viewer.
#
# Safe by default: --real-hardware alone still runs dry-run
# (connects read-only, no torque, no servo writes -- just logs the
# intended ticks). Pass --real-hardware --no-dry-run to actually
# move the arm.
# ============================================================

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--real-hardware",
    action="store_true",
    help="also drive the physical arm's servos, "
         "not just the MuJoCo sim"
)

arg_parser.add_argument(
    "--serial-port",
    default="/dev/ttyACM0",
    help="serial device for the real arm "
         "(only used with --real-hardware)"
)

arg_parser.add_argument(
    "--dry-run",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="with --real-hardware: connect read-only and never "
         "write to the servos (default). Pass --no-dry-run to "
         "move the real arm."
)

cli_args = arg_parser.parse_args()


# ============================================================
# TCP SERVER
# ============================================================

HOST = "0.0.0.0"
PORT = 5000


# ============================================================
# SO-101 JOINT LIMITS
# ============================================================

JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],   # J1
    [-1.74533,  1.74533],   # J2
    [-1.74533,  1.57080],   # J3
    [-1.65806,  1.65806],   # J4
    [-2.79253,  2.79253],   # J5
])


# ============================================================
# BUILD MUJOCO MODEL
# ============================================================

model, _ = build_model()

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# JAW BODY
# ============================================================

jaw_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "jaw"
)

if jaw_id < 0:
    raise RuntimeError(
        "Could not find MuJoCo body named 'jaw'."
    )


# ============================================================
# ACTUATORS
# ============================================================

actuator_ids = {}

for i in range(1, 7):

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"actuator_{i}"
    )

    if actuator_id < 0:

        raise RuntimeError(
            f"Could not find actuator_{i}"
        )

    actuator_ids[i] = actuator_id


print("\nDetected actuators:")

for i in range(1, 7):

    actuator_id = actuator_ids[i]

    ctrl_min = model.actuator_ctrlrange[
        actuator_id,
        0
    ]

    ctrl_max = model.actuator_ctrlrange[
        actuator_id,
        1
    ]

    print(
        f"actuator_{i}: "
        f"ctrl range = "
        f"[{ctrl_min:.4f}, {ctrl_max:.4f}]"
    )


# ============================================================
# GRIPPER RANGE
# ============================================================

GRIPPER_ACTUATOR_ID = actuator_ids[6]

GRIPPER_MIN = model.actuator_ctrlrange[
    GRIPPER_ACTUATOR_ID,
    0
]

GRIPPER_MAX = model.actuator_ctrlrange[
    GRIPPER_ACTUATOR_ID,
    1
]


# Start OPEN
gripper_closed = False


def get_gripper_command():

    # Closed maps to the MIN end of the ctrlrange and open to MAX.
    # This drives both the sim actuator and (via data.ctrl) the
    # real servo, so it is the single place to flip direction.
    if gripper_closed:

        return GRIPPER_MIN

    else:

        return GRIPPER_MAX


# ============================================================
# REAL HARDWARE (optional)
#
# lerobot_hardware's FeetechBus is joint-name-for-name and
# radian-for-radian compatible with this package's model (same
# URDF, same calibration reference), so the joint target computed
# for the MuJoCo actuators below is sent to it unchanged.
# ============================================================

bus = None

if cli_args.real_hardware:

    from ament_index_python.packages import get_package_prefix

    sys.path.insert(
        0,
        os.path.join(
            get_package_prefix("lerobot_hardware"),
            "lib",
            "lerobot_hardware"
        )
    )

    from feetech_bus import FeetechBus, ClampExceeded

    bus = FeetechBus(
        cli_args.serial_port,
        servo_ids=[1, 2, 3, 4, 5, 6]
    )

    if cli_args.dry_run:

        print(
            "REAL HARDWARE: dry-run -- connecting read-only, "
            "no servo writes."
        )

        bus.connect_read_only()

    else:

        print(
            f"REAL HARDWARE: LIVE on {cli_args.serial_port} -- "
            f"torque enabled, gesture targets will move the "
            f"physical arm."
        )

        # connect() switches torque on, and a servo's stored
        # Goal_Position can be stale (wherever it was last told to
        # go), which would make the arm lurch the instant torque
        # engages. So first set every goal to the pose the arm is
        # actually in, with torque still off, then connect.

        bus.connect_read_only()

        bus.write_positions_rad(
            bus.read_positions_rad()
        )

        bus.close()

        bus.connect()

    # ------------------------------------------------------------
    # STARTUP RAMP
    #
    # hw_cmd is the pose last commanded to the real servos. It
    # starts at the arm's ACTUAL pose (which is usually a folded
    # rest pose, nowhere near the first gesture target) and is
    # slew-limited toward each new target, so the arm eases into
    # tracking instead of swinging across its whole range on the
    # first frame. The same limit also smooths jumpy targets later.
    # ------------------------------------------------------------

    HW_MAX_SPEED = 0.25   # rad/s, about the servos' own velocity cap

    hw_cmd = dict(
        bus.read_positions_rad()
    )

    hw_reached_first_target = False

    hw_last_time = time.monotonic()

    print(
        "REAL HARDWARE: start pose (rad): "
        + "  ".join(
            f"J{i}={v:+.3f}" for i, v in hw_cmd.items()
        )
    )


# ============================================================
# INITIAL CARTESIAN TARGET
#
# This matches the actual SO-101 home/end-effector
# position we measured previously.
# ============================================================

latest_target = np.array([
    0.0394,
    -0.3009,
    0.2871
])


# ============================================================
# INITIAL JOINT TARGET
# ============================================================

latest_joints = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])


# ============================================================
# CONTROL MODE
# ============================================================

control_mode = "CARTESIAN"


# ============================================================
# THREADING
# ============================================================

target_lock = threading.Lock()

running = True


# ============================================================
# IK SOLVER
#
# The IK iterates on its own MjData so it never writes the live
# simulation's qpos. Teleporting qpos underneath the position
# servos fights the actuators and makes the arm jitter -- and that
# jitter would be handed straight to the real servos below. Here
# IK only produces a joint target; the servos in data.ctrl do all
# the actual moving.
# ============================================================

ik_data = mujoco.MjData(model)


def solve_ik(
    target_xyz,
    seed_q,
    max_iterations=12
):

    ik_data.qpos[:] = 0.0
    ik_data.qpos[:5] = seed_q
    ik_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        ik_data
    )

    step_size = 0.5

    for _ in range(max_iterations):

        current = ik_data.xpos[
            jaw_id
        ]

        error_vector = (
            np.array(target_xyz)
            - current
        )

        if np.linalg.norm(
            error_vector
        ) < 0.005:

            break

        jacobian = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacBody(
            model,
            ik_data,
            jacobian,
            None,
            jaw_id
        )

        J = jacobian[:, :5]

        dq = (
            np.linalg.pinv(J)
            @ error_vector
        )

        new_q = (
            ik_data.qpos[:5].copy()
            + step_size * dq
        )

        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )

        ik_data.qpos[:5] = new_q

        mujoco.mj_forward(
            model,
            ik_data
        )

    return ik_data.qpos[
        :5
    ].copy()


# ============================================================
# RECEIVE COMMANDS
# ============================================================

def receive_commands(connection):

    global latest_target
    global latest_joints
    global control_mode
    global gripper_closed
    global running

    receive_buffer = ""

    while running:

        try:

            data_received = connection.recv(
                1024
            )

            if not data_received:

                running = False
                break

            receive_buffer += (
                data_received.decode()
            )

            while "\n" in receive_buffer:

                message, receive_buffer = (
                    receive_buffer.split(
                        "\n",
                        1
                    )
                )

                message = message.strip()

                if not message:
                    continue


                # ==================================================
                # CARTESIAN COMMAND
                #
                # C,x,y,z,gripper
                # ==================================================

                if message.startswith("C,"):

                    parts = message.split(",")

                    if len(parts) != 5:

                        print(
                            "Invalid Cartesian command:",
                            message
                        )

                        continue

                    try:

                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])

                        gripper = int(
                            parts[4]
                        )

                    except ValueError:

                        print(
                            "Invalid Cartesian values:",
                            message
                        )

                        continue


                    new_target = np.array([
                        x,
                        y,
                        z
                    ])


                    with target_lock:

                        latest_target = (
                            new_target
                        )

                        control_mode = (
                            "CARTESIAN"
                        )

                        gripper_closed = (
                            gripper == 1
                        )


                # ==================================================
                # JOINT COMMAND
                #
                # J,j1,j2,j3,j4,j5,gripper
                # ==================================================

                elif message.startswith("J,"):

                    parts = message.split(",")

                    if len(parts) != 7:

                        print(
                            "Invalid joint command:",
                            message
                        )

                        continue

                    try:

                        new_joints = np.array([
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                            float(parts[5])
                        ])

                        gripper = int(
                            parts[6]
                        )

                    except ValueError:

                        print(
                            "Invalid joint values:",
                            message
                        )

                        continue


                    # ----------------------------------------------
                    # Safety clamp
                    # ----------------------------------------------

                    new_joints = np.clip(
                        new_joints,
                        JOINT_LIMITS[:, 0],
                        JOINT_LIMITS[:, 1]
                    )


                    with target_lock:

                        latest_joints = (
                            new_joints
                        )

                        control_mode = (
                            "JOINT"
                        )

                        gripper_closed = (
                            gripper == 1
                        )


                else:

                    print(
                        "Unknown command:",
                        message
                    )


        except Exception as e:

            print(
                "Receive error:",
                e
            )

            running = False

            break


# ============================================================
# TCP SERVER
# ============================================================

server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)

server.bind(
    (HOST, PORT)
)

server.listen(1)


print("\n====================================")
print("SO-101 Gesture Teleoperation Receiver")
print("====================================")
print(
    f"Waiting for connection on port {PORT}..."
)


# ============================================================
# ACCEPT WINDOWS CONTROLLER
# ============================================================

connection, address = server.accept()

print(
    "Connected from:",
    address
)


# ============================================================
# RECEIVE THREAD
# ============================================================

receiver_thread = threading.Thread(
    target=receive_commands,
    args=(connection,),
    daemon=True
)

receiver_thread.start()


# ============================================================
# MUJOCO VIEWER
# ============================================================

try:

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        print(
            "MuJoCo viewer started."
        )

        print(
            "Waiting for teleoperation commands..."
        )


        # ========================================================
        # MAIN SIMULATION LOOP
        # ========================================================

        while (
            running
            and viewer.is_running()
        ):

            # ----------------------------------------------------
            # Get latest command
            # ----------------------------------------------------

            with target_lock:

                target = (
                    latest_target.copy()
                )

                joints = (
                    latest_joints.copy()
                )

                mode = control_mode

                current_gripper_closed = (
                    gripper_closed
                )


            # ====================================================
            # CARTESIAN MODE
            # ====================================================

            if mode == "CARTESIAN":

                # -----------------------------------------------
                # IK turns the XYZ target into a J1-J5 target,
                # seeded from where the arm actually is now.
                # -----------------------------------------------

                joint_target = solve_ik(
                    target,
                    data.qpos[:5].copy()
                )

            else:

                # -----------------------------------------------
                # JOINT MODE: use the commanded J1-J5 as sent
                # -----------------------------------------------

                joint_target = joints


            # ====================================================
            # ARM ACTUATORS J1-J5
            #
            # Either way the joint target goes to the position
            # actuators; the servos move the arm to it.
            # ====================================================

            for i in range(5):

                data.ctrl[
                    actuator_ids[i + 1]
                ] = joint_target[i]


            # ====================================================
            # GRIPPER
            # ====================================================

            data.ctrl[
                GRIPPER_ACTUATOR_ID
            ] = get_gripper_command()


            # ====================================================
            # REAL HARDWARE
            #
            # The same joint target just sent to the sim actuators
            # is sent to the real servos too. dry_run keeps this a
            # log-only no-op unless --no-dry-run was passed.
            # ====================================================

            if bus is not None:

                # Written as two separate calls, not one dict, so a
                # gripper target outside its calibrated range (the
                # sim's jaw joint limit doesn't match the real
                # servo's flashed range) can't abort the arm
                # joints' write too.

                hw_now = time.monotonic()

                # capped so one long stall can't allow a big step
                max_step = HW_MAX_SPEED * min(
                    hw_now - hw_last_time,
                    0.1
                )

                hw_last_time = hw_now

                arm_targets_rad = {}

                for i in range(5):

                    arm_targets_rad[i + 1] = (
                        hw_cmd[i + 1]
                        + float(np.clip(
                            joint_target[i] - hw_cmd[i + 1],
                            -max_step,
                            max_step
                        ))
                    )

                try:

                    bus.write_positions_rad(
                        arm_targets_rad,
                        dry_run=cli_args.dry_run
                    )

                    hw_cmd.update(arm_targets_rad)

                except ClampExceeded as exc:

                    print(
                        "\nREAL HARDWARE target unreachable, "
                        "skipping:",
                        exc
                    )

                except Exception as exc:

                    print(
                        "\nREAL HARDWARE write failed:",
                        exc
                    )


                # Sent as-is (no mirroring around
                # GRIPPER_MIN/MAX -- that reflection had the real
                # servo closing when the sim opened and vice
                # versa). Pre-clamped to the real servo's own live
                # calibrated range so the sim's wider URDF
                # jaw-angle range doesn't trip
                # write_positions_rad's ClampExceeded abort:
                # closing as far as the real gripper mechanically
                # can is the desired behavior here, unlike an arm
                # joint mismatch, which should hard-stop instead.

                gripper_min_raw, gripper_max_raw = bus.limits[6]

                gripper_raw = max(
                    gripper_min_raw,
                    min(
                        gripper_max_raw,
                        bus.rad_to_raw(
                            6,
                            data.ctrl[GRIPPER_ACTUATOR_ID]
                        )
                    )
                )

                real_gripper_target = bus.raw_to_rad(
                    6,
                    gripper_raw
                )

                real_gripper_target = (
                    hw_cmd[6]
                    + float(np.clip(
                        real_gripper_target - hw_cmd[6],
                        -max_step,
                        max_step
                    ))
                )

                try:

                    bus.write_positions_rad(
                        {6: real_gripper_target},
                        dry_run=cli_args.dry_run
                    )

                    hw_cmd[6] = real_gripper_target

                except ClampExceeded as exc:

                    print(
                        "\nREAL HARDWARE gripper target "
                        "unreachable, skipping:",
                        exc
                    )

                except Exception as exc:

                    print(
                        "\nREAL HARDWARE gripper write failed:",
                        exc
                    )

                if not hw_reached_first_target and all(
                    abs(joint_target[i] - hw_cmd[i + 1]) < 0.01
                    for i in range(5)
                ):

                    hw_reached_first_target = True

                    print(
                        "\nREAL HARDWARE: startup ramp done, "
                        "now tracking targets."
                    )


            # ====================================================
            # STEP SIMULATION
            # ====================================================

            for _ in range(20):

                mujoco.mj_step(
                    model,
                    data
                )


            # ====================================================
            # UPDATE VIEWER
            # ====================================================

            viewer.sync()


            # ====================================================
            # CURRENT POSITION
            # ====================================================

            current_position = (
                data.xpos[jaw_id].copy()
            )


            # ====================================================
            # STATUS
            # ====================================================

            if mode == "CARTESIAN":

                error = np.linalg.norm(
                    target
                    - current_position
                )

                print(
                    f"\r"
                    f"Mode: CARTESIAN | "
                    f"Target: "
                    f"[{target[0]:.3f}, "
                    f"{target[1]:.3f}, "
                    f"{target[2]:.3f}] | "
                    f"Error: "
                    f"{error * 1000:.1f} mm | "
                    f"Gripper: "
                    f"{'CLOSED' if current_gripper_closed else 'OPEN'}",
                    end=""
                )

            else:

                print(
                    f"\r"
                    f"Mode: JOINT | "
                    f"J1: {joints[0]:.2f} | "
                    f"J2: {joints[1]:.2f} | "
                    f"J3: {joints[2]:.2f} | "
                    f"J4: {joints[3]:.2f} | "
                    f"J5: {joints[4]:.2f} | "
                    f"Gripper: "
                    f"{'CLOSED' if current_gripper_closed else 'OPEN'}",
                    end=""
                )


            time.sleep(
                0.01
            )


except KeyboardInterrupt:

    print(
        "\nStopping receiver..."
    )


finally:

    running = False

    try:

        connection.close()

    except:

        pass

    try:

        server.close()

    except:

        pass

    if bus is not None:

        try:

            bus.close()

        except Exception as exc:

            print(
                "\nREAL HARDWARE close failed:",
                exc
            )

    print(
        "\nReceiver closed."
    )
