import socket
import time
import threading

import mujoco
import mujoco.viewer
import numpy as np

from build_scene import build_model


# ============================================================
# NETWORK
# ============================================================

HOST = "0.0.0.0"
PORT = 5000


# ============================================================
# SO-101 JOINT LIMITS
# ============================================================

JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],
    [-1.74533,  1.74533],
    [-1.74533,  1.57080],
    [-1.65806,  1.65806],
    [-2.79253,  2.79253],
])


# ============================================================
# MUJOCO MODEL
# ============================================================

model, _ = build_model()

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# END-EFFECTOR BODY
# ============================================================

jaw_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "jaw"
)


# ============================================================
# ACTUATORS
# ============================================================

actuator_ids = {}

for i in range(1, 7):

    actuator_ids[i] = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"actuator_{i}"
    )


# ============================================================
# GRIPPER ACTUATOR LIMIT
# ============================================================

GRIPPER_MIN = model.actuator_ctrlrange[
    actuator_ids[6],
    0
]

GRIPPER_MAX = model.actuator_ctrlrange[
    actuator_ids[6],
    1
]

print(
    f"Gripper actuator range: "
    f"[{GRIPPER_MIN:.4f}, {GRIPPER_MAX:.4f}]"
)


# ============================================================
# SHARED CARTESIAN TARGET
#
# SO-101 jaw position at qpos =
# [0, 0, 0, 0, 0, 0]
#
# Measured from the actual MuJoCo model.
# ============================================================

latest_target = np.array([
    0.0394,
    -0.3009,
    0.2871
])


# ============================================================
# SHARED JOINT TARGET
#
# J1-J5 are in radians.
# ============================================================

latest_joint_command = np.zeros(5)


# ============================================================
# SHARED GRIPPER COMMAND
#
# Controller sends:
# 0 = open
# 1 = closed
#
# IMPORTANT:
# The MuJoCo actuator direction is reversed,
# so the values are converted before being
# sent to actuator 6.
# ============================================================

latest_gripper = 0


target_lock = threading.Lock()

running = True


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def solve_ik(
    target_xyz,
    max_iterations=3
):

    mujoco.mj_forward(
        model,
        data
    )

    step_size = 0.2

    for _ in range(max_iterations):

        current = data.xpos[
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
            data,
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
            data.qpos[:5].copy()
            + step_size * dq
        )

        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )

        data.qpos[:5] = new_q

        mujoco.mj_forward(
            model,
            data
        )

    return data.qpos[:5].copy()


# ============================================================
# TCP RECEIVER THREAD
# ============================================================

def receive_commands(connection):

    global latest_target
    global latest_joint_command
    global latest_gripper
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


                # =========================================
                # CARTESIAN COMMAND
                #
                # C,x,y,z,gripper
                # =========================================

                if message.startswith("C,"):

                    try:

                        values = message.split(",")

                        x = float(values[1])
                        y = float(values[2])
                        z = float(values[3])

                        gripper = int(values[4])

                        new_target = np.array([
                            x,
                            y,
                            z
                        ])

                        with target_lock:

                            latest_target = (
                                new_target
                            )

                            latest_gripper = (
                                gripper
                            )

                    except (
                        ValueError,
                        IndexError
                    ):

                        print(
                            "Invalid Cartesian command:",
                            message
                        )


                # =========================================
                # JOINT COMMAND
                #
                # J,j1,j2,j3,j4,j5,gripper
                #
                # Joint values are radians.
                # =========================================

                elif message.startswith("J,"):

                    try:

                        values = message.split(",")

                        new_joints = np.array([
                            float(values[1]),
                            float(values[2]),
                            float(values[3]),
                            float(values[4]),
                            float(values[5])
                        ])

                        gripper = int(
                            values[6]
                        )

                        # Apply joint safety limits
                        new_joints = np.clip(
                            new_joints,
                            JOINT_LIMITS[:, 0],
                            JOINT_LIMITS[:, 1]
                        )

                        with target_lock:

                            latest_joint_command = (
                                new_joints
                            )

                            latest_gripper = (
                                gripper
                            )

                    except (
                        ValueError,
                        IndexError
                    ):

                        print(
                            "Invalid Joint command:",
                            message
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


print(
    "SO-101 Gesture Teleoperation Receiver"
)

print(
    f"Waiting for connection on port {PORT}..."
)


connection, address = server.accept()

print(
    "Connected from:",
    address
)


# ============================================================
# START RECEIVER THREAD
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
            "Move your hand to teleoperate the SO-101."
        )


        # ====================================================
        # MAIN SIMULATION LOOP
        # ====================================================

        while (
            running
            and viewer.is_running()
        ):


            # -----------------------------------------------
            # GET LATEST COMMANDS
            # -----------------------------------------------

            with target_lock:

                target = (
                    latest_target.copy()
                )

                joint_command = (
                    latest_joint_command.copy()
                )

                gripper_command = (
                    latest_gripper
                )


            # -----------------------------------------------
            # DETERMINE CONTROL MODE
            #
            # C = Cartesian
            # J = Joint
            #
            # The controller protocol itself determines
            # which mode is active.
            # -----------------------------------------------

            # We keep track of the most recent command type
            # using a small variable outside the network thread.


            # -----------------------------------------------
            # JOINT COMMAND
            # -----------------------------------------------

            if joint_command is not None:

                pass


            # -----------------------------------------------
            # GRIPPER
            #
            # Controller:
            #   0 = OPEN
            #   1 = CLOSED
            #
            # MuJoCo actuator direction is reversed:
            #   CLOSED -> GRIPPER_MIN
            #   OPEN   -> GRIPPER_MAX
            # -----------------------------------------------

            if gripper_command == 1:

                data.ctrl[
                    actuator_ids[6]
                ] = GRIPPER_MIN

            else:

                data.ctrl[
                    actuator_ids[6]
                ] = GRIPPER_MAX


            # -----------------------------------------------
            # SIMULATION STEPS
            # -----------------------------------------------

            for _ in range(20):

                mujoco.mj_step(
                    model,
                    data
                )


            # -----------------------------------------------
            # UPDATE VIEWER
            # -----------------------------------------------

            viewer.sync()


            # -----------------------------------------------
            # SMALL DELAY
            # -----------------------------------------------

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
    except Exception:
        pass

    server.close()

    print(
        "Receiver closed."
    )
