import socket
import time
import threading

import mujoco
import mujoco.viewer
import numpy as np

from build_scene import build_model


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

    if gripper_closed:

        return GRIPPER_MAX

    else:

        return GRIPPER_MIN


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

    return data.qpos[
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

                joint_positions = solve_ik(
                    target,
                    max_iterations=3
                )

                # -----------------------------------------------
                # Arm actuators J1-J5
                # -----------------------------------------------

                for i in range(5):

                    actuator_id = (
                        actuator_ids[i + 1]
                    )

                    data.ctrl[
                        actuator_id
                    ] = joint_positions[i]


            # ====================================================
            # JOINT MODE
            # ====================================================

            elif mode == "JOINT":

                # -----------------------------------------------
                # Directly command J1-J5
                # -----------------------------------------------

                for i in range(5):

                    actuator_id = (
                        actuator_ids[i + 1]
                    )

                    data.ctrl[
                        actuator_id
                    ] = joints[i]


            # ====================================================
            # GRIPPER
            # ====================================================

            data.ctrl[
                GRIPPER_ACTUATOR_ID
            ] = get_gripper_command()


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

    print(
        "\nReceiver closed."
    )
