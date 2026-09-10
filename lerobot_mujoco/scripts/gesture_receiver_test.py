import socket
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np

from build_scene import build_model


HOST = "0.0.0.0"
PORT = 5000


# --------------------------------------------------
# JOINT LIMITS
# --------------------------------------------------

JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],
    [-1.74533,  1.74533],
    [-1.74533,  1.57080],
    [-1.65806,  1.65806],
    [-2.79253,  2.79253],
])


# --------------------------------------------------
# MUJOCO SETUP
# --------------------------------------------------

model, _ = build_model()

data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


jaw_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "jaw"
)


actuator_ids = {}

for i in range(1, 6):
    actuator_ids[i] = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"actuator_{i}"
    )


# --------------------------------------------------
# SHARED TARGET
# --------------------------------------------------

latest_target = np.array([0.0, 0.0, 0.15])

target_lock = threading.Lock()

running = True


# --------------------------------------------------
# IK SOLVER
# --------------------------------------------------

def solve_ik(target_xyz, max_iterations=3):

    mujoco.mj_forward(model, data)

    step_size = 0.2

    for _ in range(max_iterations):

        current = data.xpos[jaw_id]

        error_vector = np.array(target_xyz) - current

        if np.linalg.norm(error_vector) < 0.005:
            break

        jacobian = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            jacobian,
            None,
            jaw_id
        )

        J = jacobian[:, :5]

        dq = np.linalg.pinv(J) @ error_vector

        new_q = data.qpos[:5].copy() + step_size * dq

        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )

        data.qpos[:5] = new_q

        mujoco.mj_forward(model, data)

    return data.qpos[:5].copy()


# --------------------------------------------------
# TCP RECEIVE THREAD
# --------------------------------------------------

def receive_commands(connection):

    global latest_target
    global running

    receive_buffer = ""

    while running:

        try:

            data_received = connection.recv(1024)

            if not data_received:

                running = False
                break

            receive_buffer += data_received.decode()

            while "\n" in receive_buffer:

                message, receive_buffer = receive_buffer.split(
                    "\n",
                    1
                )

                message = message.strip()

                if not message:
                    continue

                try:

                    x, y, z = map(
                        float,
                        message.split(",")
                    )

                except ValueError:

                    print("Invalid XYZ:", message)

                    continue

                new_target = np.array([x, y, z])

                # ------------------------------------------
                # STORE ONLY THE LATEST TARGET
                # ------------------------------------------

                with target_lock:

                    latest_target = new_target

        except Exception as e:

            print("Receive error:", e)

            running = False

            break


# --------------------------------------------------
# TCP SERVER
# --------------------------------------------------

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


print("SO-101 XYZ receiver")
print(f"Waiting for connection on port {PORT}...")


connection, address = server.accept()

print("Connected from:", address)


# --------------------------------------------------
# START RECEIVE THREAD
# --------------------------------------------------

receiver_thread = threading.Thread(
    target=receive_commands,
    args=(connection,),
    daemon=True
)

receiver_thread.start()


# --------------------------------------------------
# MUJOCO VIEWER + MAIN ROBOT LOOP
# --------------------------------------------------

try:

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        print("MuJoCo viewer started.")
        print("Move your hand to teleoperate the SO-101.")

        while running and viewer.is_running():

            # ------------------------------------------
            # GET THE MOST RECENT TARGET
            # ------------------------------------------

            with target_lock:

                target = latest_target.copy()


            # ------------------------------------------
            # IK
            # ------------------------------------------

            joint_positions = solve_ik(
                target,
                max_iterations=3
            )


            # ------------------------------------------
            # SEND JOINT TARGETS TO MUJOCO
            # ------------------------------------------

            for i in range(5):

                data.ctrl[
                    actuator_ids[i + 1]
                ] = joint_positions[i]


            # ------------------------------------------
            # RUN MUJOCO
            # ------------------------------------------

            for _ in range(20):

                mujoco.mj_step(
                    model,
                    data
                )


            # ------------------------------------------
            # UPDATE VIEWER
            # ------------------------------------------

            viewer.sync()


            # ------------------------------------------
            # DISPLAY POSITION
            # ------------------------------------------

            current_position = data.xpos[jaw_id].copy()

            error = np.linalg.norm(
                target - current_position
            )


            print(
                f"Target: "
                f"[{target[0]:.3f}, "
                f"{target[1]:.3f}, "
                f"{target[2]:.3f}] | "
                f"Current: "
                f"[{current_position[0]:.3f}, "
                f"{current_position[1]:.3f}, "
                f"{current_position[2]:.3f}] | "
                f"Error: {error * 1000:.1f} mm"
            )


            # Small loop delay
            time.sleep(0.01)


except KeyboardInterrupt:

    print("\nStopping receiver...")


finally:

    running = False

    connection.close()

    server.close()

    print("Receiver closed.")
