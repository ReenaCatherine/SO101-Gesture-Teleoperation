import mujoco
import mujoco.viewer
import numpy as np

from build_scene import build_model

model, _ = build_model()
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],
    [-1.74533,  1.74533],
    [-1.74533,  1.57080],
    [-1.65806,  1.65806],
    [-2.79253,  2.79253],
])
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
print("SO-101 IK → MuJoCo test")
print("Jaw ID:", jaw_id)
print("Actuator IDs:", actuator_ids)

def solve_ik(target_xyz, max_iterations=10):

    mujoco.mj_forward(model, data)
    ik_q = data.qpos[:5].copy()

    step_size = 0.2

    for iteration in range(max_iterations):

        current = data.xpos[jaw_id]

        error = [
            target_xyz[i] - current[i]
            for i in range(3)
        ]

        error_vector = np.array(error)
        error_norm = np.linalg.norm(error_vector)

        print(f"\nIteration {iteration + 1}")
        print("Current XYZ:", current)
        print("Position error:", error)
        print("Error norm:", error_norm)

        if error_norm < 0.005:
            print("IK converged!")
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

        new_q = ik_q + step_size * dq

        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )
        ik_q = new_q.copy()

        data.qpos[:5] = new_q

        mujoco.mj_forward(model, data)

    return data.qpos[:5].copy()

targets = [
    [0.05, -0.25, 0.25],
    [0.10, -0.25, 0.25],
    [0.10, -0.30, 0.25],
    [0.05, -0.30, 0.25],
    [0.05, -0.25, 0.25]
]
for target in targets:

    print("\n==============================")
    print("Target:", target)

    joint_positions = solve_ik(
        target,
        max_iterations=1
    )

    print("IK joint positions:")
    print(joint_positions)

    for i in range(5):
        data.ctrl[actuator_ids[i + 1]] = joint_positions[i]

    for _ in range(100):
        mujoco.mj_step(model, data)

    final_position = data.xpos[jaw_id]

    final_error = np.linalg.norm(
        np.array(target) - final_position
    )

    print("Final jaw position:")
    print(final_position)

    print("Target:")
    print(target)

    print("Final position error:")
    print(final_error)


