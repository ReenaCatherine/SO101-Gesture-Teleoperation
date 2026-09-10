import mujoco
import numpy as np

JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],   # J1
    [-1.74533,  1.74533],   # J2
    [-1.74533,  1.57080],   # J3
    [-1.65806,  1.65806],   # J4
    [-2.79253,  2.79253],   # J5
])

from build_scene import build_model


model, _ = build_model()
data = mujoco.MjData(model)

jaw_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "jaw"
)

print("SO-101 IK test")
print("Jaw body ID:", jaw_id)

def solve_ik(target_xyz, max_iterations = 10):

    mujoco.mj_forward(model, data)

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

        # Stop if the jaw is close enough to the target
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

        new_q = data.qpos[:5].copy() + step_size * dq

        # Apply actual SO-101 joint limits
        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )

        data.qpos[:5] = new_q

        mujoco.mj_forward(model, data)

    return data.qpos[:5].copy()

targets = [
    [0.05, -0.25, 0.25],
    [0.06, -0.26, 0.25],
    [0.07, -0.27, 0.25],
    [0.08, -0.28, 0.25],
    [0.09, -0.29, 0.25],
    [0.10, -0.30, 0.25]
]

for target in targets:

    print("\n==============================")
    print("Target XYZ:", target)

    joint_positions = solve_ik(target, max_iterations = 1)

    print("Final joint positions (rad):")
    print(joint_positions)

    final_position = data.xpos[jaw_id]

    final_error = [
        target[i] - final_position[i]
        for i in range(3)
    ]

    print("Final jaw position:", final_position)
    print("Final position error:", final_error)
    print("Final error norm:", np.linalg.norm(final_error))

