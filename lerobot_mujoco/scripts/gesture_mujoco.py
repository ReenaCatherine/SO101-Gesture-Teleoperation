import mujoco
import numpy as np

from build_scene import build_model


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
JOINT_LIMITS = np.array([
    [-1.91986,  1.91986],
    [-1.74533,  1.74533],
    [-1.74533,  1.57080],
    [-1.65806,  1.65806],
    [-2.79253,  2.79253],
])
def solve_ik(target_xyz, max_iterations=1):
    mujoco.mj_forward(model, data)

    step_size = 0.2

    for iteration in range(max_iterations):

        current = data.xpos[jaw_id]

        error_vector = np.array(target_xyz) - current

        error_norm = np.linalg.norm(error_vector)

        print(f"IK iteration {iteration + 1}: error = {error_norm:.4f} m")

        if error_norm < 0.005:
            break

        jacobian = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model,
            data,
            jacobian,
            None,
            jaw_id
        )

        # We control J1-J5 for Cartesian positioning
        J = jacobian[:, :5]

        dq = np.linalg.pinv(J) @ error_vector

        new_q = data.qpos[:5].copy() + step_size * dq

        # Enforce the real SO-101 joint limits
        new_q = np.clip(
            new_q,
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1]
        )

        data.qpos[:5] = new_q

        mujoco.mj_forward(model, data)

    return data.qpos[:5].copy()

print("SO-101 Gesture → MuJoCo integration")
print("Jaw ID:", jaw_id)
print("Actuator IDs:", actuator_ids)
print("Initial jaw position:", data.xpos[jaw_id])

target = [0.05, -0.25, 0.25]

print("\nTarget:", target)

joint_positions = solve_ik(target, max_iterations=13)

print("IK joint positions:")
print(joint_positions)

for i in range(5):
    data.ctrl[actuator_ids[i + 1]] = joint_positions[i]

for _ in range(500):
    mujoco.mj_step(model, data)

final_position = data.xpos[jaw_id]

final_error = np.linalg.norm(
    np.array(target) - final_position
)

print("Final jaw position:")
print(final_position)

print("Final position error:")
print(final_error)
