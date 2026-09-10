import mujoco
from build_scene import build_model

J6_MIN = -0.174533
J6_MAX = 1.74533
model, _ = build_model()
data = mujoco.MjData(model)

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "6"
)

qpos_addr = model.jnt_qposadr[joint_id]

print("J6 test")
print("Joint ID:", joint_id)
print("Initial J6:", data.qpos[qpos_addr])

# Test position 1
data.qpos[qpos_addr] = max(J6_MIN, min(J6_MAX, 0.0))
mujoco.mj_forward(model, data)

print("J6 = 0.0 rad")
print("Actual:", data.qpos[qpos_addr])

# Test position 2
data.qpos[qpos_addr] = max(J6_MIN, min(J6_MAX, 1.0))
mujoco.mj_forward(model, data)

print("J6 = 1.0 rad")
print("Actual:", data.qpos[qpos_addr])

# Test position 3
data.qpos[qpos_addr] = max(J6_MIN, min(J6_MAX, 2.0))
mujoco.mj_forward(model, data)

print("J6 requested = 2.0 rad")
print("Actual:", data.qpos[qpos_addr])
