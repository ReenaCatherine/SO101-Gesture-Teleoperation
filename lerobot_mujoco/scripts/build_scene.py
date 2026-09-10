#!/usr/bin/env python3
"""
Builds the SO101 MuJoCo model from this package's own urdf/ + meshes/ --
a local copy of lerobot_description's geometry (see urdf/so101.urdf's
header), kept here so lerobot_mujoco has no runtime dependency on
lerobot_description, xacro, or a sourced ROS 2 workspace beyond
ament_index_python for locating its own share directory.

Steps: rewrite package:// mesh URIs to real filesystem paths (MuJoCo's URDF
importer doesn't understand ROS package URIs) -> load with mujoco.MjSpec ->
add position actuators (URDF <transmission> tags aren't picked up by
MuJoCo's URDF importer) plus a floor/lights/camera -> compile.

Joint names "1".."6" and the "arm"/"gripper" grouping match
lerobot_moveit/config/so101.srdf so pose_loop_mujoco.py can replay the same
named poses used by the MoveIt forward-kinematics demo (that dependency is
unavoidable -- the poses have to come from somewhere -- but it's a data
file read at runtime, not something baked into this package).
"""
import os
import tempfile

import mujoco
from ament_index_python.packages import get_package_share_directory

# joint -> (kp, kv) position-servo gains. The gripper joint (6) is lighter
# and needs less gain than the arm joints to avoid oscillating against the
# jaw's tighter range.
ACTUATOR_GAINS = {
    "1": (60.0, 4.0),
    "2": (60.0, 4.0),
    "3": (60.0, 4.0),
    "4": (30.0, 2.0),
    "5": (15.0, 1.0),
    "6": (5.0, 0.3),
}

ARM_JOINTS = ["1", "2", "3", "4", "5"]
GRIPPER_JOINTS = ["6"]


def _local_urdf_with_real_mesh_paths():
    mujoco_share = get_package_share_directory("lerobot_mujoco")
    urdf_file = os.path.join(mujoco_share, "urdf", "so101.urdf")

    with open(urdf_file) as f:
        urdf_xml = f.read()

    # MuJoCo's URDF importer resolves mesh filenames as plain filesystem
    # paths, so package:// URIs (which only ROS resolves) must become real
    # paths before MuJoCo ever sees the file.
    urdf_xml = urdf_xml.replace(
        "package://lerobot_mujoco/meshes",
        os.path.join(mujoco_share, "meshes"),
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False, prefix="so101_mujoco_"
    )
    tmp.write(urdf_xml)
    tmp.close()
    return tmp.name


def build_spec():
    """Returns a mujoco.MjSpec for the SO101 arm: robot geometry from the
    URDF plus a floor, lights, a camera, and position actuators."""
    urdf_path = _local_urdf_with_real_mesh_paths()
    try:
        spec = mujoco.MjSpec.from_file(urdf_path)
    finally:
        os.unlink(urdf_path)

    # implicitfast: the links are light (~0.1 kg) with short lever arms, so
    # holding them at kp~60 N*m/rad is numerically stiff for explicit Euler
    # at a usable timestep -- it just oscillates instead of settling.
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.degree = False

    # The URDF's visual and collision geoms are the same non-convex
    # 3D-printed-part meshes, designed to mate with small overlaps. MuJoCo
    # approximates each as a single convex hull for contacts, so adjacent
    # links (e.g. jaw against wrist_roll_follower) register large false
    # penetrations and the PD-held pose explodes. This is a kinematics/pose
    # demo, not a contact-dynamics one, so disable self-collision on every
    # imported robot geom; gravity + joint limits + the position servos are
    # still fully simulated.
    # contype=0 alone is enough: a contact needs (A.contype & B.conaffinity)
    # or (B.contype & A.conaffinity) nonzero, so zeroing contype on every
    # robot geom means none of them can ever be the colliding side against
    # each other, regardless of conaffinity. (Zeroing conaffinity too is
    # redundant for that purpose and, as of this mujoco version, silently
    # drops every mesh geom out of the compiled model -- verified by
    # comparing model.ngeom before/after; contype-only keeps all geoms.)
    for geom in spec.geoms:
        geom.contype = 0

    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=512, height=512,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2],
    )
    spec.add_material(name="grid").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    grid_mat = spec.material("grid")
    grid_mat.texrepeat = [5, 5]
    grid_mat.reflectance = 0.2

    floor = spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.05], pos=[0, 0, 0], material="grid",
    )
    # Same reasoning as the robot geoms above: some named poses (e.g.
    # "vertical", elbow bent to 1.4 rad) bring the arm mesh close to z=0,
    # and a solid floor would add real contact forces to what's meant to be
    # a pure joint-space pose demo. contype=0 is a visual reference plane
    # only, same as the grid on it.
    floor.contype = 0
    spec.worldbody.add_light(name="top_light", pos=[0.3, 0, 1.2], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_camera(
        name="front", pos=[0.9, -0.9, 0.7], xyaxes=[0.7, 0.7, 0, -0.3, 0.3, 0.9],
    )

    for name in ARM_JOINTS + GRIPPER_JOINTS:
        joint = spec.joint(name)
        kp, kv = ACTUATOR_GAINS[name]
        act = spec.add_actuator(
            name=f"actuator_{name}",
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        act.gainprm[0] = kp
        act.biasprm[0] = 0.0
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv
        act.ctrlrange = joint.range
        act.ctrllimited = True

    return spec


def build_model():
    spec = build_spec()
    model = spec.compile()
    return model, spec


def main():
    """Writes the assembled scene as a standalone MJCF file so it can be
    inspected or opened directly in `python -m mujoco.viewer`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/so101_scene.xml")
    args = parser.parse_args()

    _, spec = build_model()
    with open(args.out, "w") as f:
        f.write(spec.to_xml())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
