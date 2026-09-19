# SO-101 Gesture-Controlled Teleoperation

This project implements gesture-based teleoperation of a LeRobot SO-101 arm using:

- Windows webcam
- MediaPipe hand tracking
- Python controller
- TCP communication
- ROS 2 Humble
- MuJoCo simulation
- SO-101 URDF and meshes
- Cartesian control with inverse kinematics

## System Architecture

```text
Windows
  |
  | Webcam
  v
MediaPipe Hand Tracking
  |
  v
combined_controller.py
  |
  | TCP (XYZ commands)
  v
WSL / Ubuntu
  |
  v
gesture_receiver_test.py
  |
  | XYZ -> IK -> joint positions
  v
MuJoCo
  |
  v
SO-101 simulated arm
```

## Folder Structure

```text
SO101_Gesture_Teleoperation/
├── README.md
│
├── windows_controller/
│   ├── combined_controller.py
│   └── hand_landmarker.task
│
└── lerobot_mujoco/
    ├── CMakeLists.txt
    ├── package.xml
    ├── config/
    │   └── so101.srdf
    ├── meshes/
    │   └── so101/
    │       └── *.stl
    ├── scripts/
    │   ├── build_scene.py
    │   ├── gesture_mujoco.py
    │   ├── gesture_receiver_test.py
    │   ├── ik_mujoco_test.py
    │   ├── ik_test.py
    │   ├── j6_test.py
    │   ├── mirror_real_arm.py
    │   ├── pose_loop_mujoco.py
    │   └── view_so101.py
    └── urdf/
        └── so101.urdf
```

## Current Status

### Completed

- Real-time webcam hand tracking using MediaPipe
- Gesture recognition
- Keyboard teleoperation
- TCP communication between Windows and WSL
- SO-101 URDF and meshes in MuJoCo
- XYZ target reception
- Basic inverse-kinematics control
- MuJoCo SO-101 teleoperation simulation
- Emergency-stop logic
- Basic workspace/joint-limit constraints

### Hardware Integration

The receiver can now drive the **physical SO-101** alongside the MuJoCo
simulation, through `lerobot_hardware`'s `FeetechBus` (serial, Feetech
STS3215 servos).

- `gesture_receiver_test.py --real-hardware` sends every computed joint
  target to the real servos in addition to the sim actuators.
- **Dry-run is the default.** `--real-hardware` on its own connects
  read-only — no torque, no servo writes — and only logs the ticks it
  would have sent. Add `--no-dry-run` to actually move the arm.
- `--serial-port` selects the device (default `/dev/ttyACM0`).
- **Safe start.** On a live connect, each servo's goal is first set to the
  arm's actual pose (so torque engaging can't lurch it), and targets sent to
  the real servos are slew-limited to 0.25 rad/s starting from that pose.
  The arm therefore eases from wherever it rests to the first gesture target
  (about 11 s from the folded rest pose) instead of jumping. The sim moves
  instantly, so it leads the real arm until the log prints
  `startup ramp done`.
- `mirror_real_arm.py` is a read-only digital twin: it subscribes to
  `/joint_states` from `lerobot_hardware`'s `so101_hardware_bridge.py` and
  mirrors the real arm's pose into the viewer. It never writes to the arm,
  so it is safe to run alongside a live bridge.

Joint limits, calibration and zero offsets come from the servos' own
flashed calibration, read by `FeetechBus` at connect time; targets outside
that range raise `ClampExceeded` and are skipped rather than clamped
silently. Speed/smoothness tuning and physical collision validation are
still open.

---

# Prerequisites

## Windows

Install:

- Python 3
- OpenCV
- MediaPipe
- NumPy

The Windows controller also requires the MediaPipe hand-landmarker model:

```text
hand_landmarker.task
```

This file is already included in:

```text
windows_controller/
```

## Ubuntu / WSL

Required:

- Ubuntu 22.04
- ROS 2 Humble
- Python 3
- MuJoCo
- `colcon`

---

# Setup

## 1. Create the ROS 2 workspace

In WSL / Ubuntu:

```bash
mkdir -p ~/so101_ws/src
cd ~/so101_ws/src
```

Copy or clone the `lerobot_mujoco` folder into:

```text
~/so101_ws/src/LeRobot_mujoco
```

Then:

```bash
cd ~/so101_ws
source /opt/ros/humble/setup.bash
colcon build
```

After a successful build:

```bash
source ~/so101_ws/install/setup.bash
```

You can verify the ROS package with:

```bash
ros2 pkg list | grep lerobot_mujoco
```

Expected output:

```text
lerobot_mujoco
```

---

# How to Run the Simulation

The simulation should be started **before** the Windows controller.

## Step 1 — Start the SO-101 receiver

Open a WSL / Ubuntu terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/so101_ws/install/setup.bash

cd ~/so101_ws/src/LeRobot_mujoco/scripts

LIBGL_ALWAYS_SOFTWARE=1 python3 gesture_receiver_test.py
```

Expected output:

```text
SO-101 XYZ receiver
Waiting for connection on port 5000...
```

Keep this terminal running.

The receiver:

- listens for TCP commands on port `5000`
- receives XYZ targets
- calculates joint positions using inverse kinematics
- sends those joint positions to the MuJoCo SO-101 model

### Optional — also drive the physical SO-101

To send the same joint targets to the real servos, add `--real-hardware`.
This is **dry-run by default**: it connects read-only and logs the ticks it
would send, without enabling torque.

```bash
# dry run — safe, nothing moves
LIBGL_ALWAYS_SOFTWARE=1 python3 gesture_receiver_test.py --real-hardware

# live — the physical arm will move
LIBGL_ALWAYS_SOFTWARE=1 python3 gesture_receiver_test.py \
    --real-hardware --no-dry-run --serial-port /dev/ttyACM0
```

Do the first live run with the arm clear of obstacles and a hand on the
power switch. Ctrl-C closes the bus and disables torque.

## Step 2 — Start the Windows controller

Open Windows Command Prompt:

```cmd
cd "C:\Users\LeRobot files\so101_gesture_control"
python combined_controller.py
```

Expected output:

```text
Connected to MuJoCo bridge
```

The webcam window should open.

## Step 3 — Teleoperate the simulated arm

Use the webcam to control the arm with hand movement.

Open-palm gesture is used to enable gesture control.

Keyboard controls can also be used.

---

# Keyboard Controls

| Key | Function |
|---|---|
| `W` | Move Y+ |
| `S` | Move Y− |
| `A` | Move X− |
| `D` | Move X+ |
| `↑` | Move Z+ |
| `↓` | Move Z− |
| `←` | Move X− |
| `→` | Move X+ |
| `C` | Cartesian mode |
| `J` | Joint mode |
| `+` | Increase speed |
| `-` | Decrease speed |
| `SPACE` | Emergency stop |
| `R` | Reset emergency stop |
| `Q` | Quit |

## Gesture Control

### Open Palm

Enables gesture-based control.

### Hand Movement

The hand position detected by MediaPipe is mapped to a Cartesian XYZ target.

The target is then transmitted to the Ubuntu receiver through TCP.

---

# Safety

The controller includes an emergency-stop function.

Press:

```text
SPACE
```

to disable motion.

Press:

```text
R
```

to clear the emergency-stop state.

**Important:** The current software safety limits are intended for the simulation. They must not be assumed to be safe physical limits for the real SO-101.

Before testing physical hardware:

1. Calibrate the robot.
2. Verify each joint direction.
3. Verify physical joint limits.
4. Use low speed.
5. Test small movements first.
6. Keep the emergency-stop procedure ready.
7. Never assume that the MuJoCo workspace limits are valid for the physical robot.

---

# System Data Flow

```text
                    WINDOWS
              ┌──────────────────┐
              │     Webcam       │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │    MediaPipe     │
              │ Hand Tracking    │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │ combined_        │
              │ controller.py    │
              └────────┬─────────┘
                       │
                       │ TCP / XYZ
                       ↓
                    WSL
              ┌──────────────────┐
              │ gesture_receiver │
              │ _test.py         │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │       IK         │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │     MuJoCo       │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │   SO-101 Arm     │
              │   Simulation     │
              └──────────────────┘
```

---

# Troubleshooting

## MuJoCo viewer crashes

If MuJoCo crashes when starting the graphical viewer, try:

```bash
LIBGL_ALWAYS_SOFTWARE=1 python3 gesture_receiver_test.py
```

instead of:

```bash
python3 gesture_receiver_test.py
```

## Windows cannot connect to the receiver

Check that:

1. The WSL receiver is running.
2. The receiver is listening on port `5000`.
3. The Windows controller uses the correct WSL IP address.

In `combined_controller.py`, check:

```python
ROBOT_HOST = "172.27.7.2"
ROBOT_PORT = 5000
```

The WSL IP address may be different on another computer.

## Stopping the receiver

If `Ctrl+C` does not stop the receiver, open another WSL terminal and run:

```bash
pkill -f gesture_receiver_test.py
```

---

# Hardware Integration Notes

The hardware path is wired up; what follows is the architecture it
implements and what still needs validating on the real arm.

```text
Windows Webcam
      ↓
MediaPipe
      ↓
Hand XYZ
      ↓
TCP
      ↓
Ubuntu Receiver
      ↓
Inverse Kinematics
      ↓
Joint Positions
      ↓
LeRobot Hardware API
      ↓
Physical SO-101
```

Implemented:

- Hardware communication through `lerobot_hardware`'s `FeetechBus`
- SO-101 calibration and physical joint limits, read from the servos'
  flashed calibration at connect time
- Dry-run mode as the default, so `--real-hardware` cannot move the arm
  by accident
- Arm and gripper written separately, so a gripper target outside the
  servo's calibrated range cannot abort the arm joints' write
- Gripper target pre-clamped to the real servo's live range, because the
  sim's URDF jaw-angle range is wider than the servo's flashed one

Still to validate on the real arm:

- Joint direction verification, J1–J6
- Gripper open/closed polarity — the sim's actuator direction and the
  real servo's are not guaranteed to agree; confirm in dry-run first
- Hardware-safe workspace bounds
- Low-speed testing
- Physical collision/safety validation
- Final latency and smoothness tuning

---

# Important Simulation Note

The current MuJoCo IK is a basic proof-of-concept. The XYZ-to-joint mapping and workspace still require refinement for physically realistic hardware operation.

The current simulation is intended to demonstrate the complete software pipeline and provide a starting point for hardware integration.

---

# Quick Start

For a quick simulation test:

### WSL / Ubuntu

```bash
source /opt/ros/humble/setup.bash
source ~/so101_ws/install/setup.bash
cd ~/so101_ws/src/LeRobot_mujoco/scripts
LIBGL_ALWAYS_SOFTWARE=1 python3 gesture_receiver_test.py
```

### Windows

Open another terminal:

```cmd
cd "C:\Users\LeRobot files\so101_gesture_control"
python combined_controller.py
```

**Always start the WSL receiver before starting the Windows controller.**

---

# Project Goal

The goal is to demonstrate a low-latency gesture-controlled and keyboard-controlled teleoperation pipeline for the LeRobot SO-101 robotic arm, with simulation providing the development and testing environment before physical hardware integration.
