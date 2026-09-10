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

**Not yet completed.**

The current receiver controls the **MuJoCo simulation**, not the physical SO-101. The physical hardware interface, calibration, hardware-specific joint limits, and final hardware testing must be completed separately.

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

# Hardware Handover

The current project is a **simulation baseline** for the physical SO-101 implementation.

The intended hardware architecture is:

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

The following parts need to be implemented/validated for hardware:

- SO-101 calibration
- Physical joint limits
- Joint direction verification
- Hardware communication through LeRobot
- Hardware-safe workspace
- Low-speed testing
- Physical collision/safety validation
- Final latency and smoothness tuning

The current `gesture_receiver_test.py` should therefore be treated as a **MuJoCo simulation receiver**, not a finished hardware controller.

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
