import cv2
import mediapipe as mp
import os
import socket
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

base_options = python.BaseOptions(
    model_asset_path=os.path.join(
        SCRIPT_DIR,
        "hand_landmarker.task"
    )
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = os.environ.get("SO101_CAMERA", "0")
CAMERA_INDEX = int(CAMERA_INDEX) if CAMERA_INDEX.isdigit() else CAMERA_INDEX

cap = cv2.VideoCapture(CAMERA_INDEX)

# A V4L2 device can open() successfully but still fail to actually deliver
# frames if the index is stale (e.g. re-enumerated after a USB replug), so
# confirm with a real read, not just isOpened().
if not cap.isOpened() or not cap.read()[0]:
    print(f"ERROR: Could not open webcam at index/device {CAMERA_INDEX!r}.")
    print("Override with SO101_CAMERA=<index or /dev/videoN>.")
    raise SystemExit


# ============================================================
# CONTROL STATE
# ============================================================

mode = "CARTESIAN"

enabled = False
keyboard_control = False

# Emergency stop is latched.
# SPACE = stop
# R = reset
emergency_stop = False


# ============================================================
# SPEED
# ============================================================

speed = 50


# ============================================================
# CARTESIAN POSITION
#
# KEEPING YOUR ORIGINAL VALUES
# ============================================================

x = 0.039
y = -0.300
z = 0.287


# ============================================================
# JOINT POSITIONS
#
# J1-J5 = arm joints
# J6 = gripper state
# ============================================================

joints = [0.0, 0.0, 0.0, 0.0, 0.0]

gripper_closed = False


# ============================================================
# KEYBOARD STEP
# ============================================================

KEY_STEP = 0.16


# ============================================================
# ROBOT CONNECTION
# ============================================================

print("REACHED ROBOT CONNECTION CODE")

ROBOT_HOST = os.environ.get(
    "SO101_HOST",
    "127.0.0.1"
)
ROBOT_PORT = 5000

robot_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

try:

    robot_socket.connect(
        (ROBOT_HOST, ROBOT_PORT)
    )

    print("Connected to MuJoCo bridge")

except Exception as e:

    print("ERROR: Could not connect to MuJoCo bridge.")
    print(e)

    cap.release()
    detector.close()

    raise SystemExit


# ============================================================
# CARTESIAN SOFT LIMITS
#
# KEEPING YOUR ORIGINAL LIMITS
# ============================================================

X_MIN = -0.20
X_MAX = 0.20

Y_MIN = -0.32
Y_MAX = -0.06

Z_MIN = 0.05
Z_MAX = 0.30


# ============================================================
# SO-101 JOINT LIMITS
#
# radians
# ============================================================

JOINT_LIMITS = [
    (-1.91986,  1.91986),
    (-1.74533,  1.74533),
    (-1.74533,  1.57080),
    (-1.65806,  1.65806),
    (-2.79253,  2.79253),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(value, maximum)
    )


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def clamp_joint(index, value):

    minimum, maximum = JOINT_LIMITS[index]

    return clamp(
        value,
        minimum,
        maximum
    )


# ============================================================
# SEND CARTESIAN COMMAND
#
# C,x,y,z,gripper
#
# gripper:
# 0 = open
# 1 = closed
# ============================================================

def send_cartesian_command():

    gripper = (
        1
        if gripper_closed
        else 0
    )

    message = (
        f"C,"
        f"{x:.4f},"
        f"{y:.4f},"
        f"{z:.4f},"
        f"{gripper}\n"
    )

    try:

        robot_socket.sendall(
            message.encode()
        )

    except Exception as e:

        print(
            "Socket send error:",
            e
        )


# ============================================================
# SEND JOINT COMMAND
#
# J,j1,j2,j3,j4,j5,gripper
#
# Joint values are radians.
# ============================================================

def send_joint_command():

    gripper = (
        1
        if gripper_closed
        else 0
    )

    message = (
        f"J,"
        f"{joints[0]:.4f},"
        f"{joints[1]:.4f},"
        f"{joints[2]:.4f},"
        f"{joints[3]:.4f},"
        f"{joints[4]:.4f},"
        f"{gripper}\n"
    )

    try:

        robot_socket.sendall(
            message.encode()
        )

    except Exception as e:

        print(
            "Socket send error:",
            e
        )


# ============================================================
# GESTURE DETECTION
# ============================================================

def detect_gesture(landmarks):

    """
    OPEN PALM
        Enables gesture control.

    PINCH
        Closes gripper while control is active.

    FIST
        Stops gesture control.

    OTHER
        No action.
    """

    # --------------------------------------------------------
    # PINCH
    # --------------------------------------------------------

    thumb_index = distance(
        landmarks[4],
        landmarks[8]
    )

    if thumb_index < 0.08:

        return "PINCH"


    # --------------------------------------------------------
    # FINGER EXTENSION
    # --------------------------------------------------------

    index = (
        landmarks[8].y
        < landmarks[6].y
    )

    middle = (
        landmarks[12].y
        < landmarks[10].y
    )

    ring = (
        landmarks[16].y
        < landmarks[14].y
    )

    pinky = (
        landmarks[20].y
        < landmarks[18].y
    )

    count = sum([
        index,
        middle,
        ring,
        pinky
    ])


    # --------------------------------------------------------
    # OPEN PALM
    # --------------------------------------------------------

    if count >= 4:

        return "OPEN PALM"


    # --------------------------------------------------------
    # FIST
    # --------------------------------------------------------

    if count == 0:

        return "FIST"


    return "OTHER"


# ============================================================
# FPS / TIMESTAMP
# ============================================================

previous_time = time.time()

frame_timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read camera."
            )

            break


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # FRAME SIZE
        # ----------------------------------------------------

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # CONVERT BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        frame_timestamp_ms += 33


        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        result = detector.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )


        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        gesture = "NO HAND"

        wrist = None

        landmarks = None

        hand_depth = 0.0


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            wrist = landmarks[0]

            gesture = detect_gesture(
                landmarks
            )


            # ------------------------------------------------
            # WRIST
            # ------------------------------------------------

            wrist_x = int(
                wrist.x * w
            )

            wrist_y = int(
                wrist.y * h
            )

            cv2.circle(
                frame,
                (wrist_x, wrist_y),
                10,
                (0, 255, 0),
                -1
            )


            # ------------------------------------------------
            # DRAW HAND LANDMARKS
            # ------------------------------------------------

            for landmark in landmarks:

                px = int(
                    landmark.x * w
                )

                py = int(
                    landmark.y * h
                )

                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # HAND DEPTH
            # ------------------------------------------------

            hand_depth = distance(
                landmarks[0],
                landmarks[12]
            )


        # ====================================================
        # GESTURE ACTIONS
        # ====================================================

        if not emergency_stop:

            if gesture == "OPEN PALM":

                if not keyboard_control:

                    enabled = True

                gripper_closed = False


            elif gesture == "PINCH":

                if enabled or keyboard_control:

                    gripper_closed = True


            elif gesture == "FIST":

                enabled = False

                keyboard_control = False
            # ------------------------------------------------
            # OTHER / NO HAND
            # ------------------------------------------------

            else:

                pass


        # ====================================================
        # KEYBOARD INPUT
        # ====================================================

        key = cv2.waitKeyEx(1)


        # ====================================================
        # Q = QUIT
        # ====================================================

        if key == ord("q"):

            break


        # ====================================================
        # C = CARTESIAN MODE
        # ====================================================

        elif key == ord("c"):

            mode = "CARTESIAN"

            keyboard_control = False

            enabled = False

            print(
                "Mode changed to CARTESIAN"
            )


        # ====================================================
        # J = JOINT MODE
        # ====================================================

        elif key == ord("j"):

            mode = "JOINT"

            keyboard_control = False

            enabled = False

            print(
                "Mode changed to JOINT"
            )


        # ====================================================
        # SPEED UP
        # ====================================================

        elif key in [
            ord("+"),
            ord("=")
        ]:

            speed = min(
                100,
                speed + 5
            )

            print(
                f"Speed: {speed}%"
            )


        # ====================================================
        # SPEED DOWN
        # ====================================================

        elif key in [
            ord("-"),
            ord("_")
        ]:

            speed = max(
                5,
                speed - 5
            )

            print(
                f"Speed: {speed}%"
            )


        # ====================================================
        # SPACE = EMERGENCY STOP
        # ====================================================

        elif key == 32:

            emergency_stop = True

            enabled = False

            keyboard_control = False

            print(
                "!!! EMERGENCY STOP !!!"
            )


        # ====================================================
        # R = RESET
        # ====================================================

        elif key in [
            ord("r"),
            ord("R")
        ]:

            emergency_stop = False

            enabled = False

            keyboard_control = False

            print(
                "Emergency stop reset. "
                "Show OPEN PALM to enable control."
            )


        # ====================================================
        # O = GRIPPER OPEN
        # ====================================================

        elif key == ord("o"):

            gripper_closed = False

            print(
                "Gripper: OPEN"
            )


        # ====================================================
        # P = GRIPPER CLOSE
        # ====================================================

        elif key == ord("p"):

            gripper_closed = True

            print(
                "Gripper: CLOSED"
            )


        # ====================================================
        # CARTESIAN KEYBOARD CONTROL
        # ====================================================

        elif (
            mode == "CARTESIAN"
            and not emergency_stop
        ):

            step = (
                KEY_STEP
                * (speed / 50)
            )

            keyboard_move = False


            # ------------------------------------------------
            # W / S -> Y
            # ------------------------------------------------

            if key == ord("w"):

                y += step

                keyboard_move = True


            elif key == ord("s"):

                y -= step

                keyboard_move = True


            # ------------------------------------------------
            # A / D -> X
            # ------------------------------------------------

            elif key == ord("a"):

                x -= step

                keyboard_move = True


            elif key == ord("d"):

                x += step

                keyboard_move = True


            # ------------------------------------------------
            # UP / DOWN -> Z
            # ------------------------------------------------

            elif key in [
                82,
                2490368
            ]:

                z += step

                keyboard_move = True


            elif key in [
                84,
                2621440
            ]:

                z -= step

                keyboard_move = True


            # ------------------------------------------------
            # LEFT / RIGHT -> X
            # ------------------------------------------------

            elif key in [
                81,
                2424832
            ]:

                x -= step

                keyboard_move = True


            elif key in [
                83,
                2555904
            ]:

                x += step

                keyboard_move = True


            # ------------------------------------------------
            # Activate keyboard control
            # ------------------------------------------------

            if keyboard_move:

                keyboard_control = True

                enabled = False


            # ------------------------------------------------
            # Cartesian soft limits
            # ------------------------------------------------

            x = clamp(
                x,
                X_MIN,
                X_MAX
            )

            y = clamp(
                y,
                Y_MIN,
                Y_MAX
            )

            z = clamp(
                z,
                Z_MIN,
                Z_MAX
            )


        # ====================================================
        # JOINT KEYBOARD CONTROL
        # ====================================================

        elif (
            mode == "JOINT"
            and not emergency_stop
        ):

            joint_step = (
                0.8
                * (speed / 50)
            )

            joint_move = False


            # ------------------------------------------------
            # J1 = A / D
            # ------------------------------------------------

            if key == ord("a"):

                joints[0] -= joint_step

                joint_move = True


            elif key == ord("d"):

                joints[0] += joint_step

                joint_move = True


            # ------------------------------------------------
            # J2 = W / S
            # ------------------------------------------------

            elif key == ord("w"):

                joints[1] += joint_step

                joint_move = True


            elif key == ord("s"):

                joints[1] -= joint_step

                joint_move = True


            # ------------------------------------------------
            # J3 = UP / DOWN
            # ------------------------------------------------

            elif key in [
                82,
                2490368
            ]:

                joints[2] += joint_step

                joint_move = True


            elif key in [
                84,
                2621440
            ]:

                joints[2] -= joint_step

                joint_move = True


            # ------------------------------------------------
            # J4 = LEFT / RIGHT
            # ------------------------------------------------

            elif key in [
                81,
                2424832
            ]:

                joints[3] -= joint_step

                joint_move = True


            elif key in [
                83,
                2555904
            ]:

                joints[3] += joint_step

                joint_move = True


            # ------------------------------------------------
            # J5 = Z / X
            #
            # Using Z/X so Q remains QUIT.
            # ------------------------------------------------

            elif key == ord("z"):

                joints[4] -= joint_step

                joint_move = True


            elif key == ord("x"):

                joints[4] += joint_step

                joint_move = True


            # ------------------------------------------------
            # Keyboard joint control active
            # ------------------------------------------------

            if joint_move:

                keyboard_control = True

                enabled = False


            # ------------------------------------------------
            # Joint limits
            # ------------------------------------------------

            for i in range(5):

                joints[i] = clamp_joint(
                    i,
                    joints[i]
                )


        # ====================================================
        # GESTURE CARTESIAN CONTROL
        #
        # THIS IS THE ORIGINAL WORKING MAPPING
        # ====================================================

        if (
            mode == "CARTESIAN"
            and enabled
            and not keyboard_control
            and not emergency_stop
        ):

            if wrist is not None:

                # ------------------------------------------------
                # HAND X -> ROBOT X
                # ------------------------------------------------

                target_x = (
                    (wrist.x - 0.5)
                    * 0.3
                )


                # ------------------------------------------------
                # HAND Y -> ROBOT Y
                # ------------------------------------------------

                target_y = (
                    -0.19
                    - (wrist.y - 0.5)
                    * 0.26
                )


                # ------------------------------------------------
                # HAND DEPTH -> ROBOT Z
                # ------------------------------------------------

                hand_depth = distance(
                    landmarks[0],
                    landmarks[12]
                )

                target_z = (
                    0.20
                    + (hand_depth - 0.680)
                    * 0.60
                )


                # ------------------------------------------------
                # ORIGINAL SMOOTHING
                # ------------------------------------------------

                x += (
                    target_x - x
                ) * (
                    speed / 100
                ) * 0.1


                y += (
                    target_y - y
                ) * (
                    speed / 100
                ) * 0.1


                z += (
                    target_z - z
                ) * (
                    speed / 100
                ) * 0.1


                # ------------------------------------------------
                # ORIGINAL SOFT LIMITS
                # ------------------------------------------------

                x = clamp(
                    x,
                    X_MIN,
                    X_MAX
                )

                y = clamp(
                    y,
                    Y_MIN,
                    Y_MAX
                )

                z = clamp(
                    z,
                    Z_MIN,
                    Z_MAX
                )

        # ====================================================
        # JOINT GESTURE CONTROL
        # ====================================================

        if (
            mode == "JOINT"
            and enabled
            and not keyboard_control
            and not emergency_stop
        ):

            if landmarks is not None:

                # ------------------------------------------------
                # J1 = HAND LEFT / RIGHT
                # ------------------------------------------------

                hand_x = wrist.x - 0.5

                if abs(hand_x) > 0.05:

                    if hand_x > 0:
                        movement = hand_x - 0.05
                    else:
                        movement = hand_x + 0.05

                    joints[0] += (
                        movement
                        * 0.08
                        * (speed / 50)
                    )


                # ------------------------------------------------
                # J2 = HAND UP / DOWN
                # ------------------------------------------------

                hand_y = 0.5 - wrist.y

                if abs(hand_y) > 0.05:

                    if hand_y > 0:
                        movement = hand_y - 0.05
                    else:
                        movement = hand_y + 0.05

                    joints[1] += (
                        movement
                        * 0.06
                        * (speed / 50)
                    )


                # ------------------------------------------------
                # J3 = HAND DEPTH
                # ------------------------------------------------

                depth_error = hand_depth - 0.10

                if abs(depth_error) > 0.015:

                    joints[2] += (
                        depth_error
                        * 0.05
                        * (speed / 50)
                    )


                # ------------------------------------------------
                # J4 = HAND TILT
                # ------------------------------------------------

                palm_dx = (
                    landmarks[17].x
                    - landmarks[5].x
                )

                palm_dy = (
                    landmarks[17].y
                    - landmarks[5].y
                )

                palm_angle = math.atan2(
                    palm_dy,
                    palm_dx
                )

                joints[3] += (
                    palm_angle
                    * 0.01
                    * (speed / 50)
                )


                # ------------------------------------------------
                # J5 = HAND ROTATION
                # ------------------------------------------------

                wrist_to_index_x = (
                    landmarks[8].x
                    - landmarks[0].x
                )

                wrist_to_index_y = (
                    landmarks[8].y
                    - landmarks[0].y
                )

                # Image y points down, so an upright hand gives
                # atan2(-1, 0) = -pi/2. Measure the tilt relative to
                # upright (0 = upright, + = tilted right, - = tilted
                # left) so J5 can move in both directions.

                hand_angle = math.atan2(
                    wrist_to_index_x,
                    -wrist_to_index_y
                )

                if abs(hand_angle) > 0.15:

                    if hand_angle > 0:
                        movement = hand_angle - 0.15
                    else:
                        movement = hand_angle + 0.15

                    joints[4] += (
                        movement
                        * 0.05
                        * (speed / 50)
                    )


                # ------------------------------------------------
                # JOINT SAFETY LIMITS
                # ------------------------------------------------

                for i in range(5):

                    joints[i] = clamp_joint(
                        i,
                        joints[i]
                    )

        # ====================================================
        # SEND ROBOT COMMAND
        # ====================================================

        command = "WAIT"


        if emergency_stop:

            command = "STOP"


        elif mode == "CARTESIAN":

            if enabled or keyboard_control:

                send_cartesian_command()

                command = "MOVE"


        elif mode == "JOINT":

            if enabled or keyboard_control:

                send_joint_command()

                command = "JOINT MOVE"


        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        elapsed = (
            current_time
            - previous_time
        )

        if elapsed > 0:

            fps = 1 / elapsed

        else:

            fps = 0

        previous_time = current_time


        # ====================================================
        # DISPLAY: MODE
        # ====================================================

        cv2.putText(
            frame,
            f"MODE: {mode}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY: GESTURE
        # ====================================================

        cv2.putText(
            frame,
            f"GESTURE: {gesture}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY: CONTROL METHOD
        # ====================================================

        control_method = (
            "KEYBOARD"
            if keyboard_control
            else (
                "GESTURE"
                if enabled
                else "DISABLED"
            )
        )

        cv2.putText(
            frame,
            f"CONTROL: {control_method}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY: SPEED
        # ====================================================

        cv2.putText(
            frame,
            f"SPEED: {speed}%",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY: COMMAND
        # ====================================================

        cv2.putText(
            frame,
            f"COMMAND: {command}",
            (10, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY: POSITION
        # ====================================================

        if mode == "CARTESIAN":

            cv2.putText(
                frame,
                f"X: {x:+.3f} m",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y: {y:+.3f} m",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Z: {z:+.3f} m",
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"DEPTH: {hand_depth:.3f}",
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                f"J1: {joints[0]:+.2f} rad",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"J2: {joints[1]:+.2f} rad",
                (10, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"J3: {joints[2]:+.2f} rad",
                (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"J4: {joints[3]:+.2f} rad",
                (10, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"J5: {joints[4]:+.2f} rad",
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


        # ====================================================
        # GRIPPER DISPLAY
        # ====================================================

        gripper_text = (
            "CLOSED"
            if gripper_closed
            else "OPEN"
        )

        cv2.putText(
            frame,
            f"GRIPPER: {gripper_text}",
            (10, 385),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SAFETY STATUS
        # ====================================================

        if emergency_stop:

            safety_text = "EMERGENCY STOP"

        elif enabled or keyboard_control:

            safety_text = "CONTROL ENABLED"

        else:

            safety_text = "CONTROL DISABLED"


        cv2.putText(
            frame,
            safety_text,
            (10, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # ====================================================
        # FPS
        # ====================================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

        # ====================================================
        # SINGLE-LINE KEYBOARD SHORTCUTS
        # ====================================================

        shortcut_text = (
            "C:Cartesian | J:Joint | +/-:Speed | "
            "WASD/Arrows:Move | O:Open | P:Close | "
            "SPACE:Stop | R:Reset | Q:Quit"
        )

        cv2.putText(
            frame,
            shortcut_text,
            (5, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1
        )


        
        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "SO-101 Teleoperation Controller",
            frame
        )


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()

    detector.close()

    cv2.destroyAllWindows()

    try:

        robot_socket.close()

    except Exception:

        pass

    print(
        "Controller closed."
    )