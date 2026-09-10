import cv2
import mediapipe as mp
import socket
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# CONTROL STATE
# ============================================================

mode = "CARTESIAN"

enabled = False

# Emergency stop is LATCHED.
# SPACE activates it.
# R resets it.
emergency_stop = False


# ============================================================
# SPEED
# ============================================================

speed = 50


# ============================================================
# CARTESIAN POSITION
# ============================================================

x = 0.0
y = 0.0
z = 0.15
print("REACHED ROBOT CONNECTION CODE")
ROBOT_HOST = "172.27.7.2"
ROBOT_PORT = 5000

robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
robot_socket.connect((ROBOT_HOST, ROBOT_PORT))

print("Connected to MuJoCo bridge")

# ============================================================
# VIRTUAL CARTESIAN LIMITS
# ============================================================

X_MIN = -0.20
X_MAX = 0.20

Y_MIN = -0.20
Y_MAX = 0.20

Z_MIN = 0.05
Z_MAX = 0.30


# ============================================================
# VIRTUAL JOINT POSITIONS
# ============================================================

joints = [0, 0, 0, 0, 0, 0]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def detect_gesture(landmarks):
    """
    Simple gesture classifier.

    OPEN PALM:
        Enables normal control.

    FIST:
        Only reported as a gesture.

    PINCH:
        Only reported as a gesture.

    Emergency stop:
        Controlled independently by SPACE.
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

    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y

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

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera.")
        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # FRAME DIMENSIONS
    # --------------------------------------------------------

    h, w, _ = frame.shape


    # --------------------------------------------------------
    # CONVERT BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    frame_timestamp_ms += 33


    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # ========================================================
    # HAND DETECTION / GESTURE
    # ========================================================

    gesture = "NO HAND"
    wrist = None

    # Default value so DEPTH display works even when
    # no hand is detected.
    hand_depth = 0.0


    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        wrist = landmarks[0]

        gesture = detect_gesture(landmarks)


        # ----------------------------------------------------
        # WRIST
        # ----------------------------------------------------

        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            10,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # DRAW HAND LANDMARKS
        # ----------------------------------------------------

        for landmark in landmarks:

            px = int(landmark.x * w)
            py = int(landmark.y * h)

            cv2.circle(
                frame,
                (px, py),
                4,
                (0, 255, 0),
                -1
            )


    # ========================================================
    # NORMAL GESTURE CONTROL
    # ========================================================

    if gesture == "OPEN PALM" and not emergency_stop:

        enabled = True


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # Q = QUIT
    # ========================================================

    if key == ord("q"):
        break


    # ========================================================
    # C = CARTESIAN MODE
    # ========================================================

    if key == ord("c"):

        mode = "CARTESIAN"

        print("Mode changed to CARTESIAN")


    # ========================================================
    # J = JOINT MODE
    # ========================================================

    if key == ord("j"):

        mode = "JOINT"

        print("Mode changed to JOINT")


    # ========================================================
    # + / = = INCREASE SPEED
    # ========================================================

    if key in [ord("+"), ord("=")]:

        speed += 10

        speed = min(speed, 100)

        print(f"Speed: {speed}%")


    # ========================================================
    # - = DECREASE SPEED
    # ========================================================

    if key == ord("-"):

        speed -= 10

        speed = max(speed, 10)

        print(f"Speed: {speed}%")


    # ========================================================
    # SPACE = EMERGENCY STOP
    # ========================================================
    #
    # Works regardless of:
    #   - gesture
    #   - mode
    #   - hand visibility
    #   - speed
    #
    # Emergency stop remains active until R is pressed.
    # ========================================================

    if key == 32:

        emergency_stop = True
        enabled = False

        print("!!! EMERGENCY STOP !!!")


    # ========================================================
    # R = RESET EMERGENCY STOP
    # ========================================================

    if key == ord("r"):

        emergency_stop = False
        enabled = False

        print(
            "Emergency stop reset. "
            "Show OPEN PALM to enable control."
        )


    # ========================================================
    # ROBOT COMMAND LOGIC
    # ========================================================

    command = "WAIT"


    # ========================================================
    # EMERGENCY STOP HAS HIGHEST PRIORITY
    # ========================================================

    if emergency_stop:

        command = "STOP"


    # ========================================================
    # NORMAL CONTROL
    # ========================================================

    elif enabled:

        command = "MOVE"


        # ====================================================
        # CARTESIAN MODE
        # ====================================================

        if mode == "CARTESIAN":

            if wrist is not None:

                # ------------------------------------------------
                # APPROXIMATE DEPTH
                # ------------------------------------------------
                #
                # Landmark 0  = wrist
                # Landmark 12 = middle fingertip
                #
                # This is currently only a rough depth proxy.
                # ------------------------------------------------

                hand_depth = distance(
                    landmarks[0],
                    landmarks[12]
                )


                # ------------------------------------------------
                # HAND X -> ROBOT X
                # ------------------------------------------------

                target_x = ((wrist.x - 0.5) * 0.3)

                # ------------------------------------------------
                # APPROXIMATE HAND DEPTH -> ROBOT Y
                # ------------------------------------------------

                target_y = 0.0 - (wrist.y - 0.5) * 0.40

                # ------------------------------------------------
                # HAND VERTICAL POSITION -> ROBOT Z
                # ------------------------------------------------

                hand_depth = distance(landmarks[0], landmarks[12])
                target_z = 0.20 + (hand_depth - 0.680) * 0.60

                # ------------------------------------------------
                # GRADUAL MOVEMENT / SMOOTHING
                # ------------------------------------------------

                x += (
                    target_x - x
                ) * (speed / 100) * 0.1


                y += (
                    target_y - y
                ) * (speed / 100) * 0.1


                z += (
                    target_z - z
                ) * (speed / 100) * 0.1


                # ------------------------------------------------
                # APPLY VIRTUAL SOFT LIMITS
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

                # ------------------------------------------------
                # SEND SMOOTHED XYZ TO MUJOCO
                # ------------------------------------------------

                message = f"{x:.4f},{y:.4f},{z:.4f}\n"
                robot_socket.sendall(
                    message.encode()
)


        # ====================================================
        # JOINT MODE
        # ====================================================

        elif mode == "JOINT":

            if wrist is not None:

                # ------------------------------------------------
                # HORIZONTAL HAND POSITION -> JOINT 1
                # ------------------------------------------------

                joint_target = (
                    wrist.x - 0.5
                ) * 180


                # ------------------------------------------------
                # SMOOTH JOINT MOVEMENT
                # ------------------------------------------------

                joints[0] += (
                    joint_target - joints[0]
                ) * (speed / 100) * 0.1


                # ------------------------------------------------
                # JOINT 1 VIRTUAL LIMIT
                # ------------------------------------------------

                joints[0] = clamp(
                    joints[0],
                    -90,
                    90
                )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:
        fps = 1 / elapsed
    else:
        fps = 0

    previous_time = current_time


    # ========================================================
    # DISPLAY: MODE
    # ========================================================

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY: GESTURE
    # ========================================================

    cv2.putText(
        frame,
        f"GESTURE: {gesture}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY: SPEED
    # ========================================================

    cv2.putText(
        frame,
        f"SPEED: {speed}%",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY: COMMAND
    # ========================================================

    cv2.putText(
        frame,
        f"COMMAND: {command}",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY: CARTESIAN POSITION
    # ========================================================

    if mode == "CARTESIAN":

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"X: {x:+.3f} m",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Y: {y:+.3f} m",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Z: {z:+.3f} m",
            (10, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # STEP 60 DEBUG:
        # RAW DEPTH VALUE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"DEPTH: {hand_depth:.3f}",
            (10, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            f"Joint 1: {joints[0]:+.1f} deg",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SAFETY STATUS
    # ========================================================

    if emergency_stop:

        safety_text = "EMERGENCY STOP"

    elif enabled:

        safety_text = "CONTROL ENABLED"

    else:

        safety_text = "CONTROL DISABLED"


    cv2.putText(
        frame,
        safety_text,
        (10, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ========================================================
    # KEYBOARD INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "C: Cartesian | J: Joint | +/-: Speed | "
        "SPACE: STOP | R: Reset | Q: Quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "SO-101 Teleoperation Controller",
        frame
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
detector.close()
cv2.destroyAllWindows()