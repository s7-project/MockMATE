
import cv2
import mediapipe as mp
import time
import math
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Place face_landmarker.task beside this Python file."
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_facial_transformation_matrixes=True
)

# Approximate landmark indices used for head-pose estimation
NOSE = 1
CHIN = 152
LEFT_EYE = 33
RIGHT_EYE = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

def estimate_head_pose(landmarks, width, height):
    # 2D image points
    image_points = [
        (landmarks[NOSE].x * width, landmarks[NOSE].y * height),
        (landmarks[CHIN].x * width, landmarks[CHIN].y * height),
        (landmarks[LEFT_EYE].x * width, landmarks[LEFT_EYE].y * height),
        (landmarks[RIGHT_EYE].x * width, landmarks[RIGHT_EYE].y * height),
        (landmarks[LEFT_MOUTH].x * width, landmarks[LEFT_MOUTH].y * height),
        (landmarks[RIGHT_MOUTH].x * width, landmarks[RIGHT_MOUTH].y * height),
    ]

    # Approximate 3D reference points for a generic face
    model_points = [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ]

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = __import__("numpy").array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    distortion = __import__("numpy").zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        __import__("numpy").array(model_points, dtype="double"),
        __import__("numpy").array(image_points, dtype="double"),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = __import__("numpy").array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ])
    # Euler angles are easier to interpret using OpenCV decomposition
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
        __import__("numpy").dot(camera_matrix, __import__("numpy").hstack(
            (rotation_matrix, translation_vector)
        ))
    )
    pitch, yaw, roll = [float(v) for v in euler.flatten()]
    return pitch, yaw, roll

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Webcam not available. Close other camera apps.")

timestamp_ms = 0
last_time = time.perf_counter()

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                for landmark in landmarks:
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                pose = estimate_head_pose(landmarks, width, height)

                cv2.putText(
                    frame, "Face: VISIBLE", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

                if pose is not None:
                    pitch, yaw, roll = pose

                    # These are rough orientation cues, not gaze or confidence.
                    if yaw > 15:
                        direction = "Turned LEFT/RIGHT (approx.)"
                    elif yaw < -15:
                        direction = "Turned RIGHT/LEFT (approx.)"
                    elif pitch > 15:
                        direction = "Head tilted (approx.)"
                    elif pitch < -15:
                        direction = "Head tilted (approx.)"
                    else:
                        direction = "Facing camera (approx.)"

                    cv2.putText(
                        frame, direction, (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2
                    )
            else:
                cv2.putText(
                    frame, "Face: NOT DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            now = time.perf_counter()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.imshow("MockMATE - Stage 2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()