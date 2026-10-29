
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Find the model in this script's folder
model_path = Path(__file__).parent / "face_landmarker.task"

if not model_path.exists():
    raise FileNotFoundError(
        "Download face_landmarker.task and place it "
        "in the same folder as face_test.py"
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Close other camera apps "
        "and check Windows camera permissions."
    )

previous_time = time.perf_counter()
timestamp_ms = 0

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the preview like a selfie camera
            frame = cv2.flip(frame, 1)

            # OpenCV uses BGR; MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Use an increasing timestamp for video frames
            timestamp_ms += 33
            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            if result.face_landmarks:
                for face in result.face_landmarks:
                    height, width = frame.shape[:2]

                    # Draw a point for every detected landmark
                    for landmark in face:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame, (x, y), 1,
                                (0, 255, 0), -1
                            )

                status = "Face detected"
                color = (0, 255, 0)
            else:
                status = "No face detected"
                color = (0, 0, 255)

            now = time.perf_counter()
            fps = 1 / max(now - previous_time, 1e-6)
            previous_time = now

            cv2.putText(
                frame, status, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2
            )

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2
            )

            cv2.putText(
                frame, "Press Q to quit", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

            cv2.imshow("MockMATE - Face Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()