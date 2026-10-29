
import cv2
import mediapipe as mp
import time
import csv
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
OUTPUT_PATH = Path(__file__).parent / "mockmate_video_report.csv"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Webcam not available.")

total_frames = 0
face_frames = 0
fps_values = []
start_time = time.perf_counter()
last_time = start_time
timestamp_ms = 0
rows = []

try:
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = camera.read()
            if not ok:
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

            total_frames += 1
            detected = bool(result.face_landmarks)

            if detected:
                face_frames += 1

            now = time.perf_counter()
            elapsed = now - start_time
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            fps_values.append(fps)

            rows.append({
                "elapsed_seconds": round(elapsed, 2),
                "face_detected": int(detected),
                "processing_fps": round(fps, 2)
            })

            status = "VISIBLE" if detected else "NOT DETECTED"
            color = (0, 255, 0) if detected else (0, 0, 255)

            cv2.putText(
                frame, f"Face: {status}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            cv2.putText(
                frame, f"Duration: {elapsed:.1f}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                frame, "Press Q to finish", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

            cv2.imshow("MockMATE - Video Metrics", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()

duration = time.perf_counter() - start_time
visibility = (face_frames / total_frames * 100) if total_frames else 0
average_fps = sum(fps_values) / len(fps_values) if fps_values else 0

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["elapsed_seconds", "face_detected", "processing_fps"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n--- MockMATE Video Report ---")
print(f"Duration: {duration:.1f} seconds")
print(f"Face visibility: {visibility:.1f}%")
print(f"Average processing FPS: {average_fps:.1f}")
print(f"Detailed data saved to: {OUTPUT_PATH}")