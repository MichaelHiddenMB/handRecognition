import time
import argparse

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe connection constants moved from mp.solutions to Tasks API in newer versions.
try:
    HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
except AttributeError:
    HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)

def try_open(index):
    # CAP_AVFOUNDATION tends to behave best on macOS, but fall back if it fails.
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def draw_landmarks(image, hand_landmarks):
    height, width = image.shape[:2]
    for hand in hand_landmarks:
        pts = []
        for landmark in hand:
            x = min(max(int(landmark.x * width), 0), width - 1)
            y = min(max(int(landmark.y * height), 0), height - 1)
            pts.append((x, y))
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(image, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera index to open first (e.g. 0, 1, 2)")
    args = parser.parse_args()

    idx = max(0, args.camera_index)
    cap = try_open(idx)
    start_time = time.monotonic()

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    print("Controls: [n]=next camera index, [p]=prev, [q]=quit")
    while True:
        if cap is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Could not open camera index {idx}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("camera_debug", frame)
        else:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(franme, f"Opened index {idx} but read() failed",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.monotonic() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)
                if result.hand_landmarks:
                    draw_landmarks(frame, result.hand_landmarks)
            cv2.imshow("camera_debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key in (ord("n"), ord("p")):
            if cap is not None:
                cap.release()
            idx = idx + 1 if key == ord("n") else max(0, idx - 1)
            cap = try_open(idx)
            print(f"Switched to index {idx}. Opened? {cap is not None}")

    if cap is not None:
        cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
