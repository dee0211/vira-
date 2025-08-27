import os
import sys
import time
import warnings
import threading
import speech_recognition as sr
import cv2
import torch
import numpy as np
import geocoder
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client
from datetime import datetime
import subprocess
from ultralytics import YOLO  # ✅ ultralytics for YOLOv5n

# -----------------------
# Suppress warnings
# -----------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ===== USER SETTINGS =====
# Audio Detection Settings
MIC_INDEX = 3  # ✅ your mic is on port 3
TRIGGER_KEYWORDS = {"help", "save me", "save-me", "saveme", "hello"}
ALERT_COOLDOWN_SECS = 60
FALLBACK_LOCATION_TEXT = "Location unavailable. Please call immediately."

# Twilio Configuration
TWILIO_ACCOUNT_SID = "AC627af4eb75d3b12614d48a294026102f"
TWILIO_AUTH_TOKEN = "323046ee8236ea7fc338b70c0632532c"
TWILIO_FROM_NUMBER = "+17753645311"
SEND_TO_NUMBER = "+917411125377"

# Visual Detection Settings
STALKER_THRESHOLD_SECONDS = 60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 5
DETECTION_SKIP_FRAMES = 2
YOLO_CONFIDENCE = 0.35
YOLO_IOU_THRESHOLD = 0.5
MIN_PERSON_AREA = 2000
DEEPSORT_MAX_AGE = 50
USE_GPU = False
# =========================

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Global variables
last_audio_alert_time = 0
alert_sent_ids = set()
seen_times = {}
system_running = True
frame_count = 0
yolo_model = None
deepsort_tracker = None

def load_yolo_model():
    """Load YOLOv5 Nano model optimized for Raspberry Pi"""
    global yolo_model

    print("[YOLO] Loading YOLOv5 Nano model (Ultralytics)...")

    try:
        # Force CPU usage unless GPU is specifically enabled
        device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        print(f"[YOLO] Using device: {device}")

        # ✅ This downloads yolov5n.pt the first time
        yolo_model = YOLO("yolov5n.pt")

        print("[YOLO] YOLOv5 Nano model loaded successfully")
        return True

    except Exception as e:
        print(f"[YOLO] Failed to load YOLOv5 model: {e}")
        print("[YOLO] Make sure you have internet connection for first-time download")
        return False

def initialize_deepsort():
    """Initialize DeepSORT tracker with optimized settings for RPi"""
    global deepsort_tracker
    print("[DEEPSORT] Initializing DeepSORT tracker...")

    try:
        deepsort_tracker = DeepSort(
            max_age=DEEPSORT_MAX_AGE,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.4,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=USE_GPU
        )
        print("[DEEPSORT] DeepSORT tracker initialized successfully")
        return True

    except Exception as e:
        print(f"[DEEPSORT] Failed to initialize DeepSORT: {e}")
        return False

def visual_detection_main():
    """Main function for visual detection using YOLOv5 + DeepSORT + rpicam-apps"""
    global seen_times, alert_sent_ids, system_running, frame_count, yolo_model, deepsort_tracker

    if not load_yolo_model():
        print("[VISUAL] Cannot start visual detection without YOLO model")
        return

    if not initialize_deepsort():
        print("[VISUAL] Cannot start visual detection without DeepSORT")
        return

    print("[VISUAL] Starting rpicam-vid subprocess...")

    # ✅ Launch rpicam-vid to stream frames to stdout
    cam_process = subprocess.Popen(
        ["rpicam-vid", "--inline", "--timeout", "0", "--width", str(FRAME_WIDTH),
         "--height", str(FRAME_HEIGHT), "--framerate", str(FPS_TARGET), "-o", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    bytes_buffer = b""

    try:
        while system_running:
            # Read raw stream from rpicam-vid
            data = cam_process.stdout.read(1024)
            if not data:
                break
            bytes_buffer += data

            # Try decoding frame
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                frame_count += 1
                if frame_count % DETECTION_SKIP_FRAMES != 0:
                    continue

                # Run YOLO inference
                results = yolo_model(frame, verbose=False)
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().item()
                        cls = int(box.cls[0].cpu().item())
                        detections.append([x1, y1, x2, y2, conf, cls])

                # Convert to DeepSORT format
                deepsort_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if cls != 0:  # only person
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_PERSON_AREA:
                        continue
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    deepsort_detections.append((bbox, float(conf), "person"))

                # Track with DeepSORT
                tracks = deepsort_tracker.update_tracks(deepsort_detections, frame=frame)

                # Draw results
                for track in tracks:
                    if track.is_confirmed():
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show frame
                try:
                    cv2.imshow("RPi YOLOv5 + DeepSORT", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass

    except KeyboardInterrupt:
        print("[VISUAL] Interrupted by user.")
    finally:
        cam_process.terminate()
        cv2.destroyAllWindows()
        print("[VISUAL] Visual detection stopped.")

def main():
    """Main function optimized for Raspberry Pi with YOLOv5 + DeepSORT"""
    global system_running

    print("="*65)
    print("  RASPBERRY PI 4 EMERGENCY DETECTION WITH YOLOv5 + DeepSORT")
    print("="*65)

    try:
        # Start audio detection thread
        audio_thread = threading.Thread(target=audio_detection_thread, daemon=True)
        audio_thread.start()

        # Run visual detection
        visual_detection_main()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
    finally:
        system_running = False
        print("[SYSTEM] Emergency detection stopped.")

if __name__ == "__main__":
    main()
