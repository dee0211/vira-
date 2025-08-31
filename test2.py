import os
import sys
import time
import threading
import subprocess
import warnings
import cv2
import torch
import numpy as np
import speech_recognition as sr
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client

# -----------------------
# Suppress warnings
# -----------------------
warnings.filterwarnings("ignore")

# -----------------------
# Twilio Config
# -----------------------
TWILIO_SID = "your_twilio_sid"
TWILIO_AUTH = "your_twilio_auth"
TWILIO_FROM = "+1xxxxxxxxxx"
TWILIO_TO = "+91xxxxxxxxxx"
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# -----------------------
# Load YOLOv8 Model
# -----------------------
model = YOLO("yolov8n.pt")  # change to your trained model if needed

# -----------------------
# DeepSORT Tracker Config
# -----------------------
tracker = DeepSort(
    max_age=15,   # IDs last max 15 frames if lost
    n_init=2,     # ID confirmed after 2 detections
    nms_max_overlap=1.0,
    max_cosine_distance=0.4,
    embedder="mobilenet"
)

# -----------------------
# Twilio Alert Function
# -----------------------
def send_alert(msg="Alert! Suspicious activity detected."):
    try:
        twilio_client.messages.create(
            body=msg,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print("[INFO] Alert sent via Twilio")
    except Exception as e:
        print("[ERROR] Twilio alert failed:", e)

# -----------------------
# Voice Command Listener
# -----------------------
def voice_listener():
    r = sr.Recognizer()
    mic = sr.Microphone()
    print("[INFO] Voice listener started...")
    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio).lower()
            print(f"[VOICE] Heard: {text}")
            if "help" in text or "alert" in text:
                send_alert("Voice Trigger: Help requested!")
        except Exception:
            continue

# -----------------------
# Start Audio Thread
# -----------------------
audio_thread = threading.Thread(target=voice_listener, daemon=True)
audio_thread.start()

# -----------------------
# Video Capture Loop
# -----------------------
def capture_frames():
    print("[INFO] Starting video capture...")
    while True:
        try:
            # Capture a single frame using rpicam-vid
            subprocess.run([
                "rpicam-vid", "-t", "1000", "-o", "frame.jpg",
                "--width", "640", "--height", "480",
                "--nopreview", "-n"
            ], check=True)

            frame = cv2.imread("frame.jpg")
            if frame is None:
                print("[ERROR] Failed to read frame")
                continue

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Convert to (x, y, w, h) format for DeepSORT
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(((x1, y1, w, h), conf, cls))

            # Update DeepSORT
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("YOLOv8 + DeepSORT", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("[ERROR]", e)
            time.sleep(1)

    cv2.destroyAllWindows()

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    capture_frames()
