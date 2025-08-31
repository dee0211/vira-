#!/usr/bin/env python3
import os
import time
import warnings
import logging   # ? correct module
import cv2
import torch
import geocoder
import subprocess
import numpy as np
import threading
import speech_recognition as sr
from ultralytics import YOLO  # Changed from torch.hub to ultralytics
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client
from datetime import datetime

# Suppress ALSA / PyAudio warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["ALSA_PCM_CARD"] = "3"
os.environ["ALSA_CTL_CARD"] = "3"

# Suppress non-critical log messages
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Redirect stderr (where ALSA errors are printed) to /dev/null
import sys
sys.stderr = open(os.devnull, 'w')  # ? fixed


# -----------------------
# Config
# -----------------------
CAPTURE_PATH = "capture.jpg"
ANNOTATED_PATH = "yolo_annotated.jpg"
WIDTH, HEIGHT = 640, 480
TWILIO_ACCOUNT_SID = "AC627af4eb75d3b12614d48a294026102f"
TWILIO_AUTH_TOKEN = "f48aefdc8196d9aade7ac2352dc0e99a"
TWILIO_FROM_NUMBER = "+17753645311"
SEND_TO_NUMBER = "+917411125377"
MIC_INDEX = 1
TRIGGER_KEYWORDS = {"help", "save me", "save-me", "saveme", "hello"}
ALERT_COOLDOWN_SECS = 60
FALLBACK_LOCATION_TEXT = "Location unavailable. Please call immediately."
CAPTURE_INTERVAL = 10  # seconds
ALERT_THRESHOLD = 60  # seconds to trigger alert if ID is present
# -----------------------

warnings.filterwarnings("ignore", category=FutureWarning)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -----------------------
# Helper Functions
# -----------------------
def send_sms(body: str):
    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=SEND_TO_NUMBER
        )
        print(f"[SMS SENT] SID: {message.sid}")
    except Exception as e:
        print("[ERROR] Failed to send SMS:", e)

def get_location():
    try:
        g = geocoder.ip("me")
        if g.ok and g.latlng:
            return g.latlng
    except:
        pass
    return None

def format_location_message(lat, lon, extra=""):
    maps_link = f"https://maps.google.com/?q={lat},{lon}"
    return f"ALERT: {extra}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
           f"Location: {lat:.6f}, {lon:.6f}\n{maps_link}"

def format_fallback_message(extra=""):
    return f"ALERT: {extra}\n{FALLBACK_LOCATION_TEXT}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def normalize_text(t: str):
    return t.lower().strip()

def contains_trigger(text: str):
    t = normalize_text(text)
    return any(k in t for k in TRIGGER_KEYWORDS)

def capture_frame():
    """Capture one frame using rpicam-vid and return as OpenCV image."""
    try:
        cmd = [
            "rpicam-vid",
            "-t", "1000",                # 1 second capture
            "-o", "-",                   # output to stdout
            "--width", str(WIDTH),
            "--height", str(HEIGHT),
            "--codec", "mjpeg",          # MJPEG stream
            "--nopreview",
            "-n"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = b""
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            data += chunk
        proc.terminate()
        # Decode MJPEG to OpenCV image
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[ERROR] Capture failed: {e}")
        return None

# -----------------------
# Load YOLO8 Model (Updated from YOLOv5)
# -----------------------
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

# -----------------------
# Initialize DeepSORT with optimized settings from modeltest
# -----------------------
tracker = DeepSort(
    max_age=5,      # forget after 5 frames instead of 30
    n_init=1,       # assign ID after 1 detection instead of waiting
    max_iou_distance=0.7
)
seen_times = {}
alert_sent_ids = set()

# -----------------------
# Speech Recognition Thread (UNCHANGED)
# -----------------------
last_alert_time = 0
def speech_thread():
    global last_alert_time
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=MIC_INDEX)
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    print("[INFO] Speech thread started...")
    while True:
        with mic as source:
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=10)
                text = r.recognize_google(audio)
                print(f"[VOICE] Heard: {text}")
                if contains_trigger(text):
                    now = time.time()
                    if now - last_alert_time >= ALERT_COOLDOWN_SECS:
                        loc = get_location()
                        if loc:
                            lat, lon = loc
                            msg = format_location_message(lat, lon, extra="Voice trigger detected!")
                        else:
                            msg = format_fallback_message(extra="Voice trigger detected!")
                        send_sms(msg)
                        last_alert_time = now
                    else:
                        print("[VOICE] Alert cooldown active.")
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("[VOICE] Speech recognition error:", e)
            except Exception as e:
                print("[VOICE] Mic error:", e)

# -----------------------
# Image Capture + YOLO Thread (UPDATED with working tracking)
# -----------------------
def capture_loop():
    print("[INFO] Starting image capture loop...")
    while True:
        loop_start = time.time()
        
        # --- Capture frame using the working method from modeltest ---
        frame = capture_frame()
        if frame is None:
            print("[WARN] No frame captured.")
            time.sleep(CAPTURE_INTERVAL)
            continue

        # --- YOLO8 inference (Updated from YOLOv5) ---
        results = model(frame)[0]
        detections = []
        
        # --- Process detections with proper coordinate conversion ---
        for box in results.boxes:
            if box.cls is not None and int(box.cls[0]) == 0:  # person class
                x_center, y_center, w, h = box.xywh[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Convert YOLO (center) to DeepSORT (top-left, width, height)
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                detections.append(([x1, y1, w, h], conf, cls))

        # --- Update tracker with proper detections ---
        tracks = tracker.update_tracks(detections, frame=frame)
        current_time = time.time()

        # --- Process tracked people (Updated with working tracking logic) ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Track time seen for alerts
            if track_id not in seen_times:
                seen_times[track_id] = current_time
                print(f"[TRACKING] New person detected: ID {track_id}")
            else:
                elapsed = current_time - seen_times[track_id]
                if elapsed > ALERT_THRESHOLD and track_id not in alert_sent_ids:
                    print(f"[ALERT] Person ID {track_id} present for {elapsed:.1f} seconds")
                    loc = get_location()
                    if loc:
                        lat, lon = loc
                        msg = format_location_message(lat, lon, extra=f"Person ID {track_id} detected >{ALERT_THRESHOLD} sec")
                    else:
                        msg = format_fallback_message(extra=f"Person ID {track_id} detected >{ALERT_THRESHOLD} sec")
                    send_sms(msg)
                    alert_sent_ids.add(track_id)

        # Save annotated image
        cv2.imwrite(ANNOTATED_PATH, frame)
        print(f"[INFO] Processed frame with {len(tracks)} confirmed tracks")

        elapsed = time.time() - loop_start
        sleep_time = max(0, CAPTURE_INTERVAL - elapsed)
        time.sleep(sleep_time)

# -----------------------
# Start threads
# -----------------------
threading.Thread(target=speech_thread, daemon=True).start()
capture_loop()
