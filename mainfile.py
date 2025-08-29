#!/usr/bin/env python3
import os
import time
import warnings
import cv2
import torch
import geocoder
import subprocess
import numpy as np
import threading
import speech_recognition as sr
from deep_sort_realtime.deepsort_tracker import DeepSort
from twilio.rest import Client
from datetime import datetime

# -----------------------
# Config
# -----------------------
CAPTURE_PATH = "capture.jpg"
ANNOTATED_PATH = "yolo_annotated.jpg"
WIDTH, HEIGHT = 640, 480
TWILIO_ACCOUNT_SID = "YOUR_ACCT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM_NUMBER = "YOUR_TWILIO_NUMBER"
SEND_TO_NUMBER = "YOUR_NUMBER"
MIC_INDEX = 0
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

# -----------------------
# Load YOLOv5 Model
# -----------------------
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.classes = [0]  # only detect person

# -----------------------
# Initialize DeepSORT
# -----------------------
tracker = DeepSort(max_age=30)
seen_times = {}
alert_sent_ids = set()

# -----------------------
# Speech Recognition Thread
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
                audio = r.listen(source, timeout=3, phrase_time_limit=3)
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
# Image Capture + YOLO Thread
# -----------------------
def capture_loop():
    print("[INFO] Starting image capture loop...")
    while True:
        loop_start = time.time()
        # --- Capture image ---
        capture_cmds = [
            ["libcamera-still", "-o", CAPTURE_PATH, "-n", "--width", str(WIDTH), "--height", str(HEIGHT), "-t", "1500"],
            ["rpicam-still", "-o", CAPTURE_PATH, "-t", "1500"]
        ]
        captured = False
        for cmd in capture_cmds:
            try:
                subprocess.run(cmd, check=True)
                time.sleep(0.2)
                if os.path.exists(CAPTURE_PATH):
                    captured = True
                    break
            except:
                continue
        if not captured:
            print("[ERROR] Could not capture image. Skipping this cycle.")
            time.sleep(CAPTURE_INTERVAL)
            continue

        img = cv2.imread(CAPTURE_PATH)
        if img is None:
            print("[ERROR] Failed to read captured image. Skipping this cycle.")
            time.sleep(CAPTURE_INTERVAL)
            continue

        # --- YOLO inference ---
        results = model(img)
        detections = results.xyxy[0]

        # --- Prepare detections for DeepSORT ---
        person_dets = []
        for *box, conf, cls in detections.tolist():
            if int(cls) == 0:
                x1, y1, x2, y2 = box
                person_dets.append(([x1, y1, x2, y2], float(conf), "person"))

        tracks = tracker.update_tracks(person_dets, frame=img)
        current_time = time.time()

        # --- Process tracked people ---
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            # Track time seen
            if track_id not in seen_times:
                seen_times[track_id] = current_time
            else:
                elapsed = current_time - seen_times[track_id]
                if elapsed > ALERT_THRESHOLD and track_id not in alert_sent_ids:
                    loc = get_location()
                    if loc:
                        lat, lon = loc
                        msg = format_location_message(lat, lon, extra=f"Person ID {track_id} detected >{ALERT_THRESHOLD} sec")
                    else:
                        msg = format_fallback_message(extra=f"Person ID {track_id} detected >{ALERT_THRESHOLD} sec")
                    send_sms(msg)
                    alert_sent_ids.add(track_id)

        # Save annotated image
        try:
            annotated = results.render()[0]
            cv2.imwrite(ANNOTATED_PATH, annotated)
        except:
            cv2.imwrite(ANNOTATED_PATH, img)

        elapsed = time.time() - loop_start
        sleep_time = max(0, CAPTURE_INTERVAL - elapsed)
        time.sleep(sleep_time)

# -----------------------
# Start threads
# -----------------------
threading.Thread(target=speech_thread, daemon=True).start()
capture_loop()
