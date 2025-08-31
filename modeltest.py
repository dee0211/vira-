#!/usr/bin/env python3
import os
import time
import subprocess
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

CAPTURE_PATH = "test_capture.jpg"
ANNOTATED_PATH = "test_annotated.jpg"
WIDTH, HEIGHT = 640, 480

def capture_one(path=CAPTURE_PATH, width=WIDTH, height=HEIGHT):
    cmds = [
        ["libcamera-still", "-o", path, "-n", "--width", str(width), "--height", str(height), "-t", "1500"],
        ["rpicam-still", "-o", path, "-t", "1500"]
    ]
    for cmd in cmds:
        try:
            print(f"[CAPTURE] Trying: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            time.sleep(0.2)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print("[CAPTURE] Success!")
                return True
        except Exception as e:
            print(f"[CAPTURE] Failed with {cmd[0]}: {e}")
    print("[CAPTURE] Could not capture image with libcamera-still or rpicam-still.")
    return False

def main():
    print("[INFO] Loading YOLOv5s (person only)…")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.5
    model.classes = [0]  # person only

    print("[INFO] Capturing test image…")
    if not capture_one():
        return

    img = cv2.imread(CAPTURE_PATH)
    if img is None:
        print("[ERROR] Failed to read captured image.")
        return

    # YOLO inference
    print("[INFO] Running detection…")
    results = model(img)
    detections = results.xyxy[0]  # tensor Nx6: x1,y1,x2,y2,conf,cls
    print(f"[INFO] Raw detections: {len(detections)}")

    # Prepare for DeepSORT
    person_dets = []
    for *box, conf, cls in detections.tolist():
        if int(cls) == 0:
            x1, y1, x2, y2 = box
            person_dets.append(([x1, y1, x2, y2], float(conf), "person"))

    tracker = DeepSort(max_age=30)
    tracks = tracker.update_tracks(person_dets, frame=img)

    # Draw and list IDs
    found_ids = []
    for trk in tracks:
        if not trk.is_confirmed():
            continue
        tid = trk.track_id
        found_ids.append(tid)
        x1, y1, x2, y2 = map(int, trk.to_ltrb())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"ID:{tid}", (x1, max(0, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Save YOLO’s rendered image if possible, otherwise our own
    try:
        annotated = results.render()[0]
    except Exception:
        annotated = img
    cv2.imwrite(ANNOTATED_PATH, annotated)

    if found_ids:
        print(f"[RESULT] Person IDs detected: {sorted(found_ids)}")
    else:
        print("[RESULT] No confirmed person IDs (make sure a person is in frame and well-lit).")

    print(f"[OUTPUT] Saved capture: {CAPTURE_PATH}")
    print(f"[OUTPUT] Saved annotated: {ANNOTATED_PATH}")

if __name__ == "__main__":
    main()
