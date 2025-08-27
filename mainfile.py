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

# -----------------------
# Suppress warnings
# -----------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['YOLO_VERBOSE'] = 'False'

# ===== USER SETTINGS =====
# Audio Detection Settings
MIC_INDEX = 0  # USB microphone usually index 1 or 2 on RPi
TRIGGER_KEYWORDS = {"help", "save me", "save-me", "saveme", "hello"}
ALERT_COOLDOWN_SECS = 60
FALLBACK_LOCATION_TEXT = "Location unavailable. Please call immediately."

# Twilio Configuration
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM_NUMBER = "YOUR_TWILIO_NUMBER"
SEND_TO_NUMBER = "YOUR_PHONE_NUMBER"

# Visual Detection Settings - OPTIMIZED FOR RASPBERRY PI WITH YOLO + DEEPSORT
STALKER_THRESHOLD_SECONDS = 60
WEBCAM_INDEX = 0
FRAME_WIDTH = 640  # Slightly higher resolution for better DeepSORT features
FRAME_HEIGHT = 480
FPS_TARGET = 5  # Conservative for YOLOv5 + DeepSORT on RPi
DETECTION_SKIP_FRAMES = 2  # Process every 2nd frame for DeepSORT
YOLO_CONFIDENCE = 0.35  # Lower confidence for better detection
YOLO_IOU_THRESHOLD = 0.5
MIN_PERSON_AREA = 2000  # Minimum area to consider as person
DEEPSORT_MAX_AGE = 50  # How long to keep tracks without detection
USE_GPU = False  # Set to True if you have GPU acceleration
# =========================

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Global variables for coordination
last_audio_alert_time = 0
alert_sent_ids = set()
seen_times = {}
system_running = True
frame_count = 0
yolo_model = None
deepsort_tracker = None

def optimize_raspberry_pi():
    """Optimize Raspberry Pi settings for better performance"""
    try:
        # Set CPU governor to performance
        subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'], 
                      capture_output=True, check=False)
        print("[SYSTEM] CPU governor set to performance mode")
        
        # Check GPU memory split
        result = subprocess.run(['vcgencmd', 'get_mem', 'gpu'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            gpu_mem = result.stdout.strip()
            print(f"[SYSTEM] GPU memory: {gpu_mem}")
            if 'gpu=64' in gpu_mem.lower():
                print("[SYSTEM] Consider increasing GPU memory to 128MB: sudo raspi-config")
            
    except Exception as e:
        print(f"[SYSTEM] Performance optimization failed: {e}")

def load_yolo_model():
    """Load YOLOv5 model optimized for Raspberry Pi"""
    global yolo_model
    
    print("[YOLO] Loading YOLOv5 Nano model (optimized for RPi)...")
    
    try:
        # Force CPU usage unless GPU is specifically enabled
        device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        print(f"[YOLO] Using device: {device}")
        
        # Load the smallest YOLOv5 model
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', 
                                   pretrained=True, device=device, force_reload=False)
        
        # Optimize model settings for RPi
        yolo_model.conf = YOLO_CONFIDENCE
        yolo_model.iou = YOLO_IOU_THRESHOLD
        yolo_model.classes = [0]  # Only detect persons
        yolo_model.max_det = 15   # Maximum detections per image (reduced for DeepSORT)
        
        # Set model to evaluation mode and optimize for inference
        yolo_model.eval()
        if hasattr(yolo_model, 'half') and device != 'cpu':
            yolo_model.half()  # Use FP16 if supported
            
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
        # DeepSORT configuration optimized for Raspberry Pi
        deepsort_tracker = DeepSort(
            max_age=DEEPSORT_MAX_AGE,           # Keep tracks for longer
            n_init=3,                           # Confirm track after 3 detections
            nms_max_overlap=1.0,               # Non-max suppression overlap
            max_cosine_distance=0.4,           # Feature matching threshold
            nn_budget=None,                    # No limit on stored features
            override_track_class=None,
            embedder="mobilenet",              # Use MobileNet for faster feature extraction
            half=False,                        # Use full precision
            bgr=True,                          # OpenCV uses BGR format
            embedder_gpu=USE_GPU,              # Use GPU for embedder if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        print("[DEEPSORT] DeepSORT tracker initialized successfully")
        print(f"[DEEPSORT] Using MobileNet embedder for feature extraction")
        return True
        
    except Exception as e:
        print(f"[DEEPSORT] Failed to initialize DeepSORT: {e}")
        return False

def send_sms(body: str, alert_type: str = "GENERAL"):
    """Send SMS alert with specified body"""
    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=SEND_TO_NUMBER
        )
        print(f"[{alert_type}] SMS sent. SID: {message.sid}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send SMS: {e}")
        return False

def get_location():
    """Get current location using IP geolocation"""
    try:
        g = geocoder.ip("me")
        if g.ok and g.latlng:
            lat, lon = g.latlng
            return lat, lon
    except Exception as e:
        print(f"[ERROR] Location detection failed: {e}")
    return None

def format_audio_alert_message(lat=None, lon=None):
    """Format message for audio trigger alerts"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if lat and lon:
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        return f"🔊 AUDIO ALERT: Trigger word detected!\nTime: {timestamp}\n" \
               f"Location: {lat:.6f}, {lon:.6f}\n{maps_link}"
    else:
        return f"🔊 AUDIO ALERT: Trigger word detected!\n{FALLBACK_LOCATION_TEXT}\nTime: {timestamp}"

def format_visual_alert_message(person_id, duration):
    """Format message for visual stalker alerts"""
    g = geocoder.ip('me')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if g.latlng:
        location_url = f"https://www.google.com/maps?q={g.latlng[0]},{g.latlng[1]}"
        location_text = f"Location: {location_url}"
    else:
        location_text = "Location not available"
    
    return f"👁️ STALKER ALERT: Person (ID:{person_id}) tracked for {duration:.1f} seconds!\n" \
           f"Time: {timestamp}\n{location_text}\nDeepSORT confirmed tracking"

def normalize_text(text: str):
    """Normalize text for keyword matching"""
    return text.lower().strip()

def contains_trigger(text: str):
    """Check if text contains any trigger keywords"""
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in TRIGGER_KEYWORDS)

def prepare_deepsort_detections(yolo_detections):
    """Convert YOLO detections to DeepSORT format"""
    deepsort_detections = []
    
    for detection in yolo_detections:
        x1, y1, x2, y2, conf, cls = detection
        
        # Only process person detections (class 0)
        if int(cls) != 0:
            continue
            
        # Check minimum area
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_PERSON_AREA:
            continue
        
        # Convert to DeepSORT format: ([x1, y1, x2, y2], confidence, detection_class)
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        deepsort_detections.append((bbox, float(conf), "person"))
    
    return deepsort_detections

def audio_detection_thread():
    """Thread function for continuous audio monitoring"""
    global last_audio_alert_time, system_running
    
    print("[AUDIO] Initializing speech recognition...")
    r = sr.Recognizer()
    
    try:
        # On RPi, you might need to specify the microphone device
        try:
            mic = sr.Microphone(device_index=MIC_INDEX)
        except:
            print(f"[AUDIO] Microphone index {MIC_INDEX} not found, using default")
            mic = sr.Microphone()
        
        print(f"[AUDIO] Microphone initialized")
        
        with mic as source:
            print("[AUDIO] Adjusting for ambient noise... stay quiet.")
            r.adjust_for_ambient_noise(source, duration=2)
        
        print("[AUDIO] Audio detection started. Listening for trigger words...")
        
        while system_running:
            try:
                with mic as source:
                    # Shorter timeout for RPi
                    audio = r.listen(source, timeout=3, phrase_time_limit=3)
                
                text = r.recognize_google(audio)
                print(f"[AUDIO] Heard: {text}")
                
                if contains_trigger(text):
                    current_time = time.time()
                    if current_time - last_audio_alert_time >= ALERT_COOLDOWN_SECS:
                        print("[AUDIO] Trigger word detected! Sending alert...")
                        
                        # Get location and send alert
                        location = get_location()
                        if location:
                            lat, lon = location
                            message = format_audio_alert_message(lat, lon)
                        else:
                            message = format_audio_alert_message()
                        
                        if send_sms(message, "AUDIO ALERT"):
                            last_audio_alert_time = current_time
                    else:
                        print("[AUDIO] Alert cooldown active.")
                        
            except sr.UnknownValueError:
                pass  # Could not understand audio - this is normal
            except sr.RequestError as e:
                print(f"[AUDIO] Google Speech Recognition error: {e}")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"[AUDIO] Unexpected error: {e}")
            
            time.sleep(0.5)  # Longer sleep for RPi
            
    except Exception as e:
        print(f"[AUDIO] Failed to initialize microphone: {e}")
        print("[AUDIO] Audio detection disabled.")

def visual_detection_main():
    """Main function for visual detection using YOLOv5 + DeepSORT"""
    global seen_times, alert_sent_ids, system_running, frame_count, yolo_model, deepsort_tracker
    
    if not load_yolo_model():
        print("[VISUAL] Cannot start visual detection without YOLO model")
        return
        
    if not initialize_deepsort():
        print("[VISUAL] Cannot start visual detection without DeepSORT")
        return
    
    print("[VISUAL] Starting camera...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print(f"[VISUAL] ERROR: Could not open camera at index {WEBCAM_INDEX}")
        return
    
    # Set camera properties for better RPi performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    
    print(f"[VISUAL] YOLOv5 + DeepSORT detection started at {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[VISUAL] Processing every {DETECTION_SKIP_FRAMES} frames. Press 'q' to quit.")
    
    inference_times = []
    tracking_times = []
    
    try:
        while system_running:
            ret, frame = cap.read()
            if not ret:
                print("[VISUAL] ERROR: Failed to grab frame.")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Only process every Nth frame to save CPU
            if frame_count % DETECTION_SKIP_FRAMES == 0:
                # YOLO inference
                start_yolo = time.time()
                
                with torch.no_grad():  # Disable gradient computation for inference
                    results = yolo_model(frame)
                    detections = results.xyxy[0].cpu().numpy()  # Move to CPU if on GPU
                
                yolo_time = time.time() - start_yolo
                inference_times.append(yolo_time)
                
                # Prepare detections for DeepSORT
                deepsort_detections = prepare_deepsort_detections(detections)
                
                # DeepSORT tracking
                start_tracking = time.time()
                tracks = deepsort_tracker.update_tracks(deepsort_detections, frame=frame)
                tracking_time = time.time() - start_tracking
                tracking_times.append(tracking_time)
                
                # Keep only recent timing data
                if len(inference_times) > 10:
                    inference_times.pop(0)
                if len(tracking_times) > 10:
                    tracking_times.pop(0)
                
                # Process confirmed tracks
                confirmed_tracks = [track for track in tracks if track.is_confirmed()]
                
                for track in confirmed_tracks:
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Draw bounding box and track ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"DeepSORT ID: {track_id}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Track time for each person
                    if track_id not in seen_times:
                        seen_times[track_id] = current_time
                        print(f"[DEEPSORT] New person confirmed - ID: {track_id}")
                    
                    elapsed = current_time - seen_times[track_id]
                    
                    # Show elapsed time on video
                    cv2.putText(frame, f"Tracked: {elapsed:.1f}s", (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Check for stalker behavior
                    if elapsed > STALKER_THRESHOLD_SECONDS and track_id not in alert_sent_ids:
                        print(f"[STALKER] ALERT! Person ID {track_id} tracked for {elapsed:.1f} seconds!")
                        
                        message = format_visual_alert_message(track_id, elapsed)
                        if send_sms(message, "STALKER ALERT"):
                            alert_sent_ids.add(track_id)
                
                # Clean up old seen_times for tracks that are no longer active
                active_ids = {track.track_id for track in confirmed_tracks}
                seen_times = {tid: time_val for tid, time_val in seen_times.items() 
                             if tid in active_ids}
            
            # Add comprehensive system status
            avg_yolo = np.mean(inference_times) if inference_times else 0
            avg_tracking = np.mean(tracking_times) if tracking_times else 0
            total_time = avg_yolo + avg_tracking
            
            cv2.putText(frame, f"RPi YOLOv5 + DeepSORT Emergency System", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"YOLO: {avg_yolo:.3f}s | DeepSORT: {avg_tracking:.3f}s | Total: {total_time:.3f}s", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count} | Confirmed tracks: {len([t for t in tracks if t.is_confirmed()])}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Alert IDs: {len(alert_sent_ids)} | Active: {len(seen_times)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show frame (can be disabled for headless operation)
            try:
                cv2.imshow("RPi YOLOv5 + DeepSORT Emergency Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # Headless mode - no display available
                pass
            
            # Control frame rate
            time.sleep(0.15)  # Slightly longer for DeepSORT processing
                
    except KeyboardInterrupt:
        print("[VISUAL] Interrupted by user.")
    except Exception as e:
        print(f"[VISUAL] Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[VISUAL] Visual detection stopped.")

def main():
    """Main function optimized for Raspberry Pi with YOLOv5 + DeepSORT"""
    global system_running
    
    print("="*65)
    print("  RASPBERRY PI 4 EMERGENCY DETECTION WITH YOLOv5 + DeepSORT")
    print("="*65)
    print("Audio Detection: Listening for trigger words")
    print("Visual Detection: YOLOv5 Nano + DeepSORT person tracking")
    print("Enhanced Tracking: Deep learning-based person re-identification")
    print("Press Ctrl+C to stop system")
    print("="*65)
    
    # Optimize RPi performance
    optimize_raspberry_pi()
    
    # Check system requirements
    print(f"[SYSTEM] PyTorch version: {torch.__version__}")
    print(f"[SYSTEM] CUDA available: {torch.cuda.is_available()}")
    
    # Validate Twilio configuration
    if (TWILIO_ACCOUNT_SID == "YOUR_ACCOUNT_SID" or 
        TWILIO_AUTH_TOKEN == "YOUR_AUTH_TOKEN"):
        print("[ERROR] Please configure Twilio credentials!")
        return
    
    try:
        # Start audio detection thread
        audio_thread = threading.Thread(target=audio_detection_thread, daemon=True)
        audio_thread.start()
        
        # Run visual detection in main thread
        visual_detection_main()
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down...")
    except Exception as e:
        print(f"[SYSTEM] Critical error: {e}")
    finally:
        system_running = False
        print("[SYSTEM] Emergency detection stopped.")

if __name__ == "__main__":
    main()
