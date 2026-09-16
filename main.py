import cv2
import mediapipe as mp
import math
import threading
import time
import pyttsx3
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# 1. Shared Global Variables
latest_frames = {"dms": None, "adas": None}
safety_status = {"drowsy": False, "collision_warning": False}
running = True

# --- MODULE 0: EDGE DATABASE ENGINE ---
def init_db():
    """Creates a local SQLite database file to act as the vehicle's black box."""
    conn = sqlite3.connect("edgevision_telematics.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS safety_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  event_type TEXT, 
                  details TEXT)''')
    conn.commit()
    conn.close()

def log_alert(event_type, details):
    """Saves a safety violation timestamp into the database cleanly."""
    try:
        conn = sqlite3.connect("edgevision_telematics.db")
        c = conn.cursor()
        c.execute("INSERT INTO safety_logs (timestamp, event_type, details) VALUES (?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, details))
        conn.commit()
        conn.close()
    except Exception as e:
        pass

# --- MODULE 1: DRIVER MONITORING THREAD ---
def dms_worker():
    global latest_frames, safety_status, running
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def calc_ear(eye, landmarks):
        v1 = math.dist([landmarks[eye[1]].x, landmarks[eye[1]].y], [landmarks[eye[5]].x, landmarks[eye[5]].y])
        v2 = math.dist([landmarks[eye[2]].x, landmarks[eye[2]].y], [landmarks[eye[4]].x, landmarks[eye[4]].y])
        h = math.dist([landmarks[eye[0]].x, landmarks[eye[0]].y], [landmarks[eye[3]].x, landmarks[eye[3]].y])
        return (v1 + v2) / (2.0 * h)

    cap = cv2.VideoCapture(0)
    closed_frames = 0

    while running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                ear = (calc_ear(LEFT_EYE, face.landmark) + calc_ear(RIGHT_EYE, face.landmark)) / 2.0
                
                if ear < 0.22:
                    closed_frames += 1
                else:
                    closed_frames = 0
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                if closed_frames >= 15:
                    safety_status["drowsy"] = True
                    cv2.putText(frame, "DROWSY!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    safety_status["drowsy"] = False

        latest_frames["dms"] = frame
        time.sleep(0.01)

    cap.release()

# --- MODULE 2: ROAD AWARENESS THREAD ---
def adas_worker():
    global latest_frames, safety_status, running
    
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture('test_video.mp4') 

    while running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, classes=[0, 2, 3, 5, 7], conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        hazard = False
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 15000: 
                hazard = True
                cv2.putText(annotated_frame, "COLLISION WARNING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                break
        
        safety_status["collision_warning"] = hazard
        latest_frames["adas"] = annotated_frame
        time.sleep(0.01)

    cap.release()

# --- MODULE 3: AUDIO & TELEMATICS LOGGING THREAD ---
def speak_utterance(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        pass

def audio_worker():
    global safety_status, running
    
    last_alert_time = {"drowsy": 0, "collision": 0}
    COOLDOWN_SECONDS = 3.5

    while running:
        current_time = time.time()
        
        if safety_status["drowsy"] and (current_time - last_alert_time["drowsy"] > COOLDOWN_SECONDS):
            last_alert_time["drowsy"] = current_time
            print("[ALERT] Driver Drowsy! Logging to database...")
            log_alert("DRIVER_DROWSINESS", "Critical fatigue detected (EAR < 0.22)")
            threading.Thread(target=speak_utterance, args=("Warning! Driver Drowsy!",), daemon=True).start()
            
        elif safety_status["collision_warning"] and (current_time - last_alert_time["collision"] > COOLDOWN_SECONDS):
            last_alert_time["collision"] = current_time
            print("[ALERT] Forward Hazard! Logging to database...")
            log_alert("FORWARD_COLLISION_HAZARD", "Vehicle in immediate forward trajectory")
            threading.Thread(target=speak_utterance, args=("Warning! Forward Hazard!",), daemon=True).start()
            
        time.sleep(0.1)

# --- MAIN THREAD: DISPLAY & MASTER CONTROL ---
print("Initializing EdgeVision Master Engine & Telematics Database...")
init_db() # Create the database file if it doesn't exist yet

cv2.namedWindow("EdgeVision - Driver Monitoring (In-Cabin)", cv2.WINDOW_NORMAL)
cv2.moveWindow("EdgeVision - Driver Monitoring (In-Cabin)", 50, 100)

cv2.namedWindow("EdgeVision - Road Awareness (Out-Cabin)", cv2.WINDOW_NORMAL)
cv2.moveWindow("EdgeVision - Road Awareness (Out-Cabin)", 700, 100)

t1 = threading.Thread(target=dms_worker, daemon=True)
t2 = threading.Thread(target=adas_worker, daemon=True)
t3 = threading.Thread(target=audio_worker, daemon=True)

t1.start()
t2.start()
t3.start()

while True:
    if latest_frames["dms"] is not None:
        cv2.imshow("EdgeVision - Driver Monitoring (In-Cabin)", latest_frames["dms"])
        
    if latest_frames["adas"] is not None:
        cv2.imshow("EdgeVision - Road Awareness (Out-Cabin)", latest_frames["adas"])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Shutting down EdgeVision...")
        running = False
        break

time.sleep(0.5)
cv2.destroyAllWindows()
print("System powered off safely.")