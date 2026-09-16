import cv2
import mediapipe as mp
import math

# 1. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Define the exact points around the eyes (MediaPipe landmark indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Helper function to calculate distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Helper function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points, landmarks):
    # Vertical distances
    v1 = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    v2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal distance
    h = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

# 3. Start the Video Stream (0 is usually your laptop's built-in webcam)
cap = cv2.VideoCapture(0)

# Variables to track drowsiness
DROWSY_THRESHOLD = 0.22  # If EAR drops below this, eyes are likely closed
FRAMES_TO_WAIT = 20      # Number of consecutive frames eyes must be closed to trigger alert
closed_frames_count = 0

print("Starting EdgeVision DMS... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB before processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE, landmarks)
            right_ear = calculate_ear(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            # Drowsiness Logic
            if avg_ear < DROWSY_THRESHOLD:
                closed_frames_count += 1
            else:
                closed_frames_count = 0 # Reset if they open their eyes

            # Display EAR on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Trigger Warning if eyes closed too long
            if closed_frames_count >= FRAMES_TO_WAIT:
                cv2.putText(frame, "DROWSINESS DETECTED!", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Show the video feed
    cv2.imshow('EdgeVision - Driver Monitoring', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()