import cv2
from ultralytics import YOLO

# 1. Load the YOLOv11 Nano model (fastest version, perfect for edge devices)
# It will automatically download a small ~5MB file the first time you run this
model = YOLO('yolo11n.pt')

# 2. Start the Video Stream
# 0 is your webcam. (If you have a video file, change 0 to "video_name.mp4")
cap = cv2.VideoCapture(0)

print("Starting EdgeVision ADAS... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video stream.")
        break

    # 3. Run YOLO inference on the frame
    # We use 'classes=[0, 2, 3, 5, 7]' to only detect: 
    # Person (0), Car (2), Motorcycle (3), Bus (5), and Truck (7)
    # 'conf=0.5' means it must be 50% sure before drawing a box
    results = model(frame, classes=[0, 2, 3, 5, 7], conf=0.5)

    # 4. Draw the bounding boxes and labels onto the frame
    annotated_frame = results[0].plot()

    # 5. Show the video feed
    cv2.imshow('EdgeVision - Road Awareness', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()