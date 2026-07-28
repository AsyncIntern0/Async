from ultralytics import YOLO
import cv2
USE_PICAMERA2 = True
from picamera2 import Picamera2

# To measure the time which being taken by yolo to predict the model 
import time
# Load Model
# ---------------------------------------------------
model = YOLO("/home/radxa/Prototype_JointDetection/models/best.pt")

# ---------------------------------------------------
# camera
# ---------------------------------------------------

if USE_PICAMERA2:

    from picamera2 import Picamera2

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "BGR888"
    }
)

    picam2.configure(config)

    picam2.start()

else:

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

print("Camera Started...")
print("Press 'q' to quit.")

# ---------------------------------------------------
# Frame Skip Settings
# ---------------------------------------------------
FRAME_SKIP = 3      # Run YOLO every 3rd frame
frame_count = 0

annotated = None

# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------
while True:

    if USE_PICAMERA2:

        frame = picam2.capture_array()
        # Convert RGBA -> BGR
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    else:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

    # Always increment
    frame_count += 1

    # Run YOLO every 3rd frame
    if frame_count % FRAME_SKIP == 0:
        
        results = model(
            frame,
            imgsz=320,
            conf=0.60,
            verbose=False
        )
        

        for box in results[0].boxes:
            print(
                model.names[int(box.cls)],
                float(box.conf)
            )

        annotated = results[0].plot()
    # -----------------------------------------------
    # Display latest result
    # -----------------------------------------------
    if annotated is None:
        annotated = frame

    cv2.imshow("Prototype Detection", annotated)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ---------------------------------------------------
# Cleanup
# ---------------------------------------------------
if USE_PICAMERA2:

    picam2.stop()

else:

    cap.release()
cv2.destroyAllWindows()
