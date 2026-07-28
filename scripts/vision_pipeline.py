from ultralytics import YOLO
import cv2

USE_PICAMERA2 = True
from picamera2 import Picamera2

import time

from detect_targets import TargetDetector
from verify_targets import TargetVerifier, draw_verified_targets
from assign_joints import JointAssigner
from joint_angles import JointAngleCalculator

from track_joints import SkeletonTracker
# ---------------------------------------------------
# Load YOLO Model
# ---------------------------------------------------

model = YOLO("/home/radxa/Prototype_JointDetection/models/best.pt")

frame_count=0
FRAME_SKIP = 10      # Run YOLO every 5th frame

# ---------------------------------------------------
# Initialize Detector & Verifier
# ---------------------------------------------------

detector = TargetDetector()
verifier = TargetVerifier()
assigner = JointAssigner()

tracker = SkeletonTracker()

angle_calculator = JointAngleCalculator()

# ---------------------------------------------------
# Camera
# ---------------------------------------------------

if USE_PICAMERA2:

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

# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------

while True:

    # =====================================================
    # 1. CAPTURE CAMERA FRAME
    # =====================================================

    if USE_PICAMERA2:

        frame = picam2.capture_array()

    else:

        ret, frame = cap.read()

        if not ret:
            break

    display = frame.copy()

    # Always increment frame counter
    frame_count += 1


    # =====================================================
    # MAIN PROCESSING
    # Runs only every FRAME_SKIP frame
    # =====================================================

    if frame_count % FRAME_SKIP == 0:


        # =================================================
        # BEFORE LOCK
        # YOLO → DETECT → VERIFY → ASSIGN → LOCK
        # =================================================

        if not tracker.locked:


            # ---------------------------------------------
            # YOLO PROTOTYPE DETECTION
            # ---------------------------------------------

            yolo_start = time.perf_counter()

            results = model(
                frame,
                imgsz=256,
                conf=0.60,
                verbose=False
            )

            yolo_end = time.perf_counter()

            yolo_time = (
                yolo_end - yolo_start
            ) * 1000

            print(
                f"YOLO TIME: {yolo_time:.2f} ms"
            )


            # ---------------------------------------------
            # Draw YOLO Bounding Box
            # ---------------------------------------------

            boxes = results[0].boxes

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                conf = float(box.conf)

                cls = int(box.cls)

                label = model.names[cls]

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )


            # ---------------------------------------------
            # TARGET DETECTION
            # ---------------------------------------------

            detect_start = time.perf_counter()

            gray, thresh, candidates = detector.detect(
                frame
            )

            detect_end = time.perf_counter()

            detect_time = (
                detect_end - detect_start
            ) * 1000

            print(
                f"DETECTOR TIME: {detect_time:.2f} ms"
            )

            print(
                "Candidates:",
                len(candidates)
            )


            # ---------------------------------------------
            # Draw Candidate Markers
            # ---------------------------------------------

            for marker in candidates:

                cv2.circle(
                    display,
                    (
                        marker.center_x,
                        marker.center_y
                    ),
                    int(marker.radius),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    display,
                    (
                        marker.center_x,
                        marker.center_y
                    ),
                    3,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    "C",
                    (
                        marker.center_x + 8,
                        marker.center_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


            # ---------------------------------------------
            # TARGET VERIFICATION
            # ---------------------------------------------

            verify_start = time.perf_counter()

            verified = verifier.verify_all(
                gray,
                candidates
            )

            verify_end = time.perf_counter()

            verify_time = (
                verify_end - verify_start
            ) * 1000

            print(
                f"VERIFY TIME: {verify_time:.2f} ms"
            )


            # ---------------------------------------------
            # JOINT ASSIGNMENT
            # ---------------------------------------------

            assign_start = time.perf_counter()

            assignment = assigner.assign(
                verified
            )

            assign_end = time.perf_counter()

            assign_time = (
                assign_end - assign_start
            ) * 1000

            print(
                f"ASSIGN TIME: {assign_time:.2f} ms"
            )


            # ---------------------------------------------
            # LOCK SKELETON
            # ---------------------------------------------

            if assignment.success:

                tracker.lock(
                    assignment.skeleton
                )

                print(
                    "Skeleton successfully locked."
                )

                print(
                    "YOLO inference is now disabled."
                )


        # =================================================
        # AFTER LOCK
        # DETECT → TRACK
        #
        # YOLO DOES NOT RUN HERE
        # =================================================

        else:


            # ---------------------------------------------
            # TARGET DETECTION
            # ---------------------------------------------

            detect_start = time.perf_counter()

            gray, thresh, candidates = detector.detect(
                frame
            )

            detect_end = time.perf_counter()

            detect_time = (
                detect_end - detect_start
            ) * 1000

            print(
                f"DETECTOR TIME: {detect_time:.2f} ms"
            )

            print(
                "Candidates:",
                len(candidates)
            )


            # ---------------------------------------------
            # Draw Candidate Markers
            # ---------------------------------------------

            for marker in candidates:

                cv2.circle(
                    display,
                    (
                        marker.center_x,
                        marker.center_y
                    ),
                    int(marker.radius),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    display,
                    (
                        marker.center_x,
                        marker.center_y
                    ),
                    3,
                    (0, 0, 255),
                    -1
                )


            # ---------------------------------------------
            # TRACK JOINTS USING CANDIDATES
            # ---------------------------------------------

            track_start = time.perf_counter()

            tracker.update(
                candidates
            )

            track_end = time.perf_counter()

            track_time = (
                track_end - track_start
            ) * 1000

            print(
                f"TRACK TIME: {track_time:.2f} ms"
            )


        # ---------------------------------------------
        # Debug Information
        # ---------------------------------------------

        print(
            "Locked:",
            tracker.locked
        )

        print(
            "Tracked Joints:",
            len(tracker.tracked_joints)
        )


    # =====================================================
    # JOINT ANGLE CALCULATION
    #
    # Runs every frame AFTER skeleton lock
    # =====================================================

    if tracker.locked:

        angle_start = time.perf_counter()

        joint_angles = (
            angle_calculator.calculate_joint_angles(
                tracker.tracked_joints
            )
        )

        angle_end = time.perf_counter()

        angle_time = (
            angle_end - angle_start
        ) * 1000

        print(
            f"ANGLE TIME: {angle_time:.2f} ms"
        )


        # Print angle information

        angle_calculator.print_angles(
            tracker.tracked_joints
        )


    # =====================================================
    # DRAW TRACKED JOINTS
    # =====================================================

    tracker.draw(
        display
    )


    # =====================================================
    # DRAW JOINT ANGLES
    # =====================================================

    if tracker.locked:

        angle_calculator.draw_angles(
            display,
            tracker.tracked_joints,
            joint_angles
        )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Vision Pipeline",
        display
    )


    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if USE_PICAMERA2:

    picam2.stop()

else:

    cap.release()
cv2.destroyAllWindows()
