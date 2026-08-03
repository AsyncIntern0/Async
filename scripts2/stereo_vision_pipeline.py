"""
==========================================================
Stereo Vision Pipeline (Version 1)

Stage 1
--------
Stereo Camera
        ↓
YOLO Detection
        ↓
Target Detection
        ↓
Target Verification


==========================================================
"""



import cv2
import numpy as np
from ultralytics import YOLO

from camera.stereo_camera import StereoCamera
from camera.camera_config import (

    RECTIFICATION_FILE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

from vision.detect_targets import TargetDetector

from vision.verify_targets import (
    TargetVerifier,
    draw_verified_targets
)
from geometry.triangulate import (
    load_stereo_params,
    get_3d_coordinates,
)
from geometry.stereo_correspondence import (
    StereoCorrespondence
)
from geometry.assign_joints import (
    JointAssigner
)
# ---------------------------------------------------------
# Load YOLO
# ---------------------------------------------------------

model = YOLO(
    "/home/radxa/Prototype_JointDetection/models/best.pt"
)


# ---------------------------------------------------------
# Load Stereo Rectification Maps
# ---------------------------------------------------------

rectification = np.load(RECTIFICATION_FILE)

left_map_x = rectification["left_map_x"]
left_map_y = rectification["left_map_y"]

right_map_x = rectification["right_map_x"]
right_map_y = rectification["right_map_y"]

Q, P1, P2 = load_stereo_params(
    RECTIFICATION_FILE
)


# ---------------------------------------------------------
# Initialize Modules
# ---------------------------------------------------------

camera = StereoCamera()
camera.start()

correspondence = StereoCorrespondence()

detector = TargetDetector()

verifier = TargetVerifier()

joint_assigner = JointAssigner()


# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------

while True:

    # ==========================================
    # Capture Stereo Frames
    # ==========================================

    left_frame, right_frame = camera.capture_frames()
    #print("Left:", left_frame.shape)
    #print("Right:", right_frame.shape)
    # ==========================================
    # Rectify Stereo Frames
    # ==========================================

    left_frame = cv2.remap(
        left_frame,
        left_map_x,
        left_map_y,
        cv2.INTER_LINEAR
    )

    right_frame = cv2.remap(
        right_frame,
        right_map_x,
        right_map_y,
        cv2.INTER_LINEAR
    )

    left_display = left_frame.copy()

    right_display = right_frame.copy()
    

    # ==========================================
    # LEFT CAMERA
    # ==========================================

    left_results = model(
        left_frame,
        imgsz=256,
        conf=0.60,
        verbose=False
    )

    left_gray, left_thresh, left_candidates = detector.detect(
        left_frame
    )

    left_verified = verifier.verify_all(
        left_gray,
        left_candidates
    )
    

    left_display = draw_verified_targets(
            left_display,
            left_verified
        )

    # Draw YOLO Box

    for box in left_results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        cv2.rectangle(
            left_display,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )


    # ==========================================
    # RIGHT CAMERA
    # ==========================================

    right_results = model(
        right_frame,
        imgsz=256,
        conf=0.60,
        verbose=False
    )

    right_gray, right_thresh, right_candidates = detector.detect(
        right_frame
    )

    right_verified = verifier.verify_all(
        right_gray,
        right_candidates
    )
    print("\nLEFT")

    for t in left_verified:

        print(

            t.marker.center_x,

            t.marker.center_y

        )

    print("\nRIGHT")

    for t in right_verified:

        print(

            t.marker.center_x,

            t.marker.center_y

        )
    # ==========================================
    # Stereo Correspondence
    # ==========================================

    stereo_matches = correspondence.find_matches(

        left_verified,

        right_verified

    )
    print(f"Left Verified : {len(left_verified)}")
    print(f"Right Verified: {len(right_verified)}")
    print(f"Stereo Matches: {len(stereo_matches)}")
    
    # ==========================================
    # Joint Assignment
    # ==========================================

    assignment = joint_assigner.assign_joint_ids(

        stereo_matches

)
    if assignment.success:

        print("Joint assignment successful.")

    else:

        print(assignment.message)
    if assignment.success:

        skeleton = assignment.skeleton

        print("--------------------------------")

        print("Assigned IDs")

        print("--------------------------------")

        for joint in [

            skeleton.X1,

            skeleton.X2,

            skeleton.X3,

            skeleton.X4,

            skeleton.X5,

            skeleton.X6,

            skeleton.X7

        ]:

            print(joint.joint_id)
    DEBUG= False
    if DEBUG:
        print(f"Stereo Matches : {len(stereo_matches)}")

        for i, match in enumerate(stereo_matches, start=1):

            print(

                f"M{i} "

                f"L=({match.left_target.marker.center_x:.1f}, "

                f"{match.left_target.marker.center_y:.1f}) "

                f"R=({match.right_target.marker.center_x:.1f}, "

                f"{match.right_target.marker.center_y:.1f}) "

                f"D={match.disparity:.2f}"

            )
    points_3d, stereo_matches = get_3d_coordinates(
        stereo_matches,
        Q=Q,
        P1=P1,
        P2=P2
    )
    # ==========================================================
    # 3D Telemetry Panel
    # ==========================================================

    # Create a semi-transparent panel
    overlay = left_display.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (250, 460),
        (35, 35, 35),
        -1
    )

    left_display = cv2.addWeighted(
        overlay,
        0.65,
        left_display,
        0.35,
        0
    )

    # Panel Title
    cv2.putText(
        left_display,
        "3D TELEMETRY",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Draw each target's coordinates
    start_y = 65

    for idx, point in enumerate(points_3d, start=1):

        X, Y, Z = point

        cv2.putText(
            left_display,
            f"T{idx}",
            (20, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            left_display,
            f"X : {X:8.2f} mm",
            (60, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            left_display,
            f"Y : {Y:8.2f} mm",
            (60, start_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            left_display,
            f"Z : {Z:8.2f} mm",
            (60, start_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1,
            cv2.LINE_AA
        )

        start_y += 60

    right_display = draw_verified_targets(
        right_display,
        right_verified
    )

    for box in right_results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        cv2.rectangle(
            right_display,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

 


    # ==========================================
    # Display Stereo Images
    # ==========================================

    stereo_display = cv2.hconcat([
        left_display,
        right_display
    ])

    cv2.imshow(
        "Stereo Vision Pipeline",
        stereo_display
    )

    key = cv2.waitKey(1)

    if key == ord('q'):

        break

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

camera.stop()

cv2.destroyAllWindows()
