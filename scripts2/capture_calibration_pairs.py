#!/usr/bin/env python3

"""
capture_calibration_images.py

Capture synchronized calibration image pairs manually.

Controls
--------
SPACE : Save current image pair
Q     : Quit
"""

import cv2
import time
from pathlib import Path
from picamera2 import Picamera2


# ==========================================================
# Configuration
# ==========================================================

LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

LEFT_FOLDER = "calibration_images/left"
RIGHT_FOLDER = "calibration_images/right"

# ==========================================================


def setup_camera(camera_id):

    picam = Picamera2(camera_id)

    config = picam.create_preview_configuration(
        main={
            "size": (FRAME_WIDTH, FRAME_HEIGHT),
            "format": "RGB888"
        }
    )

    picam.configure(config)
    picam.start()

    return picam


def main():

    Path(LEFT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(RIGHT_FOLDER).mkdir(parents=True, exist_ok=True)

    print("Opening Cameras...")

    left_cam = setup_camera(LEFT_CAMERA_ID)

    time.sleep(1)

    right_cam = setup_camera(RIGHT_CAMERA_ID)

    time.sleep(2)

    print("Streaming...")
    print("SPACE : Capture")
    print("Q     : Quit\n")

    image_index = 0

    while True:

        left_frame = left_cam.capture_array()
        right_frame = right_cam.capture_array()

        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

        cv2.putText(
            left_frame,
            f"LEFT CAMERA   Pair : {image_index}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            right_frame,
            f"RIGHT CAMERA   Pair : {image_index}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):

            left_name = Path(LEFT_FOLDER) / f"pair_{image_index:03d}.jpg"
            right_name = Path(RIGHT_FOLDER) / f"pair_{image_index:03d}.jpg"

            cv2.imwrite(str(left_name), left_frame)
            cv2.imwrite(str(right_name), right_frame)

            print(f"Saved Pair {image_index:03d}")

            image_index += 1

        elif key == ord('q'):
            break

    left_cam.stop()
    right_cam.stop()

    cv2.destroyAllWindows()

    print("\nFinished.")


if __name__ == "__main__":
    main()
