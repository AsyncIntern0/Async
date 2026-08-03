#!/usr/bin/env python3

import argparse
import cv2
from picamera2 import Picamera2
from datetime import datetime


def main():

    parser = argparse.ArgumentParser(
        description="Capture image from Raspberry Pi camera"
    )

    parser.add_argument(
        "--camera",
        type=int,
        required=True,
        choices=[0, 1],
        help="Camera ID (0 or 1)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480
    )

    args = parser.parse_args()

    picam = Picamera2(args.camera)

    config = picam.create_preview_configuration(
        main={
            "size": (args.width, args.height),
            "format": "RGB888"
        }
    )

    picam.configure(config)
    picam.start()

    print("\n--------------------------------")
    print(f"Camera {args.camera} started")
    print("SPACE : Capture image")
    print("Q     : Quit")
    print("--------------------------------\n")

    image_counter = 1

    while True:

        frame = picam.capture_array()

        display = frame.copy()

        cv2.putText(
            display,
            f"Camera {args.camera}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Camera Preview", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):

            filename = (
                f"camera{args.camera}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            cv2.imwrite(filename, frame)

            print(f"Saved : {filename}")

            image_counter += 1

        elif key == ord('q'):

            break

    picam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
