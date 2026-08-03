"""
===========================================================================
Stereo Camera Module
===========================================================================

Author  : Menaka R M
Project : Prototype Joint Detection (Stereo Vision)
Purpose : Manage two CSI cameras connected to Raspberry Pi 5.

Responsibilities
----------------
1. Initialize both cameras.
2. Configure camera settings.
3. Start both cameras.
4. Capture synchronized frames.
5. Stop cameras safely.

This module DOES NOT perform:
    - Marker Detection
    - Calibration
    - Rectification
    - Tracking
    - Triangulation

===========================================================================

"""

from picamera2 import Picamera2
import time
from camera.camera_config import (
    LEFT_CAMERA_ID,
    RIGHT_CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    PIXEL_FORMAT,
    STARTUP_DELAY
)


class StereoCamera:
    """
    Stereo Camera Manager
    """



    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(self):
        """
        Initialize stereo camera objects.

        Cameras are NOT started here.
        """

        try:
            self.left_camera = Picamera2(LEFT_CAMERA_ID)
            self.right_camera = Picamera2(RIGHT_CAMERA_ID)

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize stereo cameras.\n{e}"
            )

        self.left_started = False
        self.right_started = False

    # ==========================================================
    # Start Cameras
    # ==========================================================

    def start(self):
        """
        Configure and start both cameras.
        """

        try:

            # ----------------------------------------------
            # Configure Left Camera
            # ----------------------------------------------

            left_config = self.left_camera.create_preview_configuration(
                main={
                    "size": ( FRAME_WIDTH,FRAME_HEIGHT),
                    "format": PIXEL_FORMAT,
                },
                sensor={
                    "output_size": (1296, 972)
                }
            )

            self.left_camera.configure(left_config)

            # ----------------------------------------------
            # Configure Right Camera
            # ----------------------------------------------

            right_config = self.right_camera.create_preview_configuration(
                main={
                    "size": ( FRAME_WIDTH, FRAME_HEIGHT),
                    "format": PIXEL_FORMAT,
                },
                sensor={
                    "output_size": (1296, 972)
                }
               
            )

            self.right_camera.configure(right_config)

            # ----------------------------------------------
            # Start Cameras
            # ----------------------------------------------

            self.left_camera.start()
            self.right_camera.start()

            self.left_started = True
            self.right_started = True

            # Allow sensors to stabilize
            time.sleep(STARTUP_DELAY)

            print("[INFO] Left Camera Started")
            print("[INFO] Right Camera Started")

        except Exception as e:

            self.stop()

            raise RuntimeError(
                f"Unable to start stereo cameras.\n{e}"
            )

    # ==========================================================
    # Capture Frames
    # ==========================================================

    def capture_frames(self):
        """
        Capture one frame from each camera.

        Returns
        -------
        left_frame
        right_frame
        """

        if not (self.left_started and self.right_started):

            raise RuntimeError(
                "Stereo cameras have not been started."
            )

        try:

            left_frame = self.left_camera.capture_array()

            right_frame = self.right_camera.capture_array()

            return left_frame, right_frame

        except Exception as e:

            raise RuntimeError(
                f"Failed to capture stereo frames.\n{e}"
            )

    # ==========================================================
    # Stop Cameras
    # ==========================================================

    def stop(self):
        """
        Stop both cameras safely.
        """

        try:

            if self.left_started:
                self.left_camera.stop()
                self.left_started = False

            if self.right_started:
                self.right_camera.stop()
                self.right_started = False

            print("[INFO] Stereo cameras stopped.")

        except Exception as e:

            print(f"[WARNING] Error while stopping cameras: {e}")
