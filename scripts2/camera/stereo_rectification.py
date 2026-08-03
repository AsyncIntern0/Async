"""
===============================================================================
Stereo Rectification Module
===============================================================================

Author  : Menaka R M
Project : Prototype Joint Detection using Stereo Vision

Description
-----------
This module performs stereo image rectification using the calibration
parameters obtained from stereo calibration.

Responsibilities
----------------
1. Load stereo calibration parameters.
2. Compute stereo rectification transforms.
3. Generate rectification maps.
4. Save rectification parameters for live stereo processing.

===============================================================================
"""

# =============================================================================
# Import Required Libraries
# =============================================================================

import cv2
import numpy as np

# =============================================================================
# Import Project Configuration
# =============================================================================

from camera.camera_config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    LEFT_CAMERA_FILE,
    RIGHT_CAMERA_FILE,
    STEREO_CALIBRATION_FILE,
    RECTIFICATION_FILE,
)

# =============================================================================
# Stereo Rectification Class
# =============================================================================


class StereoRectification:

    """
    Performs stereo image rectification using previously calibrated cameras.
    """

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(self):

        print("\nInitializing Stereo Rectification...\n")

        # -------------------------------------------------------------
        # Image Size
        # -------------------------------------------------------------

        self.image_size = (FRAME_WIDTH, FRAME_HEIGHT)

        # -------------------------------------------------------------
        # Camera Parameters
        # -------------------------------------------------------------

        self.left_camera_matrix = None
        self.right_camera_matrix = None

        self.left_distortion = None
        self.right_distortion = None

        # -------------------------------------------------------------
        # Stereo Parameters
        # -------------------------------------------------------------

        self.rotation_matrix = None
        self.translation_vector = None

        # -------------------------------------------------------------
        # Rectification Outputs
        # -------------------------------------------------------------

        self.R1 = None
        self.R2 = None

        self.P1 = None
        self.P2 = None

        self.Q = None

        self.left_roi = None
        self.right_roi = None

        # -------------------------------------------------------------
        # Rectification Maps
        # -------------------------------------------------------------

        self.left_map_x = None
        self.left_map_y = None

        self.right_map_x = None
        self.right_map_y = None

    # =========================================================================
    # Load Calibration Parameters
    # =========================================================================

    def load_calibration_parameters(self):

        print("Loading calibration parameters...")

        stereo_parameters = np.load(STEREO_CALIBRATION_FILE)

        self.left_camera_matrix = stereo_parameters["camera_matrix_left"]

        self.left_distortion = stereo_parameters["distortion_coefficients_left"]

        self.right_camera_matrix = stereo_parameters["camera_matrix_right"]

        self.right_distortion = stereo_parameters["distortion_coefficients_right"]

        self.rotation_matrix = stereo_parameters["rotation_matrix"]

        self.translation_vector = stereo_parameters["translation_vector"]

        self.image_size = tuple(stereo_parameters["image_size"])

        print("Calibration parameters loaded successfully.\n")

    # =========================================================================
    # Compute Stereo Rectification
    # =========================================================================

    def compute_rectification(self):

        print("Computing stereo rectification...\n")

        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.left_roi,
            self.right_roi,
        ) = cv2.stereoRectify(

            cameraMatrix1=self.left_camera_matrix,
            distCoeffs1=self.left_distortion,

            cameraMatrix2=self.right_camera_matrix,
            distCoeffs2=self.right_distortion,

            imageSize=self.image_size,

            R=self.rotation_matrix,
            T=self.translation_vector,

            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0

        )

        print("Stereo rectification completed.\n")
        print("Q Matrix:")
        print(self.Q)

    # =========================================================================
    # Generate Rectification Maps
    # =========================================================================

    def generate_rectification_maps(self):

        print("Generating rectification maps...\n")

        (
            self.left_map_x,
            self.left_map_y,
        ) = cv2.initUndistortRectifyMap(

            cameraMatrix=self.left_camera_matrix,
            distCoeffs=self.left_distortion,

            R=self.R1,
            newCameraMatrix=self.P1,

            size=self.image_size,

            m1type=cv2.CV_32FC1,

        )

        (
            self.right_map_x,
            self.right_map_y,
        ) = cv2.initUndistortRectifyMap(

            cameraMatrix=self.right_camera_matrix,
            distCoeffs=self.right_distortion,

            R=self.R2,
            newCameraMatrix=self.P2,

            size=self.image_size,

            m1type=cv2.CV_32FC1,

        )

        print("Rectification maps generated.\n")

    # =========================================================================
    # Save Rectification Parameters
    # =========================================================================

    def save_rectification_maps(self):

        print("Saving rectification maps...\n")

        np.savez(

            RECTIFICATION_FILE,

            left_map_x=self.left_map_x,
            left_map_y=self.left_map_y,

            right_map_x=self.right_map_x,
            right_map_y=self.right_map_y,

            R1=self.R1,
            R2=self.R2,

            P1=self.P1,
            P2=self.P2,

            Q=self.Q,

            left_roi=self.left_roi,
            right_roi=self.right_roi,
            image_size=self.image_size

        )

        print(f"Rectification maps saved to:\n{RECTIFICATION_FILE}\n")

    # =========================================================================
    # Run Complete Rectification Pipeline
    # =========================================================================

    def run(self):

        print("========================================================")
        print("Stereo Rectification")
        print("========================================================\n")

        self.load_calibration_parameters()

        self.compute_rectification()

        self.generate_rectification_maps()

        self.save_rectification_maps()

        print("========================================================")
        print("Stereo Rectification Completed Successfully")
        print("========================================================\n")


# =============================================================================
# Main Function
# =============================================================================

def main():

    rectification = StereoRectification()

    rectification.run()
    


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()
