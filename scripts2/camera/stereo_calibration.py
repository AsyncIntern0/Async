"""
===========================================================================
Stereo Camera Calibration
===========================================================================

Author  : Menaka R M
Project : Prototype Joint Detection (Stereo Vision)

Purpose
-------
Calibrate the stereo camera system using captured chessboard image pairs.
This module estimates the intrinsic parameters of each camera and prepares
the required data for stereo calibration.

Workflow
--------
1. Load Stereo Image Pairs
2. Generate Chessboard Object Points
3. Detect Chessboard Corners
4. Refine Corner Locations
5. Calibrate Left Camera
6. Calibrate Right Camera
7. Perform Stereo Calibration
8. Save Calibration Parameters

Outputs
-------
left_camera.npz
right_camera.npz
stereo_calibration.npz

===========================================================================
"""

# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import glob
import cv2
import numpy as np

# ==========================================================
# Import Project Configuration
# ==========================================================

from camera.camera_config import (
    LEFT_IMAGE_FOLDER,
    RIGHT_IMAGE_FOLDER,
    CHESSBOARD_SIZE,
    SQUARE_SIZE,
    CALIBRATION_OUTPUT_FOLDER,
    LEFT_CAMERA_FILE,
    RIGHT_CAMERA_FILE,
    STEREO_CALIBRATION_FILE,
)
# ==========================================================
# Stereo Calibration Class
# ==========================================================


class StereoCalibration:

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):

        print("\n==============================================")
        print("Initializing Stereo Camera Calibration")
        print("==============================================")

        # --------------------------------------------------
        # Chessboard Configuration
        # --------------------------------------------------

        self.chessboard_size = CHESSBOARD_SIZE
        self.square_size = SQUARE_SIZE

        # --------------------------------------------------
        # Image Paths
        # --------------------------------------------------

        self.left_image_paths = []
        self.right_image_paths = []

        # --------------------------------------------------
        # Calibration Data Containers
        # --------------------------------------------------

        # Real world chessboard coordinates
        self.object_points = []

        # Image points from left camera
        self.left_image_points = []

        # Image points from right camera
        self.right_image_points = []

        # Template object points
        self.object_point_template = None

        # Image Size
        self.image_size = None

        print("Initialization Completed.\n")

    # ======================================================
    # Load Stereo Image Pairs
    # ======================================================

    def load_image_pairs(self):

        print("Loading stereo image pairs...")

        self.left_image_paths = sorted(
            glob.glob(
                os.path.join(
                    LEFT_IMAGE_FOLDER,
                    "*.jpg"
                )
            )
        )

        self.right_image_paths = sorted(
            glob.glob(
                os.path.join(
                    RIGHT_IMAGE_FOLDER,
                    "*.jpg"
                )
            )
        )

        # --------------------------------------------------
        # Validate Image Count
        # --------------------------------------------------

        if len(self.left_image_paths) == 0:

            raise FileNotFoundError(
                "No left calibration images found."
            )

        if len(self.right_image_paths) == 0:

            raise FileNotFoundError(
                "No right calibration images found."
            )

        if len(self.left_image_paths) != len(self.right_image_paths):

            raise ValueError(
                "Mismatch between left and right image counts."
            )

        print(f"Total Stereo Image Pairs : {len(self.left_image_paths)}\n")

    # ======================================================
    # Generate Chessboard Object Points
    # ======================================================

    def create_object_points(self):

        print("Generating chessboard object points...")

        object_points = np.zeros(
            (
                self.chessboard_size[0] *
                self.chessboard_size[1],
                3
            ),
            np.float32
        )

        object_points[:, :2] = np.mgrid[
            0:self.chessboard_size[0],
            0:self.chessboard_size[1]
        ].T.reshape(-1, 2)

        object_points *= self.square_size

        self.object_point_template = object_points

        print(
            f"Generated {len(object_points)} chessboard corner coordinates.\n"
        )
        # ======================================================
    # Detect Chessboard Corners
    # ======================================================

    def detect_corners(self):

        print("Detecting chessboard corners...\n")

        successful_pairs = 0
        failed_pairs = 0

        # --------------------------------------------------
        # Corner Refinement Criteria
        # --------------------------------------------------

        termination_criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        # --------------------------------------------------
        # Process Every Stereo Image Pair
        # --------------------------------------------------

        for pair_index, (left_path, right_path) in enumerate(

            zip(
                self.left_image_paths,
                self.right_image_paths
            ),

            start=1

        ):

            # ----------------------------------------------
            # Read Stereo Images
            # ----------------------------------------------

            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)

            if left_image is None or right_image is None:

                print(
                    f"[Pair {pair_index:02d}] "
                    "Unable to read image pair."
                )

                failed_pairs += 1
                continue

            # ----------------------------------------------
            # Store Image Size
            # ----------------------------------------------

            if self.image_size is None:

                self.image_size = (
                    left_image.shape[1],
                    left_image.shape[0]
                )

            # ----------------------------------------------
            # Convert Images to Grayscale
            # ----------------------------------------------

            left_gray = cv2.cvtColor(
                left_image,
                cv2.COLOR_BGR2GRAY
            )

            right_gray = cv2.cvtColor(
                right_image,
                cv2.COLOR_BGR2GRAY
            )

            # ----------------------------------------------
            # Detect Chessboard Corners
            # ----------------------------------------------

            left_found, left_corners = cv2.findChessboardCorners(
                left_gray,
                self.chessboard_size
            )

            right_found, right_corners = cv2.findChessboardCorners(
                right_gray,
                self.chessboard_size
            )

            # ----------------------------------------------
            # Validate Detection
            # ----------------------------------------------

            if not (left_found and right_found):

                print(
                    f"[Pair {pair_index:02d}] "
                    "Chessboard detection failed."
                )

                failed_pairs += 1
                continue

            # ----------------------------------------------
            # Refine Corner Locations
            # ----------------------------------------------

            left_corners = cv2.cornerSubPix(

                left_gray,

                left_corners,

                (11,11),

                (-1,-1),

                termination_criteria

            )

            right_corners = cv2.cornerSubPix(

                right_gray,

                right_corners,

                (11,11),

                (-1,-1),

                termination_criteria

            )

            # ----------------------------------------------
            # Store Calibration Points
            # ----------------------------------------------

            self.object_points.append(
                self.object_point_template
            )

            self.left_image_points.append(
                left_corners
            )

            self.right_image_points.append(
                right_corners
            )

            successful_pairs += 1

            print(
                f"[Pair {pair_index:02d}] "
                "Corners detected successfully."
            )

        # --------------------------------------------------
        # Detection Summary
        # --------------------------------------------------

        print("\n==========================================")
        print("Corner Detection Summary")
        print("==========================================")
        print(f"Successful Pairs : {successful_pairs}")
        print(f"Failed Pairs     : {failed_pairs}")
        print("==========================================\n")
        
        
        # ======================================================
    # Calibrate Individual Cameras
    # ======================================================

    # ======================================================
    # Load Individual Camera Intrinsics
    # ======================================================

    def load_intrinsics(self):

        print("Loading camera intrinsic parameters...\n")

        # --------------------------------------------------
        # Verify Calibration Files Exist
        # --------------------------------------------------

        if not os.path.exists(LEFT_CAMERA_FILE):

            raise FileNotFoundError(
                f"LEFT camera calibration file not found:\n"
                f"{LEFT_CAMERA_FILE}"
            )

        if not os.path.exists(RIGHT_CAMERA_FILE):

            raise FileNotFoundError(
                f"RIGHT camera calibration file not found:\n"
                f"{RIGHT_CAMERA_FILE}"
            )

        # --------------------------------------------------
        # Load LEFT Camera Parameters
        # --------------------------------------------------

        left_data = np.load(LEFT_CAMERA_FILE)

        self.left_camera_matrix = left_data["camera_matrix"]

        self.left_distortion_coefficients = (
            left_data["distortion_coefficients"]
        )

        # --------------------------------------------------
        # Load RIGHT Camera Parameters
        # --------------------------------------------------

        right_data = np.load(RIGHT_CAMERA_FILE)

        self.right_camera_matrix = right_data["camera_matrix"]

        self.right_distortion_coefficients = (
            right_data["distortion_coefficients"]
        )

        print("LEFT camera intrinsics loaded.")

        print("RIGHT camera intrinsics loaded.")

        print("\nIntrinsic parameters loaded successfully.\n")
        # ======================================================
    # Perform Stereo Calibration
    # ======================================================

    def stereo_calibrate(self):

        print("Performing stereo calibration...\n")

        # --------------------------------------------------
        # Validate Calibration Data
        # --------------------------------------------------

        if len(self.object_points) == 0:

            raise RuntimeError(
                "Stereo calibration cannot be performed because no valid calibration points are available."
            )

        # --------------------------------------------------
        # Stereo Calibration Termination Criteria
        # --------------------------------------------------

        termination_criteria = (

            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,

            100,

            1e-5

        )

        # --------------------------------------------------
        # Stereo Calibration Flags
        # --------------------------------------------------

        calibration_flags = cv2.CALIB_FIX_INTRINSIC

        # --------------------------------------------------
        # Perform Stereo Calibration
        # --------------------------------------------------

        print("Estimating stereo camera relationship...")

        (

            stereo_reprojection_error,

            self.left_camera_matrix,
            self.left_distortion_coefficients,

            self.right_camera_matrix,
            self.right_distortion_coefficients,

            self.rotation_matrix,
            self.translation_vector,

            self.essential_matrix,
            self.fundamental_matrix

        ) = cv2.stereoCalibrate(
            

            self.object_points,

            self.left_image_points,
            self.right_image_points,

            self.left_camera_matrix,
            self.left_distortion_coefficients,

            self.right_camera_matrix,
            self.right_distortion_coefficients,

            self.image_size,

            criteria=termination_criteria,

            flags=calibration_flags


        )
        self.stereo_reprojection_error = stereo_reprojection_error

        # --------------------------------------------------
        # Display Calibration Results
        # --------------------------------------------------

        print("\nStereo Calibration Completed Successfully.\n")

        print(
            f"Stereo Reprojection Error : "
            f"{stereo_reprojection_error:.6f}"
        )

        print(
            f"Valid Stereo Pairs        : "
            f"{len(self.object_points)}"
        )

        print()
        # ======================================================
    # Save Calibration Results
    # ======================================================

    def save_results(self):

        print("Saving calibration parameters...\n")

        # --------------------------------------------------
        # Create Calibration Folder (If Required)
        # --------------------------------------------------

        os.makedirs(
            CALIBRATION_OUTPUT_FOLDER,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save Stereo Calibration Parameters
        # --------------------------------------------------

        np.savez(

                STEREO_CALIBRATION_FILE,

                # --------------------------------------------------
                # Left Camera Intrinsics
                # --------------------------------------------------

                camera_matrix_left=self.left_camera_matrix,
                distortion_coefficients_left=self.left_distortion_coefficients,

                # --------------------------------------------------
                # Right Camera Intrinsics
                # --------------------------------------------------

                camera_matrix_right=self.right_camera_matrix,
                distortion_coefficients_right=self.right_distortion_coefficients,

                # --------------------------------------------------
                # Stereo Extrinsics
                # --------------------------------------------------

                rotation_matrix=self.rotation_matrix,
                translation_vector=self.translation_vector,

                essential_matrix=self.essential_matrix,
                fundamental_matrix=self.fundamental_matrix,

                # --------------------------------------------------
                # Calibration Information
                # --------------------------------------------------

                image_size=self.image_size,
                chessboard_size=self.chessboard_size,
                square_size=self.square_size,

                stereo_rms_error=self.stereo_reprojection_error,
                valid_pairs=len(self.object_points),
            )

        print("Stereo calibration parameters saved.\n")

        # --------------------------------------------------
        # Display Saved File Locations
        # --------------------------------------------------

        print("==========================================")
        print("Calibration Files Saved Successfully")
        print("==========================================")
        print(f"Stereo Camera    : {STEREO_CALIBRATION_FILE}")
        print("==========================================\n")
        
        # ======================================================
    # Execute Complete Stereo Calibration Pipeline
    # ======================================================

    def run(self):

        print("\n====================================================")
        print("Starting Stereo Camera Calibration Pipeline")
        print("====================================================\n")

        # --------------------------------------------------
        # Load Stereo Calibration Images
        # --------------------------------------------------

        self.load_image_pairs()

        # --------------------------------------------------
        # Generate Chessboard Object Points
        # --------------------------------------------------

        self.create_object_points()

        # --------------------------------------------------
        # Detect Chessboard Corners
        # --------------------------------------------------

        self.detect_corners()

        # --------------------------------------------------
        # Calibrate Individual Cameras
        # --------------------------------------------------

        self.load_intrinsics()

        # --------------------------------------------------
        # Perform Stereo Calibration
        # --------------------------------------------------

        self.stereo_calibrate()

        # --------------------------------------------------
        # Save Calibration Results
        # --------------------------------------------------

        self.save_results()

        print("====================================================")
        print("Stereo Camera Calibration Completed Successfully")
        print("====================================================\n")


# ==========================================================
# Main Function
# ==========================================================

def main():

    stereo_calibration = StereoCalibration()

    stereo_calibration.run()


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
