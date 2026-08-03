"""
===============================================================================
Stereo Rectification Verification
===============================================================================

Author  : Menaka R M
Project : Prototype Joint Detection using Stereo Vision

Description
-----------
This utility verifies that the stereo rectification maps are working
correctly by displaying the original and rectified stereo images.

Horizontal guide lines are drawn to visually confirm that corresponding
points lie on the same image row after rectification.

Press 'q' to exit.

===============================================================================
"""

# =============================================================================
# Import Required Libraries
# =============================================================================

import cv2
import numpy as np

# =============================================================================
# Import Project Modules
# =============================================================================

from camera.stereo_camera import StereoCamera

from camera.camera_config import RECTIFICATION_FILE

# =============================================================================
# Stereo Rectification Verification Class
# =============================================================================


class VerifyRectification:

    """
    Verifies stereo rectification using live stereo cameras.
    """

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(self):

        print("\nInitializing Rectification Verification...\n")

        self.stereo_camera = StereoCamera()

        self.left_map_x = None
        self.left_map_y = None

        self.right_map_x = None
        self.right_map_y = None

    # =========================================================================
    # Load Rectification Maps
    # =========================================================================

    def load_rectification_maps(self):

        print("Loading rectification maps...")

        maps = np.load(RECTIFICATION_FILE)

        self.left_map_x = maps["left_map_x"]
        self.left_map_y = maps["left_map_y"]

        self.right_map_x = maps["right_map_x"]
        self.right_map_y = maps["right_map_y"]

        print("Rectification maps loaded successfully.\n")

    # =========================================================================
    # Draw Horizontal Guide Lines
    # =========================================================================

    def draw_horizontal_lines(self, image):

        output = image.copy()

        line_spacing = 40

        for y in range(0, output.shape[0], line_spacing):

            cv2.line(
                output,
                (0, y),
                (output.shape[1], y),
                (0, 255, 0),
                1,
            )

        return output

    # =========================================================================
    # Start Verification
    # =========================================================================

    def verify(self):

        self.load_rectification_maps()

        self.stereo_camera.start()

        print("Press 'q' to exit.\n")

        while True:

            left_frame, right_frame = self.stereo_camera.capture_frames()

            # ---------------------------------------------------------
            # Rectify Frames
            # ---------------------------------------------------------

            rectified_left = cv2.remap(
                left_frame,
                self.left_map_x,
                self.left_map_y,
                cv2.INTER_LINEAR,
            )

            rectified_right = cv2.remap(
                right_frame,
                self.right_map_x,
                self.right_map_y,
                cv2.INTER_LINEAR,
            )

            # ---------------------------------------------------------
            # Draw Guide Lines
            # ---------------------------------------------------------

            original_left = self.draw_horizontal_lines(left_frame)
            original_right = self.draw_horizontal_lines(right_frame)

            rectified_left = self.draw_horizontal_lines(rectified_left)
            rectified_right = self.draw_horizontal_lines(rectified_right)

            # ---------------------------------------------------------
            # Display Images
            # ---------------------------------------------------------

            cv2.imshow("Original Left", original_left)
            cv2.imshow("Original Right", original_right)

            cv2.imshow("Rectified Left", rectified_left)
            cv2.imshow("Rectified Right", rectified_right)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        self.stereo_camera.stop()

        cv2.destroyAllWindows()

        print("\nRectification Verification Completed.")

# =============================================================================
# Main Function
# =============================================================================


def main():

    verifier = VerifyRectification()

    verifier.verify()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()