"""
===========================================================================
Camera Configuration
===========================================================================

Author  : Menaka R M
Project : Prototype Joint Detection (Stereo Vision)

Purpose
-------
Stores all configurable camera parameters used throughout
the stereo vision pipeline.

Changing camera settings only requires editing this file.

===========================================================================
"""

# ==========================================================
# Camera IDs
# ==========================================================

LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

# ==========================================================
# Image Resolution
# ==========================================================

FRAME_WIDTH = 640
FRAME_HEIGHT =480

# Common alternatives
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720

# ==========================================================
# Camera Format
# ==========================================================

PIXEL_FORMAT = "BGR888"

# ==========================================================
# Camera Timing
# ==========================================================

STARTUP_DELAY = 2.0

# ==========================================================
# Camera Controls
# ==========================================================

AUTO_EXPOSURE = True

AUTO_WHITE_BALANCE = True

# Future manual controls
EXPOSURE_TIME = None

ANALOG_GAIN = None

BRIGHTNESS = 0.0

CONTRAST = 1.0

SATURATION = 1.0

SHARPNESS = 1.0

# ==========================================================
# Calibration Settings
# ==========================================================

CHESSBOARD_ROWS = 6
CHESSBOARD_COLUMNS = 9

# millimeters
SQUARE_SIZE = 29.0


# ==========================================================
# Calibration Image Capture
# ==========================================================

NUMBER_OF_IMAGE_PAIRS = 30

INITIAL_DELAY = 10        # seconds before first capture


# ==========================================================
# Display Settings
# ==========================================================

LEFT_WINDOW_NAME = "Left Camera"

RIGHT_WINDOW_NAME = "Right Camera"

EXIT_KEY = ord("q")

# ==========================================================
# Folder root Settings
# ==========================================================

import os

# Project root (scripts2)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CALIBRATION_FOLDER = os.path.join(PROJECT_ROOT, "calibration_images")

LEFT_IMAGE_FOLDER = os.path.join(CALIBRATION_FOLDER, "left")

RIGHT_IMAGE_FOLDER = os.path.join(CALIBRATION_FOLDER, "right")

# ==========================================================
# Chessboard Verification Settings
# ==========================================================

CHESSBOARD_SIZE = (9, 6)      # Inner corners (Columns, Rows)

DISPLAY_DELAY = 800           # milliseconds

# ==========================================================
# Calibration Output Folder
# ==========================================================

CALIBRATION_OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "calibration"
)

# ==========================================================
# Calibration Output Files
# ==========================================================

LEFT_CAMERA_FILE = os.path.join(
    CALIBRATION_OUTPUT_FOLDER,
    "left_camera.npz"
)

RIGHT_CAMERA_FILE = os.path.join(
    CALIBRATION_OUTPUT_FOLDER,
    "right_camera.npz"
)

STEREO_CALIBRATION_FILE = os.path.join(
    CALIBRATION_OUTPUT_FOLDER,
    "stereo_calibration.npz"
)

RECTIFICATION_FILE = os.path.join(
    CALIBRATION_OUTPUT_FOLDER,
    "rectification_maps.npz"
)
