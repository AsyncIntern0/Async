#!/usr/bin/env python3
"""
solve_intrinsics.py

Computes each camera's intrinsics INDEPENDENTLY from the checkerboard
images captured by capture_calibration_pairs.py. Even though the images
were captured as synchronized pairs, calibration itself treats each
camera's image set separately -- no stereo/extrinsic step here.

Usage:
    python3 solve_intrinsics.py
"""

from pathlib import Path
import time

import cv2
import numpy as np

from camera.camera_config import (
    CHESSBOARD_SIZE,
    SQUARE_SIZE,
    LEFT_IMAGE_FOLDER,
    RIGHT_IMAGE_FOLDER,
    LEFT_CAMERA_FILE,
    RIGHT_CAMERA_FILE,
    CALIBRATION_OUTPUT_FOLDER,
)


def calibrate_one_camera(image_folder, label):
    images = sorted(Path(image_folder).glob("*.jpg"))
    print(f"\n--- {label} camera: {len(images)} images in {image_folder} ---")
    if len(images) < 10:
        print(f"Only {len(images)} images -- recommend at least 15-20 for a stable result.")

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []
    img_shape = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    for i, img_path in enumerate(images, start=1):
        t0 = time.time()
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=cb_flags)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            used += 1

        elapsed_ms = (time.time() - t0) * 1000
        tag = "OK" if found else "no board"
        slow = f"  (SLOW: {elapsed_ms:.0f}ms)" if elapsed_ms > 300 else ""
        print(f"  [{i}/{len(images)}] {img_path.name}: {tag}{slow}")

    print(f"Board detected in {used}/{len(images)} images.")
    if used < 8:
        print(f"Too few valid images for {label} -- capture more before trusting this result.")
        return None

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print(f"{label} RMS reprojection error: {rms:.4f} px "
          f"(<0.5px good, ~0.5-1px workable, >1px recapture)")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs}")

    return camera_matrix, dist_coeffs, rms, used


def main():
    Path(CALIBRATION_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    left_result = calibrate_one_camera(LEFT_IMAGE_FOLDER, "LEFT")
    right_result = calibrate_one_camera(RIGHT_IMAGE_FOLDER, "RIGHT")

    if left_result:
        camera_matrix, dist_coeffs, rms, used = left_result
        np.savez(
            LEFT_CAMERA_FILE,
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            rms_reprojection_error=rms,
            num_images_used=used,
        )
        print(f"\nSaved LEFT intrinsics to {LEFT_CAMERA_FILE}")

    if right_result:
        camera_matrix, dist_coeffs, rms, used = right_result
        np.savez(
            RIGHT_CAMERA_FILE,
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            rms_reprojection_error=rms,
            num_images_used=used,
        )
        print(f"Saved RIGHT intrinsics to {RIGHT_CAMERA_FILE}")

    print("\nDone. These files use the same 'camera_matrix' / 'distortion_coefficients' "
          "keys your existing pipeline already expects -- direct drop-in replacement "
          "for the datasheet-approximated ones.")


if __name__ == "__main__":
    main()
