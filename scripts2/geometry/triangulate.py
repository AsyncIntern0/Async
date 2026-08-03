"""
triangulate.py

For a RECTIFIED stereo pair (frames already cv2.remap'd with the
calibration maps -- as your pipeline already does). After rectification,
the same physical point appears at the SAME row (y) in both images --
only x (disparity) differs. So matching is just "closest y", not
epipolar-line math.

Uses the Q matrix (disparity-to-depth) from stereo calibration.
"""

import numpy as np


# ==========================================================
# Load Q (and P1/P2 as fallback) from your rectification file.
# EDIT the key names below if yours differ -- print(list(data.keys()))
# once to confirm, then fix here.
# ==========================================================

def load_stereo_params(rectification_npz_path):
    data = np.load(rectification_npz_path)
    keys = list(data.keys())

    Q = data["Q"] if "Q" in keys else None
    P1 = data["P1"] if "P1" in keys else None
    P2 = data["P2"] if "P2" in keys else None

    if Q is None and (P1 is None or P2 is None):
        raise KeyError(
            f"Need 'Q' OR both 'P1' and 'P2' in {rectification_npz_path}. "
            f"Found keys: {keys}. Edit load_stereo_params() to match your actual key names."
        )
    return Q, P1, P2


# ==========================================================
# Match left/right markers by row proximity (rectified => same y)
# ==========================================================

def match_rectified_pairs(left_markers, right_markers, max_y_diff=5.0, min_disparity=1.0):
    if not left_markers or not right_markers:
        return []

    candidates = []
    for li, lm in enumerate(left_markers):
        for ri, rm in enumerate(right_markers):
            y_diff = abs(lm.center_y - rm.center_y)
            disparity = lm.center_x - rm.center_x  # left x > right x for real depth
            if y_diff <= max_y_diff and disparity >= min_disparity:
                candidates.append((y_diff, li, ri))

    candidates.sort(key=lambda c: c[0])

    used_l, used_r = set(), set()
    matches = []
    for y_diff, li, ri in candidates:
        if li in used_l or ri in used_r:
            continue
        used_l.add(li)
        used_r.add(ri)
        matches.append((left_markers[li], right_markers[ri]))

    return matches


# ==========================================================
# Triangulate matched pairs -> 3D points, using Q (preferred)
# or P1/P2 (fallback via cv2.triangulatePoints)
# ==========================================================

def triangulate_points(
    stereo_matches,
    Q=None,
    P1=None,
    P2=None
):
    """
    Compute 3D coordinates from matched stereo targets.

    Parameters
    ----------
    stereo_matches : List[StereoMatch]

    Returns
    -------
    List[List[float]]
        List of [X, Y, Z] coordinates.
    """

    if not stereo_matches:

        return []

    # ------------------------------------------------------
    # Q Matrix Triangulation
    # ------------------------------------------------------

    if Q is not None:

        points_3d = []

        for match in stereo_matches:

            left_marker = match.left_target.marker
            right_marker = match.right_target.marker

            x = left_marker.center_x
            y = left_marker.center_y

            disparity = (

                left_marker.center_x

                -

                right_marker.center_x

            )

            if disparity <= 0:

                continue

            vec = np.array(

                [

                    x,

                    y,

                    disparity,

                    1.0

                ]

            )

            world = Q @ vec

            world /= world[3]

            points_3d.append(

                world[:3].tolist()

            )

        return points_3d

    # Fallback: P1/P2 projection matrices
    # ------------------------------------------------------
    # Fallback: Projection Matrix Triangulation
    # ------------------------------------------------------

    if P1 is not None and P2 is not None:

        import cv2

        left_pts = np.array(
            [
                [
                    match.left_target.marker.center_x,
                    match.left_target.marker.center_y
                ]
                for match in stereo_matches
            ],
            dtype=np.float64
        ).T

        right_pts = np.array(
            [
                [
                    match.right_target.marker.center_x,
                    match.right_target.marker.center_y
                ]
                for match in stereo_matches
            ],
            dtype=np.float64
        ).T

        points_4d = cv2.triangulatePoints(

            P1,

            P2,

            left_pts,

            right_pts

        )

        points_3d = (

            points_4d[:3]

            /

            points_4d[3]

        ).T

        return points_3d.tolist()

    return []


# ==========================================================
# One-call convenience wrapper
# ==========================================================

# ==========================================================
# One-call Convenience Wrapper
# ==========================================================

def get_3d_coordinates(
    stereo_matches,
    Q=None,
    P1=None,
    P2=None
):
    """
    Compute 3D coordinates from matched stereo targets.

    Parameters
    ----------
    stereo_matches : List[StereoMatch]

    Q : ndarray, optional

    P1 : ndarray, optional

    P2 : ndarray, optional

    Returns
    -------
    points_3d : ndarray

    stereo_matches : List[StereoMatch]
    """

    points_3d = triangulate_points(

        stereo_matches,

        Q=Q,

        P1=P1,

        P2=P2

    )

    return points_3d, stereo_matches
