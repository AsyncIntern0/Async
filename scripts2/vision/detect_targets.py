import cv2
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Optional


# ==========================================================
# TARGET MARKER DATA CLASS
# ==========================================================
# ==========================================================
# DETECTION CONFIGURATION
# ==========================================================

@dataclass
class DetectionConfig:

    # Geometry

    min_area: float = 80
    max_area: float = 100000

    min_radius: float = 4
    max_radius: float = 80

    min_circularity: float = 0.80

    min_aspect_ratio: float = 0.80
    max_aspect_ratio: float = 1.20

    # Duplicate removal

    duplicate_center_distance: float = 4.0

    duplicate_radius_ratio: float = 0.60

    duplicate_area_ratio: float = 0.50

    # Preprocessing

    blur_kernel: Tuple[int, int] = (7, 7)

    adaptive_block_size: int = 31

    adaptive_C: int = 7
# ==========================================================
# TARGET MARKER
# ==========================================================

@dataclass
class TargetMarker:

    center_x: float
    center_y: float

    radius: float

    area: float
    perimeter: float

    circularity: float

    aspect_ratio: float

    contour: np.ndarray

    bounding_box: Tuple[int, int, int, int]

    parent_index: int

    child_index: int

    confidence: float = 0.0

# ==========================================================
# TARGET DETECTOR
# ==========================================================

class TargetDetector:

    def __init__(self):

        
        self.cfg = DetectionConfig()
    # ------------------------------------------------------

    def detect(self, frame):

        gray, blur, thresh = self.preprocess(frame)

        contours, hierarchy = self.extract_contours(thresh)

        candidates = self.analyze_candidates(
            contours,
            hierarchy
        )

        candidates = self._remove_duplicate_markers(
            candidates
        )

        return gray, thresh, candidates
    # ------------------------------------------------------

    # ------------------------------------------------------

    # ==========================================================
    # PREPROCESS IMAGE
    # ==========================================================

    def preprocess(self, frame):

        # Convert to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Reduce image noise
        blur = cv2.GaussianBlur(
            gray,
            self.cfg.blur_kernel,
            1.5
        )

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.cfg.adaptive_block_size,
            self.cfg.adaptive_C
        )

        return gray, blur, thresh
    # ------------------------------------------------------
    # ==========================================================
    # EXTRACT CONTOURS
    # ==========================================================

    def extract_contours(self, thresh):

        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Handle case where no contours are found
        if hierarchy is None:
            hierarchy = np.empty((0, 4), dtype=np.int32)

        return contours, hierarchy

    # ------------------------------------------------------

    # ==========================================================
    # EXTRACT CONTOUR FEATURES
    # ==========================================================

    def _extract_features(
        self,
        contour: np.ndarray
    ) -> Optional[dict]:

        # Contour Area
        area = cv2.contourArea(contour)

        # Contour Perimeter
        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            return None

        # Circularity
        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

        # Bounding Rectangle
        x_box, y_box, w_box, h_box = cv2.boundingRect(
            contour
        )

        # Aspect Ratio
        aspect_ratio = (
            float(w_box) / h_box
        )

        # Minimum Enclosing Circle
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(
            contour
        )

        # Image Moments
        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]

        else:

            center_x = circle_x
            center_y = circle_y

        return {

            "area": area,

            "perimeter": perimeter,

            "circularity": circularity,

            "radius": radius,

            "center_x": center_x,

            "center_y": center_y,

            "bounding_box": (
                x_box,
                y_box,
                w_box,
                h_box
            ),

            "aspect_ratio": aspect_ratio

        }

    # ------------------------------------------------------
    # ==========================================================
    # VALIDATE GEOMETRY
    # ==========================================================

    def _validate_geometry(
        self,
        features: dict
    ) -> bool:

        # ----------------------------------------------
        # Area Check
        # ----------------------------------------------

        if (
            features["area"] < self.cfg.min_area
            or
            features["area"] > self.cfg.max_area
        ):
            return False

        # ----------------------------------------------
        # Radius Check
        # ----------------------------------------------

        if (
            features["radius"] < self.cfg.min_radius
            or
            features["radius"] > self.cfg.max_radius
        ):
            return False

        # ----------------------------------------------
        # Circularity Check
        # ----------------------------------------------

        if (
            features["circularity"] <
            self.cfg.min_circularity
        ):
            return False

        # ----------------------------------------------
        # Aspect Ratio Check
        # ----------------------------------------------

        if not (
            self.cfg.min_aspect_ratio
            <=
            features["aspect_ratio"]
            <=
            self.cfg.max_aspect_ratio
        ):
            return False

        return True
    # ==========================================================
    # CREATE TARGET MARKER
    # ==========================================================

    def _create_marker(
        self,
        features: dict,
        contour: np.ndarray,
        hierarchy: np.ndarray,
        contour_index: int
    ) -> TargetMarker:

        # ----------------------------------------------
        # Parent / Child Information
        # ----------------------------------------------

        if len(hierarchy) > 0:

            parent_index = hierarchy[0][contour_index][3]
            child_index = hierarchy[0][contour_index][2]

        else:

            parent_index = -1
            child_index = -1

        # ----------------------------------------------
        # Create Marker
        # ----------------------------------------------

        marker = TargetMarker(

            center_x=features["center_x"],

            center_y=features["center_y"],

            radius=features["radius"],

            area=features["area"],

            perimeter=features["perimeter"],

            circularity=features["circularity"],

            aspect_ratio=features["aspect_ratio"],

            contour=contour,

            bounding_box=features["bounding_box"],

            parent_index=parent_index,

            child_index=child_index,

            confidence=0.0

        )

        return marker
    # ==========================================================
    # COMPUTE DETECTION CONFIDENCE
    # ==========================================================

    def _compute_confidence(
        self,
        marker: TargetMarker
    ) -> float:

        # ----------------------------------------------
        # Circularity Score
        # ----------------------------------------------

        circularity_score = min(
            marker.circularity,
            1.0
        )

        # ----------------------------------------------
        # Aspect Ratio Score
        # ----------------------------------------------

        aspect_error = abs(
            1.0 - marker.aspect_ratio
        )

        aspect_score = max(
            0.0,
            1.0 - aspect_error
        )

        # ----------------------------------------------
        # Radius Score
        # ----------------------------------------------

        radius_range = (
            self.cfg.max_radius -
            self.cfg.min_radius
        )

        radius_mid = (
            self.cfg.min_radius +
            self.cfg.max_radius
        ) / 2

        radius_score = 1.0 - (
            abs(marker.radius - radius_mid)
            / (radius_range / 2)
        )

        radius_score = np.clip(
            radius_score,
            0.0,
            1.0
        )

        # ----------------------------------------------
        # Area Score
        # ----------------------------------------------

        area_score = min(
            marker.area /
            self.cfg.max_area,
            1.0
        )

        # ----------------------------------------------
        # Weighted Confidence
        # ----------------------------------------------

        confidence = (

            0.40 * circularity_score +

            0.25 * aspect_score +

            0.20 * radius_score +

            0.15 * area_score

        )

        return float(confidence)
    # ==========================================================
    # REMOVE DUPLICATE MARKERS
    # ==========================================================

    def _remove_duplicate_markers(
        self,
        candidates: List[TargetMarker]
    ) -> List[TargetMarker]:

        if len(candidates) <= 1:
            return candidates

        keep = [True] * len(candidates)

        # --------------------------------------------------
        # Compare every marker with every other marker
        # --------------------------------------------------

        for i in range(len(candidates)):

            if not keep[i]:
                continue

            marker1 = candidates[i]

            for j in range(i + 1, len(candidates)):

                if not keep[j]:
                    continue

                marker2 = candidates[j]

                # ------------------------------------------
                # Distance between centres
                # ------------------------------------------

                distance = np.hypot(

                    marker1.center_x - marker2.center_x,

                    marker1.center_y - marker2.center_y

                )

                # ------------------------------------------
                # Radius Ratio
                # ------------------------------------------

                radius_ratio = min(

                    marker1.radius,

                    marker2.radius

                ) / max(

                    marker1.radius,

                    marker2.radius

                )

                # ------------------------------------------
                # Area Ratio
                # ------------------------------------------

                area_ratio = min(

                    marker1.area,

                    marker2.area

                ) / max(

                    marker1.area,

                    marker2.area

                )

                # ------------------------------------------
                # Same Physical Marker?
                # ------------------------------------------

                if (

                    distance < self.cfg.duplicate_center_distance

                    and

                    radius_ratio > self.cfg.duplicate_radius_ratio

                    and

                    area_ratio > self.cfg.duplicate_area_ratio

                ):

                    # --------------------------------------
                    # Keep Higher Confidence Marker
                    # --------------------------------------

                    if marker1.confidence >= marker2.confidence:

                        keep[j] = False

                    else:

                        keep[i] = False
                        break

        # --------------------------------------------------
        # Return filtered markers
        # --------------------------------------------------

        filtered = [

            marker

            for marker, valid

            in zip(candidates, keep)

            if valid

        ]

        return filtered
    # ==========================================================
    # ANALYZE CANDIDATE MARKERS
    # ==========================================================

    def analyze_candidates(
        self,
        contours,
        hierarchy
    ) -> List[TargetMarker]:

        candidates = []

        for contour_index, contour in enumerate(contours):

            # ----------------------------------------------
            # Extract Features
            # ----------------------------------------------

            features = self._extract_features(contour)

            if features is None:
                continue

            # ----------------------------------------------
            # Geometry Validation
            # ----------------------------------------------

            if not self._validate_geometry(features):
                continue

            # ----------------------------------------------
            # Create Marker
            # ----------------------------------------------

            marker = self._create_marker(
                features,
                contour,
                hierarchy,
                contour_index
            )

            # ----------------------------------------------
            # Compute Confidence
            # ----------------------------------------------

            marker.confidence = self._compute_confidence(
                marker
            )

            candidates.append(marker)

        return candidates


