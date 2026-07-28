import cv2
import numpy as np

from dataclasses import dataclass
from typing import List

from detect_targets import TargetMarker


# ==========================================================
# VERIFIED TARGET
# ==========================================================

@dataclass
class VerifiedTarget:

    marker: TargetMarker

    geometry_score: float = 0.0

    intensity_score: float = 0.0

    verification_score: float = 0.0

    verified: bool = False


# ==========================================================
# VERIFICATION CONFIG
# ==========================================================

class VerificationConfig:

    def __init__(self):

        # Geometry

        self.min_circularity = 0.85

        self.min_aspect = 0.90
        self.max_aspect = 1.10

        # Intensity

        self.dark_threshold = 90

        self.white_threshold = 170

        # Score

        self.pass_score = 65

        # Sampling

        self.center_ratio = 0.20

        self.white_ring_ratio = 0.55

        self.outer_ring_ratio = 0.70

        self.sample_radius = 2


# ==========================================================
# TARGET VERIFIER
# ==========================================================

class TargetVerifier:

    def __init__(self):

        self.cfg = VerificationConfig()

    # ------------------------------------------------------

    def sample_circle_intensity(

        self,

        gray,

        center_x,

        center_y,

        radius

    ):

        mask = np.zeros_like(gray)

        cv2.circle(

            mask,

            (int(center_x), int(center_y)),

            int(radius),

            255,

            -1

        )

        pixels = gray[mask == 255]

        if len(pixels) == 0:

            return 255

        return float(np.mean(pixels))

    # ------------------------------------------------------

    def sample_direction(

        self,

        gray,

        marker,

        angle_deg,

        distance_ratio

    ):

        angle = np.deg2rad(angle_deg)

        distance = marker.radius * distance_ratio

        x = marker.center_x + distance * np.cos(angle)

        y = marker.center_y + distance * np.sin(angle)

        return self.sample_circle_intensity(

            gray,

            x,

            y,

            self.cfg.sample_radius

        )

    # ------------------------------------------------------

    def sample_8_directions(

        self,

        gray,

        marker,

        distance_ratio

    ):

        values = []

        for angle in [

            0,

            45,

            90,

            135,

            180,

            225,

            270,

            315

        ]:

            value = self.sample_direction(

                gray,

                marker,

                angle,

                distance_ratio

            )

            values.append(value)

        return np.mean(values)

    # ------------------------------------------------------

    def geometry_score(

        self,

        marker

    ):

        score = 0

        if marker.circularity >= self.cfg.min_circularity:

            score += 50

        if (

            self.cfg.min_aspect

            <=

            marker.aspect_ratio

            <=

            self.cfg.max_aspect

        ):

            score += 50

        return score

    # ------------------------------------------------------

    def intensity_profile(

        self,

        gray,

        marker

    ):

        center = self.sample_circle_intensity(

            gray,

            marker.center_x,

            marker.center_y,

            marker.radius * self.cfg.center_ratio

        )

        white = self.sample_8_directions(

            gray,

            marker,

            self.cfg.white_ring_ratio

        )

        outer = self.sample_8_directions(

            gray,

            marker,

            self.cfg.outer_ring_ratio

        )

        return center, white, outer

             # ------------------------------------------------------

    # ------------------------------------------------------

    def intensity_score(self, center, white, outer):
        score = 0

        # Calculate relative contrast percentages rather than absolute differences
        # This makes it resilient to variable lighting/exposure conditions.
        contrast_center = white - center
        contrast_outer = white - outer
        center_outer_diff = abs(center - outer)

        # 1. Directional checks (White ring must be brighter than surroundings)
        if white > center:
            score += 25
        if white > outer:
            score += 25

        # 2. Relaxed relative contrast thresholds
        # Dynamic adjustments instead of absolute drop-offs
        if contrast_center > 20:
            score += 20
        elif contrast_center > 10:
            score += 10

        if contrast_outer > 15:
            score += 20
        elif contrast_outer > 5:
            score += 10

        # 3. Ambient similarity check
        if center_outer_diff < 120:  # Relaxed from 90 to allow for gradient shadows
            score += 10

        return score

    def verify_marker(self, gray, marker):
        verified = VerifiedTarget(marker)

        geometry = self.geometry_score(marker)
        center, white, outer = self.intensity_profile(gray, marker)
        intensity = self.intensity_score(center, white, outer)

        # Balanced weight adjustments if needed (e.g., 40% geometry / 60% intensity)
        final_score = geometry * 0.40 + intensity * 0.60

        print(
            f"C={center:.1f} W={white:.1f} O={outer:.1f} "
            f"G={geometry} I={intensity} F={final_score:.1f}"
        )

        verified.geometry_score = geometry
        verified.intensity_score = intensity
        verified.verification_score = final_score

        # FIX: Dynamically use the configuration pass score instead of a hardcoded 75
        verified.verified = final_score >= self.cfg.pass_score

        return verified

    # ------------------------------------------------------

    # ------------------------------------------------------

    def verify_all(self, gray, candidates):

        results = []

        # ------------------------------------------
        # Verify every detected candidate
        # ------------------------------------------

        for marker in candidates:

            result = self.verify_marker(
                gray,
                marker
            )

            results.append(result)

        # ------------------------------------------
        # Keep ONLY genuinely verified markers
        # ------------------------------------------

        passed_results = [

            result

            for result in results

            if result.verified

        ]

        # ------------------------------------------
        # Highest verification score first
        # ------------------------------------------

        passed_results.sort(

            key=lambda r: r.verification_score,

            reverse=True

        )

        # ------------------------------------------
        # Maximum 7 markers
        # ------------------------------------------

        verified_targets = passed_results[:7]

        return verified_targets
# ==========================================================
# DEBUG DRAWING
# ==========================================================

def draw_verified_targets(

    frame,

    verified_targets

):

    output = frame.copy()

    for idx, target in enumerate(verified_targets, start=1):

        marker = target.marker

        cv2.circle(

            output,

            (marker.center_x, marker.center_y),

            int(marker.radius),

            (0,255,0),

            2

        )

        cv2.circle(

            output,

            (marker.center_x, marker.center_y),

            3,

            (0,0,255),

            -1

        )

        cv2.putText(

            output,

            f"V{idx}",

            (

                marker.center_x+10,

                marker.center_y

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255,0,0),

            2

        )

        cv2.putText(

            output,

            f"{target.verification_score:.1f}",

            (

                marker.center_x+10,

                marker.center_y+18

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0,255,255),

            1

        )

    return output
