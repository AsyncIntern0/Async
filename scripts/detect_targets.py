import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

from picamera2 import Picamera2

import time


# ==========================================================
# TARGET MARKER DATA CLASS
# ==========================================================

@dataclass
class TargetMarker:

    center_x: int
    center_y: int

    radius: float

    area: float
    perimeter: float

    circularity: float
    aspect_ratio: float

    score: float = 0.0

    contour = None


# ==========================================================
# TARGET DETECTOR
# ==========================================================

class TargetDetector:

    def __init__(self):

        self.camera = None

        self.frame_width = 640
        self.frame_height = 480

        self.min_area = 80

        self.min_radius = 8
        self.max_radius = 80

        self.min_circularity = 0.80

    # ------------------------------------------------------



    def open_camera(self):

        self.camera = Picamera2()

        config = self.camera.create_preview_configuration(

            main={

                "size": (

                    self.frame_width,

                    self.frame_height

                ),

                "format": "BGR888"

            }

        )

        self.camera.configure(config)

        self.camera.start()

        print("=" * 60)
        print("Target Marker Detection Started")
        print("=" * 60)
        # ------------------------------------------------------

    def detect(self, frame):

        gray, blur, thresh = self.preprocess(frame)

        contours, hierarchy = self.extract_contours(thresh)

        candidates = self.analyze_candidates(contours)

        return gray, thresh, candidates
    # ------------------------------------------------------

    def read_frame(self):

        frame = self.camera.capture_array()

        return frame
    # ------------------------------------------------------

    def preprocess(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (7, 7),
            1.5
        )

        thresh = cv2.adaptiveThreshold(

            blur,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY_INV,

            31,

            7

        )

        return gray, blur, thresh

    # ------------------------------------------------------

    def extract_contours(self, thresh):

        contours, hierarchy = cv2.findContours(

            thresh,

            cv2.RETR_TREE,

            cv2.CHAIN_APPROX_SIMPLE

        )

        return contours, hierarchy

    # ------------------------------------------------------

    def analyze_candidates(

        self,

        contours

    ) -> List[TargetMarker]:

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area:

                continue

            perimeter = cv2.arcLength(

                contour,

                True

            )

            if perimeter == 0:

                continue

            circularity = (

                4
                * np.pi
                * area
                /
                (perimeter * perimeter)

            )

            if circularity < self.min_circularity:

                continue

            (x, y), radius = cv2.minEnclosingCircle(

                contour

            )

            if radius < self.min_radius:

                continue

            if radius > self.max_radius:

                continue

            x_box, y_box, w_box, h_box = cv2.boundingRect(

                contour

            )

            aspect_ratio = float(

                w_box

            ) / float(

                h_box

            )

            marker = TargetMarker(

                center_x=int(x),

                center_y=int(y),

                radius=radius,

                area=area,

                perimeter=perimeter,

                circularity=circularity,

                aspect_ratio=aspect_ratio

            )

            marker.contour = contour

            candidates.append(

                marker

            )

        return candidates

    # ------------------------------------------------------

    def release(self):

        if self.camera is not None:

            self.camera.stop()

        cv2.destroyAllWindows()


# ==========================================================
# MAIN
# ==========================================================

def main():

    detector = TargetDetector()

    detector.open_camera()

    try:

        while True:

            frame = detector.read_frame()

            if frame is None:

                break
            

            gray, blur, thresh = detector.preprocess(

                frame

            )

            contours, hierarchy = detector.extract_contours(

                thresh

            )

            candidates = detector.analyze_candidates(

                contours

            )
            

            # =====================================================
            # Draw Candidate Markers
            # =====================================================

            output = frame.copy()

            marker_id = 1

            for marker in candidates:

                # ------------------------------------------
                # Draw enclosing circle
                # ------------------------------------------

                cv2.circle(

                    output,

                    (marker.center_x, marker.center_y),

                    int(marker.radius),

                    (0, 255, 0),

                    2

                )

                # ------------------------------------------
                # Draw center
                # ------------------------------------------

                cv2.circle(

                    output,

                    (marker.center_x, marker.center_y),

                    3,

                    (0, 0, 255),

                    -1

                )

                # ------------------------------------------
                # Draw contour
                # ------------------------------------------

                cv2.drawContours(

                    output,

                    [marker.contour],

                    -1,

                    (255, 255, 0),

                    1

                )

                # ------------------------------------------
                # Candidate Information
                # ------------------------------------------

                cv2.putText(

                    output,

                    f"T{marker_id}",

                    (

                        marker.center_x + 10,

                        marker.center_y

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 0, 0),

                    2

                )

                cv2.putText(

                    output,

                    f"R:{marker.radius:.1f}",

                    (

                        marker.center_x + 10,

                        marker.center_y + 18

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (0, 255, 255),

                    1

                )

                cv2.putText(

                    output,

                    f"C:{marker.circularity:.2f}",

                    (

                        marker.center_x + 10,

                        marker.center_y + 34

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (0, 255, 255),

                    1

                )

                cv2.putText(

                    output,

                    f"A:{marker.area:.0f}",

                    (

                        marker.center_x + 10,

                        marker.center_y + 50

                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (0, 255, 255),

                    1

                )

                marker_id += 1

            # =====================================================
            # Status Information
            # =====================================================

            cv2.putText(

                output,

                f"Candidates : {len(candidates)}",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2

            )

            # =====================================================
            # Windows
            # =====================================================

            cv2.imshow(

                "Target Candidates",

                output

            )

            cv2.imshow(

                "Threshold",

                thresh

            )



            key = cv2.waitKey(1)

            if key == ord("q"):

                break

    finally:

        detector.release()
if __name__ == "__main__":

    main()
