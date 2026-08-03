"""
==========================================================
Stereo Correspondence Module
==========================================================

Author  : asyncIntern0
Project : Prototype Joint Detection (Stereo Vision)

Purpose
-------
Matches verified target markers detected in the left and
right rectified stereo images.

The goal of this module is to determine which marker in
the left image corresponds to the same physical marker in
the right image.

Responsibilities
----------------
1. Compare verified markers from both cameras.
2. Apply stereo correspondence constraints.
3. Find the best matching marker pairs.
4. Return matched stereo pairs for triangulation.

This module DOES NOT perform
----------------------------
- Joint ID Assignment
- Stereo Triangulation
- Tracking (Kalman Filter)
- Joint Angle Estimation

Input
-----
Left Verified Targets
Right Verified Targets

Output
------
Matched Stereo Marker Pairs

Example
-------
Left Marker 3  <------>  Right Marker 5
Left Marker 1  <------>  Right Marker 2

==========================================================
"""

import math

from dataclasses import dataclass
from typing import List, Optional

from vision.verify_targets import VerifiedTarget

# ==========================================================
# STEREO MATCH
# ==========================================================

@dataclass
class StereoMatch:
    """
    Represents one matched stereo target.

    A StereoMatch contains the corresponding
    verified target detected in both the left
    and right stereo images.
    """

    left_target: VerifiedTarget

    right_target: VerifiedTarget

    disparity: float = 0.0

    vertical_difference: float = 0.0
# ==========================================================
# CORRESPONDENCE CONFIGURATION
# ==========================================================

class CorrespondenceConfig:

    def __init__(self):

        # --------------------------------------------------
        # Maximum Allowed Vertical Difference
        # --------------------------------------------------
        # After stereo rectification, corresponding markers
        # should lie on nearly the same horizontal scan line.
        #
        # | y_left - y_right | <= max_vertical_difference
        #
        # Unit : pixels
        # --------------------------------------------------

        self.max_vertical_difference = 60.0


        # --------------------------------------------------
        # Minimum Valid Disparity
        # --------------------------------------------------
        # Disparity is:
        #
        # disparity = x_left - x_right
        #
        # The corresponding point in the right image should
        # appear slightly to the left of the point in the
        # left image.
        #
        # Very small or negative disparities are rejected.
        # --------------------------------------------------

        self.min_disparity = 1.0


        # --------------------------------------------------
        # Maximum Valid Disparity
        # --------------------------------------------------
        # Prevents impossible matches caused by incorrect
        # detections.
        #
        # This value depends on the stereo baseline,
        # camera focal length and expected working distance.
        # --------------------------------------------------

        self.max_disparity = 300.0
    
# ==========================================================
# STEREO CORRESPONDENCE
# ==========================================================

class StereoCorrespondence:
    """
    Matches verified target markers detected in the
    left and right rectified stereo images.

    The matching process uses stereo geometry
    constraints to determine corresponding targets.
    """

    # ------------------------------------------------------

    def __init__(self):

        self.cfg = CorrespondenceConfig()
        
# ------------------------------------------------------

    def vertical_difference(

        self,

        left_target,

        right_target

    ):
        """
        Compute the vertical distance between
        two verified targets.

        Returns
        -------
        float
            Absolute vertical difference in pixels.
        """

        return abs(

            left_target.marker.center_y

            -

            right_target.marker.center_y

        )
        
    # ------------------------------------------------------

    def calculate_disparity(

        self,

        left_target,

        right_target

    ):
        """
        Compute the horizontal disparity between
        two verified targets.

        disparity = x_left - x_right

        Returns
        -------
        float
            Horizontal disparity in pixels.
        """

        return (

            left_target.marker.center_x

            -

            right_target.marker.center_x

        )
     # ------------------------------------------------------

    def is_valid_match(

        self,

        left_target,

        right_target

    ):
        """
        Determine whether two verified targets satisfy
        the stereo correspondence constraints.

        Parameters
        ----------
        left_target : VerifiedTarget

        right_target : VerifiedTarget

        Returns
        -------
        bool
            True  -> Valid stereo correspondence

            False -> Invalid stereo correspondence
        """

        # --------------------------------------------------
        # Compute Vertical Difference
        # --------------------------------------------------

        vertical_error = self.vertical_difference(

            left_target,

            right_target

        )

        # --------------------------------------------------
        # Compute Horizontal Disparity
        # --------------------------------------------------

        disparity = self.calculate_disparity(

            left_target,

            right_target

        )

        # --------------------------------------------------
        # Constraint 1
        # Vertical Alignment
        # --------------------------------------------------

        if vertical_error > self.cfg.max_vertical_difference:

            print(
                f"Rejected: Vertical = {vertical_error:.2f}"
            )

            return False

        # --------------------------------------------------
        # Constraint 2
        # Positive Disparity
        # --------------------------------------------------

        if disparity < self.cfg.min_disparity:

            print(
                f"Rejected: Negative disparity = {disparity:.2f}"
            )

            return False

        # --------------------------------------------------
        # Constraint 3
        # Maximum Disparity
        # --------------------------------------------------

        if disparity > self.cfg.max_disparity:

            print(
                f"Rejected: Large disparity = {disparity:.2f}"
            )

            return False

            

        # --------------------------------------------------
        # All Constraints Satisfied
        # --------------------------------------------------

        return True
 
        # ------------------------------------------------------

    def find_best_match(

        self,

        left_target,

        right_targets

    ):
        """
        Find the best corresponding target in the
        right stereo image for one left target.

        Parameters
        ----------
        left_target : VerifiedTarget

        right_targets : List[VerifiedTarget]

        Returns
        -------
        StereoMatch | None
            Best matching stereo pair.
            Returns None if no valid match exists.
        """

        # --------------------------------------------------
        # Initialize Best Match
        # --------------------------------------------------

        best_match = None

        smallest_vertical_error = float("inf")

        # --------------------------------------------------
        # Compare Against Every Right Target
        # --------------------------------------------------

        for right_target in right_targets:

            # ----------------------------------------------
            # Check Stereo Constraints
            # ----------------------------------------------

            if not self.is_valid_match(

                left_target,

                right_target

            ):

                continue

            # ----------------------------------------------
            # Compute Matching Metrics
            # ----------------------------------------------

            vertical_error = self.vertical_difference(

                left_target,

                right_target

            )

            disparity = self.calculate_disparity(

                left_target,

                right_target

            )

            # ----------------------------------------------
            # Keep Best Candidate
            # ----------------------------------------------

            if vertical_error < smallest_vertical_error:

                smallest_vertical_error = vertical_error

                best_match = StereoMatch(

                    left_target=left_target,

                    right_target=right_target,

                    disparity=disparity,

                    vertical_difference=vertical_error

                )

        # --------------------------------------------------
        # Return Best Match
        # --------------------------------------------------

        return best_match
        # ------------------------------------------------------

    def find_matches(

        self,

        left_targets,

        right_targets

    ):
        """
        Find stereo correspondences between all verified
        targets detected in the left and right images.

        Parameters
        ----------
        left_targets : List[VerifiedTarget]

        right_targets : List[VerifiedTarget]

        Returns
        -------
        List[StereoMatch]
            List of matched stereo target pairs.
        """

        # --------------------------------------------------
        # Initialize Match List
        # --------------------------------------------------

        matches = []

        # --------------------------------------------------
        # Copy Right Targets
        # --------------------------------------------------
        # Prevent one right target from being matched
        # multiple times.
        # --------------------------------------------------

        available_right_targets = right_targets.copy()

        # --------------------------------------------------
        # Process Each Left Target
        # --------------------------------------------------

        for left_target in left_targets:

            best_match = self.find_best_match(

                left_target,

                available_right_targets

            )

            # ----------------------------------------------
            # Skip if no valid match found
            # ----------------------------------------------

            if best_match is None:

                continue

            # ----------------------------------------------
            # Store Stereo Match
            # ----------------------------------------------

            matches.append(

                best_match

            )

            # ----------------------------------------------
            # Remove Used Right Target
            # ----------------------------------------------

            available_right_targets.remove(

                best_match.right_target

            )

        # --------------------------------------------------
        # Return Stereo Correspondences
        # --------------------------------------------------
        return matches
