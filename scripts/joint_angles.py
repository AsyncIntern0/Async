"""
joint_angles.py

This module calculates 2D segment orientation angles and
relative joint angles from tracked joint-marker coordinates.

Marker Layout
-------------

                       Pelvis
                         ●

            Left Hip ●-------● Right Hip
                     |       |
                     |       |
           Left Knee ●       ● Right Knee
                     |       |
                     |       |
          Left Ankle ●       ● Right Ankle


The camera/tracker provides the (x, y) coordinate of each marker.

From these marker coordinates, we create virtual body segments:

    Pelvis Segment  : Left Hip  -> Right Hip

    Left Thigh      : Left Hip  -> Left Knee
    Left Shank      : Left Knee -> Left Ankle

    Right Thigh     : Right Hip  -> Right Knee
    Right Shank     : Right Knee -> Right Ankle


Step 1:
Calculate the absolute orientation of every segment using atan2().

Step 2:
Calculate relative joint angles by comparing adjacent
segment orientations.

Example:

    Left Knee Angle
        =
    Left Shank Orientation
        -
    Left Thigh Orientation


Important
---------

The final angle convention may later need calibration or sign
adjustment to match the STM rotary encoder convention.

This module currently calculates:

Absolute Segment Orientations:
    - Pelvis
    - Left Thigh
    - Left Shank
    - Right Thigh
    - Right Shank

Relative Joint Angles:
    - Left Hip
    - Right Hip
    - Left Knee
    - Right Knee

The ankle joint angle cannot yet be calculated because a foot
segment orientation is not available from the current markers.
"""

import math


# ==========================================================
# JOINT ANGLE CALCULATOR
# ==========================================================

class JointAngleCalculator:

    """
    Calculates:

    1. Absolute segment orientations.
    2. Relative joint angles.

    Input:
        List of TrackedJoint objects.

    Each TrackedJoint must contain:

        joint.name
        joint.x
        joint.y
    """


    # ======================================================
    # 1. GET JOINT COORDINATE
    # ======================================================

    def get_joint(self, tracked_joints, name):

        """
        Search for a joint by its name and return its
        (x, y) coordinate.

        Example:

            get_joint(tracked_joints, "Left Knee")

        may return:

            (297, 275)

        Parameters
        ----------

        tracked_joints:
            List containing the tracked joints.

        name:
            Name of the joint we want to find.

        Returns
        -------

        (x, y):
            Coordinate of the joint.

        None:
            If the joint is not available.
        """

        for joint in tracked_joints:

            if joint.name == name:

                return (
                    joint.x,
                    joint.y
                )

        return None


    # ======================================================
    # 2. CALCULATE ABSOLUTE SEGMENT ORIENTATION
    # ======================================================

    def segment_angle(self, point1, point2):

        """
        Calculate the absolute orientation of a segment.

        A segment is formed by connecting two joint markers.

        Example:

            Left Hip
                ●
                 \
                  \
                   ● Left Knee

        The line:

            Left Hip -> Left Knee

        represents the Left Thigh segment.


        point1 = (x1, y1)
        point2 = (x2, y2)


        First calculate:

            dx = x2 - x1
            dy = y2 - y1


        Then calculate:

            angle = atan2(dy, dx)


        atan2() is preferred over atan(dy/dx) because:

        1. It handles vertical lines.
        2. It avoids division-by-zero.
        3. It correctly identifies the direction/quadrant.


        IMPORTANT:

        OpenCV image coordinates are:

                -Y
                 ↑
                 |
        -X  <----+----> +X
                 |
                 ↓
                +Y


        Standard Cartesian coordinates are:

                +Y
                 ↑
                 |
        -X  <----+----> +X
                 |
                 ↓
                -Y


        Therefore, we invert the image Y difference.

        Returns
        -------

        Angle in degrees.

        The angle represents the absolute orientation of
        the segment relative to the horizontal X-axis.
        """

        x1, y1 = point1

        x2, y2 = point2


        # --------------------------------------------------
        # Calculate horizontal displacement
        # --------------------------------------------------

        dx = x2 - x1


        # --------------------------------------------------
        # Calculate vertical displacement
        #
        # Negative sign converts OpenCV image coordinates
        # into Cartesian-style coordinates.
        # --------------------------------------------------

        dy = -(y2 - y1)


        # --------------------------------------------------
        # Calculate angle in radians
        # --------------------------------------------------

        angle_radians = math.atan2(
            dy,
            dx
        )


        # --------------------------------------------------
        # Convert radians to degrees
        # --------------------------------------------------

        angle_degrees = math.degrees(
            angle_radians
        )


        return angle_degrees


    # ======================================================
    # 3. NORMALIZE ANGLE
    # ======================================================

    def normalize_angle(self, angle):

        """
        Normalize an angle to the range:

            -180° to +180°


        Example:

            350° becomes -10°

            190° becomes -170°

            30° remains 30°


        This prevents angle wrap-around problems.

        For example:

            Segment 1 = 170°
            Segment 2 = -170°

        Direct subtraction would give:

            -170 - 170 = -340°

        But geometrically the difference is only:

            20°

        Normalization corrects this problem.
        """

        return (
            angle + 180
        ) % 360 - 180


    # ======================================================
    # 4. CALCULATE ABSOLUTE SEGMENT ORIENTATIONS
    # ======================================================

    def calculate_segment_angles(self, tracked_joints):

        """
        Calculate the absolute orientation of each
        virtual body segment.


        SEGMENTS USED
        -------------

        Pelvis:

            Left Hip -------- Right Hip


        Left Thigh:

            Left Hip
                |
                |
            Left Knee


        Left Shank:

            Left Knee
                |
                |
            Left Ankle


        Right Thigh:

            Right Hip
                |
                |
            Right Knee


        Right Shank:

            Right Knee
                |
                |
            Right Ankle


        Returns
        -------

        Dictionary containing absolute segment angles.
        """


        # --------------------------------------------------
        # Get all required joint coordinates
        # --------------------------------------------------

        left_hip = self.get_joint(
            tracked_joints,
            "Left Hip"
        )


        left_knee = self.get_joint(
            tracked_joints,
            "Left Knee"
        )


        left_ankle = self.get_joint(
            tracked_joints,
            "Left Ankle"
        )


        right_hip = self.get_joint(
            tracked_joints,
            "Right Hip"
        )


        right_knee = self.get_joint(
            tracked_joints,
            "Right Knee"
        )


        right_ankle = self.get_joint(
            tracked_joints,
            "Right Ankle"
        )


        # --------------------------------------------------
        # Check whether all required joints are available
        # --------------------------------------------------

        required_joints = [

            left_hip,

            left_knee,

            left_ankle,

            right_hip,

            right_knee,

            right_ankle

        ]


        if any(
            joint is None
            for joint in required_joints
        ):

            return {}


        # --------------------------------------------------
        # Calculate absolute segment orientations
        # --------------------------------------------------

        segment_angles = {


            # ----------------------------------------------
            # PELVIS SEGMENT
            #
            # Defined using:
            #
            # Left Hip -> Right Hip
            # ----------------------------------------------

            "Pelvis":

                self.segment_angle(

                    left_hip,

                    right_hip

                ),


            # ----------------------------------------------
            # LEFT THIGH
            #
            # Left Hip -> Left Knee
            # ----------------------------------------------

            "Left Thigh":

                self.segment_angle(

                    left_hip,

                    left_knee

                ),


            # ----------------------------------------------
            # LEFT SHANK
            #
            # Left Knee -> Left Ankle
            # ----------------------------------------------

            "Left Shank":

                self.segment_angle(

                    left_knee,

                    left_ankle

                ),


            # ----------------------------------------------
            # RIGHT THIGH
            #
            # Right Hip -> Right Knee
            # ----------------------------------------------

            "Right Thigh":

                self.segment_angle(

                    right_hip,

                    right_knee

                ),


            # ----------------------------------------------
            # RIGHT SHANK
            #
            # Right Knee -> Right Ankle
            # ----------------------------------------------

            "Right Shank":

                self.segment_angle(

                    right_knee,

                    right_ankle

                )

        }


        return segment_angles


    # ======================================================
    # 5. CALCULATE RELATIVE JOINT ANGLES
    # ======================================================

    def calculate_joint_angles(self, tracked_joints):

        """
        Calculate relative joint angles.

        A relative joint angle is calculated by comparing
        the orientation of two adjacent body segments.


        LEFT HIP
        ---------

        Compare:

            Pelvis Segment
                VS
            Left Thigh


        RIGHT HIP
        ----------

        Compare:

            Pelvis Segment
                VS
            Right Thigh


        LEFT KNEE
        ----------

        Compare:

            Left Thigh
                VS
            Left Shank


        RIGHT KNEE
        -----------

        Compare:

            Right Thigh
                VS
            Right Shank


        NOTE:

        The exact positive/negative sign convention may later
        be adjusted to match the STM rotary encoder convention.
        """


        # --------------------------------------------------
        # First calculate all absolute segment orientations
        # --------------------------------------------------

        segments = self.calculate_segment_angles(

            tracked_joints

        )


        if not segments:

            return {}


        # ==================================================
        # LEFT HIP RELATIVE ANGLE
        # ==================================================

        left_hip_angle = self.normalize_angle(

            segments["Left Thigh"]

            -

            segments["Pelvis"]

        )


        # ==================================================
        # RIGHT HIP RELATIVE ANGLE
        # ==================================================

        right_hip_angle = self.normalize_angle(

            segments["Right Thigh"]

            -

            segments["Pelvis"]

        )


        # ==================================================
        # LEFT KNEE RELATIVE ANGLE
        # ==================================================

        left_knee_angle = self.normalize_angle(

            segments["Left Shank"]

            -

            segments["Left Thigh"]

        )


        # ==================================================
        # RIGHT KNEE RELATIVE ANGLE
        # ==================================================

        right_knee_angle = self.normalize_angle(

            segments["Right Shank"]

            -

            segments["Right Thigh"]

        )


        # --------------------------------------------------
        # Store all calculated relative angles
        # --------------------------------------------------

        joint_angles = {

            "Left Hip":

                left_hip_angle,


            "Right Hip":

                right_hip_angle,


            "Left Knee":

                left_knee_angle,


            "Right Knee":

                right_knee_angle

        }


        return joint_angles


    # ======================================================
    # 6. PRINT ANGLE RESULTS
    # ======================================================

    def print_angles(self, tracked_joints):

        """
        Print both:

        1. Absolute Segment Orientations
        2. Relative Joint Angles
        """


        # --------------------------------------------------
        # Calculate segment orientations
        # --------------------------------------------------

        segments = self.calculate_segment_angles(

            tracked_joints

        )


        # --------------------------------------------------
        # Calculate relative joint angles
        # --------------------------------------------------

        joints = self.calculate_joint_angles(

            tracked_joints

        )


        # --------------------------------------------------
        # Check whether calculation was successful
        # --------------------------------------------------

        if not segments:

            print(

                "Unable to calculate angles. "
                "Required joints are missing."

            )

            return


        # ==================================================
        # PRINT ABSOLUTE SEGMENT ORIENTATIONS
        # ==================================================

        print(

            "\n"
            "========== Absolute Segment Orientations =========="

        )


        for name, angle in segments.items():

            print(

                f"{name:<20} "
                f"{angle:8.2f}°"

            )


        # ==================================================
        # PRINT RELATIVE JOINT ANGLES
        # ==================================================

        print(

            "\n"
            "============== Relative Joint Angles =============="

        )


        for name, angle in joints.items():

            print(

                f"{name:<20} "
                f"{angle:8.2f}°"

            )


        print(

            "====================================================\n"

        )
    def draw_angles(self, frame, tracked_joints, angles):

        import cv2

        # Go through every calculated joint angle
        for joint_name, angle in angles.items():

            # Find the corresponding tracked joint
            for joint in tracked_joints:

                if joint.name == joint_name:

                    # Display angle near that joint
                    cv2.putText(

                        frame,

                        f"{angle:.1f} deg",

                        (
                            joint.x + 10,
                            joint.y + 30
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.5,

                        (0, 255, 255),

                        2

                    )

                    break

        return frame
