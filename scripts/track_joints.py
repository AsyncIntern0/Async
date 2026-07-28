 
from dataclasses import dataclass
from typing import Optional, List
from copy import deepcopy
import cv2


# ==========================================================
# TRACKED JOINT
# ==========================================================

@dataclass
class TrackedJoint:

    name: str

    x: int
    y: int

    locked: bool = False

    visible: bool = True

    target = None

    missing_frames: int=0


# ==========================================================
# TRACKING RESULT
# ==========================================================

@dataclass
class TrackingResult:

    success: bool

    locked: bool

    joints: List[TrackedJoint]

    message: str


# ==========================================================
# SKELETON TRACKER
# ==========================================================

class SkeletonTracker:

    def __init__(self):

        self.locked = False

        self.tracked_joints = []

        self.skeleton = None

        self.max_tracking_distance=60


    # ------------------------------------------------------

    def lock(self, skeleton):

        """
        Lock the skeleton only once.
        """

        if self.locked:

            return TrackingResult(

                success=True,

                locked=True,

                joints=self.tracked_joints,

                message="Already Locked"

            )

        self.skeleton = deepcopy(skeleton)

        self.tracked_joints = []

        joints = [

            skeleton.pelvis,

            skeleton.left_hip,

            skeleton.left_knee,

            skeleton.left_ankle,

            skeleton.right_hip,

            skeleton.right_knee,

            skeleton.right_ankle

        ]

        for joint in joints:

            if joint is None:

                continue

            tracked = TrackedJoint(

                name=joint.name,

                x=joint.target.marker.center_x,

                y=joint.target.marker.center_y,

                locked=True,

                visible=True

            )

            tracked.target = joint.target

            self.tracked_joints.append(tracked)

        self.locked = True

        print("\n==============================")
        print(" Skeleton Successfully Locked ")
        print("==============================\n")

        return TrackingResult(

            success=True,

            locked=True,

            joints=self.tracked_joints,

            message="Skeleton Locked"

        )
        # ------------------------------------------------------
    # Euclidean Distance
    # ------------------------------------------------------

    def distance(self, x1, y1, x2, y2):

        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
    def get_joint(self, name):

        for joint in self.tracked_joints:

            if joint.name == name:
                return joint

        return None


    # ------------------------------------------------------
    # Find Nearest Verified Target
    # ------------------------------------------------------

    def find_nearest(self, joint, available_targets):

        nearest = None
        nearest_distance = float("inf")

        pelvis = self.get_joint("Pelvis")

        for target in available_targets:

            # -----------------------------------------
            # LEFT / RIGHT SIDE CONSTRAINT
            # -----------------------------------------

            if pelvis is not None:

                if joint.name.startswith("Left"):

                    if target.center_x >= pelvis.x:
                        continue

                elif joint.name.startswith("Right"):

                    if target.center_x <= pelvis.x:
                        continue

            # -----------------------------------------
            # DISTANCE
            # -----------------------------------------

            d = self.distance(
                joint.x,
                joint.y,
                target.center_x,
                target.center_y
            )

            if d < nearest_distance:

                nearest_distance = d
                nearest = target

        # -----------------------------------------
        # Reject unreasonable jumps
        # -----------------------------------------

        if nearest is None:

            return None

        if nearest_distance > self.max_tracking_distance:

            return None

        return nearest



    # ------------------------------------------------------
    # Update Locked Joints
    # ------------------------------------------------------

    def update(self, candidates):

        if not self.locked:

            return TrackingResult(

                success=False,

                locked=False,

                joints=[],

                message="Skeleton not locked."

            )

        available_targets = candidates.copy()

        for joint in self.tracked_joints:

            nearest = self.find_nearest(

                joint,

                available_targets

            )

            if nearest is None:

                #joint.visible = False
                #print(f"{joint.name} temporarily lost")

                joint.missing_frames +=1

                continue

            joint.x = nearest.center_x
            joint.y = nearest.center_y

            #joint.visible = True

            joint.target = nearest

            joint.missing_frames = 0

            # Remove so another joint cannot use it

            available_targets.remove(nearest)

        return TrackingResult(

            success=True,

            locked=True,

            joints=self.tracked_joints,

            message="Tracking Updated"

        )


    # ------------------------------------------------------
    # Draw Tracked Joints
    # ------------------------------------------------------

    def draw(self, frame):

        print("Drawing: ",len(self.tracked_joints),"joints")


        if not self.locked:

            return frame

        for joint in self.tracked_joints:
            #print(joint.name,joint.x,joint.y)

            cv2.circle(

                frame,

                (joint.x, joint.y),

                6,

                (255, 255, 0),

                -1

            )

            cv2.putText(

                frame,

                joint.name,

                (joint.x + 10, joint.y - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 0),

                2

            )

            cv2.putText(

                frame,

                f"({joint.x},{joint.y})",

                (joint.x+10 , joint.y+12),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (0,255,255),

                1
                )

        return frame
