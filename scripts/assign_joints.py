from dataclasses import dataclass, field
from typing import List, Optional

from verify_targets import VerifiedTarget


# ==========================================================
# JOINT
# ==========================================================

@dataclass
class Joint:
    name: str
    target: VerifiedTarget
    parent: Optional["Joint"] = None
    children: List["Joint"] = field(default_factory=list)
    locked: bool = False
    tracked: bool = False


# ==========================================================
# SKELETON
# ==========================================================

@dataclass
class Skeleton:
    pelvis: Optional[Joint] = None
    left_hip: Optional[Joint] = None
    left_knee: Optional[Joint] = None
    left_ankle: Optional[Joint] = None
    right_hip: Optional[Joint] = None
    right_knee: Optional[Joint] = None
    right_ankle: Optional[Joint] = None
    valid: bool = False


# ==========================================================
# ASSIGNMENT RESULT
# ==========================================================

@dataclass
class AssignmentResult:
    skeleton: Skeleton
    success: bool
    confidence: float
    message: str


# ==========================================================
# JOINT ASSIGNER
# ==========================================================

class JointAssigner:

    def __init__(self):
        pass

    # ------------------------------------------------------

    def create_joint(self, name, target):
        return Joint(
            name=name,
            target=target,
            locked=True,
            tracked=False
        )

    # ------------------------------------------------------

    def connect(self, parent, child):
        child.parent = parent
        parent.children.append(child)

    # ------------------------------------------------------

    def assign(self, verified_targets):
        skeleton = Skeleton()

        # --------------------------------------------
        # Safety Checks
        # --------------------------------------------
        if len(verified_targets) != 7:
            return AssignmentResult(
                skeleton=skeleton,
                success=False,
                confidence=0,
                message=f"Expected 7 verified targets, got {len(verified_targets)}"
            )

        # --------------------------------------------
        # Execution Logic
        # --------------------------------------------
        pelvis = self.find_pelvis(verified_targets)
        left, right = self.split_left_right(verified_targets, pelvis)

        if len(left) != 3 or len(right) != 3:
            return AssignmentResult(
                skeleton=skeleton,
                success=False,
                confidence=0,
                message="Unable to split into left/right legs."
            )

        left = self.sort_leg(left)
        right = self.sort_leg(right)

        skeleton = self.build_skeleton(pelvis, left, right)

        return AssignmentResult(
            skeleton=skeleton,
            success=True,
            confidence=100,
            message="Skeleton assigned successfully."
        )

    # ------------------------------------------------------

    def find_pelvis(self, verified_targets):
        return min(
            verified_targets,
            key=lambda t: t.marker.center_y
        )

    # ------------------------------------------------------

    def split_left_right(self, verified_targets, pelvis):
        left = []
        right = []
        pelvis_x = pelvis.marker.center_x

        for target in verified_targets:
            if target == pelvis:
                continue

            if target.marker.center_x < pelvis_x:
                left.append(target)
            else:
                right.append(target)

        return left, right

    # ------------------------------------------------------

    def sort_leg(self, targets, tolerance=10):
        return sorted(
            targets,
            key=lambda t: (
                round(t.marker.center_y / tolerance),
                t.marker.center_y
            )
        )

    # ------------------------------------------------------

    def build_skeleton(self, pelvis, left, right):
        skeleton = Skeleton()

        skeleton.pelvis = self.create_joint("Pelvis", pelvis)

        skeleton.left_hip = self.create_joint("Left Hip", left[0])
        skeleton.left_knee = self.create_joint("Left Knee", left[1])
        skeleton.left_ankle = self.create_joint("Left Ankle", left[2])

        skeleton.right_hip = self.create_joint("Right Hip", right[0])
        skeleton.right_knee = self.create_joint("Right Knee", right[1])
        skeleton.right_ankle = self.create_joint("Right Ankle", right[2])

        # -----------------------------
        # Connect Skeleton
        # -----------------------------
        self.connect(skeleton.pelvis, skeleton.left_hip)
        self.connect(skeleton.left_hip, skeleton.left_knee)
        self.connect(skeleton.left_knee, skeleton.left_ankle)

        self.connect(skeleton.pelvis, skeleton.right_hip)
        self.connect(skeleton.right_hip, skeleton.right_knee)
        self.connect(skeleton.right_knee, skeleton.right_ankle)

        skeleton.valid = True
        return skeleton
