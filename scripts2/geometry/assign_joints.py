"""
==========================================================
Joint ID Assignment Module
==========================================================

Author  : asyncInten0
Project : Prototype Joint Detection (Stereo Vision)

Purpose
-------
Assign a unique joint ID (X1 - X7) to every matched
stereo target using the known geometric structure of
the lower-limb prototype.

Joint Order
-----------
X1 : Pelvis

X2 : Right Hip
X3 : Right Knee
X4 : Right Ankle

X5 : Left Hip
X6 : Left Knee
X7 : Left Ankle

Responsibilities
----------------
1. Identify the pelvis.
2. Separate left and right body chains.
3. Sort joints from proximal to distal.
4. Assign stable joint IDs.
5. Build the skeleton hierarchy.

This module DOES NOT perform
----------------------------
- Detection
- Verification
- Stereo Correspondence
- Triangulation
- Tracking

==========================================================
"""
from dataclasses import dataclass, field

from typing import List, Optional

from geometry.stereo_correspondence import StereoMatch

import numpy as np

# ==========================================================
# ASSIGNED JOINT
# ==========================================================

@dataclass
class AssignedJoint:

    joint_id: str

    stereo_match: StereoMatch

    parent: Optional["AssignedJoint"] = None

    children: List["AssignedJoint"] = field(default_factory=list)

    locked: bool = False

    tracked: bool = False
    
# ==========================================================
# SKELETON
# ==========================================================

@dataclass
class Skeleton:

    X1: Optional[AssignedJoint] = None

    X2: Optional[AssignedJoint] = None

    X3: Optional[AssignedJoint] = None

    X4: Optional[AssignedJoint] = None

    X5: Optional[AssignedJoint] = None

    X6: Optional[AssignedJoint] = None

    X7: Optional[AssignedJoint] = None

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
# GRAPH NODE
# ==========================================================

@dataclass
class GraphNode:
    """
    Graph node representing one matched stereo target.
    """

    stereo_match: StereoMatch

    neighbours: List["GraphNode"] = field(default_factory=list)

    visited: bool = False
# ==========================================================
# JOINT ASSIGNMENT CONFIGURATION
# ==========================================================

class JointAssignmentConfig:

    def __init__(self):

        # Maximum number of neighbours
        self.max_neighbours = 2

        # Maximum distance between connected joints
        self.max_connection_distance = 180

        # Maximum horizontal deviation
        self.max_horizontal_difference = 150

        # Maximum vertical deviation
        self.max_vertical_difference = 220
# ==========================================================
# JOINT ASSIGNER
# ==========================================================

class JointAssigner:

    def __init__(self):

        self.cfg = JointAssignmentConfig()
	# ------------------------------------------------------

	def create_joint(

		self,

		joint_id,

		stereo_match

	):

		return AssignedJoint(

			joint_id=joint_id,

			stereo_match=stereo_match,

			locked=True,

			tracked=False

		)
	# ------------------------------------------------------

	def connect(

		self,

		parent,

		child

	):

		child.parent = parent

		parent.children.append(child)

	# ==========================================================
	# Joint Assignment
	# ==========================================================

	def assign_joint_ids(

		self,

		stereo_matches

	):
		"""
		Assign anatomical joint IDs to stereo matches.

		The assignment process first attempts a graph-based
		approach. If the graph cannot be constructed reliably,
		a geometry-based fallback is used.

		Parameters
		----------
		stereo_matches : List[StereoMatch]

		Returns
		-------
		AssignmentResult
		"""

		# ------------------------------------------------------
		# Try Graph Assignment
		# ------------------------------------------------------

		graph_result = self.assign_using_graph(

			stereo_matches

		)

		if graph_result.success:

			return graph_result

		# ------------------------------------------------------
		# Fallback
		# ------------------------------------------------------

		return self.assign_using_geometry(

			stereo_matches

		)
		
	# ------------------------------------------------------

	def node_distance(

		self,

		node1,

		node2

	):
		"""
		Compute Euclidean distance between two graph nodes
		using the left image coordinates.
		"""

		p1 = node1.stereo_match.left_target.marker

		p2 = node2.stereo_match.left_target.marker

		return np.sqrt(

			(p1.center_x - p2.center_x) ** 2 +

			(p1.center_y - p2.center_y) ** 2

		)
	# ------------------------------------------------------

	# ------------------------------------------------------

	def build_graph(

		self,

		stereo_matches

	):
		"""
		Construct a graph from stereo matches.

		Every stereo match becomes one graph node.

		Nodes are connected only if they satisfy the
		geometric connection constraints.

		Parameters
		----------
		stereo_matches : List[StereoMatch]

		Returns
		-------
		List[GraphNode]
		"""

		# ------------------------------------------
		# Create Graph Nodes
		# ------------------------------------------

		nodes = [

			GraphNode(match)

			for match in stereo_matches

		]

		# ------------------------------------------
		# Connect Valid Neighbours
		# ------------------------------------------

		for node in nodes:

			candidates = []

			for other in nodes:

				if node is other:

					continue

				if not self.is_valid_connection(

					node,

					other

				):

					continue

				distance = self.node_distance(

					node,

					other

				)

				candidates.append(

					(

						distance,

						other

					)

				)

			# --------------------------------------
			# Sort by Distance
			# --------------------------------------

			candidates.sort(

				key=lambda item: item[0]

			)

			# --------------------------------------
			# Connect Nearest Valid Nodes
			# --------------------------------------

			connected = 0

			for _, neighbour in candidates:

				if neighbour not in node.neighbours:

					node.neighbours.append(

						neighbour

					)

				if node not in neighbour.neighbours:

					neighbour.neighbours.append(

						node

					)

				connected += 1

				if connected >= self.cfg.max_neighbours:

					break

		return nodes
	# ------------------------------------------------------

	def is_valid_connection(

		self,

		node1,

		node2

	):
		"""
		Check whether two graph nodes can be connected.

		Parameters
		----------
		node1 : GraphNode

		node2 : GraphNode

		Returns
		-------
		bool
		"""

		marker1 = node1.stereo_match.left_target.marker
		marker2 = node2.stereo_match.left_target.marker

		dx = abs(

			marker1.center_x -

			marker2.center_x

		)

		dy = abs(

			marker1.center_y -

			marker2.center_y

		)

		distance = np.sqrt(

			dx ** 2 +

			dy ** 2

		)

		# ------------------------------------------
		# Distance Check
		# ------------------------------------------

		if distance > self.cfg.max_connection_distance:

			return False

		# ------------------------------------------
		# Horizontal Check
		# ------------------------------------------

		if dx > self.cfg.max_horizontal_difference:

			return False

		# ------------------------------------------
		# Vertical Check
		# ------------------------------------------

		if dy > self.cfg.max_vertical_difference:

			return False

		return True
		# ==========================================================
		# Find Root Node
		# ==========================================================

		def find_root_node(

			self,

			graph_nodes

		):
			"""
			Find the root node of the skeleton graph.

			The root is chosen as the highest marker
			in the left image.

			Parameters
			----------
			graph_nodes : List[GraphNode]

			Returns
			-------
			GraphNode
			"""

			if not graph_nodes:

				return None

			root = min(

				graph_nodes,

				key=lambda node:
				node.stereo_match.left_target.marker.center_y

			)

			return root
	# ==========================================================
	# Split Branches
	# ==========================================================

	def split_branches(

		self,

		root

	):
		"""
		Split the graph into left and right branches.

		Parameters
		----------
		root : GraphNode

		Returns
		-------
		Tuple[List[GraphNode], List[GraphNode]]
		"""

		left_branch = []

		right_branch = []

		if root is None:

			return left_branch, right_branch

		# ----------------------------------------------
		# Examine root neighbours
		# ----------------------------------------------

		for neighbour in root.neighbours:

			x = neighbour.stereo_match.left_target.marker.center_x

			root_x = root.stereo_match.left_target.marker.center_x

			if x < root_x:

				left_branch.append(

					neighbour

				)

			else:

				right_branch.append(

					neighbour

				)

		return left_branch, right_branch
	# ==========================================================
	# Traverse Branch
	# ==========================================================

	def traverse_branch(

		self,

		start_node,

		root

	):
		"""
		Traverse a single leg branch from hip to ankle.

		Parameters
		----------
		start_node : GraphNode
			Starting node of the branch (Hip).

		root : GraphNode
			Root node (Pelvis).

		Returns
		-------
		List[GraphNode]
			Ordered nodes from Hip → Knee → Ankle.
		"""

		branch = []

		previous = root

		current = start_node

		while current is not None:

			branch.append(current)

			next_node = None

			for neighbour in current.neighbours:

				if neighbour == previous:

					continue

				next_node = neighbour

				break

			previous = current

			current = next_node

		return branch
	# ==========================================================
	# Assign Using Graph
	# ==========================================================

	def assign_using_graph(

		self,

		stereo_matches

	):
		"""
		Assign joint IDs using graph traversal.

		Parameters
		----------
		stereo_matches : List[StereoMatch]

		Returns
		-------
		AssignmentResult
		"""

		# ----------------------------------------------
		# Build Graph
		# ----------------------------------------------

		graph = self.build_graph(

			stereo_matches

		)

		if len(graph) != 7:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Graph construction failed."

			)

		# ----------------------------------------------
		# Find Root
		# ----------------------------------------------

		root = self.find_root_node(

			graph

		)

		if root is None:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Root node not found."

			)

		# ----------------------------------------------
		# Split Branches
		# ----------------------------------------------

		left_branch, right_branch = self.split_branches(

			root

		)

		if len(left_branch) != 1 or len(right_branch) != 1:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Unable to identify leg branches."

			)

		# ----------------------------------------------
		# Traverse Both Legs
		# ----------------------------------------------

		left_chain = self.traverse_branch(

			left_branch[0],

			root

		)

		right_chain = self.traverse_branch(

			right_branch[0],

			root

		)

		if len(left_chain) != 3 or len(right_chain) != 3:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Incomplete leg chain."

			)

		# ----------------------------------------------
		# Build Skeleton
		# ----------------------------------------------

		skeleton = Skeleton()

		skeleton.X1 = self.create_joint(

			"X1",

			root.stereo_match

		)

		skeleton.X2 = self.create_joint(

			"X2",

			right_chain[0].stereo_match

		)

		skeleton.X3 = self.create_joint(

			"X3",

			right_chain[1].stereo_match

		)

		skeleton.X4 = self.create_joint(

			"X4",

			right_chain[2].stereo_match

		)

		skeleton.X5 = self.create_joint(

			"X5",

			left_chain[0].stereo_match

		)

		skeleton.X6 = self.create_joint(

			"X6",

			left_chain[1].stereo_match

		)

		skeleton.X7 = self.create_joint(

			"X7",

			left_chain[2].stereo_match

		)

		skeleton.valid = True

		return AssignmentResult(

			skeleton=skeleton,

			success=True,

			confidence=1.0,

			message="Graph assignment successful."

		)
	# ==========================================================
	# Assign Using Geometry
	# ==========================================================

	def assign_using_geometry(

		self,

		stereo_matches

	):
		"""
		Assign joint IDs using image geometry.

		This method is used as a fallback when graph
		assignment cannot be completed reliably.

		Parameters
		----------
		stereo_matches : List[StereoMatch]

		Returns
		-------
		AssignmentResult
		"""

		if len(stereo_matches) != 7:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Expected 7 stereo matches."

			)

		# --------------------------------------------------
		# Root (Highest Marker)
		# --------------------------------------------------

		root = min(

			stereo_matches,

			key=lambda match:
			match.left_target.marker.center_y

		)

		# --------------------------------------------------
		# Remaining Nodes
		# --------------------------------------------------

		remaining = [

			match

			for match in stereo_matches

			if match != root

		]

		# --------------------------------------------------
		# Split Left / Right
		# --------------------------------------------------

		root_x = root.left_target.marker.center_x

		left = []

		right = []

		for match in remaining:

			if match.left_target.marker.center_x < root_x:

				left.append(match)

			else:

				right.append(match)

		if len(left) != 3 or len(right) != 3:

			return AssignmentResult(

				skeleton=Skeleton(),

				success=False,

				confidence=0.0,

				message="Unable to separate body sides."

			)

		# --------------------------------------------------
		# Sort Hip → Knee → Ankle
		# --------------------------------------------------

		left.sort(

			key=lambda m:
			m.left_target.marker.center_y

		)

		right.sort(

			key=lambda m:
			m.left_target.marker.center_y

		)

		# --------------------------------------------------
		# Build Skeleton
		# --------------------------------------------------

		skeleton = Skeleton()

		skeleton.X1 = self.create_joint(

			"X1",

			root

		)

		skeleton.X2 = self.create_joint(

			"X2",

			right[0]

		)

		skeleton.X3 = self.create_joint(

			"X3",

			right[1]

		)

		skeleton.X4 = self.create_joint(

			"X4",

			right[2]

		)

		skeleton.X5 = self.create_joint(

			"X5",

			left[0]

		)

		skeleton.X6 = self.create_joint(

			"X6",

			left[1]

		)

		skeleton.X7 = self.create_joint(

			"X7",

			left[2]

		)

		skeleton.valid = True

		return AssignmentResult(

			skeleton=skeleton,

			success=True,

			confidence=0.75,

			message="Geometry fallback assignment successful."

		)
