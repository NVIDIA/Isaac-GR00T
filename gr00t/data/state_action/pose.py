from enum import Enum
from typing import Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


# TypeVar for self-type preservation in Pose operations
PoseT = TypeVar("PoseT", bound="Pose")


def invert_transformation(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Invert a homogeneous transformation matrix.

    Args:
        T: A 4x4 homogeneous transformation matrix

    Returns:
        The inverse of the transformation matrix (4x4)
    """
    R = T[:3, :3]  # Extract the rotation matrix
    t = T[:3, 3]  # Extract the translation vector

    # Inverse of the rotation matrix is its transpose (since it's orthogonal)
    R_inv = R.T

    # Inverse of the translation is -R_inv * t
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def relative_transformation(
    T0: NDArray[np.float64], Tt: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the relative transformation between two poses.

    Args:
        T0: Initial 4x4 homogeneous transformation matrix
        Tt: Current 4x4 homogeneous transformation matrix

    Returns:
        The relative transformation matrix (4x4) from T0 to Tt
    """
    # Relative transformation is T0^{-1} * Tt
    T_relative = invert_transformation(T0) @ Tt
    return T_relative


class RotationType(Enum):
    """Supported rotation representation types"""

    QUAT = "quat"
    EULER = "euler"
    ROTVEC = "rotvec"
    MATRIX = "matrix"
    ROT6D = "rot6d"


class EulerOrder(Enum):
    """Common Euler angle conventions"""

    XYZ = "xyz"
    ZYX = "zyx"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"


class QuatOrder(Enum):
    """Quaternion ordering conventions"""

    WXYZ = "wxyz"  # scalar-first (w, x, y, z)
    XYZW = "xyzw"  # scalar-last (x, y, z, w)


class Pose:
    """
    Abstract base class for robot poses.

    This class provides common functionality for different pose representations
    including relative pose computation via the subtraction operator.
    """

    pose_type: str

    def __sub__(self: PoseT, other: PoseT) -> PoseT:
        """
        Compute relative transformation between two poses.

        For EndEffectorPose: Computes the relative transformation from other to self.
        Result represents the transformation needed to go from other's frame to self's frame.

        For JointPose: Computes the joint-space difference (self - other).

        Args:
            other: The reference pose to compute relative transformation from

        Returns:
            Relative pose (same type as self)

        Raises:
            TypeError: If poses are not of the same type

        Examples:
            # End-effector poses
            pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            relative = pose2 - pose1  # Transformation from pose1 to pose2

            # Joint poses
            joint1 = JointPose([0.0, 0.5, 1.0])
            joint2 = JointPose([0.1, 0.6, 1.2])
            joint_diff = joint2 - joint1  # Joint differences: [0.1, 0.1, 0.2]
        """
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute relative transformation between different pose types: "
                f"{type(self).__name__} and {type(other).__name__}"
            )

        return self._compute_relative(other)

    def _compute_relative(self: PoseT, other: PoseT) -> PoseT:
        """
        Internal method to compute relative transformation.
        Must be implemented by subclasses.

        Args:
            other: The reference pose

        Returns:
            Relative pose
        """
        raise NotImplementedError("Subclasses must implement _compute_relative")

    def copy(self: PoseT) -> PoseT:
        """
        Create a deep copy of this pose.
        Must be implemented by subclasses.

        Returns:
            New Pose instance with copied data
        """
        raise NotImplementedError("Subclasses must implement copy")


class JointPose(Pose):
    """
    Represents a robot configuration in joint space.

    This class stores joint angles/positions for a robot manipulator.
    Unlike end-effector poses, joint poses represent the configuration
    of all joints in the kinematic chain.

    Examples:
        # Create a 6-DOF joint configuration
        joint_pose = JointPose(
            joints=[0.0, -np.pi/4, np.pi/2, 0.0, np.pi/4, 0.0],
            joint_names=["shoulder_pan", "shoulder_lift", "elbow",
                        "wrist_1", "wrist_2", "wrist_3"]
        )

        # Create with default joint names
        joint_pose = JointPose(joints=[0.0, 0.5, 1.0])

        # Get as dictionary
        joint_dict = joint_pose.to_dict()  # {"joint_0": 0.0, ...}

        # Access individual joints
        first_joint = joint_pose.joints[0]
        num_joints = joint_pose.num_joints

        # Compute relative joint displacement
        joint1 = JointPose([0.0, 0.5, 1.0])
        joint2 = JointPose([0.1, 0.6, 1.2])
        relative = joint2 - joint1  # [0.1, 0.1, 0.2]
    """

    pose_type = "joint"

    def __init__(
        self,
        joints: Union[list, np.ndarray],
        joint_names: Optional[list] = None,
    ):
        """
        Initialize a joint pose.

        Args:
            joints: Joint angles/positions as array-like of shape (n,)
            joint_names: Optional list of names for each joint. If None,
                        defaults to ["joint_0", "joint_1", ...]
        """
        super().__init__()
        self.joints = np.array(joints, dtype=np.float64)

        # Set defaults and validate joint_names
        if joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(len(self.joints))]
        else:
            if len(joint_names) != len(self.joints):
                raise ValueError(
                    f"Number of joint names ({len(joint_names)}) must match "
                    f"number of joints ({len(self.joints)})"
                )
            self.joint_names = joint_names

    @property
    def num_joints(self) -> int:
        """
        Get the number of joints.

        Returns:
            Number of joints in the configuration
        """
        return len(self.joints)

    def to_dict(self) -> dict:
        """
        Convert joint configuration to dictionary.

        Returns:
            Dictionary mapping joint names to joint values
        """
        return dict(zip(self.joint_names, self.joints))

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative joint displacement.

        Args:
            other: Reference joint pose

        Returns:
            JointPose representing the joint-space difference (self - other)

        Raises:
            ValueError: If joint dimensions don't match
        """
        if len(self.joints) != len(other.joints):
            raise ValueError(
                f"Cannot compute relative joint pose: "
                f"joint dimensions don't match ({len(self.joints)} vs {len(other.joints)})"
            )

        relative_joints = self.joints - other.joints
        return JointPose(joints=relative_joints, joint_names=self.joint_names)

    def copy(self) -> "JointPose":
        """
        Create a deep copy of this joint pose.

        Returns:
            New JointPose instance with copied data
        """
        return JointPose(
            joints=self.joints.copy(),
            joint_names=self.joint_names.copy(),
        )

    def __repr__(self) -> str:
        if len(self.joints) <= 6:
            joints_str = np.array2string(self.joints, precision=4, suppress_small=True)
        else:
            joints_str = (
                f"[{self.joints[0]:.4f}, ..., {self.joints[-1]:.4f}] ({len(self.joints)} joints)"
            )

        return f"JointPose(joints={joints_str})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, JointPose):
            return False
        return np.allclose(self.joints, other.joints) and self.joint_names == other.joint_names

    def __getitem__(self, index) -> Union[float, NDArray[np.float64]]:
        """Allow indexing: joint_pose[0] returns first joint value"""
        return self.joints[index]

    def __len__(self) -> int:
        """Allow len(): len(joint_pose) returns number of joints"""
        return len(self.joints)


class EndEffectorPose(Pose):
    """
    Represents a single end-effector pose with translation and rotation components.

    This class handles Cartesian space representations of robot end-effector poses,
    supporting multiple rotation representations (quaternions, Euler angles, rotation
    vectors, rotation matrices, etc.).

    Examples:
        # Create with quaternion (wxyz order)
        pose = EndEffectorPose(
            translation=[1.0, 2.0, 3.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            rotation_type="quat",
            rotation_order="wxyz"
        )

        # Create with Euler angles (degrees by default)
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, 90],
            rotation_type="euler",
            rotation_order="xyz"
        )

        # Create with Euler angles in radians
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, np.pi/2],
            rotation_type="euler",
            rotation_order="xyz",
            degrees=False
        )

        # Create from homogeneous matrix
        H = np.eye(4)
        H[:3, 3] = [1, 2, 3]
        pose = EndEffectorPose(homogeneous=H)

        # Convert between representations
        quat_wxyz = pose.to_rotation("quat", "wxyz")
        euler_zyx = pose.to_rotation("euler", "zyx")
        rot6d = pose.to_rotation("rot6d")

        # Compute relative transformation
        pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        relative = pose2 - pose1  # Transformation from pose1's frame to pose2's frame
    """

    pose_type = "end_effector"

    def __init__(
        self,
        translation: Optional[Union[list, np.ndarray]] = None,
        rotation: Optional[Union[list, np.ndarray]] = None,
        rotation_type: Optional[str] = None,
        rotation_order: Optional[str] = None,
        homogeneous: Optional[np.ndarray] = None,
        degrees: bool = True,
    ):
        """
        Initialize an end-effector pose.

        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation in specified format
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            homogeneous: Homogeneous transformation matrix (4, 4)
                        If provided, overrides translation and rotation
            degrees: For Euler angles, whether the input is in degrees (default True)
        """
        super().__init__()

        # Cache for homogeneous matrix
        self._homogeneous_cache: Optional[NDArray[np.float64]] = None
        self._cache_valid = False

        # Handle homogeneous matrix input
        if homogeneous is not None:
            self._from_homogeneous(homogeneous)
            return

        # Store translation
        self._translation = np.array(translation) if translation is not None else np.zeros(3)

        # Store rotation as scipy Rotation object internally
        if rotation is not None:
            if rotation_type is None:
                raise ValueError("rotation_type must be specified when rotation is provided")
            self._set_rotation(rotation, rotation_type, rotation_order, degrees)
        else:
            self._rotation = Rotation.identity()

    def _from_homogeneous(self, homogeneous: np.ndarray):
        """Initialize from homogeneous transformation matrix"""
        homogeneous = np.array(homogeneous)

        # Extract translation (last column, first 3 rows)
        self._translation = homogeneous[:3, 3]

        # Extract rotation matrix (top-left 3x3)
        rotation_matrix = homogeneous[:3, :3]

        # Create Rotation object from matrix
        self._rotation = Rotation.from_matrix(rotation_matrix)

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation to rotation matrix.

        Args:
            rot6d: 6D rotation as (6,) array - first two rows of rotation matrix flattened

        Returns:
            Rotation matrix (3, 3)
        """
        rot6d = rot6d.reshape(2, 3)

        # First two rows
        row1 = rot6d[0]
        row2 = rot6d[1]

        # Normalize first row
        row1 = row1 / np.linalg.norm(row1)

        # Gram-Schmidt orthogonalization for second row
        row2 = row2 - np.dot(row1, row2) * row1
        row2 = row2 / np.linalg.norm(row2)

        # Third row is cross product
        row3 = np.cross(row1, row2)

        # Construct rotation matrix
        rotation_matrix = np.vstack([row1, row2, row3])

        return rotation_matrix

    @staticmethod
    def _matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to 6D rotation representation.

        Args:
            rotation_matrix: Rotation matrix (3, 3)

        Returns:
            6D rotation - (6,) array (first two rows flattened)
        """
        return rotation_matrix[:2, :].flatten()

    def _set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """Internal method to set rotation from various representations"""
        rotation = np.array(rotation)
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            if quat_order == QuatOrder.WXYZ:
                # scipy uses xyzw order, so convert
                quat_xyzw = np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
            else:
                quat_xyzw = rotation
            self._rotation = Rotation.from_quat(quat_xyzw)

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            self._rotation = Rotation.from_euler(euler_order.value, rotation, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            self._rotation = Rotation.from_rotvec(rotation)

        elif rot_type == RotationType.MATRIX:
            self._rotation = Rotation.from_matrix(rotation)

        elif rot_type == RotationType.ROT6D:
            rotation_matrix = self._rot6d_to_matrix(rotation)
            self._rotation = Rotation.from_matrix(rotation_matrix)

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

        # Invalidate cache
        self._cache_valid = False

    @property
    def translation(self) -> np.ndarray:
        """
        Get translation vector.

        Returns:
            Translation array - shape (3,)
        """
        return self._translation.copy()

    @property
    def quat_wxyz(self) -> np.ndarray:
        """Get rotation as quaternion in wxyz order (w, x, y, z)"""
        return self.to_rotation("quat", "wxyz")

    @property
    def quat_xyzw(self) -> np.ndarray:
        """Get rotation as quaternion in xyzw order (x, y, z, w)"""
        return self.to_rotation("quat", "xyzw")

    @property
    def euler_xyz(self) -> np.ndarray:
        """Get rotation as Euler angles in xyz order (degrees)"""
        return self.to_rotation("euler", "xyz")

    @property
    def rotvec(self) -> np.ndarray:
        """Get rotation as rotation vector (axis-angle)"""
        return self.to_rotation("rotvec")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation as 3x3 rotation matrix"""
        return self.to_rotation("matrix")

    @property
    def rot6d(self) -> np.ndarray:
        """Get rotation as 6D representation (first two rows of rotation matrix)"""
        return self.to_rotation("rot6d")

    @property
    def xyz_rot6d(self) -> np.ndarray:
        """Get pose as concatenated translation and 6D rotation (9,)"""
        return np.concatenate([self._translation, self.rot6d])

    @property
    def xyz_rotvec(self) -> np.ndarray:
        """Get pose as concatenated translation and rotation vector (6,)"""
        return np.concatenate([self._translation, self.rotvec])

    @property
    def homogeneous(self) -> np.ndarray:
        """
        Get homogeneous transformation matrix.

        Returns:
            Homogeneous matrix - shape (4, 4)
        """
        if not self._cache_valid:
            self._homogeneous_cache = self._compute_homogeneous()
            self._cache_valid = True
        assert self._homogeneous_cache is not None
        return self._homogeneous_cache.copy()

    def _compute_homogeneous(self) -> np.ndarray:
        """Compute homogeneous transformation matrix"""
        H = np.eye(4)
        H[:3, :3] = self._rotation.as_matrix()
        H[:3, 3] = self._translation
        return H

    def to_rotation(
        self, rotation_type: str, rotation_order: Optional[str] = None, degrees: bool = True
    ) -> np.ndarray:
        """
        Get rotation in specified representation.

        Args:
            rotation_type: Desired type ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, return in degrees (default True)

        Returns:
            Rotation in requested format
            - Shape (4,) for quat
            - Shape (3,) for euler/rotvec
            - Shape (6,) for rot6d
            - Shape (3, 3) for matrix
        """
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.ROT6D:
            rotation_matrix = self._rotation.as_matrix()
            return self._matrix_to_rot6d(rotation_matrix)

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            quat_xyzw = self._rotation.as_quat()
            if quat_order == QuatOrder.WXYZ:
                return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            else:
                return quat_xyzw

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            return self._rotation.as_euler(euler_order.value, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            return self._rotation.as_rotvec()

        elif rot_type == RotationType.MATRIX:
            return self._rotation.as_matrix()

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

    def to_homogeneous(self) -> np.ndarray:
        """
        Convert pose to homogeneous transformation matrix.
        (Alias for the homogeneous property)

        Returns:
            Homogeneous matrix - shape (4, 4)
        """
        return self.homogeneous

    def set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """
        Set rotation from specified representation.

        Args:
            rotation: Rotation data
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, whether the input is in degrees (default True)
        """
        self._set_rotation(rotation, rotation_type, rotation_order, degrees)

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative transformation from other to self.

        The result represents the transformation needed to go from other's frame to self's frame.
        Mathematically: T_relative = T_other^{-1} * T_self

        Args:
            other: Reference end-effector pose

        Returns:
            EndEffectorPose representing the relative transformation
        """
        # Get homogeneous matrices
        T_self = self.homogeneous
        T_other = other.homogeneous

        # Compute relative transformation: T_other^{-1} * T_self
        T_relative = relative_transformation(T_other, T_self)

        # Create new EndEffectorPose from relative transformation
        return EndEffectorPose(homogeneous=T_relative)

    def copy(self) -> "EndEffectorPose":
        """
        Create a deep copy of this end-effector pose.

        Returns:
            New EndEffectorPose instance with copied data
        """
        return EndEffectorPose(
            translation=self._translation.copy(),
            rotation=self._rotation.as_quat(),
            rotation_type="quat",
            rotation_order="xyzw",
        )

    def __repr__(self) -> str:
        quat = self.to_rotation("quat", "wxyz")
        return f"EndEffectorPose(translation={self.translation}, rotation_quat_wxyz={quat})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EndEffectorPose):
            return False
        return np.allclose(self._translation, other._translation) and np.allclose(
            self._rotation.as_quat(), other._rotation.as_quat()
        )
