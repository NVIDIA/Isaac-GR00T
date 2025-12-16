from typing import Generic, List, Optional, Sequence, TypeVar, Union

from gr00t.data.state_action.pose import EndEffectorPose, JointPose, Pose
from gr00t.data.types import ActionFormat
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp


PoseType = TypeVar("PoseType", bound=Pose)


class ActionChunk(Generic[PoseType]):
    """
    Abstract base class for robot action chunking.

    This class provides common functionality for different action chunking types
    including relative and delta action chunking computation with optional reference frames,
    interpolation, and format conversion.
    """

    def __init__(
        self,
        poses: Sequence[PoseType],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize action chunking from a list of poses.

        Args:
            poses: Sequence of Pose objects
            times: Optional sequence of timestamps for each pose. If None, assumes
                   uniform spacing starting from 0 with step 1.0

        Raises:
            ValueError: If action chunking is empty or times length doesn't match poses
        """
        if not poses:
            raise ValueError("ActionChunk must contain at least one pose")

        self._poses: List[PoseType] = list(poses)

        # Set up times
        if times is None:
            self._times = np.arange(len(poses), dtype=np.float64)
        else:
            if len(times) != len(poses):
                raise ValueError("Number of times must match number of poses")
            self._times = np.array(times, dtype=np.float64)

    @property
    def poses(self) -> List[PoseType]:
        """Get the list of poses"""
        return self._poses.copy()

    @property
    def times(self) -> NDArray[np.float64]:
        """Get the timestamps"""
        return self._times.copy()

    @property
    def num_poses(self) -> int:
        """Get the number of poses in the action chunking"""
        return len(self._poses)

    def relative_chunking(
        self, reference_frame: Optional[PoseType] = None
    ) -> "ActionChunk[PoseType]":
        """
        Compute the relative action chunking with respect to a reference frame.

        If reference_frame is None, uses the first pose in the action chunking as reference.
        All poses are transformed to be relative to the reference frame.

        Args:
            reference_frame: Optional reference pose. If None, uses first pose.

        Returns:
            A new ActionChunk of the same type where all poses are relative to the reference frame.
        """
        if not self._poses:
            return self.__class__([], times=[])

        # Use the first pose as the reference if one is not provided.
        ref_pose = reference_frame if reference_frame is not None else self._poses[0]

        # Use the polymorphic subtraction defined in the Pose subclasses.
        # The subtraction returns the same type as the operands
        relative_poses: List[PoseType] = [pose - ref_pose for pose in self._poses]  # type: ignore[misc]

        # Return a new instance of the same action chunking class
        # (e.g., JointActionChunk or EndEffectorActionChunk)
        return self.__class__(relative_poses, times=self.times)

    def delta_chunking(self, reference_frame: Optional[PoseType] = None) -> "ActionChunk[PoseType]":
        """
        Compute the delta action chunking where each pose represents the relative
        transformation from the previous frame.

        If reference_frame is provided, it is treated as the first frame, and the
        first delta will be from reference_frame to the first pose in the action chunking.
        Otherwise, the first pose in the delta action chunking will be the identity/zero transformation.

        Args:
            reference_frame: Optional reference pose to use as the first frame.

        Returns:
            A new ActionChunk of the same type where each pose is relative to the previous pose.
        """
        if not self._poses:
            return self.__class__([], times=[])

        delta_poses: List[PoseType] = []

        # Determine the initial reference for the very first pose.
        # If a reference_frame is given, the first delta is pose[0] - reference_frame.
        # If not, the first delta is pose[0] - pose[0], resulting in an identity/zero pose.
        prev_pose = reference_frame if reference_frame is not None else self._poses[0]

        for current_pose in self._poses:
            delta: PoseType = current_pose - prev_pose  # type: ignore[assignment]
            delta_poses.append(delta)
            prev_pose = current_pose  # Update the reference for the next step

        return self.__class__(delta_poses, times=self.times.tolist())

    def to_absolute_chunking(self, reference_frame: PoseType) -> "ActionChunk[PoseType]":
        """
        Convert a relative action chunking to an absolute action chunking by applying
        the relative poses on top of a reference frame.

        This is the inverse operation of relative_chunking(). Each relative pose
        is composed with the reference frame to produce absolute poses.

        Args:
            reference_frame: The reference pose to apply the relative action chunking on top of.

        Returns:
            A new ActionChunk of the same type with absolute poses.

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement to_absolute_chunking")

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "ActionChunk":
        """
        Interpolate the action chunking to generate intermediate poses.
        Must be implemented by subclasses.

        Args:
            num_points: Number of evenly-spaced points to generate
            times: Specific timestamps at which to interpolate

        Returns:
            A new ActionChunk with interpolated poses
        """
        raise NotImplementedError("Subclasses must implement interpolate")

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert action chunking to the specified action format.
        Must be implemented by subclasses.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement to method")

    def __len__(self) -> int:
        """Return the number of poses in the action chunking"""
        return len(self._poses)

    def __getitem__(self, index: int) -> PoseType:
        """Get a pose by index"""
        return self._poses[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_poses={len(self._poses)}, "
            f"time_range=[{self._times[0]:.2f}, {self._times[-1]:.2f}])"
        )


class JointActionChunk(ActionChunk[JointPose]):
    """
    Represents action chunking in joint space as a sequence of joint configurations.

    Examples:
        # Create a joint action chunking
        joint_poses = [
            JointPose([0.0, 0.0, 0.0]),
            JointPose([0.5, 0.5, 0.5]),
            JointPose([1.0, 1.0, 1.0]),
        ]
        action_chunking = JointActionChunk(joint_poses)

        # Get relative trajectory (all poses relative to first pose)
        relative_traj = action_chunking.relative_chunking()

        # Get relative trajectory with custom reference
        reference = JointPose([0.1, 0.1, 0.1])
        relative_traj = action_chunking.relative_chunking(reference_frame=reference)

        # Get delta trajectory (incremental changes)
        delta_traj = action_chunking.delta_chunking()

        # Convert relative trajectory back to absolute
        reference = JointPose([0.1, 0.1, 0.1])
        absolute_traj = relative_traj.to_absolute_chunking(reference_frame=reference)

        # Interpolate trajectory
        interpolated = action_chunking.interpolate(num_points=10)

        # Convert to desired format
        from gr00t.data.types import ActionFormat
        array_data = action_chunking.to(ActionFormat.DEFAULT)  # Returns joint array
    """

    def __init__(
        self,
        poses: Sequence[JointPose],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize a joint trajectory from a list of joint poses.

        Args:
            poses: Sequence of JointPose objects
            times: Optional sequence of timestamps for each pose

        Raises:
            TypeError: If poses are not all JointPose objects
        """
        # Validate all poses are JointPose
        if not all(isinstance(p, JointPose) for p in poses):
            raise TypeError("All poses must be JointPose objects for JointActionChunk")

        super().__init__(poses, times)

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "JointActionChunk":
        """
        Interpolate the joint action chunking to generate intermediate configurations.

        Uses linear interpolation for joint values.

        Args:
            num_points: Number of evenly-spaced points to generate (including endpoints).
                       Only used if times is None.
            times: Specific timestamps at which to interpolate. If provided,
                  num_points is ignored.

        Returns:
            A new JointActionChunk with interpolated poses

        Raises:
            ValueError: If neither num_points nor times is provided, or if
                       interpolation times are outside the trajectory range
        """
        if num_points is None and times is None:
            raise ValueError("Must provide either num_points or times")

        if len(self._poses) < 2:
            raise ValueError("Need at least 2 poses for interpolation")

        # Prepare data: extract joint values
        timestamps = self._times.copy()
        joint_values = np.array([pose.joints for pose in self._poses])  # (N, num_joints)

        # Find and remove non-monotonic timestamps
        drop_indices = [
            idx for idx in range(1, len(timestamps)) if timestamps[idx] <= timestamps[idx - 1]
        ]

        if drop_indices:
            for idx in drop_indices:
                print(
                    f"Dropping timestamp pair - Previous: {timestamps[idx - 1]}, "
                    f"Current: {timestamps[idx]} at index {idx}"
                )
            timestamps = np.delete(timestamps, drop_indices)
            joint_values = np.delete(joint_values, drop_indices, axis=0)

        # Check if we still have enough poses after cleanup
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 poses with monotonic timestamps for interpolation")

        # Create interpolator
        joint_interp = interpolate.interp1d(timestamps, joint_values, kind="linear", axis=0)

        # Generate interpolation times if not provided
        if times is None:
            assert num_points is not None  # Type narrowing for type checker
            interp_times = np.linspace(timestamps[0], timestamps[-1], num_points)
        else:
            interp_times = np.array(times, dtype=np.float64)

        # Check that interpolation times are within bounds
        if np.any(interp_times < timestamps[0]) or np.any(interp_times > timestamps[-1]):
            raise ValueError(
                f"Interpolation times must be within [{timestamps[0]}, {timestamps[-1]}]"
            )

        # Interpolate joint values
        interp_joint_values = joint_interp(interp_times)

        # Create interpolated poses
        joint_names = self._poses[0].joint_names
        interpolated_poses = [
            JointPose(joints=interp_joint_values[i], joint_names=joint_names)
            for i in range(len(interp_times))
        ]

        return JointActionChunk(interpolated_poses, times=interp_times)

    def to_array(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of joint values.

        Returns:
            Array with shape (N, num_joints) where N is the number of poses
        """
        return np.array([pose.joints for pose in self._poses])

    def to_absolute_chunking(self, reference_frame: JointPose) -> "JointActionChunk":
        """
        Convert a relative joint action chunking to an absolute action chunking by adding
        the relative joint positions to the reference frame.

        This is the inverse operation of relative_chunking(). Each relative joint
        configuration is added to the reference frame to produce absolute joint positions.

        Args:
            reference_frame: The reference joint pose to apply the relative trajectory on top of.

        Returns:
            A new JointActionChunk with absolute joint positions.

        Raises:
            ValueError: If joint dimensions don't match
        """
        if not self._poses:
            return JointActionChunk([], times=[])

        if len(self._poses[0].joints) != len(reference_frame.joints):
            raise ValueError(
                f"Cannot apply relative trajectory: "
                f"joint dimensions don't match ({len(self._poses[0].joints)} vs {len(reference_frame.joints)})"
            )

        # Add each relative pose to the reference frame
        absolute_poses: List[JointPose] = []
        for relative_pose in self._poses:
            absolute_joints = reference_frame.joints + relative_pose.joints
            absolute_pose = JointPose(
                joints=absolute_joints, joint_names=reference_frame.joint_names
            )
            absolute_poses.append(absolute_pose)

        return JointActionChunk(absolute_poses, times=self.times)

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert trajectory to the desired format.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            ValueError: If the action format is not supported for joint trajectories
        """
        if action_format == ActionFormat.DEFAULT:
            return self.to_array()
        else:
            raise ValueError(
                f"ActionFormat {action_format} is not supported for JointActionChunk. "
                f"Only {ActionFormat.DEFAULT} is supported."
            )


class EndEffectorActionChunk(ActionChunk[EndEffectorPose]):
    """
    Represents action chunking in Cartesian space as a sequence of end-effector poses.

    Examples:
        # Create an end-effector action chunking
        ee_poses = [
            EndEffectorPose(translation=[0, 0, 0], rotation=[1, 0, 0, 0],
                          rotation_type="quat", rotation_order="wxyz"),
            EndEffectorPose(translation=[1, 0, 0], rotation=[0.707, 0, 0, 0.707],
                          rotation_type="quat", rotation_order="wxyz"),
            EndEffectorPose(translation=[2, 0, 0], rotation=[0, 0, 0, 1],
                          rotation_type="quat", rotation_order="wxyz"),
        ]
        action_chunking = EndEffectorActionChunk(ee_poses)

        # Get relative trajectory (all poses relative to first pose)
        relative_traj = action_chunking.relative_chunking()

        # Get relative trajectory with custom reference frame
        reference = EndEffectorPose(translation=[0.5, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
        relative_traj = action_chunking.relative_chunking(reference_frame=reference)

        # Get delta trajectory
        delta_traj = action_chunking.delta_chunking()

        # Convert relative trajectory back to absolute
        reference = EndEffectorPose(translation=[0.5, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
        absolute_traj = relative_traj.to_absolute_chunking(reference_frame=reference)

        # Interpolate trajectory
        interpolated = action_chunking.interpolate(num_points=10)

        # Convert to desired format
        from gr00t.data.types import ActionFormat
        homo_matrices = action_chunking.to(ActionFormat.DEFAULT)      # (N, 4, 4) homogeneous matrices
        xyz_rot6d = action_chunking.to(ActionFormat.XYZ_ROT6D)        # (N, 9) xyz + rot6d
        xyz_rotvec = action_chunking.to(ActionFormat.XYZ_ROTVEC)      # (N, 6) xyz + rotvec
    """

    def __init__(
        self,
        poses: Sequence[EndEffectorPose],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize an end-effector trajectory from a list of end-effector poses.

        Args:
            poses: Sequence of EndEffectorPose objects
            times: Optional sequence of timestamps for each pose

        Raises:
            TypeError: If poses are not all EndEffectorPose objects
        """
        # Validate all poses are EndEffectorPose
        if not all(isinstance(p, EndEffectorPose) for p in poses):
            raise TypeError("All poses must be EndEffectorPose objects for EndEffectorActionChunk")

        super().__init__(poses, times)

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "EndEffectorActionChunk":
        """
        Interpolate the action chunking to generate intermediate poses.

        Uses linear interpolation for translation and SLERP (Spherical Linear
        Interpolation) for rotation.

        Args:
            num_points: Number of evenly-spaced points to generate (including endpoints).
                       Only used if times is None.
            times: Specific timestamps at which to interpolate. If provided,
                  num_points is ignored.

        Returns:
            A new EndEffectorActionChunk with interpolated poses

        Raises:
            ValueError: If neither num_points nor times is provided, or if
                       interpolation times are outside the trajectory range
        """
        if num_points is None and times is None:
            raise ValueError("Must provide either num_points or times")

        if len(self._poses) < 2:
            raise ValueError("Need at least 2 poses for interpolation")

        # Prepare data: extract positions and rotations
        timestamps = self._times.copy()
        homogeneous_matrices = np.array([pose.homogeneous for pose in self._poses])
        positions = homogeneous_matrices[:, :3, 3]
        rotations = Rotation.from_matrix(homogeneous_matrices[:, :3, :3])

        # Find indices where timestamps are not monotonically increasing
        drop_indices = [
            idx for idx in range(1, len(timestamps)) if timestamps[idx] <= timestamps[idx - 1]
        ]

        # Remove the problematic timestamps and corresponding data
        if drop_indices:
            for idx in drop_indices:
                print(
                    f"Dropping timestamp pair - Previous: {timestamps[idx - 1]}, "
                    f"Current: {timestamps[idx]} at index {idx}"
                )
            timestamps = np.delete(timestamps, drop_indices)
            positions = np.delete(positions, drop_indices, axis=0)
            rotations = Rotation.from_matrix(
                np.delete(homogeneous_matrices[:, :3, :3], drop_indices, axis=0)
            )

        # Check if we still have enough poses after cleanup
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 poses with monotonic timestamps for interpolation")

        # Create interpolators
        pos_interp = interpolate.interp1d(timestamps, positions, kind="linear", axis=0)
        rot_interp = Slerp(timestamps, rotations)

        # Generate interpolation times if not provided
        if times is None:
            assert num_points is not None  # Type narrowing for type checker
            interp_times = np.linspace(timestamps[0], timestamps[-1], num_points)
        else:
            interp_times = np.array(times, dtype=np.float64)

        # Check that interpolation times are within bounds
        if np.any(interp_times < timestamps[0]) or np.any(interp_times > timestamps[-1]):
            raise ValueError(
                f"Interpolation times must be within [{timestamps[0]}, {timestamps[-1]}]"
            )

        # Interpolate positions and rotations
        interp_positions = pos_interp(interp_times)
        interp_rotations = rot_interp(interp_times)

        # Create interpolated poses
        interpolated_poses = []
        for i in range(len(interp_times)):
            pose = EndEffectorPose(
                translation=interp_positions[i],
                rotation=interp_rotations[i].as_matrix(),
                rotation_type="matrix",
            )
            interpolated_poses.append(pose)

        return EndEffectorActionChunk(interpolated_poses, times=interp_times)

    def to_homogeneous_matrices(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of homogeneous transformation matrices.

        Returns:
            Array of homogeneous matrices with shape (N, 4, 4) where N is the number of poses
        """
        return np.array([pose.homogeneous for pose in self._poses])

    def to_translation_rot6d(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of translations and 6D rotations.

        Returns:
            Array with shape (N, 9) - 3 for xyz + 6 for rot6d
        """
        translations = np.array([pose.translation for pose in self._poses])  # (N, 3)
        rotations = np.array([pose.rot6d for pose in self._poses])  # (N, 6)

        # Concatenate translation and rotation
        xyz_rot6d = np.concatenate([translations, rotations], axis=1)  # (N, 9)

        return xyz_rot6d

    def to_translation_rotvec(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of translations and rotation vectors.

        Returns:
            Array with shape (N, 6) - 3 for xyz + 3 for rotvec
        """
        translations = np.array([pose.translation for pose in self._poses])  # (N, 3)
        rotations = np.array([pose.rotvec for pose in self._poses])  # (N, 3)

        # Concatenate translation and rotation
        xyz_rotvec = np.concatenate([translations, rotations], axis=1)  # (N, 6)

        return xyz_rotvec

    def to_absolute_chunking(self, reference_frame: EndEffectorPose) -> "EndEffectorActionChunk":
        """
        Convert a relative end-effector action chunking to an absolute action chunking by
        composing each relative transformation with the reference frame.

        This is the inverse operation of relative_chunking(). Each relative pose
        represents a transformation that is applied on top of the reference frame
        to produce absolute poses.

        Args:
            reference_frame: The reference end-effector pose to apply the relative trajectory on top of.

        Returns:
            A new EndEffectorActionChunk with absolute poses.
        """
        if not self._poses:
            return EndEffectorActionChunk([], times=[])

        # Get reference frame as homogeneous matrix
        T_ref = reference_frame.homogeneous

        # Compose each relative transformation with the reference frame
        absolute_poses: List[EndEffectorPose] = []
        for relative_pose in self._poses:
            # Get relative transformation as homogeneous matrix
            T_relative = relative_pose.homogeneous

            # Compose transformations: T_absolute = T_ref @ T_relative
            T_absolute = T_ref @ T_relative

            # Create absolute pose from composed transformation
            absolute_pose = EndEffectorPose(homogeneous=T_absolute)
            absolute_poses.append(absolute_pose)

        return EndEffectorActionChunk(absolute_poses, times=self.times)

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert trajectory to the desired format.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            ValueError: If the action format is not supported
        """
        if action_format == ActionFormat.DEFAULT:
            return self.to_homogeneous_matrices()
        elif action_format == ActionFormat.XYZ_ROT6D:
            return self.to_translation_rot6d()
        elif action_format == ActionFormat.XYZ_ROTVEC:
            return self.to_translation_rotvec()
        else:
            raise ValueError(f"Unsupported action format: {action_format}")
