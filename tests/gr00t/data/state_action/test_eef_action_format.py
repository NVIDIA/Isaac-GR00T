"""Tests for EEF (End Effector) action format support.

Covers:
- EndEffectorPose.from_action_format() for all ActionFormat variants
- EndEffectorActionChunk.from_array() for all ActionFormat variants
- Absolute -> relative -> absolute roundtrip consistency
"""

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.types import ActionFormat
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_pose() -> EndEffectorPose:
    """Create a random EndEffectorPose with valid rotation."""
    translation = np.random.randn(3)
    rotation = Rotation.random()
    return EndEffectorPose(
        translation=translation,
        rotation=rotation.as_quat(),
        rotation_type="quat",
        rotation_order="xyzw",
    )


# ---------------------------------------------------------------------------
# TestEndEffectorPoseFromActionFormat
# ---------------------------------------------------------------------------


class TestEndEffectorPoseFromActionFormat:
    """Test EndEffectorPose.from_action_format() for all three ActionFormat variants."""

    def test_xyz_rot6d(self):
        original = _random_pose()
        flat = original.xyz_rot6d  # shape (9,)
        reconstructed = EndEffectorPose.from_action_format(flat, ActionFormat.XYZ_ROT6D)

        np.testing.assert_allclose(reconstructed.translation, original.translation, atol=1e-6)
        np.testing.assert_allclose(
            reconstructed.rotation_matrix, original.rotation_matrix, atol=1e-6
        )

    def test_xyz_rotvec(self):
        original = _random_pose()
        flat = original.xyz_rotvec  # shape (6,)
        reconstructed = EndEffectorPose.from_action_format(flat, ActionFormat.XYZ_ROTVEC)

        np.testing.assert_allclose(reconstructed.translation, original.translation, atol=1e-6)
        np.testing.assert_allclose(
            reconstructed.rotation_matrix, original.rotation_matrix, atol=1e-6
        )

    def test_default_homogeneous(self):
        original = _random_pose()
        flat = original.homogeneous.flatten()  # shape (16,)
        reconstructed = EndEffectorPose.from_action_format(flat, ActionFormat.DEFAULT)

        np.testing.assert_allclose(reconstructed.translation, original.translation, atol=1e-6)
        np.testing.assert_allclose(
            reconstructed.rotation_matrix, original.rotation_matrix, atol=1e-6
        )

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported ActionFormat"):
            EndEffectorPose.from_action_format(np.zeros(9), "not_a_format")


# ---------------------------------------------------------------------------
# TestEndEffectorActionChunkFromArray
# ---------------------------------------------------------------------------


class TestEndEffectorActionChunkFromArray:
    """Test EndEffectorActionChunk.from_array() for all three ActionFormat variants."""

    @pytest.mark.parametrize(
        "action_format, dim",
        [
            (ActionFormat.XYZ_ROT6D, 9),
            (ActionFormat.XYZ_ROTVEC, 6),
        ],
    )
    def test_from_array_creates_correct_length(self, action_format, dim):
        n_poses = 5
        poses = [_random_pose() for _ in range(n_poses)]
        chunk = EndEffectorActionChunk(poses)
        array = chunk.to(action_format)  # shape (n_poses, dim)

        assert array.shape == (n_poses, dim)

        reconstructed = EndEffectorActionChunk.from_array(array, action_format)
        assert len(reconstructed) == n_poses

    def test_from_array_default_format(self):
        n_poses = 3
        poses = [_random_pose() for _ in range(n_poses)]
        chunk = EndEffectorActionChunk(poses)
        array = chunk.to(ActionFormat.DEFAULT)  # shape (n_poses, 4, 4)

        # Flatten each 4x4 matrix for from_array (expects 2-D input)
        flat_array = array.reshape(n_poses, 16)
        reconstructed = EndEffectorActionChunk.from_array(flat_array, ActionFormat.DEFAULT)
        assert len(reconstructed) == n_poses

    @pytest.mark.parametrize(
        "action_format, dim",
        [
            (ActionFormat.XYZ_ROT6D, 9),
            (ActionFormat.XYZ_ROTVEC, 6),
        ],
    )
    def test_from_array_roundtrip(self, action_format, dim):
        """to() -> from_array() -> to() should reproduce the same array."""
        poses = [_random_pose() for _ in range(4)]
        original_array = EndEffectorActionChunk(poses).to(action_format)

        reconstructed_array = EndEffectorActionChunk.from_array(original_array, action_format).to(
            action_format
        )

        np.testing.assert_allclose(reconstructed_array, original_array, atol=1e-5)


# ---------------------------------------------------------------------------
# TestEefRoundtrip
# ---------------------------------------------------------------------------


class TestEefRoundtrip:
    """Absolute -> relative -> absolute roundtrip tests for EEF actions."""

    @pytest.mark.parametrize(
        "action_format",
        [ActionFormat.XYZ_ROT6D, ActionFormat.XYZ_ROTVEC],
    )
    def test_absolute_relative_absolute_roundtrip(self, action_format):
        """Convert to relative and back; the result should match the original."""
        reference = _random_pose()
        absolute_poses = [_random_pose() for _ in range(5)]
        absolute_chunk = EndEffectorActionChunk(absolute_poses)

        # absolute -> relative
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)

        # relative -> absolute
        recovered_chunk = relative_chunk.to_absolute_chunking(reference_frame=reference)

        # Compare in the chosen action_format representation
        original_array = absolute_chunk.to(action_format)
        recovered_array = recovered_chunk.to(action_format)

        np.testing.assert_allclose(recovered_array, original_array, atol=1e-5)

    @pytest.mark.parametrize(
        "action_format",
        [ActionFormat.XYZ_ROT6D, ActionFormat.XYZ_ROTVEC],
    )
    def test_roundtrip_via_flat_arrays(self, action_format):
        """Full pipeline: array -> from_array -> relative -> to -> from_array -> absolute -> to."""
        reference = _random_pose()
        absolute_poses = [_random_pose() for _ in range(4)]
        original_array = EndEffectorActionChunk(absolute_poses).to(action_format)
        ref_flat = (
            reference.xyz_rot6d if action_format == ActionFormat.XYZ_ROT6D else reference.xyz_rotvec
        )

        # Reconstruct from flat arrays, compute relative, serialize
        abs_chunk = EndEffectorActionChunk.from_array(original_array, action_format)
        ref_pose = EndEffectorPose.from_action_format(ref_flat, action_format)
        rel_chunk = abs_chunk.relative_chunking(reference_frame=ref_pose)
        rel_array = rel_chunk.to(action_format)

        # Reconstruct relative from flat, convert back to absolute, serialize
        rel_chunk2 = EndEffectorActionChunk.from_array(rel_array, action_format)
        ref_pose2 = EndEffectorPose.from_action_format(ref_flat, action_format)
        recovered_chunk = rel_chunk2.to_absolute_chunking(reference_frame=ref_pose2)
        recovered_array = recovered_chunk.to(action_format)

        np.testing.assert_allclose(recovered_array, original_array, atol=1e-5)
