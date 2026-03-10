"""Tests for EmbodimentTag enum consistency with N1.7 checkpoint.

Ensures that:
- All pretrain/posttrain tags (except NEW_EMBODIMENT) have matching
  entries in the N1.7 MODALITY_CONFIGS and EMBODIMENT_TAG_TO_PROJECTOR_INDEX.
- Removed N1.6 tags (OXE_GOOGLE, OXE_WIDOWX, LIBERO_PANDA) are no longer
  in the enum or configs.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
import pytest


class TestRemovedN16Tags:
    """Verify that deprecated N1.6-only tags are fully removed."""

    @pytest.mark.parametrize("tag_name", ["OXE_GOOGLE", "OXE_WIDOWX", "LIBERO_PANDA"])
    def test_removed_from_enum(self, tag_name):
        assert not hasattr(EmbodimentTag, tag_name), (
            f"EmbodimentTag.{tag_name} should be removed (not in N1.7 checkpoint)"
        )

    @pytest.mark.parametrize("tag_value", ["oxe_google", "oxe_widowx", "libero_panda"])
    def test_removed_from_modality_configs(self, tag_value):
        assert tag_value not in MODALITY_CONFIGS, (
            f"MODALITY_CONFIGS['{tag_value}'] should be removed (not in N1.7 checkpoint)"
        )

    @pytest.mark.parametrize("tag_value", ["oxe_google", "oxe_widowx", "libero_panda"])
    def test_removed_from_projector_index(self, tag_value):
        assert tag_value not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX, (
            f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX['{tag_value}'] should be removed"
        )


class TestEmbodimentTagConsistency:
    """Verify that all EmbodimentTag enum values have matching configs."""

    # Tags that exist in the pretrained N1.7 checkpoint
    N17_PRETRAINED_TAGS = {
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
        EmbodimentTag.GR1,
        EmbodimentTag.XDOF,
        EmbodimentTag.AGIBOT,
        EmbodimentTag.UNITREE_G1,
        EmbodimentTag.SIMPLER_ENV_GOOGLE,
        EmbodimentTag.SIMPLER_ENV_WIDOWX,
        EmbodimentTag.OXE_DROID,
        EmbodimentTag.BEHAVIOR_R1_PRO,
    }

    def test_pretrained_tags_in_projector_index(self):
        """Every pretrained tag must have a projector index mapping."""
        for tag in self.N17_PRETRAINED_TAGS:
            assert tag.value in EMBODIMENT_TAG_TO_PROJECTOR_INDEX, (
                f"EmbodimentTag.{tag.name} ('{tag.value}') missing from "
                f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX"
            )

    def test_new_embodiment_in_projector_index(self):
        """NEW_EMBODIMENT is for fine-tuning and must also have a projector index."""
        assert EmbodimentTag.NEW_EMBODIMENT.value in EMBODIMENT_TAG_TO_PROJECTOR_INDEX

    def test_no_extra_projector_entries(self):
        """EMBODIMENT_TAG_TO_PROJECTOR_INDEX should not have orphan keys."""
        all_tag_values = {tag.value for tag in EmbodimentTag}
        for key in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            assert key in all_tag_values, (
                f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX has orphan key '{key}' "
                f"with no matching EmbodimentTag"
            )

    def test_no_extra_modality_config_entries(self):
        """MODALITY_CONFIGS should not have orphan keys without a matching EmbodimentTag."""
        all_tag_values = {tag.value for tag in EmbodimentTag}
        for key in MODALITY_CONFIGS:
            assert key in all_tag_values, (
                f"MODALITY_CONFIGS has orphan key '{key}' with no matching EmbodimentTag"
            )

    def test_posttrain_tags_in_modality_configs(self):
        """Pre-registered posttrain tags must have a MODALITY_CONFIGS entry."""
        # Pretrain tags get their modality config from the model checkpoint,
        # only posttrain tags need an explicit entry in MODALITY_CONFIGS.
        pretrain_tags = {
            EmbodimentTag.ROBOCASA_PANDA_OMRON,
            EmbodimentTag.GR1,
            EmbodimentTag.XDOF,
            EmbodimentTag.AGIBOT,
            EmbodimentTag.SIMPLER_ENV_GOOGLE,
            EmbodimentTag.SIMPLER_ENV_WIDOWX,
        }
        skip_tags = pretrain_tags | {EmbodimentTag.NEW_EMBODIMENT}
        for tag in EmbodimentTag:
            if tag in skip_tags:
                continue
            assert tag.value in MODALITY_CONFIGS, (
                f"EmbodimentTag.{tag.name} ('{tag.value}') is a posttrain tag "
                f"but missing from MODALITY_CONFIGS"
            )
