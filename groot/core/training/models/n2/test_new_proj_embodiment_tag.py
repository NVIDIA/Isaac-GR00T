"""The dedicated-projector G1 tag resolves everywhere inference looks it up."""

from groot.core.data.schema import EmbodimentTag
from groot.core.training.configs.data.embodiment_configs import MODALITY_CONFIGS
from groot.core.training.models.n2.processor import EMBODIMENT_TAG_TO_PROJECTOR_INDEX

BASE = "unitree_g1_whole_body_teleop_latent_gripper"
NEW_PROJ = "unitree_g1_whole_body_teleop_latent_gripper_new_proj"


def test_new_proj_tag_is_registered_with_its_own_projector_row():
    assert (
        EmbodimentTag(NEW_PROJ)
        is EmbodimentTag.UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER_NEW_PROJ
    )
    assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[NEW_PROJ] == 7
    assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[BASE] != EMBODIMENT_TAG_TO_PROJECTOR_INDEX[NEW_PROJ]


def test_new_proj_tag_shares_the_base_data_layout():
    assert MODALITY_CONFIGS[NEW_PROJ] == MODALITY_CONFIGS[BASE]
    assert MODALITY_CONFIGS[NEW_PROJ] is not MODALITY_CONFIGS[BASE]
