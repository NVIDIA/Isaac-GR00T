# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from enum import Enum


class EmbodimentTag(Enum):
    REAL_GR1_ARMS_ONLY = "real_gr1_arms_only"
    """
    The real GR1 robot embodiment with arms only.
    """

    REAL_GR1_ARMS_ONLY_ANNOTATED = "real_gr1_arms_only_annotated"
    """
    The real GR1 robot embodiment with arms only with annotations.
    """

    REAL_GR1_ARMS_WAIST = "real_gr1_arms_waist"
    """
    The real GR1 robot embodiment with arms and waist.
    """

    REAL_GR1_ARMS_WAIST_ANNOTATED = "real_gr1_arms_waist_annotated"
    """
    The real GR1 robot embodiment with arms and waist with annotations.
    """

    DEXMG_GR1_ARMS_ONLY_INSPIRE = "dexmg_gr1_arms_only_inspire"
    """
    The DEXMG GR1 dataset with arms only and inspire hand.
    """

    DEXMG_GR1_ARMS_ONLY_FOURIER = "dexmg_gr1_arms_only_fourier"
    """
    The DEXMG GR1 dataset with arms only and Fourier hand.
    """

    DEXMG_GR1_ARMS_WAIST_FOURIER = "dexmg_gr1_arms_waist_fourier"
    """
    The DEXMG GR1 dataset with arms and waist and Fourier hand.
    """

    ROBOCASA_SINGLE_ARM = "robocasa_single_arm"
    """
    The RoboCasa dataset with single arm.
    """

    ONE_X_EVE_GRIPPER = "onex_eve_gripper"
    """
    The OneX Eve Robot with gripper.
    """

    ROBOCASA_GR1_ARMS_ONLY_INSPIRE_HANDS = "robocasa_gr1_arms_only_inspire_hands"
    """
    The RoboCasa GR1 dataset with arms only and inspire hands.
    """

    ROBOCASA_GR1_ARMS_ONLY_FOURIER_HANDS = "robocasa_gr1_arms_only_fourier_hands"
    """
    The RoboCasa GR1 dataset with arms only and Fourier hands.
    """

    ROBOCASA_GR1_FIXED_LOWER_BODY_INSPIRE_HANDS = "robocasa_gr1_fixed_lower_body_inspire_hands"
    """
    The RoboCasa GR1 dataset with fixed lower body and inspire hands.
    """

    ROBOCASA_GR1_FIXED_LOWER_BODY_FOURIER_HANDS = "robocasa_gr1_fixed_lower_body_fourier_hands"
    """
    The RoboCasa GR1 dataset with fixed lower body and Fourier hands.
    """

    ROBOCASA_GR1_ARMS_WAIST_FOURIER_HANDS = "robocasa_gr1_arms_waist_fourier_hands"
    """
    The RoboCasa GR1 dataset with arms and waist and Fourier hands.
    """

    ROBOCASA365_PANDA_OMRON = "robocasa365_panda_omron"

    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa dataset with panda omron.
    """

    ROBOCASA_SINGLE_ARM_PANDA_OMRON = "robocasa_single_arm_panda_omron"
    """
    The RoboCasa dataset with single arm panda omron.
    """

    ROBOCASA_BIMANUAL_PANDA_PARALLEL_GRIPPER = "robocasa_bimanual_panda_parallel_gripper"
    """
    The dexmg bimanual panda dataset with parallel grippers.
    """

    ROBOCASA_BIMANUAL_PANDA_INSPIRE_HAND = "robocasa_bimanual_panda_inspire_hand"
    """
    The DEXMG bimanual panda dataset with inspire hands.
    """

    OXE_DROID = "oxe_droid"
    """
    The Open X-Embodiment droid dataset.
    """
    OXE_DROID_JOINT_POSITION_RELATIVE = "oxe_droid_joint_position_relative"
    """
    The Open X-Embodiment droid dataset.
    """

    OXE_DROID_RELATIVE_EEF = "oxe_droid_relative_eef"
    """
    The DROID dataset with relative EEF (xyz+rot6d) actions.
    """

    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    """
    The DROID dataset with relative EEF (xyz+rot6d) and relative joint actions.
    """

    OXE_DROID_DREAM = "oxe_droid_dream"
    """
    The Open X-Embodiment droid dataset for DreamZero.
    """

    OXE_FRACTAL = "oxe_fractal"
    """
    The Open X-Embodiment fractal (RT-1) dataset.
    """

    OXE_LANGUAGE_TABLE = "oxe_language_table"
    """
    The Open X-Embodiment language table dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The Open X-Embodiment bridge dataset.
    """

    BRIDGE_RELATIVE_EEF = "bridge_relative_eef"
    """
    The Bridge dataset (via LeRobot) with relative EEF (xyz+rot6d) actions.
    """

    FRACTAL_RELATIVE_EEF = "fractal_relative_eef"
    """
    The Fractal dataset (via LeRobot) with relative EEF (xyz+rot6d) actions.
    """

    OXE_MUTEX = "oxe_mutex"
    """
    The Open X-Embodiment mutex dataset.
    """

    OXE_ROBOSET = "oxe_roboset"
    """
    The Open X-Embodiment Roboset dataset.
    """

    OXE_PLEX = "oxe_plex"
    """
    The Open X-Embodiment Plex RoboSuite dataset.
    """

    REAL_PANDA_SINGLE_ARM = "real_panda_single_arm"
    """
    The real single arm panda robot.
    """

    HOT3D_HANDS_ONLY = "hot3d_hands_only"
    """
    The HOT3D dataset with hands only.
    """

    GR1_UNIFIED = "gr1_unified"
    """
    The GR1 unified dataset.
    """

    GR1_UNIFIED_SEQ = "gr1_unified_seq"
    """
    The GR1 unified dataset.
    """

    GR1_ISAAC = "gr1_isaac"
    """
    The GR1 Isaac dataset (Shiwei).
    """

    LAPA = "lapa"
    """
    The datasets with LAPA actions.
    """
    AGIBOT = "agibot"
    """
    The Agibot Genie-1 robot.
    """

    AGIBOT_DREAM = "agibot_dream"
    """
    The Agibot Genie-1 robot for DreamZero.
    """

    XDOF_DREAM = "xdof_dream"
    """
    The XDOF robot for DreamZero.
    """

    DREAM = "dream"
    """
    The datasets with DREAM actions.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    The new embodiment.
    """

    MECKA_HANDS = "mecka_hands"
    """
    The MECKA hands embodiment (e.g. mecka_lerobot datasets).
    """

    MECKA_RELATIVE_EEF_ACTIVE_HEAD = "mecka_relative_eef_active_head"
    """
    The MECKA robot with active head and relative EEF actions.
    """

    UNKNOWN = "unknown"

    GR1_UNIFIED_SEGMENTATION = "gr1_unified_segmentation"
    """
    The GR1 unified dataset with segmentation.
    """

    LANGUAGE_TABLE_SIM = "language_table_sim"
    """
    Simulated Language Table.
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    SimplerEnv Google.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    SimplerEnv Widowx.
    """

    LIBERO_SIM = "libero_sim"
    """
    The Libero Sim dataset.
    """

    VLABENCH_SIM = "vlabench_sim"
    """The VLABench simulation dataset."""

    DROID_SIM = "droid_sim"
    """
    The Droid dataset in sim.
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD = (
        "unitree_g1_full_body_with_waist_height_nav_cmd"
    )
    """
    The Unitree G1 dataset with full body and waist and height command and navigate command.
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD_SIM = (
        "unitree_g1_full_body_with_waist_height_nav_cmd_sim"
    )
    """
    The Unitree G1 dataset with full body and waist and height command and navigate command in sim
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD_WITH_WRIST_CAMS = (
        "unitree_g1_full_body_with_waist_height_nav_cmd_with_wrist_cams"
    )
    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT = "unitree_g1_whole_body_teleop_latent"
    """
    The Unitree G1 dataset with whole body teleop and latent actions.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_RELATIVE_EEF = (
        "unitree_g1_whole_body_teleop_latent_relative_eef"
    )
    """
    G1 latent actions with relative EEF wrist pose (rot6d) and hand joints.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_RELATIVE_EEF_NEW_PROJ = (
        "unitree_g1_whole_body_teleop_latent_relative_eef_new_proj"
    )
    """
    G1 latent actions with relative EEF wrist pose (rot6d), using a dedicated projector.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_NO_HANDS = "unitree_g1_whole_body_teleop_latent_no_hands"
    """
    The Unitree G1 dataset with whole body teleop and latent actions, without hand joints proprio.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER = "unitree_g1_whole_body_teleop_latent_gripper"
    """
    G1 latent actions with YAM gripper width actions and wrist cameras.
    Legacy compatibility only for N2; deprecated for new N2 runs. Use NEW_EMB
    by default or WRIST for pretrained G1 continuation. Keep this exact value
    for historical configs/checkpoints; it is not an alias for either new tag.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER_NEW_EMB = (
        "unitree_g1_whole_body_teleop_latent_gripper_new_emb"
    )
    """G1 SONIC gripper posttraining with a dedicated N2 projector row (15)."""

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER_WRIST = (
        "unitree_g1_whole_body_teleop_latent_gripper_wrist"
    )
    """G1 SONIC gripper continuation using the pretrained checkpoint's projector row."""

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER_NO_HAND_STATE = (
        "unitree_g1_whole_body_teleop_latent_gripper_no_hand_state"
    )
    """
    Same as UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER but without hand joints in state.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_SHARED_PROJ = (
        "unitree_g1_whole_body_teleop_latent_shared_proj"
    )
    """
    Same data layout as UNITREE_G1_WHOLE_BODY_TELEOP_LATENT, sharing projector 25.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER_SHARED_PROJ = (
        "unitree_g1_whole_body_teleop_latent_gripper_shared_proj"
    )
    """
    Same data layout as UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_GRIPPER, sharing projector 25.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_RELATIVE_EEF_GRIPPER = (
        "unitree_g1_whole_body_teleop_latent_relative_eef_gripper"
    )
    """
    G1 latent actions with relative EEF wrist pose (rot6d) and YAM gripper width actions.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_LATENT_RELATIVE_EEF_GRIPPER_NEW_PROJ = (
        "unitree_g1_whole_body_teleop_latent_relative_eef_gripper_new_proj"
    )
    """
    G1 latent actions with relative EEF and gripper width actions, using a dedicated projector.
    """

    UNITREE_G1_WHOLE_BODY_TELEOP_SMPL = "unitree_g1_whole_body_teleop_smpl"
    """
    The Unitree G1 dataset with whole body teleop and SMPL actions.
    """

    UNITREE_G1_FULL_BODY_WITH_WAIST_HEIGHT_NAV_CMD_AND_TASK_PROG = (
        "unitree_g1_full_body_with_waist_height_nav_cmd_and_task_prog"
    )
    """
    The Unitree G1 dataset with full body and waist and height command and navigate command and task progress.
    """

    G1_FIX_LOWER_RIGHT_HAND = "g1_fix_lower_right_hand"
    """
    The G1 robot with fixed lower body and right hand.
    """

    GR1_UNIFIED_OFFLINE_RL = "gr1_unified_offline_rl"
    """
    The GR1 robot with reward information for oflfine RL.
    """

    XDOF = "xdof"
    """
    The XDOF robot.
    """

    ARX_X5 = "arx_x5"
    """
    The bimanual ARX X5 robot used by RoboDojo.
    """

    XDOF_H16 = "xdof_h16"
    """
    The XDOF robot with action horizon 16.
    """

    XDOF_OSS_DATA = "xdof_oss_data"
    """
    The XDOF data with conversions via jimmywu's processing pipeline.
    """

    XDOF_RELATIVE_EEF = "xdof_relative_eef"
    """
    The XDOF robot with relative EEF actions only (no joint position prediction).
    """

    XDOF_RELATIVE_EEF_RELATIVE_JOINT = "xdof_relative_eef_relative_joint"
    """
    The XDOF robot with relative EEF actions and relative joint actions.
    """

    XDOF_RELATIVE_EEF_RELATIVE_JOINT_SUBTASK = "xdof_relative_eef_relative_joint_subtask"
    """
    The XDOF robot with relative EEF actions and relative joint actions and subtask.
    """

    XDOF_RELATIVE_EEF_RELATIVE_JOINT_RECAP_LONG = "xdof_relative_eef_relative_joint_recap_long"
    """
    The XDOF robot with relative EEF / joint actions; language uses the long
    vllm recaptions: annotation.task_recaption_long (per-episode) and
    annotation.sub_task_recaption (per-frame, first-wins).
    """

    XDOF_RELATIVE_EEF_RELATIVE_JOINT_UMI_KEYS = "xdof_relative_eef_relative_joint_umi_keys"
    """
    Same as XDOF_RELATIVE_EEF_RELATIVE_JOINT but uses the umi-style
    left_arm.eef / right_arm.eef keys instead of left_wrist_eef /
    right_wrist_eef, to align with other embodiments.
    """

    XDOF_RELATIVE_EEF_RELATIVE_JOINT_UMI_KEYS_4VIEWS = (
        "xdof_relative_eef_relative_joint_umi_keys_4views"
    )
    """
    Same as XDOF_RELATIVE_EEF_RELATIVE_JOINT_UMI_KEYS but with the additional
    bottom_camera view (4 cameras total: top, left, right, bottom).
    """

    XDOF_RELATIVE_JOINT = "xdof_relative_joint"
    """
    The XDOF robot with relative joint actions only (no EEF action prediction).
    """

    SIM_BEHAVIOR_R1_PRO = "sim_behavior_r1_pro"
    """
    The sim BEHAVIOR Galaxea R1 Pro robot with grippers.
    """

    # ------------- Deprecated G1 Embodiments -------------
    # below are deprecated G1 embodiments. Why deprecated?
    # 1. The neck wasn't locked. So the the camera angles differe slightly between episodes and robots
    # 2. Video modality was called `rs_view`. It is now called `ego_view`
    # TODO: In future if we want to unify these embodiment tags, we can do so by renaming  # noqa: TD002, TD003
    # video.rs_view -> video.ego_view. After that we can remove the below deprecated embodiment tags.

    GR1_UNIFIED_512 = "gr1_unified_512"
    """
    The GR1 unified dataset with 512*320 resolution without cropping.
    """

    R1_PRO = "r1_pro"
    """
    Real Galaxea R1 Pro.
    """

    R1_PRO_SINGLE_VIEW = "r1_pro_single-view"
    """
    Real Galaxea R1 Pro with single camera view.
    """

    MOZ1 = "moz1"
    """
    Spirit AI Moz1.
    """

    SO100_2RGB = "so100_2rgb"
    """
    The SO100 robot embodiment with 2 RGB cameras.
    """
    SO100_3RGB = "so100_3rgb"
    """
    The SO100 robot embodiment with 3 RGB cameras.
    """
    ROBOMIND_AGILEX_3RGB = "robomind_agilex_3rgb"
    """
    The RoboMIND AgileX 3RGB robot embodiment.
    """

    ROBOMIND_FRANKA_3RGB = "robomind_franka_3rgb"
    """
    The RoboMIND Franka 3RGB robot embodiment.
    """
    ROBOMIND_FRANKA_1RGB = "robomind_franka_1rgb"
    """
    The RoboMIND Franka 1RGB robot embodiment.
    """

    ROBOMIND_TIENKUNG_GELLO_1RGB = "robomind_tienkung_gello_1rgb"
    """
    The RoboMIND Tienkung Gello 1RGB robot embodiment.
    """

    ROBOMIND_UR_1RGB = "robomind_ur_1rgb"
    """
    The RoboMIND UR 1RGB robot embodiment.
    """
    ROBOMIND_FRANKA_FR3_DUAL = "robomind_franka_dual_4rgb"
    """
    The RoboMIND Franka 4RGB robot embodiment.
    """
    ROBOMIND_SIM_FRANKA_4RGB = "robomind_franka_sim_4rgb"
    """
    The RoboMIND Franka 4RGB robot embodiment in sim.
    It contains actually 4 camera streams althgouth '3rgb' appears in the path.
    """
    ROBOMIND_SIM_TIENKUNG_1RGB = "robomind_sim_tienkung_1rgb"
    """
    The RoboMIND Tienkung 1RGB robot embodiment in sim.
    """
    ROBOMIND_TIENKUNG_PROD1_GELLO_1RGB = "robomind_tienkung_prod1_gello_1rgb"
    """
    The RoboMIND Tienkung Prod1 Gello 1RGB robot embodiment.
    """
    ROBOMIND_TIENKUNG_XENS_1RGB = "robomind_tienkung_xsens_1rgb"
    """
    The RoboMIND Tienkung Xens 1RGB robot embodiment.
    """
    MOLMOACT_FRANKA_3RGB = "molmoact_franka_3rgb"
    """
    The MolmoAct Franka 3RGB robot embodiment.
    """
    # ------------- R1 Pro Sharpa Embodiments -------------
    REAL_R1_PRO_SHARPA_ADD_VIEW = "real_r1_pro_sharpa_add_view"
    """
    The real R1 Pro Sharpa robot embodiment with add view.
    """
    REAL_R1_PRO_SHARPA = "real_r1_pro_sharpa"
    """
    The real Galaxea R1 Pro robot with sharpa hands.
    """
    REAL_R1_PRO_SHARPA_DELTA_EEF = "real_r1_pro_sharpa_delta_eef"
    """
    The real Galaxea R1 Pro robot with sharpa hands and delta eef.
    """
    REAL_R1_PRO_SHARPA_ABSOLUTE_EEF = "real_r1_pro_sharpa_absolute_eef"
    """
    The real Galaxea R1 Pro robot with sharpa hands and absolute eef.
    """
    REAL_R1_PRO_SHARPA_MEANSTD = "real_r1_pro_sharpa_meanstd"
    """
    The real Galaxea R1 Pro robot with sharpa hands and meanstd.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_ARM_JOINT = "real_r1_pro_sharpa_relative_arm_joint"
    """
    The real Galaxea R1 Pro robot with relative arm joint.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF = "real_r1_pro_sharpa_relative_eef"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_ADD_VIEW = "real_r1_pro_sharpa_relative_eef_add_view"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and add view.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_RELATIVE_HAND = "real_r1_pro_sharpa_relative_eef_relative_hand"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and relative hand.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_RELATIVE_HAND_RELATIVE_JOINT = (
        "real_r1_pro_sharpa_relative_eef_relative_hand_relative_joint"
    )
    """
    The real Galaxea R1 Pro robot with relative eef, relative hand and relative joint.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_HUMAN = "real_r1_pro_sharpa_relative_eef_human"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef
    (human demonstration data with separate normalization).
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_HUMAN_ADD_VIEW = (
        "real_r1_pro_sharpa_relative_eef_human_add_view"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and add view (human demonstration data).
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_HUMAN_RELATIVE_HAND = (
        "real_r1_pro_sharpa_relative_eef_human_relative_hand"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and relative hand (human demonstration data).
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_EGODEX = "real_r1_pro_sharpa_relative_eef_egodex"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and egodex.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_EGODEX_RELATIVE_HAND = (
        "real_r1_pro_sharpa_relative_eef_egodex_relative_hand"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and egodex relative hand.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_EGODEX_WRIST_ONLY = (
        "real_r1_pro_sharpa_relative_eef_egodex_wrist_only"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and egodex wrist only.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_MAXINSIGHTS = "real_r1_pro_sharpa_relative_eef_maxinsights"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and maxinsights.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_MAXINSIGHTS_RELATIVE_HAND = (
        "real_r1_pro_sharpa_relative_eef_maxinsights_relative_hand"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and maxinsights relative hand.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_MECKA = "real_r1_pro_sharpa_relative_eef_mecka"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and mecka.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_EEF_MECKA_RELATIVE_HAND = (
        "real_r1_pro_sharpa_relative_eef_mecka_relative_hand"
    )
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative eef and mecka relative hand.
    """
    REAL_R1_PRO_SHARPA_RELATIVE_JOINT = "real_r1_pro_sharpa_relative_joint"
    """
    The real Galaxea R1 Pro robot with sharpa hands and relative joint.
    """

    # ------------- Dexmate Embodiments -------------
    REAL_DEXMATE_SHARPA_RELATIVE_EEF = "real_dexmate_sharpa_relative_eef"
    """
    Dexmate Sharpa with relative EEF, normal (non-cropped) ego view.
    """

    REAL_DEXMATE_SHARPA_RELATIVE_EEF_CROPPED = "real_dexmate_sharpa_relative_eef_cropped"
    """
    Dexmate Sharpa with relative EEF, cropped views. Uses its own projector ID.
    """

    REAL_DEXMATE_SHARPA_RELATIVE_EEF_NEW_CONTROLLER = (
        "real_dexmate_sharpa_relative_eef_new_controller"
    )
    """
    Dexmate Sharpa with relative EEF, new controller. Action distribution differs
    substantially from the old controller, so this uses its own projector ID.
    """

    REAL_DEXMATE_SHARPA_RELATIVE_EEF_RELATIVE_JOINT = (
        "real_dexmate_sharpa_relative_eef_relative_joint"
    )
    """
    Dexmate Sharpa with relative EEF + relative arm joints. Hand (finger)
    joints stay absolute. Mirrors xdof_relative_eef_relative_joint.
    """

    REAL_DEXMATE_SHARPA_RELATIVE_EEF_WITH_TACTILE = "real_dexmate_sharpa_relative_eef_with_tactile"
    """
    Dexmate Sharpa with relative EEF and tactile_f6 sensor readings in state.
    10 fingers × 6-DoF readings (60 dims) appended to the standard state.
    Use only with datasets collected with ``--require-tactile-full``.
    """

    # ------------- Tianji Embodiments -------------
    REAL_TIANJI_SHARPA_RELATIVE_EEF = "real_tianji_sharpa_relative_eef"
    """
    Tianji M6 + Sharpa hands with relative EEF, stereo ego pair + two wrist views.
    62D state/action (2x wrist EEF xyz+rot6d + 2x 22-DoF Sharpa hands). Shares the
    Sharpa projector with the dexmate/r1_pro family.
    """

    REAL_TIANJI_SHARPA_RELATIVE_EEF_WITH_TACTILE = "real_tianji_sharpa_relative_eef_with_tactile"
    """
    Tianji Sharpa with relative EEF and tactile_f6 sensor readings in state:
    10 fingers x 6-DoF readings (60 dims) appended to the standard state.

    This is the tag the CONVERTER writes into ``meta/info.json`` for every tianji
    dataset (the `_with_tactile` suffix is what enables the F6 + deform streams), so
    it must exist for conversion to run and for existing tianji checkpoints to be
    servable -- ``run_gr00t_server.py`` types ``--embodiment-tag`` as this enum.

    It DOES now have a MODALITY_CONFIGS entry, but only because ``generate_rel_stats``
    indexes MODALITY_CONFIGS with a bare ``[tag]`` and the merge step passes the
    dataset's own tag -- so every tianji merge died with KeyError at the last step,
    after writing the whole dataset. That entry is an ALIAS of the base tag: the ten
    fingertip F6 vectors are extra state COLUMNS on disk, not extra selected state.
    The vanilla configs still declare ``real_tianji_sharpa_relative_eef`` and select
    the 62D subset by named key from this same 136D data, which works because the
    loader binds modalities by NAME, never by index or by ``meta/info.json``
    ``robot_type`` (nothing in the training path reads that field at all).

    Training on the tactile block needs more than this alias: a separate ``tactile``
    modality group (the ten ``*_f6`` keys, delta_indices overridden per arm because
    the frozen Tactile-VAE consumes a 20-frame window) together with the loader
    subclass and processor seam that group requires. Do not approximate it by folding
    F6 into ``state``; that hands the state encoder 122D where it expects 62D.
    """

    # ------------- Real G1 Embodiments -------------
    REAL_G1_RELATIVE_EEF_ABSOLUTE_JOINTS = "real_g1_relative_eef_absolute_joints"
    """
    The real G1 robot with relative eef absolute joints.
    """
    REAL_G1_RELATIVE_EEF_ABSOLUTE_JOINTS_WRIST_CAM = (
        "real_g1_relative_eef_absolute_joints_wrist_cam"
    )
    """
    The real G1 robot with relative eef absolute joints and wrist camera.
    """
    REAL_G1_RELATIVE_EEF_RELATIVE_JOINTS = "real_g1_relative_eef_relative_joints"
    """
    The real G1 robot with relative eef relative joints.
    """

    GALAXEA_R1_4RGB = "galaxea_r1_4rgb"
    """
    The Galaxea R1 4RGB robot embodiment.
    """

    GRIPPER_UMI_4CAMERA = "gripper_umi_4camera"
    """
    Gripper UMI with 4 cameras (left/right UMI + left/right trigger).
    EEF + gripper distance actions, relative representation.
    """

    GRIPPER_UMI_2CAMERA = "gripper_umi_2camera"
    """
    Gripper UMI with 2 cameras (left/right UMI only).
    EEF + gripper distance actions, relative representation.
    """

    GRIPPER_UMI_2CAMERA_RECAPTION = "gripper_umi_2camera_recaption"
    """
    Gripper UMI with 2 cameras and recaptioned language annotations.
    """

    GRIPPER_UMI = "gripper_umi"
    """
    The gripper UMI embodiment (legacy, used by zarr→LeRobot conversion).
    """

    SINGLE_GRIPPER_UMI = "single_gripper_umi"
    """
    The single-arm gripper UMI embodiment (used by zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_CAMERA = "gripper_umi_trigger_camera"
    """
    The gripper UMI embodiment with left/right trigger cameras
    (used by zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_HEAD_CAMERA = "gripper_umi_trigger_head_camera"
    """
    The gripper UMI embodiment with trigger cameras + an additional head
    tracking camera (used by zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_HEAD_CAMERA_W_RELS = "gripper_umi_trigger_head_camera_w_rels"
    """
    Same as ``gripper_umi_trigger_head_camera`` but additionally emits the
    arm-to-arm relative pose ``right_arm_to_left_arm.eef`` derived from the
    two head-relative poses (used by zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_TWO_HEAD_CAMERA_W_RELS = "gripper_umi_trigger_two_head_camera_w_rels"
    """
    Same as ``gripper_umi_trigger_head_camera_w_rels`` but the head RGB
    stream is sourced from ``head_umi_camera`` instead of
    ``head_tracking_camera``. The rig has TWO head cameras: a UMI/fisheye
    ``head_umi_camera`` for the policy's RGB input, and a SLAM
    ``head_tracking_camera`` whose ``pose_interp`` is still used to compute
    ``head.eef`` and the head-relative arm poses.
    """

    GRIPPER_UMI_TRIGGER_TWO_ZED_CAMERA = "gripper_umi_trigger_two_zed_camera"
    """
    No-head bimanual gripper UMI with trigger cameras plus two table-mounted
    ZED cameras. Each ZED contributes an RGB stream and a lossless uint16
    depth stream (``{side}_zed_camera.depth``, byte-split libx264rgb). No head
    eef; emits ``right_arm_to_left_arm.eef``. The SLAM reference camera is a
    ZED, so relative poses use the general lift-and-rebase. Maps to
    ``GripperUMITriggerTwoZedCameraProcessor`` (zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_THREE_ZED_CAMERA = "gripper_umi_trigger_three_zed_camera"
    """
    Same as ``gripper_umi_trigger_two_zed_camera`` but with a third
    table-mounted ZED camera (``middle_zed_camera``). All three ZEDs
    (left/middle/right) contribute an RGB stream and a lossless uint16 depth
    stream (``{side}_zed_camera.depth``, byte-split libx264rgb). No head eef;
    emits ``right_arm_to_left_arm.eef``. Maps to
    ``GripperUMITriggerThreeZedCameraProcessor`` (zarr→LeRobot conversion).
    """

    GRIPPER_UMI_TRIGGER_FOUR_REALSENSE_CAMERA = "gripper_umi_trigger_four_realsense_camera"
    """
    No-head bimanual gripper UMI with trigger cameras plus four table-mounted
    RealSense cameras (``left`` / ``middle_left`` / ``middle_right`` / ``right``
    ``_realsense_camera``). Structurally identical to the ZED scene-camera tags
    but with RealSense scene cameras: each RealSense contributes an RGB stream
    and a lossless uint16 depth stream (``{cam}_realsense_camera.depth``,
    byte-split libx264rgb), and records per-camera ``intrinsics``. No head eef;
    emits ``right_arm_to_left_arm.eef`` via the general lift-and-rebase. Maps to
    ``GripperUMITriggerFourRealsenseCameraProcessor`` (zarr→LeRobot conversion).
    """

    SINGLE_DEXUMI = "single_dexumi"
    """
    The single right-hand DexUMI embodiment (exoskeleton joint angles + tactile,
    used by zarr→LeRobot conversion). Maps to ``SingleDexUMIProcessor``.
    """

    SINGLE_DEXUMI_WO_TAC = "single_dexumi_wo_tac"
    """
    The single right-hand DexUMI embodiment without tactile (exoskeleton joint
    angles only; tactile streams are dropped). Same EEF + hand joint pipeline
    as ``single_dexumi``. Maps to ``SingleDexUMIWoTacProcessor``.
    """

    GRIPPER_UMI_TRIGGER_TWO_HEAD_TWO_ZED_CAMERA_W_RELS = (
        "gripper_umi_trigger_two_head_two_zed_camera_w_rels"
    )
    """
    Same as ``gripper_umi_trigger_two_head_camera_w_rels`` but adds two ZED scene
    cameras (``left_zed_camera`` / ``right_zed_camera``) as additional RGB streams.
    The ZEDs carry no eef; pose computation is unchanged. Maps to
    ``GripperUMITriggerTwoHeadTwoZedCameraWithRelsProcessor``.
    """

    SINGLE_DEXUMI_TELEOP = "single_dexumi_teleop"
    """
    Single right-hand UR5e + Orange dexterous hand teleop embodiment (robot-FK eef,
    16 hand joints, 30-dim tactile). The action comes from separately recorded
    ur5e_action / right_hand_action streams. Maps to ``SingleDexUMITeleopProcessor``.
    """

    SINGLE_DEXUMI_TELEOP_WO_TAC = "single_dexumi_teleop_wo_tac"
    """
    Single right-hand UR5e + Orange dexterous hand teleop embodiment without tactile.
    The action comes from separately recorded ur5e_action / right_hand_action streams.
    Maps to ``DexUMITeleopWoTacProcessor``.
    """

    SINGLE_DEXUMI_TELEOP_WO_TAC_ABS_HAND = "single_dexumi_teleop_wo_tac_abs_hand"
    """
    Single right-hand UR5e + Orange dexterous hand teleop embodiment with absolute hand action.
    The action comes from separately recorded ur5e_action / right_hand_action streams.
    Maps to ``DexUMITeleopWoTacAbsTacProcessor``.
    """
