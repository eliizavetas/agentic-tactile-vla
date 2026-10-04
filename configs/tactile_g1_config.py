# NEW_EMBODIMENT modality config for the Unitree G1 + BrainCo Revo2 (6-DoF hand)
# Sonic-teleop dataset.
#
# Structure mirrors the stock "unitree_g1_sonic" posttrain config from newer
# Isaac-GR00T snapshots: the policy predicts the 64-dim WBC motion token
# (decoded to joint targets by the onboard WBC controller) plus direct
# per-finger hand commands. It does NOT predict action.wbc — those are the
# WBC decoder's own low-level joint outputs, not a valid policy interface.
#
# Not predicted: teleop.delta_heading and teleop.*_wrist_joints (identically
# zero in this dataset) and teleop.planner_* locomotion commands (base motion
# stays planner/operator-driven at deployment).
#
# After any change to the action delta_indices, regenerate dataset statistics:
#   python gr00t/data/stats.py <dataset_path> new_embodiment

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


tactile_g1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",  # 6-DoF BrainCo position feedback (thumb, thumb_aux, index, middle, ring, pinky)
            "right_hand",
            "projected_gravity",
        ],
    ),
    "action": ModalityConfig(
        # 50 steps @ 50 Hz = 1.0 s action horizon (stock unitree_g1 uses 30)
        delta_indices=list(range(50)),
        modality_keys=[
            "motion_token",
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            # motion_token: 64-dim continuous WBC latent; deltas are meaningless
            # in the token space, so ABSOLUTE
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand_joints: 6-DoF BrainCo command, normalized 0-1 per motor;
            # ABSOLUTE so grasp aperture cannot drift
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand_joints
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Registers under EmbodimentTag.NEW_EMBODIMENT ("new_embodiment")
register_modality_config(tactile_g1_config)
