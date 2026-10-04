# Implements the paper specification. Not re-validated on hardware since it was written.
"""Every threshold of the execution supervisor in one place.

All values are the ones reported in the paper unless a comment says otherwise.
Detector thresholds were selected from 40 calibration closures on the five
training objects (eight per object), separate from all evaluation trials, and
were frozen before evaluation. Do not tune them on evaluation objects.

Positions are normalized hand-interface positions in [0, 1]. Motor effort is a
device-specific indicator of interaction resistance. It is not calibrated
force, and none of the thresholds below is expressed in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

# Hand channel order used by the policy, the driver and the paper.
CHANNEL_NAMES: Tuple[str, ...] = (
    "thumb_flexion",     # 0
    "thumb_opposition",  # 1
    "index",             # 2
    "middle",            # 3
    "ring",              # 4
    "little",            # 5
)
NUM_CHANNELS = 6
THUMB_OPPOSITION = 1
# Monitored flexion channels, F = {0, 2, 3, 4, 5}. Thumb opposition is never
# modified by the supervisor: it preserves the policy-selected grasp arrangement.
FLEXION_CHANNELS: Tuple[int, ...] = (0, 2, 3, 4, 5)


@dataclass(frozen=True)
class SupervisorConfig:
    """Parameters of the supervisor, detector and correction (paper Sec. III)."""

    # --- timing -----------------------------------------------------------
    control_period: float = 0.02   # finger references are issued at 50 Hz

    # --- closure detection from the action chunk (Sec. III-A) --------------
    arm_delta: float = 0.015       # any flexion target advances by this much ...
    arm_window: float = 0.25       # ... over this time [s] -> ARMED
    terminal_delta: float = 0.005  # every flexion target varies by less than this ...
    terminal_window: float = 0.12  # ... over this time [s] -> grasp-assessment point

    # --- selective intervention (eq. 3) and verdict (eq. 8) ----------------
    min_support: int = 3           # at least three support latches

    # --- proprioceptive contact detector (eqs. 4-7) ------------------------
    window: float = 0.25           # W, history window [s]
    delta_u: float = 0.015         # minimum commanded advance over W
    rho_s: float = 0.3             # position-stall tracking ratio
    rho_d: float = 0.7             # effort-duration tracking ratio
    delta_e: float = 0.3           # effort deviation from the ADJUST baseline (device units)
    t_e: float = 0.06              # T_e, required continuous effort-deviation time [s]

    # --- per-finger correction (Sec. III-D) --------------------------------
    step: float = 0.004            # reference increment ...
    step_period: float = 0.02      # ... every 20 ms
    preload: float = 0.02          # normalized preload added to the measured position
    min_adjust: float = 0.6        # ADJUST lasts at least this long [s]
    max_adjust: float = 4.0        # ... and at most this long [s]
    # Permitted travel limit per channel. The paper does not report numeric
    # limits. The flexion channels (0, 2, 3, 4, 5) use 0.98, taken from the
    # teleoperation closed-hand preset [0.98, 0.90, 0.98, 0.98, 0.98, 0.98].
    # Thumb opposition (channel 1) is never driven by the supervisor, so its
    # entry is unused and left at the end of the normalized range.
    travel_limit: Tuple[float, ...] = (0.98, 1.0, 0.98, 0.98, 0.98, 0.98)

    # --- re-entry blocking (Sec. III-D) ------------------------------------
    reopen_delta: float = 0.05     # decrease in mean flexion relative to terminal assessment
    reopen_hold: float = 0.12      # ... sustained for this long [s]

    # --- hand-state freshness (not in the paper; value of the deployed system)
    # Outside the compliant branch the supervisor does not act on a hand state
    # older than this [s]. The gate is bypassed in the compliant states, where
    # no DDS state arrives because the hand server is stopped.
    max_state_age: float = 0.5

    # --- ablation switches (Sec. IV-C); the defaults are the full supervisor
    selective: bool = True         # False: intervene at every closure (U-* variants)
    use_effort: bool = True        # False: position stall only (S-position, U-stall)
    verdict_release: bool = True   # False: always retain the correction (S-retain)
    use_verdict: bool = True       # False: close-to-stall baseline, no verdict (U-stall)

    def __post_init__(self) -> None:
        if len(self.travel_limit) != NUM_CHANNELS:
            raise ValueError("travel_limit needs one value per hand channel")
        if not all(0.0 < v <= 1.0 for v in self.travel_limit):
            raise ValueError("travel limits must lie in (0, 1]")
        if self.min_adjust > self.max_adjust:
            raise ValueError("min_adjust must not exceed max_adjust")


# Ablation variants of Sec. IV-C, expressed as switches on the same code path.
VARIANTS = {
    "S-full": dict(selective=True, use_effort=True, verdict_release=True, use_verdict=True),
    "S-position": dict(selective=True, use_effort=False, verdict_release=True, use_verdict=True),
    "S-retain": dict(selective=True, use_effort=True, verdict_release=False, use_verdict=True),
    "U-full": dict(selective=False, use_effort=True, verdict_release=True, use_verdict=True),
    "U-stall": dict(selective=False, use_effort=False, verdict_release=False, use_verdict=False),
}


def variant_config(name: str, base: SupervisorConfig | None = None) -> SupervisorConfig:
    """Return the configuration of one ablation variant."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return replace(base or SupervisorConfig(), **VARIANTS[name])


@dataclass(frozen=True)
class CompliantConfig:
    """Parameters of the current-limited compliant controller (eq. 9).

    K_p and K_d are controller-interface gains, not physical stiffness or
    damping. The command limit is a device-level actuator command, not
    calibrated motor current or fingertip force.
    """

    kp: float = 5.0
    kd: float = 0.05
    period: float = 0.02           # the controller runs every 20 ms
    i_max: float = 0.15            # normalized limit; raw 150 -> 0.15, raw 200 -> 0.20
    raw_scale: float = 1000.0      # raw device units per normalized unit
    # Smoothing factor of the exponentially smoothed finger velocity. The paper
    # states that the velocity is exponentially smoothed but gives no factor.
    velocity_alpha: float = 0.3
    # Predefined grasp configuration the fingers are driven toward. The paper
    # does not list it; set it for the grasp being evaluated.
    target: Tuple[float, ...] = (0.70, 0.70, 0.75, 0.75, 0.75, 0.75)

    # --- handover to the current-mode hand service -------------------------
    # None of the three values below is reported in the paper. They are the
    # values of the deployed handover sequence.
    preopen: float = 0.8           # hand commanded fully open for this long [s]
    # The compliant branch is entered only while the measured mean flexion is
    # below this level, i.e. before the position path has closed on the object.
    early_max_flexion: float = 0.45
    min_hold: float = 8.0          # no handback sooner than this after the hold begins [s]

    def __post_init__(self) -> None:
        if len(self.target) != NUM_CHANNELS:
            raise ValueError("target needs one value per hand channel")
        if self.i_max <= 0.0:
            raise ValueError("i_max must be positive")


# Raw command limits named in the paper and their normalized equivalents.
RAW_LIMIT_TO_I_MAX = {150: 0.15, 200: 0.20}


@dataclass(frozen=True)
class ModeConfig:
    """Interaction-mode selection (Sec. III-E)."""

    # Explicit handling cues that activate compliant mode.
    handling_cues: Tuple[str, ...] = ("gently", "carefully", "fragile")
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Socket timeout for one VLM request [s]. A reply that has not arrived by
    # the time the closure is armed is late and is ignored regardless of this.
    vlm_timeout: float = 5.0
