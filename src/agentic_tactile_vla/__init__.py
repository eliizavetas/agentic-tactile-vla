"""AgenticTactileVLA: contact-guided execution-time supervision for a fixed VLA policy."""

from .config import (CHANNEL_NAMES, FLEXION_CHANNELS, CompliantConfig, ModeConfig,
                     SupervisorConfig, variant_config)
from .closure import ClosureDetector
from .compliant import CompliantController
from .contact import ContactDetector
from .correction import FingerCorrector
from .modes import InteractionMode, ModeSelector, has_handling_cue, parse_vlm_reply
from .supervisor import ExecutionSupervisor, State, SupervisorOutput
from .verdict import Verdict, verdict_from_support

__version__ = "0.1.0"

__all__ = [
    "CHANNEL_NAMES", "FLEXION_CHANNELS", "ClosureDetector", "CompliantConfig",
    "CompliantController", "ContactDetector", "ExecutionSupervisor", "FingerCorrector",
    "InteractionMode", "ModeConfig", "ModeSelector", "State", "SupervisorConfig",
    "SupervisorOutput", "Verdict", "has_handling_cue", "parse_vlm_reply",
    "variant_config", "verdict_from_support",
]
