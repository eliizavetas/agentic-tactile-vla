# Implements the paper specification. Not re-validated on hardware since it was written.
"""Grasp verdict from the final support count (paper eq. 8)."""

from __future__ import annotations

from enum import Enum

from .config import SupervisorConfig


class Verdict(str, Enum):
    """Outcome of one correction episode.

    The verdict depends only on the total number of monitored flexion channels
    with latched support evidence.

    FORMED   at least three channels have support evidence. The corrected
             configuration is retained and the body gate is released.
    EMPTY    no channel has support evidence.
    CROOKED  one or two channels have support evidence. The name is historical
             and misleading: it means insufficient support evidence for
             acceptance. It does NOT mean that the grasp is geometrically
             misaligned, and no geometric quantity enters the rule.

    A verdict is a statement about detector evidence, not about physical
    contact. A compliant object can be fully enclosed and still yield EMPTY or
    CROOKED, because its deformation lets the fingers keep tracking the command.
    """

    FORMED = "FORMED"
    EMPTY = "EMPTY"
    CROOKED = "CROOKED"


def verdict_from_support(support_count: int, cfg: SupervisorConfig | None = None) -> Verdict:
    """Map the number of latched support channels to a verdict (eq. 8)."""
    cfg = cfg or SupervisorConfig()
    if support_count < 0:
        raise ValueError("support count cannot be negative")
    if support_count >= cfg.min_support:
        return Verdict.FORMED
    if support_count == 0:
        return Verdict.EMPTY
    return Verdict.CROOKED
