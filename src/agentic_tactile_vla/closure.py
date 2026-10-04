# Implements the paper specification. Not re-validated on hardware since it was written.
"""Closure detection from the policy action chunk (paper Sec. III-A).

A closure is visible in the action chunk as progressive flexion-target motion.

* armed: any flexion target advances by 0.015 over 0.25 s
* terminal phase: every flexion target varies by less than 0.005 over 0.12 s

The terminal phase is the grasp-assessment point: the policy has completed its
commanded grasp, and the supervisor must decide whether physical support has
actually been established.

Realization notes. The paper gives the two conditions but not the buffer they
are evaluated on. Here arming looks at the upcoming part of the chunk (the
state table says "upcoming closure in the action chunk") and, as a fallback
when the chunk has too few steps left, at the targets already issued. The
terminal phase is evaluated on the targets issued over the trailing window, so
it reports a closure that has finished rather than one predicted to finish.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from .config import FLEXION_CHANNELS, SupervisorConfig


def _steps(duration: float, period: float) -> int:
    """Number of control steps that span `duration` (at least one)."""
    return max(1, int(np.ceil(duration / period - 1e-9)))


class ClosureDetector:
    """Detects arming and the terminal phase of a policy-commanded closure."""

    def __init__(self, cfg: SupervisorConfig | None = None) -> None:
        self.cfg = cfg or SupervisorConfig()
        self._flex = np.asarray(FLEXION_CHANNELS)
        self._issued: Deque[Tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self._issued.clear()

    def push(self, t: float, target: np.ndarray) -> None:
        """Record the nominal policy target issued at time `t`."""
        target = np.asarray(target, dtype=float).reshape(-1)
        self._issued.append((t, target[self._flex].copy()))
        horizon = max(self.cfg.arm_window, self.cfg.terminal_window) + 4 * self.cfg.control_period
        while self._issued and t - self._issued[0][0] > horizon:
            self._issued.popleft()

    # -- arming ---------------------------------------------------------------
    def upcoming_closure(self, chunk: Optional[np.ndarray], idx: int) -> bool:
        """True when any flexion target advances by arm_delta over arm_window."""
        cfg = self.cfg
        if chunk is not None:
            chunk = np.asarray(chunk, dtype=float)
            if chunk.ndim == 2 and chunk.shape[0] > 0:
                idx = int(np.clip(idx, 0, chunk.shape[0] - 1))
                ahead = min(idx + _steps(cfg.arm_window, cfg.control_period), chunk.shape[0] - 1)
                advance = chunk[ahead, self._flex] - chunk[idx, self._flex]
                if np.any(advance >= cfg.arm_delta - 1e-12):
                    return True
        past = self._sample_before(cfg.arm_window)
        if past is not None and self._issued:
            advance = self._issued[-1][1] - past
            if np.any(advance >= cfg.arm_delta - 1e-12):
                return True
        return False

    # -- terminal phase ---------------------------------------------------------
    def terminal_phase(self) -> bool:
        """True when every flexion target varied by less than terminal_delta
        over the trailing terminal_window. Requires a complete window."""
        cfg = self.cfg
        if not self._issued:
            return False
        t_now = self._issued[-1][0]
        if t_now - self._issued[0][0] < cfg.terminal_window - 1e-9:
            return False
        rows = [v for (t, v) in self._issued if t_now - t <= cfg.terminal_window + 1e-9]
        block = np.vstack(rows)
        variation = block.max(axis=0) - block.min(axis=0)
        return bool(np.all(variation < cfg.terminal_delta))

    def _sample_before(self, duration: float) -> Optional[np.ndarray]:
        """Issued target closest to `duration` seconds ago, if history reaches it."""
        if not self._issued:
            return None
        t_now = self._issued[-1][0]
        if t_now - self._issued[0][0] < duration - 1e-9:
            return None
        for t, v in self._issued:
            if t_now - t <= duration + 1e-9:
                return v
        return None
