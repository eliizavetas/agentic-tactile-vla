# Implements the paper specification. Not re-validated on hardware since it was written.
"""Per-finger correction with bounded travel and timing (paper Sec. III-D).

* Each unsupported finger advances independently toward its permitted travel
  limit.
* Correction starts from the MEASURED positions and advances in increments of
  0.004 every 20 ms.
* On support detection the reference is fixed at the measured position plus a
  normalized preload of 0.02, clipped to the travel limit, and that finger
  stops.
* Adjustment lasts at least 0.6 s and at most 4.0 s. After the minimum it ends
  when every monitored finger has either latched support or exhausted its
  travel, or on timeout.
* Reaching the travel limit without detector activation does NOT establish
  contact: too little remaining motion may keep the history and progress
  requirements of the stall test from being met.

Thumb opposition is not touched here; the supervisor passes the policy target
through for that channel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import FLEXION_CHANNELS, NUM_CHANNELS, SupervisorConfig

_EPS = 1e-9


class FingerCorrector:
    """Generates the override finger references during ADJUST."""

    def __init__(self, cfg: SupervisorConfig | None = None) -> None:
        self.cfg = cfg or SupervisorConfig()
        self._limit = np.asarray(self.cfg.travel_limit, dtype=float)
        self.reference = np.zeros(NUM_CHANNELS)
        self.fixed = np.zeros(NUM_CHANNELS, dtype=bool)      # stopped on support
        self.exhausted = np.zeros(NUM_CHANNELS, dtype=bool)  # reached the travel limit
        self._t_start: Optional[float] = None
        self._t_step: Optional[float] = None

    def start(self, t: float, q: np.ndarray, latched: np.ndarray) -> np.ndarray:
        """Begin a correction episode from the measured positions `q`.

        Fingers whose support latched before the assessment already satisfy the
        stopping rule, so their reference is fixed at measured + preload.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        latched = np.asarray(latched, dtype=bool).reshape(-1)
        self.reference = np.minimum(q.copy(), self._limit)
        self.fixed[:] = False
        self.exhausted[:] = False
        for i in FLEXION_CHANNELS:
            if latched[i]:
                self._fix(i, q[i])
            elif self.reference[i] >= self._limit[i] - _EPS:
                self.exhausted[i] = True
        self._t_start = t
        self._t_step = t
        return self.reference.copy()

    def _fix(self, i: int, q_i: float) -> None:
        self.reference[i] = min(q_i + self.cfg.preload, self._limit[i])
        self.fixed[i] = True

    def step(self, t: float, q: np.ndarray, latched: np.ndarray) -> np.ndarray:
        """Advance the references to time `t` and return them."""
        if self._t_start is None:
            raise RuntimeError("start() must be called before step()")
        cfg = self.cfg
        q = np.asarray(q, dtype=float).reshape(-1)
        latched = np.asarray(latched, dtype=bool).reshape(-1)

        for i in FLEXION_CHANNELS:
            if latched[i] and not self.fixed[i]:
                self._fix(i, q[i])

        n = int(np.floor((t - self._t_step) / cfg.step_period + 1e-6))
        if n > 0:
            self._t_step += n * cfg.step_period
            for i in FLEXION_CHANNELS:
                if self.fixed[i] or self.exhausted[i]:
                    continue
                self.reference[i] = min(self.reference[i] + n * cfg.step, self._limit[i])
                if self.reference[i] >= self._limit[i] - _EPS:
                    self.exhausted[i] = True
        return self.reference.copy()

    def hold(self, t: float) -> None:
        """Skip reference advancement up to time `t` (stale hand state)."""
        if self._t_step is not None:
            self._t_step = max(self._t_step, t)

    def elapsed(self, t: float) -> float:
        return 0.0 if self._t_start is None else t - self._t_start

    def all_settled(self) -> bool:
        """Every monitored finger has latched support or exhausted its travel."""
        idx = list(FLEXION_CHANNELS)
        return bool(np.all(self.fixed[idx] | self.exhausted[idx]))

    def finished(self, t: float) -> bool:
        """Termination rule of the adjustment phase."""
        dt = self.elapsed(t)
        if dt >= self.cfg.max_adjust - _EPS:
            return True
        return dt >= self.cfg.min_adjust - _EPS and self.all_settled()

    def timed_out(self, t: float) -> bool:
        return self.elapsed(t) >= self.cfg.max_adjust - _EPS and not self.all_settled()
