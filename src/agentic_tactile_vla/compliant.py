# Implements the paper specification. Not re-validated on hardware since it was written.
"""Compliant control via current-limited interaction (paper Sec. III-F, eq. 9).

    I_cmd_i(t) = sat_[-I_max, I_max]( K_p * [q_d_i(t) - q_i(t)] - K_d * qdot_smooth_i(t) )

* q_d_i and q_i are normalized hand-interface positions.
* qdot_smooth_i is the exponentially smoothed finger velocity.
* K_p = 5.0 and K_d = 0.05 are controller-interface gains. They are not
  physical stiffness or damping coefficients.
* The loop period is 20 ms.
* Raw device command limits of 150 and 200 correspond to normalized limits
  I_max = 0.15 and 0.20. These are device-level actuator commands. They are not
  calibrated motor current and not fingertip force.

Compliant mode is activated before closure, so the position path never applies
an initial high-effort squeeze. The controller supports compliant closure and
holding. It does not support autonomous transport.

This module is the reference implementation of eq. (9) and is what the unit
tests exercise. On hardware the same law runs inside the hand service under
drivers/brainco_compliant/, which adds a closing-speed cap, a setpoint ramp and
watchdogs that the paper does not describe.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .config import NUM_CHANNELS, CompliantConfig


class CurrentCommandSink(Protocol):
    """Anything that can apply a per-motor current-mode command to the hand."""

    def send_current(self, raw_command: np.ndarray) -> None:
        ...


class CompliantController:
    """Current-limited PD law on normalized finger positions."""

    def __init__(self, cfg: CompliantConfig | None = None) -> None:
        self.cfg = cfg or CompliantConfig()
        self._q_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None
        self._qdot_smooth = np.zeros(NUM_CHANNELS)

    def reset(self) -> None:
        self._q_prev = None
        self._t_prev = None
        self._qdot_smooth = np.zeros(NUM_CHANNELS)

    @property
    def smoothed_velocity(self) -> np.ndarray:
        return self._qdot_smooth.copy()

    def step(self, t: float, q: np.ndarray, q_des: Optional[np.ndarray] = None,
             qdot: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the normalized command I_cmd in [-I_max, I_max] per channel.

        `qdot` is the measured finger velocity in normalized units per second.
        When the hand does not report one, it is taken as the finite difference
        of consecutive position samples.
        """
        cfg = self.cfg
        q = np.asarray(q, dtype=float).reshape(-1)
        q_des = np.asarray(cfg.target if q_des is None else q_des, dtype=float).reshape(-1)

        if qdot is None:
            if self._q_prev is None or self._t_prev is None or t <= self._t_prev:
                qdot = np.zeros(NUM_CHANNELS)
            else:
                qdot = (q - self._q_prev) / (t - self._t_prev)
        qdot = np.asarray(qdot, dtype=float).reshape(-1)
        self._q_prev, self._t_prev = q.copy(), t

        self._qdot_smooth += cfg.velocity_alpha * (qdot - self._qdot_smooth)
        command = cfg.kp * (q_des - q) - cfg.kd * self._qdot_smooth
        return np.clip(command, -cfg.i_max, cfg.i_max)

    def to_raw(self, command: np.ndarray) -> np.ndarray:
        """Convert a normalized command to raw device units (0.15 -> 150)."""
        return np.rint(np.asarray(command, dtype=float) * self.cfg.raw_scale).astype(int)
