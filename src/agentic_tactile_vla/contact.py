# Implements the paper specification. Not re-validated on hardware since it was written.
"""Proprioceptive contact detector (paper Sec. III-C, eqs. 4-7).

Contact is estimated only from signals the hand already publishes: commanded
position u_i, measured normalized position q_i and a motor-effort proxy e_i.
Motor effort is a device-specific indicator of interaction resistance. It is
NOT calibrated force.

Over a history window W:

    du_i(t) = u_i(t) - u_i(t - W)                                   (4)
    dq_i(t) = q_i(t) - q_i(t - W)                                   (5)

Eligibility E_i(t) requires a complete history window without reopening and at
least delta_u of commanded advance. A saturated target creates no evidence.

    s_i(t) = E_i * I[ du_i >= delta_u  and  dq_i < rho_s * du_i ]   (6)

The effort-duration channel is active only inside ADJUST. On entry the current
effort sample becomes a fixed per-channel baseline b_i and the accumulator is
reset. D_i(t) is the continuous time for which |e_i(t) - b_i| > delta_e.

    d_i(t) = E_i * I[ D_i >= T_e  and  du_i >= delta_u
                      and  dq_i < rho_d * du_i ]                    (7)

Support for finger i latches when s_i or d_i fires. Position latches from
before the assessment are preserved, effort-assisted detections accumulate only
in ADJUST, and reopening resets all latches.

A missing latch is not proof of missing contact. A highly compliant object can
deform so that the finger keeps tracking its command, which removes the
command-response mismatch both tests rely on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import FLEXION_CHANNELS, NUM_CHANNELS, SupervisorConfig

_EPS = 1e-9


@dataclass
class ChannelEvidence:
    """Detector quantities of one channel at one tick (for logging)."""

    eligible: bool = False
    du: float = float("nan")
    dq: float = float("nan")
    stall: bool = False            # s_i(t)
    effort: bool = False           # d_i(t)
    effort_duration: float = 0.0   # D_i(t)
    effort_deviation: float = 0.0  # |e_i(t) - b_i|
    latched: bool = False
    source: str = ""               # "stall", "effort" or "" for the latching event


@dataclass
class _Channel:
    history: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (t, u, q)
    latched: bool = False
    source: str = ""
    baseline: Optional[float] = None
    duration: float = 0.0
    last_t: Optional[float] = None


class ContactDetector:
    """Per-finger support latches from position stall and sustained effort."""

    def __init__(self, cfg: SupervisorConfig | None = None) -> None:
        self.cfg = cfg or SupervisorConfig()
        self._ch: Dict[int, _Channel] = {i: _Channel() for i in FLEXION_CHANNELS}
        self._effort_active = False

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        """Reopening resets all latches, histories and the effort channel."""
        self._ch = {i: _Channel() for i in FLEXION_CHANNELS}
        self._effort_active = False

    def restart_history(self) -> None:
        """Drop the command/position history but keep the latches.

        Used on entry to ADJUST, where the reference restarts from the measured
        positions. A full window must then be collected again before a channel
        becomes eligible, so the reference discontinuity creates no evidence.
        """
        for ch in self._ch.values():
            ch.history.clear()

    def begin_adjust(self, effort: np.ndarray) -> None:
        """Enter ADJUST: fix the effort baseline b_i and reset D_i."""
        effort = np.asarray(effort, dtype=float).reshape(-1)
        for i, ch in self._ch.items():
            ch.baseline = float(effort[i])
            ch.duration = 0.0
            ch.last_t = None
        self._effort_active = self.cfg.use_effort

    def end_adjust(self) -> None:
        """Leave ADJUST: effort-assisted detections stop accumulating."""
        self._effort_active = False
        for ch in self._ch.values():
            ch.duration = 0.0
            ch.last_t = None

    # -- queries --------------------------------------------------------------
    @property
    def latched(self) -> np.ndarray:
        """Boolean latch per hand channel (always False for thumb opposition)."""
        out = np.zeros(NUM_CHANNELS, dtype=bool)
        for i, ch in self._ch.items():
            out[i] = ch.latched
        return out

    @property
    def support_count(self) -> int:
        return int(sum(ch.latched for ch in self._ch.values()))

    def sources(self) -> List[str]:
        out = [""] * NUM_CHANNELS
        for i, ch in self._ch.items():
            out[i] = ch.source
        return out

    # -- update ---------------------------------------------------------------
    def update(self, t: float, u: np.ndarray, q: np.ndarray,
               effort: Optional[np.ndarray] = None) -> Dict[int, ChannelEvidence]:
        """Feed one sample of issued command u, measured position q and effort e.

        `u` must be the reference actually issued to the hand at this tick: the
        policy target outside ADJUST, the supervisor reference inside it.
        """
        cfg = self.cfg
        u = np.asarray(u, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(-1)
        e = None if effort is None else np.asarray(effort, dtype=float).reshape(-1)
        report: Dict[int, ChannelEvidence] = {}

        for i, ch in self._ch.items():
            ch.history.append((t, float(u[i]), float(q[i])))
            while len(ch.history) > 2 and t - ch.history[1][0] >= cfg.window - _EPS:
                ch.history.popleft()

            ev = ChannelEvidence(latched=ch.latched, source=ch.source)

            # D_i: continuous time with |e_i - b_i| > delta_e (ADJUST only)
            if self._effort_active and e is not None and ch.baseline is not None:
                ev.effort_deviation = abs(float(e[i]) - ch.baseline)
                if ev.effort_deviation > cfg.delta_e:
                    if ch.last_t is not None:
                        ch.duration += t - ch.last_t
                else:
                    ch.duration = 0.0
                ch.last_t = t
                ev.effort_duration = ch.duration

            t0, u0, q0 = ch.history[0]
            complete = (t - t0) >= cfg.window - _EPS
            if complete:
                ev.du = float(u[i]) - u0
                ev.dq = float(q[i]) - q0
                us = [h[1] for h in ch.history]
                reopened = any(b < a - _EPS for a, b in zip(us, us[1:]))
                saturated = float(u[i]) >= cfg.travel_limit[i] - _EPS
                ev.eligible = (not reopened) and (not saturated) and ev.du >= cfg.delta_u - _EPS
                if ev.eligible:
                    ev.stall = ev.dq < cfg.rho_s * ev.du
                    ev.effort = (
                        self._effort_active
                        and ch.duration >= cfg.t_e - _EPS
                        and ev.dq < cfg.rho_d * ev.du
                    )

            if not ch.latched and (ev.stall or ev.effort):
                ch.latched = True
                ch.source = "stall" if ev.stall else "effort"
                ev.latched = True
                ev.source = ch.source
            report[i] = ev
        return report
