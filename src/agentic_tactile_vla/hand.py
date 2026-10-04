# Implements the paper specification. Not re-validated on hardware since it was written.
"""BrainCo Revo2 hand interface over Unitree DDS.

The hand has six motor channels and no touch sensors. Positions are normalized
to [0, 1]. Motor order equals the channel order used in the paper:

    0 thumb flexion, 1 thumb opposition, 2 index, 3 middle, 4 ring, 5 little

State   topic rt/brainco/{side}/state, type MotorStates_ (unitree_go IDL).
        Per motor: q (normalized position), dq, tau_est.
Command topic rt/brainco/{side}/cmd, type MotorCmds_ with one MotorCmd_ per
        motor: mode=1, q=target, dq=speed, tau=0, kp=0, kd=0. The hand's
        internal position loop tracks the target.

tau_est is used as the motor-effort proxy e. It is a device-specific indicator
of interaction resistance. It is NOT calibrated force or torque, and its scale
is only meaningful relative to a baseline taken on the same hand.

This DDS interface is position-mode only. Compliant control runs in the
current-mode hand service under drivers/brainco_compliant/, which takes over
the hand's serial port for the duration of a compliant grasp.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import CHANNEL_NAMES, NUM_CHANNELS

STATE_TOPIC = "rt/brainco/{side}/state"
CMD_TOPIC = "rt/brainco/{side}/cmd"
POSITION_MODE = 1

_dds_initialized = False
_dds_lock = threading.Lock()


def clamp01(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def init_dds(iface: Optional[str] = None, domain: int = 0) -> None:
    """Initialize the DDS channel factory once per process.

    `iface` is the network interface connected to the robot. It defaults to the
    HAND_IFACE environment variable.
    """
    global _dds_initialized
    with _dds_lock:
        if _dds_initialized:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        iface = iface or os.environ.get("HAND_IFACE") or None
        if iface:
            ChannelFactoryInitialize(domain, iface)
        else:
            ChannelFactoryInitialize(domain)
        _dds_initialized = True


@dataclass
class HandState:
    q: np.ndarray        # measured normalized positions, shape (6,)
    dq: np.ndarray       # measured velocities as reported by the hand, shape (6,)
    effort: np.ndarray   # motor-effort proxy e (tau_est), device units, not force
    stamp: float         # time.monotonic() of reception
    seq: int             # increments with every received message


class BraincoHand:
    """One BrainCo Revo2 hand: state subscription and position commands."""

    def __init__(self, side: str = "left", iface: Optional[str] = None,
                 domain: int = 0, connect: bool = True) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self._lock = threading.Lock()
        self._state: Optional[HandState] = None
        self._seq = 0
        self._pub = None
        self._sub = None
        if connect:
            self._connect(iface, domain)

    # -- DDS wiring -------------------------------------------------------------
    def _connect(self, iface: Optional[str], domain: int) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        init_dds(iface, domain)
        self._sub = ChannelSubscriber(STATE_TOPIC.format(side=self.side), MotorStates_)
        self._sub.Init(self._on_state, 10)
        self._pub = ChannelPublisher(CMD_TOPIC.format(side=self.side), MotorCmds_)
        self._pub.Init()

    def _on_state(self, msg) -> None:
        states = list(msg.states)[:NUM_CHANNELS]
        if len(states) < NUM_CHANNELS:
            return
        self.ingest(
            q=[m.q for m in states],
            dq=[m.dq for m in states],
            effort=[m.tau_est for m in states],
        )

    def ingest(self, q, dq, effort) -> None:
        """Store one state sample (also used by tests and offline replay)."""
        with self._lock:
            self._seq += 1
            self._state = HandState(
                q=np.asarray(q, dtype=float).reshape(-1)[:NUM_CHANNELS].copy(),
                dq=np.asarray(dq, dtype=float).reshape(-1)[:NUM_CHANNELS].copy(),
                effort=np.asarray(effort, dtype=float).reshape(-1)[:NUM_CHANNELS].copy(),
                stamp=time.monotonic(),
                seq=self._seq,
            )

    # -- state ------------------------------------------------------------------
    def wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._state is not None:
                return True
            time.sleep(0.02)
        return self._state is not None

    def read(self) -> Optional[HandState]:
        """Latest state sample, or None if nothing has been received yet."""
        with self._lock:
            return self._state

    # -- commands ---------------------------------------------------------------
    def build_command(self, target, speed=1.0):
        """Build a MotorCmds_ message for normalized position targets."""
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        target = clamp01(np.asarray(target, dtype=float).reshape(-1)[:NUM_CHANNELS])
        speed = clamp01(np.broadcast_to(np.asarray(speed, dtype=float), (NUM_CHANNELS,)))
        cmds = []
        for i in range(NUM_CHANNELS):
            cmd = unitree_go_msg_dds__MotorCmd_()
            cmd.mode = POSITION_MODE
            cmd.q = float(target[i])
            cmd.dq = float(speed[i])
            cmd.tau = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0
            cmd.reserve = [0, 0, 0]
            cmds.append(cmd)
        msg = MotorCmds_()
        msg.cmds = cmds
        return msg

    def send_targets(self, target, speed=1.0, dry_run: bool = False) -> np.ndarray:
        """Send normalized position targets; returns the clamped targets."""
        clamped = clamp01(np.asarray(target, dtype=float).reshape(-1)[:NUM_CHANNELS])
        if dry_run:
            return clamped
        if self._pub is None:
            raise RuntimeError("hand is not connected")
        self._pub.Write(self.build_command(clamped, speed))
        return clamped

    def send_current(self, raw_command) -> None:
        raise NotImplementedError(
            "The rt/brainco DDS interface accepts position-mode commands only. "
            "Compliant control runs in the service under drivers/brainco_compliant/."
        )


class BraincoHandController:
    """Both hands behind the two-call interface the VLA bridge uses."""

    def __init__(self, iface: Optional[str] = None, domain: int = 0) -> None:
        self.left = BraincoHand("left", iface, domain)
        self.right = BraincoHand("right", iface, domain)

    def hand(self, side: str) -> BraincoHand:
        return self.left if side == "left" else self.right

    def wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        ok_left = self.left.wait_ready(timeout)
        ok_right = self.right.wait_ready(max(0.0, deadline - time.monotonic()))
        return ok_left and ok_right

    def get_state_normalized(self) -> Tuple[np.ndarray, np.ndarray]:
        """Measured positions (left, right); zeros for a hand with no state yet."""
        out = []
        for hand in (self.left, self.right):
            state = hand.read()
            out.append(np.zeros(NUM_CHANNELS) if state is None else state.q.copy())
        return out[0], out[1]

    def send_targets_normalized(self, left, right, dry_run: bool = False) -> None:
        self.left.send_targets(left, dry_run=dry_run)
        self.right.send_targets(right, dry_run=dry_run)


__all__ = [
    "BraincoHand", "BraincoHandController", "HandState", "CHANNEL_NAMES",
    "STATE_TOPIC", "CMD_TOPIC", "clamp01", "init_dds",
]
