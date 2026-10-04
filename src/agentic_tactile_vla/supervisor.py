# Implements the paper specification. Not re-validated on hardware since it was written.
"""Execution supervisor: the finite state machine of paper Table I.

    State   b_t  Override  Exit condition
    IDLE    0    no        upcoming closure appears in the action chunk
    ARMED   0    no        terminal grasp assessment reached
    ADJUST  1    yes       grasp verdict or timeout
    HOLD    0    yes       policy reopens the hand

b_t gates progression of the body reference. When b_t = 1 only progression of
the body reference is paused; whole-body stabilization stays active throughout.
"Override" means the supervisor replaces the nominal hand target.

Selective intervention (eq. 3) is evaluated ONLY at the grasp-assessment point:

    eta_t = g_t * I[ sum_{i in F} c_hat_i(t) < 3 ]

Before that point missing contact is expected and triggers nothing. With at
least three position-support latches the supervisor stays transparent.

After ADJUST the verdict (eq. 8) decides authority:

* FORMED retains the corrected configuration and releases the body gate.
* EMPTY or CROOKED releases the hand override, so later policy actions can
  reopen the hand and generate another approach. Releasing the override does
  not itself command reopening.

Re-entry is blocked so that each closure yields at most one correction episode:

* a decrease of at least 0.05 in policy-commanded mean flexion relative to the
  terminal assessment, sustained for 0.12 s, releases a retained override;
* rearming additionally requires both the commanded and the measured mean
  flexion to stay at least 0.05 below their terminal-assessment values for
  0.12 s.

At authority transitions the output sets `authority_transition`. The caller
must then discard stale queued chunks and generate the next policy chunk from a
fresh observation, while whole-body stabilization holds the current body
reference.

Compliant branch. When the mode selector returns compliant control, the hand is
handed to the current-limited controller as soon as the closure is armed, so
the position path never applies an initial squeeze. The paper does not map this
branch onto Table I. Here it is realized as HOLD with `compliant_active` set
(override on, b_t = 0), left when the policy reopens the hand. The branch
covers compliant closure and holding only, not autonomous transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .closure import ClosureDetector
from .compliant import CompliantController
from .config import (FLEXION_CHANNELS, NUM_CHANNELS, THUMB_OPPOSITION,
                     CompliantConfig, SupervisorConfig)
from .contact import ChannelEvidence, ContactDetector
from .correction import FingerCorrector
from .modes import InteractionMode, ModeDecision, ModeSelector
from .verdict import Verdict, verdict_from_support

_EPS = 1e-9
_FLEX = list(FLEXION_CHANNELS)


class State(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    ADJUST = "ADJUST"
    HOLD = "HOLD"


@dataclass
class EpisodeRecord:
    """Summary of one assessed closure, for logging."""

    index: int
    t_assessment: float
    intervened: bool
    pre_support: int                      # position-support latches at assessment
    verdict: Optional[Verdict] = None
    support: int = 0
    latched: List[bool] = field(default_factory=lambda: [False] * NUM_CHANNELS)
    sources: List[str] = field(default_factory=lambda: [""] * NUM_CHANNELS)
    exhausted: List[bool] = field(default_factory=lambda: [False] * NUM_CHANNELS)
    q_final: List[float] = field(default_factory=lambda: [float("nan")] * NUM_CHANNELS)
    duration: float = 0.0
    timed_out: bool = False
    retained: bool = False
    mode: str = InteractionMode.NOMINAL.value
    mode_source: str = "default"


@dataclass
class SupervisorOutput:
    state: State
    hand_target: np.ndarray               # position reference to issue, shape (6,)
    override: bool                        # supervisor replaces the nominal hand target
    body_gate: int                        # b_t: 1 pauses body-reference progression
    mode: InteractionMode
    compliant_active: bool = False
    compliant_command: Optional[np.ndarray] = None   # normalized I_cmd when compliant_active
    authority_transition: bool = False    # discard queued chunks, re-infer from fresh observation
    event: Optional[str] = None
    episode: Optional[EpisodeRecord] = None           # set on the tick an episode closes
    evidence: Optional[Dict[int, ChannelEvidence]] = None


class ExecutionSupervisor:
    """Wraps a fixed VLA policy as a stateful supervisory layer (eq. 2)."""

    def __init__(self, cfg: SupervisorConfig | None = None,
                 mode_selector: Optional[ModeSelector] = None,
                 compliant_cfg: CompliantConfig | None = None) -> None:
        self.cfg = cfg or SupervisorConfig()
        self.closure = ClosureDetector(self.cfg)
        self.detector = ContactDetector(self.cfg)
        self.corrector = FingerCorrector(self.cfg)
        self.modes = mode_selector or ModeSelector(None)
        self.compliant = CompliantController(compliant_cfg)
        self.reset()

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        """Operator reset (return to the initial pose)."""
        self.state = State.IDLE
        self.mode = InteractionMode.NOMINAL
        self.mode_decision: Optional[ModeDecision] = None
        self.closure.reset()
        self.detector.reset()
        self.compliant.reset()
        self.modes.new_episode()
        self._nominal = None
        self._issued = None
        self._held = None
        self._compliant_active = False
        self._rearm_blocked = False
        self._term_cmd = 0.0
        self._term_meas = 0.0
        self._below_since: Optional[float] = None
        self._arm_targets = None
        self._arm_peak = 0.0
        self._closure_started = False
        self._no_closure_since: Optional[float] = None
        self._episode: Optional[EpisodeRecord] = None
        self._episode_count = 0

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _mean_flexion(vec: np.ndarray) -> float:
        return float(np.mean(np.asarray(vec, dtype=float)[_FLEX]))

    def _sustained(self, t: float, condition: bool) -> bool:
        """True once `condition` has held for reopen_hold seconds."""
        if not condition:
            self._below_since = None
            return False
        if self._below_since is None:
            self._below_since = t
        return (t - self._below_since) >= self.cfg.reopen_hold - _EPS

    def _with_policy_opposition(self, reference: np.ndarray, nominal: np.ndarray) -> np.ndarray:
        out = np.asarray(reference, dtype=float).copy()
        out[THUMB_OPPOSITION] = nominal[THUMB_OPPOSITION]
        return out

    def _block_rearm(self) -> None:
        self._rearm_blocked = True
        self._below_since = None

    # -- main entry -------------------------------------------------------------
    def step(self, t: float, chunk: Optional[np.ndarray], idx: int, q: np.ndarray,
             effort: Optional[np.ndarray] = None, instruction: Optional[str] = None,
             image: object = None, qdot: Optional[np.ndarray] = None) -> SupervisorOutput:
        """Advance the supervisor by one control tick.

        chunk        nominal hand targets of the current action chunk, shape (H, 6),
                     or None while a new chunk is pending
        idx          index of the action being executed in that chunk
        q            measured normalized finger positions, shape (6,)
        effort       motor-effort proxy, shape (6,); device units, not force
        instruction  operator instruction, parsed for handling cues
        image        current camera frame for the VLM (optional)
        """
        q = np.asarray(q, dtype=float).reshape(-1)[:NUM_CHANNELS]
        effort = (np.zeros(NUM_CHANNELS) if effort is None
                  else np.asarray(effort, dtype=float).reshape(-1)[:NUM_CHANNELS])

        if chunk is not None:
            chunk = np.asarray(chunk, dtype=float)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            idx = int(np.clip(idx, 0, chunk.shape[0] - 1))
            self._nominal = chunk[idx, :NUM_CHANNELS].copy()
        elif self._nominal is None:
            self._nominal = q.copy()
        nominal = self._nominal
        self.closure.push(t, nominal)
        cmd_mean = self._mean_flexion(nominal)
        meas_mean = self._mean_flexion(q)

        handler = {
            State.IDLE: self._idle,
            State.ARMED: self._armed,
            State.ADJUST: self._adjust,
            State.HOLD: self._hold,
        }[self.state]
        out = handler(t, chunk, idx, q, effort, nominal, cmd_mean, meas_mean,
                      instruction, image, qdot)
        self._issued = out.hand_target.copy()
        return out

    def _last_issued(self, nominal: np.ndarray) -> np.ndarray:
        """Reference issued on the previous tick.

        The position measured now is the response to that reference, so the
        detector pairs q(t) with it rather than with the target about to be
        issued. Otherwise one tick of transport delay would look like a stall
        at the onset of every closure.
        """
        return nominal if self._issued is None else self._issued

    # -- IDLE -----------------------------------------------------------------
    def _idle(self, t, chunk, idx, q, effort, nominal, cmd_mean, meas_mean,
              instruction, image, qdot) -> SupervisorOutput:
        cfg = self.cfg
        event = None
        if self._rearm_blocked:
            reopened = (cmd_mean <= self._term_cmd - cfg.reopen_delta + _EPS
                        and meas_mean <= self._term_meas - cfg.reopen_delta + _EPS)
            if self._sustained(t, reopened):
                self._rearm_blocked = False
                self._below_since = None
                self.detector.reset()          # reopening resets all latches
                self.modes.new_episode()
                self.mode = InteractionMode.NOMINAL
                self.mode_decision = None
                event = "REOPENED"
        else:
            # The hand is open: let the VLM look at the scene, once per episode.
            self.modes.maybe_request(image, instruction)
            if self.closure.upcoming_closure(chunk, idx):
                return self._arm(t, q, nominal, cmd_mean, meas_mean, instruction, qdot)
        return SupervisorOutput(State.IDLE, nominal.copy(), False, 0, self.mode, event=event)

    def _arm(self, t, q, nominal, cmd_mean, meas_mean, instruction, qdot) -> SupervisorOutput:
        self.detector.reset()
        self._arm_targets = nominal[_FLEX].copy()
        self._arm_peak = cmd_mean
        self._closure_started = False
        self._no_closure_since = None
        self._below_since = None
        self.mode_decision = self.modes.decide(instruction)
        self.mode = self.mode_decision.mode

        if self.mode is InteractionMode.COMPLIANT:
            # Hand the closure to the current-limited controller before any
            # position-mode squeeze is issued.
            self.state = State.HOLD
            self._compliant_active = True
            self._held = q.copy()
            self._term_cmd = cmd_mean
            self._term_meas = meas_mean
            self.compliant.reset()
            command = self.compliant.step(t, q, qdot=qdot)
            return SupervisorOutput(State.HOLD, self._with_policy_opposition(self._held, nominal),
                                    True, 0, self.mode, compliant_active=True,
                                    compliant_command=command, authority_transition=True,
                                    event="COMPLIANT_ENGAGED")

        self.state = State.ARMED
        evidence = self.detector.update(t, self._last_issued(nominal), q)
        return SupervisorOutput(State.ARMED, nominal.copy(), False, 0, self.mode,
                                event="ARMED", evidence=evidence)

    # -- ARMED ----------------------------------------------------------------
    def _armed(self, t, chunk, idx, q, effort, nominal, cmd_mean, meas_mean,
               instruction, image, qdot) -> SupervisorOutput:
        cfg = self.cfg
        evidence = self.detector.update(t, self._last_issued(nominal), q)   # position support only
        self._arm_peak = max(self._arm_peak, cmd_mean)
        if not self._closure_started:
            self._closure_started = bool(
                np.any(nominal[_FLEX] - self._arm_targets >= cfg.arm_delta - _EPS))

        # The policy reopened before finishing the closure: nothing to assess.
        if self._sustained(t, cmd_mean <= self._arm_peak - cfg.reopen_delta + _EPS):
            return self._disarm(nominal, "DISARMED_REOPEN")

        # The announced closure never started and is no longer in the chunk.
        if not self._closure_started:
            if self.closure.upcoming_closure(chunk, idx):
                self._no_closure_since = None
            else:
                if self._no_closure_since is None:
                    self._no_closure_since = t
                if t - self._no_closure_since >= cfg.arm_window - _EPS:
                    return self._disarm(nominal, "DISARMED_NO_CLOSURE")
            return SupervisorOutput(State.ARMED, nominal.copy(), False, 0, self.mode,
                                    evidence=evidence)

        if not self.closure.terminal_phase():
            return SupervisorOutput(State.ARMED, nominal.copy(), False, 0, self.mode,
                                    evidence=evidence)

        # ---- grasp-assessment point (g_t = 1) --------------------------------
        self._term_cmd = cmd_mean
        self._term_meas = meas_mean
        pre_support = self.detector.support_count
        intervene = (not cfg.selective) or pre_support < cfg.min_support   # eq. 3
        self._episode_count += 1
        self._episode = EpisodeRecord(
            index=self._episode_count, t_assessment=t, intervened=intervene,
            pre_support=pre_support, mode=self.mode.value,
            mode_source=self.mode_decision.source if self.mode_decision else "default")

        if not intervene:
            # Sufficient support: remain transparent. The closure has been
            # assessed, so re-entry is blocked until the hand reopens.
            record = self._close_episode(t, q, verdict=None, retained=False)
            self.state = State.IDLE
            self._block_rearm()
            return SupervisorOutput(State.IDLE, nominal.copy(), False, 0, self.mode,
                                    event="TRANSPARENT", episode=record, evidence=evidence)

        self.state = State.ADJUST
        self.detector.restart_history()
        self.detector.begin_adjust(effort)
        reference = self.corrector.start(t, q, self.detector.latched)
        evidence = self.detector.update(t, reference, q, effort)
        return SupervisorOutput(State.ADJUST, self._with_policy_opposition(reference, nominal),
                                True, 1, self.mode, authority_transition=True,
                                event="ADJUST", evidence=evidence)

    def _disarm(self, nominal, event) -> SupervisorOutput:
        self.state = State.IDLE
        self.detector.reset()
        self.modes.new_episode()
        self.mode = InteractionMode.NOMINAL
        self.mode_decision = None
        self._below_since = None
        return SupervisorOutput(State.IDLE, nominal.copy(), False, 0, self.mode, event=event)

    # -- ADJUST ---------------------------------------------------------------
    def _adjust(self, t, chunk, idx, q, effort, nominal, cmd_mean, meas_mean,
                instruction, image, qdot) -> SupervisorOutput:
        cfg = self.cfg
        # Evidence about the reference issued on the previous tick.
        evidence = self.detector.update(t, self.corrector.reference, q, effort)
        reference = self.corrector.step(t, q, self.detector.latched)

        if not self.corrector.finished(t):
            return SupervisorOutput(State.ADJUST, self._with_policy_opposition(reference, nominal),
                                    True, 1, self.mode, evidence=evidence)

        support = self.detector.support_count
        verdict = verdict_from_support(support, cfg) if cfg.use_verdict else None
        retain = (verdict is Verdict.FORMED) or not cfg.verdict_release or not cfg.use_verdict
        timed_out = self.corrector.timed_out(t)
        self.detector.end_adjust()
        record = self._close_episode(t, q, verdict=verdict, retained=retain, timed_out=timed_out)
        label = verdict.value if verdict else "EXTENDED"

        if retain:
            # Retain the corrected configuration and release the body gate.
            self.state = State.HOLD
            self._held = reference.copy()
            self._below_since = None
            return SupervisorOutput(State.HOLD, self._with_policy_opposition(reference, nominal),
                                    True, 0, self.mode, authority_transition=True,
                                    event=f"VERDICT_{label}", episode=record, evidence=evidence)

        # Release the hand override. This does not command reopening: the
        # nominal policy target simply becomes the hand reference again.
        self.state = State.IDLE
        self._block_rearm()
        return SupervisorOutput(State.IDLE, nominal.copy(), False, 0, self.mode,
                                authority_transition=True, event=f"VERDICT_{label}",
                                episode=record, evidence=evidence)

    def _close_episode(self, t, q, verdict, retained, timed_out=False) -> EpisodeRecord:
        record = self._episode
        record.verdict = verdict
        record.support = self.detector.support_count
        record.latched = [bool(v) for v in self.detector.latched]
        record.sources = self.detector.sources()
        record.q_final = [float(v) for v in q]
        record.retained = retained
        record.timed_out = timed_out
        if record.intervened:
            record.exhausted = [bool(v) for v in self.corrector.exhausted]
            record.duration = self.corrector.elapsed(t)
        self._episode = None
        return record

    # -- HOLD -----------------------------------------------------------------
    def _hold(self, t, chunk, idx, q, effort, nominal, cmd_mean, meas_mean,
              instruction, image, qdot) -> SupervisorOutput:
        cfg = self.cfg
        if self._compliant_active:
            # No terminal assessment exists in this branch; the reference
            # levels are the highest commanded and measured mean flexion seen.
            self._term_cmd = max(self._term_cmd, cmd_mean)
            self._term_meas = max(self._term_meas, meas_mean)

        if self._sustained(t, cmd_mean <= self._term_cmd - cfg.reopen_delta + _EPS):
            # The policy reopens the hand: release the retained override.
            self.state = State.IDLE
            self._compliant_active = False
            self._held = None
            self._block_rearm()
            return SupervisorOutput(State.IDLE, nominal.copy(), False, 0, self.mode,
                                    authority_transition=True, event="RELEASED")

        target = self._with_policy_opposition(self._held, nominal)
        if self._compliant_active:
            command = self.compliant.step(t, q, qdot=qdot)
            return SupervisorOutput(State.HOLD, target, True, 0, self.mode,
                                    compliant_active=True, compliant_command=command)
        return SupervisorOutput(State.HOLD, target, True, 0, self.mode)
