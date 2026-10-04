import threading
import time

import numpy as np
import pytest

from agentic_tactile_vla import (ClosureDetector, CompliantConfig, CompliantController,
                                 ContactDetector, FingerCorrector, InteractionMode,
                                 ModeSelector, SupervisorConfig, Verdict, has_handling_cue,
                                 parse_vlm_reply, verdict_from_support)
from agentic_tactile_vla.hand import BraincoHand, clamp01
from agentic_tactile_vla.modes import strip_handling_cues

DT = 0.02


# ---------------------------------------------------------------- verdict (eq. 8)
@pytest.mark.parametrize("count,expected", [
    (0, Verdict.EMPTY), (1, Verdict.CROOKED), (2, Verdict.CROOKED),
    (3, Verdict.FORMED), (4, Verdict.FORMED), (5, Verdict.FORMED)])
def test_verdict_rule(count, expected):
    assert verdict_from_support(count) is expected


def test_paper_parameters():
    c = SupervisorConfig()
    assert (c.window, c.delta_u, c.rho_s, c.rho_d, c.delta_e, c.t_e) == \
        (0.25, 0.015, 0.3, 0.7, 0.3, 0.06)
    assert (c.arm_delta, c.arm_window, c.terminal_delta, c.terminal_window) == \
        (0.015, 0.25, 0.005, 0.12)
    assert (c.step, c.step_period, c.preload, c.min_adjust, c.max_adjust) == \
        (0.004, 0.02, 0.02, 0.6, 4.0)
    assert (c.reopen_delta, c.reopen_hold, c.min_support, c.control_period) == \
        (0.05, 0.12, 3, 0.02)
    k = CompliantConfig()
    assert (k.kp, k.kd, k.period, k.i_max) == (5.0, 0.05, 0.02, 0.15)


# ---------------------------------------------------------------- closure
def test_arming_and_terminal_phase():
    det = ClosureDetector()
    flat = np.full((50, 6), 0.1)
    assert not det.upcoming_closure(flat, 0)
    rising = flat.copy()
    rising[:, 2] += np.linspace(0, 0.1, 50)        # 0.002 per step -> 0.026 over 0.25 s
    assert det.upcoming_closure(rising, 0)
    slow = flat.copy()
    slow[:, 2] += np.linspace(0, 0.02, 50)         # below 0.015 over 0.25 s
    assert not det.upcoming_closure(slow, 0)

    for k in range(4):
        det.push(k * DT, np.full(6, 0.5))
    assert not det.terminal_phase()                # window not complete yet
    for k in range(4, 10):
        det.push(k * DT, np.full(6, 0.5))
    assert det.terminal_phase()
    det.push(10 * DT, np.array([0.5, 0.5, 0.506, 0.5, 0.5, 0.5]))
    assert not det.terminal_phase()                # one target varied by 0.006


def test_thumb_opposition_does_not_arm():
    det = ClosureDetector()
    chunk = np.full((50, 6), 0.1)
    chunk[:, 1] += np.linspace(0, 0.5, 50)
    assert not det.upcoming_closure(chunk, 0)


# ---------------------------------------------------------------- contact
def feed(det, u_fn, q_fn, n, effort_fn=None, t0=0.0):
    last = None
    for k in range(n):
        t = t0 + k * DT
        e = None if effort_fn is None else effort_fn(t)
        last = det.update(t, u_fn(t), q_fn(t), e)
    return last


def test_position_stall_latches_only_after_full_window():
    det = ContactDetector()
    u = lambda t: np.full(6, 0.3 + 0.2 * t)        # 0.05 commanded advance per 0.25 s
    q = lambda t: np.full(6, 0.3)                  # finger does not follow
    feed(det, u, q, 12)                            # 0.22 s of history
    assert det.support_count == 0
    feed(det, u, q, 3, t0=12 * DT)
    assert det.support_count == 5
    assert not det.latched[1]                      # thumb opposition is never monitored


def test_tracking_finger_does_not_latch():
    det = ContactDetector()
    u = lambda t: np.full(6, 0.3 + 0.2 * t)
    feed(det, u, u, 60)
    assert det.support_count == 0


def test_insufficient_command_progress_creates_no_evidence():
    det = ContactDetector()
    u = lambda t: np.full(6, 0.3 + 0.04 * t)       # 0.01 per window < delta_u
    feed(det, u, lambda t: np.full(6, 0.3), 60)
    assert det.support_count == 0


def test_saturated_target_creates_no_evidence():
    det = ContactDetector(SupervisorConfig(travel_limit=(0.5,) * 6))
    u = lambda t: np.full(6, 0.5)                  # already at the limit
    feed(det, u, lambda t: np.full(6, 0.3), 60)
    assert det.support_count == 0
    det = ContactDetector(SupervisorConfig(travel_limit=(0.31,) * 6))
    u = lambda t: np.minimum(0.3 + 0.2 * t, 0.31) * np.ones(6)
    ev = feed(det, u, lambda t: np.full(6, 0.3), 60)
    assert det.support_count == 0 and not ev[2].eligible


def test_reopening_inside_window_blocks_eligibility_and_reset_clears():
    det = ContactDetector()
    def u(t):
        return np.full(6, 0.3 + 0.2 * t - (0.05 if t > 0.2 else 0.0))
    ev = feed(det, u, lambda t: np.full(6, 0.3), 15)
    assert not ev[2].eligible
    feed(det, lambda t: np.full(6, 0.3 + 0.2 * t), lambda t: np.full(6, 0.3), 40, t0=1.0)
    assert det.support_count == 5
    det.reset()
    assert det.support_count == 0


def test_effort_channel_only_inside_adjust():
    # the finger reproduces half of the commanded motion: above rho_s, below rho_d
    u = lambda t: np.full(6, 0.3 + 0.2 * t)
    q = lambda t: np.full(6, 0.3 + 0.1 * t)
    high = lambda t: np.full(6, 0.9)

    det = ContactDetector()
    feed(det, u, q, 40, effort_fn=high)            # outside ADJUST: effort ignored
    assert det.support_count == 0

    det = ContactDetector()
    det.begin_adjust(np.full(6, 0.2))              # baseline b_i
    feed(det, u, q, 40, effort_fn=high)            # |e - b| = 0.7 > delta_e
    assert det.support_count == 5 and det.sources()[2] == "effort"

    det = ContactDetector()
    det.begin_adjust(np.full(6, 0.2))
    feed(det, u, q, 40, effort_fn=lambda t: np.full(6, 0.4))   # deviation 0.2 < delta_e
    assert det.support_count == 0

    det = ContactDetector(SupervisorConfig(use_effort=False))
    det.begin_adjust(np.full(6, 0.2))
    feed(det, u, q, 40, effort_fn=high)
    assert det.support_count == 0


def test_short_effort_transient_is_suppressed():
    u = lambda t: np.full(6, 0.3 + 0.2 * t)
    q = lambda t: np.full(6, 0.3 + 0.1 * t)
    # 0.04 s bursts separated by quiet samples never reach T_e = 0.06 s
    burst = lambda t: np.full(6, 0.9 if int(round(t / DT)) % 4 < 2 else 0.2)
    det = ContactDetector()
    det.begin_adjust(np.full(6, 0.2))
    feed(det, u, q, 60, effort_fn=burst)
    assert det.support_count == 0


# ---------------------------------------------------------------- correction
def test_corrector_increment_preload_and_limit():
    cor = FingerCorrector(SupervisorConfig(travel_limit=(0.5,) * 6))
    q = np.full(6, 0.4)
    none = np.zeros(6, dtype=bool)
    ref = cor.start(0.0, q, none)
    assert np.allclose(ref, 0.4)
    ref = cor.step(0.02, q, none)
    assert ref[2] == pytest.approx(0.404)
    assert ref[1] == pytest.approx(0.4)            # thumb opposition is not advanced
    latched = none.copy(); latched[2] = True
    ref = cor.step(0.04, np.full(6, 0.401), latched)
    assert ref[2] == pytest.approx(0.421)          # measured + preload
    ref = cor.step(0.10, np.full(6, 0.9), latched)
    assert ref[2] == pytest.approx(0.421)          # a fixed finger stays fixed
    latched[3] = True
    ref = cor.step(0.12, np.full(6, 0.49), latched)
    assert ref[3] == pytest.approx(0.5)            # preload clipped to the travel limit
    for k in range(7, 60):
        cor.step(k * DT, q, latched)
    assert cor.exhausted[0] and cor.exhausted[4] and cor.exhausted[5]
    assert cor.all_settled() and cor.finished(1.2)
    assert not cor.finished(0.5)                   # never before the 0.6 s minimum


# ---------------------------------------------------------------- modes
@pytest.mark.parametrize("text,expected", [
    ("nominal", InteractionMode.NOMINAL), ("compliant", InteractionMode.COMPLIANT),
    (" Compliant\n", InteractionMode.COMPLIANT),
    ("compliant.", None), ("the object is fragile, use compliant", None),
    ('{"mode": "compliant"}', None), ("gentle", None), ("", None), (None, None),
    ("compliant, I_max=0.1", None), ("soft plastic", None), (0.15, None)])
def test_vlm_reply_parser_accepts_only_the_two_labels(text, expected):
    assert parse_vlm_reply(text) is expected


def test_handling_cues():
    assert has_handling_cue("pick up the cup gently")
    assert has_handling_cue("Carefully place it down")
    assert has_handling_cue("the object is FRAGILE")
    assert not has_handling_cue("pick up the object and place it down")
    assert not has_handling_cue(None)
    assert strip_handling_cues("pick up the cup gently") == "pick up the cup"


def wait_for(selector):
    for _ in range(200):
        if selector._reply_ready:
            return
        time.sleep(0.005)


def test_language_cue_has_priority_over_vlm():
    calls = []
    sel = ModeSelector(lambda img, prompt: calls.append(prompt) or "nominal")
    assert not sel.maybe_request("frame", "grasp it gently")   # the cue decides
    d = sel.decide("grasp it gently")
    assert d.mode is InteractionMode.COMPLIANT and d.source == "language" and not calls


def test_vlm_selects_mode_and_never_sees_cues():
    prompts = []
    sel = ModeSelector(lambda img, prompt: prompts.append(prompt) or "compliant")
    assert sel.maybe_request("frame", "pick up the cup")
    assert not sel.maybe_request("frame", "pick up the cup")   # one request per episode
    wait_for(sel)
    d = sel.decide("pick up the cup")
    assert d.mode is InteractionMode.COMPLIANT and d.source == "vlm"
    assert "pick up the cup" in prompts[0] and len(prompts) == 1


@pytest.mark.parametrize("reply", ["maybe compliant", "", None, 42, "COMPLIANT MODE"])
def test_malformed_vlm_reply_defaults_to_nominal(reply):
    sel = ModeSelector(lambda img, prompt: reply)
    sel.maybe_request("frame", "pick up the cup")
    wait_for(sel)
    d = sel.decide("pick up the cup")
    assert d.mode is InteractionMode.NOMINAL and d.source == "default"


def test_late_missing_and_failed_vlm_default_to_nominal():
    gate = threading.Event()
    sel = ModeSelector(lambda img, prompt: gate.wait(2.0) and "compliant")
    sel.maybe_request("frame", "pick up the cup")
    d = sel.decide("pick up the cup")              # the reply has not arrived yet
    assert d.mode is InteractionMode.NOMINAL and d.detail == "VLM reply late"
    gate.set(); wait_for(sel)
    assert sel.decide("pick up the cup").mode is InteractionMode.NOMINAL   # cached

    assert ModeSelector(None).decide("pick it up").mode is InteractionMode.NOMINAL

    def boom(img, prompt):
        raise RuntimeError("service down")
    sel = ModeSelector(boom)
    sel.maybe_request("frame", "pick it up"); wait_for(sel)
    assert sel.decide("pick it up").mode is InteractionMode.NOMINAL


# ---------------------------------------------------------------- compliant (eq. 9)
def test_compliant_law_and_saturation():
    ctl = CompliantController(CompliantConfig(i_max=0.15))
    q = np.full(6, 0.74)
    cmd = ctl.step(0.0, q, q_des=np.full(6, 0.75), qdot=np.zeros(6))
    assert np.allclose(cmd, 5.0 * 0.01)                        # K_p * error
    cmd = ctl.step(0.02, q, q_des=np.full(6, 0.75), qdot=np.full(6, 1.0))
    assert np.allclose(cmd, 0.05 - 0.05 * 0.3)                 # smoothed velocity, alpha 0.3
    cmd = ctl.step(0.04, np.zeros(6), q_des=np.full(6, 0.75), qdot=np.zeros(6))
    assert np.allclose(cmd, 0.15)                              # saturated at +I_max
    cmd = ctl.step(0.06, np.ones(6), q_des=np.zeros(6), qdot=np.zeros(6))
    assert np.allclose(cmd, -0.15)                             # saturated at -I_max
    assert np.all(ctl.to_raw(np.full(6, 0.15)) == 150)
    assert np.all(CompliantController(CompliantConfig(i_max=0.20)).to_raw(np.full(6, 0.2)) == 200)


# ---------------------------------------------------------------- hand
def test_hand_state_and_clamping_without_dds():
    hand = BraincoHand("left", connect=False)
    assert hand.read() is None
    hand.ingest(q=[0.1] * 6, dq=[0.0] * 6, effort=[0.5] * 6)
    state = hand.read()
    assert state.seq == 1 and np.allclose(state.effort, 0.5)
    assert np.allclose(hand.send_targets([-1, 0.5, 2, 0, 1, 0.3], dry_run=True),
                       [0, 0.5, 1, 0, 1, 0.3])
    assert np.allclose(clamp01([1.5, -0.2]), [1.0, 0.0])
    with pytest.raises(ValueError):
        BraincoHand("middle", connect=False)
