import numpy as np
import pytest

from agentic_tactile_vla import (ExecutionSupervisor, InteractionMode, State,
                                 SupervisorConfig, Verdict, variant_config)
from sim import DT, SimHand, run

FAR = 2.0   # contact position that is never reached


def events(outputs):
    return [o.event for o in outputs if o.event]


def test_transparent_when_three_fingers_stall_during_policy_closure():
    hand = SimHand(contact=[0.4, FAR, 0.4, 0.4, 0.4, 0.4])
    outs = run(ExecutionSupervisor(), hand, 3.0)
    assert "TRANSPARENT" in events(outs)
    assert not any(o.override for o in outs)
    assert all(o.body_gate == 0 for o in outs)
    assert not any(o.state is State.ADJUST for o in outs)


def test_formed_after_correction_and_retained():
    hand = SimHand(contact=[0.70, FAR, 0.70, 0.72, 0.74, FAR])
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.9,) * 6))
    outs = run(sup, hand, 6.0)
    adjust = [o for o in outs if o.state is State.ADJUST]
    assert adjust and all(o.body_gate == 1 and o.override for o in adjust)
    # thumb opposition always follows the policy target
    assert all(abs(o.hand_target[1] - 0.3) < 1e-9 for o in outs)
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.verdict is Verdict.FORMED
    assert done.episode.support == 4 and done.episode.pre_support == 0
    assert done.state is State.HOLD and done.body_gate == 0 and done.authority_transition
    assert 0.6 - 1e-9 <= done.episode.duration <= 4.0 + DT
    # fingers stopped at different depths: measured position + preload
    held = outs[-1].hand_target
    assert held[2] == pytest.approx(0.72, abs=0.01)
    assert held[4] == pytest.approx(0.76, abs=0.01)
    assert outs[-1].state is State.HOLD and outs[-1].override


def test_correction_starts_from_measured_positions():
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor()
    outs = run(sup, hand, 2.0)
    first = next(o for o in outs if o.event == "ADJUST")
    k = outs.index(first)
    assert first.authority_transition
    # reference at ADJUST entry equals the measured position, not the policy target
    assert np.allclose(first.hand_target[[0, 2, 3, 4, 5]], 0.6, atol=0.02)
    step = outs[k + 1].hand_target[2] - first.hand_target[2]
    assert step == pytest.approx(0.004, abs=1e-9)


def test_empty_releases_override_and_blocks_reentry():
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.8,) * 6))
    outs = run(sup, hand, 8.0)
    ev = events(outs)
    assert ev.count("ADJUST") == 1            # one correction episode per closure
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.verdict is Verdict.EMPTY
    assert all(done.episode.exhausted[i] for i in (0, 2, 3, 4, 5))
    assert done.state is State.IDLE and not done.override and done.authority_transition
    # releasing the override hands back the policy target; it does not open the hand
    assert np.allclose(done.hand_target[[0, 2, 3, 4, 5]], 0.6)
    assert outs[-1].state is State.IDLE


def test_travel_limit_alone_is_not_contact():
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.62,) * 6))
    outs = run(sup, hand, 4.0)
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.support == 0 and done.episode.verdict is Verdict.EMPTY


def test_crooked_means_one_or_two_supports():
    hand = SimHand(contact=[0.7, FAR, 0.7, FAR, FAR, FAR])
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.85,) * 6))
    outs = run(sup, hand, 8.0)
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.verdict is Verdict.CROOKED and done.episode.support == 2
    assert done.state is State.IDLE


def test_timeout_bounds_adjustment():
    # 4.0 s of travel at 0.2 per second cannot reach the limit from 0.1
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor()
    outs = run(sup, hand, 8.0, closed_level=0.115, ramp=0.05)
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.timed_out
    assert done.episode.duration == pytest.approx(4.0, abs=2 * DT)


def test_rearm_after_policy_reopens_and_closes_again():
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.8,) * 6))
    run(sup, hand, 4.0)
    assert sup.state is State.IDLE and sup._rearm_blocked
    # policy reopens: commanded and measured mean flexion drop
    for k in range(100):
        t = 4.0 + k * DT
        target = np.full((1, 6), 0.1)
        out = sup.step(t, target, 0, hand.q.copy(), hand.effort.copy())
        hand.apply(out.hand_target)
    assert not sup._rearm_blocked
    assert sup.detector.support_count == 0


def test_hold_released_when_policy_reopens():
    hand = SimHand(contact=[0.7, FAR, 0.7, 0.7, 0.7, 0.7])
    sup = ExecutionSupervisor(SupervisorConfig(travel_limit=(0.9,) * 6))
    outs = run(sup, hand, 12.0, t_open=4.0)
    ev = events(outs)
    assert "VERDICT_FORMED" in ev and "RELEASED" in ev and "REOPENED" in ev
    released = next(o for o in outs if o.event == "RELEASED")
    assert released.authority_transition and not released.override
    assert outs[-1].state is State.IDLE


def test_unconditional_variant_intervenes_even_with_support():
    hand = SimHand(contact=[0.4, FAR, 0.4, 0.4, 0.4, 0.4])
    outs = run(ExecutionSupervisor(variant_config("U-full")), hand, 4.0)
    assert "ADJUST" in events(outs)


def test_retain_variant_never_releases():
    hand = SimHand(contact=[FAR] * 6)
    cfg = variant_config("S-retain", SupervisorConfig(travel_limit=(0.8,) * 6))
    outs = run(ExecutionSupervisor(cfg), hand, 6.0)
    done = [o for o in outs if o.episode is not None][0]
    assert done.episode.verdict is Verdict.EMPTY and done.state is State.HOLD


def test_language_cue_engages_compliant_before_closure():
    hand = SimHand(contact=[FAR] * 6)
    sup = ExecutionSupervisor()
    outs = run(sup, hand, 3.0, instruction="pick up the cup gently")
    engaged = next(o for o in outs if o.event == "COMPLIANT_ENGAGED")
    assert engaged.mode is InteractionMode.COMPLIANT and engaged.compliant_active
    # the position path never issues the closing squeeze
    assert max(o.hand_target[2] for o in outs) < 0.2
    assert np.all(np.abs(engaged.compliant_command) <= 0.15 + 1e-12)
    assert not any(o.state is State.ADJUST for o in outs)


def test_missing_chunk_keeps_last_target():
    sup = ExecutionSupervisor()
    out1 = sup.step(0.0, np.full((5, 6), 0.2), 0, np.full(6, 0.2))
    out2 = sup.step(0.02, None, 0, np.full(6, 0.2))
    assert np.allclose(out1.hand_target, out2.hand_target)
