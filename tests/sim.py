"""Minimal finger model used only by the unit tests.

Each finger tracks its reference at a bounded speed and is blocked at a contact
position. Effort rises with the tracking error while blocked. This is a test
fixture for the supervisor logic, not a model of the BrainCo hand.
"""

import numpy as np

DT = 0.02


class SimHand:
    def __init__(self, contact, q0=0.1, speed=1.5, effort_gain=20.0, soft=None):
        self.q = np.full(6, q0, dtype=float)
        self.contact = np.asarray(contact, dtype=float)
        self.speed = speed
        self.effort_gain = effort_gain
        # soft[i]: fraction of commanded motion still reproduced after contact
        self.soft = np.zeros(6) if soft is None else np.asarray(soft, dtype=float)
        self.effort = np.zeros(6)

    def apply(self, u):
        u = np.asarray(u, dtype=float)
        step = np.clip(u - self.q, -self.speed * DT, self.speed * DT)
        new_q = self.q + step
        for i in range(6):
            if new_q[i] > self.contact[i] and step[i] > 0:
                free = max(self.contact[i] - self.q[i], 0.0)
                new_q[i] = self.q[i] + free + self.soft[i] * (step[i] - free)
        self.q = new_q
        blocked = self.q >= self.contact - 1e-9
        self.effort = np.where(blocked, self.effort_gain * np.clip(u - self.q, 0, None), 0.0)


def policy_target(t, t_close=0.5, ramp=0.6, open_level=0.1, closed_level=0.6,
                  t_open=None):
    """Nominal hand target at time t: open, ramp closed, optionally reopen."""
    if t_open is not None and t >= t_open:
        level = max(open_level, closed_level - (t - t_open) / ramp * (closed_level - open_level))
    elif t < t_close:
        level = open_level
    else:
        level = min(closed_level, open_level + (t - t_close) / ramp * (closed_level - open_level))
    target = np.full(6, level)
    target[1] = 0.3          # thumb opposition chosen by the policy
    return target


def make_chunk(t, horizon=50, **kw):
    return np.vstack([policy_target(t + k * DT, **kw) for k in range(horizon)])


def run(supervisor, hand, duration, instruction=None, **policy_kw):
    """Run the closed loop; body-reference time stops while b_t = 1."""
    outputs = []
    t_policy = 0.0
    for k in range(int(round(duration / DT))):
        t = k * DT
        chunk = make_chunk(t_policy, **policy_kw)
        out = supervisor.step(t, chunk, 0, hand.q.copy(), hand.effort.copy(),
                              instruction=instruction)
        hand.apply(out.hand_target)
        if out.body_gate == 0:
            t_policy += DT
        outputs.append(out)
    return outputs
