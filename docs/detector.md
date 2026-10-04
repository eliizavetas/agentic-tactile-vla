# Proprioceptive contact detector

Code: `src/agentic_tactile_vla/contact.py`. Paper: Sec. III-C, eqs. (4) to (7).

The hand has no touch sensors. Contact is estimated from three signals per
monitored flexion channel `i` in `F = {0, 2, 3, 4, 5}`:

| Symbol | Signal |
|---|---|
| `u_i(t)` | commanded position |
| `q_i(t)` | measured normalized position in `[0, 1]` |
| `e_i(t)` | motor-effort proxy reported by the hand |

Motor effort is a device-specific indicator of interaction resistance. It is
not calibrated force. The effort threshold below is in device units and is
only meaningful for the hand it was calibrated on.

## Equations

Progress over a history window `W`:

```
du_i(t) = u_i(t) - u_i(t - W)        (4)
dq_i(t) = q_i(t) - q_i(t - W)        (5)
```

Eligibility `E_i(t)` requires a complete history window without reopening and
at least `delta_u` of commanded advance. A saturated target creates no
evidence.

Position stall:

```
s_i(t) = E_i(t) * I[ du_i(t) >= delta_u  AND  dq_i(t) < rho_s * du_i(t) ]        (6)
```

Effort duration, active only inside ADJUST. On entry to ADJUST the current
effort sample becomes a fixed per-channel baseline `b_i` and the accumulator is
reset. `D_i(t)` is the continuous time for which `|e_i(t) - b_i| > delta_e`.

```
d_i(t) = E_i(t) * I[ D_i(t) >= T_e  AND  du_i(t) >= delta_u
                     AND  dq_i(t) < rho_d * du_i(t) ]        (7)
```

Support for finger `i` latches when `s_i(t) OR d_i(t)` fires.

* Position latches from before the assessment are preserved.
* Effort-assisted detections accumulate only in ADJUST.
* Reopening resets all latches.

## Parameters

| Symbol | Value | Field in `SupervisorConfig` | Role |
|---|---|---|---|
| `W` | 0.25 s | `window` | history window |
| `delta_u` | 0.015 | `delta_u` | minimum commanded advance over `W` |
| `rho_s` | 0.3 | `rho_s` | tracking ratio below which a finger counts as stalled |
| `rho_d` | 0.7 | `rho_d` | tracking ratio required together with sustained effort |
| `delta_e` | 0.3 | `delta_e` | effort deviation from the ADJUST baseline, device units |
| `T_e` | 0.06 s | `t_e` | continuous time the effort deviation must persist |

## Calibration

The thresholds were selected from 40 calibration closures on the five training
objects, eight per object, separate from all evaluation trials. Calibration
targeted stable finger stopping and suppression of short effort transients.
Held-out objects were not used for tuning. All thresholds, the preload, the
timing parameters and the controller gains were frozen before evaluation.

Do not tune these values on evaluation objects.

## Eligibility in practice

| Condition | Effect |
|---|---|
| Less than `W` of history | not eligible |
| The reference decreased inside the window | not eligible |
| Commanded advance below `delta_u` | not eligible |
| Reference at the travel limit | not eligible |

On entry to ADJUST the reference restarts from the measured position. The
history is restarted at that moment, so a full window must be collected again
and the reference discontinuity creates no evidence. Existing latches are kept.

A consequence: a finger that starts ADJUST close to its travel limit may run
out of travel before a full window with enough advance exists. Its support
then cannot be evaluated. Reaching the travel limit without detector
activation does not establish contact.

## Failure modes

**Compliant objects.** Both tests need a mismatch between commanded and
measured motion. A soft object can deform so that the finger keeps following
its command. The object is then physically enclosed and the detector still
reports no support. In the paper the soft ball produced no FORMED verdict in
six correction events although video showed it enclosed every time; four
events were CROOKED and two EMPTY.

**Missing evidence is not missing contact.** A latch means the tests fired. No
latch can mean no contact, a compliant contact, or too little excitation.

**Conservative rejection.** In the paper's retention audit, 18 of 44 rejected
configurations would still have retained the object (NPV 59.1%). On held-out
objects 25.6% of retainable configurations were falsely rejected. Acceptance is
the reliable direction: 70 of 76 accepted grasps retained the object.

**Effort is not force.** The effort channel compares a device signal with a
baseline taken on the same hand at the start of ADJUST. It carries no physical
unit and does not transfer across hands without recalibration.

**Short transients.** An effort deviation shorter than `T_e` is ignored by
design.
