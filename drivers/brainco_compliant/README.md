# BrainCo compliant grasp service

Hardware driver for the compliant branch: a C++ service that drives the
BrainCo Revo2 hand in current mode over its serial port, and the clients that
talk to it.

These files were recovered from the robot and are included with translation of
comments and removal of site-specific values only. They were not rebuilt or
re-run when this repository was assembled. `CMakeLists.txt` is the exception:
the original build file was not recovered, so it was written for this
repository and is untested.

| File | Role |
|---|---|
| `brainco_grasp_service.cpp` | the service: owns the serial port, runs the control loop, accepts commands on a Unix socket |
| `h3_spherical_grasp.cpp` | six-finger grasp diagnostic; also compiled into the service, which reuses its constants and helpers |
| `h2_index_impedance.cpp` | single-finger impedance diagnostic |
| `h0_index_current_visible.cpp` | single-finger current-mode probe |
| `run_brainco_grasp_service.sh` | launcher with pre-flight checks |
| `build_brainco_grasp_service.sh`, `CMakeLists.txt` | build |
| `brainco-grasp`, `brainco_grasp_cli.py` | command-line client |
| `impedance_grasp.py` | non-blocking Python client used by the bridge |

## Control law

Each 20 ms the service computes, per motor,

```
I_cmd = sat_[-I_max, I_max]( K_p * (q_d - q) - K_d * qdot_smooth )
```

with `K_p = 5.0`, `K_d = 0.05`, an exponentially smoothed velocity and raw
limits of 200 (profile `nominal`) or 150 (profile `gentle`), that is
`I_max = 0.20` or `0.15`. This is equation (9) of the paper with the reported
parameters. These are device-level actuator commands, not calibrated current
or force.

## Implementation details not stated in the paper

The service contains safety logic around the control law that the paper does
not describe:

| Detail | Value in the source |
|---|---|
| Closing-speed cap | above a closing speed of 0.25 the positive command is additionally limited, falling linearly to zero and then negative at 0.55 |
| Setpoint ramp | `q_d` moves from the starting pose toward the grasp configuration at 0.06 per second |
| Velocity smoothing factor | 0.30 |
| Grasp configuration | `sphere`: 0.70, 0.70, 0.75, 0.75, 0.75, 0.75; `cylinder` differs in thumb opposition (0.80) |
| Maximum hold | 120 s, after which the hand is opened |
| Watchdogs | read failures, command/feedback current mismatch, speed and position bounds abort the grasp, send zero current and return the hand to an open pose in position mode |
| Start gates | the service refuses to start without two gate files, see `docs/deployment.md` |

The closing-speed cap is an additional safety limit on top of equation (9). It
only ever reduces the commanded effort.

`src/agentic_tactile_vla/compliant.py` implements equation (9) alone, without
these additions. It is used by the unit tests and as a reference.

## Build and use

See `docs/deployment.md`, section "Compliant mode on hardware".
