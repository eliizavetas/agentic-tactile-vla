# Session runbook

How to run a session when everything is already installed. Installation,
gate files and background: [deployment.md](deployment.md).

In every robot terminal, load the settings first: `set -a; source .env; set +a`
from this repository.

## 1. Before you start

- [ ] Robot on the safety hoist, area around the table clear, hand empty.
- [ ] Laptop on the lab network or VPN.
- [ ] GPU server reachable from the laptop and from the robot.
- [ ] Emergency stop within reach.
- [ ] Compliant mode only: on the robot, `sudo -v` to warm the sudo ticket.

## 2. Terminals

Start them top to bottom. Do not start the next one until the "wait for"
condition is met.

| Terminal | Machine | Directory | Command | Wait for |
|---|---|---|---|---|
| T1 policy server | GPU server, in tmux | Isaac-GR00T checkout, its environment active | `python gr00t/eval/run_gr00t_server.py --model-path <checkpoint> --embodiment-tag NEW_EMBODIMENT --port $POLICY_PORT` | `Server is ready` |
| T2 port forward | GPU server host, in tmux; only if T1 runs in a container | any | `socat TCP-LISTEN:$POLICY_PORT,fork,reuseaddr TCP:<container address>:$POLICY_PORT` | no output; stays running |
| T3 camera | robot | GR00T-WholeBodyControl checkout, camera environment active | `python -m gear_sonic.camera.composed_camera --ego-view-camera usb --ego-view-device-id <video device id> --port $CAMERA_PORT` | resolution 1280x720 reported |
| T4 hand service | robot; exactly ONE instance | brainco_hand_service checkout | `sudo ./bin/brainco_hand_server -n $HAND_IFACE` | no `Failed to open`, no `No ttyUSB serial ports found` |
| T5 whole-body controller | robot | `gear_sonic_deploy` in the GR00T-WholeBodyControl checkout | `source scripts/setup_env.sh && ./deploy.sh --input-type zmq_manager real` | `Init done`; the robot stands up |
| T6 keyboard publisher | robot; exactly ONE publisher on its port | this repository, GR00T-WholeBodyControl data-collection environment active | `python scripts/keyboard_publisher.py` | `keys: k / i / p / g` |
| T7 bridge with supervisor | robot | this repository, same environment as T6 | see below | see below |

T7, first check the policy connection. It must print `True`:

```bash
python -c "import os; from agentic_tactile_vla.bridge.groot_client import Gr00tClient as C; print(C(os.environ['POLICY_HOST'], int(os.environ['POLICY_PORT'])).ping())"
```

Then start the bridge:

```bash
HAND_ENABLE=1 SLEW_ENABLE=1 python scripts/run_agent.py --hand left \
  --embodiment-tag NEW_EMBODIMENT --prompt "pick up the object and place it down" \
  2>&1 | tee bridge.log
```

- [ ] `PolicyServer is reachable`
- [ ] `[BrainCo] hand state reader ready (6 DOF)`: the hand service in T4 is publishing
- [ ] banner shows the agent enabled and the monitored channels:
  `[agent] hand=left enabled=True observe=False variant=S-full vlm=False adjust=[0, 2, 3, 4, 5]`

The prompt must be the checkpoint's training prompt, character for character.
The one above is from the recorded logs.

To follow supervisor events in a second robot terminal:
`tail -f bridge.log | grep --line-buffered "\[agent\]"`

Optional:

| Purpose | Where | What | Reference |
|---|---|---|---|
| Automatic mode selection | GPU server, in tmux | `python scripts/vlm_service.py --port $VLM_PORT` | [deployment.md](deployment.md), "Optional: VLM service" |
| | robot | `ssh -N -L $VLM_PORT:localhost:$VLM_PORT <user>@<GPU server>`, then `VLM_HOST=localhost` and `--vlm` on T7 | same |
| Fragile objects | robot, T7 | `COMPLIANT_DRIVER=1` on T7, after `sudo -v` | [deployment.md](deployment.md), "Compliant mode on hardware" |

**tmux.** T1, T2 and the VLM service must run in tmux on the GPU server, so
the laptop can be closed without stopping them. The robot terminals are
usually opened over ssh from the laptop and die with the ssh session. Run
T3, T4, T5 and T7 in tmux on the robot as well. Losing T5 drops whole-body
control. Do not close the laptop during a trial: T6 is where you press the keys.

## 3. Run a trial

Keys go into T6, followed by Enter.

| Order | Key | Effect |
|---|---|---|
| 1 | `k` | start the whole-body control loop (once per session) |
| 2 | `i` | move to the initial pose and switch to POSE mode. Required: without it the robot does not move. Also resets the supervisor |
| 3 | `p` | unpause the policy loop. `p` is a toggle: an even number of presses leaves it paused |
| — | `g` | optional: freeze or resume body-reference progression by hand |

Between trials: `p` to pause, reset the scene, `i`, `p`.

**Ordinary vs careful instruction.** No key switches it. The policy prompt
stays the same in every condition. Careful handling is a flag on T7,
`--handling-cue gently`, which reaches only the supervisor, so switching means
restarting T7 (section 4). Do not use `prompt:` lines in T6 for this: they
change the policy's prompt, which the checkpoint was not trained on.

## 4. Switch condition between trials

Procedure: `p` in T6 to pause, Ctrl+C in T7, start T7 with the new flags,
then `i` and `p`. Pressing `k` is only needed if the whole-body loop was
stopped.

| Condition | T7 flags | Restart |
|---|---|---|
| Full supervisor | none (variant `S-full`) | T7 only |
| Base VLA | `--no-agent` | T7 only |
| Observe only | `--observe` | T7 only |
| Ablation | `--variant U-stall`, `U-full`, `S-position` or `S-retain` | T7 only |
| Careful instruction | `--handling-cue gently` | T7 only |
| VLM mode selection | `--vlm`, VLM service and tunnel running | T7 only |
| Compliant hardware | `COMPLIANT_DRIVER=1` in front of the command | T7 only |
| Changed threshold | edit `src/agentic_tactile_vla/config.py`. The variants are the switches in `VARIANTS` there | T7 only |

Never restart T1 to T6 to change a condition. Keep the prompt identical
across conditions.

## 5. Logs

| File | Written by | Contains |
|---|---|---|
| `agent_logs/verdicts_<hand>_<time>.csv` | T7 (`$AGENT_LOG_DIR`) | one row per assessed closure: intervention, support count, verdict, per-finger latches, mode |
| `agent_logs/ramp_<hand>_<time>.csv` | T7 | one row per control tick inside ADJUST: reference, position, effort deviation, eligibility, latch |
| `bridge.log` | T7, through `tee` | console output of the bridge and the supervisor |
| `/tmp/impedance_grasp.log`, `drivers/brainco_compliant/data_out/grasp_service/` | compliant handover | service console and per-grasp records |

`ticks_*` and `marks_*` in `data/raw/` were written in August by a chunk probe
and a marker tool. These are not part of this repository, and a session run
from it does not write them.

Copy logs off the robot from the laptop:

```bash
scp -r <robot user>@<robot address>:<path to this repository>/agent_logs ./agent_logs_$(date +%Y%m%d)
scp <robot user>@<robot address>:<path to this repository>/bridge.log ./bridge_$(date +%Y%m%d_%H%M).log
```

Copy after every session; restarting T7 with `tee bridge.log` overwrites it.

## 6. Stop safely

- [ ] `p` in T6: policy loop paused.
- [ ] Ctrl+C in T7.
- [ ] If T7 was stopped during a compliant grasp: `sudo pkill -f brainco_grasp_service`, then check T4 is still running, and restart it if not.
- [ ] Put the robot down following the lab procedure, BEFORE stopping the whole-body controller.
- [ ] Ctrl+C in T5.
- [ ] Ctrl+C in T6.
- [ ] Ctrl+C in T4, then T3.
- [ ] Policy server last: T2, then T1.
- [ ] Copy the logs (section 5).

## 7. If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `waiting for state msg` | whole-body loop not started | press `k` |
| robot idle while actions are sent | no initial pose | press `i` |
| robot stopped mid-trial | `p` pressed an even number of times | press `p` |
| `PolicyServer not reachable` | port forward down | restart T2; check T1 |
| ping prints `False` | T1 not ready or T2 down | wait for `Server is ready`; restart T2 |
| camera will not open | stale camera process holds the device | `pkill -f composed_camera; sudo fuser -k /dev/video<id>`, then restart T3 |
| `Address already in use` | a server is already running on that port | do not start a second one; use the running one |
| `Giving up to decode frame after 20 retries` | two hand services on the same serial port | stop the extra one, keep exactly one |
| `could not free the serial port` (compliant handover) | the hand server could not be stopped: no sudo ticket, so `sudo -n pkill` failed | `sudo -v` on the robot, then retry |
| `no READY` (compliant handover) | sudoers rule lacks the environment tag, so the service never started | fix the rule, see [deployment.md](deployment.md), "Launch rules the client follows" |
| `COMPLIANT_TOO_LATE` | compliant decision arrived after the hand closed | expected behaviour; the grasp stays on the position path |
| hand completely dead, both hands | both hand processes stopped: the hand server serves both hands | restart T4; check no `brainco_grasp_service` is left running |
| arm moves but hand does not; `[BrainCo] DRY-RUN` lines | `HAND_ENABLE` not set | restart T7 with `HAND_ENABLE=1` |
| no `[agent]` lines | supervisor disabled (`--no-agent`) or no hand state | check the T7 flags and T4 |
| `[agent] step failed, disabling` | the supervisor raised an exception | T7 continues as base VLA; restart T7 and check `bridge.log` |
