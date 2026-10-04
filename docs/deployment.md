# Deployment on the real robot

Three machines are involved.

| Machine | Runs |
|---|---|
| GPU server | GR00T policy server, optionally the VLM service |
| Robot (Unitree G1 onboard computer) | camera server, BrainCo hand service, GEAR-SONIC whole-body controller, keyboard publisher, the bridge with the supervisor |
| Workstation | terminals into both |

Copy `.env.example` to `.env`, fill in the addresses and `source` it in every
terminal on the robot. Nothing in this repository contains real addresses.

The supervisor code in `src/` is a reimplementation from the paper and has not
been re-run on hardware. Treat the first runs as a bring-up: start with
`--observe`, keep the emergency stop in hand, and set the travel limits for
your hand in `SupervisorConfig.travel_limit` first.

## Launch order

Start the processes in this order. Steps marked WAIT must report ready before
the next one is started.

### 1. Policy server (GPU server) — WAIT for "Server is ready"

```bash
cd <Isaac-GR00T checkout> && source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
  --model-path <path to the fine-tuned GR00T-N1.6-3B checkpoint> \
  --embodiment-tag NEW_EMBODIMENT --port 5555
```

The checkpoint must have been fine-tuned with `configs/tactile_g1_config.py`
(action horizon 50). If the port is busy a server is already running; do not
start a second one.

### 2. Port forward (GPU server host)

Needed only when the policy server runs inside a container whose port is not
published.

```bash
socat TCP-LISTEN:5555,fork,reuseaddr TCP:<container address>:5555
```

Check from the robot:

```bash
python -c "from agentic_tactile_vla.bridge.groot_client import Gr00tClient as C; import os; print(C(os.environ['POLICY_HOST'], int(os.environ['POLICY_PORT'])).ping())"
```

This must print `True` before continuing.

### 3. Camera server (robot)

```bash
cd ~/GR00T-WholeBodyControl && source .venv_camera/bin/activate
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera usb --ego-view-device-id <video device id> --port $CAMERA_PORT
```

The camera must deliver 1280x720. The bridge resizes it to the resolution the
policy was trained on. If the camera delivers another resolution the policy
sees inputs it was not trained on and the run is not valid.

### 4. BrainCo hand service (robot) — exactly one instance

```bash
cd <brainco hand service directory>
sudo ./bin/brainco_hand_server -n $HAND_IFACE
```

The service owns the serial connection to the hands and publishes
`rt/brainco/{left,right}/state`. Two instances on the same serial port corrupt
each other's frames.

### 5. Whole-body controller (robot) — WAIT for "Init done"

The robot moves into its standing pose during this step.

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real
```

### 6. Keyboard publisher (robot)

```bash
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
python scripts/keyboard_publisher.py
```

### 7. Bridge with the supervisor (robot) — WAIT for "PolicyServer is reachable"

```bash
source ~/GR00T-WholeBodyControl/.venv_data_collection/bin/activate
pip install -e ".[robot]"          # once
HAND_ENABLE=1 SLEW_ENABLE=1 python scripts/run_agent.py \
  --hand left --embodiment-tag NEW_EMBODIMENT \
  --prompt "pick up the object and place it down" 2>&1 | tee bridge.log
```

`unitree_sdk2py` must be importable in this environment (add its checkout to
`PYTHONPATH` if it is not installed).

Without `HAND_ENABLE=1` hand commands are printed and not sent. Variants:

```bash
python scripts/run_agent.py --no-agent  ...     # base VLA
python scripts/run_agent.py --observe   ...     # supervisor computes and logs, never overrides
python scripts/run_agent.py --variant U-stall ...   # ablation variants
python scripts/run_agent.py --vlm ...           # mode selection by the VLM service
python scripts/run_agent.py --handling-cue gently ...   # cue for the interaction parser only
```

Supervisor events can be followed with

```bash
tail -f bridge.log | grep --line-buffered "\[agent\]"
```

### 8. Start moving

In the keyboard publisher terminal, with a hand on the emergency stop:

1. `k` starts the whole-body control loop
2. `i` moves to the initial pose and switches to POSE mode (required)
3. `p` releases the pause

`i` at any time returns to the initial pose and resets the supervisor. `g`
freezes body-reference progression by hand.

## Optional: VLM service (GPU server)

```bash
pip install -e ".[vlm]"
python scripts/vlm_service.py --port 5599
```

The first start downloads Qwen2.5-VL-7B-Instruct. Set `VLM_HOST` and `VLM_PORT`
on the robot and pass `--vlm` to `run_agent.py`. `--backend mock` starts the
service without a model for a connectivity check.

## Compliant mode on hardware

The compliant controller (`compliant.py`) produces a current-limited command.
The DDS hand interface used by the bridge accepts position commands only, so
applying that command needs a current-mode driver for the hand with exclusive
access to its serial port. No such driver ships with this repository. Attach
one by assigning an object with a `send_current(raw_command)` method to
`compliant_sink` in `bridge/vla_bridge.py`. Without it, the compliant branch
holds the hand at the position it had when the branch engaged, so the
position path applies no squeeze, but no compliant closure takes place either.

Compliant mode covers closure and holding only. It does not support
autonomous transport.

## Logs

| Path | Content |
|---|---|
| `$AGENT_LOG_DIR/verdicts_<hand>_<time>.csv` | one row per assessed closure |
| `$AGENT_LOG_DIR/ramp_<hand>_<time>.csv` | one row per tick inside ADJUST |
| `bridge.log` | console output of the bridge and the supervisor |

Formats: `docs/data_format.md`.

## Troubleshooting

| Symptom | Action |
|---|---|
| `PolicyServer not reachable` | policy server down or port forward missing |
| `waiting for state msg` | press `k` |
| actions are sent, robot does not move | press `i` |
| `No cached chunk yet` forever | the policy server does not answer |
| robot stopped | `p` was pressed an even number of times |
| no `[agent]` lines | the supervisor is disabled (`--no-agent`) or the hand publishes no state |
| `[agent] step failed, disabling` | the supervisor raised; the bridge continues as base VLA |
| hand does not move | `HAND_ENABLE=1` missing, or the hand service is not running |
| camera does not open | another process holds the video device |
