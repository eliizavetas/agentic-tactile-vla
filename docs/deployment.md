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

The DDS hand interface used by the bridge accepts position commands only.
Compliant closure is executed by a separate service in
`drivers/brainco_compliant/` that drives the hand in current mode over its
serial port. The service files were recovered from the robot; they were not
rebuilt or re-run when this repository was assembled.

Compliant mode covers closure and holding only. It does not support
autonomous transport.

### How the handover works

The serial port of a hand has one owner. When the supervisor engages the
compliant branch, the client `impedance_grasp.py` does the following in a
background thread:

1. remembers how `brainco_hand_server` was started and stops it
2. starts the grasp service for the supervised hand and waits for `READY`
3. sends `START sphere gentle` and waits for `HOLDING`
4. when the policy reopens the hand (or on key `i`), stops the service and
   restarts `brainco_hand_server`

The handover takes several seconds. While the service owns the port the hand
publishes no DDS state, so the hand positions in the policy observation are
the last ones received before the handover. The hand must be open when the
handover starts; the supervisor engages the branch when the closure is armed,
before the fingers close.

### Build

```bash
cd drivers/brainco_compliant
STARK_SDK_DIR=<path to the BrainCo Stark SDK> bash build_brainco_grasp_service.sh
```

This builds `brainco_grasp_service` and the three gate diagnostics into
`build/`. `CMakeLists.txt` was written for this repository because the original
build file was not recovered; adjust the SDK library name in it if your SDK
version uses another one.

### Gate files

The service refuses to start unless it is given two gate files. A gate file is
the `*_meta.txt` record of a diagnostic run on the same hand: plain
`key=value` lines with the hand's serial number, its firmware version, the
controller version, the current limit used and the result. The service reads
them at startup and checks that the hand it is connected to has passed the
six-finger current-mode diagnostic at each current limit it is about to use.
The files are specific to a hand and a firmware version. They are not shipped
with this repository and must be created on your own hardware.

The diagnostics form a chain; each stage requires the gate file of the one
before it.

| Stage | Program | What it checks | Passing record must contain |
|---|---|---|---|
| H0 | `h0_index_current_visible` | current-mode commands on the index finger produce visible motion and feedback | `test=h0_index_current_visible_v1`, a passing result |
| H2 | `h2_index_impedance` | the impedance law on the index finger at raw limit 200 | `controller_version=index_impedance_v9`, `current_limit_raw=200`, `returned_open=1`, `zero_confirmed=1`, `result=SAFE_RUN_PASS` |
| H3 | `h3_spherical_grasp` | the six-finger grasp, once per profile | `controller_version=spherical_impedance_v3`, the profile and its `current_limit_raw`, `moved_drive_count=6`, `contact_like_count=6`, `returned_open=1`, `position_mode_restored=1`, `result=SAFE_RUN_PASS` |

To create them, stop `brainco_hand_server` first, keep hands clear of the
fingers, and run each stage with the hand's serial device and serial number
(printed by `brainco_hand_server` at startup). Each program asks for an
explicit confirmation flag and writes `<output prefix>.csv` and
`<output prefix>_meta.txt`.

```bash
cd drivers/brainco_compliant && mkdir -p data_out/h0 data_out/h2 data_out/h3

sudo ./build/h0_index_current_visible --confirm-current-test \
  --port <serial device> --expected-serial <hand serial> \
  --output-prefix data_out/h0/current_visible

sudo ./build/h2_index_impedance --confirm-impedance-test \
  --port <serial device> --expected-serial <hand serial> \
  --h0-gate-meta data_out/h0/current_visible_meta.txt \
  --output-prefix data_out/h2/index_impedance

# H3 needs an object in the open palm; run it once per profile
sudo ./build/h3_spherical_grasp --confirm-spherical-test --profile hard \
  --port <serial device> --expected-serial <hand serial> \
  --h2-gate-meta data_out/h2/index_impedance_meta.txt \
  --output-prefix data_out/h3/hard

sudo ./build/h3_spherical_grasp --confirm-spherical-test --profile soft_low \
  --port <serial device> --expected-serial <hand serial> \
  --h2-gate-meta data_out/h2/index_impedance_meta.txt \
  --output-prefix data_out/h3/soft_low
```

A run that does not end with `result=SAFE_RUN_PASS` is not a gate; find the
cause before repeating it. Then point the service at the two H3 records:

```bash
export BRAINCO_GRASP_NOMINAL_GATE=$PWD/data_out/h3/hard_meta.txt      # raw limit 200
export BRAINCO_GRASP_GENTLE_GATE=$PWD/data_out/h3/soft_low_meta.txt   # raw limit 150
```

Notes on the recovered sources:

* H0 and H2 address the right hand only. H3 and the service select the hand
  through `BRAINCO_GRASP_SLAVE` (`0x7f` right, `0x7e` left).
* The service normally requires the serial number in the gate files to match
  the connected hand. `BRAINCO_GRASP_GATE_ANY_SERIAL=1` waives only that
  comparison, for gate files recorded on the other hand of the same model. All
  other gate conditions still apply.
* The commands above were written from the programs' usage text and gate
  checks. They were not executed when this repository was assembled.

### Run with the bridge

Set the per-hand values in `.env` (`BRAINCO_GRASP_PORT_LEFT`,
`BRAINCO_GRASP_SERIAL_LEFT`, the two gate variables), then:

```bash
sudo -v      # the client calls sudo non-interactively; warm the ticket first
COMPLIANT_DRIVER=1 HAND_ENABLE=1 python scripts/run_agent.py --hand left \
  --handling-cue gently --embodiment-tag NEW_EMBODIMENT --prompt "..."
```

Without `COMPLIANT_DRIVER=1` the compliant branch holds the hand at the
position it had when the branch engaged: the position path applies no squeeze,
and no compliant closure takes place.

The service can also be driven by hand, with `brainco_hand_server` stopped:

```bash
sudo BRAINCO_GRASP_PORT=<serial device> BRAINCO_GRASP_SERIAL=<hand serial> \
  BRAINCO_GRASP_NOMINAL_GATE=$BRAINCO_GRASP_NOMINAL_GATE \
  BRAINCO_GRASP_GENTLE_GATE=$BRAINCO_GRASP_GENTLE_GATE \
  bash run_brainco_grasp_service.sh          # terminal 1, leave running
./brainco-grasp run sphere gentle            # terminal 2; Ctrl+C releases
./brainco-grasp status
```

Safety behaviour of the service: `RELEASE` opens the hand normally, `ABORT`
opens it and ends in a fault state, and on any watchdog, read, current or
state error the service sends zero current and returns the hand to an open
pose in position mode. Beyond equation (9) it contains a closing-speed cap and
a setpoint ramp; see `drivers/brainco_compliant/README.md`.

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
