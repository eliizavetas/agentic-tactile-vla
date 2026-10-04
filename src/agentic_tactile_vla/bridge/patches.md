# Bridge patches

`vla_bridge.py` is the VLA inference runner of GR00T-WholeBodyControl
(`gear_sonic/scripts/run_vla_inference.py`, Apache-2.0) with the changes below.
The file keeps the upstream structure so it can be diffed against the upstream
script.

The supervisor integration in this file targets the supervisor in
`src/agentic_tactile_vla/` and has not been re-run on hardware. The other
patches were carried over from the bridge that was used on the robot.

| Patch | Where | Why |
|---|---|---|
| Policy client | `groot_client.py`, import at the top | The stock client breaks when another package calls `msgpack_numpy.patch()`. The replacement converts arrays to the server wire format before packing. |
| Action horizon and inference rate | `InferenceConfig` | `action_horizon = 50` matches the fine-tuned modality config (`configs/tactile_g1_config.py`); the policy forward pass runs at 5 Hz. |
| BrainCo hand state in the observation | `_override_hand_state` | The policy was trained on the six measured BrainCo motor positions per hand. The upstream script fills the hand state from the whole-body state message, which does not contain them. |
| BrainCo hand commands | `_send_brainco_hands`, `_to7` | Hand targets go directly to the hand over DDS, as during data collection. The whole-body controller still receives a 7-dimensional hand vector, padded from the six motor values. `HAND_ENABLE=1` arms the hand; without it commands are printed only. |
| Camera transform (CAMFIX) | `prepare_observation_from_sensors` | The camera delivers 720x1280 while the training data was exported at 480x640. The same resize is applied before inference so the policy sees the training distribution. |
| Slew limits (SLEW) | `_slew_limit`, `_slew_hand` | Consecutive chunks can disagree at their boundary. With `SLEW_ENABLE=1` the per-tick change of the motion token and of the nominal hand command is capped (`SLEW_MAX`, `SLEW_HAND`). The hand cap is not applied to supervisor references. |
| Manual body freeze | key `g` | Toggles body-reference progression by hand, for bench tests. |
| Right thumb lock (THUMBLOCK) | main loop | Optional workaround for a faulty right thumb: `RIGHT_THUMB_LOCK=1` forces right thumb flexion to zero. Off by default. |
| Execution supervisor | main loop | See below. |

## Supervisor integration

The seam is the point in the main loop where the hand part of the action chunk
is still a full `(T, 6)` array, before it is indexed by time. Enabled with
`AGENT_ENABLE=1`; without it the supervisor is not constructed and the bridge
behaves as the base VLA.

1. Each tick the supervisor receives the hand chunk of the supervised hand
   (`AGENT_HAND`, default `left`), the current chunk index, the measured finger
   positions and the motor-effort proxy read from the hand, the instruction and
   the latest camera frame.
2. `override`: the supervisor reference replaces the nominal hand target of the
   supervised hand. Thumb opposition inside that reference is always the policy
   target.
3. `body_gate` (b_t = 1): the motion token is not sent and the chunk index does
   not advance, so progression of the body reference is paused. The whole-body
   controller keeps running and holds the last body reference.
4. `authority_transition`: the cached chunk is dropped, any inference result
   computed from an observation taken before the transition is discarded, and a
   new inference is triggered. While the new chunk is pending the supervisor
   keeps stepping on the hand and no motion token is sent.
5. Compliant branch: the supervisor returns a current-limited command
   (`compliant_command`). It is forwarded to `compliant_sink` when a
   current-mode hand driver is attached. No such driver ships with this
   repository, because the DDS hand interface is position-mode only. Without a
   sink the hand is held at the position it had when the branch engaged.
6. Key `i` (return to the initial pose) resets the supervisor.

Environment variables read by the bridge:

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_HOST`, `POLICY_PORT` | `localhost`, `5555` | GR00T policy server |
| `ROBOT_HOST`, `CAMERA_PORT` | `localhost`, `5555` | camera server |
| `HAND_IFACE` | unset | network interface used for the hand DDS channel |
| `HAND_ENABLE` | `0` | `1` sends hand commands, otherwise dry run |
| `AGENT_ENABLE` | `0` | `1` enables the supervisor |
| `AGENT_HAND` | `left` | supervised hand |
| `AGENT_OBSERVE` | `0` | `1` runs and logs the supervisor without overriding anything |
| `AGENT_VARIANT` | `S-full` | `S-full`, `S-position`, `S-retain`, `U-full`, `U-stall` |
| `AGENT_HANDLING_CUE` | unset | handling text given to the interaction parser only, so the policy prompt stays identical across conditions |
| `AGENT_LOG_DIR` | `agent_logs` | directory for verdict and ramp CSV files |
| `VLM_ENABLE`, `VLM_HOST`, `VLM_PORT` | `0`, `localhost`, `5599` | mode-selection VLM service |
| `SLEW_ENABLE`, `SLEW_MAX`, `SLEW_HAND` | `0`, `0.05`, `0.03` | slew limits |
| `RIGHT_THUMB_LOCK` | `0` | right thumb workaround |
