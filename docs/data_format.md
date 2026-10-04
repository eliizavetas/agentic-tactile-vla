# Data formats

## What is in `data/`

Per-trial records for the evaluation tables are not part of this release.
`data/raw/` contains August development logs recorded with an earlier version
of the supervisor; they document the log format and can be replayed through
the detector, but do not reproduce the paper's tables.

All files are shipped exactly as recorded. Nothing was generated, resampled or
edited. Some files are empty (zero bytes) because a run was started and stopped
before any event; they are kept for completeness.

Finger names in these logs are `thumb`, `thumb_aux`, `index`, `middle`, `ring`,
`pinky`. They correspond to channels 0 to 5: thumb flexion, thumb opposition,
index, middle, ring, little. All logs are from the left hand.

## `data/raw/ticks_<date>_<time>.csv`

One row per bridge tick: what was commanded and what the hand measured.

| Column | Meaning |
|---|---|
| `t` | time since the recorder started, seconds |
| `chunk_id` | running index of the action chunk in use |
| `idx` | index of the executed action inside that chunk |
| `body_jump` | L2 distance between consecutive motion tokens; `nan` on the first row |
| `cmd_<finger>` | hand target sent for each of the six channels, normalized |
| `q_<finger>` | measured position for each of the six channels, normalized |

These files contain no effort signal.

## `data/raw/ramp_left_<date>_<time>.csv`

One row per control tick inside an ADJUST episode of the earlier supervisor.

| Column | Meaning |
|---|---|
| `t` | wall-clock time, seconds |
| `attempt` | index of the ADJUST episode in this file |
| `tick` | tick index inside the episode |
| `<finger>_cmd` | reference issued to the finger |
| `<finger>_q` | measured position |
| `<finger>_err` | reference minus measured position |
| `<finger>_taudev` | effort deviation from the baseline taken at the start of the episode, device units |
| `<finger>_taudur` | continuous time the deviation has exceeded the threshold, seconds |
| `<finger>_done` | 1 once the supervisor stopped working on the finger, for any reason |

Fingers: `thumb`, `index`, `middle`, `ring`, `pinky`. The four earliest files
(15 and 18 August) have no `thumb_*` columns, because the thumb was not
corrected at that time. `<finger>_done` is not a contact flag: it also rises on
travel exhaustion and at timeout.

## `data/raw/verdicts_left_<date>_<time>.csv`

One row per closure assessed by the earlier supervisor.

| Column | Meaning |
|---|---|
| `t` | wall-clock time of the verdict |
| `attempt` | index matching `attempt` in the ramp file of the same session |
| `label` | verdict issued by the earlier supervisor |
| `actual` | optional manual annotation, mostly empty |
| `profile`, `prompt` | handling profile and instruction; absent in the four files of 15 August |
| `n_contacts` | number of fingers credited with contact |
| `depth_mean`, `depth_spread` | mean and spread of the stopping positions of credited fingers |
| `tau_peak_max` | largest effort deviation seen in the episode, device units |
| `repeat_of_previous`, `n_since_success` | bookkeeping of consecutive verdicts |
| `timed_out` | 1 if the episode ended on the timeout |
| `<finger>_contact` | 1 if the detector credited this finger |
| `<finger>_q` | measured position when it was credited |
| `<finger>_why` | which test fired |
| `<finger>_err`, `<finger>_taupk`, `<finger>_taudur` | tracking error, peak effort deviation and its duration |

Important: the `label` column was produced by the earlier acceptance rule, not
by eq. (8) of the paper. `analysis/compute_verdicts.py` applies eq. (8) to the
recorded per-finger flags and reports how many rows differ.

## `data/raw/marks_<date>_<time>.csv`

Operator markers pressed at the start of each attempt.

| Column | Meaning |
|---|---|
| `n` | marker index |
| `t_wall` | wall-clock time |
| `t_rel` | time since the marker tool started, seconds |
| `label` | text typed by the operator, usually the attempt number |

## `data/processed/`

Tables assembled by hand and by script from the raw logs and the operator's
ledger during the August development sessions, typically five attempts per
object and condition. They are development records, not the evaluation reported in the
paper.

| File | Content |
|---|---|
| `baseline_20260817_perattempt.csv`, `baseline_20260817_summary.csv` | base VLA on training objects |
| `baseline_unseen_20260819_*.csv` | base VLA on objects outside the training set |
| `unseen_20260819_perattempt_corrected.csv` | corrected timing columns for the 19 August session |
| `agent_20260818_summary.csv`, `agent_20260818_perverdict.csv` | earlier supervisor, first session |
| `agent_20260819_perattempt.csv`, `agent_20260819_summary.csv` | earlier supervisor on training objects |
| `agent_unseen_20260820_*.csv` | earlier supervisor on objects outside the training set |
| `agent_combined_perattempt.csv`, `agent_perattempt_full.csv`, `baseline_perattempt_full.csv` | merged per-attempt tables |
| `summary_agent_all_objects.csv`, `summary_base_all_objects.csv` | per-object summaries |
| `cup_normal_20260820_perattempt.csv`, `cup_gentle_20260820_perattempt.csv` | thin-walled cup sessions |

Common columns:

| Column | Meaning |
|---|---|
| `condition` | base VLA or supervisor |
| `object`, `object_set` | object name and whether it was a training object |
| `n` | attempt number |
| `approaches_ledger`, `first_try`, `grasped`, `placed` | operator ledger; empty where not recorded |
| `robot_span_s`, `lead_in_s`, `mark_to_mark_s`, `window_s` | timing derived from the tick logs |
| `max_flex` | largest mean flexion command in the attempt |
| `closure_events`, `n_verdicts`, `verdict_labels` | closures and verdicts of the earlier supervisor |
| `object_intact`, `dropped_in_transport`, `notes` | operator observations |

Empty cells mean the quantity was not recorded. They are not zeros.

## Logs written by this repository

`recorder.py` writes two files per run into `AGENT_LOG_DIR`.

`verdicts_<hand>_<timestamp>.csv`, one row per assessed closure:

| Column | Meaning |
|---|---|
| `t` | time of the row |
| `episode` | index of the assessed closure |
| `intervened` | 1 if eq. (3) triggered ADJUST, 0 if the supervisor stayed transparent |
| `pre_support` | position-support latches at the assessment |
| `verdict` | FORMED, EMPTY, CROOKED, or empty when no correction ran |
| `support` | final support count |
| `retained`, `timed_out`, `duration_s` | outcome of the correction |
| `mode`, `mode_source` | interaction mode and what selected it: `language`, `vlm` or `default` |
| `<channel>_latched`, `<channel>_source` | support latch and the test that set it, `stall` or `effort` |
| `<channel>_exhausted`, `<channel>_q` | travel exhausted, and measured position at the verdict |

`ramp_<hand>_<timestamp>.csv`, one row per tick inside ADJUST:

| Column | Meaning |
|---|---|
| `t`, `episode`, `tick` | time, correction episode, tick index |
| `<channel>_ref`, `<channel>_q` | issued reference and measured position |
| `<channel>_effort_dev`, `<channel>_effort_dur` | `abs(e - b)` and `D` |
| `<channel>_eligible`, `<channel>_latched` | detector eligibility and latch |

Channels: `thumb_flexion`, `index`, `middle`, `ring`, `little`.
