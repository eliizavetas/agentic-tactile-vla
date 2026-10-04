# Data

Per-trial records for the evaluation tables are not part of this release.
`data/raw/` contains August development logs recorded with an earlier version
of the supervisor; they document the log format and can be replayed through
the detector, but do not reproduce the paper's tables.

| Directory | Content |
|---|---|
| `raw/` | 155 CSV logs from 11 to 20 August 2026, left hand: `ticks_*`, `ramp_left_*`, `verdicts_left_*`, `marks_*` |
| `processed/` | per-attempt and summary tables assembled during the same sessions |

The files are unchanged copies of the recorded logs. Six of them are empty
because the run produced no event.

The earlier supervisor that wrote these logs differs from the one in `src/`:
it used a different correction step, a different acceptance rule and a
different closure trigger. Its `label` column is therefore not the verdict of
eq. (8) in the paper.

The bridge console logs of the cup sessions (`bridge_cup_*.log`,
`bridge_gentle_*.log`) are not included, because they contain addresses and
device identifiers of the lab network and cannot be shipped unchanged.

Column descriptions: `docs/data_format.md`.

```bash
# detector of this repository on recorded signals
python scripts/replay_logs.py data/raw/ramp_left_20260820_191527.csv
python scripts/replay_logs.py data/raw/ticks_20260820_191250.csv

# eq. (8) applied to the recorded per-finger flags
python analysis/compute_verdicts.py data/raw/verdicts_left_*.csv

# one correction event
python scripts/figures/fig_event.py data/raw/ramp_left_20260819_022049.csv -a 9
```
