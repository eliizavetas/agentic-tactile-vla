#!/usr/bin/env python3
"""Replay recorded logs through the detector of this repository, offline.

Two kinds of log can be replayed (formats in docs/data_format.md):

ticks_*.csv   commanded and measured finger positions for a whole run.
              Closure detection and the position-stall test (eq. 6) are
              replayed. These files contain no effort signal, so the
              effort-duration test (eq. 7) cannot be evaluated on them.

ramp_*.csv    per-tick command, position and effort deviation inside each
              ADJUST episode. Both tests are replayed and the verdict rule of
              eq. 8 is applied to the result.

    python scripts/replay_logs.py data/raw/ticks_20260820_191250.csv
    python scripts/replay_logs.py data/raw/ramp_left_20260820_191527.csv

The logs under data/raw/ were recorded in August with an earlier version of
the supervisor (different correction step and acceptance rule). A replay shows
what this detector reports on those signals. It does not reproduce the paper's
tables, and the hand was driven by the earlier controller, not by this one.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from agentic_tactile_vla import (ClosureDetector, ContactDetector, SupervisorConfig,
                                 verdict_from_support)
from agentic_tactile_vla.config import CHANNEL_NAMES, FLEXION_CHANNELS

# column names used in the August logs, in paper channel order
LOG_NAMES = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]


def replay_ticks(path: str, cfg: SupervisorConfig) -> None:
    df = pd.read_csv(path)
    need = ["t"] + [f"cmd_{n}" for n in LOG_NAMES] + [f"q_{n}" for n in LOG_NAMES]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    closure, detector = ClosureDetector(cfg), ContactDetector(cfg)
    flex = list(FLEXION_CHANNELS)
    armed, started, arm_cmd, peak, below_since = False, False, None, 0.0, None
    closures = []
    prev_u = None
    for row in df.itertuples(index=False):
        t = float(row.t)
        u = np.array([getattr(row, f"cmd_{n}") for n in LOG_NAMES], dtype=float)
        q = np.array([getattr(row, f"q_{n}") for n in LOG_NAMES], dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(q))):
            continue
        closure.push(t, u)
        mean = float(u[flex].mean())
        if not armed:
            if closure.upcoming_closure(None, 0):
                armed, started, arm_cmd, peak, below_since = True, True, u[flex].copy(), mean, None
                detector.reset()
                t_arm = t
        else:
            detector.update(t, u if prev_u is None else prev_u, q)
            peak = max(peak, mean)
            reopened = mean <= peak - cfg.reopen_delta
            below_since = (below_since if below_since is not None else t) if reopened else None
            if below_since is not None and t - below_since >= cfg.reopen_hold:
                armed = False
            elif closure.terminal_phase():
                closures.append((t_arm, t, detector.support_count, detector.latched.copy()))
                armed = False
                # wait for the command to drop before arming again
                closure.reset()
        prev_u = u
    print(f"{os.path.basename(path)}: {len(df)} ticks, {len(closures)} closures reached "
          f"the terminal phase")
    print("  t_arm     t_terminal  position latches  eta (eq. 3)  latched fingers")
    for t_arm, t_term, count, latched in closures:
        names = ", ".join(CHANNEL_NAMES[i] for i in flex if latched[i]) or "none"
        eta = int(count < cfg.min_support)
        print(f"  {t_arm:8.2f}  {t_term:10.2f}  {count:16d}  {eta:11d}  {names}")
    print("  (effort is not recorded in ticks files; eq. 7 is not evaluated)")


def replay_ramp(path: str, cfg: SupervisorConfig) -> None:
    df = pd.read_csv(path)
    chan = {"thumb": 0, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
    # the earliest ramp files have no thumb columns (the thumb was not corrected)
    chan = {n: i for n, i in chan.items() if f"{n}_cmd" in df.columns}
    need = ["t", "attempt"] + [f"{n}_{s}" for n in chan for s in ("cmd", "q", "taudev")]
    missing = [c for c in need if c not in df.columns]
    if missing or not chan:
        sys.exit(f"{path}: missing columns {missing}")
    print(f"{os.path.basename(path)}: {df.attempt.nunique()} ADJUST episodes")
    print("  attempt  ticks  dur[s]  support  verdict (eq. 8)  latched (source)")
    for attempt, g in df.groupby("attempt"):
        detector = ContactDetector(cfg)
        detector.begin_adjust(np.zeros(6))      # taudev is already relative to the baseline
        prev_u = None
        for row in g.itertuples(index=False):
            u, q, e = np.zeros(6), np.zeros(6), np.zeros(6)
            for n, i in chan.items():
                u[i] = getattr(row, f"{n}_cmd")
                q[i] = getattr(row, f"{n}_q")
                e[i] = getattr(row, f"{n}_taudev")
            detector.update(float(row.t), u if prev_u is None else prev_u, q, e)
            prev_u = u
        sources = detector.sources()
        latched = ", ".join(f"{CHANNEL_NAMES[i]} ({sources[i]})"
                            for i in FLEXION_CHANNELS if detector.latched[i]) or "none"
        verdict = verdict_from_support(detector.support_count, cfg).value
        print(f"  {attempt:7d}  {len(g):5d}  {g.t.max() - g.t.min():6.2f}  "
              f"{detector.support_count:7d}  {verdict:15s}  {latched}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--position-only", action="store_true",
                    help="disable the effort-duration test (ramp files)")
    args = ap.parse_args()
    cfg = SupervisorConfig(use_effort=not args.position_only)
    for path in args.csv:
        if os.path.getsize(path) == 0:
            print(f"{os.path.basename(path)}: empty file, skipped")
            continue
        base = os.path.basename(path)
        if base.startswith("ticks_"):
            replay_ticks(path, cfg)
        elif base.startswith("ramp_"):
            replay_ramp(path, cfg)
        else:
            sys.exit(f"{base}: expected a ticks_*.csv or ramp_*.csv file")


if __name__ == "__main__":
    main()
