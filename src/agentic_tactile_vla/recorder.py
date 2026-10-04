# Implements the paper specification. Not re-validated on hardware since it was written.
"""CSV logging of supervisor decisions.

Two files are written per run, in the directory given by AGENT_LOG_DIR:

    verdicts_<hand>_<timestamp>.csv   one row per assessed closure
    ramp_<hand>_<timestamp>.csv       one row per control tick inside ADJUST

The schema is described in docs/data_format.md. It differs from the August
development logs under data/raw/, which were written by an earlier version of
the supervisor.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Optional

from .config import CHANNEL_NAMES, FLEXION_CHANNELS
from .supervisor import State, SupervisorOutput


class Recorder:
    def __init__(self, hand: str = "left", directory: Optional[str] = None) -> None:
        directory = directory or os.environ.get("AGENT_LOG_DIR", "agent_logs")
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.verdict_path = os.path.join(directory, f"verdicts_{hand}_{stamp}.csv")
        self.ramp_path = os.path.join(directory, f"ramp_{hand}_{stamp}.csv")
        self._verdict_file = open(self.verdict_path, "w", newline="")
        self._ramp_file = open(self.ramp_path, "w", newline="")
        self._verdicts = csv.writer(self._verdict_file)
        self._ramp = csv.writer(self._ramp_file)
        names = [CHANNEL_NAMES[i] for i in FLEXION_CHANNELS]

        header = ["t", "episode", "intervened", "pre_support", "verdict", "support",
                  "retained", "timed_out", "duration_s", "mode", "mode_source"]
        for n in names:
            header += [f"{n}_latched", f"{n}_source", f"{n}_exhausted", f"{n}_q"]
        self._verdicts.writerow(header)

        header = ["t", "episode", "tick"]
        for n in names:
            header += [f"{n}_ref", f"{n}_q", f"{n}_effort_dev", f"{n}_effort_dur",
                       f"{n}_eligible", f"{n}_latched"]
        self._ramp.writerow(header)
        self._tick = 0
        self._episode = 0

    def log(self, t: float, out: SupervisorOutput, q) -> None:
        if out.state is State.ADJUST and out.evidence is not None:
            if out.event == "ADJUST":
                self._tick = 0
                self._episode += 1
            row = [f"{t:.3f}", self._episode, self._tick]
            for i in FLEXION_CHANNELS:
                ev = out.evidence[i]
                row += [f"{out.hand_target[i]:.4f}", f"{q[i]:.4f}",
                        f"{ev.effort_deviation:.4f}", f"{ev.effort_duration:.3f}",
                        int(ev.eligible), int(ev.latched)]
            self._ramp.writerow(row)
            self._tick += 1

        rec = out.episode
        if rec is not None:
            row = [f"{t:.3f}", rec.index, int(rec.intervened), rec.pre_support,
                   rec.verdict.value if rec.verdict else "", rec.support,
                   int(rec.retained), int(rec.timed_out), f"{rec.duration:.3f}",
                   rec.mode, rec.mode_source]
            for i in FLEXION_CHANNELS:
                row += [int(rec.latched[i]), rec.sources[i], int(rec.exhausted[i]),
                        f"{rec.q_final[i]:.4f}"]
            self._verdicts.writerow(row)
            self._verdict_file.flush()
            self._ramp_file.flush()

    def close(self) -> None:
        self._verdict_file.close()
        self._ramp_file.close()
