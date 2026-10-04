#!/usr/bin/env python3
"""Apply the paper's verdict rule (eq. 8) to recorded per-finger support flags.

Reads verdicts_*.csv files (format in docs/data_format.md), counts the
monitored flexion channels with a recorded contact flag and maps the count to
FORMED / EMPTY / CROOKED:

    FORMED   at least three supports
    EMPTY    no support
    CROOKED  otherwise (insufficient support evidence, not misalignment)

    python analysis/compute_verdicts.py data/raw/verdicts_left_*.csv
    python analysis/compute_verdicts.py data/raw/verdicts_left_*.csv --out verdicts_eq8.csv

The August logs under data/raw/ were written by an earlier version of the
supervisor, whose own `label` column used a different acceptance rule. The
script reports both, so the difference is visible. The per-finger flags are
that version's detector decisions; they are read as recorded and not altered.
"""

import argparse
import csv
import os
from collections import Counter

MIN_SUPPORT = 3
# flexion channels as named in the August logs (thumb_aux is thumb opposition)
FLEXION = ["thumb", "index", "middle", "ring", "pinky"]
NEW_FLEXION = ["thumb_flexion", "index", "middle", "ring", "little"]


def verdict(count: int) -> str:
    if count >= MIN_SUPPORT:
        return "FORMED"
    return "EMPTY" if count == 0 else "CROOKED"


def support_count(row: dict) -> int:
    if "thumb_contact" in row:                      # August log schema
        cols = [f"{f}_contact" for f in FLEXION]
    else:                                           # schema written by recorder.py
        cols = [f"{f}_latched" for f in NEW_FLEXION]
    return sum(1 for c in cols if str(row.get(c, "")).strip() in ("1", "1.0", "True"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None, help="write one row per verdict to this CSV")
    args = ap.parse_args()

    rows, total, logged = [], Counter(), Counter()
    for path in args.csv:
        if os.path.getsize(path) == 0:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                count = support_count(row)
                v = verdict(count)
                old = row.get("label") or row.get("verdict") or ""
                total[v] += 1
                logged[old] += 1
                rows.append({"file": os.path.basename(path),
                             "attempt": row.get("attempt") or row.get("episode") or "",
                             "support": count, "verdict_eq8": v, "logged_label": old})

    n = len(rows)
    print(f"{n} verdict rows in {len(args.csv)} files")
    print("  rule of eq. 8:   " + "  ".join(f"{k} {total[k]}" for k in ("FORMED", "CROOKED", "EMPTY")))
    print("  label in the log: " + "  ".join(f"{k or '(none)'} {v}" for k, v in sorted(logged.items())))
    changed = sum(1 for r in rows if r["logged_label"] and r["logged_label"] != r["verdict_eq8"])
    print(f"  rows where the logged label differs from eq. 8: {changed}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
