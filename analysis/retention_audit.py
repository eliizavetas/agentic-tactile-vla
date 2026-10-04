#!/usr/bin/env python3
"""Retention audit: support verdict versus an external retention label.

The audit in the paper (Sec. IV-E) performs standardized closures, lifts the
grasp by 5 cm, holds for 2 s and records whether the object was retained,
independently of the supervisor's verdict.

Input is a CSV with one row per closure and the columns

    support    number of latched support channels at the verdict (0-5)
    retained   1 if the object was retained through the lift and hold, else 0
    object_set optional, e.g. "training" or "held_out"; adds a per-set breakdown

    python analysis/retention_audit.py my_audit.csv

The per-closure records behind the paper's Table V are not part of this
release, so no input file ships with the repository. To check the rates that
follow from the published table, pass its four cells:

    python analysis/retention_audit.py --counts 70 6 18 26
"""

import argparse
import csv
from collections import defaultdict

from stats import wilson_interval

MIN_SUPPORT = 3


def report(title, tp, fp, fn, tn):
    def line(label, k, n):
        if n == 0:
            print(f"  {label:38s} n/a")
            return
        lo, hi = wilson_interval(k, n)
        print(f"  {label:38s} {k}/{n} = {100 * k / n:5.1f}%   95% Wilson {100 * lo:.1f}-{100 * hi:.1f}")
    print(f"{title} (n = {tp + fp + fn + tn})")
    print(f"  {'':24s} retained  not retained")
    print(f"  {'at least three supports':24s} {tp:8d}  {fp:12d}")
    print(f"  {'below three supports':24s} {fn:8d}  {tn:12d}")
    line("retained given acceptance", tp, tp + fp)
    line("not retained given rejection (NPV)", tn, fn + tn)
    line("rejected although retainable", fn, tp + fn)
    line("accepted although not retained", fp, fp + tn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--counts", nargs=4, type=int,
                    metavar=("ACC_RET", "ACC_LOST", "REJ_RET", "REJ_LOST"))
    args = ap.parse_args()
    if args.counts:
        report("From table cells", *args.counts)
        return
    if not args.csv:
        ap.error("give a CSV file or --counts")
    cells = defaultdict(lambda: [0, 0, 0, 0])
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            accepted = int(row["support"]) >= MIN_SUPPORT
            retained = int(row["retained"]) == 1
            k = (0 if retained else 1) if accepted else (2 if retained else 3)
            for key in ("all", row.get("object_set") or None):
                if key:
                    cells[key][k] += 1
    for key, c in cells.items():
        report(f"object set: {key}", *c)


if __name__ == "__main__":
    main()
