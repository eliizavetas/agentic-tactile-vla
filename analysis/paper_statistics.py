#!/usr/bin/env python3
"""Recompute the interval estimates and paired tests reported in the paper from
the counts printed in its tables and text.

Per-trial records for the evaluation tables are not part of this release, so
this script does not regenerate the tables from raw data. It takes the
published counts as input and checks the statistics derived from them: Wilson
intervals, exact McNemar tests with Holm correction, the correction-episode
reduction and the retention-audit rates.

    python analysis/paper_statistics.py
"""

from stats import holm, mcnemar_exact, wilson_interval


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    print("Main comparison (Table II), 75 matched blocks")
    for name, k in [("Base VLA", 46), ("VLA + close-to-stall", 54), ("Supervisor", 63)]:
        lo, hi = wilson_interval(k, 75)
        print(f"  {name:22s} {k}/75 = {pct(k / 75)}%   95% Wilson {pct(lo)}-{pct(hi)}")
    # discordant block counts reported in Sec. IV-B: (first wins, second wins)
    contrasts = [("Supervisor vs base VLA", 20, 3), ("Supervisor vs close-to-stall", 11, 2),
                 ("Close-to-stall vs base VLA", 12, 4)]
    raw = [mcnemar_exact(b, c) for _, b, c in contrasts]
    for (name, b, c), p, p_adj in zip(contrasts, raw, holm(raw)):
        print(f"  {name:30s} discordant {b}/{c}   exact McNemar p = {p:.4f}   Holm p = {p_adj:.3f}")

    print("\nComponent ablation (Table III), 100 matched blocks")
    for name, k, corr in [("U-stall", 72, 135), ("U-full", 84, 134), ("S-position", 79, 49),
                          ("S-retain", 77, 47), ("S-full", 86, 46)]:
        lo, hi = wilson_interval(k, 100)
        print(f"  {name:11s} {k}/100   95% Wilson {pct(lo)}-{pct(hi)}   corrections {corr}")
    print(f"  correction episodes, U-full -> S-full: {pct((134 - 46) / 134)}% reduction")
    print("  paired tests for this table need per-block outcomes, which are not released")

    print("\nPose perturbation (Table IV), 50 matched trials per regime and condition")
    rows = [("Nominal", 32, 37, 43), ("Translation", 25, 32, 39), ("Yaw", 23, 30, 37)]
    for name, vla, stall, sup in rows:
        print(f"  {name:12s} VLA {vla}/50  stall {stall}/50  supervisor {sup}/50   "
              f"gain over VLA {2 * (sup - vla)} points")
    print(f"  pooled supervisor vs close-to-stall, discordant 26/6: "
          f"exact McNemar p = {mcnemar_exact(26, 6):.5f}")

    print("\nRetention audit (Table V), 120 closures")
    tp, fp, fn, tn = 70, 6, 18, 26   # accepted+retained, accepted+lost, rejected+retained, rejected+lost
    print(f"  retained given acceptance  {tp}/{tp + fp} = {pct(tp / (tp + fp))}%")
    print(f"  not retained given rejection (NPV)  {tn}/{fn + tn} = {pct(tn / (fn + tn))}%")
    print(f"  rejected although retainable  {fn}/{tp + fn} = {pct(fn / (tp + fn))}% (pooled)")
    print("  the held-out split (88.9% and 25.6% in the paper) needs per-closure records")


if __name__ == "__main__":
    main()
