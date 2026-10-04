#!/usr/bin/env python3
"""
Plot one correction event from a recorded ramp log, using the verdict log as
the authority on which fingers the detector credited with contact.

    python scripts/figures/fig_event.py data/raw/ramp_left_20260819_022049.csv          # list
    python scripts/figures/fig_event.py data/raw/ramp_left_20260819_022049.csv -a 9     # plot
    python scripts/figures/fig_event.py data/raw/ramp_left_20260819_022049.csv -a 9 -o out/ev

All values are read from the two CSV files; nothing is hardcoded. The logs
under data/raw/ were recorded in August with an earlier version of the
supervisor, so the contact flags are that version's decisions.

Why two files. In the ramp log, <finger>_done rises whenever the agent stops
working on that finger -- on contact, on travel exhaustion, or at timeout --
so it is not a contact flag. The verdict log records the detector's actual
per-finger decision in <finger>_contact. This script therefore reads WHICH
fingers made contact from the verdict file, and WHEN from the ramp, as the
first tick at which that finger's command stopped advancing. Fingers that only
exhausted their travel are drawn without a marker and listed in the legend
note, since distinguishing the two is the point of the figure.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

from agentic_tactile_vla import SupervisorConfig

FING = ['thumb', 'index', 'middle', 'ring', 'pinky']
NICE = {'pinky': 'little'}
COL  = {'thumb': '#7a5195', 'index': '#003f5c', 'middle': '#ef5675',
        'ring': '#2f8f6b', 'pinky': '#bc8b00'}
DELTA_E = SupervisorConfig().delta_e
STEP_EPS = 1e-4


def verdict_path(ramp_path):
    d, b = os.path.split(os.path.abspath(ramp_path))
    m = re.search(r'ramp_(\w+?)_(\d{8})_(\d{6})\.csv$', b)
    if not m:
        return None
    hand, day, hms = m.groups()
    for delta in (0, 1, -1, 2, -2):
        p = os.path.join(d, f'verdicts_{hand}_{day}_{int(hms)+delta:06d}.csv')
        if os.path.exists(p):
            return p
    return None


def load(ramp_path, attempt, vpath):
    r = pd.read_csv(ramp_path)
    need = ['t', 'attempt', 'tick'] + [f'{f}_{s}' for f in FING
                                       for s in ('cmd', 'q', 'taudev')]
    miss = [c for c in need if c not in r.columns]
    if miss:
        sys.exit(f'ramp file is missing: {miss}')

    v = None
    if vpath and os.path.exists(vpath):
        v = pd.read_csv(vpath)
        if not all(f'{f}_contact' in v.columns for f in FING):
            print('note: verdict file has no *_contact columns; no markers')
            v = None
    else:
        print('note: no matching verdict file; no markers')

    if attempt is None:
        print(f'{os.path.basename(ramp_path)}: {len(r)} ticks, '
              f'{r.attempt.nunique()} attempts')
        if v is not None:
            print(f'verdicts: {os.path.basename(vpath)}, {len(v)} rows\n')
            print('attempt  ticks   dur   label     contacts')
            for a, g in r.groupby('attempt'):
                row = v[v.attempt == a]
                if row.empty:
                    lbl, names = '--', ''
                else:
                    row = row.iloc[0]
                    lbl = row.get('label', '--')
                    names = ', '.join(NICE.get(f, f) for f in FING
                                      if row.get(f'{f}_contact', 0) == 1)
                print(f'{a:>7}  {len(g):>5}  {g.t.max()-g.t.min():>5.2f}s  '
                      f'{lbl:<9} {names or "none"}')
        else:
            for a, g in r.groupby('attempt'):
                print(f'{a:>7}  {len(g):>5}  {g.t.max()-g.t.min():>5.2f}s')
        print('\nre-run with  -a <attempt>. Pick one with two or more '
              'contacts stopping at different depths.')
        sys.exit(0)

    g = r[r.attempt == attempt].copy()
    if g.empty:
        sys.exit(f'attempt {attempt} not in the ramp file')
    g['ts'] = g.t - g.t.iloc[0]

    contacts, vrow = {}, None
    if v is not None:
        row = v[v.attempt == attempt]
        if not row.empty:
            vrow = row.iloc[0]
            contacts = {f: bool(vrow.get(f'{f}_contact', 0)) for f in FING}
    return g, contacts, vrow


def stop_tick(g, f):
    c = g[f'{f}_cmd'].to_numpy()
    idx = np.flatnonzero(np.diff(c) <= STEP_EPS)
    return int(idx[0] + 1) if idx.size else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('-a', '--attempt', type=int, default=None)
    p.add_argument('-o', '--out', default='fig_event')
    p.add_argument('-v', '--verdicts', default=None)
    args = p.parse_args()

    vpath = args.verdicts or verdict_path(args.csv)
    g, contacts, vrow = load(args.csv, args.attempt, vpath)

    mpl.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.linewidth': 0.6})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.5, 3.9), sharex=True,
        gridspec_kw={'height_ratios': [2.1, 1.0], 'hspace': 0.10})
    fig.subplots_adjust(left=0.175, right=0.98, top=0.90, bottom=0.115)

    ts = g.ts.to_numpy(); T = float(ts.max())
    print(f'attempt {args.attempt}: {len(g)} ticks, {T:.2f}s'
          + (f", label {vrow['label']}" if vrow is not None else ''))

    ceiling_only = []
    for f in FING:
        c = COL[f]
        ax1.plot(ts, g[f'{f}_cmd'], c=c, lw=0.9, ls=(0, (3, 2)), alpha=.6)
        ax1.plot(ts, g[f'{f}_q'], c=c, lw=1.4, label=NICE.get(f, f))
        ax2.plot(ts, g[f'{f}_taudev'].abs(), c=c, lw=1.0)
        k = stop_tick(g, f); made = contacts.get(f, False)
        if k is not None and made:
            tc = float(ts[k]); qv = float(g[f'{f}_q'].to_numpy()[k])
            ax1.plot(tc, qv, 'o', c=c, ms=4.5, zorder=6, mec='white', mew=0.6)
            ax1.axvline(tc, c=c, lw=0.6, alpha=.28, zorder=0)
            ax2.axvline(tc, c=c, lw=0.6, alpha=.28, zorder=0)
        elif k is not None:
            ceiling_only.append(NICE.get(f, f))
        print(f'  {NICE.get(f,f):<7} {"contact" if made else "no contact":<11}'
              f' stopped at {f"{ts[k]:.2f}s" if k is not None else "  --  "}'
              f'  final q = {g[f"{f}_q"].iloc[-1]:.3f}')

    ax1.set_ylabel('normalized finger position'); ax1.set_xlim(0, T)
    ax1.legend(ncol=5, fontsize=6.5, frameon=False, loc='upper left',
               bbox_to_anchor=(0, 1.13), columnspacing=1.0, handlelength=1.5)
    txt = ('dashed: command $u_i$\nsolid: measured $q_i$\n'
           'dot: detector-credited contact')
    if ceiling_only:
        txt += '\nno dot: travel exhausted (' + ', '.join(ceiling_only) + ')'
    ax1.text(0.985, 0.04, txt, transform=ax1.transAxes, fontsize=6.3,
             color='0.35', ha='right', va='bottom', linespacing=1.5)

    ax2.axhline(DELTA_E, c='k', lw=0.7, ls=':')
    ax2.text(0.02 * T, DELTA_E + 0.02, r'$\delta_e$', fontsize=7.5)
    ax2.set_ylabel('effort deviation\n' + r'$|e_i-b_i|$')
    ax2.set_xlabel('time since adjustment start (s)'); ax2.set_xlim(0, T)
    for a in (ax1, ax2):
        a.spines[['top', 'right']].set_visible(False)

    plt.savefig(args.out + '.pdf'); plt.savefig(args.out + '.png', dpi=200)
    print(f'wrote {args.out}.pdf / .png')


if __name__ == '__main__':
    main()
