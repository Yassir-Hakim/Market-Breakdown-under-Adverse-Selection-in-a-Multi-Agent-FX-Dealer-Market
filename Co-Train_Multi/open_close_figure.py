'''
  open_close_figure.py -- the market opens in every run, then the fee spread
  decides whether it stays open. Three saved learning curves on one figure:
  T1-T4 co-trained with the market maker at the 1% and 2% spreads, and the 2%
  run without the tau anneal (robustness_runs.py, tau_const). All three start
  from the same weights, so their first readings coincide.

  Reads each run's cotrain_multi_curve.npz (the hard game on the validation
  paths at the end of every training block); trains nothing.
  Output: Results/cotrain_multi/market_opens_then_closes.png

  Usage: click Run (python open_close_figure.py).
'''
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'Results' / 'cotrain_multi'
RUNS = [   # label, folder under Results/cotrain_multi, colour, line style
    ('1% spread', 'T1-T2-T3-T4_mm0.01_bdc0.02_bid_size_2way', '#2a78d6', '-'),
    ('2% spread', 'T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way', '#eb6834', '-'),
    ('2% spread, no τ anneal', 'checks/T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_tau_const',
     '#eb6834', (0, (4, 3))),
]
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def curve(folder):
    c = np.load(RESULTS / folder / 'cotrain_multi_curve.npz', allow_pickle=True)
    steps = (np.asarray(c['step_mm']) + np.asarray(c['step_trader'])).astype(float)
    return steps, 100 * np.asarray(c['filled'], float), 1e4 * np.asarray(c['mm_pl'], float)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 6.3), sharex=True, facecolor=SURF,
                                      gridspec_kw=dict(hspace=0.12))
    for ax in (top, bottom):
        ax.set_facecolor(SURF)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color=INK2, lw=0.8)
    for label, folder, colour, style in RUNS:
        x, filled, mm = curve(folder)
        top.plot(x, filled, color=colour, lw=2, ls=style, label=label)
        bottom.plot(x, mm, color=colour, lw=2, ls=style, label=label)
        print(f'  {label:<24} peak filled {filled[:3].max():5.1f}%   last {filled[-1]:5.2f}%   '
              f'MM last {mm[-1]:+6.2f} bp   ({len(x)} readings)')
    x, filled, _ = curve(RUNS[1][1])
    top.annotate('2% spread', (x[-1], filled[-1]), xytext=(6, 0), textcoords='offset points',
                 va='center', fontsize=9)
    top.annotate('same start in all three: ~37% of\noffered volume filled after the\n'
                 'first market-maker block', (100, 36.6), xytext=(420, 29), textcoords='data',
                 fontsize=9, color=INK2, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    top.text(1250, 2.2, '1% spread and 2% without the anneal: closed', fontsize=9, color=INK2)
    top.set_ylabel('offered volume filled (%)')
    bottom.set_ylabel('market maker P/L (bp)')
    bottom.set_xlabel('training steps, market maker and traders alternating in blocks of 100')
    top.legend(frameon=False, loc='upper right', fontsize=9, handlelength=2.6)
    top.set_xlim(0, 4300)
    fig.suptitle('From the same open start, the 1% market closes; the 2% market can end open or closed',
                 x=0.075, ha='left', fontsize=12, y=0.97)
    fig.text(0.075, 0.915, 'T1–T4 co-trained with the market maker (bid arm, MM fee 1%); '
             'hard game on the validation paths.', fontsize=8.5, color=INK2, ha='left')
    out = RESULTS / 'market_opens_then_closes.png'
    plt.savefig(out, dpi=170, bbox_inches='tight', facecolor=SURF)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
