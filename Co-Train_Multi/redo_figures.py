'''
  redo_figures.py -- the charts of the redone deck and report, drawn from
  verification/redo_numbers.json (written by redo_numbers.py).

  Every column is ALL of T1-T4's money (GBP 710k at 1.25 $/GBP), averaged over
  the same 20,000 test games, split four ways:
    blue    the MM took it to the Bureau          (Avi's measure)
    green   the MM netted it: paid one trader with what another sold it
    grey    the traders took it to the Bureau themselves
    light   left unconverted at the end
  The numbers under the columns are the same four parts, then all Bureau use:
  the MM's and the traders' own together.
  fig_spread   the MM trained together with T1-T4, one column per fee spread
  fig_traders  the MMs trained on 4, 12 or 24 random traders, each frozen and
               tested on T1-T4, 2% spread

  Usage: python redo_figures.py [numbers.json] [out folder]
  (default: verification/redo_numbers.json in, verification/redo_figures/ out)
  matplotlib + numpy only; the deck's font is used if it is in ./fonts.
'''
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fbfbf8', '#14213d', '#3f4756', '#7a8290', '#e1e0d9', '#c3c2b7'
BLUE, GREEN, GREY, LIGHT = '#2a78d6', '#008300', '#c3c2b7', '#e9e7e0'
PARTS = (('MM took to the Bureau', 'mm_bureau', BLUE),
         ('MM netted', 'netted', GREEN),
         ('Traders to the Bureau', 'cards_bureau', GREY),
         ('Left unconverted', 'left_over', LIGHT))

FONTS = HERE / 'fonts'
FACES = {}
if (FONTS / 'IBMPlexSans-400.ttf').exists():
    for w in ('400', '500', '600'):
        fm.fontManager.addfont(str(FONTS / f'IBMPlexSans-{w}.ttf'))
        FACES[w] = str(FONTS / f'IBMPlexSans-{w}.ttf')
plt.rcParams.update({'font.family': 'IBM Plex Sans' if FACES else 'DejaVu Sans',
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})

PT = 18            # on a slide the figure shows 1664 px wide at dpi 100: 18 pt = 25 px


def font(size=PT, weight='400'):
    if weight in FACES:
        return dict(fontproperties=fm.FontProperties(fname=FACES[weight], size=size))
    return dict(fontsize=size, fontweight='bold' if weight == '600' else 'normal')


def parts_of(r):
    '''The four parts, % of the game's money. Each is read on its own (a dollar
    seller's dollars at the offer's price, the round's rate or the settlement
    rate), so they add to 97-100%; each column is drawn scaled to 100%, the
    numbers under it are the parts as measured, and the report prints the sum.'''
    return np.array([r[key] for _, key, _ in PARTS]) * 100



def fmt(v):
    if abs(v) < 0.05:
        return '0%'
    return f'{v:.0f}%' if abs(v) >= 9.95 else f'{v:.1f}%'


def draw(xs, xlabels, rows, out, x_title, width, xlim, note, size=(16.64, 5.3), pt=18):
    global PT
    PT = pt
    fig = plt.figure(figsize=size, dpi=100)
    ax = fig.add_axes([0.205, 0.47, 0.785, 0.46])
    for x, r in zip(xs, rows):
        parts = np.maximum(parts_of(r), 0.0)
        heights = parts * 100.0 / parts.sum()          # the column drawn to 100%; the table keeps the raw parts
        bottom = 0.0
        for h, (_, _, colour) in zip(heights, PARTS):
            if h > 0:
                ax.bar(x, h, width=width, bottom=bottom, color=colour, edgecolor=SURFACE,
                       linewidth=2.0, zorder=3)
            bottom += h
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color=MUTED, **font(PT - 2))
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0, pad=8)
    ax.grid(axis='y', color=GRID, linewidth=1.0, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.text(0.0, 1.035, note, transform=ax.transAxes, ha='left', va='bottom', color=MUTED, **font(PT - 3))

    # the table under the columns: the x labels, then the four parts with their swatches
    trans = ax.get_xaxis_transform()
    y = -0.10
    ax.text(-0.03, y, x_title, transform=ax.transAxes, ha='right', va='center', color=INK2, **font(PT, '500'))
    for x, label in zip(xs, xlabels):
        ax.text(x, y, label, transform=trans, ha='center', va='center', color=INK, **font(PT, '600'),
                linespacing=1.15)
    y -= 0.19 if any("\n" in l for l in xlabels) else 0.14
    for k, (label, key, colour) in enumerate(PARTS):
        ax.text(-0.045, y, label, transform=ax.transAxes, ha='right', va='center', color=INK2, **font(PT - 1))
        ax.plot([-0.027], [y], transform=ax.transAxes, marker='s', ms=12, color=colour, ls='', clip_on=False)
        for x, r in zip(xs, rows):
            ax.text(x, y, fmt(r[key] * 100), transform=trans, ha='center', va='center',
                    color=INK if k < 2 else INK2, **font(PT, '500' if k < 2 else '400'))
        y -= 0.115
    # the whole game's Bureau use: the MM's and the traders' own
    ax.text(-0.045, y - 0.01, 'All Bureau use', transform=ax.transAxes, ha='right', va='center', color=INK,
            **font(PT - 1, '600'))
    for x, r in zip(xs, rows):
        ax.text(x, y - 0.01, fmt(r['total_bureau'] * 100), transform=trans, ha='center', va='center', color=INK,
                **font(PT, '600'))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'Results/cotrain_multi/verification/redo_numbers.json'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / 'redo_figures'
    rows = {r['key']: r for r in json.loads(src.read_text())}
    note = '% of all of T1–T4\'s money (£710k), average of the same 20,000 test games'

    ladder = sorted((r for r in rows.values() if r['group'] == 'ladder'), key=lambda r: r['trader_fee'])
    xs = [round((r['trader_fee'] - r['mm_fee']) * 100, 3) for r in ladder]
    keys = ['pop4_test_0.03', 'pop12_test_0.03', 'pop24_test_0.03']
    labels = ['4 random', '12 random', '24 random']
    present = [(k, l) for k, l in zip(keys, labels) if k in rows]
    for suffix, size, pt in (('', (16.64, 5.3), 18), ('_report', (11.0, 4.3), 13)):
        draw(xs, [f'{x:g}%' for x in xs], ladder, out / f'fig_spread{suffix}', 'Fee spread', width=0.34,
             xlim=(0.55, 6.45), note=note, size=size, pt=pt)
        draw(list(range(len(present))), [l for _, l in present], [rows[k] for k, _ in present],
             out / f'fig_traders{suffix}', 'MM trained on', width=0.36, xlim=(-0.55, len(present) - 0.45),
             note=note, size=size, pt=pt)
    for r in ladder + [rows[k] for k, _ in present]:
        p = parts_of(r)
        print(f"{r['label']:<64} parts {np.round(p, 2)}  sum {p.sum():.2f}%")


if __name__ == '__main__':
    main()
