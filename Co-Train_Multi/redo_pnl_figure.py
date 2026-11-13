'''
  redo_pnl_figure.py -- P/L of every player against the fee spread, with the
  population MMs overlaid, drawn from verification/redo_numbers.json (written by
  redo_numbers.py; P/L = the average over the 200,000 test games, redo_numbers.pl()).

  Left: the MM's P/L, % of the game's money. Right: each trader's P/L, % of its
  own start, with the dashed line for converting everything at the Bureau in
  round 1 (the run's own "Bureau only" P/L on the same games).
    solid, filled   MM and T1-T4 trained together (Plot 1's runs), every spread
    hollow          the MMs trained on 4 / 12 / 24 random traders (1% and 2% only),
                    tested the Plot 2 way: frozen, T1-T4 relearn against it
  fig_pnl_live is the same graph for the live game: the solid lines only, with a
  plain-words legend.

  Usage: python redo_pnl_figure.py [numbers.json] [out folder]
  (default: verification/redo_numbers.json in, verification/redo_figures/ out)
'''
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fbfbf8', '#14213d', '#3f4756', '#7a8290', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2a78d6', '#eb6834'          # categorical slots 1 and 2 (validated on this surface)

FONTS = HERE / 'fonts'
FACES = {}
if (FONTS / 'IBMPlexSans-400.ttf').exists():
    for w in ('400', '500', '600'):
        fm.fontManager.addfont(str(FONTS / f'IBMPlexSans-{w}.ttf'))
        FACES[w] = str(FONTS / f'IBMPlexSans-{w}.ttf')
plt.rcParams.update({'font.family': 'IBM Plex Sans' if FACES else 'DejaVu Sans',
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
PT = 17            # the figure shows 1664 px wide on a slide: 17 pt = 24 px


def font(size=PT, weight='400'):
    if weight in FACES:
        return dict(fontproperties=fm.FontProperties(fname=FACES[weight], size=size))
    return dict(fontsize=size, fontweight='bold' if weight == '600' else 'normal')


LADDER = ('cotrained_0.02', 'cotrained_0.025', 'cotrained_0.03', 'cotrained_0.04',
          'cotrained_0.05', 'cotrained_0.06', 'cotrained_0.07')
SAME = (('pop4', 'o', ORANGE, '4 random', -0.1),
        ('pop12', 's', ORANGE, '12 random', 0.0),
        ('pop24', '^', ORANGE, '24 random', 0.1))
CARDS = ('T1', 'T2', 'T3', 'T4')


def spread(r):
    return round((r['trader_fee'] - r['mm_fee']) * 100, 3)


def style(ax, ylim, yticks, show_y=True):
    ax.set_xlim(0.6, 6.45)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([f'{x}%' for x in range(1, 7)], color=MUTED, **font(PT - 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:g}%'.replace('-', '−') for y in yticks] if show_y else [], color=MUTED, **font(PT - 1))
    ax.tick_params(length=0, pad=6)
    ax.grid(axis='y', color=GRID, linewidth=1.0, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)


def hollow(ax, x, y, marker, colour, scale=1.0, k=1.0):
    ax.plot([x], [y], marker=marker, ms=(9.5 if marker != 'D' else 8.5) * scale * k, mfc=SURFACE, mec=colour, mew=2.2 * k,
            ls='', zorder=5)


def draw(rows, out, name, size, pt, live=False):
    global PT
    PT = pt
    k = pt / 17                                   # marks scale with the type
    ladder = [rows[key] for key in LADDER]
    xs = [spread(r) for r in ladder]
    fig = plt.figure(figsize=size, dpi=100)

    # ---- the MM ------------------------------------------------------------------------------
    ax = fig.add_axes([0.062, 0.125, 0.335, 0.705])
    style(ax, (-0.25, 3.95), [0, 1, 2, 3])
    ys = [r['pl']['mm_pl'] * 100 for r in ladder]
    ax.plot(xs, ys, color=BLUE, lw=2.4 * k, marker='o', ms=7.5 * k, mec=SURFACE, mew=1.5 * k, zorder=4)
    for key, marker, colour, _, dx in (() if live else SAME):
        for fee in ('0.02', '0.03'):
            kk = f'{key}_test_{fee}'
            if kk in rows:
                hollow(ax, spread(rows[kk]) + dx, rows[kk]['pl']['mm_pl'] * 100, marker, colour, 1.0, k)
    ax.text(6.0, ys[-1] + 0.24, f'{ys[-1]:.1f}%', ha='center', va='bottom', color=INK, **font(PT, '600'))
    if not live:
        pops = [rows[f'pop{n}_test_0.03']['pl']['mm_pl'] * 100 for n in (4, 12, 24)]
        ax.text(2.32, sum(pops) / 3 - 0.08, f'{min(pops):.2f}–{max(pops):.2f}%', ha='left', va='center', color=INK,
                **font(PT - 1, '500'))
    ax.text(0.0, 1.05, 'MM · P/L, % of the £710k', transform=ax.transAxes, ha='left', va='bottom', color=INK,
            **font(PT + 1, '600'))
    fig.text(0.062 + 0.335 / 2, 0.012, 'Fee spread', ha='center', va='bottom', color=INK2, **font(PT - 1))

    # ---- the traders -------------------------------------------------------------------------
    for i, card in enumerate(CARDS):
        col, row = i % 2, i // 2
        a = fig.add_axes([0.487 + col * 0.262, 0.545 - row * 0.42, 0.228, 0.285])
        style(a, (-9.2, -0.8), [-8, -6, -4, -2], show_y=True)
        pl = [r['pl']['cards'][card]['pl'] * 100 for r in ladder]
        bdc = [r['pl']['cards'][card]['bureau_only'] * 100 for r in ladder]
        a.plot(xs, bdc, color=MUTED, lw=2.0 * k, ls=(0, (4, 3)), zorder=3)
        a.plot(xs, pl, color=BLUE, lw=2.4 * k, marker='o', ms=6.5 * k, mec=SURFACE, mew=1.3 * k, zorder=4)
        for key, marker, colour, _, dx in (() if live else SAME):
            for fee in ('0.02', '0.03'):
                kk = f'{key}_test_{fee}'
                if kk in rows:
                    hollow(a,spread(rows[kk]) + dx, rows[kk]['pl']['cards'][card]['pl'] * 100, marker, colour, 0.8, k)
        c = ladder[0]['per_card'][card]
        side = '£' if c['side'] == 'A' else '$'
        a.text(0.0, 1.06, f'{card} · sells {side}{c["L"] / 1000:.0f}k', transform=a.transAxes,
               ha='left', va='bottom', color=INK, **font(PT + 1, '600'))
    fig.text(0.487 + 0.49 / 2, 0.012, 'Fee spread', ha='center', va='bottom', color=INK2, **font(PT - 1))
    fig.text(0.428, 0.125 + 0.705 / 2, 'Each trader · P/L, % of its start', rotation=90, ha='center', va='center',
             color=INK, **font(PT + 1, '600'))

    # ---- one legend row for every panel, placed piece by piece -------------------------------
    renderer = fig.canvas.get_renderer()
    W = fig.get_size_inches()[0] * fig.dpi
    y, x = 0.955, 0.062
    def text(t, gap=0.012, weight='400'):
        nonlocal x
        h = fig.text(x, y, t, ha='left', va='center', color=INK2, **font(PT - 1, weight))
        x += h.get_window_extent(renderer).width / W + gap
    def handle(kind, colour=None, marker=None):
        nonlocal x
        if kind == 'line':
            fig.add_artist(Line2D([x, x + 0.026], [y, y], color=BLUE, lw=2.4 * k, transform=fig.transFigure))
            fig.add_artist(Line2D([x + 0.013], [y], marker='o', ms=7.5 * k, color=BLUE, mec=SURFACE, mew=1.5 * k,
                                  transform=fig.transFigure))
            x += 0.032
        elif kind == 'dash':
            fig.add_artist(Line2D([x, x + 0.026], [y, y], color=MUTED, lw=2.0 * k, ls=(0, (4, 3)),
                                  transform=fig.transFigure))
            x += 0.032
        else:
            fig.add_artist(Line2D([x + 0.006], [y], marker=marker, ms=9 * k, mfc=SURFACE, mec=colour, mew=2.2 * k,
                                  ls='', transform=fig.transFigure))
            x += 0.015
    if live:
        handle('line'); text('P/L after training', 0.022)
        handle('dash'); text('A trader taking everything straight to the Bureau in round 1', 0.0)
    else:
        handle('line'); text('Trained together (Plot 1)', 0.022)
        handle('dash'); text('Bureau only, in round 1 (traders)', 0.022)
        text('Plot 2 test, MMs trained on', 0.008, '500')
        handle('mark', ORANGE, 'o'); text('4,', 0.01)
        handle('mark', ORANGE, 's'); text('12,', 0.01)
        handle('mark', ORANGE, '^'); text('24 random', 0.0)
    assert x < 0.995, f'legend too wide: ends at {x:.3f}'

    for ext in ('.png', '.pdf'):
        fig.savefig(out / f'{name}{ext}', dpi=200)
    plt.close(fig)
    print('wrote', out / f'{name}.png')


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'Results/cotrain_multi/verification/redo_numbers.json'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / 'Results/cotrain_multi/verification/redo_figures'
    out.mkdir(parents=True, exist_ok=True)
    rows = {r['key']: r for r in json.loads(src.read_text())}
    for name, size, pt, live in (('fig_pnl', (16.64, 6.0), 17, False), ('fig_pnl_report', (11.0, 4.6), 12, False),
                                 ('fig_pnl_live', (16.64, 6.0), 17, True)):
        draw(rows, out, name, size, pt, live)


if __name__ == '__main__':
    main()
