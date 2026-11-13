'''
  population_tables.py -- the random tables behind the population runs,
  card by card, and how long every stage trained.

  How a random trader is drawn (cotrain_multi.random_roster): side by coin
  toss, redrawn until both sides are present; start amount L from the four
  cards' amounts (50k, 100k, 200k or 500k, in the trader's own currency);
  target T = all of L converted at the opening rate 1.25, times u, with u
  uniform in 0.90-1.10; penalties A and B each from the cards' set
  (0, 2, 3, 5, 10 or 20%).

  For every population run on disk this reads the table the run saved,
  rebuilds it from its size and seed to show the saved table is exactly that
  draw, and prints each card beside the four cards: side, start, target, u,
  penalties, and whether A + B is above the traders' Bureau fee (if it is
  not, the card has no reason to use the Bureau). It also prints how many
  training steps each stage ran, on how many games each step learned and on
  how many games it was measured, and what each table's market maker did on
  the four cards (read from verification/verify_population.json, which
  verify_population.py writes).

  Output: a printout, verification/tables.json and verification/fig_tables.png
  (one bar per table: how its money splits between the two sides; MAIN_ONLY
  below picks table 0 at each size, the deck's runs, or every table).
  Usage: python population_tables.py
'''
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'Results' / 'cotrain_multi'
OUT = RESULTS / 'verification'
RUN = re.compile(r'^pop(\d+)s(\d+)_mm(0\.\d+)_bdc(0\.\d+)_bid_size_2way$')
BASELINE_TEST = 'checks/T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_frozen_cotrained/respond_on_cards'
FEE = 0.03   # the traders' Bureau fee of the 2% spread runs, the ones the figure reads
MAIN_ONLY = True   # the figure: True draws table 0 at each size (the deck's population
                   # experiment); False draws every table on disk


def card(row):
    return dict(name=str(row['name']), side=str(row['side']), L=float(row['L']),
                T=float(row['T']), A=float(row['A']), B=float(row['B']))


def describe(cards, a0, fee):
    '''Each card with its pounds at the opening rate, how demanding its target
    is (u = T / all of L converted at a0) and whether its penalties make the
    Bureau worth using at this fee (A + B above the fee).'''
    out = []
    for c in cards:
        par = c['L'] * a0 if c['side'] == 'A' else c['L'] / a0
        pounds = c['L'] if c['side'] == 'A' else c['L'] / a0
        out.append(dict(c, pounds=pounds, u=c['T'] / par, bureau_worth_it=bool(c['A'] + c['B'] > fee + 1e-12)))
    return out


def stage(folder):
    '''How long one stage trained and on how many games it was measured.'''
    curve = np.load(folder / 'cotrain_multi_curve.npz')
    z = np.load(folder / 'cotrain_multi_eval.npz')
    return dict(mm_steps=int(curve['step_mm'][-1]), trader_steps=int(curve['step_trader'][-1]),
                stopped_early=bool(int(curve['stopped_at']) >= 0), games_per_step=int(z['batch']),
                record_games=int(z['record_mm_pl'].shape[0]), eval_games=int(z['eval_mm_pl'].shape[0]),
                lr_trader=float(z['lr'][0]), lr_mm=float(z['lr'][1]))


def collect():
    sys.path.insert(0, str(HERE))
    import cotrain_multi as A
    rules = dict(start_amounts=[float(x) for x in A.ROSTER_L],
                 target_u=[float(x) for x in A.ROSTER_TARGET_RANGE],
                 penalties=[float(x) for x in A.ROSTER_CARD_FEES],
                 a0=float(A.A0), sd=float(A.SD), rounds=int(A.ROUNDS))
    audit = OUT / 'verify_population.json'
    results = {r['folder']: r for r in json.loads(audit.read_text())} if audit.exists() else {}
    runs = []
    for folder in sorted(p for p in RESULTS.rglob('pop*') if p.is_dir() and RUN.match(p.name)):
        n, seed, g, f = RUN.match(folder.name).groups()
        n, seed, g, f = int(n), int(seed), float(g), float(f)
        z = np.load(folder / 'cotrain_multi_eval.npz')
        saved = [card(dict(name=nm, side=sd, L=c[0], T=c[1], A=c[2], B=c[3]))
                 for nm, sd, c in zip(z['traders'], z['sides'], z['cards'])]
        rebuilt = [card(r) for r in A.random_roster(n, seed)]
        rel = folder.relative_to(RESULTS).as_posix()
        test = results.get(rel + '/respond_on_cards', {})
        runs.append(dict(folder=rel, n=n, seed=seed, mm_fee=g, trader_fee=f, spread=round(f - g, 4),
                         saved_table_is_the_draw=saved == rebuilt, capital=float(z['capital_pounds']),
                         cards=describe(saved, float(z['a0']), f),
                         train=stage(folder), test=stage(folder / 'respond_on_cards'),
                         flow_on_cards=test.get('flow'), share_on_cards=test.get('share')))
    base = results.get(BASELINE_TEST, {})
    return dict(rules=rules, four_cards=describe([card(r) for r in A.TRADERS], float(A.A0), FEE),
                runs=runs, baseline_same_test=dict(flow=base.get('flow'), share=base.get('share')))


def money(c):
    return ('£' if c['side'] == 'A' else '$') + f"{c['L'] / 1e3:,.0f}k"


def report(data):
    r = data['rules']
    print('How a random trader is drawn')
    print('  side            coin toss, redrawn until both sides are present')
    print('  start amount L  one of ' + ', '.join(f'{x / 1e3:,.0f}k' for x in r['start_amounts'])
          + " in the trader's own currency (the four cards' amounts)")
    print(f"  target T        all of L converted at {r['a0']}, times u; u uniform in "
          f"{r['target_u'][0]:.2f}-{r['target_u'][1]:.2f}")
    print('  penalties A, B  each one of ' + ', '.join(f'{x * 100:g}%' for x in r['penalties']))
    print(f"  market          {r['rounds']} rounds, rate opens at {r['a0']} $/GBP and moves by a "
          f"normal step of sd {r['sd']} a round")

    def show(cards, fee):
        for c in cards:
            target = ('$' if c['side'] == 'A' else '£') + f"{c['T']:,.0f}"
            print(f"    {c['name']:>4}  {'pound ' if c['side'] == 'A' else 'dollar'} seller  "
                  f"start {money(c):>6}  (£{c['pounds'] / 1e3:,.0f}k)  target {target:>9}  u {c['u']:.3f}  "
                  f"A {c['A'] * 100:>2g}%  B {c['B'] * 100:>2g}%"
                  + ('' if c['bureau_worth_it'] else f'   A+B <= {fee * 100:g}%: no reason to use the Bureau'))

    print(f'\nThe four cards T1-T4 (traders\' fee {FEE * 100:g}%)')
    show(data['four_cards'], FEE)
    seen = set()
    for run in data['runs']:
        key = (run['n'], run['seed'])
        if key not in seen:
            seen.add(key)
            a = sum(c['pounds'] for c in run['cards'] if c['side'] == 'A')
            print(f"\n{run['n']} traders, table {run['seed']}: saved table is the draw from (N, seed): "
                  f"{run['saved_table_is_the_draw']}; money £{run['capital']:,.0f}, "
                  f"{a / run['capital'] * 100:.0f}% pound sellers (traders' fee {run['trader_fee'] * 100:g}%)")
            show(run['cards'], run['trader_fee'])
        t, s = run['train'], run['test']
        print(f"  {run['spread'] * 100:g}% spread ({run['folder']}): MM trained {t['mm_steps']:,} steps and "
              f"traders {t['trader_steps']:,}, {t['games_per_step']:,} games a step"
              f"{', stopped early' if t['stopped_early'] else ''}; T1-T4 relearned {s['trader_steps']:,} steps "
              f"against it (MM steps {s['mm_steps']}){', stopped early' if s['stopped_early'] else ''}; "
              f"measured on {s['record_games']:,} games"
              + (f"; its MM on T1-T4: {run['flow_on_cards'] * 100:.2f}% of the money, "
                 f"{run['share_on_cards'] * 100:.1f}% to the Bureau"
                 if run['flow_on_cards'] is not None and run['share_on_cards'] == run['share_on_cards'] else ''))


def fig_tables(data, out):
    '''One bar per random table (and the four cards on top): 100% of that
    table's money, pound sellers left, dollar sellers right, one segment per
    card, hatched when the card has no reason to use the Bureau at the 2%
    spread. Right: the table's money and what its market maker did in the
    test on the four cards.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fbfbf8'
    POUND, DOLLAR = '#1baf7a', '#4a3aa7'
    POUND_PALE, DOLLAR_PALE = '#d4efe4', '#dcd8ef'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'figure.facecolor': SURFACE,
                         'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'hatch.linewidth': 1.6})

    rows = [('T1–T4, the four cards', 'the test · baseline MM', data['four_cards'], 710_000.0,
             data['baseline_same_test'])]
    for run in sorted((r for r in data['runs'] if abs(r['trader_fee'] - FEE) < 1e-9
                       and (r['seed'] == 0 or not MAIN_ONLY)),
                      key=lambda r: (r['n'], r['seed'])):
        rows.append((f"{run['n']} random traders" if MAIN_ONLY else f"{run['n']} traders · table {run['seed']}",
                     None, run['cards'], run['capital'],
                     dict(flow=run['flow_on_cards'], share=run['share_on_cards'])))

    top_in, bottom_in = 1.3, 0.9          # legend and headers above, axis below, in inches
    height = top_in + bottom_in + 0.3 + 0.74 * len(rows)
    fig, ax = plt.subplots(figsize=(12.4, height), dpi=200)
    fig.subplots_adjust(left=0.215, right=0.70, top=1 - top_in / height, bottom=bottom_in / height)
    for i, (label, sub, cards, capital, res) in enumerate(rows):
        y = len(rows) - 1 - i + (0.3 if i == 0 else 0.0)   # a little space under the four cards
        x = 0.0
        ordered = [c for c in cards if c['side'] == 'A'] + [c for c in cards if c['side'] == 'B']
        for c in ordered:
            w = c['pounds'] / capital * 100
            live = c['bureau_worth_it']
            colour = POUND if c['side'] == 'A' else DOLLAR
            pale = POUND_PALE if c['side'] == 'A' else DOLLAR_PALE
            ax.barh(y, w, left=x, height=0.64, color=colour if live else pale,
                    hatch=None if live else '///', edgecolor=SURFACE if live else colour,
                    linewidth=0 if live else 1.2, zorder=3)
            ax.plot([x + w, x + w], [y - 0.32, y + 0.32], color=SURFACE, lw=2.2, zorder=4)
            if len(cards) <= 4 and w >= 6.5:
                ink = INK if (c['side'] == 'A' or not live) else '#ffffff'
                ax.text(x + w / 2, y, f"{c['name']}\n{money(c)}", ha='center', va='center',
                        fontsize=11.5, color=ink, zorder=5, linespacing=1.1,
                        bbox=None if live else dict(boxstyle='round,pad=0.12', fc=SURFACE, ec='none', alpha=0.85))
            x += w
        ax.text(-2, y + (0.14 if sub else 0), label, ha='right', va='center', fontsize=14, color=INK,
                fontweight='bold' if sub else 'normal')
        if sub:
            ax.text(-2, y - 0.25, sub, ha='right', va='center', fontsize=12, color=INK2)
        ax.text(104, y, f'£{capital / 1e6:.2f}m', ha='left', va='center', fontsize=14, color=INK)
        if res.get('flow') is not None:
            ax.text(122, y, f"{res['flow'] * 100:.0f}%", ha='left', va='center', fontsize=14, color=INK)
            ax.text(142, y, f"{res['share'] * 100:.0f}%", ha='left', va='center', fontsize=14, color=INK)
    top = len(rows) - 0.18
    ax.text(104, top, 'money\nat the table', ha='left', va='bottom', fontsize=12, color=INK2)
    ax.text(122, top + 0.72, 'its MM, tested on T1–T4', ha='left', va='bottom', fontsize=12, color=INK)
    ax.text(122, top, 'money at\nthe Bureau', ha='left', va='bottom', fontsize=12, color=INK2)
    ax.text(142, top, 'share sent\nto the Bureau', ha='left', va='bottom', fontsize=12, color=INK2)
    ax.axvline(50, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(50, -0.72, 'even split', ha='center', va='top', fontsize=12, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.tick_params(axis='x', colors=INK2, length=0, pad=20, labelsize=13)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("share of the table's money (£ at the opening rate 1.25 $/£)", color=INK2, labelpad=6,
                  fontsize=13)
    handles = [Patch(color=POUND, label='pound sellers'), Patch(color=DOLLAR, label='dollar sellers'),
               Patch(facecolor='#ffffff', edgecolor=INK2, hatch='///',
                     label=f"A + B ≤ traders' fee ({FEE * 100:g}%): no reason to use the Bureau")]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.215, 0.99), ncol=3, frameon=False,
               fontsize=13, handlelength=1.4, columnspacing=1.6)
    fig.savefig(out)
    plt.close(fig)


def main():
    data = collect()
    report(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tables.json').write_text(json.dumps(data, indent=1))
    fig_tables(data, OUT / 'fig_tables.png')
    print(f"\nwrote {(OUT / 'tables.json').relative_to(HERE.parent)} and fig_tables.png")


if __name__ == '__main__':
    main()
