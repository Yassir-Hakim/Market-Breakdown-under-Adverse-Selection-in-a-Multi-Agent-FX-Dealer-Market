'''
  live_numbers.py -- every number quoted in the live-game deck, read from
  redo_numbers.json (written by redo_numbers.py), with a check on each claim.
  Usage: python live_numbers.py [numbers.json]
'''
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'Results/cotrain_multi/verification/redo_numbers.json'
rows = {r['key']: r for r in json.loads(src.read_text())}
LADDER = ('cotrained_0.02', 'cotrained_0.025', 'cotrained_0.03', 'cotrained_0.04',
          'cotrained_0.05', 'cotrained_0.06', 'cotrained_0.07')
ladder = [rows[k] for k in LADDER]
pct = lambda v: v * 100
spread = lambda r: round((r['trader_fee'] - r['mm_fee']) * 100, 3)

print('--- the game (setup slide)')
r0 = ladder[0]
print('money in the game', r0['capital_gbp'], '| rules: trader fee', r0['trader_fee'], 'MM fee', r0['mm_fee'])
for c, d in r0['per_card'].items():
    print(f"  {c}: sells {'£' if d['side'] == 'A' else '$'}{d['L']:,.0f}, wants {'$' if d['side'] == 'A' else '£'}{d['T']:,.0f}, A {d['A']:.0%}, B {d['B']:.0%}")
print('  training:', {k: v for k, v in r0['training'].items() if not isinstance(v, (list, dict))})

print('--- fee slide 1 (Plot 1)')
for r in ladder:
    print(f"  {spread(r):>4g}%  MM traded {pct(r['mm_trades']):5.1f}  MM to Bureau {pct(r['mm_bureau']):5.1f}  netted {pct(r['netted']):5.1f}"
          f"  traders to Bureau {pct(r['cards_bureau']):5.1f}  left {pct(r['left_over']):5.1f}  all Bureau {pct(r['total_bureau']):5.1f}"
          f"  any fill {pct(r['any_fill']):5.2f}% of games")
a, b, z = ladder[0], ladder[1], ladder[-1]
assert pct(a['mm_trades']) < 0.05 and round(pct(a['total_bureau'])) == 100          # 1%: nobody trades with the MM
assert pct(b['mm_trades']) > 5                                                     # 1.5%: a market opens
assert round(pct(z['netted'])) == 22 and round(pct(z['total_bureau'])) == 51 and round(pct(z['left_over'])) == 24
assert round(pct(z['mm_bureau'])) == 49

print('--- fee slide 2 (P/L)')
for r in ladder:
    g = r['pl']['cards']
    print(f"  {spread(r):>4g}%  MM {pct(r['pl']['mm_pl']):5.2f}  " + '  '.join(
        f"{c} {pct(g[c]['pl']):6.2f} (Bureau {pct(g[c]['bureau_only']):6.2f}, gain {pct(g[c]['gain']):+.2f} ± {pct(g[c]['gain_se']):.2f})"
        for c in g))
assert abs(pct(a['pl']['mm_pl'])) < 0.005 and round(pct(z['pl']['mm_pl']), 1) == 3.5
below = [(spread(r), c) for r in ladder for c, d in r['pl']['cards'].items()
         if d['gain'] < -2 * d['gain_se'] and d['gain'] < -1e-4]   # visibly below: more than 0.01 points and 2 SE
print('  below the Bureau line by more than 0.01 points and 2 SE:', below)
assert below == [(6, 'T4')]
t4 = z['pl']['cards']['T4']
print(f"  T4 at 6%: {pct(t4['gain']):.2f} points, {-t4['gain'] / t4['gain_se']:.1f} SE; to MM {pct(z['per_card']['T4']['to_mm']):.0f}%, left {pct(z['per_card']['T4']['left']):.0f}%")

print('--- closing slide (Plot 2)')
for n in (4, 12, 24):
    lo, hi = rows[f'pop{n}_test_0.02'], rows[f'pop{n}_test_0.03']
    assert pct(lo['mm_trades']) < 0.05
    print(f"  {n:>2} random  1%: MM traded {pct(lo['mm_trades']):.3f}  |  2%: MM to Bureau {pct(hi['mm_bureau']):4.1f}  netted {pct(hi['netted']):3.1f}"
          f"  traders to Bureau {pct(hi['cards_bureau']):4.1f}  all Bureau {pct(hi['total_bureau']):4.1f}  MM P/L {pct(hi['pl']['mm_pl']):.2f}")
    assert round(pct(hi['total_bureau'])) == 97
nets = [pct(rows[f'pop{n}_test_0.03']['netted']) for n in (4, 12, 24)]
assert f'{min(nets):.1f}' == '2.5' and f'{max(nets):.1f}' == '3.3'
c2 = rows['cotrained_0.03']
print(f"  trained with T1-T4 only, 2%: netted {pct(c2['netted']):.1f}  all Bureau {pct(c2['total_bureau']):.1f}  MM P/L {pct(c2['pl']['mm_pl']):.2f}")
for k in ('pop4s1_test_0.03', 'pop4s2_test_0.03'):
    r = rows[k]
    print(f"  {k}: MM to Bureau {pct(r['mm_bureau']):.1f}  netted {pct(r['netted']):.1f}  all Bureau {pct(r['total_bureau']):.1f}")
print('all checks passed')
