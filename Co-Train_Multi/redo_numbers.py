'''
  redo_numbers.py -- every number in the redone deck and report, read from the
  saved runs, each one tagged with how many games it averages over.

  Runs (all: the market maker's Bureau fee 1%, bid arm, offer size seen,
  two-way offers, starting weights seed 42):
    spread ladder   T1-T4 co-trained with the market maker, one run per
                    traders' Bureau fee on disk (0.02 ... 0.07 = spreads 1% ... 6%)
    same test, 2%   that co-trained market maker frozen, T1-T4 relearn against
                    it from fresh (robustness_runs.py, frozen_cotrained)
    robustness      the 2% co-training from two other random starts (seeds 142,
                    242) and two other draws of 4 random traders (robustness_runs.py);
                    for the report only
    population      pop4s0, pop12s0, pop24s0 at 1% and 2%: the market maker
                    trained on the random table, frozen, T1-T4 relearn
                    (respond_on_cards); and the co-training on the table itself

  From each run's cotrain_multi_eval.npz:
    the 20,000 recorded games
      MM trades        pounds the MM bought or paid out in fills (a dollar
                       seller's dollars at the offer's price), % of the game's
                       money (capital_pounds: every card's start at 1.25)
      MM to Bureau     pounds the MM moved at the Bureau, its final clear
                       included: Avi's measure (the run's own
                       bureau_pound_of_capital, rebuilt and printed beside it)
      netted           MM trades - MM to Bureau: what the MM paid out of what it
                       bought, with no Bureau
      cards to Bureau  start currency the cards sold at the Bureau themselves
                       (dollars at the true rate of that round)
      left over        start currency still held at the end (dollars at the
                       settlement rate)
      per card, in its own currency and as % of its start: sold to the MM,
      sold at the Bureau, left over (these three add to the start exactly)
      round 1, per card: the offer's amount and price (the same in every game:
      it is set before the rate is revealed) and the % of games it was filled
      both sides       % of games in which, in some round, the MM filled a pound
                       seller AND a dollar seller
      matchable        per game, twice the smaller of the pounds the MM bought and
                       the pounds it paid out for dollars: the most it could have
                       netted over the game, had it held everything to the end
    the 200,000 evaluation games
      MM P/L (% of the game's money), each card's P/L (% of its start) and its
      P/L had it gone straight to the Bureau
  From cotrain_multi_curve.npz: the blocks each agent trained (a block is 100
  steps of 1,000 games) and the late validation band.

  Output: a printout and verification/redo_numbers.json. numpy only.
  Usage: python redo_numbers.py
'''
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'Results' / 'cotrain_multi'
OUT = RESULTS / 'verification'

LADDER = (0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07)      # traders' Bureau fee; the MM's is 0.01
MM_FEE = 0.01
POP_SIZES = (4, 12, 24)
LATE = 10                                                # validation readings in the late band


def ladder_folder(trader_fee):
    return f'T1-T2-T3-T4_mm0.01_bdc{trader_fee:g}_bid_size_2way'


def runs():
    '''(key, group, label, folder relative to RESULTS, trader fee)'''
    out = []
    for f in LADDER:
        out.append((f'cotrained_{f:g}', 'ladder', f'T1-T4 co-trained, spread {(f - MM_FEE) * 100:g}%',
                    ladder_folder(f), f))
    out.append(('sametest_0.03', 'sametest', 'MM trained with T1-T4, same test, spread 2%',
                'checks/' + ladder_folder(0.03) + '_frozen_cotrained/respond_on_cards', 0.03))
    for f in (0.02, 0.03):
        for n in POP_SIZES:
            base = f'pop{n}s0_mm0.01_bdc{f:g}_bid_size_2way'
            out.append((f'pop{n}_test_{f:g}', 'population_test',
                        f'MM trained on {n} random traders, tested on T1-T4, spread {(f - MM_FEE) * 100:g}%',
                        base + '/respond_on_cards', f))
            out.append((f'pop{n}_own_{f:g}', 'population_own',
                        f'{n} random traders co-trained with their MM, spread {(f - MM_FEE) * 100:g}%',
                        base, f))
    # robustness_runs.py: the same 2% runs from other random starts, and two other 4-trader draws
    for seed in (142, 242):
        out.append((f'seed{seed}_0.03', 'robustness', f'T1-T4 co-trained, spread 2%, starting weights seed {seed}',
                    'checks/' + ladder_folder(0.03) + f'_seed{seed}', 0.03))
    for table in (1, 2):
        out.append((f'pop4s{table}_test_0.03', 'robustness',
                    f'MM trained on 4 random traders (draw {table}), tested on T1-T4, spread 2%',
                    f'checks/pop4s{table}_mm0.01_bdc0.03_bid_size_2way/respond_on_cards', 0.03))
        out.append((f'pop4s{table}_own_0.03', 'robustness_own',
                    f'4 random traders (draw {table}) co-trained with their MM, spread 2%',
                    f'checks/pop4s{table}_mm0.01_bdc0.03_bid_size_2way', 0.03))
    return out


def as_str(a):
    return [str(s) for s in np.asarray(a)]


def money_and_round1(z):
    names, sides = as_str(z['traders']), np.array(as_str(z['sides']))
    cards = np.asarray(z['cards'], float)                                 # L, T, A, B per card
    L = cards[:, 0]
    cap, a0 = float(z['capital_pounds']), float(z['a0'])
    pound = sides == 'A'
    X = np.asarray(z['record_hidden_rate'], float)                         # (games, rounds)
    a5 = np.asarray(z['record_settlement_rate'], float)                    # (games,)
    games, rounds = X.shape

    held_fill = np.asarray(z['record_trader_held_after_fill'], float)      # (games, cards, rounds)
    held_bureau = np.asarray(z['record_trader_held_after_bureau'], float)
    sold_bureau = held_fill - held_bureau                                  # start currency sold at the Bureau
    left = held_bureau[:, :, -1]                                           # start currency held at the end

    amount = np.asarray(z['record_offered_amount'], float)                 # (games, offers, rounds)
    weight = np.asarray(z['record_fill_weight'], float)
    price = np.asarray(z['record_quoted_price'], float)
    direction = np.asarray(z['record_direction'], float)
    owner = np.asarray(z['offer_trader'], int)
    filled = weight * amount                                               # in the offer's own currency
    filled_gbp = np.where(direction > 0, filled,
                          np.divide(filled, price, out=np.zeros_like(filled), where=filled > 0))

    # the MM's Bureau pounds: every change of its pounds at the Bureau, its final clear included
    after_fill = np.asarray(z['record_mm_pounds_after_fill'], float)
    after_bureau = np.asarray(z['record_mm_pounds_after_bureau'], float)
    end = np.asarray(z['record_mm_pounds_end'], float)
    mm_bureau = np.abs(after_bureau - after_fill).sum(1) + np.abs(end - after_bureau[:, -1])
    mm_trades = filled_gbp.sum((1, 2))

    to_gbp_x = np.where(pound[None, :, None], 1.0, 1.0 / X[:, None, :])    # a round's dollars at its true rate
    cards_bureau = (sold_bureau * to_gbp_x).sum((1, 2))
    left_gbp = (left * np.where(pound[None, :], 1.0, 1.0 / a5[:, None])).sum(1)

    # the MM filled a pound seller and a dollar seller in the same round
    fwd_pound = (direction > 0) & (weight > 0) & (amount > 0)
    fwd_dollar = (direction < 0) & (weight > 0) & (amount > 0)
    both = (fwd_pound.any(1) & fwd_dollar.any(1)).any(1)                  # (games,)
    any_fill = ((weight > 0) & (amount > 0)).any((1, 2))
    # over the whole game: pounds the MM bought, and pounds it paid out for dollars;
    # twice the smaller is the most it could have netted had it held everything to the end
    bought = np.where(direction > 0, filled_gbp, 0.0).sum((1, 2))
    paid = np.where(direction < 0, filled_gbp, 0.0).sum((1, 2))

    out = dict(
        recorded_games=int(games), capital_gbp=cap, a0=a0,
        mm_trades=float(mm_trades.mean() / cap), mm_bureau=float(mm_bureau.mean() / cap),
        saved_mm_bureau=(float(z['bureau_pound_of_capital']) if 'bureau_pound_of_capital' in z.files else None),
        saved_bureau_share=float(z['bureau_share']) if 'bureau_share' in z.files else None,
        cards_bureau=float(cards_bureau.mean() / cap), left_over=float(left_gbp.mean() / cap),
        both_sides_same_round=float(both.mean()), any_fill=float(any_fill.mean()),
        matchable=float((2.0 * np.minimum(bought, paid)).mean() / cap),
        pound_side_of_trades=float(bought.sum() / max(bought.sum() + paid.sum(), 1e-300)))
    out['netted'] = out['mm_trades'] - out['mm_bureau']
    out['total_bureau'] = out['mm_bureau'] + out['cards_bureau']
    out['sum_check'] = out['netted'] + out['mm_bureau'] + out['cards_bureau'] + out['left_over']

    per_card, worst_gap = {}, 0.0
    for i, name in enumerate(names):
        cols = np.nonzero(owner == i)[0]
        fwd = direction[:, cols, :] == (1.0 if pound[i] else -1.0)
        sold_mm = (filled[:, cols, :] * fwd).sum((1, 2))                   # own currency
        sold_bdc = sold_bureau[:, i, :].sum(1)
        kept = left[:, i]
        worst_gap = max(worst_gap, float(np.abs(sold_mm + sold_bdc + kept - L[i]).max() / L[i]))
        a1, p1, w1 = amount[:, cols, 0], price[:, cols, 0], weight[:, cols, 0]
        made = a1 > 0
        per_card[name] = dict(
            side=str(sides[i]), L=float(L[i]), T=float(cards[i, 1]), A=float(cards[i, 2]), B=float(cards[i, 3]),
            to_mm=float(sold_mm.mean() / L[i]), to_bureau=float(sold_bdc.mean() / L[i]),
            left=float(kept.mean() / L[i]),
            r1_offers_made=float(made.mean()),
            r1_amount_mean=float(a1[made].mean()) if made.any() else 0.0,
            r1_amount_sd=float(a1[made].std()) if made.any() else 0.0,
            r1_price_median=float(np.median(p1[made])) if made.any() else float('nan'),
            r1_price_spread=(float(p1[made].max() - p1[made].min()) if made.any() else float('nan')),
            r1_filled=float(w1[made].mean()) if made.any() else 0.0)
    out['per_card'] = per_card
    out['own_currency_gap'] = worst_gap         # sold to MM + at Bureau + left vs start, worst game and card
    return out


def pl(z):
    names = as_str(z['traders'])
    trader = np.asarray(z['eval_trader_pl'], float)
    bdc = np.asarray(z['eval_bdc_pl'], float)
    mm = np.asarray(z['eval_mm_pl'], float)
    n_eval = int(mm.shape[0])
    out = dict(eval_games=n_eval, mm_pl=float(mm.mean()), mm_pl_se=float(mm.std() / np.sqrt(n_eval)),
               cards={})
    for i, name in enumerate(names):
        gain = trader[i] - bdc[i]
        out['cards'][name] = dict(pl=float(trader[i].mean()), bureau_only=float(bdc[i].mean()),
                                  gain=float(gain.mean()), gain_se=float(gain.std() / np.sqrt(n_eval)))
    if 'record_mm_pl' in z.files:                # the same numbers on the 20,000 recorded games, as a check
        out['mm_pl_recorded'] = float(np.asarray(z['record_mm_pl'], float).mean())
    return out


def training(folder):
    file = folder / 'cotrain_multi_curve.npz'
    if not file.exists():
        return None
    c = np.load(file, allow_pickle=True)
    moved = np.array([str(m) for m in c['moved']])
    out = dict(mm_blocks=int((moved == 'mm').sum()), trader_blocks=int((moved == 'trader').sum()))
    rows = np.nonzero(moved == 'trader')[0][-LATE:]
    for key in ('mm_pl', 'filled'):
        if key in c.files and len(rows):
            v = np.asarray(c[key], float)[rows]
            out[f'late_{key}'] = [float(np.nanmean(v)), float(np.nanstd(v))]
    return out


def read_run(folder):
    z = np.load(folder / 'cotrain_multi_eval.npz', allow_pickle=False)
    out = dict(trader_fee=float(z['trader_bdc_fee']), mm_fee=float(z['fee']),
               traders=as_str(z['traders']), sides=as_str(z['sides']),
               cards=np.asarray(z['cards'], float).tolist())
    out.update(money_and_round1(z))
    out['pl'] = pl(z)
    out['training'] = training(folder)
    return out


def pct(x, d=2):
    return f'{x * 100:.{d}f}%'


def report(rows):
    print('Every run: money is % of the game\'s money (start amounts at 1.25 $/GBP), over the '
          'recorded games; P/L over the evaluation games.\n')
    for r in rows:
        saved = '' if r['saved_mm_bureau'] is None else f" (saved {pct(r['saved_mm_bureau'])})"
        t = r['training']
        blocks = '' if t is None else f"  trained {t['mm_blocks']} MM / {t['trader_blocks']} trader blocks"
        print(f"{r['label']}  [{r['folder']}]{blocks}")
        print(f"  {r['recorded_games']:,} games: MM trades {pct(r['mm_trades'])}  MM to Bureau "
              f"{pct(r['mm_bureau'])}{saved}  netted {pct(r['netted'])}  cards to Bureau "
              f"{pct(r['cards_bureau'])}  left over {pct(r['left_over'])}  (sum {pct(r['sum_check'], 1)})  "
              f"total Bureau {pct(r['total_bureau'], 1)}  both sides in a round {pct(r['both_sides_same_round'], 1)}"
              f"  matchable {pct(r['matchable'])}"
              f"  own-currency check {r['own_currency_gap']:.1e}")
        p = r['pl']
        rec = f" (recorded games {p['mm_pl_recorded'] * 1e4:+.1f} bp)" if 'mm_pl_recorded' in p else ''
        print(f"  {p['eval_games']:,} games: MM P/L {p['mm_pl'] * 1e4:+.1f} bp +/- {2 * p['mm_pl_se'] * 1e4:.1f}{rec}")
        for name, c in r['per_card'].items():
            q = p['cards'][name]
            print(f"    {name} {'sells GBP' if c['side'] == 'A' else 'sells USD'} L {c['L']:,.0f} T {c['T']:,.0f} "
                  f"A {c['A']:g} B {c['B']:g} | r1 {c['r1_amount_mean']:,.0f} @ {c['r1_price_median']:.4f} "
                  f"filled {pct(c['r1_filled'], 1)} | to MM {pct(c['to_mm'], 1)} Bureau {pct(c['to_bureau'], 1)} "
                  f"left {pct(c['left'], 1)} | P/L {pct(q['pl'])} Bureau-only {pct(q['bureau_only'])} "
                  f"gain {q['gain'] * 1e4:+.1f} bp")
        print()


def main():
    rows = []
    for key, group, label, rel, fee in runs():
        folder = RESULTS / rel
        if not (folder / 'cotrain_multi_eval.npz').exists():
            print(f'  (not on disk: {rel})')
            continue
        r = read_run(folder)
        assert abs(r['trader_fee'] - fee) < 1e-12, (rel, r['trader_fee'])
        rows.append(dict(key=key, group=group, label=label, folder=rel, **r))
    print()
    report(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'redo_numbers.json').write_text(json.dumps(rows, indent=1))
    print(f"wrote {(OUT / 'redo_numbers.json').relative_to(HERE.parent)}")


if __name__ == '__main__':
    main()
