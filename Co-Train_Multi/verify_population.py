'''
  verify_population.py -- an independent audit of the population experiment,
  read straight off the saved evaluation records. No training.

  For every run on disk (the random tables, the four cards read against each
  table's frozen market maker, the four-card co-trained runs and the
  robustness checks) it:

    settings  lists what the run was trained with, as saved in its
              evaluation file, so runs meant to be comparable can be
              seen to share them
    books     rebuilds every player's balances round by round from the
              recorded offers and fills, the traders' Bureau trades and the
              market maker's repay decisions, using only the rules:
                a fill moves money between one trader and the market maker
                at the offer's price; a Bureau trade converts at the true
                rate less the fee; the market maker clears what it owes at
                the fifth rate; nothing else moves anyone's money.
              The rebuilt balances must match what the game recorded
    scores    recomputes every player's P/L from the rebuilt balances with
              the rules' scoring, and compares with the saved P/L
    headline  recomputes Bureau flow (pounds the market maker moved at the
              Bureau per game, as a share of the game's capital) and
              Bureau share (of the pounds it filled, the share that went to
              the Bureau) from the rebuilt books, and compares with the
              saved values
    frozen    for the four cards read against a population market maker:
              the market maker they faced is the table's final market
              maker, weight for weight
    tables    each random table's cards, how lopsided it is, and the cards
              with no reason ever to convert (A + B at or below the fee)

  Writes Results/cotrain_multi/verification/verify_population.json.
  Usage: python verify_population.py
'''
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'Results' / 'cotrain_multi'
OUT = RESULTS / 'verification'
TOL = 1e-9          # money mismatch allowed, as a share of the game's capital (or the card's L)

SETTING_KEYS = ('fee', 'trader_bdc_fee', 'first', 'mm_level_init', 'n_outer', 'n_inner', 'batch',
                'lr', 'trader_plateau', 'tau_trader', 'tau_mm', 'tau_anneal_frac',
                'tau_anneal_steps', 'seed_init', 'run_tag', 'see_offer_size', 'see_direction',
                'see_queue', 'mm_arch', 'two_way', 'n_mm_feat', 'rounds', 'a0', 'sd',
                'n_population', 'roster_seed', 'capital_pounds')


def mm_bureau(P, D, repay, X, g):
    '''The market maker's Bureau trade, from the rules: cover a share of any
    short leg by selling the other leg at the true rate less its fee, never
    more than it holds.'''
    short_d, short_p = np.maximum(-D, 0.0), np.maximum(-P, 0.0)
    buy_d = np.minimum(repay * short_d, np.maximum(P, 0.0) * (1.0 - g) * X)
    buy_p = np.minimum(repay * short_p, np.maximum(D, 0.0) * (1.0 - g) / X)
    return P + buy_p - buy_d / ((1.0 - g) * X), D + buy_d - buy_p * X / (1.0 - g)


def settings_of(z):
    out = {}
    for key in SETTING_KEYS:
        if key in z.files:
            value = z[key]
            out[key] = value.tolist() if value.ndim else value.item()
        else:
            out[key] = None
    return out


def audit(folder):
    z = np.load(folder / 'cotrain_multi_eval.npz', allow_pickle=False)
    sides = [str(s) for s in z['sides']]
    names = [str(s) for s in z['traders']]
    cards = np.asarray(z['cards'], float)            # L, T, A, B per trader
    N = len(sides)
    f, g = float(z['trader_bdc_fee']), float(z['fee'])
    cap, a0 = float(z['capital_pounds']), float(z['a0'])
    p, a = z['record_quoted_price'], z['record_offered_amount']
    w, d = z['record_fill_weight'], z['record_direction']
    X, S = z['record_hidden_rate'], z['record_settlement_rate']
    owner = np.asarray(z['offer_trader'], int)
    G, K, R = p.shape
    worst = dict(fills_mm=0.0, fills_traders=0.0, bureau_traders=0.0, bureau_mm=0.0,
                 final_clear=0.0, mm_pl=0.0, trader_pl=0.0)
    hard = bool(np.all((w == 0.0) | (w == 1.0)))

    P, D = np.zeros(G), np.zeros(G)
    held = np.tile(cards[:, 0], (G, 1))            # start currency
    banked = np.zeros((G, N))                      # target currency
    bureau_pounds = np.zeros(G)                    # pounds the MM moved at the Bureau
    bought, paid = np.zeros(G), np.zeros(G)        # pounds bought from pound sellers / paid to dollar sellers
    both_sides_round = np.zeros(G, bool)
    for r in range(R):
        amount = w[:, :, r] * a[:, :, r]           # in the currency the offer sells
        price = np.where(amount > 0, p[:, :, r], 1.0)
        mm_pounds = np.where(d[:, :, r] > 0, amount, -amount / price)
        mm_dollars = np.where(d[:, :, r] > 0, -amount * price, amount)
        P, D = P + mm_pounds.sum(1), D + mm_dollars.sum(1)
        for k in range(K):
            i = owner[k]
            if sides[i] == 'A':                    # holds pounds, wants dollars
                held[:, i] -= mm_pounds[:, k]
                banked[:, i] -= mm_dollars[:, k]
            else:                                  # holds dollars, wants pounds
                held[:, i] -= mm_dollars[:, k]
                banked[:, i] -= mm_pounds[:, k]
        up = np.where(d[:, :, r] > 0, amount, 0.0).sum(1)
        down = np.where(d[:, :, r] < 0, amount / price, 0.0).sum(1)
        bought, paid = bought + up, paid + down
        both_sides_round |= (up > 0) & (down > 0)
        worst['fills_mm'] = max(worst['fills_mm'],
                                np.abs(P - z['record_mm_pounds_after_fill'][:, r]).max() / cap,
                                np.abs(D - z['record_mm_dollars_after_fill'][:, r]).max() / (a0 * cap))
        worst['fills_traders'] = max(
            worst['fills_traders'],
            (np.abs(held - z['record_trader_held_after_fill'][:, :, r]) / cards[:, 0]).max(),
            (np.abs(banked - z['record_trader_banked_after_fill'][:, :, r]) / cards[:, 0]).max())
        # traders at the Bureau: the amount each converted is its decision (read
        # from the record); what it receives for it is the rules'
        converted = z['record_trader_held_after_fill'][:, :, r] - z['record_trader_held_after_bureau'][:, :, r]
        worst['bureau_traders'] = max(worst['bureau_traders'], float(np.maximum(-converted, 0.0).max()
                                                                     / cards[:, 0].min()))
        rate = np.where(np.array(sides) == 'A', (1.0 - f) * X[:, r:r + 1], (1.0 - f) / X[:, r:r + 1])
        held, banked = held - converted, banked + converted * rate
        worst['bureau_traders'] = max(
            worst['bureau_traders'],
            (np.abs(banked - z['record_trader_banked_after_bureau'][:, :, r]) / cards[:, 0]).max())
        # the market maker at the Bureau: its repay share is its decision; the trade is the rules'
        P2, D2 = mm_bureau(P, D, z['record_repay_fraction'][:, r], X[:, r], g)
        bureau_pounds += np.abs(P2 - P)
        P, D = P2, D2
        worst['bureau_mm'] = max(worst['bureau_mm'],
                                 np.abs(P - z['record_mm_pounds_after_bureau'][:, r]).max() / cap,
                                 np.abs(D - z['record_mm_dollars_after_bureau'][:, r]).max() / (a0 * cap))
    # the fifth rate: the market maker must clear every short
    P2, D2 = mm_bureau(P, D, np.ones(G), S, g)
    bureau_pounds += np.abs(P2 - P)
    P, D = P2, D2
    worst['final_clear'] = max(np.abs(P - z['record_mm_pounds_end']).max() / cap,
                               np.abs(D - z['record_mm_dollars_end']).max() / (a0 * cap))
    mm_pl = (P + D / S) / cap
    worst['mm_pl'] = float(np.abs(mm_pl - z['record_mm_pl']).max())
    for i in range(N):
        L, T, A, B = cards[i]
        factor = 1.0 / S if sides[i] == 'A' else S
        wealth = held[:, i] * (1.0 - A) + (banked[:, i] - B * np.maximum(T - banked[:, i], 0.0)) * factor
        worst['trader_pl'] = max(worst['trader_pl'],
                                 float(np.abs((wealth - L) / L - z['record_trader_pl'][i]).max()))
    filled = bought + paid
    flow = float(bureau_pounds.mean() / cap)
    share = float(bureau_pounds.sum() / filled.sum()) if filled.sum() > 0 else float('nan')
    eval_mm = z['eval_mm_pl']
    edges = z['eval_trader_pl'] - z['eval_bdc_pl']
    offered = np.where(d > 0, a, a / X[:, None, :])     # pound-equivalent at the true rate
    out = dict(
        folder=str(folder.relative_to(RESULTS)), traders=names, sides=sides,
        cards=cards.tolist(), capital=cap, trader_fee=f, mm_fee=g,
        settings=settings_of(z), hard_verdicts=hard, games=G,
        worst_mismatch=worst,
        passed=bool(hard and max(worst.values()) < TOL),
        saved_flow=(float(z['bureau_pound_of_capital']) if 'bureau_pound_of_capital' in z.files
                    else float('nan')),     # the oldest runs did not save it
        saved_share=float(z['bureau_share']),
        flow=flow, share=share,
        filled_of_capital=float(filled.mean() / cap),
        bought_share=float(bought.sum() / filled.sum()) if filled.sum() > 0 else float('nan'),
        imbalance_floor=float(np.abs(bought - paid).sum() / filled.sum()) if filled.sum() > 0 else float('nan'),
        games_both_sides_same_round=float(both_sides_round.mean()),
        games_both_sides_any_round=float(((bought > 0) & (paid > 0)).mean()),
        mm_bp=float(eval_mm.mean() * 1e4), mm_bp_2se=float(2 * eval_mm.std() / np.sqrt(eval_mm.size) * 1e4),
        edge_pct=(edges.mean(1) * 100).tolist(),
        edge_2se_pct=(2 * edges.std(1) / np.sqrt(edges.shape[1]) * 100).tolist(),
        filled_by_trader=[float((w[:, owner == i] * offered[:, owner == i]).sum()
                                / max(offered[:, owner == i].sum(), 1e-300)) for i in range(N)],
        bid_by_round=np.asarray(z['bid_by_round']).tolist() if 'bid_by_round' in z.files else None,
        ask_by_round=np.asarray(z['ask_by_round']).tolist() if 'ask_by_round' in z.files else None,
        n_population=int(z['n_population']) if 'n_population' in z.files else 0,
        roster_seed=int(z['roster_seed']) if 'roster_seed' in z.files else 0)
    if out['n_population'] and names[0].startswith('R'):
        L, A, B = cards[:, 0], cards[:, 2], cards[:, 3]
        pounds = np.where(np.array(sides) == 'A', L, L / a0)
        dead = (A + B) <= f + 1e-12
        a_cap, b_cap = pounds[np.array(sides) == 'A'].sum(), pounds[np.array(sides) == 'B'].sum()
        live = ~dead
        a_live = pounds[live & (np.array(sides) == 'A')].sum()
        b_live = pounds[live & (np.array(sides) == 'B')].sum()
        out['table'] = dict(
            pound_sellers=int((np.array(sides) == 'A').sum()), dollar_sellers=int((np.array(sides) == 'B').sum()),
            pound_seller_capital=float(a_cap), dollar_seller_capital=float(b_cap),
            imbalance=float(abs(a_cap - b_cap) / (a_cap + b_cap)),
            dead_cards=[names[i] for i in np.nonzero(dead)[0]],
            dead_capital_share=float(pounds[dead].sum() / pounds.sum()),
            live_imbalance=float(abs(a_live - b_live) / max(a_live + b_live, 1e-300)))
    return out


def frozen_check(pop_folder):
    '''The four cards' market maker is the table's final market maker, weight
    for weight (torch only for loading the saved weights).'''
    try:
        import torch
    except ImportError:
        return None
    table = pop_folder / 'agents_final.pth'
    if not table.exists():
        table = pop_folder / 'agents_last.pth'
    cards = pop_folder / 'respond_on_cards' / 'agents_final.pth'
    if not (table.exists() and cards.exists()):
        return None
    mine, theirs = torch.load(table)['mm'], torch.load(cards)['mm']
    same = mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
    return dict(identical=bool(same), table_file=table.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted({f.parent for f in RESULTS.rglob('cotrain_multi_eval.npz')
                      if 'verification' not in f.parts})
    rows = []
    for folder in folders:
        try:
            row = audit(folder)
        except Exception as error:          # report and carry on
            row = dict(folder=str(folder.relative_to(RESULTS)), error=repr(error), passed=False)
        if (folder / 'respond_on_cards').is_dir():
            row['frozen'] = frozen_check(folder)
        rows.append(row)
        worst = max(row.get('worst_mismatch', {'x': float('nan')}).values())
        print(f"{'PASS' if row['passed'] else 'FAIL'}  worst {worst:8.1e}  "
              f"flow {row.get('flow', float('nan')) * 100:6.2f}% (saved {row.get('saved_flow', float('nan')) * 100:6.2f}%)  "
              f"share {row.get('share', float('nan')) * 100:6.2f}% (saved {row.get('saved_share', float('nan')) * 100:6.2f}%)  "
              f"frozen {row.get('frozen')}  {row['folder']}"
              + (f"  ERROR {row['error']}" if 'error' in row else ''), flush=True)
    (OUT / 'verify_population.json').write_text(json.dumps(rows, indent=1))
    print('written', OUT / 'verify_population.json', flush=True)


if __name__ == '__main__':
    main()
