'''
  replay_check.py -- was every population run trained by the code on disk
  today, with the settings its folder name says?

  Replays the opening of each run with today's cotrain_multi.py, on one
  thread as the runs were, and compares the validation readings with the
  ones the run saved in its curve. The code is deterministic on one machine
  and one thread, so the same code and settings give the same numbers to
  the last bit: a match means the saved run came from this code; a
  mismatch means something changed between the run and today.

    table stage   the table's validation before any training (fresh
                  networks), after the market maker's first block of 100
                  steps, and (four-trader tables) after the traders' first
                  block
    cards stage   the four cards' validation against the table's frozen
                  market maker before training, and after their first block

  Also compares the base four-card run (15 Sep) with the 30-outer-step
  check run (20 Sep), whose first 20 outer steps are the same schedule, row
  by row: nothing to replay, both curves are on disk.

  Writes Results/cotrain_multi/verification/replay_check.json.
  Usage: python replay_check.py
'''
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cotrain_multi as A

RESULTS = HERE.parent / 'Results' / 'cotrain_multi'
OUT = RESULTS / 'verification'
BASE = 'T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way'
RUNS = [   # folder, table size, table seed, traders' Bureau fee, replay the traders' first block too
    ('pop4s0_mm0.01_bdc0.03_bid_size_2way', 4, 0, 0.03, True),
    ('checks/pop4s1_mm0.01_bdc0.03_bid_size_2way', 4, 1, 0.03, True),
    ('checks/pop4s2_mm0.01_bdc0.03_bid_size_2way', 4, 2, 0.03, True),
    ('pop12s0_mm0.01_bdc0.03_bid_size_2way', 12, 0, 0.03, False),
    ('pop24s0_mm0.01_bdc0.03_bid_size_2way', 24, 0, 0.03, False),
]


def configure(n_pop, seed, fee):
    '''The settings every population job ran with (run_arms.py and
    robustness_runs.py set exactly these).'''
    A.MM_ARCH, A.SEE_OFFER_SIZE, A.MM_FEE = 'bid', True, 0.01
    A.TWO_WAY, A.TRADER_FEE = True, fee
    A.N_POPULATION, A.ROSTER_SEED = n_pop, seed
    A.ACCUM_GAMES = 250


def gap(v, curve, row):
    '''Largest difference between a replayed validation and the saved row.'''
    mm = abs(float(v['mm_pl'].mean()) - float(curve['mm_pl'][row]))
    traders = max(abs(float(pl.mean()) - float(saved))
                  for pl, saved in zip(v['trader_pls'], curve['trader_pl'][row]))
    return dict(row=row, moved=str(curve['moved'][row]), mm_gap=mm, trader_gap=traders,
                mm_saved=float(curve['mm_pl'][row]), mm_replayed=float(v['mm_pl'].mean()))


def opening(cfg, mm_net, trader_nets, curve, agents, label):
    '''The first validation and the first block(s) of training, compared row by row.'''
    opts = [torch.optim.AdamW(net.parameters(), lr=A.LR_TRADER) for net in trader_nets]
    scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=200)
              for opt in opts]
    opt_mm = torch.optim.AdamW(mm_net.parameters(), lr=A.LR_MM) if 'mm' in agents else None
    val_z = A.draw_paths(cfg, A.VAL_PATHS, A.SEED_PATH + 1, 'validation')
    val_order = A.draw_order(cfg, val_z.shape[0], A.SEED_PATH + A.ORDER_OFFSET + 1)
    val_bdc = [A.always_bdc_pl(spec, val_z) for spec in cfg.specs]
    rows = [gap(A.validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc), curve, 0)]
    print(f'  {label} row 0 (before training): {rows[-1]}', flush=True)
    step = {'trader': 0, 'mm': 0}
    for row, agent in enumerate(agents, start=1):
        t0 = time.time()
        step[agent], _, _ = A.train_block(agent, cfg, mm_net, trader_nets, opt_mm, opts, scheds,
                                          step[agent])
        rows.append(gap(A.validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc), curve, row))
        print(f'  {label} row {row} (after the {agent} block, {time.time() - t0:.0f} s): {rows[-1]}',
              flush=True)
    return rows


def replay(folder, n_pop, seed, fee, trader_block):
    configure(n_pop, seed, fee)
    run = RESULTS / folder
    out = dict(folder=folder)
    # the table stage: fresh networks, the market maker moves first
    cfg = A.make_config(A.random_roster(n_pop, seed))
    mm_net, trader_nets = A.build_agents(cfg)
    curve = np.load(run / 'cotrain_multi_curve.npz', allow_pickle=True)
    agents = ('mm', 'trader') if trader_block else ('mm',)
    out['table'] = opening(cfg, mm_net, trader_nets, curve, agents, f'{folder} table')
    # the cards stage: the table's final market maker, frozen; the cards fresh
    saved = run / 'agents_final.pth'
    if not saved.exists():
        saved = run / 'agents_last.pth'
    frozen = A.build_mm_net(cfg, A.SEED_INIT_MM)
    frozen.load_state_dict(torch.load(saved)['mm'])
    A.set_learning(frozen, False)
    cards = A.make_config(A.TRADERS)
    curve = np.load(run / 'respond_on_cards' / 'cotrain_multi_curve.npz', allow_pickle=True)
    out['cards'] = opening(cards, frozen, A.build_traders(cards), curve, ('trader',),
                           f'{folder} cards')
    return out


def base_against_check():
    '''The 15 Sep base run and the 20 Sep 30-outer-step run share the first 20
    outer steps of schedule (the anneal is 700 steps in both), so their first
    41 curve rows must be identical if the code did not change in between.'''
    base = np.load(RESULTS / BASE / 'cotrain_multi_curve.npz', allow_pickle=True)
    check = np.load(RESULTS / 'checks' / (BASE + '_outer30_anneal700') / 'cotrain_multi_curve.npz',
                    allow_pickle=True)
    rows = len(base['mm_pl'])
    mm = float(np.abs(base['mm_pl'] - check['mm_pl'][:rows]).max())
    traders = float(np.abs(base['trader_pl'] - check['trader_pl'][:rows]).max())
    return dict(rows=rows, mm_gap=mm, trader_gap=traders)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(threads=torch.get_num_threads(), torch=torch.__version__)
    report['base_vs_outer30'] = base_against_check()
    print('base run vs outer-30 check, first 20 outer steps:', report['base_vs_outer30'], flush=True)
    report['runs'] = []
    for folder, n_pop, seed, fee, trader_block in RUNS:
        t0 = time.time()
        report['runs'].append(replay(folder, n_pop, seed, fee, trader_block))
        print(f'{folder} done in {(time.time() - t0) / 60:.1f} min', flush=True)
        (OUT / 'replay_check.json').write_text(json.dumps(report, indent=1))
    worst = max(max(r['mm_gap'], r['trader_gap']) for run in report['runs']
                for stage in ('table', 'cards') for r in run[stage])
    print(f'largest gap over every replayed row: {worst:.3e}', flush=True)
    print('written', OUT / 'replay_check.json', flush=True)


if __name__ == '__main__':
    main()
