'''
  robustness_runs.py -- how much of a 2% result is the game, and how much is
  the training? Seeds, the tau schedule, the frozen-market-maker protocol,
  and more random tables for the smallest population. One click runs
  everything that has not finished yet, several at a time, then prints the
  comparison.

  Every run is the base run's game -- the four cards (or a four-trader random
  table), traders' Bureau fee 0.03 and the market maker's 0.01 (the 2%
  spread), bid arm, offer size seen, two-way offers -- with ONE thing changed:

    seed142, seed242    the networks start from other random weights; the
                        training and evaluation paths are the base run's
    tau_const           no anneal: tau sits at its final 0.02 for both agents
                        from the first step
    tau_const_mm        the market maker's tau fixed at 0.02 from the first
                        step; the traders' anneal as the base run
    outer30_anneal700   ten more outer steps with the anneal held at the base
                        run's 700 steps, so every extra step is at the final
                        tau (the longer run on disk also changed the traders'
                        learning rate and stretched the anneal)
    frozen_cotrained    the base run's market maker, frozen, and the four
                        cards retrained against it from fresh -- what the
                        population runs do to THEIR market maker. Separates
                        what the population did from what the protocol did
    pop4s1, pop4s2      two more random four-trader tables, run as pop4s0

  Results go under Results/cotrain_multi/checks/, so nothing the population
  report reads changes; logs under checks/logs/. A job whose evaluation is
  on disk is skipped, and one already running (its marker names a live
  process) is not started twice.

  The comparison reads each run two ways: its final evaluation, and the mean
  and spread of its last ten validation readings. The 2% runs do not settle
  (none passes the flatness test), so one final snapshot can sit anywhere in
  a band.

  Usage: python robustness_runs.py          (runs what is missing, then the comparison)
         python robustness_runs.py smoke    (tiny sizes into Results_smoke: a quick test of the runner)
'''
import os
import sys
import json
import time
import subprocess
from pathlib import Path

import numpy as np

WORKERS = 4                  # jobs in flight at once; each is one process on one thread
BASE_RUN = 'T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way'     # the base run in Results/cotrain_multi
POP = 'mm0.01_bdc0.03_bid_size_2way'
GAME = dict(MM_ARCH='bid', SEE_OFFER_SIZE=True, TWO_WAY=True, MM_FEE=0.01, TRADER_FEE=0.03,
            N_POPULATION=0, ROSTER_SEED=0, ACCUM_GAMES=0)
RESPONDED = 'respond_on_cards/cotrain_multi_eval.npz'
EVALUATED = 'cotrain_multi_eval.npz'
JOBS = [   # name, what changes, the folder it writes, the file that means it is done
    dict(name='pop4s1', settings=dict(N_POPULATION=4, ROSTER_SEED=1, ACCUM_GAMES=250),
         folder='pop4s1_' + POP, done=RESPONDED),
    dict(name='pop4s2', settings=dict(N_POPULATION=4, ROSTER_SEED=2, ACCUM_GAMES=250),
         folder='pop4s2_' + POP, done=RESPONDED),
    dict(name='frozen_cotrained', settings=dict(RUN_TAG='_frozen_cotrained'), freeze=BASE_RUN,
         folder=BASE_RUN + '_frozen_cotrained', done=RESPONDED),
    dict(name='tau_const', settings=dict(RUN_TAG='_tau_const', TAU_TRADER=[0.02, 0.02], TAU_MM=[0.02, 0.02]),
         folder=BASE_RUN + '_tau_const', done=EVALUATED),
    dict(name='tau_const_mm', settings=dict(RUN_TAG='_tau_const_mm', TAU_MM=[0.02, 0.02]),
         folder=BASE_RUN + '_tau_const_mm', done=EVALUATED),
    dict(name='outer30_anneal700', settings=dict(RUN_TAG='_outer30_anneal700', N_OUTER=30, TAU_ANNEAL_STEPS=700),
         folder=BASE_RUN + '_outer30_anneal700', done=EVALUATED),
    dict(name='seed142', settings=dict(RUN_TAG='_seed142', SEED_INIT_TRADER=142, SEED_INIT_MM=142),
         folder=BASE_RUN + '_seed142', done=EVALUATED),
    dict(name='seed242', settings=dict(RUN_TAG='_seed242', SEED_INIT_TRADER=242, SEED_INIT_MM=242),
         folder=BASE_RUN + '_seed242', done=EVALUATED),
]
LATE = 10                    # validation readings averaged at the end of a run

HERE = Path(__file__).resolve().parent
SMOKE = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
RESULTS = HERE.parent / ('Results_smoke' if SMOKE else 'Results') / 'cotrain_multi'
CHECKS = RESULTS / 'checks'
LOGS = CHECKS / 'logs'

WORKER = '''
import sys, json, time
from pathlib import Path
here, job, smoke = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] == 'smoke'
sys.path.insert(0, here)
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import cotrain_multi as A
real_path = A.results_path
smoke_root = Path(here).parent / 'Results_smoke'


def rerouted(name):
    name = name.replace('cotrain_multi/', 'cotrain_multi/checks/', 1)
    if not smoke:
        return real_path(name)
    smoke_root.mkdir(parents=True, exist_ok=True)
    return smoke_root / name


A.results_path = rerouted
for key, value in list(job['game'].items()) + list(job['settings'].items()):
    setattr(A, key, value)
if smoke:
    A.N_OUTER, A.N_INNER, A.BATCH, A.CHECKS = 2, 3, 64, False
    A.VAL_PATHS, A.EVAL_PATHS, A.RECORD_PATHS = 256, 1024, 256
    A.ACCUM_GAMES = 32 if A.ACCUM_GAMES else 0
print('job', job['name'], 'settings', job['settings'], flush=True)
t0 = time.time()
if job.get('freeze'):
    cfg = A.make_config(A.TRADERS)
    mm_net = A.build_mm_net(cfg, A.SEED_INIT_MM)
    source = real_path('cotrain_multi/' + job['freeze']) / 'agents_final.pth'
    mm_net.load_state_dict(torch.load(source)['mm'])
    print('market maker frozen, loaded from', source, flush=True)
    A.respond_to_frozen_mm(A.TRADERS, mm_net, A.run_dir(cfg) / 'respond_on_cards')
else:
    A.run_full()
print('worker done in', round((time.time() - t0) / 60, 1), 'min', flush=True)
'''


def env_single_thread():
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def is_done(job):
    return (CHECKS / job['folder'] / job['done']).exists()


def is_running(job):
    marker = LOGS / (job['name'] + '.pid')
    if not marker.exists():
        return False
    try:
        os.kill(int(marker.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def run_jobs():
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job in JOBS:
        name = job['name']
        if is_done(job) and not SMOKE:
            print(f'  [done already] {name}', flush=True)
        elif is_running(job):
            print(f'  [running already, not started twice] {name}', flush=True)
        else:
            jobs.append(job)
    print(f'{len(jobs)} job(s) to run, {WORKERS} at a time; logs in {LOGS}', flush=True)
    running, t_all = [], time.time()
    while jobs or running:
        while jobs and len(running) < WORKERS:
            job = jobs.pop(0)
            name = job['name']
            if (is_done(job) and not SMOKE) or is_running(job):   # a second launch may have taken it
                print(f'  [skipped: finished or running elsewhere] {name}', flush=True)
                continue
            log = LOGS / (name + ('_smoke' if SMOKE else '') + '.log')
            payload = json.dumps(dict(job, game=GAME))
            handle = open(log, 'w')
            proc = subprocess.Popen([sys.executable, '-u', '-c', WORKER, str(HERE), payload,
                                     'smoke' if SMOKE else 'full'], cwd=str(HERE), stdout=handle,
                                    stderr=subprocess.STDOUT, env=env_single_thread())
            (LOGS / (name + '.pid')).write_text(str(proc.pid))
            running.append((job, proc, handle, time.time(), log))
            print(f'  started  {name:<18} -> {log.name}', flush=True)
        for entry in list(running):
            job, proc, handle, t0, log = entry
            if proc.poll() is None:
                continue
            handle.close()
            running.remove(entry)
            name, minutes = job['name'], (time.time() - t0) / 60
            (LOGS / (name + '.pid')).unlink(missing_ok=True)
            if proc.returncode == 0:
                print(f'  finished {name:<18} {minutes:6.1f} min', flush=True)
            else:
                lines = log.read_text().strip().splitlines()
                last = lines[-1] if lines else 'no output'
                print(f'  FAILED   {name:<18} {minutes:6.1f} min  <- {last}', flush=True)
        time.sleep(5)
    minutes = (time.time() - t_all) / 60
    print(f'all jobs done in {minutes:.1f} min', flush=True)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def final_reading(file):
    '''The final evaluation: the market maker's P/L, its Bureau flow and share,
    how much it filled, and the pound sellers' share of what it filled.'''
    z = np.load(file, allow_pickle=False)
    cap = float(z['capital_pounds'])
    amount, d = z['record_offered_amount'], z['record_direction']
    price, weight = z['record_quoted_price'], z['record_fill_weight']
    with np.errstate(divide='ignore', invalid='ignore'):
        pounds = np.where(d > 0, amount, np.where(amount > 0, amount / price, 0.0))
    filled = weight * pounds
    bought, paid = float(filled[d > 0].sum()), float(filled[d < 0].sum())
    games = amount.shape[0]
    at_bureau = (np.abs(z['record_mm_pounds_after_bureau'] - z['record_mm_pounds_after_fill']).sum()
                 + np.abs(z['record_mm_pounds_end'] - z['record_mm_pounds_after_bureau'][:, -1]).sum())
    share = float(z['bureau_share'])
    return dict(mm_bp=float(z['eval_mm_pl'].mean() * 1e4), flow=float(at_bureau / games / cap),
                share=share, filled=(bought + paid) / games / cap,
                bought=bought / max(bought + paid, 1e-300))


def late_reading(file):
    '''Mean and spread of the last LATE validation readings taken after a
    traders' block: the band a final snapshot is drawn from.'''
    c = np.load(file, allow_pickle=True)
    moved = np.array([str(m) for m in c['moved']])
    rows = np.nonzero(moved == 'trader')[0][-LATE:]
    mm = np.asarray(c['mm_pl'], float)[rows] * 1e4
    filled = np.asarray(c['filled'], float)[rows] * 100
    share = np.asarray(c['bureau_share'], float)[rows] * 100
    return dict(mm=(float(mm.mean()), float(mm.std())), filled=(float(filled.mean()), float(filled.std())),
                share=(float(np.nanmean(share)), float(np.nanstd(share))) if np.isfinite(share).any() else None)


def row(label, folder):
    eval_file, curve_file = folder / EVALUATED, folder / 'cotrain_multi_curve.npz'
    if not eval_file.exists():
        return f'  {label:<44} (not on disk yet)'
    final, late = final_reading(eval_file), late_reading(curve_file)
    mm, flow, share, filled, bought = (final['mm_bp'], final['flow'] * 100, final['share'] * 100,
                                       final['filled'] * 100, final['bought'] * 100)
    (mm_mean, mm_sd), (fill_mean, fill_sd) = late['mm'], late['filled']
    late_share = '      -      ' if late['share'] is None else '{:5.1f} +/- {:4.1f}'.format(*late['share'])
    return (f'  {label:<44} {mm:+7.1f} {flow:6.1f}% {share:6.1f}% {filled:6.1f}% {bought:6.1f}%   |'
            f' {mm_mean:+6.1f} +/- {mm_sd:4.1f}  {fill_mean:5.1f} +/- {fill_sd:4.1f}  {late_share}')


def report():
    real = HERE.parent / ('Results_smoke' if SMOKE else 'Results') / 'cotrain_multi'
    base = real / BASE_RUN
    header = ('  ' + ' ' * 44 + '  MM bp    flow  BdC shr  filled GBP-sell   | late: MM bp          filled %'
              '        BdC share %')
    lines = ['Final evaluation (MM bp; MM Bureau flow, % capital; Bureau share of its fills; filled, % capital;',
             'pound sellers share of its fills)  |  mean +/- sd of the last ten validation readings',
             '',
             'A. Co-trained on the four cards, 2% spread: what the schedule and the seed do',
             header,
             row('base run (seed 42, anneal 700 of 2000 steps)', base),
             row('seed 142', CHECKS / (BASE_RUN + '_seed142')),
             row('seed 242', CHECKS / (BASE_RUN + '_seed242')),
             row('tau constant 0.02, both agents', CHECKS / (BASE_RUN + '_tau_const')),
             row('tau constant 0.02, market maker only', CHECKS / (BASE_RUN + '_tau_const_mm')),
             row('30 outer steps, anneal held at 700 steps', CHECKS / (BASE_RUN + '_outer30_anneal700')),
             row('on disk: longer (30 outer, trader lr 1e-4)', real / (BASE_RUN + '_longer')),
             row('on disk: lr (trader lr 1e-4)', real / (BASE_RUN + '_lr')),
             '',
             'B. The four cards retrained from fresh against a FROZEN market maker, 2% spread',
             header,
             row('frozen co-trained MM (the base run)', CHECKS / (BASE_RUN + '_frozen_cotrained') / 'respond_on_cards'),
             row('frozen MM trained on 4 random traders, table 0', real / ('pop4s0_' + POP) / 'respond_on_cards'),
             row('frozen MM trained on 4 random traders, table 1', CHECKS / ('pop4s1_' + POP) / 'respond_on_cards'),
             row('frozen MM trained on 4 random traders, table 2', CHECKS / ('pop4s2_' + POP) / 'respond_on_cards'),
             row('frozen MM trained on 12 random traders', real / ('pop12s0_' + POP) / 'respond_on_cards'),
             row('frozen MM trained on 24 random traders', real / ('pop24s0_' + POP) / 'respond_on_cards'),
             '',
             'C. The random tables themselves, co-trained, 2% spread',
             header,
             row('4 random traders, table 0', real / ('pop4s0_' + POP)),
             row('4 random traders, table 1', CHECKS / ('pop4s1_' + POP)),
             row('4 random traders, table 2', CHECKS / ('pop4s2_' + POP)),
             row('12 random traders', real / ('pop12s0_' + POP)),
             row('24 random traders', real / ('pop24s0_' + POP))]
    text = chr(10).join(lines)
    print(chr(10) + text, flush=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    (CHECKS / 'robustness_report.txt').write_text(text + chr(10))


def main():
    run_jobs()
    report()


if __name__ == '__main__':
    main()
