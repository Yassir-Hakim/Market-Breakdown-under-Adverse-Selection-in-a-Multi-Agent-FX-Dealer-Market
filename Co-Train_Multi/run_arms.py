import os
import sys
import json
import time
import subprocess
from pathlib import Path
"""
  run_arms.py -- runs the whole run matrix as PARALLEL PROCESSES, one arm per
  process, then the compares. Nothing in cotrain_multi.py changes: each
  worker imports it, sets MM_ARCH / SEE_OFFER_SIZE / MM_FEE /
  TRADERS after the import (every one of them is read at call time) and
  calls run_game, exactly as a hand-edited run does.

  Why processes and not threads. A training step is ~a hundred tiny torch
  ops on 1,000 games (matmuls of 1000 x 11 by 11 x 64); that is
  dispatch-bound, so torch's intra-op threads (it already uses every core
  by default) buy nothing and can cost. One arm per core, ONE thread per
  arm, is where the cores go. Five arms at a fee then take about one arm's
  time.

  Reproducibility. Different thread counts change the order of
  floating-point reductions inside BLAS, so a run with one thread is not
  bit-identical to the same run with the default threads (statistically
  the same). Every arm launched here uses the SAME thread setting, so the
  comparison stays paired and consistent; the checks (0-10) run inside
  each process against that process's own arithmetic. Do not mix arms run
  here with arms run by hand at another thread setting if bit-for-bit
  matters; for reading the results it does not.

  Logs: Results/cotrain_multi/logs/{traders}_mm{g}_{label}.log (the full
  run output, checks included). Then, per fee in FEES order (fee 0 first:
  its compare is the noise floor the fee-g compare reads),
  cotrain_multi_compare.py.

  Usage: python run_arms.py          (edit SETTINGS below)
         python run_arms.py smoke    (tiny sizes: a one-minute test of the runner)
"""

# ============================ SETTINGS ======================================
# ============================ SETTINGS ======================================
ARMS = (("bid", "bid", True),
        )                                        # label, MM_ARCH, SEE_OFFER_SIZE
FEES = (0.01,)                                    # MM fee; the rules' 0.01
TRADER_FEES = (0.02, 0.03)                        # the TRADERS' Bureau fee: night 1's spread ladder
TABLE = "ABAB"                                    # the canonical four cards
WORKERS = 2                                       # arms in flight at once. Fixed at 3: each pop50
                                                  # worker peaks near 3 GB, so cpu_count - 1 of them
                                                  # is what filled the machine
THREADS_PER_WORKER = 1                            # torch / BLAS threads inside each arm
TWO_WAY = True                                    # traders may sell back as well as forward; False = forward only
POPULATIONS = (12,)                      # 0: TABLE itself co-trains. n > 0: a random table of n
                                                  # co-trains and the market maker is then read on TABLE,
                                                  # frozen (cotrain_multi's population mode). Needs the bid
                                                  # or round arm: the turn arm's input grows with the table
ROSTER_SEED = 0                                   # which random table every population job uses
ACCUM_GAMES = 250                                 # population jobs backprop each training batch in
                                                  # chunks of this many games (same gradient, a
                                                  # fraction of the memory). Base-table jobs always
                                                  # run unchunked, so their runs stay bit-comparable
                                                  # with the artefacts already on disk
SKIP_DONE = True                                  # anything that finished before the crash is kept
COMPARE_AFTER = False
# ============================================================================
# ============================================================================

T1 = dict(name="T1", side="A", L=100_000, T=125_000, A=0.02, B=0.03)
T2 = dict(name="T2", side="B", L=200_000, T=155_000, A=0.03, B=0.02)
T3 = dict(name="T3", side="A", L=50_000, T=63_000, A=0.20, B=0.0)      # rules-doc cards
T4 = dict(name="T4", side="B", L=500_000, T=396_000, A=0.0, B=0.20)
TABLES = {"AB": None, "AA": [T1, T3], "ABAB": [T1, T2, T3, T4]}

HERE = Path(__file__).resolve().parent
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"

WORKER = r'''
import sys, json, time
sys.path.insert(0, {here!r})
import torch
torch.set_num_threads({threads})
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass          # some builds refuse this after import; the intra-op setting above is what matters
import cotrain_multi as A
job = json.loads({job!r})
A.MM_ARCH, A.SEE_OFFER_SIZE, A.MM_FEE = job["mm_arch"], job["see_size"], job["fee"]
A.TWO_WAY, A.TRADER_FEE = job["two_way"], job["trader_fee"]
A.N_POPULATION, A.ROSTER_SEED = job["n_pop"], job["roster_seed"]
A.ACCUM_GAMES = job["accum"]
if job["rows"] is not None:
    A.TRADERS = job["rows"]
if job["smoke"]:
    # tiny sizes, and the artefacts go to Results_smoke/, never the real folders
    A.N_OUTER, A.N_INNER, A.BATCH = 2, 3, 64
    A.VAL_PATHS, A.EVAL_PATHS, A.RECORD_PATHS = 256, 1024, 256
    from pathlib import Path as _P
    _scratch = _P({here!r}).parent / "Results_smoke"
    A.results_path = lambda name: (_scratch.mkdir(parents=True, exist_ok=True) or _scratch / name)
print(f"worker: arm {{job['label']}} ({{A.MM_ARCH}}, size {{A.SEE_OFFER_SIZE}}, "
      f"{{'two-way' if A.TWO_WAY else 'one-way'}}), fee {{A.MM_FEE}}, {{torch.get_num_threads()}} torch thread(s)", flush=True)
t0 = time.time()
s = A.run_full()
line = f"worker done in {{(time.time() - t0) / 60:.1f}} min: MM {{s['mm'] * 1e4:+.2f}} bp"
if job["n_pop"]:
    r = s["respond"]
    line += (f"; TABLE vs the frozen MM: MM {{r['mm'] * 1e4:+.2f}} bp, Bureau share "
             f"{{r['bureau_share'] * 100:.1f}}%, MM-Bureau flow {{r['bureau_of_capital'] * 100:.2f}}% of capital")
print(line, flush=True)
'''

COMPARE = r'''
import sys
sys.path.insert(0, {here!r})
import cotrain_multi as A, cotrain_multi_compare as C
A.MM_FEE = {fee}
A.TWO_WAY, A.TRADER_FEE = {two_way!r}, {trader_fee!r}
if {rows!r} is not None:
    A.TRADERS = {rows!r}
if {smoke!r}:
    from pathlib import Path as _P
    _scratch = _P({here!r}).parent / "Results_smoke"
    A.results_path = lambda name: (_scratch.mkdir(parents=True, exist_ok=True) or _scratch / name)
C.main()
'''


def env_single_thread():
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(THREADS_PER_WORKER)
    return env


def run_dir_of(A, label, mm_arch, see_size, fee, rows, trader_fee, n_pop=0):
    was_fee, was_trader = A.MM_FEE, A.TRADER_FEE
    was_pop, was_seed = A.N_POPULATION, A.ROSTER_SEED
    A.MM_FEE, A.TRADER_FEE = fee, trader_fee
    A.N_POPULATION, A.ROSTER_SEED = n_pop, ROSTER_SEED
    try:
        table = (A.random_roster(n_pop, ROSTER_SEED) if n_pop
                 else (A.TRADERS if rows is None else rows))
        cfg = A.make_config(table, mm_arch=mm_arch,
                            see_size=see_size, two_way=TWO_WAY)
        name = f"pop{n_pop}s{ROSTER_SEED}" if n_pop else cfg.name
        return A.run_dir(cfg), name
    finally:
        A.MM_FEE, A.TRADER_FEE = was_fee, was_trader
        A.N_POPULATION, A.ROSTER_SEED = was_pop, was_seed


def main():
    sys.path.insert(0, str(HERE))
    import cotrain_multi as A
    rows = TABLES[TABLE] if isinstance(TABLE, str) else list(TABLE)
    jobs = []
    for n_pop in POPULATIONS:
     for trader_fee in TRADER_FEES:
      for fee in FEES:
        for label, mm_arch, see_size in ARMS:
            if n_pop and mm_arch == "turn":
                print(f"  [skipped] pop{n_pop} {label}: the turn arm's flow block grows "
                      f"with the table, so its market maker cannot carry over to TABLE", flush=True)
                continue
            folder, name = run_dir_of(A, label, mm_arch, see_size, fee, rows, trader_fee, n_pop)
            done = (folder / "cotrain_multi_eval.npz").exists() and (
                not n_pop or (folder / "respond_on_cards" / "cotrain_multi_eval.npz").exists())
            if SKIP_DONE and not SMOKE and done:
                print(f"  [done already] {name} mm {fee:g} bdc {trader_fee:g} {label}: {folder}", flush=True)
                continue
            jobs.append(dict(label=label, mm_arch=mm_arch, see_size=see_size,
                             fee=fee, rows=rows, smoke=SMOKE, name=name, two_way=TWO_WAY,
                             trader_fee=trader_fee, n_pop=n_pop, roster_seed=ROSTER_SEED,
                             accum=ACCUM_GAMES if n_pop else 0))
    logs = (HERE.parent / "Results_smoke" if SMOKE else A.results_path("")) / "cotrain_multi" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} arm(s) to run, {WORKERS} at a time, {THREADS_PER_WORKER} thread(s) each"
          f"{' [SMOKE: tiny sizes]' if SMOKE else ''}; logs in {logs}", flush=True)
    running, results = [], []
    t_all = time.time()
    while jobs or running:
        while jobs and len(running) < WORKERS:
            job = jobs.pop(0)
            log = logs / f"{job['name']}_mm{job['fee']:g}_bdc{job['trader_fee']:g}_{job['label']}{'_smoke' if SMOKE else ''}.log"
            code = WORKER.format(here=str(HERE), threads=THREADS_PER_WORKER, job=json.dumps(job))
            handle = open(log, "w")
            proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(HERE), stdout=handle,
                                    stderr=subprocess.STDOUT, env=env_single_thread())
            running.append((job, proc, handle, time.time(), log))
            print(f"  started  {job['name']} mm {job['fee']:g} bdc {job['trader_fee']:g} {job['label']:<9} -> {log.name}", flush=True)
        for entry in list(running):
            job, proc, handle, t0, log = entry
            if proc.poll() is None:
                continue
            handle.close()
            running.remove(entry)
            minutes = (time.time() - t0) / 60
            ok = proc.returncode == 0
            results.append((job, ok, minutes))
            tail = ""
            if not ok:
                lines = log.read_text().strip().splitlines()
                tail = "  <- " + (lines[-1] if lines else "no output")
            print(f"  {'finished' if ok else 'FAILED  '} {job['name']} mm {job['fee']:g} bdc {job['trader_fee']:g} {job['label']:<9} "
                  f"{minutes:5.1f} min{tail}", flush=True)
        time.sleep(2)
    print(f"all arms done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    if COMPARE_AFTER and 0 in POPULATIONS:
        for trader_fee in TRADER_FEES:
            for fee in FEES:
                print(f"\n=== compare, MM fee {fee:g}, trader fee {trader_fee:g} ===", flush=True)
                code = COMPARE.format(here=str(HERE), fee=fee, rows=rows, smoke=SMOKE,
                                      two_way=TWO_WAY, trader_fee=trader_fee)
                subprocess.run([sys.executable, "-c", code], cwd=str(HERE), env=env_single_thread())


if __name__ == "__main__":
    main()