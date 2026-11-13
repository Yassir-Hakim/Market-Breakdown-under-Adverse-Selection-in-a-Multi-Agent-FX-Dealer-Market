'''
  more_tables.py -- two more random 12-trader tables for the population plot,
  in one click.

  The plot against the number of traders has three random tables at 4
  traders but only one at 12 and one at 24, so the spread between tables
  can only be read at 4. This adds 12-trader tables 1 and 2 (roster seeds 1
  and 2) at the 2% spread (traders' Bureau fee 0.03, MM fee 0.01), bid arm,
  offer size seen, two-way offers -- exactly the settings of every other
  population run -- through the runner robustness_runs.py already uses:
  co-train on the table, freeze the market maker, the four cards learn
  against it. Results go under Results/cotrain_multi/checks/ next to the
  4-trader tables 1 and 2, so nothing the population report reads changes.

  Afterwards it re-runs verify_population.py (which recomputes every run from
  its saved games, the new ones included) and population_simple.py (which
  redraws the three figures with a square per table), so the plot updates
  itself.

  Time: about 10 hours for both, two at a time, on the 8 GB laptop. Keep it
  plugged in; it keeps the Mac awake while it runs. A table already on disk
  is skipped and one already running is not started twice, so a second click
  is harmless. Close nothing else heavy while it runs (two 12-trader tables
  fit in 8 GB; a third job would not).

  Usage: python more_tables.py
'''
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robustness_runs as R

R.WORKERS = 2
R.JOBS = [
    dict(name='pop12s1', settings=dict(N_POPULATION=12, ROSTER_SEED=1, ACCUM_GAMES=250),
         folder='pop12s1_' + R.POP, done=R.RESPONDED),
    dict(name='pop12s2', settings=dict(N_POPULATION=12, ROSTER_SEED=2, ACCUM_GAMES=250),
         folder='pop12s2_' + R.POP, done=R.RESPONDED),
]


def main():
    if sys.platform == 'darwin':          # keep the Mac awake for as long as this process lives
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    R.run_jobs()
    for script in ('verify_population.py', 'population_simple.py'):
        print(f'\n---- {script} ----', flush=True)
        subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE))


if __name__ == '__main__':
    main()
