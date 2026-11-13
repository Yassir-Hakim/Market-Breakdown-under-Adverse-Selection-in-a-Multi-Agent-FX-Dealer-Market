'''
  spread_ladder.py -- the four cards T1-T4 co-trained with the market maker at
  more fee spreads, so the baseline has a point at every step from 1% to 6%.

  One click (Run in VS Code) runs what is missing, four at a time, each in its
  own process on one thread. It is the same runner (run_arms.py) and the same
  settings that made the baselines already on disk; only the traders' Bureau
  fee changes. Nothing in cotrain_multi.py or run_arms.py is edited: the
  settings below are set on run_arms after it is imported.

  Spread = the traders' Bureau fee - the market maker's (0.01):
      0.02 -> 1%   0.025 -> 1.5%   0.03 -> 2%   0.04 -> 3%
      0.05 -> 4%   0.06 -> 5%      0.07 -> 6%
  0.02, 0.03 and 0.07 are on disk already, so they are skipped.

  Results: Results/cotrain_multi/T1-T2-T3-T4_mm0.01_bdc{fee}_bid_size_2way/
  Logs:    Results/cotrain_multi/logs/T1-T2-T3-T4_mm0.01_bdc{fee}_bid.log
  About one to two hours on this Mac with four running at once.

  Usage: python spread_ladder.py          (the real runs)
         python spread_ladder.py smoke    (tiny sizes into Results_smoke: a one-minute test)
'''
import run_arms as R

R.ARMS = (('bid', 'bid', True),)          # the bid arm, offer size seen: every run on disk
R.FEES = (0.01,)                          # the market maker's Bureau fee
R.TRADER_FEES = (0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07)
R.TABLE = 'ABAB'                          # T1, T2, T3, T4
R.POPULATIONS = (0,)                      # the four cards co-train; no random table
R.WORKERS = 4
R.THREADS_PER_WORKER = 1                  # as every run on disk (replay_check.json: threads 1)
R.TWO_WAY = True
R.SKIP_DONE = True
R.COMPARE_AFTER = False

if __name__ == '__main__':
    R.main()
