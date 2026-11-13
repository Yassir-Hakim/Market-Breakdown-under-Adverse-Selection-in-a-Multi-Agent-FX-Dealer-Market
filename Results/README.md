# Results

Saved runs. Every training and evaluation script writes here through `Mechanics/fx_mechanics.results_path`, so the folder names below are set by the code; renaming a folder stops the paper scripts from finding it.

Sides: **A** is the pound seller (starts in GBP, wants USD), **B** the dollar seller. Fees are written as decimals (`0.01` = 1%).

## `cotrain_multi/`: four traders and one market maker (`Co-Train_Multi/`)

Run folders are named `{traders}_mm{g}_bdc{f}_{arch}_size_2way`:

| Part | Meaning |
|---|---|
| `T1-T2-T3-T4` | the game's four trader cards |
| `pop{N}s{S}` | a random table of N traders drawn with roster seed S |
| `mm{g}` | the market maker's Bureau fee |
| `bdc{f}` | the traders' Bureau fee (fee spread = f − g) |
| `bid` / `turn` / `round` | the market maker's decision architecture (paper §6.3) |
| `size_2way` | the market maker sees offer sizes; offers may go either way |
| trailing tag | a variant of the base run (`_lr`, `_longer`, `_seed142`, …) |

Inside a run folder:

| File | Contents |
|---|---|
| `agents_final.pth`, `agents_last.pth` | trained weights (final and latest block) |
| `cotrain_multi_curve.npz` | the training curve, one reading per block |
| `cotrain_multi_eval.npz` | full evaluation on 200,000 paths (not in the repository: up to 415 MB per file; available on request) |
| `cotrain_multi_summary.txt` | headline numbers |
| `*.png` | diagnostic figures (`Co-Train_Multi/cotrain_multi_diagnostics.py`) |
| `respond_on_cards/` | population runs only: T1–T4 retrained against the table's frozen market maker |

Also in `cotrain_multi/`:

- `checks/`: robustness runs (`Co-Train_Multi/robustness_runs.py`, `more_tables.py`) and their report.
- `verification/`: audits of the saved runs (`verify_population.py`, `replay_check.py`, `redo_numbers.py`).
- `logs/`: training logs from `run_arms.py`.

## Other folders

| Folder | Written by | Contents |
|---|---|---|
| `cotrain/{side}_fee{g}[_BdC_FEE{f}]` | `Co-Train/cotrain.py` | one trader and one market maker co-trained |
| `mm_phase1/trader_{side}_g{g}` | `MM_Phase_1/train_traders.py` | traders trained against a fee-aware market maker |
| `mm_phase1/torch_{side}_R4_smoothed_g{g}` | `MM_Phase_1/torch_train_fee.py` | neural trader, four rounds, market-maker fee g |
| `mm_phase1/mm_{side}_fee{g}_TFA` | `MM_Phase_1/mm_train.py` | learned market maker against frozen traders |
| `mm_phase1/MM_FEE_AWARE_TRADER_NOT/` | `MM_Phase_1/mm_train.py` | earlier run: the market maker knows its fee, the trader does not |
| `mm_phase1/MM_fee_sweep_BdC_Fee_{f}` | `Experiments/MM_fee_experiment.py` | sweep over the market maker's fee |
| `fee_aware_dp/` | `Experiments/fee_aware_dp.py` | full game with the market maker's fee: DP next to the networks |
| `torch_{side}_R4_smoothed` | `NN_one_offer_game/torch_train.py` | neural trader against the hard-coded market maker |
| `ppo_A_R1`, `ppo_A_R4` | `NN_one_offer_game/PPO_train.py` | model-free PPO baseline |
| `phase2_1/` | `Experiments/Random_init_experiment/general_train.py` | trader trained across random markets |
| `v0_*.png`, `v1_plot.png`, `v2_*.png`, `optimal_strategy.png`, `set_path_dp.png` | `DP_Models/` | figures from the exact solutions |
| `v0_fee_aware_plot.png` | `MM_Phase_1/fee_aware_v0.py` | one-offer closed form with the market maker's fee |
| `v0_sigma_sweep.png` | `Experiments/v0_sigma_sweep.py` | one-offer result against volatility |
