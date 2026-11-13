import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))           # Mechanics/, MM_Phase_1/
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""
  cotrain_multi_compare.py -- the MM's decision architectures SIDE BY
  SIDE, read PAIRED on the same evaluation paths. Reads the artefacts that
  cotrain_multi.py wrote for each arm; recomputes nothing about the
  learning, and only ONE thing about the game -- the cross-play matrix,
  which is evaluation only (each arm's MM against each arm's traders on the
  same paths). Which runs to read FOLLOWS cotrain_multi.py's TRADERS and
  MM_FEE. The (round, size off) point is this file's own "round" run with
  SEE_OFFER_SIZE = False; if that has not been run, a run of the same table
  under an older folder name is read in its place -- an artefact, not code.

  What it writes, in Results/cotrain_multi/{traders}_mm{g}_bdc{f}_arch_compare/:
    arch_compare_summary.txt   every number, and the reading rules applied
    arch_compare.png           (a) the MM's P/L per arm with the paired
                               differences; (b) each trader's edge over
                               always-Bureau per arm; (c) kappa by round --
                               the share of the Bureau fee the MM still
                               demands at its fitted cut, per direction;
                               (d) the Bureau share of filled volume and
                               each trader's filled share, per arm
    arch_crossplay.png         the MM of arm i against the traders of arm
                               j: "the MM sees more" (rows) against "the
                               traders adapted" (columns)
    arch_surface.png           the acceptance SURFACE: the fitted cut P/X
                               per direction against the offer's own size
                               and against the other side's size in the
                               same round -- "the level is a surface, not
                               a number": depth, and the reaction to the
                               other side's flow (flat in the round arm,
                               sloped in the turn arm if it learned to net)
    arch_compare.npz           the paired differences (the fee-0 file is the
                               noise floor for the fee-g reading)

  kappa. The fitted acceptance cut is P/X per direction and round (the
  eval npz's bid_by_round for pound sellers, ask_by_round for dollar
  sellers). The fee-aware rule sits at (1 - g) and 1/(1 - g); the naive
  rule at 1 and 1. kappa_A = (1 - bid) / g and kappa_B = (1 - 1/ask) / g
  therefore read: 1 = pass the whole fee through (fee-aware), 0 = zero
  spread (naive), in between = "true rate plus some proportion of the fee"
  below 0 = the MM pays up to attract flow. At g = 0 the
  cut minus 1 is reported instead.

  The reading rules (fixed before the runs; Meeting 5 summary, section 5.8):
    1. noise floor delta0 = the largest pairwise |difference| of MM P/L at
       g = 0 (nothing can be matched for profit when clearing is free); a
       difference at g > 0 is real only if |d| > max(2 SE, delta0)
    2. "wash out" -- the two architectures learn the same policy -- holds
       if turn - bid is not real AND the cross-play
       diagonal and off-diagonal agree within the same bound
    3. lookahead has value if turn - bid is real: report it in bp of capital
       and as a share of g x the filled pound volume (the fee at stake)
    4. the within-round refinement has value if bid - round is real
    5. the size refinement has value if (round, size) - nosize is real
    7. the learned MM has learned to NET if its P/L reaches the book rule's
       (arm - book+flatten within the bound, or above it: the book rule is
       myopic and a learner may beat it by carrying)
    6. reading the own price (turn, MM_FLOW all) beats the cut-only variant
       (turn-others) only if turn - turn-others is real -- the prediction
       is that it is not (an optimal rule is monotone in the own price
       here, so a cut already expresses it). Optional: the arm comparison
       does not depend on it
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cotrain_multi as CA
import MM_Phase_1.mm_train as M

# ============================ SETTINGS ======================================
ARMS = (("nosize", "round", False, "all"),     # the (round, size off) point: no size seen
        ("round", "round", True, "all"),
        ("bid", "bid", True, "all"),
        ("turn", "turn", True, "all"),           # whole round at once: every offer read
        ("turn-others", "turn", True, "others")) # optional variant: own price out of the block
                        # label, MM_ARCH, SEE_OFFER_SIZE, MM_FLOW; a missing run is skipped
RUN_DIRS = None         # None: derived from cotrain_multi.py's TRADERS and MM_FEE;
                        # else {label: path} to compare other runs on purpose
CROSS_PLAY = True       # each arm's MM against each arm's traders (evaluation only)
NOISE_FLOOR_NPZ = None  # None: this table's fee-0 compare file, if it exists
# ============================================================================

BP, PCT = 1e4, 100.0
ARM_COLOURS = {"nosize": "0.55", "round": "tab:blue", "bid": "tab:orange", "turn": "tab:green",
               "turn-others": "tab:olive"}


def _paired(a, b):
    """a minus b on the SAME paths: mean and 2 SE of the paired difference
    (mm_train.paired's convention, ddof = 1)."""
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), 2.0 * float(d.std(ddof=1)) / np.sqrt(len(d))


def arm_dir(label, mm_arch, see_size, mm_flow):
    if RUN_DIRS is not None:
        return Path(RUN_DIRS[label])
    cfg = CA.make_config(CA.TRADERS, mm_arch=mm_arch, see_size=see_size, mm_flow=mm_flow)
    run = CA.run_dir(cfg)
    if not (run / "cotrain_multi_eval.npz").exists() and (CA.run_dir_v4(cfg) / "cotrain_multi_eval.npz").exists():
        print(f"  [{label}] reading the run in its 12 Sep folder name: {CA.run_dir_v4(cfg)}", flush=True)
        return CA.run_dir_v4(cfg)
    if label == "nosize" and not cfg.two_way and not (run / "cotrain_multi_eval.npz").exists():
        legacy = CA.results_path(f"cotrain_multi/{cfg.name}_fee{CA.MM_FEE:g}")
        if (legacy / "cotrain_multi_eval.npz").exists():
            print(f"  [nosize] no (round, size off) run of this file; reading the older run in {legacy}", flush=True)
            return legacy
    return run


def load_arms():
    arms = []
    for label, mm_arch, see_size, mm_flow in ARMS:
        run = arm_dir(label, mm_arch, see_size, mm_flow)
        ev_file = run / "cotrain_multi_eval.npz"
        if not ev_file.exists():
            print(f"  [skipped] no run for arm {label} in {run}", flush=True)
            continue
        ev = np.load(ev_file, allow_pickle=False)
        ev = {key: ev[key] for key in ev.files}
        cfg = CA.make_config(CA.TRADERS, mm_arch=mm_arch, see_size=see_size, mm_flow=mm_flow)
        assert list(ev["traders"]) == list(cfg.traders), f"arm {label}: a different trader table"
        assert float(ev["fee"]) == CA.MM_FEE, f"arm {label}: a different MM fee"
        assert int(ev["n_mm_feat"]) == cfg.n_mm_feat, (
            f"arm {label}: {int(ev['n_mm_feat'])} MM inputs in the run, {cfg.n_mm_feat} expected")
        if "mm_flow" in ev:
            assert str(ev["mm_flow"]) == mm_flow or mm_arch != "turn", f"arm {label}: a different MM_FLOW"
        arms.append(dict(label=label, mm_arch=mm_arch, see_size=see_size, mm_flow=mm_flow,
                         run=run, ev=ev, cfg=cfg))
    assert len(arms) >= 2, "fewer than two arms to compare"
    ref = arms[0]["ev"]["eval_bdc_pl"]
    for arm in arms[1:]:
        assert np.array_equal(arm["ev"]["eval_bdc_pl"], ref), (
            f"arm {arm['label']} was evaluated on different paths: the comparison would not be paired")
    return arms


def filled_share_by_trader(ev):
    """Filled share of each TRADER's offered volume, in pound-equivalent at
    the true rate (two-way, a trader's offers can be in either currency)."""
    X = ev["record_hidden_rate"][:, None, :]
    amounts = np.where(ev["record_direction"] > 0, ev["record_offered_amount"], ev["record_offered_amount"] / X)
    filled = (ev["record_fill_weight"] * amounts).sum(axis=(0, 2))
    offered = amounts.sum(axis=(0, 2))
    own = ev["offer_trader"] if "offer_trader" in ev else np.arange(len(filled))
    n = int(own.max()) + 1
    filled_t = np.array([filled[own == i].sum() for i in range(n)])
    offered_t = np.array([offered[own == i].sum() for i in range(n)])
    return filled_t / np.maximum(offered_t, 1e-300)


def kappa(ev, fee):
    """(kappa_A by round, kappa_B by round); nan where the cut is unidentified."""
    bid, ask = ev["bid_by_round"], ev["ask_by_round"]
    if fee > 0.0:
        return (1.0 - bid) / fee, (1.0 - 1.0 / ask) / fee
    return bid - 1.0, ask - 1.0


def cross_play(arms):
    """MM of arm i against the traders of arm j, on the evaluation paths.
    Returns mm[i, j] (mean P/L), edge[i, j, t] (each trader's paired edge
    over always-Bureau), and the diagonal's gap to the stored evaluation."""
    n = len(arms)
    loaded = []
    for arm in arms:
        file = arm["run"] / "agents_final.pth"
        if not file.exists():
            print(f"  [cross-play skipped] no agents_final.pth in {arm['run']}", flush=True)
            return None
        saved = torch.load(file)
        mm_net, trader_nets = CA.build_agents(arm["cfg"])
        mm_net.load_state_dict(saved["mm"])
        for net, state in zip(trader_nets, saved["traders"]):
            net.load_state_dict(state)
        for net in [mm_net] + trader_nets:
            CA.set_learning(net, False)
        loaded.append((mm_net, trader_nets))
    cfg0 = arms[0]["cfg"]
    n_eval = len(arms[0]["ev"]["eval_mm_pl"])
    # the runs' evaluation paths: cotrain_mm_arch.evaluate's draw. The count
    # is taken from the run (a run at another EVAL_PATHS is still readable)
    # and the draw is CERTIFIED against the stored always-Bureau P/Ls, which
    # are a function of the paths alone
    z = CA.draw_paths(cfg0, CA.EVAL_PATHS, CA.SEED_PATH + 2, "evaluation")
    if z.shape[0] != n_eval:
        z = CA.draw_paths(cfg0, n_eval, CA.SEED_PATH + 2, "evaluation")
    order = CA.draw_order(cfg0, z.shape[0], CA.SEED_PATH + CA.ORDER_OFFSET + 2)
    bdc = [CA.always_bdc_pl(spec, z) for spec in cfg0.specs]
    if z.shape[0] != n_eval or not all(np.array_equal(b.numpy(), arms[0]["ev"]["eval_bdc_pl"][t])
                                       for t, b in enumerate(bdc)):
        print(f"  [cross-play skipped] the runs' evaluation paths cannot be re-drawn here "
              f"({n_eval:,} in the run, EVAL_PATHS {CA.EVAL_PATHS:,})", flush=True)
        return None
    mm = np.full((n, n), np.nan)
    mm_2se = np.full((n, n), np.nan)
    edge = np.full((n, n, cfg0.n), np.nan)
    diag_gap = 0.0
    with torch.no_grad():
        for i, arm in enumerate(arms):
            for j, (_, trader_nets) in enumerate(loaded):
                mm_pl, trader_pls, _ = CA.rollout(arm["cfg"], loaded[i][0], trader_nets, z, order,
                                                  tau=None, fee=CA.MM_FEE)
                mm[i, j] = float(mm_pl.mean())
                mm_2se[i, j] = 2.0 * float(mm_pl.std()) / np.sqrt(len(mm_pl))
                for t in range(cfg0.n):
                    edge[i, j, t] = _paired(trader_pls[t].numpy(), bdc[t].numpy())[0]
                if i == j:
                    diag_gap = max(diag_gap, float(np.abs(mm_pl.numpy() - arm["ev"]["eval_mm_pl"]).max()))
    return dict(mm=mm, mm_2se=mm_2se, edge=edge, diag_gap=diag_gap)


def noise_floor(cfg):
    file = (Path(NOISE_FLOOR_NPZ) if NOISE_FLOOR_NPZ is not None
            else CA.results_path(f"cotrain_multi/{cfg.name}_mm0_bdc{cfg.params.bdc_fee:g}_arch_compare") / "arch_compare.npz")
    if CA.MM_FEE == 0.0 or not file.exists():
        return None, file
    d = np.load(file, allow_pickle=False)
    return float(np.abs(d["pair_delta_mm"]).max()), file


def compare(arms):
    cfg, fee = arms[0]["cfg"], CA.MM_FEE
    names, N, rounds = list(cfg.traders), cfg.n, cfg.rounds
    labels = [arm["label"] for arm in arms]
    out = []
    w = out.append
    w(f"MM ARCHITECTURES SIDE BY SIDE: traders {cfg.name} (sides {''.join(s.side for s in cfg.specs)}), "
      f"MM fee {fee}, {len(arms[0]['ev']['eval_mm_pl']):,} evaluation paths, paired")
    w("=" * 92)
    w("arms")
    for arm in arms:
        ev = arm["ev"]
        w(f"  {arm['label']:<8} MM_ARCH {arm['mm_arch']:<6} flow {arm['mm_flow']:<6} size "
          f"{'on ' if arm['see_size'] else 'off'}  {int(ev['n_mm_feat']):>2} MM inputs  "
          f"{'two-way, ' + str(int(ev['n_offers'])) + ' offers a round' if bool(ev.get('two_way', False)) else 'one-way'}  run {arm['run']}")
    w("")
    # ---- (1) the MM, per arm and pairwise ---------------------------------
    w("(1) the MM's P/L, bp of the game's capital (hard game)")
    mm = {arm["label"]: arm["ev"]["eval_mm_pl"] for arm in arms}
    for arm in arms:
        pl = mm[arm["label"]]
        w(f"  {arm['label']:<8} {pl.mean() * BP:+8.2f} +/- {2 * pl.std(ddof=1) / np.sqrt(len(pl)) * BP:.2f}   "
          f"Bureau share of filled pound volume {float(arm['ev']['bureau_share']) * PCT:5.1f}%   "
          f"filled {float(arm['ev']['filled']) * PCT:5.2f}% of offered volume   "
          f"agrees with {str(arm['ev']['benchmark_rule'])} on {float(arm['ev']['agree']) * PCT:.1f}%")
    delta0, floor_file = noise_floor(cfg)
    pairs, pair_delta, pair_2se = [], [], []
    w("  pairwise, first minus second (paired):")
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = labels[j], labels[i]        # later arm minus earlier: turn - bid, bid - round, ...
            d, se = _paired(mm[a], mm[b])
            pairs.append(f"{a} - {b}")
            pair_delta.append(d)
            pair_2se.append(se)
            bound = max(se, delta0) if delta0 is not None else se
            verdict = "REAL" if abs(d) > bound else "within noise"
            w(f"    {a:>8} - {b:<8} {d * BP:+8.2f} +/- {se * BP:.2f} bp   [{verdict}"
              + (f", bound max(2 SE, delta0 {delta0 * BP:.2f}) = {bound * BP:.2f}]" if delta0 is not None
                 else (", bound 2 SE; no fee-0 noise floor found]" if fee > 0 else "]")))
    if fee == 0.0:
        w(f"  at fee 0 nothing can be matched for profit: the largest |difference| above, "
          f"{np.abs(pair_delta).max() * BP:.2f} bp, is the NOISE FLOOR delta0 for the fee-g reading")
    elif delta0 is None:
        w(f"  (no fee-0 compare file at {floor_file}: run the three arms at MM_FEE = 0 first "
          f"for the noise floor)")
    w("")
    # ---- (2) the traders ---------------------------------------------------
    w("(2) each trader: P/L, edge over always-Bureau (paired), filled share of its offered volume")
    for t, name in enumerate(names):
        for arm in arms:
            ev = arm["ev"]
            e, se = _paired(ev["eval_trader_pl"][t], ev["eval_bdc_pl"][t])
            w(f"  {name} ({cfg.specs[t].side})  {arm['label']:<8} P/L {ev['eval_trader_pl'][t].mean() * PCT:+.4f}%  "
              f"edge {e * PCT:+.4f}% +/- {se * PCT:.4f}%  filled {filled_share_by_trader(ev)[t] * PCT:5.2f}%")
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                d, se = _paired(arms[j]["ev"]["eval_trader_pl"][t], arms[i]["ev"]["eval_trader_pl"][t])
                w(f"      {labels[j]:>8} - {labels[i]:<8} {d * PCT:+.4f}% +/- {se * PCT:.4f}%")
    w("")
    # ---- (3) kappa ----------------------------------------------------------
    w("(3) kappa by round: the share of the Bureau fee the MM still demands at its fitted cut "
      "(1 = fee-aware pass-through, 0 = zero spread, < 0 = pays up to attract flow)"
      if fee > 0 else "(3) the fitted cut minus 1 by round (fee 0: kappa is undefined)")
    for arm in arms:
        kA, kB = kappa(arm["ev"], fee)
        w(f"  {arm['label']:<8} pound sellers (bid): " + "  ".join(f"r{r + 1} {v:+.2f}" for r, v in enumerate(kA))
          + "   dollar sellers (ask): " + "  ".join(f"r{r + 1} {v:+.2f}" for r, v in enumerate(kB)))
    w("")
    # ---- (4) rules ----------------------------------------------------------
    w("(4) the hardcoded MMs against each arm's traders, MM P/L bp (the rule faces THAT arm's traders)")
    for arm in arms:
        ev = arm["ev"]
        w(f"  {arm['label']:<8} " + "  ".join(f"{str(r):<18} {ev['eval_rule_mm_pl'][k].mean() * BP:+7.2f}"
                                             for k, r in enumerate(ev["rule_names"])))
    book = {}
    if any("book+flatten" in [str(r) for r in arm["ev"]["rule_names"]] for arm in arms):
        w("  the learned MM against the BOOK RULE (paired, arm MM - book+flatten, bp): the reference for")
        w("  'did it learn to net?' -- the book rule reads every offer and nets crossing pairs and inventory")
        for arm in arms:
            names_r = [str(r) for r in arm["ev"]["rule_names"]]
            if "book+flatten" not in names_r:
                continue
            k = names_r.index("book+flatten")
            d, se = _paired(arm["ev"]["eval_mm_pl"], arm["ev"]["eval_rule_mm_pl"][k])
            book[arm["label"]] = (d, se)
            w(f"    {arm['label']:<8} {d * BP:+8.2f} +/- {se * BP:.2f} bp")
    w("")
    # ---- (5) cross-play -----------------------------------------------------
    xp = cross_play(arms) if CROSS_PLAY else None
    if xp is not None:
        w("(5) cross-play, MM P/L bp: the MM of the ROW arm against the traders of the COLUMN arm, same paths")
        w("  " + " " * 8 + "".join(f"{l:>12}" for l in labels) + "     (row = MM, column = traders)")
        for i, arm in enumerate(arms):
            w(f"  {arm['label']:<8}" + "".join(f"{xp['mm'][i, j] * BP:+12.2f}" for j in range(len(arms))))
        w(f"  diagonal vs the stored evaluations: max |gap| {xp['diag_gap']:.1e}"
          + ("  (exact)" if xp["diag_gap"] == 0.0 else
             "  (not exact: a different machine or torch build than the run's)" if xp["diag_gap"] > 1e-9 else ""))
        for t, name in enumerate(names):
            w(f"  {name}'s edge over always-Bureau, % (row = MM, column = traders):")
            for i, arm in enumerate(arms):
                w(f"    {arm['label']:<8}" + "".join(f"{xp['edge'][i, j, t] * PCT:+12.4f}" for j in range(len(arms))))
        # rule 2's second half: does the MM's identity matter once the traders are fixed?
        w("  row spread (max - min across columns, the traders' adaptation) and column spread "
          "(across rows, the MM's information), bp:")
        w("    rows: " + "  ".join(f"{labels[i]} {np.ptp(xp['mm'][i]) * BP:.2f}" for i in range(len(arms)))
          + "   columns: " + "  ".join(f"{labels[j]} {np.ptp(xp['mm'][:, j]) * BP:.2f}" for j in range(len(arms))))
    w("")
    # ---- (6) the reading rules ------------------------------------------------
    w("(6) reading rules (fixed before the runs)")
    by = {p: (d, se) for p, d, se in zip(pairs, pair_delta, pair_2se)}

    def real(p):
        if p not in by:
            return None
        d, se = by[p]
        bound = max(se, delta0) if delta0 is not None else se
        return abs(d) > bound
    if fee == 0.0:
        w("  rule 1: this IS the fee-0 run; its largest |difference| is the noise floor for the fee-g reading")
    else:
        w(f"  rule 1: bound = max(2 SE, delta0 {delta0 * BP:.2f} bp)" if delta0 is not None
          else "  rule 1: bound = 2 SE only (no fee-0 noise floor found)")
        filled_pounds = float(CA.pound_volume({k[len("record_"):]: v for k, v in arms[-1]["ev"].items()
                                               if k.startswith("record_")}).mean()) if "record_fill_weight" in arms[-1]["ev"] else float("nan")
        for p, rule, text in (("turn - bid", 3, "lookahead (turn - bid)"),
                              ("bid - round", 4, "the within-round refinement (bid - round)"),
                              ("round - nosize", 5, "the size refinement (round - nosize)"),
                              ("turn - turn-others", 6, "the freedom to read the own price (turn - turn-others)")):
            r = real(p)
            if r is None:
                w(f"  rule {rule}: {text}: arm missing")
                continue
            d, se = by[p]
            line = f"  rule {rule}: {text} {d * BP:+.2f} +/- {se * BP:.2f} bp: {'REAL' if r else 'within noise'}"
            if p == "turn - bid" and np.isfinite(filled_pounds) and filled_pounds > 0:
                stake = fee * filled_pounds / cfg.capital
                line += (f"; the fee at stake (g x filled pound volume on the turn arm's record paths) "
                         f"is {stake * BP:.2f} bp of capital"
                         + (f", so lookahead recovered {d / stake:+.1%} of it" if stake * BP >= 0.05
                            else " (too little filled to read the difference as a share of it)"))
            w(line)
        for label in ("turn", "bid", "round"):
            if label in book:
                d, se = book[label]
                bound = max(se, delta0) if delta0 is not None else se
                verdict = ("reaches the book rule" if abs(d) <= bound else
                           ("BEATS the book rule (carrying beyond the myopic rule)" if d > 0
                            else "falls short of the book rule"))
                w(f"  rule 7: {label} vs book+flatten {d * BP:+.2f} +/- {se * BP:.2f} bp: {verdict}")
        r = real("turn - bid")
        if r is not None and xp is not None:
            spread_ok = np.ptp(xp["mm"]) * BP <= max(np.abs(pair_2se).max(), delta0 or 0.0) * BP
            w(f"  rule 2: wash out {'HOLDS' if (not r and spread_ok) else 'does NOT hold'} at N = {N}: "
              f"turn - bid {'within noise' if not r else 'real'}; cross-play spread {np.ptp(xp['mm']) * BP:.2f} bp")
    text = "\n".join(out)
    return text, dict(pairs=pairs, pair_delta=pair_delta, pair_2se=pair_2se, xp=xp)


def figure(arms, result, folder):
    cfg, fee = arms[0]["cfg"], CA.MM_FEE
    labels = [arm["label"] for arm in arms]
    colours = [ARM_COLOURS.get(l, "tab:grey") for l in labels]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    # (a) MM P/L per arm
    ax = axes[0, 0]
    means = [arm["ev"]["eval_mm_pl"].mean() * BP for arm in arms]
    errs = [2 * arm["ev"]["eval_mm_pl"].std(ddof=1) / np.sqrt(len(arm["ev"]["eval_mm_pl"])) * BP for arm in arms]
    ax.bar(labels, means, yerr=errs, color=colours, capsize=4)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("MM P/L, bp of the game's capital")
    ax.set_title("(a) the MM per arm (2 SE); paired differences in the text")
    txt = "\n".join(f"{p}: {d * BP:+.2f} +/- {se * BP:.2f} bp"
                    for p, d, se in zip(result["pairs"], result["pair_delta"], result["pair_2se"]))
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=.8))
    # (b) traders' edges per arm
    ax = axes[0, 1]
    n_arm, N = len(arms), cfg.n
    width = 0.8 / n_arm
    for k, arm in enumerate(arms):
        ev = arm["ev"]
        e = [_paired(ev["eval_trader_pl"][t], ev["eval_bdc_pl"][t]) for t in range(N)]
        ax.bar(np.arange(N) + (k - (n_arm - 1) / 2) * width, [v[0] * PCT for v in e], width,
               yerr=[v[1] * PCT for v in e], color=colours[k], capsize=3, label=arm["label"])
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels([f"{n} ({s.side})" for n, s in zip(cfg.traders, cfg.specs)])
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("edge over always-Bureau, % of the trader's capital")
    ax.set_title("(b) each trader's edge per arm (paired, 2 SE)")
    ax.legend(fontsize=8)
    # (c) kappa by round
    ax = axes[1, 0]
    r = np.arange(1, cfg.rounds + 1)
    for k, arm in enumerate(arms):
        kA, kB = kappa(arm["ev"], fee)
        ax.plot(r, kA, "o-", color=colours[k], label=f"{arm['label']} bid (pound sellers)")
        ax.plot(r, kB, "s--", color=colours[k], label=f"{arm['label']} ask (dollar sellers)")
    if fee > 0:
        ax.axhline(1, color="k", lw=.8, ls=":")
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.text(r[-1], 1.02, "fee-aware (1)", ha="right", fontsize=8)
        ax.text(r[-1], 0.02, "zero spread (0)", ha="right", fontsize=8)
        ax.set_ylabel("kappa: share of g the MM still demands at its cut")
    else:
        ax.set_ylabel("fitted cut / X - 1 (fee 0)")
    ax.set_xticks(r)
    ax.set_xlabel("round")
    ax.set_title("(c) the MM's fitted acceptance cut, by round and direction")
    ax.legend(fontsize=7, ncol=2)
    # (d) Bureau share and filled shares
    ax = axes[1, 1]
    cats = ["Bureau share"] + [f"{n} filled" for n in cfg.traders]
    for k, arm in enumerate(arms):
        ev = arm["ev"]
        vals = [float(ev["bureau_share"]) * PCT] + list(filled_share_by_trader(ev) * PCT)
        ax.bar(np.arange(len(cats)) + (k - (n_arm - 1) / 2) * width, vals, width, color=colours[k],
               label=arm["label"])
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_ylabel("%")
    ax.set_title("(d) Bureau share of filled volume; each trader's filled share")
    ax.legend(fontsize=8)
    fig.suptitle(f"MM architectures side by side -- traders {cfg.name}, MM fee {fee}, "
                 f"{len(arms[0]['ev']['eval_mm_pl']):,} paired evaluation paths", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "arch_compare.png", dpi=150)
    plt.close(fig)
    xp = result["xp"]
    if xp is not None:
        fig, ax = plt.subplots(figsize=(1.6 * len(arms) + 3, 1.2 * len(arms) + 2))
        im = ax.imshow(xp["mm"] * BP, cmap="RdYlGn")
        for i in range(len(arms)):
            for j in range(len(arms)):
                ax.text(j, i, f"{xp['mm'][i, j] * BP:+.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([f"traders of {l}" for l in labels], fontsize=8)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([f"MM of {l}" for l in labels], fontsize=8)
        ax.set_title("cross-play: MM P/L, bp of capital, same evaluation paths", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout()
        fig.savefig(folder / "arch_crossplay.png", dpi=150)
        plt.close(fig)


def _cut_by_bin(u, volume, fill, key, side, n_bins=5):
    """The fitted cut P/X (mm_train.operating_threshold) within quantile
    bins of `key`; nan where a bin has no accepted or no rejected volume."""
    if len(key) == 0:
        return np.array([]), np.array([])
    edges = np.quantile(key, np.linspace(0, 1, n_bins + 1))
    centres, cuts = [], []
    for b in range(n_bins):
        sel = (key >= edges[b]) & (key <= edges[b + 1]) if b == n_bins - 1 else (key >= edges[b]) & (key < edges[b + 1])
        centres.append(float(key[sel].mean()) if sel.any() else np.nan)
        if sel.sum() == 0 or (volume[sel] * fill[sel]).sum() <= 0 or (volume[sel] * (1 - fill[sel])).sum() <= 0:
            cuts.append(np.nan)
            continue
        cuts.append(M.operating_threshold(u[sel], volume[sel], fill[sel], side)[0])
    return np.array(centres), np.array(cuts)


def surface(arms, folder):
    """arch_surface.png: the fitted cut against the offer's own size and
    against the other side's size in the same round, per direction and arm."""
    cfg, fee = arms[0]["cfg"], CA.MM_FEE
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    out = []
    for row_i, (d, side, name) in enumerate(((1.0, "A", "pound sellers (bid)"), (-1.0, "B", "dollar sellers (ask)"))):
        for arm in arms:
            ev = arm["ev"]
            q = ev["record_quoted_price"]; a = ev["record_offered_amount"]; f = ev["record_fill_weight"]
            dirs = ev["record_direction"]; X = ev["record_hidden_rate"][:, None, :]
            sel = dirs == d
            if not sel.any():
                continue
            u = (q / X)[sel]
            vol = a[sel]
            fill = f[sel]
            unit = cfg.capital if d > 0 else cfg.params.a0 * cfg.capital
            own = (a / unit)[sel]
            # the other side's offered pound-equivalent in the same round
            other_pounds = np.where(dirs == -d, np.where(dirs > 0, a, a / X), 0.0).sum(axis=1, keepdims=True)   # at the true rate
            other = np.broadcast_to(other_pounds, a.shape)[sel] / cfg.capital
            colour = ARM_COLOURS.get(arm["label"], "tab:grey")
            for col_i, (key, label) in enumerate(((own, "own size"), (other, "other side"))):
                c, cut = _cut_by_bin(u, vol, fill, key, side)
                axes[row_i, col_i].plot(c, cut, "o-", color=colour, label=arm["label"])
                out.append(f"  {name:<20} {arm['label']:<8} cut vs {label:<10}: "
                           + "  ".join(f"{x:.2f}->{y:.4f}" if np.isfinite(y) else f"{x:.2f}->  nan " for x, y in zip(c, cut)))
        for col_i, xlabel in enumerate(("the offer's own size (offer units: capital, or a0 x capital for dollars)",
                                        "the OTHER side's offered pound-equivalent in the round, / capital")):
            ax = axes[row_i, col_i]
            ref = (1 - fee) if d > 0 else 1 / (1 - fee)
            ax.axhline(ref, color="k", lw=.8, ls=":")
            ax.axhline(1.0, color="k", lw=.8, ls="--")
            ax.set_xlabel(xlabel)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.ticklabel_format(useOffset=False, axis="x")
            ax.set_ylabel(f"fitted cut P/X, {name}")
            ax.set_title(f"{name}: cut vs {'own size' if col_i == 0 else 'the other side'}  (dotted fee-aware, dashed naive)",
                         fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle(f"the acceptance surface -- traders {cfg.name}, MM fee {fee}: a cut that moves with size and with the "
                 f"other side is a bid/ask with depth", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "arch_surface.png", dpi=150)
    plt.close(fig)
    return out


def main():
    arms = load_arms()
    cfg = arms[0]["cfg"]
    folder = CA.results_path(f"cotrain_multi/{CA.run_stem(cfg)}_arch_compare")
    folder.mkdir(parents=True, exist_ok=True)
    text, result = compare(arms)
    print("\n" + text)
    (folder / "arch_compare_summary.txt").write_text(text + "\n")
    xp = result["xp"]
    np.savez(folder / "arch_compare.npz",
             traders=np.array(cfg.traders), fee=CA.MM_FEE,
             labels=np.array([arm["label"] for arm in arms]),
             pair_labels=np.array(result["pairs"]),
             pair_delta_mm=np.array(result["pair_delta"]),
             pair_2se_mm=np.array(result["pair_2se"]),
             eval_mm_pl=np.stack([arm["ev"]["eval_mm_pl"] for arm in arms]),
             crossplay_mm=np.full((0, 0), np.nan) if xp is None else xp["mm"],
             crossplay_edge=np.full((0, 0, 0), np.nan) if xp is None else xp["edge"])
    figure(arms, result, folder)
    rows = surface(arms, folder)
    text += "\n(7) the acceptance surface: the fitted cut P/X by quantile bin of the offer's own size and of the other side's size\n" + "\n".join(rows) + "\n"
    (folder / "arch_compare_summary.txt").write_text(text + "\n")
    print("\n".join(rows[:4]) + ("\n  ..." if len(rows) > 4 else ""))
    print(f"\nartefacts in {folder}/  (arch_compare_summary.txt, arch_compare.png"
          f"{', arch_crossplay.png' if xp is not None else ''}, arch_surface.png, arch_compare.npz)")


if __name__ == "__main__":
    main()