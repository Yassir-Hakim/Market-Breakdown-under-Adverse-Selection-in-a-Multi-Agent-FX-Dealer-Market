import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "DP_Models"))     # v1 is imported by name there
sys.path.insert(0, str(_ROOT))                   # Mechanics/, DP_Models/, MM_Phase_1/
"""
  fee_aware_dp.py -- the exact single-trader game (four rounds, one offer per
  round) with the market maker's OWN Bureau fee g, on both sides, next to the
  networks already trained against the same fee-aware market maker. Nothing
  is trained; every number is produced by this run.

  1. CHECKS (stop on failure)
     - g = 0 reproduces the committed DP file (the one without the MM-fee
       setting) bit for bit: tables, replay, traces; small grids, both sides.
     - gate 1 (no fee: one round == v1 == the v0 closed form).
     - gate 2 at (f, g) = (2%, 1%) and (3%, 1%): the fee game is (1-g) x the
       no-fee game at f' = (f-g)/(1-g) -- one round against v1 and the
       fee-aware closed form, and the whole game on the rescaled card
       T' = T/(1-g), A' = 1-(1-A)/(1-g).
  2. THE DP   T1 (side A) and T2 (side B) at (f, g) = (2%,0), (2%,1%),
     (3%,0), (3%,1%); K = 1, shipped grids; gates 3-4 on 20,000 paths. Then
     the DP is replayed on the networks' own 200,000 evaluation paths, so DP,
     network and always-Bureau are compared on identical games. The quotable
     DP value is floor-anchored, as in rl_diagnostics: the analytic floor plus
     the paired (DP - always-Bureau) edge. The table value is printed too; it
     is inflated by the anchor-grid bias and is not quotable.
  3. THE NETWORKS  Results/mm_phase1/MM_fee_sweep_BdC_Fee_{f}/torch_{side}_
     g{g}/policy_best.pth, played through torch_train_fee.rollout on the hard
     game at the same (f, g).

  HYPOTHESES (written before running)
    H1  (2%, 1%): the DP sits on the always-Bureau floor on both sides --
        edge under 0.005% of capital, against about +0.026% at (2%, 0) on
        side A. Reason: the one-shot edge at (2%, 1%) is 0.006 bp per unit
        offered, and the buyer has no interior one-shot optimum at all.
    H2  (3%, 1%) plays like (2%, 0): its edge is nearer the (2%, 0) edge than
        the (3%, 0) edge, on both sides. Not equal: the exact twin is the
        no-fee game at f' = 2.02% on a card with T up 1% and A cut to
        1.01% (T1) / 2.02% (T2).
    H3  the networks match the DP: DP minus network within 2 s.e. (paired, same
        paths) at every point, as at (2%, 0) side A (0.0002 points).

  Time on the Mac: about 15 min (8 solves, 4 at a time). Keep it plugged in.
  Output: Results/fee_aware_dp/report.txt, report.json, edges.png

  Usage: click Run (python fee_aware_dp.py).
"""
import json
import math
import os
import subprocess
import tempfile
import time
import importlib.util
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

import DP_Models.v2_multiple_rounds as V
from Mechanics.fx_mechanics import GameParams, bdc_payoff_per_unit, trade_rate
from MM_Phase_1.fee_aware_v0 import quote_from_threshold

# ============================ SETTINGS ======================================
CARDS = {"A": dict(L=100_000.0, T=125_000.0, A=0.02, B=0.03),   # T1
         "B": dict(L=200_000.0, T=155_000.0, A=0.03, B=0.02)}   # T2
FEES = [(0.02, 0.0), (0.02, 0.01), (0.03, 0.0), (0.03, 0.01)]  # (f, g)
K = 1                   # one offer per round
GATE_PATHS = 20_000     # gates 3-4, the DP file's own replay
WORKERS = 4             # DP solves at a time
SWEEP = _ROOT / "Results" / "mm_phase1"          # the trained networks
OUT = _ROOT / "Results" / "fee_aware_dp"
BLOB = "f25c8df8078afae151c3b3d34dfad02705504395"  # the committed DP file
                                                   # without the MM-fee setting
# ============================================================================

SMALL = dict(n_offer=25, quad_per_cell=3, n_c=9, n_d=21, n_a=7, n_qfrac=8)


def spec_for(side, f, g):
    return V.TraderSpec(**CARDS[side], side=side, K=K, mm_fee=g,
                        params=GameParams(bdc_fee=f))


# ---------------------------------------------------------------------------
# 1. checks
# ---------------------------------------------------------------------------
def identity_check():
    """g = 0 against the committed file without the setting, bit for bit."""
    try:
        src = subprocess.run(["git", "-C", str(_ROOT), "cat-file", "-p", BLOB],
                             capture_output=True, check=True).stdout
    except Exception as e:                        # no git, or blob not found
        print(f"  skipped: the committed file could not be read ({e})")
        return None
    path = Path(tempfile.mkdtemp()) / "dp_committed.py"
    path.write_bytes(src)
    saved = list(sys.path)
    s = importlib.util.spec_from_file_location("dp_committed", path)
    old = importlib.util.module_from_spec(s)
    s.loader.exec_module(old)
    sys.path[:] = saved
    ok = True
    for side in ("A", "B"):
        for k in (1, 2):
            so = old.TraderSpec(**CARDS[side], side=side, K=k, rounds=3)
            sn = V.TraderSpec(**CARDS[side], side=side, K=k, rounds=3)
            po = old.solve_v2(so, old.Grids(so, **SMALL))
            pn = V.solve_v2(sn, V.Grids(sn, **SMALL))
            same = po.value == pn.value and np.array_equal(po.entry1, pn.entry1)
            same &= all(np.array_equal(po.Wtables[n], pn.Wtables[n])
                        for n in po.Wtables)
            ro = old.simulate_paths(po, n_paths=2000, seed=5)
            rn = V.simulate_paths(pn, n_paths=2000, seed=5)
            same &= all(np.array_equal(ro[key], rn[key]) for key in ro)
            same &= np.array_equal(old.clairvoyant_wealth(so, ro["X"], ro["a5"]),
                                   V.clairvoyant_wealth(sn, rn["X"], rn["a5"]))
            print(f"  side {side} K={k} (3 rounds, small grid): "
                  f"{'bit-identical' if same else 'DIFFERENT'}", flush=True)
            ok &= bool(same)
    assert ok, "g = 0 does not reproduce the committed file -- stop"
    return True


# ---------------------------------------------------------------------------
# 2. the DP (runs in worker processes)
# ---------------------------------------------------------------------------
def run_dp(job):
    side, f, g, X, a5 = job["side"], job["f"], job["g"], job["X"], job["a5"]
    t0 = time.time()
    spec = spec_for(side, f, g)
    sol = V.solve_v2(spec)
    t_solve = time.time() - t0
    out = dict(side=side, f=f, g=g, solve_s=t_solve, table=sol.pl())

    # gates 3-4, the file's own replay; gate 5 on the same draws
    try:
        res = V.gate3_and_4(sol, n_paths=GATE_PATHS, seed=1,
                            print_progress=False)
        out["gates34"] = "passed"
    except AssertionError as e:
        out["gates34"] = f"FAILED: {e}"
        res = V.simulate_paths(sol, n_paths=GATE_PATHS, seed=1)
    out["gate3_gap"] = float(res["W"].mean() - sol.value) / spec.L
    try:
        _, W_cv = V.gate5_brackets(sol, res, print_progress=False)
        out["gate5"] = "passed"
    except AssertionError as e:
        out["gate5"], W_cv = f"FAILED: {e}", float("nan")
    out["clairvoyant"] = (W_cv - spec.L) / spec.L
    out["floor"] = V.bdc_baseline(spec, sol.grids)[1]

    # round-1 offer, fresh book: threshold z, slip quote, fill probability
    act, _ = V.greedy_offer_action(sol, 1, K, V.NO_FLOOR, sol.grids.n_offer,
                                   spec.L, 0.0, spec.params.a0)
    if act is None:
        out["r1"] = None
    else:
        z = float(sol.grids.z[act[0]])
        thr = V._rate_at(spec, spec.params.a0, z)
        quote = quote_from_threshold(thr, side, g)
        out["r1"] = dict(z=z, offer=float(trade_rate(quote, side)),
                         size=act[1] / spec.L, fill=float(1.0 - norm.cdf(z)))

    # the DP on the networks' paths, with the flows it makes
    n = X.shape[0]
    pl = np.empty(n)
    offers = np.zeros(spec.rounds); fills = np.zeros(spec.rounds)
    to_mm = np.zeros(spec.rounds); to_bdc = np.zeros(spec.rounds)
    left = 0.0
    for i in range(n):
        tr = V.play_game(sol, path=X[i], a5=a5[i])
        pl[i] = tr["pl"]
        for r in tr["rounds"]:
            j = r["n"] - 1
            for o in r["offers"]:
                offers[j] += 1
                if o["accepted"]:
                    fills[j] += 1
                    to_mm[j] += o["q"]
            to_bdc[j] += r["bdc_dump"]
        left += tr["c"]
    out["pl"] = pl
    out["flows"] = dict(offer_rate=(offers / n).tolist(),
                        fill_given_offer=np.where(offers > 0, fills / np.maximum(offers, 1),
                                                  np.nan).tolist(),
                        to_mm=(to_mm / (n * spec.L)).tolist(),
                        to_bureau=(to_bdc / (n * spec.L)).tolist(),
                        left_at_end=left / (n * spec.L))
    out["total_s"] = time.time() - t0
    return out


def floor_pl(spec, X1, a5):
    """Always-Bureau on given paths, through the shared rules: dump the whole
    book in round 1, hold the target currency to settlement."""
    pay = bdc_payoff_per_unit(X1, spec.params.bdc_fee, spec.side) * spec.L
    return np.array([(V.terminal_wealth(spec, 0.0, d, s) - spec.L) / spec.L
                     for d, s in zip(pay, a5)])


# ---------------------------------------------------------------------------
# 3. the networks (main process only: torch is imported here and nowhere else)
# ---------------------------------------------------------------------------
def network_paths():
    """The networks' evaluation paths, exactly as torch_train_fee draws them.
    Importing torch_train_fee first matters: it sets float64 as torch's
    default, and the draws depend on it."""
    import torch
    import MM_Phase_1.torch_train_fee as T
    spec = T.build_spec("A")
    gen = torch.Generator().manual_seed(T.SEED_PATH + 2)
    moves = torch.randn(T.EVAL_PATHS, spec.rounds + 1, generator=gen)
    m = spec.params
    X = m.a0 + torch.cumsum(m.sd * moves[:, :spec.rounds], dim=1)
    a5 = X[:, -1] + m.sd * moves[:, spec.rounds]
    return moves, X.numpy().copy(), a5.numpy().copy()


def run_network(side, f, g, moves):
    import torch
    import MM_Phase_1.torch_train_fee as T
    path = (SWEEP / f"MM_fee_sweep_BdC_Fee_{f}" / f"torch_{side}_g{g:g}"
            / "policy_best.pth")
    if not path.exists():
        return dict(missing=str(path))
    T.BDC_FEE, T.MM_FEE = float(f), float(g)     # rollout and build_spec read these
    spec = T.build_spec(side)
    net = T.build_net(T.SEED_INIT)
    net.load_state_dict(torch.load(path))
    with torch.no_grad():
        pl = T.rollout(net, spec, moves, tau=None).numpy().copy()
        fl = T.rollout(None, spec, moves, tau=None,
                       forced=[(spec.params.a0, 0.0, 1.0)] * spec.rounds
                       ).numpy().copy()
        feats = T.features(1.0, 0.0, torch.full((1,), float(spec.L)),
                           torch.zeros(1), torch.full((1,), float(spec.params.a0)),
                           spec, torch.zeros(1, T.LAGS) if T.LAGS else None)
        quote, frac = T.squash(net(feats), torch.full((1,), float(spec.params.a0)))
    quote, frac = float(quote[0]), float(frac[0])
    thr = quote / (1.0 - g) if side == "A" else quote * (1.0 - g)
    z = (thr - spec.params.a0) / spec.params.sd * (1 if side == "A" else -1)
    return dict(pl=pl, floor=fl, path=str(path.relative_to(_ROOT)),
                r1=dict(z=z, offer=trade_rate(quote, side), size=frac,
                        fill=float(1.0 - norm.cdf(z))))


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------
def mean_se(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def pct(x, nd=4):
    return f"{x * 100:+.{nd}f}%"


def report(dp, nets, have_paths):
    lines = ["Exact DP with the market maker's own Bureau fee g, one offer per "
             "round, four rounds",
             "(% of starting capital; +/- is 2 s.e. of the paired difference; "
             "floor = always-Bureau, analytic)", ""]
    summary = {}
    for side in ("A", "B"):
        c = CARDS[side]
        cur = ("GBP", "$") if side == "A" else ("$", "GBP")
        lines.append(f"side {side}  card {cur[0]} {c['L']:,.0f} -> {cur[1]} "
                     f"{c['T']:,.0f} target, residue A {c['A']:.0%}, deficit B "
                     f"{c['B']:.0%}")
        fprime = "f'"
        lines.append(f"  {'(f, g)':<11} {fprime:>6} {'floor':>9} {'DP value':>10} "
                     f"{'DP edge':>18} {'network edge':>18} {'DP - network':>18} "
                     f"{'DP table':>9}  {'r1 DP: z, fill':>15}  {'r1 net: z, fill':>16}")
        for f, g in FEES:
            d = dp[(side, f, g)]
            key = f"{side} f={f:g} g={g:g}"
            row = dict(floor=d["floor"], table=d["table"], gates34=d["gates34"],
                       gate5=d["gate5"], gate3_gap=d["gate3_gap"],
                       clairvoyant=d["clairvoyant"], flows=d["flows"],
                       r1_dp=d["r1"], solve_s=d["solve_s"])
            fl_paths = d["floor_paths"]
            e, se = mean_se(d["pl"] - fl_paths)
            row.update(dp_edge=e, dp_edge_se=se, dp_value=d["floor"] + e)
            n = nets.get((side, f, g))
            net_txt, reg_txt, r1n = "(no network)", "", ""
            if n is not None and "pl" in n:
                ne, nse = mean_se(n["pl"] - n["floor"])
                r, rse = mean_se(d["pl"] - n["pl"]) if have_paths else (np.nan, np.nan)
                row.update(net_edge=ne, net_edge_se=nse, dp_minus_net=r,
                           dp_minus_net_se=rse, net_path=n["path"], r1_net=n["r1"],
                           engines_floor_gap=float(np.max(np.abs(n["floor"] - fl_paths)))
                           if have_paths else None)
                net_txt = f"{ne * 100:+.4f} +/- {2 * nse * 100:.4f}"
                reg_txt = f"{r * 100:+.4f} +/- {2 * rse * 100:.4f}"
                r1n = f"{n['r1']['z']:+.2f}sd {n['r1']['fill']:6.2%}"
            elif n is not None:
                net_txt = "(not on disk)"
            r1 = d["r1"]
            r1d = "no offer" if r1 is None else f"{r1['z']:+.2f}sd {r1['fill']:6.2%}"
            fe = (f - g) / (1 - g)
            lines.append(f"  ({f:.0%}, {g:.0%}){'':<3} {fe:6.2%} {pct(d['floor']):>9} "
                         f"{pct(d['floor'] + e):>10} "
                         f"{f'{e * 100:+.4f} +/- {2 * se * 100:.4f}':>18} "
                         f"{net_txt:>18} {reg_txt:>18} {pct(d['table'], 3):>9}  "
                         f"{r1d:>15}  {r1n:>16}")
            summary[key] = row
        lines.append("")
    lines.append("flows under the DP (share of starting capital, rounds 1-4; "
                 "offer rate = offers per game)")
    for side in ("A", "B"):
        for f, g in FEES:
            fl = dp[(side, f, g)]["flows"]
            lines.append(f"  side {side} ({f:.0%}, {g:.0%}): to MM "
                         + " ".join(f"{x:6.2%}" for x in fl["to_mm"])
                         + "   to Bureau " + " ".join(f"{x:6.2%}" for x in fl["to_bureau"])
                         + f"   left at end {fl['left_at_end']:.2%}   P(fill | offer) "
                         + " ".join("  --  " if not np.isfinite(x) else f"{x:6.2%}"
                                    for x in fl["fill_given_offer"]))
    lines.append("")
    lines.append("gates 3-4 (replay vs table, money conservation) and gate 5 "
                 "(floor < DP table < clairvoyant):")
    for side in ("A", "B"):
        for f, g in FEES:
            d = dp[(side, f, g)]
            lines.append(f"  side {side} ({f:.0%}, {g:.0%}): gates 3-4 {d['gates34']}"
                         f" (table - replay {-d['gate3_gap'] * 100:+.3f} points),"
                         f" gate 5 {d['gate5']}, clairvoyant {pct(d['clairvoyant'], 3)}")
    lines.append("")
    lines += verdicts(summary)
    return "\n".join(lines), summary


def verdicts(s):
    out = ["HYPOTHESES (stated before the run)"]
    def e(side, f, g):
        r = s[f"{side} f={f:g} g={g:g}"]
        return r["dp_edge"], r["dp_edge_se"]
    h1 = []
    for side in ("A", "B"):
        x, se = e(side, 0.02, 0.01)
        h1.append(f"side {side} {x * 100:+.4f} +/- {2 * se * 100:.4f}")
    ok1 = all(abs(e(sd, 0.02, 0.01)[0]) < 5e-5 for sd in ("A", "B"))
    out.append(f"  H1 (2%,1%) on the floor, |edge| < 0.005%: "
               f"{'SUPPORTED' if ok1 else 'NOT SUPPORTED'} ({'; '.join(h1)})")
    h2, ok2 = [], True
    for side in ("A", "B"):
        x, a, b = e(side, 0.03, 0.01)[0], e(side, 0.02, 0.0)[0], e(side, 0.03, 0.0)[0]
        ok2 &= abs(x - a) < abs(x - b)
        h2.append(f"side {side} (3%,1%) {x * 100:+.4f} vs (2%,0) {a * 100:+.4f}, "
                  f"(3%,0) {b * 100:+.4f}")
    out.append(f"  H2 (3%,1%) nearer (2%,0) than (3%,0): "
               f"{'SUPPORTED' if ok2 else 'NOT SUPPORTED'} ({'; '.join(h2)})")
    h3, ok3, have = [], True, True
    for key, r in s.items():
        if "dp_minus_net" not in r or not np.isfinite(r["dp_minus_net"]):
            have = False
            continue
        within = abs(r["dp_minus_net"]) <= 2 * r["dp_minus_net_se"]
        ok3 &= within
        h3.append(f"{key}: {r['dp_minus_net'] * 100:+.4f} +/- "
                  f"{2 * r['dp_minus_net_se'] * 100:.4f}{'' if within else ' (outside)'}")
    out.append(f"  H3 networks match the DP within 2 s.e.: "
               f"{'SUPPORTED' if ok3 and have else ('NOT SUPPORTED' if have else 'incomplete')}")
    out += [f"     {x}" for x in h3]
    return out


def figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    xs = np.arange(len(FEES))
    for ax, side in zip(axes, ("A", "B")):
        rows = [summary[f"{side} f={f:g} g={g:g}"] for f, g in FEES]
        dp = [r["dp_edge"] * 100 for r in rows]
        dpe = [2 * r["dp_edge_se"] * 100 for r in rows]
        ax.bar(xs - 0.18, dp, 0.36, yerr=dpe, capsize=3, color="tab:blue",
               label="exact DP")
        if all("net_edge" in r for r in rows):
            ne = [r["net_edge"] * 100 for r in rows]
            nee = [2 * r["net_edge_se"] * 100 for r in rows]
            ax.bar(xs + 0.18, ne, 0.36, yerr=nee, capsize=3, color="tab:orange",
                   label="network")
        ax.axhline(0.0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"f={f:.0%}\ng={g:.0%}" for f, g in FEES])
        ax.set_title(f"side {side} ({'T1' if side == 'A' else 'T2'})")
        ax.set_ylabel("edge over always-Bureau (% of capital)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("One offer per round: what the offer channel is worth, "
                 "with and without the market maker's fee")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def main():
    if sys.platform == "darwin":        # keep the Mac awake while this runs
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("1. CHECKS")
    print("=" * 74)
    print("g = 0 against the committed file without the MM-fee setting")
    identity = identity_check()
    print("\ngate 1: no fee, one round == v1 == the closed form")
    assert V.gate1_v1_anchor(K=3) < 2e-5, "gate 1 FAILED -- stop"
    print("  gate 1 PASSED")
    for f, g in ((0.02, 0.01), (0.03, 0.01)):
        print(f"\ngate 2 at f = {f:.0%}, g = {g:.0%}  (f' = {(f - g) / (1 - g):.4%})")
        V.gate2_mm_fee(g=g, f=f)
        print("  gate 2 PASSED")

    print("\n" + "=" * 74)
    print("2. THE DP, 3. THE NETWORKS")
    print("=" * 74)
    try:
        moves, X, a5 = network_paths()
        have_paths = True
        print(f"  {X.shape[0]:,} evaluation paths from torch_train_fee's own seed")
    except Exception as e:                 # no torch: the DP still runs
        print(f"  torch not importable ({e}); networks skipped, DP replayed "
              f"on 200,000 numpy paths")
        rng = np.random.default_rng(2)
        z = rng.standard_normal((200_000, 5))
        X = 1.25 + np.cumsum(0.05 * z[:, :4], axis=1)
        a5 = X[:, -1] + 0.05 * z[:, 4]
        moves, have_paths = None, False

    jobs = [dict(side=s, f=f, g=g, X=X, a5=a5) for f, g in FEES for s in ("A", "B")]
    dp = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for out in pool.map(run_dp, jobs):
            key = (out["side"], out["f"], out["g"])
            out["floor_paths"] = floor_pl(spec_for(*key), X[:, 0], a5)
            dp[key] = out
            print(f"  DP side {key[0]} f={key[1]:.0%} g={key[2]:.0%}: solved in "
                  f"{out['solve_s']:.0f}s, done in {out['total_s']:.0f}s; gates "
                  f"3-4 {out['gates34']}", flush=True)

    nets = {}
    if have_paths:
        for f, g in FEES:
            for s in ("A", "B"):
                try:
                    nets[(s, f, g)] = run_network(s, f, g, moves)
                except Exception as e:
                    nets[(s, f, g)] = dict(error=str(e))
                    print(f"  network side {s} f={f:.0%} g={g:.0%}: {e}")

    text, summary = report(dp, nets, have_paths)
    head = (f"fee_aware_dp.py, {time.strftime('%Y-%m-%d %H:%M')}, "
            f"{time.time() - t_start:.0f}s; identity check "
            f"{'passed' if identity else 'skipped'}; gates 1-2 passed\n\n")
    print("\n" + head + text)
    (OUT / "report.txt").write_text(head + text + "\n")
    (OUT / "report.json").write_text(json.dumps(summary, indent=1, default=float))
    figure(summary, OUT / "edges.png")
    print(f"\nwrote {OUT / 'report.txt'}, report.json, edges.png")


if __name__ == "__main__":
    main()
