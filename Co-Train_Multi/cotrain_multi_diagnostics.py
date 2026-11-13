import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))           # Mechanics/
sys.path.insert(0, str(Path(__file__).resolve().parent))
"""
  cotrain_multi_diagnostics.py -- the figures and the summary for ONE
  arm of the co-trained game, drawn ONLY from cotrain_multi_curve.npz and
  cotrain_multi_eval.npz (written by cotrain_multi.py). Nothing is
  recomputed from the game and nothing is plotted anywhere else. Which run
  to read FOLLOWS cotrain_multi.py's TRADERS, MM_FEE, MM_ARCH and
  SEE_OFFER_SIZE, so the two cannot drift apart; set RUN_DIR to re-plot
  another run on purpose. The arms side by side: cotrain_multi_compare.py.

    cotrain_multi_curve.png        learning curves: the MM's hard-game P/L
                                   against the block index with the 
                                   hardcoded rules as MOVING lines, then one
                                   panel per trader with its P/L, the 2 SE of
                                   its PAIRED edge over always-BdC, and that
                                   floor on the same paths. Markers say who
                                   had just moved. Do they settle, or cycle?
    cotrain_multi_policy.png       what was learned, block by block: the
                                   MM's fitted round-1 bid and ask against
                                   the naive and fee-aware cuts and their
                                   spread; each trader's round-1 quote and
                                   the share of its book it puts up; the
                                   filled shares (the deadlock detector); the
                                   Bureau share (the crossing detector); the
                                   traders' learning rates and the ramp.
    cotrain_multi_offers.png       the traders' offers round by round on the
                                   record paths: quote over the true rate
                                   (median and 10-90% band) against the MM's
                                   level for that direction, the trader's
                                   own Bureau line and the naive cut; the
                                   share of the book put up; the volume
                                   filled.
    cotrain_multi_mm_book.png      the MM's capacity through the game: both
                                   legs of its book after the fills and
                                   after the Bureau phase, round by round,
                                   as % of the game's capital (median and
                                   10-90% band); the net position; the
                                   pound volume filled and the pound volume
                                   moved at the Bureau by round, and the
                                   crossed share; the repay fraction on
                                   short books.
    cotrain_multi_final_books.png  where everyone ends: each trader's
                                   leftover start currency (% of L) and
                                   banked target currency (% of T) and its
                                   P/L against always-BdC on the same path;
                                   the MM's end book and P/L.
    cotrain_multi_set_path.png     the demo game (SET_PATH) round by round:
                                   the rates, every trader's quote and
                                   verdict against the MM's level for its
                                   direction, and every book through the
                                   game.
    cotrain_multi_median_path.png  the same three panels for the MEDIAN game:
                                   the record path whose MM P/L is the
                                   median, so the picture is typical rather
                                   than a flattering or an alarming draw.
    cotrain_multi_set_traders.png  the demo game trader by trader, in the
    cotrain_multi_median_traders.png  earlier files' format: each trader's
                                   offer ladder in its own units against the
                                   hidden rate and the Bureau's after-fee
                                   rate, and its inventory (start currency
                                   over L, banked over T) through the game.
    cotrain_multi_edges.png        the final paired comparisons with 2 SE:
                                   each trader over always-BdC and over
                                   the single-trader run on the same
                                   paths; the MM against each rule.
    cotrain_multi_netting.png      netting, the limit-order-book view: three
                                   single rounds drawn as order books in
                                   $/GBP (pound sellers' pounds on offer at
                                   or below each price against dollar
                                   sellers' pounds wanted at or above it;
                                   where the two overlap the offers cross
                                   and can be netted), and by round the
                                   pounds that could be netted, the pounds
                                   the learned MM netted, and the pounds it
                                   moved at the Bureau.
    cotrain_multi_summary.txt      every headline number in one file.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import cotrain_multi as CM
from Mechanics.fx_mechanics import (trade_rate, bdc_payoff_per_unit,   # the earlier
                                    rate_units)                        # files' units

# ============================ SETTINGS ======================================
RUN_DIR = None          # None: the run cotrain_multi.py's TRADERS, MM_FEE,
                        # MM_ARCH and SEE_OFFER_SIZE point at. A path
                        # re-plots that run instead (an older folder too)
# ============================================================================

BP = 1e4
PCT = 100.0
COLOUR = {"mm": "tab:purple", "trader": "tab:green", "init": "0.5"}
TRADER_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown",
                  "tab:pink", "tab:olive", "tab:cyan"]


def load():
    cfg = CM.make_config(CM.TRADERS)
    if RUN_DIR is None:
        run = CM.run_dir(cfg)
    else:
        run = Path(RUN_DIR)
    if RUN_DIR is None and not (run / "cotrain_multi_eval.npz").exists():
        old = CM.run_dir_v4(cfg)
        if (old / "cotrain_multi_eval.npz").exists():
            print(f"  reading the run in its 12 Sep folder name: {old}", flush=True)
            run = old
    for name in ("cotrain_multi_curve.npz", "cotrain_multi_eval.npz"):
        if not (run / name).exists():
            have = sorted(p.name for p in run.parent.glob(f"{cfg.name}_*") if (p / name).exists())
            raise AssertionError(
                f"no run to plot in {run} ({name} missing): cotrain_multi.py's SETTINGS point at an arm that "
                f"has not been run (MM_ARCH {CM.MM_ARCH}, size {CM.SEE_OFFER_SIZE}, two-way "
                f"{CM.TWO_WAY}, MM fee {CM.MM_FEE}, trader fee {cfg.params.bdc_fee:g}). Runs on disk for {cfg.name}: "
                f"{have or 'none'}. Set the SETTINGS back to the run you want, or RUN_DIR to its folder.")
    curve = np.load(run / "cotrain_multi_curve.npz", allow_pickle=False)
    ev = np.load(run / "cotrain_multi_eval.npz", allow_pickle=False)
    curve = {key: curve[key] for key in curve.files}
    ev = {key: ev[key] for key in ev.files}
    # an older eval npz named the trader's Bureau fee BDC_FEE_DEFAULT and
    # carried no architecture; this file reads trader_bdc_fee, so fall back
    ev.setdefault("trader_bdc_fee", ev.get("BDC_FEE_DEFAULT"))
    ev.setdefault("mm_arch", np.array("round" if not bool(ev["live_book_within_round"]) else "bid"))
    ev.setdefault("n_flow", np.array(0))
    ev.setdefault("two_way", np.array(False))
    ev.setdefault("offer_trader", np.arange(len(ev["traders"])))
    return run, curve, ev


def names(ev):
    return [str(t) for t in ev["traders"]]


def sides(ev):
    return [str(s) for s in ev["sides"]]


def fee_aware_cut(side, fee):
    return (1.0 - fee) if side == "A" else 1.0 / (1.0 - fee)


def bureau_line(side, f):
    """The trader's own outside option as a multiple of the true rate: a
    pound seller beats its Bureau above (1-f), a dollar seller below 1/(1-f)."""
    return (1.0 - f) if side == "A" else 1.0 / (1.0 - f)


def _who_moved(curve):
    return [str(m) for m in curve["moved"]]


def _mark_movers(ax, x, y, moved):
    for agent, marker in (("mm", "s"), ("trader", "o"), ("init", "D")):
        idx = [i for i, m in enumerate(moved) if m == agent]
        if idx:
            ax.plot(x[idx], y[idx], marker, color=COLOUR[agent], ms=4, ls="none",
                    label=("as initialised" if agent == "init"
                           else f"after the {'MM' if agent == 'mm' else 'traders'} moved"))


def _title(ev):
    return (f"{'-'.join(names(ev))} ({''.join(sides(ev))}), MM fee "
            f"{float(ev['fee']):g}, {str(ev['first'])} first, MM init "
            f"{str(ev['mm_level_init'])}")


def _band(ax, x, lo, hi, colour, label=None):
    ax.fill_between(x, lo, hi, color=colour, alpha=0.15, lw=0, label=label)


# ---------------------------------------------------------------------------
def plot_curve(run, curve, ev):
    N = len(names(ev))
    x = np.arange(len(curve["mm_pl"]))
    moved = _who_moved(curve)
    fig, axes = plt.subplots(1 + N, 1, figsize=(8, 3.0 * (1 + N)), sharex=True)
    top = axes[0]
    top.plot(x, curve["mm_pl"] * BP, color="k", lw=1.4,
             label="co-trained MM, hard game (validation paths)")
    _band(top, x, (curve["mm_pl"] - curve["mm_2se"]) * BP,
          (curve["mm_pl"] + curve["mm_2se"]) * BP, "k")
    rule_names = [str(n) for n in curve["rule_names"][0]]
    for j, name in enumerate(rule_names):
        book = "book" in name
        colour = "tab:olive" if book else "tab:red" if "fee" in name else "tab:blue"
        repay = float(ev["rule_repay"][j])
        # the book rule gets its own dash patterns (dash-dot flatten, dotted
        # carry): when nothing crosses it scores exactly the fee-aware rule,
        # and the gaps let that line show through underneath
        style = ({1.0: "-.", 0.0: ":"} if book else {1.0: "-", 0.0: "--"}).get(repay, ":")
        top.plot(x, curve["rule_pl"][:, j] * BP, ls=style, lw=1.0,
                 color=colour, label=f"hardcoded {name} against the traders at that block")
    _mark_movers(top, x, curve["mm_pl"] * BP, moved)
    top.set_ylabel("MM P/L, bp of the game's capital")
    top.set_title(f"{_title(ev)}: learning curves, {int(ev['n_inner'])} iterations per block")
    top.grid(alpha=0.3)
    top.legend(fontsize=6.5, loc="best")
    for i, (ax, name, side) in enumerate(zip(axes[1:], names(ev), sides(ev))):
        pl, band = curve["trader_pl"][:, i], curve["edge_2se"][:, i]
        ax.plot(x, pl * PCT, color="k", lw=1.4,
                label=f"{name} (side {side}), hard game (validation paths)")
        _band(ax, x, (pl - band) * PCT, (pl + band) * PCT, "k",
              "2 SE of the edge over always-BdC, paired")
        floor = float(curve["bdc_pl"][-1, i])
        ax.axhline(floor * PCT, color="tab:orange", ls="--", lw=1.4,
                   label=f"always-BdC on the same paths ({floor * PCT:+.3f}%)")
        _mark_movers(ax, x, pl * PCT, moved)
        ax.set_ylabel(f"{name} P/L, % of L")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best")
    axes[-1].set_xlabel("block (0 = as initialised; then alternating MM / traders)")
    stopped = int(curve["stopped_at"])
    if stopped >= 0:
        for ax in axes:
            ax.axvline(x[-1], color="0.6", lw=0.8, ls=":")
        axes[-1].text(x[-1], axes[-1].get_ylim()[0], f" flat: stopped at outer step {stopped}",
                      fontsize=7, color="0.4", va="bottom")
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_policy(run, curve, ev):
    fee = float(ev["fee"])
    x = np.arange(len(curve["mm_pl"]))
    moved = _who_moved(curve)
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    cuts, quotes, filled, bureau, lrs = axes

    # ---- the MM's cuts -------------------------------------------------------
    bid, ask = curve["bid"][:, 0], curve["ask"][:, 0]
    if np.isfinite(bid).any():
        cuts.plot(x, bid, color="tab:blue", lw=1.4,
                  label="fitted round-1 bid (pound sellers filled below)")
        cuts.axhline(fee_aware_cut("A", fee), color="tab:blue", lw=0.9, ls="--",
                     label=f"fee-aware bid {fee_aware_cut('A', fee):.4f}")
    if np.isfinite(ask).any():
        cuts.plot(x, ask, color="tab:red", lw=1.4,
                  label="fitted round-1 ask (dollar sellers filled above)")
        cuts.axhline(fee_aware_cut("B", fee), color="tab:red", lw=0.9, ls="--",
                     label=f"fee-aware ask {fee_aware_cut('B', fee):.4f}")
    cuts.axhline(1.0, color="k", lw=0.9, label="naive: the true rate")
    if np.isfinite(bid).any() and np.isfinite(ask).any():
        twin = cuts.twinx()
        twin.plot(x, (ask - bid) * PCT, color="0.5", lw=1.0, ls=":",
                  label="spread ask - bid, % of X (right)")
        twin.axhline(2 * fee / (1 - fee) * PCT, color="0.5", lw=0.7, ls="--")
        twin.set_ylabel("spread, % of X", color="0.4", fontsize=8)
        twin.tick_params(axis="y", labelsize=7, colors="0.4")
        lines = [l for l in cuts.get_lines() + twin.get_lines() if not l.get_label().startswith("_")]
        cuts.legend(lines, [l.get_label() for l in lines], fontsize=6.5, loc="best")
    else:
        cuts.legend(fontsize=6.5, loc="best")
    cuts.set_ylabel("MM cut / true rate, round 1\n(nan where nothing accepted or rejected)")
    cuts.set_title(f"{_title(ev)}: what was learned, block by block")
    cuts.grid(alpha=0.3)

    # ---- the traders' opening quotes -----------------------------------------
    twin = quotes.twinx()
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        quotes.plot(x, curve["offer_rate"][:, i], color=colour, lw=1.4,
                    label=f"{name} ({side}) round-1 quote, $/GBP (median)")
        twin.plot(x, curve["fraction"][:, i] * PCT, color=colour, lw=0.9, ls=":",
                  label=f"{name} share of book put up (right)")
    quotes.axhline(float(ev["a0"]), color="k", lw=0.8, ls="--", label="a0")
    twin.set_ylabel("put up, % of book", color="0.4", fontsize=8)
    twin.set_ylim(0, 102)
    twin.tick_params(axis="y", labelsize=7, colors="0.4")
    quotes.set_ylabel("round-1 quote, $/GBP")
    quotes.grid(alpha=0.3)
    lines = [l for l in quotes.get_lines() + twin.get_lines() if not l.get_label().startswith("_")]
    quotes.legend(lines, [l.get_label() for l in lines], fontsize=6.5, loc="best")

    # ---- the flow: the deadlock detector -------------------------------------
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        filled.plot(x, curve["filled_i"][:, i] * PCT, color=colour, lw=1.2,
                    label=f"{name} ({side}) volume filled")
    filled.plot(x, curve["filled"] * PCT, color="k", lw=1.4,
                label="all traders, pound-equivalent volume filled")
    filled.plot(x, curve["filled_benchmark"] * PCT, color="0.4", lw=1.0, ls="--",
                label="filled by the benchmark rule (best rule at that block)")
    filled.plot(x, curve["agree"] * PCT, color="0.7", lw=0.9,
                label="verdicts agreeing with the benchmark rule")
    _mark_movers(filled, x, curve["filled"] * PCT, moved)
    filled.set_ylabel("% of offered volume")
    filled.set_ylim(-2, 102)
    filled.grid(alpha=0.3)
    filled.legend(fontsize=6.5, loc="best")

    # ---- the crossing detector -----------------------------------------------
    share = curve["bureau_share"]
    bureau.plot(x, share * PCT, color="k", lw=1.4,
                label="Bureau share of filled pound volume (nan: nothing filled)")
    bureau.plot(x, (1 - share) * PCT, color="tab:green", lw=1.2,
                label="crossed against opposite flow, or carried out solvent")
    bureau.axhline(100, color="0.6", lw=0.8, ls="--")
    bureau.set_ylabel("% of the MM's filled volume")
    bureau.set_ylim(-2, 102)
    bureau.grid(alpha=0.3)
    bureau.legend(fontsize=6.5, loc="best")

    # ---- the schedule ----------------------------------------------------------
    for i, name in enumerate(names(ev)):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        lrs.plot(x, curve["lr_trader"][:, i], color=colour, lw=1.2,
                 label=f"{name} learning rate (plateau decay)")
    lrs.set_yscale("log")
    lrs.set_ylabel("trader learning rate")
    twin = lrs.twinx()
    tau = curve["tau"].astype(float)
    twin.plot(x, tau, color="0.5", lw=1.0, ls=":", label="ramp width tau at the block's last step (right)")
    twin.set_yscale("log")
    twin.set_ylabel("tau", color="0.4", fontsize=8)
    twin.tick_params(axis="y", labelsize=7, colors="0.4")
    lrs.set_xlabel("block (0 = as initialised; then alternating MM / traders)")
    lrs.grid(alpha=0.3)
    lines = [l for l in lrs.get_lines() + twin.get_lines() if not l.get_label().startswith("_")]
    lrs.legend(lines, [l.get_label() for l in lines], fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_policy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _held_start(ev, i, L):
    """Each trader's book at the start of every round: L, then the book
    after the previous round's Bureau phase."""
    after = ev["record_trader_held_after_bureau"][:, i, :]
    return np.concatenate([np.full((after.shape[0], 1), L), after[:, :-1]], axis=1)


def plot_offers(run, ev):
    N, rounds, fee, f = len(names(ev)), int(ev["rounds"]), float(ev["fee"]), float(ev["trader_bdc_fee"])
    X = ev["record_hidden_rate"]                              # (P, rounds)
    fig, axes = plt.subplots(3, N, figsize=(4.6 * N, 10.5), squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1, 1]})
    r = np.arange(1, rounds + 1)
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        L = float(ev["cards"][i, 0])
        u = ev["record_quoted_price"][:, i, :]        # the offer itself, $/GBP
        level = ev["record_level"][:, i, :]           # the MM's level it faced, $/GBP
        # the reference prices are fixed multiples of the true rate, so in $/GBP
        # they move with the rate: drawn at the round's median rate
        Xm = np.median(X, axis=0)
        offered = ev["record_offered_amount"][:, i, :]
        fill = ev["record_fill_weight"][:, i, :]
        share = offered / np.maximum(_held_start(ev, i, L), 1e-9)

        q = axes[0, i]
        q.plot(r, np.median(u, axis=0), color=colour, lw=1.6, label="the offer, median")
        _band(q, r, np.percentile(u, 10, axis=0), np.percentile(u, 90, axis=0), colour,
              "10-90% of games")
        q.plot(r, np.median(level, axis=0), color="k", lw=1.2, ls="-",
               label="the MM's level for this direction, median")
        q.plot(r, Xm, color="0.4", lw=0.8, ls=":", label="the true rate, median")
        q.plot(r, fee_aware_cut(side, fee) * Xm, color="tab:red", lw=0.8, ls="--",
               label="fee-aware cut")
        q.plot(r, bureau_line(side, f) * Xm, color="tab:orange", lw=0.8, ls="--",
               label=f"{name}'s own Bureau")
        q.set_title(f"{name} (side {side}): {'sells pounds, filled below the level' if side == 'A' else 'sells dollars, filled above the level'}",
                    fontsize=9)
        q.set_ylabel("$/GBP")
        q.grid(alpha=0.3)
        q.legend(fontsize=6, loc="best", framealpha=0.55)

        p = axes[1, i]
        p.plot(r, np.median(share, axis=0) * PCT, color=colour, lw=1.6, label="share of the book put up, median")
        _band(p, r, np.percentile(share, 10, axis=0) * PCT, np.percentile(share, 90, axis=0) * PCT,
              colour, "10-90% of games")
        p.set_ylabel("% of book held at the round's start")
        p.set_ylim(-2, 102)
        p.grid(alpha=0.3)
        p.legend(fontsize=6, loc="best")

        v = axes[2, i]
        vol_filled = (offered * fill).sum(axis=0) / np.maximum(offered.sum(axis=0), 1e-300)
        games_filled = (fill > 0.5).mean(axis=0)
        v.bar(r - 0.2, vol_filled * PCT, width=0.4, color=colour, label="offered volume filled")
        v.bar(r + 0.2, games_filled * PCT, width=0.4, color="0.6", label="games with a fill")
        v.set_ylabel("%")
        v.set_ylim(0, max(5, float(max(vol_filled.max(), games_filled.max()) * PCT * 1.15)))
        v.set_xlabel("round")
        v.set_xticks(r)
        v.grid(alpha=0.3, axis="y")
        v.legend(fontsize=6, loc="best")
    fig.suptitle(f"{_title(ev)}: the traders' offers round by round ({X.shape[0]:,} record paths)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_offers.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _bureau_volume_by_round(ev):
    d_pounds = ev["record_mm_pounds_after_bureau"] - ev["record_mm_pounds_after_fill"]
    final = ev["record_mm_pounds_end"] - ev["record_mm_pounds_after_bureau"][:, -1]
    return np.abs(d_pounds), np.abs(final)


def _filled_pound_volume_by_round(ev):
    filled = ev["record_fill_weight"] * ev["record_offered_amount"]
    pounds = np.where(ev["record_direction"] > 0, filled, filled / ev["record_quoted_price"])
    return pounds.sum(axis=1)                                   # (P, rounds)


def plot_mm_book(run, ev):
    rounds, cap = int(ev["rounds"]), float(ev["capital_pounds"])
    X = ev["record_hidden_rate"]
    r = np.arange(1, rounds + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (legs, net), (volume, repay) = axes

    # ---- both legs, after the fills and after the Bureau --------------------
    for leg, colour in (("pounds", "tab:blue"), ("dollars", "tab:red")):
        for phase, ls, label in (("after_fill", "-", "after the round's fills"),
                                 ("after_bureau", "--", "after the Bureau phase")):
            book = ev[f"record_mm_{leg}_{phase}"]
            if leg == "dollars":
                book = book / X                                 # pounds at the round's rate
            book = book / cap * PCT
            legs.plot(r, np.median(book, axis=0), color=colour, lw=1.5, ls=ls,
                      label=f"{leg} {label}, median")
            if phase == "after_fill":
                _band(legs, r, np.percentile(book, 10, axis=0), np.percentile(book, 90, axis=0),
                      colour, f"{leg}: 10-90% of games")
    legs.axhline(0, color="k", lw=0.8)
    legs.set_ylabel("% of the game's capital (pounds; dollars at the round's rate)")
    legs.set_title("the MM's book: negative = short", fontsize=9)
    legs.set_xticks(r)
    legs.grid(alpha=0.3)
    legs.legend(fontsize=6, loc="best")

    # ---- the net position --------------------------------------------------------
    net_fill = (ev["record_mm_pounds_after_fill"] + ev["record_mm_dollars_after_fill"] / X) / cap * PCT
    net_bureau = (ev["record_mm_pounds_after_bureau"] + ev["record_mm_dollars_after_bureau"] / X) / cap * PCT
    gross_fill = (np.abs(ev["record_mm_pounds_after_fill"]) + np.abs(ev["record_mm_dollars_after_fill"]) / X) / cap * PCT
    net.plot(r, np.median(net_fill, axis=0), color="k", lw=1.5, label="net value after the fills, median")
    _band(net, r, np.percentile(net_fill, 10, axis=0), np.percentile(net_fill, 90, axis=0), "k",
          "10-90% of games")
    net.plot(r, np.median(net_bureau, axis=0), color="k", lw=1.2, ls="--",
             label="net value after the Bureau phase, median")
    net.plot(r, np.median(gross_fill, axis=0), color="tab:purple", lw=1.2, ls=":",
             label="gross exposure |pounds| + |dollars| after the fills, median")
    net.axhline(0, color="0.5", lw=0.8)
    net.set_ylabel("% of the game's capital")
    net.set_title("net value and gross exposure through the game", fontsize=9)
    net.set_xticks(r)
    net.grid(alpha=0.3)
    net.legend(fontsize=6, loc="best")

    # ---- volumes: filled, at the Bureau, crossed -----------------------------------
    filled_r = _filled_pound_volume_by_round(ev)
    bureau_r, final = _bureau_volume_by_round(ev)
    filled_mean = filled_r.mean(axis=0) / cap * PCT
    bureau_mean = bureau_r.mean(axis=0) / cap * PCT
    volume.bar(r - 0.2, filled_mean, width=0.4, color="tab:green", label="pound volume filled, mean per game")
    volume.bar(r + 0.2, bureau_mean, width=0.4, color="tab:red", label="pound volume moved at the Bureau, mean per game")
    volume.bar([rounds + 1], [final.mean() / cap * PCT], width=0.4, color="tab:red", hatch="//",
               label="round-5 forced clear")
    total_filled, total_bureau = filled_r.sum(), bureau_r.sum() + final.sum()
    offered = np.where(ev["record_direction"] > 0, ev["record_offered_amount"],
                       ev["record_offered_amount"] / ev["record_quoted_price"]).sum()
    if offered > 0 and total_filled / offered >= CM.MIN_FILLED_SHARE:
        volume.set_title(f"volumes: {(1 - total_bureau / total_filled) * PCT:.1f}% of filled volume "
                         f"never went to the Bureau", fontsize=9)
    else:
        volume.set_title(f"volumes: {total_filled / max(offered, 1e-300) * PCT:.2f}% of offered volume "
                         f"filled -- too little to measure crossing", fontsize=9)
    volume.set_ylabel("% of the game's capital")
    volume.set_xticks(list(r) + [rounds + 1])
    volume.set_xticklabels([str(v) for v in r] + ["settle"])
    volume.set_xlabel("round")
    volume.grid(alpha=0.3, axis="y")
    volume.legend(fontsize=6, loc="best")

    # ---- the repay head -------------------------------------------------------------
    rep = ev["repay_by_round"]
    repay.bar(r, rep, color="tab:purple", width=0.5, label="repay fraction on SHORT books, short-weighted")
    repay.axhline(1.0, color="tab:blue", lw=0.8, ls="--", label="flatten (1)")
    repay.axhline(0.0, color="tab:red", lw=0.8, ls="--", label="carry (0)")
    repay.set_ylim(-0.05, 1.05)
    repay.set_ylabel("fraction of the short repaid")
    repay.set_title("the Bureau phase: flatten or carry", fontsize=9)
    repay.set_xticks(r)
    repay.set_xlabel("round")
    repay.grid(alpha=0.3, axis="y")
    repay.legend(fontsize=6, loc="best")
    fig.suptitle(f"{_title(ev)}: the MM's capacity through the game ({X.shape[0]:,} record paths)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_mm_book.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_final_books(run, ev):
    N, cap = len(names(ev)), float(ev["capital_pounds"])
    fig, axes = plt.subplots(N + 1, 3, figsize=(12, 3.0 * (N + 1)), squeeze=False)
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        L, T = float(ev["cards"][i, 0]), float(ev["cards"][i, 1])
        held = ev["record_trader_held_after_bureau"][:, i, -1] / L * PCT
        banked = ev["record_trader_banked_after_bureau"][:, i, -1] / T * PCT
        gain = (ev["record_trader_pl"][i] - ev["record_bdc_pl"][i]) * PCT
        a, b, c = axes[i]
        a.hist(held, bins=60, color=colour)
        a.set_title(f"{name} ({side}): leftover start currency at the end, % of L (median {np.median(held):.2f}%)",
                    fontsize=8)
        b.hist(banked, bins=60, color=colour)
        b.axvline(100, color="k", lw=0.8, ls="--")
        b.set_title(f"{name}: banked target currency, % of T (median {np.median(banked):.2f}%; 100 = target met)",
                    fontsize=8)
        c.hist(gain, bins=60, color=colour)
        c.axvline(0, color="k", lw=0.8, ls="--")
        c.set_title(f"{name}: P/L minus always-BdC on the same path, % of L (mean {gain.mean():+.4f}%)",
                    fontsize=8)
        for ax in (a, b, c):
            ax.grid(alpha=0.3)
    a, b, c = axes[N]
    a5 = ev["record_settlement_rate"]
    pounds_end = ev["record_mm_pounds_end"] / cap * PCT
    dollars_end = ev["record_mm_dollars_end"] / a5 / cap * PCT
    a.hist(pounds_end, bins=60, color="tab:blue")
    a.set_title(f"MM: pounds after the round-5 clear, % of capital (median {np.median(pounds_end):.3f}%)", fontsize=8)
    b.hist(dollars_end, bins=60, color="tab:red")
    b.set_title(f"MM: dollars after the clear, in pounds at a5, % of capital (median {np.median(dollars_end):.3f}%)", fontsize=8)
    mm_pl = ev["record_mm_pl"] * BP
    c.hist(mm_pl, bins=60, color="tab:purple")
    c.axvline(0, color="k", lw=0.8, ls="--")
    c.set_title(f"MM P/L, bp of capital (mean {mm_pl.mean():+.2f} bp)", fontsize=8)
    for ax in (a, b, c):
        ax.grid(alpha=0.3)
    fig.suptitle(f"{_title(ev)}: where everyone ends ({len(a5):,} record paths)", fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_final_books.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _game(ev, which):
    """The ONE game a path figure draws, as a dict of that game's arrays.
    which = "set": the demo path (SET_PATH), as before. which = "median":
    the recorded path whose MM P/L is the MEDIAN of the record set -- a
    typical game rather than a flattering or an alarming one, picked the way
    rl_diagnostics.plot_path picks its representative game. One game for
    every panel, so the MM's book and the traders' inventories can be read
    against each other; each trader's own percentile ON that game is
    reported next to it, so it is visible when the game was not typical for
    that trader."""
    if which == "set":
        idx, p = 0, "set_record_"
        pl_mm = float(ev["set_mm_pl"])
        pl_tr = np.asarray(ev["set_trader_pl"], float)
        label = "the demo game (SET_PATH)"
        pct = None
    else:
        pl = np.asarray(ev["record_mm_pl"], float)
        idx, p = int(np.argsort(pl)[len(pl) // 2]), "record_"
        pl_mm = float(pl[idx])
        all_tr = np.asarray(ev["record_trader_pl"], float)
        pl_tr = all_tr[:, idx]
        pct = [float((all_tr[i] < pl_tr[i]).mean()) for i in range(all_tr.shape[0])]
        label = f"the median game (median MM P/L of {len(pl):,} record paths, path #{idx})"
    X = np.asarray(ev[p + "hidden_rate"])[idx]
    a5 = float(np.asarray(ev[p + "settlement_rate"])[idx])
    g = dict(idx=idx, X=X, a5=a5, pl_mm=pl_mm, pl_tr=pl_tr, label=label,
             pct=pct, rates=np.concatenate([X, [a5]]))
    for key in ("quoted_price", "fill_weight", "level", "offered_amount", "position",
                "trader_held_after_fill", "trader_banked_after_fill",
                "trader_held_after_bureau", "trader_banked_after_bureau",
                "mm_pounds_after_fill", "mm_dollars_after_fill",
                "mm_pounds_after_bureau", "mm_dollars_after_bureau"):
        g[key] = np.asarray(ev[p + key])[idx]
    return g


def plot_game_path(run, ev, which, stem):
    """The three-panel view of ONE game: the rates with every quote and
    verdict against the MM's level, the MM's book, and the traders' books."""
    rounds, cap = int(ev["rounds"]), float(ev["capital_pounds"])
    g = _game(ev, which)
    rates, X = g["rates"], g["X"]
    r = np.arange(1, rounds + 1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    price, mm, traders = axes
    price.plot(np.arange(0, rounds + 2), np.concatenate([[float(ev["a0"])], rates]), color="k",
               lw=1.2, marker="o", ms=3, label="the rate: a0, rounds 1-4, settlement")
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        q, fill, level = g["quoted_price"][i], g["fill_weight"][i], g["level"][i]
        price.plot(r, level, color=colour, lw=1.0, ls="--",
                   label=f"MM level for {name} ({side}): filled {'below' if side == 'A' else 'above'} it")
        price.plot(r[fill > 0.5], q[fill > 0.5], marker="o", color=colour, ls="none", ms=7,
                   label=f"{name} quote, filled")
        price.plot(r[fill <= 0.5], q[fill <= 0.5], marker="x", color=colour, ls="none", ms=7,
                   label=f"{name} quote, rejected")
    price.set_ylabel("$/GBP")
    price.set_title(f"{_title(ev)}: {g['label']}\nrates "
                    + ", ".join(f"{float(v):.2f}" for v in rates), fontsize=8)
    price.grid(alpha=0.3)
    price.legend(fontsize=6, loc="best")

    for leg, colour in (("pounds", "tab:blue"), ("dollars", "tab:red")):
        for phase, ls in (("after_fill", "-"), ("after_bureau", "--")):
            book = g[f"mm_{leg}_{phase}"]
            if leg == "dollars":
                book = book / X
            mm.plot(r, book / cap * PCT, color=colour, lw=1.4, ls=ls, marker="o", ms=3,
                    label=f"MM {leg} {'after fills' if phase == 'after_fill' else 'after Bureau'}")
    mm.axhline(0, color="k", lw=0.8)
    mm.set_ylabel("MM book, % of capital\n(dollars at the round's rate)")
    mm.set_title(f"MM P/L on this game {g['pl_mm'] * BP:+.2f} bp", fontsize=9)
    mm.grid(alpha=0.3)
    mm.legend(fontsize=6, loc="best")

    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        L, T = float(ev["cards"][i, 0]), float(ev["cards"][i, 1])
        held = g["trader_held_after_bureau"][i] / L * PCT
        banked = g["trader_banked_after_bureau"][i] / T * PCT
        traders.plot(r, held, color=colour, lw=1.4, marker="o", ms=3, label=f"{name} held, % of L")
        traders.plot(r, banked, color=colour, lw=1.4, ls="--", marker="s", ms=3, label=f"{name} banked, % of T")
    traders.set_ylabel("% (after each round's Bureau phase)")
    traders.set_xlabel("round")
    traders.set_xticks(np.arange(0, rounds + 2))
    traders.set_title("trader P/L on this game: " + ", ".join(
        f"{name} {float(pl) * PCT:+.3f}%" + ("" if g["pct"] is None else f" ({g['pct'][i] * PCT:.0f}th pct)")
        for i, (name, pl) in enumerate(zip(names(ev), g["pl_tr"]))), fontsize=9)
    traders.grid(alpha=0.3)
    traders.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(run / f"{stem}.png", dpi=150)
    plt.close(fig)


def plot_set_path(run, ev):
    plot_game_path(run, ev, "set", "cotrain_multi_set_path")


def plot_median_path(run, ev):
    plot_game_path(run, ev, "median", "cotrain_multi_median_path")


def plot_trader_paths(run, ev, which, stem):
    """ONE game, trader by trader, in the earlier files' format
    (v2_multiple_rounds.plot_game): per trader an offer ladder in ITS OWN
    units -- target currency per unit of the currency it holds, so both
    sides read the same way -- with the hidden rate as a dashed band across
    each round, the anchor it quoted off, filled offers solid green, rejected
    ones open red, marker size with the amount offered, the Bureau's
    after-fee rate as an orange square on the rounds it used the Bureau, and
    the settlement rate as a star; and below it the inventory, start currency
    left over L and target currency banked over T, crossing 1.0 when the
    target is met. Offers sit at their ARRIVAL POSITION within the round."""
    rounds = int(ev["rounds"])
    n = len(names(ev))
    K = int(ev["n_offers"]) if "n_offers" in ev else n
    f = float(ev["trader_bdc_fee"]) if "trader_bdc_fee" in ev else float(ev["BDC_FEE_DEFAULT"])
    g = _game(ev, which)
    X, a5, a0 = g["X"], g["a5"], float(ev["a0"])
    fig, axes = plt.subplots(2 * n, 1, figsize=(9, 4.0 * n),
                             gridspec_kw={"height_ratios": [2, 1] * n})
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        ax_rate, ax_inv = axes[2 * i], axes[2 * i + 1]
        L, T = float(ev["cards"][i, 0]), float(ev["cards"][i, 1])
        anchor = np.concatenate([[a0], X[:-1]])
        held_f, held_b = g["trader_held_after_fill"][i], g["trader_held_after_bureau"][i]
        bank_f, bank_b = g["trader_banked_after_fill"][i], g["trader_banked_after_bureau"][i]
        first = dict(rate=True, anchor=True, acc=True, rej=True, bdc=True)
        for k in range(rounds):
            lo, hi = k + 0.60, k + 1.30
            ax_rate.hlines(trade_rate(X[k], side), lo, hi, color="tab:purple", lw=1.6,
                           ls="--", alpha=0.8, label="hidden true rate X" if first["rate"] else None)
            ax_rate.plot([lo - 0.1], [trade_rate(anchor[k], side)], marker="_", ms=14, color="grey",
                         label="anchor (last revealed)" if first["anchor"] else None)
            first["rate"] = first["anchor"] = False
            amount = float(g["offered_amount"][i, k])
            if amount > 1e-6:
                x = lo + (float(g["position"][i, k]) + 0.5) / max(K, 1) * (hi - lo)
                size = 40 + 320 * min(amount / L, 1.0)
                if g["fill_weight"][i, k] > 0.5:
                    ax_rate.scatter([x], [trade_rate(g["quoted_price"][i, k], side)], s=size,
                                    c="tab:green", edgecolors="k", zorder=5,
                                    label="offer accepted" if first["acc"] else None)
                    first["acc"] = False
                else:
                    ax_rate.scatter([x], [trade_rate(g["quoted_price"][i, k], side)], s=size,
                                    facecolors="none", edgecolors="tab:red", linewidths=1.6, zorder=5,
                                    label="offer rejected" if first["rej"] else None)
                    first["rej"] = False
            if abs(float(held_b[k] - held_f[k])) > 1e-6:
                ax_rate.scatter([hi + 0.12], [bdc_payoff_per_unit(X[k], f, side)], marker="s", s=70,
                                c="tab:orange", edgecolors="k", zorder=5,
                                label="Bureau (after fee)" if first["bdc"] else None)
                first["bdc"] = False
            ax_rate.axvspan(k + 0.5, k + 1.5, color="grey", alpha=0.05)
        ax_rate.plot([rounds + 1.3], [trade_rate(a5, side)], marker="*", ms=16, c="k",
                     label="settlement a5")
        ax_rate.set_ylabel(f"rate ({rate_units(side)})\ntarget per unit held")
        pct = "" if g["pct"] is None else f"  ({g['pct'][i] * PCT:.0f}th pct of its own record P/L)"
        ax_rate.set_title(f"{name} (side {side}, L={L:,.0f} T={T:,.0f}): P/L "
                          f"{float(g['pl_tr'][i]) * PCT:+.3f}%{pct}", fontsize=9)
        ax_rate.legend(fontsize=6, loc="best", ncol=2)
        ax_rate.grid(alpha=0.3)

        xs, ys_h, ys_b = [0.0], [L], [0.0]
        for k in range(rounds):
            xs += [k + 1.0, k + 1.5]
            ys_h += [float(held_f[k]), float(held_b[k])]
            ys_b += [float(bank_f[k]), float(bank_b[k])]
        ax_inv.step(xs, np.array(ys_h) / L, where="post", color="tab:blue", lw=1.8,
                    label=f"{'GBP' if side == 'A' else '$'} left / L")
        ax_inv.step(xs, np.array(ys_b) / max(T, 1.0), where="post", color="tab:green", lw=1.8,
                    label=f"{'$' if side == 'A' else 'GBP'} banked / T")
        ax_inv.axhline(1.0, ls=":", color="grey", label="target")
        ax_inv.axhline(0.0, color="k", lw=0.8)
        ax_inv.set_ylabel("fraction")
        ax_inv.set_xlim(*ax_rate.get_xlim())
        ax_inv.legend(fontsize=6, loc="best")
        ax_inv.grid(alpha=0.3)
        if i == n - 1:
            ax_inv.set_xlabel("round (offers at their arrival position; Bureau after each round)")
    fig.suptitle(f"{g['label']}, trader by trader", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(run / f"{stem}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), 2 * float(d.std()) / np.sqrt(len(d))


def plot_edges(run, ev):
    fig, (tr, mm) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels, means, errs, colours = [], [], [], []
    baseline_idx = [int(i) for i in ev["baseline_index"]]
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        colour = TRADER_COLOURS[i % len(TRADER_COLOURS)]
        m, e = _paired(ev["eval_trader_pl"][i], ev["eval_bdc_pl"][i])
        labels.append(f"{name} ({side})\nvs always-BdC"); means.append(m * PCT); errs.append(e * PCT); colours.append(colour)
        if i in baseline_idx:
            k = baseline_idx.index(i)
            labels.append(f"{name} ({side})\nvs the single-trader {name}")
            means.append(float(ev["baseline_gain"][k]) * PCT); errs.append(float(ev["baseline_2se"][k]) * PCT)
            colours.append(colour)
    tr.bar(range(len(means)), means, yerr=errs, color=colours, capsize=3)
    tr.axhline(0, color="k", lw=0.8)
    tr.set_xticks(range(len(means)))
    tr.set_xticklabels(labels, fontsize=7)
    tr.set_ylabel("paired gain, % of L (2 SE)")
    tr.set_title("the traders", fontsize=9)
    tr.grid(alpha=0.3, axis="y")
    rule_names = [str(n) for n in ev["rule_names"]]
    means = [_paired(ev["eval_mm_pl"], ev["eval_rule_mm_pl"][j])[0] * BP for j in range(len(rule_names))]
    errs = [_paired(ev["eval_mm_pl"], ev["eval_rule_mm_pl"][j])[1] * BP for j in range(len(rule_names))]
    mm.bar(range(len(rule_names)), means, yerr=errs, color=["tab:olive" if "book" in n else "tab:red" if "fee" in n else "tab:blue" for n in rule_names],
           capsize=3)
    mm.axhline(0, color="k", lw=0.8)
    mm.set_xticks(range(len(rule_names)))
    mm.set_xticklabels(rule_names, fontsize=7, rotation=15)
    mm.set_ylabel("co-trained MM minus rule, bp of capital (2 SE)")
    mm.set_title(f"the MM ({float(ev['eval_mm_pl'].mean()) * BP:+.2f} bp) against the rules replayed against its traders",
                 fontsize=9)
    mm.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{_title(ev)}: final paired comparisons, {len(ev['eval_mm_pl']):,} evaluation paths", fontsize=10)
    fig.tight_layout()
    fig.savefig(run / "cotrain_multi_edges.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _offer_pounds(ev):
    """Each recorded offer's size in pounds: a pound seller's amount as it is,
    a dollar seller's amount at its own price (the pounds the MM would pay)."""
    return np.where(ev["record_direction"] > 0, ev["record_offered_amount"],
                    ev["record_offered_amount"] / ev["record_quoted_price"])


def _nettable_pounds(ev):
    """Per record path and round, the pounds an order book would cross: pound
    sellers from the cheapest up against dollar sellers from the highest
    price down, matched while a dollar seller pays more than a pound seller
    asks. That volume is the largest, over the offers' own prices t, of the
    smaller of (pounds offered at or below t) and (pounds wanted at or above
    t). Netted pounds never meet the Bureau, whatever the true rate is."""
    price = ev["record_quoted_price"]
    live = ev["record_offered_amount"] > 0
    pounds = _offer_pounds(ev)
    sells = np.where(live & (ev["record_direction"] > 0), pounds, 0.0)[:, :, None, :]
    wants = np.where(live & (ev["record_direction"] < 0), pounds, 0.0)[:, :, None, :]
    offer_price, level = price[:, :, None, :], price[:, None, :, :]    # offer i, candidate t
    offered_below = (sells * (offer_price <= level)).sum(axis=1)       # (paths, candidates, rounds)
    wanted_above = (wants * (offer_price >= level)).sum(axis=1)
    return np.minimum(offered_below, wanted_above).max(axis=1)         # (paths, rounds)


def _netted_pounds(ev):
    """Per record path and round, the pounds the learned MM netted inside the
    round: the smaller of the pounds it bought from pound sellers and the
    pounds it paid out to dollar sellers."""
    filled = ev["record_fill_weight"] * _offer_pounds(ev)
    bought = np.where(ev["record_direction"] > 0, filled, 0.0).sum(axis=1)
    paid_out = np.where(ev["record_direction"] < 0, filled, 0.0).sum(axis=1)
    return np.minimum(bought, paid_out)


def _order_book(ax, ev, path, rnd, heading, nettable, netted):
    """One round drawn as an order book, prices in $/GBP, sizes in pounds."""
    fee = float(ev["fee"])
    X = float(ev["record_hidden_rate"][path, rnd])
    price = ev["record_quoted_price"][path, :, rnd]
    live = ev["record_offered_amount"][path, :, rnd] > 0
    pounds = _offer_pounds(ev)[path, :, rnd] / 1e3
    direction = ev["record_direction"][path, :, rnd]
    accepted = ev["record_fill_weight"][path, :, rnd] > 0.5
    owner = np.asarray(ev["offer_trader"]).astype(int)
    who = names(ev)
    shown = np.concatenate([price[live], [(1 - fee) * X, X / (1 - fee)]])
    pad = 0.15 * (shown.max() - shown.min()) + 0.004
    lo, hi = shown.min() - pad, shown.max() + pad
    ax.axvspan((1 - fee) * X, X / (1 - fee), color="0.88", lw=0)
    ax.axvline(X, color="k", lw=1.0, ls="--")
    sell, want = live & (direction > 0), live & (direction < 0)
    k_sell = np.flatnonzero(sell)[np.argsort(price[sell])]            # cheapest first
    k_want = np.flatnonzero(want)[np.argsort(price[want])]            # lowest price first
    if k_sell.size:
        up = np.cumsum(pounds[k_sell])
        ax.step(np.concatenate([[lo], price[k_sell], [hi]]), np.concatenate([[0.0], up, [up[-1]]]),
                where="post", color="tab:blue", lw=1.8)
        for k, height in zip(k_sell, up):
            ax.plot(price[k], height, "o", ms=7, color="tab:blue", mfc="tab:blue" if accepted[k] else "white")
            ax.annotate(who[owner[k]], (price[k], height), textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="tab:blue")
    if k_want.size:
        total = pounds[k_want].sum()
        down = total - np.cumsum(pounds[k_want])
        ax.step(np.concatenate([[lo], price[k_want], [hi]]), np.concatenate([[total], down, [0.0]]),
                where="post", color="tab:red", lw=1.8)
        for k, height in zip(k_want, total - np.cumsum(pounds[k_want]) + pounds[k_want]):
            ax.plot(price[k], height, "s", ms=7, color="tab:red", mfc="tab:red" if accepted[k] else "white")
            ax.annotate(who[owner[k]], (price[k], height), textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="tab:red")
    if nettable > 0:
        left, right = price[k_sell].min(), price[k_want].max()
        ax.axvspan(left, right, color="tab:green", alpha=0.12, lw=0)
        ax.hlines(nettable / 1e3, left, right, color="tab:green", lw=2.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("price, $/GBP")
    ax.set_ylabel("pounds, thousands")
    ax.set_title(f"{heading}\npath #{path}, round {rnd + 1} (true rate {X:.4f}): could net "
                 f"{nettable / 1e3:,.1f}k, the MM netted {netted / 1e3:,.1f}k", fontsize=9)
    ax.grid(alpha=0.3)


def plot_netting(run, ev):
    """Netting, the limit-order-book view. A round of this game is a tiny
    order book: every offer is a limit order, pound sellers on one side and
    dollar sellers on the other. Where a dollar seller's price is above a
    pound seller's, the two cross and the MM can pass pounds from one to the
    other without the Bureau. Three single rounds are drawn as books (a
    typical crossing round, a large one, and a typical round with offers on
    both sides that do not cross), then the by-round totals."""
    nettable, netted = _nettable_pounds(ev), _netted_pounds(ev)
    cap, rounds = float(ev["capital_pounds"]), int(ev["rounds"])
    live = ev["record_offered_amount"] > 0
    both_sides = ((live & (ev["record_direction"] > 0)).any(axis=1)
                  & (live & (ev["record_direction"] < 0)).any(axis=1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    crossing_cells = np.argwhere(nettable > 0)
    quiet_cells = np.argwhere((nettable <= 0) & both_sides)
    # fixed places: two crossing rounds, then a round that does not cross
    picks = [None, None, None]
    notes = ["no round in this run had a crossing pair:\nthere was nothing to net", "",
             "no round had offers on both sides\nthat did not cross"]
    if len(crossing_cells):
        by_size = crossing_cells[np.argsort(nettable[tuple(crossing_cells.T)], kind="stable")]
        picks[0] = ("a typical crossing round (median of the crossing rounds)", by_size[len(by_size) // 2])
        picks[1] = ("a large crossing round (95th percentile)", by_size[int(0.95 * (len(by_size) - 1))])
    if len(quiet_cells):
        offered = _offer_pounds(ev).sum(axis=1)[tuple(quiet_cells.T)]
        picks[2] = ("a typical round with both sides but no crossing",
                    quiet_cells[np.argsort(offered, kind="stable")][len(quiet_cells) // 2])
    for ax, pick, missing in zip(axes.flat[:3], picks, notes):
        if pick is None:
            ax.axis("off")
            ax.text(0.5, 0.5, missing, ha="center", va="center", fontsize=10, transform=ax.transAxes)
            continue
        heading, (path, rnd) = pick
        _order_book(ax, ev, int(path), int(rnd), heading, float(nettable[path, rnd]),
                    float(netted[path, rnd]))

    ax = axes.flat[3]
    r = np.arange(1, rounds + 1)
    bureau_r, _ = _bureau_volume_by_round(ev)
    ax.bar(r - 0.27, nettable.mean(axis=0) / cap * PCT, width=0.27, color="tab:green", alpha=0.45,
           label="could be netted: crossing offers")
    ax.bar(r, netted.mean(axis=0) / cap * PCT, width=0.27, color="tab:green",
           label="netted by the learned MM inside the round")
    ax.bar(r + 0.27, bureau_r.mean(axis=0) / cap * PCT, width=0.27, color="tab:red",
           label="moved at the Bureau (after the round)")
    if nettable.sum() > 0:
        captured = np.minimum(netted, nettable).sum() / nettable.sum()
        ax.set_title(f"by round, mean per game\ncrossing offers in {(nettable > 0).mean() * PCT:.1f}% of rounds; "
                     f"the MM netted {captured * PCT:.1f}% of what could be netted", fontsize=9)
    else:
        ax.set_title("by round, mean per game\nno crossing offers, so nothing could be netted", fontsize=9)
    ax.set_xticks(r)
    ax.set_xlabel("round")
    ax.set_ylabel("pounds, % of the game's capital")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="best")
    key = [Patch(color="0.88", label="MM fee band around the true rate"),
           Line2D([], [], color="k", ls="--", lw=1.0, label="true rate"),
           Line2D([], [], color="tab:blue", lw=1.8, label="pound sellers: pounds offered at or below this price"),
           Line2D([], [], color="tab:red", lw=1.8, label="dollar sellers: pounds wanted at or above this price"),
           Patch(color="tab:green", alpha=0.25, label="offers cross: a dollar seller pays more than a pound seller asks"),
           Line2D([], [], color="tab:green", lw=2.2, label="pounds that can be netted"),
           Line2D([], [], ls="none", marker="o", color="0.3", mfc="0.3", label="filled marker: the learned MM accepted"),
           Line2D([], [], ls="none", marker="o", color="0.3", mfc="white", label="open marker: the learned MM rejected")]
    fig.legend(handles=key, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(f"{_title(ev)}: netting, the limit-order-book view "
                 f"({ev['record_quoted_price'].shape[0]:,} record paths)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(run / "cotrain_multi_netting.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_summary(run, curve, ev):
    N, rounds = len(names(ev)), int(ev["rounds"])
    out = []
    w = out.append
    w(f"CO-TRAINED GAME {'-'.join(names(ev))}  (sides {''.join(sides(ev))})")
    w("=" * 78)
    w("traders")
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        L, T, A, B = ev["cards"][i]
        key = str(ev["card_keys"][i])
        w(f"  {name}: side {side}, L={L:,.0f} T={T:,.0f} A={A:g} B={B:g}"
          + (f"  (single-trader card {key})" if key else "  (new card)"))
    w(f"market: a0 {float(ev['a0'])}, sd {float(ev['sd'])}, rounds {rounds}, trader BdC fee "
      f"{float(ev['trader_bdc_fee'])}, MM fee {float(ev['fee'])}, game capital GBP {float(ev['capital_pounds']):,.0f}")
    w(f"MM architecture {str(ev['mm_arch']).upper()}: "
      + {"round": "one level per round per direction, set from the book as the round opens; verdicts order-free",
         "bid": "level re-set per offer from the LIVE book; the first offer of a round decided blind (the real game)",
         "turn": "every offer of the round read before any verdict; round-open book; verdicts concurrent and order-free"
                 + " (every offer read, the judged one included: the cut may depend on the own price)"
         }[str(ev["mm_arch"])])
    if bool(ev["two_way"]):
        base = ev["offer_directions"][None, :, None]
        rev = ev["record_direction"] != base
        X = ev["record_hidden_rate"][:, None, :]
        vol = np.where(ev["record_direction"] > 0, ev["record_offered_amount"],
                       ev["record_offered_amount"] / X)
        f = ev["record_fill_weight"]
        w("TWO-WAY: one offer a round, the trader choosing its direction. Share of each trader's offered "
          "pound-equivalent volume that went REVERSE (sold banked currency back), and those offers' filled share: "
          + "  ".join(f"{name} {vol[:, i][rev[:, i]].sum() / max(vol[:, i].sum(), 1e-300) * 100:.1f}% / "
                      f"{(f[:, i] * vol[:, i])[rev[:, i]].sum() / max(vol[:, i][rev[:, i]].sum(), 1e-300) * 100:.1f}%"
                      for i, name in enumerate(names(ev))))
    w(f"MM sees: {int(ev['n_mm_feat'])} inputs (direction {bool(ev['see_direction'])}, sub-round "
      f"{bool(ev['see_queue'])}, offer size {bool(ev['see_offer_size'])}, flow block {int(ev['n_flow'])} columns); "
      f"level read in direction {float(ev['ref_direction']):+.0f}")
    w(f"schedule: {int(ev['n_outer'])} x {int(ev['n_inner'])} x {int(ev['batch']):,} games, {str(ev['first'])} first, "
      f"MM init {str(ev['mm_level_init'])}, LR trader {float(ev['lr'][0]):g} MM {float(ev['lr'][1]):g}, "
      f"plateau {bool(ev['trader_plateau'])}, tau trader {tuple(float(v) for v in ev['tau_trader'])} MM {tuple(float(v) for v in ev['tau_mm'])} "
      f"over {float(ev['tau_anneal_frac']):.0%}")
    stopped = int(curve["stopped_at"])
    w(f"convergence: {len(curve['mm_pl'])} blocks; "
      + (f"flat, stopped at outer step {stopped}" if stopped >= 0 else "ran the full schedule (not flat)"))
    w("")
    w(f"FINAL (hard game, {len(ev['eval_mm_pl']):,} evaluation paths)")
    mm_pl = ev["eval_mm_pl"]
    w(f"  MM P/L {mm_pl.mean() * BP:+.2f} bp +/- {2 * mm_pl.std() / np.sqrt(len(mm_pl)) * BP:.2f}; "
      f"Bureau share of filled pound volume {float(ev['bureau_share']) * PCT:.1f}%")
    baseline_idx = [int(i) for i in ev["baseline_index"]]
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        m, e = _paired(ev["eval_trader_pl"][i], ev["eval_bdc_pl"][i])
        line = (f"  {name} ({side}) P/L {ev['eval_trader_pl'][i].mean() * PCT:+.4f}%; edge over always-BdC "
                f"{m * PCT:+.4f}% +/- {e * PCT:.4f}% (floor {ev['eval_bdc_pl'][i].mean() * PCT:+.4f}%, "
                f"analytic {float(ev['bdc_floor'][i]) * PCT:+.4f}%)")
        if i in baseline_idx:
            k = baseline_idx.index(i)
            line += (f"; vs the single-trader run {float(ev['baseline_gain'][k]) * PCT:+.4f}% +/- "
                     f"{float(ev['baseline_2se'][k]) * PCT:.4f}%")
        w(line)
    for j, rule in enumerate(ev["rule_names"]):
        m, e = _paired(mm_pl, ev["eval_rule_mm_pl"][j])
        w(f"  vs {str(rule):<18} MM {ev['eval_rule_mm_pl'][j].mean() * BP:+8.2f} bp   co MM - rule "
          f"{m * BP:+.2f} +/- {e * BP:.2f} bp   traders vs it "
          + "  ".join(f"{ev['eval_rule_trader_pl'][j, i].mean() * PCT:+.4f}%" for i in range(N)))
    w(f"  verdicts agree with {str(ev['benchmark_rule'])} on {float(ev['agree']) * PCT:.1f}% of offered "
      f"pound volume; filled {float(ev['filled']) * PCT:.2f}% (rule {float(ev['filled_benchmark']) * PCT:.2f}%)")
    if "book_filled" in ev and np.isfinite(float(ev["book_filled"])):
        w(f"  book+flatten replayed on the record paths: filled {float(ev['book_filled']) * PCT:.2f}% of offered "
          f"pound volume, Bureau share {float(ev['book_bureau_share']) * PCT:.1f}% "
          f"(learned MM: filled {float(ev['filled']) * PCT:.2f}%, Bureau share {float(ev['bureau_share']) * PCT:.1f}%)")
    w("")
    w("THE POLICY BY ROUND (record paths)")
    fee = float(ev["fee"])
    w("  bid (pound sellers filled below) / X : " + "  ".join(f"r{k + 1} {v:.4f}" for k, v in enumerate(ev["bid_by_round"]))
      + f"   (fee-aware {fee_aware_cut('A', fee):.4f}, naive 1)")
    w("  ask (dollar sellers filled above) / X: " + "  ".join(f"r{k + 1} {v:.4f}" for k, v in enumerate(ev["ask_by_round"]))
      + f"   (fee-aware {fee_aware_cut('B', fee):.4f}, naive 1)")
    spread = ev["ask_by_round"] - ev["bid_by_round"]
    w("  spread ask - bid, % of X          : " + "  ".join(f"r{k + 1} {v * PCT:+.2f}%" for k, v in enumerate(spread))
      + f"   (fee-aware {2 * fee / (1 - fee) * PCT:.2f}%, naive 0)")
    w("  repay fraction on short books     : " + "  ".join(f"r{k + 1} {v:.3f}" for k, v in enumerate(ev["repay_by_round"])))
    X = ev["record_hidden_rate"]
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        L = float(ev["cards"][i, 0])
        u = ev["record_quoted_price"][:, i, :]
        offered, fill = ev["record_offered_amount"][:, i, :], ev["record_fill_weight"][:, i, :]
        share = offered / np.maximum(_held_start(ev, i, L), 1e-9)
        vol_filled = (offered * fill).sum(axis=0) / np.maximum(offered.sum(axis=0), 1e-300)
        w(f"  {name} ({side}) offer, median $/GBP : " + "  ".join(f"r{k + 1} {v:.4f}" for k, v in enumerate(np.median(u, axis=0))))
        w(f"  {name} ({side}) book put up, median : " + "  ".join(f"r{k + 1} {v * PCT:5.1f}%" for k, v in enumerate(np.median(share, axis=0))))
        w(f"  {name} ({side}) volume filled       : " + "  ".join(f"r{k + 1} {v * PCT:5.2f}%" for k, v in enumerate(vol_filled)))
    filled_r = _filled_pound_volume_by_round(ev)
    bureau_r, final = _bureau_volume_by_round(ev)
    cap = float(ev["capital_pounds"])
    w("  MM pound volume filled, mean      : " + "  ".join(f"r{k + 1} {v:5.2f}%" for k, v in enumerate(filled_r.mean(axis=0) / cap * PCT)) + " of capital")
    w("  MM pound volume at the Bureau     : " + "  ".join(f"r{k + 1} {v:5.2f}%" for k, v in enumerate(bureau_r.mean(axis=0) / cap * PCT))
      + f"  settle {final.mean() / cap * PCT:5.2f}%")
    total_filled, total_bureau = filled_r.sum(), bureau_r.sum() + final.sum()
    offered = np.where(ev["record_direction"] > 0, ev["record_offered_amount"],
                       ev["record_offered_amount"] / ev["record_quoted_price"]).sum()
    if offered > 0 and total_filled / offered >= CM.MIN_FILLED_SHARE:
        w(f"  crossed or carried out solvent    : {(1 - total_bureau / total_filled) * PCT:.1f}% of filled volume")
    else:
        w(f"  crossed or carried out solvent    : not measurable ({total_filled / max(offered, 1e-300) * PCT:.2f}% "
          f"of offered volume filled)")
    nettable, netted = _nettable_pounds(ev), _netted_pounds(ev)
    w(f"  netting inside the round          : crossing offers in {(nettable > 0).mean() * PCT:.1f}% of rounds; "
      f"could net {nettable.sum(axis=1).mean() / cap * PCT:.2f}% of capital per game, the MM netted "
      f"{netted.sum(axis=1).mean() / cap * PCT:.2f}%"
      + (f" ({np.minimum(netted, nettable).sum() / nettable.sum() * PCT:.1f}% of what could be netted)"
         if nettable.sum() > 0 else ""))
    w("")
    w("FINAL BOOKS (record paths, medians)")
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        L, T = float(ev["cards"][i, 0]), float(ev["cards"][i, 1])
        held = ev["record_trader_held_after_bureau"][:, i, -1] / L
        banked = ev["record_trader_banked_after_bureau"][:, i, -1] / T
        w(f"  {name} ({side}): leftover {np.median(held) * PCT:.2f}% of L, banked {np.median(banked) * PCT:.2f}% of T, "
          f"target met in {(banked >= 1.0).mean() * PCT:.1f}% of games")
    a5 = ev["record_settlement_rate"]
    w(f"  MM: pounds {np.median(ev['record_mm_pounds_end'] / cap) * PCT:+.3f}% of capital, dollars (at a5) "
      f"{np.median(ev['record_mm_dollars_end'] / a5 / cap) * PCT:+.3f}%; P/L {ev['record_mm_pl'].mean() * BP:+.2f} bp")
    w("")
    w(f"THE DEMO GAME (rates {list(ev['set_path'])})")
    w(f"  MM P/L {float(ev['set_mm_pl']) * BP:+.2f} bp; traders "
      + "  ".join(f"{name} {float(pl) * PCT:+.3f}%" for name, pl in zip(names(ev), ev["set_trader_pl"])))
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        q, fill, level = ev["set_record_quoted_price"][0, i], ev["set_record_fill_weight"][0, i], ev["set_record_level"][0, i]
        w(f"  {name} ({side}): " + "  ".join(f"r{k + 1} quote {q[k]:.4f} level {level[k]:.4f} {'FILLED' if fill[k] > 0.5 else 'rejected'}"
                                                for k in range(rounds)))
    w("")
    g = _game(ev, "median")
    w(f"THE MEDIAN GAME (median MM P/L of {len(ev['record_mm_pl']):,} record paths, path #{g['idx']}; "
      f"rates {[round(float(v), 4) for v in g['rates']]})")
    w(f"  MM P/L {g['pl_mm'] * BP:+.2f} bp; traders "
      + "  ".join(f"{name} {float(pl) * PCT:+.3f}% ({g['pct'][i] * PCT:.0f}th pct)"
                  for i, (name, pl) in enumerate(zip(names(ev), g["pl_tr"]))))
    for i, (name, side) in enumerate(zip(names(ev), sides(ev))):
        q, fill, level = g["quoted_price"][i], g["fill_weight"][i], g["level"][i]
        w(f"  {name} ({side}): " + "  ".join(f"r{k + 1} quote {q[k]:.4f} level {level[k]:.4f} "
                                             f"{'FILLED' if fill[k] > 0.5 else 'rejected'}"
                                             for k in range(rounds)))
    w("")
    w("NEGATIVE RATE PATHS (share deleted, by sd; first-hit draw 1..rounds+1)")
    for sd, row in zip(ev["negative_table_sds"], ev["negative_table"]):
        w(f"  sd {float(sd):.2f}: P(deleted) {row[0]:.1e}   " + "  ".join(f"{v:.1e}" for v in row[1:]))
    w("")
    w("LEARNING CURVE (validation paths; MM bp, traders % of L; edge = paired over always-BdC)")
    for k in range(len(curve["mm_pl"])):
        w(f"  block {k:>2} after {str(curve['moved'][k]):<6} MM {curve['mm_pl'][k] * BP:+8.2f} bp  "
          + "  ".join(f"{name} {curve['trader_pl'][k, i] * PCT:+.4f}% (edge {curve['edge'][k, i] * PCT:+.4f}, filled "
                      f"{curve['filled_i'][k, i] * PCT:5.2f}%)" for i, name in enumerate(names(ev)))
          + f"  bid {curve['bid'][k, 0]:.4f} ask {curve['ask'][k, 0]:.4f} bureau {curve['bureau_share'][k] * PCT:5.1f}%"
          + "  lr " + "/".join(f"{v:.0e}" for v in curve["lr_trader"][k]))
    (run / "cotrain_multi_summary.txt").write_text("\n".join(out) + "\n")


def main():
    run, curve, ev = load()
    plot_curve(run, curve, ev)
    plot_policy(run, curve, ev)
    plot_offers(run, ev)
    plot_mm_book(run, ev)
    plot_final_books(run, ev)
    plot_set_path(run, ev)
    plot_median_path(run, ev)
    plot_trader_paths(run, ev, "set", "cotrain_multi_set_traders")
    plot_trader_paths(run, ev, "median", "cotrain_multi_median_traders")
    plot_edges(run, ev)
    plot_netting(run, ev)
    write_summary(run, curve, ev)
    print(f"{run} (MM architecture {str(ev['mm_arch'])}):\n  cotrain_multi_curve.png, cotrain_multi_policy.png, cotrain_multi_offers.png, "
          f"cotrain_multi_mm_book.png,\n  cotrain_multi_final_books.png, cotrain_multi_set_path.png, "
          f"cotrain_multi_median_path.png,\n  cotrain_multi_set_traders.png, cotrain_multi_median_traders.png, "
          f"cotrain_multi_edges.png, cotrain_multi_netting.png, cotrain_multi_summary.txt")


if __name__ == "__main__":
    main()