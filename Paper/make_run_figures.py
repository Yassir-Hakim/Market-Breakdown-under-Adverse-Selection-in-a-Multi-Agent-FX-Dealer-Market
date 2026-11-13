'''
make_run_figures.py -- the paper's learning-curve figures, drawn from the saved
training curves exactly as the repo's diagnostics draw them.

Content is the same as Co-Train/cotrain_diagnostics.py (plot_curve -> cotrain_curve.png)
and Co-Train_Multi/cotrain_multi_diagnostics.py (plot_curve -> cotrain_multi_curve.png):
every block of every run, nothing cropped or smoothed, the same series:
  - the co-trained MM's hard-game P/L on the validation paths, with its 2 s.e. band,
  - every hardcoded rule's P/L against the traders at that block
    (colour: naive blue, fee-aware red, book olive; solid/dash-dot = flatten, dashed/dotted = carry),
  - each trader's hard-game P/L, with the 2 s.e. band of its paired edge,
  - the always-Bureau P/L on the same paths (dashed orange),
  - markers for who had just moved (MM square, traders circle, as initialised diamond),
  - the stopping block, if the stopping rule fired.
Only the formatting differs: the paper's fonts and size, one shared legend, and the
paper's words (Bureau, pound/dollar seller).

Reads Results/cotrain/{A,B}_{tag}/cotrain_curve.npz + cotrain_eval.npz and
Results/cotrain_multi/T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way/cotrain_multi_curve.npz
+ cotrain_multi_eval.npz. Trains nothing. Writes Paper/figures/fig_learning.{pdf,png} (one pound seller and one dollar seller,
each co-trained with a market maker at (3%,1%)) and Paper/figures/fig_learning4.{pdf,png}
(the four-trader market at a 2% spread: the MM and T1-T4), one panel per agent.

Usage: click Run (python make_run_figures.py).
'''
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "Results"
FIGS = HERE / "figures"
BP, PCT = 1e4, 100.0

# the repo diagnostics' colours, unchanged
COLOUR = {"mm": "tab:purple", "trader": "tab:green", "init": "0.5"}
INK, INK2, GRID = "#1a1a1a", "#55534e", "#e6e5e0"


def style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((n for n in ("TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                               "Book Antiqua", "DejaVu Serif") if n in names), "serif")
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})


def rule_style(name, repay):
    """the repo's mapping from a hardcoded rule to its colour and line style"""
    book = "book" in name
    colour = "tab:olive" if book else "tab:red" if "fee" in name else "tab:blue"
    ls = ({1.0: "-.", 0.0: ":"} if book else {1.0: "-", 0.0: "--"}).get(float(repay), ":")
    return colour, ls


def mark_movers(ax, x, y, moved):
    for agent, marker in (("mm", "s"), ("trader", "o"), ("init", "D")):
        idx = [i for i, m in enumerate(moved) if m == agent]
        if idx:
            ax.plot(x[idx], y[idx], marker, color=COLOUR[agent], ms=2.6, ls="none", zorder=4)


def grid(ax):
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def mm_panel(ax, curve, rule_repay):
    x = np.arange(len(curve["mm_pl"]))
    moved = [str(m) for m in curve["moved"]]
    mm, se = curve["mm_pl"] * BP, curve["mm_2se"] * BP
    ax.plot(x, mm, color="k", lw=1.1, zorder=3)
    ax.fill_between(x, mm - se, mm + se, color="k", alpha=0.12, lw=0)
    for j, name in enumerate(str(n) for n in curve["rule_names"][0]):
        colour, ls = rule_style(name, rule_repay[j])
        ax.plot(x, curve["rule_pl"][:, j] * BP, ls=ls, lw=0.8, color=colour)
    mark_movers(ax, x, mm, moved)
    stop(ax, curve, x)
    grid(ax)


def trader_panel(ax, curve, i=None):
    x = np.arange(len(curve["mm_pl"]))
    moved = [str(m) for m in curve["moved"]]
    pl = curve["trader_pl"] if i is None else curve["trader_pl"][:, i]
    band = curve["edge_2se"] if i is None else curve["edge_2se"][:, i]
    floor = curve["bdc_pl"][0] if i is None else curve["bdc_pl"][-1, i]   # constant: fixed paths
    ax.plot(x, pl * PCT, color="k", lw=1.1, zorder=3)
    ax.fill_between(x, (pl - band) * PCT, (pl + band) * PCT, color="k", alpha=0.12, lw=0)
    ax.axhline(float(floor) * PCT, color="tab:orange", ls="--", lw=1.0)
    if i is None and "best_response_pl" in curve and np.isfinite(curve["best_response_pl"]):
        ax.axhline(float(curve["best_response_pl"]) * PCT, color="tab:purple", ls="-.", lw=1.0)
    mark_movers(ax, x, pl * PCT, moved)
    stop(ax, curve, x)
    grid(ax)


def stop(ax, curve, x):
    if int(curve["stopped_at"]) >= 0:
        ax.axvline(x[-1], color="0.6", lw=0.8, ls=":")


def legend_handles(rule_names, rule_repay, traders="trader", stopped=True):
    h = [Line2D([], [], color="k", lw=1.1, label="co-trained agent, hard game (validation games)"),
         Patch(color="k", alpha=0.12, lw=0, label="2 s.e. (trader: of its paired edge over always-Bureau)")]
    for name, rp in zip(rule_names, rule_repay):
        c, ls = rule_style(name, rp)
        h.append(Line2D([], [], color=c, ls=ls, lw=0.8, label=f"fixed rule: {name.replace('+', ', ')}"))
    h += [Line2D([], [], color="tab:orange", ls="--", lw=1.0, label="always-Bureau, same games"),
          ] + ([Line2D([], [], color="0.6", ls=":", lw=0.8, label="stopping rule fired")] if stopped else []) + [
          Line2D([], [], marker="s", color=COLOUR["mm"], ls="none", ms=3, label="after the MM moved"),
          Line2D([], [], marker="o", color=COLOUR["trader"], ls="none", ms=3, label=f"after the {traders} moved"),
          Line2D([], [], marker="D", color=COLOUR["init"], ls="none", ms=3, label="as initialised")]
    return h


def repo_legend(rule_names=(), rule_repay=(), traders="trader", stopped=False):
    h = [Line2D([], [], color="k", lw=1.1, label="co-trained agent, hard game (validation games)"),
         Patch(color="k", alpha=0.12, lw=0, label="2 s.e. (trader: of its paired edge)"),
         Line2D([], [], color="tab:orange", ls="--", lw=1.0, label="always-Bureau, same games")]
    for name, rp in zip(rule_names, rule_repay):
        c, ls = rule_style(name, rp)
        h.append(Line2D([], [], color=c, ls=ls, lw=0.8, label=f"fixed rule: {name.replace('+', ', ')}"))
    if stopped:
        h.append(Line2D([], [], color="0.6", ls=":", lw=0.8, label="stopping rule fired"))
    h += [Line2D([], [], marker="s", color=COLOUR["mm"], ls="none", ms=3, label="after the MM moved"),
          Line2D([], [], marker="o", color=COLOUR["trader"], ls="none", ms=3, label=f"after the {traders} moved"),
          Line2D([], [], marker="D", color=COLOUR["init"], ls="none", ms=3, label="as initialised")]
    return h


def fig_learning():
    """one trader and one market maker, the open channel with the MM's fee, (3%,1%):
    the pound seller settles, the dollar seller does not. One panel each, own axis."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    fig.subplots_adjust(wspace=0.32)
    for ax, side, title in ((axes[0], "A", "(a) One pound seller, (3%, 1%)"),
                            (axes[1], "B", "(b) One dollar seller, (3%, 1%)")):
        curve = np.load(RESULTS / "cotrain" / f"{side}_fee0.01_BdC_FEE0.03" / "cotrain_curve.npz",
                        allow_pickle=True)
        trader_panel(ax, curve)
        ax.set_title(title, loc="left")
        ax.set_ylabel("trader P/L (% of capital)")
        ax.set_xlabel("block")
    fig.legend(handles=repo_legend(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_learning")


def fig_learning4():
    """the four-trader market at a 2% spread: the MM and each trader on its own axis"""
    run = RESULTS / "cotrain_multi" / "T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way"
    curve = np.load(run / "cotrain_multi_curve.npz", allow_pickle=True)
    ev = np.load(run / "cotrain_multi_eval.npz", allow_pickle=True)
    traders = [str(t) for t in ev["traders"]]
    sides = [str(s) for s in ev["sides"]]
    # a full appendix page: each row about as tall as a panel of the repo's plot_curve (3 in)
    fig = plt.figure(figsize=(6.3, 8.0))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.3)
    top = fig.add_subplot(gs[0, :])
    mm_panel(top, curve, ev["rule_repay"])
    top.set_title("(a) The market maker", loc="left")
    top.set_ylabel("MM P/L (bp of the game's capital)")
    for i, (name, side) in enumerate(zip(traders, sides)):
        ax = fig.add_subplot(gs[1 + i // 2, i % 2], sharex=top)
        trader_panel(ax, curve, i)
        who = "pound seller" if side == "A" else "dollar seller"
        ax.set_title(f"({'bcde'[i]}) {name}, {who}", loc="left")
        ax.set_ylabel(f"{name} P/L (% of capital)")
        if i >= 2:
            ax.set_xlabel("block")
    rules = [str(n) for n in curve["rule_names"][0]]
    fig.legend(handles=repo_legend(rules, ev["rule_repay"], "traders", int(curve["stopped_at"]) >= 0),
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.055))
    save(fig, "fig_learning4")


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(FIGS / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


if __name__ == "__main__":
    style()
    fig_learning()
    fig_learning4()
