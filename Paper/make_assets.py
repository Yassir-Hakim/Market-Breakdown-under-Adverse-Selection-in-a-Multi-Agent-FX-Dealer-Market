import sys
from pathlib import Path
PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
for _p in (ROOT / "NN_one_offer_game", ROOT / "MM_Phase_1", ROOT / "Co-Train_Multi",
           ROOT / "DP_Models", ROOT):
    sys.path.insert(0, str(_p))
"""
  make_assets.py -- every number and every figure in the paper, from the code.

  Two sources, nothing typed by hand:
    model   the exact results, computed here from the model code (v0_game,
            fee_aware_v0, v1_sequential_offers, v2_multiple_rounds). The long
            dynamic-programming solves are cached in Paper/cache/ -- delete a
            cache file to recompute it.
    runs    the learning results, read from the saved runs under Results/
            (cotrain_multi, fee_aware_dp, mm_phase1, cotrain, ppo_*), with the
            project's own reading functions (redo_numbers.py) where they exist.

  Output:
    numbers.tex    one macro per number: \val{key} in the text
    numbers.json   the same numbers with the function and file each came from
    figures/*.pdf  every figure in the paper

  Usage: click Run (python make_assets.py). About 30 min the first time (the
  cached solves), a few minutes after that. Nothing is trained.
"""
import argparse
import json
import math
import re
import time
import traceback

import numpy as np

CACHE = PAPER / "cache"
FIGS = PAPER / "figures"
RESULTS = ROOT / "Results"
A0, SD, F_TRADER, G_MM = 1.25, 0.05, 0.02, 0.01


# =============================================================================
# 1. the registry: one entry per number in the paper
# =============================================================================
class Numbers:
    def __init__(self):
        self.items = {}

    def put(self, key, value, text, source):
        assert re.fullmatch(r"[A-Za-z0-9.]+", key), key
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.items[key] = dict(value=value, text=str(text), source=source)
        return value


def signed(s):
    return s if s.startswith("-") else "+" + s


def fnum(x, nd=2, sign=False, comma=False):
    s = f"{x:,.{nd}f}" if comma else f"{x:.{nd}f}"
    if sign:
        s = signed(s)
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def fpct(x, nd=1, sign=False):
    """x is a fraction; returns '7.4\\%'."""
    return fnum(100.0 * x, nd, sign) + "\\%"


def fpct_small(x, sig=2):
    """a fraction shown as a percentage with `sig` significant digits"""
    v = 100.0 * x
    if v == 0:
        return "0\\%"
    if abs(v) < 1e-3:
        return fsci(v, sig - 1) + "\\%"
    nd = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return fnum(v, nd) + "\\%"


def fsci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    out = fn()
    out["_seconds"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(out, indent=1))
    return out


# =============================================================================
# 2. model numbers (need scipy and the model code)
# =============================================================================
def model_params(N):
    N.put("p.a0", A0, fnum(A0, 2), "fx_mechanics.A0_DEFAULT")
    N.put("p.sd", SD, fnum(SD, 2), "fx_mechanics.sd_DEFAULT")
    N.put("p.f", F_TRADER, fpct(F_TRADER, 0), "rules: traders' Bureau fee")
    N.put("p.g", G_MM, fpct(G_MM, 0), "rules: market maker's Bureau fee")
    N.put("p.sdrel", SD / A0, fpct(SD / A0, 0), "sd / a0")
    # the single-trader trainer's budget, read from its settings
    src = (ROOT / "MM_Phase_1" / "torch_train_fee.py").read_text()
    for name, key in (("BATCH", "st.batch"), ("ITERS", "st.iters")):
        v = int(re.search(rf"^{name} = (\d+)", src, re.M).group(1))
        N.put(key, v, f"{v:,}", f"MM_Phase_1/torch_train_fee.py: {name}")


def model_v0(N, D):
    from scipy.stats import norm
    from scipy.integrate import quad
    import v0_game as V0
    out = {}
    for side in ("A", "B"):
        P = V0.closed_form_optimal_rate(A0, SD, F_TRADER, side)
        z = (P - A0) / SD
        fill = 1 - norm.cdf(z) if side == "A" else norm.cdf(z)
        val = V0.expected_target_per_unit(P, A0, SD, F_TRADER, side)
        pinf = V0.perfect_information_value(A0, SD, side)
        floor = (1 - F_TRADER) * pinf
        share = (val - floor) / (F_TRADER * pinf)
        out[side] = dict(P=P, z=abs(z), fill=fill, val=val, pinf=pinf, floor=floor,
                         share=share, edge=val - floor)
    s, b = out["A"], out["B"]
    src = "v0_game.closed_form_optimal_rate / expected_target_per_unit"
    N.put("v0.Pstar", s["P"], fnum(s["P"], 4), src)
    N.put("v0.zstar", s["z"], fnum(s["z"], 2), src)
    N.put("v0.zstarfine", s["z"], fnum(s["z"], 4), src)
    N.put("v0.fill", s["fill"], fpct(s["fill"], 1), src)
    N.put("v0.fillfine", s["fill"], fpct(s["fill"], 2), src)
    N.put("v0.value", s["val"], fnum(s["val"], 6), src)
    N.put("v0.floor", s["floor"], fnum(s["floor"], 4), src)
    N.put("v0.share", s["share"], fpct(s["share"], 1), src)
    N.put("v0.edge", s["edge"], fsci(s["edge"], 1), src)
    N.put("v0.edgebp", s["edge"] / A0, fnum(1e4 * s["edge"] / A0, 1), src + " (bp of a pound)")
    N.put("v0.pie", F_TRADER * A0, fnum(F_TRADER * A0, 3), "f * a0")
    N.put("v0B.P", b["P"], fnum(b["P"], 4), src)
    N.put("v0B.offer", 1 / b["P"], fnum(1 / b["P"], 4), src + " (pounds per dollar)")
    N.put("v0B.zstar", b["z"], fnum(b["z"], 2), src)
    N.put("v0B.fill", b["fill"], fpct(b["fill"], 1), src)
    N.put("v0B.share", b["share"], fpct(b["share"], 2), src)
    N.put("v0B.Einv", b["pinf"], fnum(b["pinf"], 6), "v0_game.perfect_information_value(side B)")
    # Monte Carlo road test of the optimum (fixed seed)
    np.random.seed(0)
    mc, se = V0.monte_carlo_expected_target_per_unit(s["P"], A0, SD, F_TRADER, 200_000, "A")
    N.put("v0.mc", mc, fnum(mc, 5), "v0_game.monte_carlo_expected_target_per_unit (200,000 games, seed 0)")
    N.put("v0.mcse", 2 * se, fnum(2 * se, 5), "same, 2 s.e.")
    N.put("v0.mcgap", (mc - s["val"]) / se, fnum((mc - s["val"]) / se, 1, sign=True), "same, gap in s.e.")
    # the curves for Figure 3 (both sides, in the trader's own units)
    curves = {}
    for side in ("A", "B"):
        if side == "A":
            Ps = np.linspace(A0 - 2 * SD, A0 + 4 * SD, 241)
        else:
            Ps = np.linspace(A0 - 4 * SD, A0 + 2 * SD, 241)
        vals = [V0.expected_target_per_unit(P, A0, SD, F_TRADER, side) for P in Ps]
        curves[side] = dict(P=Ps.tolist(), value=vals, Pstar=out[side]["P"],
                            vstar=out[side]["val"], floor=out[side]["floor"],
                            pinf=out[side]["pinf"])
    D["v0_curves"] = curves
    # the payoff's per-game spread at the optimum and what it means for sampling
    P = s["P"]
    EY = s["val"]
    EY2 = P * P * (1 - norm.cdf(s["z"])) + (1 - F_TRADER) ** 2 * quad(
        lambda x: x * x * norm.pdf(x, A0, SD), A0 - 12 * SD, P, epsabs=1e-13)[0]
    sdY = math.sqrt(EY2 - EY * EY)
    ratio = s["edge"] / sdY
    n_detect = (2.0 / ratio) ** 2
    N.put("v0.sdY", sdY, fnum(sdY, 4), "sd of the one-shot payoff at P*, quadrature")
    N.put("v0.edgeratio", ratio, fnum(ratio, 3), "edge / payoff sd")
    N.put("v0.ndetect", n_detect, f"{round(n_detect, -3):,.0f}", "(2 / ratio)^2 games to see the edge at 2 s.e.")
    # the pathwise gradient through the hard acceptance rule, at the optimum
    N.put("v0.pathwise", s["fill"], fnum(s["fill"], 4),
          "pathwise dG/dP with the hard rule = Pr(X > P*) (true derivative 0)")
    return out


def model_relaxation(N, D):
    """The optimum of the sigmoid-relaxed one-shot game, against the exact one."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm
    import v0_game as V0
    Pstar = V0.closed_form_optimal_rate(A0, SD, F_TRADER, "A")

    def G_tau(P, tau):
        w = tau * SD

        def integrand(x):
            s = 0.5 * (1 + math.tanh((x - P) / (2 * w)))
            return (s * P + (1 - s) * (1 - F_TRADER) * x) * norm.pdf(x, A0, SD)
        lo, hi = A0 - 10 * SD, A0 + 10 * SD
        pts = [p for p in (P - 20 * w, P, P + 20 * w) if lo < p < hi]
        return quad(integrand, lo, hi, points=pts, limit=400, epsabs=1e-13, epsrel=1e-12)[0]

    rows = []
    for tau in (1.0, 0.2, 0.02, 0.001):
        r = minimize_scalar(lambda P: -G_tau(P, tau), bounds=(A0, A0 + 4 * SD),
                            method="bounded", options=dict(xatol=1e-10))
        rows.append(dict(tau=tau, P=r.x, shift_sd=(r.x - Pstar) / SD,
                         value_gap=G_tau(r.x, tau) - V0.expected_target_per_unit(Pstar, A0, SD, F_TRADER, "A")))
    for r in rows:
        k = {1.0: "a", 0.2: "b", 0.02: "c", 0.001: "d"}[r["tau"]]
        text = (fnum(r["shift_sd"], 3, sign=True) if abs(r["shift_sd"]) >= 1e-3
                else fsci(r["shift_sd"], 1))
        N.put(f"relax.shift{k}", r["shift_sd"], text,
              f"argmax of the relaxed one-shot payoff minus P*, in sd, tau = {r['tau']}")
        N.put(f"relax.gain{k}", r["value_gap"], fsci(r["value_gap"], 1),
              f"relaxed optimum value minus the exact optimum value, tau = {r['tau']}")
    D["relaxation"] = rows


def model_relax_curves(N, D):
    """Curves for the acceptance-jump figure: the hard and relaxed one-shot payoff."""
    from scipy.integrate import quad
    from scipy.stats import norm
    import v0_game as V0

    def G_tau(P, tau):
        if tau == 0:
            return V0.expected_target_per_unit(P, A0, SD, F_TRADER, "A")
        w = tau * SD

        def integrand(x):
            s_ = 0.5 * (1 + math.tanh((x - P) / (2 * w)))
            return (s_ * P + (1 - s_) * (1 - F_TRADER) * x) * norm.pdf(x, A0, SD)
        lo, hi = A0 - 10 * SD, A0 + 10 * SD
        pts = [q for q in (P - 20 * w, P, P + 20 * w) if lo < q < hi]
        return quad(integrand, lo, hi, points=pts, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
    z = np.linspace(-1.0, 3.0, 121)
    curves = {}
    for tau in (0.0, 1.0, 0.2):
        curves[str(tau)] = [G_tau(A0 + SD * zz, tau) for zz in z]
    D["relax_curves"] = dict(z=z.tolist(), G=curves,
                             floor=(1 - F_TRADER) * A0,
                             Pstar=V0.closed_form_optimal_rate(A0, SD, F_TRADER, "A"))


def model_interp(N, D):
    """Linear interpolation of kappa on the DP's anchor grid, for the bias figure."""
    import inspect
    import v2_multiple_rounds as V2
    d = {k: v.default for k, v in inspect.signature(V2.Grids.__init__).parameters.items()
         if v.default is not inspect.Parameter.empty}
    half, n_a = d["a_halfwidth_sds"], d["n_a"]
    a = np.linspace(A0 - 4 * SD, A0 + 4 * SD, 801)
    kap = np.array([V2.kappa(None, x, SD) for x in a])
    out = dict(a=a.tolist(), err={})
    for n in (n_a, 2 * n_a - 1):
        nodes = A0 + SD * np.linspace(-half, half, n)
        kn = np.array([V2.kappa(None, x, SD) for x in nodes])
        out["err"][str(n)] = ((np.interp(a, nodes, kn) - kap) / kap).tolist()
        out.setdefault("step", {})[str(n)] = 2 * half / (n - 1)
    e1, e2 = (max(out["err"][str(n)]) for n in (n_a, 2 * n_a - 1))
    N.put("interp.maxerr", e1, fpct_small(e1), "max relative error of linear interpolation of kappa, default grid")
    N.put("interp.ratio", e1 / e2, fnum(e1 / e2, 1), "ratio of max errors, default grid vs half step")
    D["interp"] = out


def model_kappa(N):
    import v2_multiple_rounds as V2
    k1 = V2.kappa(None, A0, SD)
    k4 = V2.kappa(None, A0, 2 * SD)          # four steps: sd * sqrt(4)
    N.put("kappa.one", k1, fnum(k1, 6), "v2_multiple_rounds.kappa(a0, sd)")
    N.put("kappa.onebonus", k1 * A0 - 1, fpct(k1 * A0 - 1, 2), "a0 * kappa - 1")
    N.put("kappa.four", k4, fnum(k4, 6), "v2_multiple_rounds.kappa(a0, 2 sd)")
    N.put("kappa.fourbonus", k4 * A0 - 1, fpct(k4 * A0 - 1, 2), "a0 * kappa4 - 1")
    N.put("kappa.recip", 1 / A0, fnum(1 / A0, 6), "1 / a0")
    N.put("kappa.slopeshort", (1 + 0.03) * k1, fnum((1 + 0.03) * k1, 4), "(1 + B) kappa, T1")
    # Siegel: log-normal alternative, one and four steps
    s_ln = SD / A0
    N.put("siegel.lnone", math.exp(s_ln ** 2) - 1, fpct(math.exp(s_ln ** 2) - 1, 2), "exp(s^2) - 1, s = sd/a0")
    # the Bureau floor if the seller converts in round n (T1), by quadrature
    from scipy.integrate import quad
    from scipy.stats import norm
    L, T, A, B = 100_000.0, 125_000.0, 0.02, 0.03
    floors = []
    for n in (1, 2, 3, 4):
        s_n, s_rest = SD * math.sqrt(n), SD * math.sqrt(5 - n)

        def w(x):
            d = L * (1 - F_TRADER) * x
            pay = d - B * max(T - d, 0.0)
            return pay * V2.kappa(None, x, s_rest) * norm.pdf(x, A0, s_n)
        EW = quad(w, A0 - 8 * s_n, A0 + 8 * s_n, limit=200)[0]
        floors.append((EW - L) / L)
        N.put(f"floor.round{n}", floors[-1], fpct(floors[-1], 2),
              f"T1 converts everything at the Bureau in round {n}, quadrature")
    per = (floors[0] - floors[-1]) / 3
    N.put("floor.perround", per, fnum(100 * per, 2), "mean cost of a round's delay, points")
    L, T, A, B = 200_000.0, 155_000.0, 0.03, 0.02          # T2, dollar seller
    floorsB = []
    for n in (1, 2, 3, 4):
        s_n = SD * math.sqrt(n)

        def wB(x):
            d = L * (1 - F_TRADER) / x                         # pounds bought at the Bureau
            return (d - B * max(T - d, 0.0)) * x * norm.pdf(x, A0, s_n)   # E[a5 | x] = x
        EW = quad(wB, A0 - 8 * s_n, A0 + 8 * s_n, limit=200)[0]
        floorsB.append((EW - L) / L)
        N.put(f"floorB.round{n}", floorsB[-1], fpct(floorsB[-1], 3),
              f"T2 converts everything at the Bureau in round {n}, quadrature")
    perB = (floorsB[0] - floorsB[-1]) / 3
    N.put("floorB.perround", perB, fnum(100 * perB, 3), "T2: mean cost of a round's delay, points")
    # where leaving a unit unconverted beats the Bureau, short of the target: a + b < f (1 + b)
    for name, (a, b) in (("T1", (0.02, 0.03)), ("T2", (0.03, 0.02))):
        N.put(f"left.f{name}", (a + b) / (1 + b), fpct((a + b) / (1 + b), 2),
              f"{name}: trader fee above which an unconverted unit costs less than the Bureau")


def model_negative(N, D):
    """Avi's box-out: how often deleting non-positive paths bites."""
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    rows = []
    for sd in (0.05, 0.10, 0.20, 0.30, 0.40):
        term = [norm.cdf(-A0 / (sd * math.sqrt(n))) for n in range(1, 6)]
        union = float(sum(term))
        if sd >= 0.2:
            rates = A0 + sd * np.cumsum(rng.standard_normal((1_000_000, 5)), axis=1)
            ever = float((rates <= 0).any(axis=1).mean())
        else:
            ever = None
        rows.append(dict(sd=sd, last=term[-1], union=union, ever=ever))
    D["negative"] = rows
    N.put("neg.sd05", 2 * rows[0]["last"], fsci(2 * rows[0]["last"], 0),
          "Levy's inequality: Pr(some rate <= 0 in 5 draws) <= 2 Pr(a5 <= 0), sd 0.05")
    N.put("neg.sd10", rows[1]["union"], fsci(rows[1]["union"], 0), "same, sd 0.10")
    N.put("neg.cut8", 2 * norm.sf(8.0), fsci(2 * norm.sf(8.0), 0), "probability a step lies beyond 8 sd")
    N.put("neg.sd40", rows[4]["ever"], fpct(rows[4]["ever"], 1), "Monte Carlo, 1e6 paths, sd 0.40")
    t1000 = norm.cdf(-A0 / (SD * math.sqrt(1000)))
    N.put("neg.t1000", t1000, fpct(t1000, 0), "Pr(a_1000 <= 0), sd 0.05")
    N.put("neg.e1000", min(1.0, 2 * t1000), fpct(2 * t1000, 0),
          "reflection: Pr(the walk touches 0 by round 1000)")


def model_sigma_sweep(N, D):
    from scipy.stats import norm
    import v0_game as V0
    sds = np.round(np.arange(0.005, 0.1201, 0.0025), 4)
    rows = []
    for sd in sds:
        row = dict(sd=float(sd))
        for side in ("A", "B"):
            P = V0.closed_form_optimal_rate(A0, sd, F_TRADER, side)
            inter = V0.offer_is_interior(P, A0, sd, F_TRADER, side)
            z = (P - A0) / sd
            fill = 1 - norm.cdf(z) if side == "A" else norm.cdf(z)
            val = V0.expected_target_per_unit(P, A0, sd, F_TRADER, side)
            pinf = V0.perfect_information_value(A0, sd, side)
            share = (val - (1 - F_TRADER) * pinf) / (F_TRADER * pinf)
            beyond = z if side == "A" else -z          # how far out, in the trader's favour
            row[side] = dict(z=beyond if inter else None, fill=fill if inter else 0.0,
                             share=share if inter else 0.0, interior=bool(inter))
        rows.append(row)
    D["sigma_sweep"] = rows
    dead = [r["sd"] for r in rows if not r["B"]["interior"]]
    N.put("sweep.Bdead", min(dead) if dead else float("nan"),
          fnum(min(dead), 4) if dead else "none",
          "smallest sd on the grid where the buyer has no interior optimum")
    lo, hi = rows[0], rows[-1]
    N.put("sweep.sdlo", lo["sd"], fnum(lo["sd"], 3), "grid start")
    N.put("sweep.sdhi", hi["sd"], fnum(hi["sd"], 3), "grid end")


def model_sigma_star(N, D):
    """Where the dollar seller's one-offer channel closes (g = 0, f = 2%)."""
    import fee_aware_v0 as FA
    def interior(sd):
        X = FA.closed_form_optimal_rate(A0, sd, F_TRADER, "B", 0.0)
        return bool(FA.offer_is_interior(X, A0, sd, F_TRADER, "B", 0.0))
    lo, hi = 0.05, 0.12
    assert interior(lo) and not interior(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if interior(mid) else (lo, mid)
    approx = A0 * math.sqrt(F_TRADER) / 2
    N.put("sigma.Bstar", lo, fnum(lo, 3), "largest sd with an interior dollar-seller offer, bisection")
    N.put("sigma.Bapprox", approx, fnum(approx, 3), "a0 sqrt(f) / 2")


def model_carry_mm(N, D):
    """One blind dollar-seller offer in round 1 against a market maker that
    carries the dollars to round 5 instead of flattening: it accepts P when
    (1-g) kappa_4(X) > 1/P, kappa_4(x) = E[1/a5 | a1 = x] over four more steps."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar, brentq
    from scipy.stats import norm
    import v2_multiple_rounds as V2
    import fee_aware_v0 as FA
    sd4 = 2 * SD
    lo_x, hi_x = A0 - 8 * SD, A0 + 8 * SD
    xs = np.linspace(lo_x, hi_x, 801)
    kap = np.array([V2.kappa(None, x, sd4) for x in xs])
    kap_at = lambda x: float(np.interp(x, xs, kap))
    tail = lambda x, f: quad(lambda y: norm.pdf(y, A0, SD) / y, x, hi_x, limit=200)[0] * (1 - f)
    out = {}
    for (f, g) in ((0.02, 0.0), (0.02, 0.01), (0.03, 0.01)):
        def carry_val(x):          # threshold x: filled iff X < x, paid (1-g) kappa_4(x) per dollar
            return (1 - g) * kap_at(x) * norm.cdf((x - A0) / SD) + tail(x, f)
        def flat_val(x):           # flatten rule: filled iff X < x, paid (1-g)/x per dollar
            return (1 - g) / x * norm.cdf((x - A0) / SD) + tail(x, f)
        floor = tail(lo_x, f)
        res = {}
        for name, fn in (("carry", carry_val), ("flat", flat_val)):
            r = minimize_scalar(lambda x: -fn(x), bounds=(lo_x, A0 + 2 * SD), method="bounded",
                                options=dict(xatol=1e-10))
            fill = norm.cdf((r.x - A0) / SD)
            res[name] = dict(z=(r.x - A0) / SD, fill=fill, edge=fn(r.x) - floor)
        pinf = quad(lambda y: norm.pdf(y, A0, SD) / y, lo_x, hi_x, limit=200)[0]
        out[(f, g)] = res
        t = f"carrymm.f{round(100 * f)}g{round(100 * g)}"
        for name in ("carry", "flat"):
            N.put(f"{t}.{name}fill", res[name]["fill"], fpct_small(res[name]["fill"]),
                  f"one round-1 dollar-seller offer, {name} market maker, fill probability")
            share = res[name]["edge"] / (f * pinf)
            N.put(f"{t}.{name}share", share, fpct_small(share),
                  f"same, edge over the Bureau as a share of the fee f E[1/X]")
    ratio = A0 * kap_at(A0)
    N.put("carrymm.break", 1 / ((1 - G_MM) * ratio), fnum(1 / ((1 - G_MM) * ratio), 4),
          "dollar side, carry break-even price / X at X = a0, g = 1%")
    D["carry_mm"] = {f"{k[0]}_{k[1]}": v for k, v in out.items()}


def model_fee(N, D):
    """The market maker's own fee in the one-shot game."""
    from scipy.stats import norm
    import fee_aware_v0 as FA
    rhos = {}
    for (f, g) in ((0.02, 0.0), (0.02, 0.01), (0.03, 0.0), (0.03, 0.01)):
        fp = (f - g) / (1 - g)
        rhos[(f, g)] = fp / (SD / A0)
    N.put("fee.fprime21", (0.02 - 0.01) / 0.99, fpct((0.02 - 0.01) / 0.99, 2), "(f-g)/(1-g)")
    N.put("fee.fprime31", (0.03 - 0.01) / 0.99, fpct((0.03 - 0.01) / 0.99, 2), "(f-g)/(1-g)")
    N.put("fee.rho20", rhos[(0.02, 0.0)], fnum(rhos[(0.02, 0.0)], 2), "f' a0 / sd")
    N.put("fee.rho21", rhos[(0.02, 0.01)], fnum(rhos[(0.02, 0.01)], 3), "f' a0 / sd")
    N.put("fee.rho31", rhos[(0.03, 0.01)], fnum(rhos[(0.03, 0.01)], 3), "f' a0 / sd")
    N.put("fee.rho30", rhos[(0.03, 0.0)], fnum(rhos[(0.03, 0.0)], 2), "f' a0 / sd")
    # Proposition 5.1: the card of the equivalent no-fee game at g = 1%
    N.put("card.alphaA", 1 - (1 - 0.02) / (1 - G_MM), fpct(1 - (1 - 0.02) / (1 - G_MM), 2), "1 - (1-A)/(1-g), T1")
    N.put("card.alphaB", 1 - (1 - 0.03) / (1 - G_MM), fpct(1 - (1 - 0.03) / (1 - G_MM), 2), "1 - (1-A)/(1-g), T2")
    N.put("card.Tup", 1 / (1 - G_MM) - 1, fpct(1 / (1 - G_MM) - 1, 2), "1/(1-g) - 1")

    def one(f, g, sd=SD, side="A"):
        X = FA.closed_form_optimal_rate(A0, sd, f, side, g)
        inter = FA.offer_is_interior(X, A0, sd, f, side, g)
        z = (X - A0) / sd
        fill = (1 - norm.cdf(z)) if side == "A" else norm.cdf(z)
        val = FA.expected_target_per_unit(X, A0, sd, f, side, g)
        pinf = FA.perfect_information_value(A0, sd, side)
        floor = (1 - f) * pinf
        ceiling = (1 - min(f, g)) * pinf
        return dict(X=X, z=z, fill=fill if inter else 0.0, val=val, floor=floor,
                    edge=val - floor, share=(val - floor) / (ceiling - floor),
                    interior=bool(inter), quote=FA.quote_from_threshold(X, side, g))
    s21 = one(0.02, 0.01)
    N.put("fee.z21", s21["z"], fnum(s21["z"], 2), "fee_aware_v0.closed_form_optimal_rate, (2%,1%), seller")
    N.put("fee.fill21", s21["fill"], fpct(s21["fill"], 2), "same, fill probability")
    N.put("fee.quote21", s21["quote"], fnum(s21["quote"], 4), "same, quote on the slip")
    N.put("fee.edge21bp", s21["edge"] / A0, fnum(1e4 * s21["edge"] / A0, 3),
          "same, edge over the Bureau in bp of a pound")
    N.put("fee.share21", s21["share"], fpct_small(s21["share"]), "same, share of the prize")
    b21 = one(0.02, 0.01, side="B")
    N.put("fee.Binterior21", b21["interior"], "no" if not b21["interior"] else "yes",
          "fee_aware_v0.offer_is_interior, (2%,1%), buyer")
    N.put("fee.Bz21", abs(b21["z"]), fnum(abs(b21["z"]), 2), "(2%,1%) buyer threshold, sd below a0")
    N.put("fee.Bfill21", b21["fill"], fpct_small(b21["fill"]), "(2%,1%) buyer fill")
    s251 = one(0.025, 0.01)
    rho251 = ((0.025 - 0.01) / 0.99) / (SD / A0)
    N.put("fee.rho251", rho251, fnum(rho251, 2), "f' a0 / sd at (2.5%, 1%)")
    N.put("fee.fill251", s251["fill"], fpct(s251["fill"], 1), "(2.5%,1%) one-offer seller fill")
    s31 = one(0.03, 0.01)
    s20 = one(0.02, 0.0)
    N.put("fee.z31", s31["z"], fnum(s31["z"], 2), "(3%,1%) seller threshold, sd")
    N.put("fee.fill31", s31["fill"], fpct(s31["fill"], 1), "(3%,1%) seller fill")
    N.put("fee.z20", s20["z"], fnum(s20["z"], 2), "(2%,0) seller threshold, sd")
    # halving sd at (2%,1%)
    h = one(0.02, 0.01, sd=SD / 2)
    N.put("fee.fillhalfsd", h["fill"], fpct(h["fill"], 1), "(2%,1%), sd halved: fill")
    # finer rounds, same total variance over four rounds' worth of time
    finer = []
    for n in (4, 8, 16, 32, 64):
        r = one(0.02, 0.01, sd=SD * math.sqrt(4 / n))
        finer.append(dict(n=n, fill=r["fill"], share=r["share"]))
        N.put(f"fee.finer{n}", r["fill"], fpct(r["fill"], 2 if r["fill"] < 0.01 else 1),
              f"(2%,1%) one-shot fill with sd * sqrt(4/{n})")
    D["finer_rounds"] = finer
    # the three strips at the no-fee optimum quote P = 1.3222
    import v0_game as V0
    P = V0.closed_form_optimal_rate(A0, SD, F_TRADER, "A")
    for g, tag in ((0.0, "g0"), (0.01, "g1")):
        lo = P / (1 - g)                       # filled iff X > lo
        hi = P / (1 - F_TRADER)                # a fill beats the Bureau iff X < hi
        beats = max(0.0, norm.cdf((hi - A0) / SD) - norm.cdf((lo - A0) / SD))
        worse = 1 - norm.cdf((max(hi, lo) - A0) / SD)
        N.put(f"strip.beats{tag}", beats, fpct(beats, 2), f"Pr(P/(1-g) < X < P/(1-f)), g={g}")
        N.put(f"strip.worse{tag}", worse, fpct(worse, 2), f"Pr(X > P/(1-f)), g={g}")
    D["strips"] = dict(P=P, X21=s21["X"], q21=s21["quote"])
    # rho sweep: the no-fee game at f' (Corollary), sd fixed
    rows = []
    for rho in np.round(np.arange(0.05, 1.5001, 0.025), 3):
        fp = rho * SD / A0
        r = one(fp, 0.0)
        rows.append(dict(rho=float(rho), fill=r["fill"], share=r["share"], z=r["z"],
                         interior=r["interior"]))
    D["rho_sweep"] = rows
    for rho, tag in ((0.5, "a"), (0.35, "b"), (0.25, "c")):
        r = one(rho * SD / A0, 0.0)
        N.put(f"rho.fill{tag}", r["fill"], fpct_small(r["fill"]), f"one-shot fill at rho={rho}")
        N.put(f"rho.share{tag}", r["share"], fpct_small(r["share"]), f"share of the prize at rho={rho}")
    D["rho_marks"] = {"(2%, 0)": rhos[(0.02, 0.0)], "(2%, 1%)": rhos[(0.02, 0.01)],
                      "(3%, 1%)": rhos[(0.03, 0.01)], "(3%, 0)": rhos[(0.03, 0.0)]}
    # the direct-trade band (Remark 5.6) is computed in derived_numbers, from kappa.one

def model_carry(N):
    """The market maker's break-even price, flattening against carrying (n=1)."""
    k = 4 * (SD / A0) ** 2
    N.put("carry.flatA", 1 - G_MM, fnum(1 - G_MM, 4), "fee-aware cut, side A")
    N.put("carry.carryA", (1 - G_MM) / (1 + k), fnum((1 - G_MM) / (1 + k), 4), "(1-g)/(1+4 s^2), side A")
    N.put("carry.flatB", 1 / (1 - G_MM), fnum(1 / (1 - G_MM), 4), "1/(1-g), side B")
    N.put("carry.carryB", 1 / ((1 - G_MM) * (1 + k)), fnum(1 / ((1 - G_MM) * (1 + k)), 4),
          "1/((1-g)(1+4 s^2)), side B")


def model_v1(N, D):
    import v1_sequential_offers as V1
    from Mechanics.fx_mechanics import GameParams
    sol = V1.solve_v1(GameParams(), max_offers=10, side="A")
    floor, pie = (1 - F_TRADER) * A0, F_TRADER * A0
    rows = []
    for k in range(1, 11):
        rows.append(dict(K=k, value=sol.value(k), first=sol.first_offer(k),
                         share=(sol.value(k) - floor) / pie))
    D["v1"] = rows
    for r in rows:
        N.put(f"v1.share{r['K']}", r["share"], fpct(r["share"], 1),
              f"v1_sequential_offers.solve_v1, K={r['K']}")
    D["v1_ladder5"] = [float(x) for x in sol.ladder(5)]
    N.put("v1.ladder5", 0, " $\\to$ ".join(fnum(x, 4) for x in sol.ladder(5)),
          "v1 ladder, K=5, every offer rejected")


def _v2_full(side, K, g=0.0):
    """The full card game with K sized offers per round: solve, replay, facts.
    g is the market maker's own Bureau fee (flatten-rule market maker)."""
    import v2_multiple_rounds as V2
    from scipy.stats import norm
    card = dict(A=dict(L=100_000.0, T=125_000.0, A=0.02, B=0.03),
                B=dict(L=200_000.0, T=155_000.0, A=0.03, B=0.02))[side]
    spec = V2.TraderSpec(**card, side=side, K=K, mm_fee=g)
    sol = V2.solve_v2(spec)
    res = V2.simulate_paths(sol, n_paths=20_000, seed=1)
    pay = spec.L * np.array([V2.bdc_payoff_per_unit(x, spec.params.bdc_fee, side)
                             for x in res["X"][:, 0]])
    dump = np.array([V2.terminal_wealth(spec, 0.0, d, a5) for d, a5 in zip(pay, res["a5"])])
    edge = (res["W"] - dump) / spec.L
    floor = V2.bdc_baseline(spec, sol.grids)[1]
    cv = float(V2.clairvoyant_wealth(spec, res["X"], res["a5"]).mean() - spec.L) / spec.L
    first, _ = V2.greedy_offer_action(sol, 1, K, V2.NO_FLOOR, sol.grids.n_offer, spec.L, 0.0, A0)
    last_z = []
    for n in range(1, spec.rounds + 1):
        act, _ = V2.greedy_offer_action(sol, n, 1, V2.NO_FLOOR, sol.grids.n_offer, spec.L, 0.0, A0)
        last_z.append(None if act is None else float(sol.grids.z[act[0]]))
    clear1 = bureau_any = 0
    bureau_round = np.zeros(spec.rounds)
    n_trace = 5000
    for i in range(n_trace):
        tr = V2.play_game(sol, path=res["X"][i], a5=res["a5"][i])
        clear1 += tr["rounds"][0]["c"] <= 1e-6
        dumps = [r["bdc_dump"] > 1e-6 for r in tr["rounds"]]
        bureau_any += any(dumps)
        bureau_round += np.array(dumps, float)
    return dict(table=sol.pl(), floor=floor, edge=float(edge.mean()),
                edge_se=float(edge.std(ddof=1) / math.sqrt(edge.size)),
                value=floor + float(edge.mean()), clairvoyant=cv,
                first_z=None if first is None else float(sol.grids.z[first[0]]),
                first_size=None if first is None else first[1] / spec.L,
                last_z=last_z, clear_round1=clear1 / n_trace,
                bureau_any=bureau_any / n_trace,
                bureau_by_round=(bureau_round / n_trace).tolist(),
                prob_fill_first=None if first is None else float(1 - norm.cdf(sol.grids.z[first[0]])))


def _v2_one_round_sized(Kmax=4):
    import v2_multiple_rounds as V2
    spec = V2.TraderSpec(rounds=1, K=Kmax, B=0.0, terminal_mode="dollars", side="A")
    grids = V2.Grids(spec)
    _, Ulv, _ = V2.solve_round(spec, grids, np.stack(
        [V2.terminal_matrix(spec, grids, a4) for a4 in grids.a]), A0)
    vals = [float(Ulv[k][V2.NO_FLOOR + 1, grids.n_offer][grids.n_c - 1, 0] / spec.L)
            for k in range(1, Kmax + 1)]
    return dict(values=vals)


def _v2_refine(n_a):
    import v2_multiple_rounds as V2
    spec = V2.TraderSpec(K=1)
    sol = V2.solve_v2(spec, V2.Grids(spec, n_a=n_a))
    res = V2.simulate_paths(sol, n_paths=20_000, seed=1)
    pay = spec.L * (1 - F_TRADER) * res["X"][:, 0]
    dump = np.array([V2.terminal_wealth(spec, 0.0, d, a5) for d, a5 in zip(pay, res["a5"])])
    edge = (res["W"] - dump) / spec.L
    floor = V2.bdc_baseline(spec, sol.grids)[1]
    return dict(n_a=n_a, step_sd=16.0 / (n_a - 1), table=sol.pl(),
                replay=floor + float(edge.mean()),
                replay_se=float(edge.std(ddof=1) / math.sqrt(edge.size)))


def _v2_grid_check(kind, side="A"):
    """T1 at (2%,0), one offer per round: replay the base-grid policy and a
    refined-grid policy on the same 20,000 paths; return the paired difference."""
    import v2_multiple_rounds as V2
    card = dict(A=dict(L=100_000.0, T=125_000.0, A=0.02, B=0.03),
                B=dict(L=200_000.0, T=155_000.0, A=0.03, B=0.02))[side]
    spec = V2.TraderSpec(**card, side=side, K=1)
    out = {}
    for tag, kw in (("base", {}), ("fine", dict(n_a=49) if kind == "anchor" else dict(n_c=33, n_d=81))):
        sol = V2.solve_v2(spec, V2.Grids(spec, **kw))
        res = V2.simulate_paths(sol, n_paths=20_000, seed=1)
        out[tag] = np.asarray(res["W"], float) / spec.L
        out[tag + "_table"] = float(sol.pl())
    d = out["fine"] - out["base"]
    return dict(kind=kind, diff=float(d.mean()), diff_se=float(d.std(ddof=1) / math.sqrt(d.size)),
                table_base=out["base_table"], table_fine=out["fine_table"])


def _v2_gate2():
    import v2_multiple_rounds as V2
    out = {}
    for f in (0.02, 0.03):
        a, b, c = V2.gate2_mm_fee(g=0.01, f=f, print_progress=False)
        out[str(f)] = dict(same=float(a), ref=float(b), game=float(c))
    return out


def model_grids(N):
    """The dynamic programme's default grids, read from its signature."""
    import inspect
    import v2_multiple_rounds as V2
    d = {k: v.default for k, v in inspect.signature(V2.Grids.__init__).parameters.items()
         if v.default is not inspect.Parameter.empty}
    src = "v2_multiple_rounds.Grids defaults"
    N.put("grid.zmax", d["zmax"], fnum(d["zmax"], 0), src)
    N.put("grid.noffer", d["n_offer"], str(d["n_offer"]), src)
    N.put("grid.zstep", 2 * d["zmax"] / (d["n_offer"] - 1), fnum(2 * d["zmax"] / (d["n_offer"] - 1), 3), src)
    N.put("grid.quad", d["quad_per_cell"], str(d["quad_per_cell"]), src)
    N.put("grid.nc", d["n_c"], str(d["n_c"]), src)
    N.put("grid.nd", d["n_d"], str(d["n_d"]), src)
    N.put("grid.na", d["n_a"], str(d["n_a"]), src)
    N.put("grid.ahalf", d["a_halfwidth_sds"], fnum(d["a_halfwidth_sds"], 0), src)
    N.put("grid.astep", 2 * d["a_halfwidth_sds"] / (d["n_a"] - 1), fnum(2 * d["a_halfwidth_sds"] / (d["n_a"] - 1), 2), src)
    N.put("grid.nq", d["n_qfrac"], str(d["n_qfrac"]), src)
    joint = (d["n_c"] * d["n_d"]) ** 5 * d["n_a"]
    N.put("grid.joint", joint, fsci(joint, 0), src + ": (n_c n_d)^5 n_a, five agents' holdings")


def model_v2(N, D):
    full = {s: cached(f"v2_full_K3_{s}", lambda s=s: _v2_full(s, 3)) for s in ("A", "B")}
    for s, r in full.items():
        tag = "T1" if s == "A" else "T2"
        N.put(f"k3.{tag}.value", r["value"], fpct(r["value"], 2),
              "v2 K=3, floor-anchored replay (20,000 paths, seed 1)")
        N.put(f"k3.{tag}.se", 2 * r["edge_se"], fpct(2 * r["edge_se"], 2), "same, 2 s.e.")
        N.put(f"k3.{tag}.table", r["table"], fpct(r["table"], 2), "v2 K=3 table value (inflated)")
        N.put(f"k3.{tag}.edge", r["edge"], fpct(r["edge"], 2, sign=True), "v2 K=3 edge over the floor")
        N.put(f"k3.{tag}.cv", r["clairvoyant"], fpct(r["clairvoyant"], 2, sign=True), "clairvoyant, same paths")
        N.put(f"k3.{tag}.clear1", r["clear_round1"], fpct(r["clear_round1"], 1), "games clearing in round 1")
        N.put(f"k3.{tag}.bureau", r["bureau_any"], fpct(r["bureau_any"], 1), "games using the Bureau at all")
        if r["first_z"] is not None:
            N.put(f"k3.{tag}.firstz", r["first_z"], fnum(r["first_z"], 2, sign=True), "first offer, sd from a0")
            N.put(f"k3.{tag}.firstsize", r["first_size"], fpct(r["first_size"], 1), "first offer, share of the book")
        for n, z in enumerate(r["last_z"], 1):
            if z is not None:
                N.put(f"k3.{tag}.lastz{n}", z, fnum(z, 2, sign=True), f"last-offer threshold, round {n}")
    D["v2_full_K3"] = full
    one = cached("v2_one_round_sized", _v2_one_round_sized)
    floor, pie = (1 - F_TRADER) * A0, F_TRADER * A0
    for k, v in enumerate(one["values"], 1):
        N.put(f"sized.share{k}", (v - floor) / pie, fpct((v - floor) / pie, 1),
              f"one round, sized offers, K={k}")
    D["v2_one_round_sized"] = one
    ref = [cached(f"v2_refine_na{n}", lambda n=n: _v2_refine(n)) for n in (13, 25, 49)]
    for r in ref:
        gap = r["table"] - r["replay"]
        N.put(f"refine.gap{r['n_a']}", gap, fnum(100 * gap, 3, sign=True),
              f"table - replay, points, n_a={r['n_a']}")
        N.put(f"refine.step{r['n_a']}", r["step_sd"], fnum(r["step_sd"], 2), "anchor step, sd")
        N.put(f"refine.replay{r['n_a']}", r["replay"], fpct(r["replay"], 3), "floor-anchored replay")
    D["refine"] = ref
    gaps = [r["table"] - r["replay"] for r in ref]
    for tag, (a, b) in (("a", gaps[0:2]), ("b", gaps[1:3])):
        N.put(f"refine.ratio{tag}", a / b, fnum(a / b, 1), "ratio of successive table - replay gaps")
    # the offer count at the rules' fees (flatten-rule market maker, g = 1%)
    rules = {}
    for s in ("A", "B"):
        for K in (2, 3):
            r = cached(f"v2_full_K{K}_{s}_g1", lambda s=s, K=K: _v2_full(s, K, g=G_MM))
            rules[f"{s}{K}"] = r
            tag = ("T1" if s == "A" else "T2") + f".K{K}g1"
            N.put(f"k.{tag}.edge", r["edge"], fnum(100 * r["edge"], 3, sign=True),
                  f"v2 K={K}, g=1%, floor-anchored replay, edge in points (20,000 paths)")
            N.put(f"k.{tag}.edgese", 2 * r["edge_se"], fnum(200 * r["edge_se"], 3), "same, 2 s.e.")
            N.put(f"k.{tag}.value", r["value"], fpct(r["value"], 3), "same, value")
            N.put(f"k.{tag}.bureau", r["bureau_any"], fpct(r["bureau_any"], 1), "games using the Bureau")
            N.put(f"k.{tag}.clear1", r["clear_round1"], fpct(r["clear_round1"], 1), "games clearing in round 1")
    D["v2_rules_K"] = rules
    for kind, side, name in (("anchor", "A", "anchor"), ("cd", "A", "cd"), ("cd", "B", "cdB")):
        r = cached(f"v2_gridcheck_{name}", lambda kind=kind, side=side: _v2_grid_check(kind, side))
        kind = name
        N.put(f"gridchk.{kind}.diff", r["diff"], fnum(100 * r["diff"], 4, sign=True),
              "replayed value, refined-grid policy minus base-grid policy, points, paired")
        N.put(f"gridchk.{kind}.diffse", 2 * r["diff_se"], fnum(200 * r["diff_se"], 4), "same, 2 s.e.")
        same = r["diff"] == 0.0 and r["diff_se"] == 0.0
        N.put(f"gridchk.{kind}.same", float(same), "every" if same else "not every",
              "refined and base policies give the same final wealth on every path (zero mean, zero s.d.)")
        N.put(f"gridchk.{kind}.tablegap", r["table_fine"] - r["table_base"],
              fnum(100 * (r["table_fine"] - r["table_base"]), 3, sign=True), "table value, fine minus base, points")
    g2 = cached("v2_gate2", _v2_gate2)
    g2 = {k: v for k, v in g2.items() if not k.startswith("_")}
    worst = max(v["game"] for v in g2.values())
    N.put("gate2.game", worst, fsci(worst, 0), "v2 gate2_mm_fee, part (d), worst table gap / L")
    N.put("gate2.ref", max(v["ref"] for v in g2.values()),
          fsci(max(v["ref"] for v in g2.values()), 0), "gate 2 (b),(c) worst gap")
    N.put("gate2.same", max(v["same"] for v in g2.values()),
          fsci(max(v["same"] for v in g2.values()), 0), "gate 2 (a) worst gap")


MODEL_PARTS = [model_params, model_v0, model_relaxation, model_relax_curves, model_kappa, model_negative,
               model_sigma_sweep, model_sigma_star, model_carry_mm, model_fee, model_carry, model_v1, model_grids, model_interp, model_v2]


def run_model(N, D):
    for fn in MODEL_PARTS:
        t0 = time.time()
        try:
            if fn.__code__.co_argcount == 2:
                fn(N, D)
            else:
                fn(N)
            print(f"  model  {fn.__name__:<22} {time.time() - t0:6.1f}s", flush=True)
        except Exception:
            print(f"  model  {fn.__name__:<22} FAILED", flush=True)
            traceback.print_exc()


# =============================================================================
# 3. numbers from the saved runs (numpy only; the files are under Results/)
# =============================================================================
CM = RESULTS / "cotrain_multi"
LADDER = (0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07)        # traders' Bureau fee


def ladder_folder(f, arm="bid", tag=""):
    return CM / f"T1-T2-T3-T4_mm0.01_bdc{f:g}_{arm}_size_2way{tag}"


def spread_label(f):
    return f"{100 * (f - G_MM):g}\\%"


def cached_run(name, fn):
    """run-derived numbers are cached too, so a re-run only re-reads what is missing"""
    return cached(f"runs_{name}", fn)


def _read(folder):
    import redo_numbers as RN
    r = RN.read_run(folder)
    return json.loads(json.dumps(r, default=float))


def runs_cards(N, D):
    z = np.load(ladder_folder(0.03) / "cotrain_multi_eval.npz", allow_pickle=False)
    names = [str(s) for s in z["traders"]]
    sides = [str(s) for s in z["sides"]]
    cards = np.asarray(z["cards"], float)
    rows = []
    for name, side, (L, T, A, B) in zip(names, sides, cards):
        fair = T / (L * A0) if side == "A" else T / (L / A0)
        rows.append(dict(name=name, side=side, L=L, T=T, A=A, B=B, fair=fair))
    D["cards"] = rows
    cap = float(z["capital_pounds"])
    N.put("cards.capital", cap, fnum(cap, 0, comma=True), "cotrain_multi_eval.npz: capital_pounds")
    pound = sum(r["L"] for r in rows if r["side"] == "A")
    N.put("cards.poundshare", pound / cap, fpct(pound / cap, 0), "pound sellers' start / all capital")
    N.put("cards.dollarshare", 1 - pound / cap, fpct(1 - pound / cap, 0), "dollar sellers' share")
    hp = dict(n_outer=int(z["n_outer"]), n_inner=int(z["n_inner"]), batch=int(z["batch"]),
              lr_trader=float(z["lr"][0]), lr_mm=float(z["lr"][1]),
              tau_trader=np.asarray(z["tau_trader"], float).tolist(),
              tau_mm=np.asarray(z["tau_mm"], float).tolist(), anneal=float(z["tau_anneal_frac"]),
              mm_arch=str(z["mm_arch"]), first=str(z["first"]))
    D["hyper"] = hp
    src = "cotrain_multi_eval.npz of the 2% base run"
    N.put("hp.outer", hp["n_outer"], str(hp["n_outer"]), src)
    N.put("hp.inner", hp["n_inner"], str(hp["n_inner"]), src)
    N.put("hp.batch", hp["batch"], f"{hp['batch']:,}", src)
    N.put("hp.lrtrader", hp["lr_trader"], fsci(hp["lr_trader"], 0), src)
    N.put("hp.lrmm", hp["lr_mm"], fsci(hp["lr_mm"], 0), src)
    N.put("hp.tautrader0", hp["tau_trader"][0], fnum(hp["tau_trader"][0], 1), src)
    N.put("hp.tautrader1", hp["tau_trader"][1], fnum(hp["tau_trader"][1], 2), src)
    N.put("hp.taumm0", hp["tau_mm"][0], fnum(hp["tau_mm"][0], 1), src)
    N.put("hp.taumm1", hp["tau_mm"][1], fnum(hp["tau_mm"][1], 2), src)
    N.put("hp.anneal", hp["anneal"], fpct(hp["anneal"], 0), src)
    N.put("hp.steps", 2 * hp["n_outer"] * hp["n_inner"], f"{2 * hp['n_outer'] * hp['n_inner']:,}", src + " total steps, both agents")
    N.put("hp.games", hp["n_outer"] * hp["n_inner"] * hp["batch"], f"{hp['n_outer'] * hp['n_inner'] * hp['batch']:,}", src + " games per agent")
    # if every offer filled, the most the market maker could match is twice the smaller side
    N.put("cards.nettable", 2 * pound / cap, fpct(2 * pound / cap, 0), "2 x pound side / capital")


def runs_fee_dp(N, D):
    path = RESULTS / "fee_aware_dp" / "report.json"
    rep = json.loads(path.read_text())
    D["fee_dp"] = rep
    src = "Results/fee_aware_dp/report.json (Experiments/fee_aware_dp.py)"
    for key, r in rep.items():
        side, f, g = key.split()
        f, g = float(f[2:]), float(g[2:])
        t = f"dp.{side}.f{round(100 * f)}g{round(100 * g)}"
        N.put(t + ".floor", r["floor"], fpct(r["floor"], 3), src)
        N.put(t + ".value", r["dp_value"], fpct(r["dp_value"], 3), src)
        N.put(t + ".table", r["table"], fpct(r["table"], 3), src)
        N.put(t + ".edge", r["dp_edge"], fnum(100 * r["dp_edge"], 4, sign=True), src + " (points)")
        N.put(t + ".edgese", 2 * r["dp_edge_se"], fnum(200 * r["dp_edge_se"], 4), src)
        N.put(t + ".net", r["net_edge"], fnum(100 * r["net_edge"], 4, sign=True), src)
        N.put(t + ".netse", 2 * r["net_edge_se"], fnum(200 * r["net_edge_se"], 4), src)
        N.put(t + ".reg", r["dp_minus_net"], fnum(100 * r["dp_minus_net"], 4, sign=True), src)
        N.put(t + ".regse", 2 * r["dp_minus_net_se"], fnum(200 * r["dp_minus_net_se"], 4), src)
        N.put(t + ".cv", r["clairvoyant"], fpct(r["clairvoyant"], 2, sign=True), src)
        N.put(t + ".gap", -r["gate3_gap"], fnum(-100 * r["gate3_gap"], 3, sign=True), src + " table - replay")
        N.put(t + ".gapfa", r["table"] - r["dp_value"], fnum(100 * (r["table"] - r["dp_value"]), 3, sign=True),
              src + " table - floor-anchored replay, points")
        if r.get("r1_dp"):
            N.put(t + ".r1z", r["r1_dp"]["z"], fnum(r["r1_dp"]["z"], 2, sign=True), src)
            N.put(t + ".r1fill", r["r1_dp"]["fill"], fpct_small(r["r1_dp"]["fill"]), src)
        if r.get("r1_net"):
            N.put(t + ".r1znet", r["r1_net"]["z"], fnum(r["r1_net"]["z"], 2, sign=True), src)
            N.put(t + ".r1fillnet", r["r1_net"]["fill"], fpct_small(r["r1_net"]["fill"]), src)
        fl = r["flows"]
        tomm = sum(fl["to_mm"])
        N.put(t + ".tomm", tomm, fpct(tomm, 2) if tomm >= 1e-3 else fpct_small(tomm), src + " to the MM, all rounds")
        for n in range(4):
            N.put(t + f".bdc{n + 1}", fl["to_bureau"][n], fpct(fl["to_bureau"][n], 1), src)
            N.put(t + f".mm{n + 1}", fl["to_mm"][n], fpct(fl["to_mm"][n], 2), src)
        N.put(t + ".left", fl["left_at_end"], fpct(fl["left_at_end"], 2), src)


def runs_fee_dp_summary(N, D):
    rep = D.get("fee_dp", {})
    src = "Results/fee_aware_dp/report.json, derived"
    a = [abs(rep[f"A f={f:g} g={g:g}"]["dp_minus_net"]) for f, g in ((0.02, 0.0), (0.02, 0.01), (0.03, 0.0), (0.03, 0.01))]
    N.put("dp.A.maxreg", max(a), fnum(100 * max(a), 3), src + " max |DP - network|, pound seller, points")
    alive = [rep[f"B f={f:g} g={g:g}"]["dp_minus_net"] for f, g in ((0.02, 0.0), (0.03, 0.0), (0.03, 0.01))]
    N.put("dp.B.regmin", min(alive), fnum(100 * min(alive), 3), src + " dollar seller, channel alive")
    N.put("dp.B.regmax", max(alive), fnum(100 * max(alive), 3), src)


def runs_mm_phase1(N, D):
    """The learned market maker against a hard-coded trader, one card, one side."""
    out = {}
    for side in ("A", "B"):
        for g in (0.0, 0.01):
            folder = RESULTS / "mm_phase1" / f"mm_{side}_fee{g:g}_TFA"
            f = folder / "mm_eval.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            mm = np.asarray(z["eval_mm_pl"], float)
            rules = [str(s) for s in z["rule_names"]]
            rule_pl = np.asarray(z["eval_rule_mm_pl"], float)
            best = int(np.argmax(rule_pl.mean(1)))
            diff = mm - rule_pl[best]
            q = np.asarray(z["record_quoted_price"], float)
            amt = np.asarray(z["record_offered_amount"], float)
            be = np.asarray(z["record_break_even"], float)
            w = np.asarray(z["record_fill_weight"], float)
            made = amt > 0
            rule_accept = (q < be) if side == "A" else (q > be)
            net_accept = w > 0.5
            dis = made & (rule_accept != net_accept)
            dist = np.abs(q / be - 1.0)[dis]
            repay = np.asarray(z["record_repay_fraction"], float)
            out[f"{side}{g:g}"] = dict(
                mm=float(mm.mean()), mm_se=float(mm.std() / math.sqrt(mm.size)),
                rules=rules, rule_pl=rule_pl.mean(1).tolist(), best=rules[best],
                diff=float(diff.mean()), diff_se=float(diff.std() / math.sqrt(diff.size)),
                offers=int(made.sum()), disagree=int(dis.sum()),
                max_dist=float(dist.max()) if dist.size else 0.0,
                repay=repay.mean(0).tolist())
    D["mm_phase1"] = out
    src = "Results/mm_phase1/mm_{side}_fee{g}_TFA/mm_eval.npz"
    if out:
        short = max(0.0, -min(r["diff"] for r in out.values()))
        N.put("mmn.maxshort", short, fnum(1e4 * short, 3), src + " largest shortfall of the network against the best rule, bp")
    for k, r in out.items():
        t = f"mmn.{k.replace('.', '')}"
        N.put(t + ".mm", r["mm"], fnum(1e4 * r["mm"], 2, sign=True), src + " MM P/L, bp")
        N.put(t + ".best", r["best"], r["best"].replace("+", " + "), src + " best reference rule")
        N.put(t + ".bestpl", max(r["rule_pl"]), fnum(1e4 * max(r["rule_pl"]), 2, sign=True), src)
        N.put(t + ".diff", r["diff"], fnum(1e4 * r["diff"], 2, sign=True), src + " network - best rule, bp")
        N.put(t + ".diffse", 2 * r["diff_se"], fnum(2e4 * r["diff_se"], 2), src)
        N.put(t + ".offers", r["offers"], f"{r['offers']:,}", src + " offers on the recorded games")
        N.put(t + ".disagree", r["disagree"], f"{r['disagree']:,}", src)
        N.put(t + ".maxdist", r["max_dist"], fsci(r["max_dist"], 1) if r["max_dist"] > 0 else "0", src)
        N.put(t + ".repay1", r["repay"][0], fnum(r["repay"][0], 2), src + " round-1 repay fraction")
        N.put(t + ".repay4", r["repay"][-1], fnum(r["repay"][-1], 2), src + " round-4 repay fraction")
        for name, pl in zip(r["rules"], r["rule_pl"]):
            key = name.replace("fee-aware", "fa").replace("naive", "nv").replace("+", "")
            N.put(t + "." + key, pl, fnum(1e4 * pl, 2, sign=True), src + f" rule {name}, bp")


def runs_cotrain(N, D):
    """One trader and one market maker, co-trained from fresh networks."""
    out = {}
    for side in ("A", "B"):
        for tag in ("fee0", "fee0.01", "fee0.01_BdC_FEE0.03"):
            f = RESULTS / "cotrain" / f"{side}_{tag}" / "cotrain_eval.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            mm = np.asarray(z["eval_mm_pl"], float)[0]
            tr = np.asarray(z["eval_trader_pl"], float)[0]
            bdc = np.asarray(z["eval_bdc_pl"], float)
            edge = tr - bdc
            out[f"{side}_{tag}"] = dict(
                f=float(z["trader_bdc_fee"]), g=float(z["fee"]),
                mm=float(mm.mean()), mm_se=float(mm.std() / math.sqrt(mm.size)),
                edge=float(edge.mean()), edge_se=float(edge.std() / math.sqrt(edge.size)),
                filled=float(z["filled"]), agree=float(z["agree"]),
                benchmark=str(z["benchmark_rule"]),
                threshold=np.asarray(z["threshold_by_round"], float).tolist())
    D["cotrain"] = out
    src = "Results/cotrain/{side}_{tag}/cotrain_eval.npz"
    for k, r in out.items():
        t = "co." + k.replace("_BdC_FEE0.03", "f3").replace("fee0.01", "g1").replace("fee0", "g0").replace("_", "")
        N.put(t + ".mm", r["mm"], fnum(1e4 * r["mm"], 2, sign=True), src + " MM P/L, bp")
        N.put(t + ".edge", r["edge"], fnum(100 * r["edge"], 4, sign=True), src + " trader - always-Bureau, points")
        N.put(t + ".edgese", 2 * r["edge_se"], fnum(200 * r["edge_se"], 4), src)
        N.put(t + ".filled", r["filled"], fpct_small(r["filled"]), src + " share of offered volume filled")
        N.put(t + ".agree", r["agree"], fpct(r["agree"], 1), src + " agreement with the benchmark rule")
        thr = [x for x in r["threshold"] if np.isfinite(x)]
        if thr:
            N.put(t + ".thr1", thr[0], fnum(thr[0], 4), src + " fitted acceptance cut P/X, round 1")


def runs_learning(N, D):
    """Block-by-block learning curves saved by the co-training scripts:
    each trader's paired edge over always-Bureau on the validation games, and
    the market maker's P/L against the hardcoded rules on the same games."""
    out = {"one": {}, "four": None}
    for side in ("A", "B"):
        for tag in ("fee0", "fee0.01", "fee0.01_BdC_FEE0.03"):
            f = RESULTS / "cotrain" / f"{side}_{tag}" / "cotrain_curve.npz"
            if f.exists():
                z = np.load(f, allow_pickle=True)
                out["one"][f"{side}_{tag}"] = dict(
                    edge=np.asarray(z["edge"], float).ravel().tolist(),
                    edge_2se=np.asarray(z["edge_2se"], float).ravel().tolist(),
                    trader_pl=np.asarray(z["trader_pl"], float).ravel().tolist(),
                    bdc_pl=np.asarray(z["bdc_pl"], float).ravel().tolist(),
                    mm_pl=np.asarray(z["mm_pl"], float).ravel().tolist())
    f = ladder_folder(0.03) / "cotrain_multi_curve.npz"
    z = np.load(f, allow_pickle=True)
    edge = np.asarray(z["edge"], float)
    mm = np.asarray(z["mm_pl"], float)
    rule = np.asarray(z["rule_pl"], float)
    names = [str(s) for s in np.asarray(z["trader_names"]).reshape(-1)[:edge.shape[1]]]
    out["four"] = dict(edge=edge.tolist(), names=names,
                       mm=mm.tolist(), best_rule=rule.max(1).tolist(),
                       trader_pl=np.asarray(z["trader_pl"], float).tolist(),
                       edge_2se=np.asarray(z["edge_2se"], float).tolist(),
                       mm_2se=np.asarray(z["mm_2se"], float).tolist(),
                       bdc_pl=np.asarray(z["bdc_pl"], float).tolist())
    D["learning"] = out
    src = "cotrain_curve.npz / cotrain_multi_curve.npz, validation games"
    last = 10                                  # the last 10 blocks = the last 5 outer steps
    for key, r in out["one"].items():
        e = np.asarray(r["edge"][-last:])
        t = "learn." + key.replace("_BdC_FEE0.03", "f3").replace("fee0.01", "g1").replace("fee0", "g0").replace("_", "")
        N.put(t + ".lo", e.min(), fnum(100 * e.min(), 3, sign=True), src + ", min edge over the last 10 blocks, points")
        N.put(t + ".hi", e.max(), fnum(100 * e.max(), 3, sign=True), src + ", max edge over the last 10 blocks, points")
        N.put(t + ".mean", e.mean(), fnum(100 * e.mean(), 3, sign=True), src + ", mean edge over the last 10 blocks")
    for i, name in enumerate(out["four"]["names"]):
        e = edge[-last:, i]
        N.put(f"learn4.{name}.lo", e.min(), fnum(100 * e.min(), 3, sign=True), src + " (4 traders, 2%)")
        N.put(f"learn4.{name}.hi", e.max(), fnum(100 * e.max(), 3, sign=True), src)
        N.put(f"learn4.{name}.mean", e.mean(), fnum(100 * e.mean(), 3, sign=True), src)
    gap = mm - rule.max(1)
    start = 10
    N.put("learn4.mmgap", np.abs(gap[start:]).max(), fnum(1e4 * np.abs(gap[start:]).max(), 1),
          src + ", max |MM - best hardcoded rule| from block 10, bp")
    N.put("learn4.mmlo", mm[-last:].min(), fnum(1e4 * mm[-last:].min(), 0, sign=True), src + ", MM bp, last 10 blocks")
    N.put("learn4.mmhi", mm[-last:].max(), fnum(1e4 * mm[-last:].max(), 0, sign=True), src)
    N.put("learn4.nblocks", len(mm) - 1, str(len(mm) - 1), "blocks of training")


def runs_ppo(N, D):
    out = {}
    for tag in ("ppo_A_R1", "ppo_A_R4"):
        f = RESULTS / tag / "summary.txt"
        if not f.exists():
            continue
        txt = f.read_text()
        fee = float(re.search(r"BdC fee\s+([\d.]+)%", txt).group(1)) / 100
        ppo = float(re.search(r"\n\s+PPO\s+([-+\d.]+)%", txt).group(1)) / 100
        dp = float(re.search(r"DP optimum \(replayed\)\s+([-+\d.]+)%", txt).group(1)) / 100
        floor = float(re.search(r"always-BdC\s+([-+\d.]+)%", txt).group(1)) / 100
        prize = float(re.search(r"DP - always-BdC \(prize\)\s+([-+\d.]+)%", txt).group(1)) / 100
        rounds = int(re.search(r"rounds (\d+)", txt).group(1))
        out[tag] = dict(fee=fee, ppo=ppo, dp=dp, floor=floor, prize=prize, rounds=rounds)
    D["ppo"] = out
    for tag, r in out.items():
        t = "ppo." + tag.split("_")[-1]
        src = f"Results/{tag}/summary.txt (rl_diagnostics.report)"
        N.put(t + ".fee", r["fee"], fpct(r["fee"], 0), src)
        N.put(t + ".ppo", r["ppo"], fpct(r["ppo"], 2), src)
        N.put(t + ".floor", r["floor"], fpct(r["floor"], 2), src)
        N.put(t + ".prize", r["prize"], fnum(100 * r["prize"], 2, sign=True), src + " DP - always-Bureau, points")


def runs_ladder(N, D):
    rows = {}
    for f in LADDER:
        folder = ladder_folder(f)
        if (folder / "cotrain_multi_eval.npz").exists():
            rows[f"{f:g}"] = cached_run(f"ladder_{f:g}", lambda folder=folder: _read(folder))
    D["ladder"] = rows
    src = "redo_numbers.read_run on Results/cotrain_multi/T1-T2-T3-T4_mm0.01_bdc{f}_bid_size_2way"
    for fs, r in rows.items():
        f = float(fs)
        t = f"lad.s{round(100 * (f - G_MM) * 10)}"          # s10 = 1%, s15 = 1.5%, ...
        N.put(t + ".trades", r["mm_trades"], fpct(r["mm_trades"], 1), src)
        N.put(t + ".mmbdc", r["mm_bureau"], fpct(r["mm_bureau"], 1), src)
        N.put(t + ".netted", r["netted"], fpct(r["netted"], 1), src)
        N.put(t + ".cards", r["cards_bureau"], fpct(r["cards_bureau"], 1), src)
        N.put(t + ".left", r["left_over"], fpct(r["left_over"], 1), src)
        N.put(t + ".allbdc", r["total_bureau"], fpct(r["total_bureau"], 1), src)
        N.put(t + ".anyfill", r["any_fill"], fpct_small(r["any_fill"]), src + " games with any fill")
        N.put(t + ".both", r["both_sides_same_round"], fpct(r["both_sides_same_round"], 1), src)
        N.put(t + ".matchable", r["matchable"], fpct(r["matchable"], 1), src)
        share = r["netted"] / r["mm_trades"] if r["mm_trades"] > 1e-9 else 0.0
        N.put(t + ".netshare", share, fpct(share, 0), "netted / MM trades")
        bshare = r["mm_bureau"] / r["mm_trades"] if r["mm_trades"] > 1e-9 else float("nan")
        N.put(t + ".bshare", bshare, fpct(bshare, 1) if r["mm_trades"] > 1e-9 else "--",
              "MM to the Bureau / MM trades")
        N.put(t + ".mmpl", r["pl"]["mm_pl"], fnum(1e4 * r["pl"]["mm_pl"], 1, sign=True), src + " MM P/L, bp")
        N.put(t + ".mmplse", 2 * r["pl"]["mm_pl_se"], fnum(2e4 * r["pl"]["mm_pl_se"], 1), src)
        for name, c in r["pl"]["cards"].items():
            N.put(t + f".gain{name}", c["gain"], fnum(1e4 * c["gain"], 1, sign=True), src + f" {name} - Bureau-only, bp")
            N.put(t + f".gainse{name}", 2 * c["gain_se"], fnum(2e4 * c["gain_se"], 1), src)
        for name, c in r["per_card"].items():
            N.put(t + f".r1size{name}", c["r1_amount_mean"] / c["L"], fpct(c["r1_amount_mean"] / c["L"], 0),
                  src + f" {name} round-1 offer / start")
            N.put(t + f".tomm{name}", c["to_mm"], fpct(c["to_mm"], 1), src + f" {name} sold to the MM")
            N.put(t + f".left{name}", c["left"], fpct(c["left"], 1), src + f" {name} left unconverted, share of its start")
            N.put(t + f".r1fill{name}", c["r1_filled"], fpct(c["r1_filled"], 0), src + f" {name} round-1 fill rate")


def runs_arms(N, D):
    rows = {}
    for f in (0.02, 0.03):
        for arm in ("bid", "turn", "round"):
            folder = ladder_folder(f, arm)
            if (folder / "cotrain_multi_eval.npz").exists():
                rows[f"{arm}{f:g}"] = cached_run(f"arm_{arm}_{f:g}", lambda folder=folder: _read(folder))
    D["arms"] = rows
    for k, r in rows.items():
        t = "arm." + k.replace("0.0", "f")
        N.put(t + ".mmpl", r["pl"]["mm_pl"], fnum(1e4 * r["pl"]["mm_pl"], 1, sign=True), "redo_numbers.read_run, " + k)
        N.put(t + ".trades", r["mm_trades"], fpct(r["mm_trades"], 1) if r["mm_trades"] >= 1e-3
              else fpct_small(r["mm_trades"]), "same")
        N.put(t + ".allbdc", r["total_bureau"], fpct(r["total_bureau"], 1), "same")


def runs_population(N, D):
    rows = {}
    for f in (0.02, 0.03):
        for n in (4, 12, 24):
            base = CM / f"pop{n}s0_mm0.01_bdc{f:g}_bid_size_2way"
            for stage, folder in (("test", base / "respond_on_cards"), ("own", base)):
                if (folder / "cotrain_multi_eval.npz").exists():
                    rows[f"{stage}{n}_{f:g}"] = cached_run(f"pop_{stage}{n}_{f:g}",
                                                           lambda folder=folder: _read(folder))
    extra = {"frozen": CM / "checks" / "T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_frozen_cotrained" / "respond_on_cards",
             "seed142": CM / "checks" / "T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_seed142",
             "seed242": CM / "checks" / "T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_seed242"}
    for k, folder in extra.items():
        if (folder / "cotrain_multi_eval.npz").exists():
            rows[k] = cached_run(f"pop_{k}", lambda folder=folder: _read(folder))
    D["population"] = rows
    src = "redo_numbers.read_run, population runs"
    for k, r in rows.items():
        t = "pop." + k.replace("_0.02", "a").replace("_0.03", "b")
        N.put(t + ".trades", r["mm_trades"], fpct(r["mm_trades"], 1), src)
        N.put(t + ".mmbdc", r["mm_bureau"], fpct(r["mm_bureau"], 1), src)
        N.put(t + ".netted", r["netted"], fpct(r["netted"], 1), src)
        N.put(t + ".allbdc", r["total_bureau"], fpct(r["total_bureau"], 1), src)
        N.put(t + ".mmpl", r["pl"]["mm_pl"], fnum(1e4 * r["pl"]["mm_pl"], 1, sign=True), src + " bp")
        if "T4" in r.get("per_card", {}):
            N.put(t + ".t4mm", r["per_card"]["T4"]["to_mm"], fpct(r["per_card"]["T4"]["to_mm"], 1), src)
            N.put(t + ".t4fill", r["per_card"]["T4"]["r1_filled"], fpct(r["per_card"]["T4"]["r1_filled"], 0), src)


def _mechanism(folder):
    """Where fills sit relative to the rate, and how the market maker's book
    comes to be cleared, from the 20,000 recorded games of one run."""
    z = np.load(folder / "cotrain_multi_eval.npz", allow_pickle=False)
    X = np.asarray(z["record_hidden_rate"], float)                      # (games, rounds)
    amount = np.asarray(z["record_offered_amount"], float)             # (games, offers, rounds)
    weight = np.asarray(z["record_fill_weight"], float)
    price = np.asarray(z["record_quoted_price"], float)
    direction = np.asarray(z["record_direction"], float)
    owner = np.asarray(z["offer_trader"], int)
    sides = np.array([str(s) for s in z["sides"]])
    forward = np.where(sides[owner] == "A", 1.0, -1.0)[None, :, None]
    made = amount > 0
    filled = made & (weight > 0.5)
    pound_fill = (filled & (direction > 0)).any(1)                      # (games, rounds)
    dollar_fill = (filled & (direction < 0)).any(1)
    both_round = pound_fill & dollar_fill
    anchor = np.concatenate([np.full((X.shape[0], 1), float(z["a0"])), X[:, :-1]], axis=1)
    rel = price / anchor[:, None, :] - 1.0
    fwd_made = made & (direction == forward)
    out = dict(
        p_pound=float(pound_fill.mean()), p_dollar=float(dollar_fill.mean()),
        p_both=float(both_round.mean()),
        p_both_indep=float(pound_fill.mean() * dollar_fill.mean()),
        games_both=float(both_round.any(1).mean()),
        quote_pound=[float(np.percentile(rel[fwd_made & (direction > 0)], q)) for q in (5, 50, 95)],
        quote_dollar=[float(np.percentile(rel[fwd_made & (direction < 0)], q)) for q in (5, 50, 95)],
        reverse_offered=float((np.abs(amount) * (made & (direction != forward))).sum()
                              / max((np.abs(amount) * made).sum(), 1e-300)),
        reverse_filled=float((np.abs(amount) * weight * (made & (direction != forward))).sum()
                             / max((np.abs(amount) * weight * made).sum(), 1e-300)),
    )
    fwd_filled = fwd_made & (weight > 0.5)
    for key, dv in (("quote_pound_filled", 1), ("quote_dollar_filled", -1)):
        m = fwd_filled & (direction == dv)
        out[key] = float(np.median(rel[m])) if m.any() else None
    # share of forward fills that came after the rate moved the trader's way
    rose = (X > anchor)[:, None, :]
    for key, dv, cond in (("pound_fill_after_rise", 1, rose), ("dollar_fill_after_fall", -1, ~rose)):
        m = fwd_filled & (direction == dv)
        out[key] = float((m & cond).sum() / m.sum()) if m.any() else None
    # the z of the rate move in rounds where a pound seller / a dollar seller filled
    move = (X - anchor) / float(z["sd"])
    out["move_pound"] = float(np.median(move[pound_fill])) if pound_fill.any() else None
    out["move_dollar"] = float(np.median(move[dollar_fill])) if dollar_fill.any() else None
    out["n_pound_fill"] = int(pound_fill.sum())
    out["n_dollar_fill"] = int(dollar_fill.sum())
    # the market maker's Bureau use: in-round against the final clear
    after_fill = np.asarray(z["record_mm_pounds_after_fill"], float)
    after_bureau = np.asarray(z["record_mm_pounds_after_bureau"], float)
    end = np.asarray(z["record_mm_pounds_end"], float)
    in_round = np.abs(after_bureau - after_fill).sum()
    final = np.abs(end - after_bureau[:, -1]).sum()
    out["final_share"] = float(final / max(in_round + final, 1e-300))
    # fill imbalance: |pounds bought - pounds paid| over pounds filled, per game
    filled_gbp = np.where(direction > 0, weight * amount,
                          np.divide(weight * amount, price, out=np.zeros_like(amount), where=amount > 0))
    bought = np.where(direction > 0, filled_gbp, 0.0).sum((1, 2))
    paid = np.where(direction < 0, filled_gbp, 0.0).sum((1, 2))
    out["imbalance"] = float(np.abs(bought - paid).sum() / max((bought + paid).sum(), 1e-300))
    out["bureau_share"] = float(z["bureau_share"]) if "bureau_share" in z.files else None
    out["bid_by_round"] = np.asarray(z["bid_by_round"], float).tolist()
    out["ask_by_round"] = np.asarray(z["ask_by_round"], float).tolist()
    g = float(z["fee"])
    out["kappa_A"] = [(1 - b) / g for b in out["bid_by_round"]]
    out["kappa_B"] = [(1 - 1 / a) / g for a in out["ask_by_round"]]
    return out


MOVE_BINS = np.arange(-4.0, 4.0001, 0.25)


def _move_hist(folder):
    """the rate move, in units of sd, in the rounds where a pound seller / a dollar
    seller filled, binned; from the same recorded games as _mechanism"""
    z = np.load(folder / "cotrain_multi_eval.npz", allow_pickle=False)
    X = np.asarray(z["record_hidden_rate"], float)
    amount = np.asarray(z["record_offered_amount"], float)
    weight = np.asarray(z["record_fill_weight"], float)
    direction = np.asarray(z["record_direction"], float)
    filled = (amount > 0) & (weight > 0.5)
    pound_fill = (filled & (direction > 0)).any(1)
    dollar_fill = (filled & (direction < 0)).any(1)
    anchor = np.concatenate([np.full((X.shape[0], 1), float(z["a0"])), X[:, :-1]], axis=1)
    move = np.clip((X - anchor) / float(z["sd"]), MOVE_BINS[0], MOVE_BINS[-1] - 1e-9)
    hp, _ = np.histogram(move[pound_fill], MOVE_BINS)
    hd, _ = np.histogram(move[dollar_fill], MOVE_BINS)
    return dict(pound=hp.tolist(), dollar=hd.tolist(), n_rounds=int(X.size))


def runs_mechanism(N, D):
    rows = {}
    hists = {}
    for f in (0.03, 0.07):
        folder = ladder_folder(f)
        if (folder / "cotrain_multi_eval.npz").exists():
            hists[f"{f:g}"] = cached_run(f"movehist_{f:g}", lambda folder=folder: _move_hist(folder))
    D["move_hist"] = hists
    for f in LADDER:
        folder = ladder_folder(f)
        if (folder / "cotrain_multi_eval.npz").exists():
            rows[f"{f:g}"] = cached_run(f"mech_{f:g}", lambda folder=folder: _mechanism(folder))
    pop4 = CM / "pop4s0_mm0.01_bdc0.03_bid_size_2way" / "respond_on_cards"
    if (pop4 / "cotrain_multi_eval.npz").exists():
        rows["pop4"] = cached_run("mech_pop4", lambda: _mechanism(pop4))
    D["mechanism"] = rows
    src = "Paper/make_assets._mechanism on the recorded games"
    for k, r in rows.items():
        t = "mech." + ("pop4" if k == "pop4" else f"s{round(100 * (float(k) - G_MM) * 10)}")
        N.put(t + ".both", r["p_both"], fpct_small(r["p_both"]), src + " Pr(both sides fill in a round)")
        N.put(t + ".indep", r["p_both_indep"], fpct_small(r["p_both_indep"]), src + " product of the marginals")
        N.put(t + ".ppound", r["p_pound"], fpct(r["p_pound"], 1), src)
        N.put(t + ".pdollar", r["p_dollar"], fpct(r["p_dollar"], 1), src)
        N.put(t + ".gamesboth", r["games_both"], fpct(r["games_both"], 1), src)
        N.put(t + ".qpound", r["quote_pound"][1], fpct(r["quote_pound"][1], 1, sign=True), src + " median quote / anchor - 1")
        N.put(t + ".qdollar", r["quote_dollar"][1], fpct(r["quote_dollar"][1], 1, sign=True), src)
        N.put(t + ".final", r["final_share"], fpct(r["final_share"], 1), src + " share of MM Bureau volume at the end")
        N.put(t + ".imb", r["imbalance"], fpct(r["imbalance"], 1), src + " fill imbalance")
        if r["move_pound"] is not None:
            N.put(t + ".movepound", r["move_pound"], fnum(r["move_pound"], 2, sign=True), src + " median rate move (sd) when a pound seller fills")
        if r["move_dollar"] is not None:
            N.put(t + ".movedollar", r["move_dollar"], fnum(r["move_dollar"], 2, sign=True), src)
        N.put(t + ".rev", r["reverse_offered"], fpct_small(r["reverse_offered"]), src + " share of offered volume in the reverse direction")
        N.put(t + ".revfilled", r["reverse_filled"], fsci(r["reverse_filled"], 0) if r["reverse_filled"] > 0 else "0",
              src + " share of filled volume in the reverse direction")
        for key, name in (("pound_fill_after_rise", "uppound"), ("dollar_fill_after_fall", "downdollar")):
            if r.get(key) is not None:
                N.put(t + "." + name, r[key], fpct(r[key], 1), src + " share of forward fills after the rate moved that way")
        for key, name in (("quote_pound_filled", "qpoundfill"), ("quote_dollar_filled", "qdollarfill")):
            if r.get(key) is not None:
                N.put(t + "." + name, r[key], fpct(r[key], 1, sign=True), src + " median quote / anchor - 1, filled offers")
        for n in range(4):
            for side, key in (("A", "kappa_A"), ("B", "kappa_B")):
                x = r[key][n]
                N.put(t + f".k{side}{n + 1}", x, fnum(x, 2) if np.isfinite(x) else "--",
                      src + f" side {side} cut / MM fee, round {n + 1}")


def _curve(folder):
    c = np.load(folder / "cotrain_multi_curve.npz", allow_pickle=True)
    steps = (np.asarray(c["step_mm"]) + np.asarray(c["step_trader"])).astype(float)
    return dict(steps=steps.tolist(), filled=np.asarray(c["filled"], float).tolist(),
                mm_pl=np.asarray(c["mm_pl"], float).tolist(), stopped_at=int(c["stopped_at"]))


def runs_curves(N, D):
    groups = {"0.02": [], "0.03": []}
    for folder in sorted(list(CM.glob("*bdc0.0[23]_bid*")) + list((CM / "checks").glob("*bdc0.0[23]_bid*"))):
        if not (folder / "cotrain_multi_curve.npz").exists():
            continue
        if re.match(r"pop\d+s[1-9]", folder.name):   # other random draws of a table: not used in the paper
            continue
        f = "0.02" if "bdc0.02" in folder.name else "0.03"
        c = _curve(folder)
        groups[f].append(dict(name=folder.name, peak=max(c["filled"][:3]), last=c["filled"][-1],
                              mm_last=c["mm_pl"][-1]))
    D["curve_groups"] = groups
    who = {"_tau_const": "the run in which neither agent's acceptance rule was smoothed",
           "pop4s1": "a co-training run on another random draw of a table of four traders",
           "pop4s2": "a co-training run on another random draw of a table of four traders"}
    closed = [x["name"] for x in groups["0.03"] if x["last"] <= 0.02]
    names = [next((v for k, v in who.items() if n.startswith(k) or n.endswith(k)), n) for n in closed]
    N.put("curve.s20.closedwho", len(closed), " and ".join(names) if names else "none",
          "2% runs whose last validation fill is at most 2%")
    for f, tag in (("0.02", "s10"), ("0.03", "s20")):
        g = groups[f]
        open_ = [x for x in g if x["last"] > 0.02]
        N.put(f"curve.{tag}.n", len(g), str(len(g)), "runs with a saved curve")
        N.put(f"curve.{tag}.open", len(open_), str(len(open_)), "last validation fill > 2%")
        N.put(f"curve.{tag}.peaklo", min(x["peak"] for x in g), fpct(min(x["peak"] for x in g), 0), "")
        N.put(f"curve.{tag}.peakhi", max(x["peak"] for x in g), fpct(max(x["peak"] for x in g), 0), "")
        if open_:
            N.put(f"curve.{tag}.lastlo", min(x["last"] for x in open_), fpct(min(x["last"] for x in open_), 0), "")
            N.put(f"curve.{tag}.lasthi", max(x["last"] for x in open_), fpct(max(x["last"] for x in open_), 0), "")
    D["curves"] = {"1% spread": _curve(ladder_folder(0.02)), "2% spread": _curve(ladder_folder(0.03))}
    tc = CM / "checks" / "T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way_tau_const"
    if (tc / "cotrain_multi_curve.npz").exists():
        D["curves"]["2% spread, no anneal"] = _curve(tc)
    base = D["curves"]["2% spread"]
    N.put("curve.firstfill", base["filled"][1], fpct(base["filled"][1], 0), "fill after the first MM block")
    N.put("curve.firstmm", base["mm_pl"][1], fnum(1e4 * base["mm_pl"][1], 0, sign=True), "MM bp after the first block")
    one = D["curves"]["1% spread"]
    closed = next(s for s, x in zip(one["steps"], one["filled"]) if x < 1e-3)
    N.put("curve.s10.closedat", closed, f"{closed:,.0f}", "first step with fill < 0.1% at 1%")


def runs_robustness(N, D):
    rows = {}
    for tag in ("_seed142", "_seed242", "_tau_const", "_tau_const_mm", "_outer30_anneal700"):
        folder = CM / "checks" / f"T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way{tag}"
        if (folder / "cotrain_multi_eval.npz").exists():
            rows[tag.strip("_")] = cached_run(f"rob{tag}", lambda folder=folder: _read(folder))
    for tag in ("_lr", "_longer"):
        folder = CM / f"T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way{tag}"
        if (folder / "cotrain_multi_eval.npz").exists():
            rows[tag.strip("_")] = cached_run(f"rob{tag}", lambda folder=folder: _read(folder))
    D["robustness"] = rows
    for tag in ("_tau_const", "_tau_const_mm"):
        f = CM / "checks" / f"T1-T2-T3-T4_mm0.01_bdc0.03_bid_size_2way{tag}" / "cotrain_multi_eval.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            t = "rob." + tag.replace("_", "")
            N.put(t + ".tautrader0", float(z["tau_trader"][0]), fnum(float(z["tau_trader"][0]), 2), str(f.name))
            N.put(t + ".taumm0", float(z["tau_mm"][0]), fnum(float(z["tau_mm"][0]), 2), str(f.name))
    base = D.get("ladder", {}).get("0.03")
    nets = [r["netted"] for r in rows.values()] + ([base["netted"]] if base else [])
    if nets:
        N.put("rob.maxnetted", max(nets), fpct(max(nets), 1), "largest netted share, base 2% run and its checks")
    for k, r in rows.items():
        t = "rob." + k.replace("_", "")
        N.put(t + ".mmpl", r["pl"]["mm_pl"], fnum(1e4 * r["pl"]["mm_pl"], 1, sign=True), "redo_numbers.read_run, " + k)
        N.put(t + ".trades", r["mm_trades"], fpct(r["mm_trades"], 1), "same")
        N.put(t + ".mmbdc", r["mm_bureau"], fpct(r["mm_bureau"], 1), "same")
        N.put(t + ".netted", r["netted"], fpct(r["netted"], 1), "same")
        N.put(t + ".allbdc", r["total_bureau"], fpct(r["total_bureau"], 1), "same")


def runs_verification(N, D):
    v = CM / "verification"
    out = {}
    for name in ("verify_population.json", "replay_check.json"):
        if (v / name).exists():
            out[name] = json.loads((v / name).read_text())
    D["verification_files"] = sorted(out)
    vp = out.get("verify_population.json")
    if vp:
        worst = max(max(r["worst_mismatch"].values()) for r in vp)
        N.put("ver.nruns", len(vp), str(len(vp)), "verify_population.json: runs rebuilt from their records")
        N.put("ver.npassed", sum(bool(r["passed"]) for r in vp), str(sum(bool(r["passed"]) for r in vp)), "same, passed")
        N.put("ver.worst", worst, fsci(worst, 0), "same, worst mismatch between the rebuilt and the recorded books")
    rc = out.get("replay_check.json")
    if rc:
        gaps, rows = [], 0
        for run in rc["runs"]:
            for part in ("table", "cards"):
                for row in run.get(part, []):
                    gaps += [abs(row["mm_gap"]), abs(row["trader_gap"])]
                    rows += 1
        N.put("rep.nruns", len(rc["runs"]), str(len(rc["runs"])), "replay_check.json: runs replayed")
        N.put("rep.nrows", rows, str(rows), "same, training blocks replayed")
        N.put("rep.maxgap", max(gaps), fnum(max(gaps), 0) if max(gaps) == 0 else fsci(max(gaps), 0),
              "same, largest gap between saved and replayed P/L")


RUN_PARTS = [runs_cards, runs_fee_dp, runs_fee_dp_summary, runs_mm_phase1, runs_cotrain, runs_learning, runs_ppo, runs_ladder,
             runs_arms, runs_population, runs_mechanism, runs_curves, runs_robustness,
             runs_verification]


def run_runs(N, D):
    for fn in RUN_PARTS:
        t0 = time.time()
        try:
            fn(N, D)
            print(f"  runs   {fn.__name__:<22} {time.time() - t0:6.1f}s", flush=True)
        except Exception:
            print(f"  runs   {fn.__name__:<22} FAILED", flush=True)
            traceback.print_exc()


# =============================================================================
# 3b. figures (from the cached data only, so they can be redrawn quickly)
# =============================================================================
INK, INK2, GRID = "#1a1a1a", "#55534e", "#e6e5e0"
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_VIOLET, C_RED, C_GREY = "#4a3aa7", "#e34948", "#b9b7af"
FULL, HALF = 6.3, 3.05


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = next((n for n in ("TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                               "Book Antiqua", "DejaVu Serif") if n in names), "serif")
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family + ":italic",
        "mathtext.bf": family + ":bold",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 3})
    return plt


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _save(plt, fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_oneshot(plt, D):
    cur = D["v0_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for ax, side in zip(axes, ("A", "B")):
        c = cur[side]
        P = np.array(c["P"])
        v = np.array(c["value"])
        if side == "A":
            x, xstar = P, c["Pstar"]
            unit, title = r"offer $P$ (\$ per \pounds)", r"(a) Pound seller (T1)"
        else:
            x, xstar = 1 / P, 1 / c["Pstar"]
            unit, title = r"offer $1/P$ (\pounds\ per \$)", r"(b) Dollar seller (T2)"
        ax.plot(x, v, color=C_BLUE, lw=1.5, zorder=3)
        ax.axhline(c["floor"], color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(xstar, color=INK2, lw=0.6, ls=":")
        ax.plot([xstar], [c["vstar"]], "o", ms=4, color=C_ORANGE, zorder=4)
        ax.annotate("Bureau only", (x.min() + 0.02 * (x.max() - x.min()), c["floor"]),
                    xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
        ax.annotate("optimum", (xstar, c["vstar"]), xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color=INK2)
        lo = c["floor"] - 0.6 * (c["vstar"] - c["floor"]) - 0.0015 * c["floor"]
        ax.set_ylim(lo, c["floor"] + 3.2 * (c["vstar"] - c["floor"]) + 0.0004 * c["floor"])
        ax.set_xlabel(unit.replace(r"\pounds\ ", "£ ").replace(r"\pounds", "£"))
        ax.set_title(title, loc="left")
        ax.set_ylabel(r"payoff per pound (\$)" if side == "A" else "payoff per dollar (£)")
        _grid(ax)
    _save(plt, fig, "fig_oneshot")


def fig_sigma(plt, D):
    rows = D["sigma_sweep"]
    sd = np.array([r["sd"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))
    for side, col, lab in (("A", C_BLUE, "pound seller"), ("B", C_ORANGE, "dollar seller")):
        z = np.array([np.nan if r[side]["z"] is None else r[side]["z"] for r in rows], float)
        fill = np.array([r[side]["fill"] for r in rows])
        axes[0].plot(sd, z, color=col, label=lab)
        axes[1].plot(sd, 100 * fill, color=col, label=lab)
    for ax in axes:
        ax.axvline(SD, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel(r"step standard deviation $\sigma$")
        _grid(ax)
    axes[0].set_ylabel(r"optimal offer beyond $a_0$ ($\sigma$)")
    axes[0].axhline(0, color=INK2, lw=0.6)
    axes[1].set_ylabel("probability the offer fills (%)")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    axes[1].legend(loc="upper right")
    axes[0].annotate(r"game's $\sigma$", (SD, axes[0].get_ylim()[1]), xytext=(3, -10),
                     textcoords="offset points", fontsize=7.5, color=INK2)
    _save(plt, fig, "fig_sigma")


def fig_strips(plt, D):
    from scipy.stats import norm
    P, X21, q21 = D["strips"]["P"], D["strips"]["X21"], D["strips"]["q21"]
    x = np.linspace(A0 - 4 * SD, A0 + 4.5 * SD, 800)
    pdf = norm.pdf(x, A0, SD)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=True)
    panels = ((axes[0], P, P, r"(a) $g=0$, offer $P=%.4f$" % P),
              (axes[1], P, P / (1 - G_MM), r"(b) $g=1\%$, same offer"),
              (axes[2], q21, X21, r"(c) $g=1\%$, best offer"))
    for ax, quote, lo, title in panels:
        hi = quote / (1 - F_TRADER)                 # a fill beats the Bureau iff X < hi
        ax.plot(x, pdf, color=INK2, lw=0.9)
        m_rej = x <= lo
        m_win = (x > lo) & (x < hi)
        m_lose = x >= max(lo, hi)
        ax.fill_between(x[m_rej], pdf[m_rej], color=C_GREY, alpha=0.35, lw=0, label="rejected: Bureau")
        ax.fill_between(x[m_win], pdf[m_win], color=C_AQUA, alpha=0.85, lw=0, label="filled, beats the Bureau")
        ax.fill_between(x[m_lose], pdf[m_lose], color=C_RED, alpha=0.85, lw=0, label="filled, worse than the Bureau")
        ax.axvline(lo, color=INK, lw=0.6, ls=":")
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel(r"the round's rate $X$ (\$ per £)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(A0 - 4 * SD, A0 + 4.5 * SD)
    fill = 1 - norm.cdf((X21 - A0) / SD)
    ymax = pdf.max()
    axes[2].annotate("fills in %.2f%%\nof rounds" % (100 * fill), xy=(X21 + 0.3 * SD, 0.01 * ymax),
                     xytext=(X21 - 0.15 * SD, 0.5 * ymax), fontsize=7.5, color=INK2, ha="right",
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    _save(plt, fig, "fig_strips")


def fig_learning(plt, D):
    """Learning curves as the co-training scripts record them: each trader's P/L (% of its
    capital) on the validation games after every block, with its always-Bureau P/L on the same
    games dashed, and the four-trader market maker's P/L (bp) against the best fixed rule."""
    L = D["learning"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.9), sharex=True)
    fig.subplots_adjust(wspace=0.34, hspace=0.34)
    runs = (("fee0", "(2%, 0)", C_BLUE), ("fee0.01", "(2%, 1%)", C_ORANGE),
            ("fee0.01_BdC_FEE0.03", "(3%, 1%)", C_AQUA))
    for ax, side, title in ((axes[0, 0], "A", "(a) One trader: a pound seller"),
                            (axes[0, 1], "B", "(b) One trader: a dollar seller")):
        for tag, lab, col in runs:
            r = L["one"].get(f"{side}_{tag}")
            if r is None:
                continue
            pl = 100 * np.array(r["trader_pl"])
            se = 100 * np.array(r["edge_2se"])
            x = np.arange(len(pl))
            ax.plot(x, pl, color=col, lw=1.1, marker="o", ms=1.8, label=lab)
            ax.fill_between(x, pl - se, pl + se, color=col, alpha=0.15, lw=0)
            ax.plot(x, 100 * np.array(r["bdc_pl"]), color=col, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_ylabel("trader P/L (% of capital)")
        _grid(ax)
    axes[0, 0].legend(loc="lower right", fontsize=7, title=r"$(f,g)$", title_fontsize=7)
    four = L["four"]
    pl = 100 * np.array(four["trader_pl"])
    se = 100 * np.array(four["edge_2se"])
    bd = 100 * np.array(four["bdc_pl"])
    x = np.arange(pl.shape[0])
    for i, (name, col) in enumerate(zip(four["names"], (C_BLUE, C_ORANGE, C_AQUA, C_VIOLET))):
        axes[1, 0].plot(x, pl[:, i], color=col, lw=1.1, marker="o", ms=1.8, label=name)
        axes[1, 0].fill_between(x, pl[:, i] - se[:, i], pl[:, i] + se[:, i], color=col, alpha=0.15, lw=0)
        axes[1, 0].plot(x, bd[:, i], color=col, lw=0.8, ls=(0, (3, 2)))
    axes[1, 0].set_title("(c) Four traders at a 2% spread: each trader", loc="left", fontsize=8.5)
    axes[1, 0].set_ylabel("trader P/L (% of capital)")
    axes[1, 0].legend(loc="lower right", fontsize=7, ncol=2)
    axes[1, 1].plot(x, 1e4 * np.array(four["mm"]), color=INK, lw=1.1, marker="o", ms=1.8,
                    label="co-trained market maker")
    axes[1, 1].plot(x, 1e4 * np.array(four["best_rule"]), color=INK2, lw=0.9, ls=(0, (3, 2)),
                    label="best of six fixed rules")
    axes[1, 1].set_title("(d) Four traders at 2%: the market maker", loc="left", fontsize=8.5)
    axes[1, 1].set_ylabel("MM P/L (bp of the game's capital)")
    axes[1, 1].legend(loc="upper right", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("block")
        _grid(ax)
    _save(plt, fig, "fig_learning")


def fig_flows(plt, D):
    rep = D["fee_dp"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.0), sharey=True)
    for ax, side, title in ((axes[0], "A", "(a) Pound seller (T1)"), (axes[1], "B", "(b) Dollar seller (T2)")):
        fl = rep[f"{side} f=0.02 g=0"]["flows"]
        r = np.arange(1, 5)
        mm, bdc = 100 * np.array(fl["to_mm"]), 100 * np.array(fl["to_bureau"])
        ax.bar(r, mm, 0.6, color=C_BLUE, label="to the market maker (filled offers)", zorder=3)
        ax.bar(r, bdc, 0.6, bottom=mm, color=C_GREY, label="to the Bureau", zorder=3)
        for k in range(4):
            if mm[k] + bdc[k] > 0.5:
                ax.annotate(f"{mm[k] + bdc[k]:.1f}%", (r[k], mm[k] + bdc[k]), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(r)
        ax.set_xticklabels([f"round {k}" for k in r])
        ax.set_title(title, loc="left")
        _grid(ax)
    axes[0].set_ylabel("share of starting capital (%)")
    axes[0].set_ylim(0, 105)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    _save(plt, fig, "fig_flows")


def fig_interp(plt, D):
    fig, ax = plt.subplots(figsize=(HALF, 2.1))       # schematic, curvature exaggerated
    u = np.linspace(0.3, 2.2, 200)
    ax.plot(u, 1 / u, color=C_BLUE, lw=1.4, label=r"true value (convex)")
    nodes = np.array([0.45, 1.0, 1.55, 2.1])
    ax.plot(nodes, 1 / nodes, color=C_ORANGE, lw=1.1, marker="o", ms=3.5, label="table: straight lines")
    uu = np.linspace(nodes[0], nodes[-1], 300)
    ax.fill_between(uu, 1 / uu, np.interp(uu, nodes, 1 / nodes), color=C_ORANGE, alpha=0.18, lw=0)
    ax.set_xticks(nodes)
    ax.set_xticklabels(["grid point"] * 4, fontsize=6.5)
    ax.set_yticks([])
    ax.set_xlabel("anchor")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=7)
    _save(plt, fig, "fig_interp")


def fig_relax(plt, D):
    from scipy.special import expit
    rc = D["relax_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))
    ax = axes[0]
    u = np.linspace(-3, 3, 601)                     # (X - P) / sigma
    ax.step(u, (u > 0).astype(float), where="post", color=INK, lw=1.4, label="hard rule (game)")
    for tau, col in ((1.0, C_ORANGE), (0.2, C_BLUE), (0.02, C_AQUA)):
        ax.plot(u, expit(u / tau), color=col, lw=1.1, label=r"relaxed, $\tau=%g$" % tau)
    ax.set_xlabel(r"rate minus offer, $(X-P)/\sigma$")
    ax.set_ylabel("probability the offer fills")
    ax.set_title("(a) The market maker's verdict", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    _grid(ax)
    ax = axes[1]
    z = np.array(rc["z"])
    for tau, col, lab in (("0.0", INK, "hard rule"), ("1.0", C_ORANGE, r"relaxed, $\tau=1$"),
                          ("0.2", C_BLUE, r"relaxed, $\tau=0.2$")):
        g = np.array(rc["G"][tau])
        ax.plot(z, g, color=col, lw=1.3 if tau == "0.0" else 1.1, label=lab)
        k = int(np.argmax(g))
        ax.plot([z[k]], [g[k]], "o", ms=3.5, color=col)
    ax.axhline(rc["floor"], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.annotate("Bureau only", (z[-1], rc["floor"]), xytext=(-2, 3), textcoords="offset points",
                fontsize=7.5, color=INK2, ha="right")
    ax.set_xlabel(r"offer $P$, in $\sigma$ above $a_0$")
    ax.set_ylabel(r"expected payoff per pound (\$)")
    ax.set_title("(b) Expected payoff of a blind offer", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    _grid(ax)
    _save(plt, fig, "fig_relax")


def fig_rho(plt, D):
    rows = D["rho_sweep"]
    rho = np.array([r["rho"] for r in rows])
    fill = np.array([r["fill"] if r["interior"] and r["fill"] > 1e-8 else np.nan for r in rows])
    share = np.array([r["share"] if r["interior"] else 0.0 for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))
    axes[0].semilogy(rho, 100 * fill, color=C_BLUE)
    axes[1].plot(rho, 100 * share, color=C_BLUE)
    marks = D["rho_marks"]
    for ax in axes:
        for lab, r in marks.items():
            ax.axvline(r, color=INK2, lw=0.5, ls=":")
        _grid(ax)
        ax.set_xlabel(r"governing ratio $\rho = f\,'a_0/\sigma$")
    names = {}
    for lab, r in marks.items():
        names.setdefault(round(r, 1), []).append(lab)
    top = axes[1].get_ylim()[1]
    for r, labs in names.items():
        axes[1].annotate(" and ".join(labs), (r, top), xytext=(-9, -3), textcoords="offset points",
                         fontsize=6.6, color=INK2, rotation=90, va="top", ha="center")
    axes[0].set_ylabel("probability the best offer fills (%)")
    axes[1].set_ylabel("share of the fee recovered (%)")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    _save(plt, fig, "fig_rho")


def fig_edges(plt, D):
    rep = D["fee_dp"]
    fees = [(0.02, 0.0), (0.02, 0.01), (0.03, 0.0), (0.03, 0.01)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    xs = np.arange(len(fees))
    for ax, side, title in ((axes[0], "A", "(a) Pound seller (T1)"), (axes[1], "B", "(b) Dollar seller (T2)")):
        rows = [rep[f"{side} f={f:g} g={g:g}"] for f, g in fees]
        dp = [100 * r["dp_edge"] for r in rows]
        dpe = [200 * r["dp_edge_se"] for r in rows]
        ne = [100 * r["net_edge"] for r in rows]
        nee = [200 * r["net_edge_se"] for r in rows]
        ax.bar(xs - 0.19, dp, 0.36, yerr=dpe, color=C_BLUE, label="exact DP",
               error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
        ax.bar(xs + 0.19, ne, 0.36, yerr=nee, color=C_ORANGE, label="neural network",
               error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"f={100*f:g}%\ng={100*g:g}%" for f, g in fees])
        ax.set_title(title, loc="left")
        _grid(ax)
    axes[0].set_ylabel("edge over Bureau-only (points)")
    axes[1].legend(loc="upper left")
    _save(plt, fig, "fig_edges")


def fig_ladder(plt, D):
    lad = D["ladder"]
    fs = sorted(lad, key=float)
    spreads = [100 * (float(f) - G_MM) for f in fs]
    parts = [("netted", "netted by the MM", C_AQUA), ("mm_bureau", "MM to the Bureau", C_BLUE),
             ("cards_bureau", "traders to the Bureau", C_GREY), ("left_over", "left unconverted", C_YELLOW)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    xs = np.arange(len(fs))
    bottom = np.zeros(len(fs))
    for key, lab, col in parts:
        vals = np.array([lad[f][key] for f in fs])
        tot = np.array([sum(lad[f][k] for k, _, _ in parts) for f in fs])
        vals = 100 * vals / tot
        ax.bar(xs, vals, 0.68, bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.6)
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s:g}%" for s in spreads])
    ax.set_xlabel(r"spread $f-g$")
    ax.set_ylabel("share of the game's money (%)")
    ax.set_title("(a)", loc="left")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7)
    ax2 = axes[1]
    mm = [1e4 * lad[f]["pl"]["mm_pl"] for f in fs]
    ax2.plot(spreads, mm, "o-", color=C_BLUE, ms=3.5, label="market maker")
    ax2.set_xlabel(r"spread $f-g$ (%)")
    ax2.set_ylabel("market maker P/L (bp of the money)")
    ax2.set_title("(b)", loc="left")
    _grid(ax2)
    _save(plt, fig, "fig_ladder")


def fig_curves(plt, D):
    cur = D["curves"]
    style = {"1% spread": (C_BLUE, "-"), "2% spread": (C_ORANGE, "-"),
             "2% spread, no anneal": (C_ORANGE, (0, (4, 2.5)))}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.25))
    for lab, c in cur.items():
        col, ls = style.get(lab, (C_VIOLET, "-"))
        axes[0].plot(c["steps"], 100 * np.array(c["filled"]), color=col, ls=ls, label=lab)
        axes[1].plot(c["steps"], 1e4 * np.array(c["mm_pl"]), color=col, ls=ls, label=lab)
    axes[0].set_ylabel("offered volume filled (%)")
    axes[1].set_ylabel("market maker P/L (bp)")
    for ax in axes:
        ax.set_xlabel("training steps, both sides")
        _grid(ax)
        ax.axhline(0, color=INK2, lw=0.6)
    axes[0].legend(loc="upper right")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    _save(plt, fig, "fig_curves")


def fig_population(plt, D):
    pop = D["population"]
    lad = D["ladder"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2))
    ns = [4, 12, 24]
    for ax, key, lab in ((axes[0], "total_bureau", "all Bureau use (%)"), (axes[1], "netted", "netted by the MM (%)")):
        for f, col, name in (("0.02", C_BLUE, "1% spread"), ("0.03", C_ORANGE, "2% spread")):
            ys = [100 * pop[f"test{n}_{f}"][key] for n in ns if f"test{n}_{f}" in pop]
            ax.plot(ns[:len(ys)], ys, "o-", color=col, ms=3.5, label=name + ", population MM")
            base = 100 * lad[f][key]
            ax.axhline(base, color=col, lw=0.8, ls=(0, (3, 2)))
        if "frozen" in pop:
            ax.plot([4, 24], [100 * pop["frozen"][key]] * 2, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(ns)
        ax.set_xlabel("random traders the MM trained against")
        ax.set_ylabel(lab)
        _grid(ax)
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    axes[1].legend(loc="upper right", fontsize=6.8)
    _save(plt, fig, "fig_population")


def fig_mechanism(plt, D):
    """where the rate had moved in the rounds each side filled: 2% and 6% spreads"""
    H = D["move_hist"]
    centres = 0.5 * (MOVE_BINS[:-1] + MOVE_BINS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), sharey=True)
    for ax, f, title in ((axes[0], "0.03", "(a) 2% spread"), (axes[1], "0.07", "(b) 6% spread")):
        h = H[f]
        n = h["n_rounds"]
        ax.bar(centres, 100 * np.array(h["pound"]) / n, width=0.25, color=C_BLUE, alpha=0.55,
               label="a pound seller fills")
        ax.bar(centres, 100 * np.array(h["dollar"]) / n, width=0.25, color=C_ORANGE, alpha=0.55,
               label="a dollar seller fills")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"rate move in the round ($\sigma$)")
        _grid(ax)
    axes[0].set_ylabel("share of rounds (%)")
    axes[0].legend(loc="upper right")
    _save(plt, fig, "fig_mechanism")


def fig_feedback(plt, D):
    from scipy.stats import norm
    v1 = D["v1"]
    one = D.get("v2_one_round_sized")
    lad = D["v1_ladder5"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    x = np.linspace(A0 - 7 * SD, A0 + 4 * SD, 900)
    pdf = norm.pdf(x, A0, SD)
    ax.plot(x, pdf, color=INK2, lw=0.9)
    ax.set_xlim(A0 - 7 * SD, A0 + 4 * SD)
    cols = [C_BLUE, C_AQUA, C_YELLOW, C_ORANGE, C_VIOLET]
    hi = x.max()
    for k, (P, col) in enumerate(zip(lad, cols)):
        m = (x > P) & (x <= hi)
        ax.fill_between(x[m], pdf[m], color=col, alpha=0.75, lw=0, label=f"offer {k + 1}: {P:.4f}")
        ax.axvline(P, color=col, lw=0.7, ls=":")
        hi = P
    m = x <= hi
    ax.fill_between(x[m], pdf[m], color=C_GREY, alpha=0.35, lw=0, label="all rejected: Bureau")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"the round's rate $X$ (\$ per £)")
    ax.set_title("(a) Which rates each of five offers catches", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[1]
    K = [r["K"] for r in v1]
    ax.plot(K, [100 * r["share"] for r in v1], "o-", color=C_BLUE, ms=3.2, label="all-or-nothing offers")
    if one:
        floor, pie = (1 - F_TRADER) * A0, F_TRADER * A0
        ax.plot(range(1, len(one["values"]) + 1), [100 * (v - floor) / pie for v in one["values"]],
                "s-", color=C_ORANGE, ms=3.2, label="sized offers")
    ax.set_xlabel(r"offers allowed in the round, $K$")
    ax.set_ylabel("share of the fee recovered (%)")
    ax.set_xticks(K)
    ax.set_title("(b) Value of more offers", loc="left")
    ax.legend(loc="lower right")
    _grid(ax)
    _save(plt, fig, "fig_feedback")


# fig_sigma and fig_edges are not in the paper (sigma* is an artefact of the additive model near the
# truncation; the edges are in the merged Table 2). Their functions are kept for reference.
FIGURES = [fig_oneshot, fig_strips, fig_rho, fig_ladder, fig_curves, fig_population,
           fig_mechanism, fig_feedback, fig_flows, fig_interp, fig_relax]   # learning curves: make_run_figures.py


def make_figures(D):
    plt = _style()
    for fn in FIGURES:
        try:
            fn(plt, D)
            print(f"  figure {fn.__name__}")
        except KeyError as e:
            print(f"  figure {fn.__name__:<16} skipped: no data for {e}")
        except Exception:
            print(f"  figure {fn.__name__:<16} FAILED")
            traceback.print_exc()


# =============================================================================
# 3c. tables (LaTeX bodies, from the cached data only)
# =============================================================================
def _pts(x, nd=4, sign=True):
    return fnum(100 * x, nd, sign=sign)


def _pm(x, se, nd=2):
    """a paired difference in bp of capital, +- 2 s.e. (x and se are fractions of capital)"""
    return f"${fnum(1e4 * x, nd, sign=True).replace(chr(92) + 'ensuremath{-}', '-')} \\pm {fnum(2e4 * se, nd)}$"


def _bp_pm(x, se):
    """a paired difference in bp, +- 2 s.e.; values below 0.0005 bp in scientific form"""
    def one(v, sign):
        if v == 0:
            return "0"
        if abs(v) >= 5e-4:
            return fnum(v, 3, sign=sign).replace(chr(92) + "ensuremath{-}", "-")
        m, e = f"{v:.0e}".split("e")
        return f"{'+' if sign and v > 0 else ''}{m}\\times 10^{{{int(e)}}}"
    return f"${one(1e4 * x, True)} \\pm {one(2e4 * se, False)}$"


def make_tables(D):
    T = PAPER / "tables"
    T.mkdir(parents=True, exist_ok=True)
    made = []

    def write(name, lines):
        (T / f"{name}.tex").write_text("\n".join(lines) + "\n")
        made.append(name)

    if "cards" in D:
        rows = []
        for r in D["cards"]:
            start = f"\\pounds{r['L']:,.0f}" if r["side"] == "A" else f"\\${r['L']:,.0f}"
            target = f"\\${r['T']:,.0f}" if r["side"] == "A" else f"\\pounds{r['T']:,.0f}"
            who = "pound seller" if r["side"] == "A" else "dollar seller"
            rows.append(f"{r['name']} & {who} & {start} & {target} & {fpct(r['A'], 0)} & "
                        f"{fpct(r['B'], 0)} & {fpct(r['fair'], 1)} \\\\")
        write("tab_cards", rows)
    if "fee_dp" in D:
        rows = []
        for side, name in (("A", "T1, pound seller"), ("B", "T2, dollar seller")):
            rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{name}}}}} \\\\")
            # ordered by the fee spread f-g: equal spreads (2%,0) and (3%,1%) side by side (Prop 5.2)
            for f, g in ((0.02, 0.01), (0.02, 0.0), (0.03, 0.01), (0.03, 0.0)):
                r = D["fee_dp"][f"{side} f={f:g} g={g:g}"]
                gl = fpct(g, 0) if g > 0 else "0"
                rows.append(f"\\quad ({fpct(f, 0)}, {gl}) & {fpct(f - g, 0)} & {fpct(r['floor'], 3)} & {fpct(r['dp_value'], 3)} & "
                            f"{_pm(r['dp_edge'], r['dp_edge_se'])} & {_pm(r['net_edge'], r['net_edge_se'])} & "
                            f"{_pm(r['dp_minus_net'], r['dp_minus_net_se'])} \\\\")
        write("tab_feedp", rows)
    if "ladder" in D:
        rows = []
        for fs in sorted(D["ladder"], key=float):
            r = D["ladder"][fs]
            sp = 100 * (float(fs) - G_MM)
            rows.append(f"{sp:g}\\% & {fpct(r['mm_trades'], 1)} & {fpct(r['netted'], 1)} & "
                        f"{fpct(r['mm_bureau'], 1)} & {fpct(r['cards_bureau'], 1)} & {fpct(r['left_over'], 1)} & "
                        f"{fpct(r['total_bureau'], 1)} & ${fnum(1e4 * r['pl']['mm_pl'], 1, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ \\\\")
        write("tab_ladder", rows)
    if "population" in D and "ladder" in D:
        pop, lad = D["population"], D["ladder"]
        rows = []

        def line(label, r):
            return (f"{label} & {fpct(r['mm_trades'], 1)} & {fpct(r['netted'], 1)} & {fpct(r['mm_bureau'], 1)} & "
                    f"{fpct(r['total_bureau'], 1)} & "
                    f"${fnum(1e4 * r['pl']['mm_pl'], 1, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ \\\\")
        rows.append(line("co-trained with T1--T4", lad["0.03"]))
        if "frozen" in pop:
            rows.append(line("control: the same MM, frozen", pop["frozen"]))
        for n in (4, 12, 24):
            k = f"test{n}_0.03"
            if k in pop:
                rows.append(line(f"{n} random traders", pop[k]))
        write("tab_population", rows)
    if "v1" in D:
        rows = [f"{r['K']} & {fnum(r['value'], 5)} & {fnum(r['first'], 4)} & {fpct(r['share'], 1)} \\\\"
                for r in D["v1"]]
        write("tab_v1", rows)
    if "refine" in D:
        rows = [f"{r['n_a']} & {fnum(r['step_sd'], 2)} & {fpct(r['table'], 4)} & {fpct(r['replay'], 4)} & "
                f"${fnum(1e4 * (r['table'] - r['replay']), 2, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ \\\\"
                for r in D["refine"]]
        write("tab_refine", rows)
    if "negative" in D:
        rows = []
        for r in D["negative"]:
            ever = fpct(r["ever"], 1) if r["ever"] is not None else "--"
            cells = [f"{r['sd']:.2f}", fsci(r['last'], 1), ever]
            if abs(r["sd"] - SD) < 1e-12:
                cells = [c if c == "--" else f"\\boldmath\\bfseries {c}" for c in cells]
            rows.append(" & ".join(cells) + " \\\\")
        write("tab_negative", rows)
    if "robustness" in D and "ladder" in D:
        rob = dict(D["robustness"])
        order = [("base", "base run (seed 42)", D["ladder"]["0.03"]),
                 ("seed142", "seed 142", rob.get("seed142")), ("seed242", "seed 242", rob.get("seed242")),
                 ("outer30_anneal700", "30 outer steps", rob.get("outer30_anneal700")),
                 ("tau_const_mm", "MM without the anneal", rob.get("tau_const_mm")),
                 ("tau_const", "neither agent annealed", rob.get("tau_const")),
                 ("lr", "trader learning rate $10^{-4}$", rob.get("lr")),
                 ("longer", "longer anneal, trader rate $10^{-4}$", rob.get("longer"))]
        rows = []
        for _, lab, r in order:
            if r is None:
                continue
            rows.append(f"{lab} & {fpct(r['mm_trades'], 1)} & {fpct(r['netted'], 1)} & {fpct(r['mm_bureau'], 1)} & "
                        f"{fpct(r['total_bureau'], 1)} & ${fnum(1e4 * r['pl']['mm_pl'], 1, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ \\\\")
        write("tab_robust", rows)
    if "cotrain" in D:
        rows = []
        for key, lab in (("A_fee0", "pound seller, $(2\\%,0)$"), ("A_fee0.01", "pound seller, $(2\\%,1\\%)$"),
                         ("A_fee0.01_BdC_FEE0.03", "pound seller, $(3\\%,1\\%)$"),
                         ("B_fee0", "dollar seller, $(2\\%,0)$"), ("B_fee0.01", "dollar seller, $(2\\%,1\\%)$"),
                         ("B_fee0.01_BdC_FEE0.03", "dollar seller, $(3\\%,1\\%)$")):
            r = D["cotrain"].get(key)
            if r is None:
                continue
            thr = next((x for x in r["threshold"] if x == x), float("nan"))
            rows.append(f"{lab} & ${fnum(1e4 * r['mm'], 1, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ & "
                        f"{_pm(r['edge'], r['edge_se'])} & {fpct_small(r['filled'])} & {fnum(thr, 4)} & "
                        f"{fpct(r['agree'], 1)} \\\\")
        write("tab_cotrain", rows)
    if "mm_phase1" in D:
        rows = []
        for key, lab in (("A0", "pound seller, $g=0$"), ("A0.01", "pound seller, $g=1\\%$"),
                         ("B0", "dollar seller, $g=0$"), ("B0.01", "dollar seller, $g=1\\%$")):
            r = D["mm_phase1"].get(key)
            if r is None:
                continue
            rows.append(f"{lab} & ${fnum(1e4 * r['mm'], 2, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ & "
                        f"{r['best'].replace('+', ' + ')} & "
                        f"${fnum(1e4 * max(r['rule_pl']), 2, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ & "
                        f"{_bp_pm(r['diff'], r['diff_se'])} & "
                        f"{r['disagree']:,} of {r['offers']:,} \\\\")
        write("tab_mmnet", rows)
    # ---- Section 7 tables: numbers the text used to list, regrouped (same data as the macros)
    def _bp1(x, sign=True):
        return fnum(1e4 * x, 1, sign=sign).replace(chr(92) + "ensuremath{-}", "$-$")

    def _share(x):
        return fpct(x, 1) if x >= 1e-3 else "$<$0.01\\%" if x < 1e-4 else fpct_small(x)

    def _gain(x):
        v = 1e4 * x
        return "0.0" if abs(v) < 0.05 else fnum(v, 1, sign=True).replace(chr(92) + "ensuremath{-}", "-")

    if "learning" in D and "fee_dp" in D:
        # last ten blocks of co-training on the validation games: range, final block, noise (median 2 s.e.)
        L, last = D["learning"], 10
        rng = lambda e: f"{_bp1(e.min())} to {_bp1(e.max())}"
        rows = []
        for side, who in (("A", "pound seller"), ("B", "dollar seller")):
            for tag, lab, f, g in (("fee0", "$(2\\%,0)$", 0.02, 0.0),
                                   ("fee0.01_BdC_FEE0.03", "$(3\\%,1\\%)$", 0.03, 0.01)):
                r = L["one"].get(f"{side}_{tag}")
                if r is None:
                    continue
                e = np.asarray(r["edge"][-last:])
                se = np.median(np.asarray(r["edge_2se"][-last:]))
                dp = D["fee_dp"][f"{side} f={f:g} g={g:g}"]["dp_edge"]
                rows.append(f"{who}, {lab} & {rng(e)} & {_bp1(e[-1])} & {_bp1(se, False)} & "
                            f"{fnum(1e4 * dp, 2, sign=True).replace(chr(92) + 'ensuremath{-}', '$-$')} \\\\")
        write("tab_learnrange1", rows)
        four = L["four"]
        ed = np.asarray(four["edge"])
        rows = []
        for i, name in enumerate(four["names"]):
            e = ed[-last:, i]
            se = np.median(np.asarray(four["edge_2se"])[-last:, i])
            who = "pound seller" if name in ("T1", "T3") else "dollar seller"
            rows.append(f"{name}, {who}, edge & {rng(e)} & {_bp1(e[-1])} & {_bp1(se, False)} \\\\")
        mm = 1e4 * np.asarray(four["mm"])[-last:]
        mse = 1e4 * np.median(np.asarray(four["mm_2se"])[-last:])
        f0 = lambda v: fnum(v, 0, sign=True).replace(chr(92) + "ensuremath{-}", "$-$")
        rows.append(f"market maker, P/L & {f0(mm.min())} to {f0(mm.max())} & {f0(mm[-1])} & {fnum(mse, 0)} \\\\")
        write("tab_learnrange4", rows)
    if "arms" in D:
        rows = []
        for f, sp in ((0.02, "1\\%"), (0.03, "2\\%")):
            for arm, lab in (("bid", "bid-wise (the game)"), ("turn", "turn-wise"), ("round", "round-wise")):
                r = D["arms"].get(f"{arm}{f:g}")
                if r is None:
                    continue
                rows.append(f"{sp} & {lab} & {_share(r['mm_trades'])} & {_share(r['netted'])} & "
                            f"${fnum(1e4 * r['pl']['mm_pl'], 1, sign=True).replace(chr(92) + 'ensuremath{-}', '-')}$ \\\\")
        write("tab_arms", rows)
    if "mechanism" in D:
        rows = []
        for fs in ("0.025", "0.03", "0.04", "0.05", "0.06", "0.07"):
            r = D["mechanism"].get(fs)
            if r is None:
                continue
            sp = 100 * (float(fs) - G_MM)
            q = lambda v: "--" if v is None else fpct(v, 1, sign=True)
            rows.append(f"{sp:g}\\% & {q(r['quote_pound'][1])} & {q(r['quote_dollar'][1])} & "
                        f"{q(r['quote_pound_filled'])} & {q(r['quote_dollar_filled'])} & "
                        f"{fpct(r['p_pound'], 1)} & {fpct(r['p_dollar'], 1)} & {fpct_small(r['p_both'])} \\\\")
        write("tab_fills", rows)
    if "ladder" in D:
        rows = []
        for fs in sorted(D["ladder"], key=float):
            c = D["ladder"][fs]["pl"]["cards"]
            sp = 100 * (float(fs) - G_MM)
            cells = [f"${_gain(c[t]['gain'])} \\pm "
                     f"{fnum(2e4 * c[t]['gain_se'], 1)}$" for t in ("T1", "T2", "T3", "T4")]
            rows.append(f"{sp:g}\\% & " + " & ".join(cells) + " \\\\")
        write("tab_gains", rows)
    print(f"  tables {', '.join(made)}")


# =============================================================================
# 4. writing numbers.tex / numbers.json
# =============================================================================
def save_part(name, N, D):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"part_{name}_numbers.json").write_text(json.dumps(N.items, indent=1, default=float))
    (CACHE / f"part_{name}_data.json").write_text(json.dumps(D, default=float))


def load_parts():
    items, data = {}, {}
    for name in ("model", "runs"):
        fn, fd = CACHE / f"part_{name}_numbers.json", CACHE / f"part_{name}_data.json"
        if fn.exists():
            items.update(json.loads(fn.read_text()))
        if fd.exists():
            data.update(json.loads(fd.read_text()))
    return items, data


def derived_numbers(items, data):
    """Numbers computed from both parts' cached data, after loading."""
    def put(key, value, text, source):
        items[key] = dict(value=value, text=text, source=source)
    lad = data.get("ladder", {})
    if lad:
        sums = [r["sum_check"] for r in lad.values()]
        src = "ladder runs: MM trades + traders to Bureau + left, valued at trade rates"
        put("lad.summin", min(sums), fpct(min(sums), 1), src + ", min over spreads")
        put("lad.summax", max(sums), fpct(max(sums), 1), src + ", max over spreads")
    fee_dp = data.get("fee_dp", {})
    for key, r in fee_dp.items():
        side, f, g = key.split()
        f, g = float(f[2:]), float(g[2:])
        t = f"dp.{side}.f{round(100 * f)}g{round(100 * g)}"
        if r["dp_edge"] > 2 * r["dp_edge_se"]:
            capt = r["net_edge"] / r["dp_edge"]
            put(t + ".capt", capt, fpct(capt, 0), "network edge / DP edge, where the DP edge is significant")
    from scipy.stats import norm
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    f, g = F_TRADER, G_MM
    # Remark 5.6, direct trade between a pound seller and a dollar seller in one round, each scored in
    # its starting currency at the revealed rate (the rules' numeraire): a pound seller prefers a sure P
    # if P E[1/X] > 1 - f; a dollar seller if a0 / P > 1 - f.
    if "kappa.one" in items:
        k1 = items["kappa.one"]["value"]
        lo_b, hi_b = (1 - f) / k1, A0 / (1 - f)
        src = "derived from kappa.one = E[1/X] (one step, quadrature within 8 sd)"
        put("band.lo", lo_b, fnum(lo_b, 4), "(1-f) / E[1/X], " + src)
        put("band.hi", hi_b, fnum(hi_b, 4), "a0 / (1-f)")
        put("band.width", (hi_b - lo_b) / A0, fpct((hi_b - lo_b) / A0, 1), "(hi - lo) / a0")
        put("band.lonaive", (1 - f) * A0, fnum((1 - f) * A0, 4), "(1-f) a0, the pound seller's level without the premium")
    # Section 4.1 numeraire: the one-offer problem scored in pounds at the revealed rate (the rules'
    # numeraire for a pound seller) instead of in dollars; T1-type seller, (f, g) = (2%, 0).
    lo, hi = A0 - 8 * SD, A0 + 8 * SD

    def pounds_value(P):
        acc = quad(lambda x: P / x * norm.pdf(x, A0, SD), P, hi, epsabs=1e-14, limit=200)[0]
        return acc + (1 - f) * (norm.cdf((P - A0) / SD) - norm.cdf((lo - A0) / SD))

    def dollars_value(P):
        z = (P - A0) / SD
        return P * (1 - norm.cdf(z)) + (1 - f) * (A0 * norm.cdf(z) - SD * norm.pdf(z))
    rp = minimize_scalar(lambda P: -pounds_value(P), bounds=(A0, A0 + 4 * SD), method="bounded",
                         options={"xatol": 1e-10})
    rd = minimize_scalar(lambda P: -dollars_value(P), bounds=(A0, A0 + 4 * SD), method="bounded",
                         options={"xatol": 1e-10})
    Pp, Pd = rp.x, rd.x
    floor_p = (1 - f) * (1 - norm.cdf((lo - A0) / SD))
    src = "one offer, T1 side, (2%,0): optimum of the pound-scored value (quadrature within 8 sd), bounded Brent"
    put("num.Pstar", Pp, fnum(Pp, 4), src)
    put("num.dz", (Pp - Pd) / SD, fnum((Pp - Pd) / SD, 3, sign=True), "pound-scored minus dollar-scored optimum, in sd")
    put("num.fill", 1 - norm.cdf((Pp - A0) / SD), fpct(1 - norm.cdf((Pp - A0) / SD), 2), "fill probability at the pound-scored optimum")
    ep = pounds_value(Pp) - floor_p
    ed = (dollars_value(Pd) - (1 - f) * A0) / A0
    put("num.edgebp", ep, fnum(1e4 * ep, 2), "pound-scored edge over the Bureau, bp of the pound offered")
    put("num.edgedbp", ed, fnum(1e4 * ed, 2), "dollar-scored edge over the Bureau / a0, bp of the pound offered")
    # Appendix C: offers outside the DP's threshold grid (+-4 sd of the anchor). Such an offer beats the
    # coupled Bureau conversion only if the rate lands in the fee band next to its threshold; over anchors
    # within the DP's anchor range (+-8 sd of a0), at (2%,1%), the band lies at least this far out:
    zgrid, fp = 4.0, (f - g) / (1 - g)
    worst = 0.0
    for a in np.linspace(A0 - 8 * SD, A0 + 8 * SD, 161):
        xs = a + zgrid * SD          # threshold at the top of the grid
        xl = a - zgrid * SD          # threshold at the bottom
        # pound seller: fill iff X > x, beats the Bureau iff X < x (1-g)/(1-f)
        pA = max(norm.sf(zgrid), norm.cdf((xl * (1 - g) / (1 - f) - a) / SD))
        # dollar seller: fill iff X < x, beats the Bureau iff X > x (1-f)/(1-g)
        pB = max(norm.sf((xs * (1 - f) / (1 - g) - a) / SD), norm.cdf(-zgrid))
        worst = max(worst, pA, pB)
    gain = (f - g) * (1 + 0.03) * 1.01     # fee spread per unit, times the largest (1 + beta), times a margin
    bound = 4 * worst * gain
    put("grid.offp", worst, fsci(worst, 1), "largest per-round probability that an off-grid offer's fill beats the Bureau, (2%,1%)")
    put("grid.offbound", bound, fnum(100 * bound, 4), "4 rounds x offp x (f-g)(1+beta_max) x 1.01, points of capital")
    # the open channel without the market maker's fee, in bp (abstract, Section 5.4)
    if "dp.A.f2g0.edge" in items:
        v = items["dp.A.f2g0.edge"]["value"]
        put("dp.A.f2g0.edgebp", v, fnum(1e4 * v, 1), "dp.A.f2g0.edge in bp of capital")
    for side in ("A", "B"):
        k = f"dp.{side}.f2g1.edge"
        if k in items and k + "se" in items:
            hi = items[k]["value"] + items[k + "se"]["value"]
            put(f"dp.{side}.f2g1.hibp", hi, fnum(1e4 * hi, 2), "edge + 2 s.e. at (2%,1%), bp of capital")
    # units: the four-trader gains of Section 7.5 in points (they are stored in bp)
    for t in ("T1", "T2", "T3", "T4"):
        k = f"lad.s20.gain{t}"
        if k in items:
            v = items[k]["value"]
            put(k + "pt", v, fnum(100 * v, 3, sign=True), items[k]["source"].replace(", bp", ", points"))
    for k, sg in (("lad.s60.gainT4", True), ("lad.s60.gainseT4", False)):
        if k in items:
            v = items[k]["value"]
            put(k + "pt", v, fnum(100 * v, 3, sign=sg), items[k]["source"] + " (points)")
    # the frozen control of Section 7.6: fresh traders' gains against the co-trained market maker
    fz = data.get("population", {}).get("frozen", {}).get("pl", {}).get("cards", {})
    co = data.get("ladder", {}).get("0.03", {}).get("pl", {}).get("cards", {})
    if fz and co:
        for t in ("T1", "T2", "T3", "T4"):
            put(f"pop.frozen.gain{t}pt", fz[t]["gain"], fnum(100 * fz[t]["gain"], 3, sign=True),
                "population runs, frozen control: fresh trader's gain over Bureau-only, points")
        put("pop.frozen.allhigher", all(fz[t]["gain"] >= co[t]["gain"] for t in fz),
            "yes" if all(fz[t]["gain"] >= co[t]["gain"] for t in fz) else "no",
            "every fresh trader gains at least as much as the co-trained one, against the same MM")
    put("p.a0sd", A0 / SD, fnum(A0 / SD, 0), "a0 / sd")
    if "grid.ahalf" in items:
        h = items["grid.ahalf"]["value"]
        po = 2 * norm.sf(h / 2.0)          # a4 - a0 has sd 2 sd after four steps
        e = math.floor(math.log10(po))
        put("grid.anchorout", po, "\\ensuremath{%.0f\\times 10^{%d}}" % (po / 10 ** e, e),
            "Pr(|a4 - a0| > ahalf sd), four Gaussian steps")
    # display: two-sided mass beyond 8 sd with two significant figures (1.2e-15, not "1e-15")
    if "neg.cut8" in items:
        v = items["neg.cut8"]["value"]
        e = math.floor(math.log10(v))
        put("neg.cut8", v, "\\ensuremath{%.1f\\times 10^{%d}}" % (v / 10 ** e, e),
            items["neg.cut8"]["source"] + " (two-sided, 2 s.f.)")
    # Section 7.5: fees saved on the market maker's flow at a 2% spread, against the gains
    lad2 = data.get("ladder", {}).get("0.03")
    cards = data.get("cards")
    if lad2 and cards:
        saved = lad2["mm_trades"] * 0.03 - lad2["mm_bureau"] * G_MM
        put("lad.s20.feesaved", saved, fnum(1e4 * saved, 0),
            "MM trades x f - MM to Bureau x g, (f, g) = (3%, 1%), bp of the game's capital")
        cap = sum(c["L"] if c["side"] == "A" else c["L"] / A0 for c in cards)
        tg = sum(lad2["pl"]["cards"][c["name"]]["gain"] * (c["L"] if c["side"] == "A" else c["L"] / A0)
                 for c in cards) / cap
        put("lad.s20.tradersbp", tg, fnum(1e4 * tg, 0),
            "sum of the four traders' gains over Bureau-only, in pounds at a0, bp of the game's capital")
    # netting of the three market-maker designs at a 2% spread
    arms = data.get("arms", {})
    for d in ("bid", "turn", "round"):
        r = arms.get(f"{d}0.03")
        if r is not None:
            put(f"arm.{d}f3.netted", r["netted"], fpct(r["netted"], 1), "arms run, netted share of the money")
    groups = data.get("curve_groups", {})
    for f, tag in (("0.02", "s10"), ("0.03", "s20")):
        g = groups.get(f, [])
        nt = sum(1 for x in g if x["name"].startswith("T1-T2-T3-T4"))
        npop = sum(1 for x in g if x["name"].startswith("pop"))
        put(f"curve.{tag}.ncards", nt, str(nt), "runs on T1-T4 in the curve group")
        put(f"curve.{tag}.npop", npop, str(npop), "runs on random tables in the curve group")
        put(f"curve.{tag}.npop4", sum(1 for x in g if x["name"].startswith("pop4")),
            str(sum(1 for x in g if x["name"].startswith("pop4"))), "random four-trader tables")


# Differences of P/L (edges, gaps, their standard errors) are quoted in the text in basis points of
# capital. Their registry entries are written in percentage points above; this pass moves the decimal
# point two places, so every digit is unchanged and the value is exactly 100 times the points figure.
BP_KEYS = (
    ["floor.perround", "floorB.perround", "grid.offbound", "gridchk.cdB.diff", "gridchk.cdB.diffse",
     "dp.A.maxreg", "dp.B.regmin", "dp.B.regmax", "dp.A.f2g0.reg", "dp.A.f2g0.regse",
     "dp.A.f2g0.gap", "dp.A.f2g0.gapfa", "dp.B.f2g0.gapfa", "dp.A.f2g1.net", "dp.A.f2g1.netse",
     "dp.B.f3g1.net", "dp.B.f3g1.netse", "ppo.R1.prize", "ppo.R4.prize",
     "refine.gap13", "refine.gap25", "refine.gap49",
     "lad.s20.gainT1pt", "lad.s20.gainT2pt", "lad.s20.gainT3pt", "lad.s20.gainT4pt",
     "lad.s60.gainT4pt", "lad.s60.gainseT4pt"]
    + [f"dp.{s}.f{f}g{g}.{q}" for s in "AB" for f in (2, 3) for g in (0, 1) for q in ("edge", "edgese")]
    + [f"co.{k}.{q}" for k in ("Ag0", "Ag1", "Ag1f3", "Bg0", "Bg1", "Bg1f3") for q in ("edge", "edgese")]
    + [f"learn.{k}.{q}" for k in ("Ag0", "Ag1", "Ag1f3", "Bg0", "Bg1", "Bg1f3") for q in ("lo", "hi")]
    + [f"learn4.T{i}.{q}" for i in range(1, 5) for q in ("lo", "hi")]
    + [f"k.T{i}.K{K}g1.{q}" for i in (1, 2) for K in (2, 3) for q in ("edge", "edgese")])


def _shift2(text):
    """'+0.0258' -> '+2.58', '0.18' -> '18', '\\ensuremath{-}0.0004' -> '\\ensuremath{-}0.04'"""
    neg = "\\ensuremath{-}"
    pre = ""
    if text.startswith(neg):
        pre, text = neg, text[len(neg):]
    elif text[:1] in "+-":
        pre, text = text[0], text[1:]
    assert re.fullmatch(r"\d+\.\d+", text), text
    i, f = text.split(".")
    f = f + "00"[: max(0, 2 - len(f))]
    i, f = (i + f[:2]).lstrip("0") or "0", f[2:]
    return pre + i + ("." + f if f else "")


def to_bp(items):
    for k in BP_KEYS:
        if k in items:
            it = items[k]
            it["text"] = _shift2(it["text"])
            it["source"] = it["source"].replace("points", "bp") + " [shown in bp of capital]"


def write_numbers(items):
    lines = ["% numbers.tex -- written by make_assets.py; do not edit by hand.",
             "% Every value below is computed from the model code or read from a saved run;",
             "% numbers.json says which function and which file.", ""]
    for key in sorted(items):
        lines.append(f"\\expandafter\\def\\csname val@{key}\\endcsname{{{items[key]['text']}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    (PAPER / "numbers.json").write_text(json.dumps(
        {k: dict(value=v["value"], text=v["text"], source=v["source"]) for k, v in sorted(items.items())},
        indent=1, default=float))
    print(f"  wrote numbers.tex ({len(items)} values)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("model", "runs", "figures"), default=None)
    args = ap.parse_args()
    t0 = time.time()
    if args.only in (None, "model"):
        N, D = Numbers(), {}
        run_model(N, D)
        save_part("model", N, D)
    if args.only in (None, "runs"):
        N, D = Numbers(), {}
        run_runs(N, D)
        save_part("runs", N, D)
    items, data = load_parts()
    derived_numbers(items, data)
    to_bp(items)
    write_numbers(items)
    make_tables(data)
    if args.only in (None, "figures"):
        make_figures(data)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
