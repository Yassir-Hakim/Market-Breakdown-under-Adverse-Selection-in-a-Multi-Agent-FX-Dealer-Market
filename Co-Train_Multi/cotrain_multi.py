import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "NN_one_offer_game"      # as cotrain.py; edit these names
_PHASE1 = _ROOT / "MM_Phase_1"           # if a folder is ever renamed
assert (_CORE / "rl_env.py").exists(), f"core folder not found: {_CORE}"
assert (_PHASE1 / "mm_train.py").exists(), f"Phase 1 folder not found: {_PHASE1}"
sys.path.insert(0, str(_CORE))           # the certified machinery layer
sys.path.insert(0, str(_ROOT))           # Mechanics/, DP_Models/
sys.path.insert(0, str(_PHASE1))         # mm_train and train_traders import
                                         # each other by bare name
"""
  cotrain_multi.py -- N TRADERS and ONE MARKET MAKER learn against each
  other by co-training, in the four-round foreign-exchange game set out in
  the rules document.

THE GAME
  Each trader holds a card: a starting balance L in one currency, a target
  balance T in the other, a penalty A on whatever start currency is left at
  the end, and a penalty B on any shortfall against the target. Side "A"
  starts in pounds and wants dollars; side "B" starts in dollars and wants
  pounds. The true rate X is a random walk that only the market maker sees
  during a round; it is revealed to everyone at the end of the round.

  Each round has two phases.

  OFFER PHASE. Every trader presents ONE offer to the market maker, at a
  price of its choosing, for a fraction of its own balance. The offers
  arrive in a RANDOM order, reshuffled every round and every game: the
  rules give "T2, T3, T4, T1" only as an example ordering, so no trader
  owns a slot. The market maker accepts or rejects each one.

  BUREAU PHASE. The rate is revealed. Each trader may convert at the
  Bureau de Change -- the exchange of last resort -- paying fee f on the
  sale currency. The market maker may then use the Bureau too, at its own
  lower fee g. After the fourth round a fifth rate is drawn, the market
  maker must clear any negative position at it, and everyone is scored.

  Traders see only their own balances, their own offer and verdict, and the
  revealed rate. They never see each other, and are coupled ONLY through
  the market maker's prices and inventory. That coupling is the point of
  the game: the market maker can cover what it owes one trader with a fill
  from a trader on the other side, instead of paying the Bureau. With a
  single trader it cannot, and it can only pass its own fee through.

WHAT THE MARKET MAKER SEES
  Always: how much of the game is left, which phase it is in, the true
  rate, and both legs of its own book. Then, gated on the configuration:
    - the offer's SIZE (SEE_OFFER_SIZE, on by default). Matching offers
      against each other depends on their sizes, so an MM that is meant to
      net flow must see them. Every architecture gets it, so that a
      comparison between architectures is not confounded by one of them
      being blind to size.
    - the offer's DIRECTION, +1 for a pound seller and -1 for a dollar
      seller. Present only when the game can contain both, because a
      column that is constant carries no information and would only
      perturb the initialisation.
    - how many offers of the round are still to come, present only in the
      architecture that decides offer by offer (below) with more than one
      offer a round.
    - the FLOW BLOCK, the rest of the round's offers (below).
  It never sees a trader's identity, card, book or history. Two offers that
  look the same to it ARE the same to it.

THE MARKET MAKER'S OUTPUT IS A PRICE, NOT A YES/NO
  The network emits a LEVEL, and the offer is filled when it is on the
  right side of that level. This is not a restriction on what it can
  express. An accept/reject rule that is optimal in this game is monotone
  in the offer's own price -- the market maker knows the true rate, so a
  cheaper pound is always better for it whatever it does with the rest of
  the round -- and a monotone accept/reject rule IS a cut. Emitting the cut
  keeps several things a free yes/no head would have to relearn:
  monotonicity, a fill ramp that anneals to a hard verdict, an
  initialisation that starts by rejecting everything (so the learning curve
  starts at zero), a quote that can be read off and plotted, and a gradient
  path from the trader's price through a structural ramp rather than
  through a learned slope.

MM_ARCH -- WHEN THE MARKET MAKER DECIDES
  The central experiment of this file. The same game, the same paths and
  the same schedule are run under three decision architectures, and read
  side by side (cotrain_multi_compare.py):

    "bid"   One offer at a time. The level is re-set for every offer from
            the book AS IT STANDS at that moment, and the market maker also
            sees how many offers are still to come. The first offer of a
            round is decided blind. THIS IS THE REAL GAME: in the room, a
            market maker is handed one slip at a time, in a random order,
            and must answer before seeing the next.

    "turn"  The whole round at once. Before any verdict the market maker
            reads the FLOW BLOCK -- every offer of the round -- and decides
            against the book as it stood when the round opened. Its
            verdicts are therefore concurrent and cannot depend on the
            arrival order. This is NOT the real game; it is the ceiling on
            what within-round lookahead could be worth, and the point of
            measuring it is to find out whether the real game gives
            anything up.

    "round" One level per round per direction, fixed from the book as the
            round opens and applied to every offer as it arrives. No
            lookahead and no live book, so the arrival order cannot change
            a verdict. The simplest of the three and the reference point
            the other two are measured against.

  The hypothesis under test is that "bid" and "turn" wash out: with enough
  volume and a reward computed only at the end of the game, a market maker
  deciding one offer at a time should learn the same policy as one that
  sees the whole round, and should settle on accepting at the true rate
  plus some fraction of the Bureau fee it expects to save. The three arms
  differ ONLY in when the decision is made and on what information, so a
  difference between them is a difference in architecture and nothing else.

THE FLOW BLOCK ("turn" only)
  Every offer of the round, the one being judged included. The level may
  therefore depend on that offer's own price, so the policy class on this
  architecture is ANY accept/reject rule, not just a cut. That is the
  turn-wise market maker as specified: it reads all N bids and decides.

  Layout of the block: one pair of numbers per offer, sorted best-for-the-
  market-maker first, so no column is a slot or an identity. One-way, the
  pairs are (price relative to the true rate, size) grouped by direction.
  Two-way, where an offer's direction is not fixed by the table, the layout
  is direction-free: (edge, signed size), where edge = d(X - p)/X is the
  market maker's gain per pound of that offer and the sign of the size says
  whether the trade leaves it holding more pounds or fewer.

TWO_WAY -- THE TRADER CHOOSES ITS DIRECTION
  The rules say traders "do not have to trade in one direction, they can
  buy and sell either currency provided they never go into debt". With
  TWO_WAY on, each trader still makes ONE offer a round, but chooses which
  way it goes: its fraction output is read signed, s = 2f - 1 in (-1, 1).
    s > 0   sell that share of the balance it still HOLDS -- the forward
            trade, towards its target.
    s < 0   sell that share of what it has already BANKED, back again.
    s = 0   no offer.
  A side-A trader's reverse offer is therefore a dollar sale, and a side-B
  trader's a pound sale. Never in debt: an offer is always a fraction of a
  balance the trader actually has, so it cannot sell what it does not hold.
  An offer's direction becomes DATA, varying per game and per round, and
  travels with the offer through the market maker's direction input, the
  flow block, the book rule and the record.

  The trader's Bureau trade stays ONE-WAY, towards the target. Reversing
  there would pay the fee to undo a conversion the trader has to redo, and
  the only motive for doing so -- betting on the rate -- has zero
  expectation under a random walk. Reverse trading is available where it
  can actually pay: against the market maker, which may quote better than
  the Bureau.

THE BOOK RULE -- THE BENCHMARK FOR "DID IT LEARN TO NET?"
  Alongside the four simple hardcoded market makers (accept at the true
  rate or at the fee-adjusted rate, then either clear the book every round
  or carry it to the end) there is a fifth and sixth: the BOOK RULE, in
  both clearing flavours. Each round it reads every offer of the round and
  its own book, and accepts the SUBSET of offers that maximises what its
  book would be worth in pounds if every negative position were covered at
  the Bureau there and then.

  That is limit-order-book logic written down: a pound seller and a dollar
  seller whose prices cross can be put together, existing inventory nets
  against the new flow, and only the residual has to meet the Bureau's fee.
  With one offer on the table it reduces exactly to the fee-adjusted rule;
  with offers on both sides it accepts pairs that rule would reject. It is
  deliberately myopic -- one round at a time, no view of the future -- so a
  learned market maker can beat it by carrying a position, and the gap
  between the two is the measurement.

WHAT IS BEING MEASURED
  With one trader and a market maker fee of 1%, co-training ends in
  no-trade on both sides. The trader's window is about 1% of the rate wide
  against a per-round move of about 4%, so adverse selection swallows it
  and the result is a zero-gradient deadlock. With traders on both sides
  the market maker can cover one fill with another instead of paying the
  Bureau, so its own fee leaves the trader's window, which then runs on the
  full spread between the two Bureau rates.

  So: does two-sided flow revive the channel, and does the market maker's
  spread become NARROWER than twice its fee? Read off the record, never off
  the learner: the fitted acceptance cut per direction, the share of filled
  volume that still went to the Bureau (one minus that is the share it
  crossed or carried), each trader's edge over always-using-the-Bureau
  paired on the same paths, and the pair against the single-trader run on
  the SAME evaluation paths. The control is a same-direction table: if that
  revives too, the revival is flow escaping the deadlock rather than
  genuine crossing.

WHAT IS TAKEN FROM THE SINGLE-TRADER FILES, AND WHY
  - Block structure, budget, learning rates, the tau ramps, the plateau
    decay, the level initialisation and the seeds: the game each individual
    agent plays is unchanged, so its training schedule is too. With one
    trader this file's block IS the single-trader block.
  - The traders' block updates ALL traders at once, each on its own P/L
    with the market maker frozen. They never see each other, so a
    simultaneous step is well defined.
  - The gated input columns above exist only where they carry information.
    A column that is constant in a given configuration would do nothing but
    change the initialisation, and would break the exact reductions the
    checks rely on for no gain.
  - Negative rates: a random walk can in principle go through zero. Any
    path that does is deleted and the deletion counted, and the table of
    how often that happens across volatilities is printed with every run.
    At the volatility used here it never fires.

CAPITAL. Everything the market maker sees or scores is expressed as a
  fraction of the TOTAL capital in the game, in pounds at the opening rate
  with no fee, summed over the traders present. One constant per
  configuration, so its book and its P/L are comparable across tables of
  different sizes.

DISTRIBUTION KNOWLEDGE. No agent is told the distribution of the rate.
  They see realised moves only.

CHECKS -- all run before any training, and all must pass
   0  the single-trader run's saved pair, replayed here with one trader,
      reproduces that run's per-path results.
   1  with one trader this file's game IS the single-trader engine's game,
      to the bit: same P/Ls and same record, on fresh and random networks,
      hard and soft, under every hardcoded rule.
   2  books conserve: every trader's gain is the market maker's loss, every
      verdict is a clean accept or reject, every Bureau trade moves both
      legs at the fee-adjusted rate, no Bureau trade deepens a negative
      position or turns a positive leg negative, the final clearing leaves
      nothing coverable, and each round's arrival order is a permutation.
   3  the training seam: the market maker's step moves only the market
      maker, the traders' simultaneous step moves each trader by its own
      gradient and nothing else, and the frozen partner still passes
      gradient through.
   4  the arrival order is uniform over permutations.
   5  the always-use-the-Bureau floor, per path, matches the analytic value
      for each card.
   6  with one offer a round, every architecture is the "round"
      architecture to the bit.
   7  with more than one offer a round, every architecture with its extra
      input columns zero-weighted is the "round" architecture to the bit,
      and those columns do carry gradient.
   8  verdicts are order-free where the architecture says they must be, and
      order-dependent where it says they must be.
   9  the flow block is laid out as documented: one pair per offer, sorted
      best-for-the-market-maker first, every column live.
  10  the book rule never scores below the simple rules on a round, is
      strictly better on some, reduces to the fee-adjusted rule with one
      offer on the table, and takes a hand-checked crossing pair.
  11  a two-way trader pinned to trade forward plays the one-way game to
      the bit, and the reverse branch is live: pinned the other way its
      offers flip direction, fill, move the P/L and carry gradient.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

import MM_Phase_1.mm_train as M                       # the MM's net, the rules,
                                              # Bureau, paired, thresholds
import NN_one_offer_game.torch_train as T     # the trader's net
from NN_one_offer_game.rl_diagnostics import dump_all_pl   # always-BdC per
                                              # path, closed form, certified
from DP_Models.v2_multiple_rounds import bdc_baseline      # the analytic floor
from MM_Phase_1.train_traders import CARD
from NN_one_offer_game.rl_env import GameSpec
from Mechanics.fx_mechanics import GameParams, results_path, BDC_FEE_DEFAULT

# ============================ SETTINGS ======================================
# THE TRADERS IN THE GAME -- one row each; add, remove or edit rows freely.
# N is the length of this list and the arrival order is reshuffled through
# it every round and every game. A row equal to one of the cards the
# single-trader runs used (T1 = card A, T2 = card B) also gets the replay
# check and the paired single-trader baseline; any other row is a new
# trader and gets everything else.
#           name   side   L (start)   T (target)   A       B
TRADERS = [
    dict(name="T1", side="A", L=100_000, T=125_000, A=0.02, B=0.03),  # pound seller; the single-trader runs' card A
    dict(name="T2", side="B", L=200_000, T=155_000, A=0.03, B=0.02),
    dict(name="T3", side="A", L=50_000, T=63_000, A=0.20, B=0.0),    
    dict(name="T4", side="B", L=500_000, T=396_000, A=0.0, B=0.20),   # dollar seller; the single-trader runs' card B
]
# the one market every trader shares (the single-trader runs' numbers)
ROUNDS = 4
A0 = 1.25
SD = 0.05
MM_FEE = 0.01           # passed to EVERY rollout, never read from M.MM_FEE
TRADER_FEE = 0.03   # the TRADERS' Bureau fee, set HERE rather
                        # than by editing Mechanics/fx_mechanics.py, so a sweep
                        # over it touches nothing else. BDC_FEE_DEFAULT (0.02) is
                        # the rules' fee; the game's window is governed by
                        # rho = [(f - g)/(1 - g)] / (sd/a0), so raising f widens it
N_POPULATION = 0        # 0: the game is TRADERS as listed. n > 0: the game is
                        # n traders drawn at random (random_roster below), the
                        # co-training runs against THEM, and afterwards the
                        # market maker is frozen and the TRADERS table trains
                        # against it from fresh -- what the market maker
                        # learned from a large random population, read on the
                        # table of interest
ROSTER_SEED = 0         # which random table; part of the run folder's name,
                        # so two tables can never overwrite each other
ROSTER_L = (50_000, 100_000, 200_000, 500_000)   # start balances drawn from
                        # (the four cards' values)
ROSTER_TARGET_RANGE = (0.90, 1.10)   # T = par x u, u uniform in this range,
                        # where par converts all of L at a0 with no fee: u is
                        # how demanding the card is (the cards sit at 0.97-1.01)
ROSTER_CARD_FEES = (0.0, 0.02, 0.03, 0.05, 0.10, 0.20)  # each card's own A and
                        # B penalties drawn from (the cards span 0 to 0.20)
BOOK_RULE_MAX_OFFERS = 8    # the book rule enumerates every accept/reject
                        # subset of a round (2^offers) as (games, 2^offers)
                        # matrices, so past this many offers it is left out
                        # of the benchmark rules and its check is skipped:
                        # at the full evaluation paths those matrices are
                        # gigabytes from 9 offers up
N_OUTER = 20            # as cotrain.py: one outer step = one block per agent
N_INNER = 100
FIRST = "mm"            # who updates first; "trader" needs MM_LEVEL_INIT random
MM_LEVEL_INIT = "conservative"   # "conservative" | "random" as cotrain.py, or
                        # "naive": the level starts AT the true rate in both
                        # directions (the fee-0 rule, zero spread), so the
                        # game opens with flow on both sides and the MM
                        # learns whether the fee makes it widen -- or cross
BATCH = 1000
ACCUM_GAMES = 0         # 0: each inner step backprops its BATCH games in ONE
                        # graph, as always. n > 0: the same games, backpropped
                        # in chunks of n with the gradients accumulated -- the
                        # SAME gradient up to floating-point summation order
                        # (the thread-count caveat in run_arms.py), for a
                        # fraction of the memory. The graph holds every offer
                        # decision of every game, so it grows with BATCH x N:
                        # at large N one full-batch graph is gigabytes, and
                        # chunking is what makes a big table trainable
LR_TRADER = 1e-3
LR_MM = 1e-4
TRADER_PLATEAU = True   # per trader, as cotrain.py (re-derived there)
TAU_TRADER = (1.0, 0.02)
TAU_MM = (0.2, 0.02)
TAU_ANNEAL_FRAC = 0.35
TAU_ANNEAL_STEPS = None  # None: the anneal takes TAU_ANNEAL_FRAC of each agent's steps, so a
                         # longer schedule (more outer steps) also anneals more slowly. A number:
                         # the anneal takes that many steps whatever N_OUTER is, and any extra
                         # outer steps are spent at the final tau (700 is the 20 x 100 schedule's)
CLIP_NORM = 1.0
VAL_PATHS = 20_000
EVAL_PATHS = 200_000    # cotrain.py's value: the baseline comparison is paired
RECORD_PATHS = 20_000
CONVERGED_BLOCKS = 4
FLAT_TOL_MM = 2e-6
FLAT_TOL_TRADER = 5e-6
MM_ARCH = "bid"        # WHEN the market maker decides. The central experiment:
                        # the same game, paths and schedule under three
                        # architectures, read side by side.
                        # "bid":   one offer at a time. The level is re-set for
                        #          every offer from the book AS IT STANDS, and the
                        #          MM sees how many offers are still to come; the
                        #          first offer of a round is decided blind. THIS IS
                        #          THE REAL GAME -- in the room the MM is handed
                        #          one slip at a time and must answer before seeing
                        #          the next. With one offer a round it is "round"
                        # "turn":  the whole round at once. Before any verdict the
                        #          MM reads every offer of the round (the flow
                        #          block) and decides against the book as it stood
                        #          when the round OPENED, so its verdicts are
                        #          concurrent and the arrival order cannot change
                        #          them. Not the real game: the ceiling on what
                        #          within-round lookahead could be worth
                        # "round": one level per round per direction, fixed from
                        #          the book as the round opens and applied to every
                        #          offer as it arrives. No lookahead, no live book.
                        #          The simplest of the three and the reference the
                        #          other two are measured against
SEE_OFFER_SIZE = True   # the MM sees the size of the offer it is judging (and,
                        # on the "turn" arm, every other offer's too). Matching
                        # offers against each other depends on their sizes, so an
                        # MM meant to net flow must see them; giving every arm the
                        # same visibility keeps an architecture comparison from
                        # being confounded by one arm being blind to size
TWO_WAY = True          # the trader chooses its offer's DIRECTION. Still ONE
                        # offer a round, but the fraction output is read signed:
                        # + sells that share of what it still HOLDS (towards its
                        # target), - sells that share of what it has already
                        # BANKED, back again. The rules allow this ("traders do
                        # not have to trade in one direction"). False: every
                        # trader always sells towards its target
TWO_WAY_FRACTION_INIT_LOGIT = 1.1    # where the fraction output starts in two-way
                        # games. 2*sigmoid(1.1) - 1 = 0.50, so the game opens with
                        # each trader offering half of what it holds, forward --
                        # the same opening as the one-way game, so turning two-way
                        # on does not also move the starting point
DROP_NEGATIVE_PATHS = True  # a random walk can in principle pass through zero.
                        # Any path that does is deleted and the deletion counted;
                        # the table of how often it happens across volatilities is
                        # printed with every run. At sd 0.05 from a0 1.25 it never
                        # fires
NEGATIVE_TABLE_SDS = (0.05, 0.10, 0.20, 0.30, 0.40)   # the tabulated grid
EVAL_ONLY = False
RUN_TAG = ""            # appended to the run folder's name, so a variant of the same game (another
                        # seed, schedule or learning rate) gets its own folder instead of
                        # overwriting the base run's, e.g. _seed142
CHECKS = True           # every check must pass before any training starts. They
                        # are self-contained: this file imports no earlier version
SEED_INIT_TRADER = 42   # trader i is built from SEED_INIT_TRADER + i
SEED_INIT_MM = 42
SEED_PATH = 9012        # cotrain.py's: same validation/eval/record paths,
                        # so every comparison with its runs is paired
ORDER_OFFSET = 100_000  # the order generator's seeds sit apart from paths
SET_PATH = [1.30, 1.36, 1.37, 1.41, 1.33]   # the demo game's rates
# ============================================================================

torch.set_default_dtype(torch.float64)
assert N_OUTER * N_INNER <= 10_000, "batch seeds would overlap; see train_block"
assert MM_LEVEL_INIT in ("conservative", "random", "naive")
assert FIRST in ("mm", "trader")
assert MM_ARCH in ("round", "bid", "turn")
assert isinstance(TWO_WAY, bool)


# ---------------------------------------------------------------------------
# 0. the configuration -- who is in the game, and what the MM therefore sees
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    traders: tuple          # trader names, in index order
    keys: tuple             # the single-trader card ("A"/"B") a row equals, else None
    specs: tuple            # one GameSpec per row
    directions: tuple       # +1.0 pound seller (side A), -1.0 dollar seller (B)
    n: int
    see_direction: bool     # the MM gets the offer's direction: mixed configs
    see_queue: bool         # the MM gets the sub-round: N > 1 with the live book
    mm_arch: str            # "round" | "bid" | "turn": when the MM decides
    live_book: bool         # the level is re-set per offer from the live book: "bid"
    see_size: bool          # the MM gets the current offer's size
    see_flow: bool          # the MM gets the round's flow block: the "turn" arm
    n_plus: int             # pound-selling OFFERS a round (flow block layout)
    n_minus: int            # dollar-selling offers a round
    n_flow: int             # width of the flow block: 2 x n_offers in "turn", else 0
                            # (one pair of numbers per offer of the round)
    two_way: bool           # the trader chooses its offer's direction
    n_offers: int           # offers a round: N (one per trader), both ways
    offer_trader: tuple     # offer k -> the trader it belongs to (k, here)
    offer_directions: tuple # offer k -> its FORWARD direction (+1 pound seller,
                            # -1 dollar seller). Two-way: the live direction is
                            # this times the sign of the trader's signed
                            # fraction, per game and round, and travels with
                            # the offer; this tuple is only the table's side
    ref_direction: float    # the direction whose raw level is read as-is
    n_mm_feat: int
    capital: float          # total game capital, pounds at a0
    rounds: int
    params: object          # the shared market
    name: str


def spec_from_row(row):
    """A GameSpec from one row of the trader table, on the shared market."""
    assert row["side"] in ("A", "B"), f"side must be A or B: {row}"
    return GameSpec(L=float(row["L"]), T=float(row["T"]), A=float(row["A"]),
                    B=float(row["B"]), rounds=ROUNDS, K=1, side=row["side"],
                    params=GameParams(a0=A0, sd=SD, bdc_fee=TRADER_FEE))


def card_key(spec):
    """The single-trader card this spec IS (numbers and market equal), else
    None: only then do that run's saved artefacts for the side apply."""
    card = CARD[spec.side]
    same = (spec.L == card.L and spec.T == card.T and spec.A == card.A
            and spec.B == card.B and spec.rounds == card.rounds
            and spec.params == card.params)
    return spec.side if same else None


def random_roster(n, seed):
    """n random trader cards on the shared market: side by fair coin, redrawn
    until both sides are present (a one-sided table has nothing to cross);
    L from ROSTER_L; target T = par x u with u uniform in ROSTER_TARGET_RANGE,
    where par converts all of L at a0 with no fee; the card's own penalties A
    and B each drawn from ROSTER_CARD_FEES. Amounts rounded to whole units.
    Deterministic in (n, seed) and independent of every other draw in the
    run."""
    assert n >= 2, "a random table needs both sides, so at least two traders"
    gen = np.random.default_rng(seed)
    while True:
        sides = [str(side) for side in gen.choice(["A", "B"], size=n)]
        if len(set(sides)) == 2:
            break
    rows = []
    for i, side in enumerate(sides):
        L = float(gen.choice(ROSTER_L))
        u = float(gen.uniform(*ROSTER_TARGET_RANGE))
        par = L * A0 if side == "A" else L / A0
        rows.append(dict(name=f"R{i + 1:02d}", side=side, L=round(L),
                         T=round(par * u), A=float(gen.choice(ROSTER_CARD_FEES)),
                         B=float(gen.choice(ROSTER_CARD_FEES))))
    return rows


def make_config(rows, mm_arch=None, see_size=None, two_way=None):
    """The configuration of one game. mm_arch, see_size and two_way default
    to the SETTINGS (read at call time, so a driver that rebinds
    them after import is honoured) and are arguments, so that the checks can
    build the reference games and the compare script can hold several arms
    in one process."""
    mm_arch = MM_ARCH if mm_arch is None else mm_arch
    see_size = SEE_OFFER_SIZE if see_size is None else see_size
    two_way = TWO_WAY if two_way is None else bool(two_way)
    assert mm_arch in ("round", "bid", "turn")
    rows = tuple(rows)
    specs = tuple(spec_from_row(row) for row in rows)
    names = tuple(str(row.get("name", f"T{i}")) for i, row in enumerate(rows))
    assert len(set(names)) == len(names), "trader names must be distinct"
    directions = tuple(1.0 if spec.side == "A" else -1.0 for spec in specs)
    n = len(specs)
    assert n >= 1
    assert all(spec.rounds == specs[0].rounds for spec in specs), \
        "one game: every card must have the same number of rounds"
    assert all(spec.params == specs[0].params for spec in specs), \
        "one market: every card must share a0, sd and the Bureau fee"
    # one offer a round per trader, both ways; two-way, its direction is
    # the trader's side times the sign of its signed fraction, per game
    offer_trader, offer_directions = tuple(range(n)), directions
    n_offers = n
    see_direction = two_way or len(set(offer_directions)) > 1
    live_book = mm_arch == "bid"
    see_queue = n_offers > 1 and live_book
    # the flow block exists only on the "turn" arm. It always holds every
    # offer of the round, so with one offer a round the turn arm still has
    # two columns the round arm does not -- that reduction is check 7's
    see_flow = mm_arch == "turn"
    n_plus, n_minus = offer_directions.count(1.0), offer_directions.count(-1.0)
    n_flow = 2 * n_offers if see_flow else 0
    # single-direction configs read the level exactly as mm_train does for
    # that side (so N = 1 is cotrain bit for bit); mixed configs read it in
    # the pound seller's convention and mirror it for dollar sellers, so
    # ONE conservative bias rejects everything in both directions
    ref_direction = directions[0] if not see_direction else 1.0
    n_mm_feat = 5 + int(see_size) + int(see_direction) + int(see_queue) + n_flow
    capital = float(sum(M.game_capital_pounds(spec) for spec in specs))
    return Config(traders=names, keys=tuple(card_key(spec) for spec in specs),
                  specs=specs, directions=directions,
                  n=n, see_direction=see_direction, see_queue=see_queue,
                  mm_arch=mm_arch, live_book=live_book, see_size=bool(see_size),
                  see_flow=see_flow, n_plus=n_plus, n_minus=n_minus,
                  n_flow=n_flow, two_way=two_way, n_offers=n_offers,
                  offer_trader=offer_trader, offer_directions=offer_directions,
                  ref_direction=ref_direction, n_mm_feat=n_mm_feat,
                  capital=capital, rounds=specs[0].rounds,
                  params=specs[0].params, name="-".join(names))


# ---------------------------------------------------------------------------
# 1. models -- all fresh; nothing loaded, nothing pre-trained
# ---------------------------------------------------------------------------
def build_mm_net(cfg, seed):
    # mm_train.build_mm_net with the input width of THIS configuration; at
    # N = 1 the width is 5 and this is that function, op for op
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = T.AlphaNet(n_in=cfg.n_mm_feat, n_out=2)
    if MM_LEVEL_INIT in ("conservative", "naive"):
        level = (1.0 - cfg.ref_direction * M.LEVEL_INIT_OFFSET
                 if MM_LEVEL_INIT == "conservative" else 1.0)
        with torch.no_grad():
            net.head.weight[0].zero_()
            net.head.bias[0] = torch.logit(torch.tensor(level / 2.0))
    return net


def build_traders(cfg, seed=None):
    """The N traders: one network each, the same architecture either way --
    two-way trading reads the SAME fraction output signed, so a trader needs
    no extra parameters for it. Two-way, that output starts at
    TWO_WAY_FRACTION_INIT_LOGIT so the game opens where the one-way game
    opens."""
    seed = SEED_INIT_TRADER if seed is None else seed
    nets = []
    for i in range(cfg.n):
        net = T.build_net(seed + i)
        if cfg.two_way:
            with torch.no_grad():
                net.head.bias[1] = TWO_WAY_FRACTION_INIT_LOGIT
        nets.append(net)
    return nets


def signed_offer(cfg, spec, fraction, held_i, banked_i):
    """The offer, from the trader's fraction output: (amount, direction,
    is_reverse) per game. s = 2f - 1; s > 0 sells s of the HELD balance in
    the trader's forward direction, s < 0 sells |s| of the BANKED balance
    in the other one. Both branches are computed and selected, so the
    gradient flows through the choice; at s = 0 the amount is 0 either
    way."""
    s = 2.0 * fraction - 1.0
    forward = s >= 0
    amount = torch.where(forward, s * held_i, -s * banked_i)
    d = torch.where(forward, torch.full_like(s, spec_direction(spec)),
                    torch.full_like(s, -spec_direction(spec)))
    return amount, d, ~forward


def spec_direction(spec):
    """+1 if the trader's forward offer sells pounds, -1 if dollars."""
    return 1.0 if spec.side == "A" else -1.0


def build_agents(cfg):
    return build_mm_net(cfg, SEED_INIT_MM), build_traders(cfg)


def set_learning(net, learning):
    for parameter in net.parameters():
        parameter.requires_grad_(learning)
    if not learning:
        net.zero_grad(set_to_none=True)
    net.train() if learning else net.eval()


def tau_for(agent, step):
    start, end = TAU_TRADER if agent == "trader" else TAU_MM
    n_anneal = max(int(TAU_ANNEAL_FRAC * N_OUTER * N_INNER), 1)  ################################################# COULD BE CAUSING DIFFERENCE ON OB=30 COMPARED TO 20
    if TAU_ANNEAL_STEPS is not None:
        n_anneal = max(int(TAU_ANNEAL_STEPS), 1)   # fixed in steps, so N_OUTER no longer stretches it
    decay = (end / start) ** (1.0 / max(n_anneal - 1, 1))
    return start * decay ** step if step < n_anneal else end


def order_of_play():
    return ("mm", "trader") if FIRST == "mm" else ("trader", "mm")


# ---------------------------------------------------------------------------
# 2. the game -- mm_train.rollout with N traders arriving in a random order
# ---------------------------------------------------------------------------
def draw_order(cfg, n_games, seed):
    """One permutation of the OFFERS per game and round -- the arrival order
    is redrawn every round, so no trader owns a slot. With a single trader
    there is only one order, and no random number is drawn."""
    if cfg.n_offers == 1:
        return torch.zeros(n_games, cfg.rounds, 1, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(n_games, cfg.rounds, cfg.n_offers, generator=gen),
                         dim=-1)


def mm_features(cfg, rounds_remaining_fraction, phase_flag, hidden_rate_now,
                mm_pounds, mm_dollars, offered_amount, direction, offers_left,
                flow=None):
    # mm_train.mm_features' five columns, in its order and its arithmetic,
    # then this file's gated columns, then ("turn" only) the flow block;
    # flow None means zeros of that width (the Bureau-phase call, where the
    # round's offers are already in the book)
    n_games = hidden_rate_now.shape[0]
    capital, a0 = cfg.capital, cfg.params.a0

    def as_column(value):
        return (torch.full((n_games,), float(value))
                if np.isscalar(value) else value)

    feature_columns = [
        as_column(rounds_remaining_fraction),  # 1  in [0,1]
        as_column(phase_flag),                 # 2  0 = verdict, 1 = at the Bureau
        hidden_rate_now,                       # 3  the true rate
        mm_pounds / capital,                   # 4  the book as fractions of the
        mm_dollars / (a0 * capital),           # 5  game's capital; negative = short
    ]
    if cfg.see_size:
        offer_unit = torch.where(direction > 0, torch.full_like(direction, capital),
                                 torch.full_like(direction, a0 * capital))
        feature_columns.append(offered_amount / offer_unit)   # 6  offer size
    if cfg.see_direction:
        feature_columns.append(direction)      # +1 pound seller, -1 dollar seller
    if cfg.see_queue:
        feature_columns.append(as_column(offers_left))   # offers still to come
    features = torch.stack(feature_columns, dim=-1)
    if cfg.see_flow:
        features = torch.cat([features, torch.zeros(n_games, cfg.n_flow)
                              if flow is None else flow], dim=1)
    return features


def flow_block_two_way(cfg, quoted, offered, hidden_rate_now, directions):
    """(n_games, 2N): the round's flow when the trader chooses its direction
    rather than the table fixing it. One
    direction-free layout: per offer, the MM's EDGE per pound of it,
    d (X - p) / X, and its SIGNED pound-equivalent size at the TRUE rate
    (+ the MM buys pounds, - it sells them); sorted best-edge first. Same
    properties as the one-way block: fixed width, no slot, no identity.
    Every offer of the round is in it, the judged one included."""
    X = hidden_rate_now.unsqueeze(1)
    edge = directions * (X - quoted) / X
    pounds = torch.where(directions > 0, offered, offered / X)
    signed_size = directions * pounds / cfg.capital
    key = -edge                                   # ascending key = best first
    key, perm = torch.sort(key, dim=1)
    edge_sorted, size_sorted = -key, signed_size.gather(1, perm)
    return torch.cat([edge_sorted, size_sorted], dim=1)


def flow_block(cfg, quoted, offered, hidden_rate_now, live_directions=None):
    """(n_games, 2N): the round's order flow as the turn-wise MM sees it,
    built from the traders' quotes BEFORE any verdict.
    Layout, fixed by the table: pound sellers' prices, pound sellers' sizes,
    dollar sellers' prices, dollar sellers' sizes; a direction with no
    trader contributes no columns. price = quoted / X - 1 (the fee-aware cut
    is -g for pound sellers, +g/(1-g) for dollar sellers); size in the same
    offer units the MM sees elsewhere (capital for pounds, a0 * capital for dollars). Sorted
    best-for-the-MM first: pound sellers ascending in price (cheapest pounds
    first), dollar sellers descending (the MM pays 1/price pounds per
    dollar). Sorted, so no column is a slot or an identity (check 8).
    Every offer of the round is in the block, the judged one included, so
    the level may depend on that offer's own price and the policy class is
    any accept/reject rule. Not detached: the traders' prices enter the
    MM's level, which is the true game (the "bid" arm already carries that
    dependency through the live book)."""
    capital, a0 = cfg.capital, cfg.params.a0
    if cfg.two_way:
        return flow_block_two_way(cfg, quoted, offered, hidden_rate_now, live_directions)
    directions = torch.tensor(cfg.offer_directions)
    cols = []
    for d, unit in ((1.0, capital), (-1.0, a0 * capital)):
        idx = torch.nonzero(directions == d).squeeze(1)
        if idx.numel() == 0:
            continue
        p = quoted[:, idx] / hidden_rate_now.unsqueeze(1) - 1.0
        q = offered[:, idx] / unit
        key = d * p
        key, perm = torch.sort(key, dim=1)       # ascending key = best first
        cols += [d * key, q.gather(1, perm)]
    return torch.cat(cols, dim=1)


def rule_level(rule, hidden_rate_now, direction, fee):
    # mm_train.rule_verdict's break-even, per offer by its direction
    if not rule.fee_aware:
        return hidden_rate_now
    return torch.where(direction > 0, (1.0 - fee) * hidden_rate_now,
                       hidden_rate_now / (1.0 - fee))


@dataclass
class BookRule:
    """The BOOK RULE: a hardcoded MM that decides a whole round at once. It reads
    every offer of the round and its own book, and accepts the SUBSET of
    offers that maximises the pound value of its book after the round if
    every short were covered at the Bureau there and then (mm_train's
    mm_bureau with repay 1 at the true rate; longs valued at the true rate,
    as the game scores them). Limit-order-book logic written down, and the
    benchmark for "did the learned MM work out how to net?": a pound seller at p_A and a dollar seller at p_B cross
    when p_B > p_A and the MM nets them; inventory nets against the flow;
    only the residual meets the Bureau's fee -- "put you two together, a
    small residual". With one offer on the table it is the fee-aware rule
    exactly (check 10); with two sides it accepts pairs the fee-aware rule
    rejects. Myopic on purpose: one round, no Jensen, so a learned MM can
    beat it by carrying. Repay as the other rules: 1 flatten, 0 carry.
    Brute force over the 2^N subsets (N <= 8); ties go to the lowest
    subset index."""
    name: str
    repay: float
    fee_aware: bool = True    # read by benchmark_agreement's callers; the verdicts are the book's
    book: bool = True


def reference_rules(fee, n_offers=None):
    """mm_train's four hardcoded MMs, then the book rule in both repay
    flavours. Only at fee > 0: at fee 0 the round value is linear in the
    offers, every offer is judged on its own, and the book rule IS the
    naive rule. The book rule enumerates 2^(offers) subsets a round, so
    past BOOK_RULE_MAX_OFFERS offers it is left out."""
    rules = list(M.reference_rules(fee))
    if fee > 0.0 and (n_offers is None or n_offers <= BOOK_RULE_MAX_OFFERS):
        rules += [BookRule("book+flatten", 1.0), BookRule("book+carry", 0.0)]
    return rules


def _subsets(n):
    """(2^n, n) 0/1 indicator of every subset, subset s = bits of s."""
    return torch.tensor([[(s >> i) & 1 for i in range(n)] for s in range(2 ** n)],
                        dtype=torch.get_default_dtype())


def book_verdicts(cfg, quoted, offered, hidden_rate_now, mm_pounds, mm_dollars, fee,
                  live_directions=None):
    """The book rule's verdicts for one round: (fill, level), each (n_games,
    N). fill is the subset with the highest flattened pound value of the
    book after the round. level is the record's price for each offer: the
    price at which the MM would have been indifferent to it given the rest
    of its chosen set -- sign-exact (accept iff direction * (level - price)
    > 0 reproduces fill, up to ties) and first order in magnitude, since
    the marginal value of a pound is taken as 1 and of a dollar as 1/X."""
    N = cfg.n_offers
    assert N <= 12, "the book rule enumerates 2^(offers) subsets"
    d = (torch.tensor(cfg.offer_directions) if live_directions is None
         else live_directions)          # two-way: direction per game
    # each offer's book move if accepted
    dp = torch.where(d > 0, offered, -offered / quoted)        # pounds
    dd = torch.where(d > 0, -offered * quoted, offered)        # dollars
    sub = _subsets(N)                                          # (2^N, N)
    P = mm_pounds.unsqueeze(1) + dp @ sub.T                    # (n_games, 2^N)
    D = mm_dollars.unsqueeze(1) + dd @ sub.T
    X = hidden_rate_now.unsqueeze(1).expand_as(P)
    P2, D2 = M.mm_bureau(P, D, torch.ones_like(P), X, fee)     # cover every short now
    value = P2 + D2 / X
    best = value.argmax(dim=1)                                 # (n_games,)
    fill = sub[best]                                           # (n_games, N)
    # the marginal value of each offer against the rest of the chosen set
    bits = (1 << torch.arange(N)).unsqueeze(0)                 # (1, N)
    s_in = best.unsqueeze(1) | bits
    s_out = best.unsqueeze(1) & ~bits
    marginal = value.gather(1, s_in) - value.gather(1, s_out)  # pounds, (n_games, N)
    q_pos = offered > 0
    p = quoted
    level_A = p + torch.where(q_pos, marginal * hidden_rate_now.unsqueeze(1) / offered.clamp_min(1e-300), torch.zeros_like(p))
    level_B = torch.where(q_pos, p / (1.0 + marginal * p / offered.clamp_min(1e-300)), p)
    level = torch.where(d > 0, level_A, level_B)
    return fill, level


OFFER_KEYS = ("quoted_price", "offered_amount", "level", "fill_weight",
              "position", "mm_pounds_at_verdict", "mm_dollars_at_verdict")   # (n_games, offers, rounds)
TRADER_KEYS = OFFER_KEYS + ("trader_held_after_fill", "trader_banked_after_fill",
                            "trader_held_after_bureau", "trader_banked_after_bureau")   # the rest: (n_games, N, rounds)
MM_KEYS = ("hidden_rate", "repay_fraction",
           "mm_pounds_after_fill", "mm_dollars_after_fill",
           "mm_pounds_after_bureau", "mm_dollars_after_bureau")


def rollout(cfg, mm_net, trader_nets, rate_moves_sd, order, tau=None,
            rule=None, fee=None, record=False, tau_abs=False, move_scales=None):
    """tau   None: hard verdicts (evaluation). A number: the training ramp.
       move_scales  per-round ramp scale to use instead of this batch's own
             rate-move std -- so a batch backpropped in chunks plays the SAME
             soft game as the whole batch would.
       rule  a mm_train.Rule instead of the MM net.
       order (n_games, rounds, offers) offer index per slot, from draw_order.
       Returns the MM's P/L, a list of the N traders' P/Ls, and the record
       (per-offer keys shaped (n_games, offers, rounds), per-trader keys
       (n_games, N, rounds), MM keys (n_games, rounds)), all per game."""
    fee = MM_FEE if fee is None else fee
    market, n_rounds, N = cfg.params, cfg.rounds, cfg.n
    K = cfg.n_offers                          # offers a round: one per trader
    n_games = rate_moves_sd.shape[0]
    directions = torch.tensor(cfg.offer_directions)

    # traders' state
    held = [torch.full((n_games,), float(spec.L)) for spec in cfg.specs]
    banked = [torch.zeros(n_games) for _ in cfg.specs]
    # game state -- one market, shared by all
    last_revealed_rate = torch.full((n_games,), float(market.a0))
    recent_moves = torch.zeros(n_games, T.LAGS) if T.LAGS else None
    hidden_rates = market.a0 + torch.cumsum(
        market.sd * rate_moves_sd[:, :n_rounds], dim=1)
    settlement_rate = hidden_rates[:, -1] + market.sd * rate_moves_sd[:, n_rounds]
    # MM state
    mm_pounds = torch.zeros(n_games)
    mm_dollars = torch.zeros(n_games)

    log = ({key: [[] for _ in range(K if key in OFFER_KEYS else N)] for key in TRADER_KEYS}
           | {key: [] for key in MM_KEYS}) if record else None
    live_directions_by_round = []        # two-way: direction varies, so logged

    for round in range(n_rounds):
        rounds_remaining_fraction = (n_rounds - round) / n_rounds
        hidden_rate_now = hidden_rates[:, round]

        # ---- offer phase: every trader quotes from its OWN state ---------
        # one offer per trader; two-way, the trader also chooses its
        # direction (the fraction read signed), so directions are data
        quoted, offered, live_d, reverse = [], [], [], []
        for i, (net, spec) in enumerate(zip(trader_nets, cfg.specs)):
            price, fraction = T.squash(
                net(T.features(rounds_remaining_fraction, 0.0, held[i],
                               banked[i], last_revealed_rate, spec,
                               recent_moves)),
                last_revealed_rate)
            quoted.append(price)
            if cfg.two_way:
                amount, d_i, rev_i = signed_offer(cfg, spec, fraction, held[i], banked[i])
                offered.append(amount)
                live_d.append(d_i)
                reverse.append(rev_i)
            else:
                offered.append(fraction * held[i])
        quoted, offered = torch.stack(quoted, dim=1), torch.stack(offered, dim=1)
        if cfg.two_way:
            live_directions = torch.stack(live_d, dim=1)     # (n_games, N)
            is_reverse = torch.stack(reverse, dim=1)
        else:
            live_directions = is_reverse = None
        # "turn": the whole round's flow, read before any
        # verdict; one block serves every slot of the round
        flow = (flow_block(cfg, quoted, offered, hidden_rate_now,
                           live_directions=live_directions)
                if cfg.see_flow else None)

        # ---- ... and the MM answers them in this round's random order ----
        # the level is set from the book as the round OPENS (one policy per
        # round) unless the offer-by-offer arm ("bid") is on
        round_pounds, round_dollars = mm_pounds, mm_dollars
        if rule is not None and getattr(rule, "book", False):
            # the book rule: every offer and the round-open book, one
            # subset decision, read out per slot below
            book_fill, book_level = book_verdicts(cfg, quoted, offered, hidden_rate_now,
                                                  round_pounds, round_dollars, fee,
                                                  live_directions)
        if tau is not None:
            move_scale = (move_scales[round] if move_scales is not None
                          else (hidden_rate_now - last_revealed_rate).detach().std()
                          ).clamp_min(1e-12)
            ramp_width = tau if tau_abs else tau * move_scale
        fill = [torch.zeros(n_games) for _ in range(K)]
        level_seen = [torch.zeros(n_games) for _ in range(K)]
        pounds_seen = [torch.zeros(n_games) for _ in range(K)]
        dollars_seen = [torch.zeros(n_games) for _ in range(K)]
        position = [torch.zeros(n_games) for _ in range(K)]
        for slot in range(K):
            idx = order[:, round, slot]
            price = quoted.gather(1, idx.unsqueeze(1)).squeeze(1)
            amount = offered.gather(1, idx.unsqueeze(1)).squeeze(1)
            direction = (directions[idx] if not cfg.two_way
                         else live_directions.gather(1, idx.unsqueeze(1)).squeeze(1))
            if rule is not None:
                level = (book_level.gather(1, idx.unsqueeze(1)).squeeze(1)
                         if getattr(rule, "book", False)
                         else rule_level(rule, hidden_rate_now, direction, fee))
            else:
                # the level is set from what the MM knows BEFORE the price
                # is read. mm_squash's sigmoid on the head's own view (the
                # strided kernel: a contiguous copy rounds differently),
                # read as-is in the reference direction and mirrored,
                # 1 - s = sigmoid(-raw), for the other one
                book_pounds, book_dollars = ((mm_pounds, mm_dollars)
                                             if cfg.live_book
                                             else (round_pounds, round_dollars))
                raw = mm_net(mm_features(cfg, rounds_remaining_fraction, 0.0,
                                         hidden_rate_now, book_pounds, book_dollars,
                                         amount, direction,
                                         (K - 1 - slot) / K, flow))
                s = torch.sigmoid(raw[..., 0])
                level = 2.0 * hidden_rate_now * torch.where(
                    direction == cfg.ref_direction, s, 1.0 - s)
            if rule is not None and getattr(rule, "book", False):
                fill_slot = book_fill.gather(1, idx.unsqueeze(1)).squeeze(1)
            elif rule is not None or tau is None:
                fill_slot = (direction * (level - price) > 0).double()
            else:
                fill_slot = torch.sigmoid(direction * (level - price) / ramp_width)
            # the fill, from BOTH ends, trader by trader: the slot's trader
            # takes it (mask 1), the others take exactly nothing (mask 0).
            # One filled tensor per trader feeds its book and the MM's, in
            # mm_train's order of operations, so that at N = 1 the backward
            # graph is mm_train's and the gradient is the same to the bit
            # the book the MM SAW for this verdict, for the record
            mm_pounds_before, mm_dollars_before = ((mm_pounds, mm_dollars)
                                                   if cfg.live_book
                                                   else (round_pounds, round_dollars))
            for k in range(K):
                i, spec = cfg.offer_trader[k], cfg.specs[cfg.offer_trader[k]]
                mask = (idx == k).double()
                fill_i = mask * fill_slot
                filled = fill_i * offered[:, k]
                if not cfg.two_way:
                    # one-way: the trader sells what it holds, so the fill
                    # moves held -> banked and the MM's book the other way
                    held[i] = held[i] - filled
                    if spec.side == "A":
                        banked[i] = banked[i] + filled * quoted[:, k]
                        mm_pounds = mm_pounds + filled
                        mm_dollars = mm_dollars - filled * quoted[:, k]
                    else:
                        banked[i] = banked[i] + filled / quoted[:, k]
                        mm_dollars = mm_dollars + filled
                        mm_pounds = mm_pounds - filled / quoted[:, k]
                else:
                    # two-way: the same fill, split per game between the forward
                    # trade (sells the held balance) and the reverse one
                    # (sells the banked balance back at the offer's price)
                    rev = is_reverse[:, k].double() * filled
                    fwd = filled - rev
                    if spec.side == "A":
                        held[i] = held[i] - fwd + rev / quoted[:, k]
                        banked[i] = banked[i] + fwd * quoted[:, k] - rev
                        mm_pounds = mm_pounds + fwd - rev / quoted[:, k]
                        mm_dollars = mm_dollars - fwd * quoted[:, k] + rev
                    else:
                        held[i] = held[i] - fwd + rev * quoted[:, k]
                        banked[i] = banked[i] + fwd / quoted[:, k] - rev
                        mm_dollars = mm_dollars + fwd - rev * quoted[:, k]
                        mm_pounds = mm_pounds - fwd / quoted[:, k] + rev
                fill[k] = fill[k] + fill_i
                level_seen[k] = level_seen[k] + mask * level
                pounds_seen[k] = pounds_seen[k] + mask * mm_pounds_before
                dollars_seen[k] = dollars_seen[k] + mask * mm_dollars_before
                position[k] = position[k] + mask * slot
        if T.LAGS:
            recent_moves = torch.cat(
                [(hidden_rate_now - last_revealed_rate).detach().unsqueeze(1),
                 recent_moves[:, :-1]], dim=1)
        else:
            recent_moves = None
        if record:
            if cfg.two_way:
                live_directions_by_round.append(live_directions.detach().numpy().copy())
            for k in range(K):
                for key, value in (("quoted_price", quoted[:, k]),
                                   ("offered_amount", offered[:, k]),
                                   ("level", level_seen[k]),
                                   ("fill_weight", fill[k]),
                                   ("position", position[k]),
                                   ("mm_pounds_at_verdict", pounds_seen[k]),
                                   ("mm_dollars_at_verdict", dollars_seen[k])):
                    log[key][k].append(value.detach().numpy().copy())
            for i in range(N):
                for key, value in (("trader_held_after_fill", held[i]),
                                   ("trader_banked_after_fill", banked[i])):
                    log[key][i].append(value.detach().numpy().copy())
            for key, value in (("hidden_rate", hidden_rate_now),
                               ("mm_pounds_after_fill", mm_pounds),
                               ("mm_dollars_after_fill", mm_dollars)):
                log[key].append(value.detach().numpy().copy())

        # ---- Bureau phase: rate revealed; the traders first, then the MM --
        for i, (net, spec) in enumerate(zip(trader_nets, cfg.specs)):
            _, bdc_fraction = T.squash(
                net(T.features(rounds_remaining_fraction, 1.0, held[i],
                               banked[i], hidden_rate_now, spec,
                               recent_moves)),
                hidden_rate_now)
            converted_amount = bdc_fraction * held[i]
            bdc_rate_after_fee = (1.0 - market.bdc_fee) * (
                hidden_rate_now if spec.side == "A" else 1.0 / hidden_rate_now)
            held[i] = held[i] - converted_amount
            banked[i] = banked[i] + converted_amount * bdc_rate_after_fee
        if rule is not None:
            repay_fraction = torch.full((n_games,), float(rule.repay))
        else:
            raw = mm_net(mm_features(cfg, rounds_remaining_fraction, 1.0,
                                     hidden_rate_now, mm_pounds, mm_dollars,
                                     torch.zeros(n_games), torch.zeros(n_games),
                                     0.0))
            repay_fraction = torch.sigmoid(raw[..., 1])
        mm_pounds, mm_dollars = M.mm_bureau(mm_pounds, mm_dollars, repay_fraction,
                                            hidden_rate_now, fee)
        if record:
            for i in range(N):
                log["trader_held_after_bureau"][i].append(held[i].detach().numpy().copy())
                log["trader_banked_after_bureau"][i].append(banked[i].detach().numpy().copy())
            for key, value in (("repay_fraction", repay_fraction),
                               ("mm_pounds_after_bureau", mm_pounds),
                               ("mm_dollars_after_bureau", mm_dollars)):
                log[key].append(value.detach().numpy().copy())
        last_revealed_rate = hidden_rate_now

    # ---- settlement: each trader's (torch_train's terminal equation) ------
    trader_pls = []
    for i, spec in enumerate(cfg.specs):
        settlement_factor = (1.0 / settlement_rate if spec.side == "A"
                             else settlement_rate)
        shortfall = torch.clamp(spec.T - banked[i], min=0.0)
        final_wealth = (held[i] * (1.0 - spec.A)
                        + (banked[i] - spec.B * shortfall) * settlement_factor)
        trader_pls.append((final_wealth - spec.L) / spec.L)

    # ---- round 5: the MM MUST clear any short at the round-5 rate, same
    # fee; what is left is scored in pounds at that rate with no fee
    mm_pounds, mm_dollars = M.mm_bureau(mm_pounds, mm_dollars, torch.ones(n_games),
                                        settlement_rate, fee)
    mm_pl = (mm_pounds + mm_dollars / settlement_rate) / cfg.capital

    if record:
        log = ({key: np.stack([np.stack(per_round, axis=1) for per_round in log[key]],
                              axis=1) for key in TRADER_KEYS}
               | {key: np.stack(log[key], axis=1) for key in MM_KEYS})
        log["direction"] = (np.stack(live_directions_by_round, axis=2) if cfg.two_way
                            else np.broadcast_to(np.array(cfg.offer_directions)[None, :, None],
                                                 log["fill_weight"].shape).copy())
        log["settlement_rate"] = settlement_rate.detach().numpy().copy()
        log["mm_pounds_end"] = mm_pounds.detach().numpy().copy()
        log["mm_dollars_end"] = mm_dollars.detach().numpy().copy()
    return mm_pl, trader_pls, log


def touches_zero(cfg, rate_moves_sd):
    """Per path: does the rate touch <= 0 at any of the rounds + 1 draws?"""
    market = cfg.params
    rates = market.a0 + torch.cumsum(market.sd * rate_moves_sd, dim=1)
    return (rates <= 0.0).any(dim=1)


def draw_paths(cfg, n_paths, seed, label=""):
    """Rate-move draws for training, validation and evaluation, with the
    stated policy on negative rates: any path touching <= 0 is deleted and
    the deletion counted. At sigma 0.05 nothing is deleted and the tensor is
    the one cotrain.py draws from the same seed."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_paths, cfg.rounds + 1, generator=gen)
    if not DROP_NEGATIVE_PATHS:
        return z
    bad = touches_zero(cfg, z)
    if bool(bad.any()):
        print(f"  [negative rate] {int(bad.sum())} of {n_paths:,} paths touched "
              f"<= 0 and were deleted{(' (' + label + ')') if label else ''}",
              flush=True)
        return z[~bad]
    return z


def negative_path_table(cfg, n=200_000, seed=0):
    """The documented rejection probabilities: for a grid of sigma at this
    game's a0 and rounds + 1 draws, the share of paths deleted, by the draw
    at which they first touch <= 0 (negative_paths.py, kept as a run-time
    note). Returns rows (sd, P(deleted), per-draw first-hit shares)."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, cfg.rounds + 1, generator=gen)
    rows = []
    for sd in NEGATIVE_TABLE_SDS:
        rates = cfg.params.a0 + sd * torch.cumsum(z, dim=1)
        touched = rates <= 0.0
        hit = touched.any(dim=1)
        first = torch.where(hit, touched.double().argmax(dim=1) + 1, torch.zeros(n, dtype=torch.long))
        rows.append((sd, float(hit.double().mean()),
                     [float((first == r).double().mean()) for r in range(1, cfg.rounds + 2)]))
    return rows


def always_bdc_pl(spec, rate_moves_sd):
    """The trader's outside option on THESE paths (cotrain.always_bdc_pl)."""
    market = spec.params
    z = rate_moves_sd.detach().numpy()
    hidden_rates = market.a0 + np.cumsum(market.sd * z[:, :spec.rounds], axis=1)
    settlement_rate = hidden_rates[:, -1] + market.sd * z[:, spec.rounds]
    return torch.as_tensor(dump_all_pl(spec, hidden_rates[:, 0], settlement_rate))


# ---------------------------------------------------------------------------
# 3. measurement -- read off the record, never off the learner
# ---------------------------------------------------------------------------
def pound_volume(log):
    """Filled volume in pounds, per game: pounds bought from pound sellers
    plus pounds paid to dollar sellers, at the fill's price."""
    filled = log["fill_weight"] * log["offered_amount"]
    return np.where(log["direction"] > 0, filled,
                    filled / log["quoted_price"]).sum(axis=(1, 2))


def bureau_pound_volume(log):
    """Pounds the MM moved at the Bureau, per game: every round's repay and
    the round-5 clear, whichever leg it sold."""
    d_pounds = log["mm_pounds_after_bureau"] - log["mm_pounds_after_fill"]
    final = log["mm_pounds_end"] - log["mm_pounds_after_bureau"][:, -1]
    return np.abs(d_pounds).sum(axis=1) + np.abs(final)


MIN_FILLED_SHARE = 0.005   # below this share of offered pound volume filled,
                           # the Bureau share is noise and is reported as nan


def offered_pound_volume(log):
    """Offered volume in pound-equivalent at the TRUE rate, per offer.
    Converting at the QUOTED price would be wrong: an unfillable reverse
    quote near zero then counted as an infinite pound offer and swamped
    every volume-weighted statistic (Bureau share nan, "filled 0.0%" next
    to 15% fills). Fills stay at the fill's price: that is money that moved."""
    X = log["hidden_rate"][:, None, :]
    return np.where(log["direction"] > 0, log["offered_amount"],
                    log["offered_amount"] / X)


def bureau_share(log):
    """Of the filled pound volume, the share the MM took to the Bureau.
    1 - this is the share it crossed against opposite flow (or carried out
    of the game solvent). Ratio of sums over games; nan when almost nothing
    was filled (a share of nothing said 97% in the first AB run)."""
    offered = offered_pound_volume(log).sum()
    filled, bureau = pound_volume(log).sum(), bureau_pound_volume(log).sum()
    if offered <= 0 or filled / offered < MIN_FILLED_SHARE:
        return float("nan")
    return float(bureau / filled)


def thresholds_by_direction(log, rounds):
    """The fitted acceptance cut P/X per direction and round (mm_train's
    operating_threshold on that direction's offers). Returns {+1: [...],
    -1: [...]} with nan where the direction is absent or has no volume."""
    out = {}
    for d, side in ((1.0, "A"), (-1.0, "B")):
        rows = []
        for r in range(rounds):
            sel = log["direction"][:, :, r] == d
            volume = log["offered_amount"][:, :, r][sel]
            fill = log["fill_weight"][:, :, r][sel]
            # a cut is identified only between accepted and rejected volume;
            # with nothing accepted (or nothing rejected) the fit sits at an
            # extreme quote and says nothing about the policy
            if (volume * fill).sum() <= 0 or (volume * (1.0 - fill)).sum() <= 0:
                rows.append(np.nan)
                continue
            u = (log["quoted_price"][:, :, r] / log["hidden_rate"][:, r][:, None])[sel]
            rows.append(M.operating_threshold(u, volume, fill, side)[0])
        out[d] = np.array(rows)
    return out


def book_verdicts_from_log(cfg, log, fee):
    """The book rule's verdicts on a recorded play, offer by offer: what the
    book rule would have done with the learner's offers and the learner's
    own book at each round's open (the counterfactual the agreement
    statistic needs). (n_games, N, rounds)."""
    n_games, N, rounds = log["fill_weight"].shape
    out = np.zeros((n_games, N, rounds))
    with torch.no_grad():
        for r in range(rounds):
            if r == 0:
                P0, D0 = torch.zeros(n_games), torch.zeros(n_games)
            else:
                P0 = torch.tensor(log["mm_pounds_after_bureau"][:, r - 1])
                D0 = torch.tensor(log["mm_dollars_after_bureau"][:, r - 1])
            fill, _ = book_verdicts(cfg, torch.tensor(log["quoted_price"][:, :, r]),
                                    torch.tensor(log["offered_amount"][:, :, r]),
                                    torch.tensor(log["hidden_rate"][:, r]), P0, D0, fee,
                                    torch.tensor(log["direction"][:, :, r]) if cfg.two_way else None)
            out[:, :, r] = fill.numpy()
    return out


def benchmark_agreement(log, rule, fee, cfg=None):
    """Share of the offered volume (in pounds, so directions add) on which
    the recorded verdicts agree with a reference rule, and the filled share
    for both. mm_train.volume_agreement, per offer by its direction. For
    the book rule the benchmark verdicts are the book's own, recomputed on
    the learner's offers and book (cfg needed)."""
    X = log["hidden_rate"][:, None, :]
    d = log["direction"]
    if getattr(rule, "book", False):
        assert cfg is not None, "benchmark_agreement needs cfg for the book rule"
        benchmark = book_verdicts_from_log(cfg, log, fee)
    else:
        level = X if not rule.fee_aware else np.where(d > 0, (1 - fee) * X, X / (1 - fee))
        benchmark = (d * (level - log["quoted_price"]) > 0).astype(float)
    volume = offered_pound_volume(log)        # at the true rate (see there)
    total = volume.sum()
    if total <= 0:
        return float("nan"), 0.0, 0.0
    agree = (volume * (log["fill_weight"] == benchmark)).sum() / total
    filled = (volume * log["fill_weight"]).sum() / total
    filled_benchmark = (volume * benchmark).sum() / total
    return float(agree), float(filled), float(filled_benchmark)


def repay_on_short_books(log):
    """mm_train.repay_on_short_books on this file's record (same keys)."""
    return M.repay_on_short_books(log)


# ---------------------------------------------------------------------------
# 4. checks -- the reduction to cotrain.py, then the seams this file adds
# ---------------------------------------------------------------------------
def _snapshot(net):
    return {key: value.detach().clone() for key, value in net.state_dict().items()}


def _unchanged(before, net):
    return all(torch.equal(before[key], value)
               for key, value in net.state_dict().items())


def check_reduces_to_mm_train(row, n_games=2000, seed=3):
    """check 1 -- the N = 1 configuration IS mm_train.rollout: same nets,
    same paths, the hard game, the soft game and every rule must agree to
    the bit (max |gap| == 0), P/L and record alike. Two MMs: the training
    init (its zeroed head row makes the level a constant, which would hide
    any arithmetic on it) and a random-head net whose level moves with
    every input. Run at the single-trader engine's own visibility, so it
    tests exactly that engine's game; the extra input columns this file can
    add are certified separately by the reduction gate (check 7)."""
    cfg = make_config((row,), mm_arch="round", see_size=M.SEE_OFFER_SIZE, two_way=False)
    spec, side = cfg.specs[0], cfg.specs[0].side
    trader_net = T.build_net(SEED_INIT_TRADER)
    mm_net = build_mm_net(cfg, SEED_INIT_MM)
    reference_mm = M.build_mm_net(SEED_INIT_MM, side)
    assert _unchanged(_snapshot(reference_mm), mm_net), \
        "check 1: the N = 1 MM is not mm_train's build_mm_net"
    with torch.random.fork_rng():
        torch.manual_seed(SEED_INIT_MM + 1)
        random_mm = T.AlphaNet(n_in=cfg.n_mm_feat, n_out=2)
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_games, spec.rounds + 1, generator=gen)
    order = draw_order(cfg, n_games, seed)
    plays = ([(f"{name} hard", net, dict(tau=None)) for name, net in
              (("init net", mm_net), ("random net", random_mm))]
             + [(f"{name} soft tau 0.2", net, dict(tau=0.2)) for name, net in
                (("init net", mm_net), ("random net", random_mm))]
             + [(rule.name, None, dict(rule=rule)) for rule in M.reference_rules(0.01)])
    worst = 0.0
    for label, net, kw in plays:
        with torch.no_grad():
            ref_mm, ref_tr, ref_log = M.rollout(net, trader_net, spec, z,
                                                fee=0.01, record=True, **kw)
            mm_pl, trader_pls, log = rollout(cfg, net, [trader_net], z, order,
                                             fee=0.01, record=True, **kw)
        gap = max(float((mm_pl - ref_mm).abs().max()),
                  float((trader_pls[0] - ref_tr).abs().max()))
        for mine, theirs in (("quoted_price", "quoted_price"),
                             ("offered_amount", "offered_amount"),
                             ("level", "break_even"), ("fill_weight", "fill_weight"),
                             ("trader_held_after_bureau", "trader_held_after_bureau"),
                             ("mm_pounds_after_bureau", "mm_pounds_after_bureau"),
                             ("mm_dollars_after_bureau", "mm_dollars_after_bureau")):
            a = log[mine][:, 0, :] if mine in TRADER_KEYS else log[mine]   # one offer, one trader
            gap = max(gap, float(np.abs(a - ref_log[theirs]).max()))
        gap = max(gap, float(np.abs(log["mm_pounds_end"] - ref_log["mm_pounds_end"]).max()))
        worst = max(worst, gap)
        print(f"  check 1  N=1 ({cfg.traders[0]}, side {side}) vs mm_train.rollout, {label:<24} "
              f"{n_games} games: max |gap| {gap:.1e}  "
              f"[{'ok' if gap == 0.0 else 'FAIL'}]", flush=True)
    assert worst == 0.0, "the N = 1 game is not mm_train's game"


def check_books_conserve(cfg, n_games=2000, seed=5):
    """check 2 -- the seam this file adds: with N traders filling in a
    random order, the sum of the traders' book moves must equal minus the
    MM's, every fill 0 or 1, every Bureau trade at the fee-adjusted rate,
    no repay deepening a short, and round 5 leaving no coverable short.
    Two players, as mm_train: naive+flatten and a generous untrained net."""
    f = 0.01
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_games, cfg.rounds + 1, generator=gen)
    order = draw_order(cfg, n_games, seed)
    trader_nets = build_traders(cfg)
    generous = build_mm_net(cfg, SEED_INIT_MM)
    with torch.no_grad():   # generous in the reference direction
        generous.head.bias[0] = torch.logit(torch.tensor(
            (1.0 + cfg.ref_direction * M.LEVEL_INIT_OFFSET) / 2.0))
    for label, kw in (("naive+flatten", dict(rule=M.Rule("naive+flatten", False, 1.0))),
                      ("untrained net", dict())):
        with torch.no_grad():
            _, _, log = rollout(cfg, None if "rule" in kw else generous, trader_nets,
                                z, order, tau=None, fee=f, record=True, **kw)
        fill = log["fill_weight"]
        assert np.all((fill == 0.0) | (fill == 1.0)), "hard verdict gave a fraction"
        filled_share = ((log["offered_amount"] * fill).sum()
                        / log["offered_amount"].sum())
        # each trader's physical pounds and dollars before and after the fills
        d_pounds = np.zeros((n_games, cfg.rounds))
        d_dollars = np.zeros((n_games, cfg.rounds))
        for i, spec in enumerate(cfg.specs):
            start_held = np.concatenate([np.full((n_games, 1), spec.L),
                                         log["trader_held_after_bureau"][:, i, :-1]], axis=1)
            start_banked = np.concatenate([np.zeros((n_games, 1)),
                                           log["trader_banked_after_bureau"][:, i, :-1]], axis=1)
            dh = log["trader_held_after_fill"][:, i] - start_held
            db = log["trader_banked_after_fill"][:, i] - start_banked
            d_pounds += dh if spec.side == "A" else db
            d_dollars += db if spec.side == "A" else dh
        start_pounds = np.concatenate([np.zeros((n_games, 1)),
                                       log["mm_pounds_after_bureau"][:, :-1]], axis=1)
        start_dollars = np.concatenate([np.zeros((n_games, 1)),
                                        log["mm_dollars_after_bureau"][:, :-1]], axis=1)
        gap_fill = max(
            np.abs(d_pounds + (log["mm_pounds_after_fill"] - start_pounds)).max(),
            np.abs(d_dollars + (log["mm_dollars_after_fill"] - start_dollars)).max())
        # the MM's Bureau trades
        bp = log["mm_pounds_after_bureau"] - log["mm_pounds_after_fill"]
        bd = log["mm_dollars_after_bureau"] - log["mm_dollars_after_fill"]
        rate = log["hidden_rate"]
        sold_pounds = bp < 0
        gap_bureau = max(
            np.abs(np.where(sold_pounds, bp * (1 - f) * rate + bd, 0.0)).max(),
            np.abs(np.where(~sold_pounds, bd * (1 - f) / rate + bp, 0.0)).max())
        before = (log["mm_pounds_after_fill"], log["mm_dollars_after_fill"])
        after = (log["mm_pounds_after_bureau"], log["mm_dollars_after_bureau"])
        deepest_before = np.minimum(np.minimum(before[0], 0.0), np.minimum(before[1], 0.0))
        deepest_after = np.minimum(np.minimum(after[0], 0.0), np.minimum(after[1], 0.0))
        assert np.all(deepest_after >= deepest_before - 1e-6), "a repay deepened a short"
        for was, now in zip(before, after):
            assert not np.any((was >= -1e-6) & (now < -1e-6)), "a repay turned a long leg short"
        end_pounds, end_dollars, a5 = log["mm_pounds_end"], log["mm_dollars_end"], log["settlement_rate"]
        gap_clear = max(
            np.minimum(np.clip(-end_pounds, 0, None), np.clip(end_dollars, 0, None) / a5).max(),
            np.minimum(np.clip(-end_dollars, 0, None) / a5, np.clip(end_pounds, 0, None)).max())
        # the order: a permutation per game and round, every slot used once
        pos = log["position"]
        assert np.all(np.sort(pos, axis=1) == np.arange(cfg.n_offers)[None, :, None]), \
            "a round's order is not a permutation of the offers"
        passed = gap_fill < 1e-6 and gap_bureau < 1e-6 and gap_clear < 1e-6
        print(f"  check 2  {cfg.name:<4} {label:<14} books conserve, {n_games} games "
              f"(fee {f}, filled {filled_share:.1%} of volume): fill gap "
              f"{gap_fill:.2e}  Bureau gap {gap_bureau:.2e}  round-5 gap "
              f"{gap_clear:.2e} GBP  [{'ok' if passed else 'FAIL'}]", flush=True)
        assert passed, "the books do not conserve value"


def _trader_grads(losses, trader_nets, retain=False):
    """Every trader's gradient of ITS OWN loss, all computed before any
    weight moves (a step would invalidate the shared graph). Returns a list
    of per-parameter gradient lists."""
    grads = []
    for i, (loss, net) in enumerate(zip(losses, trader_nets)):
        params = list(net.parameters())
        g = torch.autograd.grad(loss, params,
                                retain_graph=retain or (i < len(losses) - 1),
                                allow_unused=True)
        grads.append([torch.zeros_like(p) if gi is None else gi
                      for p, gi in zip(params, g)])
    return grads


def check_freeze_seam(cfg, seed_offset=4):
    """check 3 -- the MM's step moves only the MM; the traders' simultaneous
    step moves every trader by its OWN gradient and nothing else; the
    frozen partner passes gradient through. At N = 1 the traders' gradient
    via autograd.grad must equal loss.backward's to the bit -- that is what
    makes this file's block cotrain.py's block."""
    mm_net, trader_nets = build_agents(cfg)
    gen = torch.Generator().manual_seed(SEED_PATH + seed_offset)
    batch = torch.randn(BATCH, cfg.rounds + 1, generator=gen)
    order = draw_order(cfg, BATCH, SEED_PATH + ORDER_OFFSET + seed_offset)
    # the MM's step
    set_learning(mm_net, True)
    for net in trader_nets:
        set_learning(net, False)
    before = [_snapshot(net) for net in trader_nets]
    mm_pl, _, _ = rollout(cfg, mm_net, trader_nets, batch, order, tau=tau_for("mm", 0))
    loss = -mm_pl.mean()
    loss.backward()
    assert all(p.grad is None for net in trader_nets for p in net.parameters()), \
        "check 3: a frozen trader received gradient"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in mm_net.parameters()), "check 3: the MM received NO gradient"
    torch.optim.AdamW(mm_net.parameters(), lr=1e-3).step()
    assert all(_unchanged(b, net) for b, net in zip(before, trader_nets)), \
        "check 3: a frozen trader moved"
    # the traders' step
    set_learning(mm_net, False)
    for net in trader_nets:
        set_learning(net, True)
    mm_before = _snapshot(mm_net)
    _, trader_pls, _ = rollout(cfg, mm_net, trader_nets, batch, order,
                               tau=tau_for("trader", 0))
    losses = [-pl.mean() for pl in trader_pls]
    grads = _trader_grads(losses, trader_nets, retain=True)
    assert all(p.grad is None for p in mm_net.parameters()), \
        "check 3: the frozen MM received gradient"
    for i, (net, g) in enumerate(zip(trader_nets, grads)):
        assert any(gi.abs().sum() > 0 for gi in g), \
            f"check 3: trader {i} received NO gradient from its own P/L"
    if cfg.n == 1:
        # the same gradient by loss.backward, to the bit
        net = trader_nets[0]
        net.zero_grad(set_to_none=True)
        losses[0].backward()
        assert all(torch.equal(p.grad, g) for p, g in zip(net.parameters(), grads[0])), \
            "check 3: autograd.grad and backward disagree at N = 1"
    for net, g in zip(trader_nets, grads):
        for p, gi in zip(net.parameters(), g):
            p.grad = gi
        torch.optim.AdamW(net.parameters(), lr=1e-3).step()
    assert _unchanged(mm_before, mm_net), "check 3: the frozen MM moved"
    print(f"  check 3  freeze seam ({cfg.name}): the MM's step moves the MM only; "
          f"each of {cfg.n} trader(s) moves by its own gradient only"
          + ("; autograd.grad == backward at N = 1" if cfg.n == 1 else "")
          + "  [ok]", flush=True)


def check_order_uniform(cfg, n_games=20_000, seed=11):
    """check 4 -- every offer takes every slot with frequency 1/K (K offers
    a round), and the order is fresh per game and round (so nothing about
    the game leaks into the slot)."""
    K = cfg.n_offers
    if K == 1:
        return
    order = draw_order(cfg, n_games, seed)
    worst = 0.0
    for i in range(K):
        for slot in range(K):
            share = float((order[:, :, slot] == i).double().mean())
            worst = max(worst, abs(share - 1.0 / K))
    se = np.sqrt((1 / K) * (1 - 1 / K) / (n_games * cfg.rounds))
    assert worst < 4 * se, "check 4: the arrival order is not uniform"
    # fresh per round: the round-to-round agreement of slot 0 is ~1/K
    same = float((order[:, 1:, 0] == order[:, :-1, 0]).double().mean())
    assert abs(same - 1.0 / K) < 4 * se * np.sqrt(cfg.rounds / (cfg.rounds - 1)), \
        "check 4: the order is not reshuffled every round"
    print(f"  check 4  order ({cfg.name}): every one of {K} offer(s) in every slot with "
          f"frequency 1/{K} +/- {worst:.4f}; reshuffled every round  [ok]",
          flush=True)


def check_bdc_floor(cfg):
    """check 5 -- each card's per-path always-use-the-Bureau result must
    average to the analytic floor, on the evaluation paths (cotrain.check_bdc_floor)."""
    gen = torch.Generator().manual_seed(SEED_PATH + 2)
    z = torch.randn(EVAL_PATHS, cfg.rounds + 1, generator=gen)
    for key, spec in zip(cfg.traders, cfg.specs):
        _, floor = bdc_baseline(spec)
        pl = always_bdc_pl(spec, z)
        mean, two_se = float(pl.mean()), 2 * float(pl.std()) / np.sqrt(EVAL_PATHS)
        assert abs(mean - floor) <= two_se, (
            f"check 5: always-BdC per path {mean * 100:+.4f}% +/- "
            f"{two_se * 100:.4f}% disagrees with the analytic floor "
            f"{floor * 100:+.4f}% for trader {key}")
        print(f"  check 5  always-BdC floor ({key}, side {spec.side}): analytic {floor * 100:+.4f}%, "
              f"per path on {EVAL_PATHS:,} eval paths {mean * 100:+.4f}% +/- "
              f"{two_se * 100:.4f}%  [ok]", flush=True)


def check_replays_cotrain_pair(row):
    """check 0 -- if cotrain.py has run this side at this fee, its saved
    pair replayed through the N = 1 configuration on the same evaluation
    paths must give cotrain_eval.npz's per-path P/Ls: the reduction on the
    REAL artefact, not a fresh net. Skipped when the artefacts are absent
    or the row is not one of that run's cards. Built at that run's own
    game and visibility, as check 1."""
    cfg = make_config((row,), mm_arch="round", see_size=M.SEE_OFFER_SIZE, two_way=False)
    side = cfg.keys[0]
    if side is None:
        print(f"  [check 0 skipped] {cfg.traders[0]} is not a single-trader card; no "
              f"cotrain artefact applies to it", flush=True)
        return
    run = results_path(f"cotrain/{side}_g_{MM_FEE:g}_BdC_FEE_{cfg.params.bdc_fee}")
    pair, ev = run / "pair_final.pth", run / "cotrain_eval.npz"
    if not (pair.exists() and ev.exists()):
        print(f"  [check 0 skipped] no cotrain artefacts for side {side} at fee "
              f"{MM_FEE:g} in {run}", flush=True)
        return
    saved = torch.load(pair)
    trader_net, mm_net = T.build_net(SEED_INIT_TRADER), build_mm_net(cfg, SEED_INIT_MM)
    trader_net.load_state_dict(saved["trader"])
    mm_net.load_state_dict(saved["mm"])
    theirs = np.load(ev, allow_pickle=False)
    n = theirs["eval_mm_pl"].shape[1]
    if n != EVAL_PATHS:
        print(f"  [check 0 skipped] cotrain evaluated {n:,} paths, this file "
              f"{EVAL_PATHS:,}", flush=True)
        return
    gen = torch.Generator().manual_seed(SEED_PATH + 2)
    z = torch.randn(EVAL_PATHS, cfg.rounds + 1, generator=gen)
    with torch.no_grad():
        mm_pl, trader_pls, _ = rollout(cfg, mm_net, [trader_net], z,
                                       draw_order(cfg, EVAL_PATHS, 0), tau=None)
    gap = max(float(np.abs(mm_pl.numpy() - theirs["eval_mm_pl"][0]).max()),
              float(np.abs(trader_pls[0].numpy() - theirs["eval_trader_pl"][0]).max()))
    print(f"  check 0  cotrain's side-{side} pair replayed at N=1 on its "
          f"{EVAL_PATHS:,} eval paths: max |gap| {gap:.1e}  "
          f"[{'exact' if gap == 0.0 else ('ok' if gap < 1e-9 else 'FAIL')}]",
          flush=True)
    assert gap < 1e-9, "the N = 1 game does not reproduce cotrain's evaluation"


# ---- the architecture checks: reduction, the column gate, order, the block --
RECORD_COMPARE_KEYS = TRADER_KEYS + MM_KEYS + ("direction", "settlement_rate",
                                               "mm_pounds_end", "mm_dollars_end")
MONEY_KEYS = ("offered_amount", "mm_pounds_at_verdict", "mm_dollars_at_verdict",
              "trader_held_after_fill", "trader_banked_after_fill",
              "trader_held_after_bureau", "trader_banked_after_bureau",
              "mm_pounds_after_fill", "mm_dollars_after_fill",
              "mm_pounds_after_bureau", "mm_dollars_after_bureau",
              "mm_pounds_end", "mm_dollars_end")   # read per unit of capital


def _record_gap(mine, theirs, scale=None):
    """max |gap| over the P/Ls and every record key; scale, if given, maps
    a key to a divisor (used to read raw pound/dollar books per unit of
    capital)."""
    (mm_pl, trader_pls, log), (ref_mm, ref_tr, ref_log) = mine, theirs
    gap = max(float((mm_pl - ref_mm).abs().max()),
              max(float((a - b).abs().max()) for a, b in zip(trader_pls, ref_tr)))
    for key in RECORD_COMPARE_KEYS:
        divisor = 1.0 if scale is None else scale(key)
        gap = max(gap, float(np.abs(log[key] - ref_log[key]).max()) / divisor)
    return gap


def _plays(cfg, seed_init=None):
    """The plays every replay check uses: the training-init net (constant
    level, hides arithmetic on the level), a random-head net (its level
    moves with every input), hard and soft, then every reference rule."""
    seed_init = SEED_INIT_MM if seed_init is None else seed_init
    init_net = build_mm_net(cfg, seed_init)
    with torch.random.fork_rng():
        torch.manual_seed(seed_init + 1)
        random_net = T.AlphaNet(n_in=cfg.n_mm_feat, n_out=2)
    return ([(f"{name} hard", net, dict(tau=None)) for name, net in
             (("init net", init_net), ("random net", random_net))]
            + [(f"{name} soft tau 0.2", net, dict(tau=0.2)) for name, net in
               (("init net", init_net), ("random net", random_net))]
            + [(rule.name, None, dict(rule=rule)) for rule in reference_rules(0.01)])


def _money_scale(cfg):
    return lambda key: cfg.capital if key in MONEY_KEYS else 1.0


def check_arms_coincide_at_one(row, n_games=2000, seed=6):
    """check 6 -- at N = 1 the "bid" arm IS the "round" arm to the bit (at
    the run's own size setting): with a single offer its live book is the
    round-open book and there is nothing still to come. Same nets (state
    dicts equal), same paths, hard and soft games and every rule, P/L and
    record alike. The "turn" arm reads every offer of the round including
    the one being judged, so even at N = 1 it has two columns the round arm
    does not; its reduction is check 7's, which zero-weights them."""
    cfg_ref = make_config((row,), mm_arch="round", two_way=False)
    trader_nets = [T.build_net(SEED_INIT_TRADER)]
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_games, cfg_ref.rounds + 1, generator=gen)
    order = draw_order(cfg_ref, n_games, seed)
    worst = 0.0
    for arch in ("bid",):
        cfg = make_config((row,), mm_arch=arch, two_way=False)
        assert cfg.n_mm_feat == cfg_ref.n_mm_feat, f"check 6: the {arch} arm has extra columns at N = 1"
        assert _unchanged(_snapshot(build_mm_net(cfg_ref, SEED_INIT_MM)),
                          build_mm_net(cfg, SEED_INIT_MM)), f"check 6: build_mm_net differs ({arch})"
        for label, net, kw in _plays(cfg_ref):
            with torch.no_grad():
                theirs = rollout(cfg_ref, net, trader_nets, z, order, fee=0.01, record=True, **kw)
                mine = rollout(cfg, net, trader_nets, z, order, fee=0.01, record=True, **kw)
            gap = _record_gap(mine, theirs)
            worst = max(worst, gap)
            print(f"  check 6  N=1 ({cfg.traders[0]}, side {cfg.specs[0].side}) {arch:<4} vs round, "
                  f"{label:<24} {n_games} games: max |gap| {gap:.1e}  "
                  f"[{'ok' if gap == 0.0 else 'FAIL'}]", flush=True)
    assert worst == 0.0, "check 6: the arms do not coincide at N = 1"


def _column_map(cfg, cfg_ref):
    """(reference column, this arm's column) for every column both layouts
    have: the five base columns, then direction, then the sub-round. This
    file puts the size column right after the base five, so the reference's
    direction and sub-round columns sit one place further right when the
    arm sees size and the reference does not."""
    pairs = [(j, j) for j in range(5)]
    j_ref, j = 5 + int(cfg_ref.see_size), 5 + int(cfg.see_size)
    if cfg.see_direction:
        pairs.append((j_ref, j))
        j_ref, j = j_ref + 1, j + 1
    if cfg_ref.see_queue:
        assert cfg.see_queue, "check 7: the reference has a sub-round column this arm lacks"
        pairs.append((j_ref, j))
        j_ref, j = j_ref + 1, j + 1
    assert j_ref == cfg_ref.n_mm_feat, "check 7: the reference layout moved"
    return pairs


def _copy_with_columns_zeroed(net_from, cfg_to, pairs):
    """A net of cfg_to's input width carrying net_from's weights on the
    mapped input columns and ZERO weight on every other input column; every
    other parameter copied. Only fc_in and shortcut read the input."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        net_to = T.AlphaNet(n_in=cfg_to.n_mm_feat, n_out=2)
    with torch.no_grad():
        for (name, p_from), (name_to, p_to) in zip(net_from.named_parameters(),
                                                   net_to.named_parameters()):
            assert name == name_to
            if name in ("fc_in.weight", "shortcut.weight"):
                p_to.zero_()
                for j_from, j_to in pairs:
                    p_to[:, j_to] = p_from[:, j_from]
            else:
                p_to.copy_(p_from)
    return net_to


def check_columns_reduce(rows, n_games=2000, seed=8, tol=1e-12):
    """check 7 -- the reduction gate for the added input columns: the run's
    arm, with every column beyond the plain ones (size, flow block) given
    ZERO weight, plays the (round or bid, size off) game exactly. So each
    column is a pure addition and cannot be silently changing anything else.
    Random-head nets, hard and soft.
    Tolerance 1e-12, not 0: a wider matmul may accumulate the shared terms
    in a different order (BLAS-dependent), and the raw books are read per
    unit of capital. Then: those columns are LIVE, not decoration -- a
    random-head net's MM loss has non-zero gradient on every one of them."""
    rows = tuple(rows)
    cfg = make_config(rows)                       # the run's arm, size and flow
    ref_arch = "bid" if cfg.live_book else "round"
    cfg_ref = make_config(rows, mm_arch=ref_arch, see_size=False)
    pairs = _column_map(cfg, cfg_ref)
    with torch.random.fork_rng():
        torch.manual_seed(SEED_INIT_MM + 1)
        net_ref = T.AlphaNet(n_in=cfg_ref.n_mm_feat, n_out=2)
    net = _copy_with_columns_zeroed(net_ref, cfg, pairs)
    trader_nets = build_traders(cfg)
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_games, cfg.rounds + 1, generator=gen)
    order = draw_order(cfg, n_games, seed)
    scale = _money_scale(cfg)
    worst = 0.0
    for label, kw in (("hard", dict(tau=None)), ("soft tau 0.2", dict(tau=0.2))):
        with torch.no_grad():
            theirs = rollout(cfg_ref, net_ref, trader_nets, z, order, fee=0.01, record=True, **kw)
            mine = rollout(cfg, net, trader_nets, z, order, fee=0.01, record=True, **kw)
        gap = _record_gap(mine, theirs, scale)
        worst = max(worst, gap)
        print(f"  check 7  {cfg.name:<4} ({cfg.mm_arch}, "
              f"size {'on' if cfg.see_size else 'off'}, {cfg.n_mm_feat} inputs) with the "
              f"{cfg.n_mm_feat - cfg_ref.n_mm_feat} extra column(s) zero-weighted vs ({ref_arch}, size off), "
              f"random net {label:<12} {n_games} games: max |gap| {gap:.1e}  "
              f"[{'exact' if gap == 0.0 else ('ok' if gap <= tol else 'FAIL')}]", flush=True)
    assert worst <= tol, "check 7: an extra column is not wired as a pure addition"
    extra = sorted(set(range(cfg.n_mm_feat)) - {j for _, j in pairs})
    if extra:
        with torch.random.fork_rng():
            torch.manual_seed(SEED_INIT_MM + 2)
            live = T.AlphaNet(n_in=cfg.n_mm_feat, n_out=2)
        set_learning(live, True)
        for net in trader_nets:
            set_learning(net, False)
        mm_pl, _, _ = rollout(cfg, live, trader_nets, z[:500], order[:500], fee=0.01, tau=0.2)
        (-mm_pl.mean()).backward()
        g = (live.fc_in.weight.grad[:, extra].abs().sum(dim=0)
             + live.shortcut.weight.grad[:, extra].abs().sum(dim=0))
        print(f"  check 7  {cfg.name:<4} ({cfg.mm_arch}) the {len(extra)} extra column(s) carry gradient "
              f"to the MM (min |grad| {float(g.min()):.1e}, soft game, 500 games)  "
              f"[{'ok' if bool((g > 0).all()) else 'FAIL'}]", flush=True)
        assert bool((g > 0).all()), "check 7: an extra input column receives no gradient"


def check_order_invariance(rows, n_games=2000, seeds=(11, 12), tol=1e-12):
    """check 8 -- with the level set from the round-open book ("round",
    "turn"), the arrival order cannot change a verdict: two different order
    draws give the same hard fills exactly and the same levels, P/Ls, soft
    fills and books (money per unit of capital) to 1e-12 -- not 0, because
    the book's floating-point sum order differs between the two orders and
    that roundoff reaches the next round's level. For "turn" this is also
    the proof that the flow block is sorted and leaks no slot. For "bid"
    the live book must make the soft-game levels DIFFER between orders, or
    the book is not live. N = 1 has one order and the check is vacuous."""
    rows = tuple(rows)
    cfg = make_config(rows)
    if cfg.n_offers == 1:
        print(f"  [check 8 skipped] one offer a round has one arrival order", flush=True)
        return
    with torch.random.fork_rng():
        torch.manual_seed(SEED_INIT_MM + 1)
        net = T.AlphaNet(n_in=cfg.n_mm_feat, n_out=2)
    trader_nets = build_traders(cfg)
    gen = torch.Generator().manual_seed(seeds[0])
    z = torch.randn(n_games, cfg.rounds + 1, generator=gen)
    orders = [draw_order(cfg, n_games, s) for s in seeds]
    assert not torch.equal(*orders), "check 8: the two order draws coincide"
    scale = _money_scale(cfg)
    for label, kw in (("hard", dict(tau=None)), ("soft tau 0.2", dict(tau=0.2))):
        with torch.no_grad():
            a, b = [rollout(cfg, net, trader_nets, z, order, fee=0.01, record=True, **kw)
                    for order in orders]
        fills_equal = bool(np.array_equal(a[2]["fill_weight"], b[2]["fill_weight"]))
        level_gap = float(np.abs(a[2]["level"] - b[2]["level"]).max())
        keys = tuple(k for k in RECORD_COMPARE_KEYS if k != "position")
        gap = max(float((a[0] - b[0]).abs().max()),
                  max(float((x - y).abs().max()) for x, y in zip(a[1], b[1])),
                  max(float(np.abs(a[2][k] - b[2][k]).max()) / scale(k) for k in keys))
        if cfg.live_book:
            ok = (label == "hard") or level_gap > 0.0
            print(f"  check 8  {cfg.name:<4} ({cfg.mm_arch}) two arrival orders, {label:<12}: "
                  f"levels differ by up to {level_gap:.1e} (must be > 0 in the soft game: "
                  f"the book is live)  [{'ok' if ok else 'FAIL'}]", flush=True)
            assert ok, "check 8: the bid arm's level ignores the live book"
        else:
            ok = gap <= tol and (fills_equal or label != "hard")
            print(f"  check 8  {cfg.name:<4} ({cfg.mm_arch}) two arrival orders, {label:<12}: "
                  + (f"hard fills {'identical' if fills_equal else 'DIFFER'}, " if label == "hard"
                     else f"soft fills within {float(np.abs(a[2]['fill_weight'] - b[2]['fill_weight']).max()):.1e}, ")
                  + f"max |gap| {gap:.1e} (levels {level_gap:.1e})  [{'ok' if ok else 'FAIL'}]", flush=True)
            assert ok, f"check 8: the {cfg.mm_arch} arm's verdicts depend on the arrival order"


def check_flow_block(rows, n_games=500, seed=13):
    """check 9 -- the flow block, on random quotes. One-way layout:
    grouped by direction, sorted best-for-the-MM first, sizes carried with
    their prices. Two-way layout: (edge, signed pound size) per offer at the
    TRUE rate, sorted best-edge first, direction-free. The block holds EVERY
    offer of the round, the judged one included: moving any offer's price or
    size changes it, the judged offer's included. That the block carries no
    slot is check 8 (order invariance) for the turn arm."""
    rows = tuple(rows)
    cfg = make_config(rows, mm_arch="turn")
    N = cfg.n_offers
    gen = torch.Generator().manual_seed(seed)
    X = cfg.params.a0 * torch.ones(n_games)
    quoted = cfg.params.a0 * (1.0 + 0.1 * torch.randn(n_games, N, generator=gen))
    offered = torch.rand(n_games, N, generator=gen) * torch.tensor(
        [cfg.specs[t].L for t in cfg.offer_trader])
    base = torch.tensor(cfg.offer_directions)
    if cfg.two_way:
        # directions are data: a random mix, as the traders would produce
        flip = torch.where(torch.rand(n_games, N, generator=gen) < 0.5, -1.0, 1.0)
        live = base.unsqueeze(0) * flip
    else:
        live = None
    kw = dict(live_directions=live)
    all_block = flow_block(cfg, quoted, offered, X, **kw)
    # (i) the layout, by direct construction
    if cfg.two_way:
        edge = live * (X.unsqueeze(1) - quoted) / X.unsqueeze(1)
        pounds = torch.where(live > 0, offered, offered / X.unsqueeze(1))
        perm = torch.argsort(-edge, dim=1)
        expect = torch.cat([edge.gather(1, perm),
                            (live * pounds / cfg.capital).gather(1, perm)], dim=1)
        sorted_ok = bool((all_block[:, :N].diff(dim=1) <= 1e-15).all())   # best edge first
    else:
        expect_cols = []
        for d, unit in ((1.0, cfg.capital), (-1.0, cfg.params.a0 * cfg.capital)):
            idx = torch.nonzero(base == d).squeeze(1)
            if idx.numel() == 0:
                continue
            p = quoted[:, idx] / X.unsqueeze(1) - 1.0
            perm = torch.argsort(d * p, dim=1)
            expect_cols += [p.gather(1, perm), (offered[:, idx] / unit).gather(1, perm)]
        expect = torch.cat(expect_cols, dim=1)
        sorted_ok = True
    layout_gap = float((all_block - expect).abs().max())
    assert layout_gap == 0.0 and sorted_ok, "check 9: the flow block's layout is not the documented one"
    # every offer is in the block: moving ANY of them, including the one the
    # MM is judging, must move it -- that is what makes the arm's policy
    # class wider than a cut
    big = 1.0 + 0.5 * torch.rand(n_games, generator=gen)
    rows_ix = torch.arange(n_games)
    moved = []
    for j in range(N):
        q2, o2 = quoted.clone(), offered.clone()
        q2[rows_ix, j] = quoted[rows_ix, j] * big
        o2[rows_ix, j] = offered[rows_ix, j] * big
        moved.append(float((flow_block(cfg, q2, o2, X, **kw) - all_block).abs().max()))
    ok = min(moved) > 0.0
    print(f"  check 9  flow block ({cfg.name}, {'two-way' if cfg.two_way else 'one-way'} layout): "
          f"layout as documented (gap {layout_gap:.1e}); every offer of the round is in it -- "
          f"moving any one moves the block (smallest move {min(moved):.1e} over {N} offers)  "
          f"[{'ok' if ok else 'FAIL'}]", flush=True)
    assert ok, "check 9: the flow block is not built as documented"


def check_book_rule(rows, n_games=2000, seed=14):
    """check 10 -- the book rule. (i) At N = 1 (each side) its hard fills
    are the fee-aware rule's exactly and so is its P/L (both flavours).
    (ii) At N > 1, on random offers its round value is >= the naive and
    the fee-aware subsets' (they are among the candidates) on every game
    and round, and on a hand example it accepts a crossing pair that the
    fee-aware rule rejects and rejects a pair that does not pay: the pound
    seller at 1.24 (> (1-g)X) and the dollar seller at 1.26 (< X/(1-g))
    of equal size cross for +1,600 GBP and are both taken; the same pound
    seller against a dollar seller a thirtieth the size is not.
    (iii) On the record, the recorded level reproduces the fill by sign."""
    rows = tuple(rows)
    if len(rows) > BOOK_RULE_MAX_OFFERS:
        print(f"  [check 10 skipped] {len(rows)} offers a round: the book rule "
              f"enumerates 2^(offers) subsets and is not run past "
              f"{BOOK_RULE_MAX_OFFERS} offers", flush=True)
        return
    fee = 0.01
    # (i) one offer on the table: the fee-aware rule to the bit
    first = {}
    for row in rows:
        first.setdefault(row["side"], row)
    for row in first.values():
        cfg = make_config((row,), mm_arch="round", two_way=False)
        trader_nets = [T.build_net(SEED_INIT_TRADER)]
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n_games, cfg.rounds + 1, generator=gen)
        order = draw_order(cfg, n_games, seed)
        for repay in (1.0, 0.0):
            fa = M.Rule("fee-aware+" + ("flatten" if repay else "carry"), True, repay)
            bk = BookRule("book+" + ("flatten" if repay else "carry"), repay)
            with torch.no_grad():
                a = rollout(cfg, None, trader_nets, z, order, fee=fee, rule=fa, record=True)
                b = rollout(cfg, None, trader_nets, z, order, fee=fee, rule=bk, record=True)
            same_fills = bool(np.array_equal(a[2]["fill_weight"], b[2]["fill_weight"]))
            gap = float((a[0] - b[0]).abs().max())
            sign_ok = bool(np.all((b[2]["direction"] * (b[2]["level"] - b[2]["quoted_price"]) > 0)
                                  == (b[2]["fill_weight"] > 0.5)))
            print(f"  check 10 N=1 ({cfg.traders[0]}, side {cfg.specs[0].side}) {bk.name} vs {fa.name}: "
                  f"fills {'identical' if same_fills else 'DIFFER'}, MM P/L max |gap| {gap:.1e}; "
                  f"level reproduces the fill by sign  [{'ok' if same_fills and gap == 0.0 and sign_ok else 'FAIL'}]",
                  flush=True)
            assert same_fills and gap == 0.0 and sign_ok, "check 10: the book rule is not fee-aware at N = 1"
    cfg = make_config(rows, mm_arch="round", two_way=False)     # one offer per trader here
    if cfg.n == 1:
        return
    # (ii) dominance on random offers, and the hand example
    gen = torch.Generator().manual_seed(seed + 1)
    X = cfg.params.a0 * (1.0 + 0.04 * torch.randn(n_games, generator=gen))
    quoted = X.unsqueeze(1) * (1.0 + 0.03 * torch.randn(n_games, cfg.n, generator=gen))
    offered = torch.rand(n_games, cfg.n, generator=gen) * torch.tensor([s.L for s in cfg.specs])
    P0 = torch.zeros(n_games)
    D0 = torch.zeros(n_games)
    d = torch.tensor(cfg.offer_directions)
    fill, level = book_verdicts(cfg, quoted, offered, X, P0, D0, fee)
    sign = (d * (level - quoted) > 0).double()
    sign_ok = bool(torch.equal(sign[offered > 0], fill[offered > 0]))
    assert sign_ok, "check 10: the recorded level does not reproduce the book rule's fill by sign"

    def value_of(fills):
        dp = torch.where(d > 0, offered, -offered / quoted)
        dd = torch.where(d > 0, -offered * quoted, offered)
        P = P0 + (dp * fills).sum(dim=1)
        D = D0 + (dd * fills).sum(dim=1)
        P2, D2 = M.mm_bureau(P, D, torch.ones_like(P), X, fee)
        return P2 + D2 / X
    naive = (d * (X.unsqueeze(1) - quoted) > 0).double()
    cut = torch.where(d > 0, (1 - fee) * X.unsqueeze(1), X.unsqueeze(1) / (1 - fee))
    fee_aware = (d * (cut - quoted) > 0).double()
    v_book, v_naive, v_fa = value_of(fill), value_of(naive), value_of(fee_aware)
    worst = float(torch.maximum(v_naive - v_book, v_fa - v_book).max())
    beats = int(((v_book > v_fa + 1e-9)).sum())
    print(f"  check 10 {cfg.name:<4} book rule on {n_games} random rounds: round value >= naive and "
          f"fee-aware subsets on every round (worst shortfall {worst:.1e} GBP); strictly better on "
          f"{beats} of them; level reproduces the fill by sign  [{'ok' if worst <= 1e-6 else 'FAIL'}]", flush=True)
    assert worst <= 1e-6, "check 10: the book rule's subset is not the best of the candidates"
    sides = [s.side for s in cfg.specs]
    if "A" in sides and "B" in sides:
        iA, iB = sides.index("A"), sides.index("B")
        X1 = torch.tensor([1.25, 1.25])
        q = torch.zeros(2, cfg.n)
        p = torch.full((2, cfg.n), 1.25)
        q[:, iA] = 100_000.0
        p[:, iA] = 1.24                       # above (1-g)X = 1.2375: rejected alone
        p[:, iB] = 1.26                       # below X/(1-g) = 1.2626: rejected alone
        q[0, iB] = 125_000.0                  # 100k GBP worth: the pair crosses for +1,600 GBP
        q[1, iB] = 12_600.0 / 3               # a thirtieth of it: the pair does not pay
        f, _ = book_verdicts(cfg, p, q, X1, torch.zeros(2), torch.zeros(2), fee)
        takes_pair = bool(f[0, iA] == 1 and f[0, iB] == 1)
        rejects_pair = bool(f[1, iA] == 0)
        print(f"  check 10 hand example: the crossing pair 1.24 / 1.26 (fee-aware rejects both) is taken "
              f"{'[ok]' if takes_pair else '[FAIL]'}; the same pound seller against a tiny dollar seller "
              f"is rejected {'[ok]' if rejects_pair else '[FAIL]'}", flush=True)
        assert takes_pair and rejects_pair, "check 10: the book rule does not net a crossing pair"


def _pin_fraction(nets, logit):
    """Every trader's fraction head pinned to a constant: sigmoid(+-50) is
    exactly 1.0 or 0.0 in float64, so the signed fraction 2f - 1 is exactly
    +1 or -1 and the one-way fraction f is exactly 1 or 0 -- the two games
    then offer the same amounts, which is what makes check 11 a bit-for-bit
    reduction rather than a tolerance."""
    for net in nets:
        with torch.no_grad():
            net.head.weight[1].zero_()
            net.head.bias[1] = logit


def check_two_way_reduces(rows, n_games=2000, seed=15):
    """check 11 -- the two-way reduction. A trader whose signed fraction is
    pinned at +1 (it offers all of what it HOLDS, forward, every round) and
    a one-way trader pinned to the same amount play the SAME game: same
    prices, same amounts, same directions, so P/L and record agree to the
    bit on every rule, hard and soft. Then the reverse branch is live:
    pinned at -1 the offers sell the banked balance back, they carry the
    opposite direction, they fill, they move the traders' P/L, and the
    loss has gradient on the fraction head that chooses the direction."""
    rows = tuple(rows)
    cfg2 = make_config(rows, two_way=True)
    cfg1 = make_config(rows, two_way=False)
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_games, cfg1.rounds + 1, generator=gen)
    order = draw_order(cfg1, n_games, seed)
    assert torch.equal(order, draw_order(cfg2, n_games, seed)), "check 11: the orders differ"
    nets1, nets2 = build_traders(cfg1), build_traders(cfg2)
    _pin_fraction(nets1, 50.0)        # sigmoid(50) == 1.0 exactly: offers all it holds
    _pin_fraction(nets2, 50.0)        # 2 * 1.0 - 1 == 1.0 exactly: the same offer
    worst = 0.0
    for rule in reference_rules(0.01, cfg1.n_offers):
        with torch.no_grad():
            a = rollout(cfg1, None, nets1, z, order, fee=0.01, rule=rule, record=True)
            b = rollout(cfg2, None, nets2, z, order, fee=0.01, rule=rule, record=True)
        gap = _record_gap(b, a)
        worst = max(worst, gap)
        print(f"  check 11 {cfg1.name:<4} two-way pinned forward vs one-way, {rule.name:<18} "
              f"{n_games} games: max |gap| {gap:.1e}  [{'ok' if gap == 0.0 else 'FAIL'}]", flush=True)
        assert gap == 0.0, "check 11: the two-way game with a forward-pinned trader is not the one-way game"
    # the reverse branch: pinned at -1, the offers are reverse offers
    rev_nets = build_traders(cfg2)
    _pin_fraction(rev_nets, -50.0)
    rule = reference_rules(0.01, cfg1.n_offers)[0]
    with torch.no_grad():
        c = rollout(cfg2, None, rev_nets, z, order, fee=0.01, rule=rule, record=True)
    base = np.array(cfg2.offer_directions)[None, :, None]
    flipped = bool((c[2]["direction"] == -base).all())
    volume = float((c[2]["fill_weight"] * c[2]["offered_amount"]).sum())
    moved = max(float((x - y).abs().max()) for x, y in zip(b[1], c[1]))
    # and the direction choice carries gradient
    live = build_traders(cfg2)
    mm_net = build_mm_net(cfg2, SEED_INIT_MM)
    set_learning(mm_net, False)
    for net in live:
        set_learning(net, True)
    _, trader_pls, _ = rollout(cfg2, mm_net, live, z[:500], order[:500], fee=0.01, tau=0.2)
    grads = _trader_grads([-pl.mean() for pl in trader_pls], live)
    head = min(float(sum(g.abs().sum() for g, (name, _) in zip(gs, net.named_parameters())
                         if name.startswith("head.")))
               for gs, net in zip(grads, live))
    ok = flipped and volume > 0.0 and moved > 0.0 and head > 0.0
    print(f"  check 11 {cfg1.name:<4} the reverse branch is live: pinned at -1 every offer flips "
          f"direction {'[yes]' if flipped else '[NO]'}, {volume:,.2f} units filled, the traders' P/L "
          f"moves by up to {moved:.1e}, min |grad| on the fraction head {head:.1e}  "
          f"[{'ok' if ok else 'FAIL'}]", flush=True)
    assert ok, "check 11: the reverse branch is not live"


# ---------------------------------------------------------------------------
# 5. training -- alternating blocks: the MM, then all traders at once
# ---------------------------------------------------------------------------
def _batch(cfg, agent, step):
    offset = 10_000 if agent == "trader" else 20_000
    z = draw_paths(cfg, BATCH, SEED_PATH + offset + step, f"{agent} step {step}")
    return z, draw_order(cfg, z.shape[0], SEED_PATH + ORDER_OFFSET + offset + step)


def train_block(agent, cfg, mm_net, trader_nets, opt_mm, opts, scheds, step):
    if agent == "mm":
        set_learning(mm_net, True)
        for net in trader_nets:
            set_learning(net, False)
    else:
        set_learning(mm_net, False)
        for net in trader_nets:
            set_learning(net, True)
    last_pl = float("nan")
    for _ in range(N_INNER):
        tau = tau_for(agent, step)
        z, order = _batch(cfg, agent, step)     # a FRESH batch, as cotrain
        step += 1
        chunk = ACCUM_GAMES if ACCUM_GAMES else z.shape[0]
        parts = [(z[i:i + chunk], order[i:i + chunk])
                 for i in range(0, z.shape[0], chunk)]
        weights = [zc.shape[0] / z.shape[0] for zc, _ in parts]
        # the soft ramp is scaled by the batch's rate-move std, so each chunk
        # must be handed the WHOLE batch's scales to play the same game
        scales = (cfg.params.sd * z[:, :cfg.rounds].std(dim=0)).detach() \
            if len(parts) > 1 else None
        if agent == "mm":
            opt_mm.zero_grad()
            finite, pl_sum = True, 0.0
            for (zc, oc), w in zip(parts, weights):
                mm_pl, trader_pls, _ = rollout(cfg, mm_net, trader_nets, zc, oc,
                                               tau=tau, move_scales=scales)
                loss = -mm_pl.mean() * w        # sum over chunks = the batch mean
                if not torch.isfinite(loss):
                    finite = False
                    break
                loss.backward()                 # gradients ACCUMULATE across chunks
                pl_sum += -loss.item()
            if not finite:
                print(f"  [guard] mm loss non-finite at its step {step - 1}. "
                      f"Skipping step.", flush=True)
                opt_mm.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(mm_net.parameters(), CLIP_NORM)
            opt_mm.step()
            last_pl = pl_sum
        else:
            finite = True
            totals = [None] * len(trader_nets)
            loss_vals = [0.0] * len(trader_nets)
            for (zc, oc), w in zip(parts, weights):
                mm_pl, trader_pls, _ = rollout(cfg, mm_net, trader_nets, zc, oc,
                                               tau=tau, move_scales=scales)
                losses = [-pl.mean() * w for pl in trader_pls]
                if not all(torch.isfinite(loss) for loss in losses):
                    finite = False
                    break
                grads = _trader_grads(losses, trader_nets)    # all before any step
                for i, g in enumerate(grads):
                    totals[i] = (g if totals[i] is None else
                                 [a + b for a, b in zip(totals[i], g)])
                    loss_vals[i] += losses[i].item()
            if not finite:
                print(f"  [guard] a trader loss non-finite at step {step - 1}. "
                      f"Skipping step.", flush=True)
                continue
            for net, opt, sched, g, loss_val in zip(trader_nets, opts, scheds,
                                                    totals, loss_vals):
                opt.zero_grad()
                for p, gi in zip(net.parameters(), g):
                    p.grad = gi
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
                opt.step()
                if sched is not None:
                    sched.step(loss_val)
            last_pl = float(np.mean([-loss_val for loss_val in loss_vals]))
    return step, tau, last_pl


def validate(cfg, mm_net, trader_nets, z, order, bdc_pls):
    """The HARD game on the fixed validation paths; the rules recomputed
    against the CURRENT traders; each trader paired with its floor."""
    set_learning(mm_net, False)
    for net in trader_nets:
        set_learning(net, False)
    rules = reference_rules(MM_FEE, cfg.n_offers)
    with torch.no_grad():
        mm_pl, trader_pls, log = rollout(cfg, mm_net, trader_nets, z, order,
                                         tau=None, fee=MM_FEE, record=True)
        rule_pl = {rule.name: rollout(cfg, None, trader_nets, z, order, rule=rule,
                                      fee=MM_FEE)[0] for rule in rules}
    benchmark_rule = max(rules, key=lambda rule: float(rule_pl[rule.name].mean()))
    agree, filled, filled_benchmark = benchmark_agreement(log, benchmark_rule, MM_FEE, cfg)
    thresholds = thresholds_by_direction(log, cfg.rounds)
    edges = [M.paired(pl, floor) for pl, floor in zip(trader_pls, bdc_pls)]
    fraction = np.array([float(np.median(log["offered_amount"][:, i, 0])) / spec.L
                         for i, spec in enumerate(cfg.specs)])      # the forward offer
    own = np.array(cfg.offer_trader)
    # two-way: a trader's offers can be in either currency, so its filled
    # share is taken in pound-equivalent at the true rate; one-way, directly
    amounts = offered_pound_volume(log) if cfg.two_way else log["offered_amount"]
    filled_i = np.array([
        float((log["fill_weight"][:, own == i] * amounts[:, own == i]).sum()
              / max(amounts[:, own == i].sum(), 1e-300)) for i in range(cfg.n)])
    return dict(mm_pl=mm_pl, trader_pls=trader_pls, rule_pl=rule_pl,
                edge=np.array([e for e, _ in edges]),
                edge_2se=np.array([s for _, s in edges]),
                offer_rate=np.array([float(np.median(log["quoted_price"][:, i, 0]))
                                     for i in range(cfg.n)]),
                fraction=fraction, filled_i=filled_i,
                bid=thresholds[1.0], ask=thresholds[-1.0],
                agree=agree, filled=filled, filled_benchmark=filled_benchmark,
                benchmark=benchmark_rule.name, bureau_share=bureau_share(log),
                log=log)


def flat_run(recent):
    if len(recent) < CONVERGED_BLOCKS:
        return False
    for earlier, later in zip(recent[:-1], recent[1:]):
        pairs = [(later[0], earlier[0], FLAT_TOL_MM)] + [
            (later[1][i], earlier[1][i], FLAT_TOL_TRADER) for i in range(len(later[1]))]
        for a, b, tolerance in pairs:
            change, two_se = M.paired(a, b)
            if abs(change) > max(two_se, tolerance):
                return False
    return True


def cotrain(cfg, run):
    mm_net, trader_nets = build_agents(cfg)
    opts = [torch.optim.AdamW(net.parameters(), lr=LR_TRADER) for net in trader_nets]
    scheds = [(torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=200)
               if TRADER_PLATEAU else None) for opt in opts]
    opt_mm = torch.optim.AdamW(mm_net.parameters(), lr=LR_MM)

    val_z = draw_paths(cfg, VAL_PATHS, SEED_PATH + 1, "validation")
    val_order = draw_order(cfg, val_z.shape[0], SEED_PATH + ORDER_OFFSET + 1)
    val_bdc = [always_bdc_pl(spec, val_z) for spec in cfg.specs]
    rule_names = [rule.name for rule in reference_rules(MM_FEE, cfg.n_offers)]

    floors = np.array([float(b.mean()) for b in val_bdc])   # constant: fixed paths
    v = validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc)
    history = [_row(cfg, -1, "init", {"trader": 0, "mm": 0}, v, rule_names,
                    floors=floors)]
    _print_row(cfg, history[-1])

    step = {"trader": 0, "mm": 0}
    recent, stopped_at = [], None
    for outer in range(N_OUTER):
        for agent in order_of_play():
            step[agent], tau, batch_pl = train_block(
                agent, cfg, mm_net, trader_nets, opt_mm, opts, scheds, step[agent])
            v = validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc)
            history.append(_row(cfg, outer, agent, step, v, rule_names,
                                tau=tau, batch_pl=batch_pl,
                                lr_trader=[opt.param_groups[0]["lr"] for opt in opts],
                                floors=floors))
            _print_row(cfg, history[-1])
        torch.save({"traders": [net.state_dict() for net in trader_nets],
                    "mm": mm_net.state_dict(), "outer": outer,
                    "steps": dict(step)}, run / "agents_last.pth")
        recent = (recent + [(v["mm_pl"], v["trader_pls"])])[-CONVERGED_BLOCKS:]
        if flat_run(recent):
            stopped_at = outer
            print(f"  flat over the last {CONVERGED_BLOCKS} outer steps: stopping "
                  f"at outer step {outer}", flush=True)
            break
    if stopped_at is None:
        print(f"  ran all {N_OUTER} outer steps; last {CONVERGED_BLOCKS} "
              f"{'flat' if flat_run(recent) else 'NOT flat'} by the paired "
              f"max(2 SE, tolerance) test", flush=True)
    curve = {key: np.array([row[key] for row in history]) for key in history[0]}
    curve["stopped_at"] = -1 if stopped_at is None else stopped_at
    np.savez(run / "cotrain_multi_curve.npz", **curve)
    return mm_net, trader_nets, curve


def _row(cfg, outer, moved, step, v, rule_names, tau=float("nan"),
         batch_pl=float("nan"), lr_trader=None, floors=None):
    mm_pl = v["mm_pl"]
    return dict(
        outer=outer, moved=moved, step_trader=step["trader"], step_mm=step["mm"],
        tau=tau, batch_pl=batch_pl,
        bdc_pl=np.array([np.nan] * cfg.n if floors is None else floors),
        trader_names=np.array(cfg.traders),
        lr_trader=np.array([LR_TRADER] * cfg.n if lr_trader is None else lr_trader),
        mm_pl=float(mm_pl.mean()), mm_2se=2 * float(mm_pl.std()) / np.sqrt(len(mm_pl)),
        trader_pl=np.array([float(pl.mean()) for pl in v["trader_pls"]]),
        trader_2se=np.array([2 * float(pl.std()) / np.sqrt(len(pl))
                             for pl in v["trader_pls"]]),
        edge=v["edge"], edge_2se=v["edge_2se"],
        rule_pl=np.array([float(v["rule_pl"][name].mean()) for name in rule_names]),
        rule_names=np.array(rule_names),
        offer_rate=v["offer_rate"], fraction=v["fraction"], filled_i=v["filled_i"],
        bid=v["bid"], ask=v["ask"], agree=v["agree"], filled=v["filled"],
        filled_benchmark=v["filled_benchmark"], benchmark=v["benchmark"],
        bureau_share=v["bureau_share"])


def _print_row(cfg, row):
    best = row["rule_pl"].max()
    tau_label = "  init " if row["moved"] == "init" else f"tau {row['tau']:7.4f}"
    traders = "  ".join(
        f"{key}({cfg.specs[i].side}) {row['trader_pl'][i] * 100:+.4f}% (vs BdC "
        f"{row['edge'][i] * 100:+.4f}) up {row['fraction'][i] * 100:4.1f}% "
        f"filled {row['filled_i'][i] * 100:5.2f}%"
        for i, key in enumerate(cfg.traders))
    cuts = ""
    if np.isfinite(row["bid"][0]):
        cuts += f"  bid r1 {row['bid'][0]:.4f}"
    if np.isfinite(row["ask"][0]):
        cuts += f"  ask r1 {row['ask'][0]:.4f}"
    if np.isfinite(row["bid"][0]) and np.isfinite(row["ask"][0]):
        cuts += f"  spread {(row['ask'][0] - row['bid'][0]) * 100:+.2f}%"
    lrs = "/".join(f"{lr:.0e}" for lr in row["lr_trader"])
    print(f"  outer {row['outer']:>3} after {row['moved']:<6} {tau_label}  "
          f"MM {row['mm_pl'] * 1e4:+8.2f} bp (best rule {best * 1e4:+7.2f})  "
          f"bureau {row['bureau_share'] * 100:5.1f}%  {traders}{cuts}  "
          f"agree({row['benchmark']}) {row['agree'] * 100:5.1f}%  lr {lrs}", flush=True)


# ---------------------------------------------------------------------------
# 6. main -- checks, training, a paired evaluation, artefacts
# ---------------------------------------------------------------------------
def load_agents(cfg, run):
    file = run / "agents_final.pth"
    if not file.exists():
        file = run / "agents_last.pth"
    assert file.exists(), f"EVAL_ONLY: no saved agents in {run}"
    saved = torch.load(file)
    mm_net, trader_nets = build_agents(cfg)
    mm_net.load_state_dict(saved["mm"])
    for net, state in zip(trader_nets, saved["traders"]):
        net.load_state_dict(state)
    curve = np.load(run / "cotrain_multi_curve.npz", allow_pickle=False)
    curve = {key: curve[key] for key in curve.files}
    print(f"  EVAL_ONLY: agents loaded from {file.name}, curve of "
          f"{len(curve['mm_pl'])} blocks", flush=True)
    return mm_net, trader_nets, curve


def load_baseline(key, cfg):
    """cotrain.py's single-trader evaluation for this card at this fee, per
    path on the SAME evaluation paths: the no-trade baseline. None if the
    trader is not one of that run's cards, the artefact is absent, or the fee
    is not cotrain's 0.02."""
    if key is None:
        return None
    file = results_path(f"cotrain/{key}_fee{MM_FEE:g}") / "cotrain_eval.npz"
    if abs(cfg.params.bdc_fee - 0.02) > 1e-12:
        # cotrain.py's folder carries no trader fee and its runs were at 0.02:
        # at another fee the comparison would be apples to oranges
        print(f"  [baseline skipped] cotrain's card {key} was played at trader Bureau fee 0.02, "
              f"this game at {cfg.params.bdc_fee:g}", flush=True)
        return None
    if not file.exists():
        return None
    theirs = np.load(file, allow_pickle=False)
    if theirs["eval_trader_pl"].shape[1] != EVAL_PATHS:
        return None
    return theirs["eval_trader_pl"][0], theirs["eval_mm_pl"][0]


def run_dir(cfg):
    """The folder carries the table, the MM's Bureau fee, the traders' Bureau
    fee, the arm, the flow mode (turn-all), the size flag and _2way:
    {traders}_mm{g}_bdc{f}_{arch}[_size][_2way][RUN_TAG]. Every setting that
    changes what was run is in the name, so two variants can never overwrite
    each other's results."""
    return results_path(f"cotrain_multi/{run_stem(cfg)}_{cfg.mm_arch}"

                        f"{'_size' if cfg.see_size else ''}"
                        f"{'_2way' if cfg.two_way else ''}{RUN_TAG}")


def run_dir_v4(cfg):
    """An older folder name that carried no trader fee: a read-only fallback
    so runs already on disk under it can still be read."""
    return results_path(f"cotrain_multi/{cfg.name}_fee{MM_FEE:g}_{cfg.mm_arch}"

                        f"{'_size' if cfg.see_size else ''}"
                        f"{'_2way' if cfg.two_way else ''}")


def run_stem(cfg):
    """The table and BOTH fees: the MM's Bureau fee (MM_FEE) and the
    traders' (the GameParams' bdc_fee), e.g. T1-T2_mm0.01_bdc0.02. A random
    table is named by its size and seed rather than its many card names."""
    table = f"pop{N_POPULATION}s{ROSTER_SEED}" if N_POPULATION else cfg.name
    return f"{table}_mm{MM_FEE:g}_bdc{cfg.params.bdc_fee:g}"


def run_game(rows):
    cfg = make_config(rows)
    run = run_dir(cfg)
    run.mkdir(parents=True, exist_ok=True)
    print(f"\n################ {cfg.n} trader(s) {cfg.name}: "
          + ", ".join(f"{key}(side {spec.side}, L={spec.L:,.0f} T={spec.T:,.0f} "
                      f"A={spec.A:g} B={spec.B:g}{'' if k is None else ', single-trader card ' + k})"
                      for key, spec, k in zip(cfg.traders, cfg.specs, cfg.keys))
          + f"  rounds={cfg.rounds}  a0={cfg.params.a0}  sd={cfg.params.sd}  "
          f"trader BdC fee {cfg.params.bdc_fee}  MM fee {MM_FEE}  "
          f"capital GBP {cfg.capital:,.0f} ################", flush=True)
    print(f"  MM architecture {cfg.mm_arch.upper()}: "
          + {"round": "one level per round per direction, set from the book as the "
                      "round opens; the arrival order cannot change a verdict",
             "bid": "level re-set per offer from the LIVE book; the first offer of a "
                    "round is decided blind (this is the real game)",
             "turn": "every offer of the round read before any verdict; round-open "
                     "book; verdicts concurrent and order-free (not the real game: "
                     "the ceiling on within-round lookahead)"
                     + " -- every offer read, the judged one included, so the policy "
                       "class is any accept/reject rule"
             }[cfg.mm_arch], flush=True)
    print(f"  MM sees: rounds left, phase, X, book"
          f"{', offer size' if cfg.see_size else ''}"
          f"{', offer direction' if cfg.see_direction else ''}"
          f"{', offers still to come' if cfg.see_queue else ''}"
          f"{f', the flow block ({cfg.n_flow} columns)' if cfg.see_flow else ''} "
          f"({cfg.n_mm_feat} inputs); level read in direction "
          f"{cfg.ref_direction:+.0f} and mirrored for the other", flush=True)
    print(f"  offers a round: {cfg.n_offers} (one per trader)" + (
        f" -- TWO-WAY: each trader chooses its offer's DIRECTION, its fraction read signed "
        f"(+ sells what it holds, - sells what it has banked back); it opens at "
        f"{2 / (1 + np.exp(-TWO_WAY_FRACTION_INIT_LOGIT)) - 1:.0%} of what it holds, forward"
        if cfg.two_way else " -- one-way: every trader sells towards its target"), flush=True)
    if DROP_NEGATIVE_PATHS:
        print(f"  negative rates: any path touching <= 0 is deleted. Share deleted "
              f"at a0 {cfg.params.a0} over {cfg.rounds + 1} draws (first-hit draw "
              f"1..{cfg.rounds + 1}):", flush=True)
        for sd, p_del, by_draw in negative_path_table(cfg):
            print(f"    sd {sd:.2f}  P(deleted) {p_del:.1e}  "
                  + "  ".join(f"{share:.1e}" for share in by_draw)
                  + ("   <- this game" if abs(sd - cfg.params.sd) < 1e-12 else ""),
                  flush=True)
    print(f"  co-training: {N_OUTER} x {N_INNER} x {BATCH:,} games per block, "
          f"{FIRST} first, MM level init {MM_LEVEL_INIT}; the traders' block "
          f"updates all {cfg.n} at once, each on its own P/L", flush=True)

    if CHECKS:
        first_of_side = {}
        for row in rows:
            first_of_side.setdefault(row["side"], row)
        for row in first_of_side.values():
            check_reduces_to_mm_train(row)
        check_books_conserve(cfg)
        check_freeze_seam(cfg)
        check_order_uniform(cfg)
        check_bdc_floor(cfg)
        for row in first_of_side.values():
            check_replays_cotrain_pair(row)
        # the arms coincide with one offer a round, the column gate and order
        # invariance for the run's own arm, the flow block itself
        for row in first_of_side.values():
            check_arms_coincide_at_one(row)
        check_columns_reduce(rows)
        check_order_invariance(rows)
        check_flow_block(rows)
        check_book_rule(rows)
        if cfg.two_way:
            check_two_way_reduces(rows)
    start_time = time.time()

    if EVAL_ONLY:
        mm_net, trader_nets, curve = load_agents(cfg, run)
    else:
        mm_net, trader_nets, curve = cotrain(cfg, run)
    set_learning(mm_net, False)
    for net in trader_nets:
        set_learning(net, False)
    return evaluate(cfg, run, mm_net, trader_nets, curve, start_time)


def evaluate(cfg, run, mm_net, trader_nets, curve, start_time):
    rules = reference_rules(MM_FEE, cfg.n_offers)
    print(f"  evaluating: {cfg.n} trader(s), {EVAL_PATHS:,} paths, "
          f"{len(rules)} benchmark rules", flush=True)
    eval_z = draw_paths(cfg, EVAL_PATHS, SEED_PATH + 2, "evaluation")
    eval_order = draw_order(cfg, eval_z.shape[0], SEED_PATH + ORDER_OFFSET + 2)
    n_eval = eval_z.shape[0]
    eval_bdc = [always_bdc_pl(spec, eval_z) for spec in cfg.specs]
    floors = [float(bdc_baseline(spec)[1]) for spec in cfg.specs]
    with torch.no_grad():
        mm_pl, trader_pls, _ = rollout(cfg, mm_net, trader_nets, eval_z, eval_order,
                                       tau=None, fee=MM_FEE)
        rule_pl = {rule.name: rollout(cfg, None, trader_nets, eval_z, eval_order,
                                      rule=rule, fee=MM_FEE)[:2] for rule in rules}
        rec_z = draw_paths(cfg, RECORD_PATHS, SEED_PATH + 3, "record")
        rec_order = draw_order(cfg, rec_z.shape[0], SEED_PATH + ORDER_OFFSET + 3)
        rec_mm_pl, rec_trader_pls, log = rollout(cfg, mm_net, trader_nets, rec_z,
                                                 rec_order, tau=None, fee=MM_FEE,
                                                 record=True)
        rec_bdc = [always_bdc_pl(spec, rec_z) for spec in cfg.specs]
        set_rates = np.asarray(SET_PATH)
        set_z = torch.tensor((set_rates - np.concatenate(
            ([cfg.params.a0], set_rates[:-1]))) / cfg.params.sd).unsqueeze(0)
        set_mm_pl, set_trader_pls, set_log = rollout(
            cfg, mm_net, trader_nets, set_z, draw_order(cfg, 1, SEED_PATH + ORDER_OFFSET + 9),
            tau=None, fee=MM_FEE, record=True)

    print(f"  eval P/Ls and recorded plays done; benchmark agreement next", flush=True)
    benchmark_rule = max(rules, key=lambda r: float(rule_pl[r.name][0].mean()))
    agree, filled, filled_benchmark = benchmark_agreement(log, benchmark_rule, MM_FEE, cfg)
    thresholds = thresholds_by_direction(log, cfg.rounds)
    repay_by_round = repay_on_short_books(log)
    share = bureau_share(log)
    bureau_of_capital = float(bureau_pound_volume(log).mean()) / cfg.capital

    def pct(x):
        return f"{float(x.mean()) * 100:+.4f}%"

    def bp(x):
        return f"{float(x.mean()) * 1e4:+.2f} bp"

    two_se = 2 * float(mm_pl.std()) / np.sqrt(n_eval)
    print(f"\ntraders {cfg.name} final (hard game, {n_eval:,} paths, torch engine):")
    print(f"  co-trained MM P/L {bp(mm_pl)} +/- {two_se * 1e4:.2f} of the game's "
          f"capital (GBP {cfg.capital:,.0f}); Bureau share of filled pound volume "
          f"{share * 100:.1f}% (crossed or carried out {(1 - share) * 100:.1f}%)")
    print(f"  MM-Bureau pound flow per game {bureau_of_capital * 100:.2f}% of the "
          f"game's capital -- how much of the game's money the market maker "
          f"exchanged at the Bureau")
    baselines = {}
    for i, (key, spec) in enumerate(zip(cfg.traders, cfg.specs)):
        edge, edge_2se = M.paired(trader_pls[i], eval_bdc[i])
        line = (f"  {key} (side {spec.side}) P/L {pct(trader_pls[i])}; edge over "
                f"always-BdC {edge * 100:+.4f}% +/- {edge_2se * 100:.4f}% (paired; "
                f"floor here {pct(eval_bdc[i])}, analytic {floors[i] * 100:+.4f}%)")
        base = load_baseline(cfg.keys[i], cfg)
        if base is not None and len(base[0]) != n_eval:
            base = None
        if base is not None:
            gain, gain_2se = M.paired(trader_pls[i], torch.as_tensor(base[0]))
            baselines[i] = (gain, gain_2se, float(base[0].mean()), float(base[1].mean()))
            line += (f"; vs cotrain's single-trader card {cfg.keys[i]} on these paths "
                     f"{gain * 100:+.4f}% +/- {gain_2se * 100:.4f}% (paired; that "
                     f"trader {base[0].mean() * 100:+.4f}%)")
        print(line)
    for rule in rules:
        gap, gap_2se = M.paired(mm_pl, rule_pl[rule.name][0])
        print(f"  vs {rule.name:<18} MM {bp(rule_pl[rule.name][0]):>10}   co MM - rule "
              f"{gap * 1e4:+.2f} +/- {gap_2se * 1e4:.2f} bp (paired)   traders vs "
              f"this rule " + "  ".join(pct(pl) for pl in rule_pl[rule.name][1]))
    print(f"  verdicts agree with the {benchmark_rule.name} rule on {agree * 100:.1f}% "
          f"of offered pound volume; filled {filled * 100:.1f}% "
          f"({benchmark_rule.name} {filled_benchmark * 100:.1f}%)")
    for d, label in ((1.0, "bid (pound sellers, accept below)"),
                     (-1.0, "ask (dollar sellers, accept above)")):
        if np.isfinite(thresholds[d]).any():
            print(f"  {label} P/X by round: "
                  + "  ".join(f"r{r + 1} {thresholds[d][r]:.4f}" for r in range(cfg.rounds)))
    if cfg.see_direction:
        spread = thresholds[-1.0] - thresholds[1.0]
        print("  spread ask - bid, x true rate: "
              + "  ".join(f"r{r + 1} {spread[r] * 100:+.2f}%" for r in range(cfg.rounds))
              + f"   (fee-aware flatten would be {2 * MM_FEE * 100:.2f}%, naive 0)")
    print("  Bureau repay fraction on SHORT books, short-weighted, by round: "
          + "  ".join(f"r{r + 1} {repay_by_round[r]:.3f}" for r in range(cfg.rounds)))
    if cfg.two_way:
        base = np.array(cfg.offer_directions)[None, :, None]
        rev = log["direction"] != base
        vol = offered_pound_volume(log)
        print("  direction chosen, share of each trader's offered pound-equivalent volume that went "
              "REVERSE, and the reverse fills' share: "
              + "  ".join(f"{name} {float(vol[:, i][rev[:, i]].sum() / max(vol[:, i].sum(), 1e-300)) * 100:.1f}% / "
                          f"{float((log['fill_weight'][:, i] * vol[:, i])[rev[:, i]].sum() / max(vol[:, i][rev[:, i]].sum(), 1e-300)) * 100:.1f}%"
                          for i, name in enumerate(cfg.traders)))
    print("  round-1 quotes, $/GBP (median): "
          + "  ".join(f"{key} {np.median(log['quoted_price'][:, i, 0]):.4f}"
                      for i, key in enumerate(cfg.traders)))

    print(f"  writing artefacts to {run.name}/", flush=True)
    np.savez(run / "cotrain_multi_eval.npz",
             traders=np.array(cfg.traders),
             card_keys=np.array(["" if k is None else k for k in cfg.keys]),
             sides=np.array([s.side for s in cfg.specs]),
             directions=np.array(cfg.directions), fee=MM_FEE, first=FIRST,
             mm_level_init=MM_LEVEL_INIT, n_outer=N_OUTER, n_inner=N_INNER,
             batch=BATCH, lr=np.array([LR_TRADER, LR_MM]), trader_plateau=TRADER_PLATEAU,
             tau_trader=np.array(TAU_TRADER), tau_mm=np.array(TAU_MM),
             tau_anneal_frac=TAU_ANNEAL_FRAC,
             tau_anneal_steps=-1 if TAU_ANNEAL_STEPS is None else int(TAU_ANNEAL_STEPS),
             seed_init=np.array([SEED_INIT_TRADER, SEED_INIT_MM]), run_tag=RUN_TAG,
             see_offer_size=cfg.see_size, see_direction=cfg.see_direction,
             see_queue=cfg.see_queue, ref_direction=cfg.ref_direction,
             live_book_within_round=cfg.live_book,
             mm_arch=cfg.mm_arch, see_flow=cfg.see_flow, n_flow=cfg.n_flow,
             two_way=cfg.two_way, n_offers=cfg.n_offers,
             offer_trader=np.array(cfg.offer_trader),
             offer_directions=np.array(cfg.offer_directions),
             flow_columns=np.array(
                 ([f"{what}_{k}" for what in ("edge", "signed_size")
                   for k in range(cfg.n_flow // 2)]
                  if cfg.two_way else
                  [f"{d}_{what}_{k}" for d, n_d in (("plus", cfg.n_plus), ("minus", cfg.n_minus))
                   for what in ("price", "size") for k in range(n_d)]) if cfg.see_flow else []),
             drop_negative_paths=DROP_NEGATIVE_PATHS,
             negative_table_sds=np.array(NEGATIVE_TABLE_SDS),
             negative_table=np.array([[p] + by for _, p, by in negative_path_table(cfg)]),
             n_mm_feat=cfg.n_mm_feat,
             cards=np.array([[s.L, s.T, s.A, s.B] for s in cfg.specs]),
             rounds=cfg.rounds, a0=cfg.params.a0, sd=cfg.params.sd,
             BDC_FEE_DEFAULT=cfg.params.bdc_fee, trader_bdc_fee=cfg.params.bdc_fee,
             capital_pounds=cfg.capital,
             bdc_floor=np.array(floors),
             eval_bdc_pl=np.stack([b.numpy() for b in eval_bdc]),
             record_bdc_pl=np.stack([b.numpy() for b in rec_bdc]),
             eval_mm_pl=mm_pl.numpy(),
             eval_trader_pl=np.stack([pl.numpy() for pl in trader_pls]),
             rule_names=np.array([rule.name for rule in rules]),
             rule_repay=np.array([rule.repay for rule in rules]),
             eval_rule_mm_pl=np.stack([rule_pl[r.name][0].numpy() for r in rules]),
             eval_rule_trader_pl=np.stack([np.stack([pl.numpy() for pl in rule_pl[r.name][1]])
                                           for r in rules]),
             benchmark_rule=benchmark_rule.name, agree=agree, filled=filled,
             filled_benchmark=filled_benchmark, bureau_share=share,
             bureau_pound_of_capital=bureau_of_capital,
             n_population=N_POPULATION, roster_seed=ROSTER_SEED,
             bid_by_round=thresholds[1.0], ask_by_round=thresholds[-1.0],
             repay_by_round=np.array(repay_by_round),
             baseline_index=np.array(list(baselines)),
             baseline_gain=np.array([v[0] for v in baselines.values()]),
             baseline_2se=np.array([v[1] for v in baselines.values()]),
             baseline_trader_pl=np.array([v[2] for v in baselines.values()]),
             baseline_mm_pl=np.array([v[3] for v in baselines.values()]),
             record_mm_pl=rec_mm_pl.numpy(),
             record_trader_pl=np.stack([pl.numpy() for pl in rec_trader_pls]),
             set_path=np.array(SET_PATH), set_mm_pl=float(set_mm_pl[0]),
             set_trader_pl=np.array([float(pl[0]) for pl in set_trader_pls]),
             **{f"set_record_{key}": value for key, value in set_log.items()},
             **{f"record_{key}": value for key, value in log.items()})
    torch.save({"traders": [net.state_dict() for net in trader_nets],
                "mm": mm_net.state_dict()}, run / "agents_final.pth")
    print(f"artefacts in {run}/  ({(time.time() - start_time) / 60:.2f} min)", flush=True)
    return {"name": cfg.name, "mm": float(mm_pl.mean()),
            "traders": [float(pl.mean()) for pl in trader_pls],
            "edges": [M.paired(pl, b)[0] for pl, b in zip(trader_pls, eval_bdc)],
            "bureau_share": share, "bureau_of_capital": bureau_of_capital,
            "bid": thresholds[1.0], "ask": thresholds[-1.0]}


def respond_to_frozen_mm(rows, mm_net, run):
    """The rows' traders train FROM FRESH against a market maker that never
    moves -- the trader blocks of cotrain, with the market maker's block left
    out -- then the full evaluation. This is how a market maker trained on
    one table is read on another: the table of interest best-responds to
    what it learned, so the training population and the game being measured
    stay separate."""
    cfg = make_config(rows)
    run.mkdir(parents=True, exist_ok=True)
    print(f"\n---- market maker FROZEN; {cfg.n} trader(s) {cfg.name} respond "
          f"from fresh (up to {N_OUTER} trader blocks) ----", flush=True)
    start_time = time.time()
    set_learning(mm_net, False)
    if EVAL_ONLY and (run / "agents_last.pth").exists():
        _, trader_nets, curve = load_agents(cfg, run)
    else:
        trader_nets = build_traders(cfg)
        opts = [torch.optim.AdamW(net.parameters(), lr=LR_TRADER) for net in trader_nets]
        scheds = [(torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=200)
                   if TRADER_PLATEAU else None) for opt in opts]
        val_z = draw_paths(cfg, VAL_PATHS, SEED_PATH + 1, "validation")
        val_order = draw_order(cfg, val_z.shape[0], SEED_PATH + ORDER_OFFSET + 1)
        val_bdc = [always_bdc_pl(spec, val_z) for spec in cfg.specs]
        rule_names = [rule.name for rule in reference_rules(MM_FEE, cfg.n_offers)]
        floors = np.array([float(b.mean()) for b in val_bdc])
        v = validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc)
        history = [_row(cfg, -1, "init", {"trader": 0, "mm": 0}, v, rule_names,
                        floors=floors)]
        _print_row(cfg, history[-1])
        step, recent, stopped_at = {"trader": 0, "mm": 0}, [], None
        for outer in range(N_OUTER):
            step["trader"], tau, batch_pl = train_block(
                "trader", cfg, mm_net, trader_nets, None, opts, scheds, step["trader"])
            v = validate(cfg, mm_net, trader_nets, val_z, val_order, val_bdc)
            history.append(_row(cfg, outer, "trader", step, v, rule_names, tau=tau,
                                batch_pl=batch_pl,
                                lr_trader=[opt.param_groups[0]["lr"] for opt in opts],
                                floors=floors))
            _print_row(cfg, history[-1])
            torch.save({"traders": [net.state_dict() for net in trader_nets],
                        "mm": mm_net.state_dict(), "outer": outer,
                        "steps": dict(step)}, run / "agents_last.pth")
            recent = (recent + [(v["mm_pl"], v["trader_pls"])])[-CONVERGED_BLOCKS:]
            if flat_run(recent):
                stopped_at = outer
                print(f"  flat over the last {CONVERGED_BLOCKS} trader blocks: "
                      f"stopping at block {outer}", flush=True)
                break
        if stopped_at is None:
            print(f"  ran all {N_OUTER} trader blocks; last {CONVERGED_BLOCKS} "
                  f"{'flat' if flat_run(recent) else 'NOT flat'} by the paired "
                  f"max(2 SE, tolerance) test", flush=True)
        curve = {key: np.array([row[key] for row in history]) for key in history[0]}
        curve["stopped_at"] = -1 if stopped_at is None else stopped_at
        np.savez(run / "cotrain_multi_curve.npz", **curve)
    for net in trader_nets:
        set_learning(net, False)
    return evaluate(cfg, run, mm_net, trader_nets, curve, start_time)


def run_full():
    """The whole run the SETTINGS ask for. N_POPULATION = 0: the TRADERS
    table co-trains, as always. N_POPULATION = n: a random table of n
    co-trains, then the market maker is frozen and the TRADERS table
    responds to it from fresh -- trained on the population, read on the
    cards. Its artefacts sit under the population run's folder,
    respond_on_cards/."""
    if not N_POPULATION:
        return run_game(TRADERS)
    roster = random_roster(N_POPULATION, ROSTER_SEED)
    pop_cfg, cards_cfg = make_config(roster), make_config(TRADERS)
    assert pop_cfg.n_mm_feat == cards_cfg.n_mm_feat, (
        f"the market maker reads {pop_cfg.n_mm_feat} input columns on the random "
        f"table but {cards_cfg.n_mm_feat} on TRADERS, so it cannot carry over: "
        f"the turn architecture's flow block grows with the table (two columns "
        f"per offer). Population runs need the bid or round architecture.")
    s = run_game(roster)
    run = run_dir(pop_cfg)
    file = run / "agents_final.pth"
    if not file.exists():
        file = run / "agents_last.pth"
    mm_net = build_mm_net(pop_cfg, SEED_INIT_MM)
    mm_net.load_state_dict(torch.load(file)["mm"])
    print(f"\n  population co-training done; market maker frozen, loaded from "
          f"{file.name}", flush=True)
    s["respond"] = respond_to_frozen_mm(TRADERS, mm_net, run / "respond_on_cards")
    return s


def main():
    s = run_full()
    label = f"pop{N_POPULATION}s{ROSTER_SEED}" if N_POPULATION else s["name"]
    print("\n" + "=" * 72)
    print(f"CO-TRAINED GAME {label}, fee {MM_FEE}, MM architecture {MM_ARCH}"
          f"{' (offer size seen)' if SEE_OFFER_SIZE else ''}"
          f"{', TWO-WAY offers' if TWO_WAY else ''}, hard game, {EVAL_PATHS:,} paths")
    print(f"  MM {s['mm'] * 1e4:+8.2f} bp   Bureau share "
          f"{s['bureau_share'] * 100:5.1f}%   traders "
          + "  ".join(f"{pl * 100:+.4f}% (edge {e * 100:+.4f})"
                      for pl, e in zip(s["traders"], s["edges"]))
          + f"   r1 bid {s['bid'][0]:.4f} ask {s['ask'][0]:.4f}")
    if N_POPULATION:
        r = s["respond"]
        print(f"  TRADERS responding to that MM, frozen: MM {r['mm'] * 1e4:+8.2f} bp   "
              f"Bureau share {r['bureau_share'] * 100:5.1f}%   MM-Bureau flow "
              f"{r['bureau_of_capital'] * 100:.2f}% of capital   traders "
              + "  ".join(f"{pl * 100:+.4f}% (edge {e * 100:+.4f})"
                          for pl, e in zip(r["traders"], r["edges"]))
              + f"   r1 bid {r['bid'][0]:.4f} ask {r['ask'][0]:.4f}")
    print("  read: vs the single-trader baseline (no-trade at fee 0.01), do the "
          "traders trade and gain, does the MM's Bureau share fall below 100%, "
          "and is ask - bid narrower than the fee-aware 2g?")
    print("  figures and the summary: python cotrain_multi_diagnostics.py; "
          "the arms side by side: python cotrain_multi_compare.py")


if __name__ == "__main__":
    main()