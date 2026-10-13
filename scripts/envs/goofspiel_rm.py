"""Near-Nash Goofspiel teacher via Regret-Matching / backward-induction Nash.

``choose_action(observation, player_id, legal_action_ids, rng=None) -> int``

WHY THIS EXISTS. Goofspiel (GOPS) has NO winning pure strategy: any deterministic
bidder — including the repo's default "bid the card equal to the prize" matcher —
is a PURE strategy and is provably exploitable (Rhoads-Bartholdi 2012: a pure
value-matcher loses 78-13 to the "overbid-one" counter). The evidence-backed
win/loss policy is a MIXED near-Nash produced by regret matching, which is exactly
the "principled Nash policy" the tournament winner planned but never built (they
shipped a hand-tuned rank-relative Gaussian whose centre offset they admit is "the
open question"). This module BUILDS that Nash policy and caches it to disk, mirror-
ing ``liar_dice_expert.py``'s solve-once/cache/lookup shape.

== Game model (verified against pvp_selfplay._load_game: imp_info=False,
   points_order="random", returns_type="win_loss") ==
- N cards each (N in {5, 8, 10, 13}); N rounds. Deck of prizes 1..N revealed in a
  RANDOM order (a chance node each round).
- Each round: a prize (value 1..N) is shown; both players SIMULTANEOUSLY bid one
  card from their remaining hand. Higher bid takes the prize's points; a tie
  discards it. action id = bid card VALUE - 1.
- imp_info=False => BOTH hands, both scores and the revealed-prize sequence are
  PUBLIC. The ONLY hidden information is the opponent's CURRENT simultaneous bid
  and the FUTURE (random) prize order. This makes the game a perfect-information
  simultaneous-move game, solvable EXACTLY by backward induction with a per-round
  zero-sum matrix-game (regret-matching) stage solve.
- Objective is WIN/LOSS: terminal value 1.0 (win) / 0.5 (tie) / 0.0 (loss) by the
  SIGN of the score margin — so margin-optimal solved tables are the wrong
  equilibrium; we solve the win/loss game directly.

== Two solvers (both self-contained, no pyspiel/numpy; both cached per N) ==
1. EXACT backward-induction Nash for small N (default N <= GOOFSPIEL_RM_EXACT_MAX_N
   = 5, wall-clock + state guarded). ``value(my_hand, opp_hand, remaining_prizes,
   score_diff)`` is memoised; each round is a zero-sum matrix game M[a][b] =
   continuation value, solved by CFR+ regret matching (``_solve_matrix_game``).
   This IS exact subgame-perfect Nash for the public-information game. We cache the
   value memo; at inference we rebuild the ONE relevant stage matrix from cached
   continuation values and sample the row (our) Nash mix.
2. SAMPLED Monte-Carlo regret matching over a RANK ABSTRACTION for large N
   (8/10/13, iteration + wall-clock budgeted). The info set is (hand_size, prize's
   rank among the remaining prizes, coarse score bucket) and the action is which
   RANK of our hand to play — a compact, recurring abstraction that GENERALISES
   across states (so a budgeted solve is still useful at N=13, where exact solving
   is infeasible) and is the self-play-learned upgrade of the winner's hand-tuned
   rank kernel. Each action value is a one-roll-out estimate, so it is O(N^3) per
   game rather than the intractable N! of full simultaneous-move MCCFR.

NEVER-RAISES: every parse / lookup / budget failure falls back to a self-contained
prize-MATCHING softmax (the repo's proven default), so this module degrades to the
current teacher's behaviour rather than ever crashing or returning an illegal id.

Public API (mirrors the other expert teachers):
    choose_action(observation, player_id, legal_action_ids, rng=None) -> int
"""

from __future__ import annotations

import math
import os
import pickle
import random
import re
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration (env-tunable; all guarded so a garbage value can never raise on
# import — same discipline as goofspiel_trajectories / liar_dice_expert)
# ---------------------------------------------------------------------------

_VALID_N: Tuple[int, ...] = (5, 8, 10, 13)


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


# Largest deck we attempt to solve EXACTLY (backward induction). N=5 is fast and
# fully exact; N=8 exact is ~30M+ states in pure Python (infeasible in budget) so
# it defaults to the sampled solver. Bump this ONLY with a matching seconds budget.
_EXACT_MAX_N = _env_int("GOOFSPIEL_RM_EXACT_MAX_N", 5)

# CFR+ regret-matching iterations for each per-round zero-sum matrix stage game in
# the EXACT solve. A <=5x5 matrix game reaches a strong near-Nash (well-mixed, which
# is the whole point) in ~120 iters; the VALUE it feeds upward converges faster
# still. 120 keeps the whole N=5 build inside the default 120s cap. Raise for a
# tighter solve (and raise GOOFSPIEL_RM_SECONDS to match). If N=5 logs "exceeded
# budget; using sampled", lower this or raise the seconds cap.
_STAGE_ITERS = max(1, _env_int("GOOFSPIEL_RM_STAGE_ITERS", 120))

# Monte-Carlo regret-matching iterations for the SAMPLED (rank-abstraction) solve.
_ITERS = max(1, _env_int("GOOFSPIEL_RM_ITERS", 40_000))
# Check the wall-clock every this many sampled games (avoids a time syscall per iter).
_TIME_CHECK = 64

# Wall-clock budget (seconds) for a COLD build of either solver. CFR+ average
# strategies (and a partially-filled value memo) are anytime-valid, but we DISCARD
# a timed-out exact build (partial memo would give wrong lookups) and fall through
# to the sampled solver. The persistent disk cache means this cold cost is paid at
# most once. 0 / "none" / "off" disables the cap.
def _seconds() -> Optional[float]:
    raw = os.environ.get("GOOFSPIEL_RM_SECONDS")
    if raw is None:
        return 120.0
    if raw == "" or raw.strip().lower() in ("0", "none", "off", "inf"):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        return 120.0


# Hard state-count guard on the exact memo (defends against an accidental large-N
# exact build eating all RAM before the wall-clock cap trips).
_EXACT_MAX_STATES = _env_int("GOOFSPIEL_RM_EXACT_MAX_STATES", 4_000_000)

# Fixed seed for the sampled solver so the cached blueprint is reproducible.
_SEED = _env_int("GOOFSPIEL_RM_SEED", 1234567)

# Internal prize-matching fallback temperature (matches the default teacher's
# near-greedy 0.1 so a fallback bid is the proven-strong matching bid).
_FALLBACK_TEMP = max(1e-6, _env_float("GOOFSPIEL_RM_FALLBACK_TEMP", 0.1))

_CACHE_DIR = os.getenv(
    "GOOFSPIEL_RM_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".expert_cache"),
)


class _BudgetExceeded(Exception):
    """Raised inside the exact recursion when the wall-clock / state budget trips."""


# ---------------------------------------------------------------------------
# Observation parsing (superset of goofspiel_expert's parsing; also recovers the
# opponent hand, both scores and the remaining prizes that the Nash lookup needs)
# ---------------------------------------------------------------------------

_RE_PRIZE = re.compile(r"point card:\s*(\d+)", re.IGNORECASE)
_RE_ACTION_VALUE = re.compile(r"^\s*(\d+)\s*->.*?Bid:\s*(\d+)", re.IGNORECASE)
_RE_POINTS = re.compile(r"Points?:\s*(\d+)\s+(\d+)", re.IGNORECASE)
# Remaining prizes still in play (current + future), the primary Nash signal. The
# head-to-head simulator emits this line explicitly; OpenSpiel's imp_info=False
# observation exposes the same set (see _remaining_prizes for the fallbacks).
_RE_REMAINING = re.compile(r"Remaining [Pp]oint [Cc]ards?:\s*([\d ]+)")


def _coerce_rng(rng) -> random.Random:
    if rng is None:
        return random.Random(0x600F5)
    if isinstance(rng, int):
        return random.Random(rng)
    return rng


def _hand_for(observation: str, player_id: int) -> List[int]:
    """Values (1..N) in ``P{player_id} hand: ...``; empty list if not present."""
    m = re.search(rf"P{player_id}\s*hand:\s*([\d ]+)", observation, re.IGNORECASE)
    if not m:
        return []
    try:
        return [int(x) for x in m.group(1).split()]
    except ValueError:
        return []


def _legal_values(observation: str, legal_action_ids: List[int]) -> Dict[int, int]:
    """action id -> bid card value, from the ``<id> -> ...Bid: <v>`` lines with the
    ``value = id + 1`` fallback (goofspiel action id == 0-based card index)."""
    values: Dict[int, int] = {}
    for line in observation.splitlines():
        m = _RE_ACTION_VALUE.match(line)
        if m:
            values[int(m.group(1))] = int(m.group(2))
    return {aid: values.get(aid, aid + 1) for aid in legal_action_ids}


def _remaining_prizes(observation: str, prize: int, hand_size: int, n: int) -> Optional[List[int]]:
    """Best-effort recover of the prizes still in play (current + future), which
    must have size == hand_size and contain ``prize``. Priority:
      1. an explicit ``Remaining point cards: ...`` line (simulator / reformatted);
      2. derive from a revealed ``point card sequence`` line: remaining = {1..N} minus
         the already-resolved prizes.
    Returns a sorted list, or None if nothing consistent is found (caller falls back).
    """
    m = _RE_REMAINING.search(observation)
    if m:
        try:
            rem = sorted({int(x) for x in m.group(1).split()})
        except ValueError:
            rem = []
        if rem and prize in rem and len(rem) == hand_size:
            return rem
    # Fallback: reconstruct from the revealed prize sequence when present.
    seq = re.search(r"(?:point card sequence|prize sequence|cards? revealed)\s*:?\s*([\d ]+)",
                    observation, re.IGNORECASE)
    if seq:
        try:
            revealed = [int(x) for x in seq.group(1).split()]
        except ValueError:
            revealed = []
        # revealed may or may not include the current prize; the resolved prizes are
        # everything revealed EXCEPT the current one.
        resolved = list(revealed)
        if resolved and resolved[-1] == prize:
            resolved = resolved[:-1]
        rem = sorted(set(range(1, n + 1)) - set(resolved))
        if prize in rem and len(rem) == hand_size:
            return rem
    return None


def _parse_state(observation: str, player_id: int, legal_action_ids: List[int]) -> Optional[dict]:
    """Parse the full public state the Nash lookup needs. Returns a dict with keys
    prize, my_hand, opp_hand (or None), sd (my score - opp, or None), rem_incl
    (remaining prizes incl. current, or None) and n; or None if even the prize is
    unreadable. Never raises."""
    pm = _RE_PRIZE.search(observation)
    if not pm:
        return None
    prize = int(pm.group(1))

    id_to_value = _legal_values(observation, legal_action_ids)
    my_hand = sorted(id_to_value[aid] for aid in legal_action_ids)
    if not my_hand:
        return None

    opp_id = 1 - player_id
    opp_hand_raw = _hand_for(observation, opp_id)
    opp_hand = sorted(opp_hand_raw) if opp_hand_raw else None

    sd: Optional[int] = None
    pts = _RE_POINTS.search(observation)
    if pts:
        p0, p1 = int(pts.group(1)), int(pts.group(2))
        sd = (p0 - p1) if player_id == 0 else (p1 - p0)

    seen = set(my_hand) | {prize}
    if opp_hand:
        seen |= set(opp_hand)
    n = _infer_n(seen, len(my_hand))

    rem_incl = _remaining_prizes(observation, prize, len(my_hand), n)
    return {"prize": prize, "my_hand": my_hand, "opp_hand": opp_hand,
            "sd": sd, "rem_incl": rem_incl, "n": n}


def _infer_n(values_seen: set, hand_size: int) -> int:
    """Snap the deck size to the smallest valid N >= every observed value and the
    current hand size (exact at game start; conservative mid-game)."""
    mx = max(values_seen) if values_seen else hand_size
    for c in _VALID_N:
        if c >= mx and c >= hand_size:
            return c
    return _VALID_N[-1]


# ---------------------------------------------------------------------------
# Shared game mechanics
# ---------------------------------------------------------------------------

def _round_delta(my_card: int, opp_card: int, prize: int) -> int:
    """Change in (my_score - opp_score) from bidding ``my_card`` vs ``opp_card``."""
    if my_card > opp_card:
        return prize
    if my_card < opp_card:
        return -prize
    return 0


def _terminal_winloss(score_diff: int) -> float:
    """WIN/LOSS terminal value in [0, 1] by the SIGN of the margin."""
    if score_diff > 0:
        return 1.0
    if score_diff < 0:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Zero-sum matrix game solver (CFR+ regret matching; stdlib only, deterministic)
# ---------------------------------------------------------------------------

def _rm_strategy(regret: List[float]) -> List[float]:
    pos = [r if r > 0.0 else 0.0 for r in regret]
    tot = sum(pos)
    if tot > 0.0:
        return [p / tot for p in pos]
    n = len(regret)
    return [1.0 / n] * n


def _solve_matrix_game(payoff: List[List[float]], iters: int) -> Tuple[float, List[float], List[float]]:
    """Solve a 2-player zero-sum matrix game (row player MAXIMISES ``payoff``) by
    CFR+ self-play regret matching. Returns (game value, row mix, col mix). The
    average strategies converge to a Nash equilibrium; the value is x^T M y.
    Deterministic (no RNG) so a cached solve is reproducible."""
    rows = len(payoff)
    cols = len(payoff[0]) if rows else 0
    if rows == 1 and cols == 1:
        return payoff[0][0], [1.0], [1.0]

    row_regret = [0.0] * rows
    col_regret = [0.0] * cols
    row_sum = [0.0] * rows
    col_sum = [0.0] * cols

    for _ in range(iters):
        rs = _rm_strategy(row_regret)
        cs = _rm_strategy(col_regret)
        for i in range(rows):
            row_sum[i] += rs[i]
        for j in range(cols):
            col_sum[j] += cs[j]

        # Row action values vs the column mix; row maximises.
        u_row = [sum(payoff[i][j] * cs[j] for j in range(cols)) for i in range(rows)]
        v_row = sum(rs[i] * u_row[i] for i in range(rows))
        for i in range(rows):
            row_regret[i] = max(0.0, row_regret[i] + (u_row[i] - v_row))

        # Column action values vs the row mix; column MINIMISES payoff.
        u_col = [sum(payoff[i][j] * rs[i] for i in range(rows)) for j in range(cols)]
        v_col = sum(cs[j] * u_col[j] for j in range(cols))
        for j in range(cols):
            col_regret[j] = max(0.0, col_regret[j] + (v_col - u_col[j]))

    def _avg(acc: List[float]) -> List[float]:
        t = sum(acc)
        if t > 0.0:
            return [a / t for a in acc]
        n = len(acc)
        return [1.0 / n] * n

    row_strat = _avg(row_sum)
    col_strat = _avg(col_sum)
    value = sum(row_strat[i] * payoff[i][j] * col_strat[j]
                for i in range(rows) for j in range(cols))
    return value, row_strat, col_strat


# ---------------------------------------------------------------------------
# EXACT backward-induction Nash (small N)
# ---------------------------------------------------------------------------

class ExactSolver:
    """Exact subgame-perfect Nash of the public-information goofspiel game via
    backward induction, each round a zero-sum matrix stage game. ``value`` is
    memoised over (my_hand, opp_hand, remaining_prizes, score_diff) in player-0
    perspective."""

    def __init__(self, n: int, stage_iters: int, deadline: Optional[float],
                 max_states: int):
        self.n = n
        self.stage_iters = stage_iters
        self.deadline = deadline
        self.max_states = max_states
        self.value_memo: Dict[Tuple, float] = {}

    def value(self, hand0: frozenset, hand1: frozenset, rem: frozenset, sd0: int) -> float:
        key = (hand0, hand1, rem, sd0)
        cached = self.value_memo.get(key)
        if cached is not None:
            return cached
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise _BudgetExceeded("wall-clock")
        if len(self.value_memo) > self.max_states:
            raise _BudgetExceeded("state count")

        if not rem:
            v = _terminal_winloss(sd0)
        else:
            # Chance node: the next prize is drawn uniformly from ``rem``.
            total = 0.0
            for prize in rem:
                total += self._stage_value(hand0, hand1, rem - {prize}, sd0, prize)
            v = total / len(rem)

        self.value_memo[key] = v
        return v

    def _stage_value(self, hand0: frozenset, hand1: frozenset, rem_after: frozenset,
                     sd0: int, prize: int) -> float:
        rows = sorted(hand0)
        cols = sorted(hand1)
        matrix = [
            [self.value(hand0 - {a}, hand1 - {b}, rem_after, sd0 + _round_delta(a, b, prize))
             for b in cols]
            for a in rows
        ]
        val, _, _ = _solve_matrix_game(matrix, self.stage_iters)
        return val


def _value_as_me(memo: Dict[Tuple, float], my_hand: frozenset, opp_hand: frozenset,
                 rem: frozenset, sd_me: int) -> Optional[float]:
    """Look up a continuation value from MY perspective, using the memo's player-0
    symmetry: value(h0,h1,rem,sd0) = 1 - value(h1,h0,rem,-sd0)."""
    v = memo.get((my_hand, opp_hand, rem, sd_me))
    if v is not None:
        return v
    v = memo.get((opp_hand, my_hand, rem, -sd_me))
    if v is not None:
        return 1.0 - v
    return None


def _exact_distribution(memo: Dict[Tuple, float], state: dict, stage_iters: int,
                        legal_action_ids: List[int]) -> Dict[int, float]:
    """Rebuild the current round's Nash stage game from cached continuation values
    and return {action_id: probability} for OUR (row) Nash mix. {} on any miss."""
    my_hand = state["my_hand"]
    opp_hand = state["opp_hand"]
    sd_me = state["sd"]
    rem_incl = state["rem_incl"]
    prize = state["prize"]
    if opp_hand is None or sd_me is None or rem_incl is None:
        return {}
    if len(my_hand) != len(opp_hand) or prize not in rem_incl:
        return {}

    mh = frozenset(my_hand)
    oh = frozenset(opp_hand)
    rem_after = frozenset(rem_incl) - {prize}

    rows = sorted(mh)
    cols = sorted(oh)
    matrix: List[List[float]] = []
    for a in rows:
        row: List[float] = []
        for b in cols:
            v = _value_as_me(memo, mh - {a}, oh - {b}, rem_after, sd_me + _round_delta(a, b, prize))
            if v is None:
                return {}  # incomplete memo -> safe fallback
            row.append(v)
        matrix.append(row)

    _, row_strat, _ = _solve_matrix_game(matrix, stage_iters)
    legal = set(legal_action_ids)
    dist: Dict[int, float] = {}
    for card, p in zip(rows, row_strat):
        aid = card - 1
        if aid in legal and p > 0.0:
            dist[aid] = dist.get(aid, 0.0) + p
    return _renormalize(dist)


# ---------------------------------------------------------------------------
# SAMPLED regret matching over a rank abstraction (large N)
#
# Full external-sampling MCCFR is INTRACTABLE here: goofspiel has simultaneous
# moves every round, so a traverser that enumerates its own action at all N rounds
# fans out to N! leaves per iteration (13! for N=13). Instead we use Monte-Carlo
# regret matching over the abstraction: at each visited info set we enumerate the m
# rank-actions and estimate each action's value with ONE sampled roll-out of the
# continuation (both players following the current average-regret strategy). That is
# O(N^3) per game — tractable at every N — and its averaged strategy converges toward
# the abstraction's equilibrium. Anytime-valid, so the wall-clock cap can stop it
# early with a usable (if noisier) blueprint.
# ---------------------------------------------------------------------------

def _score_state(sd: int, stake: int) -> int:
    """Coarse win/loss-relevant score bucket. stake = points still in play."""
    if sd > stake:
        return 4   # lead already uncatchable (won)
    if sd < -stake:
        return 0   # deficit already uncatchable (lost)
    if sd > 0:
        return 3   # ahead
    if sd < 0:
        return 1   # behind
    return 2       # even


class _RankNode:
    __slots__ = ("regret", "strat_sum", "n")

    def __init__(self, n: int):
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n
        self.n = n

    def strategy(self) -> List[float]:
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        return [p / s for p in pos] if s > 0.0 else [1.0 / self.n] * self.n

    def average(self) -> List[float]:
        s = sum(self.strat_sum)
        return [x / s for x in self.strat_sum] if s > 0.0 else [1.0 / self.n] * self.n


class SampledSolver:
    """Monte-Carlo regret matching over the (hand_size, prize_rank, score_bucket)
    info abstraction; the action is which RANK of the current hand to bid. Both
    symmetric players share the info-set space (self-play)."""

    def __init__(self, n: int):
        self.n = n
        self.full = frozenset(range(1, n + 1))
        self.nodes: Dict[Tuple[int, int, int], _RankNode] = {}

    def _node(self, key: Tuple[int, int, int], m: int) -> _RankNode:
        nd = self.nodes.get(key)
        if nd is None:
            nd = _RankNode(m)
            self.nodes[key] = nd
        return nd

    @staticmethod
    def _sample(strategy: List[float], rng: random.Random) -> int:
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(strategy):
            acc += p
            if r <= acc:
                return i
        return len(strategy) - 1

    def _rollout(self, hand0: frozenset, hand1: frozenset, rem: frozenset,
                 sd0: int, rng: random.Random) -> float:
        """Play the continuation to a terminal, both players sampling their current
        regret-matching strategy. Returns player-0's zero-sum utility (+1/0/-1)."""
        while rem:
            rem_sorted = sorted(rem)
            m = len(rem_sorted)
            prize = rng.choice(rem_sorted)
            prize_rank = rem_sorted.index(prize)
            stake = sum(rem_sorted)
            cards0 = sorted(hand0)
            cards1 = sorted(hand1)
            s0 = self._node((m, prize_rank, _score_state(sd0, stake)), m).strategy()
            s1 = self._node((m, prize_rank, _score_state(-sd0, stake)), m).strategy()
            a = cards0[self._sample(s0, rng)]
            b = cards1[self._sample(s1, rng)]
            sd0 += _round_delta(a, b, prize)
            hand0 = hand0 - {a}
            hand1 = hand1 - {b}
            rem = rem - {prize}
        return _terminal_pm1(sd0)

    def _train_iter(self, traverser: int, rng: random.Random) -> None:
        """One self-play game. At each round the ``traverser``'s info set is updated
        with regret matching over Monte-Carlo action values (one roll-out per action);
        the other player and chance are sampled to advance the single main trajectory."""
        hand0 = hand1 = self.full
        rem = self.full
        sd0 = 0
        while rem:
            rem_sorted = sorted(rem)
            m = len(rem_sorted)
            prize = rng.choice(rem_sorted)
            prize_rank = rem_sorted.index(prize)
            rem_after = rem - {prize}
            stake = sum(rem_sorted)
            cards0 = sorted(hand0)
            cards1 = sorted(hand1)
            s0 = self._node((m, prize_rank, _score_state(sd0, stake)), m).strategy()
            s1 = self._node((m, prize_rank, _score_state(-sd0, stake)), m).strategy()

            if traverser == 0:
                b = cards1[self._sample(s1, rng)]          # opponent's fixed action
                node = self._node((m, prize_rank, _score_state(sd0, stake)), m)
                strat = s0
                util = [self._rollout(hand0 - {cards0[i]}, hand1 - {b}, rem_after,
                                      sd0 + _round_delta(cards0[i], b, prize), rng)
                        for i in range(m)]                 # player-0 utility of each action
                nu = sum(strat[i] * util[i] for i in range(m))
                for i in range(m):
                    node.regret[i] = max(0.0, node.regret[i] + util[i] - nu)
                    node.strat_sum[i] += strat[i]
                a = cards0[self._sample(s0, rng)]          # advance main trajectory
            else:
                a = cards0[self._sample(s0, rng)]          # opponent's fixed action
                node = self._node((m, prize_rank, _score_state(-sd0, stake)), m)
                strat = s1
                util = [-self._rollout(hand0 - {a}, hand1 - {cards1[i]}, rem_after,
                                       sd0 + _round_delta(a, cards1[i], prize), rng)
                        for i in range(m)]                 # player-1 utility (= -u0)
                nu = sum(strat[i] * util[i] for i in range(m))
                for i in range(m):
                    node.regret[i] = max(0.0, node.regret[i] + util[i] - nu)
                    node.strat_sum[i] += strat[i]
                b = cards1[self._sample(s1, rng)]          # advance main trajectory

            sd0 += _round_delta(a, b, prize)
            hand0 = hand0 - {a}
            hand1 = hand1 - {b}
            rem = rem_after

    def train(self, iters: int, rng: random.Random, deadline: Optional[float]) -> int:
        done = 0
        # Re-check the clock every _TIME_CHECK games so a long build stops promptly
        # without paying time.monotonic() on every single iteration.
        while done < iters:
            self._train_iter(0, rng)
            self._train_iter(1, rng)
            done += 1
            if deadline is not None and (done % _TIME_CHECK == 0) and time.monotonic() > deadline:
                break
        return done

    def policy(self) -> Dict[Tuple[int, int, int], List[float]]:
        return {key: nd.average() for key, nd in self.nodes.items()}


def _terminal_pm1(score_diff: int) -> float:
    """Zero-sum terminal utility for player 0: +1 win / 0 tie / -1 loss."""
    if score_diff > 0:
        return 1.0
    if score_diff < 0:
        return -1.0
    return 0.0


def _sampled_distribution(policy: Dict[Tuple[int, int, int], List[float]], state: dict,
                          legal_action_ids: List[int]) -> Dict[int, float]:
    """Map the rank-abstraction policy for this state to {action_id: probability}.
    {} on any miss (caller falls back)."""
    my_hand = state["my_hand"]
    sd_me = state["sd"]
    rem_incl = state["rem_incl"]
    prize = state["prize"]
    if sd_me is None or rem_incl is None or prize not in rem_incl:
        return {}
    rem_sorted = sorted(rem_incl)
    m = len(rem_sorted)
    if m != len(my_hand):
        return {}
    prize_rank = rem_sorted.index(prize)
    stake = sum(rem_sorted)
    key = (m, prize_rank, _score_state(sd_me, stake))
    probs = policy.get(key)
    if not probs or len(probs) != m:
        return {}
    cards = sorted(my_hand)  # rank i -> cards[i]
    legal = set(legal_action_ids)
    dist: Dict[int, float] = {}
    for i, p in enumerate(probs):
        aid = cards[i] - 1
        if aid in legal and p > 0.0:
            dist[aid] = dist.get(aid, 0.0) + p
    return _renormalize(dist)


def _renormalize(dist: Dict[int, float]) -> Dict[int, float]:
    total = sum(dist.values())
    if total <= 0.0:
        return {}
    return {aid: p / total for aid, p in dist.items()}


# ---------------------------------------------------------------------------
# Build + cache (solve once, reuse forever) — mirrors liar_dice_expert's shape
# ---------------------------------------------------------------------------

def _cache_path(n: int) -> str:
    return os.path.join(_CACHE_DIR, f"goofspiel_rm_n{n}.pkl")


def build_for_n(n: int, verbose: bool = False) -> dict:
    """Solve (exact for small N, sampled otherwise) and return the strategy payload:
    {"n", "type": "exact"|"sampled"|"none", "value_memo"?, "policy"?, ...}."""
    seconds = _seconds()

    if n <= _EXACT_MAX_N:
        deadline = time.monotonic() + seconds if seconds else None
        try:
            solver = ExactSolver(n, _STAGE_ITERS, deadline, _EXACT_MAX_STATES)
            full = frozenset(range(1, n + 1))
            solver.value(full, full, full, 0)
            if verbose:
                print(f"[goofspiel_rm] exact N={n}: {len(solver.value_memo)} states, "
                      f"stage_iters={_STAGE_ITERS}")
            return {"n": n, "type": "exact", "value_memo": solver.value_memo,
                    "stage_iters": _STAGE_ITERS}
        except _BudgetExceeded as exc:
            if verbose:
                print(f"[goofspiel_rm] exact N={n} exceeded budget ({exc}); using sampled")
        except Exception as exc:  # noqa: BLE001 — build must never propagate
            if verbose:
                print(f"[goofspiel_rm] exact N={n} failed ({exc}); using sampled")

    # Sampled (rank-abstraction) Monte-Carlo regret-matching path.
    try:
        deadline = time.monotonic() + seconds if seconds else None
        solver = SampledSolver(n)
        done = solver.train(_ITERS, random.Random(_SEED), deadline)
        if verbose:
            print(f"[goofspiel_rm] sampled N={n}: {done}/{_ITERS} iters, "
                  f"{len(solver.nodes)} info sets")
        return {"n": n, "type": "sampled", "policy": solver.policy(), "iters": done}
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[goofspiel_rm] sampled N={n} failed ({exc}); matching-only")
        return {"n": n, "type": "none"}


def _save_payload(path: str, payload: dict) -> None:
    """Atomic write so a concurrent worker never reads a half-written pickle."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)


_STRATEGY_MEM: Dict[int, dict] = {}


def _load_or_build(n: int) -> dict:
    """In-process + disk cached strategy payload for deck size ``n``."""
    if n in _STRATEGY_MEM:
        return _STRATEGY_MEM[n]
    path = _cache_path(n)
    payload: Optional[dict] = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:  # noqa: BLE001 — rebuild on any corruption
            payload = None
    if payload is None:
        payload = build_for_n(n)
        try:
            _save_payload(path, payload)
        except Exception:  # noqa: BLE001 — cache write is best-effort
            pass
    _STRATEGY_MEM[n] = payload
    return payload


# ---------------------------------------------------------------------------
# Self-contained prize-matching fallback (never-raise guarantee)
# ---------------------------------------------------------------------------

def _matching_pick(observation: str, legal_action_ids: List[int], rng: random.Random) -> int:
    """Prize-matching softmax over legal ids — the repo's proven default, reproduced
    locally so this module degrades gracefully instead of ever raising."""
    legal = list(legal_action_ids)
    prize_m = _RE_PRIZE.search(observation)
    if not prize_m:
        return rng.choice(legal)
    prize = int(prize_m.group(1))
    id_to_value = _legal_values(observation, legal)
    scores = [-abs(id_to_value[aid] - prize) for aid in legal]
    top = max(scores)
    weights = [math.exp((s - top) / _FALLBACK_TEMP) for s in scores]
    return rng.choices(legal, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def choose_action(observation: str, player_id: int, legal_action_ids: List[int],
                  rng=None) -> int:
    """Sample a legal bid from the near-Nash regret-matching strategy for this deck
    size; fall back to prize-matching on any parse / lookup / budget miss. Always
    returns an element of ``legal_action_ids``; never raises (except on an empty
    legal set, matching the other teachers)."""
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    rng = _coerce_rng(rng)
    legal = list(legal_action_ids)

    try:
        state = _parse_state(observation, player_id, legal)
        if state is not None:
            payload = _load_or_build(state["n"])
            ptype = payload.get("type")
            dist: Dict[int, float] = {}
            if ptype == "exact":
                dist = _exact_distribution(payload["value_memo"], state,
                                           payload.get("stage_iters", _STAGE_ITERS), legal)
            elif ptype == "sampled":
                dist = _sampled_distribution(payload["policy"], state, legal)
            if dist:
                ids = list(dist)
                return rng.choices(ids, weights=[dist[a] for a in ids], k=1)[0]
    except Exception:  # noqa: BLE001 — never-raise: fall back to matching
        pass

    return _matching_pick(observation, legal, rng)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build + inspect the goofspiel RM strategy for a deck size.")
    ap.add_argument("--n", type=int, default=5, choices=list(_VALID_N))
    ap.add_argument("--rebuild", action="store_true", help="ignore any on-disk cache")
    args = ap.parse_args()

    if args.rebuild:
        p = _cache_path(args.n)
        if os.path.exists(p):
            os.remove(p)
        _STRATEGY_MEM.pop(args.n, None)

    t0 = time.monotonic()
    payload = build_for_n(args.n, verbose=True)
    dt = time.monotonic() - t0
    kind = payload["type"]
    size = len(payload.get("value_memo") or payload.get("policy") or {})
    print(f"N={args.n}: type={kind}, entries={size}, build={dt:.1f}s")
    try:
        _save_payload(_cache_path(args.n), payload)
        print(f"cached -> {_cache_path(args.n)}")
    except Exception as exc:  # noqa: BLE001
        print(f"cache write skipped: {exc}")
