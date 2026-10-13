"""Liar's Dice (Dudo) teacher.

``choose_action`` labels with a deterministic visible-bid reader by default: it
uses only what the eval prompt shows (own dice, latest bid, seat), reads the
opponent's hand from the latest bid and plays the claim most likely to be true
(section "Visible-bid reader" below). The CFR+ / MCCFR solvers and the sampled
flat-EV policy remain as opt-in backstops (``LIARS_TEACHER=ev``).

The CFR+ solver builds the 2-player Dudo game tree (6 faces, 6s WILD) for the
env's dice configuration, solves it with Counterfactual Regret Minimization to a
near-Nash average strategy and caches that strategy once to disk.

== Game model ==
- Two players, ``d`` dice each, ``total_dice = 2*d``.
- Chance deals each player a hidden hand (multiset of ``d`` faces in 1..6).
- Player 0 bids first. A bid is (quantity Q in 1..total_dice, face F in 1..6).
  Action-id encoding matches the env / liar_dice_policy:
      bid_id = (Q-1)*6 + (F-1);   liar_id = total_dice*6.
  A bid is "higher" iff its id is strictly greater (Dudo order: ties broken by
  quantity then face), so legal bids after current id ``c`` are every id in
  ``c+1 .. liar_id-1``, plus ``liar_id`` (challenge) when a bid exists.
- On a challenge, count dice across ALL ``total_dice`` dice matching the bid's
  face. 6s are wild and count for any face EXCEPT a bid on face 6 (then only
  literal 6s count) -- this mirrors ``liar_dice_env.bid_probability`` /
  ``liar_dice_policy._prob_bid_true``. If count >= quantity the bid was true
  (the challenger loses), else the challenger wins. Zero-sum +/-1 utility.

== Information set ==
An info set for the acting player is (their own hand multiset, public bid
history). The opponent's hand is hidden chance. We enumerate hand PAIRS with
their joint probability and run vanilla CFR+ over the shared betting tree,
which is exact for small configs.

== Tractability / abstraction ==
Exact CFR+ is run for ``total_dice <= EXACT_MAX_TOTAL_DICE`` (default 4, i.e.
up to 2 dice each). For larger configs the full tree is intractable
(hands grow like C(d+5,5) per player and the betting tree like 6*total_dice
bids), so we fall back to a DICE-COUNT BUCKET ABSTRACTION: the solve is run on
a reduced ``d_solve = EXACT_MAX_TOTAL_DICE // 2`` dice-per-player game and, at
lookup time, the real hand and bids are projected onto that smaller game by
bucketing dice counts. The projection is documented in ``_abstract_*`` below.
This keeps the teacher near-optimal on the strategically-decisive low-count
region while remaining solvable with the standard library alone.

Public API (called by the SFT generator):
    choose_action(observation, player_id, legal_action_ids, rng=None) -> int
    opponent_action(kind, observation, legal_action_ids, rng) -> int   (generation pool)
    blueprint_needed() -> bool   (whether generation must warm the MCCFR blueprint)
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_FACES = 6
WILD_FACE = 6  # 6s are wild

# Dice-per-player of the *deployed* env variant. Standard Dudo here is 5 each
# (total 10), matching the reference harness (BayesianOpponentInference n_dice=5).
DEPLOY_DICE_PER_PLAYER = 5

# Largest total-dice count we solve EXACTLY. At/below this we build the full
# tree; above it we project onto an abstraction of this size.
#
# NOTE on tractability: each CFR+ iteration here does a FULL chance enumeration
# (every hand pair, both player perspectives), so a single iteration is a
# complete high-quality sweep — convergence needs ~1e2 iterations, not 1e3+.
# The 1-die-each game (total_dice=2) already has ~24.6k information sets and
# costs ~1.4s/iteration; total_dice=4 is orders of magnitude larger and is not
# solvable in a usable time budget. We therefore solve the 1-die-each game
# exactly and project real (5-dice) hands onto it via the abstraction below.
EXACT_MAX_TOTAL_DICE = 2  # solve 1 die per player exactly; abstract above that

# CFR+ iterations for the cached solve. Full-sweep iterations converge fast;
# 150 drives exploitability for the 1-die game well below 0.05 (utility units).
CFR_ITERATIONS = 150

_CACHE_DIR = os.getenv(
    "LIARS_DICE_EXPERT_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".expert_cache"),
)


# ---------------------------------------------------------------------------
# Action-id encoding (matches liar_dice_env / liar_dice_policy)
# ---------------------------------------------------------------------------

def decode_bid(action_id: int) -> tuple[int, int]:
    """action_id -> (quantity, face)."""
    q, f = divmod(action_id, NUM_FACES)
    return q + 1, f + 1


def encode_bid(quantity: int, face: int) -> int:
    return (quantity - 1) * NUM_FACES + (face - 1)


def liar_id(total_dice: int) -> int:
    return total_dice * NUM_FACES


def legal_action_ids_for(current_id: "int | None", total_dice: int) -> list[int]:
    """Legal action ids given the current bid id (None if no bid yet)."""
    lid = liar_id(total_dice)
    if current_id is None:
        return list(range(lid))  # any opening bid, no challenge
    return list(range(current_id + 1, lid)) + [lid]


# ---------------------------------------------------------------------------
# Dice / hand helpers
# ---------------------------------------------------------------------------

def _hands(dice_per_player: int) -> list[tuple[int, ...]]:
    """All sorted hand multisets (ascending tuples) for one player."""
    return [tuple(c) for c in itertools.combinations_with_replacement(range(1, NUM_FACES + 1), dice_per_player)]


def _hand_probability(hand: tuple[int, ...]) -> float:
    """Probability of rolling this sorted multiset with fair dice."""
    n = len(hand)
    counts: dict[int, int] = defaultdict(int)
    for d in hand:
        counts[d] += 1
    perms = math.factorial(n)
    for c in counts.values():
        perms //= math.factorial(c)
    return perms / (NUM_FACES ** n)


def _matching_count(hand: tuple[int, ...], face: int) -> int:
    """Dice in `hand` counting toward `face` (6s wild unless face==6)."""
    if face == WILD_FACE:
        return sum(1 for d in hand if d == WILD_FACE)
    return sum(1 for d in hand if d == face or d == WILD_FACE)


# ---------------------------------------------------------------------------
# CFR+ node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("n_actions", "regret_sum", "strategy_sum")

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.regret_sum = [0.0] * n_actions
        self.strategy_sum = [0.0] * n_actions

    def strategy(self) -> list[float]:
        """Current strategy from positive regrets (regret matching)."""
        pos = [r if r > 0.0 else 0.0 for r in self.regret_sum]
        total = sum(pos)
        if total > 0.0:
            return [r / total for r in pos]
        return [1.0 / self.n_actions] * self.n_actions

    def average_strategy(self) -> list[float]:
        total = sum(self.strategy_sum)
        if total > 0.0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / self.n_actions] * self.n_actions


# ---------------------------------------------------------------------------
# CFR+ solver for a 2-player Dudo game with `dice_per_player` dice each
# ---------------------------------------------------------------------------

class DudoSolver:
    """Exact CFR+ over the full Dudo tree for the given dice-per-player."""

    def __init__(self, dice_per_player: int):
        self.dice_per_player = dice_per_player
        self.total_dice = 2 * dice_per_player
        self.liar = liar_id(self.total_dice)
        self.hands = _hands(dice_per_player)
        self.hand_prob = {h: _hand_probability(h) for h in self.hands}
        # nodes keyed by (player_hand, history_tuple)
        self.nodes: dict[tuple, _Node] = {}

    # -- info-set key --------------------------------------------------------
    def _key(self, hand: tuple[int, ...], history: tuple[int, ...]) -> tuple:
        return (hand, history)

    def _get_node(self, key: tuple, n_actions: int) -> _Node:
        node = self.nodes.get(key)
        if node is None:
            node = _Node(n_actions)
            self.nodes[key] = node
        return node

    # -- terminal utility (challenge resolution) -----------------------------
    def _challenge_utility(self, hands: tuple, last_bid_id: int, challenger: int) -> float:
        """Utility for player 0 when `challenger` calls liar on `last_bid_id`."""
        q, f = decode_bid(last_bid_id)
        count = _matching_count(hands[0], f) + _matching_count(hands[1], f)
        bid_true = count >= q
        # If bid true, the challenger loses; else challenger wins.
        challenger_wins = not bid_true
        winner = challenger if challenger_wins else 1 - challenger
        return 1.0 if winner == 0 else -1.0

    # -- CFR+ recursion ------------------------------------------------------
    def _cfr(self, hands: tuple, history: tuple[int, ...], player: int,
             reach0: float, reach1: float) -> float:
        """Return expected utility for player 0 from this node."""
        last_bid = history[-1] if history else None

        # terminal: a challenge was the last action
        if last_bid == self.liar:
            # the player who just acted (challenger) is `1 - player`
            challenger = 1 - player
            bid_before = history[-2]
            return self._challenge_utility(hands, bid_before, challenger)

        current_id = last_bid  # may be None
        actions = legal_action_ids_for(current_id, self.total_dice)
        n = len(actions)

        my_hand = hands[player]
        key = self._key(my_hand, history)
        node = self._get_node(key, n)
        strat = node.strategy()

        util = [0.0] * n
        node_util = 0.0
        for i, a in enumerate(actions):
            next_hist = history + (a,)
            if player == 0:
                u = self._cfr(hands, next_hist, 1, reach0 * strat[i], reach1)
            else:
                u = self._cfr(hands, next_hist, 0, reach0, reach1 * strat[i])
            util[i] = u
            node_util += strat[i] * u

        # regret + strategy accumulation (CFR+: floor regrets at 0)
        if player == 0:
            cf_reach = reach1
            my_reach = reach0
        else:
            cf_reach = reach0
            my_reach = reach1
            node_util = node_util  # util is for player 0 throughout

        for i in range(n):
            if player == 0:
                regret = util[i] - node_util
            else:
                regret = -(util[i] - node_util)  # player 1 minimizes player-0 utility
            new_r = node.regret_sum[i] + cf_reach * regret
            node.regret_sum[i] = new_r if new_r > 0.0 else 0.0  # CFR+ flooring
            node.strategy_sum[i] += my_reach * strat[i]

        return node_util

    def train(self, iterations: int, log_every: int = 0) -> list[float]:
        """Run CFR+ for `iterations`. Returns exploitability samples (every
        log_every iters; empty if log_every==0)."""
        expl_samples: list[float] = []
        hand_pairs = [(h0, h1) for h0 in self.hands for h1 in self.hands]
        pair_prob = {(h0, h1): self.hand_prob[h0] * self.hand_prob[h1]
                     for (h0, h1) in hand_pairs}
        for t in range(1, iterations + 1):
            for hands in hand_pairs:
                w = pair_prob[hands]
                # weight chance reach into the reach probabilities
                self._cfr(hands, (), 0, w, 1.0)
                self._cfr(hands, (), 1, 1.0, w)
            if log_every and (t % log_every == 0 or t == iterations):
                expl_samples.append(self.exploitability())
        return expl_samples

    # -- exploitability via best response ------------------------------------
    def exploitability(self) -> float:
        """Sum of both players' best-response value against the average
        strategy. 0.0 == exact Nash (zero-sum, so BR values sum to 0 at Nash)."""
        avg: dict[tuple, list[float]] = {
            key: node.average_strategy() for key, node in self.nodes.items()
        }
        br0 = self._best_response_value(avg, br_player=0)
        br1 = self._best_response_value(avg, br_player=1)
        # At Nash both BR values equal the game value v and -v -> sum 0.
        return br0 + br1

    def _best_response_value(self, avg: dict, br_player: int) -> float:
        """Expected value (for `br_player`) of best-responding to `avg`,
        averaged over chance. Returns value in [-1, 1]."""
        total = 0.0
        for h0 in self.hands:
            for h1 in self.hands:
                hands = (h0, h1)
                w = self.hand_prob[h0] * self.hand_prob[h1]
                total += w * self._br_recurse(avg, hands, (), 0, br_player)
        return total

    def _br_recurse(self, avg: dict, hands: tuple, history: tuple[int, ...],
                    player: int, br_player: int) -> float:
        """Value for br_player at this node (br_player maximizes; opponent
        follows avg)."""
        last_bid = history[-1] if history else None
        if last_bid == self.liar:
            challenger = 1 - player
            bid_before = history[-2]
            u0 = self._challenge_utility(hands, bid_before, challenger)
            return u0 if br_player == 0 else -u0

        current_id = last_bid
        actions = legal_action_ids_for(current_id, self.total_dice)

        if player == br_player:
            best = -math.inf
            for a in actions:
                v = self._br_recurse(avg, hands, history + (a,), 1 - player, br_player)
                if v > best:
                    best = v
            return best
        else:
            key = self._key(hands[player], history)
            node = avg.get(key)
            if node is None:
                strat = [1.0 / len(actions)] * len(actions)
            else:
                strat = node
            v = 0.0
            for i, a in enumerate(actions):
                v += strat[i] * self._br_recurse(avg, hands, history + (a,), 1 - player, br_player)
            return v

    # -- export --------------------------------------------------------------
    def export_average(self) -> dict[tuple, list[tuple[int, float]]]:
        """Map info-set key -> list of (action_id, probability)."""
        out: dict[tuple, list[tuple[int, float]]] = {}
        for (hand, history), node in self.nodes.items():
            last = history[-1] if history else None
            actions = legal_action_ids_for(last, self.total_dice)
            avg = node.average_strategy()
            out[(hand, history)] = list(zip(actions, avg))
        return out


# ---------------------------------------------------------------------------
# Cache: solve once, reuse forever
# ---------------------------------------------------------------------------

def _cache_path(dice_per_player: int) -> str:
    total = 2 * dice_per_player
    return os.path.join(_CACHE_DIR, f"liar_dice_{total}dice_cfrplus.pkl")


def _solve_dice_per_player() -> int:
    """The dice-per-player we actually solve (exact if small, else abstraction
    size). Capped so the tree stays solvable with stdlib only."""
    deploy_total = 2 * DEPLOY_DICE_PER_PLAYER
    if deploy_total <= EXACT_MAX_TOTAL_DICE:
        return DEPLOY_DICE_PER_PLAYER
    return EXACT_MAX_TOTAL_DICE // 2


def build_and_cache(dice_per_player: "int | None" = None,
                    iterations: int = CFR_ITERATIONS,
                    verbose: bool = False) -> dict:
    """Solve and cache the CFR+ average strategy for `dice_per_player`."""
    if dice_per_player is None:
        dice_per_player = _solve_dice_per_player()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    solver = DudoSolver(dice_per_player)
    log_every = max(1, iterations // 10) if verbose else 0
    expl = solver.train(iterations, log_every=log_every)
    payload = {
        "dice_per_player": dice_per_player,
        "total_dice": solver.total_dice,
        "iterations": iterations,
        "exploitability": expl[-1] if expl else solver.exploitability(),
        "strategy": solver.export_average(),
    }
    with open(_cache_path(dice_per_player), "wb") as f:
        pickle.dump(payload, f)
    return payload


_STRATEGY_CACHE: "dict | None" = None
_SOLVE_DPP: int = _solve_dice_per_player()


def _load_strategy() -> dict:
    """Load (or build) the cached strategy for the solved dice-per-player."""
    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is not None:
        return _STRATEGY_CACHE
    path = _cache_path(_SOLVE_DPP)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _STRATEGY_CACHE = pickle.load(f)
                return _STRATEGY_CACHE
        except Exception:
            pass  # rebuild on any corruption
    _STRATEGY_CACHE = build_and_cache(_SOLVE_DPP)
    return _STRATEGY_CACHE


# ---------------------------------------------------------------------------
# Observation parsing (mirrors liar_dice_policy regexes)
# ---------------------------------------------------------------------------

_RE_DICE = re.compile(r"Your dice:\s*\[([^\]]+)\]")
_RE_TOTAL = re.compile(r"Total dice in game:\s*(\d+)")
_RE_BID = re.compile(r'Current bid:\s*"(\d+)-(\d+)"')


def _parse_observation(observation: str) -> tuple[list[int], int, "int | None"]:
    """Return (our_dice, total_dice, current_bid_id)."""
    dice_match = _RE_DICE.search(observation)
    our_dice = [int(x.strip()) for x in dice_match.group(1).split(",")] if dice_match else []

    total_match = _RE_TOTAL.search(observation)
    total_dice = int(total_match.group(1)) if total_match else len(our_dice) * 2

    bid_match = _RE_BID.search(observation)
    current_bid_id = (
        encode_bid(int(bid_match.group(1)), int(bid_match.group(2))) if bid_match else None
    )
    return our_dice, total_dice, current_bid_id


# ---------------------------------------------------------------------------
# Abstraction: project a real (hand, history) onto the solved game
# ---------------------------------------------------------------------------

def _abstract_hand(our_dice: list[int], solve_dpp: int) -> tuple[int, ...]:
    """Bucket a real hand down to `solve_dpp` representative dice.

    Keep the `solve_dpp` most informative dice: prefer wilds (6s) and the
    largest matching groups, so the abstracted hand preserves the player's
    strongest faces (the ones that most drive bidding). Returned sorted to
    match solver hand keys.
    """
    if len(our_dice) <= solve_dpp:
        # pad with uniform-ish low dice is wrong; just sort what we have.
        return tuple(sorted(our_dice))
    # rank dice: wilds first, then by how many of that face we hold
    counts: dict[int, int] = defaultdict(int)
    for d in our_dice:
        counts[d] += 1
    def rank(d: int) -> tuple:
        return (d == WILD_FACE, counts[d], d)
    chosen = sorted(our_dice, key=rank, reverse=True)[:solve_dpp]
    return tuple(sorted(chosen))


def _abstract_bid_id(real_id: "int | None", real_total: int, solve_total: int) -> "int | None":
    """Project a real bid id onto the solved game's id space by scaling the
    quantity proportionally (face preserved). Challenge maps to challenge."""
    if real_id is None:
        return None
    if real_id == liar_id(real_total):
        return liar_id(solve_total)
    q, f = decode_bid(real_id)
    # scale quantity from real_total to solve_total, clamp to [1, solve_total]
    scaled_q = max(1, min(solve_total, round(q * solve_total / real_total)))
    return encode_bid(scaled_q, f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strategy_for(observation: str) -> tuple[list[int], int, "int | None", dict]:
    our_dice, total_dice, current_bid_id = _parse_observation(observation)
    strategy = _load_strategy()
    return our_dice, total_dice, current_bid_id, strategy


def _lookup_distribution(
    our_dice: list[int],
    total_dice: int,
    current_bid_id: "int | None",
    strategy: dict,
    legal_action_ids: list[int],
) -> dict[int, float]:
    """Return {real_action_id: probability} from the cached strategy, restricted
    to and renormalized over `legal_action_ids`."""
    solve_dpp = strategy["dice_per_player"]
    solve_total = strategy["total_dice"]
    exact = (total_dice == solve_total)

    if exact:
        hand_key = tuple(sorted(our_dice))
        # reconstruct the public history that leads here. We only know the
        # current bid (the env shows just the latest), so use a single-bid
        # history when a bid exists, else the empty (opening) history.
        history = () if current_bid_id is None else (current_bid_id,)
        entry = strategy["strategy"].get((hand_key, history))
        if entry is not None:
            return _restrict({aid: p for aid, p in entry}, legal_action_ids)
        # fall through to abstraction if exact key missing
    # --- abstraction path ---
    abs_hand = _abstract_hand(our_dice, solve_dpp)
    abs_bid = _abstract_bid_id(current_bid_id, total_dice, solve_total)
    history = () if abs_bid is None else (abs_bid,)
    entry = strategy["strategy"].get((abs_hand, history))
    if entry is None:
        return {}
    # map abstracted action ids back to the real id space
    mapped: dict[int, float] = {}
    for abs_aid, p in entry:
        real_aid = _expand_action_id(abs_aid, solve_total, total_dice)
        mapped[real_aid] = mapped.get(real_aid, 0.0) + p
    return _restrict(mapped, legal_action_ids)


def _expand_action_id(abs_id: int, solve_total: int, real_total: int) -> int:
    """Inverse of _abstract_bid_id: map a solved-game action id to the real
    game's id space (scale quantity up, preserve face / challenge)."""
    if abs_id == liar_id(solve_total):
        return liar_id(real_total)
    q, f = decode_bid(abs_id)
    scaled_q = max(1, min(real_total, round(q * real_total / solve_total)))
    return encode_bid(scaled_q, f)


def _restrict(dist: dict[int, float], legal_action_ids: list[int]) -> dict[int, float]:
    """Keep only legal actions and renormalize. If none of the strategy's mass
    is legal, return {} (caller falls back)."""
    legal = set(legal_action_ids)
    kept = {aid: p for aid, p in dist.items() if aid in legal and p > 0.0}
    total = sum(kept.values())
    if total <= 0.0:
        return {}
    return {aid: p / total for aid, p in kept.items()}


def _ev_scores(
    our_dice: list[int], total_dice: int, current_bid_id: "int | None",
    legal_action_ids: list[int],
) -> dict[int, float]:
    """Flat-prior score per legal action: a bid by P(bid true) with the opponent's dice
    uniform, a challenge by P(current bid false). Wild-aware, like the env."""
    n_hidden = total_dice - len(our_dice)
    lid = liar_id(total_dice)
    current_bid = decode_bid(current_bid_id) if current_bid_id is not None else None

    scores: dict[int, float] = {}
    for aid in legal_action_ids:
        if aid == lid and current_bid is not None:
            scores[aid] = 1.0 - _bid_true_prob(our_dice, current_bid, n_hidden)
        else:
            q, f = decode_bid(aid)
            scores[aid] = _bid_true_prob(our_dice, (q, f), n_hidden)
    return scores


def _heuristic_distribution(
    our_dice: list[int], total_dice: int, current_bid_id: "int | None",
    legal_action_ids: list[int],
) -> dict[int, float]:
    """Fallback when the cached strategy has no usable mass for this info set:
    a sharpened distribution over P(action is good), so we still mix sensibly.
    Uses the same wild-aware probability model as the env."""
    scores = _ev_scores(our_dice, total_dice, current_bid_id, legal_action_ids)
    # softmax with a sharp temperature; preserves a mix but favours good moves
    temp = 0.1
    m = max(scores.values())
    exps = {aid: math.exp((s - m) / temp) for aid, s in scores.items()}
    z = sum(exps.values())
    return {aid: e / z for aid, e in exps.items()}


def _bid_true_prob(our_dice: list[int], bid: tuple[int, int], n_hidden: int) -> float:
    q, f = bid
    known = _matching_count(tuple(our_dice), f)
    p_hit = (1 / 6) if f == WILD_FACE else (2 / 6)
    still_needed = q - known
    if still_needed <= 0:
        return 1.0
    if n_hidden <= 0:
        return 0.0
    # P(X >= still_needed), X ~ Binomial(n_hidden, p_hit)
    s = 0.0
    for k in range(still_needed, n_hidden + 1):
        s += math.comb(n_hidden, k) * (p_hit ** k) * ((1 - p_hit) ** (n_hidden - k))
    return s


# ───────────────────────────────────────────────────────────────────────────
# Memoryless external-sampling MCCFR teacher (robust, low-exploitability)
# ───────────────────────────────────────────────────────────────────────────
# The deep 5-dice Dudo tree has no tractable exact Nash, and the agent only ever
# observes (own dice, current bid) — never the full bid history — so we solve the
# game under an (own-dice, current-bid) INFORMATION ABSTRACTION using external-
# sampling Monte-Carlo CFR. Each iteration samples one deal + one trajectory, so
# it scales to any dice count. The average strategy is a near-equilibrium of the
# abstracted game and is far less exploitable than the EV heuristic — measured at
# 1 die each, full-history best-response exploitability is ~0.51 vs ~0.79 (EV) and
# ~1.56 (uniform) — hence more robust against a strong PvP opponent. Blueprints
# are solved per total-dice count and cached to disk; the EV policy remains the
# fallback for unsolved / asymmetric configs.

_MCCFR_ITERS = 60_000          # external-sampling iterations per dice config
_MCCFR_MAX_TOTAL = 10          # solve symmetric configs with total dice <= this
# Wall-clock cap on a COLD blueprint build. The 10-dice tree runs at ~12 it/s,
# so the 60k target alone is ~80 min of single-threaded, BLOCKING compute -- far
# more than any trajectory-generation budget. CFR+ is anytime-valid (the average
# strategy is usable after any number of iterations), so we train in chunks and
# stop at whichever of (iters, seconds) comes first, then cache what we have.
# Override with LIARS_DICE_MCCFR_SECONDS / LIARS_DICE_MCCFR_ITERS. The persistent
# disk cache means this cold cost is paid at most once across runs.
_MCCFR_MAX_SECONDS = 90.0
_MCCFR_CHUNK = 200             # iterations per time-check chunk
_UNSET = object()              # sentinel: "caller did not pass max_seconds"
_BLUEPRINT_MEM: "dict[int, dict | None]" = {}   # in-process: total_dice -> blueprint


def _mccfr_iters() -> int:
    raw = os.getenv("LIARS_DICE_MCCFR_ITERS")
    if raw is None or raw == "":
        return _MCCFR_ITERS
    try:
        return max(1, int(raw))
    except ValueError:
        return _MCCFR_ITERS


def _mccfr_seconds() -> "float | None":
    """Wall-clock budget for a cold build. None disables the time cap (build the
    full iteration target)."""
    raw = os.getenv("LIARS_DICE_MCCFR_SECONDS")
    if raw is None:
        return _MCCFR_MAX_SECONDS
    if raw == "" or raw.lower() in ("0", "none", "off", "inf"):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        return _MCCFR_MAX_SECONDS


class _MccfrNode:
    __slots__ = ("regret", "strat_sum", "n")

    def __init__(self, n: int):
        self.regret = [0.0] * n
        self.strat_sum = [0.0] * n
        self.n = n

    def strategy(self) -> list[float]:
        pos = [r if r > 0.0 else 0.0 for r in self.regret]
        s = sum(pos)
        return [p / s for p in pos] if s > 0.0 else [1.0 / self.n] * self.n

    def average(self) -> list[float]:
        s = sum(self.strat_sum)
        return [x / s for x in self.strat_sum] if s > 0.0 else [1.0 / self.n] * self.n


class DudoMCCFR:
    """External-sampling MCCFR over 1v1 Dudo with (hand, current-bid) info sets."""

    def __init__(self, dice_per_player: int):
        self.dpp = dice_per_player
        self.total = 2 * dice_per_player
        self.liar = liar_id(self.total)
        self.hands = _hands(dice_per_player)
        self.hp = [_hand_probability(h) for h in self.hands]
        self.nodes: dict = {}

    def _node(self, key, n):
        nd = self.nodes.get(key)
        if nd is None:
            nd = _MccfrNode(n)
            self.nodes[key] = nd
        return nd

    def _challenge_u0(self, hands, bid_id, challenger):
        q, f = decode_bid(bid_id)
        count = _matching_count(hands[0], f) + _matching_count(hands[1], f)
        winner = challenger if count < q else 1 - challenger
        return 1.0 if winner == 0 else -1.0

    def _traverse(self, hands, history, traverser, rng):
        player = len(history) % 2
        last = history[-1] if history else None
        if last == self.liar:
            challenger = (len(history) - 1) % 2
            u0 = self._challenge_u0(hands, history[-2], challenger)
            return u0 if traverser == 0 else -u0
        actions = legal_action_ids_for(last, self.total)
        node = self._node((hands[player], last), len(actions))  # MEMORYLESS key
        strat = node.strategy()
        if player == traverser:
            util = [0.0] * len(actions)
            nu = 0.0
            for i, a in enumerate(actions):
                util[i] = self._traverse(hands, history + (a,), traverser, rng)
                nu += strat[i] * util[i]
            for i in range(len(actions)):
                node.regret[i] = max(0.0, node.regret[i] + util[i] - nu)  # CFR+
                node.strat_sum[i] += strat[i]
            return nu
        # external sampling: sample a single opponent action
        r = rng.random()
        acc = 0.0
        idx = len(actions) - 1
        for i, p in enumerate(strat):
            acc += p
            if r <= acc:
                idx = i
                break
        return self._traverse(hands, history + (actions[idx],), traverser, rng)

    def train(self, iters, rng):
        for _ in range(iters):
            h0 = rng.choices(self.hands, weights=self.hp, k=1)[0]
            h1 = rng.choices(self.hands, weights=self.hp, k=1)[0]
            self._traverse((h0, h1), (), 0, rng)
            self._traverse((h0, h1), (), 1, rng)

    def blueprint(self) -> dict:
        """{(sorted-hand, current-bid-or-None): {action_id: prob}}."""
        bp = {}
        for (hand, current), nd in self.nodes.items():
            acts = legal_action_ids_for(current, self.total)
            avg = nd.average()
            bp[(hand, current)] = {acts[i]: avg[i] for i in range(len(acts))}
        return bp


def build_mccfr_blueprint(
    total_dice: int,
    iters: "int | None" = None,
    seed: int = 0,
    max_seconds: "float | None | object" = _UNSET,
) -> "tuple[dict, int]":
    """Train an external-sampling MCCFR blueprint for ``total_dice``.

    Trains in ``_MCCFR_CHUNK``-sized batches up to ``iters`` (default
    ``_mccfr_iters()``), stopping early once ``max_seconds`` of wall-clock has
    elapsed. CFR+'s average strategy is valid after any number of iterations, so
    an early stop just yields a slightly more exploitable (but still usable)
    blueprint. Returns ``(blueprint, iterations_done)``.
    """
    iters = iters if iters is not None else _mccfr_iters()
    if max_seconds is _UNSET:
        max_seconds = _mccfr_seconds()
    solver = DudoMCCFR(total_dice // 2)
    rng = random.Random(seed)
    start = time.monotonic()
    done = 0
    while done < iters:
        chunk = min(_MCCFR_CHUNK, iters - done)
        solver.train(chunk, rng)
        done += chunk
        if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
            break
    return solver.blueprint(), done


def _mccfr_cache_path(total_dice: int) -> str:
    return os.path.join(_CACHE_DIR, f"liar_dice_mccfr_total{total_dice}.pkl")


def _load_blueprint_payload(path: str):
    """Return (blueprint_dict, iters) from a cache file, or None on miss/error.
    Accepts both the rich ``{"blueprint", "iters", ...}`` payload and the legacy
    bare-blueprint-dict format."""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return None
    if isinstance(obj, dict) and "blueprint" in obj:
        return obj["blueprint"], int(obj.get("iters", 0))
    if isinstance(obj, dict):  # legacy: the bare blueprint dict
        return obj, -1
    return None


def _save_blueprint_payload(path: str, bp: dict, iters: int, total_dice: int) -> None:
    """Atomically persist a blueprint payload (write-temp-then-rename) so a
    concurrent reader/worker never sees a half-written pickle."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    payload = {"blueprint": bp, "iters": iters, "total_dice": total_dice}
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)


def _get_blueprint(total_dice: int):
    """Lazy per-total-dice blueprint (in-process + disk cache). Returns None for
    unsupported configs (odd total, or larger than _MCCFR_MAX_TOTAL).

    On a cache miss this builds a blueprint within the ``LIARS_DICE_MCCFR_SECONDS``
    wall-clock budget (so it never blocks generation indefinitely) and persists it
    to the (mounted, persistent) cache dir, so the cold cost is paid at most once."""
    if total_dice in _BLUEPRINT_MEM:
        return _BLUEPRINT_MEM[total_dice]
    if total_dice < 2 or total_dice > _MCCFR_MAX_TOTAL or total_dice % 2 != 0:
        _BLUEPRINT_MEM[total_dice] = None
        return None
    bp = None
    path = _mccfr_cache_path(total_dice)
    if os.path.exists(path):
        loaded = _load_blueprint_payload(path)
        if loaded is not None:
            bp = loaded[0]
    if bp is None:
        try:
            bp, done = build_mccfr_blueprint(total_dice)
            try:
                _save_blueprint_payload(path, bp, done, total_dice)
            except Exception:
                pass  # cache write is best-effort; in-process copy still usable
        except Exception:
            bp = None
    _BLUEPRINT_MEM[total_dice] = bp
    return bp


# --- exploitability via a correct full-history best response (for tests) -----

def _dudo_br(total_dice, sigma_fn, br_player, br_hand, history, opp_reach):
    liar = liar_id(total_dice)
    player = len(history) % 2
    last = history[-1] if history else None
    if last == liar:
        challenger = (len(history) - 1) % 2
        q, f = decode_bid(history[-2])
        tot = 0.0
        for oh, w in opp_reach.items():
            hands = (br_hand, oh) if br_player == 0 else (oh, br_hand)
            count = _matching_count(hands[0], f) + _matching_count(hands[1], f)
            winner = challenger if count < q else 1 - challenger
            u0 = 1.0 if winner == 0 else -1.0
            tot += w * (u0 if br_player == 0 else -u0)
        return tot
    actions = legal_action_ids_for(last, total_dice)
    if player == br_player:
        return max(_dudo_br(total_dice, sigma_fn, br_player, br_hand, history + (a,), opp_reach)
                   for a in actions)
    tot = 0.0
    for ai, a in enumerate(actions):
        new_reach = {}
        for oh, w in opp_reach.items():
            p = sigma_fn(oh, last, actions)[ai]
            if p > 0.0:
                new_reach[oh] = w * p
        if new_reach:
            tot += _dudo_br(total_dice, sigma_fn, br_player, br_hand, history + (a,), new_reach)
    return tot


def dudo_exploitability(total_dice: int, sigma_fn) -> float:
    """Full-history best-response exploitability of a memoryless strategy
    ``sigma_fn(hand, current_bid, actions) -> probs``. ~0 == Nash of the
    (unabstracted) game; larger == more exploitable by a strong opponent."""
    hs = _hands(total_dice // 2)
    hp = [_hand_probability(h) for h in hs]
    prior = {h: hp[i] for i, h in enumerate(hs)}
    total = 0.0
    for bp_player in (0, 1):
        for i, bh in enumerate(hs):
            total += hp[i] * _dudo_br(total_dice, sigma_fn, bp_player, bh, (), prior)
    return total


# ───────────────────────────────────────────────────────────────────────────
# Visible-bid reader: the default label
# ───────────────────────────────────────────────────────────────────────────
# The eval prompt shows our dice, the LATEST bid (never the bid history) and our seat, and the model
# decodes greedily, so the label is a deterministic function of exactly those three.
#
# Opponent model: the player who made the latest bid is read from that bid alone. Its likelihood under
# each opponent hand is a softmax of that hand's own P(bid true) over every opening bid (the prompt never
# shows what the bid raised from), averaged over _LIKELIHOOD_TEMPS, then mixed LIARS_PRIOR_MIX toward the
# prior-mean likelihood so bluffs and non-reading bidders do not over-steer the posterior.
# Decision: a bid scores P(true | our dice, posterior), Liar scores 1 - P(current bid true). Take the
# argmax, settle ties on the HIGHEST bid id (the highest claim we are equally sure of) and call Liar only
# when it is strictly better.
#
# Measured at teacher level, exact over all 252 x 252 deals and both seats: 63-69% vs non-reading
# policies (68.4% vs the sampled flat-EV label it replaces), 58-66% vs latest-bid readers (where the
# flat-EV label scores 30-42%), 97% vs uniform random; with 10% near-miss errors on both sides it keeps
# 55-57% vs the readers and 60-62% vs non-readers. The mix plateaus at 0.20-0.25 (0.15: -2pp, 0.40: -9pp
# in the worst matchup).

LIARS_PRIOR_MIX = 0.25
_LIKELIHOOD_TEMPS = (0.01, 0.05, 0.1, 0.3)
_SCORE_TOL = 1e-9

# LIARS_TEACHER picks the label: "reader" (default), "ev_argmax" (flat-EV scores, deterministic argmax,
# ties high) or "ev" (the sampled flat-EV label, MCCFR blueprint when LIARS_DICE_FORCE_EV=0, shipped
# before the reader). One-line rollback: _TEACHER_DEFAULT = "ev".
_TEACHER_DEFAULT = "reader"
_TEACHER_MODES = ("reader", "ev_argmax", "ev")


def _teacher_mode() -> str:
    mode = os.getenv("LIARS_TEACHER", _TEACHER_DEFAULT).strip().lower()
    return mode if mode in _TEACHER_MODES else _TEACHER_DEFAULT


def _force_ev() -> bool:
    return os.getenv("LIARS_DICE_FORCE_EV", "1").strip() == "1"


def blueprint_needed() -> bool:
    """True only when choose_action can read the MCCFR blueprint (LIARS_TEACHER=ev with
    LIARS_DICE_FORCE_EV=0); generation skips the cold build otherwise."""
    return _teacher_mode() == "ev" and not _force_ev()


@functools.lru_cache(maxsize=None)
def _opening_likelihoods(n_opp: int, n_hidden: int) -> tuple[tuple[float, ...], ...]:
    """lik[h][b]: P(an opponent holding _hands(n_opp)[h] opens with bid b) when it scores every bid by
    its own P(true) with ``n_hidden`` dice unseen, softmax averaged over _LIKELIHOOD_TEMPS."""
    n_bids = liar_id(n_opp + n_hidden)
    rows = []
    for hand in _hands(n_opp):
        scores = [_bid_true_prob(list(hand), decode_bid(b), n_hidden) for b in range(n_bids)]
        top = max(scores)
        acc = [0.0] * n_bids
        for temp in _LIKELIHOOD_TEMPS:
            weights = [math.exp((s - top) / temp) for s in scores]
            z = sum(weights)
            acc = [a + w / z for a, w in zip(acc, weights)]
        rows.append(tuple(a / len(_LIKELIHOOD_TEMPS) for a in acc))
    return tuple(rows)


@functools.lru_cache(maxsize=None)
def _opp_match_dist(bid_id: "int | None", total_dice: int, n_own: int,
                    prior_mix: float = LIARS_PRIOR_MIX) -> tuple[tuple[float, ...], ...]:
    """md[f][k]: P(the opponent holds exactly k dice counting toward face f) given that its latest
    visible bid is ``bid_id`` (None: no bid yet, i.e. the prior). Row 0 is unused."""
    n_opp = total_dice - n_own
    hands = _hands(n_opp)
    prior = [_hand_probability(h) for h in hands]
    if bid_id is None:
        weights = prior
    else:
        lik = [row[bid_id] for row in _opening_likelihoods(n_opp, n_own)]
        mean = sum(p * x for p, x in zip(prior, lik))
        weights = [p * ((1.0 - prior_mix) * x + prior_mix * mean) for p, x in zip(prior, lik)]
    z = sum(weights)
    md = [[0.0] * (n_opp + 1) for _ in range(NUM_FACES + 1)]
    for hand, w in zip(hands, weights):
        for f in range(1, NUM_FACES + 1):
            md[f][_matching_count(hand, f)] += w / z
    return tuple(tuple(row) for row in md)


def _p_true_post(q: int, f: int, our_dice: list[int], md) -> float:
    """P(at least q dice count toward face f) given our dice and the opponent match distribution md."""
    need = q - _matching_count(tuple(our_dice), f)
    if need <= 0:
        return 1.0
    return sum(md[f][need:])


def _pick(bid_scores: dict[int, float], liar_score: "float | None", lid: int,
          tie_high: bool = True) -> "int | None":
    """Best-scoring bid, ties to the highest id (lowest when not tie_high); Liar instead when its score
    is strictly higher. None when nothing is scored."""
    best_bid, best = None, -1.0
    if bid_scores:
        best = max(bid_scores.values())
        tied = [a for a, s in bid_scores.items() if s >= best - _SCORE_TOL]
        best_bid = max(tied) if tie_high else min(tied)
    if liar_score is not None and (best_bid is None or liar_score > best + _SCORE_TOL):
        return lid
    return best_bid


def _readable(our_dice: list[int], total_dice: int, current_bid_id: "int | None") -> bool:
    """The reader needs a parsed symmetric deal (the eval game) and an in-range current bid."""
    return (bool(our_dice) and total_dice == 2 * len(our_dice)
            and all(1 <= d <= NUM_FACES for d in our_dice)
            and (current_bid_id is None or 0 <= current_bid_id < liar_id(total_dice)))


def _reader_action(our_dice: list[int], total_dice: int, current_bid_id: "int | None",
                   legal_action_ids: list[int], prior_mix: float = LIARS_PRIOR_MIX,
                   tie_high: bool = True) -> "int | None":
    """Argmax of P(claim true) over the legal actions under the latest-bid posterior."""
    lid = liar_id(total_dice)
    md = _opp_match_dist(current_bid_id, total_dice, len(our_dice), prior_mix)
    bids = {a: _p_true_post(*decode_bid(a), our_dice, md) for a in legal_action_ids if a != lid}
    liar = None
    if current_bid_id is not None and lid in legal_action_ids:
        liar = 1.0 - _p_true_post(*decode_bid(current_bid_id), our_dice, md)
    return _pick(bids, liar, lid, tie_high)


def _ev_argmax(our_dice: list[int], total_dice: int, current_bid_id: "int | None",
               legal_action_ids: list[int]) -> "int | None":
    """Deterministic flat-EV label: the argmax of _ev_scores with the reader's tie rule."""
    lid = liar_id(total_dice)
    scores = _ev_scores(our_dice, total_dice, current_bid_id, legal_action_ids)
    liar = scores.pop(lid) if (current_bid_id is not None and lid in scores) else None
    return _pick(scores, liar, lid)


# Non-hero seats of the liars generation pool (envs.liar_dice_pool), i.e. the styles the reader meets in
# the field: "reader" is the label itself; "ev_sampled" is the flat-EV label the reader replaced;
# "reader_u0_lo" / "reader_u0_hi" read the latest bid without the prior mix and settle ties on the lowest
# (cautious minimum raises) / highest id (aggressive jumps); "ev_noisy" is a flat-EV argmax student that
# plays a uniformly random legal move with probability _NOISY_EPS.
OPPONENT_KINDS = ("reader", "ev_sampled", "reader_u0_lo", "reader_u0_hi", "ev_noisy")
_NOISY_EPS = 0.2


def opponent_action(kind: str, observation: str, legal_action_ids: list[int], rng) -> int:
    """Move of pool opponent ``kind`` (one of OPPONENT_KINDS). Always a legal id."""
    if kind == "reader":
        return choose_action(observation, 0, legal_action_ids, rng=rng)
    our_dice, total_dice, current_bid_id = _parse_observation(observation)
    if kind == "ev_sampled":
        dist = _heuristic_distribution(our_dice, total_dice, current_bid_id, legal_action_ids)
        ids = list(dist)
        action = rng.choices(ids, weights=[dist[a] for a in ids], k=1)[0]
    elif kind == "ev_noisy" and rng.random() < _NOISY_EPS:
        action = rng.choice(list(legal_action_ids))
    elif kind in ("reader_u0_lo", "reader_u0_hi") and _readable(our_dice, total_dice, current_bid_id):
        action = _reader_action(our_dice, total_dice, current_bid_id, legal_action_ids,
                                prior_mix=0.0, tie_high=kind == "reader_u0_hi")
    else:
        action = _ev_argmax(our_dice, total_dice, current_bid_id, legal_action_ids)
    return action if action in legal_action_ids else rng.choice(list(legal_action_ids))


def choose_action(
    observation: str,
    player_id: int,
    legal_action_ids: list[int],
    rng=None,
) -> int:
    """Pick a legal action id with the LIARS_TEACHER label.

    Default "reader": the deterministic visible-bid reader above (``rng`` unused); an
    observation that does not parse into a symmetric deal gets the flat-EV argmax.
    "ev_argmax": that flat-EV argmax. "ev" (the previous label): sample the MCCFR
    blueprint of the (own-dice, current-bid) abstraction when LIARS_DICE_FORCE_EV=0
    and one exists, else the exact wild-aware binomial-EV policy
    (`_heuristic_distribution`): a bid (q,f) scored by P(>=q show f) and a challenge
    by P(current bid is false), sharp-softmax sampled.
    """
    if rng is None:
        rng = random
    if not legal_action_ids:
        raise ValueError("legal_action_ids is empty")
    if len(legal_action_ids) == 1:
        return legal_action_ids[0]

    our_dice, total_dice, current_bid_id = _parse_observation(observation)
    mode = _teacher_mode()
    if mode != "ev":
        try:
            if mode == "reader" and _readable(our_dice, total_dice, current_bid_id):
                action = _reader_action(our_dice, total_dice, current_bid_id, legal_action_ids)
            else:
                action = _ev_argmax(our_dice, total_dice, current_bid_id, legal_action_ids)
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001 - never raise in generation; the sampler below is the backstop
            pass

    # Primary: the MCCFR Nash-abstraction blueprint for this (symmetric) config.
    # MEASURED 2026-07-20 — the blueprint is DEFAULT-OFF: at the deploy count (total_dice=10)
    # a single MCCFR chunk (200 iters) already blows past the 90s cold-build cap, so the gen
    # only ever gets a handful of iterations -> a near-untrained, near-uniform blueprint that
    # is WORSE than the exact wild-aware binomial-EV policy (and it wastes ~90s of gen time).
    # So we default to EV; set LIARS_DICE_FORCE_EV=0 to opt the blueprint back in (small dice
    # counts where it actually converges, or once a fully-trained blueprint is pre-baked).
    if not _force_ev():
        bp = _get_blueprint(total_dice)
        if bp is not None and len(our_dice) == total_dice // 2:
            entry = bp.get((tuple(sorted(our_dice)), current_bid_id))
            if entry:
                kept = {a: p for a, p in entry.items() if a in legal_action_ids and p > 0.0}
                tot = sum(kept.values())
                if tot > 0.0:
                    ids = list(kept)
                    return rng.choices(ids, weights=[kept[a] / tot for a in ids], k=1)[0]

    # Fallback: exact-EV policy.
    dist = _heuristic_distribution(our_dice, total_dice, current_bid_id, legal_action_ids)
    ids = list(dist.keys())
    return rng.choices(ids, weights=[dist[a] for a in ids], k=1)[0]


# NOTE: the live teacher is the visible-bid reader; the memoryless MCCFR blueprint
# and the EV sampler are the LIARS_TEACHER=ev backstop. The full-history DudoSolver /
# CFR+ machinery earlier in this file is retained for reference only — it is correct
# in structure but enumerates the full bid history, which is both intractable at real
# dice counts and finer than the agent can observe (current bid only); it is
# intentionally not invoked.


if __name__ == "__main__":
    payload = build_and_cache(verbose=True)
    print(f"Solved dice_per_player={payload['dice_per_player']} "
          f"(total_dice={payload['total_dice']}), "
          f"iterations={payload['iterations']}, "
          f"exploitability={payload['exploitability']:.6f}, "
          f"info_sets={len(payload['strategy'])}")
