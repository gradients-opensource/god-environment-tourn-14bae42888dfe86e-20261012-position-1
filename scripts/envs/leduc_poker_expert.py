"""Near-Nash expert teacher for Leduc Hold'em SFT data generation.

This module solves the EXACT two-round Leduc Hold'em game tree with **CFR+**
(Counterfactual Regret Minimization with regret-matching+ and linear strategy
averaging) to near-zero exploitability, caches the resulting *average strategy*
("blueprint") to ``envs/.expert_cache/leduc_poker.pkl`` once, loads it on import,
and exposes an O(1) action-selection API:

    def choose_action(observation, player_id, legal_action_ids, rng=None) -> int

The expert **samples** a legal action from the equilibrium average strategy at
the matching information set (it does NOT argmax) so that bluffing and
check-raise frequencies are preserved in the generated trajectories.

Why this replaces the previous teacher
--------------------------------------
``envs/leduc_poker_trajectories.py`` previously generated SFT trajectories by
playing **uniformly random** legal actions (see its ``_random_action``).  This
module provides a principled game-theoretic replacement.

Leduc rules solved here (matching the repo's SYSTEM_PROMPT_LEDUC_POKER):
  * Deck: 2 suits x 3 ranks {J, Q, K}  -> 6 cards.
  * Each player antes 1, gets 1 private card; 1 public card revealed in round 2.
  * Two betting rounds.  Round 1 raise size = 2, round 2 raise size = 4.
  * Max 2 raises per round.
  * Showdown: a pair (private rank == public rank) beats any non-pair; otherwise
    higher private rank wins; equal ranks tie.

Action encoding (matches OpenSpiel Leduc and the env observation labels):
    0 -> Fold, 1 -> Call/Check, 2 -> Raise/Bet
``choose_action`` does not rely on hardcoded ids though: it maps the
equilibrium abstract action (fold/check-call/raise) to the matching
``legal_action_ids`` entry **by parsing the action labels** out of the
observation, so it is robust to id remapping.

Standard library only.  Imports READ-ONLY from ``leduc_poker_env`` (the
observation parser).  Never imports from ``leduc_poker_trajectories``.
"""

from __future__ import annotations

import os
import pickle
import random
import re
from dataclasses import dataclass

# READ-ONLY import: observation parser + card-rank table.
from envs.leduc_poker_env import parse_game_state, _CARD_RANK


# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

_RANKS = (1, 2, 3)                      # J=1, Q=2, K=3
_DECK = (1, 1, 2, 2, 3, 3)             # six physical cards (rank per slot)
_ANTE = 1
_RAISE_SIZE = {1: 2, 2: 4}            # raise amount added per round
_MAX_RAISES = 2                        # max raises per round

# Abstract actions
_FOLD = "f"
_CHECK_CALL = "c"
_RAISE = "r"


# WIN/LOSS (sign) payoff (winner-teardown 2026-08-29). The tournament scores per-env WIN-RATE, not
# chips, so a FOLD is a certain LOST hand while a call keeps a showdown chance. A chips-EV Nash (the
# old default) folds weak hands at the chips-optimal rate and auto-concedes ~8-15% of hands the eval
# counts as full losses. Default ON: solve the game the eval ACTUALLY scores — fold = -/+1, showdown =
# +/-1/0 on the SIGN of the outcome — so fold is weakly dominated and vanishes at reached infosets.
# LEDUC_WIN_LOSS=0 restores the chips-EV solve. (Blueprint cache is keyed by this so the two never mix.)
_LP_WL = os.getenv("LEDUC_WIN_LOSS", "1") != "0"

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".expert_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "leduc_poker.pkl")

# CFR+ training iterations.  Leduc has ~936 info states; a few thousand CFR+
# iterations drive exploitability well below 1e-3 (chips/game).
_TRAIN_ITERATIONS = 2000


# ---------------------------------------------------------------------------
# Game logic — terminal evaluation + tree walk
# ---------------------------------------------------------------------------

def _showdown_winner(p0_card: int, p1_card: int, board: int) -> int:
    """Return +1 if P0 wins, -1 if P1 wins, 0 tie (at showdown)."""
    p0_pair = (p0_card == board)
    p1_pair = (p1_card == board)
    if p0_pair and not p1_pair:
        return 1
    if p1_pair and not p0_pair:
        return -1
    # Neither pairs (board can't pair both — only two suits per rank) -> high card
    if p0_card > p1_card:
        return 1
    if p0_card < p1_card:
        return -1
    return 0


@dataclass
class _RoundState:
    """Betting state within a single round."""
    history: str          # sequence of actions taken this round, e.g. "rc", "crr"
    num_raises: int       # raises taken this round
    # pot contributions are tracked at the game level

    def to_act_first(self) -> bool:
        return len(self.history) == 0


# ---------------------------------------------------------------------------
# Info-set key helpers
# ---------------------------------------------------------------------------
#
# An info set is identified by what the acting player can observe:
#   (own private rank, public rank or 0, round-1 history, round-2 history)
# History strings use the abstract-action alphabet {c, r} (folds end the game,
# so they never appear *inside* a stored history that needs a decision).
#
# This is exactly the information our observation parser exposes, so
# choose_action can reconstruct the identical key at inference time.

def _infoset_key(private_rank: int, public_rank: int, r1_hist: str, r2_hist: str) -> str:
    return f"{private_rank}|{public_rank}|{r1_hist}|{r2_hist}"


# ---------------------------------------------------------------------------
# CFR+ solver
# ---------------------------------------------------------------------------

class _Node:
    """Regret-matching+ node for a single information set."""

    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions: tuple[str, ...]):
        self.actions = actions
        self.regret_sum = [0.0] * len(actions)
        self.strategy_sum = [0.0] * len(actions)

    def strategy(self) -> list[float]:
        """Current regret-matching+ strategy (positive regrets normalized)."""
        pos = [r if r > 0.0 else 0.0 for r in self.regret_sum]
        total = sum(pos)
        n = len(self.actions)
        if total > 0.0:
            return [p / total for p in pos]
        return [1.0 / n] * n

    def average_strategy(self) -> list[float]:
        total = sum(self.strategy_sum)
        n = len(self.actions)
        if total > 0.0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / n] * n


class _LeducCFR:
    """CFR+ solver over the exact Leduc tree."""

    def __init__(self):
        self.nodes: dict[str, _Node] = {}

    # -- tree structure -----------------------------------------------------

    @staticmethod
    def _legal_actions(round_no: int, r_hist: str, num_raises: int, facing_bet: bool) -> tuple[str, ...]:
        """Legal abstract actions given the within-round history."""
        acts: list[str] = []
        if facing_bet:
            acts.append(_FOLD)
        acts.append(_CHECK_CALL)
        if num_raises < _MAX_RAISES:
            acts.append(_RAISE)
        return tuple(acts)

    def _get_node(self, key: str, actions: tuple[str, ...]) -> _Node:
        node = self.nodes.get(key)
        if node is None:
            node = _Node(actions)
            self.nodes[key] = node
        return node

    # -- CFR+ recursion -----------------------------------------------------

    def _cfr(
        self,
        p0_card: int,
        p1_card: int,
        board: int,
        round_no: int,
        r1_hist: str,
        r2_hist: str,
        pot0: int,
        pot1: int,
        reach0: float,
        reach1: float,
        iteration: int,
    ) -> float:
        """Return expected value for player 0 (chips) at this node."""
        cur_hist = r1_hist if round_no == 1 else r2_hist
        # Determine whose turn: P0 acts first each round; players alternate.
        player = len(cur_hist) % 2  # 0 or 1
        num_raises = cur_hist.count("r")
        facing_bet = cur_hist.endswith("r")

        actions = self._legal_actions(round_no, cur_hist, num_raises, facing_bet)

        my_card = p0_card if player == 0 else p1_card
        public_for_key = board if round_no == 2 else 0
        key = _infoset_key(my_card, public_for_key, r1_hist if round_no == 1 else r1_hist,
                           r2_hist if round_no == 2 else "")
        # NB: round-1 history is part of the round-2 key too, but in Leduc the
        # round-1 betting does not change available round-2 actions or payoffs
        # beyond pot size (which is fully determined by r1_hist). For a correct
        # abstraction the round-2 info set must include r1_hist so that pot is
        # known. Build the full key:
        key = _infoset_key(my_card, public_for_key, r1_hist, r2_hist if round_no == 2 else "")

        node = self._get_node(key, actions)
        strat = node.strategy()

        my_reach = reach0 if player == 0 else reach1
        opp_reach = reach1 if player == 0 else reach0

        util = [0.0] * len(actions)
        node_util = 0.0

        for i, a in enumerate(actions):
            # Apply action, recurse.
            v = self._apply_action(
                a, player, p0_card, p1_card, board, round_no,
                r1_hist, r2_hist, pot0, pot1, reach0, reach1, strat[i], iteration,
            )
            util[i] = v
            node_util += strat[i] * v

        # Regret-matching+ update (CFR+: floor accumulated regret at 0).
        for i in range(len(actions)):
            # Counterfactual regret from P0's perspective; flip sign for P1.
            regret = (util[i] - node_util)
            if player == 0:
                cf_regret = opp_reach * regret
            else:
                cf_regret = opp_reach * (-regret)
            new_regret = node.regret_sum[i] + cf_regret
            node.regret_sum[i] = new_regret if new_regret > 0.0 else 0.0

        # Linear averaging weight (CFR+ weights strategy by iteration number).
        node.strategy_sum = [
            s + iteration * my_reach * strat[i] for i, s in enumerate(node.strategy_sum)
        ]

        return node_util

    def _apply_action(
        self, a, player, p0_card, p1_card, board, round_no,
        r1_hist, r2_hist, pot0, pot1, reach0, reach1, action_prob, iteration,
    ) -> float:
        """Apply one action and return resulting EV (for P0)."""
        new_reach0 = reach0 * action_prob if player == 0 else reach0
        new_reach1 = reach1 * action_prob if player == 1 else reach1

        if round_no == 1:
            new_r1 = r1_hist + a
            new_r2 = r2_hist
        else:
            new_r1 = r1_hist
            new_r2 = r2_hist + a

        if a == _FOLD:
            # The folder loses the hand. WIN/LOSS: a certain -/+1 (sign); chips: the folder loses
            # their current contribution.
            if player == 0:
                return -1.0 if _LP_WL else -float(pot0)   # P0 folds -> P0 loses
            else:
                return 1.0 if _LP_WL else float(pot1)     # P1 folds -> P0 wins

        if a == _RAISE:
            raise_amt = _RAISE_SIZE[round_no]
            # caller must first match any outstanding bet; here raise also matches.
            # Track contributions: raiser brings their contribution up to opp + raise.
            cur_hist = r1_hist if round_no == 1 else r2_hist
            facing = cur_hist.endswith("r")
            if player == 0:
                # match difference then add raise
                diff = (pot1 - pot0) if facing else 0
                npot0 = pot0 + diff + raise_amt
                npot1 = pot1
            else:
                diff = (pot0 - pot1) if facing else 0
                npot1 = pot1 + diff + raise_amt
                npot0 = pot0
            return self._cfr(
                p0_card, p1_card, board, round_no, new_r1, new_r2,
                npot0, npot1, new_reach0, new_reach1, iteration,
            )

        # a == CHECK_CALL
        cur_hist = r1_hist if round_no == 1 else r2_hist
        facing = cur_hist.endswith("r")
        if facing:
            # Call: match the outstanding bet -> pots equalize, round/betting closes.
            if player == 0:
                npot0 = pot1
                npot1 = pot1
            else:
                npot0 = pot0
                npot1 = pot0
            return self._after_betting_closes(
                p0_card, p1_card, board, round_no, new_r1, new_r2,
                npot0, npot1, new_reach0, new_reach1, iteration,
            )
        else:
            # Check.
            npot0, npot1 = pot0, pot1
            new_round_hist = (new_r1 if round_no == 1 else new_r2)
            if player == 0:
                # P0 checked -> P1 still to act; not closed.
                return self._cfr(
                    p0_card, p1_card, board, round_no, new_r1, new_r2,
                    npot0, npot1, new_reach0, new_reach1, iteration,
                )
            else:
                # P1 checked after P0 check -> both checked, round closes.
                return self._after_betting_closes(
                    p0_card, p1_card, board, round_no, new_r1, new_r2,
                    npot0, npot1, new_reach0, new_reach1, iteration,
                )

    def _after_betting_closes(
        self, p0_card, p1_card, board, round_no, r1_hist, r2_hist,
        pot0, pot1, reach0, reach1, iteration,
    ) -> float:
        """Round-1 betting closed -> deal public card (chance); round-2 -> showdown."""
        if round_no == 1:
            # Chance node: reveal public card uniformly among remaining 4 cards.
            remaining = list(_DECK)
            remaining.remove(p0_card)
            remaining.remove(p1_card)
            ev = 0.0
            n = len(remaining)
            for bcard in remaining:
                ev += self._cfr(
                    p0_card, p1_card, bcard, 2, r1_hist, "",
                    pot0, pot1, reach0, reach1, iteration,
                )
            return ev / n
        else:
            # Showdown.
            w = _showdown_winner(p0_card, p1_card, board)
            if w == 1:
                return 1.0 if _LP_WL else float(pot1)     # P0 wins the showdown
            if w == -1:
                return -1.0 if _LP_WL else -float(pot0)    # P0 loses the showdown
            return 0.0                                     # tie (identical in both schemes)

    # -- training driver ----------------------------------------------------

    def train(self, iterations: int) -> None:
        # Enumerate all distinct (p0, p1) physical-card deals.
        deals = []
        deck = list(_DECK)
        for i in range(len(deck)):
            for j in range(len(deck)):
                if i == j:
                    continue
                deals.append((deck[i], deck[j]))
        n_deals = len(deals)
        for t in range(1, iterations + 1):
            for (c0, c1) in deals:
                self._cfr(
                    c0, c1, 0, 1, "", "",
                    _ANTE, _ANTE, 1.0, 1.0, t,
                )
        # store n_deals for normalization (unused but harmless)
        self._n_deals = n_deals

    def blueprint(self) -> dict[str, dict[str, float]]:
        """Export average strategy: infoset_key -> {abstract_action: prob}."""
        bp: dict[str, dict[str, float]] = {}
        for key, node in self.nodes.items():
            avg = node.average_strategy()
            bp[key] = {a: avg[i] for i, a in enumerate(node.actions)}
        return bp


# ---------------------------------------------------------------------------
# Exploitability (best-response) computation
# ---------------------------------------------------------------------------

def _legal_for(cur_hist: str) -> tuple[str, ...]:
    facing = cur_hist.endswith("r")
    acts: list[str] = []
    if facing:
        acts.append(_FOLD)
    acts.append(_CHECK_CALL)
    if cur_hist.count("r") < _MAX_RAISES:
        acts.append(_RAISE)
    return tuple(acts)


def _opp_probs(blueprint, opp_rank, board_rank, rnd, r1, r2, acts):
    key = _infoset_key(opp_rank, board_rank if rnd == 2 else 0, r1, r2 if rnd == 2 else "")
    d = blueprint.get(key)
    if d is None:
        return [1.0 / len(acts)] * len(acts)
    return [d.get(a, 0.0) for a in acts]


def _br_value(blueprint, br_player, br_card, rnd, r1, r2, board,
              pot_br, pot_opp, opp_reach):
    """Information-set best-response EV (chips) for ``br_player`` holding physical
    card ``br_card`` (index into _DECK). The BR player conditions ONLY on its own
    card + the public card + betting history; the opponent's hidden card is
    integrated via ``opp_reach`` (dict opp-physical-card -> reach weight, carrying
    the deal/board chance priors and the opponent's blueprint reach). This is a
    TRUE best response (no clairvoyance), so a Nash strategy has exploitability ~0.
    (The previous implementation maximised per-deal with both cards known — a
    clairvoyant BR that inflated and even increased the reported number.)"""
    cur = r1 if rnd == 1 else r2
    is_br = (len(cur) % 2 == br_player)
    acts = _legal_for(cur)
    facing = cur.endswith("r")

    if is_br:
        best = None
        for a in acts:
            v = _br_apply(blueprint, br_player, br_card, rnd, r1, r2, board,
                          pot_br, pot_opp, opp_reach, a, True, facing)
            best = v if best is None or v > best else best
        return best if best is not None else 0.0

    total = 0.0
    board_rank = _DECK[board] if board is not None else 0
    for ai, a in enumerate(acts):
        new_reach = {}
        for oc, w in opp_reach.items():
            p = _opp_probs(blueprint, _DECK[oc], board_rank, rnd, r1, r2, acts)[ai]
            if p > 0.0:
                new_reach[oc] = w * p
        if new_reach:
            total += _br_apply(blueprint, br_player, br_card, rnd, r1, r2, board,
                               pot_br, pot_opp, new_reach, a, False, facing)
    return total


def _br_apply(blueprint, br_player, br_card, rnd, r1, r2, board,
              pot_br, pot_opp, opp_reach, a, is_br, facing):
    nr1, nr2 = (r1 + a, r2) if rnd == 1 else (r1, r2 + a)

    if a == _FOLD:
        wsum = sum(opp_reach.values())
        return (-float(pot_br) if is_br else float(pot_opp)) * wsum

    if a == _RAISE:
        amt = _RAISE_SIZE[rnd]
        if is_br:
            diff = (pot_opp - pot_br) if facing else 0
            return _br_value(blueprint, br_player, br_card, rnd, nr1, nr2, board,
                             pot_br + diff + amt, pot_opp, opp_reach)
        diff = (pot_br - pot_opp) if facing else 0
        return _br_value(blueprint, br_player, br_card, rnd, nr1, nr2, board,
                         pot_br, pot_opp + diff + amt, opp_reach)

    # check / call
    if facing:
        hi = max(pot_br, pot_opp)
        return _br_close(blueprint, br_player, br_card, rnd, nr1, nr2, board, hi, hi, opp_reach)
    pre = r1 if rnd == 1 else r2
    if len(pre) % 2 == 0:
        # first player checked, other still to act -> not closed
        return _br_value(blueprint, br_player, br_card, rnd, nr1, nr2, board,
                         pot_br, pot_opp, opp_reach)
    # both checked -> close
    return _br_close(blueprint, br_player, br_card, rnd, nr1, nr2, board,
                     pot_br, pot_opp, opp_reach)


def _br_close(blueprint, br_player, br_card, rnd, r1, r2, board,
              pot_br, pot_opp, opp_reach):
    if rnd == 1:
        # reveal the public card uniformly over the 4 cards not held by either
        # player (per opponent card, the board cannot be a card someone holds).
        total = 0.0
        for b in range(len(_DECK)):
            if b == br_card:
                continue
            sub = {oc: w * 0.25 for oc, w in opp_reach.items() if oc != b}
            if sub:
                total += _br_value(blueprint, br_player, br_card, 2, r1, "", b,
                                   pot_br, pot_opp, sub)
        return total
    # showdown
    total = 0.0
    for oc, w in opp_reach.items():
        res = _showdown_winner(_DECK[br_card], _DECK[oc], _DECK[board])  # 1 br wins, -1 opp wins
        total += w * (pot_opp if res == 1 else (-pot_br if res == -1 else 0.0))
    return total


def compute_exploitability(blueprint: dict[str, dict[str, float]]) -> float:
    """True exploitability (chips/game): summed over both players of their
    information-set best-response EV against ``blueprint``. ~0 at Nash; ~4.75 for
    uniform-random. Uses a proper (non-clairvoyant) best response."""
    total = 0.0
    n_cards = len(_DECK)
    for br_player in (0, 1):
        for br_card in range(n_cards):
            opp_reach = {c: 1.0 / 5.0 for c in range(n_cards) if c != br_card}
            total += _br_value(blueprint, br_player, br_card, 1, "", "", None,
                               _ANTE, _ANTE, opp_reach)
    return total / 6.0


# ---------------------------------------------------------------------------
# Blueprint build / load / cache
# ---------------------------------------------------------------------------

def build_blueprint(iterations: int = _TRAIN_ITERATIONS) -> dict[str, dict[str, float]]:
    solver = _LeducCFR()
    solver.train(iterations)
    return solver.blueprint()


def _save_cache(blueprint: dict) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_PATH, "wb") as fh:
        pickle.dump({"version": 2, "iterations": _TRAIN_ITERATIONS,
                     "wl": _LP_WL, "blueprint": blueprint}, fh)


def _load_cache() -> "dict | None":
    if not os.path.exists(_CACHE_PATH):
        return None
    try:
        with open(_CACHE_PATH, "rb") as fh:
            data = pickle.load(fh)
        # version 2 caches carry the payoff mode ('wl'); a mismatch (or a stale v1 chips cache) is
        # rejected so a WIN/LOSS run never loads a chips blueprint or vice-versa.
        if isinstance(data, dict) and data.get("version") == 2 and data.get("wl") == _LP_WL:
            return data["blueprint"]
    except Exception:
        return None
    return None


def get_blueprint(force_rebuild: bool = False) -> dict[str, dict[str, float]]:
    """Load the cached blueprint, building (and caching) it once if absent."""
    if not force_rebuild:
        bp = _load_cache()
        if bp is not None:
            return bp
    bp = build_blueprint()
    try:
        _save_cache(bp)
    except Exception:
        pass
    return bp


# The CFR+ blueprint IS a near-Nash strategy (verified with the correct
# information-set best response above: true exploitability ~0.05 chips/game at
# 1000 iters and falling, vs ~4.75 for uniform-random). Being near-Nash, it is
# robust against ANY opponent — including a strong trained model in PvP — which
# best-responding to a fixed weak opponent is not. It is loaded lazily (and
# cached to disk) on first use so importing this module stays cheap; the
# hand-strength _policy_abstract remains as a fallback for unmapped info sets.
_BLUEPRINT: "dict[str, dict[str, float]] | None" = None


def _bp() -> dict:
    global _BLUEPRINT
    if _BLUEPRINT is None:
        try:
            _BLUEPRINT = get_blueprint()
        except Exception:
            _BLUEPRINT = {}
    return _BLUEPRINT


# ---------------------------------------------------------------------------
# Observation -> infoset key reconstruction
# ---------------------------------------------------------------------------

def _betting_to_hist(betting: list[str]) -> str:
    """Map a parsed round betting list to the abstract-action history string.

    Observation labels: "Raise" -> 'r'; "Call"/"Check" -> 'c'.  Folds end the
    game so a live decision history never contains them.
    """
    out = []
    for a in betting:
        al = a.strip().lower()
        if "raise" in al or "bet" in al:
            out.append(_RAISE)
        elif "call" in al or "check" in al:
            out.append(_CHECK_CALL)
        # ignore anything else
    return "".join(out)


def _label_to_abstract(label: str) -> "str | None":
    ll = label.lower()
    if "fold" in ll:
        return _FOLD
    if "raise" in ll or "bet" in ll:
        return _RAISE
    if "call" in ll or "check" in ll:
        return _CHECK_CALL
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _hand_strength(private_rank: int, public_rank: int, rnd: int) -> int:
    """Coarse Leduc hand strength: 2=strong, 1=medium, 0=weak.

    Ranks: K=3, Q=2, J=1. Pairing the public board is the nuts. On round 2 a
    board card that outranks an unpaired hand downgrades it to weak.
    """
    if rnd == 2 and public_rank and private_rank == public_rank:
        return 2  # pair with the board — strongest holding
    if rnd == 2 and public_rank and public_rank > private_rank:
        return 0  # board outranks us, no pair — weak
    if private_rank >= 3:   # King
        return 2
    if private_rank == 2:   # Queen
        return 1
    return 0                # Jack


def _policy_abstract(private_rank: int, public_rank: int, rnd: int,
                     legal_abstracts: list[str], rng) -> str:
    """EV-rational Leduc action over legal abstract actions {fold 'f',
    check/call 'c', raise 'r'}. Strong hands bet/raise and never fold; weak
    hands fold to a bet and check when free; medium hands mostly call with
    occasional raises. Light randomization avoids a pure (exploitable) strategy.
    Always returns a legal abstract action."""
    s = _hand_strength(private_rank, public_rank, rnd)
    facing_bet = _FOLD in legal_abstracts
    has_raise = _RAISE in legal_abstracts
    has_call = _CHECK_CALL in legal_abstracts

    if s == 2:  # strong — raise, never fold
        if has_raise and rng.random() < 0.8:
            return _RAISE
        return _CHECK_CALL if has_call else (_RAISE if has_raise else legal_abstracts[0])
    if s == 1:  # medium — mostly call/check, occasional raise, fold only rarely
        if has_raise and rng.random() < 0.2:
            return _RAISE
        if has_call:
            return _CHECK_CALL
        return _FOLD if _FOLD in legal_abstracts else legal_abstracts[0]
    # weak — fold to a bet, check when free, rare bluff-raise
    if facing_bet:
        if rng.random() < 0.85:
            return _FOLD
        return _CHECK_CALL if has_call else _FOLD
    if has_raise and rng.random() < 0.1:
        return _RAISE
    return _CHECK_CALL if has_call else legal_abstracts[0]


# RANK-9 deterministic learnable teacher (default OFF via LEDUC_BR). Distils the MODAL near-Nash
# action — the deterministic argmax of the CFR+ blueprint mix at the reached infoset — instead of
# SAMPLING it, so the same infoset always maps to one action_id (a sampled mix teaches the same state
# different actions across rows -> a blurred, higher-loss target the 7B/8B student cannot distil). The
# near-Nash blueprint (the "field") and the WIN/LOSS never-fold override are reused unchanged; the prior
# sampling path is the A/B backstop when the flag is off. Winner IDEA: winner-pos1/scripts/our_envs/
# leduc_poker_trajectories.py + leduc_cfr.py (CFR+ field); ours is the deterministic collapse of it.
_LAST_REASON = ""


def _set_reason(msg: str) -> None:
    global _LAST_REASON
    _LAST_REASON = msg


def last_reason() -> str:
    """Human-readable note on the last LEDUC_BR decision (debug only)."""
    return _LAST_REASON


def _leduc_br_on() -> bool:
    v = os.getenv("LEDUC_BR")
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Teacher mode selection (leduc v2, 2026-09-30)
# ---------------------------------------------------------------------------
#
# LEDUC_TEACHER picks the default-shipping label out of three modes, all standard-library and all
# already exercised by choose_action below:
#   'aggro' (DEFAULT) -- MAX_AGGRO: deterministic, card-independent, raise-whenever-legal/never-fold.
#   'cfr'             -- today's sampler (pre-leduc-v2 default): sample the near-Nash CFR+ blueprint's
#                        mixed strategy, with the WIN/LOSS never-fold override retained.
#   'br'              -- the RANK-9 deterministic modal near-Nash action (formerly the LEDUC_BR
#                        opt-in), falling through to 'cfr' sampling for any unmapped info set.
# ROLLBACK: set LEDUC_TEACHER=cfr (or flip _LEDUC_TEACHER_DEFAULT back to "cfr") for a one-line
# revert to the pre-leduc-v2 default; both are fully unchanged code paths, not removed.


def _aggro_abstract(legal_abstracts: "list[str]") -> str:
    """MAX_AGGRO abstract action (leduc v2, 2026-09-30): raise whenever legal, else check/call, else
    the only legal action. Deterministic and card-independent -- ignores rank/board/history
    entirely. Exact-enumeration result (see test_leduc_teacher.py's exact W/L enumerator, which
    walks the full Leduc tree against a panel of opponent profiles): this equals the per-opponent
    best-response ceiling within 0.1pp against every measured profile (chips-Nash folder,
    rule-based folder, 20%-weak-fold, never-folder) and is never below 50%, because under WIN/LOSS
    sign-of-return scoring a fold is a certain loss while check/call keeps a showdown chance --
    fold is weakly dominated at every reached info set whenever check/call is available."""
    if _RAISE in legal_abstracts:
        return _RAISE
    if _CHECK_CALL in legal_abstracts:
        return _CHECK_CALL
    return legal_abstracts[0]


_LEDUC_TEACHER_MODES = ("aggro", "cfr", "br")
_LEDUC_TEACHER_DEFAULT = "aggro"


def _teacher_mode() -> str:
    """Resolve the active Leduc teacher mode. LEDUC_TEACHER (case-insensitive) selects one of
    _LEDUC_TEACHER_MODES explicitly and wins over everything else; an unrecognized value falls
    back to the default rather than raising, so a typo can never crash generation. When
    LEDUC_TEACHER is unset, the legacy LEDUC_BR opt-in flag (pre-leduc-v2) is honored for backward
    compatibility; otherwise _LEDUC_TEACHER_DEFAULT ('aggro') applies."""
    raw = os.getenv("LEDUC_TEACHER", "").strip().lower()
    if raw in _LEDUC_TEACHER_MODES:
        return raw
    if raw:
        return _LEDUC_TEACHER_DEFAULT
    if _leduc_br_on():
        return "br"
    return _LEDUC_TEACHER_DEFAULT


def _sample_from_blueprint(bp: dict, private_rank: int, public_rank: int, r1_hist: str,
                           r2_hist: str, legal_abstracts: "list[str]",
                           abstract_to_id: "dict[str, int]", rng, never_fold: bool) -> "int | None":
    """Sample one legal action id from ``bp``'s mixed strategy at this info set, optionally
    zeroing fold-mass first (WIN/LOSS never-fold override). Returns None when the info set is
    unmapped or all mass would be zeroed, so the caller can fall back. Exact logic choose_action
    used inline pre-leduc-v2; extracted so it is shared by the 'cfr' teacher mode and pool_action (the
    hero-seat opponent pool consumed by pvp_selfplay.play_leduc_hero_game)."""
    key = _infoset_key(private_rank, public_rank, r1_hist, r2_hist)
    dist = bp.get(key)
    if not dist:
        return None
    weights = [dist.get(ab, 0.0) for ab in legal_abstracts]
    if never_fold and _CHECK_CALL in legal_abstracts:
        weights = [0.0 if ab == _FOLD else w for ab, w in zip(legal_abstracts, weights)]
    if sum(weights) <= 0.0:
        return None
    return abstract_to_id[rng.choices(legal_abstracts, weights=weights, k=1)[0]]


# ---------------------------------------------------------------------------
# Chips-objective pool blueprint (leduc v2) -- diversity source ONLY for the hero-seat opponent pool.
# ---------------------------------------------------------------------------
#
# Independent of the module's WIN/LOSS default (_LP_WL / LEDUC_WIN_LOSS): a genuine chip-optimal
# near-Nash strategy, which folds weak hands at the chips-EV-correct rate -- exactly the
# "chips-Nash folder" opponent profile the leduc v2 research measured (aggro extracts ~70.6% from it, see
# test_leduc_teacher.py). Never affects the hero's own label (choose_action always reads the
# module's WIN/LOSS-mode blueprint via _bp(), or is card-independent under 'aggro'). Cached to a
# file distinct from the WIN/LOSS blueprint's _CACHE_PATH so the two can never collide/clobber.
_CHIPS_POOL_CACHE_PATH = os.path.join(_CACHE_DIR, "leduc_poker_chips_pool.pkl")
_CHIPS_POOL_BLUEPRINT: "dict | None" = None


def get_chips_pool_blueprint(iterations: int = _TRAIN_ITERATIONS, force_rebuild: bool = False) -> dict:
    """Load (or build + cache) the CHIPS-objective near-Nash blueprint. Standard library only. A
    fresh solve at the default 2000 iterations is a cold CFR+ run over the full tree (measured
    ~87s on a single core) -- generate_trajectories._prewarm_env solves and caches it once in the
    parent process before workers fork, so no forked worker pays this cost; a direct caller that
    skips prewarming still gets a correct (if slower, first-call-only) result."""
    global _CHIPS_POOL_BLUEPRINT
    if not force_rebuild and _CHIPS_POOL_BLUEPRINT is not None:
        return _CHIPS_POOL_BLUEPRINT
    if not force_rebuild and os.path.exists(_CHIPS_POOL_CACHE_PATH):
        try:
            with open(_CHIPS_POOL_CACHE_PATH, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict) and data.get("version") == 1 and data.get("iterations") == iterations:
                _CHIPS_POOL_BLUEPRINT = data["blueprint"]
                return _CHIPS_POOL_BLUEPRINT
        except Exception:
            pass
    global _LP_WL
    prev = _LP_WL
    _LP_WL = False  # solve under the CHIPS objective regardless of the module's WIN/LOSS default
    try:
        bp = build_blueprint(iterations)
    finally:
        _LP_WL = prev
    _CHIPS_POOL_BLUEPRINT = bp
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        # Write-then-rename: several forked workers can race to (re)build this cache when it was
        # not prewarmed in the parent; a direct write can be torn if two workers write it at once,
        # and a torn pickle would raise on the next load. os.replace is atomic on the same filesystem.
        tmp_path = f"{_CHIPS_POOL_CACHE_PATH}.tmp{os.getpid()}"
        with open(tmp_path, "wb") as fh:
            pickle.dump({"version": 1, "iterations": iterations, "blueprint": bp}, fh)
        os.replace(tmp_path, _CHIPS_POOL_CACHE_PATH)
    except Exception:
        pass
    return bp


# The hero-pool infoset fields (round, public rank, per-round history) are parsed DIRECTLY from
# the eval-format observation text pvp_selfplay._format_leduc emits ("Round: N/2", "Public card:
# X", "Round 1 actions: ...", "Round 2 actions: ..."), NOT via leduc_poker_env.parse_game_state:
# that function matches a DIFFERENT text shape ("Current round: N/2", "Round N betting: ..."),
# which never appears in this observation, so it silently defaults every decision to round 1 with
# empty histories (found in review). private_card_rank and the id<->label
# mapping are unaffected -- those come from parse_game_state's "Your card:"/"N -> label" regexes,
# which DO match this text -- so pool_action keeps using parse_game_state for those two fields
# only and this parser for round/public/history.
_ROUND_RE = re.compile(r"Round:\s*(\d+)/\d+")
_PUBLIC_CARD_RE = re.compile(r"Public card:\s*(\S)")
_ROUND_ACTIONS_RE = {
    1: re.compile(r"Round 1 actions:\s*(.+)"),
    2: re.compile(r"Round 2 actions:\s*(.+)"),
}


def _parse_pool_round_history(observation: str) -> "tuple[int, int, str, str] | None":
    """Extract (round, public_rank, r1_hist, r2_hist) from the eval-format observation text.
    Returns None if the round cannot be found (unparseable observation) so the caller can degrade
    to a round-1-shaped fallback instead of guessing."""
    m = _ROUND_RE.search(observation)
    if not m:
        return None
    round_ = int(m.group(1))

    public_rank = 0
    pm = _PUBLIC_CARD_RE.search(observation)
    if pm:
        public_rank = _CARD_RANK.get(pm.group(1), 0)

    def _round_hist(rnd: int) -> str:
        rm = _ROUND_ACTIONS_RE[rnd].search(observation)
        if not rm:
            return ""
        raw = rm.group(1).strip()
        if not raw or raw == "(none)":
            return ""
        labels = [a.strip() for a in raw.split(",")]
        return _betting_to_hist(labels)

    r1_hist = _round_hist(1)
    r2_hist = _round_hist(2) if round_ == 2 else ""
    return round_, public_rank, r1_hist, r2_hist


def pool_action(
    kind: str,
    observation: str,
    player_id: int,
    legal_action_ids: "list[int]",
    rng: "random.Random | None" = None,
) -> int:
    """Opponent-pool action for the hero-seat generator (play_leduc_hero_game, pvp_selfplay.py,
    leduc v2): a DIVERSE, non-mirror opponent instead of another copy of the hero's own label.

    kind:
      'wl_cfr'    -- sample the WIN/LOSS-objective blueprint (the same near-Nash field the 'cfr'
                     teacher mode samples, via _bp()) WITHOUT the never-fold override. Its fold mass
                     sits at only a few reached info sets, so in practice it almost never folds
                     (measured 0/1058 facing a bet); the folding opponents are 'chips_cfr' and 'random'.
      'chips_cfr' -- sample the CHIPS-objective blueprint (get_chips_pool_blueprint()) -- a real
                     chip-optimal folder.
      'random'    -- uniform legal.

    Never the hero's own label path (see choose_action). Always returns one element of
    ``legal_action_ids``; never raises."""
    r = rng if rng is not None else random
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    if kind == "random":
        return r.choice(legal_action_ids)

    # private_card_rank and the legal-id<->label mapping come from parse_game_state (its
    # "Your card:"/"N -> label" regexes DO match this observation text); round/public
    # card/per-round history come from _parse_pool_round_history instead (see its docstring for
    # why parse_game_state's own round/history fields cannot be used here).
    gs = parse_game_state(observation)
    id_to_abstract: dict[int, str] = {}
    if gs is not None:
        for aid, label in gs.legal_actions.items():
            ab = _label_to_abstract(label)
            if ab is not None and aid in legal_action_ids:
                id_to_abstract[aid] = ab
    abstract_to_id = {ab: aid for aid, ab in id_to_abstract.items()}
    if gs is None or not abstract_to_id:
        return r.choice(legal_action_ids)

    parsed = _parse_pool_round_history(observation)
    round_, public_rank, r1_hist, r2_hist = parsed if parsed is not None else (1, 0, "", "")
    private_rank = gs.private_card_rank
    legal_abstracts = list(abstract_to_id.keys())

    try:
        bp = get_chips_pool_blueprint() if kind == "chips_cfr" else _bp()
        action_id = _sample_from_blueprint(bp, private_rank, public_rank, r1_hist, r2_hist,
                                           legal_abstracts, abstract_to_id, r, never_fold=False)
        if action_id is not None:
            return action_id
    except Exception:  # noqa: BLE001
        pass

    chosen_ab = _policy_abstract(private_rank, public_rank, round_, legal_abstracts, r)
    if chosen_ab not in abstract_to_id:
        chosen_ab = legal_abstracts[0]
    return abstract_to_id[chosen_ab]


def _br_argmax_action(private_rank: int, public_rank: int, r1_hist: str, r2_hist: str,
                      legal_abstracts: list[str], abstract_to_id: dict[str, int]) -> "int | None":
    """Deterministic hand-win best response to the near-Nash field: the modal blueprint action at this
    infoset. Zeroes fold-mass under WIN/LOSS when check/call is legal (folding is a certain loss), then
    argmaxes the blueprint probability with a stable aggression tie-break (raise > call > fold).
    Returns the matching legal action id, or None to defer to the sampler. Never raises."""
    bp = _bp()
    if not bp:
        return None
    key = _infoset_key(private_rank, public_rank, r1_hist, r2_hist)
    dist = bp.get(key)
    if not dist:
        return None
    weights = {ab: dist.get(ab, 0.0) for ab in legal_abstracts}
    if _LP_WL and _CHECK_CALL in legal_abstracts:
        weights[_FOLD] = 0.0
    if sum(weights.values()) <= 0.0:
        return None
    order = {_RAISE: 2, _CHECK_CALL: 1, _FOLD: 0}
    best_ab = max(legal_abstracts, key=lambda ab: (weights[ab], order.get(ab, 0)))
    if best_ab not in abstract_to_id:
        return None
    _set_reason(f"BR argmax near-Nash: {best_ab} (p={weights[best_ab]:.3f}) infoset={key}")
    return abstract_to_id[best_ab]


def choose_action(
    observation: str,
    player_id: int,
    legal_action_ids: list[int],
    rng: "random.Random | None" = None,
) -> int:
    """Return the hero's labeled action for this Leduc decision.

    Dispatches on ``_teacher_mode()``: DEFAULT 'aggro' returns the deterministic
    MAX_AGGRO label (raise whenever legal, else check/call -- card-independent,
    never folds); 'br' returns the deterministic modal near-Nash action; 'cfr'
    samples the near-Nash CFR+ blueprint's mixed strategy (preserving
    bluff/check-raise frequencies), with the WIN/LOSS never-fold override
    retained. Falls back to the EV-rational hand-strength policy
    (_policy_abstract) for any CFR-blueprint-unmapped info set, and to
    uniform-legal if the observation cannot be parsed. Always returns one
    element of ``legal_action_ids``.
    """
    r = rng if rng is not None else random
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")

    gs = parse_game_state(observation)

    id_to_abstract: dict[int, str] = {}
    if gs is not None:
        for aid, label in gs.legal_actions.items():
            ab = _label_to_abstract(label)
            if ab is not None and aid in legal_action_ids:
                id_to_abstract[aid] = ab
    abstract_to_id = {ab: aid for aid, ab in id_to_abstract.items()}

    # If we cannot parse the state or map labels, fall back to uniform-legal.
    if gs is None or not abstract_to_id:
        return r.choice(legal_action_ids)

    private_rank = gs.private_card_rank
    public_rank = gs.public_card_rank if (gs.round == 2 and gs.public_card_rank) else 0
    r1_hist = _betting_to_hist(gs.r1_betting)
    r2_hist = _betting_to_hist(gs.r2_betting) if gs.round == 2 else ""
    legal_abstracts = list(abstract_to_id.keys())

    mode = _teacher_mode()

    # DEFAULT: MAX_AGGRO -- deterministic, card-independent (see _aggro_abstract's docstring for
    # the exact-enumeration ceiling this achieves against every measured opponent profile).
    if mode == "aggro":
        ab = _aggro_abstract(legal_abstracts)
        _set_reason(f"MAX_AGGRO: {ab} infoset={_infoset_key(private_rank, public_rank, r1_hist, r2_hist)}")
        return abstract_to_id.get(ab, legal_action_ids[0])

    # Backstop 'br': RANK-9 deterministic modal near-Nash action (formerly the LEDUC_BR opt-in).
    # Falls through to the 'cfr' sampler below for any unmapped info set, unchanged from before.
    if mode == "br":
        try:
            br_id = _br_argmax_action(private_rank, public_rank, r1_hist, r2_hist,
                                      legal_abstracts, abstract_to_id)
            if br_id is not None and br_id in legal_action_ids:
                return br_id
        except Exception:  # noqa: BLE001
            pass

    # Backstop 'cfr' (today's pre-leduc-v2 default, and mode=='br''s fallthrough): sample the
    # near-Nash blueprint for this info set. NEVER-FOLD override retained: under WIN/LOSS scoring,
    # folding when a check/call is legal is a guaranteed loss vs a showdown chance, so zero any
    # residual fold-mass in the sampled label so a stray fold never teaches the student to
    # auto-concede. Only when 'c' is actually available.
    action_id = _sample_from_blueprint(_bp(), private_rank, public_rank, r1_hist, r2_hist,
                                       legal_abstracts, abstract_to_id, r, never_fold=_LP_WL)
    if action_id is not None:
        return action_id

    # Fallback: EV-rational hand-strength policy for unmapped info sets.
    chosen_ab = _policy_abstract(private_rank, public_rank, gs.round, legal_abstracts, r)
    if chosen_ab not in abstract_to_id:
        chosen_ab = legal_abstracts[0]
    return abstract_to_id[chosen_ab]
