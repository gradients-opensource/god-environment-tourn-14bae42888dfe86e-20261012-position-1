"""Strong Gin Rummy expert teacher via Perfect-Information Monte Carlo (PIMC).

This module is a *self-contained* expert policy used to generate SFT data
for the Gin Rummy game-training harness.  It does NOT import meld / deadwood
logic from ``gin_rummy_trajectories.py`` (per the design brief): meld
detection and the optimal-deadwood DP are reimplemented here so the engine
can stand on its own.

Public contract
---------------
    choose_action(observation: str,
                  player_id: int,
                  legal_action_ids: list[int],
                  rng=None) -> int

Returns exactly one element of ``legal_action_ids``.

Design summary
--------------
Gin Rummy has far too many information states for tabular solving, so we use
PIMC (a.k.a. determinized search):

  1. Parse our own hand + all *publicly visible* cards (upcard + discard pile)
     from the observation text.
  2. The remaining 52 - (visible) cards form the *unseen pool* = opponent hand
     + stock, in unknown order.
  3. Sample ``N_WORLDS`` deterministic "worlds": a concrete opponent hand and
     stock order drawn uniformly from the unseen pool.
  4. For each legal action, simulate a short greedy continuation in each world
     using a fast deadwood-minimisation heuristic with *correct* knock/gin
     rules (knock only when deadwood <= knock threshold, usually 10; gin = 0
     deadwood, worth a bonus).  Average the action's value across all worlds.
  5. Pick the best-valued legal action (optionally softmax-sample for mild
     diversity when an rng is supplied).

Only the genuinely hidden-information decisions (which card to *draw*, which
card to *discard*) actually need world sampling.  Decisions that are fully
determined by our own hand -- whether to knock / go gin, and which cards to
lay off -- are resolved deterministically from the deadwood DP, which is both
faster and exactly correct.

Latency: with ``N_WORLDS`` worlds and short greedy rollouts the per-move cost
is a few milliseconds of pure-Python work, comfortably fast enough to generate
many games offline.  See ``N_WORLDS`` / ``ROLLOUT_DEPTH`` below to tune.

Standard library only.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Optional


# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
RANK_IDX = {r: i for i, r in enumerate(RANK_ORDER)}
SUITS = ('s', 'h', 'd', 'c')

FULL_DECK = tuple(r + s for s in SUITS for r in RANK_ORDER)  # 52 cards

# Tuning knobs ---------------------------------------------------------------
N_WORLDS = 24          # determinized worlds sampled per hidden-info decision
ROLLOUT_DEPTH = 6      # greedy plies simulated per world when scoring an action
GIN_BONUS = 25         # standard gin bonus (points)
KNOCK_DEFAULT = 10     # default deadwood threshold to knock when unparsed
SOFTMAX_TEMP = 0.0     # >0 enables mild softmax sampling over action values


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def _norm_card(card: str) -> str:
    """Normalise a 2-char card to canonical 'Rs' form (upper rank, lower suit)."""
    return card[0].upper() + card[1].lower()


def card_value(card: str) -> int:
    if len(card) < 2:
        return 0
    return CARD_VALUES.get(card[0].upper(), 0)


def get_rank(card: str) -> str:
    return card[0].upper()


def get_suit(card: str) -> str:
    return card[1].lower()


# ---------------------------------------------------------------------------
# Meld detection + optimal-deadwood DP (self-contained)
# ---------------------------------------------------------------------------

def find_all_melds(hand: tuple[str, ...] | list[str]) -> list[frozenset]:
    """Enumerate every valid meld (set of 3-4 same rank, run of >=3 same suit)
    that can be formed from ``hand``.  Returns frozensets of card strings."""
    melds: list[frozenset] = []

    # Sets: same rank, size 3 or 4.
    rank_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        rank_groups[get_rank(c)].append(c)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            # All size-3 subsets (so the DP can pick the best of two 3-of-a-kind
            # when a 4th card might be more valuable elsewhere) + the full set.
            if len(cards) == 3:
                melds.append(frozenset(cards))
            else:  # exactly 4
                melds.append(frozenset(cards))
                for i in range(len(cards)):
                    melds.append(frozenset(cards[:i] + cards[i + 1:]))

    # Runs: same suit, consecutive ranks, length >= 3.
    suit_groups: dict[str, list[str]] = defaultdict(list)
    for c in hand:
        suit_groups[get_suit(c)].append(c)
    for cards in suit_groups.values():
        ordered = sorted(set(cards), key=lambda c: RANK_IDX[get_rank(c)])
        i = 0
        while i < len(ordered):
            run = [ordered[i]]
            j = i + 1
            while j < len(ordered) and \
                    RANK_IDX[get_rank(ordered[j])] == RANK_IDX[get_rank(run[-1])] + 1:
                run.append(ordered[j])
                j += 1
            # all contiguous sub-runs of length >= 3
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def _deadwood_dp(hand_key: tuple[str, ...]) -> tuple[int, frozenset]:
    """Return (optimal_deadwood, set_of_melded_cards) for ``hand_key``.

    Bitmask DP over which cards are already consumed by chosen melds; melds are
    non-overlapping.  We pick the subset of disjoint melds that minimises the
    remaining (unmelded) card-value sum.
    """
    hand = list(hand_key)
    n = len(hand)
    if n == 0:
        return 0, frozenset()

    idx = {c: i for i, c in enumerate(hand)}
    vals = [card_value(c) for c in hand]

    meld_masks: list[int] = []
    for meld in find_all_melds(hand):
        mask = 0
        ok = True
        for c in meld:
            if c not in idx:
                ok = False
                break
            mask |= (1 << idx[c])
        if ok:
            meld_masks.append(mask)

    memo: dict[int, tuple[int, int]] = {}

    def dp(used: int) -> tuple[int, int]:
        """Returns (best_deadwood, melded_mask) for cards not yet `used`."""
        cached = memo.get(used)
        if cached is not None:
            return cached
        # baseline: meld nothing further -> all unused cards are deadwood
        best_dw = sum(vals[i] for i in range(n) if not (used >> i) & 1)
        best_melded = 0
        for mm in meld_masks:
            if mm & used:
                continue
            child_dw, child_melded = dp(used | mm)
            if child_dw < best_dw:
                best_dw = child_dw
                best_melded = child_melded | mm
        result = (best_dw, best_melded)
        memo[used] = result
        return result

    best_dw, melded_mask = dp(0)
    melded = frozenset(hand[i] for i in range(n) if (melded_mask >> i) & 1)
    return best_dw, melded


@lru_cache(maxsize=20000)
def _deadwood_dp_cached(hand_key: tuple[str, ...]) -> tuple[int, frozenset]:
    # Order-independent cache key.
    return _deadwood_dp(hand_key)


# Optimal deadwood over 52-bit card masks: the same value as _deadwood_dp at ~10x less cost.
# compute_optimal_deadwood is ~90% of a PIMC move (24 worlds x 6 look-ahead plies x 11 discards), so this
# multiplies gin rows per generation second without changing a label; gin's unused slice rolls over to
# the other envs. f(h) = min(value(c) + f(h - c), f(h - m) for each meld m containing c), c = lowest card
# of h, memoised on the mask. _FAST_DEADWOOD = False restores the meld-enumeration DP for values too.
_FAST_DEADWOOD = True
_CARD_BIT = {c: i for i, c in enumerate(FULL_DECK)}
_BIT_VALUE = [card_value(c) for c in FULL_DECK]
_DW_MEMO_MAX = 200_000
_DW_MEMO: dict[int, int] = {0: 0}


def _melds_by_bit() -> list[list[int]]:
    """For each card bit, the masks of every set (3 or 4 of a rank) and run (>= 3 in a suit) containing it."""
    masks: list[int] = []
    for rank in RANK_ORDER:
        bits = [_CARD_BIT[rank + s] for s in SUITS]
        masks.append(sum(1 << b for b in bits))
        masks.extend(sum(1 << b for b in bits if b != skip) for skip in bits)
    for suit in SUITS:
        for start in range(len(RANK_ORDER)):
            for end in range(start + 3, len(RANK_ORDER) + 1):
                masks.append(sum(1 << _CARD_BIT[RANK_ORDER[i] + suit] for i in range(start, end)))
    return [[m for m in masks if (m >> b) & 1] for b in range(len(FULL_DECK))]


_MELDS_BY_BIT = _melds_by_bit()


def _deadwood_mask(h: int) -> int:
    cached = _DW_MEMO.get(h)
    if cached is not None:
        return cached
    low = h & -h
    bit = low.bit_length() - 1
    best = _BIT_VALUE[bit] + _deadwood_mask(h ^ low)
    for m in _MELDS_BY_BIT[bit]:
        if m & h == m:
            best = min(best, _deadwood_mask(h ^ m))
    if len(_DW_MEMO) >= _DW_MEMO_MAX:
        _DW_MEMO.clear()
        _DW_MEMO[0] = 0
    _DW_MEMO[h] = best
    return best


def _hand_mask(hand) -> Optional[int]:
    """52-bit mask of ``hand``, or None for a duplicate or unknown card (left to the DP)."""
    h = 0
    for c in hand:
        bit = _CARD_BIT.get(c)
        if bit is None:
            bit = _CARD_BIT.get(_norm_card(c)) if len(c) == 2 else None
        if bit is None or (h >> bit) & 1:
            return None
        h |= 1 << bit
    return h


def compute_optimal_deadwood(hand: list[str] | tuple[str, ...]) -> int:
    """Minimum achievable deadwood (unmelded card-value sum) for ``hand``."""
    if not hand:
        return 0
    if _FAST_DEADWOOD:
        mask = _hand_mask(hand)
        if mask is not None:
            return _deadwood_mask(mask)
    key = tuple(sorted(_norm_card(c) for c in hand))
    return _deadwood_dp_cached(key)[0]


def get_optimal_meld_cards(hand: list[str] | tuple[str, ...]) -> set:
    """Set of cards that participate in the deadwood-optimal melding."""
    if not hand:
        return set()
    key = tuple(sorted(_norm_card(c) for c in hand))
    return set(_deadwood_dp_cached(key)[1])


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

_RE_PLAYER = re.compile(r'You are Player (\d+)')
_RE_PHASE = re.compile(r'Phase:\s*(\w+)')
_RE_UPCARD = re.compile(r'Upcard:\s*(\w+)')
_RE_DEADWOOD = re.compile(r'Deadwood=(\d+)')
_RE_KNOCK_CARD = re.compile(r'Knock card:\s*(\d+)')
_RE_STOCK = re.compile(r'Stock size:\s*(\d+)')
_RE_DISCARD_PILE = re.compile(r'Discard pile[:\s]+([^\n]+)', re.IGNORECASE)
_RE_CARD = re.compile(r'([A2-9TJQK][shdc])')
_RE_CARD_EXACT = re.compile(r'^([A2-9TJQK][shdc])$')
_RE_MELD_GROUP = re.compile(r'^([A2-9TJQK][shdc]){2,}$')
# Legal-action line. Accept BOTH envelopes: the self-play format (pvp_selfplay._teacher_obs)
# "N -> LABEL" AND the OLD env-server format "N -> Player: P Action: LABEL". The rigid old-only
# regex matched NOTHING on the self-play format, so parse_legal_actions() returned [] and
# choose_action() silently fell back to legal_action_ids[0] (near-random) on every move — i.e.
# the whole PIMC engine was dead during self-play generation. The optional prefix strips
# "Player: P Action:" when present so the captured LABEL is clean in both formats.
_RE_LEGAL = re.compile(
    r'^\s*(\d+)\s*->\s*(?:Player:\s*\d+\s*Action:\s*)?(.+?)\s*$', re.MULTILINE
)


def parse_phase(obs: str) -> str:
    m = _RE_PHASE.search(obs)
    return m.group(1) if m else ''


def parse_player_id(obs: str) -> int:
    m = _RE_PLAYER.search(obs)
    return int(m.group(1)) if m else 0


def parse_hand(obs: str, player_id: Optional[int] = None) -> list[str]:
    pid = player_id if player_id is not None else parse_player_id(obs)
    section = re.search(
        rf'Player{pid}: Deadwood=\d+.*?\n\+-+\+\n(.*?)\n\+-+\+', obs, re.DOTALL
    )
    if not section:
        return []
    cards: list[str] = []
    for row in section.group(1).strip().split('\n'):
        cards.extend(_norm_card(c) for c in _RE_CARD.findall(row))
    return cards


def parse_upcard(obs: str) -> str:
    m = _RE_UPCARD.search(obs)
    if not m:
        return 'XX'
    up = m.group(1)
    return _norm_card(up) if len(up) == 2 and up != 'XX' else 'XX'


def parse_deadwood(obs: str) -> int:
    m = _RE_DEADWOOD.search(obs)
    return int(m.group(1)) if m else 99


def parse_knock_card(obs: str) -> int:
    m = _RE_KNOCK_CARD.search(obs)
    return int(m.group(1)) if m else KNOCK_DEFAULT


def parse_stock_size(obs: str) -> int:
    m = _RE_STOCK.search(obs)
    return int(m.group(1)) if m else 0


def parse_discard_pile(obs: str, upcard: Optional[str] = None) -> set:
    """All publicly seen discarded cards (includes the current upcard)."""
    cards: set = set()
    m = _RE_DISCARD_PILE.search(obs)
    if m:
        cards.update(_norm_card(c) for c in _RE_CARD.findall(m.group(1)))
    if upcard is None:
        upcard = parse_upcard(obs)
    if upcard and upcard != 'XX' and len(upcard) == 2:
        cards.add(_norm_card(upcard))
    return cards


def parse_legal_actions(obs: str) -> list[tuple[int, str]]:
    """List of (action_id, label) parsed from the 'Legal Actions:' block."""
    out: list[tuple[int, str]] = []
    for aid, label in _RE_LEGAL.findall(obs):
        out.append((int(aid), label.strip()))
    return out


# ---------------------------------------------------------------------------
# Legal-action label classification
# ---------------------------------------------------------------------------

def _label_kind(label: str) -> tuple[str, Optional[str]]:
    """Classify a legal-action label -> (kind, card_or_none).

    kinds: 'draw_upcard', 'draw_stock', 'knock', 'gin', 'pass',
           'discard' (card given), 'meld' (group), 'other'.
    """
    low = label.strip().lower()
    if 'draw' in low and 'upcard' in low:
        return 'draw_upcard', None
    if 'draw' in low and 'stock' in low:
        return 'draw_stock', None
    if low == 'gin':
        return 'gin', None
    if low == 'knock' or low.startswith('knock'):
        return 'knock', None
    if low == 'pass':
        return 'pass', None
    stripped = label.strip()
    cm = _RE_CARD_EXACT.match(stripped)
    if cm:
        return 'discard', _norm_card(cm.group(1))
    # Sometimes labelled "Discard 5s"
    dm = re.match(r'^discard\s+([A2-9TJQK][shdc])$', stripped, re.IGNORECASE)
    if dm:
        return 'discard', _norm_card(dm.group(1))
    if _RE_MELD_GROUP.match(stripped):
        return 'meld', None
    return 'other', None


# ---------------------------------------------------------------------------
# Fast greedy helpers used inside rollouts
# ---------------------------------------------------------------------------

def _best_discard_from_hand(hand: list[str], knock_card: int) -> tuple[str, int]:
    """Greedily choose the discard from an 11-card hand that minimises the
    resulting 10-card deadwood.  Returns (card_to_discard, resulting_deadwood)."""
    best_card = hand[0]
    best_dw = None
    # Evaluate discarding each distinct card.
    seen = set()
    for c in hand:
        if c in seen:
            continue
        seen.add(c)
        remaining = list(hand)
        remaining.remove(c)
        dw = compute_optimal_deadwood(remaining)
        # Tie-break: prefer discarding higher-value, less-connected cards.
        if best_dw is None or dw < best_dw or (
            dw == best_dw and card_value(c) > card_value(best_card)
        ):
            best_dw = dw
            best_card = c
    return best_card, (best_dw if best_dw is not None else 0)


def _draw_helps(hand: list[str], card: str) -> int:
    """Deadwood reduction obtained by adding ``card`` then optimally discarding."""
    if not card or card == 'XX' or len(card) != 2:
        return 0
    base = compute_optimal_deadwood(hand)
    _, after = _best_discard_from_hand(hand + [card], KNOCK_DEFAULT)
    return max(0, base - after)


# ---------------------------------------------------------------------------
# PIMC world sampling
# ---------------------------------------------------------------------------

def _unseen_pool(hand: list[str], seen: set) -> list[str]:
    """Cards neither in our hand nor publicly visible."""
    known = set(_norm_card(c) for c in hand) | set(_norm_card(c) for c in seen)
    return [c for c in FULL_DECK if c not in known]


def _sample_world(pool: list[str], opp_hand_size: int, rng: random.Random
                  ) -> tuple[list[str], list[str]]:
    """Partition the unseen pool into (opponent_hand, stock_order)."""
    shuffled = pool[:]
    rng.shuffle(shuffled)
    opp = shuffled[:opp_hand_size]
    stock = shuffled[opp_hand_size:]
    return opp, stock


# ---------------------------------------------------------------------------
# Action evaluation (PIMC value of each legal action)
# ---------------------------------------------------------------------------

def _rollout_value_after_draw(hand_after_draw: list[str], stock: list[str],
                              knock_card: int) -> float:
    """Score a hand of 11 cards: discard optimally, then estimate prospects.

    Value = -(resulting deadwood) + bonus for being able to knock/gin, plus a
    small look-ahead over the next ``ROLLOUT_DEPTH`` stock draws (greedy improve
    by drawing stock and keeping the best discard each ply).
    """
    discard, dw = _best_discard_from_hand(hand_after_draw, knock_card)
    hand = list(hand_after_draw)
    hand.remove(discard)

    value = -float(dw)
    if dw == 0:
        value += GIN_BONUS
    elif dw <= knock_card:
        # Knockable: reward proportional to how far below threshold.
        value += 10.0 + (knock_card - dw) * 0.5

    # Short greedy look-ahead drawing from the (determinized) stock.
    cur = list(hand)
    cur_dw = dw
    for ply in range(ROLLOUT_DEPTH):
        if ply >= len(stock):
            break
        drawn = stock[ply]
        cand = cur + [drawn]
        d2, ndw = _best_discard_from_hand(cand, knock_card)
        if ndw < cur_dw:
            cur = cand
            cur.remove(d2)
            cur_dw = ndw
            # discount future improvement
            value += (dw - ndw) * (0.6 ** (ply + 1)) * 0.5
        # if no improvement we'd keep the drawn-and-redROPPED card, no change
    return value


def _evaluate_draw_action(kind: str, hand: list[str], upcard: str,
                          worlds: list[tuple[list[str], list[str]]],
                          knock_card: int) -> float:
    """Average PIMC value of a draw action across sampled worlds."""
    total = 0.0
    for opp, stock in worlds:
        if kind == 'draw_upcard':
            if not upcard or upcard == 'XX':
                total += -999.0
                continue
            total += _rollout_value_after_draw(hand + [upcard], stock, knock_card)
        else:  # draw_stock: the top of this world's determinized stock
            if not stock:
                # no stock left in this world; drawing yields nothing useful
                total += _rollout_value_after_draw(hand, stock, knock_card)
                continue
            drawn = stock[0]
            total += _rollout_value_after_draw(hand + [drawn], stock[1:], knock_card)
    return total / max(1, len(worlds))


def _evaluate_discard_action(card: str, hand: list[str],
                             worlds: list[tuple[list[str], list[str]]],
                             knock_card: int) -> float:
    """Average PIMC value of discarding ``card`` (from current >=11 hand)."""
    if card not in hand:
        return -999.0
    remaining = list(hand)
    remaining.remove(card)
    base_dw = compute_optimal_deadwood(remaining)

    total = 0.0
    for _opp, stock in worlds:
        value = -float(base_dw)
        if base_dw == 0:
            value += GIN_BONUS
        elif base_dw <= knock_card:
            value += 10.0 + (knock_card - base_dw) * 0.5
        # short look-ahead on this world's stock
        cur = list(remaining)
        cur_dw = base_dw
        for ply in range(ROLLOUT_DEPTH):
            if ply >= len(stock):
                break
            cand = cur + [stock[ply]]
            d2, ndw = _best_discard_from_hand(cand, knock_card)
            if ndw < cur_dw:
                cur = cand
                cur.remove(d2)
                value += (cur_dw - ndw) * (0.6 ** (ply + 1)) * 0.5
                cur_dw = ndw
        total += value
    return total / max(1, len(worlds))


# ---------------------------------------------------------------------------
# Deterministic handlers (no hidden info needed)
# ---------------------------------------------------------------------------

def _choose_knock_or_gin(legal: list[tuple[int, str]], hand: list[str],
                         deadwood: int, knock_card: int) -> Optional[int]:
    """If a gin/knock action is legal AND correct by the rules, return its id.

    Gin requires deadwood == 0; knock requires deadwood <= knock_card.  We
    prefer gin (bigger bonus) over knock.  Returns None if neither should be
    chosen here.
    """
    gin_id = next((aid for aid, lbl in legal if _label_kind(lbl)[0] == 'gin'), None)
    knock_id = next((aid for aid, lbl in legal if _label_kind(lbl)[0] == 'knock'), None)
    eff_dw = deadwood if deadwood != 99 else compute_optimal_deadwood(hand)
    if gin_id is not None and eff_dw == 0:
        return gin_id
    if knock_id is not None and eff_dw <= knock_card:
        return knock_id
    return None


def _choose_meld_or_layoff(legal: list[tuple[int, str]], hand: list[str]
                           ) -> int:
    """Knock/Layoff phase: declare an explicit meld group if offered, otherwise
    lay off / discard the highest-value card outside the optimal melding, else
    pass."""
    # Prefer explicit meld groups (they only ever help).
    for aid, lbl in legal:
        if _label_kind(lbl)[0] == 'meld':
            return aid

    meld_cards = get_optimal_meld_cards(hand)
    pass_id = next((aid for aid, lbl in legal if _label_kind(lbl)[0] == 'pass'), None)

    best_id, best_score = None, None
    for aid, lbl in legal:
        kind, card = _label_kind(lbl)
        if kind != 'discard' or card is None:
            continue
        # Score: prefer shedding high-value deadwood; never break a meld.
        score = card_value(card)
        if card in meld_cards:
            score -= 100
        if best_score is None or score > best_score:
            best_score, best_id = score, aid
    if best_id is not None:
        return best_id
    if pass_id is not None:
        return pass_id
    return legal[0][0]


# ---------------------------------------------------------------------------
# Exact terminal plies (Wall / Knock / Layoff)
# ---------------------------------------------------------------------------
# The plies that settle the score depend only on our own hand and the public melds, so they are solved
# exactly against OpenSpiel's gin_rummy rules: a player's final deadwood is the value of the cards still in
# hand when it passes (undeclared melds count), the knocker may declare only melds of an arrangement within
# the knock card, and the defender lays off first, then declares. The previous handler
# (_choose_meld_or_layoff / _choose_meld_or_layoff_exact), probe-confirmed on OpenSpiel states:
#   Wall    passed a legal upcard-enabled Knock: a certain draw instead of a likely win;
#   Layoff  always laid off a legal card, even one breaking its own run (final deadwood 14 -> 25);
#   Melds   declared the lowest-id meld, a set or 3-card sub-run that strands run ends (knocker 9 vs 5);
#   Discard after a knock shed the highest-value melded card when nothing was deadwood, throwing away gin.
# _EXACT_TERMINAL_PLIES = False restores the previous labels.
_EXACT_TERMINAL_PLIES = True
_TERMINAL_PHASES = ('Wall', 'Knock', 'Layoff')


def _parse_layed_melds(obs: str, pid: int) -> list[frozenset]:
    """Melds player ``pid`` has declared (its 'Layed melds:' line, printed above its hand grid)."""
    m = re.search(rf'Player{pid}:[^\n]*\n(?:[^\n+|][^\n]*\n)*?Layed melds:([^\n]*)', obs)
    if not m:
        return []
    return [frozenset(_norm_card(c) for c in _RE_CARD.findall(group)) for group in m.group(1).split()]


def _parse_layoffs(obs: str) -> list[str]:
    """Cards the defender has laid off so far, in order (the 'Layoffs:' line)."""
    m = re.search(r'Layoffs:([^\n]*)', obs)
    return [_norm_card(c) for c in _RE_CARD.findall(m.group(1))] if m else []


def _is_set(meld: frozenset) -> bool:
    return len({get_rank(c) for c in meld}) == 1


def _run_ends(run: "set | frozenset") -> set:
    """The (at most two) cards extending a run by one rank at either end."""
    suit = get_suit(next(iter(run)))
    ranks = sorted(RANK_IDX[get_rank(c)] for c in run)
    return {RANK_ORDER[i] + suit for i in (ranks[0] - 1, ranks[-1] + 1) if 0 <= i <= 12}


def _layoff_targets(melds, prev_layoffs) -> set:
    """Cards OpenSpiel accepts as the next layoff onto ``melds`` (gin_rummy_utils AllLayoffs): the 4th card
    of a 3-card set, and a run's end cards after the run absorbs the earlier layoffs that touch its ORIGINAL
    ends (one pass, so 8h9hTh + 7h offers 6h, and + 7h + 6h still offers only 6h and Jh)."""
    prev = set(prev_layoffs)
    out: set = set()
    for meld in melds:
        if not meld:
            continue
        if _is_set(meld):
            if len(meld) == 3:
                rank = get_rank(next(iter(meld)))
                out |= {rank + s for s in SUITS} - set(meld)
            continue
        out |= _run_ends(set(meld) | (prev & _run_ends(meld)))
    return out


@lru_cache(maxsize=20000)
def _optimal_partitions(hand_key: tuple[str, ...]) -> tuple[tuple[frozenset, ...], ...]:
    """Every deadwood-optimal partition of ``hand_key`` (a sorted card tuple) into disjoint melds."""
    best = compute_optimal_deadwood(hand_key)
    total = sum(card_value(c) for c in hand_key)
    melds = sorted(set(find_all_melds(hand_key)), key=lambda m: sorted(m))
    values = [sum(card_value(c) for c in m) for m in melds]
    out: list[tuple[frozenset, ...]] = []

    def walk(start: int, used: frozenset, chosen: tuple, melded: int) -> None:
        if total - melded == best:  # optimal; no further meld can fit without beating the optimum
            out.append(chosen)
            return
        for i in range(start, len(melds)):
            if not (melds[i] & used):
                walk(i + 1, used | melds[i], chosen + (melds[i],), melded + values[i])

    walk(0, frozenset(), (), 0)
    return tuple(out)


def _label_cards(label: str) -> frozenset:
    return frozenset(_norm_card(c) for c in _RE_CARD.findall(label))


def _choose_knock_discard(parsed: list[tuple[int, str]], hand: list[str]) -> Optional[int]:
    """Knocker's forced discard: the legal card leaving the least optimal deadwood, then the higher value."""
    best: Optional[tuple] = None
    for aid, lbl in parsed:
        kind, card = _label_kind(lbl)
        if kind != 'discard' or card not in hand:
            continue
        rest = list(hand)
        rest.remove(card)
        key = (compute_optimal_deadwood(rest), -card_value(card), aid)
        if best is None or key < best:
            best = key
    return best[2] if best else None


def _layoff_final_deadwood(hand: tuple, melds: tuple, prev: tuple) -> int:
    """Least deadwood reachable by laying off any further sequence of cards, then melding optimally."""
    best = compute_optimal_deadwood(hand)
    for card in sorted(_layoff_targets(melds, prev) & set(hand)):
        rest = tuple(c for c in hand if c != card)
        best = min(best, _layoff_final_deadwood(rest, melds, prev + (card,)))
    return best


def _choose_layoff(parsed: list[tuple[int, str]], hand: list[str], knocker_melds: list[frozenset],
                   prev_layoffs: list[str]) -> Optional[int]:
    """Defender's layoff: the legal card whose layoff (and any chain it opens onto a run) ends at the least
    deadwood, only when that is strictly below keeping the hand; otherwise Pass."""
    pass_id = next((aid for aid, lbl in parsed if _label_kind(lbl)[0] == 'pass'), None)
    melds, prev = tuple(knocker_melds), tuple(prev_layoffs)
    best: Optional[tuple] = None
    for aid, lbl in parsed:
        kind, card = _label_kind(lbl)
        if kind != 'discard' or card not in hand:
            continue
        rest = tuple(c for c in hand if c != card)
        key = (_layoff_final_deadwood(rest, melds, prev + (card,)), -card_value(card), aid)
        if best is None or key < best:
            best = key
    if best is not None and (pass_id is None or best[0] < compute_optimal_deadwood(hand)):
        return best[2]
    return pass_id


def _choose_meld_declaration(parsed: list[tuple[int, str]], hand: list[str], dead: set,
                             is_knocker: bool) -> Optional[int]:
    """Declare a meld of a deadwood-optimal partition of the hand; Pass once that partition is declared.
    Among optimal partitions the knocker prefers the one leaving the fewest live layoff cards for the
    defender (dead = discard pile, upcard and our own cards), then the lowest meld ids."""
    pass_id = next((aid for aid, lbl in parsed if _label_kind(lbl)[0] == 'pass'), None)
    meld_ids = {_label_cards(lbl): aid for aid, lbl in parsed if _label_kind(lbl)[0] == 'meld'}
    best: Optional[tuple] = None
    for part in _optimal_partitions(tuple(sorted(hand))):
        if not part:
            return pass_id
        ids = sorted(meld_ids.get(m, 10 ** 6) for m in part)
        exposure = len(_layoff_targets(part, ()) - dead) if is_knocker else 0
        key = (ids[-1] >= 10 ** 6, exposure, ids)  # a partition we can declare in full comes first
        if best is None or key < best:
            best = key
    if best is None or best[2][0] >= 10 ** 6:
        return pass_id
    return best[2][0]


def _choose_terminal(observation: str, player_id: int, phase: str, parsed: list[tuple[int, str]],
                     hand: list[str], seen: set) -> Optional[int]:
    """Exact label for a Wall / Knock / Layoff decision, or None to defer to the previous handler (other
    phases, the backstop flag, or an unexpected action set). Never raises."""
    if not _EXACT_TERMINAL_PLIES or phase not in _TERMINAL_PHASES:
        return None
    try:
        kinds = {_label_kind(lbl)[0] for _aid, lbl in parsed}
        if kinds & {'draw_upcard', 'draw_stock'}:
            return None
        pass_id = next((aid for aid, lbl in parsed if _label_kind(lbl)[0] == 'pass'), None)
        if phase == 'Wall':
            # OpenSpiel offers Knock here only when the upcard brings our deadwood within the knock card;
            # Pass ends the hand as a draw.
            knock_id = next((aid for aid, lbl in parsed if _label_kind(lbl)[0] in ('knock', 'gin')), None)
            return knock_id if knock_id is not None else pass_id
        if 'discard' in kinds:
            if phase == 'Knock':
                return _choose_knock_discard(parsed, hand)
            return _choose_layoff(parsed, hand, _parse_layed_melds(observation, 1 - player_id),
                                  _parse_layoffs(observation))
        own = {c for m in _parse_layed_melds(observation, player_id) for c in m}
        dead = {_norm_card(c) for c in seen} | set(hand) | own
        return _choose_meld_declaration(parsed, hand, dead, is_knocker=(phase == 'Knock'))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _deterministic_teacher_enabled() -> bool:
    """True iff the OPT-IN deterministic hand engine is requested via
    ``GR_DETERMINISTIC_TEACHER``.

    DEFAULT OFF (unset / '' / '0' / 'false' / 'no' / 'off') -> the repo's
    existing stochastic PIMC teacher, so repo behavior is UNCHANGED until this
    flag is set. HARD RULE 1 forbids flipping a default before the VM gate, so
    the deterministic engine ships OPT-IN and is promoted to default only after
    student-vs-MCTS-25 on both bases (co-trained) confirms the win. Read at call
    time so a harness that exports the flag before generation takes effect
    without a re-import.
    """
    return os.getenv("GR_DETERMINISTIC_TEACHER", "0").strip().lower() in (
        "1", "true", "yes", "on"
    )


def choose_action(observation: str,
                  player_id: int,
                  legal_action_ids: list[int],
                  rng=None) -> int:
    """Choose the expert action id for ``observation``.

    DEFAULT = the existing stochastic PIMC teacher (``_choose_action_pimc``),
    which samples ``N_WORLDS`` worlds from ``rng``; repo behavior is UNCHANGED
    with no flags set.

    Set ``GR_DETERMINISTIC_TEACHER=1`` (or true/yes/on) to opt in to the new
    deterministic hand engine (``_choose_action_deterministic``): every decision
    is a pure function of the *visible* state (hand, upcard, parsed discard pile,
    phase), so identical states map to identical ``action_id`` labels and a
    7B/8B student can distil the policy without the cross-row label noise the
    stochastic teacher injects. Promote it to default only after the VM gate
    passes. Always returns one element of ``legal_action_ids``.
    """
    if _deterministic_teacher_enabled():
        return _choose_action_deterministic(
            observation, player_id, legal_action_ids, rng
        )
    return _choose_action_pimc(observation, player_id, legal_action_ids, rng)


# ---------------------------------------------------------------------------
# Stochastic PIMC path (repo DEFAULT teacher)
# ---------------------------------------------------------------------------

def _choose_action_pimc(observation: str,
                        player_id: int,
                        legal_action_ids: list[int],
                        rng=None) -> int:
    """Stochastic determinized-search (PIMC) path — the repo DEFAULT teacher.

    Runs whenever ``GR_DETERMINISTIC_TEACHER`` is NOT set (see ``choose_action``).
    This is the body that used to be ``choose_action`` plus the exact terminal
    plies (``_choose_terminal``); sampling ``N_WORLDS`` worlds with ``rng`` makes it stochastic — the cross-row
    label-noise root cause the opt-in deterministic engine addresses.
    Always returns one element of ``legal_action_ids``.
    """
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    legal_set = set(legal_action_ids)

    if rng is None:
        rng = random.Random(0xC0FFEE)
    elif isinstance(rng, int):
        rng = random.Random(rng)

    # --- Parse state -------------------------------------------------------
    phase = parse_phase(observation)
    hand = parse_hand(observation, player_id)
    upcard = parse_upcard(observation)
    deadwood = parse_deadwood(observation)
    knock_card = parse_knock_card(observation)
    seen = parse_discard_pile(observation, upcard)

    # Legal actions parsed from the observation, filtered to those actually
    # offered in legal_action_ids (the caller's list is authoritative).
    parsed = [(aid, lbl) for aid, lbl in parse_legal_actions(observation)
              if aid in legal_set]
    if not parsed:
        # We couldn't parse labels; fall back to a safe deterministic id.
        return legal_action_ids[0]

    # --- 0) Wall / Knock / Layoff plies: exact (ahead of the displayed-deadwood
    # knock check, which cannot see the upcard that enables a Wall knock) ---
    term = _choose_terminal(observation, player_id, phase, parsed, hand, seen)
    if term is not None and term in legal_set:
        return term

    # --- 1) Knock / Gin: deterministic, rule-correct -----------------------
    kg = _choose_knock_or_gin(parsed, hand, deadwood, knock_card)
    if kg is not None and kg in legal_set:
        return kg

    # --- 2) Layoff / meld phase: deterministic -----------------------------
    kinds = {_label_kind(lbl)[0] for _aid, lbl in parsed}
    if phase in ('Knock', 'Layoff', 'Wall') or kinds & {'meld'}:
        # If the only knock/gin options were the wrong call, fall through to
        # meld/layoff handling.
        if not (kinds & {'draw_upcard', 'draw_stock'}):
            choice = _choose_meld_or_layoff(parsed, hand)
            return choice if choice in legal_set else legal_action_ids[0]

    # --- Build determinized worlds for hidden-info decisions ---------------
    # Opponent hand size in standard gin = 10 (until they draw).  Use a robust
    # default; exact value is not critical to the relative action ranking.
    pool = _unseen_pool(hand, seen)
    opp_size = min(10, len(pool))
    n_worlds = N_WORLDS if len(pool) > opp_size else 1
    worlds = [_sample_world(pool, opp_size, rng) for _ in range(n_worlds)]

    # --- 3) Draw phase: PIMC over draw-stock vs draw-upcard ----------------
    draw_actions = [(aid, _label_kind(lbl)[0]) for aid, lbl in parsed
                    if _label_kind(lbl)[0] in ('draw_upcard', 'draw_stock')]
    pass_id = next((aid for aid, lbl in parsed if _label_kind(lbl)[0] == 'pass'), None)
    if draw_actions:
        scored: list[tuple[int, float]] = []
        for aid, kind in draw_actions:
            v = _evaluate_draw_action(kind, hand, upcard, worlds, knock_card)
            scored.append((aid, v))
        # FirstUpcard phase may also allow Pass: treat passing as value of the
        # best non-upcard alternative minus a tiny cost (declining a useless
        # upcard is correct; declining a useful one is bad).
        if pass_id is not None and pass_id in legal_set:
            up_help = _draw_helps(hand, upcard) if upcard != 'XX' else 0
            # Pass is good only when the upcard does not help.
            pass_val = -float(compute_optimal_deadwood(hand)) + (
                2.0 if up_help == 0 else -5.0
            )
            scored.append((pass_id, pass_val))
        return _pick(scored, legal_set, legal_action_ids, rng)

    # --- 4) Discard phase: PIMC over which card to discard -----------------
    discard_actions = [(aid, card) for aid, lbl in parsed
                       for k, card in [_label_kind(lbl)]
                       if k == 'discard' and card is not None]
    if discard_actions:
        scored = []
        for aid, card in discard_actions:
            v = _evaluate_discard_action(card, hand, worlds, knock_card)
            scored.append((aid, v))
        return _pick(scored, legal_set, legal_action_ids, rng)

    # --- Fallback: deterministic meld/layoff or first legal id -------------
    if pass_id is not None and pass_id in legal_set:
        return pass_id
    choice = _choose_meld_or_layoff(parsed, hand)
    return choice if choice in legal_set else legal_action_ids[0]


def _pick(scored: list[tuple[int, float]], legal_set: set,
          legal_action_ids: list[int], rng: random.Random) -> int:
    """Pick the best-scored action id, optionally softmax-sampling."""
    scored = [(aid, v) for aid, v in scored if aid in legal_set]
    if not scored:
        return legal_action_ids[0]
    if SOFTMAX_TEMP and SOFTMAX_TEMP > 0:
        import math
        mx = max(v for _, v in scored)
        weights = [math.exp((v - mx) / SOFTMAX_TEMP) for _, v in scored]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for (aid, _), w in zip(scored, weights):
            acc += w
            if r <= acc:
                return aid
        return scored[-1][0]
    # argmax with deterministic tie-break (lowest action id)
    best_v = max(v for _, v in scored)
    best = sorted(aid for aid, v in scored if v == best_v)
    return best[0]


# ---------------------------------------------------------------------------
# Deterministic hand engine (OPT-IN via GR_DETERMINISTIC_TEACHER) — pure state
# ---------------------------------------------------------------------------
# WHY: our stochastic PIMC teacher samples N_WORLDS opponent/stock layouts from
# rng, so the SAME visible state can be labelled with DIFFERENT action_ids across
# rows; a 7B/8B student then distils a blurred average -> a higher loss floor ->
# the 4th-place gin that decided tourn_e119d8158386fa26. This engine removes all
# sampling: DRAW/DISCARD/KNOCK/MELD/LAYOFF are each argmax over deterministic,
# rule-grounded features of the *visible* state alone, so identical states map to
# identical labels (distillable). Idea (NOT code) ported from the pos-1 repo's
# deterministic gin thesis (our_envs/gin_rummy_env.py); re-implemented over OUR
# own primitives: compute_optimal_deadwood, get_optimal_meld_cards,
# _best_discard_from_hand, parse_discard_pile, _choose_knock_or_gin.

# Discard scorer weights (higher score == MORE discardable). Tuned so the
# never-break-a-meld veto dominates, meld-potential "keep" bonuses are meaningful,
# and pile-safety is only a small bounded tiebreak.
_DET_MELD_KEEP = 15.0      # card sits in the deadwood-optimal melding -> never shed
_DET_LIVE_PAIR_KEEP = 8.0  # live same-rank partner (a set can still complete)
_DET_LIVE_ADJ_KEEP = 5.0   # live same-suit run neighbour (a run can still complete)
_DET_SAFETY_WEIGHT = 0.5   # per-unit weight of the bounded pile-safety bonus
_DET_SAFETY_CAP = 3        # bound so safety never overrides card-value / meld logic


def _known_cards(hand: list[str], seen: set) -> set:
    """Normalised set of cards we can already account for (our hand + everything
    publicly discarded). A meld connector NOT in this set is still drawable = 'live'."""
    return {_norm_card(c) for c in hand} | {_norm_card(c) for c in seen}


def _has_live_pair(card: str, hand: list[str], known: set) -> bool:
    """Card has a *live* set potential: we hold >=2 of its rank AND at least one
    remaining copy of that rank is unseen (still drawable). If every other copy is
    dead (in the discard pile), the set can never complete -> not live -> shed OK."""
    r = get_rank(card)
    if sum(1 for c in hand if get_rank(_norm_card(c)) == r) < 2:
        return False
    all_of_rank = {r + s for s in SUITS}
    return len(all_of_rank - known) >= 1


def _has_live_adjacency(card: str, hand: list[str], known: set) -> bool:
    """Card has a *live* run potential: a same-suit neighbour (rank +-1) or a
    single-gap partner (rank +-2) is in hand AND a connector that would extend the
    pair/gap to a >=3 run is still unseen. Dead connectors veto the bonus."""
    r, s = get_rank(card), get_suit(card)
    ri = RANK_IDX[r]
    suit_ranks = {RANK_IDX[get_rank(_norm_card(c))]
                  for c in hand if get_suit(_norm_card(c)) == s}

    def unseen(rank_idx: int) -> bool:
        if rank_idx < 0 or rank_idx > 12:
            return False
        return (RANK_ORDER[rank_idx] + s) not in known

    live = False
    if (ri - 1) in suit_ranks:      # hold (r-1, r): complete with r-2 or r+1
        live = live or unseen(ri - 2) or unseen(ri + 1)
    if (ri + 1) in suit_ranks:      # hold (r, r+1): complete with r-1 or r+2
        live = live or unseen(ri - 1) or unseen(ri + 2)
    if (ri - 2) in suit_ranks:      # hold (r-2, r) gap: complete with r-1
        live = live or unseen(ri - 1)
    if (ri + 2) in suit_ranks:      # hold (r, r+2) gap: complete with r+1
        live = live or unseen(ri + 1)
    return live


def _pile_safety(card: str, seen: set) -> int:
    """Bounded [0, _DET_SAFETY_CAP] safety of shedding ``card``: the more of its
    same-rank / same-suit-neighbour cards are ALREADY discarded, the less likely
    the opponent can use it, so it is safer to throw. Pure function of the pile."""
    r, s = get_rank(card), get_suit(card)
    ri = RANK_IDX[r]
    seen_norm = {_norm_card(c) for c in seen}
    safety = 0
    # same-rank copies already gone -> a set around this rank is unlikely for them
    safety += sum(1 for su in SUITS if (r + su) != card and (r + su) in seen_norm)
    # immediate run neighbours already gone -> a run around it is unlikely for them
    for d in (-1, 1):
        rj = ri + d
        if 0 <= rj <= 12 and (RANK_ORDER[rj] + s) in seen_norm:
            safety += 1
    return min(safety, _DET_SAFETY_CAP)


def _choose_draw_deterministic(parsed: list[tuple[int, str]], hand: list[str],
                               upcard: str, legal_set: set) -> int:
    """Free-roll strict-drop: take the upcard iff it STRICTLY lowers our optimal
    deadwood after the forced discard; otherwise draw from stock (or pass in the
    first-upcard phase). Pure function of hand + upcard."""
    draw_up = next((aid for aid, lbl in parsed
                    if _label_kind(lbl)[0] == 'draw_upcard'), None)
    draw_stock = next((aid for aid, lbl in parsed
                       if _label_kind(lbl)[0] == 'draw_stock'), None)
    pass_id = next((aid for aid, lbl in parsed
                    if _label_kind(lbl)[0] == 'pass'), None)

    if draw_up is not None and draw_up in legal_set \
            and upcard and upcard != 'XX' and len(upcard) == 2:
        base = compute_optimal_deadwood(hand)
        _, after = _best_discard_from_hand(hand + [upcard], KNOCK_DEFAULT)
        if after < base:  # STRICT drop == free-roll take
            return draw_up
    if draw_stock is not None and draw_stock in legal_set:
        return draw_stock
    if pass_id is not None and pass_id in legal_set:
        return pass_id
    if draw_up is not None and draw_up in legal_set:
        return draw_up
    return parsed[0][0]


def _choose_discard_deterministic(discard_actions: list[tuple[int, str]],
                                  hand: list[str], seen: set,
                                  knock_card: int) -> int:
    """Deterministic multi-factor argmax over legal discards (higher == shed).

    score = card_value                       (high deadwood -> more discardable)
            - _DET_MELD_KEEP  if in optimal meld     (never break a formed meld)
            - _DET_LIVE_PAIR_KEEP  if live pair       (keep set potential)
            - _DET_LIVE_ADJ_KEEP   if live adjacency  (keep run potential)
            + _DET_SAFETY_WEIGHT * bounded pile-safety
    Tie-break: higher card_value, then lowest action id -> fully deterministic."""
    meld_cards = get_optimal_meld_cards(hand)
    known = _known_cards(hand, seen)
    scored: list[tuple[float, int, int, int]] = []
    for aid, card in discard_actions:
        cn = _norm_card(card)
        score = float(card_value(cn))
        if cn in meld_cards:
            score -= _DET_MELD_KEEP
        if _has_live_pair(cn, hand, known):
            score -= _DET_LIVE_PAIR_KEEP
        if _has_live_adjacency(cn, hand, known):
            score -= _DET_LIVE_ADJ_KEEP
        score += _DET_SAFETY_WEIGHT * _pile_safety(cn, seen)
        # sort key: max score, then max card_value, then min action id (-aid)
        scored.append((score, card_value(cn), -aid, aid))
    if not scored:
        return discard_actions[0][0]
    scored.sort(reverse=True)
    return scored[0][3]


def _choose_meld_or_layoff_exact(parsed: list[tuple[int, str]],
                                 hand: list[str]) -> int:
    """Knock/Layoff phase, deterministic: declare an offered meld group (only ever
    helps); else lay off / discard the card whose removal yields the minimum
    EXACT resulting deadwood (via the cached DP); else pass; else first legal id.
    Separate from the PIMC path's ``_choose_meld_or_layoff`` so that path is
    untouched."""
    for aid, lbl in parsed:
        if _label_kind(lbl)[0] == 'meld':
            return aid
    pass_id = next((aid for aid, lbl in parsed
                    if _label_kind(lbl)[0] == 'pass'), None)
    scored: list[tuple[int, int, int, int]] = []
    for aid, lbl in parsed:
        kind, card = _label_kind(lbl)
        if kind != 'discard' or card is None:
            continue
        cn = _norm_card(card)
        if cn in hand:
            remaining = list(hand)
            remaining.remove(cn)
            resulting = compute_optimal_deadwood(remaining)
        else:
            resulting = compute_optimal_deadwood(hand)
        # minimise resulting deadwood (-resulting), then shed higher value, then low aid
        scored.append((-resulting, card_value(cn), -aid, aid))
    if scored:
        scored.sort(reverse=True)
        return scored[0][3]
    if pass_id is not None:
        return pass_id
    return parsed[0][0]


def _choose_action_deterministic(observation: str,
                                 player_id: int,
                                 legal_action_ids: list[int],
                                 rng=None) -> int:
    """OPT-IN expert path (``GR_DETERMINISTIC_TEACHER``): a pure function of the
    visible state (``rng`` is accepted for signature parity with the teacher
    contract but intentionally UNUSED — no sampling, so identical states ->
    identical action_ids). Always returns one element of ``legal_action_ids``
    (never-raise legal-id fallback)."""
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    legal_set = set(legal_action_ids)

    # --- Parse state (visible information only) ----------------------------
    phase = parse_phase(observation)
    hand = parse_hand(observation, player_id)
    upcard = parse_upcard(observation)
    deadwood = parse_deadwood(observation)
    knock_card = parse_knock_card(observation)
    seen = parse_discard_pile(observation, upcard)

    parsed = [(aid, lbl) for aid, lbl in parse_legal_actions(observation)
              if aid in legal_set]
    if not parsed:
        return legal_action_ids[0]

    # --- 0) Wall / Knock / Layoff plies: exact, shared with the PIMC path ---
    term = _choose_terminal(observation, player_id, phase, parsed, hand, seen)
    if term is not None and term in legal_set:
        return term

    # --- 1) Knock / Gin ASAP: rule-correct, deterministic ------------------
    kg = _choose_knock_or_gin(parsed, hand, deadwood, knock_card)
    if kg is not None and kg in legal_set:
        return kg

    kinds = {_label_kind(lbl)[0] for _aid, lbl in parsed}

    # --- 2) Knock / Layoff / meld phase: deterministic exact-delta ----------
    if (phase in ('Knock', 'Layoff', 'Wall') or (kinds & {'meld'})) \
            and not (kinds & {'draw_upcard', 'draw_stock'}):
        choice = _choose_meld_or_layoff_exact(parsed, hand)
        return choice if choice in legal_set else legal_action_ids[0]

    # --- 3) Draw phase: free-roll strict-drop -------------------------------
    if kinds & {'draw_upcard', 'draw_stock'}:
        choice = _choose_draw_deterministic(parsed, hand, upcard, legal_set)
        return choice if choice in legal_set else legal_action_ids[0]

    # --- 4) Discard phase: deterministic multi-factor argmax ----------------
    discard_actions = [(aid, card) for aid, lbl in parsed
                       for k, card in [_label_kind(lbl)]
                       if k == 'discard' and card is not None]
    if discard_actions:
        choice = _choose_discard_deterministic(discard_actions, hand, seen, knock_card)
        return choice if choice in legal_set else legal_action_ids[0]

    # --- Fallback: pass, else exact meld/layoff, else first legal id --------
    pass_id = next((aid for aid, lbl in parsed
                    if _label_kind(lbl)[0] == 'pass'), None)
    if pass_id is not None and pass_id in legal_set:
        return pass_id
    choice = _choose_meld_or_layoff_exact(parsed, hand)
    return choice if choice in legal_set else legal_action_ids[0]


# ---------------------------------------------------------------------------
# One-sentence rule-grounded CoT rationale (learnability device; opt-in)
# ---------------------------------------------------------------------------
# Mirrors othello_expert.get_label_comment (commit 6cbb36f): naming the visible
# feature the chosen move exploits lets the student learn state -> [feature] ->
# move at higher fidelity than a raw state -> move map. GATED behind PVP_LABEL_COT
# (default OFF -> returns ''), so with the flag unset the row builder produces the
# exact same content-less row as today. Deterministic and never-raises. The
# rationale describes the CALLER'S ``action_id`` (byte-consistent with the
# game_action tool call it precedes) — the format-anchor dimension passes the
# ROW's action_id.

def _label_cot_enabled() -> bool:
    return os.getenv("PVP_LABEL_COT", "0").strip().lower() in ("1", "true", "yes", "on")


def get_label_comment(observation: str, action_id: int, player_id: int) -> str:
    """Deterministic, <=140-char, never-raises rationale for ``action_id`` on this
    gin state (mover = ``player_id``). Returns '' when PVP_LABEL_COT is unset or on
    any failure, so the caller can fall back to a content-less row."""
    if not _label_cot_enabled():
        return ""
    try:
        label = next((lbl for aid, lbl in parse_legal_actions(observation)
                      if aid == action_id), None)
        if label is None:
            return ""
        hand = parse_hand(observation, player_id)
        upcard = parse_upcard(observation)
        knock_card = parse_knock_card(observation)
        kind, card = _label_kind(label)

        if kind == 'gin':
            return "Going gin: zero deadwood, every card melded for the bonus."[:140]
        phase = parse_phase(observation)
        if kind == 'knock':
            dw = parse_deadwood(observation)
            if phase == 'Wall' and upcard != 'XX':
                dw = _best_discard_from_hand(hand + [upcard], knock_card)[1]
            elif dw == 99:
                dw = compute_optimal_deadwood(hand)
            return (f"Knocking at {dw} deadwood (<= {knock_card}); "
                    f"ending the hand while ahead.")[:140]
        if kind == 'draw_upcard':
            if upcard and upcard != 'XX' and len(upcard) == 2:
                base = compute_optimal_deadwood(hand)
                _, after = _best_discard_from_hand(hand + [upcard], knock_card)
                return (f"Taking the {upcard}: it cuts my deadwood "
                        f"from {base} to {after}.")[:140]
            return "Taking the upcard: it lowers my deadwood."[:140]
        if kind == 'draw_stock':
            if upcard and upcard != 'XX' and len(upcard) == 2:
                return (f"Drawing from stock; the {upcard} would not "
                        f"lower my deadwood.")[:140]
            return "Drawing a fresh card from the stock."[:140]
        if kind == 'pass':
            if phase in ('Knock', 'Layoff'):
                return "Passing: no further layoff or meld lowers my deadwood."[:140]
            if phase == 'Wall':
                return "Passing at the wall: the upcard does not let me knock."[:140]
            return "Passing: the upcard does not improve my hand."[:140]
        if kind == 'discard' and card is not None:
            if card in get_optimal_meld_cards(hand):
                return (f"Discarding {card}; no lower card is free of a "
                        f"meld this turn.")[:140]
            known = _known_cards(hand, parse_discard_pile(observation, upcard))
            if not _has_live_pair(card, hand, known) \
                    and not _has_live_adjacency(card, hand, known):
                return (f"Discarding {card} ({card_value(card)} pts): a dead card "
                        f"with no live meld.")[:140]
            return (f"Discarding {card}, the highest card least likely to "
                    f"complete a meld.")[:140]
        if kind == 'meld':
            return "Declaring this meld from my lowest-deadwood arrangement."[:140]
        return ""
    except Exception:
        return ""
