"""Goofspiel SFT teacher — a robust, value-matching bid policy.

``choose_action(observation, player_id, legal_action_ids, rng=None) -> int``

Goofspiel is a simultaneous-move, zero-sum bidding game. Each round a prize
card of known value is revealed and both players sealed-bid one card from an
identical hand; the higher bid takes the prize. The full game is far too large
to solve exactly (num_cards=13 → an enormous simultaneous-move tree), and unlike
the sequential games there is no cheap exact/blueprint solver to lean on.

We therefore use the well-known strong, robust heuristic that the rest of this
repo already treats as goofspiel's reference strategy: **bid the card whose
value matches the current prize card** (spend high cards on high-value prizes,
dump low cards on low-value prizes). It self-balances spending across the game
and is hard to exploit: deviating to overbid a prize wins it but wastes a high
card, leaving you weak on a later prize you would otherwise have taken with the
card you spent.

Two deliberate departures from pure greedy matching, mirroring the othello
teacher's lesson that *exploration is load-bearing* for SFT coverage — a purely
deterministic teacher makes both seats play identically every round (constant
ties) and produces a degenerate, narrow state distribution:

  * SOFTMAX sampling over ``-|card_value - prize|`` (temperature ``_TEMPERATURE``)
    so near-prize cards are sometimes chosen, covering off-corridor states;
  * EPSILON uniform-random bids (``_EPSILON``) for broader coverage.

Both are RNG-seeded by the caller, so generation stays deterministic per game.

Parsing is defensive: the teacher reads the normalized observation envelope
built by ``pvp_selfplay._teacher_obs`` (``Current point card: X`` from the
state text + ``<id> -> [Pn]Bid: V`` legal-action lines). If the prize cannot be
parsed it falls back to a uniform-legal bid; an unparsable card label falls back
to ``value = action_id + 1`` (goofspiel's action id is the 0-based card index).
Always returns one element of ``legal_action_ids``.

Strategy is FLAG-GATED by ``GOOFSPIEL_TEACHER`` (read at call time), so A/B runs
toggle it without a reimport. All modes fall back to ``matching`` on any failure:
  * ``skimup`` (DEFAULT, unset) — deterministic skim-up v2 (``_choose_action_skimup``):
    concede prizes 1-2 with the lowest card, overbid the rest (three-up in big-hand openings,
    else two-up), and spend the 4th-lowest card on a prize no card can beat; ``twoup`` is the
    13759ee default.
  * ``matching`` — the prize-matching sampler described above.
  * ``rm`` — the near-Nash regret-matching / backward-induction solver in
    ``goofspiel_rm.py`` (the principled Nash policy; escapes the 78-13 pure-matcher
    trap). Lazily imported only when selected.
  * ``rankmix`` — a rank-relative Gaussian bid kernel centred on the RANK match
    (a mixed, less-exploitable cousin of value-matching).
  * ``overbid`` — the same kernel with the mode shifted ABOVE the match toward the
    winning "overbid-one" (the winner's hand-tuned lever, ported for A/B).
The last three are compared cheaply, with zero GPU, by ``verify_goofspiel_teacher.py``.
"""

from __future__ import annotations

import math
import os
import random
import re

# MEASURED 2026-08-02 (verify_vs_winner --env goofspiel --our-expert vs the goofspiel champion model):
# the old noisy defaults (temp 0.5, eps 0.08) crippled the teacher to 34% (BEHIND), but near-greedy
# prize-matching WON 71% (15-6-9, AHEAD). The strategy is strong — the softmax/epsilon noise was making
# it bid the wrong card ~a third of the time. Position diversity for SFT comes from the RANDOM prize-order
# chance node (not per-bid noise), so a near-greedy teacher keeps coverage while demonstrating STRONG bids.
_TEMPERATURE = float(os.getenv("GOOFSPIEL_TEACHER_TEMP", "0.1"))   # near-greedy: ~98% the prize-matched card
_EPSILON = float(os.getenv("GOOFSPIEL_TEACHER_EPSILON", "0.02"))  # tiny exploration (was 0.08 = too random)

_RE_PRIZE = re.compile(r"point card:\s*(\d+)", re.IGNORECASE)
# A legal-action line: "<id> -> ...Bid: <value>" (value optional; id authoritative).
_RE_ACTION_VALUE = re.compile(r"^\s*(\d+)\s*->.*?Bid:\s*(\d+)", re.IGNORECASE)

# --- rank-relative Gaussian kernel knobs (rankmix / overbid modes) ------------
# Ported IDEAS (not code) from the tournament winner's goofspiel kernel: weight
# each legal card by a broad Gaussian in RANK space (its rank within our remaining
# hand vs the prize's strength). rankmix centres on the rank match; overbid shifts
# the peak one notch above it toward the winning "overbid-one". All env-tunable.
# GUARDED floats: a garbage A/B env value must never raise on import — that would
# break the DEFAULT matching path too (same module). (Winner's audit P1 lesson.)
def _gfloat(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


_RANK_SPREAD = max(1e-6, _gfloat("GOOFSPIEL_RANK_SPREAD", 0.32))
_OVERBID_SHIFT = _gfloat("GOOFSPIEL_OVERBID_SHIFT", 0.12)
_EPSILON_FLOOR = min(max(_gfloat("GOOFSPIEL_EPSILON_FLOOR", 0.08), 0.0), 1.0)
# ENDGAME shift-floor numerator: the overbid peak is shifted by max(_OVERBID_SHIFT, this/(m-1)) so in
# SMALL hands (endgame) the shift stays a genuine ~half-rank instead of rounding back onto the exact
# prize-match — the exploitable "matching line". Mirrors the winner's max(0.12, 0.6/(m-1)) (idea, our code).
_ENDGAME_SHIFT_NUM = _gfloat("GOOFSPIEL_ENDGAME_SHIFT_NUM", 0.6)
# Remaining point cards (current + future prizes) — score the prize by its RANK among what's LEFT
# (more accurate than an absolute-value proxy). Optional in our obs; we fall back when absent.
_RE_REMAINING = re.compile(r"Remaining [Pp]oint [Cc]ards?:\s*([\d ]+)")


_TEACHER_MODES = ("matching", "rm", "rankmix", "overbid", "twoup", "skimup")
_DEFAULT_TEACHER_MODE = "skimup"  # 13759ee shipped "twoup": set it back here for a one-line rollback


def _teacher_mode() -> str:
    """Selected strategy, read at CALL time (so A/B / tests toggle without reimport).
    DEFAULT 'skimup' (v1 2026-09-30, v2 2026-10-02); see the SKIM-UP v2 block for the measurements.
    'twoup' (2026-09-27 default). Measured on the EVAL game (turn-based imp_info, 5/8/10/13 cards,
    400 games per pair): the reigning champion plays deterministic one-up (probed on its boss
    model: cheapest card above the prize, else concede the lowest), which beats pure matching 400-0.
    two-up (cheapest card >= prize+2) beats one-up 360-4 (99%), matching 79%, our old overbid kernel
    61%, ~even vs uniform random (48%); the old default overbid lost to one-up (29%) and matching (23%).
    Deterministic labels are also the most learnable. Other modes stay opt-in via GOOFSPIEL_TEACHER."""
    mode = (os.getenv("GOOFSPIEL_TEACHER") or _DEFAULT_TEACHER_MODE).strip().lower()
    return mode if mode in _TEACHER_MODES else _DEFAULT_TEACHER_MODE


def _coerce_rng(rng) -> random.Random:
    if rng is None:
        return random.Random(0x600F5)
    if isinstance(rng, int):
        return random.Random(rng)
    return rng


def _parse_prize(observation: str) -> "int | None":
    m = _RE_PRIZE.search(observation)
    return int(m.group(1)) if m else None


def _parse_action_values(observation: str, legal_action_ids: list[int]) -> dict[int, int]:
    """Map each legal action id -> bid card value.

    Reads the value from each ``<id> -> ...Bid: <value>`` line when present;
    falls back to ``value = id + 1`` (goofspiel action id == 0-based card index)
    for any legal id not found in the text.
    """
    values: dict[int, int] = {}
    for line in observation.splitlines():
        m = _RE_ACTION_VALUE.match(line)
        if m:
            values[int(m.group(1))] = int(m.group(2))
    return {aid: values.get(aid, aid + 1) for aid in legal_action_ids}


def _softmax_sample(action_ids: list[int], scores: list[float], temperature: float,
                    rng: random.Random) -> int:
    if temperature <= 1e-6:
        # Greedy: best score, ties broken by the rng for diversity across games.
        best = max(scores)
        candidates = [a for a, s in zip(action_ids, scores) if s == best]
        return rng.choice(candidates)
    m = max(scores)
    weights = [math.exp((s - m) / temperature) for s in scores]
    return rng.choices(action_ids, weights=weights, k=1)[0]


def _choose_action_matching(observation: str, player_id: int, legal_action_ids: list[int],
                            rng=None) -> int:
    """DEFAULT strategy: pick a legal bid id via prize-matching softmax (see module
    docstring). This is the byte-identical legacy ``choose_action`` body — the
    dispatcher below routes here whenever ``GOOFSPIEL_TEACHER`` is unset/'matching',
    and every other mode falls back to it on failure."""
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    rng = _coerce_rng(rng)
    legal = list(legal_action_ids)

    prize = _parse_prize(observation)
    if prize is None:
        # No prize visible — nothing principled to match against.
        return rng.choice(legal)

    if rng.random() < _EPSILON:
        return rng.choice(legal)

    id_to_value = _parse_action_values(observation, legal)
    # Higher score = closer to the prize value (exact match is best).
    scores = [-abs(id_to_value[aid] - prize) for aid in legal]
    return _softmax_sample(legal, scores, _TEMPERATURE, rng)


def _prize_rank_frac(observation: str, prize: int) -> "float | None":
    """Prize strength in [0, 1] = its rank among the REMAINING point cards, or None if the
    'Remaining point cards:' line is absent/unparseable (caller falls back to a hand-relative
    estimate). More accurate than the absolute scale: a 7 is 'weak' early but 'strong' if it's
    the top card still in play."""
    m = _RE_REMAINING.search(observation)
    if not m:
        return None
    try:
        rem = sorted({int(x) for x in m.group(1).split()})
    except ValueError:
        return None
    if prize not in rem or len(rem) < 2:
        return None
    return rem.index(prize) / (len(rem) - 1)


def _rank_kernel_choose(observation: str, legal_action_ids: list[int],
                        rng: random.Random, overbid: bool) -> int:
    """rankmix / overbid strategy: SAMPLE a rank-relative Gaussian bid kernel.

    Score each legal card by how well its RANK within our remaining hand matches the
    prize's strength (rank-relative, so spent high cards correctly shift our bids),
    with the peak on the match (rankmix) or shifted one notch above toward the
    winning overbid (overbid), plus an epsilon-uniform floor so every card keeps
    mass. Strictly mixed — never argmax — so it escapes the exploitable pure line.
    Raises on an unparsable prize (caller falls back to matching)."""
    legal = list(legal_action_ids)
    prize = _parse_prize(observation)
    if prize is None:
        raise ValueError("no prize to rank against")
    id_to_value = _parse_action_values(observation, legal)

    cards = sorted(id_to_value[aid] for aid in legal)
    m = len(cards)
    if m <= 1:
        return legal[0]
    n_est = max(cards[-1], prize)                     # deck size N (exact at game start)
    prize_frac = _prize_rank_frac(observation, prize)  # rank among REMAINING prizes (winner's PRIZE_RANK_MODE)
    if prize_frac is None:
        prize_frac = (prize - 1) / max(n_est - 1, 1)  # fallback: prize strength on the hand-relative scale
    rank_frac = {c: i / (m - 1) for i, c in enumerate(cards)}
    # Overbid peak shifted one notch ABOVE the match, with an ENDGAME FLOOR so small hands don't round
    # the mode back onto the exact matching line (the exploitable pure strategy).
    shift = max(_OVERBID_SHIFT, _ENDGAME_SHIFT_NUM / (m - 1)) if overbid else 0.0

    weights: list[float] = []
    for aid in legal:
        gap = rank_frac[id_to_value[aid]] - prize_frac
        weights.append(math.exp(-((gap - shift) ** 2) / (2.0 * _RANK_SPREAD ** 2)))
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(legal)
    floor = _EPSILON_FLOOR / m
    weights = [(1.0 - _EPSILON_FLOOR) * (w / total) + floor for w in weights]
    return rng.choices(legal, weights=weights, k=1)[0]


# RANK-9 deterministic learnable teacher (default OFF via GOOFSPIEL_ONEUP). Bids the CHEAPEST card
# that still STRICTLY beats the prize (the deterministic collapse of the winner's overbid-one
# Gaussian kernel), else concedes the lowest card. A pure state->id function: the same visible state
# always yields the same action_id, so a 7B/8B student distils it cleanly (the stochastic overbid /
# matching samplers flip labels across rows -> a blurred, higher-loss target). Winner IDEA:
# winner-pos1/scripts/our_envs/goofspiel_trajectories.py:78-100,304-351 (SAMPLED overbid-shift);
# our deterministic re-implementation over _parse_prize / _parse_action_values.
_LAST_REASON = ""  # debuggability: why the last GOOFSPIEL_ONEUP decision was made


def _set_reason(msg: str) -> None:
    global _LAST_REASON
    _LAST_REASON = msg


def last_reason() -> str:
    """Human-readable note on the last GOOFSPIEL_ONEUP decision (debug only)."""
    return _LAST_REASON


def _flag_on(name: str) -> bool:
    """True iff env var ``name`` is truthy. Unset/''/'0'/'false'/'no'/'off' => False (default OFF).
    Read at CALL time so an A/B run or a unit test toggles it without a reimport. Never raises."""
    v = os.getenv(name)
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")


def _choose_action_oneup(observation: str, legal_action_ids: list[int]) -> int:
    """Deterministic ONE-UP bid: the cheapest legal card whose value strictly exceeds the prize (win
    it with the least-valuable sufficient card), else the lowest card (concede cheaply). Reuses the
    same prize / id->value parsing as the sampled modes. Always returns a legal id."""
    legal = list(legal_action_ids)
    id_to_value = _parse_action_values(observation, legal)
    prize = _parse_prize(observation)
    if prize is None:
        aid = min(legal, key=lambda a: (id_to_value[a], a))
        _set_reason(f"oneup: prize unreadable -> concede lowest card {id_to_value[aid]} (id {aid})")
        return aid
    beaters = [a for a in legal if id_to_value[a] > prize]
    if beaters:
        aid = min(beaters, key=lambda a: (id_to_value[a], a))
        _set_reason(f"oneup: prize {prize} -> cheapest winning card {id_to_value[aid]} (id {aid})")
        return aid
    aid = min(legal, key=lambda a: (id_to_value[a], a))
    _set_reason(f"oneup: prize {prize} unbeatable -> concede lowest card {id_to_value[aid]} (id {aid})")
    return aid


def _by_value(legal: list[int], id_to_value: dict[int, int]) -> list[int]:
    """Legal ids ordered by card value (ties by id): index 0 is the lowest card."""
    return sorted(legal, key=lambda a: (id_to_value[a], a))


def _choose_action_upk(observation: str, legal_action_ids: list[int], margin: int,
                       concede_rank: int = 0) -> int:
    """Deterministic K-UP bid: the cheapest card >= prize + margin; if none, the cheapest card that still
    beats the prize (one-up); if no card beats the prize, the card at index ``concede_rank`` of the hand
    sorted low to high (0 = the lowest; the highest card when the hand is shorter). Always a legal id."""
    legal = list(legal_action_ids)
    id_to_value = _parse_action_values(observation, legal)
    prize = _parse_prize(observation)
    if prize is None:
        return min(legal, key=lambda a: (id_to_value[a], a))
    for threshold in (prize + margin, prize + 1):
        beaters = [a for a in legal if id_to_value[a] >= threshold]
        if beaters:
            aid = min(beaters, key=lambda a: (id_to_value[a], a))
            _set_reason(f"{margin}-up: prize {prize} -> card {id_to_value[aid]} (id {aid})")
            return aid
    ordered = _by_value(legal, id_to_value)
    aid = ordered[min(max(concede_rank, 0), len(ordered) - 1)]
    _set_reason(f"{margin}-up: prize {prize} unbeatable -> concede card #{concede_rank} "
                f"{id_to_value[aid]} (id {aid})")
    return aid


# SKIM-UP v2 (default since 2026-10-02). A probe of the previous round's field (three tournament models on
# CPU with eval-exact prompts, ~1000 bids) found three ways to play a prize nobody can win outright: a
# strict two-up bidder plays its lowest card (every such bid), a one-up bidder ties with the equal card,
# and a deck-scaled overbidder plays its 3rd-lowest card (it never plays its lowest). v1 conceded such
# prizes - usually the top prize - with its lowest card, so the overbidder took 10-13 points for a 3 (v1 vs
# that model: 6/24 games). v2 spends the 4th-lowest card there, skims only prizes 1-2, and bids three-up
# while holding 9+ cards (the openings of 10/13-card games, where the overbidder bids +2..+3).
# Teacher-level eval-rules sim (2000 games per cell, seat-swapped, 5/8/10/13 cards), v1 -> v2: overbidder
# clones .30 -> .93 (an upper bound; replaying the real model's deals gives ~.80-.85), the exact strict
# two-up clone .75 -> .91, one-up-tie clones .87 -> .89, weighted field .65 -> .89. Cost: a prize-matching
# bidder drops from .77 to .60. Concession rank 3 over 2 is a clone-evidence call (k2, via
# GOOFSPIEL_CONCEDE_RANK=2: field .885 and ahead vs the rule pool); k3 still holds if a rival moves its own
# concession up a card, and an off-by-one slip to k2/k4 costs little. All knobs are read at call time;
# GOOFSPIEL_CONCEDE_RANK=0 GOOFSPIEL_SKIM_MAX=3 GOOFSPIEL_OPEN_MARGIN=0 restores v1 (up2abs3) exactly.
_SKIM_MAX_PRIZE = 2
_SKIM_MARGIN = 2
_CONCEDE_RANK = 3   # no card beats the prize -> the 4th-lowest card (index 3; the highest in a shorter hand)
_OPEN_MARGIN = 3    # margin while holding >= _OPEN_HAND cards (big-deck openings); 0 disables
_OPEN_HAND = 9


def _gint(name: str, default: int) -> int:
    """Guarded int env knob (a garbage value never raises; falls back to the default)."""
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _choose_action_skimup(observation: str, legal_action_ids: list[int], margin: "int | None" = None,
                          skim_max: "int | None" = None, concede_rank: "int | None" = None,
                          open_margin: "int | None" = None, open_hand: "int | None" = None) -> int:
    """Deterministic SKIM-UP bid: the lowest card on a prize <= ``skim_max`` (when there is a choice), else
    the K-up bid with K = ``open_margin`` while holding >= ``open_hand`` cards (when open_margin > 0) and
    K = ``margin`` otherwise; when no card beats the prize, the ``concede_rank``-th lowest card. None means
    the module default (env-overridable at call time)."""
    margin = _gint("GOOFSPIEL_SKIM_MARGIN", _SKIM_MARGIN) if margin is None else margin
    skim_max = _gint("GOOFSPIEL_SKIM_MAX", _SKIM_MAX_PRIZE) if skim_max is None else skim_max
    concede_rank = _gint("GOOFSPIEL_CONCEDE_RANK", _CONCEDE_RANK) if concede_rank is None else concede_rank
    open_margin = _gint("GOOFSPIEL_OPEN_MARGIN", _OPEN_MARGIN) if open_margin is None else open_margin
    open_hand = _gint("GOOFSPIEL_OPEN_HAND", _OPEN_HAND) if open_hand is None else open_hand
    legal = list(legal_action_ids)
    prize = _parse_prize(observation)
    if prize is not None and prize <= skim_max and len(legal) > 1:
        id_to_value = _parse_action_values(observation, legal)
        aid = min(legal, key=lambda a: (id_to_value[a], a))
        _set_reason(f"skim-up: prize {prize} <= {skim_max} -> concede lowest card {id_to_value[aid]} (id {aid})")
        return aid
    k = open_margin if open_margin > 0 and len(legal) >= open_hand else margin
    return _choose_action_upk(observation, legal, k, concede_rank)


# --- field-style rule bidders (generation opponents + meta gate). Our own simple rules for the bidding
# styles seen in the probed field (see SKIM-UP v2 above); no other participant's code or weights. ---------
_RE_WINSEQ = re.compile(r"Win sequence:([^\n]*)")


def _choose_action_strict_up(observation: str, legal_action_ids: list[int], margin: int) -> int:
    """Strict K-up bidder: the cheapest card >= prize + margin, else the lowest card (no one-up fallback)."""
    legal = list(legal_action_ids)
    id_to_value = _parse_action_values(observation, legal)
    prize = _parse_prize(observation)
    ordered = _by_value(legal, id_to_value)
    if prize is None:
        return ordered[0]
    return next((a for a in ordered if id_to_value[a] >= prize + margin), ordered[0])


def _choose_action_oneup_tie(observation: str, legal_action_ids: list[int]) -> int:
    """One-up bidder that ties on the equal card: the 1 on prize 1; else the cheapest card above the prize;
    else the equal card (a sure tie) when held; else the lowest card."""
    legal = list(legal_action_ids)
    id_to_value = _parse_action_values(observation, legal)
    prize = _parse_prize(observation)
    ordered = _by_value(legal, id_to_value)
    if prize is None or len(ordered) == 1:
        return ordered[0]
    if prize == 1 and id_to_value[ordered[0]] == 1:
        return ordered[0]
    beat = next((a for a in ordered if id_to_value[a] > prize), None)
    if beat is not None:
        return beat
    return next((a for a in ordered if id_to_value[a] == prize), ordered[0])


_OVERBID_BY_DECK = {5: 1, 8: 1, 10: 2, 13: 3}


def _choose_action_overbid_deck(observation: str, legal_action_ids: list[int]) -> int:
    """Deck-scaled overbidder, a generation and gate opponent (a simplified rule in the observed style, not a
    fitted clone of any model): the cheapest card >= prize + d with d growing with the deck size (cards in
    hand + rounds played: +1/+1/+2/+3 for 5/8/10/13 cards), else the cheapest winner; never the lowest card
    while holding two or more; no card beats the prize -> the 3rd-lowest card while holding >= 5 cards,
    else the highest."""
    legal = list(legal_action_ids)
    id_to_value = _parse_action_values(observation, legal)
    prize = _parse_prize(observation)
    ordered = _by_value(legal, id_to_value)
    if prize is None or len(ordered) == 1:
        return ordered[0]
    if id_to_value[ordered[-1]] <= prize:
        return ordered[min(2, len(ordered) - 1)] if len(ordered) >= 5 else ordered[-1]
    m = _RE_WINSEQ.search(observation)
    rounds = len(m.group(1).split()) if m else 0
    d = _OVERBID_BY_DECK.get(rounds + len(ordered), 2)
    aid = next((a for th in (prize + d, prize + 1) for a in ordered if id_to_value[a] >= th), ordered[0])
    return ordered[1] if aid == ordered[0] else aid


def choose_action(observation: str, player_id: int, legal_action_ids: list[int],
                  rng=None) -> int:
    """Dispatch to the ``GOOFSPIEL_TEACHER`` strategy (default ``_DEFAULT_TEACHER_MODE``).

    Every mode is NEVER-RAISE: any failure (import, parse, illegal id) falls back to
    ``matching``, so selecting a mode can never crash generation or return an illegal
    action."""
    if not legal_action_ids:
        raise ValueError("legal_action_ids must be non-empty")
    # RANK-9 deterministic teacher (default OFF): GOOFSPIEL_ONEUP overrides every sampled mode with
    # the learnable cheapest-winning-card label. Prior modes (default 'overbid') stay the A/B backstop
    # when the flag is off. Never-raise: any failure falls through to the sampled dispatch below.
    if _flag_on("GOOFSPIEL_ONEUP"):
        try:
            action = _choose_action_oneup(observation, legal_action_ids)
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001
            pass
    mode = _teacher_mode()

    if mode == "skimup":
        try:
            action = _choose_action_skimup(observation, legal_action_ids)
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001 — never-raise: fall back to matching
            pass
    elif mode == "twoup":
        try:
            action = _choose_action_upk(observation, legal_action_ids, margin=2)
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001 — never-raise: fall back to matching
            pass
    elif mode == "rm":
        try:
            from envs.goofspiel_rm import choose_action as _rm_choose  # lazy: only when selected
            action = _rm_choose(observation, player_id, legal_action_ids, rng=rng)
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001 — never-raise: fall back to matching
            pass
    elif mode in ("rankmix", "overbid"):
        try:
            action = _rank_kernel_choose(observation, legal_action_ids,
                                         _coerce_rng(rng), overbid=(mode == "overbid"))
            if action in legal_action_ids:
                return action
        except Exception:  # noqa: BLE001
            pass

    return _choose_action_matching(observation, player_id, legal_action_ids, rng)


# --- rule bidders: the generation opponent pool (goofspiel_hero.py) and the verify_goofspiel_meta field ---
# Same signature as choose_action: policy(observation, player_id, legal_action_ids, rng) -> legal id.
# A broad, uniform mix of NON-mirror opponents puts the recorded seat ahead, behind and level: the K-up
# ladder (one-up = the champion lineage, two-up = our 13759ee label), both v1 skim labels, the three
# field-style bidders (strict two-up, one-up with ties, deck-scaled overbid), the matching sampler, uniform
# random, and our own label with 25% random bids.
_NOISY_SELF_EPS = 0.25


def _noisy_self(observation: str, player_id: int, legal_action_ids: list[int], rng) -> int:
    rng = _coerce_rng(rng)
    aid = choose_action(observation, player_id, legal_action_ids, rng=rng)
    if len(legal_action_ids) > 1 and rng.random() < _NOISY_SELF_EPS:
        return rng.choice([a for a in legal_action_ids if a != aid])
    return aid


OPPONENT_POOL = (
    ("one-up", lambda obs, pid, legal, rng: _choose_action_upk(obs, legal, 1)),
    ("two-up", lambda obs, pid, legal, rng: _choose_action_upk(obs, legal, 2)),
    ("three-up", lambda obs, pid, legal, rng: _choose_action_upk(obs, legal, 3)),
    # v1 labels (concede the lowest card, no opening margin), kept as field members
    ("up2abs3", lambda obs, pid, legal, rng: _choose_action_skimup(obs, legal, 2, 3, 0, 0)),
    ("up3abs2", lambda obs, pid, legal, rng: _choose_action_skimup(obs, legal, 3, 2, 0, 0)),
    # field-style bidders: the bidding styles seen in the probed field
    ("strict-two-up", lambda obs, pid, legal, rng: _choose_action_strict_up(obs, legal, 2)),
    ("one-up-tie", lambda obs, pid, legal, rng: _choose_action_oneup_tie(obs, legal)),
    ("overbid-deck", lambda obs, pid, legal, rng: _choose_action_overbid_deck(obs, legal)),
    ("matching", lambda obs, pid, legal, rng: _choose_action_matching(obs, pid, legal, rng)),
    ("random", lambda obs, pid, legal, rng: _coerce_rng(rng).choice(list(legal))),
    ("noisy-self", _noisy_self),
)
