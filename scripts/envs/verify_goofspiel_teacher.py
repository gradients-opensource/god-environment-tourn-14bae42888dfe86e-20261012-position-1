"""Head-to-head simulator for the Goofspiel teacher strategies — pure Python, NO
model / GPU / pyspiel — so we can find the STRONGEST bidder CHEAPLY before spending
any training compute.

WHY THIS EXISTS. Goofspiel has no winning pure strategy (Rhoads-Bartholdi 2012: a
pure value-matcher loses 78-13 to the "overbid-one" counter), so the repo's default
prize-``matching`` teacher is, in symmetric PvP, an exploitable pure line. We built a
near-Nash regret-matching / backward-induction teacher (``goofspiel_rm``) plus two
ported rank-kernel baselines (``rankmix`` / ``overbid``). This script plays any two of
the four strategies against each other over many SEEDED games at EACH deck size
N in {5, 8, 10, 13} (alternating seats so neither gets a first-move edge) and reports
per-N and overall win-rates — the zero-GPU way to decide whether ``rm`` actually beats
``matching`` before wiring it into generation.

The four strategies (selected by name, exactly the ``GOOFSPIEL_TEACHER`` modes):
  * matching  — prize-matching softmax (the current default teacher).
  * rankmix   — rank-relative Gaussian kernel centred on the rank match.
  * overbid   — same kernel, mode shifted toward the winning "overbid-one".
  * rm        — the near-Nash regret-matching / backward-induction solver.

``goof_spiel_env.py`` is the training-time HTTP env wrapper (it imports trl/torch via
``shared_env`` and holds NO local game rules), so — like ``verify_liars_dice_teacher``
— this simulator implements the standard, trivial goofspiel rules self-contained and
reuses only the real strategy modules (goofspiel_expert / goofspiel_rm), which are
pure-stdlib.

Run from scripts/:
    python envs/verify_goofspiel_teacher.py --strategy-a rm --strategy-b matching --games 400
    python envs/verify_goofspiel_teacher.py --strategy-a overbid --strategy-b rm --games 400
    python envs/verify_goofspiel_teacher.py --strategy-a rm --strategy-b matching --cards 5

READ THE RESULT:
  * A wins clearly (>~55%)  -> strategy A is the stronger bidder at that N.
  * ~50%                    -> a wash (both near-equivalent under this matchup).
Cold ``rm`` builds its per-N Nash cache once (see the [build] lines); it is cached to
.expert_cache and reused, so re-runs are fast.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types
from typing import Callable, List, Tuple

# Make envs.* importable when run as a file (python envs/verify_X.py puts scripts/envs
# on sys.path, not scripts/). Shim a lightweight `envs` pkg so the heavy real
# envs/__init__.py (trl/torch) is skipped — same bootstrap the other verify_* scripts
# and offline tests use. Must precede any `import envs.*`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.goofspiel_expert as ge      # noqa: E402  (pure stdlib)
import envs.goofspiel_rm as grm         # noqa: E402  (pure stdlib)

_ALL_CARDS: Tuple[int, ...] = (5, 8, 10, 13)
_STRATEGIES = ("matching", "rankmix", "overbid", "rm")

# A strategy is choose(observation, player_id, legal_action_ids, rng) -> action id.
Strategy = Callable[[str, int, List[int], random.Random], int]


def _make_strategy(name: str) -> Strategy:
    """Return the choose-action callable for a named strategy. rankmix/overbid wrap
    the kernel with the same matching fallback the deployed dispatcher uses."""
    name = name.lower()
    if name == "matching":
        return lambda obs, pid, legal, rng: ge._choose_action_matching(obs, pid, legal, rng)
    if name in ("rankmix", "overbid"):
        overbid = name == "overbid"

        def _kernel(obs: str, pid: int, legal: List[int], rng: random.Random) -> int:
            try:
                action = ge._rank_kernel_choose(obs, legal, rng, overbid=overbid)
                if action in legal:
                    return action
            except Exception:  # noqa: BLE001 — mirror the dispatcher's matching fallback
                pass
            return ge._choose_action_matching(obs, pid, legal, rng)

        return _kernel
    if name == "rm":
        return lambda obs, pid, legal, rng: grm.choose_action(obs, pid, legal, rng=rng)
    raise ValueError(f"unknown strategy {name!r}; choose from {_STRATEGIES}")


# ---------------------------------------------------------------------------
# Self-contained goofspiel rules + observation builder (the format the deployed
# teachers parse: the pvp_selfplay._teacher_obs envelope + goofspiel state fields)
# ---------------------------------------------------------------------------

def _build_obs(me: int, hands: Tuple[frozenset, frozenset], scores: Tuple[int, int],
               remaining_incl: List[int], prize: int) -> Tuple[str, List[int]]:
    """Build ``me``'s observation string and legal action-id list for the round.

    Includes BOTH hands, both scores, the remaining prize pool (incl. current) and
    the current prize, so every strategy can parse what it needs: matching/rankmix/
    overbid read the prize + legal card values; rm additionally reads the opponent
    hand, the running score and the remaining prizes to reconstruct the Nash state."""
    my_hand = sorted(hands[me])
    legal_ids = [c - 1 for c in my_hand]
    action_lines = "\n".join(f"{c - 1} -> [P{me}]Bid: {c}" for c in my_hand)
    obs = (
        "Current State:\n"
        f"Current point card: {prize}\n"
        f"P0 hand: {' '.join(str(c) for c in sorted(hands[0]))}\n"
        f"P1 hand: {' '.join(str(c) for c in sorted(hands[1]))}\n"
        f"Points: {scores[0]} {scores[1]}\n"
        f"Remaining point cards: {' '.join(str(c) for c in sorted(remaining_incl))}\n"
        f"You are Player {me}.\n"
        "Legal Actions:\n"
        f"{action_lines}\n"
        "Your choice (ID only):"
    )
    return obs, legal_ids


def _play_game(n: int, p0_strat: Strategy, p1_strat: Strategy,
               deck_rng: random.Random,
               move_rngs: Tuple[random.Random, random.Random]) -> Tuple[int, int]:
    """Play one full N-card goofspiel game. Prizes are revealed in a random order
    (the chance node); each seat samples from its own independent RNG stream so the
    two mixed strategies' bids are statistically independent. Returns (score0, score1)."""
    order = list(range(1, n + 1))
    deck_rng.shuffle(order)                 # random points_order
    hands = [frozenset(range(1, n + 1)), frozenset(range(1, n + 1))]
    scores = [0, 0]
    strategies = (p0_strat, p1_strat)

    for r, prize in enumerate(order):
        remaining_incl = order[r:]          # current + future prizes still in play
        cards = [0, 0]
        for me in (0, 1):
            obs, legal = _build_obs(me, (hands[0], hands[1]), (scores[0], scores[1]),
                                    remaining_incl, prize)
            try:
                aid = strategies[me](obs, me, legal, move_rngs[me])
            except Exception:  # noqa: BLE001 — a broken strategy forfeits the round cheaply
                aid = min(legal)
            if aid not in legal:
                aid = min(legal)
            cards[me] = aid + 1
        if cards[0] > cards[1]:
            scores[0] += prize
        elif cards[1] > cards[0]:
            scores[1] += prize
        hands[0] = hands[0] - {cards[0]}
        hands[1] = hands[1] - {cards[1]}

    return scores[0], scores[1]


def _run_matchup(n: int, strat_a: Strategy, strat_b: Strategy, games: int,
                 base_seed: int) -> Tuple[int, int, int]:
    """Play ``games`` seeded games at deck size N, alternating which seat A takes.
    Returns (a_wins, b_wins, draws)."""
    a_wins = b_wins = draws = 0
    for i in range(games):
        a_is_p0 = (i % 2 == 0)
        deck_rng = random.Random(base_seed + i)
        move_rngs = (random.Random((base_seed + i) * 2654435761 & 0x7FFFFFFF),
                     random.Random((base_seed + i) * 40503 + 17 & 0x7FFFFFFF))
        p0, p1 = (strat_a, strat_b) if a_is_p0 else (strat_b, strat_a)
        s0, s1 = _play_game(n, p0, p1, deck_rng, move_rngs)
        a_score, b_score = (s0, s1) if a_is_p0 else (s1, s0)
        if a_score > b_score:
            a_wins += 1
        elif b_score > a_score:
            b_wins += 1
        else:
            draws += 1
    return a_wins, b_wins, draws


def _prebuild(name: str, n: int) -> None:
    """Prime the rm Nash cache for N (a cold build is one-time and cached), printing
    what kind of solve backs this N so the result is interpretable."""
    if name.lower() != "rm":
        return
    payload = grm._load_or_build(n)
    ptype = payload.get("type", "?")
    size = len(payload.get("value_memo") or payload.get("policy") or {})
    extra = ""
    if ptype == "sampled":
        extra = f", {payload.get('iters', '?')} iters"
    elif ptype == "none":
        extra = " (matching-only fallback)"
    print(f"  [build] rm N={n}: type={ptype}, entries={size}{extra}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Head-to-head goofspiel teacher simulator (pure Python, no GPU).")
    p.add_argument("--strategy-a", default="rm", choices=_STRATEGIES)
    p.add_argument("--strategy-b", default="matching", choices=_STRATEGIES)
    p.add_argument("--games", type=int, default=400, help="games per deck size N")
    p.add_argument("--cards", type=int, nargs="+", default=list(_ALL_CARDS),
                   choices=list(_ALL_CARDS), help="which deck sizes N to test")
    p.add_argument("--seed", type=int, default=20260817)
    args = p.parse_args()

    strat_a = _make_strategy(args.strategy_a)
    strat_b = _make_strategy(args.strategy_b)

    print(f"Goofspiel teacher head-to-head: A={args.strategy_a}  vs  B={args.strategy_b}"
          f"   ({args.games} games / N, seats alternated)\n")

    total_a = total_b = total_draw = 0
    for n in args.cards:
        _prebuild(args.strategy_a, n)
        _prebuild(args.strategy_b, n)
        a_wins, b_wins, draws = _run_matchup(n, strat_a, strat_b, args.games,
                                             args.seed + n * 1000)
        total_a += a_wins
        total_b += b_wins
        total_draw += draws
        decisive = a_wins + b_wins
        wr = 100.0 * a_wins / decisive if decisive else 50.0
        print(f"  N={n:>2}:  A {a_wins:>4}  |  B {b_wins:>4}  |  draws {draws:>4}"
              f"   ->  A win-rate {wr:5.1f}% of decisive")

    decisive = total_a + total_b
    wr = 100.0 * total_a / decisive if decisive else 50.0
    print("\nOVERALL")
    print(f"  A={args.strategy_a}: {total_a} wins   B={args.strategy_b}: {total_b} wins"
          f"   draws: {total_draw}")
    print(f"  A win-rate of decisive games: {wr:.1f}%")
    if total_a > total_b * 1.05:
        print(f"\n[A STRONGER] '{args.strategy_a}' clearly beats '{args.strategy_b}'. ✅")
    elif total_b > total_a * 1.05:
        print(f"\n[B STRONGER] '{args.strategy_b}' clearly beats '{args.strategy_a}'. ⚠️")
    else:
        print(f"\n[WASH] '{args.strategy_a}' and '{args.strategy_b}' are ~even under this matchup.")


if __name__ == "__main__":
    main()
