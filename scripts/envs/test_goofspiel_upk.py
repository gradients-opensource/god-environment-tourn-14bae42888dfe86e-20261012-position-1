"""Tests for the deterministic goofspiel teachers: K-up ('twoup', HEAD 13759ee) and skim-up v2 ('skimup', default).

Run from scripts/envs:  python test_goofspiel_upk.py
"""
import os
import random
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.goofspiel_expert as ge  # noqa: E402
from envs.goofspiel_expert import (OPPONENT_POOL, _choose_action_skimup,  # noqa: E402
                                   _choose_action_upk, _teacher_mode, choose_action)

FAILS = []
_V2_KNOBS = ("GOOFSPIEL_SKIM_MARGIN", "GOOFSPIEL_SKIM_MAX", "GOOFSPIEL_CONCEDE_RANK", "GOOFSPIEL_OPEN_MARGIN",
             "GOOFSPIEL_OPEN_HAND")
_V1_ENV = {"GOOFSPIEL_CONCEDE_RANK": "0", "GOOFSPIEL_SKIM_MAX": "3", "GOOFSPIEL_OPEN_MARGIN": "0"}


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def obs(prize: int, hand: list[int], player: int = 0, points: str = "0 0", seq: str = "",
        remaining: str = "123") -> tuple[str, list[int]]:
    """Teacher envelope like pvp_selfplay._teacher_obs on the eval format (action id = card - 1)."""
    lines = "\n".join(f"{c - 1} -> [P{player}]Bid: {c}" for c in hand)
    state = (f"You are Player {player} (P{player}).\nCurrent player: {player}\nCurrent point card: {prize}\n"
             f"Remaining Point Cards: {remaining}\nPoints: {points} \n"
             f"P{player} hand: {' '.join(map(str, hand))} \nWin sequence: {seq}\n")
    return (f"Current State:\n{state}\nYou are Player {player}.\nLegal Actions:\n{lines}\nYour choice (ID only):",
            [c - 1 for c in hand])


def card(aid: int) -> int:
    return aid + 1


def _pid0(pair: tuple[str, list[int]]) -> tuple[str, int, list[int]]:
    """(observation, player 0, legal ids): positional args for a policy call on an obs() state."""
    return pair[0], 0, pair[1]


def skim(prize: int, hand: list[int]) -> int:
    """The default skim-up v2 bid (card value) on a hand-crafted state."""
    o, legal = obs(prize, hand)
    return card(_choose_action_skimup(o, legal))


def v1_label(prize: int, hand: list[int]) -> int:
    """Frozen oracle of skim-up v1 (up2abs3, default 2026-09-30): the lowest card on a prize <= 3 with 2+
    cards; else the cheapest card >= prize + 2; else the cheapest card > prize; else the lowest card."""
    if prize <= 3 and len(hand) > 1:
        return min(hand)
    for threshold in (prize + 2, prize + 1):
        winners = [c for c in hand if c >= threshold]
        if winners:
            return min(winners)
    return min(hand)


def state_sample(n: int, seed: int) -> list[tuple[int, list[int]]]:
    """A fixed, seeded sample of (prize, sorted hand) states over every eval deck size."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        deck = rng.choice((5, 8, 10, 13))
        out.append((rng.randint(1, deck), sorted(rng.sample(range(1, deck + 1), rng.randint(1, deck)))))
    return out


def clear_knobs() -> None:
    for k in _V2_KNOBS:
        os.environ.pop(k, None)


def test_twoup_backstop() -> None:
    o, legal = obs(5, [1, 2, 3, 4, 5, 6, 7, 8])
    check("twoup: prize 5, full hand -> 7 (cheapest >= prize+2)", card(_choose_action_upk(o, legal, 2)) == 7)
    o, legal = obs(7, [2, 6, 8])
    check("twoup: prize 7, no card >= 9 -> 8 (one-up fallback)", card(_choose_action_upk(o, legal, 2)) == 8)
    o, legal = obs(8, [1, 3, 5])
    check("twoup: prize 8 unbeatable -> concede lowest 1", card(_choose_action_upk(o, legal, 2)) == 1)
    os.environ["GOOFSPIEL_TEACHER"] = "twoup"
    o, legal = obs(2, [1, 2, 3, 4, 5, 6, 7, 8])
    check("GOOFSPIEL_TEACHER=twoup still selectable (prize 2 -> 4, no skim)",
          _teacher_mode() == "twoup" and card(choose_action(o, 0, legal)) == 4)
    os.environ.pop("GOOFSPIEL_TEACHER", None)


def test_v2_skim_branch() -> None:
    for prize in (1, 2):
        check(f"skim: prize {prize}, hand [2,5,7,8] -> lowest 2", skim(prize, [2, 5, 7, 8]) == 2)
    check("skim: prize 2 with 13 cards -> lowest 1 (skim wins over the opening margin)",
          skim(2, list(range(1, 14))) == 1)
    check("skim line is 2: prize 3, hand [2,5,7,8] -> two-up 5", skim(3, [2, 5, 7, 8]) == 5)
    o, legal = obs(2, [9])
    check("skim needs 2+ cards: single legal card on a skim prize -> it",
          _choose_action_skimup(o, legal) == legal[0])


def test_v2_open_three_up_branch() -> None:
    check("open: 13 cards, prize 5 -> three-up 8", skim(5, list(range(1, 14))) == 8)
    check("open: 10 cards, prize 3 -> three-up 6", skim(3, list(range(1, 11))) == 6)
    check("open: 9 cards (the opening line), prize 5 -> three-up 8", skim(5, list(range(1, 10))) == 8)


def test_v2_two_up_branch() -> None:
    check("two-up: 8 cards (below the opening line), prize 5 -> 7", skim(5, list(range(1, 9))) == 7)
    check("two-up: 8 cards, prize 4 -> 6", skim(4, list(range(1, 9))) == 6)
    check("two-up: 4 cards [2,5,7,8], prize 6 -> 8", skim(6, [2, 5, 7, 8]) == 8)


def test_v2_one_up_fallback_branch() -> None:
    check("one-up fallback: prize 7, hand [2,6,8] -> 8", skim(7, [2, 6, 8]) == 8)
    check("one-up fallback in an opening: 13 cards, prize 11 -> 12", skim(11, list(range(1, 14))) == 12)
    check("one-up fallback in an opening: 13 cards, prize 12 -> 13", skim(12, list(range(1, 14))) == 13)


def test_v2_k3_concession_branch() -> None:
    check("k3: unbeatable prize 12, hand [1,3,5,9] -> 4th-lowest 9", skim(12, [1, 3, 5, 9]) == 9)
    check("k3: prize 13 only tieable, hand [1,2,3,4,5,13] -> 4th-lowest 4", skim(13, [1, 2, 3, 4, 5, 13]) == 4)
    check("k3: opening, prize 13, full 13-card hand -> 4th-lowest 4", skim(13, list(range(1, 14))) == 4)
    check("k3: prize 10, hand [1,2,3,4,6,9] -> 4th-lowest 4", skim(10, [1, 2, 3, 4, 6, 9]) == 4)


def test_v2_short_hand_concession() -> None:
    check("< 4 cards: unbeatable prize 10, hand [1,4,7] -> the highest 7", skim(10, [1, 4, 7]) == 7)
    check("< 4 cards: unbeatable prize 9, hand [2,6] -> the highest 6", skim(9, [2, 6]) == 6)
    o, legal = obs(10, [3])
    check("< 4 cards: single legal card on an unbeatable prize -> it", _choose_action_skimup(o, legal) == legal[0])


def test_v2_label_reads_only_prize_and_hand() -> None:
    o, legal = obs(2, [4, 9, 13], player=1, points="7 12")
    check("P1, prize 2, behind on points -> still skim with the lowest 4", card(choose_action(o, 1, legal)) == 4)
    o, legal = obs(6, list(range(1, 9)), points="20 3")
    check("label ignores the Points line (ahead) -> two-up 8", card(choose_action(o, 0, legal)) == 8)
    same = True
    for prize, hand in state_sample(300, seed=11):
        base = choose_action(*_pid0(obs(prize, hand)))
        for variant in (obs(prize, hand, player=1, points="9 2", seq="0 1 -3 ", remaining="13579"),
                        obs(prize, hand, points="0 15", seq="1 1 ")):
            o, legal = variant
            pid = 1 if "Player 1" in o else 0
            same &= choose_action(o, pid, legal) == base
    check("label is a function of (prize, own hand) only on 300 states (seat/points/history/remaining vary)",
          same)


def test_determinism() -> None:
    o, legal = obs(5, list(range(1, 9)))
    ids = {choose_action(o, 0, legal, rng=random.Random(s)) for s in range(20)}
    check("default label is deterministic (same obs, 20 rngs -> one id)", len(ids) == 1)
    sample = state_sample(500, seed=5)
    first = [choose_action(*_pid0(obs(p, h)), rng=random.Random(1)) for p, h in sample]
    second = [choose_action(*_pid0(obs(p, h)), rng=random.Random(2)) for p, h in sample]
    third = [choose_action(*_pid0(obs(p, h))) for p, h in sample]
    check("default label identical across rng seeds and repeat calls on 500 fixed states",
          first == second == third)


def test_backstop_knobs_restore_v1() -> None:
    sample = state_sample(2000, seed=99)
    pool = dict(OPPONENT_POOL)
    os.environ.update(_V1_ENV)
    try:
        env_ok = all(card(choose_action(*_pid0(obs(p, h)))) == v1_label(p, h) for p, h in sample)
    finally:
        clear_knobs()
    check("env knobs CONCEDE_RANK=0 SKIM_MAX=3 OPEN_MARGIN=0 == the frozen v1 oracle on 2000 fixed states",
          env_ok)
    pool_ok = all(card(pool["up2abs3"](*_pid0(obs(p, h)), None)) == v1_label(p, h) for p, h in sample)
    check("pool 'up2abs3' == the frozen v1 oracle on the same 2000 states", pool_ok)
    saved = (ge._CONCEDE_RANK, ge._SKIM_MAX_PRIZE, ge._OPEN_MARGIN)
    try:
        ge._CONCEDE_RANK, ge._SKIM_MAX_PRIZE, ge._OPEN_MARGIN = 0, 3, 0
        const_ok = all(card(choose_action(*_pid0(obs(p, h)))) == v1_label(p, h) for p, h in sample)
    finally:
        ge._CONCEDE_RANK, ge._SKIM_MAX_PRIZE, ge._OPEN_MARGIN = saved
    check("in-code constants (_CONCEDE_RANK=0, _SKIM_MAX_PRIZE=3, _OPEN_MARGIN=0) == the v1 oracle", const_ok)
    differs = sum(card(choose_action(*_pid0(obs(p, h)))) != v1_label(p, h) for p, h in sample)
    check(f"default v2 differs from v1 on part of the sample ({differs}/2000 states)", 0 < differs < 2000)


def test_variant_knobs() -> None:
    os.environ["GOOFSPIEL_CONCEDE_RANK"] = "2"
    check("k2 knob: unbeatable prize 12, hand [1,3,5,9] -> 3rd-lowest 5", skim(12, [1, 3, 5, 9]) == 5)
    clear_knobs()
    os.environ["GOOFSPIEL_OPEN_MARGIN"] = "0"
    check("OPEN_MARGIN=0: 13 cards, prize 5 -> two-up 7 (no opening three-up)", skim(5, list(range(1, 14))) == 7)
    check("OPEN_MARGIN=0 keeps the k3 concession: prize 12, [1,3,5,9] -> 9", skim(12, [1, 3, 5, 9]) == 9)
    clear_knobs()
    os.environ["GOOFSPIEL_CONCEDE_RANK"] = "garbage"
    check("garbage knob falls back to the default (no raise) -> 9", skim(12, [1, 3, 5, 9]) == 9)
    clear_knobs()
    o, legal = obs(12, [1, 3, 5, 9])
    check("explicit args override the defaults (concede_rank=1 -> 3)",
          card(_choose_action_skimup(o, legal, concede_rank=1)) == 3)


def test_default_dispatch() -> None:
    os.environ.pop("GOOFSPIEL_TEACHER", None)
    os.environ.pop("GOOFSPIEL_ONEUP", None)
    check("default mode is skimup", _teacher_mode() == "skimup")
    check("default constants are skim-up v2 k3 (skim <= 2, margin 2, open 3 at >= 9 cards, concede rank 3)",
          (ge._SKIM_MAX_PRIZE, ge._SKIM_MARGIN, ge._OPEN_MARGIN, ge._OPEN_HAND, ge._CONCEDE_RANK)
          == (2, 2, 3, 9, 3))
    o, legal = obs(2, list(range(1, 9)))
    check("default dispatch skims prize 2 -> 1", card(choose_action(o, 0, legal)) == 1)
    o, legal = obs(5, list(range(1, 9)))
    check("default dispatch prize 5 -> 7", card(choose_action(o, 0, legal)) == 7)
    o, legal = obs(12, [1, 3, 5, 9])
    check("default dispatch concedes the 4th-lowest card -> 9", card(choose_action(o, 0, legal)) == 9)
    os.environ["GOOFSPIEL_TEACHER"] = "matching"
    check("explicit matching still selectable", _teacher_mode() == "matching")
    os.environ.pop("GOOFSPIEL_TEACHER", None)
    check("unparsable prize -> concede lowest legal, never raises",
          choose_action("garbage without a point card", 0, [7, 3, 9]) == 3)


def test_never_illegal() -> None:
    rng = random.Random(1234)
    bad = 0
    for _ in range(1000):
        n = rng.choice((5, 8, 10, 13))
        hand = sorted(rng.sample(range(1, n + 1), rng.randint(1, n)))
        o, legal = obs(rng.randint(1, n), hand, player=rng.randint(0, 1))
        for aid in (choose_action(o, 0, legal), _choose_action_skimup(o, legal)):
            bad += aid not in legal
    check("skimup never illegal on 1000 random eval-shaped observations", bad == 0)


def test_opponent_pool() -> None:
    names = [name for name, _ in OPPONENT_POOL]
    check("opponent pool = 11 distinct rule bidders", len(names) == 11 and len(set(names)) == 11)
    check("opponent pool holds the v1 labels, the field-style rules and the noisy default",
          {"one-up", "two-up", "three-up", "up2abs3", "up3abs2", "strict-two-up", "one-up-tie", "overbid-deck",
           "matching", "random", "noisy-self"} == set(names))
    rng = random.Random(7)
    bad = 0
    for _ in range(300):
        deck = rng.choice((5, 8, 10, 13))
        hand = sorted(rng.sample(range(1, deck + 1), rng.randint(1, deck)))
        seq = "".join(f"{rng.choice((0, 1, -3))} " for _ in range(deck - len(hand)))
        o, legal = obs(rng.randint(1, deck), hand, player=rng.randint(0, 1), seq=seq)
        for _, policy in OPPONENT_POOL:
            bad += policy(o, 0, legal, rng) not in legal
    check("every pool bidder returns a legal id (300 states, history consistent with the deck)", bad == 0)
    o, legal = obs(5, list(range(1, 9)))
    pool = dict(OPPONENT_POOL)
    check("pool one-up/two-up/three-up bid 6/7/8 on prize 5",
          [card(pool[k](o, 0, legal, rng)) for k in ("one-up", "two-up", "three-up")] == [6, 7, 8])


def test_field_style_bidders() -> None:
    pool, rng = dict(OPPONENT_POOL), random.Random(3)

    def bid(name: str, prize: int, hand: list[int], seq: str = "") -> int:
        o, legal = obs(prize, hand, seq=seq)
        return card(pool[name](o, 0, legal, rng))

    check("strict-two-up: prize 5, 8 cards -> 7", bid("strict-two-up", 5, list(range(1, 9))) == 7)
    check("strict-two-up: no card >= 9 -> concede lowest 2 (no one-up fallback)",
          bid("strict-two-up", 7, [2, 6, 8]) == 2)
    check("one-up-tie: prize 1 -> the 1", bid("one-up-tie", 1, [1, 2, 5]) == 1)
    check("one-up-tie: prize 5 -> cheapest card above it 6", bid("one-up-tie", 5, list(range(1, 9))) == 6)
    check("one-up-tie: prize 13, holds the 13 -> ties with 13", bid("one-up-tie", 13, [1, 4, 13]) == 13)
    check("one-up-tie: unbeatable, no equal card -> lowest 1", bid("one-up-tie", 12, [1, 4, 9]) == 1)
    check("overbid-deck: 13 cards, prize 1 -> cheapest >= 1+3 = 4",
          bid("overbid-deck", 1, list(range(1, 14))) == 4)
    check("overbid-deck: 5 cards, prize 2 -> cheapest >= 2+1 = 3", bid("overbid-deck", 2, list(range(1, 6))) == 3)
    check("overbid-deck: deck size read from hand + win sequence (10 in hand, 3 played -> 13: prize 4 -> 7)",
          bid("overbid-deck", 4, list(range(1, 11)), seq="0 1 -3 ") == 7)
    check("overbid-deck: same hand, no history (a 10-card deck: prize 4 -> 6)",
          bid("overbid-deck", 4, list(range(1, 11))) == 6)
    check("overbid-deck: prize 13, full 13-card hand -> 3rd-lowest 3",
          bid("overbid-deck", 13, list(range(1, 14))) == 3)
    check("overbid-deck: unbeatable with < 5 cards -> the highest 9", bid("overbid-deck", 12, [1, 4, 9]) == 9)
    check("overbid-deck: never the lowest card (5 cards, prize 1: +1 would pick the lowest 3) -> 4",
          bid("overbid-deck", 1, [3, 4, 5, 6, 7]) == 4)


def test_meta_sim_render() -> None:
    """The meta sim renders the observation exactly as pyspiel does (strings captured from the real game)."""
    import verify_goofspiel_meta as meta
    check("meta sim render == pyspiel text (P0, a tied round then a win)",
          meta.render_state(0, 4, [1, 3, 6, 7, 8], (5, 0), [2, 3, 5, 6, 7, 8], [-3, 0])
          == "You are Player 0 (P0).\nCurrent player: 0\nCurrent point card: 4\nRemaining Point Cards: 13678\n"
             "Points: 5 0 \nP0 hand: 2 3 5 6 7 8 \nWin sequence: -3 0 \n")
    check("meta sim render == pyspiel text (P1, opening state)",
          meta.render_state(1, 2, [1, 3, 4, 5, 6, 7, 8], (0, 0), list(range(1, 9)), [])
          == "You are Player 1 (P1).\nCurrent player: 1\nCurrent point card: 2\nRemaining Point Cards: 1345678\n"
             "Points: 0 0 \nP1 hand: 1 2 3 4 5 6 7 8 \nWin sequence: \n")


def test_teacher_level_head_to_head() -> None:
    """Seeded teacher-level check on the eval-rules sim (400 seat-swapped games over 5/8/10/13 cards, no
    noise): skim-up v2 beats the v1 label head-to-head and holds >= .85 vs a strict two-up bidder."""
    import verify_goofspiel_meta as meta
    pol = meta.policies()
    vs_v1 = meta.rate(pol["default"], pol["up2abs3"], games=400, eps=0.0, seed=1)
    vs_strict = meta.rate(pol["default"], pol["strict-two-up"], games=400, eps=0.0, seed=1)
    check(f"teacher: skim-up v2 vs up2abs3 = {vs_v1:.3f} > 0.5 (400 games, seed 1)", vs_v1 > 0.5)
    check(f"teacher: skim-up v2 vs strict-two-up = {vs_strict:.3f} >= 0.85 (400 games, seed 1)", vs_strict >= 0.85)
    again = meta.rate(pol["default"], pol["up2abs3"], games=400, eps=0.0, seed=1)
    check("teacher-level rate is reproducible for a fixed seed", again == vs_v1)


def test_meta_gate() -> None:
    """Teacher-level regression gate (pure-python eval-rules sim, both seats with student-style noise)."""
    import verify_goofspiel_meta as meta
    for name, rate, floor in meta.gate_checks(games=240):
        check(f"meta: default vs {name} = {rate:.3f} >= {floor:.2f}", rate >= floor)


if __name__ == "__main__":
    for knob in ("GOOFSPIEL_TEACHER", "GOOFSPIEL_ONEUP") + _V2_KNOBS:
        os.environ.pop(knob, None)
    test_twoup_backstop()
    test_v2_skim_branch()
    test_v2_open_three_up_branch()
    test_v2_two_up_branch()
    test_v2_one_up_fallback_branch()
    test_v2_k3_concession_branch()
    test_v2_short_hand_concession()
    test_v2_label_reads_only_prize_and_hand()
    test_determinism()
    test_backstop_knobs_restore_v1()
    test_variant_knobs()
    test_default_dispatch()
    test_never_illegal()
    test_opponent_pool()
    test_field_style_bidders()
    test_meta_sim_render()
    test_teacher_level_head_to_head()
    test_meta_gate()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
