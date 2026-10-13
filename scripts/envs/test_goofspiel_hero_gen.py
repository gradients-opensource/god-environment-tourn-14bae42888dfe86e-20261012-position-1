"""Hero-seat goofspiel generation vs the rule-opponent pool (envs/goofspiel_hero.py), on the real eval game.

Checks the rows the student trains on: one recorded seat per game, labels always equal to the teacher on
that row's own observation (DART only changes the executed card), non-tied score states, every deck
size, both outcomes kept, legal bids from every field-style pool bidder on the real game, and the default
routing through make_selfplay_generator / play_vs_eval_game.

Needs open_spiel (skips otherwise). Run from scripts/envs:  python test_goofspiel_hero_gen.py
"""
import json
import os
import random
import re
import sys
import types
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

try:
    import pyspiel  # noqa: F401
except ImportError:
    print("SKIP: open_spiel not installed")
    sys.exit(0)

import envs.goofspiel_expert as ge  # noqa: E402
import envs.goofspiel_hero as gh  # noqa: E402
import envs.pvp_selfplay as sp  # noqa: E402

FAILS = []
SEEDS = range(1000, 1400)
FIELD_STYLE = ("strict-two-up", "one-up-tie", "overbid-deck")


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def parse_row(row: dict) -> tuple[str, int, list[tuple[int, str]], int]:
    """(state_desc, player_id, legal labels, labelled action id) from one SFT row."""
    user = row["messages"][1]["content"]
    body = user[len("Current state:\n"):]
    state_desc, _, rest = body.partition("\n\nYou are Player ")
    pid, _, after = rest.partition(".\nLegal actions:\n")
    legal = []
    for line in after.splitlines():
        if "->" in line:
            aid, _, label = line.partition("->")
            legal.append((int(aid.strip()), label.strip()))
    call = row["messages"][2]["tool_calls"][0]["function"]
    return state_desc, int(pid), legal, json.loads(call["arguments"])["action_id"]


def hand_of(state_desc: str) -> list[int]:
    return [int(x) for x in re.search(r"hand: ([\d ]+)", state_desc).group(1).split()]


GAMES: dict = {}  # seed -> views, filled once in __main__ under the default teacher


def test_one_recorded_seat() -> None:
    ok = True
    for seed, views in GAMES.items():
        ok &= len(views) == 1
        rows, score = views[0]
        ok &= score in (0.0, 0.5, 1.0) and {parse_row(r)[1] for r in rows} == {seed % 2}
    check("one view per game, every row from the hero seat (seed % 2)", ok)


def test_row_shape() -> None:
    ok = all(r["messages"][2]["content"] is None and len(r["messages"][2]["tool_calls"]) == 1
             and r["messages"][2]["tool_calls"][0]["function"]["name"] == "game_action"
             for views in GAMES.values() for r in views[0][0])
    check("rows are exactly one game_action tool call with content None", ok)


def test_labels_equal_teacher() -> None:
    total = mismatches = 0
    for views in GAMES.values():
        for row in views[0][0]:
            state_desc, pid, legal, label = parse_row(row)
            ids = [a for a, _ in legal]
            mismatches += label != ge.choose_action(sp._teacher_obs(state_desc, pid, legal), pid, ids)
            total += 1
    check(f"every label == teacher(obs) ({total} rows, {mismatches} mismatches)", total > 0 and mismatches == 0)


def test_points_diversity() -> None:
    rows = [parse_row(r)[0] for views in GAMES.values() for r in views[0][0]]
    moved = sum("Points: 0 0 " not in s for s in rows) / len(rows)
    check(f">= 50% of rows have Points != '0 0' ({moved:.1%})", moved >= 0.5)


def test_dart_changes_only_the_executed_card() -> None:
    steps = deviated = 0
    for views in GAMES.values():
        rows = [parse_row(r) for r in views[0][0]]
        for (desc, _, legal, label), (nxt, _, _, _) in zip(rows, rows[1:]):
            if len(legal) < 2:
                continue
            played = set(hand_of(desc)) - set(hand_of(nxt))
            steps += 1
            deviated += played != {label + 1}
    rate = deviated / max(steps, 1)
    check(f"DART executes a non-label card on ~{gh.GOOF_DART_EPS:.0%} of hero turns ({rate:.1%})",
          0.5 * gh.GOOF_DART_EPS <= rate <= 1.5 * gh.GOOF_DART_EPS)


def test_deck_balance_and_outcomes() -> None:
    decks = Counter(len(hand_of(parse_row(v[0][0][0])[0])) for v in GAMES.values())
    shares = {n: decks[n] / len(GAMES) for n in (5, 8, 10, 13)}
    check(f"deck sizes 5/8/10/13 each ~25% of games {shares}", all(0.18 <= s <= 0.32 for s in shares.values()))
    outcomes = Counter(v[0][1] for v in GAMES.values())
    check(f"both outcomes kept (wins, losses, draws: {dict(outcomes)})", outcomes[1.0] > 0 and outcomes[0.0] > 0)


def test_opponent_pool_coverage() -> None:
    names = Counter(gh.pick_opponent(seed)[0] for seed in range(20_000, 22_000))
    pool = len(ge.OPPONENT_POOL)
    check(f"all {pool} pool opponents drawn, each >= 70% of a uniform share {dict(names)}",
          len(names) == pool and min(names.values()) >= 0.7 * 2000 / pool)
    check("the field-style bidders are pool opponents", set(FIELD_STYLE) <= set(names))
    check("opponent draw is deterministic per seed", gh.pick_opponent(4242)[0] == gh.pick_opponent(4242)[0])


def _counting_draw(draw, tally: Counter):
    """pick_opponent wrapper that counts the drawn opponent's bids and the illegal ones (generation would
    silently replace an illegal bid with a random card, so the count is the only way to see it)."""
    def pick(seed: int):
        name, policy = draw(seed)

        def counted(obs: str, pid: int, legal: list[int], rng) -> int:
            aid = policy(obs, pid, legal, rng)
            tally["bids"] += 1
            tally["illegal"] += aid not in legal
            return aid
        return name, counted
    return pick


def test_field_style_opponents_on_real_game() -> None:
    """Each field-style bidder plays legal bids on real eval states at every deck size, and its games still
    give hero rows whose labels equal the teacher."""
    draw = gh.pick_opponent
    for name in FIELD_STYLE:
        seeds = [s for s in range(30_000, 34_000) if draw(s)[0] == name][:32]
        tally: Counter = Counter()
        gh.pick_opponent = _counting_draw(draw, tally)
        try:
            games = [gh.play_goofspiel_hero_game(s, max_turn=30) for s in seeds]
        finally:
            gh.pick_opponent = draw
        rows = [parse_row(r) for g in games if g for r in g[0][0]]
        decks = {len(hand_of(parse_row(g[0][0][0])[0])) for g in games if g}
        labels_ok = all(label == ge.choose_action(sp._teacher_obs(desc, pid, legal), pid, [a for a, _ in legal])
                        for desc, pid, legal, label in rows)
        check(f"{name}: {tally['bids']} bids on real eval states, {tally['illegal']} illegal; {len(rows)} hero "
              f"rows over decks {sorted(decks)}, labels == teacher",
              tally["bids"] > 0 and tally["illegal"] == 0 and len(rows) > 0 and decks == {5, 8, 10, 13}
              and labels_ok)


def test_determinism() -> None:
    check("seeded determinism (same seed -> identical rows and score)",
          gh.play_goofspiel_hero_game(1234, 30) == gh.play_goofspiel_hero_game(1234, 30))


def test_meta_sim_envelope_parity() -> None:
    import verify_goofspiel_meta as meta
    state_desc, pid, legal, _ = parse_row(GAMES[SEEDS[0]][0][0][3])
    check("verify_goofspiel_meta sim envelope == pvp_selfplay._teacher_obs",
          meta._envelope(state_desc, pid, legal) == sp._teacher_obs(state_desc, pid, legal))


def test_meta_sim_render_all_decks() -> None:
    """The sim engine's state text == pyspiel's on the opening bid of both seats at every eval deck size
    (two-digit cards included), so its policies read the same observation as the generator's."""
    import verify_goofspiel_meta as meta
    ok = True
    for config_id, n in enumerate(meta.DECKS):
        game = sp._load_game("goofspiel", config_id)
        state = game.new_initial_state()
        rng = random.Random(n)
        while not state.is_terminal() and state.current_player() != 1:
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
                continue
            pid = state.current_player()
            text = sp._FORMATTERS["goofspiel"](state, pid)
            prize = int(re.search(r"Current point card: (\d+)", text).group(1))
            ok &= text == meta.render_state(pid, prize, [c for c in range(1, n + 1) if c != prize], (0, 0),
                                            list(range(1, n + 1)), [])
            state.apply_action(state.legal_actions(pid)[0])
        text = sp._FORMATTERS["goofspiel"](state, 1)
        prize = int(re.search(r"Current point card: (\d+)", text).group(1))
        ok &= text == meta.render_state(1, prize, [c for c in range(1, n + 1) if c != prize], (0, 0),
                                        list(range(1, n + 1)), [])
    check("verify_goofspiel_meta render_state == pyspiel opening text for P0/P1 at 5/8/10/13 cards", ok)


def test_default_routing() -> None:
    os.environ.pop("GOOFSPIEL_OPP_MCTS", None)
    gen = sp.make_selfplay_generator("goofspiel")
    check("make_selfplay_generator('goofspiel') -> hero game by default",
          gen(1234, None, 30) == gh.play_goofspiel_hero_game(1234, 30))
    check("play_vs_eval_game('goofspiel') -> hero game",
          sp.play_vs_eval_game("goofspiel", 1235, 30) == gh.play_goofspiel_hero_game(1235, 30))
    saved = sp._GOOFSPIEL_HERO_POOL
    try:
        sp._GOOFSPIEL_HERO_POOL = False
        views = sp.make_selfplay_generator("goofspiel")(1234, None, 30)
        check("rollback constant False -> HEAD mirror self-play (both seats recorded)", len(views) == 2)
    finally:
        sp._GOOFSPIEL_HERO_POOL = saved


if __name__ == "__main__":
    os.environ.pop("GOOFSPIEL_TEACHER", None)
    GAMES.update({seed: gh.play_goofspiel_hero_game(seed, max_turn=30) for seed in SEEDS})
    test_one_recorded_seat()
    test_row_shape()
    test_labels_equal_teacher()
    test_points_diversity()
    test_dart_changes_only_the_executed_card()
    test_deck_balance_and_outcomes()
    test_opponent_pool_coverage()
    test_field_style_opponents_on_real_game()
    test_determinism()
    test_meta_sim_envelope_parity()
    test_meta_sim_render_all_decks()
    test_default_routing()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
