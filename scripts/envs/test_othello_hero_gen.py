"""Hero-seat othello generation (envs/othello_hero.py) vs the MCTS / noisy-positional opponent pool.

Checks the rows the student trains on: one recorded seat per game (seed % 2) with about half the plies,
every label equal to the rule teacher on that row's own observation, the opponent kind and its strength
deterministic per seed, no crash when open_spiel's MCTS is missing, both outcomes kept, and the default
wiring (rule teacher, hero-seat dispatch) with its in-code backstops.

Needs open_spiel (skips otherwise). Run from scripts/envs:  python test_othello_hero_gen.py
"""
import json
import os
import random
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

import envs.othello_hero as oh  # noqa: E402
import envs.othello_rule as orl  # noqa: E402
import envs.pvp_selfplay as sp  # noqa: E402

FAILS = []
SEEDS = range(400_000_000, 400_000_064)
MAX_TURN = 70


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


GAMES: dict = {}  # seed -> views, filled once in __main__ under the default teacher


def test_one_recorded_seat() -> None:
    ok = True
    for seed, views in GAMES.items():
        ok &= len(views) == 1
        rows, score = views[0]
        ok &= score in (0.0, 0.5, 1.0) and {parse_row(r)[1] for r in rows} == {seed % 2}
    check("one view per game, every row from the hero seat (seed % 2)", ok)


def test_about_half_the_plies() -> None:
    per_game = [len(views[0][0]) for views in GAMES.values()]
    mean = sum(per_game) / len(per_game)
    check(f"hero rows ~ half of a ~56-ply game (mean {mean:.1f}, max {max(per_game)})",
          24 <= mean <= 33 and max(per_game) <= 36)


def test_row_shape() -> None:
    ok = all(r["messages"][2]["content"] is None and len(r["messages"][2]["tool_calls"]) == 1
             and r["messages"][2]["tool_calls"][0]["function"]["name"] == "game_action"
             and r["messages"][1]["content"].startswith("Current state:\nYou play ")
             for views in GAMES.values() for r in views[0][0])
    check("rows are one game_action tool call on the eval othello prompt", ok)


def test_labels_equal_rule() -> None:
    total = mismatches = passes = 0
    for views in GAMES.values():
        for row in views[0][0]:
            state_desc, pid, legal, label = parse_row(row)
            ids = [a for a, _ in legal]
            mismatches += label != orl.choose_action(sp._teacher_obs(state_desc, pid, legal), pid, ids)
            passes += ids == [orl.PASS_ACTION_ID]
            total += 1
    check(f"every label == rule(obs) ({total} rows, {passes} pass rows, {mismatches} mismatches)",
          total > 0 and mismatches == 0)


def test_opponent_pool_deterministic() -> None:
    seeds = range(400_100_000, 400_102_000)
    specs = [oh.opponent_spec(s) for s in seeds]
    check("opponent kind and strength identical on a second draw",
          specs == [oh.opponent_spec(s) for s in seeds])
    kinds = Counter(k for k, _ in specs)
    share = kinds["mcts"] / len(specs)
    check(f"pool ~{oh.OPP_MCTS_SHARE:.0%} MCTS ({share:.1%})", abs(share - oh.OPP_MCTS_SHARE) < 0.05)
    sims = [v for k, v in specs if k == "mcts"]
    eps = [v for k, v in specs if k == "positional"]
    check(f"MCTS sims in {oh.OPP_MCTS_SIMS} ({min(sims)}..{max(sims)})",
          (min(sims), max(sims)) == oh.OPP_MCTS_SIMS)
    check(f"positional eps in {oh.OPP_RULE_EPS} ({min(eps):.3f}..{max(eps):.3f})",
          oh.OPP_RULE_EPS[0] <= min(eps) < 0.02 and 0.18 < max(eps) <= oh.OPP_RULE_EPS[1])
    seen = Counter(oh.opponent_spec(s)[0] for s in GAMES)
    check(f"both kinds played in the sample {dict(seen)}", seen["mcts"] > 0 and seen["positional"] > 0)


def test_same_seed_same_rows() -> None:
    mcts_seed = next(s for s in SEEDS if oh.opponent_spec(s)[0] == "mcts")
    rule_seed = next(s for s in SEEDS if oh.opponent_spec(s)[0] == "positional")
    ok = all(oh.play_hero_game(s, MAX_TURN, random.Random(s)) == GAMES[s] for s in (mcts_seed, rule_seed))
    check("a seed replays to identical rows (MCTS and positional opponents)", ok)


def test_no_mcts_falls_back_to_positional() -> None:
    seed = next(s for s in SEEDS if oh.opponent_spec(s)[0] == "mcts")
    saved = oh._make_mcts_bot
    oh._make_mcts_bot = lambda *a, **k: None
    try:
        game = sp._load_game("othello", sp._config_id_for_seed(seed, "othello"))
        kind, _ = oh._opponent(game, seed)
        views = oh.play_hero_game(seed, MAX_TURN, random.Random(seed))
    except Exception as exc:  # noqa: BLE001
        print(f"  raised {exc!r}")
        kind, views = None, []
    finally:
        oh._make_mcts_bot = saved
    check(f"without open_spiel MCTS the opponent is positional ({kind}) and rows still come",
          kind == "positional" and len(views) == 1 and len(views[0][0]) > 0)


def test_both_outcomes_kept() -> None:
    outcomes = Counter(v[0][1] for v in GAMES.values())
    check(f"both outcomes kept (wins, losses, draws: {dict(outcomes)})", outcomes[1.0] > 0 and outcomes[0.0] > 0)


def test_default_wiring_and_backstops() -> None:
    check("rule is the default othello teacher", sp._teacher("othello") is orl.choose_action)
    os.environ["OTHELLO_TEACHER"] = "expert"
    try:
        expert = sp._teacher("othello")
    finally:
        del os.environ["OTHELLO_TEACHER"]
    check("OTHELLO_TEACHER=expert selects the search expert", expert is not orl.choose_action
          and expert.__code__.co_filename.endswith("pvp_selfplay.py"))
    check("othello is a hero-seat env by default", "othello" in sp._HERO_SEAT_DEFAULT_ENVS)
    seed = next(iter(GAMES))
    check("make_selfplay_generator('othello') plays the hero game",
          sp.make_selfplay_generator("othello")(seed, None, MAX_TURN) == GAMES[seed])
    saved = sp._HERO_SEAT_DEFAULT_ENVS
    sp._HERO_SEAT_DEFAULT_ENVS = ("clobber",)
    try:
        views = sp.make_selfplay_generator("othello")(seed, None, MAX_TURN)
    finally:
        sp._HERO_SEAT_DEFAULT_ENVS = saved
    check("dropping othello from _HERO_SEAT_DEFAULT_ENVS restores both-seat self-play", len(views) == 2)


if __name__ == "__main__":
    GAMES.update({s: oh.play_hero_game(s, MAX_TURN, random.Random(s)) for s in SEEDS})
    for t in (test_one_recorded_seat, test_about_half_the_plies, test_row_shape, test_labels_equal_rule,
              test_opponent_pool_deterministic, test_same_seed_same_rows, test_no_mcts_falls_back_to_positional,
              test_both_outcomes_kept, test_default_wiring_and_backstops):
        t()
    print(f"\n{'RESULT PASS' if not FAILS else 'RESULT FAIL: ' + str(len(FAILS)) + ' failure(s)'}")
    sys.exit(1 if FAILS else 0)
