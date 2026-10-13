"""Offline test for the goofspiel local self-play SFT path (pvp_selfplay).

goofspiel is the first SIMULTANEOUS-move self-play env, so this checks the new
``is_simultaneous_node()`` branch in ``play_selfplay_game`` as well as the
teacher and row shape. Unlike the other self-play envs we can test this end to
end here: pyspiel (open_spiel) is pip-installable, so we run the REAL game.

This box's ``envs/__init__.py`` eagerly imports trl/torch, so we shim a fake
``envs`` package that skips the heavy __init__ but still resolves real
submodules (same trick as test_othello_gen.py), then import pvp_selfplay.

Run from ``scripts/``:
    uv run --no-project --with open_spiel --with pyyaml --with pydantic \
        python envs/test_goofspiel_gen.py
"""

import json as _json
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# --- Shim the `envs` package (skip the heavy __init__) -------------------------
_envs_pkg = types.ModuleType("envs")
_envs_pkg.__path__ = [_HERE]
sys.modules["envs"] = _envs_pkg

import envs.goofspiel_expert as ge          # noqa: E402
import envs.pvp_selfplay as sp              # noqa: E402
from envs.pvp_format import build_full_system_prompt  # noqa: E402

ENV = "goofspiel"


def _parse_user_prompt(user):
    """Parse a build_user_prompt() string into (state_desc, player_id, legal)."""
    assert user.startswith("Current state:\n"), user[:40]
    body = user[len("Current state:\n"):]
    state_desc, _, rest = body.partition("\n\nYou are Player ")
    pid_str, _, after = rest.partition(".\nLegal actions:\n")
    legal = []
    for line in after.splitlines():
        line = line.strip()
        if "->" in line:
            aid, _, lab = line.partition("->")
            legal.append((int(aid.strip()), lab.strip()))
    return state_desc, int(pid_str), legal


def _check_example(ex):
    """A flattened per-turn example: single game_action over its legal enum."""
    assert "messages" in ex and "tools" in ex, "missing messages/tools"
    msgs = ex["messages"]
    assert len(msgs) == 3, f"expected 3 messages, got {len(msgs)}"
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    tcs = msgs[2].get("tool_calls") or []
    assert len(tcs) == 1 and tcs[0]["function"]["name"] == "game_action", "not one game_action"
    chosen = _json.loads(tcs[0]["function"]["arguments"])["action_id"]
    _, pid, legal = _parse_user_prompt(msgs[1]["content"])
    return chosen, pid, [aid for aid, _ in legal]


def test_system_prompt_builds():
    spmt = build_full_system_prompt(ENV)
    assert "GOOFSPIEL" in spmt.upper(), "goofspiel rules missing from system prompt"
    print("[ok] system_prompt_builds: goofspiel rules present")


def test_basic_shape_legality_and_both_seats():
    views = sp.play_selfplay_game(ENV, seed=1, max_turn=20)
    assert isinstance(views, list) and views, "expected non-empty list of (examples, score) views"
    seats = set()
    total = 0
    for examples, score in views:
        assert 0.0 <= score <= 1.0, f"score out of range: {score}"
        assert examples, "a view must carry at least one example"
        for ex in examples:
            chosen, pid, legal_ids = _check_example(ex)
            assert chosen in legal_ids, f"labeled action {chosen} not in legal {legal_ids}"
            seats.add(pid)
            total += 1
    assert seats == {0, 1}, f"expected both seats, got {seats}"
    print(f"[ok] basic_shape_legality_and_both_seats: {total} examples, seats {sorted(seats)}")


def test_simultaneous_round_count():
    # 13-card game => up to 13 simultaneous rounds; each seat gets one row per
    # round it had a legal bid. Both seats should have the same number of rows,
    # and it should be > 1 (the loop actually advanced through simultaneous nodes).
    views = sp.play_selfplay_game(ENV, seed=5, max_turn=20)
    counts = {}
    for examples, _ in views:
        _, pid, _ = _check_example(examples[0])
        counts[pid] = len(examples)
    assert counts.get(0) == counts.get(1), f"seat row counts differ: {counts}"
    assert counts.get(0, 0) > 1, f"expected multiple simultaneous rounds, got {counts}"
    print(f"[ok] simultaneous_round_count: {counts[0]} rounds per seat")


def test_determinism():
    a = sp.play_selfplay_game(ENV, seed=11, max_turn=20)
    b = sp.play_selfplay_game(ENV, seed=11, max_turn=20)
    assert a == b, "generation is not deterministic for a fixed seed"
    print("[ok] determinism: identical output across two runs")


def test_teacher_matches_prize_when_greedy():
    # With the MATCHING teacher at temperature 0 and no epsilon, it must bid the card whose value
    # equals the current prize when that card is available. NOTE: the DEFAULT teacher is now 'overbid'
    # (a mixed sampler that deliberately does NOT match — that is the whole point of escaping the
    # matching trap), and the overbid kernel ignores _TEMPERATURE/_EPSILON, so this test explicitly
    # pins GOOFSPIEL_TEACHER=matching to exercise the greedy path it asserts on.
    import os
    import random
    obs = (
        "Current State:\n"
        "Current point card: 7\nP0 hand: 1 2 3 4 5 6 7 8\n"
        "You are Player 0.\n"
        "Legal Actions:\n"
        + "\n".join(f"{i} -> [P0]Bid: {i + 1}" for i in range(8))
        + "\nYour choice (ID only):"
    )
    legal = list(range(8))
    saved_mode = os.environ.get("GOOFSPIEL_TEACHER")
    saved_t, saved_e = ge._TEMPERATURE, ge._EPSILON
    try:
        os.environ["GOOFSPIEL_TEACHER"] = "matching"
        ge._TEMPERATURE, ge._EPSILON = 0.0, 0.0
        a = ge.choose_action(obs, 0, legal, rng=random.Random(0))
    finally:
        ge._TEMPERATURE, ge._EPSILON = saved_t, saved_e
        if saved_mode is None:
            os.environ.pop("GOOFSPIEL_TEACHER", None)
        else:
            os.environ["GOOFSPIEL_TEACHER"] = saved_mode
    # prize 7 -> card value 7 -> action id 6
    assert a == 6, f"greedy teacher should match prize 7 with action id 6, got {a}"
    print("[ok] teacher_matches_prize_when_greedy: bid the matching card")


def test_teacher_always_legal():
    import random
    # Prize present but the matching card already spent: still returns a legal id.
    obs = (
        "Current State:\nCurrent point card: 9\nP0 hand: 1 2 3\n"
        "You are Player 0.\nLegal Actions:\n0 -> [P0]Bid: 1\n1 -> [P0]Bid: 2\n2 -> [P0]Bid: 3\n"
    )
    for s in range(50):
        a = ge.choose_action(obs, 0, [0, 1, 2], rng=random.Random(s))
        assert a in (0, 1, 2), f"illegal action {a}"
    # No prize parseable -> uniform legal, still always legal.
    a = ge.choose_action("garbage with no point card", 0, [3, 7], rng=random.Random(1))
    assert a in (3, 7), f"illegal fallback action {a}"
    print("[ok] teacher_always_legal: legal under spent-card and unparseable obs")


if __name__ == "__main__":
    test_system_prompt_builds()
    test_basic_shape_legality_and_both_seats()
    test_simultaneous_round_count()
    test_determinism()
    test_teacher_matches_prize_when_greedy()
    test_teacher_always_legal()
    print("\nALL GOOFSPIEL SELF-PLAY TESTS PASSED")
