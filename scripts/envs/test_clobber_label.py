"""Offline tests for the deterministic clobber label (clobber_label) and the hero-seat clobber
generator (clobber_hero) that records it.

Label: same position -> same move whatever the clock says, proven wins when the solve fits,
the swindle move in lost positions, action-id invariance, the node budget as a hard bound and
a throughput floor. Generator (needs pyspiel): one recorded seat per game, recorded label ==
label_move(board), K rows per game, deterministic per seed, both outcomes kept, and the
default wiring in pvp_selfplay / generate_trajectories.

Run from scripts/envs:
    python test_clobber_label.py
"""
import ast
import json
import os
import random
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Shim the `envs` package (skip its heavy __init__) but resolve real submodules from here.
_pkg = types.ModuleType("envs")
_pkg.__path__ = [_HERE]
sys.modules["envs"] = _pkg

import envs.clobber_endgame as ce  # noqa: E402
import envs.clobber_label as cl  # noqa: E402
from envs.clobber_expert import _label_to_indices, _parse_board  # noqa: E402

_SIZES = ((4, 5), (5, 5), (5, 6))
_THROUGHPUT_FLOOR = 8.0  # labels/s, one core, default budgets (measured ~20+ on the dev box)


def _stones(board):
    return sum(row.count("x") + row.count("o") for row in board)


def _positions(seed, n, lo, hi):
    """n random-play positions (board, me, opp) with lo..hi stones and >= 2 legal moves."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        rows, cols = rng.choice(_SIZES)
        board = tuple("".join("xo"[(r + c) % 2] for c in range(cols)) for r in range(rows))
        me, opp = "o", "x"
        target = rng.randint(lo, hi)
        while _stones(board) > target:
            moves = ce._legal_moves(board, me, opp)
            if not moves:
                break
            board = ce._apply(board, *rng.choice(moves), me)
            me, opp = opp, me
        if _stones(board) == target and len(ce._legal_moves(board, me, opp)) >= 2:
            out.append((board, me, opp))
    return out


def _wins(board, me, opp):
    return ce._wins(board, me, opp, {}, [10 ** 9], float("inf"))


def _label_of(move, rows):
    r, c, nr, nc = move
    return f"{chr(97 + c)}{rows - r}{chr(97 + nc)}{rows - nr}"


def _observation(board, me, ids):
    """Teacher envelope for `board` with `me` to move; ids[i] is the action id of legal move i."""
    rows, cols = len(board), len(board[0])
    grid = "\n".join(f"{rows - r}{row}" for r, row in enumerate(board)) + "\n " + "abcdefgh"[:cols]
    moves = ce._legal_moves(board, me, "x" if me == "o" else "o")
    lines = "\n".join(f"{aid} -> {_label_of(m, rows)}" for aid, m in zip(ids, moves))
    pid = 0 if me == "o" else 1
    return (f"Current State:\nYou play {'o (White)' if pid == 0 else 'x (Black)'}.\n{grid}\n"
            f"You are Player {pid}.\nLegal Actions:\n{lines}\nYour choice (ID only):"), moves, pid


def test_determinism_ignores_the_clock():
    positions = _positions(11, 8, 10, 17) + _positions(12, 8, 18, 28)
    first = [cl.label_move(b, me, opp) for b, me, opp in positions]
    saved = time.monotonic, time.time, time.perf_counter
    ticks = [0.0]

    def jumping():
        ticks[0] += 1e6
        return ticks[0]

    try:
        time.monotonic = time.time = time.perf_counter = jumping
        for _ in range(2):
            again = [cl.label_move(b, me, opp) for b, me, opp in positions]
            assert again == first, "label changed across calls / clock jumps"
    finally:
        time.monotonic, time.time, time.perf_counter = saved
    assert not hasattr(cl, "time"), "clobber_label must not import a clock"
    print("PASS determinism_ignores_the_clock: 16 positions x 3 calls identical under clock jumps")


def test_label_is_a_proven_win_when_one_exists():
    checked = won = 0
    for board, me, opp in _positions(21, 60, 8, 16):
        moves = ce._legal_moves(board, me, opp)
        winning = {m for m in moves if not _wins(ce._apply(board, *m, me), opp, me)}
        label = cl.label_move(board, me, opp)
        assert label in moves, f"illegal label {label}"
        if winning:
            won += 1
            assert label in winning, f"{board} {me}: label {label} not in winning set {winning}"
        checked += 1
    print(f"PASS label_is_a_proven_win_when_one_exists: {won}/{checked} won positions, label in W every time")


def test_winning_ties_use_board_features_not_order():
    # Among the proven wins the label is the first in tie-break order (fewest opponent replies,
    # most own moves, coordinates), whatever order the moves were generated in.
    for board, me, opp in _positions(23, 30, 8, 14):
        moves = ce._legal_moves(board, me, opp)
        winning = [m for m in moves if not _wins(ce._apply(board, *m, me), opp, me)]
        if not winning:
            continue

        def key(m):
            child = ce._apply(board, *m, me)
            return (len(ce._legal_moves(child, opp, me)), -len(ce._legal_moves(child, me, opp)), m)

        assert cl.label_move(board, me, opp) == min(winning, key=key)
    print("PASS winning_ties_use_board_features_not_order")


def test_action_id_invariance():
    rng = random.Random(31)
    for board, me, opp in _positions(32, 25, 10, 26):
        n = len(ce._legal_moves(board, me, opp))
        picks = set()
        for ids in (list(range(n)), rng.sample(range(1000, 5000), n), list(range(n))[::-1]):
            obs, moves, pid = _observation(board, me, ids)
            aid = cl.choose_action(obs, pid, list(ids))
            picks.add(moves[ids.index(aid)])
        assert len(picks) == 1, f"move depends on action ids: {picks}"
        assert picks == {cl.label_move(board, me, opp)}
    print("PASS action_id_invariance: 25 positions x 3 id relabellings -> same board move")


def test_swindle_in_a_lost_position():
    board, me, opp = (".o..", ".o.x", "xoxx", "o.ox"), "o", "x"
    moves = ce._legal_moves(board, me, opp)
    winning_replies = {}
    for m in moves:
        child = ce._apply(board, *m, me)
        assert _wins(child, opp, me), "fixture must be lost for the mover"
        winning_replies[m] = sum(1 for r in ce._legal_moves(child, opp, me)
                                 if not _wins(ce._apply(child, *r, opp), me, opp))
    best = min(winning_replies.values())
    assert list(winning_replies.values()).count(best) == 1
    label = cl.label_move(board, me, opp)
    assert winning_replies[label] == best, f"label {label} leaves {winning_replies[label]} winning replies"
    assert label != cl.greedy_move(board, me, opp), "fixture must separate swindle from 1-ply order"
    print(f"PASS swindle_in_a_lost_position: label {label} leaves {best} winning reply (others {sorted(winning_replies.values())})")


def test_node_budget_is_a_hard_bound():
    seen = []

    class _Recording(cl._Search):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

    saved = cl._Search
    cl._Search = _Recording
    try:
        dense = _positions(41, 6, 19, 28)
        sparse = _positions(42, 6, 14, 17)
        for budget in (1, 10, 100, 1000):
            for board, me, opp in dense + sparse:
                move = cl.label_move(board, me, opp, node_budget=budget, exact_node_budget=budget)
                assert move in ce._legal_moves(board, me, opp)
    finally:
        cl._Search = saved
    assert seen and all(s.nodes <= s.budget for s in seen), "node counter exceeded its budget"
    print(f"PASS node_budget_is_a_hard_bound: {len(seen)} searches, max nodes/budget "
          f"{max(s.nodes / s.budget for s in seen):.2f}")


def test_throughput_floor():
    positions = _positions(51, 60, 10, 28)
    t0 = time.perf_counter()
    for board, me, opp in positions:
        cl.label_move(board, me, opp)
    rate = len(positions) / (time.perf_counter() - t0)
    assert rate >= _THROUGHPUT_FLOOR, f"{rate:.1f} labels/s < floor {_THROUGHPUT_FLOOR}"
    print(f"PASS throughput_floor: {rate:.1f} labels/s at node budget {cl.NODE_BUDGET} "
          f"(floor {_THROUGHPUT_FLOOR})")


# --- hero-seat generator (pyspiel) ------------------------------------------------------

def _parse_row(row):
    user = row["messages"][1]["content"]
    body = user[len("Current state:\n"):]
    state_desc, _, rest = body.partition("\n\nYou are Player ")
    pid_str, _, after = rest.partition(".\nLegal actions:\n")
    legal = {}
    for line in after.splitlines():
        aid, _, label = line.partition("->")
        legal[int(aid.strip())] = label.strip()
    call = row["messages"][2]["tool_calls"][0]["function"]
    return state_desc, int(pid_str), legal, json.loads(call["arguments"])["action_id"]


def test_hero_rows_record_one_seat_with_the_label():
    import envs.clobber_hero as ch

    k = ch.LABELS_PER_GAME
    full = outcomes = 0
    scores = set()
    seeds = range(700_000_101, 700_000_125)
    for seed in seeds:
        views = ch.play_hero_game(seed, 60)
        assert len(views) == 1, "exactly one recorded seat per game"
        rows, score = views[0]
        scores.add(score)
        assert 1 <= len(rows) <= k
        full += len(rows) == k
        for row in rows:
            state_desc, pid, legal, aid = _parse_row(row)
            assert pid == seed % 2, "only the hero seat is recorded"
            assert len(legal) >= 2 and aid in legal
            grid, grid_rows, _ = _parse_board(state_desc)
            me, opp = ("o", "x") if pid == 0 else ("x", "o")
            (r, c), (nr, nc) = _label_to_indices(legal[aid], grid_rows)
            assert (r, c, nr, nc) == cl.label_move(grid, me, opp), "recorded label != label_move(board)"
        outcomes += 1
    # Fewer than K only when the hero had fewer than K real choices (short 4x5 games).
    assert full >= 0.75 * outcomes, f"only {full}/{outcomes} games gave K={k} rows"
    assert scores == {0.0, 1.0}, f"both outcomes must be kept, saw {scores}"
    print(f"PASS hero_rows_record_one_seat_with_the_label: {outcomes} games, {full} with K={k} rows, "
          f"outcomes {sorted(scores)}")


def test_hero_game_is_deterministic_per_seed():
    import envs.clobber_hero as ch

    for seed in (700_000_201, 700_000_202, 700_000_203):
        a = json.dumps(ch.play_hero_game(seed, 60))
        b = json.dumps(ch.play_hero_game(seed, 60))
        assert a == b, f"seed {seed} not deterministic"
    print("PASS hero_game_is_deterministic_per_seed")


def _clobber_gen_args():
    src = open(os.path.join(_HERE, "generate_trajectories.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_ENV_DEFAULT_ARGS":
            return ast.literal_eval(node.value)["clobber"]
    raise AssertionError("_ENV_DEFAULT_ARGS not found")


def test_default_wiring():
    import envs.clobber_hero as ch
    import envs.pvp_selfplay as sp

    assert "clobber" not in sp._MCTS_TEACHER_DEFAULT_ENVS, "MCTS label must be opt-in only"
    assert sp._teacher("clobber") is cl.choose_action
    assert "clobber" in sp._HERO_SEAT_DEFAULT_ENVS
    gen = sp.make_selfplay_generator("clobber")
    seed = 700_000_301
    assert json.dumps(gen(seed, None, 60)) == json.dumps(ch.play_hero_game(seed, 60))
    args = _clobber_gen_args()
    assert not args.get("sample_by_score") and not args.get("wins_only"), "clobber keeps both outcomes"
    assert args["num_games"] >= 12000
    print(f"PASS default_wiring: det label teacher, hero-seat generator, gen args {args}")


if __name__ == "__main__":
    tests = [
        test_determinism_ignores_the_clock,
        test_label_is_a_proven_win_when_one_exists,
        test_winning_ties_use_board_features_not_order,
        test_action_id_invariance,
        test_swindle_in_a_lost_position,
        test_node_budget_is_a_hard_bound,
        test_throughput_floor,
        test_hero_rows_record_one_seat_with_the_label,
        test_hero_game_is_deterministic_per_seed,
        test_default_wiring,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print("\nALL CLOBBER-LABEL TESTS PASSED" if not failed else f"\n{failed} CLOBBER-LABEL TEST(S) FAILED")
    sys.exit(1 if failed else 0)
