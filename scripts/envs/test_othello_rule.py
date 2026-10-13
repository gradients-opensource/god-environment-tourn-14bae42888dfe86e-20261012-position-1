"""Offline tests for the deterministic othello rule label (othello_rule, the default othello teacher)
and the endgame fallback of the search expert it replaces (othello_expert, OTHELLO_TEACHER=expert).

Rule: corner first, X then C veto beside an EMPTY corner (lifted once either colour holds it), IAGO
table, lowest id on ties; pass-only -> pass; always legal; identical across 8 processes; never an
X/C square beside an empty corner at 11-16 empties while a safer move exists. Expert: when the exact
endgame solve runs out of time it labels the table's best move, not the lowest legal id.

Pure python (no open_spiel). Run from scripts/envs:  python test_othello_rule.py
"""
import os
import random
import sys
import time
import types
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Shim the `envs` package (skip its heavy __init__) but resolve real submodules from here.
_pkg = types.ModuleType("envs")
_pkg.__path__ = [_HERE]
sys.modules["envs"] = _pkg

import envs.othello_expert as oe  # noqa: E402
import envs.othello_rule as orl  # noqa: E402
from envs.othello_board import PASS_ACTION_ID, render_board  # noqa: E402
from envs.othello_minimax import _apply_move, _legal_moves, _WEIGHTS_DEFAULT  # noqa: E402

FAILS = []
_LABELS_PER_S_FLOOR = 2000.0  # one core; the rule is a few list scans (measured ~50k/s on the dev box)


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def _empty_board():
    return [["-"] * 8 for _ in range(8)]


def _set(board, sq, sym):
    board[sq // 8][sq % 8] = sym
    return board


def _raw(grid):
    """othello_minimax int grid (1 = x, -1 = o) -> parsed symbol board."""
    return [["x" if v == 1 else "o" if v == -1 else "-" for v in row] for row in grid]


def _ids(grid, color):
    return [r * 8 + c for r, c in _legal_moves(grid, color)]


def _field_positions(n_games, seed=0, eps=0.15):
    """Decisions with >= 2 legal moves from field-proxy games (positional rule with eps random moves on
    both seats, after 2-6 random opening plies like the eval): [(raw board, mover symbol, legal ids)]."""
    out = []
    for g in range(n_games):
        rng = random.Random(seed * 100_003 + g)
        grid = [[0] * 8 for _ in range(8)]
        grid[3][3] = grid[4][4] = -1
        grid[3][4] = grid[4][3] = 1
        color = 1
        for _ in range(rng.randint(2, 6)):
            ids = _ids(grid, color)
            if not ids:
                break
            a = rng.choice(ids)
            grid = _apply_move(grid, divmod(a, 8), color)
            color = -color
        passes = 0
        while passes < 2:
            ids = _ids(grid, color)
            if not ids:
                passes += 1
                color = -color
                continue
            passes = 0
            if len(ids) > 1:
                out.append((_raw(grid), "x" if color == 1 else "o", ids))
            a = rng.choice(ids) if rng.random() < eps else orl.positional_label(ids)
            grid = _apply_move(grid, divmod(a, 8), color)
            color = -color
    return out


POSITIONS = _field_positions(300)


def _empties(board):
    return sum(row.count("-") for row in board)


def _beside_empty_corner(board, a):
    corner = orl.X_SQUARES.get(a, orl.C_SQUARES.get(a))
    return corner is not None and board[corner // 8][corner % 8] == "-"


# --------------------------------------------------------------------------------- rule label


def test_corner_taken() -> None:
    b = _empty_board()
    check("legal corner beats a higher id and any edge", orl.rule_label(b, "x", [2, 5, 63, 20])[0] == 63)
    check("two legal corners -> the lower id", orl.rule_label(b, "x", [63, 7, 3])[0] == 7)
    ok = all(orl.rule_label(b_, me, ids)[0] == min(a for a in ids if a in orl.CORNERS)
             for b_, me, ids in POSITIONS if any(a in orl.CORNERS for a in ids))
    check("field positions: a legal corner is always the label", ok)


def test_x_then_c_veto() -> None:
    b = _empty_board()
    check("X square vetoed while a plain move exists", orl.rule_label(b, "x", [9, 19])[0] == 19)
    check("C square vetoed while a plain move exists", orl.rule_label(b, "x", [1, 27])[0] == 27)
    check("only X and C left -> the C square (X vetoed first)", orl.rule_label(b, "x", [9, 1])[0] == 1)
    check("only X squares -> the lowest id", orl.rule_label(b, "x", [54, 14])[0] == 14)
    label, reason = orl.rule_label(b, "x", [9, 19])
    check(f"reason names the veto ({reason})", reason == "avoid_x")


def test_relaxation_only_for_occupied_corner() -> None:
    # 49 is a1-side X square of corner 56, 6 is the C square of corner 7.
    empty = _empty_board()
    check("corner 56 empty: X 49 vetoed, C 6 kept as the only non-X", orl.rule_label(empty, "x", [49, 6])[0] == 6)
    for sym in ("x", "o"):
        held = _set(_empty_board(), 56, sym)
        check(f"corner 56 held by {sym}: X 49 no longer vetoed, C 6 is", orl.rule_label(held, "x", [49, 6])[0] == 49)
    held7 = _set(_empty_board(), 7, "o")
    check("corner 7 held: its C square 6 beats the vetoed X 49", orl.rule_label(held7, "x", [49, 6])[0] == 6)
    check("positional_label never relaxes", orl.positional_label([49, 6]) == 6)


def test_ties_lowest_id_and_order_invariance() -> None:
    b = _empty_board()
    check("equal weights (c1=10, f1=10) -> the lower id", orl.rule_label(b, "x", [5, 2])[0] == 2)
    ok = True
    for board, me, ids in POSITIONS[:2000]:
        shuffled = ids[:]
        random.Random(len(ids)).shuffle(shuffled)
        ok &= orl.rule_label(board, me, ids) == orl.rule_label(board, me, shuffled)
        top = max(_WEIGHTS_DEFAULT[a // 8][a % 8] for a in ids)
        label = orl.rule_label(board, me, ids)[0]
        if label not in orl.CORNERS and not any(a in orl.X_SQUARES or a in orl.C_SQUARES for a in ids):
            ok &= label == min(a for a in ids if _WEIGHTS_DEFAULT[a // 8][a % 8] == top)
    check("label independent of legal-id order, table max with lowest-id ties", ok)


def test_pass_and_legality() -> None:
    b = _empty_board()
    check("pass-only -> pass", orl.rule_label(b, "x", [PASS_ACTION_ID]) == (PASS_ACTION_ID, "pass"))
    check("empty legal set -> pass", orl.rule_label(b, "o", [])[0] == PASS_ACTION_ID)
    ok = all(orl.rule_label(board, me, ids)[0] in ids for board, me, ids in POSITIONS)
    check(f"label always in the legal set ({len(POSITIONS)} field positions)", ok)
    ok = all(orl.rule_label(board, "x", ids) == orl.rule_label(board, "o", ids) for board, _, ids in POSITIONS[:500])
    check("label does not depend on the mover's colour", ok)


def test_never_raises_and_fallback() -> None:
    try:
        a = orl.rule_label(None, "x", [9, 19, "bad", 64])
        b = orl.choose_action("garbage, no board here", 0, [49, 6])
        c = orl.rule_label([["-"] * 3], "x", [49, 6])
        ok = a[0] == 19 and b == orl.positional_label([49, 6]) and c[0] == orl.positional_label([49, 6])
    except Exception as exc:  # noqa: BLE001
        print(f"  raised {exc!r}")
        ok = False
    check("malformed board / ids degrade to the positional rule, never raise", ok)


def test_choose_action_reads_the_eval_observation() -> None:
    ok = True
    for board, me, ids in POSITIONS[:1500]:
        obs = (f"Current State:\nYou play {'x (Black)' if me == 'x' else 'o (White)'}.\n"
               f"{render_board(board, me)}\nYou are Player {0 if me == 'x' else 1}.\n"
               "Legal Actions:\n" + "\n".join(f"{a} -> {a}" for a in ids) + "\nYour choice (ID only):")
        ok &= orl.choose_action(obs, 0 if me == "x" else 1, ids) == orl.rule_label(board, me, ids)[0]
    check("choose_action(observation) == rule_label(parsed board)", ok)


def test_no_x_or_c_beside_empty_corner_at_11_16_empties() -> None:
    late = [(b, me, ids) for b, me, ids in POSITIONS if 11 <= _empties(b) <= 16]
    bad_x = bad_c = 0
    for board, me, ids in late:
        label = orl.rule_label(board, me, ids)[0]
        safe_from_x = [a for a in ids if not (a in orl.X_SQUARES and _beside_empty_corner(board, a))]
        safe_from_c = [a for a in safe_from_x if not (a in orl.C_SQUARES and _beside_empty_corner(board, a))]
        bad_x += label in orl.X_SQUARES and _beside_empty_corner(board, label) and bool(safe_from_x)
        bad_c += label in orl.C_SQUARES and _beside_empty_corner(board, label) and bool(safe_from_c)
    check(f"11-16 empties: never X beside an empty corner while a safer move exists "
          f"({len(late)} decisions, {bad_x} bad)", len(late) > 300 and bad_x == 0)
    check(f"11-16 empties: never C beside an empty corner while a safer move exists ({bad_c} bad)", bad_c == 0)


def _labels(chunk):
    return [orl.rule_label(b, me, ids) for b, me, ids in chunk]


def test_identical_labels_across_8_processes() -> None:
    sample = POSITIONS[:1600]
    ref = _labels(sample)
    with Pool(8) as pool:
        runs = pool.map(_labels, [sample] * 8)
    check("8 parallel processes produce identical labels", all(r == ref for r in runs))


def test_throughput() -> None:
    sample = POSITIONS[:5000]
    t = time.perf_counter()
    for board, me, ids in sample:
        orl.rule_label(board, me, ids)
    rate = len(sample) / max(time.perf_counter() - t, 1e-9)
    check(f"rule throughput {rate:.0f} labels/s >= {_LABELS_PER_S_FLOOR:.0f}", rate >= _LABELS_PER_S_FLOOR)


# --------------------------------------------------------------------- expert endgame fallback


def _grid(board, me):
    sym = {"x": 1, "o": -1} if me == "x" else {"x": -1, "o": 1}
    return [[sym.get(v, 0) for v in row] for row in board]


def test_expert_endgame_timeout_labels_the_table_best() -> None:
    """Force the root exact solve to time out on its first candidate (deadline already passed, clock
    checked on every node). The fallback must be the table's best move (= the positional rule), not
    the lowest legal id (the HEAD bug: best_move was taken before the weight sort)."""
    late = [(b, me, ids) for b, me, ids in POSITIONS if 12 <= _empties(b) <= 16]
    saved = oe._TIME_CHECK_INTERVAL
    oe._TIME_CHECK_INTERVAL = 1
    try:
        mismatches = lowest = 0
        for board, me, ids in late:
            move = oe.best_move_internal(_grid(board, me), time_budget=-1.0, endgame_empties=16)
            label = move[0] * 8 + move[1]
            mismatches += label != orl.positional_label(ids)
            lowest += label == min(ids) != orl.positional_label(ids)
    finally:
        oe._TIME_CHECK_INTERVAL = saved
    check(f"expert endgame timeout -> table-best move ({len(late)} decisions, {mismatches} mismatches, "
          f"{lowest} lowest-id)", len(late) > 200 and mismatches == 0)


def test_expert_solved_endgame_unchanged() -> None:
    """With time to finish, the root solve still returns a move with the optimal disc differential."""
    small = [(b, me, ids) for b, me, ids in POSITIONS if _empties(b) <= 7][:60]
    ok = True
    for board, me, ids in small:
        grid = _grid(board, me)
        move = oe.best_move_internal(grid, time_budget=30.0, endgame_empties=16)
        deadline = time.monotonic() + 30.0
        values = {a: -oe._endgame(_apply_move(grid, divmod(a, 8), 1), -1, -(10 ** 9), 10 ** 9, deadline, [0], {})
                  for a in ids}
        ok &= values[move[0] * 8 + move[1]] == max(values.values())
    check(f"expert solved endgame still optimal ({len(small)} positions <= 7 empties)", len(small) >= 30 and ok)


if __name__ == "__main__":
    for t in (test_corner_taken, test_x_then_c_veto, test_relaxation_only_for_occupied_corner,
              test_ties_lowest_id_and_order_invariance, test_pass_and_legality, test_never_raises_and_fallback,
              test_choose_action_reads_the_eval_observation, test_no_x_or_c_beside_empty_corner_at_11_16_empties,
              test_identical_labels_across_8_processes, test_throughput,
              test_expert_endgame_timeout_labels_the_table_best, test_expert_solved_endgame_unchanged):
        t()
    print(f"\n{'RESULT PASS' if not FAILS else 'RESULT FAIL: ' + str(len(FAILS)) + ' failure(s)'}")
    sys.exit(1 if FAILS else 0)
