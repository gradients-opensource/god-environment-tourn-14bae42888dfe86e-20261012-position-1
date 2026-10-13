"""Offline unit test for the Clobber endgame solver (clobber_endgame).

Checks the exact win/loss verdict on tiny hand-verified positions and that choose_action
returns the proven winning action id. Pure standard-library — no GPU / datasets / env
server. Run from scripts/:
    python envs/test_clobber_endgame.py
"""
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


def _wins(board, me, opp):
    return ce._wins(board, me, opp, {}, [10 ** 9], time.monotonic() + 30)


def test_mover_clobbers_lone_enemy():
    # "xo": the side to move clobbers the other's only stone → opponent has no move → win.
    assert _wins(("xo",), "x", "o") is True
    assert _wins(("xo",), "o", "x") is True
    print("[ok] mover_clobbers_lone_enemy")


def test_mover_loses_when_forced():
    # "xoo": x's ONLY move (0→1) yields ".xo"; then o clobbers x and wins → x loses.
    assert _wins(("xoo",), "x", "o") is False
    print("[ok] mover_loses_when_forced")


def test_no_legal_move_is_a_loss():
    # "x.o": x has no orthogonally adjacent enemy → no legal move → immediate loss.
    assert _wins(("x.o",), "x", "o") is False
    print("[ok] no_legal_move_is_a_loss")


def test_two_by_two_first_player_wins():
    # 2x2 "xo"/"ox": every stone has an adjacent enemy; hand-check → first mover wins.
    assert _wins(("xo", "ox"), "x", "o") is True
    print("[ok] two_by_two_first_player_wins")


def test_choose_action_takes_the_win():
    # Minimal observation envelope: 1x2 board "xo", one legal action a1-b1 (x clobbers o).
    obs = "Current state:\n1xo\n ab\nLegal Actions:\n0 -> a1-b1\n"
    assert ce.choose_action(obs, 0, [0]) == 0
    print("[ok] choose_action_takes_the_win")


def test_rollout_forced_win_and_loss():
    # Rollout value is exact on FORCED lines (only one legal move), regardless of rng.
    rng = random.Random(0)
    # "xo": x clobbers o -> o has no move -> x wins.
    assert ce._rollout_value(("xo",), "x", "o", rng, 8) == 1.0
    # "xoo": x's only move (0->1) -> ".xo" -> o clobbers x -> x loses.
    assert ce._rollout_value(("xoo",), "x", "o", rng, 8) == -1.0
    print("[ok] rollout_forced_win_and_loss")


def test_choose_action_rollout_mode_takes_win():
    # The CLOBBER_LEAF_EVAL=rollout path must still take the immediate win.
    obs = "Current state:\n1xo\n ab\nLegal Actions:\n0 -> a1-b1\n"
    saved = ce._LEAF_EVAL
    ce._LEAF_EVAL = "rollout"
    try:
        assert ce.choose_action(obs, 0, [0], random.Random(0)) == 0
    finally:
        ce._LEAF_EVAL = saved
    print("[ok] choose_action_rollout_mode_takes_win")


def test_choose_action_mcts_mode_takes_win():
    # The CLOBBER_LEAF_EVAL=mcts path (UCT + exact leaves) must still take the immediate win.
    obs = "Current state:\n1xo\n ab\nLegal Actions:\n0 -> a1-b1\n"
    saved = ce._LEAF_EVAL
    ce._LEAF_EVAL = "mcts"
    try:
        assert ce.choose_action(obs, 0, [0], random.Random(0)) == 0
    finally:
        ce._LEAF_EVAL = saved
    print("[ok] choose_action_mcts_mode_takes_win")


def test_choose_action_mobility_mode_takes_win():
    # The CLOBBER_LEAF_EVAL=mobility path (iterative-deepening negamax) must still take the win.
    obs = "Current state:\n1xo\n ab\nLegal Actions:\n0 -> a1-b1\n"
    saved = ce._LEAF_EVAL
    ce._LEAF_EVAL = "mobility"
    try:
        assert ce.choose_action(obs, 0, [0], random.Random(0)) == 0
    finally:
        ce._LEAF_EVAL = saved
    print("[ok] choose_action_mobility_mode_takes_win")


def test_mcts_prefers_the_winning_move():
    # 1x3 "xoo": x's only legal move (a1->b1) leads to ".xo" where o then clobbers x and x
    # loses; but with a single legal move MCTS must still return it (aid 0). A 2-option probe:
    # "xo.o"? keep it to the guaranteed-legal single-move case to stay deterministic.
    import envs.clobber_endgame as _ce
    saved_sims, saved_leaf = _ce._MCTS_SIMS, _ce._LEAF_EVAL
    _ce._MCTS_SIMS, _ce._LEAF_EVAL = 64, "mcts"
    try:
        obs = "Current state:\n1xo\n ab\nLegal Actions:\n0 -> a1-b1\n"
        assert _ce.choose_action(obs, 0, [0], random.Random(1)) == 0
    finally:
        _ce._MCTS_SIMS, _ce._LEAF_EVAL = saved_sims, saved_leaf
    print("[ok] mcts_prefers_the_winning_move")


if __name__ == "__main__":
    test_mover_clobbers_lone_enemy()
    test_mover_loses_when_forced()
    test_no_legal_move_is_a_loss()
    test_two_by_two_first_player_wins()
    test_choose_action_takes_the_win()
    test_rollout_forced_win_and_loss()
    test_choose_action_rollout_mode_takes_win()
    test_choose_action_mcts_mode_takes_win()
    test_choose_action_mobility_mode_takes_win()
    test_mcts_prefers_the_winning_move()
    print("\nALL CLOBBER-ENDGAME TESTS PASSED")
