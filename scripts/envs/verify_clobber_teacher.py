"""Verify the strengthened Clobber teacher actually plays BETTER — pit the NEW teacher
(depth-limited negamax with a mobility leaf eval + exact endgame, `clobber_endgame`)
head-to-head vs the OLD 1-ply greedy teacher (`clobber_expert`) that shipped the Round-2
result (clobber lost 46-66 / 33-53).

If NEW wins the clear majority, the fix is real: a stronger teacher generates stronger
clobber training data. This measures teacher STRENGTH directly (the thing we changed),
not just that the pipeline runs. Runs on pyspiel (CPU, no GPU) — run it on the VM.

A 5x6 Clobber start is a single fixed position and the game is deterministic, so every
game from the standard start would be identical. To get position variety we play a few
SEEDED random opening plies first, then hand BOTH teachers the same midgame position and
alternate which side each teacher takes, so neither gets a first-move/opening edge.

Run from scripts/:
    python envs/verify_clobber_teacher.py --games 40
    python envs/verify_clobber_teacher.py --games 40 --opening 6   # deeper random opening
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types

import pyspiel

# Make envs.* importable when run as a file (python envs/verify_X.py puts scripts/envs on
# sys.path, not scripts/). Shim a lightweight `envs` pkg so the real envs/__init__.py is
# skipped — the same bootstrap the offline unit tests use. Must precede any `from envs` import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

from envs.clobber_endgame import (  # teacher mode set by CLOBBER_LEAF_EVAL
    choose_action as new_choose,
    _MIDGAME_DEPTH,
    _LEAF_EVAL,
    _ROLLOUTS,
    _MCTS_SIMS,
    _MOVE_TIME_BUDGET,
)
from envs.clobber_expert import choose_action as old_choose        # 1-ply greedy (the old teacher)
from envs.pvp_selfplay import _format_clobber, _load_game, _teacher_obs


def _random_opening(state: "pyspiel.State", plies: int, rng: random.Random) -> None:
    """Play `plies` seeded-random legal moves so each seed starts from a different
    (still early) midgame position — the only way to get variety in a deterministic game."""
    for _ in range(plies):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(rng.choice(legal))


def _play_one(seed: int, new_first: bool, opening_plies: int) -> str:
    """Play one full Clobber game, NEW teacher vs OLD teacher, from a seeded random
    opening. Returns 'new'/'old'/'draw' from NEW's perspective."""
    game = _load_game("clobber", seed)
    state = game.new_initial_state()
    rng = random.Random(seed)
    _random_opening(state, opening_plies, rng)
    new_player = 0 if new_first else 1

    while not state.is_terminal():
        cp = state.current_player()
        if cp < 0:
            break
        legal = state.legal_actions(cp)
        if not legal:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal]
        obs = _teacher_obs(_format_clobber(state, cp), cp, labels)
        chooser = new_choose if cp == new_player else old_choose
        try:
            action = chooser(obs, cp, legal, rng=rng)
        except Exception:
            action = rng.choice(legal)
        if action not in legal:
            action = rng.choice(legal)
        state.apply_action(action)

    returns = state.returns()
    if returns[new_player] > returns[1 - new_player]:
        return "new"
    if returns[new_player] < returns[1 - new_player]:
        return "old"
    return "draw"


def main() -> None:
    p = argparse.ArgumentParser(description="Head-to-head: strengthened Clobber teacher vs the old greedy one.")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--opening", type=int, default=6,
                   help="seeded random opening plies before the teachers take over (position variety)")
    args = p.parse_args()

    tally = {"new": 0, "old": 0, "draw": 0}
    for i in range(args.games):
        tally[_play_one(seed=2000 + i, new_first=(i % 2 == 0), opening_plies=args.opening)] += 1

    decisive = tally["new"] + tally["old"]
    new_wr = 100 * tally["new"] / decisive if decisive else 0.0
    if _LEAF_EVAL == "rollout":
        mode = f"flat-MC rollout ({_ROLLOUTS}/move) + exact endgame"
    elif _LEAF_EVAL == "mcts":
        mode = f"UCT MCTS ({_MCTS_SIMS} sims, {_MOVE_TIME_BUDGET:g}s/move cap) + exact endgame leaves"
    elif _LEAF_EVAL == "mobility":
        mode = f"negamax depth {_MIDGAME_DEPTH} (mobility leaf) + exact endgame"
    else:
        mode = "greedy midgame + exact endgame (default)"
    print(f"\nClobber teacher head-to-head — {args.games} games (opening {args.opening} plies)")
    print(f"  NEW [CLOBBER_LEAF_EVAL={_LEAF_EVAL}]: {mode} — {tally['new']} wins")
    print(f"  OLD (1-ply greedy): {tally['old']} wins")
    print(f"  draws: {tally['draw']}   |   NEW win-rate of decisive games: {new_wr:.0f}%")
    if tally["new"] > tally["old"]:
        print("\n[PASS] NEW teacher is clearly stronger — the R2 clobber fix is real. ✅")
    else:
        print("\n[WEAK] NEW not clearly ahead — raise _MIDGAME_DEPTH (e.g. 6) / _SOLVE_MAX_PIECES and retry. ⚠️")


if __name__ == "__main__":
    main()
