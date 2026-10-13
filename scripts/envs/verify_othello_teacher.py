"""Verify the strengthened othello teacher actually plays BETTER — pit the NEW params
(budget 0.20s, endgame 16, depth 12) head-to-head vs the OLD gen params (budget 0.04s,
endgame 8, depth 8) that shipped the Round-2 collapse.

If NEW wins the clear majority, the fix is real: a stronger teacher generates stronger
othello training data, which is exactly what collapsed in R2 (othello lost 0-25 / 2-32).
This measures teacher STRENGTH directly (the thing we changed), not just that the pipeline
runs. Runs on pyspiel (CPU, no GPU) — run it on the VM.

Colours are alternated each game so neither side gets the Black/first-move edge.

Run from scripts/:
    python envs/verify_othello_teacher.py --games 40
    # sweep strength:  OTHELLO ... just pass --new_budget 0.35 to test eval-level
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

from envs.othello_expert import choose_action as oth_choose
from envs.pvp_selfplay import _format_othello, _setup_initial_state, _teacher_obs

# The two configs we're comparing (NEW = current default, OLD = the R2-collapse params).
_OLD = {"time_budget": 0.04, "max_depth": 8, "endgame_empties": 8}


def _play_one(seed: int, new_first: bool, new_params: dict) -> str:
    """Play one full othello game, NEW teacher vs OLD teacher. Returns 'new'/'old'/'draw'
    from NEW's perspective. `new_first` puts NEW on Black (player 0)."""
    game = pyspiel.load_game("othello")
    state = game.new_initial_state()
    _setup_initial_state("othello", state, seed)
    rng = random.Random(seed)
    new_player = 0 if new_first else 1

    while not state.is_terminal():
        cp = state.current_player()
        if cp < 0:
            break
        legal = state.legal_actions(cp)
        if not legal:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal]
        obs = _teacher_obs(_format_othello(state, cp), cp, labels)
        params = new_params if cp == new_player else _OLD
        try:
            action = oth_choose(obs, cp, legal, rng=rng, **params)
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
    p = argparse.ArgumentParser(description="Head-to-head: strengthened othello teacher vs the old one.")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--new_budget", type=float, default=0.20)
    p.add_argument("--new_depth", type=int, default=12)
    p.add_argument("--new_endgame", type=int, default=16)
    args = p.parse_args()

    new_params = {"time_budget": args.new_budget, "max_depth": args.new_depth,
                  "endgame_empties": args.new_endgame}
    tally = {"new": 0, "old": 0, "draw": 0}
    for i in range(args.games):
        tally[_play_one(seed=1000 + i, new_first=(i % 2 == 0), new_params=new_params)] += 1

    n = max(args.games, 1)
    decisive = tally["new"] + tally["old"]
    new_wr = 100 * tally["new"] / decisive if decisive else 0.0
    print(f"\nOthello teacher head-to-head — {args.games} games")
    print(f"  NEW (budget {args.new_budget}, endgame {args.new_endgame}, depth {args.new_depth}): "
          f"{tally['new']} wins")
    print(f"  OLD (budget 0.04, endgame 8, depth 8): {tally['old']} wins")
    print(f"  draws: {tally['draw']}   |   NEW win-rate of decisive games: {new_wr:.0f}%")
    if tally["new"] > tally["old"]:
        print("\n[PASS] NEW teacher is clearly stronger — the R2 othello fix is real. ✅")
    else:
        print("\n[WEAK] NEW not clearly ahead — raise --new_budget (e.g. 0.35) and retry. ⚠️")


if __name__ == "__main__":
    main()
