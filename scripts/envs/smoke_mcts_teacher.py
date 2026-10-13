"""Smoke test the DEFAULT MCTS teacher for othello + clobber: confirm it actually drives
play_selfplay_game (not a silent fallback to the weak hand-written teacher), produces training
rows, and measure games/sec — i.e. whether it yields enough games in the time-boxed gen.

CPU-only (pyspiel + open_spiel). Run on the VM from scripts/:
    python envs/smoke_mcts_teacher.py
"""
import os
import sys
import time
import types

# Same bootstrap the verify scripts use: skip the heavy envs/__init__, resolve submodules here.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

from envs.pvp_selfplay import play_selfplay_game, _openspiel_mcts_teacher, _load_game


def main() -> None:
    budget_s = 300  # project game count for a 5-minute per-env gen window
    for env in ("clobber", "othello"):
        game = _load_game(env, 0)
        bot = _openspiel_mcts_teacher(env, game, seed=1)
        mode = "MCTS (default ON)" if bot is not None else "hand-written FALLBACK ⚠️"
        n, t0, rows, wins = 8, time.monotonic(), 0, 0
        for g in range(n):
            views = play_selfplay_game(env, seed=5000 + g, max_turn=400)
            rows += sum(len(ex) for ex, _ in views)
            wins += sum(1 for _, s in views if s == 1.0)
        dt = time.monotonic() - t0
        per = dt / n
        print(f"{env:8s} teacher={mode:22s} {n} games in {dt:5.1f}s ({per:4.2f}s/game) "
              f"rows={rows} winning-seats={wins}")
        print(f"         → ~{int(budget_s / per)} games / 5-min gen window\n")


if __name__ == "__main__":
    main()
