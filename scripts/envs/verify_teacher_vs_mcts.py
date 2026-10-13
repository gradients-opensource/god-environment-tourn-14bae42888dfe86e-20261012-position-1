"""Measure each env teacher's ABSOLUTE strength by pitting it head-to-head against
OpenSpiel's own MCTSBot — the bot FAMILY the validator uses as its scoring opponent
(generate_openspiel_experts.py records the validator's `"opponent": "mcts"`).

WHY THIS EXISTS (the R1 meta-lesson): our per-game verify scripts pit NEW-vs-OUR-OWN-greedy,
which sits at ~50% and gave FALSE confidence — we shipped and placed LAST in R1 because our
teachers were genuinely weaker than the field. Beating a strong OpenSpiel MCTS is a real,
UNBIASED strength signal, not a mirror match. This is the "strong external baseline" the
post-mortem demanded, and one harness covers every game.

It drives the EXACT teacher gen uses (pvp_selfplay._teacher) through the same observation
envelope (_format_* + _teacher_obs), so it measures the DEPLOYED teacher, not a reimplementation.

Signal quality by game type:
  * Perfect-info (othello, clobber) — CLEAN: MCTS plays the true game, so win-rate > 50%
    means our teacher genuinely out-searches a UCT bot. THIS is where we must WIN.
  * Imperfect-info (leduc_poker, liars_dice, gin_rummy) — MCTS searches the CONCRETE state,
    i.e. it CHEATS (sees hidden cards/dice). Treat its win-rate as a hard UPPER-BOUND
    opponent: staying close is good, a blowout is expected and inconclusive. For a FAIR read
    on these, use policy exploitability (a separate check) — they draw for everyone anyway.
  * Simultaneous (goofspiel) — needs a turn-based transform; skipped here with a clear notice.

CPU-only (pyspiel + open_spiel), run on the VM from scripts/:
    python envs/verify_teacher_vs_mcts.py --env clobber --games 40 --mcts-sims 400
    CLOBBER_LEAF_EVAL=mcts python envs/verify_teacher_vs_mcts.py --env clobber --games 40
    python envs/verify_teacher_vs_mcts.py --env othello --games 40 --mcts-sims 400
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import types

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

# Make envs.* importable when run as a file (python envs/verify_X.py puts scripts/envs on
# sys.path, not scripts/). Shim a lightweight `envs` pkg so the heavy envs/__init__.py is
# skipped — the same bootstrap the other verify scripts use. Must precede any `from envs` import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

from envs.pvp_selfplay import _teacher, _teacher_obs, _load_game, _FORMATTERS

# Perfect-information games where an MCTS baseline is a CLEAN (non-cheating) strength probe.
_PERFECT_INFO = {"othello", "clobber"}


def _make_mcts_bot(game: "pyspiel.Game", sims: int, seed: int,
                   rollouts: int = 1, uct_c: float = 2.0) -> "mcts.MCTSBot":
    """OpenSpiel MCTSBot with a random-rollout evaluator. Defaults (n_rollouts=1, uct_c=2.0) match
    generate_openspiel_experts.py / the validator's opponent family. rollouts + uct_c are exposed so
    we can measure a stronger-evaluator teacher (e.g. more rollouts to de-noise othello) head-to-head."""
    np_rng = np.random.RandomState(seed)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=rollouts, random_state=np_rng)
    return mcts.MCTSBot(game, uct_c=uct_c, max_simulations=sims, evaluator=evaluator,
                        random_state=np_rng)


def _teacher_move(env_name: str, state: "pyspiel.State", cp: int, teacher, rng: random.Random) -> int:
    """Ask our deployed teacher for a move through the exact obs envelope gen uses."""
    legal = state.legal_actions(cp)
    labels = [(a, state.action_to_string(cp, a)) for a in legal]
    obs = _teacher_obs(_FORMATTERS[env_name](state, cp), cp, labels)
    try:
        action = teacher(obs, cp, legal, rng=rng)
    except Exception:
        action = rng.choice(legal)
    return action if action in legal else rng.choice(legal)


def _play_one(env_name: str, teacher, game: "pyspiel.Game", sims: int,
              seed: int, teacher_first: bool, our_sims: int = 0,
              our_rollouts: int = 1, our_uct: float = 2.0) -> str:
    """One full game: our teacher vs a fresh MCTSBot. Returns 'teacher'/'mcts'/'draw'.
    If our_sims > 0, OUR side plays an OpenSpiel MCTSBot (our_sims / our_rollouts / our_uct) instead
    of the hand-written teacher — this measures the MCTS teacher we distil into the model. The
    OPPONENT stays at the eval-opponent defaults (n_rollouts=1, uct_c=2.0)."""
    state = game.new_initial_state()
    bot = _make_mcts_bot(game, sims, seed)
    our_bot = (_make_mcts_bot(game, our_sims, seed + 10_000_000, rollouts=our_rollouts, uct_c=our_uct)
               if our_sims > 0 else None)
    rng = random.Random(seed)
    teacher_seat = 0 if teacher_first else 1
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choices(list(outcomes), weights=list(probs))[0]))
            continue
        cp = state.current_player()
        if cp == teacher_seat:
            if our_bot is not None:
                state.apply_action(int(our_bot.step(state)))
            else:
                state.apply_action(_teacher_move(env_name, state, cp, teacher, rng))
        else:
            state.apply_action(int(bot.step(state)))
    returns = state.returns()
    if returns[teacher_seat] > returns[1 - teacher_seat]:
        return "teacher"
    if returns[teacher_seat] < returns[1 - teacher_seat]:
        return "mcts"
    return "draw"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Head-to-head: our env teacher vs OpenSpiel MCTSBot (the validator's opponent family).")
    p.add_argument("--env", required=True, choices=sorted(_FORMATTERS))
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--mcts-sims", type=int, default=400,
                   help="UCT simulations per MCTS move (raise for a harder baseline; lower for gin_rummy speed)")
    p.add_argument("--our-mcts-sims", type=int, default=0,
                   help="if >0, OUR side plays OpenSpiel MCTS at this sim count instead of the "
                        "hand-written teacher — measures the MCTS teacher we distil into the model "
                        "(also skips the teacher import, so it needs no trl)")
    p.add_argument("--our-mcts-rollouts", type=int, default=1,
                   help="n_rollouts for OUR MCTS teacher (>1 de-noises the value estimate; the "
                        "lever we test to strengthen othello). Opponent stays at 1 (eval opponent).")
    p.add_argument("--our-uct-c", type=float, default=2.0, help="uct_c exploration constant for OUR MCTS teacher")
    p.add_argument("--config-id", type=int, default=0, help="game config id passed to _load_game")
    args = p.parse_args()

    game = _load_game(args.env, args.config_id)
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        print(f"[SKIP] '{args.env}' is a SIMULTANEOUS-move game — MCTSBot needs a turn-based "
              f"transform, not supported here.\n       Measure it with policy exploitability instead.")
        return

    teacher = None if args.our_mcts_sims > 0 else _teacher(args.env)
    imperfect = args.env not in _PERFECT_INFO

    tally = {"teacher": 0, "mcts": 0, "draw": 0}
    for i in range(args.games):
        result = _play_one(args.env, teacher, game, args.mcts_sims,
                           seed=3000 + i, teacher_first=(i % 2 == 0), our_sims=args.our_mcts_sims,
                           our_rollouts=args.our_mcts_rollouts, our_uct=args.our_uct_c)
        tally[result] += 1

    decisive = tally["teacher"] + tally["mcts"]
    wr = 100 * tally["teacher"] / decisive if decisive else 0.0
    if args.our_mcts_sims > 0:
        tag = (f" [our teacher = OpenSpiel MCTS {args.our_mcts_sims} sims, "
               f"{args.our_mcts_rollouts} rollouts, uct_c {args.our_uct_c:g}]")
    else:
        leaf = os.getenv("CLOBBER_LEAF_EVAL", "").strip() if args.env == "clobber" else ""
        tag = f" [CLOBBER_LEAF_EVAL={leaf}]" if leaf else ""
    print(f"\n{args.env} teacher{tag} vs OpenSpiel MCTSBot ({args.mcts_sims} sims) — {args.games} games")
    print(f"  teacher wins: {tally['teacher']}   MCTS wins: {tally['mcts']}   draws: {tally['draw']}")
    print(f"  teacher win-rate of decisive games: {wr:.0f}%")
    if imperfect:
        print("  NOTE: imperfect-info env — MCTS searches the concrete state (it CHEATS: sees hidden\n"
              "  info), so this is a hard UPPER-BOUND opponent. Staying close = good; a blowout is\n"
              "  expected and inconclusive. Use exploitability for a fair read on this env.")
    if imperfect:
        verdict = ("[OK] holds up vs a cheating MCTS. ✅" if wr >= 40
                   else "[CHECK] crushed by cheating MCTS — confirm with exploitability, don't assume weak. 🟡")
    elif wr >= 55:
        verdict = "[STRONG] teacher clearly out-plays MCTS — this env is a weapon. ✅"
    elif wr >= 45:
        verdict = "[EVEN] teacher ~matches MCTS (coin-flip) — a draw env, not yet a win. 🟡"
    else:
        verdict = "[WEAK] MCTS out-plays our teacher — this env NEEDS strengthening. 🔴"
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
