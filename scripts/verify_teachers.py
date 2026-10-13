"""Audit every game env's expert teacher by the only metric that matters:
does it actually beat a random player?

For each env we play the teacher vs a uniform-random opponent (seats swapped
each game to cancel first-move advantage) and report the teacher's win rate.

  * A genuinely strong teacher crushes random (win rate well above 50%).
  * A teacher that silently falls back to legal_action_ids[0] (broken parser,
    unbuilt blueprint, wrong observation format, ...) plays ~random and lands
    near 50% — a red flag that its SFT data is low quality.

This is stronger than checking env-specific parse internals: it works uniformly
across every teacher and measures real playing strength.

Run INSIDE the training image (pyspiel + teachers installed), no rebuild:

    cd ~/env-destroy && git pull
    docker run --rm --entrypoint python \
      -w /workspace/scripts \
      -v "$(pwd)/scripts:/workspace/scripts" \
      standalone-text-trainer verify_teachers.py

Options:
    verify_teachers.py --games 40 --envs gin_rummy othello leduc_poker
NOTE: liars_dice builds a large MCCFR blueprint on first use (can take minutes);
it is omitted from the default set — add it explicitly with --envs if needed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

# Stream results live even when stdout is a pipe (docker run captures it, so Python
# would otherwise block-buffer and the run looks "stuck" until it finishes).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import pyspiel  # noqa: F401  (fail loudly if pyspiel is missing)

import trl_compat  # noqa: F401 — inject trl.experimental shim before env teacher imports

from envs.pvp_selfplay import (
    _FORMATTERS,
    _config_id_for_seed,
    _load_game,
    _score,
    _setup_initial_state,
    _teacher,
    _teacher_obs,
)

DEFAULT_ENVS = ["clobber", "gin_rummy", "goofspiel", "othello", "leduc_poker"]
MAX_TURNS = 300


def _act(policy, env: str, state, pid: int, rng: random.Random) -> int:
    """policy is either the string 'random' or a teacher choose_action fn."""
    legal = list(state.legal_actions(pid))
    if not legal:
        return 0
    if policy == "random":
        return rng.choice(legal)
    labels = [(a, state.action_to_string(pid, a)) for a in legal]
    obs = _teacher_obs(_FORMATTERS[env](state, pid), pid, labels)
    try:
        a = policy(obs, pid, legal, rng=rng)
    except Exception:
        a = rng.choice(legal)
    return a if a in legal else rng.choice(legal)


def _play(env: str, seed: int, teacher, teacher_seat: int):
    """One teacher-vs-random game. Returns (teacher_outcome in [0,1], chose_nonzero, decisions)."""
    rng = random.Random(seed)
    game = _load_game(env, _config_id_for_seed(seed, env))
    state = game.new_initial_state()
    _setup_initial_state(env, state, seed)

    pol = {teacher_seat: teacher, 1 - teacher_seat: "random"}
    chose_nonzero = decisions = 0
    turns = 0
    while not state.is_terminal() and turns < MAX_TURNS:
        if state.is_chance_node():
            outs, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(list(outs), weights=list(probs), k=1)[0])
            continue
        if state.is_simultaneous_node():
            joint = []
            for p in range(game.num_players()):
                legal = list(state.legal_actions(p))
                a = _act(pol[p], env, state, p, rng) if legal else 0
                if p == teacher_seat and len(legal) > 1:
                    decisions += 1
                    if a != legal[0]:
                        chose_nonzero += 1
                joint.append(a)
            state.apply_actions(joint)
            turns += 1
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal = list(state.legal_actions(cp))
        if not legal:
            break
        a = _act(pol[cp], env, state, cp, rng)
        if cp == teacher_seat and len(legal) > 1:
            decisions += 1
            if a != legal[0]:
                chose_nonzero += 1
        state.apply_action(a)
        turns += 1

    outcome = _score(state.returns(), teacher_seat) if state.is_terminal() else 0.5
    return outcome, chose_nonzero, decisions


def audit(env: str, n_games: int) -> None:
    print(f"\n### {env} " + "-" * (60 - len(env)))
    try:
        t0 = time.monotonic()
        teacher = _teacher(env)
        # Warm any lazy blueprint with one throwaway game so the timing below is play-only.
        _play(env, seed=0, teacher=teacher, teacher_seat=0)
        build_s = time.monotonic() - t0
    except Exception as exc:
        print(f"  ERROR building teacher: {exc!r}")
        return

    outcomes = []
    cnz = dec = 0
    t0 = time.monotonic()
    for g in range(n_games):
        o, c, d = _play(env, seed=1000 + g, teacher=teacher, teacher_seat=g % 2)
        outcomes.append(o)
        cnz += c
        dec += d
    play_s = time.monotonic() - t0

    win_rate = 100.0 * sum(outcomes) / len(outcomes)
    wins = sum(1 for o in outcomes if o > 0.5)
    draws = sum(1 for o in outcomes if o == 0.5)
    losses = sum(1 for o in outcomes if o < 0.5)
    think = 100.0 * cnz / dec if dec else 0.0

    print(f"  teacher vs random : {win_rate:.0f}%  (W{wins}/D{draws}/L{losses} over {n_games} games)")
    print(f"  chose != first id : {think:.0f}%   |  build {build_s:.1f}s, play {play_s:.1f}s")
    if win_rate >= 62:
        print("  VERDICT: STRONG — teacher clearly beats random; SFT data is good.")
    elif win_rate >= 55:
        print("  VERDICT: MODERATE — better than random but not dominant; usable, could improve.")
    else:
        print("  VERDICT: WEAK/SUSPECT — barely better than random. Likely silent fallback")
        print("           or a weak policy. This env's SFT data quality is doubtful.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    args = ap.parse_args()

    print(f"Auditing teachers (games/env={args.games}): {', '.join(args.envs)}")
    for env in args.envs:
        if env not in _FORMATTERS:
            print(f"\n### {env} — SKIP (no formatter; not a pvp game env)")
            continue
        audit(env, args.games)
    print("\nDone. STRONG = good data, WEAK/SUSPECT = investigate before trusting that env.")


if __name__ == "__main__":
    main()
