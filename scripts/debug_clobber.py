"""Find WHY the clobber teacher loses to random (35% in verify_teachers).

For a couple of mid-game clobber states it prints, side by side:
  * the real pyspiel board            -> str(state)
  * what the teacher is fed           -> _format_clobber(state, pid) (observation_string)
  * the board the teacher RECONSTRUCTS -> clobber_expert._parse_board(obs)
  * each legal action's label + the (from,to) grid cells the teacher maps it to
    -> action_to_string() + clobber_expert._label_to_indices()
  * the move the teacher finally picks

If the reconstructed board or the move mapping doesn't match the real board, the
greedy search is evaluating garbage -> that's the bug (parse / orientation).

Run (no rebuild):
  docker run --rm --entrypoint python -w /workspace/scripts \
    -v "$(pwd)/scripts:/workspace/scripts" \
    standalone-text-trainer -u debug_clobber.py
"""

from __future__ import annotations

import random
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import pyspiel  # noqa: F401
import trl_compat  # noqa: F401

from envs.pvp_selfplay import (
    _config_id_for_seed,
    _format_clobber,
    _load_game,
    _setup_initial_state,
    _teacher_obs,
)
from envs import clobber_expert

ENV = "clobber"


def dump(state, pid: int) -> None:
    print("\n--- REAL pyspiel board: str(state) ---")
    print(str(state))

    obs = _format_clobber(state, pid)
    print(f"\n--- observation_string fed to teacher (player {pid}) ---")
    print(obs)

    print("\n--- clobber_expert._parse_board() reconstruction ---")
    grid = rows = None
    try:
        grid, rows, cols = clobber_expert._parse_board(obs)
        print(f"parsed {rows}x{cols}:")
        for row in grid:
            print("    " + " ".join(row))
    except Exception as e:
        print(f"PARSE FAILED: {e!r}")

    legal = list(state.legal_actions(pid))
    labels = [(a, state.action_to_string(pid, a)) for a in legal]
    full_obs = _teacher_obs(obs, pid, labels)
    extracted = clobber_expert._extract_action_labels(full_obs)

    print(f"\n--- legal actions ({len(legal)}) + teacher's move mapping ---")
    for a, lbl in labels[:12]:
        ex = extracted.get(a)
        mapped = ""
        if rows:
            try:
                fr, to = clobber_expert._label_to_indices(ex if ex else lbl, rows)
                mapped = f"   maps to from={fr} to={to}"
            except Exception as e:
                mapped = f"   MAP FAIL: {e!r}"
        print(f"    id={a}  action_to_string={lbl!r}  extracted={ex!r}{mapped}")

    choice = clobber_expert.choose_action(full_obs, pid, legal, rng=random.Random(7))
    print(f"\n--- teacher picks id={choice} (label {dict(labels).get(choice)!r}) ---")


def main() -> None:
    for seed in (1, 2):
        print("=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)
        rng = random.Random(seed)
        game = _load_game(ENV, _config_id_for_seed(seed, ENV))
        state = game.new_initial_state()
        _setup_initial_state(ENV, state, seed)
        plies = 0
        while not state.is_terminal():
            if state.is_chance_node():
                outs, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(outs), weights=list(probs), k=1)[0])
                continue
            cp = state.current_player()
            if cp < 0:
                break
            if plies >= 4:            # dump a non-trivial mid-game position
                dump(state, cp)
                break
            state.apply_action(rng.choice(list(state.legal_actions(cp))))
            plies += 1
    print("\nDone.")


if __name__ == "__main__":
    main()
