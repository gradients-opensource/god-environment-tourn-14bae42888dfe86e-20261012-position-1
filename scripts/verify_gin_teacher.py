"""Diagnose whether the gin_rummy PIMC teacher actually parses pyspiel
observations and makes expert decisions — or silently falls back to
legal_action_ids[0] (which would make the gin_rummy SFT data near-random).

It replays a few teacher-vs-teacher gin_rummy games exactly the way
pvp_selfplay.play_selfplay_game does (same _load_game / config / observation /
_teacher_obs), then at every decision node checks the teacher's own parsers:

  * parse_hand           -> did it see our hand? (empty => can't think)
  * parse_legal_actions  -> did it parse the offered actions? (empty => instant
                            fallback to legal_action_ids[0])
  * parse_phase          -> did it read the game phase?
  * choose_action        -> did it pick something other than the first legal id?

Run INSIDE the training image (pyspiel is installed there), no rebuild needed:

    cd ~/env-destroy && git pull
    docker run --rm --entrypoint python \
      -v "$(pwd)/scripts:/workspace/scripts" \
      standalone-text-trainer /workspace/scripts/verify_gin_teacher.py
"""

from __future__ import annotations

import random

import pyspiel  # noqa: F401  (import guard — fails loudly if pyspiel is missing)

from envs.pvp_selfplay import (
    _config_id_for_seed,
    _format_gin_rummy,
    _load_game,
    _setup_initial_state,
    _teacher_obs,
)
from envs import gin_rummy_expert

ENV = "gin_rummy"
N_GAMES = 20
MAX_TURNS = 60


def _decision_nodes(seed: int):
    """Play one teacher-vs-teacher game; yield (obs, player_id, legal_ids, labels)
    at each non-chance decision node, exactly as generation feeds the teacher."""
    rng = random.Random(seed)
    config_id = _config_id_for_seed(seed, ENV)
    game = _load_game(ENV, config_id)
    state = game.new_initial_state()
    _setup_initial_state(ENV, state, seed)

    nodes = []
    turns = 0
    while not state.is_terminal() and turns < MAX_TURNS:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = list(state.legal_actions(cp))
        if not legal_ids:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
        obs = _teacher_obs(_format_gin_rummy(state, cp), cp, labels)
        nodes.append((obs, cp, legal_ids, labels))

        # Advance the game using the teacher, so downstream states are realistic.
        try:
            action = gin_rummy_expert.choose_action(obs, cp, legal_ids, rng=rng)
        except Exception:
            action = rng.choice(legal_ids)
        if action not in legal_ids:
            action = rng.choice(legal_ids)
        state.apply_action(action)
        turns += 1
    return nodes


def main() -> None:
    total = 0
    hand_ok = 0
    legal_ok = 0
    phase_ok = 0
    chose_nonzero = 0
    printed_sample = False

    for g in range(N_GAMES):
        for obs, cp, legal_ids, labels in _decision_nodes(seed=1000 + g):
            total += 1
            hand = gin_rummy_expert.parse_hand(obs, cp)
            parsed_legal = gin_rummy_expert.parse_legal_actions(obs)
            phase = gin_rummy_expert.parse_phase(obs)
            choice = gin_rummy_expert.choose_action(obs, cp, legal_ids, rng=random.Random(cp))

            parsed_ids = {a for a, _ in parsed_legal}
            if hand:
                hand_ok += 1
            if legal_ids and len(set(legal_ids) & parsed_ids) == len(legal_ids):
                legal_ok += 1
            if phase:
                phase_ok += 1
            if len(legal_ids) > 1 and choice != legal_ids[0]:
                chose_nonzero += 1

            if not printed_sample:
                printed_sample = True
                print("=" * 72)
                print("SAMPLE — exactly what the teacher is fed at one decision node:")
                print("-" * 72)
                print(obs[:1000])
                print("-" * 72)
                print("Raw legal-action labels (from state.action_to_string):")
                for a, lbl in labels[:6]:
                    print(f"    {a} -> {lbl!r}")
                print("-" * 72)
                print(f"parse_hand           -> {len(hand)} cards: {hand}")
                print(f"parse_legal_actions  -> parsed {len(parsed_legal)} ids "
                      f"(actual legal: {len(legal_ids)})")
                print(f"parse_phase          -> {phase!r}")
                print("=" * 72)

    if total == 0:
        print("No decision nodes collected — check _load_game / setup for gin_rummy.")
        return

    def pct(x):
        return 100.0 * x / total

    print(f"\nDecision nodes analyzed: {total}")
    print(f"  parse_hand non-empty      : {hand_ok}/{total}  ({pct(hand_ok):.0f}%)")
    print(f"  parse_legal fully matched : {legal_ok}/{total}  ({pct(legal_ok):.0f}%)")
    print(f"  parse_phase non-empty     : {phase_ok}/{total}  ({pct(phase_ok):.0f}%)")
    print(f"  teacher chose != first id : {chose_nonzero}/{total}  ({pct(chose_nonzero):.0f}%)")
    print()

    hand_pct, legal_pct = pct(hand_ok), pct(legal_ok)
    if hand_pct < 20 or legal_pct < 20:
        print("VERDICT: BROKEN  (silent fallback)")
        print("  The teacher can't parse the hand and/or legal actions, so choose_action")
        print("  returns legal_action_ids[0] almost every move => gin_rummy SFT data is")
        print("  near-random. Rebalancing data will NOT help. Fix the parser (align")
        print("  gin_rummy_expert's regexes to the pyspiel observation format shown above)")
        print("  BEFORE retraining.")
    elif hand_pct > 80 and legal_pct > 80:
        print("VERDICT: WORKS")
        print("  Parsing succeeds and PIMC is thinking => gin_rummy SFT data is")
        print("  expert-quality. The remaining lever is data BALANCE (raise gin_rummy's")
        print("  ~19% share). No teacher fix needed.")
    else:
        print("VERDICT: PARTIAL")
        print("  Parsing works only some of the time. Inspect the SAMPLE above to see")
        print("  which formats slip through, and tighten the parser before retraining.")


if __name__ == "__main__":
    main()
