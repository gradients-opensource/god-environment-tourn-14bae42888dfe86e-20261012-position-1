"""Liar's Dice rows from games against a field of opponent styles (the default liars_dice generator).

The reader label (liar_dice_expert.choose_action) is a pure function of the visible state, so the rows
should cover the states the field reaches, not only the reader's mirror games: aggressive readers jump
to high bids early, non-readers raise on their own dice only, noisy students make moves no teacher makes.
Each game draws its non-hero seat from OPPONENT_POOL by seed. Half the pool is the reader itself (plain
self-play, both seats recorded); against every other style only the hero seat's reader-labelled rows are
recorded. The hero seat flips every pool cycle, so each style meets the reader in both seats.

LIARS_GEN=selfplay restores reader self-play for every game; rollback of the default: _GEN_DEFAULT.
"""

from __future__ import annotations

import os
import random

import envs.pvp_selfplay as sp
from envs import liar_dice_expert as ld

ENV_NAME = "liars_dice"

# Styles are liar_dice_expert.OPPONENT_KINDS; "reader" fills every other slot (>= 50% self-play).
OPPONENT_POOL = ("reader", "ev_sampled", "reader", "reader_u0_lo",
                 "reader", "reader_u0_hi", "reader", "ev_noisy")
_GEN_DEFAULT = "pool"


def pool_on() -> bool:
    return os.getenv("LIARS_GEN", _GEN_DEFAULT).strip().lower() != "selfplay"


def opponent_for(seed: int) -> tuple[str, int]:
    """(opponent style, hero seat) of game ``seed``."""
    n = len(OPPONENT_POOL)
    return OPPONENT_POOL[seed % n], (seed // n) % 2


def _legal_or_random(pick, legal_ids: list[int], rng: random.Random) -> int:
    try:
        action = pick()
    except Exception:  # noqa: BLE001 - a bad move must never stop generation
        return rng.choice(legal_ids)
    return action if action in legal_ids else rng.choice(legal_ids)


def play_liars_pool_game(seed: int, max_turn: int, rng: "random.Random | None" = None):
    """One game in the pvp_selfplay ``[(examples, score)]`` views shape: both seats for reader
    self-play, else the hero seat only (the opponent's moves are played, never recorded)."""
    if rng is None:
        rng = random.Random(seed)
    kind, hero = opponent_for(seed)
    if kind == "reader":
        return sp.play_selfplay_game(ENV_NAME, seed, max_turn, rng)

    choose_action = sp._teacher(ENV_NAME)
    state = sp._load_game(ENV_NAME, sp._config_id_for_seed(seed, ENV_NAME)).new_initial_state()
    rows: list[dict] = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cp = state.current_player()
        legal_ids = state.legal_actions(cp)
        labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
        state_desc = sp._format_liars_dice(state, cp)
        observation = sp._teacher_obs(state_desc, cp, labels)
        if cp == hero:
            action_id = _legal_or_random(lambda: choose_action(observation, cp, legal_ids, rng=rng),
                                         legal_ids, rng)
            sp._emit_rows(rows, ENV_NAME, state_desc, cp, labels, action_id)
        else:
            action_id = _legal_or_random(lambda: ld.opponent_action(kind, observation, legal_ids, rng),
                                         legal_ids, rng)
        state.apply_action(action_id)
        turns += 1

    score = sp._score(state.returns(), hero) if state.is_terminal() else 0.5
    return [(rows, score)] if rows else []


def generate(game_id: int, env_endpoint: "str | None", max_turn: int):
    """SFT generator signature (see sft_env_configs); ``env_endpoint`` is unused."""
    return play_liars_pool_game(game_id, max_turn, random.Random(game_id))
