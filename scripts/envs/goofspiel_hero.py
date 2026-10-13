"""Goofspiel SFT generation: our teacher on ONE seat vs a seeded rule opponent.

Eval goofspiel is a simultaneous game wrapped turn-based. The MCTS-opponent generator does not fit it
(an MCTS bot moving second searches the real state, pending bid included), and mirrored self-play with a
deterministic teacher ties every round: 100% of the old goofspiel rows read "Points: 0 0" with both seats
recorded, so the student never saw a position where it was ahead or behind.

Here each game draws one opponent from ``goofspiel_expert.OPPONENT_POOL`` by seed and records only the
hero seat (``seed % 2``, mirroring the eval's seat swap). With probability ``GOOF_DART_EPS`` the hero's
EXECUTED card is swapped for another legal card while the row keeps the teacher label (DART, see
pvp_selfplay._dart_executed_action), so hands after a student slip are covered too. Every game is kept,
won or lost: the label is a function of the visible state, not of the outcome.
"""

from __future__ import annotations

import random

from envs.goofspiel_expert import OPPONENT_POOL
from envs.pvp_selfplay import (_FORMATTERS, _config_id_for_seed, _dart_executed_action, _emit_rows,
                               _load_game, _score, _teacher, _teacher_obs)

ENV = "goofspiel"
GOOF_DART_EPS = 0.1
_POOL_SALT = 0x600F5EA7  # decorrelates the opponent draw from the seed's deck size and seat parity


def pick_opponent(seed: int):
    """``(name, policy)`` of this game's pool opponent; deterministic per seed."""
    return random.Random(seed ^ _POOL_SALT).choice(OPPONENT_POOL)


def play_goofspiel_hero_game(seed: int, max_turn: int, rng: "random.Random | None" = None):
    """Play one eval-variant game (deck size from the seed's config id, random prize order). The teacher
    bids for seat ``seed % 2`` and each of its decisions becomes a row; the pool opponent's bids are not
    recorded. Returns ``[(rows, score)]`` like play_vs_eval_game, or ``[]`` if no row was produced."""
    if rng is None:
        rng = random.Random(seed)
    game = _load_game(ENV, _config_id_for_seed(seed, ENV))
    teacher = _teacher(ENV)
    _, opponent = pick_opponent(seed)
    hero = seed % 2
    state = game.new_initial_state()
    rows: list[dict] = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cp = state.current_player()
        legal_ids = state.legal_actions(cp) if cp >= 0 else []
        if not legal_ids:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
        state_desc = _FORMATTERS[ENV](state, cp)
        policy = teacher if cp == hero else opponent
        try:
            action_id = policy(_teacher_obs(state_desc, cp, labels), cp, legal_ids, rng)
        except Exception:  # noqa: BLE001 — a policy bug must never stop generation
            action_id = rng.choice(legal_ids)
        if action_id not in legal_ids:
            action_id = rng.choice(legal_ids)
        if cp == hero:
            _emit_rows(rows, ENV, state_desc, cp, labels, action_id)
            action_id = _dart_executed_action(action_id, legal_ids, GOOF_DART_EPS, rng)
        state.apply_action(action_id)
        turns += 1

    score = _score(state.returns(), hero) if state.is_terminal() else 0.5
    return [(rows, score)] if rows else []
