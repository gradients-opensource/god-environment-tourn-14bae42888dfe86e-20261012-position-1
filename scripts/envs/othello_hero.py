"""Hero-seat othello generation (the default othello generator in pvp_selfplay).

One recorded seat per game (seed % 2, like the eval's seat swap) plays the othello teacher's label
(othello_rule by default) against an opponent drawn per game from a pool: OpenSpiel MCTS with
U[25, 75] sims and one random rollout (the eval-opponent family), or the board-blind positional
rule with U[0, 0.2] random moves, a proxy for the field's rule-imitating students. Every hero
decision becomes a row and every game is kept, won or lost: the label is a function of the visible
board, not of the outcome. The old symmetric self-play recorded both seats of one teacher playing
itself, so the student never saw the positions a different opponent leads to.
"""
from __future__ import annotations

import random

from envs.othello_rule import positional_label
from envs.pvp_selfplay import (
    _FORMATTERS,
    _config_id_for_seed,
    _dart_eps,
    _dart_executed_action,
    _emit_rows,
    _load_game,
    _make_mcts_bot,
    _openspiel_mcts_teacher,
    _score,
    _setup_initial_state,
    _teacher,
    _teacher_obs,
)

ENV = "othello"
OPP_MCTS_SHARE = 0.5
OPP_MCTS_SIMS = (25, 75)
OPP_RULE_EPS = (0.0, 0.2)
_POOL_SALT = 0x0E7E110  # decorrelates the opponent draw from the opening plies and the seat parity
_NOISE_SALT = 0x0E7E5EED


def describe() -> str:
    """One terse generation log line."""
    return (f"othello gen: hero rows vs mcts{OPP_MCTS_SIMS[0]}-{OPP_MCTS_SIMS[1]} "
            f"({OPP_MCTS_SHARE:.0%}) | positional eps{OPP_RULE_EPS[0]:g}-{OPP_RULE_EPS[1]:g}")


def opponent_spec(seed: int) -> "tuple[str, float]":
    """This game's pool opponent, deterministic per seed: ('mcts', sims) or ('positional', eps)."""
    pool = random.Random((seed ^ _POOL_SALT) & 0x7FFFFFFF)
    if pool.random() < OPP_MCTS_SHARE:
        return "mcts", pool.randint(*OPP_MCTS_SIMS)
    return "positional", pool.uniform(*OPP_RULE_EPS)


def _opponent(game, seed: int):
    """(kind, policy) for this game; policy(state, player, legal) -> action id. Without open_spiel's
    MCTS the game falls back to the positional opponent at an eps from the same seeded range."""
    kind, value = opponent_spec(seed)
    if kind == "mcts":
        bot = _make_mcts_bot(game, seed ^ 0x5EED, int(value))
        if bot is not None:
            return kind, lambda state, player, legal: int(bot.step(state))
        kind, value = "positional", random.Random((seed ^ _POOL_SALT) & 0x7FFFFFFF).uniform(*OPP_RULE_EPS)
    noise = random.Random((seed ^ _NOISE_SALT) & 0x7FFFFFFF)

    def noisy_positional(state, player, legal):
        if noise.random() < value:
            return noise.choice(legal)
        return positional_label(legal)

    return kind, noisy_positional


def play_hero_game(seed: int, max_turn: int, rng: "random.Random | None" = None):
    """Play one eval-shaped othello game (2-6 seeded opening plies) and return ``[(rows, score)]`` for
    the hero seat: one row per hero decision (forced moves and passes included, as at eval), score in
    [0, 1]. [] when the hero never moved."""
    if rng is None:
        rng = random.Random(seed)
    game = _load_game(ENV, _config_id_for_seed(seed, ENV))
    state = game.new_initial_state()
    _setup_initial_state(ENV, state, seed)
    hero = seed % 2
    _, opponent = _opponent(game, seed)
    format_state = _FORMATTERS[ENV]
    teacher = _teacher(ENV)
    # None by default; OTHELLO_TEACHER=mcts labels with the OpenSpiel MCTS bot instead (same data shape).
    mcts_label = _openspiel_mcts_teacher(ENV, game, seed)
    dart_eps = _dart_eps(ENV)

    rows: list[dict] = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        player = state.current_player()
        legal = state.legal_actions(player)
        if not legal:
            break
        if player == hero:
            labels = [(a, state.action_to_string(player, a)) for a in legal]
            state_desc = format_state(state, player)
            label = None
            if mcts_label is not None:
                try:  # a search failure falls back to the default label instead of dropping the game
                    label = int(mcts_label.step(state))
                except Exception:
                    label = None
            if label not in legal:
                try:
                    label = teacher(_teacher_obs(state_desc, player, labels), player, legal, rng=rng)
                except Exception:  # noqa: BLE001 — a teacher bug must never stop generation
                    label = rng.choice(legal)
                if label not in legal:
                    label = rng.choice(legal)
            _emit_rows(rows, ENV, state_desc, player, labels, label)
            action = _dart_executed_action(label, legal, dart_eps, rng)
        elif len(legal) == 1:
            action = legal[0]  # forced move or pass: no search needed
        else:
            try:
                action = opponent(state, player, legal)
            except Exception:
                action = rng.choice(legal)
            if action not in legal:
                action = rng.choice(legal)
        state.apply_action(action)
        turns += 1

    if not rows:
        return []
    score = _score(state.returns(), hero) if state.is_terminal() else 0.5
    return [(rows, score)]
