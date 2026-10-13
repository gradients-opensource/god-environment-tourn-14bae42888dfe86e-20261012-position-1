"""Hero-seat clobber generation (the default clobber generator in pvp_selfplay).

One recorded seat per game (seed % 2, like the eval's seat swap) plays the deterministic
clobber_label move against an opponent drawn per game from a pool: OpenSpiel MCTS with
U[25, 100] sims and one random rollout, or a noisy 1-ply greedy player that makes the kind of
blunders a weak student makes. The hero's executed move is the label with DART noise, so
recovery positions get labelled too; the row always carries the label. K seeded hero
positions per game become rows (more distinct games per row), and rows are kept whatever
the outcome: half of all eval seats start lost, and those positions need swindle rows too.
"""
from __future__ import annotations

import random

from envs import clobber_label
from envs.clobber_expert import _label_to_indices, _parse_board
from envs.pvp_selfplay import (
    _FORMATTERS,
    _build_row,
    _config_id_for_seed,
    _dart_executed_action,
    _load_game,
    _make_mcts_bot,
    _openspiel_mcts_teacher,
    _score,
    _setup_initial_state,
    _teacher_obs,
)

ENV = "clobber"
LABELS_PER_GAME = 3
DART_EPS = 0.1
OPP_MCTS_SHARE = 0.7
OPP_MCTS_SIMS = (25, 100)
OPP_GREEDY_NOISE = 0.25


def describe() -> str:
    """One terse generation log line."""
    return f"clobber gen: hero rows k={LABELS_PER_GAME} nodes={clobber_label.NODE_BUDGET}"


def _action_for_move(state, player: int, legal: "list[int]", move) -> "int | None":
    rows = len(_parse_board(state.observation_string(player))[0])
    for aid in legal:
        (r, c), (nr, nc) = _label_to_indices(state.action_to_string(player, aid), rows)
        if (r, c, nr, nc) == move:
            return aid
    return None


def _opponent(game, seed: int):
    """The game's opponent policy, drawn from the pool by a seed-derived stream:
    policy(state, player, legal) -> action id."""
    pool = random.Random((seed ^ 0xC10BBE) & 0x7FFFFFFF)
    if pool.random() < OPP_MCTS_SHARE:
        bot = _make_mcts_bot(game, seed ^ 0x5EED, pool.randint(*OPP_MCTS_SIMS))
        if bot is not None:
            return lambda state, player, legal: int(bot.step(state))

    def noisy_greedy(state, player, legal):
        if pool.random() < OPP_GREEDY_NOISE:
            return pool.choice(legal)
        grid = _parse_board(state.observation_string(player))[0]
        me, opp = ("o", "x") if player == 0 else ("x", "o")
        aid = _action_for_move(state, player, legal, clobber_label.greedy_move(grid, me, opp))
        return aid if aid is not None else pool.choice(legal)

    return noisy_greedy


def play_hero_game(seed: int, max_turn: int, rng: "random.Random | None" = None):
    """Play one eval-shaped clobber game (board and 2-6 opening plies from the seed) and return
    ``[(rows, score)]`` for the hero seat: up to LABELS_PER_GAME rows sampled from the hero's
    decisions with >= 2 legal moves, score in [0, 1]. [] when the hero never had a choice."""
    if rng is None:
        rng = random.Random(seed)
    game = _load_game(ENV, _config_id_for_seed(seed, ENV))
    state = game.new_initial_state()
    _setup_initial_state(ENV, state, seed)
    hero = seed % 2
    opponent = _opponent(game, seed)
    format_state = _FORMATTERS[ENV]
    # None by default; CLOBBER_TEACHER=mcts labels with the 13759ee MCTS-100 bot instead (A/B arm
    # that keeps this data shape and swaps only the label).
    mcts_label = _openspiel_mcts_teacher(ENV, game, seed)

    decisions = []
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
                label = clobber_label.choose_action(_teacher_obs(state_desc, player, labels), player, legal)
            if len(legal) > 1:
                decisions.append((state_desc, labels, label))
            action = _dart_executed_action(label, legal, DART_EPS, rng)
        else:
            try:
                action = opponent(state, player, legal)
            except Exception:
                action = rng.choice(legal)
            if action not in legal:
                action = rng.choice(legal)
        state.apply_action(action)
        turns += 1

    picks = random.Random((seed ^ 0x4E80) & 0x7FFFFFFF).sample(
        range(len(decisions)), min(LABELS_PER_GAME, len(decisions)))
    rows = [_build_row(ENV, decisions[i][0], hero, decisions[i][1], decisions[i][2]) for i in sorted(picks)]
    if not rows:
        return []
    score = _score(state.returns(), hero) if state.is_terminal() else 0.5
    return [(rows, score)]
