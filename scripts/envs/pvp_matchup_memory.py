"""Matchup memory for the PvP SFT rows: LONG_TERM notes in game prompts + post-game reflection rows.

At eval the validator gives each player a per-game WORKING memory (4 slots) and a per-matchup LONG_TERM
memory (8 slots), renders both into every turn's system prompt (core/pvp/bot.py _memory_block), and after
each game runs a reflection call whose only tools are the 4 memory tools (bot.py reflect;
validator/evaluation/pvp/game_runner.py _game_memories / _play_game). Our deployed model writes a
LONG_TERM note in every reflection (faithful replay with the validator's own matchup code: 132 writes in
132 games, both seats) and never during a turn (one game_action call per turn). So from game 2 of every
matchup on, i.e. on 97-99.5% of turns at the 32-214 games a 900 s matchup plays, its prompt carries
notes, while every game row it trained on showed the all-empty block.

Two levers, closing the loop between what the model writes and what it later reads:
  * reflection rows teach ONE note, long_term_memory_rewrite(slot 1, record_note(...)): the matchup
    record read from the previous note plus this game's result, and a constant plan line;
  * a share of game rows render that note (at a sampled matchup position) in the LONG_TERM block, a
    smaller part render free-form notes, with the user prompt, tools and move label unchanged, so the
    move does not depend on the notes.
WORKING memory stays empty in both: our turns write nothing, and working memory is reset every game.
Rendering reuses pvp_format's mirror of the validator's SlotMemory / _memory_block, byte-checked against
the validator's LLMBot in test_pvp_eval_parity.py. Reflection games are seeded from the env's generation
id range above the validator's eval seed span (1..1,000,000), never from inside it.

RISK: default ON, but the win-rate effect is unmeasured; the lever is pending a GPU A/B. Reflection rows
are the only rows whose call is not game_action, and one memory call leaking into a game turn costs that
turn (a forfeit when no legal game_action follows). Before relying on it, run one faithful gauntlet pass
against the previous commit that logs the memory tool-call contents and keeps the turn-forfeit counter at
0, and compare win rates. Rollback: PVP_MEMORY_ROW_SHARE=0 PVP_REFLECTION_ROW_SHARE=0 restores the old
rows exactly; any error in this module also keeps them (generate_trajectories._with_matchup_memory).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from typing import NamedTuple

import pyspiel

from envs.pvp_format import REFLECTION_GUIDANCE
from envs.pvp_format import TOOL_GUIDANCE
from envs.pvp_format import build_memory_tool_list
from envs.pvp_format import build_reflection_user_prompt
from envs.pvp_format import build_tool_system_prompt
from envs.pvp_format import default_memories
from envs.pvp_format import memory_block
from envs.pvp_format import tools_to_openai
from envs.pvp_models import MemoryArea
from envs.pvp_models import MemoryOp
from envs.pvp_selfplay import _FORMATTERS
from envs.pvp_selfplay import _config_id_for_seed
from envs.pvp_selfplay import _load_game
from envs.pvp_selfplay import _setup_initial_state
from envs.pvp_tools import memory_tool_name
from envs.shared_env import GAMES_TO_TASK_ID_RANGE

# Share of each PvP env's game rows whose system prompt carries a filled LONG_TERM block. At eval ~98% of
# turns carry notes (module docstring); the other 20% keep the empty block of game 1 and of the rows
# earlier rounds trained on. PVP_MEMORY_ROW_SHARE=0 restores the always-empty rows.
MEMORY_ROW_SHARE_DEFAULT = 0.8
# Part of those rows that render free-form notes in 1-3 slots instead of the record note, in case the
# deployed model writes something other than the note the reflection rows teach.
FREEFORM_NOTE_SHARE = 0.25
# Reflection rows added per env as a share of its game rows, capped. They are the only rows that call a
# memory tool, so they stay small next to the game_action rows. PVP_REFLECTION_ROW_SHARE=0 adds none.
REFLECTION_ROW_SHARE_DEFAULT = 0.02
REFLECTION_MAX_ROWS = 300
# Part of the reflection rows written after the FIRST game of a matchup (empty memory), where the note
# format is set; every later reflection only updates it.
REFLECTION_FIRST_GAME_SHARE = 0.25
LONG_TERM_NOTE_SLOT = 1

_MATCHUP_GAMES = (20, 160)  # games per matchup for a sampled record (live R2 + faithful replays: 32-214)
_WIN_RATE = (0.15, 0.85)
_DRAW_RATE_MAX = 0.3
_DRAWABLE = frozenset({"goofspiel", "leduc_poker", "gin_rummy", "othello"})  # liars_dice/clobber never draw
_APPENDED_NOTES_SHARE = 0.3  # free-form notes joined in slot 1, the way SlotMemory.append stores them
_REFLECTION_DRAW_SHARE = 0.1  # reflection rows after a drawn game, on games that can draw
_OUTCOME_TRIES = 30  # seeds tried for the wanted decisive/drawn result (a gin knock game is decisive 26%)
# The validator draws eval game seeds from 1..this (core/pvp/constants.py PVP_SEED_RANGE_MAX). A seed fixes
# the eval config variant and opening, so no generated game may use one.
_EVAL_SEED_MAX = 1_000_000
_REWRITE_TOOL = memory_tool_name(MemoryArea.LONG_TERM, MemoryOp.REWRITE)

_FREEFORM_NOTES = (
    "Opponent plays fast and fairly safe; no clear weakness yet.",
    "Opponent is aggressive early. Stay solid and punish over-extension.",
    "Opponent tends to repeat the same ideas from game to game.",
    "Opponent made errors late in the game; keep the position complex.",
    "Lost the last game after an early mistake. Play carefully.",
    "Won the last game with my normal plan. Keep it.",
    "Opponent seems strong; avoid risky moves.",
    "No reliable read on this opponent yet.",
)
_NOTE_RE = re.compile(
    r"Record vs this opponent: (\d+) won, (\d+) lost, (\d+) drawn\. "
    r"Last game: (?:WIN|LOSS|DRAW)\. Plan: keep my standard strategy\."
)


class ReflectionExample(NamedTuple):
    """One reflection row plus the pieces it was rendered from (the parity test re-renders them)."""

    row: dict
    state: pyspiel.State
    player_id: int
    prior_slots: dict[int, str]
    outcome: str


def record_note(wins: int, losses: int, draws: int, last_outcome: str) -> str:
    """The LONG_TERM note the reflection rows teach (well under the 128-token slot budget)."""
    return (f"Record vs this opponent: {wins} won, {losses} lost, {draws} drawn. "
            f"Last game: {last_outcome.upper()}. Plan: keep my standard strategy.")


def parse_record_note(text: str) -> tuple[int, int, int] | None:
    """(wins, losses, draws) of a record note, or None for any other text."""
    match = _NOTE_RE.fullmatch(text or "")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def tallied(record: tuple[int, int, int] | None, outcome: str) -> tuple[int, int, int]:
    """The record after one more game with `outcome` ("win"/"loss"/"draw"); None = no games yet."""
    wins, losses, draws = record or (0, 0, 0)
    return wins + (outcome == "win"), losses + (outcome == "loss"), draws + (outcome == "draw")


def _sample_history(rng: random.Random, env_name: str) -> list[str]:
    """Outcomes of the games already played in a sampled matchup (at least one)."""
    games = rng.randint(1, rng.randint(*_MATCHUP_GAMES) - 1)
    win_rate = rng.uniform(*_WIN_RATE)
    draw_rate = rng.uniform(0.0, _DRAW_RATE_MAX) if env_name in _DRAWABLE else 0.0
    history = []
    for _ in range(games):
        u = rng.random()
        history.append("draw" if u < draw_rate else "win" if u < draw_rate + (1 - draw_rate) * win_rate else "loss")
    return history


def _record_slots(history: list[str]) -> dict[int, str]:
    record = (history.count("win"), history.count("loss"), history.count("draw"))
    return {LONG_TERM_NOTE_SLOT: record_note(*record, history[-1])}


def game_row_long_term(rng: random.Random, env_name: str) -> dict[int, str]:
    """LONG_TERM slot contents for one game row: the record note, or 1-3 free-form notes."""
    if rng.random() >= FREEFORM_NOTE_SHARE:
        return _record_slots(_sample_history(rng, env_name))
    notes = rng.sample(_FREEFORM_NOTES, rng.randint(1, 3))
    if len(notes) > 1 and rng.random() < _APPENDED_NOTES_SHARE:
        return {1: "\n".join(notes)}
    return {slot: note for slot, note in enumerate(notes, start=1)}


_RULES_CACHE: dict[str, str] = {}


def _system_prompt(env_name: str, long_term_slots: dict[int, str], guidance: str) -> str:
    """rules + memory block + guidance, as LLMBot._system_prompt / _reflection_system_prompt build it."""
    rules = _RULES_CACHE.get(env_name)
    if rules is None:
        rules = _RULES_CACHE[env_name] = build_tool_system_prompt(env_name)
    memories = default_memories()
    for slot, text in long_term_slots.items():
        memories[MemoryArea.LONG_TERM].rewrite(slot, text)
    return "\n\n".join([rules, memory_block(memories), guidance])


def with_long_term_memory(row: dict, env_name: str, long_term_slots: dict[int, str]) -> dict | None:
    """A copy of a game row whose system prompt shows `long_term_slots`; None when the row does not carry
    the standard empty-memory turn prompt of `env_name` (left to the caller untouched)."""
    messages = row.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return None
    if messages[0].get("content") != _system_prompt(env_name, {}, TOOL_GUIDANCE):
        return None
    system = {**messages[0], "content": _system_prompt(env_name, long_term_slots, TOOL_GUIDANCE)}
    return {**row, "messages": [system, *messages[1:]]}


def reflection_seed_bounds(env_name: str) -> tuple[int, int]:
    """[low, high) span reflection game seeds are drawn from: the env's generation id range (as
    generate_for_env samples it) above the eval seed span, with room for the outcome retries."""
    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[env_name]
    return max(task_id_min + 1, _EVAL_SEED_MAX + 1), task_id_max - _OUTCOME_TRIES


def _play_to_terminal(env_name: str, game_seed: int) -> pyspiel.State:
    """A random game from the eval's seeded variant and opening to its end. A legal knock is always taken:
    uniformly random gin rummy almost never knocks and ends drawn on 96% of seeds (26% with this rule)."""
    rng = random.Random(game_seed)
    game = _load_game(env_name, _config_id_for_seed(game_seed, env_name))
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, game_seed)
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
            continue
        player = state.current_player()
        legal = state.legal_actions(player)
        knocks = [a for a in legal if "Knock" in state.action_to_string(player, a)] if env_name == "gin_rummy" else []
        state.apply_action(knocks[0] if knocks else rng.choice(legal))
    return state


def _outcome(returns: list[float], player_id: int) -> str:
    """GameOutcome value for a two-player zero-sum game (core/pvp/scoring.py determine_outcome)."""
    mine, theirs = returns[player_id], returns[1 - player_id]
    return "win" if mine > theirs else "loss" if mine < theirs else "draw"


def _game_with_outcome(env_name: str, game_seed: int, rng: random.Random) -> tuple[pyspiel.State, int, str]:
    """(final state, seat, outcome) with the outcome drawn as win/loss 50/50 (plus a draw share on drawable
    games) rather than taken from random play; tries consecutive seeds, keeping the last game on a miss."""
    want_draw = env_name in _DRAWABLE and rng.random() < _REFLECTION_DRAW_SHARE
    want = "draw" if want_draw else rng.choice(("win", "loss"))
    for k in range(_OUTCOME_TRIES):
        state = _play_to_terminal(env_name, game_seed + k)
        returns = state.returns()
        if (returns[0] == returns[1]) == want_draw:
            break
    seat = 0 if want_draw or _outcome(returns, 0) == want else 1
    return state, seat, _outcome(returns, seat)


def build_reflection_example(env_name: str, game_seed: int) -> ReflectionExample:
    """The reflection call after one game, answered with one long_term_memory_rewrite of the record note.
    `game_seed` (and its outcome retries up to game_seed + _OUTCOME_TRIES - 1) must lie above the eval seed
    span; ValueError otherwise."""
    if game_seed <= _EVAL_SEED_MAX:
        raise ValueError(f"reflection game seed {game_seed} lies in the eval seed span 1..{_EVAL_SEED_MAX}")
    rng = random.Random(f"reflection:{env_name}:{game_seed}")
    state, player_id, outcome = _game_with_outcome(env_name, game_seed, rng)
    first_game = rng.random() < REFLECTION_FIRST_GAME_SHARE
    prior_slots = {} if first_game else _record_slots(_sample_history(rng, env_name))
    record = tallied(parse_record_note(prior_slots.get(LONG_TERM_NOTE_SLOT, "")), outcome)
    call = {"type": "function", "function": {
        "name": _REWRITE_TOOL,
        "arguments": json.dumps({"slot": LONG_TERM_NOTE_SLOT, "content": record_note(*record, outcome)})}}
    row = {
        "messages": [
            {"role": "system", "content": _system_prompt(env_name, prior_slots, REFLECTION_GUIDANCE)},
            {"role": "user", "content": build_reflection_user_prompt(_FORMATTERS[env_name](state, player_id), outcome)},
            {"role": "assistant", "content": None, "tool_calls": [call]},
        ],
        "tools": json.dumps(tools_to_openai(build_memory_tool_list())),
    }
    return ReflectionExample(row, state, player_id, prior_slots, outcome)


def _share(env_var: str, default: float) -> float:
    """The env var as a share in [0, 1]; unset/empty -> `default`. A value that does not parse as a number
    (e.g. "off") counts as 0: setting the variable at all means someone reached for the kill switch."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.0


def _digest(*parts: object) -> int:
    key = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def add_matchup_memory(
    env_name: str,
    examples: list[dict],
    seed: int = 42,
    memory_share: float | None = None,
    reflection_share: float | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """New row list: a share of the game rows re-rendered with LONG_TERM notes, then the reflection rows.

    Each row's draw is seeded from its own content and its index among identical rows (not its position),
    so every copy of a repeated state draws independently and the output, as a multiset, does not depend
    on the order the generator workers returned games in. Input rows are never mutated.
    """
    if memory_share is None:
        memory_share = _share("PVP_MEMORY_ROW_SHARE", MEMORY_ROW_SHARE_DEFAULT)
    if reflection_share is None:
        reflection_share = _share("PVP_REFLECTION_ROW_SHARE", REFLECTION_ROW_SHARE_DEFAULT)
    stats = {"memory_rows": 0, "freeform_rows": 0, "foreign_rows": 0, "reflection_rows": 0}
    out: list[dict] = []
    copies: Counter[str] = Counter()
    for row in examples:
        content = json.dumps(row.get("messages", [])[1:], sort_keys=True, default=str)
        copies[content] += 1
        rng = random.Random(_digest(seed, env_name, content, copies[content]))
        if memory_share <= 0 or rng.random() >= memory_share:
            out.append(row)
            continue
        slots = game_row_long_term(rng, env_name)
        rendered = with_long_term_memory(row, env_name, slots)
        if rendered is None:
            stats["foreign_rows"] += 1
            out.append(row)
            continue
        stats["memory_rows"] += 1
        stats["freeform_rows"] += parse_record_note(slots.get(LONG_TERM_NOTE_SLOT, "")) is None
        out.append(rendered)
    n_reflection = min(REFLECTION_MAX_ROWS, round(reflection_share * len(examples)))
    seed_low, seed_high = reflection_seed_bounds(env_name) if n_reflection else (0, 0)
    for i in range(n_reflection):
        game_seed = seed_low + _digest(seed, env_name, "reflection", i) % (seed_high - seed_low)
        out.append(build_reflection_example(env_name, game_seed).row)
    stats["reflection_rows"] = n_reflection
    return out, stats
