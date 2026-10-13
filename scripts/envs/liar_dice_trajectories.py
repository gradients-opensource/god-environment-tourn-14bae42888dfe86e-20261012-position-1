"""Expert trajectory generator for Liar's Dice SFT training.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action).

Expert teacher: action selection is delegated to ``envs.liar_dice_expert``'s
``choose_action``, which samples a single legal action id from a CFR
equilibrium. This module owns only the env-server I/O loop, observation
parsing, and the (system, user, assistant) example construction; the teacher
itself lives in the sibling engine module (built in parallel).
"""

import json
import re

import requests

from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import tools_to_openai
from envs.shared_env import _log

_TIMEOUT = 2400
_GAME_NAME = "liars_dice"

# --- Opponent diversity (one-shot, NOT iterative DAgger) -------------------
# To cover a broader opponent-induced state distribution, we vary the MCTS
# opponent's strength across games instead of pinning it to a single
# simulation budget. Each game deterministically picks one budget from this
# fixed grid, keyed by game_id (game_id % len(grid)). This is a static,
# one-pass diversification of the env reset payload — there is no feedback
# loop / re-labelling of the model's own rollouts (that would be DAgger).
# Weaker opponents (25 sims) yield looser, more varied bidding lines; stronger
# opponents (400 sims) push the expert into tighter, more adversarial states.
_MCTS_SIMS_GRID = (25, 75, 150, 225, 400)

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


# ---------------------------------------------------------------------------
# Raw env-server observation -> eval-format (state_desc, player_id, legal_actions)
# ---------------------------------------------------------------------------
#
# The env server's raw `observation` text is built from
# core/pvp/agents.py::LiarsDiceAgent.format_state (same "Your dice: ...",
# "Total dice in game: ...", `Current bid: "Q-F"` lines parse_game_state already
# relies on), wrapped with a "Current State:\n"/"You are Player N."/
# "Legal Actions:\n..." envelope. eval_time's _user_prompt
# (pvp_format.build_user_prompt) wants exactly the inner state description text
# plus (player_id, legal_actions) — this splits the raw observation into those
# pieces so build_user_prompt can reproduce the eval-time text byte-for-byte
# (mod the "Current State"/"Legal Actions" header casing, which build_user_prompt
# itself supplies).

_RE_CURRENT_STATE_PREFIX = re.compile(r"^Current State:\n", re.MULTILINE)
_RE_LEGAL_ACTIONS_HEADER = re.compile(r"^[ \t]*Legal Actions:[ \t]*$", re.MULTILINE)
_RE_PLAYER_LINE = re.compile(r"^You are Player (\d+)\.[ \t]*$", re.MULTILINE)
_RE_CURRENT_PLAYER = re.compile(r"Current player: Player (\d+)")
_RE_LEGAL_ACTION = re.compile(r"^\s*(\d+)\s*->\s*(.+)$", re.MULTILINE)


def _split_observation(observation: str) -> tuple[str, int, list[tuple[int, str]]]:
    """Split a raw env-server observation into (state_desc, player_id, legal_actions).

    The env server's raw observation is
    "# Game Rules\n...\n# Current Game State\nGame: liars_dice\nYou are Player N.\n\n
     Current State:\n...\nLegal Actions:\n  N -> ...\nYour choice (action ID only):"
    — the "You are Player N." line lives in the preamble, BEFORE "Current State:",
    so it must be matched against the full raw text (mirrors gin_rummy/othello/leduc's
    _format_observation, which all search the raw text for this line).
    """
    player_match = _RE_PLAYER_LINE.search(observation)
    current_player_match = _RE_CURRENT_PLAYER.search(observation)

    text = observation
    prefix = _RE_CURRENT_STATE_PREFIX.search(text)
    if prefix:
        text = text[prefix.end():]

    header = _RE_LEGAL_ACTIONS_HEADER.search(text)
    if not header:
        raise ValueError(f"Could not find 'Legal Actions:' header in observation: {observation[:200]!r}")
    before, after = text[:header.start()], text[header.end():]

    if player_match:
        player_id = int(player_match.group(1))
        before = _RE_PLAYER_LINE.sub("", before)
    elif current_player_match:
        player_id = int(current_player_match.group(1))
    else:
        raise ValueError(f"Could not determine player_id from observation: {observation[:200]!r}")

    state_desc = before.strip("\n")

    legal_actions = [(int(aid), label.strip()) for aid, label in _RE_LEGAL_ACTION.findall(after)]
    if not legal_actions:
        raise ValueError(f"Could not parse legal actions from observation: {observation[:200]!r}")

    return state_desc, player_id, legal_actions


def _build_tool_example_from_parts(
    state_desc: str, player_id: int, legal_actions: list[tuple[int, str]], action_id: int
) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "game_action", "arguments": json.dumps({"action_id": action_id})}},
            ]},
        ],
        # JSON-encoded: see intercode_dataset.py's _INTERCODE_TOOLS_JSON for why
        # heterogeneous tool `parameters.properties` must not go through
        # Dataset.from_list as a list-of-dicts column.
        "tools": json.dumps(tools_to_openai(tools)),
    }


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> list[dict]:
    """
    Run one Liar's Dice game against the env server using the expert teacher.

    The teacher is ``envs.liar_dice_expert.choose_action`` (a CFR-equilibrium
    sampler). Each turn we parse the server observation into the eval-format
    pieces, hand the raw observation + parsed legal action ids to the teacher,
    record the chosen action as a bare-integer ``game_action`` tool call, and
    POST it back to advance the game.

    Returns a list of independent per-turn training examples (one per action
    taken by the expert), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}``.
    Returns ``[]`` on failure (e.g. reset failed).
    """
    # Imported lazily so this module still imports even before the sibling
    # engine module exists, and so per-process import cost is paid once.
    from envs.liar_dice_expert import choose_action

    # The env server seats the controlled expert at its default seat: it does not
    # accept a player_id at reset (sending one 500s, and the reference client
    # never sends it). State diversity comes from seed/game_id; the expert reads
    # the actual seat back from each observation, so action selection stays correct.
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        # Opponent diversity (one-shot): deterministic per-game MCTS strength
        # from a fixed grid keyed by game_id (see _MCTS_SIMS_GRID above).
        "mcts_max_simulations": _MCTS_SIMS_GRID[game_id % len(_MCTS_SIMS_GRID)],
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        _log(f"[liar_dice_trajectories] Reset failed (game {game_id}): {exc}")
        return []

    examples: list[dict] = []

    for _ in range(max_turn):
        try:
            state_desc, player_id, legal_actions = _split_observation(observation)
            legal_action_ids = [aid for aid, _ in legal_actions]
            # Expert teacher: sample one legal action id from the CFR equilibrium.
            action_id = choose_action(observation, player_id, legal_action_ids)
            examples.append(_build_tool_example_from_parts(state_desc, player_id, legal_actions, action_id))
        except Exception as exc:
            _log(f"[liar_dice_trajectories] Failed to choose/build example (game {game_id}): {exc}")
            break

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": str(action_id), "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = step_block.get("observation", "")
            done = step_block.get("done", False)
        except Exception as exc:
            _log(f"[liar_dice_trajectories] Step failed (game {game_id}): {exc}")
            break

        if done:
            break
    else:
        _log(f"[liar_dice_trajectories] max_turn={max_turn} reached (game {game_id})")

    return examples
