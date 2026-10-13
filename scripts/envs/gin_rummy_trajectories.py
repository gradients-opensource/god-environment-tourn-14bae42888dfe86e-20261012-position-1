"""Expert trajectory generator for Gin Rummy SFT training.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action).

Action selection is delegated to the PIMC teacher in envs/gin_rummy_expert.py
(``choose_action``); this module owns only the env-server I/O loop and the
training-row construction (prompt building via pvp_format builders, bare
integer action-id tool-call target, no chain-of-thought).
"""

import json
import random

import requests

from envs.gin_rummy_env import extract_and_format_observation
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import split_normalized_observation
from envs.pvp_format import tools_to_openai
from envs.shared_env import _log

_TIMEOUT = 2400
_GAME_NAME = "gin_rummy"

# Opponent diversity (one-shot, NOT DAgger): each game's env-reset payload picks
# an MCTS strength keyed by game_id, so the generated dataset contains expert
# play against a spread of opponent skill levels (weak → strong) rather than a
# single fixed opponent. This is a static schedule over games — there is no
# iterative dataset aggregation / relabelling against the learner's own rollouts.
_MCTS_SIM_LADDER = (25, 75, 150, 225)

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


def _mcts_sims_for_game(game_id: int) -> int:
    """Pick this game's opponent MCTS strength from the ladder, keyed by game_id."""
    return _MCTS_SIM_LADDER[game_id % len(_MCTS_SIM_LADDER)]


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
    max_turn: int = 200,
) -> list[dict]:
    """
    Run one Gin Rummy game against the env server using the PIMC expert policy.

    Returns a list of independent per-turn training examples (one per action
    taken by the expert), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}``.
    Returns ``[]`` on failure (e.g. reset failed).

    The expert plays the env server's default seat (the server does not accept a
    player_id at reset). Opponent diversity is varied per game via
    ``mcts_max_simulations`` (see ``_MCTS_SIM_LADDER``).
    """
    # Lazy import so this module loads even when the PIMC engine is absent or
    # stubbed (offline tests stub envs.gin_rummy_expert); mirrors the other
    # games' generators, which import their teacher inside the episode runner.
    from envs.gin_rummy_expert import choose_action

    # The env server seats the controlled expert at its default seat: it does not
    # accept a player_id at reset (sending one 500s, and the reference client never
    # sends it). The server only surfaces the controlled seat's turns, so the loop
    # acts on every observation it returns; diversity comes from seed/game_id.
    rng = random.Random(game_id)

    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        # one-shot opponent diversity keyed by game_id (see _MCTS_SIM_LADDER)
        "mcts_max_simulations": _mcts_sims_for_game(game_id),
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = extract_and_format_observation(block.get("observation", ""))
    except Exception as exc:
        _log(f"[gin_rummy_trajectories] Reset failed (game {game_id}): {exc}")
        return []

    examples: list[dict] = []

    for _ in range(max_turn):
        try:
            state_desc, player_id, legal_actions = split_normalized_observation(observation)
            legal_action_ids = [aid for aid, _ in legal_actions]
            # PIMC teacher picks one legal action id (parses ids from the obs).
            action_id = choose_action(observation, player_id, legal_action_ids, rng=rng)
            examples.append(
                _build_tool_example_from_parts(state_desc, player_id, legal_actions, action_id)
            )
        except Exception as exc:
            _log(f"[gin_rummy_trajectories] Failed to choose/build example (game {game_id}): {exc}")
            break

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": str(action_id), "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = extract_and_format_observation(step_block.get("observation", ""))
            done = step_block.get("done", False)
        except Exception as exc:
            _log(f"[gin_rummy_trajectories] Step failed (game {game_id}): {exc}")
            break

        if done:
            break
    else:
        _log(f"[gin_rummy_trajectories] max_turn={max_turn} reached (game {game_id})")

    return examples
