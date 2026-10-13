"""Nash-blueprint trajectory generator for Leduc Poker SFT training.

Plays games with a CFR Nash blueprint expert against the MCTS opponent and
returns ``(examples, final_reward)`` so that generate_trajectories.py can apply
score-based sampling to bias toward games with positive outcomes.

Produces per-turn tool-calling examples (§3.2/§3.3 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action). The assistant target is a bare
integer action id (no chain-of-thought).

Expert: actions come from envs.leduc_poker_expert.choose_action, which samples
one legal action id from a CFR Nash blueprint. Leduc Poker's tiny state space
makes a near-optimal blueprint tractable, so unlike the previous random policy
we now teach near-equilibrium play; final-reward filtering via --sample-by-score
still biases the kept set toward winning games.

The exported entry point is named ``generate_random_episode`` (signature and
``(messages, score)`` tuple return type) for backward compatibility with
envs/sft_env_configs.py and generate_trajectories.py's score-based sampling.
"""

import json
import random

import requests

from envs.leduc_poker_env import _format_observation
from envs.leduc_poker_expert import choose_action
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import split_normalized_observation
from envs.pvp_format import tools_to_openai
from envs.shared_env import _log

_TIMEOUT = 2400
_GAME_NAME = "leduc_poker"

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)

# Opponent diversity (one-shot, NOT DAgger): vary the MCTS opponent strength
# across games so the blueprint is demonstrated against a spread of opponents
# rather than a single fixed-strength MCTS. We deterministically key the
# simulation budget off game_id so generation stays reproducible. This is a
# fixed env-reset choice per game (no online opponent adaptation / no
# re-querying the policy on the opponent's distribution), hence one-shot.
_MCTS_SIM_CHOICES = (25, 50, 150, 225)


def _build_tool_example_from_parts(
    state_desc: str, player_id: int, legal_actions: list[tuple[int, str]], action_id: int,
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


def generate_random_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 10,
) -> "tuple[list[dict], float]":
    """
    Run one Leduc Poker game using the CFR Nash blueprint against MCTS.

    Despite the legacy name (kept for the sft_env_configs registry + score
    sampling contract), actions are now chosen by the Nash-blueprint expert,
    not at random.

    Returns ``(examples, final_reward)``, where ``examples`` is a list of
    independent per-turn training examples (one per action taken), each shaped
    ``{"messages": [system, user, assistant(tool_calls=[game_action])], "tools": [...]}``.
    ``final_reward`` is the env's already-normalized terminal score in [0, 1]
    (0.0 = max loss, 0.5 = draw, 1.0 = max win) for our seat, used directly by
    generate_trajectories.py as a sampling probability. Returns ``([], 0.0)``
    on failure (e.g. reset failed).
    """
    # Our seat alternates by game so the blueprint is taught from both P0 and P1.
    our_player_id = game_id % 2
    # One-shot opponent diversity: deterministic per-game MCTS strength.
    mcts_max_simulations = _MCTS_SIM_CHOICES[game_id % len(_MCTS_SIM_CHOICES)]
    # Seed the blueprint sampler once per episode so draws advance across turns
    # (a fresh per-turn Random(game_id) would replay the same sample every turn).
    rng = random.Random(game_id)

    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": mcts_max_simulations,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block       = res.json()["result"]
        episode_id  = block.get("episode_id", "")
        observation = _format_observation(block.get("observation", ""))
    except Exception as exc:
        _log(f"[leduc_poker_trajectories] Reset failed (game {game_id}): {exc}")
        return [], 0.0

    examples: list[dict] = []
    final_reward = 0.0

    for _ in range(max_turn):
        # Parse legal action ids (and seat) from the observation, then let the
        # Nash blueprint pick one. choose_action returns a legal int action id.
        try:
            state_desc, player_id, legal_actions = split_normalized_observation(observation)
            legal_action_ids = [aid for aid, _ in legal_actions]
            action_id = choose_action(
                observation,
                player_id,
                legal_action_ids,
                rng=rng,
            )
            action = str(action_id)
        except Exception as exc:
            _log(f"[leduc_poker_trajectories] choose_action failed (game {game_id}): {exc}")
            return examples, final_reward

        try:
            examples.append(
                _build_tool_example_from_parts(state_desc, player_id, legal_actions, action_id)
            )
        except Exception as exc:
            _log(f"[leduc_poker_trajectories] Failed to build example (game {game_id}): {exc}")

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            done        = step_block.get("done", False)
            if done:
                final_reward = float(step_block.get("reward", 0.0))
        except Exception as exc:
            _log(f"[leduc_poker_trajectories] Step failed (game {game_id}): {exc}")
            return examples, final_reward

        if done:
            break
    else:
        _log(f"[leduc_poker_trajectories] max_turn={max_turn} reached (game {game_id})")

    return examples, final_reward
