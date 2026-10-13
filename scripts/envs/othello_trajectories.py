"""Trajectory generator for Othello SFT training.

Produces per-turn tool-calling examples (§3.4 of docs/SFT_ALIGNMENT_PLAN.md):
each turn of a game becomes its own standalone
``{"messages": [system, user, assistant(tool_calls=game_action)], "tools": ...}``
example — a stateless [system, user] reconstruction matching
``core/pvp/bot.py::LLMBot._run_turn`` at eval time, with an always-empty memory
block and all 5 tools (4 memory + game_action).

Policy: a strong perfect-information teacher engine (othello_expert.choose_action,
built by a sibling module). It returns the BEST move's action id given the
observation — no CoT, no sampling. The generator wires this engine in and keeps
the DOUBLE-VIEW dataset: one game yields both seats' perspectives with
complementary rewards. Games are returned with their final reward
(``(examples, final_reward)``) for optional score-based sampling, mirroring
leduc_poker_trajectories.py.

Othello-specific quirk (see G.O.D's core/pvp/agents.py OthelloAgent.format_state):
the eval-time state description is prefixed with "You play x (Black)." or
"You play o (White)." based on player_id (0=Black/x, 1=White/o) — small models
otherwise misjudge their own colour. _build_tool_example_from_parts reproduces this prefix.
"""

import json
import re

import requests

from envs.othello_board import apply_action_symbols, legal_actions_symbols, render_board
from envs.othello_expert import choose_action
from envs.othello_minimax import _my_symbol, _parse_board
from envs.pvp_format import build_full_system_prompt
from envs.pvp_format import build_pvp_tools
from envs.pvp_format import build_user_prompt
from envs.pvp_format import split_normalized_observation
from envs.pvp_format import tools_to_openai
from envs.shared_env import _log

_TIMEOUT = 2400
_GAME_NAME = "othello"

# Opponent diversity (one-shot, NOT DAgger): vary the MCTS opponent's search
# budget per game so the dataset mixes weak and strong opponents in a single
# generation pass. The strong perfect-info teacher (othello_expert) plays our
# seat regardless; only the env-side MCTS opponent's strength changes here.
# Keyed deterministically by game_id (round-robin over a fixed ladder) so the
# distribution is reproducible and balanced across a generation run rather than
# left to per-call RNG.
_MCTS_SIM_LADDER = (25, 75, 150, 225)

# rules + empty memory block + tool guidance, mirrors LLMBot._system_prompt
# with a fresh (all-empty) memory state — see §3.2.
_SYSTEM_PROMPT = build_full_system_prompt(_GAME_NAME)


# ---------------------------------------------------------------------------
# Raw env-server observation -> eval-format (state_desc, player_id, legal_actions)
# ---------------------------------------------------------------------------
#
# Mirrors leduc_poker_env._format_observation: the env server returns
#   "# Game Rules\n...\n# Current Game State\nGame: othello\n
#    You are Player X.\n\nCurrent State:\n...\nLegal Actions:\n  N -> ...\n
#    Your choice (action ID only):"
# and split_normalized_observation (pvp_format.py) wants
#   "Current State:\n...\nYou are Player N.\nLegal Actions:\nN -> ...\n..."

_RE_PLAYER = re.compile(r"You are Player (\d+)\.")


def _format_observation(raw: str) -> str:
    player_match = _RE_PLAYER.search(raw)
    player_line = f"You are Player {player_match.group(1)}." if player_match else ""

    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw

    body = raw[state_start:]

    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body

    state_block = body[:legal_start].rstrip()
    actions_block = body[legal_start:]
    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)
    actions_block = actions_block.replace("Your choice (action ID only):", "Your choice (ID only):")

    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Example builder + episode runner
# ---------------------------------------------------------------------------

def _build_tool_example_from_parts(
    state_desc: str, player_id: int, legal_actions: list[tuple[int, str]], action_id: int,
) -> dict:
    """Build one stateless {system, user, assistant(game_action)} training example."""
    colour = "x (Black)" if player_id == 0 else "o (White)"
    state_desc = f"You play {colour}.\n{state_desc}"
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


def generate_heuristic_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 70,
) -> "tuple[list[dict], float] | list[tuple[list[dict], float]]":
    """
    Run one Othello game using the strong perfect-information teacher engine
    (othello_expert.choose_action) against the env's MCTS opponent.

    Produces two "views" of the game:
      - ours: one example per turn we played, ``(examples_ours, final_reward)``.
      - opponent: one example per MCTS response move, reconstructed from
        ``info.last_opponent_action`` + a small board simulator
        (othello_board.py), ``(examples_opp, 1.0 - final_reward)``.

    ``final_reward`` is the env's already-normalized terminal score in [0, 1]
    (0.0 = loss, 0.5 = draw, 1.0 = win); the opponent's score is its
    complement (draw stays 0.5). generate_trajectories.py applies score-based
    sampling to each view independently.

    Returns ``[(examples_ours, final_reward), (examples_opp, 1.0 - final_reward)]``
    on success, or ``([], 0.0)`` on failure (e.g. reset failed or a mid-game
    error — in which case any examples collected so far are discarded along
    with their (unknown) outcome).
    """
    # Opponent diversity: round-robin the MCTS sim budget over the ladder,
    # deterministically keyed by game_id (see _MCTS_SIM_LADDER above).
    mcts_sims = _MCTS_SIM_LADDER[game_id % len(_MCTS_SIM_LADDER)]
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": mcts_sims,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block        = res.json()["result"]
        episode_id   = block.get("episode_id", "")
        observation  = _format_observation(block.get("observation", ""))
        step_count   = block.get("info", {}).get("step_count", 0)
    except Exception as exc:
        _log(f"[othello_trajectories] Reset failed (game {game_id}): {exc}")
        return [], 0.0

    examples: list[dict] = []
    opp_examples: list[dict] = []
    final_reward = 0.0

    for _ in range(max_turn):
        try:
            state_desc, player_id, legal_actions = split_normalized_observation(observation)
            legal_action_ids = [aid for aid, _ in legal_actions]
            # Strong teacher: perfect-info best move (no sampling, no CoT).
            action_id = choose_action(observation, player_id, legal_action_ids)
            examples.append(_build_tool_example_from_parts(state_desc, player_id, legal_actions, action_id))

            # Pre-compute the opponent's (state, legal_actions) by simulating
            # our move on the board, so it can be paired with
            # info.last_opponent_action below once we know the opponent
            # actually got to move (see step_count check).
            our_symbol      = _my_symbol(state_desc)
            opp_symbol      = "o" if our_symbol == "x" else "x"
            board_after_us  = apply_action_symbols(_parse_board(state_desc), action_id, our_symbol)
            opp_legal       = legal_actions_symbols(board_after_us, opp_symbol)
            opp_state_desc  = render_board(board_after_us, opp_symbol)
            opp_player_id   = 1 - player_id
        except Exception as exc:
            _log(f"[othello_trajectories] Failed to build example (game {game_id}): {exc}")
            return [], 0.0

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": str(action_id), "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block     = step_res.json()["result"]
            info            = step_block.get("info", {})
            observation     = _format_observation(step_block.get("observation", ""))
            done            = step_block.get("done", False)
            if done:
                final_reward = float(step_block.get("reward", 0.0))
        except Exception as exc:
            _log(f"[othello_trajectories] Step failed (game {game_id}): {exc}")
            return [], 0.0

        # The opponent only got a real turn this step if step_count advanced
        # by 2 (our move + their move). On the final step of a game that ends
        # on our move, step_count advances by only 1 and
        # info.last_opponent_action is stale (unchanged from the prior step).
        new_step_count = info.get("step_count", step_count)
        if new_step_count - step_count >= 2:
            last_opponent_action = info.get("last_opponent_action")
            opp_legal_ids = [aid for aid, _ in opp_legal]
            if last_opponent_action in opp_legal_ids:
                opp_examples.append(_build_tool_example_from_parts(
                    opp_state_desc, opp_player_id, opp_legal, last_opponent_action,
                ))
            else:
                _log(
                    f"[othello_trajectories] last_opponent_action={last_opponent_action!r} "
                    f"not in reconstructed legal actions {opp_legal_ids} (game {game_id}), "
                    "skipping opponent example"
                )
        step_count = new_step_count

        if done:
            break
    else:
        _log(f"[othello_trajectories] max_turn={max_turn} reached (game {game_id})")

    return [(examples, final_reward), (opp_examples, 1.0 - final_reward)]
