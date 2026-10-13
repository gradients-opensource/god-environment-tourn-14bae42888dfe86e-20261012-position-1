"""Clobber GRPO rollout environment.

Training strategy
-----------------
Clobber is a pure combinatorial game: the player who cannot move LOSES.
The dominant strategic principle is MOBILITY CONTROL — always try to leave
the opponent with as few future moves as possible while keeping your own options
open.

Two rollout variants are implemented, matching the pattern of goof_spiel_env.py:

  - rollout_last   ("strategy-forcing"): plays with the mobility-greedy heuristic
    for the first (max_turn-1) turns, then trains the model on a single target
    turn. This rapidly exposes diverse board states and reinforces the
    mobility-greedy principle.

  - rollout_full   (full-game training): runs the entire game with the model
    generating every action. Reward = final env outcome + shaped mobility bonus.
    This is the primary mode for learning to WIN.
"""
from __future__ import annotations

import functools
import json
import os
import random
import re
from concurrent.futures import as_completed
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.clobber_expert import choose_action as _expert_choose
from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    _log,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME   = "clobber"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN  = 4096
_TIMEOUT         = 2400

_INVALID_PENALTY     = -0.1
_STRATEGY_REWARD     = 1.0         # last-variant: reward for mobility-greedy action
_STRATEGY_WEIGHT     = 0.4         # full-variant: weight of env outcome vs strategy ratio
_MOBILITY_BONUS_STEP = 0.05        # per-turn mobility-following bonus (full variant)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]


def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag, close in REASONING_TAG_PAIRS:
        cleaned = re.sub(rf"<{tag}>.*?</{close}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        close_tag = f"</{close}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    return re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned).strip()


def _extract_legal_action_ids(obs_text: str) -> list[int]:
    """Parse legal action IDs from observation text."""
    ids: list[int] = []
    in_actions = False
    for line in obs_text.split("\n"):
        if "Legal Actions:" in line or "Legal actions:" in line:
            in_actions = True
            continue
        if in_actions:
            m = re.match(r"\s*(\d+)\s*->", line)
            if m:
                ids.append(int(m.group(1)))
    return ids


def _format_observation(obs_text: str) -> str:
    """Append action format reminder if not already present."""
    if "Your choice" in obs_text or "Legal Actions:" in obs_text:
        return obs_text
    return obs_text + '\nRespond with ONLY the action ID (a single number).'


def _pick_greedy_action(obs_text: str, player_id: int) -> str | None:
    """Mobility-greedy action selection from observation text."""
    legal_ids = _extract_legal_action_ids(obs_text)
    if not legal_ids:
        return None
    try:
        best = _expert_choose(obs_text, player_id, legal_ids, rng=None)
        return str(best)
    except Exception:
        return str(random.choice(legal_ids))


def _is_invalid(obs_text: str) -> bool:
    return "Invalid" in obs_text or "Nothing happens" in obs_text


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are playing clobber.

# Game Rules
CLOBBER RULES:

Setup: Pieces (x for Black/Player-0, o for White/Player-1) alternate on a grid.
Goal: Force your opponent to have NO legal moves. The player who cannot move LOSES.
How to move: Move ONE of YOUR pieces to an ADJACENT square (up/down/left/right).
That square MUST contain an OPPONENT piece — it is captured (clobbered) and removed.
You CANNOT move to an empty square.

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.

Examples:
- For action "0 -> a1-b1": respond "0"
- For action "5 -> c2-c3": respond "5"\
"""

_HINT_PROMPT = """\


# Strategy Hints
WINNING STRATEGY — Mobility Control:
The player who runs out of moves first LOSES. Your goal: minimize your opponent's
future moves while maximising your own.

Decision process:
1. For each legal action, ask: "After I play this, how many moves does my opponent have?"
2. Pick the action that leaves your opponent with the FEWEST moves.
3. Tiebreak: pick the action that leaves YOU with the MOST moves.
4. Avoid leaving your own pieces isolated (no adjacent enemy = useless piece).
5. Capture pieces that are adjacent to MANY of your other pieces — these enable chains.

Quick check: If any action leaves opponent with 0 moves, play it immediately (instant win).\
"""


# ---------------------------------------------------------------------------
# Module-level state (per-rank init)
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=30,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.80,
        final_hint_prob=0.10,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)
    curriculum = _curriculum_factory(trainer.args)

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
    )


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _reset_episode(env_endpoint: str, game_id: int) -> tuple[str, str] | None:
    """Reset env and return (episode_id, formatted_observation) or None on error."""
    try:
        r = requests.post(
            f"{env_endpoint}/reset",
            json={"task_id": game_id, "seed": 42, "opponent": "mcts"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        block = r.json()["result"]
        obs   = _format_observation(block.get("observation", ""))
        return block.get("episode_id", ""), obs
    except Exception as exc:
        _log(f"[clobber] Reset failed (game {game_id}): {exc}")
        return None


def _step_episode(env_endpoint: str, action: str, episode_id: str
                  ) -> tuple[str, float, bool] | None:
    """Step env; return (observation, reward, done) or None on error."""
    try:
        r = requests.post(
            f"{env_endpoint}/step",
            json={"action": action, "episode_id": episode_id},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        block = r.json()["result"]
        obs   = _format_observation(block.get("observation", ""))
        return obs, float(block.get("reward", 0)), block.get("done", False)
    except Exception as exc:
        _log(f"[clobber] Step failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# "Last" variant (strategy-forcing)
# ---------------------------------------------------------------------------

def _run_episode_last(
    index: int,
    prompt: str,
    *,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, dict | None]:
    if num_servers == 0:
        return index, None
    game_id     = int(prompt)
    server_idx  = (index + rank) % num_servers
    endpoint    = env_pool[server_idx]["base_url"]
    use_hints   = random.random() < current_hint_prob
    target_turn = max(0, current_max_turn - 1)
    player_id   = game_id % 2  # alternate seat per game

    reset_out = _reset_episode(endpoint, game_id)
    if reset_out is None:
        return index, None
    episode_id, obs = reset_out

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [{"role": "system", "content": system_prompt}]

    done        = False
    turn_number = 0

    # --- Strategy-forcing phase ---
    while not done and turn_number < target_turn:
        messages.append({"role": "user", "content": obs})
        action = _pick_greedy_action(obs, player_id)
        if action is None:
            return index, None
        messages.append({"role": "assistant", "content": action})
        step_out = _step_episode(endpoint, action, episode_id)
        if step_out is None:
            done = False
            break
        obs, _, done = step_out
        turn_number += 1
        player_id = 1 - player_id  # alternate after each step

    if done:
        return index, None

    # --- Training turn ---
    messages.append({"role": "user", "content": obs})
    greedy_action = _pick_greedy_action(obs, player_id)

    with generation_semaphore:
        rollout = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

    prompt_ids     = rollout.get("prompt_ids", [])
    completion_ids = rollout.get("completion_ids", [])
    logprobs       = rollout.get("logprobs", [])
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    action_text = remove_reasoning_tags(completion_text)
    if "Action:" in action_text:
        action_text = action_text.split("Action:")[-1].strip()

    # Check validity
    legal_ids  = _extract_legal_action_ids(obs)
    try:
        played_id = int(action_text.strip())
        valid     = played_id in legal_ids
    except Exception:
        valid = False

    strategy_followed = False
    if valid and greedy_action is not None:
        try:
            strategy_followed = int(action_text.strip()) == int(greedy_action)
        except Exception:
            pass

    if not valid:
        reward = _INVALID_PENALTY
    elif strategy_followed:
        len_scale  = max(0.2, min(5, len(prompt_ids) / max(len(completion_ids), 1)))
        reward     = _STRATEGY_REWARD * len_scale
    else:
        reward = 0.2  # valid move but not greedy-optimal — small positive signal

    metric = (
        "[clobber-LAST] G:{:<6} T:{:>2d} | Strat:{} Valid:{} | R:{:>6.3f} | Hints:{}".format(
            str(game_id)[:6], target_turn, int(strategy_followed), int(valid),
            reward, int(use_hints),
        )
    )

    return index, {
        "prompt_ids":        prompt_ids,
        "completion_ids":    completion_ids,
        "logprobs":          logprobs,
        "reward":            reward,
        "strategy_followed": strategy_followed,
        "done":              False,
        "metric_line":       metric,
        "messages":          messages,
    }


# ---------------------------------------------------------------------------
# "Full" variant (full-game, env reward + shaping)
# ---------------------------------------------------------------------------

def _run_episode_full(
    index: int,
    prompt: str,
    *,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
) -> tuple[int, dict | None]:
    if num_servers == 0:
        return index, None
    game_id    = int(prompt)
    server_idx = (index + rank) % num_servers
    endpoint   = env_pool[server_idx]["base_url"]
    use_hints  = random.random() < current_hint_prob
    player_id  = game_id % 2

    reset_out = _reset_episode(endpoint, game_id)
    if reset_out is None:
        return index, None
    episode_id, obs = reset_out

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages      = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": obs},
    ]

    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: list[int] | None     = None

    done            = False
    train_reward    = 0.0
    turn_number     = 0
    invalid_count   = 0
    strategy_count  = 0
    total_turns     = 0

    while not done and turn_number < current_max_turn:
        greedy_action = _pick_greedy_action(obs, player_id)
        legal_ids     = _extract_legal_action_ids(obs)

        if len(messages) > 0 and messages[-1]["role"] == "user":
            pass  # already appended
        else:
            messages.append({"role": "user", "content": obs})

        if len(episode_prompt_ids) > _MAX_PROMPT_LEN:
            _log(f"[clobber] Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}")
            break

        with generation_semaphore:
            rollout = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids      = rollout.get("prompt_ids", [])
        completion_ids  = rollout.get("completion_ids", [])
        logprobs        = rollout.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Token accumulation (full-prompt style)
        if turn_number == 0:
            episode_prompt_ids = prompt_ids
            prev_full_ids      = list(prompt_ids)
        else:
            if prev_full_ids and prompt_ids[: len(prev_full_ids)] == prev_full_ids:
                delta = prompt_ids[len(prev_full_ids):]
                if delta:
                    episode_completion_ids.extend(delta)
                    episode_logprobs.extend([0.0] * len(delta))
                    episode_action_mask.extend([0] * len(delta))
            prev_full_ids = list(prompt_ids)

        if completion_ids:
            episode_completion_ids.extend(completion_ids)
            episode_logprobs.extend(logprobs)
            episode_action_mask.extend([1] * len(completion_ids))
            if prev_full_ids is not None:
                prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        action_text = remove_reasoning_tags(completion_text)
        if "Action:" in action_text:
            action_text = action_text.split("Action:")[-1].strip()

        # Validate
        try:
            played_id = int(action_text.strip())
            valid     = played_id in legal_ids
        except Exception:
            valid = False

        if not valid:
            invalid_count += 1

        total_turns += 1
        if valid and greedy_action is not None:
            try:
                if int(action_text.strip()) == int(greedy_action):
                    strategy_count += 1
            except Exception:
                pass

        # Step env
        action_to_send = action_text if valid else (str(legal_ids[0]) if legal_ids else "0")
        step_out = _step_episode(endpoint, action_to_send, episode_id)
        if step_out is None:
            invalid_count += 1
            obs  = ""
            done = False
        else:
            obs, step_reward, done = step_out
            if done:
                train_reward = step_reward
            else:
                messages.append({"role": "user", "content": obs})
                player_id = 1 - player_id

        if _is_invalid(obs):
            invalid_count += 1

        turn_number += 1

    # Clip tokens
    if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
        episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
        episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
        episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]

    strategy_ratio  = strategy_count / max(total_turns, 1)
    mobility_bonus  = strategy_ratio * _MOBILITY_BONUS_STEP * total_turns

    if done:
        shaped_reward = (
            _STRATEGY_WEIGHT * strategy_ratio
            + (1 - _STRATEGY_WEIGHT) * train_reward
            + mobility_bonus
        )
    else:
        shaped_reward = strategy_ratio + mobility_bonus

    shaped_reward -= 0.05 * invalid_count

    metric = (
        "[clobber-FULL] G:{:<6} Done:{} T:{:>2d} | EnvR:{:>5.2f} | R:{:>5.2f} | Strat:{:.1%} | Inv:{:<2}".format(
            str(game_id)[:6], int(done), turn_number,
            train_reward, shaped_reward, strategy_ratio, invalid_count,
        )
    )

    return index, {
        "prompt_ids":     episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "action_mask":    episode_action_mask,
        "logprobs":       episode_logprobs,
        "reward":         shaped_reward,
        "strategy_ratio": strategy_ratio,
        "final_score":    train_reward,
        "done":           done,
        "metric_line":    metric,
        "messages":       messages,
    }


# ---------------------------------------------------------------------------
# Public rollout functions (referenced by env_configs registry)
# ---------------------------------------------------------------------------

def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    _ensure_initialized(trainer)
    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()

    run = functools.partial(
        _run_episode_last,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = {
        "prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0],
        "reward": 0.0, "strategy_followed": False, "done": False,
        "metric_line": "[FALLBACK]", "messages": [],
    }

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    valid = [r for r in results if r is not None]
    if valid:
        avg_strat  = sum(1 for r in valid if r["strategy_followed"]) / len(valid)
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        lines = [f"[clobber-LAST-BATCH] Strategy:{avg_strat:.1%} AvgR:{avg_reward:.3f}"]
        for r in valid:
            lines.append(r.get("metric_line", ""))
        _log("\n".join(lines), flush=True)

    return {
        "prompt_ids":     [r["prompt_ids"]     for r in results],
        "completion_ids": [r["completion_ids"] for r in results],
        "logprobs":       [r["logprobs"]       for r in results],
        "env_rewards":    [r["reward"]         for r in results],
    }


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    _ensure_initialized(trainer)
    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()

    run = functools.partial(
        _run_episode_full,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
    )

    _fallback = {
        "prompt_ids": [1], "completion_ids": [1], "action_mask": [0],
        "logprobs": [1.0], "reward": 0.0, "strategy_ratio": 0.0,
        "final_score": 0.0, "done": False, "metric_line": "[FALLBACK]", "messages": [],
    }

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    valid   = [r for r in results if r is not None]
    finished = sum(1 for r in valid if r.get("done", False))
    avg_r    = sum(r["reward"] for r in valid) / len(valid) if valid else 0
    avg_strat = sum(r["strategy_ratio"] for r in valid) / len(valid) if valid else 0

    lines = [f"[clobber-FULL-BATCH] Done:{finished}/{len(valid)} AvgR:{avg_r:.3f} Strat:{avg_strat:.1%}"]
    for r in valid:
        lines.append(r.get("metric_line", ""))
    _log("\n".join(lines), flush=True)

    return {
        "prompt_ids":     [r["prompt_ids"]     for r in valid],
        "completion_ids": [r["completion_ids"] for r in valid],
        "action_mask":    [r["action_mask"]    for r in valid],
        "logprobs":       [r["logprobs"]       for r in valid],
        "env_rewards":    [r["reward"]         for r in valid],
    }
