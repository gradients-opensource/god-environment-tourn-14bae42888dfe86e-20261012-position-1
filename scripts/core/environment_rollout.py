"""HTTP client and reward-function factory for EnvTask environment rollouts.

EnvTask has no static dataset: the validator exposes one or more environment
sidecars via the ENVIRONMENT_SERVER_URLS env var (comma-separated base URLs),
each speaking a /reset + /step HTTP contract. The GRPO trainer in this repo
generates only the *first* turn of an episode through its own sampling step;
every following turn is generated here locally (no grad) purely to advance
the episode until `done`, so a final episode reward can be collected and fed
back as the GRPO reward signal — mirroring the pattern used in G.O.D's
dockerfiles/environment_functions/*.py reference rollouts.
"""

import os
import re

import requests
import torch

from core.constants import ENVIRONMENT_DEFAULT_MAX_TURNS
from core.constants import ENVIRONMENT_SERVER_URLS_ENV_VAR


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

GAME_ENVIRONMENTS = {"gin_rummy", "liars_dice", "leduc_poker", "othello", "goofspiel"}
INTERCODE_ENVIRONMENT = "intercode"

RESET_TIMEOUT_SECONDS = 300
STEP_TIMEOUT_SECONDS = 120

ACTION_PATTERN = re.compile(r"Action:\s*(-?\d+)")
BASH_PATTERN = re.compile(r"execute_bash\(command=(\"|')(.*?)(\"|')\)", re.DOTALL)
SUBMIT_PATTERN = re.compile(r"submit\(\)")


def get_environment_server_urls() -> list[str]:
    raw = os.environ.get(ENVIRONMENT_SERVER_URLS_ENV_VAR, "")
    return [url.strip() for url in raw.split(",") if url.strip()]


def pick_environment_server_url() -> str:
    urls = get_environment_server_urls()
    if not urls:
        raise RuntimeError(
            f"{ENVIRONMENT_SERVER_URLS_ENV_VAR} is not set or empty; cannot run environment rollouts"
        )
    return urls[LOCAL_RANK % len(urls)]


def build_reset_payload(env_name: str, seed: int) -> dict:
    if env_name in GAME_ENVIRONMENTS:
        return {
            "task_id": seed,
            "seed": seed,
            "opponent": "mcts",
            "mcts_max_simulations": 25,
            "mcts_num_rollouts": 1,
        }
    if env_name == INTERCODE_ENVIRONMENT:
        # NOTE: intercode's miner-facing sidecar wire format is unconfirmed against
        # public sources (the validator's own eval_intercode.py restores local
        # filesystem snapshots rather than calling an HTTP server). This minimal
        # payload assumes the same /reset contract is exposed to miners and must
        # be validated against a real environment server before tournament use.
        return {"task_id": seed, "seed": seed}
    raise ValueError(f"Unknown environment name: {env_name}")


def reset_episode(env_name: str, seed: int, base_url: str | None = None) -> dict:
    base_url = base_url or pick_environment_server_url()
    payload = build_reset_payload(env_name, seed)
    response = requests.post(f"{base_url}/reset", json=payload, timeout=RESET_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def step_episode(episode_id: str, action: str, base_url: str | None = None) -> dict:
    base_url = base_url or pick_environment_server_url()
    payload = {"action": action, "episode_id": episode_id}
    response = requests.post(f"{base_url}/step", json=payload, timeout=STEP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def extract_action(env_name: str, completion_text: str) -> str:
    if env_name in GAME_ENVIRONMENTS:
        match = ACTION_PATTERN.search(completion_text)
        return match.group(1) if match else "0"
    if env_name == INTERCODE_ENVIRONMENT:
        if SUBMIT_PATTERN.search(completion_text):
            return "submit()"
        match = BASH_PATTERN.search(completion_text)
        if match:
            return f'execute_bash(command="{match.group(2)}")'
        return "submit()"
    raise ValueError(f"Unknown environment name: {env_name}")


def generate_next_turn(model, tokenizer, messages: list[dict], max_new_tokens: int = 256) -> str:
    """Generate one assistant turn locally (no grad) to advance an episode past turn 1."""
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_first_turn(model, tokenizer, observation_text: str) -> str:
    """Generate the opening turn locally when there's no GRPOTrainer to produce it
    (used by the offline trajectory generator; during GRPO training the trainer's
    own sampling step produces turn 1 instead, see run_episode)."""
    return generate_next_turn(model, tokenizer, [{"role": "user", "content": observation_text}])


def _play_episode_loop(
    env_name: str,
    seed: int,
    episode_id: str,
    messages: list[dict],
    action: str,
    model,
    tokenizer,
    max_turns: int,
) -> float:
    """Shared turn-by-turn loop: send `action`, generate the next turn locally (no
    grad) from the resulting observation, repeat until `done` or `max_turns`.
    Mutates `messages` in place and returns the final episode reward."""
    final_reward = 0.0

    for turn in range(max_turns):
        try:
            step_response = step_episode(episode_id, action)
        except Exception as exc:
            print(f"[env_rollout] step failed for {env_name}/{seed} turn {turn}: {exc}", flush=True)
            break

        final_reward = float(step_response.get("reward", final_reward))
        if bool(step_response.get("done", False)):
            break

        observation = step_response.get("observation", "")
        messages.append({"role": "user", "content": observation})
        next_completion = generate_next_turn(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": next_completion})
        action = extract_action(env_name, next_completion)

    return final_reward


def run_episode(
    env_name: str,
    seed: int,
    model,
    tokenizer,
    first_completion_text: str,
    max_turns: int = ENVIRONMENT_DEFAULT_MAX_TURNS,
) -> float:
    try:
        reset_response = reset_episode(env_name, seed)
    except Exception as exc:
        print(f"[env_rollout] reset failed for {env_name}/{seed}: {exc}", flush=True)
        return 0.0

    episode_id = reset_response.get("episode_id", str(seed))
    observation = reset_response.get("observation", "")
    messages = [
        {"role": "user", "content": observation},
        {"role": "assistant", "content": first_completion_text},
    ]
    action = extract_action(env_name, first_completion_text)

    return _play_episode_loop(env_name, seed, episode_id, messages, action, model, tokenizer, max_turns)


def play_full_episode(
    env_name: str,
    seed: int,
    model,
    tokenizer,
    max_turns: int = ENVIRONMENT_DEFAULT_MAX_TURNS,
) -> tuple[list[dict], float]:
    """Play a full episode from scratch (no externally-provided first turn) —
    used by scripts/envs/generate_trajectories.py for offline data generation,
    where there's no GRPOTrainer to produce turn 1."""
    try:
        reset_response = reset_episode(env_name, seed)
    except Exception as exc:
        print(f"[env_rollout] reset failed for {env_name}/{seed}: {exc}", flush=True)
        return [], 0.0

    episode_id = reset_response.get("episode_id", str(seed))
    observation = reset_response.get("observation", "")
    first_completion = generate_first_turn(model, tokenizer, observation)
    messages = [
        {"role": "user", "content": observation},
        {"role": "assistant", "content": first_completion},
    ]
    action = extract_action(env_name, first_completion)

    final_reward = _play_episode_loop(env_name, seed, episode_id, messages, action, model, tokenizer, max_turns)
    return messages, final_reward


def make_environment_reward_func(model, tokenizer, max_turns: int = ENVIRONMENT_DEFAULT_MAX_TURNS):
    """Build a reward_func compatible with trl's GRPOTrainer reward_funcs contract.

    Each dataset row's extra_data carries {"env_name": ..., "seed": ...}, set by
    tokenize_env.py — a single reward_func handles every environment in the batch
    by dispatching per-row instead of binding to one environment name.
    """

    def reward_func(completions: list[str], extra_data: list[dict], **kwargs) -> list[float]:
        rewards = []
        for completion, meta in zip(completions, extra_data):
            reward = run_episode(
                meta["env_name"], meta["seed"], model, tokenizer, completion, max_turns=max_turns
            )
            rewards.append(reward)
        return rewards

    return reward_func
