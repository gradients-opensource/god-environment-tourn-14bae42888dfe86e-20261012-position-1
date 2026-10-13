"""Dataset prep for EnvTask: there's no static dataset to download, so we build one
by calling /reset on the environment server(s) for a sample of seeds per requested
environment, mirroring tokenize_grpo.py's train/dev split convention so train_env.py
can load it the same way GRPO loads its prompt dataset.
"""

import json
import os
import random
from datetime import datetime

import typer
from datasets import Dataset, load_dataset

from core.constants import ENVIRONMENT_TASK_ID_RANGES
from core.environment_rollout import reset_episode


DEFAULT_N_SEEDS_PER_ENV = 256
DEFAULT_DEV_SIZE_PER_ENV = 16


def _sample_seeds(env_name: str, n_seeds: int, seed: int = 42) -> list[int]:
    env_range = ENVIRONMENT_TASK_ID_RANGES[env_name]
    pool_size = env_range["task_id_max"] - env_range["task_id_min"] + 1
    n_seeds = min(n_seeds, pool_size)
    rng = random.Random(seed)
    return rng.sample(range(env_range["task_id_min"], env_range["task_id_max"] + 1), n_seeds)


def build_environment_rows(env_names: list[str], n_seeds_per_env: int = DEFAULT_N_SEEDS_PER_ENV) -> list[dict]:
    rows = []
    for env_name in env_names:
        seeds = _sample_seeds(env_name, n_seeds_per_env)
        for seed in seeds:
            try:
                reset_response = reset_episode(env_name, seed)
            except Exception as exc:
                print(f"[tokenize_env] skipping {env_name}/{seed}, reset failed: {exc}", flush=True)
                continue
            observation = reset_response.get("observation", "")
            if not observation:
                continue
            rows.append({"prompt": observation, "extra_data": {"env_name": env_name, "seed": seed}})
    return rows


def split_rows(rows: list[dict], dev_size_per_env: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_env: dict[str, list[dict]] = {}
    for row in rows:
        by_env.setdefault(row["extra_data"]["env_name"], []).append(row)

    train_rows, dev_rows = [], []
    for env_name, env_rows in by_env.items():
        shuffled = env_rows[:]
        rng.shuffle(shuffled)
        dev_count = min(dev_size_per_env, max(0, len(shuffled) - 1))
        dev_rows.extend(shuffled[:dev_count])
        train_rows.extend(shuffled[dev_count:])
    return train_rows, dev_rows


def get_dataset(path: str, dataset_type: dict) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def main(training_request_path: str):
    with open(training_request_path, "r") as file:
        training_request = json.load(file)

    t1 = datetime.now()
    train_request = training_request["train_request"]
    task_id = train_request["task_id"]
    env_names = train_request["dataset_type"]["environment_names"]
    n_seeds_per_env = train_request.get("n_seeds_per_env", DEFAULT_N_SEEDS_PER_ENV)
    dev_size_per_env = train_request.get("dev_size_per_env", DEFAULT_DEV_SIZE_PER_ENV)

    rows = build_environment_rows(env_names, n_seeds_per_env)
    if not rows:
        raise RuntimeError(
            f"No environment rows could be built for task {task_id} (environments: {env_names}); "
            "check ENVIRONMENT_SERVER_URLS and environment server availability"
        )

    train_rows, dev_rows = split_rows(rows, dev_size_per_env)

    train_path = os.path.join("datasets", f"env_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"env_dev_{task_id}.json")
    os.makedirs("datasets", exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train_rows, f, ensure_ascii=False)
    with open(dev_path, "w") as f:
        json.dump(dev_rows, f, ensure_ascii=False)

    t2 = datetime.now()
    print(
        f"[tokenize_env] built {len(train_rows)} train / {len(dev_rows)} dev rows "
        f"across {len(env_names)} environment(s) in {(t2 - t1).seconds}s",
        flush=True,
    )


if __name__ == "__main__":
    typer.run(main)
