"""Fetch a dataset baked into the trainer image (Dockerfile COPY of
scripts/envs/prebaked/{env_name}/) into the path train_sft_env.py reads from.

Kept deliberately simple per the plan: just a path check + a copy, no
fallback logic here — sft_env_config.py decides what to do when nothing
is found.
"""

import os
import shutil


PREBAKED_DATASETS_DIR = "/workspace/scripts/envs/prebaked"


def get_prebaked_dataset_path(env_name: str) -> str | None:
    candidate = os.path.join(PREBAKED_DATASETS_DIR, env_name, "data.json")
    return candidate if os.path.isfile(candidate) else None


def fetch_prebaked_dataset(env_name: str, dest_dir: str) -> str:
    source_path = get_prebaked_dataset_path(env_name)
    if source_path is None:
        raise FileNotFoundError(f"No prebaked dataset found for environment '{env_name}' under {PREBAKED_DATASETS_DIR}")

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"sft_env_{env_name}.json")
    shutil.copy(source_path, dest_path)
    return dest_path
