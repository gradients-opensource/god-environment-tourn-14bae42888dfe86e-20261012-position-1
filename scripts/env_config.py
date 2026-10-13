"""
EnvTask orchestrator: delegates sizing/scheduling to sft_env_config + grpo_env_config.

Two modes:
  * Default (SFT warm-start → GRPO RL): text_trainer.py runs generate_cmd and sft_run_cmd
    inline (outside the retry loop), then run_cmd (GRPO) via run_training (retries + success.txt).
  * ENV_SKIP_GRPO set (SFT-only): the SFT warm-start IS the final model. It runs as the main
    run_cmd (retries + success.txt), generate_cmd still runs inline, no GRPO phase. Chosen for
    tight wall-clock budgets where RL can't converge and would only starve the SFT pass.
"""

import os

from sft_env_config import get_training_json as _sft_training_json
from grpo_env_config import get_training_json as _grpo_training_json


def _skip_grpo() -> bool:
    """SFT-only mode toggle. Baked into the trainer image via `ENV ENV_SKIP_GRPO=1`
    (see standalone-text-trainer.dockerfile); set to 0/unset to restore SFT→GRPO."""
    return os.getenv("ENV_SKIP_GRPO", "").strip().lower() in ("1", "true", "yes")


def get_training_json(train_info: dict) -> dict:
    # Phase 1+2: SFT warm-start (data generation + SFT training)
    sft_result = _sft_training_json(train_info)
    generate_cmd = sft_result.get("generate_cmd", "")
    sft_run_cmd = sft_result["run_cmd"]
    sft_request = sft_result["train_request"]

    # SFT-only: run the SFT as the main train_cmd. train_sft_env.py already saves to
    # submission_dir + writes success.txt (same contract as GRPO), so it's a valid final
    # submission. generate_cmd still runs inline first; no separate sft_run_cmd (so text_trainer
    # doesn't run SFT twice) and no GRPO phase. SFT's WSD clock callback then uses the full
    # wall-clock budget up to end_time.
    if _skip_grpo():
        print("[env_config] ENV_SKIP_GRPO set → SFT-only mode (skipping GRPO RL phase)")
        return {
            "train_request": sft_request,
            "run_cmd": sft_run_cmd,
            "generate_cmd": generate_cmd,
        }

    # Phase 3: GRPO RL training (the main training loop with success.txt check)
    grpo_result = _grpo_training_json(train_info)
    grpo_run_cmd = grpo_result["run_cmd"]

    # text_trainer.py writes ONE base request file (training_request_<task_id>.json) from the
    # dict below, and the SFT warm-start step (train_sft_env.py) reads it. That step needs
    # dataset_path (to locate the generated trajectories) plus a few save/schedule fields that
    # ONLY sft_env_config sets (save_before_remaining_time, lr_schedule=wsd, wsd_* fracs, ...).
    # Without them train_sft_env either skips warm-start ("No dataset_path in train_request") or
    # KeyErrors on the first required field. Backfill every SFT key that the GRPO request lacks —
    # setdefault never overwrites a GRPO value, and GRPO ignores the extra keys (it reads its own
    # per-attempt request file and CLI args), so GRPO behaviour is unchanged.
    train_request = grpo_result["train_request"]
    for key, value in sft_request.items():
        train_request.setdefault(key, value)

    # Return all three phases separately so text_trainer.py can:
    # - Run generate_cmd and sft_run_cmd inline (no retry, failures are graceful)
    # - Run grpo_run_cmd via run_training (with retries and success.txt check)
    return {
        "train_request": train_request,
        "run_cmd": grpo_run_cmd,
        "generate_cmd": generate_cmd,
        "sft_run_cmd": sft_run_cmd,
    }
