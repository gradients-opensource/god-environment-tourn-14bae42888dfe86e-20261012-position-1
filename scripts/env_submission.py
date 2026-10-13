"""EnvTask failure-path backstop: the submission is always a LoRA adapter, never full weights.

Full weights get ``parser_model_id=<repo id>`` at the intercode evaluator with no tool-call parser,
so every turn forfeits (see submission_utils.py's module docstring). text_trainer.py's EnvTask branch
used to fall through to add_random_noise.py (a bf16 FULL-weight fallback) whenever nothing loadable
existed yet; it no longer can (text_trainer.py guards that call out for EnvTask) and calls
``finalize_env_submission`` here instead, which layers two recoveries on top of the periodic/insurance
checkpoint saves (submission_utils.insurance_save_step):

  1. ``repair_submission(..., adapter_only=True)`` — publish the newest ADAPTER checkpoint already on
     disk (a full-weight checkpoint is invisible to this search, so it can never be "recovered" into
     the submission).
  2. ``rescue_lora`` — nothing recoverable at all (crash before step INSURANCE_MIN_STEPS, or before
     that): launch one short LoRA SFT (RESCUE_MAX_STEPS) so the submission is a real, if lightly
     trained, adapter. Tries the dataset already generated for this task first; if that yields no
     checkpoint (empty split, immediate crash), falls back to rows built directly from the requested
     intercode dataset. Time-boxed by RESCUE_RESERVE_MINUTES so a doomed attempt never eats the last
     minutes of the pod's budget.
"""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Callable, Optional

from submission_utils import latest_complete_checkpoint, repair_submission

# RESCUE_LORA = False restores the previous behaviour: insurance/periodic checkpoints only, no
# rescue run when none exist (repair_submission then just clears the broken submission).
RESCUE_LORA = True
RESCUE_MAX_STEPS = 100
# Minimum wall-clock left to even attempt a rescue: launching train_sft_env.py, generating a small
# dataset when needed, training RESCUE_MAX_STEPS and saving all cost real time. Below this, shipping
# nothing (an empty submission fails cleanly) is safer than a run that never reaches its own save.
RESCUE_RESERVE_MINUTES = 10
# run_training attempts per rescue dataset. The rescue most often follows a main run that died of OOM
# with the same batch size; run_training halves per_device_train_batch_size (and exports OOM_HALVINGS
# for the packed batch) on an OOM retry, so 2 gives the rescue one halving instead of the same OOM.
RESCUE_RETRIES = 2


def is_lora_cmd(cmd: str) -> bool:
    """True iff ``cmd`` requests a LoRA run (``--use_lora True``); false for a full fine-tune or an
    explicit ``--use_lora False``."""
    match = re.search(r"--use_lora\s+(\S+)", cmd)
    return bool(match) and match.group(1).strip().lower() == "true"


def _set_flag(cmd: str, name: str, value: str) -> str:
    """Set ``--name`` to ``value`` in a (possibly multi-line, backslash-continued) CLI command:
    replace it in place if present, else append it at the end."""
    pattern = re.compile(rf"(--{name}\s+)(\S+)")
    if pattern.search(cmd):
        return pattern.sub(lambda m: m.group(1) + value, cmd, count=1)
    return cmd.rstrip() + f" --{name} {value}"


def _minutes_left(end_time: str) -> float:
    end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    return (end - now).total_seconds() / 60.0


def _default_run_training(cmd: str, log_path: str, retries: int) -> bool:
    from text_trainer import run_training  # heavy import: only touched by the real (non-test) path
    return run_training(cmd, log_path, "rescue", retries, "EnvTask", "rescue")


def _default_run_cmd(cmd: str, log_path: str) -> None:
    from text_trainer import run_cmd_with_log
    run_cmd_with_log(cmd, log_path)


def rescue_lora(
    train_info: dict,
    base_cmd: str,
    output_dir: str,
    ds_folder: str,
    run_training: Optional[Callable[[str, str, int], bool]] = None,
    run_cmd: Optional[Callable[[str, str], None]] = None,
) -> list[str]:
    """Train a short LoRA SFT so the submission is a real adapter instead of untouched base weights.
    Returns the run dir(s) to search for the resulting checkpoint (for the caller's
    ``repair_submission``), or ``[]`` when rescue is disabled, out of time, or ``base_cmd`` is not a
    LoRA run to begin with (rescuing a full-weight run would just reproduce the problem).

    Tries the dataset already generated for this task (``train_request["dataset_path"]``) first — its
    rows are ready, nothing to rebuild. If that path doesn't exist yet, or training on it produces no
    checkpoint (e.g. an empty split), falls back once to rows built directly from the requested
    intercode dataset via ``envs.intercode_dataset`` (independent of whatever is/was at the generated
    path, so a barren generated dataset never poisons the fallback attempt too).
    """
    run_training = run_training or _default_run_training
    run_cmd = run_cmd or _default_run_cmd
    if not RESCUE_LORA or not is_lora_cmd(base_cmd):
        print(f"[rescue] skipped (RESCUE_LORA={RESCUE_LORA}, lora_cmd={is_lora_cmd(base_cmd)})", flush=True)
        return []

    request = train_info["train_request"]
    minutes_left = _minutes_left(request["end_time"])
    if minutes_left < RESCUE_RESERVE_MINUTES:
        print(f"[rescue] {minutes_left:.1f} min left (< {RESCUE_RESERVE_MINUTES}) -> skipped", flush=True)
        return []

    task_id = request["task_id"]
    generated_path = request["dataset_path"]
    fallback_path = f"{generated_path}.rescue_intercode"
    candidates = ([generated_path] if os.path.isdir(generated_path) else []) + [fallback_path]

    for attempt, dataset_path in enumerate(candidates):
        if attempt > 0:  # the previous attempt spent real time: re-check the floor before starting another
            minutes_left = _minutes_left(request["end_time"])
            if minutes_left < RESCUE_RESERVE_MINUTES:
                print(f"[rescue] {minutes_left:.1f} min left (< {RESCUE_RESERVE_MINUTES}) -> attempt {attempt} "
                      f"skipped", flush=True)
                break
        if not os.path.isdir(dataset_path):
            build_cmd = f"python -m envs.intercode_dataset --output_path {dataset_path}"
            print(f"[rescue] building rows from the requested intercode dataset -> {dataset_path}", flush=True)
            run_cmd(build_cmd, os.path.join(ds_folder, f"rescue_build_{task_id}_{attempt}.log"))
            if not os.path.isdir(dataset_path):
                print(f"[rescue] build produced nothing at {dataset_path}", flush=True)
                continue

        run_dir = f"{output_dir}_rescue_{attempt}"
        cmd = _set_flag(base_cmd, "output_dir", run_dir)
        cmd = _set_flag(cmd, "max_steps", str(RESCUE_MAX_STEPS))
        rescue_request = dict(request)
        rescue_request["dataset_path"] = dataset_path
        rescue_request["max_steps"] = RESCUE_MAX_STEPS
        # train_info (and so train_request) starts "first_time" in production (text_trainer.py); the main
        # GRPO/EnvTask loop overrides that to "none" on its own deepcopy (c_train_info) before training, but
        # never touches train_info/train_request itself, so it still leaks in here. Left alone, the rescue's
        # own CustomEvalSaveCallback would run the LR-continuation probe at checking_step and stop the rescue
        # (should_training_stop, save_strategy="no") long before RESCUE_MAX_STEPS, shipping an early/empty
        # checkpoint instead of the intended rescue adapter — mirror the main loop's override here too.
        rescue_request["checking_mode"] = "none"
        request_path = os.path.join(ds_folder, f"rescue_request_{task_id}_{attempt}.json")
        with open(request_path, "w") as f:
            json.dump({"train_request": rescue_request}, f, indent=4, ensure_ascii=False)
        cmd = _set_flag(cmd, "request_path", request_path)

        print(f"[rescue] attempt {attempt}: {RESCUE_MAX_STEPS} steps on {dataset_path} -> {run_dir}", flush=True)
        log_path = os.path.join(ds_folder, f"rescue_train_{task_id}_{attempt}.log")
        run_training(cmd, log_path, RESCUE_RETRIES)
        if latest_complete_checkpoint([run_dir], adapter_only=True):
            return [run_dir]
        print(f"[rescue] attempt {attempt} produced no adapter checkpoint", flush=True)

    return []


def finalize_env_submission(submission_dir: str, checkpoint_roots, rescue: Callable[[], list[str]]) -> str:
    """Make sure ``submission_dir`` holds a real LoRA adapter, escalating through: the submission as
    it stands -> the newest adapter checkpoint under ``checkpoint_roots`` -> a rescue LoRA run.
    Returns ``repair_submission``'s outcome ("ok" / "recovered" / "cleared" / "empty"), or
    ``"rescued"`` when the rescue run is what supplied the adapter. Never raises — a broken rescue
    must not take the whole upload down; the submission is then whatever repair_submission already
    left it as."""
    outcome = repair_submission(submission_dir, checkpoint_roots, adapter_only=True)
    if outcome in ("ok", "recovered"):
        return outcome

    try:
        rescue_dirs = rescue()
    except Exception as exc:
        print(f"[rescue] crashed ({exc!r}), no rescue adapter available", flush=True)
        rescue_dirs = []
    if not rescue_dirs:
        return outcome

    rescue_outcome = repair_submission(submission_dir, rescue_dirs, adapter_only=True)
    return "rescued" if rescue_outcome in ("ok", "recovered") else outcome
