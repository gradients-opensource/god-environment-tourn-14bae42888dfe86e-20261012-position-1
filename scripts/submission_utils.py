"""Crash-safe submission publishing, shared by the trainer callback and the orchestrator.

Why: a submission used to be published with ``rmtree(submission)`` followed by
``copytree(checkpoint, submission)``. For the whole copy (the checkpoint includes the optimizer
state, often several GB) the submission directory was missing or half-written, so a crash or a
kill in that window shipped an EMPTY or CORRUPT model. The orchestrator's end-of-run check only
counted ``>= 2 files``, so a partial copy (e.g. ``adapter_config.json`` without its weights)
passed as a success and then failed to load at eval time -> forfeit. We hit this for real (an
othello flood run ended with an empty submission dir and had to be recovered by hand).

These helpers:
  * publish a checkpoint ATOMICALLY: copy into a sibling temp dir first, then swap it in with two
    renames, so the previous good submission stays intact until the new copy is complete;
  * verify that a directory really contains loadable weights (not just "some files");
  * find the newest COMPLETE checkpoint, so a broken submission can be repaired from it;
  * schedule the EnvTask insurance publish and check that an EnvTask submission is an adapter only
    (the intercode evaluator serves full weights without a tool-call parser: every turn forfeits).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
from typing import Iterable, Optional

_LORA_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_FULL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_SHARD_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
# Full-model weight files, sharded or not, plus their index (never part of a LoRA submission).
_FULL_WEIGHT_RE = re.compile(r"^(model|pytorch_model)(-\d+-of-\d+)?\.(safetensors|bin)(\.index\.json)?$")

# Insurance publish: one extra save early in the SFT, so a crash before the first periodic save (step 200,
# ~14 min into an 8B run) still leaves a trained adapter to ship. Late enough (20% of the planned steps) that
# the adapter is a real early checkpoint rather than a near-base one. INSURANCE_SAVE = False restores the
# previous schedule (first publish at the periodic step).
INSURANCE_SAVE = True
INSURANCE_MIN_STEPS = 50
INSURANCE_FRACTION = 0.2


def _nonempty_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def has_adapter(path: Optional[str]) -> bool:
    """True iff ``path`` holds a complete LoRA adapter: ``adapter_config.json`` + non-empty weights."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "adapter_config.json")) and any(
        _nonempty_file(os.path.join(path, f)) for f in _LORA_WEIGHT_FILES
    )


def submission_has_weights(path: Optional[str]) -> bool:
    """True iff ``path`` holds a loadable model: a LoRA adapter (config + non-empty weights) or a
    full model (config + non-empty weights, with EVERY shard present when sharded)."""
    if not path or not os.path.isdir(path):
        return False
    files = set(os.listdir(path))

    if has_adapter(path):
        return True

    if "config.json" in files:
        for index_name in _SHARD_INDEX_FILES:
            if index_name in files:
                try:
                    with open(os.path.join(path, index_name)) as f:
                        shards = set(json.load(f).get("weight_map", {}).values())
                except (OSError, ValueError):
                    return False
                return bool(shards) and all(_nonempty_file(os.path.join(path, s)) for s in shards)
        return any(_nonempty_file(os.path.join(path, f)) for f in _FULL_WEIGHT_FILES)

    return False


def _write_loss_line(directory: str, loss_line: Optional[str]) -> None:
    if loss_line is not None:
        with open(os.path.join(directory, "loss.txt"), "w") as f:
            f.write(loss_line)


def publish_dir_atomic(source_dir: str, submission_dir: str, loss_line: Optional[str] = None) -> None:
    """Publish ``source_dir`` as ``submission_dir`` without ever leaving it empty or half-written.

    Copy into ``<submission>.publish_tmp`` (a sibling, so the renames stay on one filesystem),
    then rename the old submission aside, rename the temp copy into place and drop the old one.
    Any failure restores the previous submission if it was moved aside and then falls back to the
    legacy direct copy, so this is never worse than the old ``rmtree`` + ``copytree``.
    """
    base = submission_dir.rstrip("/")
    tmp_dir, old_dir = base + ".publish_tmp", base + ".publish_old"
    try:
        for leftover in (tmp_dir, old_dir):
            if os.path.exists(leftover):
                shutil.rmtree(leftover)
        shutil.copytree(source_dir, tmp_dir)
        _write_loss_line(tmp_dir, loss_line)
        if os.path.exists(submission_dir):
            os.rename(submission_dir, old_dir)
        os.rename(tmp_dir, submission_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as exc:  # never lose the submission because of the safer path
        print(f"[publish] atomic publish failed ({exc!r}) -> legacy direct copy", flush=True)
        if not os.path.exists(submission_dir) and os.path.exists(old_dir):
            os.rename(old_dir, submission_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(submission_dir):
            shutil.rmtree(submission_dir)
        shutil.copytree(source_dir, submission_dir)
        _write_loss_line(submission_dir, loss_line)


def latest_complete_checkpoint(roots: Iterable[Optional[str]], adapter_only: bool = False) -> Optional[str]:
    """Newest ``checkpoint-<step>`` under any of ``roots`` that passes ``submission_has_weights``
    (``has_adapter`` when ``adapter_only``)."""
    complete = has_adapter if adapter_only else submission_has_weights
    best: Optional[tuple[int, str]] = None
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for ckpt in glob.glob(os.path.join(root, "checkpoint-*")):
            match = re.search(r"checkpoint-(\d+)$", ckpt.rstrip("/"))
            if match and complete(ckpt):
                step = int(match.group(1))
                if best is None or step > best[0]:
                    best = (step, ckpt)
    return best[1] if best else None


def repair_submission(submission_dir: str, checkpoint_roots: Iterable[Optional[str]],
                      adapter_only: bool = False) -> str:
    """Make sure ``submission_dir`` holds loadable weights (a LoRA adapter when ``adapter_only``).
    Returns what happened:

    ``"ok"``        - it already did (nothing touched);
    ``"recovered"`` - it was empty/partial and the newest complete checkpoint was published;
    ``"cleared"``   - it was partial (or not an adapter) and no complete checkpoint exists, so it was
                      emptied for the caller's fallback — a loadable model beats a corrupt one that
                      forfeits at load time;
    ``"empty"``     - nothing there and nothing to recover.
    """
    if (has_adapter if adapter_only else submission_has_weights)(submission_dir):
        return "ok"
    ckpt = latest_complete_checkpoint(checkpoint_roots, adapter_only=adapter_only)
    if ckpt:
        step = re.search(r"checkpoint-(\d+)", ckpt).group(1)
        publish_dir_atomic(ckpt, submission_dir, loss_line=f"{step},recovered")
        return "recovered"
    if os.path.isdir(submission_dir) and os.listdir(submission_dir):
        shutil.rmtree(submission_dir)
        os.makedirs(submission_dir, exist_ok=True)
        return "cleared"
    return "empty"


def insurance_save_step(planned_steps) -> Optional[int]:
    """Step of the insurance publish: max(INSURANCE_MIN_STEPS, INSURANCE_FRACTION of the planned steps).
    None when disabled, when the step count is unknown, or when the run ends first (its final save is then
    the first publish anyway). R1 shape: 811 planned -> 162; R2 shape: 1292 -> 258."""
    if not INSURANCE_SAVE or not planned_steps or planned_steps <= 0:
        return None
    step = max(INSURANCE_MIN_STEPS, math.floor(INSURANCE_FRACTION * planned_steps))
    return step if step < planned_steps else None


def reanchor_insurance_step(insurance_step: Optional[int], stop_step) -> Optional[int]:
    """Insurance step once the trainer's clock stop (budget_controller.ClockStop) has fixed the run's last
    step. Kept while it still lands before ``stop_step``; otherwise re-derived from ``stop_step``, so a box
    too slow to reach the insurance step of the PLANNED steps still gets its early save instead of none
    (R1 shape: 811 planned -> 162, a stop at 120 -> 50). None stays None when INSURANCE_SAVE is off."""
    if not stop_step or stop_step <= 0:
        return insurance_step
    if insurance_step is not None and insurance_step < stop_step:
        return insurance_step
    return insurance_save_step(stop_step)


def adapter_submission_problems(path: str) -> list[str]:
    """Pre-upload check for an EnvTask submission: a complete LoRA adapter and no full-model weights.
    Returns the problems found ([] = ready to upload)."""
    if not os.path.isdir(path):
        return [f"{path} does not exist"]
    problems = []
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        problems.append("adapter_config.json missing")
    if not any(_nonempty_file(os.path.join(path, f)) for f in _LORA_WEIGHT_FILES):
        problems.append("adapter weights missing or empty")
    full = sorted(f for f in os.listdir(path) if _FULL_WEIGHT_RE.match(f))
    if full:
        problems.append(f"full-model weight files present: {', '.join(full)}")
    return problems


def enforce_adapter_submission(path: str) -> list[str]:
    """Drop stray full-model weight files sitting next to a complete adapter (the evaluator would treat the
    repo as a LoRA and delete them itself, after a multi-GB upload), then re-check. A directory without a
    complete adapter is left untouched: those files may be its only weights."""
    if has_adapter(path):
        for name in os.listdir(path):
            if _FULL_WEIGHT_RE.match(name):
                os.remove(os.path.join(path, name))
    return adapter_submission_problems(path)
