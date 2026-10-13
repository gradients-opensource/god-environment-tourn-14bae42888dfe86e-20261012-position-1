"""Load the whitelisted SWE-ZERO trajectories dataset → plain-text SFT rows for `swe_infinite`.

``gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered`` is the OFFICIAL whitelisted
SFT source for the `swe_infinite` env (validator PR #1299 — see [[project-swe-infinite-incoming]]).

Format note — this env is DIFFERENT from the game/intercode/pvp sources:
    swe_infinite is evaluated by the **MiniSWE agent in PLAIN TEXT**. Each assistant turn is

        THOUGHT: <reasoning>
        ```bash
        <exactly one command>
        ```

    i.e. a normal chat message whose ``content`` is text — NOT an OpenAI ``tool_calls`` entry.
    So we pass the dataset's ``messages`` (already ``[{role, content}, ...]``) straight through
    with NO ``tools`` and NO ``tool_calls``: the training distribution then matches how the
    miniswe evaluator parses the model's output (it greps the ```bash``` block from the text).

Row schema (SWE-ZERO): ``instance_id, repo, messages:[{role, content}], trajectory_format,
exit_status, duration_sec`` (50k filtered trajectories, ~1.7 GB).

Output row: ``{"messages": [{role, content}, ...]}`` — plain multi-turn chat, no tools.

Because these rows carry no ``tool_calls``/``tools`` while the game & intercode sources do,
``generate_trajectories.merge_datasets`` aligns the Arrow schemas before concatenating.

Dataset path convention (follows miner_dataset_loader.py):
    MINER_DATASETS_DIR/<hf_org>--<hf_repo>/   (-- replaces / in the HF repo name)

Usage (run from /workspace/scripts/):
    python -m envs.swe_infinite_dataset --output_path /path/to/dataset

NOTE: like intercode_dataset.py, the exact on-disk schema is not verifiable offline — run
against the real validator-mounted dataset once; a schema mismatch degrades gracefully
(rows skipped, not a crash).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from envs.miner_dataset_loader import _get_miner_datasets_inventory, _load_one_dataset

SWE_INFINITE_HF_REPO = "gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered"

_VALID_ROLES = {"system", "user", "assistant"}

# Quality gate. SWE-ZERO-50k is a RANDOM diverse subset of the 12M upstream, so it inherits
# the raw outcome mix — measured on the full 50k: only 2008 (4%) trajectories reach a
# submitted patch ("Submitted"); the other 47992 (96%) are "incomplete" (the agent ran out of
# turns / gave up WITHOUT solving). SFT is imitation, so training on the incomplete majority
# teaches the model to explore aimlessly and never fix the bug. By default we keep ONLY the
# completed ("Submitted") trajectories — imitate the winners, not the failures. This is the
# single biggest quality lever for the boss-round swe_infinite env (and 25x fewer rows keeps
# tokenize fast). Pass keep_exit_statuses=None to train on everything.
_DEFAULT_KEEP_EXIT_STATUSES = frozenset({"submitted"})

# Fit-to-window (anti-truncation). SWE trajectories are long multi-turn conversations and the
# ACTUAL FIX (the edit + submit) is at the END, but the trainer truncates each row to
# max_length tokens FROM THE FRONT — so a raw long trajectory trains the model on early
# exploration and throws the fix away. Two cheap, low-risk moves keep the fix inside the window:
#   1. cap each OBSERVATION (the command-output / file-dump user turns — the bloat; the model's
#      THOUGHT+bash turns are short) to _OBS_CAP_CHARS,
#   2. if the trajectory is still over _MAX_TRAJECTORY_CHARS, keep the [system, issue] anchor
#      plus as many turns FROM THE END (where the fix lives) as fit.
# Char budgets are a proxy for tokens (~3-4 chars/token for code); kept deliberately under
# 4096 tokens so the trainer's own truncation never fires and clips the fix. Both are
# tunable/disable-able (max_chars=None keeps trajectories whole).
_MAX_TRAJECTORY_CHARS = 11_000   # ~2.8-3.7k tokens — safely under the default max_length 4096
_OBS_CAP_CHARS = 2_000           # per observation (command output / file dump)
_OBS_TRUNCATION_MARKER = "\n...[observation truncated]..."


def _find_swe_dataset() -> "tuple[str, Path] | None":
    """Return (hf_name, local_path) for the SWE-ZERO dataset in the miner inventory, or None.

    Matches on 'swe' — the only whitelisted SWE source is SWE-ZERO, and none of the other
    whitelisted datasets (intercode, pvp-tool-calling, games) contain 'swe'.
    """
    for hf_name, local in _get_miner_datasets_inventory():
        if "swe" in hf_name.lower():
            return hf_name, local
    return None


def _clean_messages(messages: Any) -> "list[dict[str, str]] | None":
    """Validate + normalize one trajectory's messages to plain ``{role, content}``.

    Drops malformed turns; returns None if the trajectory has no usable assistant turn
    (nothing to compute loss on) or collapses to fewer than 2 messages.
    """
    if not isinstance(messages, list) or not messages:
        return None
    out: list[dict[str, str]] = []
    has_assistant = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in _VALID_ROLES or content is None:
            continue
        if role == "assistant":
            if not str(content).strip():
                continue  # an empty assistant turn carries no training signal
            has_assistant = True
        out.append({"role": role, "content": content})
    if not has_assistant or len(out) < 2:
        return None
    return out


def _total_chars(messages: "list[dict[str, str]]") -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _cap_observations(
    messages: "list[dict[str, str]]", obs_cap: int
) -> "list[dict[str, str]]":
    """Truncate each OBSERVATION — the user turns AFTER the first (the issue) — to ``obs_cap``
    chars. Those carry the command output / file dumps (the bloat); the issue and every
    assistant (THOUGHT+bash) turn are kept whole, so no command/fix is ever shortened."""
    out: list[dict[str, str]] = []
    seen_issue = False
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "user" and not seen_issue:
            seen_issue = True            # the issue — keep full
            out.append(m)
        elif role == "user" and len(content) > obs_cap:
            out.append({"role": "user", "content": content[:obs_cap] + _OBS_TRUNCATION_MARKER})
        else:
            out.append(m)
    return out


def _fit_messages(
    messages: "list[dict[str, str]]", max_chars: int, obs_cap: int
) -> "list[dict[str, str]] | None":
    """Shrink a trajectory so the trainer's front-truncation can't clip the fix (see the
    module comment). Cap observations first; if still too long, keep the [system, issue]
    anchor + as many turns from the END as fit ``max_chars``. Returns None only if nothing
    usable survives (no assistant turn left)."""
    msgs = _cap_observations(messages, obs_cap)
    if _total_chars(msgs) <= max_chars:
        return msgs

    anchor: list[dict[str, str]] = []
    idx = 0
    if msgs and msgs[0].get("role") == "system":
        anchor.append(msgs[0]); idx = 1
    if idx < len(msgs) and msgs[idx].get("role") == "user":
        anchor.append(msgs[idx]); idx += 1          # the issue
    rest = msgs[idx:]
    budget = max_chars - _total_chars(anchor)
    if budget < 500 or not rest:
        return msgs                                  # anchor alone is huge (rare) — let the trainer truncate

    kept: list[dict[str, str]] = []
    total = 0
    for m in reversed(rest):                         # walk backward from the fix/submit
        c = len(m.get("content") or "")
        if kept and total + c > budget:
            break
        kept.append(m); total += c
    kept.reverse()
    while kept and kept[0].get("role") != "assistant":
        kept.pop(0)                                  # a window after the issue must open on an assistant turn
    if not kept:
        return None
    return anchor + kept


def _passes_exit_filter(
    row: dict[str, Any], keep: "set[str] | frozenset[str] | None"
) -> bool:
    """True if the row's ``exit_status`` is allowed. ``keep=None`` allows all; an absent/empty
    exit_status is allowed (nothing to filter on). Compared case-insensitively."""
    if keep is None:
        return True
    status = str(row.get("exit_status") or "").strip().lower()
    return (not status) or status in keep


def build_swe_infinite_sft_dataset(
    keep_exit_statuses: "set[str] | frozenset[str] | None" = _DEFAULT_KEEP_EXIT_STATUSES,
    max_chars: "int | None" = _MAX_TRAJECTORY_CHARS,
    obs_cap: int = _OBS_CAP_CHARS,
) -> "DatasetDict | None":
    """Find, load, and convert the SWE-ZERO miner dataset to plain-messages SFT rows.

    Args:
        keep_exit_statuses: keep only trajectories whose ``exit_status`` is in this set
            (compared case-insensitively). Defaults to ``{"submitted"}`` — the completed
            trajectories — because ~96% of the raw set is "incomplete" (failed attempts) and
            imitating those hurts. Pass None to keep every trajectory.
        max_chars: fit each trajectory under this many chars so the trainer's front-truncation
            can't clip the fix at the end (see _fit_messages). Pass None to keep trajectories
            whole (raw), letting max_length do the truncating.
        obs_cap: truncate each observation (command-output user turn) to this many chars.
    """
    found = _find_swe_dataset()
    if found is None:
        print(
            f"[swe_infinite_dataset] '{SWE_INFINITE_HF_REPO}' not found in miner dataset "
            "inventory.\n  Add it to requested_datasets and ensure MINER_DATASETS_DIR is set.",
            flush=True,
        )
        return None

    hf_name, local = found
    print(f"[swe_infinite_dataset] Loading {hf_name} from {local}", flush=True)

    raw = _load_one_dataset(local)
    if raw is None:
        print(f"[swe_infinite_dataset] Failed to load dataset from {local}", flush=True)
        return None

    print(f"[swe_infinite_dataset] Loaded {len(raw)} trajectories", flush=True)

    keep = {s.strip().lower() for s in keep_exit_statuses} if keep_exit_statuses else None
    examples: list[dict[str, Any]] = []
    dropped_status = 0
    dropped_malformed = 0
    fitted = 0
    for raw_row in raw:
        row = dict(raw_row)
        if not _passes_exit_filter(row, keep):
            dropped_status += 1
            continue
        messages = _clean_messages(row.get("messages"))
        if messages is None:
            dropped_malformed += 1
            continue
        if max_chars:
            n_before = len(messages)
            messages = _fit_messages(messages, max_chars, obs_cap)
            if messages is None:
                dropped_malformed += 1
                continue
            if len(messages) < n_before:
                fitted += 1                       # trajectory was over-length → trimmed to keep the fix
        examples.append({"messages": messages})

    print(
        f"[swe_infinite_dataset] Kept {len(examples)}/{len(raw)} trajectories "
        f"(quality filter exit_status∈{sorted(keep) if keep else 'ALL'}: "
        f"dropped {dropped_status} by status, {dropped_malformed} malformed; "
        f"{fitted} trimmed to fit ~{max_chars} chars so the fix survives)",
        flush=True,
    )

    if not examples:
        print(
            "[swe_infinite_dataset] No trajectories left after filtering — check exit_status "
            "values (expected 'Submitted') or pass keep_exit_statuses=None.",
            flush=True,
        )
        return None

    return DatasetDict({"train": Dataset.from_list(examples)})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert the SWE-ZERO whitelisted dataset to plain-messages SFT format."
    )
    p.add_argument("--output_path", required=True, help="Where to save the DatasetDict")
    p.add_argument(
        "--keep_exit_statuses",
        nargs="*",
        default=None,
        help="Keep only these exit_status values (case-insensitive). Omit for the default "
             "quality filter (Submitted only); pass 'all' (or no values) to keep everything.",
    )
    p.add_argument(
        "--max_chars", type=int, default=_MAX_TRAJECTORY_CHARS,
        help=f"Fit each trajectory under this many chars so the fix isn't truncated away "
             f"(default {_MAX_TRAJECTORY_CHARS}). Pass 0 to keep trajectories whole (raw).",
    )
    p.add_argument(
        "--obs_cap", type=int, default=_OBS_CAP_CHARS,
        help=f"Truncate each observation (command output) to this many chars (default {_OBS_CAP_CHARS}).",
    )
    args = p.parse_args()

    if args.keep_exit_statuses is None:
        keep_arg: "set[str] | frozenset[str] | None" = _DEFAULT_KEEP_EXIT_STATUSES
    else:
        vals = {s.strip().lower() for s in args.keep_exit_statuses}
        keep_arg = None if (not vals or "all" in vals) else vals
    dd = build_swe_infinite_sft_dataset(
        keep_exit_statuses=keep_arg,
        max_chars=args.max_chars or None,   # 0 → None (keep whole)
        obs_cap=args.obs_cap,
    )
    if dd is None:
        sys.exit(1)

    dd.save_to_disk(args.output_path)
    print(
        f"[swe_infinite_dataset] Saved {len(dd['train'])} trajectories → {args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
