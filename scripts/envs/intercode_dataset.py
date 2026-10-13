"""Prepare InterCode SFT training dataset from the validator-mounted miner dataset.

Reads ``gradients-io-tournaments/intercode_bigcode_combined_12k`` from the
validator-mounted miner datasets directory and converts each example to a
tool-calling training row that exactly matches the eval format.

Each output row:
    {"messages": [
        {"role": "system",    "content": INTERCODE_TOOL_SYSTEM_PROMPT},
        {"role": "user",      "content": "<build_user_prompt(question, [], 1)>"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function",
                          "function": {"name": "execute_bash", "arguments": "<json: {\"command\": \"<gold_command>\"}>"}}]},
     ],
     "tools": [...build_intercode_action_tools(), as OpenAI dicts...]}

Each bash dataset row yields a turn-1 execute example; a deterministic fraction of
rows (INTERCODE_SUBMIT_FRAC, default 0.35) also yield a turn-2 submit example so the
model learns to TERMINATE the episode — the eval scores the submitted observation,
so an execute-only model that never calls submit scores 0. Both are clean stateless
single-assistant-turn rows (the eval reconstructs [system, user] fresh each turn with
the step history embedded as text), so there is no multi-turn tool-message rendering.
See the submit-teaching note below. Disable with INTERCODE_TEACH_SUBMIT=0.

The system/user messages match what eval_intercode._build_tool_messages() sends to
the model for turn 1 (and turn 2) of an episode, so the model's training distribution
aligns with its inference distribution.

Dataset path convention (follows miner_dataset_loader.py):
    MINER_DATASETS_DIR/<hf_org>--<hf_repo>/   (-- replaces / in the HF repo name)
    MINER_DATASETS is a comma-separated list of those directory names.

Usage (run from /workspace/scripts/):
    python -m envs.intercode_dataset --output_path /path/to/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from budget_controller import intercode_row_budget
from envs.intercode_format import INTERCODE_EXECUTE_TOOL_NAME
from envs.intercode_format import INTERCODE_SUBMIT_TOOL_NAME
from envs.intercode_format import INTERCODE_TOOL_SYSTEM_PROMPT
from envs.intercode_format import build_intercode_action_tools
from envs.intercode_format import build_user_prompt
from envs.intercode_format import format_history_step
from envs.miner_dataset_loader import _get_miner_datasets_inventory, _load_one_dataset
from envs.pvp_format import tools_to_openai
from envs.shared_env import recipe_log

INTERCODE_HF_REPO = "gradients-io-tournaments/intercode_bigcode_combined_12k"


# ---------------------------------------------------------------------------
# Submit-step teaching (winner-teardown port, 2026-08-30)
# ---------------------------------------------------------------------------
# The eval is a multi-turn execute -> observe -> SUBMIT loop (see
# intercode_format.INTERCODE_TOOL_SYSTEM_PROMPT: every few-shot ends in submit()).
# The score is read off the observation the model SUBMITS; a run that never calls
# submit never terminates cleanly and scores 0 ("No patch"). Our warm-start data was
# execute-only, so SFT reinforces "always execute, never submit" and can suppress the
# termination the base model would otherwise do from the system prompt. The winner's
# fix is to also teach the submit step.
#
# We exploit the eval's STATELESS per-turn format (intercode_format.build_user_prompt
# embeds the step history AS TEXT in the user message, exactly like the game envs'
# per-turn reconstruction — there are no accumulating tool-role messages). So a submit
# example is a clean, independent single-assistant-turn row:
#     [system, user(query, history=[turn-1 step], turn=2), assistant(submit())]
# No multi-turn tool-message rendering (no chat-template risk), and the turn-1 command
# + its observation live in the user text, which tokenize_and_mask MASKS from the loss
# (assistant-only) — so we never train the model to emit a fabricated observation.
#
# Turn 1 examples (empty history) only ever emit execute; turn 2 examples (history
# present) emit submit. So the model learns "never submit before executing" for free.
#
# Two adversarial-review findings shape HOW the turn-1 observation is faked:
#  1. The observation must NOT be a single constant string. Loss is on the submit
#     tokens conditioned on the whole (masked) context, so a constant, distinctive
#     marker ("<command ran successfully>") that NEVER appears at eval becomes a
#     spurious trigger: the model keys on that exact string, and at eval — where the
#     real stdout is there instead — the trigger never fires and it never submits
#     (score 0, the very failure this fixes). So we draw the turn-1 observation from a
#     VARIED pool of realistic, success-shaped stdout snippets (hash-keyed), pushing the
#     observation marginal toward "arbitrary real stdout" and forcing the model to key
#     on the turn-2 STRUCTURE — the true eval-time invariant.
#  2. If 100% of turn-2 signal were "submit", the model would learn "history present ⇒
#     submit" unconditionally and could submit a FAILED observation on a multi-command
#     task. So a small share of queries instead get a contrastive turn-2 example whose
#     turn-1 observation is a TRANSIENT error and whose action is EXECUTE (retry) — never
#     submit. This teaches "error observation ⇒ keep executing, don't submit".
#
# Selections are deterministic hash slices of the query (reproducible, non-overlapping):
# submit = bottom INTERCODE_SUBMIT_FRAC of the 0..999 bucket, retry = top
# INTERCODE_RETRY_FRAC. A query gets at most one extra (submit OR retry) example.
_TEACH_SUBMIT = os.getenv("INTERCODE_TEACH_SUBMIT", "1").strip() != "0"


def _frac_env(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except ValueError:
        v = default
    return min(1.0, max(0.0, v))


_SUBMIT_FRAC_DEFAULT = 0.35
_SUBMIT_FRAC = _frac_env("INTERCODE_SUBMIT_FRAC", _SUBMIT_FRAC_DEFAULT)
_RETRY_FRAC = _frac_env("INTERCODE_RETRY_FRAC", 0.12)

# Varied, SUCCESS-shaped stdout snippets for a completed turn-1 step (masked context,
# never a label — see note above). Realistic single-command outputs so the turn-2
# context resembles a real post-success state without a memorizable constant.
_SUCCESS_OBS: tuple[str, ...] = (
    "42",
    "128\n",
    "report.txt\nnotes.md\ndata.csv",
    "24K\t/workspace",
    "  17  152  1043 main.py",
    "546578742066696c6520372077697468",
    "-rw-r--r-- 1 root root 1024 Jan  1 12:00 file.txt",
    "src/app.py:42:def main():",
    "2 total",
    "/usr/bin/python3",
    "3",
    "total 8\ndrwxr-xr-x 2 root root 4096 dir",
)

# TRANSIENT (retryable) error shapes for the contrastive "error ⇒ retry, don't submit"
# example. Only transient errors, so retrying the same command is a logical action.
_TRANSIENT_ERR_OBS: tuple[str, ...] = (
    "Resource temporarily unavailable",
    "Connection timed out",
    "error: could not lock config file: File exists",
    "Temporary failure in name resolution",
    "database is locked",
)


# ---------------------------------------------------------------------------
# Dataset lookup
# ---------------------------------------------------------------------------

def _find_intercode_dataset() -> tuple[str, Path] | None:
    """Return (hf_name, local_path) for the intercode dataset, or None."""
    for hf_name, local in _get_miner_datasets_inventory():
        if "intercode" in hf_name.lower():
            return hf_name, local
    return None


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------


# JSON-encoded: a `datasets.Dataset` unifies struct schemas across a list
# column's elements, which would corrupt the heterogeneous `parameters.properties`
# of execute_bash (has "command") vs submit (empty) — e.g. submit's `{}` becomes
# `{"command": None}`. Storing as a string and decoding in tokenize_and_mask
# avoids that. See train_sft_env.py::tokenize_and_mask.
_INTERCODE_TOOLS_JSON = json.dumps(tools_to_openai(build_intercode_action_tools()))


def _hash_bucket(query: str) -> int:
    """Stable 0..999 bucket for a query (builtin hash is per-process salted → md5)."""
    return int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16) % 1000


def _should_teach_submit(query: str, frac: "float | None" = None) -> bool:
    """True for the bottom ``frac`` (default ``_SUBMIT_FRAC``) slice of the bucket → a submit example."""
    frac = _SUBMIT_FRAC if frac is None else frac
    if not _TEACH_SUBMIT or frac <= 0.0:
        return False
    return _hash_bucket(query) < int(round(frac * 1000))


def _should_teach_retry(query: str) -> bool:
    """True for the TOP ``_RETRY_FRAC`` slice → a contrastive error⇒retry example.

    Disjoint from the submit slice (bottom vs top of the 0..999 bucket) so a query
    gets at most one extra example; if the two fractions together would exceed the
    bucket the submit slice wins (checked first in _gold_to_examples)."""
    if not _TEACH_SUBMIT or _RETRY_FRAC <= 0.0:
        return False
    return _hash_bucket(query) >= 1000 - int(round(_RETRY_FRAC * 1000))


# Passthrough hygiene (INTERCODE_PASSTHROUGH_FILTER, default ON). The eval runs each command with
# /bin/bash -c as the evaluator's own user, in the evaluator's container, 30 s per action, 10 turns,
# and a task that runs out of time scores 0.0. Some whitelisted NL2Bash golds are commands a model
# must never learn to emit there: ones that never return (tail -f, watch, top without -n, bare yes,
# ping without -c, long sleeps), process killers (kill -9 -1, pkill, killall — they can take down the
# serving stack, i.e. the whole env eval), disk fillers, system-destructive commands and placeholder
# paths (/path/to, <file>) that only produce errors. They are dropped as execute TARGETS (general
# shell-safety classes, not fitted to any eval statistic); hang golds are reused as history in a
# "Command timed out -> submit" row so a hang costs one 30 s turn instead of the whole task.
_PASSTHROUGH_FILTER = os.getenv("INTERCODE_PASSTHROUGH_FILTER", "1").strip() != "0"
_TIMEOUT_OBS = "Command timed out"  # the evaluator's literal observation on a 30 s action timeout
_HYGIENE_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (cls, re.compile(pat)) for cls, pat in (
        ("killer", r"\bkill\s+(-9\s+|-KILL\s+|-s\s+\w+\s+)?-1\b|\bkillall5?\b|\bpkill\b|\bkill\b[^;&|]*\$\((ps|pgrep|pidof)\b"),
        ("destructive", r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\*|\s|$)|\bmkfs\b|\b(shutdown|reboot|halt|poweroff)\b"
                        r"|\bchmod\s+-R\s+\S+\s+/(\s|$)|:\(\)\s*\{"),
        ("diskfill", r"\byes\b[^|;&]*>|\bfallocate\b|\btruncate\s+-s\s*\+?\d+[GT]"),
        ("hang", r"\btail\s+(-[a-zA-Z]*[fF]\b|--follow)|\bwatch\s|\bnc\b[^|;&]*\s-[a-zA-Z]*l|/dev/u?random"
                 r"|\b(vi|vim|nano|emacs|less|more)\b"),
        ("network", r"\b(ssh|scp|sftp|wget|curl|ftp|telnet|mailx?|sendmail)\b|\brsync\b[^|;&]*\s\S+:\S*"
                    r"|\bgit\s+(clone|pull|push|fetch)\b|\b(apt|apt-get|yum|dnf|pip3?|npm)\s+install\b"),
        ("placeholder", r"/path/to\b|<[A-Za-z_][A-Za-z_ ]*>|\byour_\w+|\bdbname\b"),
    )
)


def _passthrough_class(gold: str) -> "str | None":
    """The hygiene class of a whitelisted gold command, or None if it is fine to train on."""
    for cls, pat in _HYGIENE_RULES:
        if pat.search(gold):
            return cls
    if re.search(r"\btop\b", gold) and not re.search(r"\btop\b[^|;&]*\s-[a-zA-Z]*n", gold):
        return "hang"      # top only returns in batch mode with an iteration count
    if re.search(r"\byes\b", gold) and not re.search(r"\byes\b[^;&]*\|\s*(head|sed\s+-n\s+\S*q)", gold):
        return "hang"      # yes never ends unless something downstream stops reading
    if re.search(r"\bping\b", gold) and not re.search(r"\bping\b[^|;&]*\s-[a-zA-Z]*c", gold):
        return "hang"
    m = re.search(r"\bsleep\s+(\d+)", gold)
    if m and int(m.group(1)) >= 10:
        return "hang"
    if re.search(r"\bdd\b", gold) and "count=" not in gold:
        return "hang"
    return None


def _timeout_submit_example(query: str, hang_cmd: str) -> dict[str, Any]:
    """After an action that timed out (the evaluator's literal observation), SUBMIT instead of
    re-running it: every re-run burns another 30 s turn and a task out of time scores 0.0."""
    return _submit_after(query, [(hang_cmd, _TIMEOUT_OBS)])


def _submit_after(query: str, steps: "list[tuple[str, str]]") -> dict[str, Any]:
    """SUBMIT at turn len(steps)+1 after the given (command, observation) steps in the history."""
    history = [format_history_step(i + 1, cmd, obs) for i, (cmd, obs) in enumerate(steps)]
    return {
        "messages": [
            {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, history, len(steps) + 1)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": INTERCODE_SUBMIT_TOOL_NAME, "arguments": json.dumps({})}},
            ]},
        ],
        "tools": _INTERCODE_TOOLS_JSON,
    }


def _pick(pool: "tuple[str, ...]", query: str) -> str:
    """Deterministically pick one pool entry for this query."""
    return pool[_hash_bucket(query) % len(pool)]


def _execute_example(query: str, gold: str, history: "list[str] | None" = None, turn: int = 1) -> dict[str, Any]:
    """An execute_bash example. Turn 1 (empty history) = the primary command example;
    turn 2 (history present) = the contrastive "keep executing after an error" example."""
    return {
        "messages": [
            {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, history or [], turn)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": INTERCODE_EXECUTE_TOOL_NAME, "arguments": json.dumps({"command": gold})}},
            ]},
        ],
        "tools": _INTERCODE_TOOLS_JSON,
    }


def _submit_example(query: str, gold: str) -> dict[str, Any]:
    """Turn-2 example: after a SUCCESSFUL turn-1 gold step (in the history text), SUBMIT.

    A clean single-assistant-turn row matching the eval's stateless per-turn format.
    The turn-1 command + a varied, success-shaped observation are carried as text in the
    (masked) user message, so termination is taught without training on any fabricated
    observation — and the varied observation avoids a memorizable constant trigger."""
    obs = _pick(_SUCCESS_OBS, query)
    history = [format_history_step(1, gold, obs)]
    return {
        "messages": [
            {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, history, 2)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": INTERCODE_SUBMIT_TOOL_NAME, "arguments": json.dumps({})}},
            ]},
        ],
        "tools": _INTERCODE_TOOLS_JSON,
    }


def _retry_example(query: str, gold: str) -> dict[str, Any]:
    """Turn-2 contrastive example: after a TRANSIENT-error turn-1 observation, EXECUTE
    (retry) rather than submit — so "submit" never becomes an unconditional turn-2
    reflex that would submit a failed observation."""
    obs = _pick(_TRANSIENT_ERR_OBS, query)
    history = [format_history_step(1, gold, obs)]
    return _execute_example(query, gold, history=history, turn=2)


def _gold_to_examples(query: str, gold: str, submit_frac: "float | None" = None) -> list[dict[str, Any]]:
    """Build the training example(s) for one (question, gold-command) pair.

    Always the execute (turn-1) example; then, for a deterministic hash slice of queries,
    EITHER a submit (turn-2) example — so the model learns to terminate the episode (the
    eval scores the SUBMITTED observation; an execute-only model that never submits scores
    0) — OR a contrastive error⇒retry (turn-2 execute) example. See the module-level note.
    INTERCODE_TEACH_SUBMIT=0 disables both; INTERCODE_SUBMIT_FRAC / INTERCODE_RETRY_FRAC
    retune the shares."""
    examples = [_execute_example(query, gold)]
    if _should_teach_submit(query, submit_frac):
        examples.append(_submit_example(query, gold))
    elif _should_teach_retry(query):
        examples.append(_retry_example(query, gold))
    return examples


# ---------------------------------------------------------------------------
# COMPLIANT self-gen (INTERCODE_SELFGEN, default OFF) — real command stdout
# ---------------------------------------------------------------------------
# When enabled AND inside the container sandbox (envs.intercode_selfgen enforces the
# host guard), we build extra rows whose observations are REAL stdout from confined,
# tempdir-only general-bash tasks — replacing the FAKE _SUCCESS_OBS pool so the student
# can contest the eval's tfidf(last_obs, gold) reward term (p3), the one we underscored
# on (0.783 vs the winner's 0.871). Every row reuses the SAME eval-exact tool-call shape
# as the passthrough (content=None + structured execute/submit), so there is no new
# dual-model / parser risk. Default OFF => selfgen_enabled() is False => nothing added =>
# byte-identical passthrough. See envs/intercode_selfgen.py for the fd4d6f6 compliance
# story (no eval-fs, no manifest reads, no frequency-fitting, /testbed+/workspace only).

def _selfgen_n() -> int:
    """How many self-gen tasks to attempt (INTERCODE_SELFGEN_N, default 6000)."""
    try:
        return max(0, int(os.getenv("INTERCODE_SELFGEN_N", "6000")))
    except ValueError:
        return 6000


def _submit_example_real(query: str, gold: str, obs: str) -> dict[str, Any]:
    """Turn-2 SUBMIT after a REAL successful gold step — the real stdout sits in the
    (masked) history, so termination is taught on genuine observation, not a fake pool."""
    history = [format_history_step(1, gold, obs)]
    return {
        "messages": [
            {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, history, 2)},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": INTERCODE_SUBMIT_TOOL_NAME, "arguments": json.dumps({})}},
            ]},
        ],
        "tools": _INTERCODE_TOOLS_JSON,
    }


def _refine_example(query: str, verbose_cmd: str, verbose_obs: str, gold: str) -> dict[str, Any]:
    """REFINE: after a broad verbose step (real obs in the masked history), narrow to the
    minimal gold command — the run-broad-then-narrow behavior the eval few-shots demonstrate."""
    history = [format_history_step(1, verbose_cmd, verbose_obs)]
    return _execute_example(query, gold, history=history, turn=2)


def _recovery_example(query: str, near_miss_cmd: str, near_miss_obs: str, gold: str) -> dict[str, Any]:
    """RECOVERY: after a REAL near-miss error (its real stderr/obs in the masked history),
    run the correct gold — teaches fail -> correct, which the single-shot passthrough never does."""
    history = [format_history_step(1, near_miss_cmd, near_miss_obs)]
    return _execute_example(query, gold, history=history, turn=2)


def _selfgen_to_examples() -> list[dict[str, Any]]:
    """Build SFT examples from REAL confined self-gen tasks. Returns [] when INTERCODE_SELFGEN
    is off OR the host guard refuses (not in a container) OR the module is unavailable — so the
    default path stays the byte-identical whitelisted passthrough."""
    # Cheap env gate FIRST, so the default path (flag unset) never even imports the module.
    if os.getenv("INTERCODE_SELFGEN", "").strip().lower() not in ("1", "true", "yes", "on"):
        return []
    try:
        from envs.intercode_selfgen import generate_selfgen_tasks
    except Exception as exc:  # module missing/broken must never abort the passthrough build
        print(f"[intercode_dataset] self-gen unavailable ({exc!r}); passthrough only.", flush=True)
        return []
    try:
        seed = int(os.getenv("INTERCODE_SELFGEN_SEED", "0"))
    except ValueError:
        seed = 0
    tasks = generate_selfgen_tasks(_selfgen_n(), seed=seed)  # [] if the host guard fails
    examples: list[dict[str, Any]] = []
    chains = os.getenv("INTERCODE_CHAINS", "1").strip() != "0"
    n_chain = n_deloop = 0
    for t in tasks:
        examples.append(_execute_example(t.query, t.gold))                     # single-shot execute (eval-exact)
        examples.append(_submit_example_real(t.query, t.gold, t.gold_obs))     # REAL-obs submit (the p3 lever)
        if t.verbose_cmd and t.verbose_obs:
            examples.append(_refine_example(t.query, t.verbose_cmd, t.verbose_obs, t.gold))
            if chains:  # ... and the chain ends: after the minimal answer, submit (turn 3)
                examples.append(_submit_after(t.query, [(t.verbose_cmd, t.verbose_obs), (t.gold, t.gold_obs)]))
                n_chain += 1
        if t.near_miss_cmd and t.near_miss_obs:
            examples.append(_recovery_example(t.query, t.near_miss_cmd, t.near_miss_obs, t.gold))
            if chains:
                examples.append(_submit_after(t.query, [(t.near_miss_cmd, t.near_miss_obs), (t.gold, t.gold_obs)]))
                n_chain += 1
                # De-loop: the eval shell is deterministic, so a command that failed twice with the same
                # output fails forever; stop and submit instead of burning the remaining turns.
                if _hash_bucket(t.query) % 4 == 0:
                    examples.append(_submit_after(t.query, [(t.near_miss_cmd, t.near_miss_obs)] * 2))
                    n_deloop += 1
    recipe_log(
        f"[intercode_dataset] self-gen produced {len(examples)} examples from {len(tasks)} tasks "
        f"(+{n_chain} turn-3 submits, +{n_deloop} repeat->submit).",
        tag="[intercode] self-gen built",
    )
    return examples



_PYTHON_PROSE_MARKERS = (
    "```",       # markdown code block (Python explanations use ```python ... ```)
    "def ",      # Python function definition
    "import ",   # Python import statement
    "class ",    # Python class definition
)


def _is_bash_response(response: str) -> bool:
    """Return True if response looks like a bash command rather than Python/prose.

    The dataset is 50% Python coding problems (BigCode) and 50% bash commands
    (Tellina/NL2Bash). Python responses contain markdown code fences, def/import/class
    keywords, and verbose prose explanation. Bash commands are compact and lack these.
    """
    if any(marker in response for marker in _PYTHON_PROSE_MARKERS):
        return False
    # Prose explanations are long multi-paragraph text; bash commands are compact
    if len(response) > 400:
        return False
    return True


def _row_to_tool_examples(row: dict[str, Any], submit_frac: "float | None" = None) -> list[dict[str, Any]]:
    """Convert one intercode_bigcode_combined_12k row to a tool-calling training example.

    The dataset is ~50% Python coding problems (response = prose + code) and ~50%
    bash commands (response = shell command). Only bash rows are kept; Python rows
    are silently skipped because wrapping prose in an execute_bash call would be wrong.
    """
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        return []
    if not _is_bash_response(response):
        return []
    cls = _passthrough_class(response) if _PASSTHROUGH_FILTER else None
    if cls == "hang":
        return [_timeout_submit_example(instruction, response)]
    if cls is not None:
        return []
    return _gold_to_examples(instruction, response, submit_frac)


_SELFGEN_MIN_SHARE = 0.5


def _fit_row_budget(passthrough: list, selfgen: list, budget: int) -> "tuple[list, list]":
    """Trim to an INTERCODE_ROW_BUDGET: at least half the rows are real-observation self-gen rows (more when
    the passthrough is short), the rest a uniform sample of the passthrough rows."""
    n_self = min(len(selfgen), max(math.ceil(budget * _SELFGEN_MIN_SHARE), budget - len(passthrough)))
    n_pass = min(len(passthrough), budget - n_self)
    rng = random.Random(1234)
    recipe_log(
        f"[intercode_dataset] {budget}-row budget: {n_self} self-gen + {n_pass} passthrough rows",
        tag="[intercode] row budget fit",
    )
    return rng.sample(passthrough, n_pass), rng.sample(selfgen, n_self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_intercode_sft_dataset() -> DatasetDict | None:
    """Convert the whitelisted InterCode miner dataset (always) and, when INTERCODE_SELFGEN
    is on, ADD rows with REAL confined command stdout. Returns the DatasetDict or None.

    Default (INTERCODE_SELFGEN unset): behaves exactly as before — passthrough only, and
    None when the whitelisted dataset is absent."""
    all_examples: list[dict[str, Any]] = []
    # Rounds after R1 set INTERCODE_ROW_BUDGET together with INTERCODE_SUBMIT_FRAC=0 (submits only after real
    # observations). Self-gen is built first so that, if it yields nothing, the passthrough keeps its submit
    # rows: a model that never learns to submit scores 0 on every task.
    row_budget = intercode_row_budget()
    selfgen = _selfgen_to_examples()
    submit_frac = _SUBMIT_FRAC_DEFAULT if (row_budget and not selfgen and _SUBMIT_FRAC <= 0.0) else None

    # 1) Whitelisted bigcode passthrough — the default source, and the ~anchor when self-gen is on.
    found = _find_intercode_dataset()
    if found is None:
        print(
            f"[intercode_dataset] '{INTERCODE_HF_REPO}' not found in miner dataset inventory.\n"
            "  Make sure MINER_DATASETS includes "
            "'gradients-io-tournaments--intercode_bigcode_combined_12k' "
            "and MINER_DATASETS_DIR is set.",
            flush=True,
        )
    else:
        hf_name, local = found
        print(f"[intercode_dataset] Loading {hf_name} from {local}", flush=True)
        raw = _load_one_dataset(local)
        if raw is None:
            print(f"[intercode_dataset] Failed to load dataset from {local}", flush=True)
        else:
            print(f"[intercode_dataset] Loaded {len(raw)} rows", flush=True)
            skipped = 0
            hygiene: dict[str, int] = {}
            for row in raw:
                d = dict(row)
                resp = (d.get("response") or "").strip()
                if _PASSTHROUGH_FILTER and resp and _is_bash_response(resp):
                    cls = _passthrough_class(resp)
                    if cls:
                        hygiene[cls] = hygiene.get(cls, 0) + 1
                examples = _row_to_tool_examples(d, submit_frac)
                if examples:
                    all_examples.extend(examples)
                else:
                    skipped += 1
            recipe_log(
                f"[intercode_dataset] Converted {len(all_examples)} passthrough step-examples "
                f"({skipped}/{len(raw)} rows skipped; hygiene-filtered golds by class: {hygiene or 'none'}, "
                f"hang golds -> 'Command timed out -> submit' rows)",
                tag="[intercode] passthrough converted",
            )

    # 2) COMPLIANT self-gen (INTERCODE_SELFGEN, default OFF) — additive real-stdout rows (built above).
    if row_budget and selfgen and len(all_examples) + len(selfgen) > row_budget:
        all_examples, selfgen = _fit_row_budget(all_examples, selfgen, row_budget)
    # R1: the merge later caps intercode by a UNIFORM subsample; trim the self-gen rows first so the
    # whitelisted passthrough rows (the ones closest to the eval's NL queries) keep their full share.
    try:
        budget = 0 if row_budget else int(os.getenv("R1_MIN_ENV_ROWS", "0") or 0)
    except ValueError:
        budget = 0
    if budget > 0 and selfgen and len(all_examples) + len(selfgen) > budget:
        keep = max(0, budget - len(all_examples))
        selfgen = random.Random(1234).sample(selfgen, keep)
        recipe_log(
            f"[intercode_dataset] self-gen trimmed to {keep} rows to fit the {budget}-row intercode budget "
            f"with all {len(all_examples)} passthrough rows kept",
            tag="[intercode] self-gen trimmed",
        )
    all_examples.extend(selfgen)

    if not all_examples:
        return None

    recipe_log(f"[intercode_dataset] Total {len(all_examples)} step-examples.", tag="[intercode] dataset built")
    return DatasetDict({"train": Dataset.from_list(all_examples)})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert the InterCode whitelisted dataset to SFT training format."
    )
    p.add_argument("--output_path", required=True, help="Where to save the DatasetDict")
    args = p.parse_args()

    dd = build_intercode_sft_dataset()
    if dd is None:
        sys.exit(1)

    dd.save_to_disk(args.output_path)
    recipe_log(
        f"[intercode_dataset] Saved {len(dd['train'])} examples → {args.output_path}",
        tag="[intercode] dataset saved",
    )


if __name__ == "__main__":
    main()
