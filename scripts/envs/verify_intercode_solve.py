"""Intercode solve-rate eval harness (FIRST-CUT — VM-debug required).

Measures our trained model's intercode ability so the maturation A/B (testing/INTERCODE_MATURATION.md)
has a metric. Runs the SAME execute_bash -> observe -> submit agent loop the tournament eval uses
(intercode_format is copied from the validator's eval_intercode), executing the model's commands in a
REAL bash subprocess and reading real stdout, then scores.

⚠️ FIDELITY CAVEATS (read before trusting a number):
  * The true IC-Bash eval (princeton-nlp/intercode) mounts a per-task FILESYSTEM SNAPSHOT + gold answer
    and scores reward = 0.34*TF-IDF-cos(submitted_output, gold_output) + 0.33*fs-match + 0.33*md5. We do
    NOT have that sandbox, so we approximate: run the GOLD command in the same fresh temp dir to get a
    gold output, and compare the model's SUBMITTED observation to it (token-F1 + exact).
  * Tasks that depend on pre-existing files (e.g. /testbed/foo, f1/f2) will make BOTH the model AND the
    gold command error (missing file) -> UNSCOREABLE; we exclude those and report coverage. Self-contained
    NL2Bash tasks (compute/format/pipe) score faithfully.
  * Commands run per-turn in a fresh `bash -lc` (NOT a persistent shell) — fine for the ~1-command tasks
    the submit-teaching targets, a fidelity gap for multi-cd tasks (noted).
  So this is a DIRECTIONAL A/B proxy (does oversampling raise solve on the scoreable subset + command-match
  rate?), NOT the exact tournament score. Two metrics are reported: OUTPUT-match (faithful, partial
  coverage) and COMMAND-match (robust, full coverage). Use the DELTA between configs, not the absolute.

Run on the VM (GPU + the intercode dataset mounted at MINER_DATASETS_DIR), from scripts/:
    python envs/verify_intercode_solve.py --our-model <path> --n 200 --max-turns 4
Add --peek 3 to print the first few agent transcripts for sanity before trusting the rate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Make envs.* importable when run as a file (same bootstrap as the other verify scripts).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

from envs.intercode_format import (
    INTERCODE_EXECUTE_TOOL_NAME,
    INTERCODE_SUBMIT_TOOL_NAME,
    INTERCODE_TOOL_SYSTEM_PROMPT,
    build_intercode_action_tools,
    build_user_prompt,
    format_history_step,
)
from envs.intercode_dataset import _find_intercode_dataset, _is_bash_response
from envs.miner_dataset_loader import _load_one_dataset
from envs.pvp_format import sglang_template_tools, tools_to_openai

_BASE = "unsloth/Llama-3.2-3B-Instruct"
_TOOLS_OPENAI = tools_to_openai(build_intercode_action_tools())
# The eval server (SGLang v0.5.14) renders tools as Tool.model_dump(); EVAL_SGLANG_TOOLS=0 = our plain render.
if os.getenv("EVAL_SGLANG_TOOLS", "1").strip() != "0":
    _TOOLS_OPENAI = sglang_template_tools(_TOOLS_OPENAI)
_OBS_TRUNCATE = 350  # matches intercode_format.DEFAULT_OBS_TRUNCATE_CHARS


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def _is_lora_repo(repo: str) -> bool:
    if os.path.isdir(repo):
        return os.path.exists(os.path.join(repo, "adapter_config.json"))
    try:
        from huggingface_hub import HfApi
        return any(s.rfilename == "adapter_config.json" for s in HfApi().model_info(repo).siblings)
    except Exception:
        return False


def _load(repo: str, base: str):
    is_lora = _is_lora_repo(repo)
    tok = AutoTokenizer.from_pretrained(base if is_lora else repo)
    if is_lora:
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, repo).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok, ("LoRA" if is_lora else "full")


# A token repeated 4+ times ("- - - -", "x x x x"), or one char run 10+ long ("xxxxxxxxxx") — the
# signature of greedy-decode degeneration that fills max_new_tokens after the real command.
_DEGEN_RE = re.compile(r'(\S+)(?:\s+\1){3,}|(.)\2{9,}')


def _clean_command(cmd: str) -> str:
    """Trim degenerate trailing repetition so the real head command still runs (an undertrained model
    loops on a token after emitting the command). Cuts at the first long repeated run."""
    m = _DEGEN_RE.search(cmd)
    if m:
        cmd = cmd[:m.start()]
    return cmd.strip()


def _extract_command(text: str) -> "str | None":
    """Pull the execute_bash command out of the model's tool-call output. Handles a properly-terminated
    JSON string first, then a TRUNCATED one (greedy loops fill the token budget so the closing quote
    never arrives) — take what's there and trim the degenerate tail."""
    m = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', text)  # terminated JSON string
    if m:
        try:
            cmd = json.loads('"' + m.group(1) + '"')
        except Exception:
            cmd = m.group(1)
        return _clean_command(cmd) or None
    m = re.search(r'"command"\s*:\s*"(.+)', text, re.DOTALL)  # unterminated (truncated)
    if m:
        raw = m.group(1).replace('\\"', '"').replace("\\\\", "\\")
        return _clean_command(raw) or None
    return None


def _model_tool_call(model, tok, query: str, history: "list[str]", turn: int, gen_kwargs: dict):
    """One agent turn: build the eval-identical [system, user] prompt, generate, parse the tool call.
    Returns (tool_name, command_or_None, raw_text)."""
    messages = [
        {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, history, turn)},
    ]
    inputs = tok.apply_chat_template(messages, tools=_TOOLS_OPENAI, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, pad_token_id=tok.eos_token_id, **gen_kwargs)
    text = tok.decode(out[0][plen:], skip_special_tokens=True)

    cmd = _extract_command(text)
    if cmd:
        return INTERCODE_EXECUTE_TOOL_NAME, cmd, text
    # no command -> submit if the model asked for it, else unparseable
    if INTERCODE_SUBMIT_TOOL_NAME in text:
        return INTERCODE_SUBMIT_TOOL_NAME, None, text
    return None, None, text


# --------------------------------------------------------------------------------------------
# Bash execution + scoring
# --------------------------------------------------------------------------------------------
def _run_bash(cmd: str, workdir: str) -> str:
    """Run one command in a fresh bash (NOT persistent — see module caveat). Returns stdout+stderr,
    truncated like the eval observation."""
    try:
        r = subprocess.run(["bash", "-lc", cmd], cwd=workdir, capture_output=True, text=True,
                           timeout=15, stdin=subprocess.DEVNULL)
        obs = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        obs = "<timeout>"
    except Exception as e:
        obs = f"<exec error: {e}>"
    return obs.strip()[:_OBS_TRUNCATE]


def _looks_failed(obs: str) -> bool:
    o = obs.lower()
    return (not obs) or ("command not found" in o) or ("no such file" in o) or obs.startswith("<")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _token_f1(a: str, b: str) -> float:
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def _score_task(model, tok, query: str, gold_cmd: str, max_turns: int, peek: bool, gen_kwargs: dict):
    """Run the agent loop; return dict(scoreable, output_f1, output_exact, command_match, model_cmd, obs)."""
    with tempfile.TemporaryDirectory() as wd, tempfile.TemporaryDirectory() as gd:
        gold_obs = _run_bash(gold_cmd, gd)  # gold output in a fresh, identical-empty dir
        history: list[str] = []
        model_first_cmd = None
        last_obs = ""
        trace: list[dict] = []  # per-turn actions, for --dump failure analysis
        end = "max_turns"
        for turn in range(1, max_turns + 1):
            name, cmd, raw = _model_tool_call(model, tok, query, history, turn, gen_kwargs)
            if peek:
                print(f"  [peek turn {turn}] name={name} cmd={cmd!r} raw={raw[:120]!r}", flush=True)
            if name == INTERCODE_SUBMIT_TOOL_NAME:
                trace.append({"turn": turn, "action": "submit"})
                end = "submit"
                break
            if name == INTERCODE_EXECUTE_TOOL_NAME and cmd:
                if model_first_cmd is None:
                    model_first_cmd = cmd
                obs = _run_bash(cmd, wd)
                last_obs = obs
                history.append(format_history_step(turn, cmd, obs))
                trace.append({"turn": turn, "action": "execute", "cmd": cmd, "obs": obs[:300]})
            else:
                trace.append({"turn": turn, "action": "unparseable", "raw": (raw or "")[:300]})
                end = "unparseable"
                break  # unparseable -> stop this episode

        # command-match (robust, full coverage): normalized equality of the model's first command to gold
        command_match = 1.0 if model_first_cmd and _norm(model_first_cmd) == _norm(gold_cmd) else 0.0
        # output-match (faithful, partial coverage): only when the GOLD command itself produced real output
        scoreable = not _looks_failed(gold_obs)
        out_f1 = _token_f1(last_obs, gold_obs) if scoreable else None
        out_exact = 1.0 if (scoreable and _norm(last_obs) == _norm(gold_obs)) else (0.0 if scoreable else None)
        return {
            "scoreable": scoreable, "output_f1": out_f1, "output_exact": out_exact,
            "command_match": command_match, "model_cmd": model_first_cmd, "gold_obs": gold_obs,
            "trace": trace, "end": end,
        }


# --------------------------------------------------------------------------------------------
def _load_tasks(n: int, seed: int):
    """Held-out (instruction, gold_command) pairs from the mounted intercode dataset (bash rows only)."""
    found = _find_intercode_dataset()
    if found is None:
        print("[verify_intercode_solve] intercode dataset not in MINER_DATASETS — mount it first.", flush=True)
        sys.exit(2)
    _, local = found
    raw = _load_one_dataset(local)
    if raw is None:
        print("[verify_intercode_solve] could not load the dataset", flush=True)
        sys.exit(2)
    pairs = []
    for row in raw:
        d = dict(row)
        instr = (d.get("instruction") or "").strip()
        resp = (d.get("response") or "").strip()
        if instr and resp and _is_bash_response(resp):
            pairs.append((instr, resp))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Intercode solve-rate proxy eval (see module caveats).")
    p.add_argument("--our-model", required=True, help="path/repo of the trained model (or base for a baseline)")
    p.add_argument("--base-model", default=_BASE)
    p.add_argument("--n", type=int, default=200, help="held-out tasks to eval")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--peek", type=int, default=0, help="print the first N agent transcripts")
    p.add_argument("--rep-penalty", type=float, default=1.15,
                   help="repetition_penalty: 1.0 = pure greedy (an undertrained model loops into ' - - -' "
                        "garbage); 1.15 (default) breaks the loop to reveal the model's real command knowledge. "
                        "Sweep both to separate 'undertrained model' from 'greedy-decoding artifact'.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--dump", default="", help="write one JSON line per task (query, gold, trace, scores) here")
    args = p.parse_args()

    print(f"Loading model: {args.our_model}", flush=True)
    model, tok, kind = _load(args.our_model, args.base_model)
    print(f"  -> {kind}", flush=True)
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    if args.rep_penalty and args.rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.rep_penalty
    print(f"  gen: {gen_kwargs}", flush=True)

    tasks = _load_tasks(args.n, args.seed)
    print(f"Loaded {len(tasks)} held-out intercode tasks", flush=True)

    n_score = 0
    sum_f1 = 0.0
    sum_exact = 0.0
    sum_cmd = 0.0
    dump = open(args.dump, "w") if args.dump else None
    for i, (query, gold) in enumerate(tasks):
        r = _score_task(model, tok, query, gold, args.max_turns, peek=(i < args.peek), gen_kwargs=gen_kwargs)
        if dump:
            dump.write(json.dumps({"i": i, "query": query, "gold": gold, **r}) + "\n")
        sum_cmd += r["command_match"]
        if r["scoreable"]:
            n_score += 1
            sum_f1 += r["output_f1"]
            sum_exact += r["output_exact"]
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(tasks)}  cmd-match={sum_cmd/(i+1):.3f}  "
                  f"out-f1={sum_f1/max(1,n_score):.3f} (n_score={n_score})", flush=True)

    if dump:
        dump.close()
    total = len(tasks)
    print("\n=== INTERCODE SOLVE (proxy) ===")
    print(f"tasks={total}  scoreable(output)={n_score} ({100*n_score/max(1,total):.0f}%)")
    print(f"COMMAND-match rate  : {sum_cmd/max(1,total):.3f}   (robust, full coverage — did the model emit the gold command)")
    print(f"OUTPUT-F1 (scoreable): {sum_f1/max(1,n_score):.3f}   (faithful, partial coverage — submitted obs vs gold obs)")
    print(f"OUTPUT-exact         : {sum_exact/max(1,n_score):.3f}")
    print("\nA/B usage: compare these three across weight configs (baseline vs intercode:1.3 vs :2.5). A real")
    print("maturation shows COMMAND-match and OUTPUT-F1 RISE together; a rise in command-match with flat")
    print("output-F1 = memorizing gold strings without solving. Read the DELTA, not the absolute (proxy).")


if __name__ == "__main__":
    main()
