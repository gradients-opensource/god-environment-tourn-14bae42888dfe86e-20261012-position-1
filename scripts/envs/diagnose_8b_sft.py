#!/usr/bin/env python
"""Diagnose whether our SFT tool_calls rows RENDER + SUPERVISE on a given base model.

WHY THIS EXISTS
    We scored an exact 0.0 (a forfeit signature — every rival >=0.667) in the
    2026-09-07 env tournament on unsloth/Meta-Llama-3.1-8B-Instruct, having
    validated the recipe only on Llama-3.2-3B. Our SFT already emits the
    champion's structured tool_calls format (verified), so the 0.0 is most likely
    either:
      (A) the 8B chat template REJECTS our rows -> tokenize_and_mask's try/except
          drops them -> few/zero rows survive -> the UNTRAINED base is shipped, or
      (B) 3B->8B non-transfer / undertraining.
    This script tests (A) directly and needs NO GPU: it builds rows in our EXACT
    emitted shape and renders them through the REAL base tokenizer, then reports
    keep-rate + assistant-token coverage. A clean render here rules (A) out and
    points at (B); a throw/zero-coverage here is the smoking gun.

RUN (inside the standalone-text-trainer image; only the tokenizer is downloaded):
    cd /workspace/scripts && python envs/diagnose_8b_sft.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct
    # compare against the base we validated on:
    cd /workspace/scripts && python envs/diagnose_8b_sft.py \
        --model unsloth/Llama-3.2-3B-Instruct
    # optional: run the REAL tokenize_and_mask over a generated dataset dir
    cd /workspace/scripts && python envs/diagnose_8b_sft.py \
        --model unsloth/Meta-Llama-3.1-8B-Instruct --dataset-path /path/to/generated_ds

Recipe/approach only — never copies competitor code (see [[feedback-port-ideas-not-verbatim]]).
"""
import argparse
import json
import os
import sys

# Make `envs.*` importable no matter how this is launched (`python envs/diagnose_8b_sft.py`
# puts sys.path[0]=envs/, not scripts/). Mirror verify_intercode_solve.py's convention.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _decode_row_for_template(row):
    """Mirror train_sft_env.tokenize_and_mask: tools + each tool_call's arguments are
    stored as JSON STRINGS on disk; decode them back to dicts before templating so the
    template's `tojson` filter renders once (not double-encoded)."""
    tools = row.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    msgs = []
    for m in row["messages"]:
        if m.get("tool_calls"):
            m = {
                **m,
                "tool_calls": [
                    {
                        **c,
                        "function": {
                            **c["function"],
                            "arguments": (
                                json.loads(c["function"]["arguments"])
                                if isinstance(c["function"].get("arguments"), str)
                                else c["function"].get("arguments")
                            ),
                        },
                    }
                    for c in m["tool_calls"]
                ],
            }
        msgs.append(m)
    return msgs, tools


def _build_sample_rows():
    """Authentic rows in our EXACT emitted shape, arguments+tools as JSON strings
    (matching the on-disk row). The game row uses the real pvp_format builders; the
    intercode-style row is a self-contained execute_bash/submit pair (string-arg +
    no-arg tools) so the check does not depend on intercode_format's private API."""
    rows = []
    # --- game_action (perfect-info othello; 4 memory tools + game_action present) ---
    try:
        from envs.pvp_format import (
            build_full_system_prompt,
            build_user_prompt,
            build_pvp_tools,
            tools_to_openai,
        )
        legal = [(19, "D3"), (26, "C4"), (37, "F5")]
        tools = build_pvp_tools([a for a, _ in legal])
        rows.append({
            "name": "game/othello:game_action",
            "messages": [
                {"role": "system", "content": build_full_system_prompt("othello")},
                {"role": "user", "content": build_user_prompt("(current board omitted)", 0, legal)},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"type": "function", "function": {
                        "name": "game_action", "arguments": json.dumps({"action_id": 19})}}]},
            ],
            "tools": json.dumps(tools_to_openai(tools)),
        })
    except Exception as exc:
        rows.append({"name": "game/othello:game_action", "build_error": repr(exc)})

    # --- intercode execute_bash then submit (string-arg + no-arg tool) ---
    ic_tools = [
        {"type": "function", "function": {
            "name": "execute_bash",
            "description": "Run a bash command in the shell.",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string", "description": "the command to run"}},
                           "required": ["command"]}}},
        {"type": "function", "function": {
            "name": "submit",
            "description": "Submit your final answer.",
            "parameters": {"type": "object", "properties": {}}}},
    ]
    rows.append({
        "name": "intercode:execute_bash",
        "messages": [
            {"role": "system", "content": "You solve shell tasks. Call exactly one tool per turn."},
            {"role": "user", "content": "Question: how many .txt files are under /testbed?\nTurn 1: call exactly one tool now."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {
                    "name": "execute_bash", "arguments": json.dumps({"command": "ls /testbed/*.txt | wc -l"})}}]},
        ],
        "tools": json.dumps(ic_tools),
    })
    rows.append({
        "name": "intercode:submit",
        "messages": [
            {"role": "system", "content": "You solve shell tasks. Call exactly one tool per turn."},
            {"role": "user", "content": "Question: ...\nPrevious steps produced the answer on screen.\nTurn 2: call exactly one tool now."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "submit", "arguments": json.dumps({})}}]},
        ],
        "tools": json.dumps(ic_tools),
    })
    return rows


def _render_check(tokenizer, row):
    """Render one row through the model's own template and measure assistant coverage
    using the SAME prefix-length diffing tokenize_and_mask uses."""
    if row.get("build_error"):
        # A diagnostic-script issue (e.g. an import), NOT a chat-template rejection —
        # must not be counted as a forfeit mechanism / must not trip the abort gate.
        return {"ok": False, "skip": True, "error": f"row-build failed: {row['build_error']}"}
    msgs, tools = _decode_row_for_template(row)
    try:
        ids = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    n_sup = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        try:
            p = len(tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=True, add_generation_prompt=True))
            r = len(tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=True, add_generation_prompt=False))
            n_sup += max(0, r - p)
        except Exception:
            pass
    return {"ok": True, "n_tokens": len(ids), "n_supervised": n_sup}


def _generated_assistant_text(tokenizer, msgs, tools, i):
    """The exact TEXT surface the model is trained to EMIT for assistant turn i.

    This is what SGLang's ``--tool-call-parser`` must read back at eval time. We compute
    it as the delta between the prompt render (with generation prompt) and the full render
    of msgs[:i+1] — i.e. the assistant turn's rendered bytes, trailing end markers included.
    """
    prefix = tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, add_generation_prompt=False)
    return full[len(prefix):] if full.startswith(prefix) else full


def _parse_toolcall_llama3(text):
    """Tolerant reimplementation of what SGLang's ``llama3`` tool-call parser extracts.

    Llama-3.1 emits a tool call as a JSON object ``{"name": ..., "parameters": {...}}``,
    optionally prefixed by ``<|python_tag|>`` and terminated by ``<|eom_id|>``/``<|eot_id|>``.
    We strip the Llama control markers, find the first balanced ``{...}`` object, json.loads
    it, and return ``(name, args_dict)`` — or ``None`` if nothing parses (which is exactly
    the case where SGLang would return the turn as plain text and the eval would forfeit).

    This is a FAITHFUL-BY-SHAPE check, not the byte-exact SGLang detector; the definitive
    confirmation is a live SGLang serve (Fase 4). Its job here is to catch the obvious
    break — a surface with no parseable JSON tool object — without a GPU.
    """
    if not text:
        return None
    s = text
    # Strip only Llama control markers (they sit OUTSIDE the JSON object); do NOT strip the
    # bare word "assistant" — the leading '{' search already skips the header, and stripping
    # it could corrupt a JSON string value that legitimately contains it.
    for marker in ("<|python_tag|>", "<|eom_id|>", "<|eot_id|>",
                   "<|start_header_id|>", "<|end_header_id|>"):
        s = s.replace(marker, " ")
    # First balanced top-level {...} object.
    start = s.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    blob = s[start:j + 1]
                    try:
                        obj = json.loads(blob)
                    except Exception:
                        break  # not JSON; advance to next '{'
                    if isinstance(obj, dict) and "name" in obj:
                        args = obj.get("parameters", obj.get("arguments", {}))
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        return (obj["name"], args if isinstance(args, dict) else {})
                    break
        start = s.find("{", start + 1)
    return None


def _check_parse(tokenizer, row, model_id, show):
    """Render each assistant tool-call turn to its emit surface and confirm a llama3-style
    parser recovers the SAME tool name (+ arg keys). A miss here is the runtime forfeit gate
    that training-time render success does NOT catch."""
    if row.get("build_error"):
        return {"skip": True}
    msgs, tools = _decode_row_for_template(row)
    results = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        want = m["tool_calls"][0]["function"]
        want_name = want["name"]
        want_args = want.get("arguments")
        if isinstance(want_args, str):
            try:
                want_args = json.loads(want_args)
            except Exception:
                want_args = {}
        try:
            surface = _generated_assistant_text(tokenizer, msgs, tools, i)
        except Exception as exc:
            results.append({"ok": False, "err": f"render: {exc!r}"})
            continue
        parsed = _parse_toolcall_llama3(surface)
        if show:
            print(f"        surface[{i}]: {surface.strip()[:200]!r}")
        if parsed is None:
            results.append({"ok": False, "err": "NO tool call parsed from surface (SGLang would forfeit)"})
        elif parsed[0] != want_name:
            results.append({"ok": False, "err": f"name mismatch: emitted→parsed {parsed[0]!r} != {want_name!r}"})
        else:
            want_keys = set((want_args or {}).keys())
            got_keys = set((parsed[1] or {}).keys())
            results.append({"ok": True, "name": want_name,
                            "args_ok": want_keys == got_keys, "want_keys": sorted(want_keys),
                            "got_keys": sorted(got_keys)})
    return {"results": results}


def _dataset_mode(tokenizer, path, max_length, sample):
    """Run the REAL train_sft_env.tokenize_and_mask over a generated dataset dir and
    report the survivor count + assistant coverage — the production path end to end."""
    from datasets import DatasetDict, load_from_disk
    from envs.chat_template_compat import assistant_tool_calls_ok
    from train_sft_env import tokenize_and_mask  # heavy import; only in this mode

    raw = load_from_disk(path)
    if "train" not in raw:
        raw = DatasetDict({"train": raw})
    n0 = len(raw["train"])
    kept = raw["train"].filter(lambda ex: assistant_tool_calls_ok(ex["messages"]))
    print(f"  compat-filter (single tool-call): kept {len(kept)}/{n0}")
    if sample and len(kept) > sample:
        kept = kept.shuffle(seed=42).select(range(sample))
    dd = DatasetDict({"train": kept})
    out = tokenize_and_mask(dd, tokenizer, max_length=max_length)
    n_out = len(out["train"])
    print(f"  tokenize_and_mask: kept {n_out}/{len(kept)} rows through the chat template")
    if n_out:
        import statistics
        fr = []
        for ids, m in zip(out["train"]["input_ids"][:256], out["train"]["assistant_masks"][:256]):
            if ids:
                fr.append(sum(m) / len(ids))
        print(f"  assistant-token coverage (<=256 rows): mean={statistics.mean(fr):.3f} "
              f"min={min(fr):.3f} zero-mask-rows={sum(1 for x in fr if x == 0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct",
                    help="base model whose chat template to test (the tournament base)")
    ap.add_argument("--dataset-path", default=None,
                    help="optional: a generated HF dataset dir to run the REAL tokenize_and_mask over")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=2000, help="rows to sample in --dataset-path mode")
    ap.add_argument("--show-render", action="store_true", help="print the rendered (text) surface of each sample row")
    ap.add_argument("--check-parse", action="store_true",
                    help="round-trip each assistant tool-call turn through a llama3-style parser to confirm "
                         "SGLang would extract it at eval (the runtime forfeit gate render-success does NOT catch)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"== loading tokenizer: {args.model} ==")
    tok = AutoTokenizer.from_pretrained(args.model)
    has_tmpl = bool(getattr(tok, "chat_template", None))
    print(f"  chat_template present: {has_tmpl}")

    print("== synthetic-row render check (no GPU; the decisive template test) ==")
    rows = _build_sample_rows()
    n_ok = n_reject = n_skip = 0
    for row in rows:
        res = _render_check(tok, row)
        if res.get("ok"):
            n_ok += 1
            print(f"  [OK]   {row['name']:28s} tokens={res['n_tokens']:4d} supervised={res['n_supervised']:3d}")
            if res["n_supervised"] == 0:
                print(f"         ^^ WARNING: zero supervised tokens — the assistant tool_call did not "
                      f"land in the masked span on this template")
        elif res.get("skip"):
            n_skip += 1
            print(f"  [SKIP] {row['name']:28s} -> {res['error']} "
                  f"(diagnostic-script issue, NOT a template rejection)")
        else:
            n_reject += 1
            print(f"  [FAIL] {row['name']:28s} -> {res['error']}")
            print(f"         ^^ THIS is a forfeit mechanism: such a row is DROPPED by "
                  f"tokenize_and_mask's try/except on '{args.model}'")
    tested = n_ok + n_reject
    print(f"  => rendered {n_ok}/{tested} testable rows on {args.model}"
          + (f"  ({n_skip} skipped due to diagnostic-script issues)" if n_skip else ""))
    if n_reject > 0:
        print("  VERDICT: the base model's template REJECTS some of our rows — on the real "
              "dataset those rows are silently dropped; if that hits enough rows the warm-start "
              "under-trains or falls back to the base model (the 0.0 signature). This is fixable "
              "in the row shape / template handling, NOT a base-transfer problem.")
    elif n_ok == 0:
        print("  VERDICT: INCONCLUSIVE — no rows could be built/tested (diagnostic-script issue, "
              "e.g. an import); fix the diagnostic before trusting the gate.")
    else:
        print("  VERDICT: all sample rows render + supervise on this base. Template rejection is "
              "NOT the 0.0 cause here; look at 3B->8B transfer / undertraining / data volume next.")

    if args.check_parse:
        print("== tool-call PARSE round-trip (emit surface -> llama3-style parser) ==")
        print("   confirms SGLang would EXTRACT our tool calls at eval; a FAIL here is a runtime")
        print("   forfeit gate that the render check above does NOT detect.")
        n_pass = n_fail = 0
        for row in rows:
            if row.get("build_error"):
                continue
            rep = _check_parse(tok, row, args.model, args.show_render)
            if rep.get("skip"):
                continue
            for res in rep["results"]:
                if res.get("ok"):
                    n_pass += 1
                    argnote = "args-keys OK" if res["args_ok"] else f"ARG-KEYS DIFFER want={res['want_keys']} got={res['got_keys']}"
                    print(f"  [PARSE-OK] {row['name']:28s} name={res['name']!r} {argnote}")
                else:
                    n_fail += 1
                    print(f"  [PARSE-FAIL] {row['name']:28s} -> {res['err']}")
        print(f"  => {n_pass} parsed, {n_fail} failed")
        if n_fail:
            print("  VERDICT: our trained emit surface does NOT round-trip through a llama3-style parser "
                  "→ SGLang would return these as plain text and every turn FORFEITS. THIS is a real "
                  "format bug to fix (and would explain a 0.0). Confirm against a live SGLang serve.")
        elif n_pass:
            print("  VERDICT: emit surface round-trips (name + arg-keys recovered). Format is NOT the "
                  "forfeit cause — the levers are grounding (stdout match) + capacity/mix + 8B transfer. "
                  "Still confirm once on a live SGLang `--tool-call-parser llama3` serve.")

    if args.show_render:
        print("== rendered surface (first sample row, tokenize=False) ==")
        msgs, tools = _decode_row_for_template(rows[0])
        try:
            print(tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False))
        except Exception as exc:
            print(f"  (render failed: {exc!r})")

    if args.dataset_path:
        print(f"== dataset mode: real tokenize_and_mask over {args.dataset_path} ==")
        _dataset_mode(tok, args.dataset_path, args.max_length, args.sample)


if __name__ == "__main__":
    main()
