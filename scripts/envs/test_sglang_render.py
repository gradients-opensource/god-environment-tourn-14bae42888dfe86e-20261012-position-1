"""Tests for sglang_template_tools: the tool dicts must match what the validator's SGLang v0.5.14 server
hands the chat template (Tool.model_dump()), captured from the real server image (G.O.D pvp-eval.dockerfile).

Run from scripts/:  python envs/test_sglang_render.py
The golden values below were printed by sglang.srt.entrypoints.openai.protocol.Tool in that image; the
full cross-check against the live SGLang class (game + intercode tools, Qwen2.5 / Llama-3.1 / Llama-3.2
templates, render byte-identical) passed on the VM on 2026-09-26.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def test_golden_minimal_tool() -> None:
    from envs.pvp_format import sglang_template_tools
    wire = [{"type": "function", "function": {"name": "submit", "description": "D",
                                              "parameters": {"type": "object", "properties": {}}}}]
    golden = ('[{"type": "function", "function": {"description": "D", "name": "submit", "parameters": '
              '{"type": "object", "properties": {}}, "strict": false}, "defer_loading": null}]')
    check("minimal tool == SGLang Tool.model_dump() (keys + order)", json.dumps(sglang_template_tools(wire)) == golden)


def test_real_tool_sets() -> None:
    from envs.intercode_format import build_intercode_action_tools
    from envs.pvp_format import build_pvp_tools, sglang_template_tools, tools_to_openai
    for name, tools in (("game", tools_to_openai(build_pvp_tools([1, 2]))),
                        ("intercode", tools_to_openai(build_intercode_action_tools()))):
        before = json.dumps(tools)
        out = sglang_template_tools(tools)
        check(f"{name}: input not mutated", json.dumps(tools) == before)
        check(f"{name}: same tool count", len(out) == len(tools))
        check(f"{name}: names preserved in order",
              [t["function"]["name"] for t in out] == [t["function"]["name"] for t in tools])
        check(f"{name}: parameters preserved",
              all(o["function"]["parameters"] == t["function"]["parameters"] for o, t in zip(out, tools)))
        check(f"{name}: function key order is SGLang's",
              all(list(o["function"]) == ["description", "name", "parameters", "strict"] for o in out))
        check(f"{name}: tool key order is SGLang's",
              all(list(o) == ["type", "function", "defer_loading"] for o in out))


def test_flat_function_input() -> None:
    from envs.pvp_format import sglang_template_tools
    out = sglang_template_tools([{"name": "f", "description": "d", "parameters": {}}])
    check("flat function dict is wrapped", out[0]["type"] == "function" and out[0]["function"]["name"] == "f")


def test_malformed_entries_pass_through() -> None:
    from envs.pvp_format import sglang_template_tools
    out = sglang_template_tools([None, "x", {"type": "function", "function": None}])
    check("malformed entries pass through without raising", out == [None, "x", {"type": "function", "function": None}])


if __name__ == "__main__":
    test_golden_minimal_tool()
    test_real_tool_sets()
    test_flat_function_input()
    test_malformed_entries_pass_through()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
