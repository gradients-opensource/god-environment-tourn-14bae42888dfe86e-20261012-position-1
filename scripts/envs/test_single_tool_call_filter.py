"""Offline unit test for assistant_tool_calls_ok() — the Llama chat-template guard.

Verifies WITHOUT a GPU / tokenizer / Docker rebuild that the filter in
train_sft_env.main() keeps single-tool-call rows (what every generator we own emits) and
drops the stray multi-call / empty-call rows that make a Llama chat template raise
"This model only supports single tool-calls at once!".

The guard is dependency-free, so this imports it directly (no sys.modules shims).

Run from scripts/:
    python envs/test_single_tool_call_filter.py
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load chat_template_compat.py DIRECTLY (not `from envs.chat_template_compat import`) so we
# don't drag in envs/__init__.py -> shared_env (which imports `requests`). The guard is
# dependency-free by design; this keeps the test the same — runnable on a bare Python.
_spec = importlib.util.spec_from_file_location(
    "chat_template_compat", os.path.join(_HERE, "chat_template_compat.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
assistant_tool_calls_ok = _mod.assistant_tool_calls_ok


def _call(name="game_action", args='{"action_id": 3}'):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def _asst(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def test_single_call_is_ok():
    # Arrange: the exact shape pvp_selfplay / the game modules / intercode emit.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "your move"},
        _asst(_call()),
    ]
    # Act / Assert
    assert assistant_tool_calls_ok(msgs) is True
    print("[ok] single_call_is_ok")


def test_multi_call_is_dropped():
    msgs = [
        {"role": "user", "content": "your move"},
        _asst(_call(), _call(args='{"action_id": 7}')),  # 2 calls -> Llama raises
    ]
    assert assistant_tool_calls_ok(msgs) is False
    print("[ok] multi_call_is_dropped")


def test_empty_tool_calls_is_dropped():
    # 'tool_calls' present but length 0 also trips the Llama template (length != 1).
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": None, "tool_calls": []}]
    assert assistant_tool_calls_ok(msgs) is False
    print("[ok] empty_tool_calls_is_dropped")


def test_plain_assistant_text_is_ok():
    # No tool_calls at all (plain reasoning turn) is valid — only a PRESENT non-single fails.
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "sure, here goes"}]
    assert assistant_tool_calls_ok(msgs) is True
    print("[ok] plain_assistant_text_is_ok")


def test_multiturn_all_single_is_ok():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "turn1"},
        _asst(_call()),
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "turn2"},
        _asst(_call(args='{"action_id": 1}')),
    ]
    assert assistant_tool_calls_ok(msgs) is True
    print("[ok] multiturn_all_single_is_ok")


def test_multi_call_in_later_turn_is_dropped():
    # A good first turn does NOT rescue a bad later turn — the whole row must go.
    msgs = [
        {"role": "user", "content": "turn1"},
        _asst(_call()),
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "turn2"},
        _asst(_call(), _call()),  # bad
    ]
    assert assistant_tool_calls_ok(msgs) is False
    print("[ok] multi_call_in_later_turn_is_dropped")


def test_non_list_messages_is_dropped():
    assert assistant_tool_calls_ok(None) is False
    assert assistant_tool_calls_ok("not a list") is False
    print("[ok] non_list_messages_is_dropped")


if __name__ == "__main__":
    test_single_call_is_ok()
    test_multi_call_is_dropped()
    test_empty_tool_calls_is_dropped()
    test_plain_assistant_text_is_ok()
    test_multiturn_all_single_is_ok()
    test_multi_call_in_later_turn_is_dropped()
    test_non_list_messages_is_dropped()
    print("\nALL SINGLE-TOOL-CALL FILTER TESTS PASSED")
