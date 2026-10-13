"""Offline unit test for swe_infinite_dataset row conversion (no datasets/GPU/Docker needed).

Shims the heavy imports (datasets, envs.miner_dataset_loader) so the pure logic
(_clean_messages, _passes_exit_filter) runs in isolation — same trick as test_merge_balance.py.

Run from scripts/:
    python envs/test_swe_infinite_dataset.py
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# --- Shim heavy imports so swe_infinite_dataset imports standalone -------------
_ds = types.ModuleType("datasets")
_ds.Dataset = object
_ds.DatasetDict = object
sys.modules["datasets"] = _ds

_envs_pkg = types.ModuleType("envs")
_envs_pkg.__path__ = [_HERE]
sys.modules["envs"] = _envs_pkg

_mdl = types.ModuleType("envs.miner_dataset_loader")
_mdl._get_miner_datasets_inventory = lambda: []
_mdl._load_one_dataset = lambda p: None
sys.modules["envs.miner_dataset_loader"] = _mdl

import envs.swe_infinite_dataset as swe  # noqa: E402


def _m(role, content):
    return {"role": role, "content": content}


_ASST = _m("assistant", "THOUGHT: look\n```bash\nls\n```")


def test_valid_trajectory_passes():
    msgs = [_m("system", "sys"), _m("user", "fix the bug"), _ASST]
    out = swe._clean_messages(msgs)
    assert out is not None and len(out) == 3
    assert out[-1]["role"] == "assistant"
    print("[ok] valid_trajectory_passes")


def test_no_assistant_dropped():
    # Nothing to compute loss on -> drop the whole trajectory.
    assert swe._clean_messages([_m("system", "s"), _m("user", "u")]) is None
    print("[ok] no_assistant_dropped")


def test_empty_assistant_dropped():
    assert swe._clean_messages([_m("user", "u"), _m("assistant", "   ")]) is None
    print("[ok] empty_assistant_dropped")


def test_malformed_turns_skipped():
    msgs = ["notadict", _m("user", "u"), {"role": "bogus", "content": "y"}, _ASST]
    out = swe._clean_messages(msgs)
    assert out is not None
    assert [m["role"] for m in out] == ["user", "assistant"]  # non-dict + bad role dropped
    print("[ok] malformed_turns_skipped")


def test_none_content_turn_skipped():
    msgs = [_m("user", "u"), {"role": "assistant", "content": None}, _ASST]
    out = swe._clean_messages(msgs)
    assert out is not None and sum(m["role"] == "assistant" for m in out) == 1
    print("[ok] none_content_turn_skipped")


def test_clean_messages_is_plain_no_tools():
    out = swe._clean_messages([_m("user", "u"), _ASST])
    assert out is not None
    # plain-text swe_infinite: every turn is {role, content} — NO tool_calls / tools keys
    assert all(set(m.keys()) == {"role", "content"} for m in out)
    print("[ok] clean_messages_is_plain_no_tools")


def test_exit_filter_drops_incomplete():
    # ~96% of SWE-ZERO is exit_status="incomplete" (failed attempts) — the quality gate drops them.
    assert swe._passes_exit_filter({"exit_status": "Submitted"}, {"submitted"}) is True   # case-insensitive keep
    assert swe._passes_exit_filter({"exit_status": "incomplete"}, {"submitted"}) is False  # drop the failures
    assert swe._passes_exit_filter({"exit_status": "incomplete"}, None) is True  # keep=None → keep all
    assert swe._passes_exit_filter({}, {"submitted"}) is True  # missing status → lenient keep
    print("[ok] exit_filter_drops_incomplete")


def test_default_quality_gate_is_submitted_only():
    assert swe._DEFAULT_KEEP_EXIT_STATUSES == frozenset({"submitted"})
    print("[ok] default_quality_gate_is_submitted_only")


# ---- Fix #2: fit-to-window (anti-truncation) ---------------------------------

def test_cap_observations_only_hits_observations():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "ISSUE " + "x" * 5000},       # the issue — keep full
        {"role": "assistant", "content": "THOUGHT\n```bash\nls\n```"},  # keep full
        {"role": "user", "content": "OBS " + "y" * 5000},         # observation — cap
    ]
    out = swe._cap_observations(msgs, obs_cap=100)
    assert out[1]["content"] == msgs[1]["content"]                 # issue untouched
    assert out[2]["content"] == msgs[2]["content"]                 # assistant untouched
    assert out[3]["content"].startswith("OBS ")                    # observation truncated (head kept)
    assert len(out[3]["content"]) <= 100 + len(swe._OBS_TRUNCATION_MARKER)
    print("[ok] cap_observations_only_hits_observations")


def test_fit_keeps_short_trajectory_whole():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "cmd1"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "cmd2 submit"},
    ]
    assert swe._fit_messages(msgs, max_chars=10000, obs_cap=2000) == msgs  # under budget → unchanged
    print("[ok] fit_keeps_short_trajectory_whole")


def test_fit_trims_long_but_keeps_issue_and_fix():
    # A long trajectory whose FIX/submit is the LAST assistant turn — must survive the trim.
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "ISSUE-anchor"}]
    for i in range(20):
        msgs.append({"role": "assistant", "content": f"cmd{i} " + "a" * 500})
        msgs.append({"role": "user", "content": f"obs{i} " + "b" * 500})
    msgs.append({"role": "assistant", "content": "FINAL FIX + submit"})

    out = swe._fit_messages(msgs, max_chars=3000, obs_cap=2000)
    assert out is not None
    assert out[0] == {"role": "system", "content": "S"}            # anchor: system
    assert out[1]["content"] == "ISSUE-anchor"                     # anchor: the issue
    assert out[-1]["content"] == "FINAL FIX + submit"              # the fix at the END survives
    assert out[2]["role"] == "assistant"                          # a window opens on an assistant turn
    assert swe._total_chars(out) <= 3000                          # actually fits the budget
    assert len(out) < len(msgs)                                    # it really did trim
    print("[ok] fit_trims_long_but_keeps_issue_and_fix")


if __name__ == "__main__":
    test_valid_trajectory_passes()
    test_no_assistant_dropped()
    test_empty_assistant_dropped()
    test_malformed_turns_skipped()
    test_none_content_turn_skipped()
    test_clean_messages_is_plain_no_tools()
    test_exit_filter_drops_incomplete()
    test_default_quality_gate_is_submitted_only()
    test_cap_observations_only_hits_observations()
    test_fit_keeps_short_trajectory_whole()
    test_fit_trims_long_but_keeps_issue_and_fix()
    print("\nALL SWE-INFINITE-DATASET TESTS PASSED")
