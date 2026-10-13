"""Chat-template compatibility guards for the EnvTask SFT tokenizer.

Kept dependency-free (pure dict/list logic, no torch/trl/transformers) so it can be
unit-tested offline — ``train_sft_env.py`` imports the heavy training stack, this module
does not, and imports the guard from here.

Why this exists: Llama-family chat templates raise

    jinja2.exceptions.TemplateError: This model only supports single tool-calls at once!

for any assistant turn whose ``tool_calls`` list does not hold EXACTLY one call (an empty
list trips it too). Qwen's template tolerates any count, so this only surfaces when the
task assigns a Llama model. Every trajectory WE generate (pvp_selfplay + the per-game
modules + intercode_dataset) emits exactly one ``game_action``/``execute_bash`` call per
assistant turn, so this is a defensive guard: it drops nothing today and only prunes a
row if a generator ever emits a multi-call (or empty-call) turn — malformed for a
turn-based game anyway (one turn == one action). Dropping such a row lets the warm-start
tokenize on every whitelisted env model, not just Qwen.
"""

from __future__ import annotations

from typing import Any


def assistant_tool_calls_ok(messages: "list[dict[str, Any]]") -> bool:
    """True unless some assistant turn carries a ``tool_calls`` that is not exactly one.

    A message with no ``tool_calls`` (plain assistant text) is fine — only a present but
    non-single (0, ≥2, or non-list) ``tool_calls`` fails, because that is exactly what a
    Llama chat template rejects.
    """
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list) or len(calls) != 1:
            return False
    return True
