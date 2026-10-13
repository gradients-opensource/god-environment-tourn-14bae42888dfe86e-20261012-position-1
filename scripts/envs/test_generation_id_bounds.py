"""CPU test: generation id ranges must never overlap the validator's eval seed range.

shared_env.GAMES_TO_TASK_ID_RANGE is the single source of truth for every game env's task-id
span, but goofspiel's raw span starts at 0 — squarely inside the validator's own eval seed
range (1..1_000_000). A 2026-09-30 sample found 72/8000 goofspiel generation ids landing
inside that span. ``generation_id_bounds`` offsets the GENERATION sampling floor above the
eval ceiling without touching ``GAMES_TO_TASK_ID_RANGE`` itself (other callers, e.g. the
env-server warm-up sentinel and the GRPO task-id index, keep reading the raw table).

Run from scripts/envs/:  python test_generation_id_bounds.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from envs.shared_env import (  # noqa: E402
    GAMES_TO_TASK_ID_RANGE,
    VALIDATOR_EVAL_ID_CEILING,
    generation_id_bounds,
)

EVAL_RANGE = range(1, VALIDATOR_EVAL_ID_CEILING + 1)  # the validator's own eval seed span


def test_no_env_generation_range_overlaps_eval_seeds():
    for env_name in GAMES_TO_TASK_ID_RANGE:
        low, high = generation_id_bounds(env_name)
        assert low < high, f"{env_name}: empty generation range ({low}, {high})"
        overlap_lo = max(low, EVAL_RANGE.start)
        overlap_hi = min(high - 1, EVAL_RANGE.stop - 1)  # generation range is exclusive-high
        assert overlap_lo > overlap_hi, (
            f"{env_name}: generation range [{low}, {high}) overlaps the validator's eval seed "
            f"range 1..{VALIDATOR_EVAL_ID_CEILING} at [{overlap_lo}, {overlap_hi}]"
        )
    print(f"[ok] no_env_generation_range_overlaps_eval_seeds: {len(GAMES_TO_TASK_ID_RANGE)} envs clear")


def test_goofspiel_floor_is_pushed_above_the_eval_ceiling():
    # goofspiel is the one env whose raw range (starts at 0) actually needed the offset.
    low, high = generation_id_bounds("goofspiel")
    raw_low, raw_high = GAMES_TO_TASK_ID_RANGE["goofspiel"]
    assert low == VALIDATOR_EVAL_ID_CEILING + 1, f"expected floor pushed to {VALIDATOR_EVAL_ID_CEILING + 1}, got {low}"
    assert high == raw_high, "upper bound must stay exactly as the raw table defines it"
    print(f"[ok] goofspiel_floor_is_pushed_above_the_eval_ceiling: {raw_low}->{low}, high unchanged at {high}")


def test_other_envs_are_unaffected_by_the_offset():
    # Every other env already starts at 100_000_000+, well above the eval ceiling, so the
    # offset must be a complete no-op for them (byte-identical low bound to today).
    for env_name, (raw_low, _raw_high) in GAMES_TO_TASK_ID_RANGE.items():
        if env_name == "goofspiel":
            continue
        low, _high = generation_id_bounds(env_name)
        assert low == raw_low + 1, f"{env_name}: expected unaffected floor {raw_low + 1}, got {low}"
    print("[ok] other_envs_are_unaffected_by_the_offset: unchanged for every non-goofspiel env")


def test_table_itself_is_untouched():
    # generation_id_bounds must not mutate the shared single-source-of-truth table — other
    # callers (env-server warm-up sentinel, train_grpo_env.py's task-id index) read it raw.
    before = dict(GAMES_TO_TASK_ID_RANGE)
    generation_id_bounds("goofspiel")
    assert GAMES_TO_TASK_ID_RANGE == before, "GAMES_TO_TASK_ID_RANGE must not be mutated"
    assert GAMES_TO_TASK_ID_RANGE["goofspiel"] == (0, 99_999_999), "raw goofspiel range must stay (0, 99_999_999)"
    print("[ok] table_itself_is_untouched: GAMES_TO_TASK_ID_RANGE unchanged")


if __name__ == "__main__":
    test_no_env_generation_range_overlaps_eval_seeds()
    test_goofspiel_floor_is_pushed_above_the_eval_ceiling()
    test_other_envs_are_unaffected_by_the_offset()
    test_table_itself_is_untouched()
    print("\nALL GENERATION ID BOUNDS TESTS PASSED")
