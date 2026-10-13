"""Offline test for pvp_matchup_memory: LONG_TERM notes in game rows + post-game reflection rows.

Checks, on REAL pyspiel games (no GPU, no validator checkout):
  * the record note format and its tally (what the reflection rows teach);
  * game rows keep their user prompt, tools and move label byte-for-byte and only gain a filled
    LONG_TERM block (WORKING stays empty), at the configured share, deterministically and
    independent of row order;
  * reflection rows are single long_term_memory_rewrite calls over the 4 memory tools (no
    game_action), whose note is the previous note's record plus this game's result;
  * every game the module plays is seeded from the env's generation id range, never from the
    validator's eval seed span 1..1,000,000;
  * the kill switches restore the input rows exactly (a malformed value counts as off); foreign rows
    are never touched.
The byte-exact match against the validator's own LLMBot render lives in test_pvp_eval_parity.py.

Run from scripts/envs:
    python test_pvp_matchup_memory.py
"""

import json
import os
import random
import sys
import types
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.pvp_matchup_memory as mm  # noqa: E402
import envs.pvp_selfplay as sp  # noqa: E402
from envs.chat_template_compat import assistant_tool_calls_ok  # noqa: E402
from envs.pvp_format import EMPTY_MEMORY_BLOCK, REFLECTION_GUIDANCE, TOOL_GUIDANCE  # noqa: E402
from envs.shared_env import GAMES_TO_TASK_ID_RANGE  # noqa: E402

GAMES = ["liars_dice", "leduc_poker", "gin_rummy", "othello", "clobber", "goofspiel"]
MEMORY_TOOLS = ["working_memory_rewrite", "working_memory_append",
                "long_term_memory_rewrite", "long_term_memory_append"]
EVAL_SEED_MAX = 1_000_000  # validator core/pvp/constants.py PVP_SEED_RANGE_MAX (eval seeds are 1..this)
TEST_SEED0 = EVAL_SEED_MAX + 1  # fixtures stay out of the eval span too


def _game_rows(env: str, n_games: int, seed0: int = TEST_SEED0) -> list[dict]:
    """Eval-shaped game rows (pvp_selfplay._build_row) along random play of real games."""
    rows = []
    for g in range(n_games):
        seed = seed0 + 7919 * g
        rng = random.Random(seed)
        game = sp._load_game(env, sp._config_id_for_seed(seed, env))
        state = game.new_initial_state()
        sp._setup_initial_state(env, state, seed)
        while not state.is_terminal():
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            p = state.current_player()
            legal = state.legal_actions(p)
            labels = [(a, state.action_to_string(p, a)) for a in legal]
            action = rng.choice(legal)
            rows.append(sp._build_row(env, sp._FORMATTERS[env](state, p), p, labels, action))
            state.apply_action(action)
    return rows


def _long_term_slot(system: str, slot: int) -> str:
    head = "LONG_TERM_MEMORY (your notes):\n"
    block = system[system.index(head) + len(head):]
    line = next(ln for ln in block.split("\n  [") if ln.lstrip(" [").startswith(f"{slot}] "))
    return line.split("] ", 1)[1].split("\n\n")[0]


def test_record_note_round_trip():
    note = mm.record_note(2, 1, 0, "win")
    assert note == "Record vs this opponent: 2 won, 1 lost, 0 drawn. Last game: WIN. Plan: keep my standard strategy.", note
    assert mm.parse_record_note(note) == (2, 1, 0)
    assert mm.parse_record_note("Opponent bluffs a lot.") is None
    assert mm.tallied(None, "loss") == (0, 1, 0)
    assert mm.tallied((2, 1, 0), "draw") == (2, 1, 1)
    assert len(note.split()) <= 64
    print("PASS record_note_round_trip")


def test_game_rows_keep_label_and_gain_long_term_only():
    rows = _game_rows("liars_dice", 30)
    out, stats = mm.add_matchup_memory("liars_dice", rows, seed=42, memory_share=1.0, reflection_share=0.0)
    assert len(out) == len(rows) and stats["memory_rows"] == len(rows), stats
    for before, after in zip(rows, out):
        assert after["tools"] == before["tools"]
        assert after["messages"][1:] == before["messages"][1:], "user prompt or move label changed"
        sys_before, sys_after = before["messages"][0]["content"], after["messages"][0]["content"]
        assert EMPTY_MEMORY_BLOCK in sys_before and EMPTY_MEMORY_BLOCK not in sys_after
        assert sys_after.endswith("\n\n" + TOOL_GUIDANCE)
        assert "WORKING_MEMORY (your notes):\n  [1] (empty)\n  [2] (empty)\n  [3] (empty)\n  [4] (empty)" in sys_after
        assert _long_term_slot(sys_after, 1) != "(empty)"
    assert all(r["messages"][0]["content"] == rows[0]["messages"][0]["content"] for r in rows), "input mutated"
    print(f"PASS game_rows_keep_label_and_gain_long_term_only ({len(rows)} rows)")


def test_share_determinism_and_order_independence():
    rows = _game_rows("leduc_poker", 300)
    out_a, stats = mm.add_matchup_memory("leduc_poker", rows, seed=42, memory_share=0.8, reflection_share=0.02)
    out_b, _ = mm.add_matchup_memory("leduc_poker", rows, seed=42, memory_share=0.8, reflection_share=0.02)
    assert out_a == out_b, "not deterministic"
    n = len(rows)
    share = stats["memory_rows"] / n
    assert 0.74 <= share <= 0.86, f"memory share {share:.3f} off 0.8 on {n} rows"
    assert stats["reflection_rows"] == round(0.02 * n), stats
    assert 0.15 <= stats["freeform_rows"] / stats["memory_rows"] <= 0.35, stats
    shuffled = list(rows)
    random.Random(5).shuffle(shuffled)
    out_c, _ = mm.add_matchup_memory("leduc_poker", shuffled, seed=42, memory_share=0.8, reflection_share=0.0)

    def pairs(out: list[dict]) -> Counter:
        return Counter((json.dumps(r["messages"][1:]), r["messages"][0]["content"]) for r in out)

    assert pairs(out_a[:n]) == pairs(out_c), "the rendered rows depend on the input order"
    print(f"PASS share_determinism_and_order_independence (memory {share:.3f}, "
          f"reflection {stats['reflection_rows']}/{n}, free-form {stats['freeform_rows']})")


def test_duplicate_states_draw_independently():
    """Copies of one state (leduc rows are mostly repeats) each get their own draw: the share holds per
    state and a repeated state is seen with more than one note."""
    row = _game_rows("leduc_poker", 1)[0]
    out, stats = mm.add_matchup_memory("leduc_poker", [row] * 200, seed=42, memory_share=0.8, reflection_share=0.0)
    share = stats["memory_rows"] / 200
    assert 0.7 <= share <= 0.9, f"memory share {share:.3f} on 200 copies of one state"
    systems = {r["messages"][0]["content"] for r in out}
    assert len(systems) > 10, f"only {len(systems)} distinct memory blocks over 200 copies"
    print(f"PASS duplicate_states_draw_independently (share {share:.3f}, {len(systems)} distinct blocks)")


def test_reflection_rows_teach_the_tally():
    for env in GAMES:
        firsts = 0
        for i in range(24):
            ex = mm.build_reflection_example(env, game_seed=TEST_SEED0 + 1000 + i)
            msgs = ex.row["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[0]["content"].endswith("\n\n" + REFLECTION_GUIDANCE)
            assert msgs[1]["content"].startswith(f"The game is over. Result for you: {ex.outcome.upper()}.\n\nFinal state:\n")
            assert ex.state.is_terminal()
            names = [t["function"]["name"] for t in json.loads(ex.row["tools"])]
            assert names == MEMORY_TOOLS, names
            assert assistant_tool_calls_ok(msgs)
            (call,) = msgs[2]["tool_calls"]
            assert call["function"]["name"] == "long_term_memory_rewrite"
            args = json.loads(call["function"]["arguments"])
            assert list(args) == ["slot", "content"] and args["slot"] == 1
            prior = mm.parse_record_note(ex.prior_slots.get(1, ""))
            firsts += prior is None
            assert prior is not None or ex.prior_slots == {}
            assert mm.parse_record_note(args["content"]) == mm.tallied(prior, ex.outcome)
            assert args["content"].endswith(f"Last game: {ex.outcome.upper()}. Plan: keep my standard strategy.")
            assert _long_term_slot(msgs[0]["content"], 1) == (ex.prior_slots.get(1) or "(empty)")
        assert firsts >= 1, f"{env}: no first-game (empty memory) reflection in 24 draws"
    print("PASS reflection_rows_teach_the_tally (6 games)")


def test_reflection_outcomes_and_draws_follow_the_game():
    seen = {}
    for env in GAMES:
        outs = {mm.build_reflection_example(env, game_seed=TEST_SEED0 + s).outcome for s in range(40)}
        assert outs <= {"win", "loss", "draw"}
        assert {"win", "loss"} <= outs, f"{env}: outcomes {outs}"
        seen[env] = outs
    for env in ("liars_dice", "clobber"):
        assert "draw" not in seen[env], f"{env} cannot draw"
    print(f"PASS reflection_outcomes_and_draws_follow_the_game {sorted((e, sorted(o)) for e, o in seen.items())}")


def test_reflection_seeds_stay_outside_the_eval_seed_span():
    """Every game seed the module plays (first try and outcome retries) is a generation id of that env,
    above the eval seed span 1..1,000,000, so no reflection row replays an eval config and opening.
    build_reflection_example refuses an eval-span seed from any caller."""
    tried: dict[str, list[int]] = {}
    real = mm._play_to_terminal

    def spy(env_name: str, game_seed: int):
        tried.setdefault(env_name, []).append(game_seed)
        return real(env_name, game_seed)

    mm._play_to_terminal = spy
    try:
        for env in GAMES:
            row = _game_rows(env, 1)[0]
            mm.add_matchup_memory(env, [row] * 6, seed=42, memory_share=0.0, reflection_share=1.0)
    finally:
        mm._play_to_terminal = real
    for env in GAMES:
        lo, hi = GAMES_TO_TASK_ID_RANGE[env]
        seeds = tried.get(env, [])
        assert len(seeds) >= 6, f"{env}: {len(seeds)} reflection games played"
        assert min(seeds) > EVAL_SEED_MAX, f"{env}: seed {min(seeds)} inside the eval seed span"
        assert lo < min(seeds) and max(seeds) < hi, f"{env}: seeds {min(seeds)}..{max(seeds)} outside ({lo}, {hi})"
    for bad in (1, EVAL_SEED_MAX):
        try:
            mm.build_reflection_example("liars_dice", game_seed=bad)
        except ValueError:
            continue
        raise AssertionError(f"eval-span seed {bad} accepted")
    print(f"PASS reflection_seeds_stay_outside_the_eval_seed_span "
          f"({sum(map(len, tried.values()))} games, min seed {min(min(s) for s in tried.values())})")


def test_kill_switches_and_foreign_rows():
    rows = _game_rows("othello", 2)
    off, stats = mm.add_matchup_memory("othello", rows, seed=42, memory_share=0.0, reflection_share=0.0)
    assert off == rows and stats["memory_rows"] == 0 and stats["reflection_rows"] == 0
    os.environ["PVP_MEMORY_ROW_SHARE"] = "0"
    os.environ["PVP_REFLECTION_ROW_SHARE"] = "0"
    try:
        env_off, _ = mm.add_matchup_memory("othello", rows, seed=42)
        assert env_off == rows
        os.environ["PVP_MEMORY_ROW_SHARE"] = "off"
        os.environ["PVP_REFLECTION_ROW_SHARE"] = "no"
        malformed_off, _ = mm.add_matchup_memory("othello", rows, seed=42)
        assert malformed_off == rows, "a malformed kill-switch value must switch the lever off"
    finally:
        del os.environ["PVP_MEMORY_ROW_SHARE"], os.environ["PVP_REFLECTION_ROW_SHARE"]
    foreign = {"messages": [{"role": "system", "content": "custom prompt"}, *rows[0]["messages"][1:]],
               "tools": rows[0]["tools"]}
    out, stats = mm.add_matchup_memory("othello", [foreign], seed=42, memory_share=1.0, reflection_share=0.0)
    assert out == [foreign] and stats["memory_rows"] == 0 and stats["foreign_rows"] == 1
    on, stats = mm.add_matchup_memory("othello", rows * 50, seed=42)
    assert stats["memory_rows"] > 0 and stats["reflection_rows"] == round(mm.REFLECTION_ROW_SHARE_DEFAULT * len(rows) * 50)
    print("PASS kill_switches_and_foreign_rows (defaults on, 0 or malformed restores input, foreign rows untouched)")


def test_reflection_row_cap():
    rows = _game_rows("leduc_poker", 2)
    _, stats = mm.add_matchup_memory("leduc_poker", rows * 4000, seed=42, memory_share=0.0, reflection_share=0.5)
    assert stats["reflection_rows"] == mm.REFLECTION_MAX_ROWS, stats
    print(f"PASS reflection_row_cap ({mm.REFLECTION_MAX_ROWS})")


def test_notes_fit_the_slot_budget():
    rng = random.Random(3)
    for env in GAMES:
        for _ in range(200):
            for text in mm.game_row_long_term(rng, env).values():
                assert 0 < len(text.split()) <= 64, text
    print("PASS notes_fit_the_slot_budget")


def test_generate_for_env_saves_memory_and_reflection_rows():
    """Through the real entrypoint: the saved PvP dataset mixes LONG_TERM-note rows, plain rows and
    reflection rows in one Arrow table, every row a single tool call."""
    import tempfile

    from datasets import DatasetDict

    import envs.generate_trajectories as gt

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ds_leduc_poker")
        gt.generate_for_env("leduc_poker", path, num_games=150, max_turn=10, num_workers=2)
        rows = DatasetDict.load_from_disk(path)["train"].to_list()
    names = [r["messages"][-1]["tool_calls"][0]["function"]["name"] for r in rows]
    n_reflection = names.count("long_term_memory_rewrite")
    n_game = names.count("game_action")
    assert n_game + n_reflection == len(rows), set(names)
    assert n_reflection == round(mm.REFLECTION_ROW_SHARE_DEFAULT * n_game) and n_reflection > 0, (n_reflection, n_game)
    with_notes = sum(EMPTY_MEMORY_BLOCK not in r["messages"][0]["content"] for r in rows if
                     r["messages"][-1]["tool_calls"][0]["function"]["name"] == "game_action")
    assert 0.7 <= with_notes / n_game <= 0.9, with_notes / n_game
    assert all(len(r["messages"][-1]["tool_calls"]) == 1 for r in rows)
    print(f"PASS generate_for_env_saves_memory_and_reflection_rows ({n_game} game rows, "
          f"{with_notes} with notes, {n_reflection} reflection rows)")


if __name__ == "__main__":
    test_record_note_round_trip()
    test_game_rows_keep_label_and_gain_long_term_only()
    test_share_determinism_and_order_independence()
    test_duplicate_states_draw_independently()
    test_reflection_rows_teach_the_tally()
    test_reflection_outcomes_and_draws_follow_the_game()
    test_reflection_seeds_stay_outside_the_eval_seed_span()
    test_kill_switches_and_foreign_rows()
    test_reflection_row_cap()
    test_notes_fit_the_slot_budget()
    test_generate_for_env_saves_memory_and_reflection_rows()
    print("ALL PASS test_pvp_matchup_memory")
