"""Offline unit test for merge_datasets() cap + floor rebalancing (utah balance policy).

Verifies, WITHOUT a GPU / env server / Docker rebuild, that the per-env balance in
``generate_trajectories.merge_datasets`` does what the utah balance policy needs:
  * a flooding scored env is CAPPED down to scored_cap,
  * an under-represented SCORED env is OVERSAMPLED up to the floor,
  * a small COVERAGE (non-scored) env is left thin (never floored, so it can't
    dilute the scored envs),
  * an env already between floor and cap is passed through unchanged,
  * floor_frac=None disables oversampling; a set floor is RELATIVE to the largest scored env,
  * a round with >= 3 scored envs and a total_rows budget is sized by the share table instead
    (fixed total, leduc floor, an under-filled env hands its quota to the others), while the
    R1 2-env composition stays exactly as it was.

generate_trajectories imports trl + heavy env modules at top level, so we shim
those in sys.modules before importing it (same trick as test_goofspiel_gen.py);
merge_datasets itself only needs ``datasets`` and ``_log``.

Run from scripts/:
    uv run --no-project --with datasets python envs/test_merge_balance.py
"""

import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# --- Shim heavy imports so merge_datasets can be imported in isolation ---------
sys.modules["trl_compat"] = types.ModuleType("trl_compat")

_envs_pkg = types.ModuleType("envs")
_envs_pkg.__path__ = [_HERE]  # resolve real submodules (generate_trajectories) from here
sys.modules["envs"] = _envs_pkg

_shared = types.ModuleType("envs.shared_env")
_shared.GAMES_TO_TASK_ID_RANGE = {}
_shared.init_env_pool = lambda *a, **k: None
_shared._log = lambda *a, **k: print(*a, **k)  # keep the [balance] lines visible
_shared.recipe_log = lambda detail, tag="": print(tag or detail)
_shared.generation_id_bounds = lambda env_name: (0, 1)  # unused by merge_datasets under test
sys.modules["envs.shared_env"] = _shared

_cfg = types.ModuleType("envs.sft_env_configs")
_cfg._OFFLINE_ENVS = {"intercode"}
_cfg._SELFPLAY_ENVS = set()
_cfg.get_sft_trajectory_generator = lambda *a, **k: None
sys.modules["envs.sft_env_configs"] = _cfg

from datasets import Dataset, DatasetDict  # noqa: E402
import envs.generate_trajectories as gt  # noqa: E402
import budget_controller as bc  # noqa: E402


def _make_env(output_path: str, env_name: str, n_train: int, n_val: int = 2) -> str:
    """Save a fake per-env DatasetDict at f"{output_path}_{env_name}" with an `env`
    marker column so each env's contribution is countable after the merge."""
    train = Dataset.from_dict({"env": [env_name] * n_train, "x": list(range(n_train))})
    val = Dataset.from_dict({"env": [env_name] * n_val, "x": list(range(n_val))})
    path = f"{output_path}_{env_name}"
    DatasetDict({"train": train, "validation": val}).save_to_disk(path)
    return path


def _env_counts(output_path: str) -> dict:
    merged = DatasetDict.load_from_disk(output_path)["train"]
    counts: dict = {}
    for e in merged["env"]:
        counts[e] = counts.get(e, 0) + 1
    return counts


def test_cap_floor_rebalance():
    # Legacy (<3-env / no total_rows) path: cap = scored_cap x weapon weight, floor = floor_frac x the
    # largest capped scored env, scaled per env by weight / max weight.
    scored = {"clobber", "intercode", "leduc_poker"}
    w = gt._env_weights(scored)
    cap = 7000
    clobber_cap = int(round(cap * w["clobber"]))
    base_floor = int(0.5 * clobber_cap)
    w_max = max(w[e] for e in scored)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        paths = [
            _make_env(out, "clobber", 20000),     # scored, floods   -> weighted cap
            _make_env(out, "intercode", 1000),    # scored, starved  -> weighted floor
            _make_env(out, "leduc_poker", 5000),  # scored           -> weighted cap
            _make_env(out, "othello", 800),       # COVERAGE, small  -> keep 800 (not floored)
        ]
        gt.merge_datasets(paths, out, scored_envs=scored, scored_cap=cap, coverage_cap=6000, floor_frac=0.5)
        counts = _env_counts(out)

        want = {
            "clobber": clobber_cap,
            "intercode": int(base_floor * w["intercode"] / w_max),
            "leduc_poker": min(5000, int(round(cap * w["leduc_poker"]))),
            "othello": 800,
        }
        assert counts == want, f"legacy cap/floor mismatch: {counts} != {want}"
        print(f"[ok] cap_floor_rebalance: {counts}")


def test_floor_none_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        paths = [_make_env(out, "intercode", 1000)]
        gt.merge_datasets(
            paths, out,
            scored_envs={"intercode"}, scored_cap=7000, coverage_cap=6000, floor_frac=None,
        )
        counts = _env_counts(out)
        assert counts["intercode"] == 1000, f"floor_frac=None should not oversample: {counts}"
        print("[ok] floor_none_is_noop: small env left as-is when floor disabled")


def test_coverage_never_floored():
    # A starved COVERAGE env (below the floor, not in scored_envs) must stay thin.
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        paths = [
            _make_env(out, "clobber", 5000),   # scored, in-range -> keep
            _make_env(out, "gin_rummy", 500),  # coverage, starved -> stays 500 (thin replay)
        ]
        gt.merge_datasets(
            paths, out,
            scored_envs={"clobber"}, scored_cap=7000, coverage_cap=6000, floor_frac=0.5,
        )
        counts = _env_counts(out)
        assert counts["gin_rummy"] == 500, f"coverage gin_rummy was floored: {counts}"
        assert counts["clobber"] == 5000, f"clobber changed: {counts}"
        print(f"[ok] coverage_never_floored: {counts}")


def test_dynamic_floor_scales_with_max():
    # No scored env hits its cap, so the floor is RELATIVE to the largest scored env (0.5 x 2000),
    # scaled per env by weight / max weight -- not a fixed number.
    scored = {"clobber", "leduc_poker", "intercode"}
    w = gt._env_weights(scored)
    w_max = max(w[e] for e in scored)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        paths = [
            _make_env(out, "clobber", 2000),      # scored, largest -> keep 2000
            _make_env(out, "leduc_poker", 1000),  # scored, above its floor -> keep
            _make_env(out, "intercode", 400),     # scored, below its floor -> oversampled
        ]
        gt.merge_datasets(paths, out, scored_envs=scored, scored_cap=7000, coverage_cap=6000, floor_frac=0.5)
        counts = _env_counts(out)
        ic_floor = int(1000 * w["intercode"] / w_max)
        assert counts["clobber"] == 2000, f"clobber changed: {counts}"
        assert counts["leduc_poker"] == 1000, f"leduc_poker changed: {counts}"
        assert counts["intercode"] == ic_floor, f"intercode not floored to {ic_floor}: {counts}"
        print(f"[ok] dynamic_floor_scales_with_max: {counts} (floor={ic_floor})")


# ---------------------------------------------------------------------------------------------
# Round-shape share table (>= 3 scored envs + --total_rows): shares of a fixed total, not absolute caps.
# ---------------------------------------------------------------------------------------------
R2_ENVS = ["clobber", "liars_dice", "goofspiel", "leduc_poker"]
BOSS_ENVS = ["goofspiel", "gin_rummy", "clobber", "leduc_poker", "intercode", "othello"]


def test_share_caps_r2_shape():
    caps = bc.share_caps(R2_ENVS, 18000)
    want = {"clobber": 7200, "liars_dice": 4500, "goofspiel": 4500, "leduc_poker": 1800}
    assert caps == want, f"R2 share caps {caps} != {want}"
    print(f"[ok] share_caps_r2_shape: {caps}")


def test_merge_share_r2_production_counts():
    # The production R2 generation kept 22,200 of these rows under the old absolute caps.
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        sizes = {"clobber": 31203, "liars_dice": 16804, "goofspiel": 24006, "leduc_poker": 16802}
        paths = [_make_env(out, e, n) for e, n in sizes.items()]
        gt.merge_datasets(paths, out, scored_envs=set(R2_ENVS), scored_cap=6000,
                          coverage_cap=gt.BALANCE_COVERAGE_CAP_PER_ENV, floor_frac=0.5, total_rows=18000)
        counts = _env_counts(out)
        want = {"clobber": 7200, "liars_dice": 4500, "goofspiel": 4500, "leduc_poker": 1800}
        assert counts == want, f"R2 merged counts {counts} != {want}"
        print(f"[ok] merge_share_r2_production_counts: {counts} total={sum(counts.values())}")


def test_share_six_env_underfilled_othello():
    total = 32000
    base = bc.share_caps(BOSS_ENVS, total)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "merged")
        sizes = {e: 20000 for e in BOSS_ENVS}
        sizes["othello"] = 2000  # generation-limited
        paths = [_make_env(out, e, n) for e, n in sizes.items()]
        gt.merge_datasets(paths, out, scored_envs=set(BOSS_ENVS), scored_cap=6000,
                          coverage_cap=gt.BALANCE_COVERAGE_CAP_PER_ENV, floor_frac=0.5, total_rows=total)
        counts = _env_counts(out)
    n = sum(counts.values())
    assert counts["othello"] == 2000, f"under-filled othello must keep all its rows: {counts}"
    assert abs(n - total) <= len(BOSS_ENVS), f"quota not redistributed (total {n}): {counts}"
    assert counts["leduc_poker"] >= 0.07 * total, f"leduc below its 7% floor: {counts}"
    assert counts["gin_rummy"] / n >= 0.20, f"gin below 20% share: {counts}"
    assert counts["clobber"] >= base["clobber"], f"clobber below its own target {base['clobber']}: {counts}"
    assert all(counts[e] >= base[e] for e in BOSS_ENVS if e != "othello"), f"an env lost quota: {counts}"
    print(f"[ok] share_six_env_underfilled_othello: {counts} total={n}")


def test_share_six_env_othello_cap_le_12k():
    for total in (32000, 54000):
        caps = bc.share_caps(BOSS_ENVS, total)
        assert caps["othello"] <= 12000, f"6-env othello cap {caps['othello']} > 12k at {total}"
        assert caps["leduc_poker"] >= max(0.07 * total, 1000) - 1, f"leduc floor broken: {caps}"
        assert abs(sum(caps.values()) - total) <= len(caps), f"caps do not sum to {total}: {caps}"
    print(f"[ok] share_six_env_othello_cap_le_12k: {bc.share_caps(BOSS_ENVS, 32000)}")


def test_share_leduc_row_floor_small_total():
    caps = bc.share_caps(R2_ENVS, 9000)  # 10% = 900 < the 1000-row floor
    assert caps["leduc_poker"] == 1000, f"leduc 1000-row floor not applied: {caps}"
    assert abs(sum(caps.values()) - 9000) <= len(caps), f"caps do not sum: {caps}"
    print(f"[ok] share_leduc_row_floor_small_total: {caps}")


def test_share_unknown_env_falls_back():
    caps = bc.share_caps(["clobber", "mystery_env", "leduc_poker"], 10000)
    assert set(caps) == {"clobber", "mystery_env", "leduc_poker"}, caps
    assert abs(sum(caps.values()) - 10000) <= 3, f"caps do not sum: {caps}"
    assert caps["mystery_env"] > 0 and caps["clobber"] > caps["mystery_env"], caps
    print(f"[ok] share_unknown_env_falls_back: {caps}")


def test_r1_path_unchanged_by_total_rows():
    # R1 [intercode, game] keeps the 13759ee composition (20k floor per env, othello's 36k flood cap),
    # whether or not a total is passed: the share table never applies to a 2-env round.
    os.environ["R1_MIN_ENV_ROWS"] = "20000"
    try:
        for game, n_game, want_game in (("clobber", 31203, 20000), ("othello", 50000, 36000)):
            for total in (None, 40000):
                with tempfile.TemporaryDirectory() as tmp:
                    out = os.path.join(tmp, "merged")
                    paths = [_make_env(out, "intercode", 20000), _make_env(out, game, n_game)]
                    gt.merge_datasets(paths, out, scored_envs={"intercode", game}, scored_cap=6000,
                                      coverage_cap=gt.BALANCE_COVERAGE_CAP_PER_ENV, floor_frac=0.5,
                                      total_rows=total)
                    counts = _env_counts(out)
                want = {"intercode": 20000, game: want_game}
                assert counts == want, f"R1 [intercode, {game}] total_rows={total}: {counts} != {want}"
        print("[ok] r1_path_unchanged_by_total_rows: clobber 20000/20000, othello 36000/20000")
    finally:
        os.environ.pop("R1_MIN_ENV_ROWS", None)


def test_intercode_budget_kept_in_share_merge():
    # A dosed intercode budget above its share quota is kept whole; the game caps do not move.
    total = 32000
    base = bc.share_caps(BOSS_ENVS, total)
    budget = base["intercode"] + 2500
    os.environ["INTERCODE_ROW_BUDGET"] = str(budget)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "merged")
            sizes = {e: 20000 for e in BOSS_ENVS}
            sizes["intercode"] = budget
            paths = [_make_env(out, e, n) for e, n in sizes.items()]
            gt.merge_datasets(paths, out, scored_envs=set(BOSS_ENVS), scored_cap=6000,
                              coverage_cap=gt.BALANCE_COVERAGE_CAP_PER_ENV, floor_frac=0.5, total_rows=total)
            counts = _env_counts(out)
    finally:
        os.environ.pop("INTERCODE_ROW_BUDGET", None)
    assert counts["intercode"] == budget, f"intercode budget {budget} not kept: {counts}"
    games = {e: counts[e] for e in BOSS_ENVS if e != "intercode"}
    assert games == {e: base[e] for e in games}, f"game caps moved: {games} vs {base}"
    print(f"[ok] intercode_budget_kept_in_share_merge: intercode {budget}, games {games}")


def test_unused_slice_rolls_over():
    # Games generate one after another; one that stops early (row target reached) hands the rest of its
    # time slice to the games after it. Share round [clobber, liars, goofspiel], 90 s, weights 40/25/25.
    limits = []

    def fake_generate(env_name, env_path, **kw):
        limits.append((env_name, kw.get("time_limit_seconds")))
        DatasetDict({"train": Dataset.from_dict({"env": [env_name] * 10, "x": list(range(10))})}).save_to_disk(env_path)

    saved_gen, saved_argv = gt.generate_for_env, sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        gt.generate_for_env = fake_generate
        sys.argv = ["generate_trajectories", "--environment_names", "clobber", "liars_dice", "goofspiel",
                    "--output_path", os.path.join(tmp, "out"), "--time_limit_seconds", "90", "--total_rows", "9000"]
        try:
            gt.main()
        finally:
            gt.generate_for_env, sys.argv = saved_gen, saved_argv
    got = dict(limits)
    # Time weight = share cap x _GEN_TIME_WEIGHT (clobber's is gt._gtw("clobber"), the others 1.0). Clobber stops
    # at once and hands its slice on: liars gets its own share plus half of clobber's (= 45 s whatever the
    # clobber weight), goofspiel the rest.
    g = gt._gtw("clobber")
    want = {"clobber": 90.0 * 40 * g / (40 * g + 50), "liars_dice": 45.0, "goofspiel": 90.0}
    assert all(abs(got[e] - want[e]) < 1.0 for e in want), f"slices {got} != ~{want}"
    print(f"[ok] unused_slice_rolls_over: {got}")


def test_share_row_targets():
    assert bc.share_row_target("othello", 6000, 4.0) == 7500
    assert bc.share_row_target("gin_rummy", 6000, 4.0) == 7500
    assert bc.share_row_target("goofspiel", 4500, 4.0) == 6750
    assert bc.share_row_target("leduc_poker", 1800, 4.0) == 3600
    assert bc.share_row_target("clobber", 7200, 4.0) == 28800
    print("[ok] share_row_targets: othello/gin x1.25, goofspiel x1.5, leduc x2, others x GEN_ROW_TARGET_MULT")


def test_concat_aligned_mixes_toolcall_and_plain():
    """_concat_aligned must merge a tool-call source (games/intercode shape) with a plain
    swe_infinite source (no per-message tool_calls, no tools column) WITHOUT raising
    'features can't be aligned' — the whole reason swe_infinite integration needed a
    schema-aware concat. It normalizes rows and rebuilds via Dataset.from_list."""
    tool_ds = Dataset.from_list([{
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"type": "function", "function": {"name": "game_action", "arguments": "{}"}}]},
        ],
        "tools": "[]",
    }] * 3)
    plain_ds = Dataset.from_list([{
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "THOUGHT: x\n```bash\nls\n```"},
        ],
    }] * 2)  # NO tools column; messages carry NO tool_calls
    assert tool_ds.features != plain_ds.features, "precondition: the schemas must really diverge"
    merged = gt._concat_aligned([tool_ds, plain_ds])
    assert len(merged) == 5, f"expected 5 rows, got {len(merged)}"
    assert "messages" in merged.column_names and "tools" in merged.column_names
    print(f"[ok] concat_aligned_mixes_toolcall_and_plain: {len(merged)} rows unified")


def test_concat_aligned_fast_path_when_identical():
    """No swe_infinite in the mix (identical schemas) → the fast concatenate_datasets path."""
    row = {"messages": [{"role": "user", "content": "x"}], "tools": "[]"}
    merged = gt._concat_aligned([Dataset.from_list([row] * 2), Dataset.from_list([row] * 3)])
    assert len(merged) == 5
    print("[ok] concat_aligned_fast_path_when_identical")


if __name__ == "__main__":
    test_cap_floor_rebalance()
    test_floor_none_is_noop()
    test_coverage_never_floored()
    test_dynamic_floor_scales_with_max()
    test_share_caps_r2_shape()
    test_merge_share_r2_production_counts()
    test_share_six_env_underfilled_othello()
    test_share_six_env_othello_cap_le_12k()
    test_share_leduc_row_floor_small_total()
    test_share_unknown_env_falls_back()
    test_r1_path_unchanged_by_total_rows()
    test_intercode_budget_kept_in_share_merge()
    test_share_row_targets()
    test_unused_slice_rolls_over()
    test_concat_aligned_mixes_toolcall_and_plain()
    test_concat_aligned_fast_path_when_identical()
    print("\nALL MERGE-BALANCE TESTS PASSED")
