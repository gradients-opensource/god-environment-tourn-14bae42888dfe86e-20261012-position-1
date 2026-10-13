"""CPU test: the scored-env generation-time split fills every env's cap without relying on rollover.

generate_trajectories.main() gives each scored game env the slice
    scored_budget * w(env) / sum(w),   w(env) = _gtw(env) * share_cap(env)   (share rounds)
after the leduc blueprint pre-warm is taken out of --time_limit_seconds. A game that reaches its row target
early passes the rest of its slice on (rollover), but a slow box can leave nothing to pass on, so the BASE
slice alone must fill each cap. Rates are the rows/s measured on 8 WSL cores in the 2026-09-30 generation
smoke (clobber 72.5, gin 31.6, othello 45.4; cold leduc pre-warm 92 s), and clobber's slice must be >= 120 s.

Run from scripts/envs/:  python test_gen_time_slices.py
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Same import shims as test_merge_balance.py: only the pure time-split helpers are exercised.
sys.modules["trl_compat"] = types.ModuleType("trl_compat")
_envs_pkg = types.ModuleType("envs")
_envs_pkg.__path__ = [_HERE]
sys.modules["envs"] = _envs_pkg
_shared = types.ModuleType("envs.shared_env")
_shared.GAMES_TO_TASK_ID_RANGE = {}
_shared.init_env_pool = lambda *a, **k: None
_shared._log = lambda *a, **k: None
_shared.recipe_log = lambda *a, **k: None
_shared.generation_id_bounds = lambda env_name: (0, 1)  # unused by the time-split helpers under test
sys.modules["envs.shared_env"] = _shared
_cfg = types.ModuleType("envs.sft_env_configs")
_cfg._OFFLINE_ENVS = {"intercode"}
_cfg._SELFPLAY_ENVS = set()
_cfg.get_sft_trajectory_generator = lambda *a, **k: None
sys.modules["envs.sft_env_configs"] = _cfg

import envs.generate_trajectories as gt  # noqa: E402

FAILS = []

MEASURED_ROWS_PER_S = {"clobber": 72.5, "gin_rummy": 31.6, "othello": 45.4,
                       "liars_dice": 772.0, "goofspiel": 750.0, "leduc_poker": 883.0}
LEDUC_PREWARM_S = 92.0
CLOBBER_MIN_SLICE_S = 120.0

R1 = ["intercode", "clobber"]
R2 = ["clobber", "liars_dice", "goofspiel", "leduc_poker"]
BOSS = ["goofspiel", "gin_rummy", "clobber", "leduc_poker", "intercode", "othello"]


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def base_slices(envs, limit_s, total_rows):
    """Mirror of main()'s scored split (no coverage envs, no rollover)."""
    scored = [e for e in envs if e not in gt._OFFLINE_ENVS]
    budget = max(60.0, limit_s - (LEDUC_PREWARM_S if "leduc_poker" in scored else 0.0))
    share = (gt.share_caps(envs, total_rows)
             if total_rows and gt.is_share_round(envs) else None)
    tw = ({e: float(share.get(e, 0)) for e in scored} if share is not None else gt._env_weights(scored))
    w = {e: gt._gtw(e) * gt._env_weight(e, tw) for e in scored}
    total_w = sum(w.values()) or 1.0
    return share, {e: budget * w[e] / total_w for e in scored}


def test_boss_slices_fill_caps() -> None:
    for limit, total in ((810, 32000), (1620, 54000)):
        share, sl = base_slices(BOSS, limit, total)
        tag = f"boss 6-env {limit}s/{total} rows"
        check(f"{tag}: clobber base slice {sl['clobber']:.0f}s >= {CLOBBER_MIN_SLICE_S:.0f}s",
              sl["clobber"] >= CLOBBER_MIN_SLICE_S)
        for env in ("clobber", "gin_rummy", "othello", "goofspiel", "leduc_poker"):
            need = share[env] / MEASURED_ROWS_PER_S[env]
            check(f"{tag}: {env} base slice {sl[env]:.0f}s fills its cap {share[env]} (needs {need:.0f}s)",
                  sl[env] >= need)


def test_r2_slices_fill_caps() -> None:
    share, sl = base_slices(R2, 810, 18000)
    for env in R2:
        need = share[env] / MEASURED_ROWS_PER_S[env]
        check(f"R2: {env} base slice {sl[env]:.0f}s fills its cap {share[env]} (needs {need:.0f}s)",
              sl[env] >= need)


def test_r1_clobber_gets_the_whole_budget() -> None:
    # R1 = intercode (offline, not time-boxed) + ONE game: that game's weight cancels out, so the R1 split
    # is the same as at 13759ee whatever its _GEN_TIME_WEIGHT entry.
    _, sl = base_slices(R1, 810, None)
    check(f"R1 [intercode, clobber]: clobber slice {sl['clobber']:.0f}s == the whole 810s",
          abs(sl["clobber"] - 810.0) < 1e-9)


if __name__ == "__main__":
    for t in (test_boss_slices_fill_caps, test_r2_slices_fill_caps, test_r1_clobber_gets_the_whole_budget):
        t()
    print(f"\n{'RESULT PASS' if not FAILS else 'RESULT FAIL: ' + str(len(FAILS)) + ' failure(s)'}")
    sys.exit(1 if FAILS else 0)
