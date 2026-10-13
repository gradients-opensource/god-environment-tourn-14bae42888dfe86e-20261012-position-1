"""CPU tests for the round-shape recipe: epoch bounds, the clock-stop planner, the total-row shape table and
the generation-time ratio.

Run from scripts/ (no torch / GPU needed; model_utility is shimmed):
    python test_epoch_planner.py
"""
import ast
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []

R1_IC = ["intercode", "clobber"]
R1_OTHELLO = ["intercode", "othello"]
R2 = ["clobber", "liars_dice", "goofspiel", "leduc_poker"]
BOSS = ["goofspiel", "gin_rummy", "clobber", "leduc_poker", "intercode", "othello"]


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def _shim_model_utility(params: float = 8.03e9, arch: str = "LlamaForCausalLM", gpus: int = 2) -> None:
    mu = types.ModuleType("model_utility")
    mu.get_gpu_count = lambda: gpus
    mu.get_model_architecture = lambda p: arch
    mu.get_model_num_params = lambda p: int(params)
    mu.disable_flash_attention = lambda a: False
    mu.get_use_liger = lambda a: True
    sys.modules["model_utility"] = mu


def _load_function(path: str, name: str):
    """Exec one top-level function out of a module that imports torch (train_sft_env) so its pure math
    can be checked on a CPU box."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns[name]


def test_epoch_bounds() -> None:
    import budget_controller as bc
    os.environ.pop("SFT_EPOCHS", None)
    check("R1 [intercode, clobber] -> 1 epoch", bc.recommend_epochs(1.5, R1_IC) == 1)
    check("R1 [intercode, othello] (flood) -> 1 epoch", bc.recommend_epochs(1.5, R1_OTHELLO) == 1)
    check("R1 epochs stay an int (run_cmd '--num_train_epochs 1')", isinstance(bc.recommend_epochs(1.5, R1_IC), int))
    check("R2 4-env 1.5h -> bounds (1.5, 2.5)", bc.plan_epoch_bounds(1.5, R2) == (1.5, 2.5))
    check("R2 4-env 1.5h -> hi 2.5 epochs", bc.recommend_epochs(1.5, R2) == 2.5)
    check("6-env 1.5h with othello -> hi 2.5 (othello no longer forces 1)", bc.recommend_epochs(1.5, BOSS) == 2.5)
    check("6-env 3h -> bounds (2, 3)", bc.plan_epoch_bounds(3.0, BOSS) == (2, 3))
    check("6-env 3h -> hi 3 epochs", bc.recommend_epochs(3.0, BOSS) == 3)
    check("4-env othello+intercode without a game weapon is a share round, not a flood",
          bc.recommend_epochs(1.5, ["othello", "intercode", "liars_dice", "leduc_poker"]) == 2.5)
    os.environ["SFT_EPOCHS"] = "4"
    try:
        check("SFT_EPOCHS overrides every shape", bc.plan_epoch_bounds(1.5, BOSS) == (4, 4))
    finally:
        os.environ.pop("SFT_EPOCHS", None)
    saved = bc.TIME_AWARE_EPOCHS
    bc.TIME_AWARE_EPOCHS = False
    try:
        check("rollback constant restores 13759ee: othello round -> 1", bc.recommend_epochs(1.5, BOSS) == 1)
        check("rollback constant restores 13759ee: R2 -> 2", bc.recommend_epochs(1.5, R2) == 2)
    finally:
        bc.TIME_AWARE_EPOCHS = saved
    plan = bc.plan_training(1.5, R2, ["clobber", "liars_dice", "goofspiel", "leduc_poker"], read_gpu=False)
    check(f"plan_training carries the bounds ({plan.epochs}, {plan.min_epochs})", plan.epochs == 2.5 and plan.min_epochs == 1.5)


def test_stop_step_math() -> None:
    import budget_controller as bc
    # R1 production: 811 planned steps at 4.40 s/step in a 79.4-min SFT window -> fits, capped at 811.
    r1 = bc.plan_stop_step(811, 20, 4.40, 79.4 * 60 - 20 * 4.40)
    check(f"R1 numbers -> stop capped at 811 ({r1})", r1 == 811)
    # R2 production: 1292 planned (2 epochs x 646) at 4.35 s/step, 4677 s SFT window -> ~1055 (1.63 epochs).
    r2 = bc.plan_stop_step(1292, 20, 4.35, 4677 - 20 * 4.35)
    check(f"R2 numbers -> stop ~1055 ({r2})", 1045 <= r2 <= 1060)
    check("no timing -> keep the planned total", bc.plan_stop_step(500, 20, 0.0, 1000) == 500)
    check("deadline already inside the margin -> stop right after the probe",
          bc.plan_stop_step(500, 20, 4.0, 30) == 21)


def test_clock_stop_fake_clock() -> None:
    import budget_controller as bc
    for label, planned, s_step, window, lo in (("R1", 811, 4.40, 79.4 * 60, 811), ("R2", 1292, 4.35, 4677, 1045)):
        planner = bc.ClockStop(total_steps=planned)
        stop = None
        for step in range(1, 25):
            now = step * s_step
            got = planner.observe(step, now, window - now)
            if got is not None:
                check(f"{label}: planner decides once, at the probe end (step {step})", step == bc.CLOCK_STOP_PROBE[1])
                stop = got
        check(f"{label}: fake-clock stop {stop} (s/step {planner.s_per_step:.2f})",
              stop is not None and lo <= stop <= min(planned, 1060) and abs(planner.s_per_step - s_step) < 1e-6)
    m = _load_function(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_sft_env.py"),
                       "clock_cosine_multiplier")
    stop = 1054
    check("clock_cosine reaches min_lr exactly at the moved horizon (stop_step)",
          abs(m(stop, stop, 35, 0.0, 0.25) - 0.25) < 1e-12 and m(stop - 50, stop, 35, 0.0, 0.25) > 0.25)


def test_total_rows_table() -> None:
    import budget_controller as bc
    check("R1 has no total (absolute 20k floor per env)", bc.round_total_rows(R1_IC, 1.5, 2) is None)
    check("R2 4-env 1.5h on 2 GPUs -> 18000", bc.round_total_rows(R2, 1.5, 2) == 18000)
    check("6-env 1.5h on 4 GPUs -> 32000", bc.round_total_rows(BOSS, 1.5, 4) == 32000)
    check("6-env 3h on 4 GPUs -> 54000", bc.round_total_rows(BOSS, 3.0, 4) == 54000)
    check("6-env on 2 GPUs (fewer than the validator assigns) -> scaled to 16000",
          bc.round_total_rows(BOSS, 1.5, 2) == 16000)
    check("unknown GPU count keeps the shape total", bc.round_total_rows(BOSS, 1.5, 0) == 32000)


def test_generation_ratio() -> None:
    _shim_model_utility()
    import sft_env_config as S
    check("3.2B at 3.0h -> generation ratio capped at 0.25",
          S.get_generation_time_ratio(3_200_000_000, BOSS, 3.0) == 0.25)
    check("3.2B at 1.5h -> 0.40 unchanged", S.get_generation_time_ratio(3_200_000_000, BOSS, 1.5) == 0.40)
    check("8B at 1.5h -> 0.15 unchanged", S.get_generation_time_ratio(8_030_000_000, R1_IC, 1.5) == 0.15)
    check("hours omitted -> old ratio", S.get_generation_time_ratio(3_200_000_000) == 0.40)
    # R1 (intercode + one game) is not a share round: the long-task cap must not touch it (13759ee parity).
    check("3.2B R1 [intercode, clobber] at 3.0h -> 0.40 unchanged (no long-task cap in R1)",
          S.get_generation_time_ratio(3_200_000_000, R1_IC, 3.0) == 0.40)
    check("1.2B R1 [intercode, othello] at 2.5h -> size ratio unchanged",
          S.get_generation_time_ratio(1_235_814_400, R1_OTHELLO, 2.5)
          == S._size_generation_ratio(1_235_814_400, R1_OTHELLO))


def test_training_json_shapes() -> None:
    _shim_model_utility()
    import sft_env_config as S

    def tj(envs, hours=1.5):
        return S.get_training_json({
            "task_id": "T1", "model_path": "/nonexistent", "model_name": "m", "hours_to_complete": hours,
            "reg_ratio": 1.0, "output_dir": "/o", "request_path": "/r.json",
            "dataset_type": {"environment_names": envs}})

    r2 = tj(R2)
    check("R2 generate_cmd carries --total_rows 18000", " --total_rows 18000" in r2["generate_cmd"])
    check("R2 run_cmd trains up to 2.5 epochs", "--num_train_epochs 2.5 " in r2["run_cmd"])
    check("R2 train_request carries the 1.5-epoch floor", r2["train_request"].get("min_epochs") == 1.5)
    r1 = tj(R1_IC)
    check("R1 generate_cmd has no --total_rows", "--total_rows" not in r1["generate_cmd"])
    check("R1 train_request floor = 1 epoch", r1["train_request"].get("min_epochs") == 1)
    # 13759ee R1 generate_cmd on a 3.2B base at 3.0h: 3h x 0.40 = 4320 s (the long-task cap is share-only).
    real_params = S.get_model_num_params
    S.get_model_num_params = lambda p: 3_212_749_824
    try:
        r1_3b = tj(R1_IC, 3.0)
    finally:
        S.get_model_num_params = real_params
    check("R1 3.2B at 3.0h keeps --time_limit_seconds 4320 (13759ee)",
          " --time_limit_seconds 4320" in r1_3b["generate_cmd"])


if __name__ == "__main__":
    for t in (test_epoch_bounds, test_stop_step_math, test_clock_stop_fake_clock, test_total_rows_table,
              test_generation_ratio, test_training_json_shapes):
        t()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL(S)'}")
    sys.exit(1 if FAILS else 0)
