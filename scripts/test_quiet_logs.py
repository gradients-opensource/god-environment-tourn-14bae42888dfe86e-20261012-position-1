"""CPU test: the planning/generation entrypoints must not leak the recipe to stdout.

Our training container's stdout reaches the public training Loki. Two entrypoints used to
print the WHOLE recipe there: sft_env_config.get_training_json printed the full run command
(every --learning_rate / --lora_r / ... flag) and intercode_dataset.build_intercode_sft_dataset
printed its row budgets / self-gen counts unconditionally. Both now route through
envs.shared_env.recipe_log, which is QUIET by default (full detail -> a local recipe.log file,
a short coded tag -> stdout).

Run from scripts/:  python test_quiet_logs.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FAILS = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


# --- Shim the heavy (torch/transformers) model_utility module BEFORE anything imports it ------
_mu = types.ModuleType("model_utility")
_mu.get_gpu_count = lambda: 1
_mu.get_model_architecture = lambda p: "LlamaForCausalLM"
_mu.get_model_num_params = lambda p: int(8.03e9)
_mu.disable_flash_attention = lambda a: False
_mu.get_use_liger = lambda a: True
sys.modules["model_utility"] = _mu

# Recipe-revealing tokens that must never reach stdout (P_13: row targets, LR/LoRA flags,
# teacher modes, INTERCODE_* env prefixes).
_BANNED_TOKENS = ("--learning_rate", "--lora_r", "INTERCODE_")


def test_get_training_json_run_command_is_quiet():
    import sft_env_config as sc

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SFT_RECIPE_LOG_DIR"] = tmp
        os.environ.pop("SFT_QUIET_RECIPE_LOGS", None)  # exercise the default (ON)
        train_info = {
            "model_path": "/fake/model",
            "task_id": "unit-test-task",
            "dataset_type": {"environment_names": ["liars_dice"]},
            "hours_to_complete": 1.5,
            "output_dir": os.path.join(tmp, "out"),
            "request_path": os.path.join(tmp, "request.json"),
            "reg_ratio": 1.0,
        }
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                result = sc.get_training_json(train_info)
        finally:
            os.environ.pop("SFT_RECIPE_LOG_DIR", None)

        stdout_text = buf.getvalue()
        check(
            "get_training_json: stdout carries no banned recipe token",
            not any(tok in stdout_text for tok in _BANNED_TOKENS),
        )
        check(
            "get_training_json: stdout gets the short coded tag instead",
            "[sft-config] run command ready" in stdout_text,
        )
        recipe_path = os.path.join(tmp, "recipe.log")
        check("get_training_json: local recipe.log was written", os.path.exists(recipe_path))
        if os.path.exists(recipe_path):
            full_detail = open(recipe_path).read()
            check(
                "get_training_json: full run command (with --learning_rate) IS in the local file",
                "--learning_rate" in full_detail and "Run command:" in full_detail,
            )
        check("get_training_json: run_cmd still built (logging change only)", bool(result.get("run_cmd")))
        check(
            "get_training_json: run_cmd itself still carries --learning_rate (R1 behaviour unchanged)",
            "--learning_rate" in result.get("run_cmd", ""),
        )


def test_build_intercode_sft_dataset_is_quiet():
    import envs.intercode_dataset as idd

    with tempfile.TemporaryDirectory() as tmp:
        # Fabricate a tiny whitelisted-passthrough fixture so the function actually
        # exercises its row-conversion + budget path (an empty/missing dataset would
        # short-circuit before reaching the quieted print sites).
        ds_dir = os.path.join(tmp, "miner", "gradients-io-tournaments--intercode_bigcode_combined_12k")
        os.makedirs(ds_dir, exist_ok=True)
        rows = [
            {"instruction": "list files in the current directory", "response": "ls -la"},
            {"instruction": "print the current working directory", "response": "pwd"},
            {"instruction": "show disk usage of the home dir", "response": "du -sh ~"},
        ]
        with open(os.path.join(ds_dir, "data.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        os.environ["SFT_RECIPE_LOG_DIR"] = tmp
        os.environ["MINER_DATASETS_DIR"] = os.path.join(tmp, "miner")
        os.environ["MINER_DATASETS"] = "gradients-io-tournaments--intercode_bigcode_combined_12k"
        os.environ.pop("INTERCODE_SELFGEN", None)
        os.environ.pop("SFT_QUIET_RECIPE_LOGS", None)  # exercise the default (ON)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                dd = idd.build_intercode_sft_dataset()
        finally:
            for k in ("SFT_RECIPE_LOG_DIR", "MINER_DATASETS_DIR", "MINER_DATASETS"):
                os.environ.pop(k, None)

        stdout_text = buf.getvalue()
        check("build_intercode_sft_dataset: dataset actually built (fixture is live)", dd is not None)
        check(
            "build_intercode_sft_dataset: stdout carries no banned recipe token",
            not any(tok in stdout_text for tok in _BANNED_TOKENS),
        )
        check(
            "build_intercode_sft_dataset: stdout has the short coded tags instead",
            "[intercode] passthrough converted" in stdout_text and "[intercode] dataset built" in stdout_text,
        )
        recipe_path = os.path.join(tmp, "recipe.log")
        check("build_intercode_sft_dataset: local recipe.log was written", os.path.exists(recipe_path))
        if os.path.exists(recipe_path):
            full_detail = open(recipe_path).read()
            check(
                "build_intercode_sft_dataset: full counts ARE in the local file",
                "Converted" in full_detail and "Total" in full_detail and "step-examples" in full_detail,
            )


def test_quiet_flag_opt_out_restores_old_stdout_behaviour():
    # QUIET_RECIPE_LOGS is read once at import time (same style as the module's other
    # env-backed constants, e.g. _VERBOSE / _PASSTHROUGH_FILTER) — a real opt-out happens
    # via SFT_QUIET_RECIPE_LOGS=0 in a fresh process. Flip the already-imported module's
    # flag directly to exercise that branch here without a subprocess.
    import envs.shared_env as shared_env

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SFT_RECIPE_LOG_DIR"] = tmp
        saved = shared_env.QUIET_RECIPE_LOGS
        shared_env.QUIET_RECIPE_LOGS = False
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                shared_env.recipe_log("--learning_rate 4.3e-05 full detail line", tag="[x] short tag")
        finally:
            shared_env.QUIET_RECIPE_LOGS = saved
            os.environ.pop("SFT_RECIPE_LOG_DIR", None)
        check(
            "recipe_log: QUIET_RECIPE_LOGS=False prints full detail on stdout (opt-out works)",
            "--learning_rate 4.3e-05 full detail line" in buf.getvalue(),
        )


def test_errors_and_tracebacks_stay_on_real_print():
    # recipe_log is only for recipe-revealing lines. The genuine failure/absence markers in
    # intercode_dataset.py / generate_trajectories.py (e.g. "not found", "Failed to load",
    # exception reprs) must stay on a real print() so they are never silenced — verified here
    # as a source-level guard against accidentally routing an error line through recipe_log.
    import inspect

    import envs.intercode_dataset as idd

    src = inspect.getsource(idd)
    check(
        "intercode_dataset: the 'not found' dataset warning still uses a real print()",
        "not found in miner dataset inventory" in src and 'print(\n            f"[intercode_dataset] \'{INTERCODE_HF_REPO}\'' in src,
    )
    check(
        "intercode_dataset: 'Failed to load dataset' still uses a real print()",
        'print(f"[intercode_dataset] Failed to load dataset from {local}"' in src,
    )


if __name__ == "__main__":
    test_get_training_json_run_command_is_quiet()
    test_build_intercode_sft_dataset_is_quiet()
    test_quiet_flag_opt_out_restores_old_stdout_behaviour()
    test_errors_and_tracebacks_stay_on_real_print()
    if FAILS:
        print(f"\n{len(FAILS)} FAILURE(S): {FAILS}")
        sys.exit(1)
    print("\nALL QUIET LOGS TESTS PASSED")
