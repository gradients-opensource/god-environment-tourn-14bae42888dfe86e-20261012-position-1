"""Fault-injection tests for the EnvTask failure path: insurance publish, rescue LoRA, adapter-only upload.

Run from scripts/:  python test_submission_insurance.py
CPU only. The trainer callbacks run inside a scripted step loop (transformers / torch / wandb / redis are
stubbed when the image does not ship them), the rescue gets fake runners, and text_trainer.py /
train_sft_env.py are checked at source level because they import the GPU stack at module load.
"""
import ast
import datetime
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []
FAR_END = "2099-01-01 00:00:00"
BASE_CMD = (
    "torchrun --nproc_per_node=2 train_sft_env.py \\\n"
    "    --request_path datasets/training_request_t_0.json \\\n"
    "    --bf16 True \\\n"
    "    --output_dir /workspace/scripts/soutputs/t_0 \\\n"
    "    --num_train_epochs 1 \\\n"
    "    --per_device_train_batch_size 32 \\\n"
    "    --learning_rate 3e-05 \\\n"
    "    --max_length 4096 --use_lora True --lora_r 128 --lora_alpha 512 --lora_dropout 0.05 --padding_free True"
)


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------------

def _install_stubs() -> None:
    """Minimal stand-ins for the GPU-image modules customized_trainer imports, only when they are absent."""
    def missing(name: str) -> bool:
        try:
            __import__(name)
            return False
        except ImportError:
            return True

    if missing("transformers"):
        tf = types.ModuleType("transformers")
        tu = types.ModuleType("transformers.trainer_utils")

        class TrainerState:
            def __init__(self):
                self.global_step, self.epoch, self.log_history = 0, 0.0, []

        class TrainerControl:
            def __init__(self):
                self.should_save = self.should_evaluate = self.should_training_stop = False

        tf.GenerationConfig, tf.TrainerCallback = object, object
        tf.TrainerState, tf.TrainerControl = TrainerState, TrainerControl
        tu.is_main_process = lambda local_rank: local_rank in (-1, 0)
        tf.trainer_utils = tu
        sys.modules.update({"transformers": tf, "transformers.trainer_utils": tu})
    if missing("torch"):
        torch = types.ModuleType("torch")
        dist = types.ModuleType("torch.distributed")
        dist.is_available = lambda: False
        torch.distributed = dist
        sys.modules.update({"torch": torch, "torch.distributed": dist})
    if missing("wandb"):
        sys.modules["wandb"] = types.ModuleType("wandb")
    if missing("redis"):
        redis = types.ModuleType("redis")
        redis.Redis = object  # state_manager annotates with it; the callbacks under test never open Redis
        sys.modules["redis"] = redis


def _write_adapter(path: str, tag: bytes = b"") -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": 128, "lora_alpha": 512}, f)
    with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
        f.write(b"\x01" * 64 + tag)


def _write_full(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "llama"}, f)
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {"a": shards[0], "b": shards[1]}}, f)
    for shard in shards:
        with open(os.path.join(path, shard), "wb") as f:
            f.write(b"\x02" * 64)


def _write_dataset(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "dataset_dict.json"), "w") as f:
        json.dump({"splits": ["train"]}, f)


def _snapshot(path: str) -> dict:
    out = {}
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                out[os.path.relpath(full, path)] = f.read()
    return out


def _cli_value(cmd: str, name: str):
    match = re.search(rf"--{name}\s+(\S+)", cmd)
    return match.group(1) if match else None


def _layout():
    """(root, text_trainer output_dir, its first run dir, submission dir, ds_folder) in a fresh temp dir."""
    root = tempfile.mkdtemp()
    output_dir = os.path.join(root, "soutputs", "t")
    submission = os.path.join(root, "ckpt", "t", "repo")
    ds_folder = os.path.join(root, "datasets")
    os.makedirs(submission)
    os.makedirs(ds_folder)
    return root, output_dir, output_dir + "_0", submission, ds_folder


def _train_info(submission: str, dataset_path: str, end_time: str) -> dict:
    # checking_mode is "first_time" here, matching production: text_trainer.py's flat train_info always
    # starts "first_time" (its GRPO/EnvTask -> "none" override only ever lands on a *separate* deepcopy,
    # c_train_info, never on train_info itself), and get_env_training_json's train_request is a deepcopy of
    # this same train_info. rescue_lora must override it back to "none" itself (see test below) or the
    # rescue run's LR-continuation probe (checking_step=70) fires and truncates it before RESCUE_MAX_STEPS.
    return {
        "train_request": {"task_id": "t", "end_time": end_time, "dataset_path": dataset_path,
                          "submission_dir": submission, "max_steps": -1, "save_before_remaining_time": 5,
                          "periodic_save_steps": 200, "checking_mode": "first_time", "checking_step": 70},
        "run_cmd": BASE_CMD,
        "generate_cmd": "python -m envs.generate_trajectories --environment_names intercode clobber",
    }


class FakeRunners:
    """Records every command; a training run writes adapter checkpoints 50 and 100 into its --output_dir unless
    its dataset is listed in ``barren``; a dataset build creates the --output_path dataset."""

    def __init__(self, barren=()):
        self.trained, self.commands, self.barren, self.retries = [], [], set(barren), []

    def run_training(self, cmd: str, log_path: str, retries: int) -> bool:
        self.trained.append(cmd)
        self.retries.append(retries)
        with open(_cli_value(cmd, "request_path")) as f:
            dataset = json.load(f)["train_request"]["dataset_path"]
        if dataset in self.barren:
            return False
        for step in (50, 100):
            _write_adapter(os.path.join(_cli_value(cmd, "output_dir"), f"checkpoint-{step}"), str(step).encode())
        return True

    def run_cmd(self, cmd: str, log_path: str) -> None:
        self.commands.append(cmd)
        if "envs.intercode_dataset" in cmd:
            _write_dataset(_cli_value(cmd, "output_path"))


def _rescue(train_info, output_dir, ds_folder, runners):
    from env_submission import rescue_lora
    return lambda: rescue_lora(train_info, BASE_CMD, output_dir, ds_folder,
                               run_training=runners.run_training, run_cmd=runners.run_cmd)


def _scripted_sft(run_dir: str, submission: str, planned: int, crash_at: int, insurance_step) -> None:
    """Drive CustomEvalSaveCallback + WhenToEvalHandler through steps 1..crash_at-1 the way the HF loop does,
    writing an adapter checkpoint whenever a save is requested; the process then dies at ``crash_at``."""
    _install_stubs()
    import customized_trainer as CT
    from transformers import TrainerControl, TrainerState
    handler = CT.WhenToEvalHandler(FAR_END, 5, periodic_save_steps=200, steps_per_epoch=planned, max_steps=-1,
                                   insurance_step=insurance_step)
    callback = CT.CustomEvalSaveCallback(handler, submission, run_dir, "base", end_time=FAR_END,
                                         publish_policy="latest")
    state = TrainerState()
    for step in range(1, crash_at):
        state.global_step, state.epoch, state.log_history = step, step / planned, [{"loss": 0.1}]
        control = TrainerControl()
        callback.on_step_end(None, state, control)
        if control.should_evaluate:
            callback.on_evaluate(None, state, control, {"eval_loss": 0.1})
        if control.should_save:
            _write_adapter(os.path.join(run_dir, f"checkpoint-{step}"), str(step).encode())
            callback.on_save(None, state, control)


def _loss_line(submission: str) -> str:
    try:
        with open(os.path.join(submission, "loss.txt")) as f:
            return f.read()
    except OSError:
        return ""


# ---------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------

def test_insurance_step() -> None:
    from submission_utils import insurance_save_step
    check("insurance step: R1 shape 811 planned steps -> 162", insurance_save_step(811) == 162)
    check("insurance step: R2 shape 1292 planned steps -> 258", insurance_save_step(1292) == 258)
    check("insurance step: float step count (num_train_epochs is a float) -> 162", insurance_save_step(811.0) == 162)
    check("insurance step: short run -> floor of 50", insurance_save_step(120) == 50)
    check("insurance step: run no longer than the floor -> none (its own final save comes first)",
          insurance_save_step(50) is None and insurance_save_step(30) is None)
    check("insurance step: unknown step count -> none",
          all(insurance_save_step(v) is None for v in (0, -1, None)))


def test_handler_schedule() -> None:
    _install_stubs()
    import customized_trainer as CT
    handler = CT.WhenToEvalHandler(FAR_END, 5, periodic_save_steps=200, steps_per_epoch=811, max_steps=-1,
                                   insurance_step=162)
    check("handler: insurance step requests a save", handler(162) == {"eval": True, "reason": "insurance"})
    check("handler: other steps keep their schedule", not handler(161)["eval"] and handler(200)["reason"] == "periodic")
    legacy = CT.WhenToEvalHandler(FAR_END, 5, periodic_save_steps=200, steps_per_epoch=811, max_steps=-1)
    check("handler: no insurance step -> first save stays at the periodic step", not legacy(162)["eval"])


def _load_clock_cosine():
    """ClockCosineCallback + _reanchor_insurance exec'd out of train_sft_env.py (which imports torch /
    bitsandbytes at load time) against a settable fake clock."""
    import budget_controller as bc
    import submission_utils as SU
    src = open(os.path.join(HERE, "train_sft_env.py"), encoding="utf-8").read()
    clock = {"t": None}

    class _Now(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["t"]

    ns = {"datetime": types.SimpleNamespace(datetime=_Now, timezone=datetime.timezone, timedelta=datetime.timedelta),
          "math": math, "transformers": types.SimpleNamespace(TrainerCallback=object), "torch": None,
          "LOCAL_RANK": 0, "is_main_process": lambda rank: True, "log_info": lambda *a, **k: None,
          "ClockStop": bc.ClockStop, "CLOCK_STOP": bc.CLOCK_STOP, "wsd_lr_multiplier": bc.wsd_lr_multiplier,
          "reanchor_insurance_step": getattr(SU, "reanchor_insurance_step", None)}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
                "clock_cosine_multiplier", "ClockCosineCallback", "_reanchor_insurance"):
            exec(ast.get_source_segment(src, node), ns)
    return ns, clock


def test_clock_stop_reanchors_insurance() -> None:
    """The trainer's clock stop (budget_controller.ClockStop) ends a slow run at the last step that fits; when
    that lands before the insurance step computed from the PLANNED steps, the insurance never fires and a crash
    before the stop leaves nothing to ship. Once the stop step is known the insurance is re-derived from it."""
    import submission_utils as SU
    reanchor = getattr(SU, "reanchor_insurance_step", None)
    check("reanchor: submission_utils.reanchor_insurance_step exists", reanchor is not None)
    if reanchor is None:
        return
    check("reanchor: insurance before the stop is kept", reanchor(162, 811) == 162)
    check("reanchor: stop before the insurance -> re-derived from the stop step", reanchor(162, 120) == 50)
    check("reanchor: stop exactly at the insurance step -> re-derived", reanchor(162, 162) == 50)
    check("reanchor: stop too early for any insurance -> none", reanchor(162, 40) is None)
    check("reanchor: no stop step -> unchanged", reanchor(162, None) == 162)
    saved = SU.INSURANCE_SAVE
    SU.INSURANCE_SAVE = False
    try:
        check("reanchor: INSURANCE_SAVE = False stays off", reanchor(None, 120) is None)
    finally:
        SU.INSURANCE_SAVE = saved

    _install_stubs()
    import customized_trainer as CT
    from transformers import TrainerControl, TrainerState
    ns, clock = _load_clock_cosine()
    planned, s_step, stop_expected, crash_at = 811, 30.0, 120, 100
    handler = CT.WhenToEvalHandler(FAR_END, 5, periodic_save_steps=200, steps_per_epoch=planned, max_steps=-1,
                                   insurance_step=SU.insurance_save_step(planned))
    root = tempfile.mkdtemp()
    run_dir, submission = os.path.join(root, "out_0"), os.path.join(root, "submission")
    save_cb = CT.CustomEvalSaveCallback(handler, submission, run_dir, "base", end_time=FAR_END, publish_policy="latest")
    start = datetime.datetime(2026, 10, 5, 9, 0, tzinfo=datetime.timezone.utc)
    # the deadline fits stop_expected steps at s_step (plus the clock stop's save margin), 5-min save buffer
    end = start + datetime.timedelta(seconds=stop_expected * s_step + 90 + 5 * 60)
    clk = ns["ClockCosineCallback"](3e-5, 35, 0.25, planned, end.strftime("%Y-%m-%d %H:%M:%S"), 5,
                                    steps_per_epoch=planned, min_epochs=1,
                                    on_stop_step=lambda stop: ns["_reanchor_insurance"](handler, stop))
    clock["t"] = start
    clk.on_train_begin(None, None, None)
    state = TrainerState()
    for step in range(1, crash_at):  # train_sft_env's callback order: CustomEvalSaveCallback, then ClockCosine
        clock["t"] = start + datetime.timedelta(seconds=step * s_step)
        state.global_step, state.epoch, state.log_history = step, step / planned, [{"loss": 0.1}]
        control = TrainerControl()
        save_cb.on_step_end(None, state, control)
        clk.on_step_end(None, state, control)
        if control.should_evaluate:
            save_cb.on_evaluate(None, state, control, {"eval_loss": 0.1})
        if control.should_save:
            _write_adapter(os.path.join(run_dir, f"checkpoint-{step}"), str(step).encode())
            save_cb.on_save(None, state, control)
    check(f"slow box: clock stop lands before the planned insurance step "
          f"({clk._clock_stop.stop_step} < {SU.insurance_save_step(planned)})",
          clk._clock_stop.stop_step == stop_expected)
    check(f"slow box: insurance re-anchored to the stop step (step {handler.insurance_step})",
          handler.insurance_step == SU.insurance_save_step(stop_expected))
    shipped = SU.latest_complete_checkpoint([run_dir], adapter_only=True)
    check("slow box, crash before the stop: the re-anchored insurance adapter is recoverable",
          shipped is not None and shipped.endswith("checkpoint-50") and _loss_line(submission).startswith("50,"))

    with open(os.path.join(HERE, "train_sft_env.py")) as f:
        sft_tree = ast.parse(f.read())
    clock_calls = [n for n in ast.walk(sft_tree) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "ClockCosineCallback"]
    check("train_sft_env: ClockCosineCallback gets the insurance re-anchor hook",
          bool(clock_calls) and all(any(k.arg == "on_stop_step" for k in c.keywords) for c in clock_calls))


def test_crash_after_insurance() -> None:
    from env_submission import finalize_env_submission
    from submission_utils import insurance_save_step
    _, output_dir, run_dir, submission, _ = _layout()
    _scripted_sft(run_dir, submission, planned=811, crash_at=170, insurance_step=insurance_save_step(811))
    check("crash at step 170: the insurance checkpoint was published first", _loss_line(submission).startswith("162,"))
    rescues = []
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir],
                                      rescue=lambda: rescues.append(1) or [])
    with open(os.path.join(run_dir, "checkpoint-162", "adapter_model.safetensors"), "rb") as f:
        insured = f.read()
    with open(os.path.join(submission, "adapter_model.safetensors"), "rb") as f:
        shipped = f.read()
    check("crash after the insurance save -> that adapter is the submission", outcome == "ok" and shipped == insured)
    check("crash after the insurance save -> no rescue run", not rescues)

    shutil.rmtree(submission)
    os.makedirs(submission)
    with open(os.path.join(submission, "adapter_config.json"), "w") as f:
        f.write("{}")
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir], rescue=lambda: rescues.append(1) or [])
    check("publish cut mid-copy -> insurance checkpoint recovered, no rescue",
          outcome == "recovered" and _loss_line(submission).startswith("162,") and not rescues)


def test_crash_before_any_checkpoint() -> None:
    from env_submission import finalize_env_submission
    from submission_utils import has_adapter
    _, output_dir, run_dir, submission, ds_folder = _layout()
    dataset = os.path.join(ds_folder, "sft_env_t")
    _write_dataset(dataset)
    runners = FakeRunners()
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir],
                                      rescue=_rescue(_train_info(submission, dataset, FAR_END), output_dir,
                                                     ds_folder, runners))
    check("crash before any checkpoint -> rescue LoRA trained and shipped",
          outcome == "rescued" and has_adapter(submission) and _loss_line(submission).startswith("100,"))
    check("rescue: exactly one short run", len(runners.trained) == 1 and "--max_steps 100" in runners.trained[0])
    cmd = runners.trained[0]
    check("rescue: same launcher and LoRA flags as the SFT run",
          cmd.startswith("torchrun --nproc_per_node=2 train_sft_env.py") and "--use_lora True --lora_r 128" in cmd)
    check("rescue: trains into its own output dir", _cli_value(cmd, "output_dir") == output_dir + "_rescue_0")
    with open(_cli_value(cmd, "request_path")) as f:
        request = json.load(f)["train_request"]
    check("rescue: request trains the already-generated rows for 100 steps",
          request["dataset_path"] == dataset and request["max_steps"] == 100)
    check("rescue: request forces checking_mode='none' (production train_info starts 'first_time'; "
          "left alone, the checking_step=70 LR-continuation probe would cut the rescue short)",
          request["checking_mode"] == "none")
    check("rescue: the generated dataset exists -> nothing rebuilt", not runners.commands)
    check("rescue: add_random_noise never runs",
          not any("add_random_noise" in c for c in runners.trained + runners.commands))


def test_rescue_on_requested_rows() -> None:
    from env_submission import finalize_env_submission
    _, output_dir, run_dir, submission, ds_folder = _layout()
    missing = os.path.join(ds_folder, "sft_env_t")
    runners = FakeRunners()
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir],
                                      rescue=_rescue(_train_info(submission, missing, FAR_END), output_dir,
                                                     ds_folder, runners))
    builds = [c for c in runners.commands if "envs.intercode_dataset" in c]
    with open(_cli_value(runners.trained[0], "request_path")) as f:
        request = json.load(f)["train_request"]
    check("no generated dataset -> rescue rows built from the requested intercode dataset",
          outcome == "rescued" and len(builds) == 1 and request["dataset_path"] == _cli_value(builds[0], "output_path"))

    _, output_dir, run_dir, submission, ds_folder = _layout()
    dataset = os.path.join(ds_folder, "sft_env_t")
    _write_dataset(dataset)
    runners = FakeRunners(barren={dataset})
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir],
                                      rescue=_rescue(_train_info(submission, dataset, FAR_END), output_dir,
                                                     ds_folder, runners))
    check("generated rows yield no checkpoint -> second rescue attempt on the requested rows",
          outcome == "rescued" and len(runners.trained) == 2
          and _cli_value(runners.trained[1], "output_dir") == output_dir + "_rescue_1")


def test_rescue_limits() -> None:
    import env_submission as ES
    _, output_dir, run_dir, submission, ds_folder = _layout()
    dataset = os.path.join(ds_folder, "sft_env_t")
    _write_dataset(dataset)
    soon = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    runners = FakeRunners()
    outcome = ES.finalize_env_submission(submission, [None, output_dir, run_dir],
                                         rescue=_rescue(_train_info(submission, dataset, soon), output_dir,
                                                        ds_folder, runners))
    check("5 min left -> no rescue launched and nothing shipped (never full weights)",
          not runners.trained and outcome == "empty" and not os.listdir(submission))

    saved = ES.RESCUE_LORA
    ES.RESCUE_LORA = False
    try:
        runners = FakeRunners()
        roots = _rescue(_train_info(submission, dataset, FAR_END), output_dir, ds_folder, runners)()
        check("RESCUE_LORA = False -> insurance-only backstop, no rescue run", roots == [] and not runners.trained)
    finally:
        ES.RESCUE_LORA = saved

    boom = []

    def failing_rescue():
        boom.append(1)
        raise RuntimeError("rescue crashed")

    outcome = ES.finalize_env_submission(submission, [output_dir], rescue=failing_rescue)
    check("a crashing rescue never raises out of finalize", boom and outcome == "empty")


def test_rescue_retries_and_clock_per_attempt() -> None:
    """The rescue gets an OOM batch halving (run_training halves per_device_train_batch_size on an OOM retry),
    and the time floor is checked before EVERY attempt, not only once up front."""
    import env_submission as ES
    _, output_dir, run_dir, submission, ds_folder = _layout()
    dataset = os.path.join(ds_folder, "sft_env_t")
    _write_dataset(dataset)
    runners = FakeRunners()
    _rescue(_train_info(submission, dataset, FAR_END), output_dir, ds_folder, runners)()
    check(f"rescue run_training gets retries={ES.RESCUE_RETRIES} (>= 2: one OOM halving)",
          runners.retries == [ES.RESCUE_RETRIES] and ES.RESCUE_RETRIES >= 2)

    _, output_dir, run_dir, submission, ds_folder = _layout()
    dataset = os.path.join(ds_folder, "sft_env_t")
    _write_dataset(dataset)
    runners = FakeRunners(barren={dataset})
    clock = iter([60.0, 5.0, 5.0])
    real = ES._minutes_left
    ES._minutes_left = lambda end_time: next(clock)
    try:
        roots = _rescue(_train_info(submission, dataset, FAR_END), output_dir, ds_folder, runners)()
    finally:
        ES._minutes_left = real
    check("attempt 0 used the time up -> attempt 1 is skipped (below RESCUE_RESERVE_MINUTES)",
          roots == [] and len(runners.trained) == 1 and not runners.commands)


def test_full_weights_never_ship() -> None:
    from env_submission import finalize_env_submission
    from submission_utils import has_adapter, latest_complete_checkpoint
    _, output_dir, run_dir, submission, _ = _layout()
    _write_full(os.path.join(run_dir, "checkpoint-300"))
    _write_adapter(os.path.join(run_dir, "checkpoint-200"), b"200")
    check("adapter-only checkpoint search skips a newer full-weight checkpoint",
          latest_complete_checkpoint([run_dir], adapter_only=True).endswith("checkpoint-200"))
    check("default checkpoint search unchanged (newest complete checkpoint of any kind)",
          latest_complete_checkpoint([run_dir]).endswith("checkpoint-300"))
    _write_full(submission)
    rescues = []
    outcome = finalize_env_submission(submission, [run_dir], rescue=lambda: rescues.append(1) or [])
    names = set(os.listdir(submission))
    check("full-weight submission -> replaced by the newest adapter checkpoint",
          outcome == "recovered" and has_adapter(submission) and not any(n.startswith("model") for n in names)
          and not rescues)

    _, output_dir, run_dir, submission, _ = _layout()
    _write_full(submission)
    outcome = finalize_env_submission(submission, [run_dir], rescue=lambda: rescues.append(1) or [])
    check("full-weight submission and no adapter anywhere -> cleared, rescue asked",
          outcome == "cleared" and rescues and not os.listdir(submission))


def test_happy_path_untouched() -> None:
    from env_submission import finalize_env_submission
    _, output_dir, run_dir, submission, _ = _layout()
    _write_adapter(os.path.join(run_dir, "checkpoint-820"), b"820")
    _write_adapter(submission, b"820")
    for name, body in (("loss.txt", b"820,no_eval"), ("tokenizer.json", b"{}"), ("README.md", b"base_model: x\n")):
        with open(os.path.join(submission, name), "wb") as f:
            f.write(body)
    before = _snapshot(submission)
    rescues = []
    outcome = finalize_env_submission(submission, [None, output_dir, run_dir], rescue=lambda: rescues.append(1) or [])
    check("happy path -> 'ok', submission byte-identical, no rescue",
          outcome == "ok" and _snapshot(submission) == before and not rescues)


def test_adapter_assert() -> None:
    from submission_utils import adapter_submission_problems, enforce_adapter_submission
    root = tempfile.mkdtemp()
    good, bare, full, mixed = (os.path.join(root, n) for n in ("good", "bare", "full", "mixed"))
    _write_adapter(good)
    os.makedirs(bare)
    with open(os.path.join(bare, "adapter_config.json"), "w") as f:
        f.write("{}")
    _write_full(full)
    _write_adapter(mixed, b"m")
    _write_full(mixed)
    check("adapter check: complete adapter -> no problems", adapter_submission_problems(good) == [])
    check("adapter check: config without weights -> flagged", adapter_submission_problems(bare) != [])
    check("adapter check: full weights only -> flagged", adapter_submission_problems(full) != [])
    problems = adapter_submission_problems(mixed)
    check("adapter check: adapter + model-*.safetensors -> the shards are flagged",
          any("model-00001-of-00002.safetensors" in p for p in problems))
    adapter_before = _snapshot(mixed)["adapter_model.safetensors"]
    check("enforce: stray full-weight files removed next to a complete adapter",
          enforce_adapter_submission(mixed) == [] and _snapshot(mixed)["adapter_model.safetensors"] == adapter_before
          and not any(n.startswith("model") for n in os.listdir(mixed)))
    check("enforce: never deletes the only weights of a non-adapter dir",
          enforce_adapter_submission(full) != [] and "model-00001-of-00002.safetensors" in os.listdir(full))


def test_lora_cmd_detection() -> None:
    from env_submission import is_lora_cmd
    check("LoRA run command detected", is_lora_cmd(BASE_CMD))
    check("full fine-tune command not a LoRA run", not is_lora_cmd(BASE_CMD.replace(" --use_lora True", "")))
    check("--use_lora False not a LoRA run", not is_lora_cmd(BASE_CMD.replace("--use_lora True", "--use_lora False")))


def _enclosing_ifs(tree: ast.AST, target: ast.AST) -> list:
    """If-tests (as source) whose BODY contains ``target``."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    tests, child, node = [], target, parents.get(target)
    while node is not None:
        if isinstance(node, ast.If) and any(child is n for n in node.body):
            tests.append(ast.unparse(node.test))
        child, node = node, parents.get(node)
    return tests


def test_source_guards() -> None:
    with open(os.path.join(HERE, "text_trainer.py")) as f:
        trainer_src = f.read()
    with warnings.catch_warnings():  # text_trainer's own regex literals ("\s") warn on newer Pythons
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(trainer_src)
    noise = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and "add_random_noise" in n.value]
    guarded = [any("!=" in t and "ENVIRONMENTTASK" in t for t in _enclosing_ifs(tree, n)) for n in noise]
    check("text_trainer: the EnvTask path cannot reach add_random_noise", bool(noise) and all(guarded))
    check("text_trainer: EnvTask end of run goes through finalize_env_submission",
          "finalize_env_submission(" in trainer_src)

    with open(os.path.join(HERE, "train_sft_env.py")) as f:
        sft_tree = ast.parse(f.read())
    handler_calls = [n for n in ast.walk(sft_tree) if isinstance(n, ast.Call)
                     and getattr(n.func, "id", "") == "WhenToEvalHandler"]
    check("train_sft_env: WhenToEvalHandler receives the insurance step",
          bool(handler_calls) and all(any(k.arg == "insurance_step" for k in c.keywords) for c in handler_calls))

    with open(os.path.join(HERE, "sft_env_config.py")) as f:
        config_src = f.read()
    check("sft_env_config: SFT_SMALL_LORA stays default ON (LoRA on every base)",
          re.search(r'os\.getenv\("SFT_SMALL_LORA", "1"\)', config_src) is not None)


if __name__ == "__main__":
    for t in (test_insurance_step, test_handler_schedule, test_clock_stop_reanchors_insurance, test_crash_after_insurance,
              test_crash_before_any_checkpoint, test_rescue_on_requested_rows, test_rescue_limits,
              test_rescue_retries_and_clock_per_attempt,
              test_full_weights_never_ship, test_happy_path_untouched, test_adapter_assert,
              test_lora_cmd_detection, test_source_guards):
        try:
            t()
        except Exception as exc:  # one broken test must not hide the others
            check(f"{t.__name__} raised {exc!r}", False)
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL(S)'}")
    sys.exit(1 if FAILS else 0)
