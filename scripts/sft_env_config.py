"""
Orchestrator config for EnvTask SFT training.
Counterpart of instruct_config.py for the SFT environment mode.
No GRPO content; no vLLM/num_generations/beta.
"""

import json
import os
from copy import deepcopy

from budget_controller import LONG_TASK_HOURS, is_share_round
from envs.shared_env import _log, recipe_log
from lrs_lookup import get_lr_from_ar_instruct
from model_utility import (
    disable_flash_attention,
    get_gpu_count,
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
)

SFT_ENV_SIZE_CONFIG: dict[str, dict] = {
    "0_1_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "1_2_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": False},
    "2_4_b":  {"lr": 5e-5,   "distributed": "ddp", "gpu_count": 1, "batch_size": 24,  "gradient_accumulation_steps": 1, "use_lora": False},
    "4_5_b":  {"lr": 7e-5,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "5_9_b":  {"lr": 3.5e-5, "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "9_12_b": {"lr": 1e-4,   "distributed": "ddp", "gpu_count": 2, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "12_15_b":{"lr": 1e-4,   "distributed": "ds",  "gpu_count": 4, "batch_size": 32,  "gradient_accumulation_steps": 1, "use_lora": True},
    "15_40_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 4, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
    "40_80_b":{"lr": 8e-5,   "distributed": "ds",  "gpu_count": 8, "batch_size": 16,  "gradient_accumulation_steps": 2, "use_lora": True},
}

for _key in SFT_ENV_SIZE_CONFIG:
    SFT_ENV_SIZE_CONFIG[_key]["label"] = _key


def get_sft_env_config(param_nums: int) -> dict:
    if param_nums is None:
        raise ValueError("Cannot determine model size: weight counting failed")
    result = {"lr": 4e-5, "distributed": "ds", "gpu_count": 8, "batch_size": 6, "use_lora": True}
    if param_nums < 1_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["12_15_b"]
    elif param_nums < 40_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = SFT_ENV_SIZE_CONFIG["40_80_b"]
    else:
        _log(f"Model size {param_nums} is not supported, using 40_80_b")
    result = deepcopy(result)
    # Per-device batch / grad-accum OVERRIDE for smaller boxes. The size bands' per-device
    # batch is tuned for the tournament's multi-GPU hosts; on a single H100 an 8B model at
    # batch 32 OOMs (73GB). SFT_BATCH / SFT_GA let a test box fit memory while preserving a
    # sane effective batch via grad-accum (e.g. batch 8 x ga 4 = 32). DEFAULT = band values,
    # so the deployed tournament recipe is UNCHANGED unless these env vars are set.
    _b = os.getenv("SFT_BATCH", "").strip()
    _g = os.getenv("SFT_GA", "").strip()
    if _b:
        result["batch_size"] = int(_b)
        _log(f"[sft-config] SFT_BATCH override: per_device_train_batch_size -> {result['batch_size']}")
    if _g:
        result["gradient_accumulation_steps"] = int(_g)
        _log(f"[sft-config] SFT_GA override: gradient_accumulation_steps -> {result['gradient_accumulation_steps']}")
    # AUTO 1-GPU eff-batch (rank-12, OPT-IN via AUTO_1GPU_BATCH=1; default OFF -> band values UNCHANGED).
    # The size bands are tuned for the tournament's multi-GPU hosts; on a SINGLE H100 an >=8B model at the
    # band's micro-batch (32) OOMs, and even when it fits the effective batch collapses with no anneal
    # headroom. When EXACTLY ONE GPU is visible, cap the micro-batch for memory (>=8B) and rebuild a small
    # validated effective batch via grad-accum. STRICTLY guarded on get_gpu_count()==1 so any >=2-GPU
    # allocation is byte-identical to today. SFT_BATCH/SFT_GA above still take precedence (explicit wins).
    # MUST be A/B'd on the VM (eff-24 vs the band's eff, BOTH bases) before the default is flipped.
    if os.getenv("AUTO_1GPU_BATCH", "0") == "1" and not _b and not _g:
        try:
            if get_gpu_count() == 1:
                _eff = int(os.getenv("AUTO_1GPU_EFF_BATCH", "24"))
                _micro_cap = int(os.getenv("AUTO_1GPU_MICRO_BATCH", "6"))
                if param_nums >= 8_000_000_000:
                    result["batch_size"] = max(1, min(result["batch_size"], _micro_cap))
                result["gradient_accumulation_steps"] = max(1, round(_eff / max(1, result["batch_size"])))
                _log(f"[auto-1gpu] 1 GPU: micro-batch {result['batch_size']} x ga "
                     f"{result['gradient_accumulation_steps']} = effective "
                     f"{result['batch_size'] * result['gradient_accumulation_steps']}", flush=True)
        except Exception as _e:
            _log(f"[auto-1gpu] skipped: {_e}", flush=True)
    return result


# Generation-time cap for tasks of LONG_TASK_HOURS or more (the 3h boss from-scratch task, drawn on any
# base). Its fixed total of rows is generated well inside a quarter of 3h, while the <4B ratio 0.40 would
# spend 72 min generating. 1.0 restores the plain size ratio.
LONG_TASK_GEN_RATIO = 0.25


def get_generation_time_ratio(param_nums: int, env_names: "list[str] | None" = None,
                              hours: "float | None" = None) -> float:
    """Fraction of the time budget spent on data generation: the model-size ratio below, capped at
    LONG_TASK_GEN_RATIO for share rounds (fixed total of rows) of LONG_TASK_HOURS or more. R1 (intercode +
    one game, absolute row floor) keeps the plain size ratio at every length."""
    ratio = _size_generation_ratio(param_nums, env_names)
    if hours is not None and hours >= LONG_TASK_HOURS and is_share_round(env_names):
        return min(ratio, LONG_TASK_GEN_RATIO)
    return ratio


def _size_generation_ratio(param_nums: int, env_names: "list[str] | None" = None) -> float:
    """Fraction of total time budget spent on data generation, by model size.

    Smaller models train faster, so they can afford to spend more of the
    fixed wall-clock budget generating data.

    OTHELLO 8B FLOOD (rank-2, OPT-IN via OTHELLO_8B_GENTIME=1): the 72% othello flood was
    VM-validated at 3B (ratio 0.40 fills the 36k flood). On 8B the default 0.15 (~810s of a 1.5h
    budget) physically CANNOT fill the weight-6.0 / 36k othello flood (othello gen is compute-heavy),
    so both SFT othello levers go inert and the student collapses to ~16-20% despite an ~85% teacher.
    When 'othello' is in the round AND OTHELLO_8B_GENTIME=1 on a >=6B model, use
    OTHELLO_8B_GENTIME_RATIO (default 0.40, matching the validated 3B value) so the flood FILLS.
    OPT-IN so committing changes NOTHING; A/B on the VM (gen must fill ~36k AND ~50min training must
    remain so the single cosine epoch anneals to min LR), then flip the default. GEN_SECONDS at the
    call site remains the hard manual override and takes precedence over this.
    """
    if (env_names and "othello" in env_names and param_nums >= 6_000_000_000
            and os.getenv("OTHELLO_8B_GENTIME", "").strip() == "1"):
        try:
            return float(os.getenv("OTHELLO_8B_GENTIME_RATIO", "0.40"))
        except ValueError:
            return 0.40
    if param_nums < 2_000_000_000:
        return 0.35
    elif param_nums < 4_000_000_000:
        # 0.25→0.30→0.40 (2026-09-03, VM-VALIDATED othello flood @ 72% vs the champion):
        # the winning run used GEN_SECONDS=2400 (≈0.44 of 1.5h) and gen produced 455k othello
        # → capped to 36k, then trained ~60-70min/1-epoch → 72%. 0.40 bakes ~2160s gen so the
        # 36k flood reliably FILLS (othello gen is compute-heavy: ~15-40min for 8000 games),
        # while still leaving ~50min training for the flood's single annealed epoch. Paired with
        # flood-aware epochs=1 (budget_controller.recommend_epochs) for othello rounds.
        return 0.40
    elif param_nums < 6_000_000_000:
        return 0.20
    else:
        return 0.15


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger_kernel",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = "deepspeed"
    else:
        start_cmd = "python"

    # LR-schedule flags. For WSD the wall-clock WSDClockCallback in train_sft_env.py
    # owns the LR (it forces a constant base + drives warmup/stable/decay by the
    # clock), so we must NOT pass cosine_with_min_lr kwargs here or they'd conflict.
    if config.get("lr_schedule") == "wsd":
        lr_flags = """ \
    --lr_scheduler_type constant \
    --warmup_steps 0"""
    else:
        lr_flags = """ \
    --warmup_steps 35 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}\""""

    # --report_to none: wandb gives us nothing here (training runs on the validator's
    # infra, we never see the run) and its background service intermittently fails to
    # start within 30s -> ServiceStartTimeoutError kills a training attempt and burns
    # wall-clock on the retry (observed on the Llama-8B smoke). Our callbacks
    # (CustomEvalSaveCallback/WhenToEvalHandler) don't touch wandb, so dropping it is free.
    template = (
        start_cmd
        + """ train_sft_env.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to none \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0."""
        + lr_flags
        + """ \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger_kernel {use_liger_kernel} \
    --packing {packing} \
    --disable_fa {disable_fa} \
    --max_length 4096"""
    )

    if run_type == "ds":
        template += " --deepspeed ds_config/zero3.json"

    if config.get("use_lora", False):
        template += " --use_lora True"
        if "lora_r" in config:  # A/B lever set an explicit LoRA config (else LoraArguments defaults apply)
            template += (f" --lora_r {config['lora_r']} --lora_alpha {config['lora_alpha']}"
                         f" --lora_dropout {config['lora_dropout']}")

    if not config.get("disable_fa", False):
        template += " --padding_free True"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


def _is_merged_continuation(model_path: str) -> bool:
    """A continuation base is the previous round's LoRA merged and re-saved by the validator's transformers-5
    downloader, whose config.json carries the v5 ``rope_parameters`` block (see base_config_compat, which
    keeps it in the shadow config); foundation configs do not."""
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            return isinstance(json.load(f).get("rope_parameters"), dict)
    except (OSError, ValueError, AttributeError):
        return False


def _intercode_dose(model_path: str, param_nums: int, record_path: str) -> float:
    """Intercode row-budget multiplier from the validator's baseline of the starting model
    (budget_controller.intercode_dose). The numbers go to a local file, not the public log."""
    from budget_controller import baseline_env_mean, intercode_dose, record_baseline
    mean = baseline_env_mean(os.getenv("BASELINE_STATS_PATH"), "intercode")
    continuation = _is_merged_continuation(model_path)
    dose = intercode_dose(mean, continuation, param_nums)
    record_baseline(record_path, {"intercode_mean": mean, "continuation": continuation, "dose": dose})
    return dose


def get_training_json(train_info: dict) -> dict:
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_path)
    config = get_sft_env_config(param_nums)

    # EFFECTIVE-BATCH target (adopted from the R1 winner — port, not copy). They hold the GLOBAL
    # batch at ~128 via GPU-scaled grad-accum, having found effective-64 UNDER-trained the
    # game_action emission (-> forfeit) and effective-128 fixed it. Our per-bucket batch x gpu gives
    # a much smaller effective batch (3B: 24 x 1-2 GPU = 24-48; 7B: 32 x 2 = 64), so this scales
    # gradient_accumulation_steps to hit the target. The pos-1 winner ships eff=128 unconditionally
    # (their forfeit autopsy: eff-64 under-trains game_action emission). We TRIED default-128 overnight
    # but REVERTED to flag-gated: othello (our confirmed rank-1 weapon) is validated at 58% with the
    # current effective 32, and eff-128 is UNVALIDATED on our compute — on a slow validator its 4x-fewer
    # optimizer steps could REDUCE othello. Protect the validated weapon; A/B eff-128 with SFT_EFFECTIVE_BATCH=128.
    _eff_target = os.getenv("SFT_EFFECTIVE_BATCH")
    if _eff_target:
        try:
            _target = int(_eff_target)
            _gpus = max(1, get_gpu_count())
            _per_step = max(1, config["batch_size"] * _gpus)
            config["gradient_accumulation_steps"] = max(1, round(_target / _per_step))
            _eff = config["batch_size"] * _gpus * config["gradient_accumulation_steps"]
            _log(f"[eff-batch] target={_target}: batch {config['batch_size']} x {_gpus} gpu x "
                 f"ga {config['gradient_accumulation_steps']} = effective {_eff}", flush=True)
        except Exception as _e:  # bad env value -> leave config untouched, never block training
            _log(f"[eff-batch] ignoring SFT_EFFECTIVE_BATCH={_eff_target!r}: {_e}", flush=True)

    # SMALL-MODEL LoRA, DEFAULT ON for EnvTask (2026-09-26). A FULL-weight submission gets NO tool-call
    # parser in the validator's intercode evaluator: it serves full weights with parser_model_id=<the opaque
    # tournament repo id> (G.O.D validator/evaluation/evaluators/intercode.py full-weight branch), which maps
    # to no family, so every intercode turn forfeits and scores the no-op floor — the real 08-31 Llama-3.2-3B
    # tournament shows 11/15 miners at the identical 0.7410 (R2 group: all 0.6725) while a LoRA entrant
    # scored .7745/.837/.853 and won intercode every round. LoRA submissions get
    # parser_model_id=original_model, and PvP resolves the parser from the base either way. R1 always has
    # intercode and 3 of the 6 whitelisted bases are <4B, so the <4B bands must ship a LoRA. The regularised
    # r128 / alpha512 / dropout0.1 is the 3-peat 3B champion's recipe. SFT_SMALL_LORA=0 restores
    # the old full fine-tune. Rank/alpha/dropout env-overridable.
    _small_lora = os.getenv("SFT_SMALL_LORA", "1").strip().lower() not in ("", "0", "false", "no", "off")
    _small_lora_applied = bool(_small_lora and not config.get("use_lora", False))
    if _small_lora_applied:
        config["use_lora"] = True
        config["lora_r"] = int(os.getenv("SFT_SMALL_LORA_R") or "128")
        config["lora_alpha"] = int(os.getenv("SFT_SMALL_LORA_ALPHA") or "512")
        config["lora_dropout"] = float(os.getenv("SFT_SMALL_LORA_DROPOUT") or "0.1")
        _log(f"[small-lora] size-config full-FT -> LoRA r{config['lora_r']}/"
             f"a{config['lora_alpha']}/drop{config['lora_dropout']}", flush=True)

    # LoRA RANK for LoRA buckets (e.g. the 8B bucket). The optimal rank is ROUND-DEPENDENT (2026-09-11
    # 8B VM campaign, 1 GPU): r256 WON the 6-game boss round (othello 49->61% vs champ,
    # liars 65->72% — capacity holds many envs co-trained) but LOST the 2-env R1 othello+intercode round
    # (othello 63%->48% — r256 over-fits/undertrains a small focused pool). Since R1 is the GUARANTEED
    # round you must win to advance, the DEFAULT stays the R1-safe r128; r256 is OPT-IN (SFT_LORA_R=256)
    # for wide boss rounds. NOTE: the 1-GPU box undertrains r256 (fewer steps), so the R1 drop may be
    # partly an artifact — re-validate on 2-4 GPU (and consider a round-aware rank: r128 for <=4 envs,
    # r256 for >=5) before flipping the default. The champion runs r256 (on 2-4 GPU with enough steps).
    if config.get("use_lora", False) and "lora_r" not in config:
        config["lora_r"] = int(os.getenv("SFT_LORA_R") or "128")
        config["lora_alpha"] = int(os.getenv("SFT_LORA_ALPHA") or "512")
        config["lora_dropout"] = float(os.getenv("SFT_LORA_DROPOUT") or "0.05")
        _log(f"[lora-rank] LoRA bucket -> r{config['lora_r']}/a{config['lora_alpha']}/"
             f"drop{config['lora_dropout']}", flush=True)

    task_id = train_info["task_id"]
    env_names = train_info.get("dataset_type", {}).get("environment_names") or ["liars_dice"]
    dataset_path = f"/workspace/scripts/datasets/sft_env_{task_id}"

    # Self-calibrating budget plan: WSD fracs, coverage/replay env layer, epochs.
    from budget_controller import plan_training
    try:
        from envs.sft_env_configs import _SFT_REGISTRY
        all_game_envs = list(_SFT_REGISTRY.keys())
    except Exception:
        all_game_envs = ["liars_dice", "leduc_poker", "gin_rummy", "othello", "goofspiel"]
    plan = plan_training(float(train_info["hours_to_complete"]), env_names, all_game_envs)
    for _n in plan.notes:
        _log(f"[budget] {_n}", flush=True)

    run_config = {
        "epoch_num": plan.epochs,
        "lr_schedule": "cosine",  # epoch-based cosine decay — see note at train_request below
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.25,
        "use_liger_kernel": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture),
        "packing": "False",  # pre-tokenised dataset; TRL packing not used
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
    }
    # carry the small-LoRA hyperparams (if the A/B lever set them) into the run command
    for _lk in ("lora_r", "lora_alpha", "lora_dropout"):
        if _lk in config:
            run_config[_lk] = config[_lk]

    if train_info.get("find_lk_lr"):
        lr = get_lr_from_ar_instruct(model_architecture, param_nums)
        if lr is not None:
            _log(f"Using lr from architecture config: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            _log(f"Using lr from config: {run_config['learning_rate']}", flush=True)

    run_config["learning_rate"] *= train_info["reg_ratio"]

    # Family-aware ENV LR (2026-09-23) — robustness for "either base can be assigned" (unsloth/Llama
    # OR Qwen). The size band (7-9B → 3.5e-5) is family-BLIND, but the game-recipe optimum differs by
    # base family: Qwen2.5-7B winners (the field's top finishers) peaked ~4.3e-5, while the
    # Llama-3.1-8B champion ran ~1.46e-5. SFT_FAMILY_LR=1 auto-detects the architecture and picks the
    # harvested per-family LR, so ONE submission self-tunes to whichever base the tournament draws.
    # Gated OFF (needs a per-family Qwen+Llama VM A/B first); unknown families fall through to the band;
    # SFT_LR below still overrides everything.
    if os.getenv("SFT_FAMILY_LR", "").strip() == "1":
        _arch = (model_architecture or "").lower()
        # Per-family values env-overridable (defaults = current harvested LRs, so SFT_FAMILY_LR=1 behaves
        # exactly as before). Lets the VM A/B raise the Llama EnvTask LR (e.g. SFT_FAMILY_LR_LLAMA=4.3e-5)
        # WITHOUT editing lrs/instruct_ar.csv (which is shared with InstructTextTask via find_lk_lr).
        try:
            _qwen_lr = float(os.getenv("SFT_FAMILY_LR_QWEN", "4.3e-5"))
            _llama_lr = float(os.getenv("SFT_FAMILY_LR_LLAMA", "1.46e-5"))
        except ValueError:
            _qwen_lr, _llama_lr = 4.3e-5, 1.46e-5
        _fam_lr = _qwen_lr if "qwen" in _arch else (_llama_lr if "llama" in _arch else None)
        if _fam_lr is not None:
            run_config["learning_rate"] = _fam_lr
            _log(f"[sft-config] SFT_FAMILY_LR: arch={_arch!r} -> learning_rate {_fam_lr}", flush=True)
    # The two family LoRA LRs below were validated on the 7B/8B LoRA bands only. Bases that the
    # small-LoRA default switched from full fine-tune keep the csv x reg_ratio lineage LR (Llama-3.2-3B
    # 6.1e-5, Qwen2.5-3B 7.9e-5, Qwen2.5-1.5B 1.05e-4): a LoRA wants at least the full-FT LR, not half.
    elif (run_config.get("use_lora") and not _small_lora_applied
          and (model_architecture or "").lower() == "qwen2forcausallm"
          and os.getenv("SFT_FAMILY_LR", "").strip() != "0"):
        # Qwen2.5 LoRA default = 4.3e-5: EVERY Qwen2.5-7B VM validation of the assistant-only packed
        # recipe (2026-09-24/25, h2h vs the real R1 winners) ran at 4.3e-5, the harvested Qwen-winner LR.
        # Scoped to what was validated: the Qwen2.5 full fine-tune bands (1.5B/3B, bf16 weights) and
        # Qwen3 keep the csv lookup x reg_ratio lineage, like Llama. SFT_FAMILY_LR=0 turns it off.
        try:
            run_config["learning_rate"] = float(os.getenv("SFT_FAMILY_LR_QWEN", "4.3e-5"))
        except ValueError:
            run_config["learning_rate"] = 4.3e-5
        _log(f"[sft-config] Qwen default LR -> {run_config['learning_rate']} (validated packed recipe)", flush=True)
    elif (run_config.get("use_lora") and not _small_lora_applied
          and (model_architecture or "").lower() == "llamaforcausallm"
          and os.getenv("SFT_FAMILY_LR", "").strip() != "0"):
        # Llama LoRA (Llama-3.1-8B) default = 3e-5. VM A/B 2026-09-26, real entrypoint, R1 [intercode, gin],
        # same code, only the LR differs: gin STRICT h2h vs the new champion's boss-round Llama model
        # 51% (48-47-5) at the csv lineage 1.556e-5 -> 61% (60-39-1) at 3e-5, intercode proxy F1/exact
        # .328/.224 -> .343/.276. The csv LR was tuned for full-sequence loss; the packed assistant-only
        # loss supervises ~2% of the tokens, which wants a higher LR (same finding as Qwen). The Llama-3.2-3B
        # full fine-tune (use_lora False) keeps its lineage LR. SFT_FAMILY_LR=0 turns it off.
        try:
            run_config["learning_rate"] = float(os.getenv("SFT_FAMILY_LR_LLAMA_LORA", "3e-5"))
        except ValueError:
            run_config["learning_rate"] = 3e-5
        _log(f"[sft-config] Llama LoRA default LR -> {run_config['learning_rate']} (VM A/B 2026-09-26)", flush=True)

    # Winning-recipe LR knob (2026-09-21 harvest). SFT_LR sets the peak LR directly, final precedence
    # over the band + reg_ratio + SFT_FAMILY_LR. Unset = the value chosen above (default unchanged).
    _lr_override = os.getenv("SFT_LR", "").strip()
    if _lr_override:
        try:
            run_config["learning_rate"] = float(_lr_override)
            _log(f"[sft-config] SFT_LR override: learning_rate -> {run_config['learning_rate']}", flush=True)
        except ValueError:
            _log(f"[sft-config] ignoring invalid SFT_LR={_lr_override!r}", flush=True)

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    train_request = deepcopy(train_info)
    train_request["dataset_path"] = dataset_path
    train_request["save_before_remaining_time"] = 5
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 200
    train_request["checking_step"] = 70
    train_request["min_steps"] = max(
        int(train_info["hours_to_complete"] * 70),
        train_info.get("min_steps", 100),
    )
    # LR schedule = cosine-with-min-lr, annealed over the epochs (our tuned schedule: decays
    # the peak LR to min_lr_rate 0.25 by the last epoch, e.g. ~4.3e-05 -> ~1.6e-05). We
    # switched OFF wall-clock WSD here: WSD was chosen back when the dataset was huge and the
    # stop-step was unknown, so it annealed on the clock. With the new SMALL balanced set the
    # step count IS known — but a few-env task can now finish its 2 epochs BEFORE the clock's
    # decay window opens, which would skip the anneal entirely and leave the model at full LR.
    # Cosine anneals on step/epoch progress so it always completes; the deadline guard in
    # customized_trainer.py still stops+saves before the budget ends. (wsd_* below are inert
    # unless lr_schedule is flipped back to "wsd".)
    # "cosine_clock" (default since 2026-09-25) = the same cosine_with_min_lr curve (CLI warmup and
    # min_lr_rate unchanged) driven by max(step progress, clock progress) in train_sft_env, so a run the
    # deadline cuts mid-epoch still publishes an annealed checkpoint. On the validated 1-GPU runs the cut
    # came at ~87% of the epoch; the real 2-GPU budget cuts earlier (fixed costs do not halve), which the
    # plain step cosine would publish at ~0.4x peak LR. A run that finishes its steps early is unchanged.
    # SFT_LR_SCHEDULE=cosine restores the pure step schedule.
    train_request["lr_schedule"] = os.getenv("SFT_LR_SCHEDULE", "cosine_clock").strip().lower() or "cosine_clock"
    train_request["wsd_warmup_frac"] = plan.wsd_warmup_frac
    train_request["wsd_decay_frac"] = plan.wsd_decay_frac
    train_request["wsd_min_mult"] = plan.wsd_min_mult
    train_request["min_epochs"] = plan.min_epochs  # epoch floor of the plan; the trainer's clock stop warns below it

    gen_seconds = int(train_info["hours_to_complete"] * 3600
                      * get_generation_time_ratio(param_nums, env_names, float(train_info["hours_to_complete"])))
    gen_seconds = int(os.getenv("GEN_SECONDS") or gen_seconds)  # override for the data-volume experiment (empty/unset -> computed)
    generate_cmd = (
        f"python -m envs.generate_trajectories"
        f" --environment_names {' '.join(env_names)}"
        f" --output_path {dataset_path}"
        f" --time_limit_seconds {gen_seconds}"
    )
    # SMALL per-env cap so the merged set is winner-sized (~24k total, per-env weapon-weighted) and
    # the cosine LR ANNEALS within the 1.5h deadline. MEASURED 2026-08-29: the old 50000 cap produced
    # a 65-173k set that the deadline-guarded trainer only reached ~0.05-0.19 epoch of — LR frozen near
    # peak, models half-baked. Our othello student was 2% vs the specialist champion at cap 50000; retrained
    # focused at a SMALL cap (full epochs + anneal) it jumped to 58% vs the same champion (teacher is 92%) —
    # the distillation gap was pure undertraining. The winner's proven ~24.7k trains full epochs+anneal
    # on the validator box; 6000 x the weapon weights lands ~14k (2-env R1) / ~24k (4-env R2). Override
    # with SCORED_CAP. (Simple/commodity envs saturate in <20 steps, so this only helps.)
    # Per-env scored cap. SCORED_CAP is the EXACT-cap escape hatch (no fit reduction). Otherwise the
    # base 6000 is multiplied by the rank-6 data-fit reducer so a slow (1-GPU) / many-env draw shrinks
    # the cap to what the SFT window can ANNEAL over. DATA_FIT_SCALE unset -> factor 1.0 -> cap 6000
    # (byte-identical to the previous behavior).
    _scored_cap_override = os.getenv("SCORED_CAP", "").strip()
    if _scored_cap_override:
        scored_cap = int(_scored_cap_override)
    else:
        from budget_controller import data_fit_scale
        _base_cap = 6000
        _gpu = get_gpu_count()
        _fit = data_fit_scale(len(env_names), _gpu, float(train_info["hours_to_complete"]), epochs=plan.epochs)
        scored_cap = max(1, int(round(_base_cap * _fit)))
        if _fit != 1.0:
            _log(f"[data-fit] scored_cap {_base_cap} x {_fit:.3f} -> {scored_cap} "
                 f"(n_envs={len(env_names)} gpu={_gpu} h={train_info['hours_to_complete']} "
                 f"epochs={plan.epochs})", flush=True)
    generate_cmd += f" --scored_cap {scored_cap}"
    # Rounds with >= 3 scored envs train a fixed total split by the share table (budget_controller.share_caps);
    # R1 and smaller rounds get no total and keep the per-env caps above.
    from budget_controller import round_total_rows
    total_rows = round_total_rows(env_names, float(train_info["hours_to_complete"]), get_gpu_count())
    if total_rows:
        generate_cmd += f" --total_rows {total_rows}"
    # Coverage / replay layer: a thin slice of the non-scored game envs, generated
    # AFTER the scored envs, so the continuation model carries them forward /
    # doesn't forget (see budget_controller.plan_training). Empty in the final
    # round (all envs scored) -> no-op.
    if plan.coverage_envs and plan.coverage_games > 0:
        generate_cmd += (
            f" --coverage_envs {' '.join(plan.coverage_envs)}"
            f" --coverage_games {plan.coverage_games}"
        )

    # R1 VOLUME RECIPE (budget_controller.is_r1_volume_round — VM-validated vs the real R1 winners):
    # a per-env row FLOOR (game and intercode each >= R1_MIN_ENV_ROWS; never lowers othello's flood)
    # and the compliant intercode self-gen (general bash tasks in a temp dir, real stdout — see
    # envs/intercode_selfgen.py). Threaded as an env prefix like PVP_BASE_FAMILY below; the generate step
    # runs inside our training container, so the self-gen host guard is satisfied explicitly. Any other
    # round shape (or R1_VOLUME=0) leaves generate_cmd byte-identical. recommend_epochs pairs it with 1 epoch.
    from budget_controller import is_r1_volume_round, R1_MIN_ENV_ROWS, R1_SELFGEN_TASKS
    if is_r1_volume_round(env_names):
        # INTERCODE_RETRY_FRAC=0: the passthrough "retry" rows re-run the same command after an invented
        # transient error that the deterministic eval shell never produces; the eval-validator analysis
        # (2026-09-25) found our model re-running identical failing commands in 174/218 post-error turns.
        generate_cmd = (
            f"R1_MIN_ENV_ROWS={R1_MIN_ENV_ROWS} INTERCODE_SELFGEN=1 INTERCODE_SELFGEN_CONTAINER=1 "
            f"INTERCODE_SELFGEN_N={R1_SELFGEN_TASKS} INTERCODE_RETRY_FRAC=0 " + generate_cmd
        )
        _log(f"[sft-config] R1 volume recipe: >= {R1_MIN_ENV_ROWS} rows/env + intercode self-gen "
             f"({R1_SELFGEN_TASKS} tasks), {plan.epochs} epoch(s)", flush=True)
    elif "intercode" in env_names and total_rows:
        # The same self-gen recipe in every other round that scores intercode, sized to intercode's share
        # and dosed up when the starting model has forgotten intercode (budget_controller).
        from budget_controller import intercode_recipe_prefix
        _dose = _intercode_dose(train_info["model_path"], param_nums,
                                os.path.join(os.path.dirname(dataset_path), f"baseline_{task_id}.json"))
        generate_cmd = intercode_recipe_prefix(env_names, total_rows, _dose) + generate_cmd

    # Small bases learn the gin upcard rows at a lower oversampling k (budget_controller.gin_generate_prefix).
    from budget_controller import gin_generate_prefix
    _gin_prefix = gin_generate_prefix(env_names, param_nums, plan.coverage_envs)
    if _gin_prefix:
        generate_cmd = _gin_prefix + generate_cmd
        _log(f"[sft-config] {_gin_prefix.strip()} threaded to generate_cmd (base {param_nums} params)", flush=True)

    # Dual-model emit-surface split (format_emit_dual_model). The anchor/CoT rows need the DRAWN
    # base family at GENERATION time so pvp_selfplay._build_row picks the right emit surface (Qwen
    # renders assistant `content` beside a structured tool_call; Llama drops it and must text-encode).
    # Thread the detected family via a POSIX env prefix (run_cmd_with_log uses shell=True) ONLY when
    # an anchor/CoT flag is on — default (both flags unset) leaves generate_cmd byte-identical.
    _anchor_on = os.getenv("PVP_ACTION_ANCHOR", "").strip().lower() in ("1", "true", "yes", "on")
    _cot_on = os.getenv("PVP_LABEL_COT", "").strip().lower() in ("1", "true", "yes", "on")
    if _anchor_on or _cot_on:
        _archl = (model_architecture or "").lower()
        _fam = "qwen" if "qwen" in _archl else ("llama" if "llama" in _archl else "")
        if _fam:
            generate_cmd = f"PVP_BASE_FAMILY={_fam} " + generate_cmd
            recipe_log(
                f"[sft-config] PVP_BASE_FAMILY={_fam} threaded to generate_cmd (anchor/CoT on)",
                tag="[sft-config] base family threaded",
            )

    recipe_log(f"Run command: {run_cmd}", tag="[sft-config] run command ready")

    return {
        "train_request": train_request,
        "run_cmd": run_cmd,
        "generate_cmd": generate_cmd,
    }
