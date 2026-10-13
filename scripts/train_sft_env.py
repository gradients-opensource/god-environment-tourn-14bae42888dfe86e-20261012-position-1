"""
SFT trainer for EnvTask imitation learning.

Baseline: train_instruct.py (production machinery: callbacks, batch-size adjustment,
  success.txt, LoRA helpers).
Differences from train_instruct.py:
  1. Dataset: loaded via load_from_disk (HF DatasetDict) instead of MyDataset.
  2. Trainer: SFTTrainer with tokenize_and_mask (assistant-only loss) instead of Trainer.
"""

import datetime
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
import transformers
import datasets as hf_datasets
from datasets import DatasetDict, Features, Sequence, Value, load_from_disk
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import is_main_process
from trl import SFTConfig, SFTTrainer

from customized_trainer import (
    CustomEvalSaveCallback,
    WhenToEvalHandler,
    resize_if_needed,
    set_generation_config,
)
from state_manager import get_state, set_state
from submission_utils import insurance_save_step, reanchor_insurance_step
from utility import log_info
from budget_controller import CLOCK_STOP, ClockStop, wsd_lr_multiplier
from envs.chat_template_compat import assistant_tool_calls_ok

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def _mark_sft_skipped(output_dir: str, reason: str) -> None:
    """Make a silent SFT warm-start skip LOUD and post-hoc visible.

    Both skip paths in main() ``return`` BEFORE any training, so the container
    ships the UNTRAINED base model — which emits no eval-format moves and scores
    ~0 (the exact-0.0 forfeit signature we hit on the Llama-3.1-8B arena while
    validating only on 3B; see [[live-tournament-20260907]]). A single log_info
    scrolls past in a busy trainer log, so ALSO print a banner and drop a
    ``SFT_SKIPPED.txt`` marker next to the output — a post-mortem can then SEE at
    a glance that the warm-start never ran instead of guessing from a 0.0 score.
    """
    banner = (
        "\n================ SFT WARM-START SKIPPED ================\n"
        f"  reason: {reason}\n"
        "  -> the UNTRAINED base model will be returned (expect ~0 eval score)\n"
        "========================================================\n"
    )
    log_info(banner)
    try:
        if is_main_process(LOCAL_RANK):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "SFT_SKIPPED.txt"), "w") as f:
                f.write(reason + "\n")
    except Exception as exc:  # marker is diagnostic-only — never let it abort
        log_info(f"[sft-skip] could not write SFT_SKIPPED.txt marker: {exc}")


class WSDClockCallback(transformers.TrainerCallback):
    """Wall-clock-driven WSD learning rate: warmup -> stable -> cosine decay, keyed
    on the SFT time window [train start, anneal deadline] rather than a step count.

    This decouples the LR schedule from how many steps the (unknown) validator
    hardware actually manages: we run the stable phase and trigger the decay near
    the deadline, so the final/saved checkpoint is always properly annealed inside
    the budget — fast box => more steps before decay, slow box => fewer, same
    shape either way. Sets the optimizer LR absolutely on every step (so it
    overrides the HF 'constant' scheduler underneath)."""

    def __init__(self, base_lr, end_time_str, save_before_minutes,
                 warmup_frac, decay_frac, min_mult):
        self.base_lr = float(base_lr)
        self.end_time_str = end_time_str
        self.save_before_minutes = float(save_before_minutes or 0)
        self.warmup_frac = float(warmup_frac)
        self.decay_frac = float(decay_frac)
        self.min_mult = float(min_mult)
        self._start = None
        self._deadline = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = datetime.datetime.now(datetime.timezone.utc)
        try:
            end = datetime.datetime.strptime(self.end_time_str, "%Y-%m-%d %H:%M:%S")
            end = end.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            end = self._start + datetime.timedelta(hours=1)
        # Anneal must complete by the time the periodic-save callback writes the
        # final checkpoint (save_before_remaining_time minutes before the deadline).
        self._deadline = end - datetime.timedelta(minutes=self.save_before_minutes)

    def _progress(self):
        if self._start is None or self._deadline is None:
            return 0.0
        total = (self._deadline - self._start).total_seconds()
        if total <= 0:
            return 1.0
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - self._start).total_seconds() / total

    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return
        lr = self.base_lr * wsd_lr_multiplier(
            self._progress(), self.warmup_frac, self.decay_frac, self.min_mult
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def clock_cosine_multiplier(step: int, total_steps: int, warmup_steps: int, time_progress: float,
                            min_lr_rate: float) -> float:
    """cosine_with_min_lr (HF: linear warmup, then min + (1-min)*0.5*(1+cos(pi*p))) where the decay
    progress p = max(step progress, wall-clock progress), clamped to [0, 1]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    step_p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    p = min(1.0, max(0.0, step_p, time_progress))
    return min_lr_rate + (1.0 - min_lr_rate) * 0.5 * (1.0 + math.cos(math.pi * p))


class ClockCosineCallback(transformers.TrainerCallback):
    """The validated cosine_with_min_lr schedule, but annealed against the SFT deadline too.

    The step schedule assumes the whole planned run fits the budget. When the deadline guard cuts the
    run mid-epoch (the tournament's 2-GPU budget does: fixed costs like data gen do not halve), the
    published checkpoint would still sit at a high LR. Here the decay progress is the max of step
    progress and clock progress over [train start, end_time - save buffer], so the last checkpoint is
    annealed either way. A run that finishes its steps before the clock follows the plain step cosine.
    Sets the optimizer LR on every step (HF's scheduler is switched to 'constant' by the caller).

    With ``steps_per_epoch`` given (budget_controller.CLOCK_STOP), the planned epochs are an upper bound:
    after the probe steps it measures s/step, moves the cosine horizon to the last step that fits before
    the deadline and saves + stops there, so a slow box anneals exactly at its stop step. ``on_stop_step``
    (optional) is called once with that stop step on every rank (see _reanchor_insurance)."""

    def __init__(self, base_lr, warmup_steps, min_lr_rate, total_steps, end_time_str, save_before_minutes,
                 steps_per_epoch=None, min_epochs=None, on_stop_step=None):
        self.on_stop_step = on_stop_step
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps or 0)
        self.min_lr_rate = float(min_lr_rate)
        self.total_steps = max(1, int(total_steps or 1))
        self.end_time_str = end_time_str
        self.save_before_minutes = float(save_before_minutes or 0)
        self.steps_per_epoch = int(steps_per_epoch or 0)
        self.min_epochs = float(min_epochs or 0)
        self._clock_stop = ClockStop(self.total_steps) if (CLOCK_STOP and self.steps_per_epoch > 0) else None
        self._start = None
        self._deadline = None
        self._last_logged = -1

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = datetime.datetime.now(datetime.timezone.utc)
        try:
            end = datetime.datetime.strptime(self.end_time_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=datetime.timezone.utc)
            self._deadline = end - datetime.timedelta(minutes=self.save_before_minutes)
        except Exception:
            self._deadline = None  # no usable deadline -> plain step cosine

    def _time_progress(self) -> float:
        if self._start is None or self._deadline is None:
            return 0.0
        total = (self._deadline - self._start).total_seconds()
        if total <= 0:
            return 1.0
        return (datetime.datetime.now(datetime.timezone.utc) - self._start).total_seconds() / total

    @staticmethod
    def _rank0_value(value: float, args) -> float:
        """Under torchrun every rank must apply the SAME LR (DDP never re-syncs parameters), but each
        reads its own clock; take rank 0's value (one scalar broadcast per optimizer step). Callbacks
        run at the same program point on every rank, so the collective is safe. No-op when not DDP."""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                device = getattr(args, "device", None) or (
                    torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
                t = torch.tensor([float(value)], dtype=torch.float64, device=device)
                dist.broadcast(t, src=0)
                return float(t.item())
        except Exception as exc:
            log_info(f"[clock-cosine] rank sync skipped: {exc}")
        return value

    def on_step_begin(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return
        tp = self._rank0_value(self._time_progress(), args)
        mult = clock_cosine_multiplier(state.global_step, self.total_steps, self.warmup_steps, tp,
                                       self.min_lr_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = self.base_lr * mult
        # HF logs 'learning_rate' from the (constant) scheduler's last LR; report the applied one.
        sched = kwargs.get("lr_scheduler")
        if sched is not None and hasattr(sched, "_last_lr"):
            sched._last_lr = [pg["lr"] for pg in optimizer.param_groups]
        if state.global_step - self._last_logged >= 50 and is_main_process(LOCAL_RANK):
            self._last_logged = state.global_step
            step_p = max(0.0, (state.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            log_info(f"[clock-cosine] step {state.global_step}/{self.total_steps} "
                     f"(decay progress: step {step_p:.2f}, clock {tp:.2f}) lr={self.base_lr * mult:.3e} (x{mult:.3f})")

    def on_step_end(self, args, state, control, **kwargs):
        plan = self._clock_stop
        if plan is None or self._deadline is None:
            return
        if plan.stop_step is not None:
            if state.global_step >= plan.stop_step:
                control.should_save = True
                control.should_training_stop = True
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        stop = plan.observe(state.global_step, now.timestamp(), (self._deadline - now).total_seconds())
        if stop is None:
            return
        # Every rank reaches the probe end at the same step; all of them stop where rank 0's clock says.
        plan.stop_step = int(self._rank0_value(float(stop), args))
        planned, self.total_steps = self.total_steps, plan.stop_step
        if is_main_process(LOCAL_RANK):
            epochs = plan.stop_step / self.steps_per_epoch
            warn = f" WARN below the {self.min_epochs:g}-epoch floor" if epochs < self.min_epochs else ""
            log_info(f"[epoch-plan] {plan.s_per_step:.2f} s/step: stop at step {plan.stop_step}/{planned} "
                     f"({epochs:.2f} epochs){warn}")
        if self.on_stop_step is not None:
            try:  # a hook failure must never take the training step down
                self.on_stop_step(plan.stop_step)
            except Exception as exc:
                log_info(f"[epoch-plan] on_stop_step hook failed: {exc!r}")


def _reanchor_insurance(handler, stop_step: int) -> None:
    """ClockCosineCallback on_stop_step hook: the insurance step was computed from the PLANNED steps; a clock
    stop at or before it would end the run with no early save at all (a crash before the stop then ships
    nothing), so re-derive it from the stop step (submission_utils.reanchor_insurance_step). Every rank gets
    the same rank-0 stop step, so the extra eval/save stays in lockstep under DDP."""
    new_step = reanchor_insurance_step(handler.insurance_step, stop_step)
    if new_step != handler.insurance_step:
        if is_main_process(LOCAL_RANK):
            log_info(f"[insurance] clock stop at step {stop_step} is not after the planned insurance step "
                     f"{handler.insurance_step} -> insurance save at step {new_step}")
        handler.insurance_step = new_step


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFTEnvTrainingArgs(SFTConfig):
    request_path: Optional[str] = field(default=None)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


# ---------------------------------------------------------------------------
# Model helpers (mirrored from train_instruct.py)
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, torch.nn.Linear)):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def load_lora_model(training_args: SFTEnvTrainingArgs, model_path: str,
                    lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )

    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split() if m.strip()]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=PeftTaskType.CAUSAL_LM,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    return model


def load_model(training_args: SFTEnvTrainingArgs, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        log_info("Using LIGER kernel")
        model_class = AutoLigerKernelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
    )
    return model


# ---------------------------------------------------------------------------
# Tokenisation + masking (from game-trajectories-sft/train.py)
# ---------------------------------------------------------------------------

# --- experimental sequence PACKING (ENV_PACK=1, default OFF) --------------------------------
# Short game rows (one board + one move) waste most of a 4096-token sequence to padding, so only
# ~10k of the 150k warm-start pool trains in a 1.5h round. Packing several rows per sequence trains
# MANY more of the pool in the same wall-clock. Rows in a pack must NOT attend across each other:
# we reset position_ids to 0 at each row boundary so FlashAttention-2 takes the position_ids-packing
# path and builds a BLOCK-DIAGONAL mask (attention_mask stays all-ones so FA2 does NOT take the
# unpad path, which would ignore the resets). Loss stays assistant-only via the per-row masks.
# DEFAULT ON since the 2026-09-24 VM A/B (same 40k rows, Qwen2.5-7B, h2h vs the real R1 winners):
# gin vs our group's winner 36% -> 52%, vs another past winner 33% -> 47%, intercode proxy up. Packing is also the ONLY
# assistant-only-loss path: the unpacked path loses `assistant_masks` to HF column pruning and
# silently trains FULL-SEQUENCE loss, where decision tokens are just ~2% of all tokens. main()
# turns it back off when flash-attention is disabled (row isolation needs FA2). ENV_PACK=0 = old path.
_PACK = os.getenv("ENV_PACK", "1").strip() == "1"

# Render each row's tools the way the validator's SGLang v0.5.14 server does (Tool.model_dump():
# description-first + strict/defer_loading keys), so the training prompt's <tools> block is byte-identical
# to the eval prompt. OFF until A/B'd: models trained on the plain render already emit identical calls
# under the SGLang render (CPU peek 2026-09-26), so this only removes a small train/eval prompt shift.
_SGLANG_TOOLS = os.getenv("SFT_SGLANG_TOOLS", "0").strip() == "1"


def _publish_policy() -> str:
    """SFT_PUBLISH_POLICY (latest|best, default latest). A typo falls back to latest, the default this
    trainer is validated with, instead of silently reviving the noisy lowest-eval_loss selection."""
    raw = os.getenv("SFT_PUBLISH_POLICY", "latest").strip().lower()
    if raw not in ("latest", "best"):
        log_info(f"[publish] SFT_PUBLISH_POLICY={raw!r} not in (latest, best) -> latest")
        return "latest"
    return raw


def _pack_batch_size(world_size: int = 1) -> int:
    """Per-device batch of PACKED sequences. PACK_BATCH (default 16) is the GLOBAL number of packs per
    micro-step (x gradient_accumulation_steps, which is 1 on the default path) — the operating point every
    VM validation ran at (1 GPU x 16) — so it is split across the ranks (2 GPUs -> 8 each) instead of
    doubling the global batch and halving the optimizer steps. The opt-in eff-batch knobs
    (SFT_EFFECTIVE_BATCH / AUTO_1GPU_BATCH / SFT_GA) size GA from the unpacked CLI batch, not from packs.
    Halved once per OOM retry: text_trainer halves the CLI --per_device_train_batch_size after an OOM
    and exports the count as OOM_HALVINGS; the packed path overrides the CLI batch, so it must apply
    the same halving or every retry would OOM again at the same step."""
    try:
        halvings = max(0, int(os.getenv("OOM_HALVINGS", "0") or 0))
    except ValueError:
        halvings = 0
    per_device = max(1, int(os.getenv("PACK_BATCH", "16")) // max(1, int(world_size or 1)))
    return max(1, per_device >> halvings)


def _pack_rows(rows, max_length):
    """Greedy first-fit: concat tokenized (input_ids, assistant_masks) rows into <=max_length packs.
    Builds `labels` HERE (id where assistant else -100) rather than carrying `assistant_masks` — labels
    is a model-forward arg so the Trainer never drops it (assistant_masks IS dropped). position_ids
    reset to 0 at each row boundary = the block-diagonal signal for FA2."""
    packs, cur_ids, cur_lab, cur_pos = [], [], [], []
    for ids, mask in rows:
        if cur_ids and len(cur_ids) + len(ids) > max_length:
            packs.append({"input_ids": cur_ids, "labels": cur_lab, "position_ids": cur_pos})
            cur_ids, cur_lab, cur_pos = [], [], []
        ids = list(ids)
        cur_ids = cur_ids + ids
        cur_lab = cur_lab + [ids[i] if mask[i] else -100 for i in range(len(ids))]
        cur_pos = cur_pos + list(range(len(ids)))
    if cur_ids:
        packs.append({"input_ids": cur_ids, "labels": cur_lab, "position_ids": cur_pos})
    return packs


class _PackedCollator:
    """Pad packed sequences to the batch max. all-ones attention_mask so FA2 segments by position_ids
    (block-diagonal), NOT by unpadding; the pad tail is its own isolated position segment with -100
    labels, so it costs no loss and leaks into no real row. Reads only model-forward columns
    (input_ids/labels/position_ids) so the Trainer's column pruning can't drop them."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

    def __call__(self, features):
        import torch
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn, position_ids = [], [], [], []
        for f in features:
            ids, lab, pos = list(f["input_ids"]), list(f["labels"]), list(f["position_ids"])
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * (len(ids) + pad))          # all ones -> FA2 uses position_ids for segmentation
            position_ids.append(pos + list(range(pad)))  # pad tail = its own isolated position segment
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }


def _render_probe(tokenizer) -> list[str]:
    """Confirm the DRAWN base's chat template still emits a parseable tool call for BOTH
    surfaces we ship — a game_action (games) and execute_bash/submit (intercode) — and does
    NOT double-encode the arguments as a JSON string literal.

    WHY: on a Llama draw a DROPPED tool call forfeits every turn (the exact-0.0 we hit
    2026-09-07), and the blind try/except in _process returns an EMPTY row instead of raising,
    so a template that silently mangles our emit surface is invisible until the 0.0 lands. This
    probe surfaces it BEFORE tokenization. It is a by-shape check (no GPU); the definitive gate
    is a live SGLang serve under BOTH --tool-call-parser llama3 AND the Qwen parser.

    Returns a list of human-readable problems ('' -> clean). The caller decides warn vs. fail.
    """
    try:
        from envs.pvp_format import build_pvp_tools, tools_to_openai
        from envs.intercode_format import build_intercode_action_tools
        _game_tools = tools_to_openai(build_pvp_tools([0, 1, 2]))
        _bash_tools = tools_to_openai(build_intercode_action_tools())
    except Exception as exc:
        # Probe SETUP failure is diagnostic-only: never let it abort the warm-start. The per-row
        # try/except downstream is still the belt-and-braces net for real render breaks.
        log_info(f"[render-probe] skipped (setup failed: {exc!r}) — relying on per-row render net")
        return []

    _base_family = os.getenv("PVP_BASE_FAMILY", "").strip().lower()
    # arguments as DICTS — the same shape _process feeds the template AFTER it decodes the
    # JSON-string arguments below, so the probe exercises the real render path.
    probes = [
        ("game_action", {"action_id": 1}, _game_tools),
        ("execute_bash", {"command": "ls /testbed"}, _bash_tools),
        # Multi-line command: a template/parser that emits the embedded newline raw (instead of the
        # JSON-escaped \n) produces an unparseable call -> the model learns a surface that forfeits.
        ("execute_bash", {"command": "cd /testbed\nfind . -name '*.log' | wc -l"}, _bash_tools),
        ("submit", {}, _bash_tools),
    ]
    problems: list[str] = []
    for name, args, tools in probes:
        # Render a PROMPT (user only, generation prompt open) and the FULL turn, then diff:
        # the ASSISTANT EMIT DELTA is the exact bytes the model is trained to produce and the
        # only bytes SGLang's parser reads at eval. Checking the whole render instead would
        # (a) always find every tool NAME in the system-prompt tool schema (masking a dropped
        # assistant call) and (b) feed the first tool-SCHEMA object to the llama3 parser, whose
        # recovered name is a schema name, not the call — a guaranteed false positive on
        # game_action/submit. Mirrors diagnose_8b_sft._generated_assistant_text (OUR primitive).
        prompt_msgs = [{"role": "user", "content": "probe"}]
        full_msgs = prompt_msgs + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": args}}]},
        ]
        try:
            prefix = tokenizer.apply_chat_template(
                prompt_msgs, tools=tools, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                full_msgs, tools=tools, tokenize=False, add_generation_prompt=False)
        except Exception as exc:
            problems.append(f"{name}: chat template raised ({exc!r})")
            continue
        emitted = full[len(prefix):] if full.startswith(prefix) else full
        if name not in emitted:
            problems.append(f"{name}: tool-call name absent from the ASSISTANT emit surface "
                            f"(template dropped the call — parser would forfeit)")
        # Double-encode signature: arguments rendered as an ESCAPED JSON string literal
        # (e.g. "{\"action_id\": 1}") instead of a bare object — only appears when a dict was
        # stringified then re-serialized by the template's tojson filter on a str.
        if args and '\\"' in emitted and any(f'\\"{k}\\"' in emitted for k in args):
            problems.append(f"{name}: arguments double-encoded as a string literal (template tojson on a str)")
        # Round-trip through OUR llama3 parser primitive when the base is Llama-shaped, on the
        # SAME assistant delta the eval parser reads.
        if _base_family != "qwen":
            try:
                from envs.diagnose_8b_sft import _parse_toolcall_llama3
                parsed = _parse_toolcall_llama3(emitted)
                if parsed is None or parsed[0] != name:
                    problems.append(f"{name}: llama3 parser did not recover the call from the assistant emit surface")
            except Exception:
                pass  # parser primitive unavailable — name/double-encode checks already ran
    return problems


def tokenize_and_mask(dataset: DatasetDict, tokenizer, max_length: int = 4096) -> DatasetDict:
    """Apply chat template and mask non-assistant tokens so loss is assistant-only."""
    # Render-probe forfeit guard (runs ONCE, before any row is tokenized). Default: LOUD
    # warning only — a false positive must NOT abort a live run (a silent eval forfeit is bad,
    # but hard-failing training guarantees a 0). SFT_RENDER_PROBE_STRICT=1 HARD-FAILS instead:
    # use it on the VM validation run, where a definite fail is the signal we WANT before we
    # flip any default-on. This is orthogonal to the per-row try/except below (which drops a
    # single unrenderable row); the probe catches a base-wide template break the drop hides.
    # Diagnostic-only (warn by default; it never alters training data). Runs on every build so a
    # format-dropping base is caught BEFORE the 0.0 forfeit; SFT_RENDER_PROBE=0 disables it as an
    # escape hatch if it ever misbehaves on a live run.
    _probe_problems = [] if os.getenv("SFT_RENDER_PROBE", "1").strip() == "0" else _render_probe(tokenizer)
    if _probe_problems:
        _banner = ("render-probe: base chat template may DROP/MANGLE our tool-call emit "
                   "surface:\n  - " + "\n  - ".join(_probe_problems))
        if os.getenv("SFT_RENDER_PROBE_STRICT", "").strip() == "1":
            raise RuntimeError(
                _banner + "\n(SFT_RENDER_PROBE_STRICT=1 -> hard fail; confirm the fix on a live "
                "SGLang serve under both the llama3 and Qwen tool-call parsers)")
        log_info("[WARN] " + _banner + "\n  -> continuing (set SFT_RENDER_PROBE_STRICT=1 to "
                 "hard-fail; confirm on a live SGLang serve before relying on this base)")

    def _process(example):
        msgs = example["messages"]
        tools = example.get("tools") or None
        if isinstance(tools, str):
            # Tools are stored as a JSON string, not a list of dicts: Arrow unifies
            # struct schemas across a list column's elements, which corrupts
            # heterogeneous tool `parameters.properties` (e.g. a no-arg tool's `{}`
            # becomes `{"some_other_tool_arg": None}`). JSON-encoding avoids that.
            tools = json.loads(tools)
        if _SGLANG_TOOLS and tools:
            from envs.pvp_format import sglang_template_tools
            tools = sglang_template_tools(tools)
        # Same Arrow-struct-unification issue applies to tool_calls[].function.arguments
        # across envs (e.g. game envs' {"action_id": int} vs intercode's {"command": str}):
        # generators store `arguments` JSON-encoded as a string. Decode back to a dict
        # here so the chat template's `tojson` filter renders it correctly (it would
        # double-encode an already-JSON string).
        msgs = [
            {
                **m,
                "tool_calls": [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": (
                                json.loads(call["function"]["arguments"])
                                if isinstance(call["function"].get("arguments"), str)
                                else call["function"].get("arguments")
                            ),
                        },
                    }
                    for call in m["tool_calls"]
                ],
            }
            if m.get("tool_calls")
            else m
            for m in msgs
        ]
        try:
            ids = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=False)
            mask = [0] * len(ids)
            for i, msg in enumerate(msgs):
                if msg["role"] != "assistant":
                    continue
                p = len(tokenizer.apply_chat_template(msgs[:i],   tools=tools, tokenize=True, add_generation_prompt=True))
                r = len(tokenizer.apply_chat_template(msgs[:i+1], tools=tools, tokenize=True, add_generation_prompt=False))
                for j in range(p, r):
                    mask[j] = 1
        except Exception:
            # A single row the chat template can't render must NOT abort the whole
            # warm-start — that aborts SFT and falls back to the base model (0 score).
            # Emit an empty row and drop it right after the map. The
            # assistant_tool_calls_ok pre-filter already removes the known multi-tool-call
            # case; this is the belt-and-braces net for any other model-specific template
            # restriction we haven't seen (the assigned env model is not always Qwen).
            return {"input_ids": [], "assistant_masks": []}
        # rank-8 truncation fix, OPT-IN behind SFT_TAIL_TRUNCATE (default OFF => the ORIGINAL
        # head-keep with no empty-mask drop, byte-identical to HEAD). The supervised assistant
        # span sits at the END of the sequence, so head-truncation (ids[:max_length]) can drop
        # the whole target span -> an all-zero mask that trains nothing (observed on long
        # liars/memory rows). With the flag ON we keep the TAIL and drop any row whose supervised
        # tokens still didn't survive. Flip the default only after a VM A/B confirms the
        # co-trained student holds on both bases (this shapes training data, so it is gated like
        # every other recipe lever rather than changing the default silently).
        _tail_fix = os.getenv("SFT_TAIL_TRUNCATE", "").strip() == "1"
        if len(ids) > max_length:
            if _tail_fix:
                ids, mask = ids[-max_length:], mask[-max_length:]
            else:
                ids, mask = ids[:max_length], mask[:max_length]
        if _tail_fix and sum(mask) == 0:
            # Paired with the opt-in fix: a row with no supervised tokens trains nothing — drop
            # it via the empty-row path (the same schema-pinned drop the try/except above uses).
            return {"input_ids": [], "assistant_masks": []}
        return {"input_ids": ids, "assistant_masks": mask}

    # Use more of the container's CPUs (docker run gives --cpus=28) — num_proc=4 badly
    # underutilizes them for this per-row apply_chat_template work. Disable inner tokenizer
    # threading so the fork-based num_proc workers don't deadlock or oversubscribe.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    n_proc = min(16, os.cpu_count() or 4)
    log_info(f"Tokenizing + masking dataset ({len(dataset['train'])} rows, num_proc={n_proc})...")
    # FORCE the output Arrow schema instead of letting datasets INFER it. The try/except in _process
    # emits {"input_ids": [], "assistant_masks": []} for a row the chat template can't render; if such
    # an empty list is a num_proc shard's FIRST example, datasets infers that column as list<null>,
    # then the next real row (list<int64>) crashes with "Couldn't cast array of type int64 to null"
    # (non-deterministic by shard — one attempt reaches 99%, the next dies at 6%). Pinning the feature
    # types makes the empties valid-but-empty int lists. Kept columns reuse their own features
    # (unchanged by _process), so nothing downstream loses messages/tools.
    _in_features = next(iter(dataset.values())).features
    _out_features = Features({
        **{k: _in_features[k] for k in _in_features},
        "input_ids": Sequence(Value("int32")),
        "assistant_masks": Sequence(Value("int8")),
    })
    result = dataset.map(_process, num_proc=n_proc, desc="tokenize+mask", features=_out_features)
    # Drop rows the template rejected (emitted empty above) so a stray unrenderable row
    # can't slip an empty example into training — and so we see the count if any dropped.
    for split in list(result.keys()):
        n0 = len(result[split])
        result[split] = result[split].filter(lambda ex: len(ex["input_ids"]) > 0)
        if len(result[split]) < n0:
            log_info(f"[tokenize] dropped {n0 - len(result[split])} unrenderable row(s) from '{split}' → {len(result[split])}")
    log_info("Tokenizing + masking done.")
    if _PACK:
        from datasets import Dataset
        for split in list(result.keys()):
            rows = list(zip(result[split]["input_ids"], result[split]["assistant_masks"]))
            packs = _pack_rows(rows, max_length)
            result[split] = Dataset.from_list(packs)
        log_info(f"[pack] ENV_PACK=1 → packed '{list(result.keys())}' into "
                 f"{len(result['train'])} sequences (<= {max_length} tokens each)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _PACK
    argument_parser = transformers.HfArgumentParser((SFTEnvTrainingArgs, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()

    # Packing guards (packing is the default assistant-only-loss path, see _PACK above):
    if _PACK and training_args.disable_fa:
        # Rows in a pack are kept apart by resetting position_ids, which only flash_attention_2
        # honours; under eager/sdpa attention they would attend to each other -> unpacked path.
        _PACK = False
        log_info("[pack] flash-attention disabled for this model -> packing OFF (unpacked path)")
    if _PACK and getattr(training_args, "padding_free", False):
        # The tournament command passes --padding_free True (meant for the unpacked path). TRL does
        # not accept a custom collator together with padding_free, and _PackedCollator already
        # flattens the batch, so padding_free must be off whenever we pack.
        training_args.padding_free = False
        log_info("[pack] packing active -> padding_free forced False (the packed collator flattens)")

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load pre-generated trajectories and apply assistant masking
    if "dataset_path" not in train_request:
        _mark_sft_skipped(
            training_args.output_dir,
            "no dataset_path in train_request — SFT data generation produced nothing "
            "(check the generate_trajectories log for env-gen failures)",
        )
        return
    raw: DatasetDict = load_from_disk(train_request["dataset_path"])
    log_info(f"Dataset: train={len(raw['train'])}")
    # Drop rows a Llama-family chat template can't render: an assistant turn must carry
    # EXACTLY one tool_call or apply_chat_template raises "This model only supports single
    # tool-calls at once!" (Qwen is lenient, Llama is not). Every trajectory we generate is
    # single-call, so this is a defensive guard that drops nothing on our own rows — see
    # envs.chat_template_compat.assistant_tool_calls_ok. Model-agnostic: applied always so the
    # warm-start tokenizes identically on any whitelisted env model.
    _before = len(raw["train"])
    raw = raw.filter(lambda ex: assistant_tool_calls_ok(ex["messages"]))
    _dropped = _before - len(raw["train"])
    if _dropped:
        log_info(
            f"Filtered {_dropped}/{_before} row(s) with non-single-tool-call assistant turns "
            f"(Llama chat-template compat) → train={len(raw['train'])}"
        )
    if len(raw["train"]) == 0:
        _mark_sft_skipped(
            training_args.output_dir,
            f"0 of {_before} rows survived the single-tool-call compat filter — every "
            "generated row had a non-single-tool-call assistant turn (a format/generator bug, "
            "NOT expected: our generators emit exactly one game_action/execute_bash per turn)",
        )
        return
    # Cap the warm-start set. Trajectory generation can emit >1M windowed examples, but this SFT is
    # a time-budgeted warm-start (WSDClockCallback stops before the deadline) that only consumes a
    # fraction — tokenizing all of them upfront burns tournament wall-clock for rows never trained
    # on. Shuffle + cap to a representative slice. Override via train_request["max_sft_warmstart_examples"].
    max_warmstart = train_request.get("max_sft_warmstart_examples", 150_000)
    if max_warmstart and len(raw["train"]) > max_warmstart:
        log_info(f"Capping warm-start dataset {len(raw['train'])} → {max_warmstart} (shuffled)")
        raw["train"] = raw["train"].shuffle(seed=42).select(range(max_warmstart))
    dataset = tokenize_and_mask(raw, tokenizer, max_length=training_args.max_length or 4096)

    train_ds = dataset["train"]
    log_info(f"train_size: {len(train_ds)}")

    # LOUD row-keep summary through the chat template. If the assigned base model's
    # template can't render our rows (the try/except in tokenize_and_mask emits an
    # empty row that is then filtered), this ratio collapses — the exact failure mode
    # that silently ships a barely-trained / base model on an untested base (we
    # validated only on 3B). A large drop here on 8B is the #1 0.0 suspect.
    # (non-PACK only: in the PACK path the row count legitimately shrinks by packing,
    # which is not a template-reject signal.)
    _rows_in = len(raw["train"])
    _rows_out = len(train_ds)
    if not _PACK and _rows_in and _rows_out < _rows_in:
        _kept = 100.0 * _rows_out / _rows_in
        log_info(f"[sft-tokenize] chat-template KEPT {_rows_out}/{_rows_in} rows ({_kept:.1f}%) "
                 f"— a large drop means the base model's template rejects our tool_calls rows")

    # Assistant-mask coverage sanity check (non-PACK path). If a row's mask is empty
    # (no supervised assistant tokens) or covers ~everything, the assistant-only loss
    # is NOT doing what we expect — a silent training defect (e.g. assistant_masks
    # pruned by the Trainer, or the assistant span truncated past max_length). Sampled
    # to stay cheap. Expect a small positive supervised fraction; 0.000 or many zero-mask
    # rows => masking/truncation broken.
    if not _PACK and "assistant_masks" in train_ds.column_names and len(train_ds):
        _n = min(256, len(train_ds))
        _sample = train_ds.select(range(_n))
        _fracs, _empty = [], 0
        for _ids, _m in zip(_sample["input_ids"], _sample["assistant_masks"]):
            _tot = len(_ids)
            _sup = sum(_m)
            if _tot == 0 or _sup == 0:
                _empty += 1
            if _tot:
                _fracs.append(_sup / _tot)
        _mean = sum(_fracs) / len(_fracs) if _fracs else 0.0
        log_info(f"[sft-mask] assistant-token coverage over {_n} sampled rows: "
                 f"mean supervised fraction={_mean:.3f}, rows with ZERO supervised tokens={_empty}/{_n} "
                 f"(expect a small positive fraction; 0.000 / many-zero => masking or truncation broken)")

    if _PACK:
        # Packed sequences are full ~max_length (vs the short unpacked rows), so the unpacked
        # per-device batch OOMs. Scale it down (env-tunable PACK_BATCH); all step/deadline math below
        # uses this value. 16 keeps ~1.35x the real-row throughput of unpacked-batch-32 while fitting.
        training_args.per_device_train_batch_size = _pack_batch_size(training_args.world_size)
        log_info(f"[pack] per_device_train_batch_size -> {training_args.per_device_train_batch_size} "
                 f"(packed seqs are full-length; lower PACK_BATCH if it still OOMs)")

    # Batch size adjustment (mirrors train_instruct.py)
    original_steps = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    log_info(f"original_steps: {original_steps}")

    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = max(int(max_batch_size_theory), 1)

    if (training_args.per_device_train_batch_size > max_batch_size_theory
            and train_request.get("adjust_batch_size", True)):
        log_info(
            f"Reducing batch size {training_args.per_device_train_batch_size} → {max_batch_size_theory}"
        )
        training_args.per_device_train_batch_size = max_batch_size_theory

    # Load model
    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    try:
        model.config.use_cache = False
    except Exception:
        pass

    set_generation_config(train_request["model_name"], model)

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Output dir: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    max_steps = train_request.get("max_steps", -1)
    training_args.save_only_model = True

    # LoRA + DDP + gradient checkpointing: the default (reentrant) checkpoint re-runs forward in
    # backward, firing a LoRA param's DDP grad hook twice → "Expected to mark a variable ready
    # only once ... lora_B.default.weight has been marked as ready twice". Non-reentrant
    # checkpointing is DDP-safe. This mirrors train_grpo_env.py, which runs backward fine on this
    # same VM with use_reentrant=False and ddp_find_unused_parameters left as None.
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # LoRA freezes the base weights, so every trainable (LoRA) param gets a gradient every step —
    # there are never unused parameters. HF defaults DDP to find_unused_parameters=True for PEFT,
    # which the runtime warns is a wasted extra autograd traversal per step. Turn it off to speed
    # up each step (more steps fit in the fixed wall-clock budget).
    training_args.ddp_find_unused_parameters = False

    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(
        f"total_steps_per_epoch: {total_steps_per_epoch}; "
        f"total_steps_all_epochs: {total_steps_all_epochs}"
    )

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)

    success_file = os.path.join(training_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)

    checking_step = train_request.get("checking_step", 70)
    if checking_step >= total_steps_per_epoch:
        checking_step = max(total_steps_per_epoch - 2, 1)

    # Insurance publish: one extra save at ~20% of the planned steps (submission_utils.insurance_save_step),
    # so a crash before the first periodic save (200 steps, ~14 min into an 8B run) still leaves a trained
    # adapter to ship instead of nothing for env_submission.finalize_env_submission to recover.
    _planned_steps = max_steps if (max_steps or -1) > 0 else total_steps_all_epochs
    _insurance_step = insurance_save_step(_planned_steps)
    when_to_eval = WhenToEvalHandler(
        train_request["end_time"],
        train_request["save_before_remaining_time"],
        periodic_save_steps=periodic_save_steps,
        # max(1, ...): a near-empty dataset (<1 batch per rank) must not crash the
        # `global_step % steps_per_epoch` check at step 1 and burn every retry.
        steps_per_epoch=max(1, total_steps_per_epoch),
        max_steps=max_steps,
        insurance_step=_insurance_step,
    )

    callbacks = [
        CustomEvalSaveCallback(
            when_to_eval,
            train_request["submission_dir"],
            training_args.output_dir,
            train_request["model_name"],
            max_steps,
            checking_step=checking_step,
            total_steps_all_epochs=max(1, total_steps_all_epochs),
            end_time=train_request["end_time"],
            checking_mode=train_request.get("checking_mode", "none"),
            # The eval set below is 8 packs of the TRAIN set, so "lowest eval_loss" is noise: publish
            # the newest checkpoint instead (most data, most-annealed LR, and the deadline save).
            # SFT_PUBLISH_POLICY=best restores the lowest-eval_loss selection.
            publish_policy=_publish_policy(),
        )
    ]

    # Wall-clock WSD: let the clock-driven callback own the LR so the schedule is
    # robust to the validator's unknown hardware / step count. Neutralize HF's
    # step-count scheduler + warmup so they don't fight the callback.
    if train_request.get("lr_schedule") == "wsd":
        training_args.lr_scheduler_type = "constant"
        training_args.warmup_steps = 0
        training_args.warmup_ratio = 0.0
        callbacks.append(
            WSDClockCallback(
                base_lr=training_args.learning_rate,
                end_time_str=train_request["end_time"],
                save_before_minutes=train_request.get("save_before_remaining_time", 0),
                warmup_frac=train_request.get("wsd_warmup_frac", 0.05),
                decay_frac=train_request.get("wsd_decay_frac", 0.20),
                min_mult=train_request.get("wsd_min_mult", 0.10),
            )
        )
        log_info("WSD clock-triggered LR enabled (constant base + wall-clock warmup/stable/decay)")
    elif train_request.get("lr_schedule") == "cosine_clock":
        # Same cosine_with_min_lr curve as the CLI asks for (its warmup_steps and min_lr_rate are read
        # here), decayed by max(step, clock) progress — see ClockCosineCallback. HF's scheduler is set
        # to plain 'constant' (and its kwargs cleared: min_lr_rate is not a constant-schedule kwarg).
        _kw = training_args.lr_scheduler_kwargs or {}
        if isinstance(_kw, str):
            try:
                _kw = json.loads(_kw)
            except ValueError:
                _kw = {}
        _min_rate = float(_kw.get("min_lr_rate", 0.25)) if isinstance(_kw, dict) else 0.25
        _sched_total = max_steps if (max_steps or -1) > 0 else total_steps_all_epochs
        _warmup = training_args.warmup_steps or int(round((training_args.warmup_ratio or 0.0) * _sched_total))
        callbacks.append(
            ClockCosineCallback(
                base_lr=training_args.learning_rate,
                warmup_steps=_warmup,
                min_lr_rate=_min_rate,
                total_steps=_sched_total,
                end_time_str=train_request["end_time"],
                save_before_minutes=train_request.get("save_before_remaining_time", 0),
                steps_per_epoch=total_steps_per_epoch,
                min_epochs=train_request.get("min_epochs"),
                # a clock stop at/before the planned insurance step moves the insurance save before it
                on_stop_step=lambda stop_step: _reanchor_insurance(when_to_eval, stop_step),
            )
        )
        training_args.lr_scheduler_type = "constant"
        training_args.lr_scheduler_kwargs = {}
        training_args.warmup_steps = 0
        training_args.warmup_ratio = 0.0
        log_info(f"[clock-cosine] enabled: base lr {training_args.learning_rate:.3e}, warmup {_warmup}, "
                 f"min_lr_rate {_min_rate}, {_sched_total} steps, anneal by the earlier of steps / "
                 f"end_time-{train_request.get('save_before_remaining_time', 0)}min")

    # CustomEvalSaveCallback sets control.should_evaluate at periodic_save_steps (200) and before
    # the deadline; HF Trainer.evaluate() then needs an eval_dataset or it raises
    # "Trainer: evaluation requires an eval_dataset" — which crashed every run at step 200 and
    # burned all 5 retries. Mirror train_grpo_env.py: pass a tiny slice of the already-tokenized
    # train set. eval_strategy="no", so eval only fires from the callback (not on its own
    # schedule); a small slice keeps the periodic eval cost negligible.
    eval_slice = train_ds.select(range(min(8, len(train_ds))))

    # ENV_PACK: keep the packed columns (position_ids / assistant_masks) through to the custom
    # collator — the Trainer would otherwise drop columns not in the model's forward signature —
    # and use the packed collator that builds block-diagonal FA2 inputs. None => TRL's default.
    _collator = None
    if _PACK:
        training_args.remove_unused_columns = False
        # Skip TRL's dataset prep (it re-maps + can drop non-forward columns); our packed dataset is
        # already {input_ids, labels, position_ids} and the custom collator handles the rest.
        try:
            training_args.dataset_kwargs = {**(getattr(training_args, "dataset_kwargs", None) or {}),
                                            "skip_prepare_dataset": True}
        except Exception:
            pass
        _collator = _PackedCollator(tokenizer)

    # LOUD summary of the effective loss configuration. Assistant-only loss depends on the
    # per-row `assistant_masks` column surviving into the collator. In the default (non-PACK)
    # path with assistant_only_loss unset, HF's remove_unused_columns can prune the non-forward
    # `assistant_masks` column BEFORE the collator (this file's own _pack_rows comment notes
    # "assistant_masks IS dropped") -> the loss silently falls back to FULL-SEQUENCE. That is a
    # quality defect, not the 0.0 forfeit cause (full-seq loss still learns valid moves — 3B
    # scored non-zero), but it is worth surfacing. ENV_PACK=1 builds `labels` directly (a
    # model-forward arg) = provably assistant-only. Verify the installed TRL 0.18.0 behavior
    # with testing/diagnose_8b_sft.sh before trusting the default mask.
    _aol = bool(getattr(training_args, "assistant_only_loss", False))
    log_info(f"[sft-loss] packing={_PACK} assistant_only_loss={_aol} "
             f"remove_unused_columns={getattr(training_args, 'remove_unused_columns', None)}")
    if not _PACK and not _aol:
        log_info("[sft-loss] NOTE: non-PACK + assistant_only_loss=False — if the installed TRL "
                 "drops the `assistant_masks` column this trains FULL-SEQUENCE loss. "
                 "Proven-assistant-only alternative: ENV_PACK=1.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_slice,
        processing_class=tokenizer,
        data_collator=_collator,
        callbacks=callbacks,
    )

    trainer.train()

    if is_main_process(LOCAL_RANK):
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Training successfully done", "finish")


if __name__ == "__main__":
    main()
