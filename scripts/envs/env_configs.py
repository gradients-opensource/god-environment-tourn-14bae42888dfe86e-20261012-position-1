"""
Environment training configuration registry.

Hyperparameter resolution order for (env, mode, size_label):
  1. mode_cfg.per_size[size_label]  → all 5 SizeHyperparams fields (explicit)
  2. DEFAULT_HYPERPARAMS[size_label] + mode_cfg.num_generations if set
  Then mode_cfg scalars (temperature, top_k, initial_max_turn, …) on top.
"""

# ---------------------------------------------------------------------------
# trl.experimental.openenv shim — injected before env imports so every env
# file that does `from trl.experimental.openenv import generate_rollout_completions`
# gets this implementation when TRL < the version that ships experimental/.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types


def _generate_rollout_completions(trainer, prompts, as_chat=False, **kwargs):
    import torch
    import torch.nn.functional as F

    tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    model = trainer.model
    results = []
    for prompt in prompts:
        text = (
            tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if as_chat
            else str(prompt)
        )
        enc = tok(text, return_tensors="pt")
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        max_new = getattr(getattr(trainer, "args", None), "max_completion_length", 512) or 512
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )
        pids = enc["input_ids"][0].tolist()
        cids = out.sequences[0][plen:].tolist()
        lps = [F.log_softmax(s[0], dim=-1)[c].item() for s, c in zip(out.scores or [], cids)]
        results.append({"prompt_ids": pids, "completion_ids": cids, "logprobs": lps})
    return results


if "trl.experimental" not in _sys.modules:
    import trl as _trl
    _exp = _types.ModuleType("trl.experimental")
    _exp.__path__ = []
    _sys.modules["trl.experimental"] = _exp
    _trl.experimental = _exp

if "trl.experimental.openenv" not in _sys.modules:
    import trl as _trl
    _oe = _types.ModuleType("trl.experimental.openenv")
    _oe.generate_rollout_completions = _generate_rollout_completions
    _sys.modules["trl.experimental.openenv"] = _oe
    _trl.experimental.openenv = _oe
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field, fields
from typing import Callable

from envs.alf_world_env import (
    alfworld_rollout_first_prompt_and_completion_parallelized as _alf_rollout_last,
    alfworld_rollout_full_prompt_and_completion_parallelized  as _alf_rollout_full,
    alfworld_rollout_reward_func                              as _alf_reward,
)
from envs.gin_rummy_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_rollout_last,
    rollout_reward_func                                        as _gin_reward,
    _curriculum_factory                                        as _gin_curriculum,
)
from envs.gin_rummy_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_opp_rollout_last,
    rollout_reward_func                                        as _gin_opp_reward,
    _curriculum_factory                                        as _gin_opp_curriculum,
)
from envs.gin_rummy_refined import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _gin_ref_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _gin_ref_rollout_last,
    rollout_reward_func                                        as _gin_ref_reward,
    _curriculum_factory                                        as _gin_ref_curriculum,
)
from envs.goof_spiel_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _goof_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _goof_rollout_last,
    rollout_reward_func                                        as _goof_reward,
    _curriculum_factory                                        as _goof_curriculum,
)
from envs.leduc_poker_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_rollout_last,
    rollout_reward_func                                        as _leduc_reward,
    _curriculum_factory                                        as _leduc_curriculum,
)
from envs.leduc_poker_opponent_modeling import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _leduc_opp_rollout_last,
    rollout_reward_func                                        as _leduc_opp_reward,
    _curriculum_factory                                        as _leduc_opp_curriculum,
)
from envs.clobber_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _clob_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _clob_rollout_last,
    rollout_reward_func                                        as _clob_reward,
    _curriculum_factory                                        as _clob_curriculum,
)
from envs.liar_dice_env import (
    rollout_full_prompt_and_completion_parallelized_curriculum as _liar_rollout_full,
    rollout_last_prompt_and_completion_parallelized_curriculum as _liar_rollout_last,
    rollout_reward_func                                        as _liar_reward,
    _curriculum_factory                                        as _liar_curriculum,
)


@dataclass
class SizeHyperparams:
    """Co-tuned VRAM/dynamics params for one (env, mode, size). All fields required."""
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations:             int
    vllm_gpu_memory_utilization: float
    beta:                        float

    def apply(self, args) -> None:
        for f in fields(self):
            setattr(args, f.name, getattr(self, f.name))


@dataclass
class ModeConfig:
    """Per-mode overrides for one environment. None means use the default."""
    initial_max_turn:     int | None   = None
    rollouts_per_stage:   int | None   = None
    trainer_class:        "type | None" = None  # default: GRPOTrainer or ActionMaskedGRPOTrainer
    max_completion_length: int | None  = None   # default: 2048 (reasoning) or 16
    num_generations:      int | None   = None
    temperature:          float | None = None
    top_k:                int | None   = None
    per_size: dict[str, SizeHyperparams] = field(default_factory=dict)

    def apply_scalars(self, args) -> None:
        for attr in ("initial_max_turn", "rollouts_per_stage", "temperature", "top_k"):
            val = getattr(self, attr)
            if val is not None:
                setattr(args, attr, val)


@dataclass
class EnvTrainingConfig:
    rollout_full: Callable
    rollout_last: Callable
    reward_func:  Callable
    curriculum_factory:   Callable | None = None
    vllm_max_model_length: int = 5248  # reasoning mode adds 2048 on top at runtime
    num_iterations:       int = 2
    reasoning:   ModeConfig = field(default_factory=ModeConfig)
    no_mask:     ModeConfig = field(default_factory=ModeConfig)
    full_prompt: ModeConfig = field(default_factory=ModeConfig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EnvTrainingConfig] = {
    "clobber": EnvTrainingConfig(
        rollout_full=_clob_rollout_full,
        rollout_last=_clob_rollout_last,
        reward_func=_clob_reward,
        curriculum_factory=_clob_curriculum,
        reasoning=ModeConfig(initial_max_turn=4, num_generations=4, temperature=1.0, top_k=-1),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=1024, num_generations=4, temperature=1.0, top_k=-1),
        full_prompt=ModeConfig(initial_max_turn=6, num_generations=4, temperature=1.0, top_k=-1),
    ),
    "goof_spiel": EnvTrainingConfig(
        rollout_full=_goof_rollout_full,
        rollout_last=_goof_rollout_last,
        reward_func=_goof_reward,
        curriculum_factory=_goof_curriculum,
        reasoning=ModeConfig(initial_max_turn=1, num_generations=4, temperature=1.1, top_k=-1),
        no_mask=ModeConfig(initial_max_turn=1, num_generations=4, temperature=1.1, top_k=-1),
        full_prompt=ModeConfig(num_generations=4, temperature=1.1, top_k=-1),
    ),
    "gin_rummy": EnvTrainingConfig(
        rollout_full=_gin_rollout_full,
        rollout_last=_gin_rollout_last,
        reward_func=_gin_reward,
        curriculum_factory=_gin_curriculum,
        num_iterations=3,
        reasoning=ModeConfig(initial_max_turn=10, num_generations=4, temperature=1.0, top_k=-1),
        no_mask=ModeConfig(initial_max_turn=6, rollouts_per_stage=640, num_generations=4, temperature=1.0, top_k=-1),
        full_prompt=ModeConfig(initial_max_turn=10, num_generations=4, temperature=1.0, top_k=-1),
    ),
    "gin_rummy_opponent_modeling": EnvTrainingConfig(
        rollout_full=_gin_opp_rollout_full,
        rollout_last=_gin_opp_rollout_last,
        reward_func=_gin_opp_reward,
        curriculum_factory=_gin_opp_curriculum,
        reasoning=ModeConfig(initial_max_turn=8, num_generations=4, temperature=1.0, top_k=-1),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=4, temperature=1.0, top_k=-1),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=4, temperature=1.0, top_k=-1),
    ),
    "gin_rummy_refined": EnvTrainingConfig(
        rollout_full=_gin_ref_rollout_full,
        rollout_last=_gin_ref_rollout_last,
        reward_func=_gin_ref_reward,
        curriculum_factory=_gin_ref_curriculum,
        reasoning=ModeConfig(initial_max_turn=8, num_generations=4, temperature=1.0, top_k=-1),
        no_mask=ModeConfig(initial_max_turn=4, rollouts_per_stage=512, num_generations=4, temperature=1.0, top_k=-1),
        full_prompt=ModeConfig(initial_max_turn=8, num_generations=4, temperature=1.0, top_k=-1),
    ),
    "liars_dice": EnvTrainingConfig(
        rollout_full=_liar_rollout_full,
        rollout_last=_liar_rollout_last,
        reward_func=_liar_reward,
        curriculum_factory=_liar_curriculum,
        reasoning=ModeConfig(rollouts_per_stage=1280, initial_max_turn=1, num_generations=4, temperature=2.0, top_k=5),
        no_mask=ModeConfig(rollouts_per_stage=1536, initial_max_turn=1, num_generations=4, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(rollouts_per_stage=1280, initial_max_turn=2, num_generations=4, temperature=2.0, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=4, gradient_accumulation_steps=4, num_generations=4, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "leduc_poker": EnvTrainingConfig(
        rollout_full=_leduc_rollout_full,
        rollout_last=_leduc_rollout_last,
        reward_func=_leduc_reward,
        curriculum_factory=_leduc_curriculum,
        reasoning=ModeConfig(num_generations=10, temperature=1.8, top_k=5),
        no_mask=ModeConfig(num_generations=10, temperature=1.8, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
        full_prompt=ModeConfig(num_generations=10, temperature=1.8, top_k=5, per_size={
            "2_4_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.3,  beta=0.01),
            "6_9_b": SizeHyperparams(per_device_train_batch_size=2, gradient_accumulation_steps=8, num_generations=8, vllm_gpu_memory_utilization=0.35, beta=0.01),
        }),
    ),
    "leduc_poker_opponent_modeling": EnvTrainingConfig(
        rollout_full=_leduc_opp_rollout_full,
        rollout_last=_leduc_opp_rollout_last,
        reward_func=_leduc_opp_reward,
        curriculum_factory=_leduc_opp_curriculum,
        reasoning=ModeConfig(num_generations=8, temperature=2.0, top_k=5),
        no_mask=ModeConfig(num_generations=8, temperature=2.0, top_k=5),
        full_prompt=ModeConfig(num_generations=8, temperature=2.0, top_k=5),
    ),
    "alfworld": EnvTrainingConfig(
        rollout_full=_alf_rollout_full,
        rollout_last=_alf_rollout_last,
        reward_func=_alf_reward,
        reasoning=ModeConfig(num_generations=4, temperature=1.0, top_k=-1),
        no_mask=ModeConfig(num_generations=4, temperature=1.0, top_k=-1),
        full_prompt=ModeConfig(num_generations=4, temperature=1.0, top_k=-1),
    ),
}


# ---------------------------------------------------------------------------
# Variant routing
# ---------------------------------------------------------------------------

# Change this to select a non-default variant for a base environment name.
#
# ⚠️ GRPO-DEAD under our SFT-only run — do NOT uncomment to chase the boss-vs-past-winner diff's
# "#1: re-enable opponent-modeling variants" recommendation. get_env_config (the ONLY consumer of
# _VARIANT_OVERRIDES, below) is imported only by the GRPO path (grpo_env_config.py, train_grpo_env.py).
# We ship ENV_SKIP_GRPO=1, so the SFT trajectory path (generate_trajectories.py → sft_env_configs →
# make_selfplay_generator) uses the BASE env name and NEVER reads this dict: uncommenting changes ZERO
# SFT bytes (rows byte-identical) and only wastes a paid Docker rebuild + false "we fixed it" confidence.
# The only SFT-reachable opponent modeling is baking it INTO the teacher (e.g. gin PIMC world-sampling
# priors from public pick/decline signal), not this override. Verified 2026-09-20 (per-game audit).
_VARIANT_OVERRIDES: dict[str, str] = {
    # "gin_rummy": "gin_rummy_opponent_modeling",
    # "gin_rummy": "gin_rummy_refined",
    # "leduc_poker": "leduc_poker_opponent_modeling",
}


def get_env_config(name: str) -> EnvTrainingConfig:
    """Look up the training config for a named environment.

    If ``name`` has an entry in ``_VARIANT_OVERRIDES``, that registry key is
    used instead — allowing a single code-level switch between implementations
    without changing the caller's environment name.

    Raises ``ValueError`` with a helpful message if the name is unknown.
    """
    resolved = _VARIANT_OVERRIDES.get(name, name)
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown environment: {name!r} (resolved to {resolved!r}). "
            f"Known environments: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[resolved]
