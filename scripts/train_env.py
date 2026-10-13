"""Standalone training script for EnvTask (environment tournament).

Mirrors train_grpo.py's GRPOTrainer setup almost exactly, since trl 0.18.0's
GRPOTrainer has no built-in multi-turn rollout hook in this repo's pinned
version. The difference is the reward signal: instead of validating
user-submitted reward-function code strings against a static dataset, the
reward comes from core.environment_rollout.make_environment_reward_func,
which replays each episode against a live environment server.
"""

import json
import os

import torch
import transformers
from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import AutoTokenizer
from transformers.modeling_utils import is_deepspeed_zero3_enabled
from transformers.trainer_utils import is_main_process
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from trl import get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
from typing import Optional

from core.environment_rollout import get_environment_server_urls, make_environment_reward_func
from customized_trainer import CustomEvalSaveCallback, WhenToEvalHandler, resize_if_needed, set_generation_config
from tokenize_env import get_dataset
from utility import log_info


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(GRPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


def truncate_prompts(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def truncation_function(examples):
        batch_encoding = tokenizer(
            examples["prompt"], truncation=False, padding=False, add_special_tokens=True
        )
        truncated_token_ids_batch = [
            token_ids[-max_length:] if len(token_ids) > max_length else token_ids
            for token_ids in batch_encoding["input_ids"]
        ]
        return {"prompt": tokenizer.batch_decode(truncated_token_ids_batch, skip_special_tokens=True)}

    return dataset.map(truncation_function, batched=True)


def main():
    argument_parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig))
    training_args, model_args = argument_parser.parse_args_into_dataclasses()

    if not get_environment_server_urls():
        raise RuntimeError(
            "ENVIRONMENT_SERVER_URLS is not set or empty; EnvTask training cannot run "
            "rollouts without at least one environment server."
        )

    miner_datasets_dir = os.environ.get("MINER_DATASETS_DIR")
    miner_datasets = os.environ.get("MINER_DATASETS")
    baseline_stats_path = os.environ.get("BASELINE_STATS_PATH")
    log_info(f"MINER_DATASETS_DIR={miner_datasets_dir} MINER_DATASETS={miner_datasets} BASELINE_STATS_PATH={baseline_stats_path}")

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    # model_path normally points at the SFT warm-start stage's output
    # (env_config.py chains SFT -> RL). That stage can fail independently
    # (e.g. axolotl API drift) without blocking RL training — fall back to
    # the original base model if no warm-started checkpoint was produced.
    model_path = train_request["model_path"]
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        fallback_path = train_request.get("fallback_model_path")
        log_info(f"SFT warm-start output not found at {model_path}, falling back to base model {fallback_path}")
        model_path = fallback_path
        train_request["model_path"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = os.path.join("datasets", f"env_train_{task_id}.json")
    dev_path = os.path.join("datasets", f"env_dev_{task_id}.json")

    train_ds = get_dataset(train_path, train_request["dataset_type"])
    dev_ds = get_dataset(dev_path, train_request["dataset_type"])

    log_info(f"world_size: {training_args.world_size}")

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    device_map = get_kbit_device_map() if quantization_config is not None else {"": device_string}
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        device_map = None

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)
    set_generation_config(train_request["model_name"], model)

    peft_config = get_peft_config(model_args)
    if "lora_model" in train_request:
        model = PeftModelForCausalLM.from_pretrained(
            model, train_request["lora_model"], is_trainable=True, **model_kwargs
        )

    if peft_config is None:
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Created output directory: {training_args.output_dir}")

    training_args.save_only_model = True
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    max_prompt_length = train_request.get("max_prompt_length", 512)
    train_ds = truncate_prompts(train_ds, tokenizer, max_prompt_length)
    dev_ds = truncate_prompts(dev_ds, tokenizer, max_prompt_length)

    max_steps = train_request.get("max_steps", -1)
    env_reward_func = make_environment_reward_func(model, tokenizer)

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    total_steps_per_epoch = (
        len(train_ds)
        * training_args.num_generations
        // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    ) or 1
    if periodic_save_steps > total_steps_per_epoch:
        periodic_save_steps = -1

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[env_reward_func],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            CustomEvalSaveCallback(
                WhenToEvalHandler(
                    train_request["end_time"],
                    train_request["save_before_remaining_time"],
                    periodic_save_steps=periodic_save_steps,
                    steps_per_epoch=total_steps_per_epoch,
                    max_steps=max_steps,
                ),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
            )
        ],
    )

    trainer.train()

    if is_main_process(LOCAL_RANK):
        with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
            f.write("Success")


if __name__ == "__main__":
    main()
