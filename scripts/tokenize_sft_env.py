"""Tokenize EnvTask trajectory data (from prebaked or freshly-generated rollouts)
into the exact files train_instruct.py expects, reusing tokenize_instruct.py's
axolotl chat-template tokenization instead of reimplementing loss masking.

train_request["dataset"] must point to a JSON file of
{"messages": [{"role": "user"/"assistant", "content": "..."}]} rows — the same
shape scripts/core/environment_rollout.py.play_full_episode() produces.
"""

import json
from datetime import datetime

import typer
from transformers import AutoTokenizer

from tokenize_instruct import split_dataset, tokenize_dataset


CHAT_DATASET_TYPE = {
    "chat_template": "chatml",
    "chat_column": "messages",
    "chat_role_field": "role",
    "chat_content_field": "content",
    "chat_user_reference": "user",
    "chat_assistant_reference": "assistant",
}


def main(training_request_path: str):
    t1 = datetime.now()
    with open(training_request_path, "r") as file:
        training_request = json.load(file)

    train_request = training_request["train_request"]
    task_id = train_request["task_id"]
    total_path = train_request["dataset"]
    train_path = f"datasets/train_{task_id}.json"
    dev_path = f"datasets/dev_{task_id}.json"

    with open(total_path, "r") as f:
        total_rows = len(json.load(f))
    # trajectory datasets from rejection sampling are usually much smaller than
    # the default dev_size=200 text datasets assume, so scale it down
    dev_size = max(1, min(200, total_rows // 10))

    split_dataset(total_path, train_path, dev_path, dev_size=dev_size)

    config_path = "test_axolotl.yml"
    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = train_request.get("max_length", -1)

    tokenize_dataset(
        tokenizer, train_path, CHAT_DATASET_TYPE, config_path,
        f"datasets/train_tokenized_{task_id}.json", max_length=max_length,
    )
    tokenize_dataset(
        tokenizer, dev_path, CHAT_DATASET_TYPE, config_path,
        f"datasets/dev_tokenized_{task_id}.json", max_length=max_length,
    )

    t2 = datetime.now()
    print(f"[tokenize_sft_env] tokenization completed in {(t2 - t1).seconds}s", flush=True)


if __name__ == "__main__":
    typer.run(main)
