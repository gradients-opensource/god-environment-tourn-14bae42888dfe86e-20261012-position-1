#!/bin/bash

TASK_ID="19fccc14-8df6-4085-86ee-ce740ccdff30"
MODEL="unsloth/Qwen2-1.5B-Instruct"
DATASET="https://s3.eu-central-003.backblazeb2.com/gradients-validator/bf533eae17a9e498_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260313%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260313T133453Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0ea188624d2910a9b599c4a55fdfd3fc4bbde966c518e49ffc5dce9d98caba2d"
DATASET_TYPE='{
  "field_prompt": "prompt",
  "field_system": null,
  "field_chosen": "chosen",
  "field_rejected": "rejected",
  "prompt_format": null,
  "chosen_format": null,
  "rejected_format": null
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="Masnuy"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"


CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "DpoTask"


docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name dpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \

echo "Uploading model to HuggingFace..."
docker run --rm --gpus all \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --name hf-uploader \
  hf-uploader
