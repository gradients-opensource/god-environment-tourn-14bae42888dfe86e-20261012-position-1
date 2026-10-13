#!/bin/bash
# ============================================================
# Test: Qwen/Qwen2.5-1.5B-Instruct | InstructTextTask
# Dataset: 413f24dca0fc83b4 (coding)
# ============================================================

TASK_ID="413f24dc-a0fc-83b4-aee5-1faf9aaeb03b"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="https://s3.eu-central-003.backblazeb2.com/gradients-validator/413f24dca0fc83b4_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260521%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260521T141010Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c3a8d94071e2c32adaf6baad576485d5c60a920878482a737ee4e1ea424df7de"
DATASET_TYPE='{
  "field_system": null,
  "field_instruction": "instruct",
  "field_input": null,
  "field_output": "output",
  "format": null,
  "no_input_format": null,
  "system_format": null
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=2
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="YOUR_HF_USERNAME_HERE"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME="qwen-1.5b-coding-instruct"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : InstructTextTask"
echo " Task ID   : $TASK_ID"
echo " Hours     : $HOURS_TO_COMPLETE"
echo " Repo Name : $EXPECTED_REPO_NAME"
echo "================================================"

echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "InstructTextTask"

echo "Training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name instruct-text-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"

echo "Uploading model to HuggingFace..."
docker run --rm --gpus all \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --env MODEL="$MODEL" \
  --env WANDB_LOGS_PATH="/app/checkpoints/$TASK_ID" \
  --name hf-uploader \
  hf-uploader

echo "✅ Selesai! Model di-upload ke HuggingFace: $HUGGINGFACE_USERNAME/$EXPECTED_REPO_NAME"
