#!/bin/bash
# ============================================================
# Test: Qwen/Qwen2.5-1.5B-Instruct | DpoTask
# Dataset: b2c3d4e5 (ganti dengan dataset DPO dari validator)
# ============================================================

TASK_ID="b2c3d4e5-f6a7-b8c9-d0e1-f2a3b4c5d6e7"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="GANTI_DENGAN_S3_URL_DATASET_DPO"
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
HUGGINGFACE_USERNAME="YOUR_HF_USERNAME"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME="qwen-1.5b-coding-dpo"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : DpoTask"
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
  --task-type "DpoTask"

echo "Training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name dpo-text-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
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
