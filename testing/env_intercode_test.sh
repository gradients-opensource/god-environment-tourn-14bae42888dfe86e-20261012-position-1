#!/bin/bash
# ============================================================
# Test: Qwen/Qwen2.5-1.5B-Instruct | EnvTask (intercode)
#
# WARNING: intercode's miner-facing /reset and /step wire format is NOT
# confirmed against public G.O.D sources (the validator's own eval_intercode.py
# restores local filesystem snapshots rather than calling a sidecar over HTTP).
# This script and scripts/core/environment_rollout.py assume the same /reset
# + /step contract as the game environments is exposed to miners — verify this
# against a real environment server before relying on this for a tournament.
#
# Same as env_othello_test.sh otherwise: no static dataset, --network none is
# intentionally NOT set so the trainer container can reach
# ENVIRONMENT_SERVER_URLS.
# ============================================================

TASK_ID="e1234567-aaaa-bbbb-cccc-intercode0001"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="not-used-for-env-tasks"
DATASET_TYPE='{
  "environment_names": ["intercode"]
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1.5
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="YOUR_HF_USERNAME_HERE"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME="qwen-1.5b-env-intercode"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
ENVIRONMENT_SERVER_URLS="GANTI_DENGAN_URL_ENVIRONMENT_SERVER"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask (intercode)"
echo " Task ID   : $TASK_ID"
echo " Hours     : $HOURS_TO_COMPLETE"
echo " Repo Name : $EXPECTED_REPO_NAME"
echo "================================================"

echo "Downloading model (no dataset to download for EnvTask)..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

echo "Training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env ENVIRONMENT_SERVER_URLS="$ENVIRONMENT_SERVER_URLS" \
  --name env-intercode-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
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
