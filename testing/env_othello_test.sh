#!/bin/bash
# ============================================================
# Test: Qwen/Qwen2.5-1.5B-Instruct | EnvTask (othello)
# Unlike text tasks, EnvTask has no static dataset — training data comes from
# live /reset + /step calls to ENVIRONMENT_SERVER_URLS during training. Set
# ENVIRONMENT_SERVER_URLS below to a real (or local dummy) environment server
# before running this — without it, tokenize_env.py will fail to build any
# rows and training will refuse to start.
#
# NOTE: --network none is intentionally NOT used for the trainer container
# here (unlike the text-task test scripts), because the container must be able
# to reach ENVIRONMENT_SERVER_URLS over the network.
# ============================================================

TASK_ID="e1234567-aaaa-bbbb-cccc-othello000001"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="not-used-for-env-tasks"
DATASET_TYPE='{
  "environment_names": ["othello"]
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1.5
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="YOUR_HF_USERNAME_HERE"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME="qwen-1.5b-env-othello"
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
echo " Task      : EnvTask (othello)"
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
  --name env-othello-trainer \
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
