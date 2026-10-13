#!/bin/bash
# ============================================================
# Test: Qwen/Qwen3-4B-Instruct-2507 | EnvTask (clobber + gin_rummy)
# Payload dari task asli validator (task_id: ad0a55e7-...)
#
# Behaviour:
#   - clobber: dipilih sebagai active environment (pertama di list)
#   - SFT warm-start: clobber via local pyspiel self-play (no env server)
#   - GRPO RL: clobber dengan mobility-control strategy
# ============================================================

TASK_ID="ad0a55e7-6e0d-4f7d-b3d7-d37e6c7f4f50"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
DATASET="env_task_dummy_dataset"
DATASET_TYPE='{"environment_names": ["clobber", "gin_rummy"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1.5
WANDB_TOKEN="${WANDB_TOKEN:-YOUR_WANDB_TOKEN_HERE}"
HUGGINGFACE_USERNAME="${HUGGINGFACE_USERNAME:-YOUR_HF_USERNAME_HERE}"
HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-YOUR_HF_TOKEN_HERE}"
EXPECTED_REPO_NAME="qwen3-4b-clobber-ginrummy"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
# Tidak butuh ENVIRONMENT_SERVER_URLS untuk SFT phase (selfplay lokal)
# Untuk GRPO rollout, isi URL server environment asli kalau ada
ENVIRONMENT_SERVER_URLS="${ENVIRONMENT_SERVER_URLS:-}"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask (clobber -> active, gin_rummy -> fallback)"
echo " Task ID   : $TASK_ID"
echo " Hours     : $HOURS_TO_COMPLETE"
echo "================================================"

echo "[1/3] Downloading model..."
docker rm -f downloader-clobber-ginrummy 2>/dev/null || true
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-clobber-ginrummy \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

# SFT-only by default (ENV_SKIP_GRPO=1 is also baked into the image). Override without a
# rebuild by exporting ENV_SKIP_GRPO=0 to run the full SFT warm-start → GRPO RL pipeline.
ENV_SKIP_GRPO="${ENV_SKIP_GRPO:-1}"
echo "[2/3] Training (EnvTask, ENV_SKIP_GRPO=$ENV_SKIP_GRPO → $([ "$ENV_SKIP_GRPO" = "1" ] && echo 'SFT-only' || echo 'SFT + GRPO'))..."
docker rm -f env-clobber-ginrummy-trainer 2>/dev/null || true
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --shm-size=2g \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env ENVIRONMENT_SERVER_URLS="$ENVIRONMENT_SERVER_URLS" \
  --env ENV_SKIP_GRPO="$ENV_SKIP_GRPO" \
  --name env-clobber-ginrummy-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"

# Skip the upload step during local testing when HF creds are still placeholders.
# In the real tournament the orchestrator injects valid HUGGINGFACE_* creds, so this
# guard is a no-op there. Training output is already saved locally in $LOCAL_FOLDER.
if [ "$HUGGINGFACE_USERNAME" = "YOUR_HF_USERNAME_HERE" ] || [ "$HUGGINGFACE_TOKEN" = "YOUR_HF_TOKEN_HERE" ]; then
  echo "[3/3] Skipping HuggingFace upload — HUGGINGFACE_USERNAME/HUGGINGFACE_TOKEN not set."
  echo "      Model saved locally at: $LOCAL_FOLDER"
  echo "      To enable upload: export HUGGINGFACE_USERNAME=... HUGGINGFACE_TOKEN=... (token needs WRITE)"
  echo "Selesai (training only)! Model lokal: $LOCAL_FOLDER"
  exit 0
fi

echo "[3/3] Uploading to HuggingFace..."
docker rm -f hf-uploader 2>/dev/null || true
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

echo "Selesai! Model: $HUGGINGFACE_USERNAME/$EXPECTED_REPO_NAME"
