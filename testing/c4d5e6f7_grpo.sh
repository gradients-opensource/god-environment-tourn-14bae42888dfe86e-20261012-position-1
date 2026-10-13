#!/bin/bash
# ============================================================
# Test: Qwen/Qwen2.5-1.5B-Instruct | GrpoTask
# Dataset: c4d5e6f7 (ganti dengan dataset GRPO dari validator)
# ============================================================

TASK_ID="c4d5e6f7-a8b9-c0d1-e2f3-a4b5c6d7e8f9"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATASET="GANTI_DENGAN_S3_URL_DATASET_GRPO"
DATASET_TYPE='{
  "field_prompt": "prompt",
  "reward_functions": [
    {
      "reward_func": "def reward_short_words(completions, **kwargs):\n    \"\"\"Rewards text with fewer characters per word.\"\"\"\n    import textstat\n    scores = [textstat.avg_character_per_word(comp) for comp in completions]\n    return [-s for s in scores]\n",
      "reward_weight": 1,
      "func_hash": "4bf7186f64a37f397f81c27739e360a08818899a3ad6ba31d96cefc8d75ec675",
      "is_generic": true
    }
  ]
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=2.0
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="YOUR_HF_USERNAME"
HUGGINGFACE_TOKEN="YOUR_HF_TOKEN_HERE"
EXPECTED_REPO_NAME="qwen-1.5b-coding-grpo"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : GrpoTask"
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
  --task-type "GrpoTask"

echo "Training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name grpo-text-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "GrpoTask" \
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
