#!/bin/bash

TASK_ID="618b077e-aa2c-47f5-9d42-8dfff38a682b"
MODEL="unsloth/Phi-3-mini-4k-instruct"
DATASET="https://s3.eu-central-003.backblazeb2.com/gradients-validator/e803524b6fd48ddf_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260313%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260313T225823Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=b2cb1a3d7ce85082812b9f9433a6fa0a2f88fd5166326f817726829e13b7a3fa"
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
HOURS_TO_COMPLETE=6
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
  --task-type "GrpoTask"


docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name grpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "GrpoTask" \
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