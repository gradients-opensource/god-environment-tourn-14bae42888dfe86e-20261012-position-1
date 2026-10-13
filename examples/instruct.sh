#!/bin/bash

TASK_ID="a0b6e564-b8fd-4183-aee5-1faf9aaeb03b"
MODEL="microsoft/Phi-3-mini-128k-instruct"
DATASET="https://s3.eu-central-003.backblazeb2.com/gradients-validator/687a16e4fbff5a30_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260402%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260402T145147Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a5b4b0647678cd12733ccb4c2be17fb9ee754204a93a76b64dac155f113d0a6b"
DATASET_TYPE='{
  "field_system": null,
  "field_instruction": "instruct",
  "field_input": "input",
  "field_output": "output",
  "format": null,
  "no_input_format": null,
  "system_format": null
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=2
WANDB_TOKEN="YOUR_WANDB_TOKEN_HERE"
HUGGINGFACE_USERNAME="Jordansky"
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
  --task-type "InstructTextTask"


docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network none \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --name instruct-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \

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
