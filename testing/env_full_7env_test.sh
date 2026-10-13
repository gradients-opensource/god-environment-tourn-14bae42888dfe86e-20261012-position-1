#!/bin/bash
# ============================================================
# EnvTask FULL run: all 7 envs, 3h budget (real tournament shape)
# task a2b8265c — intercode + leduc_poker + othello + clobber + goofspiel
#                 + gin_rummy + liars_dice.  All 7 are SCORED → no coverage layer.
#
# Validates the utah recipe end-to-end at full scale:
#   - 6 games via local pyspiel self-play (capped 7k each)
#   - intercode via mounted miner dataset (coding anchor)
#   - cosine LR over ~1-2 epochs, deadline guard, clean finish
#
# No ENVIRONMENT_SERVER_URLS needed (SFT = local pyspiel + offline datasets).
# Requires the intercode miner dataset downloaded (see env_clobber_intercode_test.sh header).
#
# RUN IN BACKGROUND (survives logout while you rest):
#   nohup bash testing/env_full_7env_test.sh > /tmp/full7.log 2>&1 &
#   tail -f /tmp/full7.log        # watch;  Ctrl-C to stop watching (train keeps running)
# ============================================================

TASK_ID="a2b8265c-cc8b-4e80-a9d0-517cf7c6d0e3"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET="env_task_dummy_dataset"
DATASET_TYPE='{"environment_names": ["intercode", "leduc_poker", "othello", "clobber", "goofspiel", "gin_rummy", "liars_dice"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-3.0}"
EXPECTED_REPO_NAME="qwen3-4b-env-full-7env"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$INTERCODE_NAME"

if [ ! -d "$MINER_DATASETS_HOST/$INTERCODE_NAME" ]; then
  echo "!!! Intercode dataset (REQUIRED) NOT found: $MINER_DATASETS_HOST/$INTERCODE_NAME"
  echo "    See testing/env_clobber_intercode_test.sh header for the download command."
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask (7 envs, ALL scored)"
echo " Hours     : $HOURS_TO_COMPLETE   (deadline guard saves ~5min before end)"
echo " MinerDS   : $MINER_DATASETS_LIST"
echo "================================================"

echo "Downloading model (no static dataset for EnvTask)..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

echo "Training (utah recipe, full 7-env / ${HOURS_TO_COMPLETE}h)..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=160g \
  --cpus=28 \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --volume "$MINER_DATASETS_HOST:/miner_datasets:ro" \
  --env MINER_DATASETS_DIR=/miner_datasets \
  --env MINER_DATASETS="$MINER_DATASETS_LIST" \
  --env WANDB_MODE=offline \
  --name env-full-7env-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"

echo ""
echo "================================================"
echo " Full 7-env training done. Check for:"
echo "   [balance] each game -> 7000 (capped) ; intercode kept ; NO coverage line"
echo "   all 7 envs present (watch othello: it must generate data as a SCORED env)"
echo "   [combine] train=<the merged size>"
echo "   cosine LR decays ; epoch reaches ~1-2 (real multi-epoch at 3h)"
echo "   Training successfully done  (NOT the base-model fallback)"
echo " Model output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "================================================"
