#!/bin/bash
# ============================================================
# EnvTask SMOKE: Llama-3.2-3B-Instruct, the 6 envs of dummy task 20886bcc, 1.5h budget.
# Validates the MCTS-teacher fix (clobber+othello) end-to-end on the REAL R1 base model:
#   - clobber + othello via the OpenSpiel MCTS-100 teacher (2026-07-26 fix). WATCH the log for
#     the '[pvp_selfplay] WARNING ... MCTS teacher unavailable' line — it must NOT appear
#     (its absence = the strong MCTS teacher is active, not a silent fallback to the weak one).
#   - gin_rummy / leduc_poker / goofspiel via their hand-written teachers.
#   - intercode via the mounted miner dataset (coding anchor, REQUIRED).
#
# Needs: H100 (present), Docker + the two images built, intercode dataset downloaded.
#
# RUN IN BACKGROUND (survives logout while it trains ~1.5h):
#   nohup bash testing/env_llama_6env_test.sh > /tmp/llama6.log 2>&1 &
#   tail -f /tmp/llama6.log        # Ctrl-C stops watching; training keeps running
# ============================================================

TASK_ID="20886bcc-430c-4a28-9c14-2f39ceba117a"
MODEL="${MODEL:-unsloth/Llama-3.2-3B-Instruct}"
DATASET="env_task_dummy_dataset"
DATASET_TYPE='{"environment_names": ["clobber", "gin_rummy", "intercode", "leduc_poker", "othello", "goofspiel"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-1.5}"
EXPECTED_REPO_NAME="llama32-3b-env-6env"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$INTERCODE_NAME"

if [ ! -d "$MINER_DATASETS_HOST/$INTERCODE_NAME" ]; then
  echo "!!! Intercode dataset (REQUIRED) NOT found: $MINER_DATASETS_HOST/$INTERCODE_NAME"
  echo "    Download it first (see testing/env_clobber_intercode_test.sh header for the command)."
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask 6 envs (clobber gin_rummy intercode leduc_poker othello goofspiel)"
echo " Hours     : $HOURS_TO_COMPLETE"
echo " FIX CHECK : clobber+othello must use the MCTS teacher (NO [pvp_selfplay] WARNING line)"
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

echo "Training (utah recipe, 6-env / ${HOURS_TO_COMPLETE}h)..."
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
  --name env-llama-6env-trainer \
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
echo " Llama-3.2-3B 6-env smoke done. CHECK:"
echo "   NO '[pvp_selfplay] WARNING ... MCTS teacher unavailable' line (fallback)"
echo "   clobber + othello both generate data (SCORED envs)"
echo "   [combine] train=<merged size>"
echo "   'Training successfully done'  (NOT the base-model fallback)"
echo " Output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "================================================"
