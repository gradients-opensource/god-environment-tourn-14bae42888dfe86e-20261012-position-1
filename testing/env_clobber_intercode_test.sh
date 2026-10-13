#!/bin/bash
# ============================================================
# EnvTask smoke test: clobber + leduc_poker + intercode  (covers cap + anchor)
#
# Verifies the utah balance recipe END-TO-END with intercode present:
#   - clobber floods via local pyspiel self-play      -> capped to 7k
#   - intercode loaded from the mounted miner dataset -> floored/kept (coding anchor)
#   - LR = cosine_with_min_lr (decays), multi-epoch, clean finish
#
# No ENVIRONMENT_SERVER_URLS needed: SFT gen is local pyspiel (clobber) + offline
# miner datasets. Only the miner datasets must be present on the host.
#
# ---- ONE-TIME: download the miner datasets to the host ----
#   # intercode (REQUIRED):
#   uv run --with "huggingface_hub[cli]" huggingface-cli download \
#     gradients-io-tournaments/intercode_bigcode_combined_12k --repo-type dataset \
#     --local-dir "./miner_datasets/gradients-io-tournaments--intercode_bigcode_combined_12k"
#   # if a dataset is gated:  uv run --with "huggingface_hub[cli]" huggingface-cli login
#
# ---- RUN (from repo root) ----
#   HOURS_TO_COMPLETE=0.3 bash testing/env_clobber_intercode_test.sh 2>&1 | tee /tmp/smoke.log
# ============================================================

TASK_ID="c10bbe12-1234-4567-89ab-clobberint001"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET="env_task_dummy_dataset"
# clobber       = OpenSpiel self-play gen -> exercises the CAP
# leduc_poker   = a second game env alongside clobber
# intercode     = offline coding anchor -> exercises the FLOOR
DATASET_TYPE='{"environment_names": ["clobber", "leduc_poker", "intercode"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-0.3}"
# Which GPU(s) the trainer gets. Default: all. On a shared/multi-GPU box where only one
# card is free, pin it, e.g.  TRAIN_GPUS=device=3  (single) or  TRAIN_GPUS='"device=1,3"'.
TRAIN_GPUS="${TRAIN_GPUS:-all}"
EXPECTED_REPO_NAME="qwen3-4b-clobber-intercode-smoke"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

# Miner datasets. Download once to $MINER_DATASETS_HOST (see header).
#  - intercode:            REQUIRED (the coding anchor)
MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$INTERCODE_NAME"

if [ ! -d "$MINER_DATASETS_HOST/$INTERCODE_NAME" ]; then
  echo "!!! Intercode dataset (REQUIRED) NOT found: $MINER_DATASETS_HOST/$INTERCODE_NAME"
  echo "    Download it (one-time):"
  echo "      uv run --with 'huggingface_hub[cli]' huggingface-cli download \\"
  echo "        gradients-io-tournaments/intercode_bigcode_combined_12k --repo-type dataset \\"
  echo "        --local-dir \"$MINER_DATASETS_HOST/$INTERCODE_NAME\""
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask (clobber + leduc_poker + intercode)"
echo " Hours     : $HOURS_TO_COMPLETE"
echo " MinerDS   : $MINER_DATASETS_LIST"
echo "================================================"

# Clear any container left over from a previous/aborted run so --name can't collide
# ("container name already in use"). Only ONE smoke may run at a time (fixed names).
docker rm -f downloader-image env-clobber-intercode-trainer 2>/dev/null || true

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

echo "Training (utah balance recipe: cap + floor + cosine)... [GPUs: $TRAIN_GPUS]"
docker run --rm --gpus "$TRAIN_GPUS" \
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
  --name env-clobber-intercode-trainer \
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
echo " Training done. Check the log above for:"
echo "   [balance] clobber ... -> 7000 (capped)"
echo "   [balance] intercode ...           (floored/kept, coding anchor)"
echo "   [combine] ... train=<small ~10-15k>"
echo "   --lr_scheduler_type cosine_with_min_lr   (not constant/wsd)"
echo "   learning_rate decreasing + epoch ~1-2 + loss going down"
echo " Model output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "================================================"
