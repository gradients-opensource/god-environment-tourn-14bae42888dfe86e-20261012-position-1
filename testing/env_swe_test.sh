#!/bin/bash
# ============================================================
# EnvTask smoke test: swe_infinite (+ clobber + intercode)
#
# Verifies the NEW swe_infinite env end-to-end (validator PR #1299):
#   - SWE-ZERO trajectories load via envs/swe_infinite_dataset.py (PLAIN messages, no tools)
#   - a MIXED task exercises merge_datasets._concat_aligned: plain SWE rows + tool-call
#     game/intercode rows unify into one schema (no "features can't be aligned")
#   - tokenize renders the plain-text MiniSWE turns (tool_calls=None) on Qwen AND Llama
#   - LR = cosine_with_min_lr, clean finish
#
# For a swe_infinite-ONLY (boss-round) check, set:
#   DATASET_TYPE='{"environment_names": ["swe_infinite"]}'
#
# ---- ONE-TIME: download the miner datasets to the host ----
#   # SWE-ZERO (REQUIRED here, ~1.7GB / 50k trajectories):
#   uv run --with "huggingface_hub[cli]" huggingface-cli download \
#     gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered --repo-type dataset \
#     --local-dir "./miner_datasets/gradients-io-tournaments--SWE-ZERO-12M-trajectories-filtered"
#   # intercode (for the mixed-schema merge; optional if you set the swe-only DATASET_TYPE):
#   uv run --with "huggingface_hub[cli]" huggingface-cli download \
#     gradients-io-tournaments/intercode_bigcode_combined_12k --repo-type dataset \
#     --local-dir "./miner_datasets/gradients-io-tournaments--intercode_bigcode_combined_12k"
#
# ---- RUN (from repo root) ----
#   TRAIN_GPUS=device=0 HOURS_TO_COMPLETE=0.3 bash testing/env_swe_test.sh 2>&1 | tee /tmp/swe_smoke.log
# ============================================================

TASK_ID="c10bbe12-1234-4567-89ab-sweinfinite01"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET="env_task_dummy_dataset"
# swe_infinite = offline SWE-ZERO loader (plain text) -> exercises the SWE loader + _concat_aligned
# clobber      = OpenSpiel self-play (tool-call)      -> the tool-call side of the schema merge
# intercode    = offline coding anchor (tool-call)    -> second tool-call source
_DEFAULT_DT='{"environment_names": ["swe_infinite", "clobber", "intercode"]}'
DATASET_TYPE="${DATASET_TYPE:-$_DEFAULT_DT}"
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-0.3}"
# Which GPU(s) the trainer gets. Default all; pin one on a shared box, e.g. TRAIN_GPUS=device=0.
TRAIN_GPUS="${TRAIN_GPUS:-all}"
EXPECTED_REPO_NAME="swe-infinite-smoke"

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
SWE_NAME="gradients-io-tournaments--SWE-ZERO-12M-trajectories-filtered"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$SWE_NAME,$INTERCODE_NAME"

if [ ! -d "$MINER_DATASETS_HOST/$SWE_NAME" ]; then
  echo "!!! SWE-ZERO dataset (REQUIRED) NOT found: $MINER_DATASETS_HOST/$SWE_NAME"
  echo "    Download it (one-time, ~1.7GB):"
  echo "      uv run --with 'huggingface_hub[cli]' huggingface-cli download \\"
  echo "        gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered --repo-type dataset \\"
  echo "        --local-dir \"$MINER_DATASETS_HOST/$SWE_NAME\""
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " Model     : $MODEL"
echo " Task      : EnvTask $DATASET_TYPE"
echo " Hours     : $HOURS_TO_COMPLETE   GPUs: $TRAIN_GPUS"
echo " MinerDS   : $MINER_DATASETS_LIST"
echo "================================================"

# Clear any container left over from a previous/aborted run (fixed names → only one at a time).
docker rm -f downloader-image env-swe-trainer 2>/dev/null || true

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
  --name env-swe-trainer \
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
echo "   [swe_infinite_dataset] Loaded <N> trajectories / Converted to <M>"
echo "   [generate_trajectories] SWE-Infinite dataset saved"
echo "   merge: aligned <k> datasets w/ divergent schemas  (the _concat_aligned path)"
echo "   Dataset: train=<...>   +  Filtered ... (Llama compat)  +  tokenize NO crash"
echo "   'loss': ... going down   +   Training successfully done"
echo " Model output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "================================================"
