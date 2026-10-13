#!/bin/bash
# ============================================================
# 3B OTHELLO SMOKE — Llama-3.2-3B-Instruct, self-play (no env server), WANDB offline.
# Measures the flood + surpass levers on the DECISIVE env in isolation.
#
# Usage (run INSIDE screen):
#   screen -dmS smoke bash testing/env_othello_3b_smoke.sh baseline
#   screen -dmS smoke bash testing/env_othello_3b_smoke.sh flood
#   screen -dmS smoke bash testing/env_othello_3b_smoke.sh surpass
#   (attach: screen -r smoke ; detach: Ctrl-a d)
#
# CONFIG presets:
#   baseline  = 843e71f-equivalent: othello weight 1.3, 2 epochs, default teacher.
#   flood     = SFT_EPOCHS=1 + more gen time + faster teacher (ENDGAME 12) to lift othello ROW volume.
#   surpass   = flood + train vs the ACTUAL eval opponent MCTS-25 (OTHELLO_OPP_MCTS=1).
#
# After each run, PvP the model vs the othello champ 5GU4 to see if othello improved / surpassed:
#   .venv/bin/python scripts/envs/verify_vs_winner.py --env othello --games 40 --our-model <OUT>
# (needs the host venv — see testing/setup_host_venv.sh)
# ============================================================
set -u
CONFIG="${1:-flood}"
MODEL="${MODEL:-unsloth/Llama-3.2-3B-Instruct}"
ENVS="${ENVS:-othello}"                 # set ENVS='othello,intercode' for the full round (needs intercode mount)
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-1.5}"
TASK_ID="smoke-3b-othello-${CONFIG}-$(date +%H%M)"
EXPECTED_REPO_NAME="llama32-3b-othello-${CONFIG}"
DATASET="not-used-for-env-tasks"
FILE_FORMAT="s3"
DATASET_TYPE="{\"environment_names\": [$(echo "$ENVS" | sed 's/,/","/g; s/^/"/; s/$/"/')]}"

# --- per-CONFIG env-vars (the levers) ---
COMMON_ENV=( --env WANDB_MODE=offline )
case "$CONFIG" in
  baseline)
    LEVERS=( --env 'ENV_WEIGHTS={"othello":1.3}' --env SFT_EPOCHS=2 ) ;;
  flood)
    LEVERS=( --env SFT_EPOCHS=1 --env GEN_SECONDS=2400 --env OTHELLO_TEACHER_ENDGAME=12 ) ;;
  surpass)
    LEVERS=( --env SFT_EPOCHS=1 --env GEN_SECONDS=2400 --env OTHELLO_TEACHER_ENDGAME=12 \
             --env OTHELLO_OPP_MCTS=1 --env OTHELLO_OPP_MCTS_SIMS=25 --env OTHELLO_OPP_MCTS_ROLLOUTS=1 ) ;;
  *) echo "unknown CONFIG '$CONFIG' (use baseline|flood|surpass)"; exit 2 ;;
esac

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"; OUTPUTS_DIR="$(pwd)/outputs"
MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$MINER_DATASETS_HOST"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
LOG="/root/smoke_${CONFIG}.log"

echo "================================================" | tee "$LOG"
echo " 3B OTHELLO SMOKE  CONFIG=$CONFIG" | tee -a "$LOG"
echo " Model=$MODEL  Envs=$ENVS  Hours=$HOURS_TO_COMPLETE" | tee -a "$LOG"
echo " Levers: ${LEVERS[*]}" | tee -a "$LOG"
echo " Out=$OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME" | tee -a "$LOG"
echo "================================================" | tee -a "$LOG"

echo "[1/2] Downloading model..." | tee -a "$LOG"
docker run --rm --volume "$CHECKPOINTS_DIR:/cache:rw" --name dl-$CONFIG \
  trainer-downloader --task-id "$TASK_ID" --model "$MODEL" --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" --task-type "EnvTask" 2>&1 | tee -a "$LOG"

echo "[2/2] Training (CONFIG=$CONFIG)..." | tee -a "$LOG"
docker run --rm --gpus all --security-opt=no-new-privileges --cap-drop=ALL \
  --memory=160g --cpus=28 \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --volume "$MINER_DATASETS_HOST:/miner_datasets:ro" \
  --env MINER_DATASETS_DIR=/miner_datasets \
  --env MINER_DATASETS="gradients-io-tournaments--intercode_bigcode_combined_12k" \
  "${COMMON_ENV[@]}" "${LEVERS[@]}" \
  --name smoke-$CONFIG-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" --model "$MODEL" --dataset "$DATASET" --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" --file-format "$FILE_FORMAT" --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== SMOKE DONE ($CONFIG). CHECK (log: $LOG): ====" | tee -a "$LOG"
echo " 1. OTHELLO merged rows (>> ~4-7.8k = flood worked):" | tee -a "$LOG"
echo "      grep -Ei '\[balance\]|\[combine\]|othello.*(row|example|conversation)' $LOG | tail" | tee -a "$LOG"
echo " 2. gen command shows '--scored_cap 6000' and NOT stale 7000 (image fresh)." | tee -a "$LOG"
echo " 3. NO '[pvp_selfplay] WARNING ... unavailable' (teacher active)." | tee -a "$LOG"
echo " 4. cosine LR reaches min at the end (anneal completed); 'Training successfully done'." | tee -a "$LOG"
echo " 5. PvP vs the othello champ (surpass check):" | tee -a "$LOG"
echo "      .venv/bin/python scripts/envs/verify_vs_winner.py --env othello --games 40 --our-model $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME" | tee -a "$LOG"
