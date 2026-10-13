#!/bin/bash
# Intercode maturation A/B — trains othello+intercode at 3 intercode weights, evals each with the
# proxy solve-harness (verify_intercode_solve.py) AND othello-hold (verify_vs_winner vs 5GU4).
# Answers: does raising intercode weight (0.6 -> 1.3 -> 2.5) improve intercode solve WITHOUT collapsing
# othello? See testing/INTERCODE_MATURATION.md. RUN MONITORED — the harness is a first-cut (debug it on
# a --peek smoke FIRST, below, before trusting the sweep).
set -u
ML=/root/ic_ab_master.log
CKPT=/root/env-destroy/secure_checkpoints; OUT=/root/env-destroy/outputs; MINER=/root/env-destroy/miner_datasets
SCR=/root/env-destroy/scripts   # fresh checkout mounted OVER the image's baked /workspace/scripts (harness lives here)
VSW=/root/env-destroy/scripts/envs/verify_vs_winner.py
VIS=/root/env-destroy/scripts/envs/verify_intercode_solve.py
MODEL=unsloth/Llama-3.2-3B-Instruct
echo "[$(date +%H:%M)] === INTERCODE A/B (othello+intercode, weights 0.6/1.3/2.5) ===" > "$ML"

# ---- STEP 0: harness smoke FIRST (debug before spending on the sweep) ----
# Uses the base model just to confirm the harness RUNS end-to-end (loads tasks, runs bash, scores).
# If this errors, FIX the harness before the sweep — do not burn training runs on a broken eval.
echo "[$(date +%H:%M)] harness smoke (base model, n=8, peek=3)..." >> "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$MINER:/miner_datasets:ro" \
  -v /root/hf_cache:/root/.cache/huggingface:rw \
  --env MINER_DATASETS_DIR=/miner_datasets \
  --env MINER_DATASETS=gradients-io-tournaments--intercode_bigcode_combined_12k \
  -v "$SCR:/workspace/scripts:ro" \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_intercode_solve.py --our-model $MODEL --n 8 --max-turns 4 --peek 3" \
  > /root/ic_smoke.log 2>&1
if grep -q "INTERCODE SOLVE" /root/ic_smoke.log; then
  echo "[$(date +%H:%M)] harness smoke OK ✅" >> "$ML"
else
  echo "[$(date +%H:%M)] harness smoke FAILED — FIX before sweep. tail:" >> "$ML"
  tail -20 /root/ic_smoke.log >> "$ML"
  echo "[$(date +%H:%M)] === STOPPED (harness needs a fix) ===" >> "$ML"
  exit 1
fi

run_cfg() { # $1=weight $2=label
  local W="$1" LBL="$2" TID="icab-${2}-$(date +%H%M)" REPO="llama32-3b-icab-$2"
  echo "[$(date +%H:%M)] TRAIN $LBL: ENV_WEIGHTS intercode=$W ..." >> "$ML"
  docker run --rm --gpus all --security-opt=no-new-privileges --cap-drop=ALL --memory=160g --cpus=28 \
    -v "$CKPT:/cache:rw" -v "$OUT:/app/checkpoints/:rw" -v "$MINER:/miner_datasets:ro" \
    --env MINER_DATASETS_DIR=/miner_datasets --env WANDB_MODE=offline \
    --env SFT_EPOCHS=1 --env GEN_SECONDS=2400 --env OTHELLO_TEACHER_ENDGAME=16 \
    --env "ENV_WEIGHTS={\"intercode\": $W}" \
    standalone-text-trainer --task-id "$TID" --model "$MODEL" --dataset not-used \
    --dataset-type '{"environment_names": ["othello", "intercode"]}' \
    --task-type EnvTask --file-format s3 --hours-to-complete 1.5 --expected-repo-name "$REPO" \
    > "/root/train_icab_$2.log" 2>&1
  local M="$OUT/$TID/$REPO"; local MC="/host_outputs/$TID/$REPO"
  if [ ! -f "$M/config.json" ]; then echo "[$(date +%H:%M)] $LBL TRAIN FAILED" >> "$ML"; return 1; fi
  # intercode row count actually used (the merge log)
  local ROWS=$(grep -oE "balance: intercode train [0-9]+ → [0-9]+|intercode: [0-9]+ examples" "/root/train_icab_$2.log" | tail -1)
  echo "[$(date +%H:%M)] $LBL trained. intercode rows: ${ROWS:-?}" >> "$ML"
  # eval 1: intercode solve (proxy)
  docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$MINER:/miner_datasets:ro" \
    -v /root/hf_cache:/root/.cache/huggingface:rw \
    --env MINER_DATASETS_DIR=/miner_datasets \
    --env MINER_DATASETS=gradients-io-tournaments--intercode_bigcode_combined_12k \
    -v "$SCR:/workspace/scripts:ro" \
    --entrypoint bash standalone-text-trainer \
    -c "cd /workspace/scripts && python envs/verify_intercode_solve.py --our-model $MC --n 200 --max-turns 4" \
    > "/root/eval_icsolve_$2.log" 2>&1
  local CMD=$(grep "COMMAND-match" "/root/eval_icsolve_$2.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  local F1=$(grep "OUTPUT-F1" "/root/eval_icsolve_$2.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  # eval 2: othello HOLD (must not collapse)
  docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
    -v "$SCR:/workspace/scripts:ro" \
    --entrypoint bash standalone-text-trainer \
    -c "cd /workspace/scripts && python envs/verify_vs_winner.py --env othello --games 40 --our-model $MC" \
    > "/root/eval_oth_$2.log" 2>&1
  local OTH=$(grep -oE "win-rate of decisive games: [0-9]+%" "/root/eval_oth_$2.log" | grep -oE "[0-9]+" | tail -1)
  echo "[$(date +%H:%M)] $LBL RESULT: intercode cmd-match=${CMD:-?} out-f1=${F1:-?} | othello vs 5GU4=${OTH:-?}%" >> "$ML"
}

run_cfg 0.6 baseline
run_cfg 1.3 step1
run_cfg 2.5 step2
echo "[$(date +%H:%M)] === INTERCODE A/B DONE ===" >> "$ML"
echo "READ: intercode cmd-match/out-f1 should RISE 0.6→1.3→2.5 while othello HOLDS ≥72%. Best config =" >> "$ML"
echo "highest intercode WITHOUT othello dropping. If othello collapses at 2.5, step1 (1.3) is the safe pick." >> "$ML"
