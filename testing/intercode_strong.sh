#!/bin/bash
# SINGLE-SHOT "intercode strong" attempt for TODAY (deadline ~15:00 WIB to swap the commit).
# Tests the BAKED round-aware boost: intercode weight 2.5 in an othello+intercode round (env var here = the
# baked default; the tournament BUILDS the image from our repo so it gets the bake), at epochs=1 (= the
# deploy's flood-aware default, so this test == the tournament). Goal: give intercode enough capacity that
# the othello flood stops breaking it (weight 0.6 -> game_action, 1.3 -> degenerate; 2.5 = the bet).
# NOTE: epochs stays 1 (NOT 2) on purpose — epochs=2 risks not fitting 1.5h and dragging OTHELLO (our carry);
# weight-only is the safe lever. If 2.5 still degenerates, that's the signal epochs is needed (next tourney).
# Then eval intercode-solve (FIXED harness + PEEK to SEE the commands) AND othello-hold (must stay strong).
#
# DECISION (compare to last night's baseline: weight 0.6 = game_action/0.000, weight 1.3 = execute_bash but
# DEGENERATE garbage):
#   PASS  -> intercode commands look CLEAN (not '- - -'/'x x x' spam) AND intercode solve clearly > 0 AND
#            othello vs 5GU4 >= 70%  => swap training_repo.py commit to this SHA, deploy TODAY.
#   FAIL  -> degenerate still, OR intercode ~0, OR othello < 70%  => KEEP e6f8e23 (already live via fiber).
# NO PANIC: this script never kills anything; the monitor is READ-ONLY. Fallback (e6f8e23) is already safe.
set -u
ML=/root/ic_strong_master.log
CKPT=/root/env-destroy/secure_checkpoints; OUT=/root/env-destroy/outputs; MINER=/root/env-destroy/miner_datasets
SCR=/root/env-destroy/scripts
MODEL=unsloth/Llama-3.2-3B-Instruct
DS_MOUNT=gradients-io-tournaments--intercode_bigcode_combined_12k
echo "[$(date +%H:%M)] === INTERCODE-STRONG single-shot: epochs=2 intercode_weight=2.5 (deadline ~15:00) ===" > "$ML"

# ---- STEP 0: harness smoke (FIXED harness is untested locally — confirm it runs before the real eval) ----
echo "[$(date +%H:%M)] harness smoke (base model, n=4)..." >> "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$MINER:/miner_datasets:ro" \
  -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --env MINER_DATASETS_DIR=/miner_datasets --env MINER_DATASETS="$DS_MOUNT" \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_intercode_solve.py --our-model $MODEL --n 4 --max-turns 3 --rep-penalty 1.15" \
  > /root/ics_smoke.log 2>&1
if ! grep -q "INTERCODE SOLVE" /root/ics_smoke.log; then
  echo "[$(date +%H:%M)] harness smoke FAILED — fixing needed, NOT training. tail:" >> "$ML"; tail -20 /root/ics_smoke.log >> "$ML"
  echo "[$(date +%H:%M)] === STOPPED (harness) — deploy stays e6f8e23 ===" >> "$ML"; exit 1
fi
echo "[$(date +%H:%M)] harness smoke OK" >> "$ML"

# ---- STEP 1: train the ONE config ----
TID="icstrong-$(date +%H%M)"; REPO="llama32-3b-icstrong"
echo "[$(date +%H:%M)] TRAIN $TID: SFT_EPOCHS=2 ENV_WEIGHTS intercode=2.5 GEN=1800 ..." >> "$ML"
docker run --rm --gpus all --security-opt=no-new-privileges --cap-drop=ALL --memory=160g --cpus=28 \
  -v "$CKPT:/cache:rw" -v "$OUT:/app/checkpoints/:rw" -v "$MINER:/miner_datasets:ro" \
  --env MINER_DATASETS_DIR=/miner_datasets --env WANDB_MODE=offline \
  --env SFT_EPOCHS=1 --env GEN_SECONDS=2400 --env OTHELLO_TEACHER_ENDGAME=16 \
  --env 'ENV_WEIGHTS={"intercode": 2.5}' \
  standalone-text-trainer --task-id "$TID" --model "$MODEL" --dataset not-used \
  --dataset-type '{"environment_names": ["othello", "intercode"]}' \
  --task-type EnvTask --file-format s3 --hours-to-complete 1.5 --expected-repo-name "$REPO" \
  > /root/train_icstrong.log 2>&1
M="$OUT/$TID/$REPO"; MC="/host_outputs/$TID/$REPO"
if [ ! -f "$M/config.json" ]; then echo "[$(date +%H:%M)] TRAIN FAILED — keep e6f8e23. tail:" >> "$ML"; tail -15 /root/train_icstrong.log >> "$ML"; exit 1; fi
EP=$(grep -oE "num_train_epochs[^0-9]*[0-9]+|epoch [0-9.]+/[0-9]|[0-9]+/[0-9]+ \[.*epoch" /root/train_icstrong.log | tail -1)
ROWS=$(grep -oiE "intercode[^0-9]{0,20}[0-9]+ (examples|rows)|balance: intercode[^0-9]*[0-9]+ . [0-9]+|per_env_target=\{[^}]*\}|Train: [0-9]+" /root/train_icstrong.log | tail -3 | tr '\n' ' ')
echo "[$(date +%H:%M)] trained. epochs=${EP:-?} | rows: ${ROWS:-?}" >> "$ML"

# ---- STEP 2: eval intercode-solve (FIXED harness, PEEK to see commands) + othello-hold ----
echo "[$(date +%H:%M)] eval intercode-solve (n=150, peek=8, rep-penalty 1.15)..." >> "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$MINER:/miner_datasets:ro" \
  -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --env MINER_DATASETS_DIR=/miner_datasets --env MINER_DATASETS="$DS_MOUNT" \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_intercode_solve.py --our-model $MC --n 150 --max-turns 3 --peek 8 --rep-penalty 1.15" \
  > /root/eval_ics_intercode.log 2>&1
CMD=$(grep "COMMAND-match" /root/eval_ics_intercode.log | grep -oE "[0-9]+\.[0-9]+" | head -1)
F1=$(grep "OUTPUT-F1" /root/eval_ics_intercode.log | grep -oE "[0-9]+\.[0-9]+" | head -1)
# degeneration check: count degenerate ('- - -' / 'x x x' style) among the peeked commands
DEGEN=$(grep -cE "cmd='[^']*( -){4,}| x x x x|( [a-z-]){10,}" /root/eval_ics_intercode.log 2>/dev/null)
CLEAN=$(grep -cE "peek turn 1. name=execute_bash cmd='[a-z]" /root/eval_ics_intercode.log 2>/dev/null)
echo "[$(date +%H:%M)] intercode: cmd-match=${CMD:-?} out-f1=${F1:-?} | peek(turn1) clean~${CLEAN:-?} degen~${DEGEN:-?}" >> "$ML"

echo "[$(date +%H:%M)] eval othello-hold (n=40 vs 5GU4)..." >> "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_vs_winner.py --env othello --games 40 --our-model $MC" \
  > /root/eval_ics_othello.log 2>&1
OTH=$(grep -oE "win-rate of decisive games: [0-9]+%" /root/eval_ics_othello.log | grep -oE "[0-9]+" | tail -1)
echo "[$(date +%H:%M)] othello vs 5GU4 = ${OTH:-?}%" >> "$ML"

# ---- STEP 3: VERDICT ----
echo "[$(date +%H:%M)] === RESULT: SHA to deploy if PASS = $(cd /root/env-destroy && git rev-parse HEAD 2>/dev/null | cut -c1-12) (this run's code == e6f8e23 training path; recipe change is the ENV_WEIGHTS/epochs you'd bake) ===" >> "$ML"
echo "DECISION GUIDE:" >> "$ML"
echo "  PASS (deploy today) if: peek commands CLEAN (clean~ high, degen~ low) AND cmd-match/out-f1 clearly > 0 AND othello >= 70%" >> "$ML"
echo "  FAIL (keep e6f8e23)  if: degen~ still high / intercode ~0 / othello < 70%" >> "$ML"
echo "  NOTE: to actually DEPLOY, the winning ENV_WEIGHTS+epochs must be BAKED into the recipe (a code commit)," >> "$ML"
echo "        not just env vars — the tournament training won't get our ad-hoc env vars. Flag me to make that commit." >> "$ML"
echo "[$(date +%H:%M)] === INTERCODE-STRONG DONE ===" >> "$ML"
