#!/bin/bash
# FIRST 8B TRAIN — re-fight the exact round we scored 0.0 on: othello+intercode on
# unsloth/Meta-Llama-3.1-8B-Instruct (the tournament base; we validated only on 3B).
# One command: diagnose (gate) -> train full 1.5h -> eval othello-hold + intercode-solve -> verdict.
#
# WHY THIS ORDER (cost-safe): Step 0 is the ~1-min, no-GPU render diagnostic. If the 8B chat
# template REJECTS our tool_calls rows, training would just reproduce the 0.0 (rows get dropped)
# — so we ABORT before spending an H100 hour. Only if rows render do we train.
#
# READS TO WATCH in the train log (added by commit 2b7f805):
#   [sft-tokenize] KEPT x/y   — a big drop = the 8B template rejects our rows (the #1 0.0 suspect)
#   [sft-mask] ... coverage   — mean supervised fraction (0.000/many-zero = masking/truncation broken)
#   SFT_SKIPPED.txt in OUT    — the warm-start silently returned the BASE model (= guaranteed 0.0)
#
# LR: the tournament path sets find_lk_lr=True, so sft_env_config auto-looks-up the per-arch 8B LR
#   (~1.46e-5 for Llama-3.1-8B) — do NOT hardcode 3.5e-5. This run uses the deployed default recipe
#   (round-aware othello flood + flood-aware epochs), i.e. exactly what the tournament would run.
#
# NO PANIC: this script never kills anything; the fallback deploy stays whatever is live.
set -u
ML=/root/first_train_8b_master.log
CKPT=/root/env-destroy/secure_checkpoints; OUT=/root/env-destroy/outputs; MINER=/root/env-destroy/miner_datasets
SCR=/root/env-destroy/scripts
MODEL=${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}
BASE3B=${BASE3B:-unsloth/Llama-3.2-3B-Instruct}
DS_MOUNT=${DS_MOUNT:-gradients-io-tournaments--intercode_bigcode_combined_12k}
echo "[$(date +%H:%M)] === FIRST 8B TRAIN: othello+intercode on $MODEL (re-fight the 0.0 round) ===" > "$ML"

# ---- STEP 0: render diagnostic GATE (no GPU; ~1 min). Abort the train if the 8B template rejects us. ----
echo "[$(date +%H:%M)] STEP 0 — 8B render diagnostic (gate)..." | tee -a "$ML"
docker run --rm --gpus all -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/diagnose_8b_sft.py --model $MODEL" \
  > /root/diag_8b.log 2>&1
tail -25 /root/diag_8b.log | tee -a "$ML"
if grep -q "template REJECTS" /root/diag_8b.log; then
  echo "[$(date +%H:%M)] === ABORT: 8B template REJECTS our rows — FIX row-shape first, do NOT train (saved an H100 hour). ===" | tee -a "$ML"
  exit 1
fi
if ! grep -q "render + supervise" /root/diag_8b.log; then
  echo "[$(date +%H:%M)] === ABORT: diagnostic did not confirm a clean render — inspect /root/diag_8b.log before training. ===" | tee -a "$ML"
  exit 1
fi
echo "[$(date +%H:%M)] STEP 0 OK — rows render + supervise on 8B; proceeding to train." | tee -a "$ML"

# ---- STEP 1: train othello+intercode @ 8B, full 1.5h, DEPLOYED default recipe (no ad-hoc overrides) ----
TID="ft8b-$(date +%H%M)"; REPO="llama31-8b-oth-ic"
echo "[$(date +%H:%M)] STEP 1 — TRAIN $TID (full 1.5h, deployed recipe, LR auto per-arch)..." | tee -a "$ML"
docker run --rm --gpus all --security-opt=no-new-privileges --cap-drop=ALL --memory=160g --cpus=28 \
  -v "$CKPT:/cache:rw" -v "$OUT:/app/checkpoints/:rw" -v "$MINER:/miner_datasets:ro" \
  --env MINER_DATASETS_DIR=/miner_datasets --env MINER_DATASETS="$DS_MOUNT" --env WANDB_MODE=offline \
  standalone-text-trainer --task-id "$TID" --model "$MODEL" --dataset not-used \
  --dataset-type '{"environment_names": ["othello", "intercode"]}' \
  --task-type EnvTask --file-format s3 --hours-to-complete 1.5 --expected-repo-name "$REPO" \
  > /root/train_ft8b.log 2>&1
M="$OUT/$TID/$REPO"; MC="/host_outputs/$TID/$REPO"
# Surface the new diagnostics no matter what.
echo "[$(date +%H:%M)] --- key train-log signals ---" | tee -a "$ML"
grep -E "\[sft-tokenize\]|\[sft-mask\]|\[sft-loss\]|SFT WARM-START SKIPPED|Training successfully done" /root/train_ft8b.log | tee -a "$ML"
if [ -f "$OUT/$TID/$REPO/SFT_SKIPPED.txt" ] || ls "$OUT/$TID"/*/SFT_SKIPPED.txt >/dev/null 2>&1; then
  echo "[$(date +%H:%M)] ⚠️ SFT_SKIPPED.txt present — warm-start returned the BASE model (this is the 0.0 mechanism)." | tee -a "$ML"
fi
if [ ! -f "$M/config.json" ] && [ ! -f "$M/adapter_config.json" ]; then
  echo "[$(date +%H:%M)] TRAIN produced no model — tail:" | tee -a "$ML"; tail -20 /root/train_ft8b.log | tee -a "$ML"; exit 1
fi

# ---- STEP 2: eval othello-hold (vs champ 5GU4) + intercode-solve ----
echo "[$(date +%H:%M)] STEP 2 — eval othello vs 5GU4 (n=40)..." | tee -a "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_vs_winner.py --env othello --games 40 --our-model $MC" \
  > /root/eval_ft8b_othello.log 2>&1
OTH=$(grep -oE "win-rate of decisive games: [0-9]+%" /root/eval_ft8b_othello.log | grep -oE "[0-9]+" | tail -1)
echo "[$(date +%H:%M)] othello vs 5GU4 = ${OTH:-?}%" | tee -a "$ML"

echo "[$(date +%H:%M)] STEP 2 — eval intercode-solve (n=150, peek=8, rep-penalty 1.15)..." | tee -a "$ML"
docker run --rm --gpus all -v "$OUT:/host_outputs:ro" -v "$MINER:/miner_datasets:ro" \
  -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
  --env MINER_DATASETS_DIR=/miner_datasets --env MINER_DATASETS="$DS_MOUNT" \
  --entrypoint bash standalone-text-trainer \
  -c "cd /workspace/scripts && python envs/verify_intercode_solve.py --our-model $MC --n 150 --max-turns 3 --peek 8 --rep-penalty 1.15" \
  > /root/eval_ft8b_intercode.log 2>&1
CMD=$(grep "COMMAND-match" /root/eval_ft8b_intercode.log | grep -oE "[0-9]+\.[0-9]+" | head -1)
F1=$(grep "OUTPUT-F1" /root/eval_ft8b_intercode.log | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "[$(date +%H:%M)] intercode: cmd-match=${CMD:-?} out-f1=${F1:-?}" | tee -a "$ML"

# ---- STEP 3: VERDICT ----
echo "[$(date +%H:%M)] === VERDICT (8B, othello+intercode) ===" | tee -a "$ML"
echo "  othello vs 5GU4 = ${OTH:-?}%   | intercode cmd-match=${CMD:-?} out-f1=${F1:-?}" | tee -a "$ML"
echo "  READ: othello >=50% AND intercode >0  => the 0.0 is FIXED + othello flood TRANSFERS to 8B → advance to multi-env + gin_rummy." | tee -a "$ML"
echo "        othello <50% (esp ~0)           => flood does NOT transfer to 8B → re-tune othello on 8B (data volume / teacher)." | tee -a "$ML"
echo "        intercode ~0 / game-contaminated => intercode broken on 8B → eval-grounded-synthetic intercode (R1 lever)." | tee -a "$ML"
echo "[$(date +%H:%M)] === FIRST 8B TRAIN DONE — see $ML, /root/train_ft8b.log, /root/eval_ft8b_*.log ===" | tee -a "$ML"
