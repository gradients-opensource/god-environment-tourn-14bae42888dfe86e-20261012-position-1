#!/bin/bash
# One-shot diagnostic for the 2026-09-07 exact-0.0 (Llama-3.1-8B). NO training, NO GPU needed,
# ~1 min (downloads only tokenizers). Answers the #1 question: does the 8B chat template RENDER
# our structured tool_calls SFT rows, or drop them (-> few/zero rows survive -> base model
# shipped = 0.0)? Compares 8B vs the 3B we validated on. See scripts/envs/diagnose_8b_sft.py.
#
#   READ the verdict lines:
#   - "all sample rows render + supervise" on 8B  => template is NOT the cause; chase 3B->8B
#     transfer / undertraining / data volume next.
#   - "[FAIL] ... template REJECTS"               => THE smoking gun; rows are silently dropped.
#
# Optional 3rd arg = a generated dataset dir (inside the container) to run the REAL
# train_sft_env.tokenize_and_mask over and report the true survivor count + mask coverage.
set -u
SCR=${SCR:-/root/env-destroy/scripts}     # fresh checkout mounted over the image's baked copy
M8=${M8:-unsloth/Meta-Llama-3.1-8B-Instruct}
M3=${M3:-unsloth/Llama-3.2-3B-Instruct}
DS=${1:-}                                  # optional generated-dataset dir (container path)
ML=/root/diag_8b_sft.log
echo "[$(date +%H:%M)] === 8B SFT render diagnostic (8B=$M8 vs 3B=$M3) ===" | tee "$ML"

run() {  # run() <model> [--dataset-path DIR]
  docker run --rm --gpus all \
    -v "$SCR:/workspace/scripts:ro" -v /root/hf_cache:/root/.cache/huggingface:rw \
    --entrypoint bash standalone-text-trainer \
    -c "cd /workspace/scripts && python envs/diagnose_8b_sft.py --model $1 ${2:-} ${3:-} --show-render"
}

echo "[$(date +%H:%M)] --- TOURNAMENT BASE (8B) ---" | tee -a "$ML"
run "$M8" | tee -a "$ML"
echo "[$(date +%H:%M)] --- BASE WE VALIDATED ON (3B) ---" | tee -a "$ML"
run "$M3" | tee -a "$ML"

if [ -n "$DS" ]; then
  echo "[$(date +%H:%M)] --- REAL tokenize_and_mask over $DS on 8B ---" | tee -a "$ML"
  run "$M8" "--dataset-path" "$DS" | tee -a "$ML"
fi
echo "[$(date +%H:%M)] === DONE — see $ML ===" | tee -a "$ML"
