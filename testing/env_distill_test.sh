#!/bin/bash
# ============================================================
# DISTILLATION EXPERIMENT — Llama-3.1-8B, clobber+othello, HIGH data cap + upgraded othello teacher.
#
# The 2026-08-02 gold-standard PvP diagnostic proved our TEACHERS beat the R1 winner's MODEL
# (clobber teacher 93%, othello teacher 87% at rollouts=8) while our R1 MODELS lost (clobber 33%,
# othello 3%) — a DISTILLATION gap from too little data (7k cap -> ~14k rows vs the winner's ~260k).
# This run tests the fix: BALANCE_SCORED_CAP=50000 (more data) + OTHELLO_MCTS_ROLLOUTS=8 (the
# upgraded teacher). If the model learns the teachers, its PvP win-rate vs #2 should JUMP.
#
# Needs: H100, Docker + the two images built, intercode dataset present (mounted, unused by 2 games).
#
# RUN IN BACKGROUND (survives logout, ~1.5h):
#   nohup bash testing/env_distill_test.sh > /tmp/distill.log 2>&1 &
#   tail -f /tmp/distill.log
# ============================================================

TASK_ID="d1571100-de57-4000-8000-000000000001"
MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"        # 8B = comparable to #2 in verify_vs_winner
DATASET="env_task_dummy_dataset"
DATASET_TYPE='{"environment_names": ["clobber", "othello"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-1.5}"
EXPECTED_REPO_NAME="llama31-8b-distill-clobber-othello"

# --- the experiment knobs (env-tunable) ---
# THE REAL data lever = --scored_cap (the GEN cap). Round 1 of this experiment proved the pipeline
# computed only scored_cap=3500 for 1.5h -> train=13000 (~= R1); the merge cap (BALANCE_SCORED_CAP)
# was NOT the binding limit. Raise SCORED_CAP to actually generate more games.
SCORED_CAP="${SCORED_CAP:-50000}"
BALANCE_SCORED_CAP="${BALANCE_SCORED_CAP:-50000}"      # merge cap (it DID help clobber 33->47%; othello stayed gen-limited below it)
# Othello teacher choice. "off" = the hand-written negamax + PERFECT-endgame EXPERT engine; "mcts" = the
# OpenSpiel MCTS teacher. MEASURED 2026-08-02 (verify_vs_winner --our-expert vs the REAL othello champ,
# task-5ccbf224 Boss 5GU4Xkd3): the expert @ budget 0.08s = 46% (EVEN with the champ) AND is FAST for gen
# (self-play has no model inference). That's champ-level strength WITHOUT the MCTS-rollouts=8 slowdown that
# starved othello gen to 0% in round 1. So the expert is the better othello teacher for distillation → default off.
OTHELLO_TEACHER="${OTHELLO_TEACHER:-off}"
OTHELLO_MCTS_ROLLOUTS="${OTHELLO_MCTS_ROLLOUTS:-1}"   # only used if OTHELLO_TEACHER=mcts
# GEN time (seconds). Empty -> the pipeline's computed default (hours * 3600 * gen_ratio, ~2160 for 4h).
# othello gen is SLOW; if its merged row count is starved, raise this (e.g. 3600 = 1h of pure gen) so
# othello actually PRODUCES enough games for the scored_cap to keep. This is the direct throughput knob.
GEN_SECONDS="${GEN_SECONDS:-}"
# To reach the winner's ~260k games you likely need MORE gen time (raise GEN_SECONDS/HOURS_TO_COMPLETE)
# or an offline PREBAKE — the time-boxed MCTS gen cannot hit 260k in a 1.5h round.

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$INTERCODE_NAME"   # clobber+othello don't need it; mounted harmlessly if present

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " DISTILLATION EXPERIMENT"
echo " Model      : $MODEL"
echo " Envs       : clobber + othello"
echo " SCORED_CAP : $SCORED_CAP   (the REAL merged-row cap; R1 computed only 3500)"
echo " GEN_SECONDS: ${GEN_SECONDS:-<computed>}   (othello gen throughput knob)"
echo " Othello    : teacher=$OTHELLO_TEACHER (off = fast expert engine, 46% vs the othello champ)"
echo " Hours      : $HOURS_TO_COMPLETE"
echo "================================================"
echo " NB: if SCORED_CAP still shows 7000 in the gen command below, the Docker image is"
echo "     STALE — rebuild it: docker build -f dockerfiles/standalone-text-trainer.dockerfile -t standalone-text-trainer ."
echo "================================================"

echo "Downloading model..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

echo "Training (distillation experiment)..."
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
  --env SCORED_CAP="$SCORED_CAP" \
  --env GEN_SECONDS="$GEN_SECONDS" \
  --env BALANCE_SCORED_CAP="$BALANCE_SCORED_CAP" \
  --env OTHELLO_TEACHER="$OTHELLO_TEACHER" \
  --env OTHELLO_MCTS_ROLLOUTS="$OTHELLO_MCTS_ROLLOUTS" \
  --name env-distill-trainer \
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
echo " Distillation experiment done. CHECK the log — in ORDER:"
echo "   1. THE KEY DIAGNOSTIC — per-env merged row counts (is OTHELLO starved?):"
echo "        grep -Ei '\[balance\]|\[combine\]|othello|clobber' /tmp/distill3.log | grep -Ei 'row|train|cap|=|:' "
echo "      othello 3%/0% at eval smells like STARVATION: if othello contributes FAR fewer rows than"
echo "      clobber, gen THROUGHPUT is the bottleneck (raise GEN_SECONDS) — not the cap, not the teacher."
echo "   2. The gen command line shows '--scored_cap $SCORED_CAP' (NOT 7000 = image was rebuilt)."
echo "   3. NO '[pvp_selfplay] WARNING ... MCTS teacher unavailable'  (strong teachers active)."
echo "   4. 'Training successfully done'."
echo ""
echo " NEXT — PvP the NEW model vs the R1 winner (#2). Use 40 games (30 has ~+/-9% noise):"
echo "   cd $(pwd)"
echo "   ~/env-destroy/.venv/bin/python envs/verify_vs_winner.py --env clobber --games 40 --our-model $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "   ~/env-destroy/.venv/bin/python envs/verify_vs_winner.py --env othello --games 40 --our-model $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "================================================"
