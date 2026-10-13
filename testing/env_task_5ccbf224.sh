#!/bin/bash
# ============================================================
# TASK 5ccbf224 REPLICA — the REAL R2 / Group-Stage-2 task, run locally to validate vs the 4 champs.
#
# Exact task config (from the Gradients arena task JSON the user pasted):
#   base  = unsloth/Meta-Llama-3.1-8B-Instruct
#   hours = 1.5
#   type  = EnvTask
#   envs  = [intercode, goofspiel, othello, clobber]
#
# WHY this is the best pre-tournament test: it's the SAME shape a real round uses (4 envs, 1.5h budget),
# and all 4 opponents are public (task-5ccbf224 repos), so after training we PvP our model vs the per-game
# CHAMPS — clobber vs 5EEa, othello vs the Boss 5GU4Xkd3 (verify_vs_winner picks them automatically).
#
# Our improvements baked in (all tunable):
#   - othello uses the fast champ-level EXPERT engine (OTHELLO_TEACHER=off) — 46% vs the othello champ,
#     and fast in self-play (no model inference) so it doesn't starve gen like the slow MCTS-rollouts=8 did.
#   - clobber uses the MCTS-100 teacher (93% vs 5EEa) — unchanged default.
#   - SCORED_CAP + GEN_SECONDS raised to pack MORE strong data into the budget (the distillation-volume fix).
#     NB: SCORED_CAP only bites if the image was REBUILT after commit cd63246 — rebuild first!
#
# Needs: H100, both Docker images BUILT, and the intercode dataset mounted (intercode env REQUIRES it).
#
# RUN IN BACKGROUND (survives disconnect):
#   nohup bash testing/env_task_5ccbf224.sh > /tmp/task5ccbf224.log 2>&1 &
#   tail -f /tmp/task5ccbf224.log      # Ctrl-C stops watching; training keeps running
# ============================================================

TASK_ID="5ccbf224-23a0-4d12-adeb-7023e07ed06e"          # the REAL task id (used as the local output-dir label)
MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"     # the task's exact base
DATASET="env_task_dummy_dataset"
DATASET_TYPE='{"environment_names": ["intercode", "goofspiel", "othello", "clobber"]}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-1.5}"            # the task's REAL budget (bump only to explore the volume ceiling)
EXPECTED_REPO_NAME="llama31-8b-task5ccbf224"

# --- our improvements are now BAKED INTO THE CODE DEFAULTS (2026-08-02) ---
# Leaving these EMPTY makes the run use the exact config the tournament validator gets (no env vars):
#   othello teacher = expert (pvp_selfplay._MCTS_TEACHER_DEFAULT_ENVS dropped othello),
#   scored_cap = 50000 (sft_env_config default), goofspiel = near-greedy, gen_seconds = computed.
# THIS IS THE TRUE VALIDATOR REPLICA — a clean clobber ~57% / goofspiel win here = deploy-ready.
# Set any knob to force an override for experiments.
OTHELLO_TEACHER="${OTHELLO_TEACHER:-}"
SCORED_CAP="${SCORED_CAP:-}"
GEN_SECONDS="${GEN_SECONDS:-}"
ENV_PACK="${ENV_PACK:-}"   # experimental: ENV_PACK=1 packs short game rows -> trains MUCH more of the pool (default OFF)

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
MINER_DATASETS_LIST="$INTERCODE_NAME"

if [ ! -d "$MINER_DATASETS_HOST/$INTERCODE_NAME" ]; then
  echo "!!! Intercode dataset (REQUIRED by the intercode env) NOT found: $MINER_DATASETS_HOST/$INTERCODE_NAME"
  echo "    Download it first (see testing/env_clobber_intercode_test.sh header), or the intercode env can't run."
  exit 1
fi

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " TASK 5ccbf224 REPLICA (R2 / Group-Stage 2)"
echo " Model      : $MODEL"
echo " Envs       : intercode + goofspiel + othello + clobber"
echo " Hours      : $HOURS_TO_COMPLETE   (the task's real budget)"
echo " Othello    : teacher=$OTHELLO_TEACHER (off = fast expert engine, 46% vs the othello champ)"
echo " SCORED_CAP : $SCORED_CAP   GEN_SECONDS: ${GEN_SECONDS:-<computed>}"
echo " NB: if the gen log shows '--scored_cap 3500' not $SCORED_CAP, the image is STALE -> rebuild it."
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

echo "Training (task 5ccbf224 replica)..."
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
  --env OTHELLO_TEACHER="$OTHELLO_TEACHER" \
  --env SCORED_CAP="$SCORED_CAP" \
  --env GEN_SECONDS="$GEN_SECONDS" \
  --env ENV_PACK="$ENV_PACK" \
  --name env-task5ccbf224-trainer \
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
echo " Task 5ccbf224 replica DONE. VALIDATE vs the per-game CHAMPS (clobber+othello are PvP-measurable):"
echo "   cd $(pwd)/scripts"
echo "   OUT=$OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "   ~/env-destroy/.venv/bin/python envs/verify_vs_winner.py --env clobber --games 40 --our-model \$OUT   # vs 5EEa (clobber champ)"
echo "   ~/env-destroy/.venv/bin/python envs/verify_vs_winner.py --env othello --games 40 --our-model \$OUT   # vs Boss 5GU4Xkd3 (othello champ)"
echo " >55% = we out-play the champ on that differentiator. (intercode/goofspiel aren't locally PvP-measurable.)"
echo "================================================"
