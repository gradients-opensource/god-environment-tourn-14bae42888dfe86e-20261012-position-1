#!/bin/bash
# ============================================================
# GENERIC EnvTask replica — validate the pipeline on ANY base model + env combo, exactly like the
# validator (clone repo -> generate -> train -> save). Built to test Qwen2.5-7B (the current tournament
# base — we'd only ever verified Llama-3.1-8B) on the real R1/R2 env combos, then PvP the result vs the
# champ 5EEaxgnm.
#
# USAGE (env-driven):
#   # R1 of the current tourney (leduc+liars, Qwen) — no dataset needed (game envs = self-play):
#   MODEL="Qwen/Qwen2.5-7B-Instruct" ENVS="leduc_poker liars_dice" nohup bash testing/env_task_custom.sh > /tmp/qwen_r1.log 2>&1 &
#
#   # R2 (liars+goofspiel+leduc+intercode, Qwen) — needs the intercode dataset mounted:
#   MODEL="Qwen/Qwen2.5-7B-Instruct" ENVS="liars_dice goofspiel leduc_poker intercode" nohup bash testing/env_task_custom.sh > /tmp/qwen_r2.log 2>&1 &
#
# All the winning DEFAULTS are baked in (ENV_PACK off), so this is the true validator experience.
# Needs: H100, both Docker images BUILT (rebuild after code changes!), intercode dataset ONLY if 'intercode' in ENVS.
# ============================================================

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"          # the current tournament base (was Llama-3.1-8B before)
ENVS="${ENVS:-leduc_poker liars_dice}"               # space-separated scored envs for this task
HOURS_TO_COMPLETE="${HOURS_TO_COMPLETE:-1.5}"        # the real tournament budget
TASK_ID="${TASK_ID:-cc000000-0000-4000-8000-000000000001}"
DATASET="env_task_dummy_dataset"
FILE_FORMAT="s3"

# Build the JSON env list + a readable repo name from ENVS.
_envs_json=$(printf '"%s",' $ENVS | sed 's/,$//')
DATASET_TYPE="{\"environment_names\": [$_envs_json]}"
_model_short=$(echo "$MODEL" | sed 's#.*/##; s/[^A-Za-z0-9]/-/g' | tr '[:upper:]' '[:lower:]')
_envs_short=$(echo $ENVS | tr ' ' '-')
EXPECTED_REPO_NAME="${EXPECTED_REPO_NAME:-custom-${_model_short}-${_envs_short}${REPO_SUFFIX:+-$REPO_SUFFIX}}"

MINER_DATASETS_HOST="${MINER_DATASETS_HOST:-$(pwd)/miner_datasets}"
INTERCODE_NAME="gradients-io-tournaments--intercode_bigcode_combined_12k"
SWE_NAME="gradients-io-tournaments--SWE-ZERO-12M-trajectories-filtered"
# Build the comma-separated MINER_DATASETS list from the OFFLINE envs present in ENVS
# (intercode and/or swe_infinite). Game envs self-play and need no dataset.
_ds_entries=()
if echo " $ENVS " | grep -qw intercode; then
  if [ ! -d "$MINER_DATASETS_HOST/$INTERCODE_NAME" ]; then
    echo "!!! 'intercode' is in ENVS but the intercode dataset is NOT found: $MINER_DATASETS_HOST/$INTERCODE_NAME"
    echo "    Download it first, or drop intercode from ENVS."
    exit 1
  fi
  _ds_entries+=("$INTERCODE_NAME")
fi
if echo " $ENVS " | grep -qw swe_infinite; then
  if [ ! -d "$MINER_DATASETS_HOST/$SWE_NAME" ]; then
    echo "!!! 'swe_infinite' is in ENVS but the SWE-ZERO dataset is NOT found: $MINER_DATASETS_HOST/$SWE_NAME"
    echo "    Download it first (snapshot_download SWE-ZERO-12M-trajectories-filtered), or drop swe_infinite from ENVS."
    exit 1
  fi
  _ds_entries+=("$SWE_NAME")
fi
# comma-separated -> matches miner_dataset_loader._get_miner_datasets_inventory; empty for pure game envs
MINER_DATASETS_LIST=$(IFS=,; echo "${_ds_entries[*]}")

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 700 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

echo "================================================"
echo " GENERIC EnvTask replica (validator experience, PURE defaults)"
echo " Model : $MODEL"
echo " Envs  : $ENVS"
echo " Hours : $HOURS_TO_COMPLETE"
echo " Repo  : $EXPECTED_REPO_NAME"
echo " Data  : ${MINER_DATASETS_LIST:-<none (game envs self-play)>}"
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

# Forward A/B knobs into the container. Generation + training run INSIDE the container, so host
# env vars must be passed with --env or they never reach clobber_endgame.py / sft_env_config.py /
# goofspiel_expert.py. Only set ones are forwarded (unset -> default behaviour unchanged).
EXTRA_ENV_ARGS=()
[ -n "$CLOBBER_LEAF_EVAL" ]       && EXTRA_ENV_ARGS+=(--env "CLOBBER_LEAF_EVAL=$CLOBBER_LEAF_EVAL")
[ -n "$CLOBBER_MOBILITY_LADDER" ] && EXTRA_ENV_ARGS+=(--env "CLOBBER_MOBILITY_LADDER=$CLOBBER_MOBILITY_LADDER")
[ -n "$CLOBBER_TIE_MARGIN" ]      && EXTRA_ENV_ARGS+=(--env "CLOBBER_TIE_MARGIN=$CLOBBER_TIE_MARGIN")
[ -n "$SFT_EFFECTIVE_BATCH" ]     && EXTRA_ENV_ARGS+=(--env "SFT_EFFECTIVE_BATCH=$SFT_EFFECTIVE_BATCH")
[ -n "$SCORED_CAP" ]              && EXTRA_ENV_ARGS+=(--env "SCORED_CAP=$SCORED_CAP")
[ -n "$GOOFSPIEL_TEACHER" ]       && EXTRA_ENV_ARGS+=(--env "GOOFSPIEL_TEACHER=$GOOFSPIEL_TEACHER")
[ -n "$SFT_SMALL_LORA" ]          && EXTRA_ENV_ARGS+=(--env "SFT_SMALL_LORA=$SFT_SMALL_LORA")
[ -n "$SFT_SMALL_LORA_R" ]        && EXTRA_ENV_ARGS+=(--env "SFT_SMALL_LORA_R=$SFT_SMALL_LORA_R")
[ -n "$SFT_SMALL_LORA_ALPHA" ]    && EXTRA_ENV_ARGS+=(--env "SFT_SMALL_LORA_ALPHA=$SFT_SMALL_LORA_ALPHA")
[ -n "$SFT_SMALL_LORA_DROPOUT" ]  && EXTRA_ENV_ARGS+=(--env "SFT_SMALL_LORA_DROPOUT=$SFT_SMALL_LORA_DROPOUT")
[ -n "$BALANCE_FLOOR_FRAC" ]      && EXTRA_ENV_ARGS+=(--env "BALANCE_FLOOR_FRAC=$BALANCE_FLOOR_FRAC")
# WEAPON-FOCUS recipe (ported from the R2 winner): per-env weighting, coverage toggle, LoRA rank.
[ -n "$ENV_WEIGHTS" ]             && EXTRA_ENV_ARGS+=(--env "ENV_WEIGHTS=$ENV_WEIGHTS")
[ -n "$ENV_WEAPON_FOCUS" ]        && EXTRA_ENV_ARGS+=(--env "ENV_WEAPON_FOCUS=$ENV_WEAPON_FOCUS")
[ -n "$ENV_COVERAGE" ]            && EXTRA_ENV_ARGS+=(--env "ENV_COVERAGE=$ENV_COVERAGE")
[ -n "$SFT_LORA_R" ]              && EXTRA_ENV_ARGS+=(--env "SFT_LORA_R=$SFT_LORA_R")
[ -n "$SFT_LORA_ALPHA" ]          && EXTRA_ENV_ARGS+=(--env "SFT_LORA_ALPHA=$SFT_LORA_ALPHA")
[ -n "$SFT_LORA_DROPOUT" ]        && EXTRA_ENV_ARGS+=(--env "SFT_LORA_DROPOUT=$SFT_LORA_DROPOUT")

echo "Training..."
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
  "${EXTRA_ENV_ARGS[@]}" \
  --name env-custom-trainer \
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
echo " DONE. CHECK the log:"
echo "   - Model loads on $MODEL (no chat-template / tokenizer crash = Qwen-compatible!)"
echo "   - NO '[pvp_selfplay] WARNING ... MCTS teacher unavailable'"
echo "   - 'Training successfully done' + model saved"
echo " Output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo ""
echo " Then PvP the model vs the champ (per-game champ auto-picked; add leduc/liars once verify supports them):"
echo "   cd $(pwd)/scripts && OUT=$OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
echo "   ~/env-destroy/.venv/bin/python envs/verify_vs_winner.py --env goofspiel --games 40 --our-model \$OUT"
echo "================================================"
