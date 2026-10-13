"""Hardware-agnostic training budget controller for EnvTask SFT.

The validator runs our training on hardware we cannot profile ahead of time and
hard-kills the container at ``hours_to_complete``. So instead of hardcoding game
counts / step counts (which are wrong on any box but the one we measured on), we
express the whole pipeline as fractions of the wall-clock budget and let the
schedule self-calibrate at run time:

  * GENERATION is time-boxed (a fraction of the budget) and priority-ordered, so
    a slow box still produces the scored envs first; whatever it gets is the data.
  * The LR uses **WSD** (warmup -> stable -> decay) keyed on the WALL CLOCK, not a
    step count: we run the stable phase and trigger decay near the deadline, so we
    always finish on a properly annealed model regardless of how many steps the
    box managed. ``wsd_lr_multiplier`` is that curve.
  * The DATA MIX is round-shape-agnostic: the task's scored envs get the bulk, and
    every other game env gets a thin *coverage / replay* slice. This degrades
    correctly across rounds with no round detection — in R1 (2 scored envs) the
    "other" envs are a thin forward-seed; in later rounds they are replay that
    keeps the continuation model from forgetting; in the final round (all envs
    scored) the coverage set is empty and it's just balanced full coverage.

This module is intentionally torch-free except for an optional, guarded GPU read,
so its logic is unit-testable on any machine.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

# Fractions of the wall-clock budget (sum < 1; the remainder is overhead:
# model load, tokenize, checkpoint save, upload margin).
GEN_FRACTION = 0.30          # time-box for trajectory generation
RESERVE_FRACTION = 0.12      # tail kept free for the final anneal + save/flush
# SFT gets whatever is left between gen end and the reserve tail.

# WSD shape (as fractions of the *SFT* window, i.e. of progress in [0, 1]).
WSD_WARMUP_FRAC = 0.05
WSD_DECAY_FRAC = 0.20
WSD_MIN_MULT = 0.10          # floor LR multiplier at the end of decay (== min_lr_rate)

# Coverage / replay layer.
COVERAGE_FRACTION = 0.15     # share of effort spent on non-scored game envs
COVERAGE_MIN_GAMES = 2_000   # per coverage env, floor so it's not pointless
COVERAGE_MAX_GAMES = 12_000  # per coverage env, cap so it never crowds the scored envs

# ── Data-fit reducer (rank-6) ───────────────────────────────────────────────
# A slow draw (1 GPU) or a many-env round demands MORE trained rows than the SFT
# window can anneal over, which FREEZES the cosine LR near its peak (the othello
# "stuck at 2%" failure) and RE-CREATES undertraining the moment we raise volume.
# This reducer shrinks the per-env scored_cap so the merged set FITS the window and
# the LR fully decays. It re-implements, in OUR style over OUR budget fractions, the
# IDEA behind the R1 winner's hardware/epoch fit-scaling (winner sft_env_config.py:41
# _VALIDATED_ENV_GPU_COUNT and :205-217 scale grad-accum to a validated global batch on
# any GPU allocation) — as a DATA-side cap reducer, with OUR OWN constants (NOT the
# winner's 1.4742 / 0.45 / 2.0, which are unported).
#
# DEFAULT-SAFE: returns 1.0 (no reduction) unless DATA_FIT_SCALE=1. The constants below
# are CONSERVATIVE PLACEHOLDERS and MUST be re-measured on the VM (rows the deadline-
# guarded trainer actually anneals over, per gpu-hour, on BOTH bases) before the default
# is flipped. All overridable via env for the A/B.
DATA_FIT_FLOOR = 0.35              # never shrink a per-env cap below this fraction
DATA_FIT_ROWS_PER_GPU_HOUR = 20_000   # PLACEHOLDER rows/gpu-hour the trainer anneals over — MEASURE ON VM
DATA_FIT_TARGET_ROWS_PER_ENV = 13_500 # PLACEHOLDER per-env rows the winner regime implies (rank-7 target)


def data_fit_scale(
    n_envs: int,
    gpu_count: int,
    hours: float,
    *,
    sft_fraction: float = 1.0 - GEN_FRACTION - RESERVE_FRACTION,
    epochs: int = 1,
) -> float:
    """Multiplicative factor in [DATA_FIT_FLOOR, 1] for the per-env scored cap.

    factor 1.0 = leave the cap untouched (the ONLY behavior unless DATA_FIT_SCALE=1).
    When enabled, shrink the cap when the box can't anneal the demanded rows inside the
    SFT window: capacity = rows/gpu-hour x gpu_count x (sft-window hours); demand =
    n_envs x target_rows/env x epochs; factor = clamp(capacity / demand, floor, 1).
    Never raises; on an absent/zero GPU read (gpu_count <= 0) or nonsense inputs -> 1.0
    (the safe no-op).
    """
    if os.getenv("DATA_FIT_SCALE", "0") != "1":
        return 1.0
    try:
        floor = float(os.getenv("DATA_FIT_FLOOR", DATA_FIT_FLOOR))
        rows_per_gpu_hour = float(os.getenv("DATA_FIT_ROWS_PER_GPU_HOUR", DATA_FIT_ROWS_PER_GPU_HOUR))
        target_per_env = float(os.getenv("DATA_FIT_TARGET_ROWS_PER_ENV", DATA_FIT_TARGET_ROWS_PER_ENV))
        n = max(1, int(n_envs))
        g = int(gpu_count)
        h = float(hours)
        if g <= 0 or h <= 0 or rows_per_gpu_hour <= 0 or target_per_env <= 0:
            return 1.0
        sft_hours = max(0.0, h * sft_fraction)
        capacity = rows_per_gpu_hour * g * sft_hours
        demand = n * target_per_env * max(1, int(epochs))
        if demand <= 0:
            return 1.0
        return max(floor, min(1.0, capacity / demand))
    except Exception:
        return 1.0


def wsd_lr_multiplier(
    progress: float,
    warmup_frac: float = WSD_WARMUP_FRAC,
    decay_frac: float = WSD_DECAY_FRAC,
    min_mult: float = WSD_MIN_MULT,
) -> float:
    """WSD learning-rate multiplier in [min_mult, 1] for wall-clock ``progress`` in
    [0, 1] (0 = SFT start, 1 = anneal deadline).

      * progress < warmup_frac                 -> linear warmup 0 -> 1
      * warmup_frac <= progress < 1-decay_frac -> stable at 1
      * progress >= 1-decay_frac               -> cosine decay 1 -> min_mult
    """
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return min_mult
    if warmup_frac > 0.0 and progress < warmup_frac:
        return max(0.0, progress / warmup_frac)
    decay_start = 1.0 - decay_frac
    if decay_frac > 0.0 and progress >= decay_start:
        frac = (progress - decay_start) / decay_frac          # 0 -> 1 across decay
        cosine = 0.5 * (1.0 + math.cos(math.pi * frac))        # 1 -> 0
        return min_mult + (1.0 - min_mult) * cosine
    return 1.0


def gpu_profile() -> tuple[int, str]:
    """(gpu_count, gpu_name). Guarded — returns (0, "cpu") when torch/CUDA absent,
    so callers (and tests) work on any machine. Used only for coarse logging /
    priors; the schedule is robust without it."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.device_count(), torch.cuda.get_device_name(0)
    except Exception:
        pass
    return 0, "cpu"


# R1 VOLUME RECIPE (2026-09-24 VM A/B vs the REAL tourn_e119 R1 winners, Qwen2.5-7B). R1 is always
# intercode + one game and is decided by per-env RANKS, and our R1 losses came from starved volume:
# gin 7k rows (a past winner flooded gin) and intercode down-weighted to 0.6 as a "commodity" (it
# decided the round). The validated recipe: >= 20k rows per scored env (the game + intercode incl.
# the compliant self-gen), ONE fully annealed epoch, assistant-only packed loss. With it our model
# went 34% -> 52% vs our group's winner and 33% -> 47% vs another past winner at gin, intercode proxy up.
# R1_VOLUME=0 restores the previous small-cap / 2-epoch behavior.
R1_MIN_ENV_ROWS = 20_000
R1_SELFGEN_TASKS = 3_000


def is_r1_volume_round(envs: "list[str] | None") -> bool:
    """True for an R1-shaped round (exactly two scored envs, one of them intercode) unless R1_VOLUME=0."""
    if os.getenv("R1_VOLUME", "1").strip() == "0" or not envs:
        return False
    scored = [e for e in envs if e]
    return len(scored) == 2 and "intercode" in scored


# ── Round-shape share table (2026-09-30) ────────────────────────────────────────────────────────
# Rounds with >= 3 scored envs (R2 = 4 envs, both 6-env boss tasks) are sized as a FIXED total of rows
# split by share, instead of scored_cap x weapon weight per env: the absolute caps made the total grow
# with the env count (22.2k at 4 envs, 36.6k at 6) and the shares implicit (R2 gave leduc 18.9%, an env
# the whole field finished within 46.8-50.8%, while clobber fell to 35% and finished last). The R1 2-env round
# keeps its validated absolute recipe (20k per env). ROUND_SHARES = False restores the 13759ee caps.
ROUND_SHARES = True
SHARE_MIN_ENVS = 3
SHARE_WEIGHT: dict[str, float] = {
    "clobber": 1.6,       # R2 4-env -> 40%: R1 at 50% share reached parity with the champion, R2 at 35% lost
    "othello": 1.4,
    "gin_rummy": 1.4,     # 6-env -> ~20%: at 16% (1 epoch) gin went 6-10 vs a model the 2-env recipe beat 61%
    "intercode": 1.2,     # ~17% of a 6-env round (see intercode_recipe_prefix)
    "liars_dice": 1.0,
    "goofspiel": 1.0,
    "leduc_poker": 0.4,   # near-Nash: rows buy no rank; floored below so a from-scratch base still learns it
}
LEDUC_MIN_SHARE = 0.07
LEDUC_MIN_ROWS = 1_000
# Generation early-stop multiple of the share cap. Slow teachers (othello, gin) are time-bound, so a 4x
# target only burns their slice on rows the merge discards; the fast envs keep GEN_ROW_TARGET_MULT.
SHARE_ROW_TARGET_MULT: dict[str, float] = {"othello": 1.25, "gin_rummy": 1.25, "goofspiel": 1.5, "leduc_poker": 2.0}
LONG_TASK_HOURS = 2.5
# Total rows per shape, sized for ~2 epochs of the Llama-3.1-8B LoRA on the GPUs the validator assigns the
# shape (<= 4 envs: 2x H100, more envs or the 3h from-scratch task: 4x H100). Measured cell: 14.6-14.7k
# packed tok/s on 2x H100 at ~1860 tokens/row (R1/R2 production); the 4-GPU cells assume ~1.9x (unverified).
# Smaller bases train faster but are not scaled up: the epoch cap and generation bound them, and the clock
# stop covers the other direction. A box with fewer GPUs than the shape assumes scales the total down.
_SHAPE_ROWS_2GPU = 18_000
_SHAPE_ROWS_4GPU = 32_000
_SHAPE_ROWS_LONG = 54_000


def is_share_round(envs: "list[str] | set[str] | None") -> bool:
    """True when the round is sized by the share table: >= SHARE_MIN_ENVS scored envs, never R1."""
    names = {e for e in (envs or []) if e}
    return ROUND_SHARES and len(names) >= SHARE_MIN_ENVS and not is_r1_volume_round(sorted(names))


def round_total_rows(envs: "list[str] | None", hours: float, gpu_count: int = 0) -> "int | None":
    """Fixed total of trained rows for a share round (None otherwise: R1 keeps its absolute floor)."""
    if not is_share_round(envs):
        return None
    n = len({e for e in envs if e})
    if hours >= LONG_TASK_HOURS:
        rows, shape_gpus = _SHAPE_ROWS_LONG, 4
    elif n > 4:
        rows, shape_gpus = _SHAPE_ROWS_4GPU, 4
    else:
        rows, shape_gpus = _SHAPE_ROWS_2GPU, 2
    if 0 < gpu_count < shape_gpus:
        rows = int(rows * gpu_count / shape_gpus)
    return rows


def share_caps(envs: "list[str]", total_rows: int, available: "dict[str, int] | None" = None) -> "dict[str, int]":
    """Per-env row caps: ``total_rows`` split by SHARE_WEIGHT (unknown env -> 1.0), leduc floored at
    max(LEDUC_MIN_SHARE, LEDUC_MIN_ROWS), and an env that generated fewer rows than its quota
    (``available``) keeps what it has while the rest of its quota goes to the others by weight."""
    names = list(dict.fromkeys(e for e in envs if e))
    if not names or total_rows <= 0:
        return {}
    avail = available or {}
    weight = {e: SHARE_WEIGHT.get(e, 1.0) for e in names}
    fixed: dict[str, float] = {}
    if "leduc_poker" in weight and len(names) > 1:
        floor = min(total_rows, max(LEDUC_MIN_SHARE * total_rows, LEDUC_MIN_ROWS))
        if total_rows * weight["leduc_poker"] / sum(weight.values()) < floor:
            fixed["leduc_poker"] = min(floor, avail.get("leduc_poker", floor))
    while True:
        free = [e for e in names if e not in fixed]
        rest = max(0.0, total_rows - sum(fixed.values()))
        w_free = sum(weight[e] for e in free) or 1.0
        quota = {e: rest * weight[e] / w_free for e in free}
        short = {e: float(avail[e]) for e in free if e in avail and avail[e] < quota[e]}
        if not short:
            break
        fixed.update(short)
    quota.update(fixed)
    return {e: int(round(quota[e])) for e in names}


def share_row_target(env_name: str, cap: int, default_mult: float) -> int:
    """Generation early-stop target (rows) for an env of a share round."""
    return int(cap * SHARE_ROW_TARGET_MULT.get(env_name, default_mult))


# ── Epoch planner + clock stop (2026-09-30) ─────────────────────────────────────────────────────
# Share rounds plan an UPPER bound of epochs and let the trainer stop on the clock (ClockStop below), so
# the run neither idles (R1 left 25% of its SFT window unused) nor gets cut mid-anneal (R2 stopped at
# 1.66 of 2 epochs). R1 stays at its validated single epoch (its clock stop and the insurance save still run
# there: the stop is a no-op when the planned steps fit). TIME_AWARE_EPOCHS = False restores 13759ee
# (othello anywhere -> 1 epoch, else 2), under which both 6-env boss tasks trained a single epoch.
TIME_AWARE_EPOCHS = True
SHARE_EPOCH_BOUNDS = (1.5, 2.5)
LONG_TASK_EPOCH_BOUNDS = (2, 3)


def plan_epoch_bounds(hours: float, envs: "list[str] | None" = None) -> "tuple[float, float]":
    """(floor, planned) epochs for the SFT pass. The planned value is what the trainer is asked for; the
    clock stop ends earlier when the box is slower, and the trainer warns below the floor. SFT_EPOCHS
    overrides both."""
    override = os.getenv("SFT_EPOCHS")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n, n
        except ValueError:
            pass
    if TIME_AWARE_EPOCHS and is_share_round(envs):
        return LONG_TASK_EPOCH_BOUNDS if hours >= LONG_TASK_HOURS else SHARE_EPOCH_BOUNDS
    n = _fixed_epochs(hours, envs)
    return n, n


def recommend_epochs(hours: float, envs: "list[str] | None" = None) -> "int | float":
    """Epochs passed to the trainer (--num_train_epochs): the planned bound of plan_epoch_bounds."""
    return plan_epoch_bounds(hours, envs)[1]


def _fixed_epochs(hours: float, envs: "list[str] | None" = None) -> int:
    """Epochs for R1 and the rounds outside the share table (all rounds when TIME_AWARE_EPOCHS is off).

    FLOOD-AWARE (2026-09-03, VM-VALIDATED at 72% vs the champion): when the othello
    data-flood is active (othello in the round → weight 6.0 → ~36k rows), run **1 epoch**
    — the big flooded set fills the 1.5h budget in a single annealed pass, and this recipe
    SURPASSED the champion on othello (72%, 29-11-0, up from 13% starved / 5-33). The #1
    lever was FULL TRAINING TIME: the flood + 1 epoch at the real 1.5h budget trains ~60-70
    min (vs the ~9 min a 0.4h smoke gave). A SMALL non-othello round at 1 epoch UNDERtrains
    (finishes its few-hundred steps early, wastes the budget), so those keep **2 epochs** —
    this is the env-BLINDness an earlier review flagged, now fixed by passing the round's
    scored envs in.
    """
    # FLOOD MODE (rank-14, OPT-IN). Default "epoch": an othello round runs 1 epoch (the current
    # VM-VALIDATED behavior — 72% vs the champion). "datacap": keep 2 epochs for the co-trained
    # non-othello envs and instead SHRINK othello's ROW share (see _env_weights OTHELLO_DATACAP_FACTOR
    # in generate_trajectories) so the flood FITS 2 epochs. HIGHEST-RISK config change: do NOT flip the
    # default to "datacap" until a VM A/B confirms 2-epoch + data-cap still holds othello vs MCTS-25.
    # SFT_EPOCHS (plan_epoch_bounds) still overrides everything.
    if envs and "othello" in envs and os.getenv("FLOOD_MODE", "epoch").strip().lower() != "datacap":
        return 1  # othello flood → 1 epoch (VALIDATED 72% vs champ)
    if is_r1_volume_round(envs):
        return 1  # R1 volume recipe: ~40k rows fill the budget in ONE annealed epoch (VALIDATED, see above)
    if hours >= 2.5:
        return 2
    if hours >= 1.0:
        return 2
    return 1


# The trainer times optimizer steps CLOCK_STOP_PROBE[0]..[1] (past the first-step compile / cache warm-up)
# and then ends the run, cosine horizon included, at the last step that fits before the deadline minus
# CLOCK_STOP_MARGIN_S (final save + publish). A run that fits is untouched; a slower box anneals exactly at
# its stop step instead of being cut by the deadline guard. CLOCK_STOP = False restores the plain schedule.
CLOCK_STOP = True
CLOCK_STOP_PROBE = (5, 20)
CLOCK_STOP_MARGIN_S = 90.0


def plan_stop_step(total_steps: int, probe_step: int, s_per_step: float, seconds_left: float,
                   margin_s: float = CLOCK_STOP_MARGIN_S) -> int:
    """Last optimizer step that fits in ``seconds_left`` (measured at ``probe_step``), capped at the plan."""
    if not (s_per_step > 0 and math.isfinite(s_per_step)):
        return total_steps
    fit = probe_step + int(max(0.0, seconds_left - margin_s) // s_per_step)
    return max(probe_step + 1, min(total_steps, fit))


@dataclass
class ClockStop:
    """Measures s/step over the probe window and decides the stop step once. Feed it every step with a
    monotonic clock and the seconds left to the deadline; returns the stop step at the end of the probe."""
    total_steps: int
    probe: "tuple[int, int]" = CLOCK_STOP_PROBE
    margin_s: float = CLOCK_STOP_MARGIN_S
    s_per_step: "float | None" = None
    stop_step: "int | None" = None
    _t_probe: "float | None" = None

    def observe(self, step: int, now_s: float, seconds_left: float) -> "int | None":
        if step == self.probe[0]:
            self._t_probe = now_s
        elif step == self.probe[1] and self._t_probe is not None and self.stop_step is None:
            self.s_per_step = (now_s - self._t_probe) / (self.probe[1] - self.probe[0])
            self.stop_step = plan_stop_step(self.total_steps, step, self.s_per_step, seconds_left, self.margin_s)
            return self.stop_step
        return None


# ── Intercode recipe outside R1 (2026-09-30) ────────────────────────────────────────────────────
# Our R1 intercode (0.9175, #1 of 13 on the same seed) came from real-observation self-gen + no invented
# retries; every other round trained passthrough only (3,600 rows, ~24% submits after an UNRELATED fake
# stdout, 0 real observations). Share rounds that score intercode now run the same self-gen recipe with a
# row budget = intercode's share cap, and submit only after real observations. The R1 prefix is untouched.
# INTERCODE_RECIPE_ALL_ROUNDS = False restores the passthrough-only data outside R1.
INTERCODE_RECIPE_ALL_ROUNDS = True
IC_SELFGEN_TASKS = 1_500       # ~6 rows per task: >= 50% of the largest budget comes from real executions
# BASELINE_STATS_PATH carries the starting model's measured intercode mean. Base Llama-3.1-8B measures
# ~0.756; our R1 model scored 0.9175. A continuation that starts below this has forgotten intercode.
IC_FORGET_THRESHOLD = 0.78
IC_FORGET_DOSE = 1.5
IC_FORGET_MIN_PARAMS = 6_000_000_000   # threshold calibrated on the 8B base only


def intercode_row_budget() -> int:
    """INTERCODE_ROW_BUDGET set by intercode_recipe_prefix (0 when unset or invalid)."""
    try:
        return max(0, int(os.getenv("INTERCODE_ROW_BUDGET", "0") or 0))
    except ValueError:
        return 0


def intercode_recipe_prefix(envs: "list[str]", total_rows: "int | None", dose: float = 1.0) -> str:
    """Env prefix for generate_cmd in a share round that scores intercode ('' otherwise)."""
    if not (INTERCODE_RECIPE_ALL_ROUNDS and total_rows and "intercode" in envs and is_share_round(envs)):
        return ""
    budget = int(share_caps(envs, total_rows).get("intercode", 0) * dose)
    return (f"INTERCODE_SELFGEN=1 INTERCODE_SELFGEN_CONTAINER=1 INTERCODE_SELFGEN_N={IC_SELFGEN_TASKS} "
            f"INTERCODE_RETRY_FRAC=0 INTERCODE_SUBMIT_FRAC=0 INTERCODE_ROW_BUDGET={budget} ")


# Gin "draw upcard" rows are written k times (pvp_selfplay._row_copies, default 4). At k=4 the 3B students
# over-took the upcard (28-38 false upcards per eval run on Qwen-3B and Llama-3B), so bases under
# GIN_SMALL_BASE_PARAMS default to GIN_SMALL_BASE_UPCARD_K; 7B/8B keep the validated k=4.
# GIN_SMALL_BASE_UPCARD_K = 4 restores the uniform k.
GIN_SMALL_BASE_UPCARD_K = 2
GIN_SMALL_BASE_PARAMS = 4_000_000_000


def gin_generate_prefix(envs: "list[str]", param_nums: "int | None",
                        coverage_envs: "list[str] | None" = None) -> str:
    """Env prefix for generate_cmd carrying the small-base upcard k when gin rows are generated ('' otherwise)."""
    if not param_nums or param_nums >= GIN_SMALL_BASE_PARAMS:
        return ""
    if "gin_rummy" not in set(envs or []) | set(coverage_envs or []):
        return ""
    return f"GIN_UPCARD_BASE_K={GIN_SMALL_BASE_UPCARD_K} "


def baseline_env_mean(path: "str | None", env: str) -> "float | None":
    """mean_score of ``env`` in the validator's EnvBaselineStats JSON (env_stats.<env>.mean_score);
    None when the file is absent, unreadable or has no such entry."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            mean = float(json.load(fh)["env_stats"][env]["mean_score"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return mean if math.isfinite(mean) else None


def intercode_dose(mean: "float | None", is_continuation: bool, param_nums: int) -> float:
    """Intercode row-budget multiplier: IC_FORGET_DOSE for a continuation that starts below the threshold."""
    if mean is None or not is_continuation or param_nums < IC_FORGET_MIN_PARAMS:
        return 1.0
    return IC_FORGET_DOSE if mean < IC_FORGET_THRESHOLD else 1.0


def record_baseline(path: str, payload: dict) -> None:
    """Keep what the baseline said for the post-mortem, in a local file (never the public log)."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError:
        pass


@dataclass
class BudgetPlan:
    hours: float
    gen_seconds: int
    sft_seconds: int
    reserve_seconds: int
    epochs: "int | float"
    wsd_warmup_frac: float
    wsd_decay_frac: float
    wsd_min_mult: float
    scored_envs: list[str]            # generated first, full budget
    coverage_envs: list[str]          # generated after, thin slice (replay/seed)
    coverage_games: int               # per coverage env
    gpu_count: int = 0
    gpu_name: str = "cpu"
    notes: list[str] = field(default_factory=list)
    min_epochs: "int | float" = 1     # floor of the epoch plan; the clock stop warns below it


def plan_training(
    hours: float,
    task_envs: list[str],
    all_game_envs: list[str],
    *,
    base_games_per_env: int = 100_000,
    coverage_fraction: float = COVERAGE_FRACTION,
    read_gpu: bool = True,
) -> BudgetPlan:
    """Build the full self-calibrating plan for one round.

    ``task_envs``       : the scored envs for this round (from the validator).
    ``all_game_envs``   : every SFT *game* env we have a teacher for (excl. offline
                          envs like intercode). Coverage = these minus task_envs.
    """
    total_s = max(1, int(hours * 3600))
    gen_seconds = int(total_s * GEN_FRACTION)
    reserve_seconds = int(total_s * RESERVE_FRACTION)
    sft_seconds = max(1, total_s - gen_seconds - reserve_seconds)

    scored = [e for e in task_envs if e in all_game_envs] or list(task_envs)
    # WEAPON-FOCUS (R1 2026-08-17 autopsy): the coverage/replay layer — every NON-scored game env
    # added as a thin slice — SPREAD our R1 pool across many tasks and diluted the env we actually
    # win. The env score is a RELATIVE per-env rank, and the validator supplies the base model each
    # round (continuation is validator-controlled, not ours), so coverage buys little anti-forget
    # while costing rank. Default OFF (focused, matching the R2 winner's small weapon-weighted set);
    # set ENV_COVERAGE=1 to restore the replay layer for an A/B.
    _coverage_on = os.getenv("ENV_COVERAGE", "0") == "1"
    coverage = [e for e in all_game_envs if e not in set(scored)] if _coverage_on else []

    # Per-coverage-env game count: a thin slice of the scored per-env budget.
    n_cov = max(1, len(coverage))
    coverage_games = int(base_games_per_env * coverage_fraction / n_cov)
    coverage_games = max(COVERAGE_MIN_GAMES, min(COVERAGE_MAX_GAMES, coverage_games))
    if not coverage:
        coverage_games = 0

    gpu_count, gpu_name = gpu_profile() if read_gpu else (0, "cpu")

    notes = [
        f"budget={hours:.2f}h -> gen={gen_seconds}s sft={sft_seconds}s reserve={reserve_seconds}s",
        f"scored={scored} coverage={coverage} coverage_games={coverage_games}/env",
        f"gpu={gpu_count}x {gpu_name}",
    ]
    # Full round env list (NOT `scored`, which drops offline envs like intercode): the R1 volume
    # rule keys on intercode being in the round. The othello/hours rules see the same games.
    min_epochs, epochs = plan_epoch_bounds(hours, list(task_envs))

    return BudgetPlan(
        hours=hours,
        gen_seconds=gen_seconds,
        sft_seconds=sft_seconds,
        reserve_seconds=reserve_seconds,
        epochs=epochs,
        wsd_warmup_frac=WSD_WARMUP_FRAC,
        wsd_decay_frac=WSD_DECAY_FRAC,
        wsd_min_mult=WSD_MIN_MULT,
        scored_envs=scored,
        coverage_envs=coverage,
        coverage_games=coverage_games,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        notes=notes,
        min_epochs=min_epochs,
    )
