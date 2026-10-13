"""
Generate game trajectories against env servers and save as an HF DatasetDict
(train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Run from /workspace/scripts/:
  # Single environment (--num_games / --max_turn override the built-in defaults)
  python -m envs.generate_trajectories --environment_names liars_dice \
      --output_path /path/to/dataset --num_games 50000

  # Multiple environments — uses built-in per-env defaults for num_games/max_turn,
  # generates each to a staging path then merges into output_path
  python -m envs.generate_trajectories \
      --environment_names gin_rummy liars_dice leduc_poker \
      --output_path /path/to/dataset

  # Mixed: intercode (offline, reads from MINER_DATASETS) + game envs (env server)
  python -m envs.generate_trajectories \
      --environment_names gin_rummy intercode \
      --output_path /path/to/dataset

Score-based sampling:
  Some generators (e.g. leduc_poker) return (messages, score) tuples.  When
  --sample-by-score is set, each game is kept with probability
  clamp(score, 0, 1) ** score_power.  --wins-only is a stricter filter that
  discards any game where score <= 0.  For generators that return only
  messages (no score), all games are kept regardless of these flags.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

import trl_compat  # noqa: F401 — inject trl.experimental shim before env imports
from budget_controller import intercode_row_budget, is_share_round, share_caps, share_row_target

from envs.shared_env import GAMES_TO_TASK_ID_RANGE, _log, generation_id_bounds, init_env_pool, recipe_log
from envs.sft_env_configs import _OFFLINE_ENVS, _SELFPLAY_ENVS, get_sft_trajectory_generator


# ── Process-pool worker ───────────────────────────────────────────────────────
# Each worker process loads the expert generator once via _worker_init, then
# handles multiple games sequentially. Using processes (not threads) gives each
# worker its own GIL so CPU-bound expert computation runs truly in parallel
# without contention. --num_workers controls how many concurrent env server
# connections are open, letting you tune without overloading either side.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)
    # Each worker process needs its own in-memory MCCFR blueprint; disk cache
    # (written by the parent pre-warm) makes this a fast load rather than a
    # multi-minute 60k-iteration solve per worker.
    if env_name == "liars_dice":
        try:
            from envs.liar_dice_expert import _get_blueprint, blueprint_needed

            if blueprint_needed():
                _get_blueprint(10)
        except Exception:
            pass


def _prewarm_env(env_name: str) -> None:
    """Warm expensive per-env teacher blueprints IN THE PARENT before spawning
    workers. The pool forks on Linux, so a blueprint loaded/solved here is
    inherited (copy-on-write) by every worker — avoiding a redundant per-worker
    solve (liars_dice MCCFR ~minutes cold; leduc CFR+ ~seconds)."""
    try:
        t0 = time.monotonic()
        if env_name == "liars_dice":
            from envs.liar_dice_expert import _get_blueprint, blueprint_needed

            if blueprint_needed():  # only the opt-in MCCFR label reads it
                _log("Pre-warming liars_dice MCCFR blueprint (total_dice=10)...", flush=True)
                _get_blueprint(10)
                _log(f"liars_dice blueprint ready in {time.monotonic() - t0:.1f}s", flush=True)
        elif env_name == "leduc_poker":
            from envs.leduc_poker_expert import _bp, get_chips_pool_blueprint

            from envs.pvp_selfplay import _leduc_hero_pool_on

            _log("Pre-warming leduc_poker CFR+ blueprints...", flush=True)
            _bp()
            # Hero-seat generation's opponent pool also samples a CHIPS-objective blueprint
            # (pool_action kind='chips_cfr'); solving it here too avoids every forked worker
            # racing to solve it concurrently on first use (~87s cold, see get_chips_pool_blueprint).
            # Only that pool reads it, so LEDUC_HERO_POOL=0 does not pay for it.
            if _leduc_hero_pool_on():
                get_chips_pool_blueprint()
            _log(f"leduc_poker blueprints ready in {time.monotonic() - t0:.1f}s", flush=True)
    except Exception as exc:
        _log(f"Blueprint pre-warm failed for {env_name} (workers will retry): {exc}", flush=True)


def _worker_play(
    game_id: int, endpoint: str, max_turn: int
) -> "list[dict] | tuple[list[dict], float] | None":
    return _GENERATE_FN(game_id, endpoint, max_turn)

# ─────────────────────────────────────────────────────────────────────────────

MIN_ASSISTANT_TURNS = 1

# Envs whose SFT generator returns a list of already-flattened, stateless
# per-turn examples (``{"messages": [...], "tools": ...}``, §3.3/§3.4) instead
# of one growing conversation per game. These skip _clean/_sliding_windows
# entirely — each item is already a complete row. A generator may instead
# return ``(examples, score)`` (e.g. leduc_poker) — see the score-filter
# handling in generate_for_env, which applies wins_only/sample_by_score to the
# whole game's examples at once before flattening.
#
# Every self-play env (pvp_selfplay.play_selfplay_game) returns the flat
# ``[(examples, score), ...]`` views format, so _FLAT_EXAMPLE_ENVS must be a
# superset of _SELFPLAY_ENVS. Union them explicitly so a newly-added self-play
# env (e.g. clobber) can't silently fall through to the raw-conversation path
# and hit ``'tuple' object has no attribute 'get'`` in _clean.
_FLAT_EXAMPLE_ENVS: frozenset[str] = frozenset(
    {"liars_dice", "gin_rummy", "leduc_poker", "othello", "goofspiel"}
) | _SELFPLAY_ENVS


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """
    Split a conversation into overlapping sub-conversations.
    Each window: [system] + window_turns × (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    """Normalize a raw conversation, preserving tool_calls and content: None.

    Plain-text generators emit string content, which passes through unchanged.
    Tool-calling generators may emit content: None alongside tool_calls — both
    are kept as-is (no stringification, no dropping) so the row matches the
    eval-time message shape.
    """
    if not messages:
        return None
    cleaned = []
    for m in messages:
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)
        out = {"role": m["role"], "content": content}
        if m.get("tool_calls") is not None:
            out["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id") is not None:
            out["tool_call_id"] = m["tool_call_id"]
        cleaned.append(out)
    messages = cleaned
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _clean_tool_example(example: "dict | None") -> "dict | None":
    """Validate one flattened per-turn example for _FLAT_EXAMPLE_ENVS.

    Mirrors _clean's role for these envs: keep tool_calls/tools/content as-is
    (no stringification), but fail loudly (return None, counted as skipped) if
    the example isn't shaped as exactly one assistant message ending in a
    single game_action tool call — a malformed example here is exactly the
    defect this restructuring is meant to eliminate from the training
    distribution.
    """
    if not example or "messages" not in example or "tools" not in example:
        return None
    messages = example["messages"]
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        return None
    tool_calls = messages[-1].get("tool_calls") or []
    if len(tool_calls) != 1:
        return None
    if tool_calls[0].get("function", {}).get("name") != "game_action":
        return None
    return example


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


# Per-environment generation defaults. Applied for both single and multi-env paths;
# CLI flags (--num_games, --max_turn, --score-power) override when explicitly provided.
#
# Tuned for LOCAL self-play (pvp_selfplay.py): both seats are the teacher, so we
# KEEP BOTH seats' moves (no wins-only/sample_by_score — a strong teacher's moves
# are good even from the losing seat, and we want 2x yield). num_games for the
# very fast envs (liars_dice/leduc) is capped so they don't swamp the merged mix:
# liars_dice self-plays ~hundreds of games/s, so a few thousand games already
# gives tens of thousands of per-turn examples. The slow envs (gin_rummy/othello)
# are time-bounded well before their num_games cap, and get the bulk of the
# generation budget via _GEN_TIME_WEIGHT below. mcts_simulations is unused by
# self-play (no server opponent) but kept for the legacy env-server path.
_ENV_DEFAULT_ARGS: dict[str, dict] = {
    "gin_rummy":   {"num_games": 8000,  "max_turn": 200, "mcts_simulations": 25},
    "liars_dice":  {"num_games": 8000,  "max_turn": 30,  "mcts_simulations": 225},
    # num_games 8500 (leduc v2 2026-09-30): hero-seat generation (play_leduc_hero_game,
    # pvp_selfplay.py) records only ONE seat per game instead of both, at 2.57 rows/game measured
    # (8000 games -> 20579 rows in 55.4s on this CPU, single-process), so games must stay high
    # enough that raw row volume clears R1's R1_MIN_ENV_ROWS=20000 floor (_scored_cap_for below)
    # with real margin rather than landing right at it. 8500 games -> ~21.9k raw rows in well
    # under a minute -- trivial against the ~810s R1 per-env generation slice. Row weight/floor/cap
    # are recipe-owned (_DEFAULT_ENV_WEIGHT / _scored_cap_for) and are not touched here.
    "leduc_poker": {"num_games": 8500,  "max_turn": 10,  "mcts_simulations": 50},
    # othello: hero-seat rows (othello_hero.py) record one seat, ~28 rows per game, so 8000 games stay far
    # above every row target and the slice's clock is what bounds generation.
    "othello":     {"num_games": 8000,  "max_turn": 70,  "mcts_simulations": 25},
    # goofspiel: one turn per prize card, so num_cards (13) + small buffer caps
    # max_turn. Fast (heuristic teacher), so capped num_games like the other
    # near-instant envs to avoid swamping the merged mix; each game yields one row per
    # hero-seat bid (~8 on the 5/8/10/13 mix; envs/goofspiel_hero.py records one seat).
    # max_turn counts decision turns; eval goofspiel is turn-based (2 per prize), 13 cards = 26.
    "goofspiel":   {"num_games": 8000,  "max_turn": 30,  "mcts_simulations": 25},
    # clobber: hero-seat rows (clobber_hero.py) carry the deterministic label on every recorded
    # move, so rows from lost games are kept too (swindle play for the half of eval seats that
    # start lost). K=3 rows per game -> ~2x the distinct games per row. 13759ee: num_games 8000 +
    # sample_by_score True, which on draw-free clobber kept only the winning seat.
    "clobber":     {"num_games": 12000, "max_turn": 60,  "mcts_simulations": 25},
}
# Relative share of the scored-env generation time budget. Self-play throughput
# varies ~100x across envs (liars_dice/leduc are near-instant; gin_rummy/othello
# run a heuristic/minimax per move), so an equal split would let the fast envs
# dominate the dataset. Weight time toward the slow envs to balance example
# counts. Fast envs also hit their (small) num_games cap quickly and return
# early. Default weight 1.
_GEN_TIME_WEIGHT: dict[str, float] = {
    "liars_dice": 1.0,
    "leduc_poker": 1.0,
    "gin_rummy": 6.0,
    "othello": 3.0,   # was 8.0. Only rounds with several games split time by these weights (in share rounds
                      # x the share cap); R1's single game gets the whole budget whatever its weight.
    "goofspiel": 1.0,
    # Clobber's hero-seat label runs ~72 rows/s on 8 cores (0.09-0.10 CPU-s/row). At 1.0 the 6-env 1.5h
    # round gave it a 73 s base slice after the leduc pre-warm, short of its 7215-row cap without rollover
    # from gin; 2.0 gives ~133 s there (and 1.5 x the cap-fill time in every shape at 0.75x CPU speed),
    # while gin and othello still fill their caps (test_gen_time_slices.py).
    "clobber": 2.0,
}
# REBALANCED generation-time weights (rank-7, OPT-IN via GEN_TIME_WEIGHT_V2=1). Raising gin/othello
# ROW targets without also giving clobber/goofspiel their share of gen time would starve them (rank-7:
# "do NOT just inflate gin and starve othello/clobber/goofspiel"). These are CONSERVATIVE PLACEHOLDERS
# — measure per-env self-play rows/sec on the validator box and retune before flipping the default.
# Default (flag unset) uses _GEN_TIME_WEIGHT above, so committing this changes NOTHING.
_GEN_TIME_WEIGHT_V2: dict[str, float] = {
    "liars_dice": 1.0,
    "leduc_poker": 1.0,
    "gin_rummy": 6.0,
    "othello": 8.0,
    "goofspiel": 2.0,
    "clobber": 3.0,
}


def _gtw(env_name: str) -> float:
    """Generation-time weight for an env. GEN_TIME_WEIGHT_V2=1 selects the rebalanced map (rank-7);
    default returns the current _GEN_TIME_WEIGHT value (unchanged)."""
    src = _GEN_TIME_WEIGHT_V2 if os.getenv("GEN_TIME_WEIGHT_V2", "0") == "1" else _GEN_TIME_WEIGHT
    return src.get(env_name, 1.0)


_DEFAULT_ARGS: dict = {"num_games": 50000, "max_turn": 30, "mcts_simulations": 10}


def _generate_offline(env_name: str, output_path: str) -> bool:
    """Generate dataset for offline envs that don't use an env server.

    Returns True if data was successfully saved, False if the source was
    unavailable (caller should skip this env rather than abort).
    """
    if env_name == "intercode":
        # COMPLIANT source only. SFT rows come from the validator's WHITELISTED passthrough
        # dataset (gradients-io-tournaments/intercode_bigcode_combined_12k), converted to
        # eval-exact execute_bash/submit tool-call rows by intercode_dataset.py. We do NOT
        # replicate the eval's filesystem — no /testbed bake into the image, and no runtime
        # trajectory generation executed against a copy of it. Per the validator's 2026-09
        # rule, deriving training signal from a replica of the eval environment reveals the
        # test set and is DQ-able REGARDLESS of the mechanism (image-bake vs trajectory-gen):
        # what's banned is grounding on the eval's own filesystem, not the file format.
        from envs.intercode_dataset import build_intercode_sft_dataset

        dd = build_intercode_sft_dataset()
        if dd is None:
            print(
                "[generate_trajectories] No data for env 'intercode', skipping it: "
                "whitelisted passthrough dataset unavailable (training continues with other "
                "envs). Ensure MINER_DATASETS contains "
                "'gradients-io-tournaments--intercode_bigcode_combined_12k' and "
                "MINER_DATASETS_DIR is set.",
                flush=True,
            )
            return False
        dd.save_to_disk(output_path)
        _log(f"Intercode dataset saved → {output_path} ({len(dd['train'])} examples; passthrough)")
        return True
    if env_name == "swe_infinite":
        from envs.swe_infinite_dataset import build_swe_infinite_sft_dataset
        dd = build_swe_infinite_sft_dataset()
        if dd is None:
            print(
                "[generate_trajectories] No data for env 'swe_infinite', skipping it: "
                "SWE-ZERO dataset unavailable (training will continue with other envs). "
                "To include it, ensure MINER_DATASETS contains "
                "gradients-io-tournaments--SWE-ZERO-12M-trajectories-filtered "
                "and MINER_DATASETS_DIR is set.",
                flush=True,
            )
            return False
        dd.save_to_disk(output_path)
        _log(f"SWE-Infinite dataset saved → {output_path} ({len(dd['train'])} trajectories)")
        return True
    else:
        raise ValueError(f"Unknown offline env: {env_name!r}")


# Per-env balance for the merged SFT set (utah balance policy). Generation is time-boxed,
# so a fast self-play env (e.g. clobber) emits several times more windowed examples than a
# slow one — left unbalanced, one env floods the mix, the model over-fits its game moves
# and FORGETS how to code.
#
# We measured this in our f0d0e591 run: a big per-env cap let clobber flood the set (~91k
# clobber vs the fixed ~6k intercode ≈ 15:1). Intercode solve-rate collapsed from the base
# model's 89% to 76% and we placed 2nd — yet all that game data barely helped win-rate (the
# 91k clobber still scored only 52%; our leaner clobber trials scored higher on less data).
# So the policy is:
#   - cap each env SMALL so no game/self-play env can flood the mix;
#   - FLOOR each SCORED env by oversampling the under-represented ones (especially the fixed
#     intercode set) up to the floor, so coding stays well-represented and strong.
# A small, balanced total also lets training finish multiple epochs with a full LR decay
# (vs the ~0.1-0.6 epoch we got on the old ~105k game-flooded set).
# Per-env row cap in the merged set. Env-tunable (BALANCE_SCORED_CAP) so we can run the
# DATA-VOLUME experiment: the 2026-08-02 gold-standard PvP diagnostic proved our clobber/othello
# TEACHERS beat the R1 winner's MODEL (clobber teacher 93%, othello 50%) but our MODELS lost
# (33% / 3%) — a DISTILLATION gap from too little data (this 7k cap → ~14k-row R1 set vs the
# winner's ~260k). Raise this (e.g. 50_000) + generate more/faster to close it. Default unchanged.
BALANCE_SCORED_CAP_PER_ENV = int(os.getenv("BALANCE_SCORED_CAP", "7000"))
BALANCE_COVERAGE_CAP_PER_ENV = 1_500  # thin replay slice (was 6_000): coverage envs are NOT scored, so a
                                      # big cap just dilutes the scored envs and bloats the set. In a smoke
                                      # (clobber+leduc+intercode) coverage at 6k was 39% of the merged data;
                                      # for a 2-scored task (clobber+intercode) it was ~65%. Keep it thin.
BALANCE_FLOOR_FRAC = float(os.getenv("BALANCE_FLOOR_FRAC", "0.5"))  # floor = this × the largest scored env (RELATIVE):
                                     # every scored env is oversampled up to at least this fraction of the biggest one,
                                     # so none is starved AND the floor auto-scales with the round's data volume.
                                     # ENV-TUNABLE (default 0.5 UNCHANGED). Lever for multi-env rounds: at 0.5 a fast
                                     # WEAK env (clobber ~46% vs the field) can still hold ~2x the rows of a STRONG one
                                     # (othello ~87% vs peers) — backwards, since it over-trains the env we lose. Raise
                                     # toward 1.0 to give every scored env EQUAL footing (spend the mix on what we win).

# ---------------------------------------------------------------------------------------------
# WEAPON-FOCUS per-env weighting (idea ported from the R2 winner's adaptive baseline-headroom
# weighting — our own code, not theirs). R1 (2026-08-17) autopsy: our balance SPREAD the pool
# ~evenly across many tasks (a fast COMMODITY env like intercode + a thin 5-env coverage layer),
# so the ENV WE WIN (goofspiel 62% / othello 87% vs peers) was starved of model capacity while a
# rival who FOCUSED his weapon ranked 1st (goofspiel 93%). The env score is a RELATIVE per-env
# rank: rows spent on a COMMODITY env (base model already strong, or the game draws to ~Nash
# regardless of training) buy ~no rank; rows spent on a WEAPON (base weak, high headroom, we
# out-play peers) buy rank. So we WEIGHT the merged pool — down-weight commodity, up-weight our
# weapons. The env CAP and the FLOOR both scale by this weight so a down-weighted env can't be
# floored back up to parity.
#
# Weights are RELATIVE (only ratios matter). Env-tunable via ENV_WEIGHTS (JSON, e.g.
# '{"goofspiel":1.5}') so we retune per the round's MEASURED baselines; ENV_WEAPON_FOCUS=0
# flattens to the legacy uniform balance (all 1.0) for an A/B. Default 1.0 for any env not listed.
_DEFAULT_ENV_WEIGHT: dict[str, float] = {
    "intercode":    0.6,   # COMMODITY: base already ~89% solve — distillation barely moves the rank
    "leduc_poker":  0.7,   # COMMODITY: near-Nash, draws to ~parity regardless of training
    "liars_dice":   0.7,   # COMMODITY: near-Nash bluff equilibrium
    "gin_rummy":    1.0,
    "clobber":      1.3,   # WEAPON single-env (greedy 60% vs the champion, 8B+r128; mobility_pure label
                           # DISPROVEN 48%<60% → greedy stays) BUT FRAGILE co-trained. 2026-08-31 multi-env
                           # DILUTION validation [goofspiel clobber]: at weight 1.0 clobber COLLAPSED 60%→35%
                           # (goofspiel wins the shared-LoRA capacity battle; clobber plays legal-but-weak). The
                           # FIX (this line): weight 1.3 gives clobber 56.5% of the pool → clobber RECOVERS to
                           # 50% (parity) while goofspiel HOLDS 62% (was 63%). Net win, VALIDATED — so 1.3 is
                           # committed, not deferred. (Single-env is unaffected; this only helps clobber survive
                           # co-training.)
    "swe_infinite": 1.0,   # boss gate: parity (its own strict-win logic decides it, not row share)
    "othello":      6.0,   # WEAPON + FLOOD TARGET (2026-09-02, decoded from the tourn_31f2..._20260831 CHAMPIONS).
                           # ALL THREE top miners FLOODED othello: the champion (pos-1 repo) 171k windows/6000
                           # games; the runner-up 50k; a third contender 27k→12k. WE starved othello to ~4-7.8k and LOST
                           # EVERY othello match → dead-last 0.25. This raises othello's MERGE cap to 6000×6.0 = 36000.
                           # ⚠️ BUT the cap raise is INERT ON ITS OWN: an adversarial review + this repo's own note
                           # (line ~331, sft_env_config gen-ratio comment) confirm othello is GENERATION-THROUGHPUT-
                           # limited (~4-5k rows/1.5h), NOT cap-limited — the 36k cap only truncates, it can't
                           # conjure rows. The flood MATERIALIZES only with gen-throughput tuning: raise GEN_SECONDS
                           # several-fold AND/OR lower OTHELLO_TEACHER_BUDGET/OTHELLO_TEACHER_ENDGAME (faster games),
                           # AND/OR an OFFLINE PREBAKE (pre-gen othello outside the 1.5h window, mount as a dataset) —
                           # the champion hit 171k with a FASTER teacher, not a bigger cap. Weight 6.0 is HARMLESS-
                           # INERT as a default on the real (gen-limited) validator box (othello still ~4-5k → same
                           # as 843e71f), and only floods once gen is tuned. THE FLOOD RECIPE (VM smoke, opt-in):
                           # SFT_EPOCHS=1 + GEN_SECONDS=<large> + lower OTHELLO_TEACHER_BUDGET — validate othello
                           # merged rows ~36k AND cosine reaches min LR AND student othello vs MCTS-25 improves, THEN
                           # bake epochs=1 as default too. Was 1.3; A/B baseline = ENV_WEIGHTS='{"othello":1.3}'.
    # goofspiel: WEAPON at the STUDENT level. 2026-08-30 student A/B (single-env, r128, --our-model vs a
    # real rival): the DEFAULT config = 61% (36-23), a clear win. The TEACHER --our-expert reads only
    # 43% — the trained 8B model GENERALIZES BEYOND the hand-teacher (a smoother, less-exploitable policy),
    # so the teacher number badly under-represents the deployed strength (same student>>teacher gap clobber
    # showed). A "sharper" teacher (tighter spread 0.18) tested BETTER at teacher-level (60%) but the sharp
    # STUDENT was WORSE (52%) — tighter spread cuts SFT coverage -> worse generalization; REJECTED, default
    # kept. Kept at 1.0 (NOT 1.3): the weapon comes from teacher+base-model quality, not the data weight, and
    # goofspiel is already favored over every commodity (all <1.0). Bumping to 1.3 is a validated-A/B
    # candidate (would re-balance the othello+goofspiel two-weapon draw), not churned in the night before.
    "goofspiel":    1.0,
}


# ROUND-AWARE othello flood (2026-09-04, VM-VALIDATED). The othello merge weight 6.0 (~36k rows) is
# the winning FLOOD when othello is the round's only game weapon (othello+intercode → 72% vs the champion),
# but in a MULTI-WEAPON round it OVER-DOMINATES: it eats the shared 3B/1-epoch capacity + gen-time and
# COLLAPSES the co-trained weapons (6-env at weight 6.0: goofspiel 78%→10%, clobber 42%, avg 37%). The
# fix — MODERATE othello to 2.0 when another game weapon is in the round — recovers them WITHOUT costing
# othello (6-env weight 2.0: othello 58% held, goofspiel 78% recovered, clobber 50%, AVG 62%). othello
# saturates ~58% co-trained regardless of 12k vs 36k rows, so its extra flood data is wasted in a
# multi-weapon round AND harmful. So flood only when alone-with-commodities; moderate when co-trained
# with a win-capable game weapon (goofspiel/clobber/gin_rummy; leduc/liars are near-Nash commodity that
# draw regardless, intercode/swe are offline — none trigger moderation).
_WEAPON_GAME_ENVS = frozenset({"goofspiel", "clobber", "gin_rummy"})
# VM-VALIDATED sweep (2026-09-04, 6-env avg of othello/goofspiel/clobber vs their champs):
# othello weight 6.0(flood)→AVG37% [goofspiel COLLAPSES 10%], 3.0→53%, 2.0→62%, 1.5→65% [BEST].
# 1.5 is the sweet spot: othello HOLDS 58% (safely beats the boss >50%) while goofspiel recovers
# to 85% and clobber to 52%. Lower (1.0) risks othello <50% = losing the boss-gate, so 1.5 is the floor.
_OTHELLO_MULTIWEAPON_WEIGHT = float(os.getenv("OTHELLO_MULTIWEAPON_WEIGHT", "1.5"))
# Intercode is BROKEN by the othello flood: co-trained under a weight-6.0 othello flood at 1 epoch, the 0.6
# intercode weight leaves intercode undertrained → the model emits game_action / degenerate garbage on
# intercode instead of clean execute_bash (VM-verified 2026-09-07: weight 0.6 → game_action/0.000, 1.3 →
# execute_bash but degenerate). So in an OTHELLO+INTERCODE round, boost intercode HARD so it gets enough
# signal to emit clean commands. Round-aware (only when othello co-trained — elsewhere intercode stays its
# 0.6 commodity weight). Env-tunable; default 2.5. SET TO 0.6 (or INTERCODE_OTHELLO_WEIGHT=0.6) to revert.
# DEFAULT 0.6 = NO-OP (reverted 2026-09-07): the 2.5 boost was VM-tested and is a REGRESSION — intercode
# stayed broken (0.000, model emits game tools not execute_bash) AND othello dropped 72%→55% (lose-lose;
# naive weight-shuffle just steals othello capacity without recovering intercode). Kept env-tunable for
# future A/B (set INTERCODE_OTHELLO_WEIGHT=2.5+ to re-enable), but the committed default must stay 0.6 so
# the branch HEAD == e6f8e23 training behavior and can never be deployed as a regression by accident.
_INTERCODE_OTHELLO_WEIGHT = float(os.getenv("INTERCODE_OTHELLO_WEIGHT", "0.6"))


def _env_weights(env_names: "set[str] | tuple[str, ...] | list[str] | None" = None) -> "dict[str, float]":
    """Relative per-env weight map for the merged pool. ENV_WEIGHTS (JSON) overrides entries;
    ENV_WEAPON_FOCUS=0 flattens to the legacy uniform 1.0. When ``env_names`` (the round's envs) is
    given and othello is co-trained with another game weapon, othello is MODERATED to
    OTHELLO_MULTIWEAPON_WEIGHT; and intercode co-trained with othello is BOOSTED to
    INTERCODE_OTHELLO_WEIGHT so the flood stops breaking it (both round-aware). Never raises."""
    if os.getenv("ENV_WEAPON_FOCUS", "1") == "0":
        return {}
    weights = dict(_DEFAULT_ENV_WEIGHT)
    weights = _apply_headroom_weights(weights, env_names)  # rank-13 OPT-IN; NEUTRAL when disabled/absent
    if env_names and "othello" in env_names and (_WEAPON_GAME_ENVS & set(env_names)):
        weights["othello"] = min(weights.get("othello", 1.0), _OTHELLO_MULTIWEAPON_WEIGHT)
    if (env_names and "othello" in env_names and "intercode" in env_names
            and not (_WEAPON_GAME_ENVS & set(env_names))):
        # Only in the othello-FLOOD scenario (othello + intercode, no game weapon → othello stays weight 6.0
        # and DOMINATES, which is what breaks intercode). In a multi-weapon round othello is already moderated
        # above, so intercode is less starved — leave it at its validated 0.6 there (don't dilute the weapons).
        weights["intercode"] = max(weights.get("intercode", 1.0), _INTERCODE_OTHELLO_WEIGHT)
    # FLOOD-MODE datacap (rank-14, OPT-IN, pairs with budget_controller FLOOD_MODE=datacap): when othello
    # is co-trained at 2 epochs instead of the epoch-drop, shrink its row share so the flood fits. Only
    # active under FLOOD_MODE=datacap (default "epoch" -> skipped, othello keeps its validated weight).
    if env_names and "othello" in env_names and os.getenv("FLOOD_MODE", "epoch").strip().lower() == "datacap":
        try:
            weights["othello"] = weights.get("othello", 1.0) * float(os.getenv("OTHELLO_DATACAP_FACTOR", "0.5"))
        except ValueError:
            pass
    raw = os.getenv("ENV_WEIGHTS")  # explicit override wins (for A/B / manual control)
    if raw:
        try:
            weights.update({str(k): float(v) for k, v in json.loads(raw).items()})
        except Exception as exc:
            _log(f"  balance: ignoring bad ENV_WEIGHTS={raw!r}: {exc}")
    return weights


def _env_weight(env_name: str, weights: "dict[str, float] | None" = None) -> float:
    w = (weights if weights is not None else _env_weights()).get(env_name, 1.0)
    # Fall back to 1.0 for any non-finite / non-positive weight (0, negative, NaN, or a JSON
    # Infinity that json.loads accepts) so a bad ENV_WEIGHTS entry can never poison the cap
    # (int(round(cap * inf)) -> OverflowError) — this guard covers every use (cap, floor, gen-time).
    return w if 0.0 < w < float("inf") else 1.0


# Per-env row TARGET table (rank-7, OPT-IN via ENV_TARGET_ROWS=1). Raises the scored envs toward the
# WINNER volume regime (gin/othello ~13.5k at 2-env/1.5h; intercode toward ~27k) WITHOUT just inflating
# gin and starving the other weapons. When the flag is unset the table is IGNORED and the cap stays the
# scored_cap x weapon-weight product (default behavior UNCHANGED). MUST be paired with the rank-6
# data_fit_scale reducer — raising volume on a slow/many-env box WITHOUT the reducer RE-CREATES the
# undertraining (LR frozen near peak). Placeholders: MEASURE on the VM before enabling. The intercode
# entry is a passthrough VOLUME target only (rows still come from the whitelisted intercode_dataset.py
# passthrough — no eval-fs grounding; it cannot raise rows past the ~12k the passthrough supplies).
_ENV_TARGET_ROWS: dict[str, int] = {
    "gin_rummy": 13_500,
    "othello": 13_500,
    "clobber": 10_000,
    "goofspiel": 10_000,
    "intercode": 27_000,
    "leduc_poker": 4_000,
    "liars_dice": 4_000,
}


def _scored_cap_for(env_name: str, scored_cap: "int | None", weight: float) -> "int | None":
    """Per-env row cap for a SCORED env. Default = scored_cap x weapon-weight (UNCHANGED). When
    ENV_TARGET_ROWS=1 and the env has a target, the explicit target OVERRIDES the product (rank-7)."""
    if scored_cap is None:
        return None
    if os.getenv("ENV_TARGET_ROWS", "0") == "1" and env_name in _ENV_TARGET_ROWS:
        return int(_ENV_TARGET_ROWS[env_name])
    cap = int(round(scored_cap * weight))
    # R1 volume recipe (set by sft_env_config only for R1-shaped rounds, see budget_controller
    # .is_r1_volume_round): a per-env FLOOR, so the round's game and intercode each get >= this many
    # rows. A floor never LOWERS a cap, so an othello round keeps its validated 36k flood.
    try:
        floor = int(os.getenv("R1_MIN_ENV_ROWS", "0") or 0)
    except ValueError:
        floor = 0
    return max(cap, floor) if floor > 0 else cap


def _load_base_winrates() -> "dict[str, float]":
    """Per-env base-model win-rate vs MCTS (rank-13). Reads BASE_WINRATES (inline JSON) or
    BASE_WINRATES_PATH (a JSON file). Returns {} on any absence/parse failure -> NEUTRAL fallback."""
    raw = os.getenv("BASE_WINRATES", "").strip()
    path = os.getenv("BASE_WINRATES_PATH", "").strip()
    try:
        if raw:
            data = json.loads(raw)
        elif path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            return {}
        return {str(k): float(v) for k, v in data.items() if 0.0 <= float(v) <= 1.0}
    except Exception:
        return {}


def _apply_headroom_weights(weights: "dict[str, float]", env_names) -> "dict[str, float]":
    """Base-model HEADROOM weighting (rank-13, OPT-IN via ENV_HEADROOM_WEIGHTS=1). headroom = 1 - base
    win-rate; a low-win-rate (high-headroom) env — e.g. gin/othello — is up-weighted, since that is
    where distillation buys the most rank. Normalized so the mean multiplier over the round's stated
    envs is ~1.0, then clipped to a CONSERVATIVE band [ENV_HEADROOM_LO, ENV_HEADROOM_HI] and MULTIPLIED
    onto the static weapon weights. Disabled OR stats-absent -> returns weights UNCHANGED (NEUTRAL)."""
    if os.getenv("ENV_HEADROOM_WEIGHTS", "0") != "1":
        return weights
    stats = _load_base_winrates()
    if not stats:
        return weights  # NEUTRAL fallback
    try:
        lo = float(os.getenv("ENV_HEADROOM_LO", "0.6"))
        hi = float(os.getenv("ENV_HEADROOM_HI", "1.8"))
    except ValueError:
        lo, hi = 0.6, 1.8
    scope = [e for e in (list(env_names) if env_names else list(stats.keys())) if e in stats]
    if not scope:
        return weights
    headrooms = {e: max(0.0, 1.0 - stats[e]) for e in scope}
    mean_h = (sum(headrooms.values()) / len(headrooms)) or 1.0
    out = dict(weights)
    for e, h in headrooms.items():
        factor = max(lo, min(hi, (h / mean_h) if mean_h > 0 else 1.0))
        out[e] = out.get(e, 1.0) * factor
    return out


def _json_str_field(v):
    """Serialize a ``tools`` / tool-call ``arguments`` value to a JSON STRING so the merged
    Arrow schema stays a uniform string column — matching what ``tokenize_and_mask`` expects
    (it ``json.loads`` these back). ``None`` and already-``str`` values pass through unchanged.
    Most generators already store these JSON-encoded; this only rescues a straggler env that
    stored a RAW dict/list, whose heterogeneous nested ints/keys are exactly what made Arrow
    choke with 'Couldn't cast array of type int64 to null' once a plain-text swe_infinite row
    forced the schema-unifying ``Dataset.from_list`` merge path."""
    if v is None or isinstance(v, str):
        return v
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return None


def _canon_tool_calls(tool_calls):
    """Coerce a message's ``tool_calls`` to a FIXED all-string struct
    ``{id, type, function:{name, arguments}}`` so every row unifies to ONE Arrow schema — no
    per-env struct drift (an int ``id`` or a raw-dict ``arguments`` in one env vs a string in
    another is what produced the int64-vs-null cast crash). Returns None for an empty list so a
    non-tool turn stays a plain message (tokenize renders it as ordinary text)."""
    out = []
    for call in (tool_calls or []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        out.append({
            "id": str(call["id"]) if call.get("id") is not None else None,
            "type": call.get("type") or "function",
            "function": {
                "name": fn.get("name"),
                "arguments": _json_str_field(fn.get("arguments")),
            },
        })
    return out or None


def _to_canonical_row(row: dict) -> dict:
    """Coerce one row to the canonical superset shape ``{messages:[{role, content, tool_calls}], tools}``.

    Plain (swe_infinite) rows have no per-message ``tool_calls`` and no ``tools`` column; defaulting
    both to None lets ``Dataset.from_list`` infer ONE schema across mixed sources. ``tool_calls`` and
    ``tools`` are normalized to a fixed ALL-STRING shape (see _canon_tool_calls / _json_str_field) so
    heterogeneous per-env tool schemas can't collide into an 'int64 -> null' Arrow cast at tokenize
    time. tokenize_and_mask renders a tool_calls=None assistant turn as ordinary text (the plain-text
    MiniSWE format swe_infinite is evaluated in) and ``json.loads`` the string tools/arguments back.
    """
    canon_msgs = []
    for m in (row.get("messages") or []):
        if not isinstance(m, dict):
            continue
        canon_msgs.append({
            "role": m.get("role"),
            "content": m.get("content"),
            "tool_calls": _canon_tool_calls(m.get("tool_calls")),
        })
    return {"messages": canon_msgs, "tools": _json_str_field(row.get("tools"))}


# EXPLICIT canonical Arrow schema for the merged set. Passed to Dataset.from_list so the schema is
# FORCED, never inferred — inference is what broke: with swe_infinite (tool_calls/tools all None) it
# guessed a nested field as `null`, then a game row's int/struct there crashed tokenize with
# "Couldn't cast array of type int64 to null". Every value is coerced to this shape by _to_canonical_row
# (tool_calls -> {id,type,function{name,arguments-as-json-str}}, tools -> json str), which is exactly
# what tokenize_and_mask reads back (json.loads the string tools/arguments).
_CANON_FEATURES = Features({
    "messages": [{
        "role": Value("string"),
        "content": Value("string"),
        "tool_calls": [{
            "id": Value("string"),
            "type": Value("string"),
            "function": {"name": Value("string"), "arguments": Value("string")},
        }],
    }],
    "tools": Value("string"),
})


def _concat_aligned(dsets: list) -> Dataset:
    """``concatenate_datasets`` that tolerates sources with divergent Arrow schemas.

    The game/intercode/pvp sources share one schema ({messages:[{role,content,tool_calls}], tools});
    swe_infinite emits plain {messages:[{role,content}]} with no tools. concatenate_datasets requires
    identical features, so when they diverge we normalize every row (_to_canonical_row) and rebuild
    with Dataset.from_list under an EXPLICIT _CANON_FEATURES schema (no inference — inference is what
    produced the int64-vs-null cast crash). When all schemas already match (no swe_infinite in the
    mix) we keep the fast, unchanged concatenate_datasets path.
    """
    dsets = [d for d in dsets if d is not None and len(d) > 0]
    if not dsets:
        return Dataset.from_list([])
    if len(dsets) == 1:
        return dsets[0]
    if all(d.features == dsets[0].features for d in dsets):
        return concatenate_datasets(dsets)
    rows: list[dict] = []
    for d in dsets:
        for row in d:
            rows.append(_to_canonical_row(dict(row)))
    _log(f"  merge: aligned {len(dsets)} datasets w/ divergent schemas → {len(rows)} rows "
         f"(from_list, explicit canonical features)")
    return Dataset.from_list(rows, features=_CANON_FEATURES)


def _merge_share_caps(per_env_paths: list[str], prefix: str, scored_envs, total_rows: "int | None") -> "dict[str, int] | None":
    """Share-table caps for the scored envs that produced data (None outside share rounds). An intercode
    row budget (INTERCODE_ROW_BUDGET, possibly dosed above its share) is kept whole."""
    if not total_rows or not is_share_round(sorted(scored_envs)):
        return None
    available = {}
    for p in per_env_paths:
        env_name = p[len(prefix):] if p.startswith(prefix) else ""
        if env_name in scored_envs:
            train = DatasetDict.load_from_disk(p).get("train")
            available[env_name] = len(train) if train is not None else 0
    caps = share_caps(list(available), total_rows, available)
    if "intercode" in caps:
        caps["intercode"] = max(caps["intercode"], intercode_row_budget())
    return caps


def merge_datasets(
    per_env_paths: list[str],
    output_path: str,
    scored_envs: "set[str] | tuple[str, ...]" = (),
    scored_cap: "int | None" = None,
    coverage_cap: "int | None" = None,
    floor_frac: "float | None" = None,
    total_rows: "int | None" = None,
) -> None:
    """Concatenate per-environment DatasetDicts into one and save to output_path.

    A round with >= 3 scored envs and a ``total_rows`` budget is sized by the share table instead of
    the two passes below: each scored env is capped at its share of the total, computed on the rows it
    actually generated (budget_controller.share_caps), with no floor oversampling.

    Two passes rebalance the train splits so no fast env dominates and no scored env is
    starved:
      1. CAP: each env is shuffled and truncated to its cap so a flooding self-play env can't
         dominate. Scored-env cap = scored_cap × the env's WEAPON-FOCUS weight (weapons bigger,
         commodity smaller); coverage envs use the flat coverage_cap.
      2. FLOOR (scored envs only): after capping, a base floor = floor_frac × the LARGEST scored
         env, then SCALED per env by weight/max_weight so a down-weighted commodity env floors
         lower and stays down-weighted, while no scored env is starved. Any scored env below its
         floor is oversampled (rows repeated) up to it. A RELATIVE floor auto-scales with the
         round's data volume. Coverage (non-scored) envs are never floored — thin replay slice.
    The env name is recovered from the path suffix (paths are f"{output_path}_{env_name}").
    """
    prefix = output_path + "_"
    weights = _env_weights(scored_envs)  # weapon-focus + round-aware othello (empty = legacy uniform)
    shares = _merge_share_caps(per_env_paths, prefix, scored_envs, total_rows)
    if shares is not None:
        floor_frac = None
    # Pass 1 — load + cap; keep each env's capped splits, its weight and whether it is scored. The
    # scored-env cap scales by the env WEIGHT (weapon-focus): a WEAPON gets a bigger cap, a COMMODITY
    # a smaller one, so the model spends capacity where rank is actually won. Coverage envs unweighted.
    entries: list = []  # (env_name, is_scored, weight, {split_name: ds})
    for p in per_env_paths:
        env_name = p[len(prefix):] if p.startswith(prefix) else ""
        is_scored = env_name in scored_envs
        weight = _env_weight(env_name, weights) if is_scored else 1.0
        if is_scored and shares is not None:
            cap = shares.get(env_name)
        elif is_scored:
            cap = _scored_cap_for(env_name, scored_cap, weight)
        else:
            cap = coverage_cap
        env_splits: dict = {}
        for split_name, ds in DatasetDict.load_from_disk(p).items():
            if split_name == "train" and cap and len(ds) > cap:
                _log(f"  balance: {env_name} train {len(ds)} → {cap} (capped, weight={weight:.2f})")
                ds = ds.shuffle(seed=42).select(range(cap))
            env_splits[split_name] = ds
        entries.append((env_name, is_scored, weight, env_splits))
    # Pass 2 — dynamic floor relative to the largest (post-cap) scored env, then SCALED per env by its
    # weight so the floor can't undo the weapon-focus: a down-weighted commodity env floors lower and
    # stays down-weighted, while every scored env still keeps a sane minimum (no starvation).
    scored_entries = [(w, sp) for (_e, sc, w, sp) in entries if sc and "train" in sp]
    max_scored = max((len(sp["train"]) for (_w, sp) in scored_entries), default=0)
    max_weight = max((w for (w, _sp) in scored_entries), default=1.0) or 1.0
    base_floor = int(floor_frac * max_scored) if (floor_frac and max_scored) else 0
    splits: dict[str, list] = {}
    for (env_name, is_scored, weight, env_splits) in entries:
        env_floor = int(base_floor * weight / max_weight) if (is_scored and base_floor) else 0
        for split_name, ds in env_splits.items():
            if split_name == "train" and env_floor and 0 < len(ds) < env_floor:
                reps = -(-env_floor // len(ds))  # ceil(env_floor / len) — repeat rows to reach the floor
                idx = (list(range(len(ds))) * reps)[:env_floor]
                _log(f"  balance: {env_name} train {len(ds)} → {env_floor} (floored, oversampled x{env_floor / len(ds):.2f})")
                ds = ds.select(idx)
            splits.setdefault(split_name, []).append(ds)
    DatasetDict({k: _concat_aligned(v) for k, v in splits.items()}).save_to_disk(output_path)
    _log(f"Merged {len(per_env_paths)} env datasets → {output_path} (weighted floor: base={base_floor})")


def _with_matchup_memory(env_name: str, examples: list[dict], seed: int) -> list[dict]:
    """PvP game rows plus eval matchup memory (pvp_matchup_memory): a share of the rows show LONG_TERM
    notes and reflection rows are appended. Any failure keeps the plain rows — never lose an env's data."""
    try:
        from envs.pvp_matchup_memory import add_matchup_memory

        started = time.monotonic()
        out, stats = add_matchup_memory(env_name, examples, seed=seed)
    except Exception as exc:  # noqa: BLE001
        # Printed directly (not _log, which is silent by default) so a run shows the step was skipped;
        # the line names only the env and the error type.
        print(f"[WARN] gen post-step skipped for {env_name}: {type(exc).__name__}", flush=True)
        _log(f"[WARN] matchup memory skipped for {env_name} ({exc!r}); rows keep the empty memory block")
        return examples
    _log(f"Matchup memory : {stats['memory_rows']}/{len(examples)} game rows show LONG_TERM notes "
         f"({stats['freeform_rows']} free-form), +{stats['reflection_rows']} reflection rows, "
         f"{stats['foreign_rows']} non-standard rows left as-is ({time.monotonic() - started:.1f}s)")
    return out


def generate_for_env(
    env_name: str,
    output_path: str,
    num_games: int,
    max_turn: int,
    window_turns: int = 10,
    window_step: int = 0,
    num_workers: int = 0,
    seed: int = 42,
    wins_only: bool = False,
    sample_by_score: bool = False,
    score_power: float = 1.0,
    mcts_simulations: int = 10,
    time_limit_seconds: float | None = None,
    row_target: int | None = None,
) -> None:
    """Generate and save a trajectory dataset for a single environment.

    row_target (flat per-turn envs only): stop submitting games once this many rows are collected.
    The merge keeps a shuffled cap of each env anyway, so rows beyond a few x the cap only cost time,
    and that time comes straight out of training (the SFT deadline is fixed)."""
    is_flat = env_name in _FLAT_EXAMPLE_ENVS

    if window_step == 0:
        window_step = window_turns // 2 or 1

    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[env_name]
    is_selfplay = env_name in _SELFPLAY_ENVS

    if is_selfplay:
        # Local pyspiel teacher self-play: no env server, no HTTP. Play is
        # CPU-bound in-process, so parallelism is over CPU cores. Default to all
        # available cores (capped) — override with --num_workers / GEN_WORKERS.
        env_pool = None
        if not num_workers:
            try:
                num_workers = max(1, int(os.getenv("GEN_WORKERS", str(os.cpu_count() or 8))))
            except ValueError:
                num_workers = os.cpu_count() or 8
    else:
        reset_payload = {
            "task_id": task_id_min,
            "seed": 42,
            "opponent": "mcts",
            "mcts_max_simulations": mcts_simulations,
            "mcts_num_rollouts": 1,
        }
        _, env_pool, num_servers, _, _ = init_env_pool(reset_payload)
        if not num_workers:
            try:
                mult = max(1, int(os.getenv("GEN_WORKER_MULTIPLIER", "3")))
            except ValueError:
                mult = 3
            num_workers = max(1, num_servers * mult)

    _log(f"Environment  : {env_name}")
    _log(f"Output       : {output_path}")
    _log(f"Num games    : {num_games}")
    _log(f"Window turns : {window_turns}  step {window_step}")
    _log(f"Mode         : {'local self-play' if is_selfplay else 'env-server'}   Workers: {num_workers}")
    if is_selfplay and env_name == "clobber":
        from envs.clobber_hero import describe

        _log(describe())
    elif is_selfplay and env_name == "othello":
        from envs.othello_hero import describe as describe_othello
        from envs.pvp_selfplay import _HERO_SEAT_DEFAULT_ENVS, _othello_teacher_mode

        if env_name in _HERO_SEAT_DEFAULT_ENVS:
            _log(f"{describe_othello()} | label={_othello_teacher_mode()}")

    random.seed(seed)
    gen_id_min, gen_id_max = generation_id_bounds(env_name)
    game_ids = random.sample(range(gen_id_min, gen_id_max), num_games)
    tasks = [
        (gid, None if is_selfplay else env_pool[i % num_servers]["base_url"], max_turn)
        for i, gid in enumerate(game_ids)
    ]

    use_score_filter = wins_only or sample_by_score
    _log(f"Playing {num_games} games..." + (f" (limit {time_limit_seconds:.0f}s)" if time_limit_seconds else ""))
    if use_score_filter:
        _log(f"Score filter: wins_only={wins_only}  sample_by_score={sample_by_score}"
              f"  score_power={score_power}")
    conversations: list[list[dict]] = []
    examples: list[dict] = []  # _FLAT_EXAMPLE_ENVS only — already-flattened per-turn rows
    skipped = 0
    score_filtered = 0
    all_scores: list[float] = []
    deadline = time.monotonic() + time_limit_seconds if time_limit_seconds else None
    _prewarm_env(env_name)

    def _ingest_result(result) -> None:
        nonlocal completed, skipped, score_filtered
        if is_flat:
            if isinstance(result, tuple):
                views = [result]
            elif result and isinstance(result[0], tuple):
                views = result
            else:
                views = [(result, None)]

            for flat_examples, score in views:
                if score is not None:
                    all_scores.append(score)

                if score is not None and use_score_filter:
                    if wins_only and score <= 0:
                        score_filtered += 1
                        flat_examples = []
                    elif sample_by_score:
                        prob = max(0.0, min(1.0, score)) ** score_power
                        if random.random() >= prob:
                            score_filtered += 1
                            flat_examples = []

                for ex in (flat_examples or []):
                    cleaned = _clean_tool_example(ex)
                    if cleaned is None:
                        skipped += 1
                    else:
                        examples.append(cleaned)
            completed += 1
            return

        if isinstance(result, tuple):
            raw_messages, score = result
            all_scores.append(score)
        else:
            raw_messages, score = result, None

        if score is not None and use_score_filter:
            if wins_only and score <= 0:
                score_filtered += 1
                completed += 1
                return
            if sample_by_score:
                prob = max(0.0, min(1.0, score)) ** score_power
                if random.random() >= prob:
                    score_filtered += 1
                    completed += 1
                    return

        cleaned = _clean(raw_messages)
        if cleaned is None:
            skipped += 1
        else:
            conversations.append(cleaned)
        completed += 1

    def _remaining_timeout() -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _deadline_reached() -> bool:
        remaining = _remaining_timeout()
        return remaining is not None and remaining <= 0

    pending: dict = {}
    task_queue = deque(tasks)
    completed = 0
    max_in_flight = max(1, num_workers * 2)
    pool = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(env_name,),
    )
    timed_out = False
    target_reached = False

    def _target_reached() -> bool:
        return bool(row_target) and is_flat and len(examples) >= row_target

    try:
        def _submit_more() -> None:
            while (len(pending) < max_in_flight and task_queue and not _deadline_reached()
                   and not _target_reached()):
                gid, ep, mt = task_queue.popleft()
                pending[pool.submit(_worker_play, gid, ep, mt)] = gid

        _submit_more()
        while pending:
            timeout = _remaining_timeout()
            if timeout is not None and timeout <= 0:
                timed_out = True
                break

            done, _ = wait(
                pending,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                break

            for future in done:
                pending.pop(future)
                _ingest_result(future.result())
                if completed % 100 == 0:
                    _log(f"  {completed}/{num_games} games done", flush=True)

            if _target_reached():
                target_reached = True
                break

            if _deadline_reached():
                timed_out = True
                break

            _submit_more()
    finally:
        # After an early stop, wait for the few games still running (about one game-length) so they do
        # not keep burning CPU inside the NEXT env's fixed time slice; on a timeout, drop them as before.
        pool.shutdown(wait=target_reached, cancel_futures=True)

    if timed_out:
        _log(f"Time limit reached, stopping at {completed}/{num_games} games", flush=True)
    if target_reached:
        left = _remaining_timeout()
        recipe_log(
            f"[generate_trajectories] {env_name}: row target reached ({len(examples)} >= {row_target}) at "
            f"{completed}/{num_games} games"
            + (f", {left:.0f}s of its time slice left for training" if left is not None else ""),
            tag=f"[gen] {env_name}: target reached",
        )

    if is_flat:
        if use_score_filter:
            _log(f"Valid examples : {len(examples)}   Skipped : {skipped}   Score-filtered games : {score_filtered}")
        else:
            _log(f"Valid examples : {len(examples)}   Skipped : {skipped}")
        if not examples:
            raise RuntimeError("No valid examples generated. Check ENVIRONMENT_SERVER_URLS.")
        if env_name in _SELFPLAY_ENVS:
            examples = _with_matchup_memory(env_name, examples, seed)
        dataset = Dataset.from_list(examples)
        dd = DatasetDict({"train": dataset})
        _log(f"Train: {len(dd['train'])}")
        dd.save_to_disk(output_path)
        _log(f"Dataset saved → {output_path}")
        return

    score_summary = ""
    if all_scores:
        wins = sum(1 for s in all_scores if s > 0)
        score_summary = (
            f"   Score stats: min={min(all_scores):.3f}  max={max(all_scores):.3f}"
            f"  wins(>0)={wins}/{len(all_scores)} ({100*wins/len(all_scores):.1f}%)\n"
            f"   Score-filtered: {score_filtered}"
        )
    _log(f"Valid : {len(conversations)}   Skipped : {skipped}{chr(10) + score_summary if score_summary else ''}")

    if not conversations:
        raise RuntimeError("No valid conversations generated. Check ENVIRONMENT_SERVER_URLS.")

    # Raw game length stats — helps diagnose max_turn being hit or unexpectedly long games.
    raw_lengths = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    max_turn_hits = sum(1 for l in raw_lengths if l >= max_turn)
    length_buckets = Counter(l // 10 * 10 for l in raw_lengths)
    _log(f"\n  ── Raw game stats (before windowing) ──────────────────")
    _log(f"  Avg turns/game   : {sum(raw_lengths)/len(raw_lengths):.1f}")
    _log(f"  Min/Max turns    : {min(raw_lengths)} / {max(raw_lengths)}")
    _log(f"  Hit max_turn={max_turn}  : {max_turn_hits} / {len(conversations)} games"
          f"  ({100*max_turn_hits/len(conversations):.1f}%)")
    _log(f"  Length buckets   : " +
          "  ".join(f"{k}-{k+9}:{v}" for k, v in sorted(length_buckets.items())))

    # Apply sliding window — expands long games into overlapping sub-conversations.
    # Short games (< window_turns pairs) are kept whole as a single window.
    windowed: list[list[dict]] = []
    for conv in conversations:
        windows = _sliding_windows(conv, window_turns, window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed
    _log(f"\n  ── After windowing (turns={window_turns} step={window_step}) ──")
    _log(f"  Total examples   : {len(conversations)}")

    for k, v in _stats(conversations).items():
        _log(f"  {k}: {v}")

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    dd = DatasetDict({"train": dataset})
    _log(f"Train: {len(dd['train'])}")

    dd.save_to_disk(output_path)
    _log(f"Dataset saved → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_names", nargs="+", required=True,
                   help="One or more environment names. Single entry uses --num_games / "
                        "--max_turn overrides; multiple entries use built-in per-env defaults.")
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=None,
                   help="Override num_games (single-env only; defaults to per-env built-in).")
    p.add_argument("--max_turn",    type=int, default=None,
                   help="Override max_turn (single-env only; defaults to per-env built-in).")
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = num_servers. "
                        "Each process holds one concurrent env server connection; "
                        "raise to increase throughput, lower to reduce env server load.")
    p.add_argument("--seed", type=int, default=42)
    # Score-based sampling (for generators that return (messages, score) tuples)
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games where score <= 0. Only applies when the "
                        "generator returns a (messages, score) tuple.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Keep each game with probability clamp(score, 0, 1) ** score-power. "
                        "Only applies when the generator returns a (messages, score) tuple.")
    p.add_argument("--score-power", type=float, default=None,
                   help="Exponent for score-based sampling (single-env override; "
                        "defaults to per-env built-in, or 1.0 if not set).")
    p.add_argument("--time_limit_seconds", type=float, default=None,
                   help="Total generation budget in seconds. For multiple envs the budget "
                        "is divided equally. None = unlimited (generate all num_games).")
    p.add_argument("--coverage_envs", nargs="*", default=None,
                   help="Non-scored game envs to add as a thin coverage/replay layer, "
                        "generated AFTER the scored environment_names, each with "
                        "--coverage_games games. Keeps a continuation model from "
                        "forgetting / seeds future rounds. Merged into the same output.")
    p.add_argument("--coverage_games", type=int, default=4000,
                   help="Games per coverage env (small; the scored envs get the bulk).")
    p.add_argument("--scored_cap", type=int, default=BALANCE_SCORED_CAP_PER_ENV,
                   help="Per-env row cap for the SCORED envs in the merged set. Sized upstream "
                        "(sft_env_config) to the round's budget so the merged total trains ~2 "
                        "epochs; defaults to the module constant when run standalone.")
    p.add_argument("--total_rows", type=int, default=None,
                   help="Fixed total of merged rows for a round with >= 3 scored envs, split by the share "
                        "table (budget_controller.share_caps) in place of --scored_cap.")
    args = p.parse_args()

    # Coverage / replay layer: thin slice of non-scored game envs (offline envs and
    # already-scored envs are excluded), generated AFTER the scored envs.
    coverage_envs = [
        e for e in (args.coverage_envs or [])
        if e not in _OFFLINE_ENVS and e not in set(args.environment_names)
    ]

    # Fast path: a single scored env and no coverage layer -> write directly.
    if len(args.environment_names) == 1 and not coverage_envs:
        env_name = args.environment_names[0]
        if env_name in _OFFLINE_ENVS:
            ok = _generate_offline(env_name, args.output_path)
            if not ok:
                _log(f"[generate_trajectories] No data for sole env {env_name!r}; nothing to train on.")
            return
        env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
        # Generate to a per-env path, then run it through merge_datasets so the SCORED_CAP
        # (weapon-focus weighted) caps it HERE too. The old direct-write path silently ignored
        # the cap, so a fast single self-play env (e.g. othello ~170k windows) flooded the trained
        # set and the model under-trained to ~0.05 epoch (LR never annealed) within the budget.
        env_path = f"{args.output_path}_{env_name}"
        generate_for_env(
            env_name, env_path,
            num_games=args.num_games if args.num_games is not None else env_cfg["num_games"],
            max_turn=args.max_turn if args.max_turn is not None else env_cfg["max_turn"],
            window_turns=args.window_turns,
            window_step=args.window_step,
            num_workers=args.num_workers,
            seed=args.seed,
            wins_only=args.wins_only or env_cfg.get("wins_only", False),
            sample_by_score=args.sample_by_score or env_cfg.get("sample_by_score", False),
            score_power=args.score_power if args.score_power is not None else env_cfg.get("score_power", 1.0),
            mcts_simulations=env_cfg["mcts_simulations"],
            time_limit_seconds=args.time_limit_seconds,
        )
        merge_datasets(
            [env_path], args.output_path,
            scored_envs={env_name},
            scored_cap=args.scored_cap,
            coverage_cap=BALANCE_COVERAGE_CAP_PER_ENV,
            floor_frac=BALANCE_FLOOR_FRAC,
        )
        return

    # Staged path: scored envs first (priority), then the coverage layer, then merge.
    # Time budget split ~85% across scored game envs / ~15% across coverage envs, so
    # the scored (this-round-evaluated) envs always get the bulk and total gen time
    # stays within --time_limit_seconds.
    game_scored = [e for e in args.environment_names if e not in _OFFLINE_ENVS]
    n_s, n_c = len(game_scored), len(coverage_envs)

    # Cold-start fix: build the expensive per-env teacher blueprints ONCE up
    # front, BEFORE the budget is split. A cold MCCFR/CFR solve (liars_dice ~90s,
    # leduc ~20s) would otherwise run inside that env's small weighted time slice
    # and consume it entirely, starving the env to zero examples. Building here in
    # the parent makes every env's slice pure play time, and the per-env
    # ProcessPoolExecutor (fork) inherits the in-memory blueprint for free. The
    # build cost is then charged against the WHOLE budget (subtracted below), not
    # one env's slice, so a cold/no-cache run still produces all envs uniformly.
    prewarm_t0 = time.monotonic()
    for e in game_scored + coverage_envs:
        if e in _SELFPLAY_ENVS:
            _prewarm_env(e)
    prewarm_elapsed = time.monotonic() - prewarm_t0
    effective_budget = args.time_limit_seconds
    if effective_budget is not None:
        effective_budget = max(60.0, effective_budget - prewarm_elapsed)
        if prewarm_elapsed > 1.0:
            _log(f"[budget] blueprint pre-warm took {prewarm_elapsed:.0f}s up front; "
                 f"{effective_budget:.0f}s left for play", flush=True)

    # Scored envs split ~85% of the (post-prewarm) budget WEIGHTED by
    # _GEN_TIME_WEIGHT (slow envs get more) rather than equally; coverage envs
    # split the remaining ~15%.
    scored_budget = (effective_budget * (0.85 if n_c > 0 else 1.0)) if effective_budget else None
    # Time weight = throughput-comp (_GEN_TIME_WEIGHT: slow envs need more wall-clock per row) ×
    # weapon-focus target weight (_env_weight: weapons want MORE rows). The product gives each env
    # the time it needs to actually FILL its (weighted) cap instead of a fast weapon capping out
    # early while a slow one starves.
    _tw = _env_weights(game_scored)  # round-aware: moderate othello's gen-time share in multi-weapon rounds too
    # Share rounds: the target rows of each env are its share cap, so those caps weight the time split.
    _share = (share_caps(args.environment_names, args.total_rows)
              if args.total_rows and is_share_round(args.environment_names) else None)
    if _share is not None:
        _tw = {e: float(_share.get(e, 0)) for e in game_scored}
    scored_w_sum = sum(_gtw(e) * _env_weight(e, _tw) for e in game_scored) or 1.0
    # Games generate one after another; one that reaches its row target early leaves the rest of its slice
    # to the games after it (by the same weights), so the round's generation budget does not sit idle.
    unused_s = 0.0

    def _scored_time_for(env_name: str):
        nonlocal unused_s
        if scored_budget is None:
            return None
        w = _gtw(env_name) * _env_weight(env_name, _tw)
        if env_name not in game_scored:
            return scored_budget * w / scored_w_sum
        later_w = sum(_gtw(e) * _env_weight(e, _tw) for e in game_scored[game_scored.index(env_name):]) or 1.0
        bonus = unused_s * w / later_w
        unused_s -= bonus
        return scored_budget * w / scored_w_sum + bonus

    if effective_budget and n_c > 0:
        coverage_time = effective_budget * 0.15 / n_c
    else:
        coverage_time = None

    per_env_paths: list[str] = []
    # Early stop for scored envs (GEN_ROW_TARGET_MULT, default 4, 0 = off): stop generating once an
    # env holds this multiple of the rows the merge will keep for it (same cap the merge computes).
    # Measured 2026-09-25 on the tournament command: othello filled its whole slice (169k rows for a
    # 36k cap; on a <4B draw that slice is 0.40 x 1.5h = 36 min), the rest of the rows being discarded.
    _merge_weights = _env_weights(set(args.environment_names))
    try:
        _row_mult = float(os.getenv("GEN_ROW_TARGET_MULT", "4") or 0)
    except ValueError:
        _row_mult = 4.0
    # inf/nan would raise inside _gen's try and DROP the env; a value below 1 would stop short of the cap.
    _row_mult = max(1.0, _row_mult) if (math.isfinite(_row_mult) and _row_mult > 0) else 0.0

    def _row_target_for(env_name: str) -> "int | None":
        if _row_mult <= 0 or env_name not in args.environment_names:
            return None
        if _share is not None:
            return share_row_target(env_name, _share.get(env_name, 0), _row_mult) or None
        cap = _scored_cap_for(env_name, args.scored_cap, _env_weight(env_name, _merge_weights))
        return int(cap * _row_mult) if cap else None

    def _gen(env_name, num_games, time_limit):
        env_path = f"{args.output_path}_{env_name}"
        if env_name in _OFFLINE_ENVS:
            if _generate_offline(env_name, env_path):
                per_env_paths.append(env_path)
            return
        env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
        try:
            generate_for_env(
                env_name, env_path,
                num_games=num_games if num_games is not None else env_cfg["num_games"],
                max_turn=env_cfg["max_turn"],
                window_turns=args.window_turns,
                window_step=args.window_step,
                num_workers=args.num_workers,
                seed=args.seed,
                wins_only=env_cfg.get("wins_only", False),
                sample_by_score=env_cfg.get("sample_by_score", False),
                score_power=env_cfg.get("score_power", 1.0),
                mcts_simulations=env_cfg["mcts_simulations"],
                time_limit_seconds=time_limit,
                row_target=_row_target_for(env_name),
            )
            per_env_paths.append(env_path)
        except Exception as exc:
            print(f"[generate_trajectories] No data for env {env_name!r}, skipping it: {exc}", flush=True)

    for env_name in args.environment_names:        # scored first (priority)
        limit, t0 = _scored_time_for(env_name), time.monotonic()
        _gen(env_name, None, limit)
        if limit is not None and env_name in game_scored:
            unused_s += max(0.0, limit - (time.monotonic() - t0))
    for env_name in coverage_envs:                 # thin coverage/replay layer
        _gen(env_name, args.coverage_games, coverage_time)

    if not per_env_paths:
        raise RuntimeError(
            "No valid data generated for any environment "
            f"({', '.join(args.environment_names + coverage_envs)}). Check "
            "ENVIRONMENT_SERVER_URLS and/or MINER_DATASETS."
        )
    merge_datasets(
        per_env_paths, args.output_path,
        scored_envs=set(args.environment_names),
        scored_cap=args.scored_cap,
        coverage_cap=BALANCE_COVERAGE_CAP_PER_ENV,
        floor_frac=BALANCE_FLOOR_FRAC,
        total_rows=args.total_rows,
    )


if __name__ == "__main__":
    main()
