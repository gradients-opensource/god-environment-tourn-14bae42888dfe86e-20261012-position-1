# Othello-flood recipe — MANDATORY VM smoke before deploy

**Why:** we placed LAST (env-score 0.25) in tourn_31f2e0fe36783f71_20260831 (othello+intercode @ Llama-3.2-3B).
Decoded all 3 advancers — they FLOODED othello (the champion 171k windows, the runner-up 50k, a third contender 27k→12k);
we starved it to ~4-7.8k and lost every othello match. This recipe floods othello + pairs it with 1 epoch.

## What changed (branch clobber-mobility-winner-recipe) — safe defaults + an OPT-IN flood
An adversarial review caught that a blind flood default is net-NEGATIVE (cap raise is inert without gen-throughput
tuning; an unconditional epochs 2→1 regresses non-othello rounds). So the committed DEFAULT stays ≈ 843e71f-safe,
and the full flood is an OPT-IN recipe (env vars) you enable during this smoke, then bake once validated.
- `generate_trajectories.py` `_DEFAULT_ENV_WEIGHT['othello']` 1.3 → **6.0** = the TARGET merge cap (6000×6.0=36k).
  **Harmless-INERT by default** — othello is gen-THROUGHPUT-limited (~4-5k rows), not cap-limited, so the cap only
  fills once you raise gen throughput below. Never regresses on its own.
- `sft_env_config.get_generation_time_ratio` 3B 0.25 → **0.30** (mild: +20% othello gen even by default).
- `budget_controller.recommend_epochs` — **UNCHANGED default (2 epochs)**. The champion's big-set+1-epoch pairing is
  OPT-IN via `SFT_EPOCHS=1` (set it ONLY together with the gen-throughput vars below, else 1 epoch just halves training).
- `othello_minimax.py` `SFT_OTHELLO_ANCHOR=1` flag + `_WEIGHTS_ANCHOR` (OFF by default; sharper corner table).
- `intercode_dataset.py` docstring 0.4 → 0.35 (matches code default).

## The flood is INERT without gen-throughput tuning (read this)
Weight 6.0 raises the cap to 36k but othello generation only yields ~4-5k rows in the 1.5h window (the perfect-endgame
negamax teacher is seconds/game). The champion hit 171k with a FASTER (MCTS ~1s/game) teacher → more games, NOT a
bigger cap. So to make the flood REAL you must ALSO do one or more of:
- raise `GEN_SECONDS` several-fold (steals train time — pair with `SFT_EPOCHS=1` so it still fits),
- lower `OTHELLO_TEACHER_BUDGET` (default 0.08s) / `OTHELLO_TEACHER_ENDGAME` (default 16 — the perfect-solve depth is
  the likely throughput killer; try 12) for faster games,
- or an **offline PREBAKE**: pre-generate othello outside the 1.5h window and mount it as a dataset (the only path to
  full ~50-171k champion scale in-budget).

## Measured facts (from env_distill_test.sh — don't re-litigate)
- Our EXPERT othello teacher (negamax + perfect endgame, budget 0.08s) = **46% vs the champ** and is FAST.
  **Keep OTHELLO_TEACHER=off** — MCTS-rollouts=8 starved othello gen to 0%. The fix is DATA VOLUME, not a teacher swap.
- othello gen **throughput** is the binding limit (stays below the cap even at 50k). Full champion scale (≥50k)
  likely needs an **offline prebake** (pre-generate othello outside the 1.5h window, mount as a dataset).
- Leduc already has a champion-level CFR+ solver (`_LeducCFR`, WIN/LOSS-tuned) — do NOT touch it.

## Smoke — Llama-3.2-3B, othello+intercode (the round we lost)
Adapt `testing/env_othello_test.sh` (or `env_distill_test.sh`) with MODEL=unsloth/Llama-3.2-3B-Instruct,
DATASET_TYPE='{"environment_names":["othello","intercode"]}', HOURS_TO_COMPLETE=1.5, on an H100.
Run the FLOOD (defaults) and the OLD recipe as an A/B:

```bash
# A/B baseline (current 843e71f-equivalent, 2 epochs, no flood):
ENV_WEIGHTS='{"othello":1.3}' SFT_EPOCHS=2 bash testing/env_othello_test.sh

# FLOOD recipe (opt-in): weight 6.0 is already the default cap; the flood needs gen-throughput + 1 epoch.
# Start here and TUNE GEN_SECONDS / OTHELLO_TEACHER_BUDGET until othello merged rows approach ~36k:
SFT_EPOCHS=1 GEN_SECONDS=3000 OTHELLO_TEACHER_BUDGET=0.05 bash testing/env_othello_test.sh
#   ...if othello rows still short, lower OTHELLO_TEACHER_ENDGAME (16→12) and/or raise GEN_SECONDS again,
#   or build an offline othello prebake dataset.

# Anchor-teacher A/B (offline, CPU): compare teacher strength BEFORE enabling the flag
SFT_OTHELLO_ANCHOR=1 python scripts/envs/verify_othello_teacher.py   # vs default (unset)
python scripts/envs/verify_teacher_vs_mcts.py --mcts-sims 25         # vs the real eval opponent
```

## PASS/FAIL gates (deploy ONLY if all pass)
1. **othello merged rows** in the gen log ≫ old ~4-7.8k (target ≥20k). If short → raise `GEN_SECONDS`, or prebake.
2. **cosine LR reaches its min** at run end (anneal completed — grep the LR schedule / final logging_steps).
3. **train finishes inside 1.5h** — no mid-epoch deadline truncation (1 epoch makes a stop graceful regardless).
4. **student othello win-rate vs OpenSpiel MCTS-25** (the real eval opponent) IMPROVES vs the 843e71f baseline.
5. Deploy the flood only if (4) holds. Enable `SFT_OTHELLO_ANCHOR` only if its offline A/B wins.

Deploy target after smoke passes: a new commit pushed to **env-destroy** (never origin).

---

# SURPASS the champion (not just match) — in-window only

**Mount is validator-only** (confirmed 2026-09-02): the validator mounts ONLY its own whitelisted
datasets (`core/whitelisted_sft_datasets.json`, upstream); a miner CANNOT mount a self-generated
othello dataset. So the "prebake a huge othello set + mount it" path is CLOSED. Surpass must be IN-WINDOW.

**The structural edge (real, code-confirmed):** our othello teacher (`othello_expert._endgame`) solves
the decisive last ~16 plies EXACTLY (true disc-differential negamax, ×10000). The champion's MCTS-50 and
the eval opponent MCTS-25 play the endgame with RANDOM rollouts. So our teacher plays the winning phase
PERFECTLY while MCTS guesses — a quality edge the champion lacks. Keep `OTHELLO_TEACHER=off` (the expert),
never MCTS.

## Experiment S1 — teacher-speed sweet spot (lift volume, keep the endgame edge)
The perfect-endgame solve is the throughput killer. Lowering `OTHELLO_TEACHER_ENDGAME` shrinks the solve
tree (faster) while staying perfect for the last N plies (still beats MCTS-25's random endgame). The
midgame is SATURATED, so lowering budget/depth loses ~no strength. A/B for max (volume × student-strength):
```bash
# baseline endgame 16:
bash testing/env_othello_test.sh
# faster endgame (more games) — try 14 then 12:
OTHELLO_TEACHER_ENDGAME=12 OTHELLO_TEACHER_BUDGET=0.05 bash testing/env_othello_test.sh
```
Measure: othello rows generated (should RISE as ENDGAME drops) AND student othello win-rate vs MCTS-25.
Pick the ENDGAME that MAXIMISES student win-rate (more data usually wins until the endgame gets too shallow).

## Experiment S2 — train vs the ACTUAL eval opponent (MCTS-25) [NEW capability]
`OTHELLO_OPP_MCTS=1` switches othello gen from self-play to OUR expert vs an OpenSpiel MCTS-25 opponent,
recording only our seat → the student is distilled to BEAT the eval opponent, not to mirror itself.
```bash
OTHELLO_OPP_MCTS=1 OTHELLO_OPP_MCTS_SIMS=25 bash testing/env_othello_test.sh
```
Off by default (self-play, byte-identical). SMOKE-GATE: training vs one opponent can TARGET the metric
(good) or overfit (bad) — deploy only if student othello vs MCTS-25 beats both the 843e71f baseline AND
the S1 self-play best. Note: this records ~half the rows/game (one seat), so pair with S1's higher volume.

## Experiment S3 — othello quality filter (optional)
Othello currently keeps both seats unfiltered. A wins-only filter (keep only our-teacher WINS) sharpens the
policy but cuts volume — A/B only if S1+S2 volume is already high. (Not yet wired; add
`sample_by_score`/`wins_only` to othello in generate_trajectories `_ENV_DEFAULT_ARGS` behind a flag if S1/S2
show volume to spare.)

## SURPASS pass/fail
The goal is not "othello improves" but **student othello win-rate vs MCTS-25 EXCEEDS the champion's level**
(our E=16 expert teacher was ~46% vs the champion; target the STUDENT clears >50% vs MCTS-25 and beats the 843e71f
baseline decisively). Deploy the combination (ENDGAME sweet spot ± OPP_MCTS) that wins that A/B.
