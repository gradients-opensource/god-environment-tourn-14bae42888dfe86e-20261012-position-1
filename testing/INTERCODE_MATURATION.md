# Intercode Maturation — raise it from commodity-starved toward the field leader's level (VM-gated A/B)

**Goal:** stop starving our own intercode. Move it from ~parity (commodity, weight 0.6) toward the
field-leader level (a rival's intercode 77.5% R1 / 83.7% R2) WITHOUT collapsing our othello weapon.

**Status:** the lever is CONFIG (already env-tunable) — NOT a code change. Behavior at the default is
byte-identical, so today's deploy `e6f8e23` is untouched. Everything below is a POST-tournament A/B
(needs the VM). Do not deploy blind.

## What the rival does (decoded from their live Loki training logs, 2026-09-06)
- filter `wins_only`: 12000 → 5993 (~50%, ≈ our bash-only 5996 — negligible difference, NOT the lever)
- intercode weight **0.70–0.78** (weight FLOOR 0.70: `weight floor: 0.66→0.70`)
- **oversamples** the ~5993 unique up to a per-round target: `num_games 25000→17700`, `25000→19460`
- per-round targets (task ids match the user's screenshots):
  - `d0b77b47` (R1, othello+intercode): `othello 50000, intercode 17700` → **ratio ≈ 2.8:1**, intercode 77.5%
  - `bc9d501a` (R2, 4-env): `intercode 19460` → intercode 83.7%

## What WE do now (generate_trajectories.py)
- `_DEFAULT_ENV_WEIGHT["intercode"] = 0.6` (line 361, "COMMODITY")
- cap = `BALANCE_SCORED_CAP (7000) × weight` = **7000×0.6 = 4200**
- our intercode SFT set = **8871 rows** (5996 execute + 738 retry + 2137 submit) → **CAPPED DOWN to 4200 = we DISCARD 4671 (53%) of our own intercode data**
- in the othello flood (othello ~36k target, weight 6.0): ratio **≈ 8.6:1** (vs the rival's 2.8:1)

## The mechanism (already exists — no code change)
- merge caps each scored env to `7000×weight`, then FLOORS (oversamples, repeats rows) up to
  `env_floor = BALANCE_FLOOR_FRAC(0.5) × max_scored × weight/max_weight` (lines 597-611).
- So intercode volume is driven by its **weight** (via ENV_WEIGHTS JSON) + `BALANCE_FLOOR_FRAC` + `BALANCE_SCORED_CAP` — all env vars, no code.

## The A/B sweep (run when the VM is back; othello+intercode round on Llama-3.2-3B)

| Step | Config | intercode rows (expect) | Risk | Confidence |
|---|---|---|---|---|
| Baseline | (default) | 4200 (capped, half discarded) | — | current |
| **1 (safe win)** | `ENV_WEIGHTS='{"intercode":1.3}'` | ~8871 (use ALL unique, cap=9100) | low | HIGH — we're literally throwing this data away today |
| 2 (rival-ish) | `ENV_WEIGHTS='{"intercode":2.5}'` | ~15-17k (oversample) | med — steals othello share | needs A/B |
| 3 (aggressive) | `ENV_WEIGHTS='{"intercode":6.0}'` (= othello) + higher `BALANCE_FLOOR_FRAC` | ~17-19k (rival level) | high — dilutes the flood | needs A/B |

Note: to OVERSAMPLE above 8871 (step 2/3), intercode's weight must approach othello's so the floor
(`weight/max_weight`) pushes it there; raising `BALANCE_FLOOR_FRAC` (0.5→0.7) is the gentler alt lever.

## GATES (each step must pass before the next)
1. **Intercode rows** reach the target (read the merge log `balance: intercode ... → N`).
2. **Othello HOLDS** — othello merge rows ≈ unchanged AND othello vs the champion stays ≥ its flood number
   (72-77%). The weapon must NOT collapse — this is the whole risk. (verify_vs_winner --env othello)
3. **Fits 1.5h** — total merged rows still finish ≥1 epoch with full LR decay (a rival's 67k total may NOT
   fit our throughput; find OUR max intercode that keeps othello full + trains in-window).
4. **Intercode solve-rate improves** — ⚠️ BLOCKED: no in-repo intercode eval harness. Interim proxy =
   row-count + othello-hold; TRUE validation needs the harness (see below). Target: approach the rival's 83%.

## Prerequisite for step 2+ (the real blocker)
Build an **intercode solve-rate eval harness** (mount princeton-nlp/intercode IC-Bash, run the model's
execute→observe→submit loop, score the submitted observation). Without it, oversampling is unmeasurable
and a blind crank risks over-fitting the gold commands / regressing below base 89%. Step 1 (use all 8871)
is safe to ship without the harness (just un-discards our own data); steps 2/3 are harness-gated.

## Honest caveats
- The rival LOST overall (over-engineered GRPO — see reference_diff_report_20260831_final). Copy their
  intercode-volume IDEA, not their whole recipe.
- We're othello gen-throughput-limited (~4-5k/1.5h without prebake); we may not be able to run othello
  50k AND intercode 17.7k. Our optimum is likely a MIDDLE ground (step 1-2), found by A/B, not step 3.
- Intercode is still a 2-env-round HALVED contributor — this is a supporting refinement, not a round-flipper.
  The primary lever remains the second game env (clobber ~60%). Do this AFTER, or alongside, that.
