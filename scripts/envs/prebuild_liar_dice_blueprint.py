"""Pre-build (warm) the Liar's Dice MCCFR blueprint into the persistent cache.

The live teacher (`liar_dice_expert._get_blueprint`) solves an external-sampling
MCCFR blueprint for the deploy dice config (total_dice=10). A well-converged
solve is ~tens of thousands of iterations of single-threaded Python (~12 it/s),
which does NOT fit inside a training job's trajectory-generation budget. So a
cold build inside the trainer is time-capped (LIARS_DICE_MCCFR_SECONDS) and only
yields a thin blueprint.

This script builds a STRONG blueprint once, offline, by running many independent
CFR+ instances in parallel (different seeds) and averaging their average
strategies -- summing each info set's `strat_sum` across runs is exactly the
average-strategy combination CFR uses, and the extra seeds also broaden info-set
coverage. The merged blueprint is written (atomically) to the blueprint cache
dir, which the trainer mounts persistently (god/secure_checkpoints/expert_cache
-> /cache/expert_cache via LIARS_DICE_EXPERT_CACHE_DIR), so every subsequent
training run loads a full-quality teacher in milliseconds.

Usage:
    python -m envs.prebuild_liar_dice_blueprint \
        --total-dice 10 --workers 20 --iters-per-worker 6000 \
        --cache-dir /path/to/expert_cache
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Import the sibling module directly (NOT `from envs import ...`): the `envs`
# package __init__ pulls in heavy deps (trl, etc.) that aren't needed -- and may
# not be installed -- for this CPU-only solver. Run as a plain script:
#   python envs/prebuild_liar_dice_blueprint.py ...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import liar_dice_expert as e  # noqa: E402


def _train_worker(args: "tuple[int, int, int]") -> "dict[tuple, list[float]]":
    """Run one CFR+ instance and return {info_set_key: strat_sum}."""
    dice_per_player, iters, seed = args
    solver = e.DudoMCCFR(dice_per_player)
    solver.train(iters, random.Random(seed))
    return {key: list(nd.strat_sum) for key, nd in solver.nodes.items()}


def build_parallel(total_dice: int, workers: int, iters_per_worker: int,
                   base_seed: int = 0) -> "tuple[dict, int]":
    dpp = total_dice // 2
    jobs = [(dpp, iters_per_worker, base_seed + w) for w in range(workers)]
    merged: "dict[tuple, list[float]]" = {}
    t0 = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_train_worker, jobs):
            for key, ss in part.items():
                acc = merged.get(key)
                if acc is None:
                    merged[key] = list(ss)
                else:
                    for i, v in enumerate(ss):
                        acc[i] += v
            done += 1
            print(f"  merged worker {done}/{workers} "
                  f"({len(merged)} info sets, {time.monotonic() - t0:.0f}s)",
                  flush=True)

    # Normalize merged strat_sum into the blueprint format
    # {(hand, current): {action_id: prob}}.
    blueprint: dict = {}
    for (hand, current), ss in merged.items():
        acts = e.legal_action_ids_for(current, total_dice)
        s = sum(ss)
        if s > 0.0:
            blueprint[(hand, current)] = {acts[i]: ss[i] / s for i in range(len(acts))}
        else:
            n = len(acts)
            blueprint[(hand, current)] = {acts[i]: 1.0 / n for i in range(n)}
    return blueprint, workers * iters_per_worker


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-dice", type=int, default=2 * e.DEPLOY_DICE_PER_PLAYER)
    p.add_argument("--workers", type=int, default=min(20, (os.cpu_count() or 4)))
    p.add_argument("--iters-per-worker", type=int, default=6000)
    p.add_argument("--cache-dir", type=str, default=e._CACHE_DIR)
    p.add_argument("--base-seed", type=int, default=0)
    args = p.parse_args()

    total = args.total_dice
    if total < 2 or total % 2 != 0:
        raise SystemExit(f"total-dice must be even and >= 2 (got {total})")

    path = os.path.join(args.cache_dir, f"liar_dice_mccfr_total{total}.pkl")
    print(f"Pre-building MCCFR blueprint: total_dice={total} "
          f"workers={args.workers} iters/worker={args.iters_per_worker} "
          f"(~{args.workers * args.iters_per_worker} effective iters)")
    print(f"Cache target: {path}")
    t0 = time.monotonic()
    blueprint, eff_iters = build_parallel(
        total, args.workers, args.iters_per_worker, args.base_seed
    )
    e._save_blueprint_payload(path, blueprint, eff_iters, total)
    print(f"DONE in {time.monotonic() - t0:.0f}s: {len(blueprint)} info sets, "
          f"{eff_iters} effective iterations -> {path}")


if __name__ == "__main__":
    main()
