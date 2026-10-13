"""Measure whether the liars_dice teacher is actually strong AT THE DEPLOYED CONFIG.

WHY THIS EXISTS: liars_dice is the one game teacher that never appeared in a live
tournament (R1 = gin_rummy+leduc, R2 = othello+clobber+goofspiel+intercode), so its
real strength is unverified — exactly the blind spot that let othello ship weak and
collapse 0-25. The teacher's design IS strong (external-sampling MCCFR blueprint), BUT
the deployed game is 5 dice/player (total_dice=10) and the COLD blueprint build is capped
at LIARS_DICE_MCCFR_SECONDS (default 90s ≈ ~1000 of the 60k target iterations), with NO
pre-baked cache. So the teacher the tournament actually gets is a possibly under-trained
blueprint — which could be WEAKER than the simple exact-EV fallback it's meant to beat.

This does NOT guess. It builds the blueprint under the real gen budget and plays it
head-to-head vs the EV fallback over many total_dice=10 Dudo games (colors alternated),
and also prints the exploitability of the small (total_dice=2) game where an exact
best-response is tractable, as a sanity check that the MCCFR machinery converges at all.

Pure standard-library (NO pyspiel / GPU) — liars_dice_expert is self-contained. Fast.
Run from scripts/:
    python envs/verify_liars_dice_teacher.py --games 400
    python envs/verify_liars_dice_teacher.py --games 400 --build-seconds 90   # gen default
    python envs/verify_liars_dice_teacher.py --games 400 --build-seconds 600  # does more help?

READ THE RESULT:
  * blueprint WINS clearly  -> the 90s cold build already beats EV; liars_dice is fine.
  * blueprint ~= or < EV     -> the shipped teacher is NOT better than the trivial fallback
                                -> force the EV fallback for this config (safe, 1-line) OR
                                   raise the build budget until blueprint clearly wins.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types

# Make envs.* importable when run as a file (python envs/verify_X.py puts scripts/envs on
# sys.path, not scripts/). Shim a lightweight `envs` pkg so the real envs/__init__.py is
# skipped — the same bootstrap the offline unit tests use. Must precede any `import envs`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.liar_dice_expert as ld

_DEPLOY_TOTAL = 2 * ld.DEPLOY_DICE_PER_PLAYER  # 10 for standard 5-dice Dudo


def _bp_policy(bp: dict):
    """Blueprint sampler: sigma(hand, current_bid, legal, rng) -> action id.
    Falls back to a uniform-legal pick when the info set has no usable mass (exactly
    what choose_action does)."""
    def pol(hand, current_bid, legal, rng):
        entry = bp.get((tuple(sorted(hand)), current_bid)) if bp else None
        if entry:
            kept = {a: p for a, p in entry.items() if a in legal and p > 0.0}
            tot = sum(kept.values())
            if tot > 0.0:
                ids = list(kept)
                return rng.choices(ids, weights=[kept[a] / tot for a in ids], k=1)[0]
        return rng.choice(legal)
    return pol


def _ev_policy():
    """Exact wild-aware binomial-EV sampler — the fallback the blueprint must beat."""
    def pol(hand, current_bid, legal, rng):
        dist = ld._heuristic_distribution(list(hand), _DEPLOY_TOTAL, current_bid, legal)
        ids = list(dist)
        return rng.choices(ids, weights=[dist[a] for a in ids], k=1)[0]
    return pol


# Deal table computed once (C(10,5)=252 hands); reused across all games.
_HANDS5 = ld._hands(ld.DEPLOY_DICE_PER_PLAYER)
_HP5 = [ld._hand_probability(h) for h in _HANDS5]
_LIAR10 = ld.liar_id(_DEPLOY_TOTAL)


def _resolve_challenge(hands, bid_id, challenger) -> float:
    """Player-0 utility when `challenger` calls liar on `bid_id`. Mirrors
    DudoMCCFR._challenge_u0 (6s wild unless the bid face is 6): the challenger wins
    iff the bid is false (matching count < quantity)."""
    q, f = ld.decode_bid(bid_id)
    count = ld._matching_count(hands[0], f) + ld._matching_count(hands[1], f)
    winner = challenger if count < q else 1 - challenger
    return 1.0 if winner == 0 else -1.0


def _play_game(pol0, pol1, rng) -> float:
    """One Dudo game at the deploy config. Returns player-0 utility (+1 win / -1 loss)."""
    hands = (rng.choices(_HANDS5, weights=_HP5, k=1)[0],
             rng.choices(_HANDS5, weights=_HP5, k=1)[0])
    current = None
    player = 0
    while True:
        legal = ld.legal_action_ids_for(current, _DEPLOY_TOTAL)
        a = (pol0 if player == 0 else pol1)(hands[player], current, legal, rng)
        if a == _LIAR10:                   # `player` challenges the last bid (`current`)
            return _resolve_challenge(hands, current, player)
        current = a
        player = 1 - player


def _sanity_small_game_exploitability(build_seconds) -> None:
    """total_dice=2 is small enough for an exact full-history best response — print the
    blueprint's exploitability there so we know the MCCFR converges (0 == Nash)."""
    bp, iters = ld.build_mccfr_blueprint(2, max_seconds=build_seconds, seed=1)

    def sigma(hand, current_bid, actions):
        entry = bp.get((tuple(sorted(hand)), current_bid))
        if entry:
            probs = [entry.get(a, 0.0) for a in actions]
            s = sum(probs)
            if s > 0.0:
                return [p / s for p in probs]
        return [1.0 / len(actions)] * len(actions)

    expl = ld.dudo_exploitability(2, sigma)
    print(f"  [sanity] total_dice=2 blueprint ({iters} iters): exploitability={expl:.3f} "
          f"(0=Nash; EV≈0.79, uniform≈1.56)")


def main() -> None:
    p = argparse.ArgumentParser(description="Is the liars_dice teacher strong at the deployed 10-dice config?")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--build-seconds", type=float, default=None,
                   help="cold-build wall-clock budget (default = gen's LIARS_DICE_MCCFR_SECONDS, 90s)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    # None -> use gen's own default budget (LIARS_DICE_MCCFR_SECONDS, 90s), NOT uncapped.
    build_seconds = ld._UNSET if args.build_seconds is None else args.build_seconds
    budget_note = "default 90s" if args.build_seconds is None else f"{args.build_seconds}s"
    print(f"Building the deploy blueprint (total_dice={_DEPLOY_TOTAL}) under the gen budget ({budget_note})...")
    bp, iters = ld.build_mccfr_blueprint(_DEPLOY_TOTAL, max_seconds=build_seconds, seed=args.seed)
    tgt = ld._mccfr_iters()
    print(f"  cold build did {iters}/{tgt} iterations "
          f"({'FULL' if iters >= tgt else 'CAPPED by time — under-trained'}).")

    bp_pol, ev_pol = _bp_policy(bp), _ev_policy()
    bp_wins = ev_wins = draws = 0
    for i in range(args.games):
        rng = random.Random(args.seed + 1 + i)
        if i % 2 == 0:
            u0 = _play_game(bp_pol, ev_pol, rng)         # blueprint = player 0
            bp_wins += u0 > 0; ev_wins += u0 < 0
        else:
            u0 = _play_game(ev_pol, bp_pol, rng)         # EV = player 0
            ev_wins += u0 > 0; bp_wins += u0 < 0
        draws += u0 == 0

    decisive = bp_wins + ev_wins
    wr = 100 * bp_wins / decisive if decisive else 0.0
    print(f"\nLiars_dice teacher head-to-head — {args.games} games (total_dice={_DEPLOY_TOTAL})")
    print(f"  BLUEPRINT (shipped teacher): {bp_wins} wins")
    print(f"  EV fallback (baseline):      {ev_wins} wins")
    print(f"  blueprint win-rate: {wr:.0f}%")
    try:
        _sanity_small_game_exploitability(build_seconds)
    except Exception as e:  # sanity is best-effort; never fail the head-to-head on it
        print(f"  [sanity skipped: {e}]")

    if bp_wins > ev_wins * 1.05:
        print("\n[PASS] The shipped blueprint clearly beats EV — liars_dice teacher is strong. ✅")
    else:
        print("\n[WEAK] The 90s cold blueprint does NOT clearly beat the trivial EV fallback. ⚠️")
        print("       Fix options: force the EV fallback for this config (safe, 1-line), or")
        print("       raise LIARS_DICE_MCCFR_SECONDS until the blueprint wins (costs gen time).")


if __name__ == "__main__":
    main()
