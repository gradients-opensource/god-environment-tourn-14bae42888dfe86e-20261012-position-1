#!/usr/bin/env bash
# Run every CPU-only teacher-strength check in one shot, BEFORE the (paid) GPU smoke.
# These prove the game teachers actually got stronger — the gate that would have caught
# othello's R2 collapse. No GPU needed: they run pyspiel/pure-Python self-play on CPU,
# so you can run this the moment you can SSH into the VM, even before a GPU is allocated.
#
#   Run from anywhere on the VM (script lives in scripts/envs/):
#       bash scripts/envs/verify_all_teachers.sh
#   Optional: pass a game count, e.g.  bash scripts/envs/verify_all_teachers.sh 60
#
# Exit code 0 iff ALL checks pass. Each check still prints its own [PASS]/[WEAK] line.
set -uo pipefail

GAMES="${1:-40}"
LD_GAMES="${2:-400}"          # liars_dice games are cheap (pure python) -> more of them
cd "$(dirname "$0")/.."       # -> scripts/  (verifies import `envs.*` and run from here)

RUN="uv run --no-project --with open_spiel python"   # open_spiel needed by othello/clobber
fail=0

banner() { echo; echo "════════════════════════════════════════════════════════"; echo " $1"; echo "════════════════════════════════════════════════════════"; }

banner "1/4  Clobber endgame unit test (exact-solver regression guard)"
$RUN envs/test_clobber_endgame.py || { echo "[FAIL] clobber unit test errored"; fail=1; }

banner "2/4  Othello teacher — strengthened vs R2-collapse params"
$RUN envs/verify_othello_teacher.py --games "$GAMES" || { echo "[FAIL] othello verify errored"; fail=1; }

banner "3/4  Clobber teacher — negamax+exact vs old greedy"
$RUN envs/verify_clobber_teacher.py --games "$GAMES" || { echo "[FAIL] clobber verify errored"; fail=1; }

banner "4/4  Liars_dice teacher — cold blueprint vs EV fallback (deploy config)"
$RUN envs/verify_liars_dice_teacher.py --games "$LD_GAMES" || { echo "[FAIL] liars_dice verify errored"; fail=1; }

banner "SUMMARY"
echo "Scroll up and confirm EACH check printed [PASS]:"
echo "  • clobber unit test  -> ALL CLOBBER-ENDGAME TESTS PASSED"
echo "  • othello            -> [PASS] NEW teacher is clearly stronger"
echo "  • clobber            -> [PASS] NEW teacher is clearly stronger"
echo "  • liars_dice         -> [PASS] blueprint clearly beats EV"
echo
if [ "$fail" -eq 0 ]; then
  echo "No check ERRORED. If all four say [PASS], proceed to the GPU smoke:"
  echo "    bash testing/env_full_7env_test.sh"
  echo
  echo "A [WEAK] is NOT an error (exit 0) but MEANS that teacher needs its contingent fix"
  echo "(both are pre-built + env-gated, default OFF — just re-verify with the flag ON):"
  echo
  echo "  clobber [WEAK]  -> test the rollout-to-terminal leaf:"
  echo "      CLOBBER_LEAF_EVAL=rollout $RUN envs/verify_clobber_teacher.py --games $GAMES"
  echo "      if it now [PASS]es, set CLOBBER_LEAF_EVAL=rollout as the code default before deploy."
  echo
  echo "  liars_dice [WEAK] -> the verify ALREADY measured blueprint-vs-EV directly, so a [WEAK]"
  echo "      IS the proof EV is better. Fix = set LIARS_DICE_FORCE_EV=1 as the code default"
  echo "      before deploy (the production teacher then skips the blueprint). No re-verify needed."
else
  echo "Some check ERRORED (crash/import) — fix that before trusting the [PASS]/[WEAK] verdicts."
fi
exit "$fail"
