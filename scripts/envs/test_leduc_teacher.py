"""Tests for the Leduc Hold'em MAX_AGGRO deterministic never-fold teacher (LEDUC_TEACHER, default
'aggro') and its exact win-rate ceiling vs a panel of opponent profiles (leduc v2 research: "Leduc v2:
never-fold, raise-whenever-legal deterministic label + hero-seat vs diverse pool").

Covers:
  (A) _aggro_abstract: pure never-fold/raise-whenever-legal unit tests.
  (B) _teacher_mode() resolution (LEDUC_TEACHER default 'aggro', explicit 'cfr'/'br', legacy
      LEDUC_BR back-compat) and choose_action's mode dispatch (parse_game_state monkeypatched with
      a fake GameState so this needs no pyspiel/trl).
  (C) pool_action's chips_cfr opponent folds a genuinely weak hand (J facing a round-1 raise) at a
      non-zero rate -- a direct regression check for the infoset-parsing bug the leduc v2 review
      review found (pool_action was reading round/history through leduc_poker_env.parse_game_state,
      whose regexes never match this module's own eval-format observation text, so every decision
      silently sampled the round-1 opening infoset instead of the actual one).
  (D) An exact (non-sampled) W/L enumerator over the full Leduc tree -- reuses this module's own
      tree-walk constants/showdown logic so it can never drift from the production CFR+ solver --
      that checks aggro >= 0.5 and >= the exact pre-leduc-v2 default label vs a panel of opponent
      profiles, and >= 0.70 vs a chips-Nash folder (validator W/L rule, all 30 deals x both seat
      orders).
  (E) A scoring canary against the validator's own core.pvp.scoring.determine_outcome (GOD_ROOT),
      confirming the sign-of-return semantics MAX_AGGRO relies on. Skips if GOD_ROOT/validator code
      is not importable.
  (F) Hero-seat pyspiel generation (play_leduc_hero_game): hero-only rows, every recorded label
      equal to the exact MAX_AGGRO action for its own legal-action set (never fold), decision
      histories covering the full reachable set per round ('cr' is a P0-only decision point that
      MAX_AGGRO's own never-check rule makes unreachable for the hero seat -- see the test for the
      turn-order argument), and no crash when the optional open_spiel MCTS submodule is
      unavailable. Skips if pyspiel, or leduc_poker_env's own non-trl dependencies, are not
      importable here.

Run from scripts/:  python envs/test_leduc_teacher.py
Needs pyspiel + numpy; `trl` itself is stubbed automatically when absent (see the import block
below) so this runs in a lean CPU venv without pulling in the full training image.
"""
from __future__ import annotations

import os
import random
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

# leduc_poker_env.py pulls in `trl` (heavy ML dep) purely for its env-server rollout helpers;
# leduc_poker_expert only needs its pure-python parse_game_state + _CARD_RANK. Stub ONLY `trl`
# (never leduc_poker_env itself) when it is not already installed, so the REAL module -- with its
# real regexes, not a test double -- imports cleanly in a lean CPU venv that has pyspiel/requests/
# yaml but not trl. Parts E/F need the REAL parser and skip themselves (with a clear message) if
# leduc_poker_env still cannot be imported (e.g. requests/yaml missing) or pyspiel is unavailable.
_REAL_PARSE_GAME_STATE = None
if "envs.leduc_poker_env" not in sys.modules:
    if "trl" not in sys.modules:
        try:
            import trl  # noqa: F401  -- real trl already installed (training/eval image): use as-is
            # trl 0.18 (the training image's pin) has no trl.experimental; production injects it through
            # trl_compat before any env import (generate_trajectories.py), so do the same here -- otherwise
            # leduc_poker_env fails to import and Parts E/F silently SKIP.
            import trl_compat  # noqa: F401
        except Exception:
            _trl_openenv_stub = types.ModuleType("trl.experimental.openenv")
            _trl_openenv_stub.generate_rollout_completions = lambda *a, **k: None
            _trl_experimental_stub = types.ModuleType("trl.experimental")
            _trl_experimental_stub.openenv = _trl_openenv_stub
            _trl_stub = types.ModuleType("trl")
            _trl_stub.experimental = _trl_experimental_stub
            sys.modules["trl"] = _trl_stub
            sys.modules["trl.experimental"] = _trl_experimental_stub
            sys.modules["trl.experimental.openenv"] = _trl_openenv_stub
    try:
        import envs.leduc_poker_env as _lpenv  # noqa: F401
        _REAL_PARSE_GAME_STATE = _lpenv.parse_game_state
    except Exception:
        _stub = types.ModuleType("envs.leduc_poker_env")
        _stub.parse_game_state = lambda obs: None
        _stub._CARD_RANK = {}
        sys.modules["envs.leduc_poker_env"] = _stub
else:
    _REAL_PARSE_GAME_STATE = getattr(sys.modules["envs.leduc_poker_env"], "parse_game_state", None)

import envs.leduc_poker_expert as lpe  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def _fake_gs(private_rank: int, legal: list[tuple[int, str]], round_no: int = 1,
            public_rank: "int | None" = None, r1=(), r2=()):
    """Minimal duck-typed stand-in for leduc_poker_env.GameState -- only the fields choose_action
    / pool_action actually read."""
    return types.SimpleNamespace(
        private_card_rank=private_rank,
        public_card_rank=public_rank,
        round=round_no,
        r1_betting=list(r1),
        r2_betting=list(r2),
        legal_actions=dict(legal),
    )


# ---------------------------------------------------------------------------------------------
# (A) _aggro_abstract -- pure, no observation parsing involved.
# ---------------------------------------------------------------------------------------------

def test_aggro_abstract():
    check("raises when legal (facing a bet)", lpe._aggro_abstract(["f", "c", "r"]) == "r")
    check("raises when legal (free)", lpe._aggro_abstract(["c", "r"]) == "r")
    check("calls when raise illegal, not facing a bet", lpe._aggro_abstract(["c"]) == "c")
    check("calls when raise illegal, facing a bet (never folds)", lpe._aggro_abstract(["f", "c"]) == "c")
    check("falls back to the only legal action", lpe._aggro_abstract(["f"]) == "f")


# ---------------------------------------------------------------------------------------------
# (B) _teacher_mode() resolution + choose_action dispatch (fake parser, no pyspiel/trl needed).
# ---------------------------------------------------------------------------------------------

def test_teacher_mode_resolution():
    for var in ("LEDUC_TEACHER", "LEDUC_BR"):
        os.environ.pop(var, None)
    check("default mode is aggro", lpe._teacher_mode() == "aggro")
    os.environ["LEDUC_TEACHER"] = "cfr"
    check("explicit LEDUC_TEACHER=cfr", lpe._teacher_mode() == "cfr")
    os.environ["LEDUC_TEACHER"] = "BR"
    check("explicit LEDUC_TEACHER=BR is case-insensitive -> br", lpe._teacher_mode() == "br")
    os.environ["LEDUC_TEACHER"] = "bogus"
    check("unknown LEDUC_TEACHER falls back to the default (aggro)", lpe._teacher_mode() == "aggro")
    os.environ.pop("LEDUC_TEACHER", None)
    os.environ["LEDUC_BR"] = "1"
    check("legacy LEDUC_BR=1 with LEDUC_TEACHER unset -> br (back-compat)", lpe._teacher_mode() == "br")
    os.environ.pop("LEDUC_BR", None)
    check("mode resolution restored to default", lpe._teacher_mode() == "aggro")


def test_choose_action_aggro_default_never_folds_deterministic():
    for var in ("LEDUC_TEACHER", "LEDUC_BR"):
        os.environ.pop(var, None)

    lpe.parse_game_state = lambda obs: _fake_gs(3, [(0, "Fold"), (1, "Call"), (2, "Raise")])
    ids = {lpe.choose_action("x", 0, [0, 1, 2], rng=random.Random(seed)) for seed in range(30)}
    check("aggro deterministically raises when legal (30 rng seeds)", ids == {2})

    lpe.parse_game_state = lambda obs: _fake_gs(1, [(0, "Fold"), (1, "Call")])
    ids = {lpe.choose_action("x", 0, [0, 1], rng=random.Random(seed)) for seed in range(30)}
    check("aggro calls -- never folds -- when raise is unavailable (weak hand, facing a bet)", ids == {1})

    lpe.parse_game_state = lambda obs: _fake_gs(2, [(0, "Call"), (1, "Raise")])
    check("aggro raises pre-bet too (free action)", lpe.choose_action("x", 0, [0, 1]) == 1)


def test_choose_action_cfr_and_br_modes_never_fold_under_wl():
    lpe.parse_game_state = lambda obs: _fake_gs(1, [(0, "Fold"), (1, "Call"), (2, "Raise")])
    for mode in ("cfr", "br"):
        os.environ["LEDUC_TEACHER"] = mode
        seen = {lpe.choose_action("x", 0, [0, 1, 2], rng=random.Random(seed)) for seed in range(20)}
        check(f"{mode} mode never folds under WIN/LOSS scoring (weak hand facing a bet)", 0 not in seen)
        check(f"{mode} mode always returns a legal id", seen <= {0, 1, 2})
    os.environ.pop("LEDUC_TEACHER", None)


def test_pool_action_always_legal():
    lpe.parse_game_state = lambda obs: _fake_gs(1, [(0, "Fold"), (1, "Call"), (2, "Raise")])
    for kind in ("wl_cfr", "chips_cfr", "random"):
        ids = {lpe.pool_action(kind, "x", 0, [0, 1, 2], rng=random.Random(seed)) for seed in range(15)}
        check(f"pool_action kind={kind!r} always returns a legal id", ids <= {0, 1, 2} and ids)


# ---------------------------------------------------------------------------------------------
# (C) Regression check for the leduc v2 review's infoset-parsing finding: pool_action used
#     to read round/history via leduc_poker_env.parse_game_state, whose regexes ("Current round:",
#     "Round N betting:") never match this module's own eval-format observation text ("Round:
#     N/2", "Round N actions: ..."), so every pool decision silently sampled the round-1 OPENING
#     infoset regardless of the real game state. Needs the REAL parse_game_state (private
#     card/legal-action-label parsing); skips if unavailable.
# ---------------------------------------------------------------------------------------------

def _leduc_pool_observation(card: str, round_num: int, r1_actions: str, legal: list[tuple[int, str]],
                            player_id: int = 0) -> str:
    """Build an observation byte-identical in SHAPE to what pvp_selfplay._teacher_obs(
    _format_leduc(...), ...) emits, for a hand-picked round-1-facing-a-raise scenario."""
    lines = [f"Your card: {card}", f"Round: {round_num}/2", "Pot: 4 chips"]
    if r1_actions:
        lines.append(f"Round 1 actions: {r1_actions}")
    state_desc = "\n".join(lines)
    action_lines = "\n".join(f"{aid} -> {label}" for aid, label in legal)
    return (
        f"Current State:\n{state_desc}\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{action_lines}\n"
        f"Your choice (ID only):"
    )


def test_pool_action_chips_cfr_folds_weak_hand_facing_raise():
    if _REAL_PARSE_GAME_STATE is None:
        print("SKIP pool_action chips_cfr fold-rate test: real leduc_poker_env.parse_game_state "
              "unavailable here")
        return
    orig_parse = lpe.parse_game_state
    lpe.parse_game_state = _REAL_PARSE_GAME_STATE
    try:
        obs = _leduc_pool_observation(
            "J♠", round_num=1, r1_actions="Raise",
            legal=[(0, "Fold"), (1, "Call"), (2, "Raise")],
        )
        # Sanity: the observation really does describe round 1, J, facing exactly one raise --
        # otherwise this test would not be exercising the scenario it claims to.
        parsed = lpe._parse_pool_round_history(obs)
        check("test fixture parses as round 1, J facing a single raise",
              parsed == (1, 0, "r", ""))

        n_seeds = 400
        fold_count = 0
        seen_ids = set()
        for seed in range(n_seeds):
            action_id = lpe.pool_action("chips_cfr", obs, 0, [0, 1, 2], rng=random.Random(seed))
            seen_ids.add(action_id)
            if action_id == 0:
                fold_count += 1
        check("pool_action chips_cfr always returns a legal id", seen_ids <= {0, 1, 2})
        check(f"pool_action chips_cfr folds a weak J facing a round-1 raise at a non-zero rate "
              f"({fold_count}/{n_seeds})", fold_count > 0)
    finally:
        lpe.parse_game_state = orig_parse


# ---------------------------------------------------------------------------------------------
# (D) Exact W/L enumerator, built directly on this module's own tree constants/showdown logic so
#     results can never silently drift from the production CFR+ solver / choose_action tree.
# ---------------------------------------------------------------------------------------------

def _legal(hist: str) -> list[str]:
    acts = []
    if hist.endswith(lpe._RAISE):
        acts.append(lpe._FOLD)
    acts.append(lpe._CHECK_CALL)
    if hist.count(lpe._RAISE) < lpe._MAX_RAISES:
        acts.append(lpe._RAISE)
    return acts


def _play(pol, cards, board, rnd, r1, r2, pots, prob, acc):
    """pol[player](rank, pub, r1, r2, legal) -> {action: prob}. Recursively distributes ``prob``
    mass into acc[+1/0/-1] (hero==player0's WIN/LOSS outcome), or, at a round-1 close, defers the
    board chance node into acc['_r2']."""
    hist = r1 if rnd == 1 else r2
    player = len(hist) % 2
    acts = _legal(hist)
    dist = pol[player](cards[player], board if rnd == 2 else 0, r1, r2 if rnd == 2 else "", acts)
    for a, p in dist.items():
        if p <= 0.0:
            continue
        q = prob * p
        nr1, nr2 = (r1 + a, r2) if rnd == 1 else (r1, r2 + a)
        if a == lpe._FOLD:
            acc[-1 if player == 0 else 1] += q
            continue
        np_ = list(pots)
        if a == lpe._RAISE:
            other = 1 - player
            np_[player] = np_[other] + lpe._RAISE_SIZE[rnd]
            _play(pol, cards, board, rnd, nr1, nr2, np_, q, acc)
            continue
        facing = hist.endswith(lpe._RAISE)
        if facing:
            np_[player] = np_[1 - player]
            closed = True
        else:
            closed = player == 1
        if not closed:
            _play(pol, cards, board, rnd, nr1, nr2, np_, q, acc)
        elif rnd == 1:
            acc["_r2"].append((nr1, np_, q))
        else:
            acc[lpe._showdown_winner(cards[0], cards[1], board)] += q


def _expected_score(pol_hero, pol_opp) -> float:
    """Exact hero W/L score in [0,1] (win + 0.5*draw), averaged over all 30 ordered physical-card
    deals AND both seat orders (mirrors the validator's seed-swapped pairing, leduc v2)."""
    tot = {1: 0.0, 0: 0.0, -1: 0.0}
    n = 0
    deck = list(lpe._DECK)
    for i in range(len(deck)):
        for j in range(len(deck)):
            if i == j:
                continue
            cards = (deck[i], deck[j])
            remaining = [deck[k] for k in range(len(deck)) if k not in (i, j)]
            for hero_first in (True, False):
                pol = [pol_hero, pol_opp] if hero_first else [pol_opp, pol_hero]
                acc = {1: 0.0, 0: 0.0, -1: 0.0, "_r2": []}
                _play(pol, cards, None, 1, "", "", [lpe._ANTE, lpe._ANTE], 1.0, acc)
                for (nr1, pots, q) in acc["_r2"]:
                    for b in remaining:
                        sub = {1: 0.0, 0: 0.0, -1: 0.0, "_r2": []}
                        _play(pol, cards, b, 2, nr1, "", pots, q / len(remaining), sub)
                        for k in (1, 0, -1):
                            acc[k] += sub[k]
                sign = 1 if hero_first else -1
                tot[1] += acc[sign]
                tot[-1] += acc[-sign]
                tot[0] += acc[0]
                n += 1
    return (tot[1] + 0.5 * tot[0]) / n


def _pol_aggro(rank, pub, r1, r2, acts):
    return {lpe._aggro_abstract(acts): 1.0}


def _pol_passive(rank, pub, r1, r2, acts):
    return {lpe._CHECK_CALL: 1.0} if lpe._CHECK_CALL in acts else {acts[0]: 1.0}


def _pol_rule_llm(rank, pub, r1, r2, acts):
    """Textbook rule-based folder: J folds to a raise; K (or a paired board) raises; a board that
    outranks an unpaired hand folds to a bet; everything else calls/checks."""
    facing = lpe._FOLD in acts
    if pub:
        if rank == pub:
            return {lpe._RAISE: 1.0} if lpe._RAISE in acts else {lpe._CHECK_CALL: 1.0}
        if pub > rank and facing:
            return {lpe._FOLD: 1.0}
        return {lpe._CHECK_CALL: 1.0}
    if rank == 3:
        return {lpe._RAISE: 1.0} if lpe._RAISE in acts else {lpe._CHECK_CALL: 1.0}
    if rank == 1 and facing:
        return {lpe._FOLD: 1.0}
    return {lpe._CHECK_CALL: 1.0}


def _pol_uniform(rank, pub, r1, r2, acts):
    return {a: 1.0 / len(acts) for a in acts}


def _bp_policy(bp: dict, never_fold: bool = False):
    def pol(rank, pub, r1, r2, acts):
        dist = bp.get(lpe._infoset_key(rank, pub, r1, r2))
        weights = {a: (dist.get(a, 0.0) if dist else 1.0 / len(acts)) for a in acts}
        if never_fold and lpe._CHECK_CALL in acts:
            weights[lpe._FOLD] = 0.0
        total = sum(weights.values())
        if total <= 0.0:
            return {a: 1.0 / len(acts) for a in acts}
        return {a: w / total for a, w in weights.items()}
    return pol


def test_exact_enumeration_aggro_ceiling():
    bp_wl = lpe._bp()  # default WIN/LOSS blueprint (module cache; 2000 iters)
    # Fully-converged solve at the module's own production default (2000 iters --
    # get_chips_pool_blueprint()'s default) gives a robust, deterministic margin above the 0.70
    # ceiling instead of a knife's edge (a partially-converged, fewer-iteration solve measured
    # ~69.95%, just short of 70.0 on convergence noise alone).
    bp_chips = lpe.get_chips_pool_blueprint()

    old_default = _bp_policy(bp_wl, never_fold=True)  # exact (non-sampled) pre-leduc-v2 default label
    panel = {
        "passive": _pol_passive,
        "wl_cfr_sampled": _bp_policy(bp_wl, never_fold=False),
        "chips_cfr": _bp_policy(bp_chips, never_fold=False),
        "rule_llm": _pol_rule_llm,
        "uniform_random": _pol_uniform,
    }
    for name, opp in panel.items():
        aggro_score = _expected_score(_pol_aggro, opp)
        old_score = _expected_score(old_default, opp)
        check(f"aggro >= 0.5 vs {name} (aggro={aggro_score:.4f})", aggro_score >= 0.5 - 1e-9)
        check(f"aggro >= old-default-1e-9 vs {name} (aggro={aggro_score:.4f} old={old_score:.4f})",
              aggro_score >= old_score - 1e-9)
    chips_score = _expected_score(_pol_aggro, panel["chips_cfr"])
    check(f"aggro >= 0.70 vs chips-Nash folder (aggro={chips_score:.4f})", chips_score >= 0.70)

    never_folder_score = _expected_score(_pol_aggro, _pol_aggro)
    check(f"aggro vs any never-folder == 0.5 (mirror ties, got {never_folder_score:.4f})",
          abs(never_folder_score - 0.5) < 1e-9)


# ---------------------------------------------------------------------------------------------
# (E) Scoring canary -- validator's own core.pvp.scoring.determine_outcome (GOD_ROOT), confirming
#     the sign-of-return semantics MAX_AGGRO is built on. Skips if not importable.
# ---------------------------------------------------------------------------------------------

def test_scoring_canary():
    god_root = os.getenv("GOD_ROOT", "").strip()
    if not god_root:
        print("SKIP scoring canary: GOD_ROOT not set")
        return
    saved_path = list(sys.path)
    sys.path = [god_root] + [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    try:
        from core.models.pvp_models import GameOutcome, GameScoringContext
        from core.pvp.scoring import determine_outcome
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP scoring canary: validator code not importable from {god_root} ({exc})")
        sys.path = saved_path
        return
    sys.path = saved_path
    small = GameScoringContext(returns=[1.0, -1.0], player_id=0, is_zero_sum=True,
                               min_utility=-13.0, max_utility=13.0)
    big = GameScoringContext(returns=[13.0, -13.0], player_id=0, is_zero_sum=True,
                             min_utility=-13.0, max_utility=13.0)
    check("scoring canary: a 1-chip showdown win and a 13-chip raised-pot win both == WIN "
          "(sign-of-return; leduc v2)", determine_outcome(small) == GameOutcome.WIN == determine_outcome(big))


# ---------------------------------------------------------------------------------------------
# (F) Hero-seat pyspiel generation (play_leduc_hero_game). Needs the REAL parse_game_state and
#     pyspiel; skips (with a clear message) otherwise.
#
# Decision-history coverage note: of the six pre-action histories the round's betting tree can
# reach ('', 'c', 'r', 'cr', 'rr', 'crr'), 'cr' is P0's decision (it follows P0-checks,
# P1-raises), and P0 always acts first in BOTH rounds (see leduc_poker_env.GameState's own
# docstring). Under MAX_AGGRO the hero never checks while a raise is legal, so a hero playing P0
# never produces the leading 'c' that 'cr' requires -- the hero's OWN decision histories are
# exactly {'', 'rr'} when hero_seat==0 and {'c', 'r', 'crr'} when hero_seat==1, a union of five,
# never 'cr'. This is a property of the label, not a generation bug, so 'cr' is deliberately
# excluded from ``required`` below.
# ---------------------------------------------------------------------------------------------

def test_hero_gen_pyspiel():
    if _REAL_PARSE_GAME_STATE is None:
        print("SKIP hero-gen pyspiel test: real leduc_poker_env.parse_game_state unavailable here")
        return
    try:
        import pyspiel  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP hero-gen pyspiel test: pyspiel unavailable ({exc})")
        return

    lpe.parse_game_state = _REAL_PARSE_GAME_STATE
    for var in ("LEDUC_TEACHER", "LEDUC_BR", "LEDUC_HERO_POOL"):
        os.environ.pop(var, None)

    import envs.pvp_selfplay as ps  # noqa: E402  -- needs real pyspiel

    # (round, r1_hist, r2_hist, chosen_abstract, expected_aggro_abstract)
    decisions: list[tuple[int, str, str, "str | None", str]] = []
    orig_choose = lpe.choose_action

    def _tracking_choose(observation, player_id, legal_action_ids, rng=None):
        action_id = orig_choose(observation, player_id, legal_action_ids, rng=rng)
        # round/history: from the eval-format text directly (see leduc_poker_expert
        # ._parse_pool_round_history's docstring for why parse_game_state's own round/history
        # fields cannot be used here). private card / legal-action labels: parse_game_state,
        # which DOES parse this text correctly for those two fields.
        parsed = lpe._parse_pool_round_history(observation)
        gs = _REAL_PARSE_GAME_STATE(observation)
        if parsed is not None and gs is not None:
            round_, _pub, r1h, r2h = parsed
            label = gs.legal_actions.get(action_id, "")
            ab = lpe._label_to_abstract(label)
            legal_abstracts = [lpe._label_to_abstract(lbl) for lbl in gs.legal_actions.values()]
            legal_abstracts = [a for a in legal_abstracts if a is not None]
            expected = lpe._aggro_abstract(legal_abstracts) if legal_abstracts else ""
            decisions.append((round_, r1h, r2h, ab, expected))
        return action_id

    lpe.choose_action = _tracking_choose
    try:
        n_seeds = 400
        total_rows = 0
        multi_view_seeds = 0
        t0 = time.monotonic()
        for seed in range(n_seeds):
            views = ps.play_leduc_hero_game(seed, max_turn=10, rng=random.Random(seed))
            if len(views) > 1:
                multi_view_seeds += 1
                continue
            if views:
                total_rows += len(views[0][0])
        elapsed = time.monotonic() - t0
    finally:
        lpe.choose_action = orig_choose

    check("hero-gen records at most one view per game (hero-only rows)", multi_view_seeds == 0)
    check(f"hero-gen produced rows across {n_seeds} seeds (got {total_rows})", total_rows > 0)
    check("hero never folds (100% labels == aggro's never-fold rule)",
          all(ab != lpe._FOLD for _rnd, _r1, _r2, ab, _exp in decisions))
    check("every hero label == the exact MAX_AGGRO action for its own legal-action set",
          all(ab == exp for _rnd, _r1, _r2, ab, exp in decisions))

    r1_hists = {r1 for rnd, r1, _r2, _ab, _exp in decisions if rnd == 1}
    r2_hists = {r2 for rnd, _r1, r2, _ab, _exp in decisions if rnd == 2}
    required = {"", "c", "r", "rr", "crr"}  # 'cr' excluded -- see the section comment above
    check(f"round-1 decision histories cover {sorted(required)} (saw {sorted(r1_hists)})",
          required <= r1_hists)
    check(f"round-2 decision histories cover {sorted(required)} (saw {sorted(r2_hists)})",
          required <= r2_hists)
    check("'cr' never appears as a hero decision history (P0-only, unreachable under never-check)",
          "cr" not in r1_hists and "cr" not in r2_hists)

    print(f"[perf] leduc hero-seat self-play: {n_seeds} games, {total_rows} hero rows in "
          f"{elapsed:.2f}s -> {n_seeds / elapsed:.1f} games/s, {total_rows / elapsed:.1f} rows/s")


def test_hero_gen_survives_missing_mcts_submodule():
    """Structural-only check (crash-free + rows produced): needs pyspiel but NOT the real
    (trl-backed) parser -- when only the stub parser is available (as in a lean CPU test venv),
    choose_action degrades to uniform-random internally, which is still enough to prove
    play_leduc_hero_game's game loop / pool routing / MCTS-fallback never crashes."""
    try:
        import pyspiel  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP hero-gen-no-mcts test: pyspiel unavailable ({exc})")
        return

    if _REAL_PARSE_GAME_STATE is not None:
        lpe.parse_game_state = _REAL_PARSE_GAME_STATE
    for var in ("LEDUC_TEACHER", "LEDUC_BR", "LEDUC_HERO_POOL"):
        os.environ.pop(var, None)
    import envs.pvp_selfplay as ps  # noqa: E402

    orig_mcts_bot = ps._make_mcts_bot
    ps._make_mcts_bot = lambda *a, **k: None  # simulate: open_spiel.python.algorithms.mcts unavailable
    try:
        crashed = False
        rows_total = 0
        for seed in range(40):
            try:
                views = ps.play_leduc_hero_game(seed, max_turn=10, rng=random.Random(seed))
                rows_total += sum(len(rows) for rows, _score in views)
            except Exception:  # noqa: BLE001
                crashed = True
        check("hero-gen never crashes when the MCTS submodule is unavailable", not crashed)
        check("hero-gen still produces rows when the MCTS submodule is unavailable", rows_total > 0)
    finally:
        ps._make_mcts_bot = orig_mcts_bot


if __name__ == "__main__":
    test_aggro_abstract()
    test_teacher_mode_resolution()
    test_choose_action_aggro_default_never_folds_deterministic()
    test_choose_action_cfr_and_br_modes_never_fold_under_wl()
    test_pool_action_always_legal()
    test_pool_action_chips_cfr_folds_weak_hand_facing_raise()
    test_exact_enumeration_aggro_ceiling()
    test_scoring_canary()
    test_hero_gen_pyspiel()
    test_hero_gen_survives_missing_mcts_submodule()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
