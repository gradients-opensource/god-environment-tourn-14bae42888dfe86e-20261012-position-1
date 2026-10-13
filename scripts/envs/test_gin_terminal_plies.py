"""Tests for the gin teacher's exact terminal plies (Wall / Knock / Layoff) and the gin row oversampling.

The plies that settle a gin hand depend only on our own hand and the public melds, so the teacher solves
them exactly (gin_rummy_expert._choose_terminal) instead of with the old first-meld / always-lay-off
heuristic. Covers:
  (A) Hand-built observations in the exact OpenSpiel observation_string format (stdlib only):
      Wall knock, layoff that breaks our own run, layoff of a pure deadwood card, a layoff chain whose first
      card is free, 4-run over its 3-card sub-run, set-vs-run conflict, exposure tie-break for the knocker,
      knock-phase discard that keeps gin, the gin path, legality of every return, the backstop constant
      and the opt-in deterministic path.
  (B) The same defects on REAL pyspiel states built by scripting the deal, run end to end through the
      self-play envelope (pvp_selfplay._teacher_obs) and scored by OpenSpiel itself. Skips without pyspiel.
  (C) pvp_selfplay._row_copies: terminal rows with >= 2 legal ids, forced rows, the upcard k and its
      small-base default threaded from sft_env_config (budget_controller.gin_generate_prefix).

Run from scripts/envs:  python test_gin_terminal_plies.py
"""
from __future__ import annotations

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.gin_rummy_expert as G  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# OpenSpiel-format observation builder (layout copied from real observation_string output)
# ---------------------------------------------------------------------------

SUITS = "scdh"
RANKS = "A23456789TJQK"
PASS, KNOCK = 54, 55


def cid(card: str) -> int:
    return SUITS.index(card[1]) * 13 + RANKS.index(card[0])


def meld_aid(cards: list[str]) -> int:
    """OpenSpiel meld action id: sets rank*5 + missing suit (4 = all four), then runs by length, suit, rank."""
    if len({c[0] for c in cards}) == 1:
        rank = RANKS.index(cards[0][0])
        missing = [i for i, s in enumerate(SUITS) if cards[0][0] + s not in cards]
        return 56 + rank * 5 + (missing[0] if missing else 4)
    length, suit = len(cards), SUITS.index(cards[0][1])
    start = min(RANKS.index(c[0]) for c in cards)
    base = 65 + sum(4 * (14 - n) for n in range(3, length))
    return 56 + base + suit * (14 - length) + start


def meld_label(cards: list[str]) -> str:
    if len({c[0] for c in cards}) == 1:
        return "".join(sorted(cards, key=lambda c: SUITS.index(c[1])))
    return "".join(sorted(cards, key=lambda c: RANKS.index(c[0])))


def grid(cards: list[str]) -> str:
    border = "+" + "-" * 26 + "+"
    rows = ["|" + "".join((r + s) if (r + s) in cards else "  " for r in RANKS) + "|" for s in SUITS]
    return "\n".join([border] + rows + [border])


def section(pid: int, hand: list[str], deadwood: "int | None", melds=(), layoffs=()) -> str:
    head = f"Player{pid}:" + (f" Deadwood={deadwood}" if deadwood is not None else "")
    extra = ""
    if melds:
        extra += "\nLayed melds: " + " ".join(meld_label(list(m)) for m in melds)
    if layoffs:
        extra += "\nLayoffs: " + "".join(layoffs)
    return f"{head}{extra}\n{grid(hand)}"


def obs(phase: str, me: int, hand: list[str], legal: list[tuple[int, str]], *, upcard="XX", pile=(),
        knock=10, stock=20, my_melds=(), opp_melds=(), layoffs=(), deadwood=None) -> str:
    """Self-play teacher envelope around an OpenSpiel gin observation_string for player ``me``. The displayed
    deadwood is the card total, as OpenSpiel shows it once a hand is knocked (pass ``deadwood`` otherwise)."""
    dw = sum(G.card_value(c) for c in hand) if deadwood is None else deadwood
    mine = section(me, hand, dw, my_melds, layoffs)  # OpenSpiel prints layoffs in the defender's section
    theirs = section(1 - me, [], None, opp_melds)
    p0, p1 = (mine, theirs) if me == 0 else (theirs, mine)
    state = (f"\nKnock card: {knock}\nPrev upcard: XX\nRepeated move: 0\nCurrent player: {me}\n"
             f"Phase: {phase}\n\n{p0}\n\nStock size: {stock}  Upcard: {upcard}\n"
             f"Discard pile: {''.join(pile)}\n\n{p1}\n")
    lines = "\n".join(f"{a} -> Player: {me} Action: {lbl}" for a, lbl in legal)
    return f"Current State:\n{state}\nYou are Player {me}.\nLegal Actions:\n{lines}\nYour choice (ID only):"


def card_legal(cards: list[str], with_pass: bool) -> list[tuple[int, str]]:
    legal = [(cid(c), c) for c in cards] + ([(PASS, "Pass")] if with_pass else [])
    return sorted(legal)


def meld_legal(melds: list[list[str]], with_pass: bool) -> list[tuple[int, str]]:
    legal = [(meld_aid(m), meld_label(m)) for m in melds] + ([(PASS, "Pass")] if with_pass else [])
    return sorted(legal)


def teach(o: str, me: int, legal: list[tuple[int, str]]) -> int:
    return G.choose_action(o, me, [a for a, _ in legal], rng=1)


def final_deadwood(hand: list[str], declared: list[list[str]]) -> int:
    used = {c for m in declared for c in m}
    return sum(G.card_value(c) for c in hand if c not in used)


# ---------------------------------------------------------------------------
# (A) stdlib scenarios
# ---------------------------------------------------------------------------

def _scenarios() -> dict:
    """name -> (observation, player, legal, expected_ids, old_bad_id). ``old_bad_id`` is the label the
    previous handler produced (probe-confirmed), or None when the scenario is a pure regression check."""
    out = {}
    # Wall: 10-card deadwood 15 > knock 10; the upcard 4h completes 2h3h4h so OpenSpiel offers Knock.
    hand = ["4s", "5s", "6s", "9h", "9d", "9c", "2h", "3h", "Kd"]
    legal = [(PASS, "Pass"), (KNOCK, "Knock")]
    out["wall_knock"] = (obs("Wall", 0, hand, legal, upcard="4h", pile=["7c", "Jd"], stock=2,
                             deadwood=G.compute_optimal_deadwood(hand)), 0, legal,
                         {KNOCK}, PASS)
    legal = [(PASS, "Pass")]
    out["wall_pass_only"] = (obs("Wall", 0, hand, legal, upcard="Qc", pile=["7c"], stock=2,
                                 deadwood=G.compute_optimal_deadwood(hand)), 0, legal,
                             {PASS}, None)
    # Layoff: knocker melded 8h9hTh; our 5h6h7h is a run, 7h is the only legal layoff and would strand 5h6h.
    hand = ["5h", "6h", "7h", "2c", "3d", "9s"]
    legal = card_legal(["7h"], True)
    out["layoff_meld_breaker"] = (obs("Layoff", 1, hand, legal, pile=["Kc"], opp_melds=[["8h", "9h", "Th"]]),
                                  1, legal, {PASS}, cid("7h"))
    # Layoff of a pure deadwood card.
    hand = ["Jh", "2c", "3d", "9s", "Ks", "Kc"]
    legal = card_legal(["Jh"], True)
    out["layoff_deadwood"] = (obs("Layoff", 1, hand, legal, pile=["4c"], opp_melds=[["8h", "9h", "Th"]]),
                              1, legal, {cid("Jh")}, None)
    # Layoff chain: 7h is free to lay off (our 7-set keeps three) and enables 6h onto 7h-Th.
    hand = ["7h", "6h", "7s", "7c", "7d", "Kc", "Ks"]
    legal = card_legal(["7h"], True)
    out["layoff_chain_first"] = (obs("Layoff", 1, hand, legal, pile=["4c"], opp_melds=[["8h", "9h", "Th"]]),
                                 1, legal, {cid("7h")}, None)
    hand = ["6h", "7s", "7c", "7d", "Kc", "Ks"]
    legal = card_legal(["6h"], True)
    out["layoff_chain_second"] = (obs("Layoff", 1, hand, legal, pile=["4c"], opp_melds=[["8h", "9h", "Th"]],
                                      layoffs=["7h"]), 1, legal, {cid("6h")}, None)
    # Same chain with the seats swapped: the knocker's melds are read from Player1's section.
    hand = ["7h", "6h", "7s", "7c", "7d", "Kc", "Ks"]
    legal = card_legal(["7h"], True)
    out["layoff_chain_first_p0"] = (obs("Layoff", 0, hand, legal, pile=["4c"], opp_melds=[["8h", "9h", "Th"]]),
                                    0, legal, {cid("7h")}, None)
    # Knock phase, after the forced discard: the 4-run (deadwood 5) beats its 3-card sub-run (deadwood 9).
    hand = ["4h", "5h", "6h", "7h", "2c", "3d"]
    legal = meld_legal([["5h", "6h", "7h"], ["4h", "5h", "6h", "7h"]], False)
    out["knock_4run_over_subrun"] = (obs("Knock", 0, hand, legal, pile=["Kc", "9h"], my_melds=[["9s", "9c", "9d"]]),
                                     0, legal, {meld_aid(["4h", "5h", "6h", "7h"])}, meld_aid(["5h", "6h", "7h"]))
    # Defender meld stage, set-vs-run conflict: the 7-set (lowest id) leaves 29, the 5h-8h run leaves 24.
    hand = ["7s", "7c", "7h", "5h", "6h", "8h", "Kc"]
    legal = meld_legal([["7s", "7c", "7h"], ["5h", "6h", "7h"], ["6h", "7h", "8h"], ["5h", "6h", "7h", "8h"]], True)
    out["defender_set_vs_run"] = (obs("Layoff", 1, hand, legal, pile=["4c"], opp_melds=[["Ts", "Tc", "Td"]]),
                                  1, legal, {meld_aid(["5h", "6h", "7h", "8h"])}, meld_aid(["7s", "7c", "7h"]))
    # Knocker exposure tie-break: {7s7c7d7h}+{8h9hTh} and {7s7c7d}+{7h8h9hTh} both leave deadwood 2; the first
    # exposes only Jh to layoffs, the second exposes 6h and Jh.
    hand = ["7s", "7c", "7d", "7h", "8h", "9h", "Th", "2c"]
    legal = meld_legal([["7s", "7c", "7d"], ["7s", "7c", "7d", "7h"], ["8h", "9h", "Th"], ["7h", "8h", "9h", "Th"],
                        ["7c", "7d", "7h"], ["7s", "7d", "7h"], ["7s", "7c", "7h"], ["7h", "8h", "9h"]], False)
    out["knocker_exposure"] = (obs("Knock", 0, hand, legal, pile=["Kc"]), 0, legal,
                               {meld_aid(["7s", "7c", "7d", "7h"]), meld_aid(["8h", "9h", "Th"])}, None)
    # Knock-phase forced discard: every card is melded; dropping a 5 leaves 10, dropping Ah or 4h keeps gin.
    hand = ["Ah", "2h", "3h", "4h", "2c", "3c", "4c", "5s", "5d", "5c"]
    legal = card_legal(hand, False)
    out["knock_discard_keeps_gin"] = (obs("Knock", 0, hand, legal, pile=["Kc"], knock=10), 0, legal,
                                      {cid("4h")}, cid("5s"))
    # Gin path: the knocker has melded everything; only Pass remains.
    legal = [(PASS, "Pass")]
    out["gin_pass"] = (obs("Knock", 0, [], legal, pile=["Kc"], my_melds=[["Ah", "2h", "3h"], ["2c", "3c", "4c"]]),
                       0, legal, {PASS}, None)
    # Knocker after its partition is fully declared: Pass, not a leftover meld from another arrangement.
    hand = ["2c", "3d"]
    legal = [(PASS, "Pass")]
    out["knocker_done_pass"] = (obs("Knock", 0, hand, legal, pile=["Kc"], my_melds=[["4h", "5h", "6h", "7h"]]),
                                0, legal, {PASS}, None)
    return out


def test_stdlib_scenarios():
    for name, (o, me, legal, want, _old) in _scenarios().items():
        got = teach(o, me, legal)
        check(f"[A] {name}: teacher -> {got} in {sorted(want)}", got in want)
        check(f"[A] {name}: label is legal", got in {a for a, _ in legal})


def test_backstop_restores_previous_labels():
    """_EXACT_TERMINAL_PLIES = False must reproduce the probe-confirmed old labels (the defects)."""
    saved = G._EXACT_TERMINAL_PLIES
    G._EXACT_TERMINAL_PLIES = False
    try:
        for name, (o, me, legal, _want, old) in _scenarios().items():
            if old is None:
                continue
            got = teach(o, me, legal)
            check(f"[A] backstop {name}: previous handler -> {got} == {old}", got == old)
    finally:
        G._EXACT_TERMINAL_PLIES = saved


def test_deterministic_path_shares_terminal_plies():
    saved = os.environ.get("GR_DETERMINISTIC_TEACHER")
    os.environ["GR_DETERMINISTIC_TEACHER"] = "1"
    try:
        for name, (o, me, legal, want, _old) in _scenarios().items():
            got = teach(o, me, legal)
            check(f"[A] deterministic {name}: -> {got} in {sorted(want)}", got in want)
    finally:
        if saved is None:
            os.environ.pop("GR_DETERMINISTIC_TEACHER", None)
        else:
            os.environ["GR_DETERMINISTIC_TEACHER"] = saved


def test_knock_meld_sequence_reaches_optimal_deadwood():
    """Declare until Pass on the probe hand: the knocker ends at the optimal deadwood 5, never 9."""
    hand = ["4h", "5h", "6h", "7h", "9s", "9c", "9d", "2c", "3d"]
    all_melds = [["9s", "9c", "9d"], ["5h", "6h", "7h"], ["4h", "5h", "6h", "7h"], ["4h", "5h", "6h"]]
    declared: list[list[str]] = []
    for _ in range(6):
        rest = [c for c in hand if c not in {x for m in declared for x in m}]
        dw_rest = sum(G.card_value(c) for c in rest)
        # OpenSpiel LegalMelds(hand, knock): a meld is legal when some group of disjoint melds containing it
        # leaves deadwood <= knock.
        avail = [m for m in all_melds if set(m) <= set(rest)]
        groups = [[avail[i] for i in range(len(avail)) if mask >> i & 1] for mask in range(1, 1 << len(avail))]
        groups = [g for g in groups if sum(len(m) for m in g) == len({c for m in g for c in m})
                  and dw_rest - sum(G.card_value(c) for m in g for c in m) <= 10]
        legal_melds = [m for m in avail if any(m in g for g in groups)]
        legal = meld_legal(legal_melds, dw_rest <= 10)
        got = teach(obs("Knock", 0, rest, legal, my_melds=declared), 0, legal)
        if got == PASS:
            break
        declared.append(next(m for m in legal_melds if meld_aid(m) == got))
    check(f"[A] knock sequence final deadwood {final_deadwood(hand, declared)} == 5",
          final_deadwood(hand, declared) == 5)


def test_optimal_partitions():
    parts = G._optimal_partitions(tuple(sorted(["4h", "5h", "6h", "7h", "2c", "3d"])))
    check("[A] partitions: 4-run is the only optimal partition",
          parts == ((frozenset({"4h", "5h", "6h", "7h"}),),))
    parts = G._optimal_partitions(tuple(sorted(["7s", "7c", "7d", "7h", "8h", "9h", "Th", "2c"])))
    check(f"[A] partitions: two tied partitions found ({len(parts)})", len(parts) == 2)
    parts = G._optimal_partitions(tuple(sorted(["2c", "9d", "Ks"])))
    check("[A] partitions: no meld -> one empty partition", parts == ((),))


def test_layoff_targets_match_openspiel():
    """AllLayoffs: a 3-card set takes its 4th card, a 4-card set nothing, a run its ends; earlier layoffs
    that touch a run's ORIGINAL ends extend it once (so 8-T + 7h offers 6h, but + 7h + 6h never offers 5h)."""
    run = frozenset({"8h", "9h", "Th"})
    check("[A] targets: run ends", G._layoff_targets((run,), ()) == {"7h", "Jh"})
    check("[A] targets: run + 7h", G._layoff_targets((run,), ("7h",)) == {"6h", "Jh"})
    check("[A] targets: run + 7h + 6h stops at 6h", G._layoff_targets((run,), ("7h", "6h")) == {"6h", "Jh"})
    check("[A] targets: 3-set", G._layoff_targets((frozenset({"7s", "7c", "7d"}),), ()) == {"7h"})
    check("[A] targets: 4-set", G._layoff_targets((frozenset({"7s", "7c", "7d", "7h"}),), ()) == set())
    check("[A] targets: run at the ace edge", G._layoff_targets((frozenset({"As", "2s", "3s"}),), ()) == {"4s"})


def test_fast_deadwood_matches_dp():
    """The mask solver behind compute_optimal_deadwood returns the meld-enumeration DP's value on random and
    meld-dense hands; duplicates / unknown cards fall back to the DP; the backstop constant selects the DP."""
    import random
    rng = random.Random(56)
    deck = list(G.FULL_DECK)
    hands = [rng.sample(deck, rng.randint(1, 11)) for _ in range(15000)]
    for _ in range(5000):  # long same-suit stretches and rank clusters, where sets and runs compete
        suit, start = rng.choice("shdc"), rng.randint(0, 6)
        run = [G.RANK_ORDER[r] + suit for r in range(start, start + rng.randint(3, 7))]
        rank = rng.choice(G.RANK_ORDER)
        extra = [rank + s for s in "shdc" if rank + s not in run][:rng.randint(1, 3)]
        hands.append((run + extra + rng.sample(deck, 3))[:11])
    bad = [h for h in hands if G.compute_optimal_deadwood(h) != G._deadwood_dp(tuple(sorted(set(h))))[0]
           and len(set(h)) == len(h)]
    check(f"[A] fast deadwood == DP on {len(hands)} hands ({len(bad)} mismatches)", not bad)
    check("[A] fast deadwood accepts unnormalised cards", G.compute_optimal_deadwood(["4H", "5h", "6h", "kd"]) == 10)
    dup = ["4h", "4h", "5h", "6h"]
    check("[A] duplicate card -> DP fallback", G.compute_optimal_deadwood(dup) == G._deadwood_dp(tuple(sorted(dup)))[0])
    check("[A] unknown card -> left to the DP", G._hand_mask(["XX", "Kd"]) is None)
    saved = G._FAST_DEADWOOD
    G._FAST_DEADWOOD = False
    try:
        check("[A] backstop _FAST_DEADWOOD=False uses the DP", G.compute_optimal_deadwood(hands[0])
              == G._deadwood_dp_cached(tuple(sorted(hands[0])))[0])
    finally:
        G._FAST_DEADWOOD = saved


def test_label_comment_terminal_phases():
    """The opt-in rationale (PVP_LABEL_COT) describes the terminal labels it now accompanies."""
    saved = os.environ.get("PVP_LABEL_COT")
    os.environ["PVP_LABEL_COT"] = "1"
    try:
        sc = _scenarios()
        o, me, _legal, _want, _old = sc["wall_knock"]
        text = G.get_label_comment(o, KNOCK, me)
        check(f"[A] CoT wall knock uses the post-upcard deadwood ({text!r})", text.startswith("Knocking at 0 "))
        o, me, _legal, _want, _old = sc["layoff_meld_breaker"]
        text = G.get_label_comment(o, PASS, me)
        check(f"[A] CoT layoff pass ({text!r})", "upcard" not in text and text.startswith("Passing"))
    finally:
        if saved is None:
            os.environ.pop("PVP_LABEL_COT", None)
        else:
            os.environ["PVP_LABEL_COT"] = saved


def test_never_raises_on_garbage():
    legal = [(PASS, "Pass"), (cid("7h"), "7h")]
    o = "Current State:\nPhase: Layoff\nYou are Player 1.\nLegal Actions:\n54 -> Pass\n" + f"{cid('7h')} -> 7h\n"
    got = teach(o, 1, legal)
    check(f"[A] unparseable hand -> legal id ({got})", got in {a for a, _ in legal})


# ---------------------------------------------------------------------------
# (B) real pyspiel states
# ---------------------------------------------------------------------------

def _pyspiel():
    try:
        import pyspiel  # noqa: F401
        from envs.pvp_selfplay import _teacher_obs  # noqa: F401
        return pyspiel
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP [B] pyspiel scenarios ({exc})")
        return None


class Script:
    """Deal exact hands (OpenSpiel deals P0's hand, then P1's, then the upcard) and drive moves by label."""

    def __init__(self, pyspiel, hand_size: int, knock: int, p0: list[str], p1: list[str], upcard: str):
        self.state = pyspiel.load_game("gin_rummy", {"hand_size": hand_size, "knock_card": knock}).new_initial_state()
        for c in p0 + p1 + [upcard]:
            self.state.apply_action(cid(c))

    def labels(self) -> list[tuple[int, str]]:
        cp = self.state.current_player()
        return [(a, self.state.action_to_string(cp, a)) for a in self.state.legal_actions()]

    def do(self, *steps: str) -> "Script":
        for st in steps:
            if self.state.is_chance_node():
                self.state.apply_action(cid(st))
                continue
            by_label = {lbl.split("Action: ")[-1]: a for a, lbl in self.labels()}
            assert st in by_label, (st, sorted(by_label))
            self.state.apply_action(by_label[st])
        return self

    def teacher(self) -> tuple[int, str]:
        from envs.pvp_selfplay import _teacher_obs
        cp = self.state.current_player()
        labels = self.labels()
        a = G.choose_action(_teacher_obs(self.state.observation_string(cp), cp, labels), cp,
                            [x for x, _ in labels], rng=1)
        return a, dict(labels)[a].split("Action: ")[-1]

    def teacher_until(self, player: int) -> list[str]:
        """Let the teacher play ``player``'s consecutive turns; return the labels it chose."""
        out = []
        while not self.state.is_terminal() and self.state.current_player() == player:
            a, lbl = self.teacher()
            out.append(lbl)
            self.state.apply_action(a)
        return out


def _with_backstop(flag: bool, fn):
    saved = G._EXACT_TERMINAL_PLIES
    G._EXACT_TERMINAL_PLIES = flag
    try:
        return fn()
    finally:
        G._EXACT_TERMINAL_PLIES = saved


def test_pyspiel_wall():
    pyspiel = _pyspiel()
    if pyspiel is None:
        return
    hero = ["4s", "5s", "6s", "9h", "9d", "9c", "2h", "3h", "Kd"]
    opp = ["As", "2c", "5d", "7d", "Jc", "Qc", "Ks", "8d", "Th"]
    burn = [r + s for s in SUITS for r in RANKS if r + s not in set(hero + opp + ["Qd", "4h"])]

    def build():
        sc = Script(pyspiel, 9, 10, opp, hero, "Qd").do("Pass", "Pass")
        # 33 cards in stock: P0 and P1 alternately draw and throw back the drawn card; P0's 31st draw is 4h,
        # leaving 2 in stock, so the hero (P1) faces the Wall with upcard 4h.
        for i in range(31):
            card = burn[i] if i < 30 else "4h"
            sc.do("Draw stock", card, card)
        return sc

    sc = build()
    phase = sc.state.observation_string(1).split("Phase: ")[1].split("\n")[0]
    check(f"[B] wall reached (phase {phase}, legal {[l for _, l in sc.labels()]})",
          phase == "Wall" and any(l.endswith("Knock") for _, l in sc.labels()))
    old = _with_backstop(False, lambda: build().teacher()[1])
    check(f"[B] wall: previous handler passes ({old})", old == "Pass")
    labels = sc.teacher_until(1)
    check(f"[B] wall: teacher knocks then plays out ({labels})", labels[:1] == ["Knock"])
    sc.teacher_until(0)  # defender layoffs/melds by the teacher too
    ret = sc.state.returns() if sc.state.is_terminal() else None
    check(f"[B] wall knock wins the hand (returns {ret})", ret is not None and ret[1] > 0)


def test_pyspiel_layoff_and_melds():
    pyspiel = _pyspiel()
    if pyspiel is None:
        return
    knocker = ["8h", "9h", "Th", "Js", "Jc", "Jd", "2s"]
    open_seq = ("Pass", "Pass", "Draw stock", "Qd", "Knock", "Qd", "8h9hTh", "JsJcJd", "Pass")

    def breaker():
        return Script(pyspiel, 7, 10, knocker, ["5h", "6h", "7h", "2c", "3d", "9s", "Kc"], "Qs").do(*open_seq)

    check(f"[B] layoff meld-breaker: previous handler lays off 7h",
          _with_backstop(False, lambda: breaker().teacher()[1]) == "7h")
    sc = breaker()
    check(f"[B] layoff meld-breaker: teacher passes", sc.teacher()[1] == "Pass")
    sc.teacher_until(1)
    check(f"[B] meld-breaker: defender keeps 5h6h7h, ends at 24 (returns {sc.state.returns()})",
          sc.state.returns() == [22.0, -22.0])

    sc = Script(pyspiel, 7, 10, knocker, ["7h", "6h", "7s", "7c", "7d", "Kc", "Ks"], "Qs").do(*open_seq)
    labels = sc.teacher_until(1)
    check(f"[B] layoff chain: teacher lays 7h then 6h ({labels})", labels[:2] == ["7h", "6h"])
    check(f"[B] layoff chain: defender ends at 20, not 26 (returns {sc.state.returns()})",
          sc.state.returns() == [18.0, -18.0])

    def conflict():
        return Script(pyspiel, 7, 10, ["Ts", "Tc", "Td", "2s", "3d", "As", "Ad"],
                      ["7s", "7c", "7h", "5h", "6h", "8h", "Kc"], "Qs").do(
            "Pass", "Pass", "Draw stock", "Qd", "Knock", "Qd", "TsTcTd", "Pass")

    sc = conflict()
    sc.teacher_until(1)
    check(f"[B] set-vs-run: defender ends at 24 (returns {sc.state.returns()})",
          sc.state.returns() == [17.0, -17.0])
    sc = conflict()
    _with_backstop(False, lambda: sc.teacher_until(1))
    check(f"[B] set-vs-run: previous handler strands the run at 29 (returns {sc.state.returns()})",
          sc.state.returns() == [22.0, -22.0])


def test_pyspiel_knocker():
    pyspiel = _pyspiel()
    if pyspiel is None:
        return
    junk = ["As", "Kc", "Qs", "Jd", "8s", "6c", "Ts", "Th", "Js"]

    def probe_hand():
        return Script(pyspiel, 9, 10, ["4h", "5h", "6h", "7h", "9s", "9c", "9d", "2c", "3d"], junk, "Qd").do(
            "Pass", "Pass", "Draw stock", "Kd", "Knock")

    sc = probe_hand()
    sc.teacher_until(0)
    check("[B] knock melds: knocker passes at deadwood 5, defender to move",
          not sc.state.is_terminal() and "Deadwood=5" in sc.state.observation_string(0))
    sc = probe_hand()
    _with_backstop(False, lambda: sc.teacher_until(0))
    check("[B] knock melds: previous handler strands 4h (deadwood 9)",
          "Deadwood=9" in sc.state.observation_string(0))

    def gin_hand():
        return Script(pyspiel, 9, 10, ["Ah", "2h", "3h", "2c", "3c", "4c", "5s", "5d", "5c"], junk, "Qd").do(
            "Pass", "Pass", "Draw stock", "4h", "Knock")

    sc = gin_hand()
    labels = sc.teacher_until(0)
    check(f"[B] knock discard keeps gin ({labels[:1]}), defender gets no layoffs",
          labels[:1] in (["Ah"], ["4h"]) and "Deadwood=0" in sc.state.observation_string(0))
    sc.teacher_until(1)
    check(f"[B] gin scored with the bonus (returns {sc.state.returns()})",
          sc.state.is_terminal() and sc.state.returns()[0] >= 25.0)
    sc = gin_hand()
    old = _with_backstop(False, lambda: sc.teacher_until(0))
    check(f"[B] knock discard: previous handler throws a 5 ({old[:1]})", old[:1] in (["5s"], ["5c"], ["5d"]))


# ---------------------------------------------------------------------------
# (C) row oversampling
# ---------------------------------------------------------------------------

def test_row_copies():
    try:
        import envs.pvp_selfplay as P
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP [C] row copies ({exc})")
        return
    saved = {k: os.environ.pop(k, None) for k in ("GIN_UPCARD_OVERSAMPLE", "GIN_UPCARD_BASE_K")}
    try:
        two = [(PASS, "Player: 0 Action: Pass"), (KNOCK, "Player: 0 Action: Knock")]
        one = [(PASS, "Player: 0 Action: Pass")]
        layoff = [(cid("7h"), "Player: 1 Action: 7h"), (PASS, "Player: 1 Action: Pass")]
        melds = [(meld_aid(["5h", "6h", "7h"]), "Player: 0 Action: 5h6h7h"),
                 (meld_aid(["4h", "5h", "6h", "7h"]), "Player: 0 Action: 4h5h6h7h")]
        draw = [(52, "Player: 0 Action: Draw upcard"), (53, "Player: 0 Action: Draw stock")]
        discard = [(cid("Kd"), "Player: 0 Action: Kd"), (cid("2c"), "Player: 0 Action: 2c")]
        k = P._GIN_TERMINAL_K
        check(f"[C] terminal k is 2 ({k})", k == 2)
        check("[C] Wall with Knock legal -> k", P._row_copies("gin_rummy", two, KNOCK, "Phase: Wall\n") == k)
        check("[C] Wall Pass-only (forced) -> 1", P._row_copies("gin_rummy", one, PASS, "Phase: Wall\n") == 1)
        check("[C] Layoff choice -> k", P._row_copies("gin_rummy", layoff, PASS, "Phase: Layoff\n") == k)
        check("[C] Knock meld choice -> k", P._row_copies("gin_rummy", melds, melds[1][0], "Phase: Knock\n") == k)
        check("[C] Knock forced Pass -> 1", P._row_copies("gin_rummy", one, PASS, "Phase: Knock\n") == 1)
        check("[C] Discard phase -> 1", P._row_copies("gin_rummy", discard, cid("Kd"), "Phase: Discard\n") == 1)
        check("[C] Draw stock -> 1", P._row_copies("gin_rummy", draw, 53, "Phase: Draw\n") == 1)
        check("[C] Draw upcard -> 4 by default", P._row_copies("gin_rummy", draw, 52, "Phase: Draw\n") == 4)
        check("[C] other env -> 1", P._row_copies("othello", two, KNOCK, "Phase: Wall\n") == 1)
        check("[C] no state text (old callers) -> terminal rule off",
              P._row_copies("gin_rummy", two, KNOCK) == 1)
        os.environ["GIN_UPCARD_BASE_K"] = "2"
        check("[C] small-base default k=2 for upcard", P._row_copies("gin_rummy", draw, 52, "Phase: Draw\n") == 2)
        os.environ["GIN_UPCARD_OVERSAMPLE"] = "4"
        check("[C] explicit GIN_UPCARD_OVERSAMPLE wins over the base default",
              P._row_copies("gin_rummy", draw, 52, "Phase: Draw\n") == 4)
        saved_k = P._GIN_TERMINAL_K
        P._GIN_TERMINAL_K = 1
        check("[C] backstop _GIN_TERMINAL_K=1 -> 1",
              P._row_copies("gin_rummy", layoff, PASS, "Phase: Layoff\n") == 1)
        P._GIN_TERMINAL_K = saved_k
    finally:
        for key, val in saved.items():
            os.environ.pop(key, None)
            if val is not None:
                os.environ[key] = val


def test_small_base_prefix():
    from budget_controller import GIN_SMALL_BASE_UPCARD_K, gin_generate_prefix
    check(f"[C] small-base upcard k is 2 ({GIN_SMALL_BASE_UPCARD_K})", GIN_SMALL_BASE_UPCARD_K == 2)
    check("[C] 3B gin round -> prefix", gin_generate_prefix(["intercode", "gin_rummy"], 3_210_000_000)
          == "GIN_UPCARD_BASE_K=2 ")
    check("[C] 1.5B gin coverage env -> prefix", gin_generate_prefix(["othello"], 1_540_000_000, ["gin_rummy"])
          == "GIN_UPCARD_BASE_K=2 ")
    check("[C] 8B gin round -> no prefix", gin_generate_prefix(["intercode", "gin_rummy"], 8_030_000_000) == "")
    check("[C] 7B gin round -> no prefix", gin_generate_prefix(["gin_rummy"], 7_610_000_000) == "")
    check("[C] 3B round without gin -> no prefix", gin_generate_prefix(["intercode", "othello"], 3_210_000_000) == "")
    check("[C] unknown size -> no prefix", gin_generate_prefix(["gin_rummy"], None) == "")


def test_training_json_threads_small_base_k():
    try:
        import test_epoch_planner as tep
        tep._shim_model_utility()
        import sft_env_config as S
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP [C] get_training_json ({exc})")
        return

    def gen_cmd(envs, params):
        real = S.get_model_num_params
        S.get_model_num_params = lambda p: params
        try:
            return S.get_training_json({
                "task_id": "T1", "model_path": "/nonexistent", "model_name": "m", "hours_to_complete": 1.5,
                "reg_ratio": 1.0, "output_dir": "/o", "request_path": "/r.json",
                "dataset_type": {"environment_names": envs}})["generate_cmd"]
        finally:
            S.get_model_num_params = real

    check("[C] 3B [intercode, gin] generate_cmd carries GIN_UPCARD_BASE_K=2",
          "GIN_UPCARD_BASE_K=2 " in gen_cmd(["intercode", "gin_rummy"], 3_212_749_824))
    check("[C] 8B [intercode, gin] generate_cmd unchanged",
          "GIN_UPCARD_BASE_K" not in gen_cmd(["intercode", "gin_rummy"], 8_030_261_248))


if __name__ == "__main__":
    for fn in (test_fast_deadwood_matches_dp, test_optimal_partitions, test_layoff_targets_match_openspiel,
               test_stdlib_scenarios,
               test_backstop_restores_previous_labels, test_deterministic_path_shares_terminal_plies,
               test_knock_meld_sequence_reaches_optimal_deadwood, test_label_comment_terminal_phases,
               test_never_raises_on_garbage,
               test_pyspiel_wall, test_pyspiel_layoff_and_melds, test_pyspiel_knocker,
               test_row_copies, test_small_base_prefix, test_training_json_threads_small_base_k):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised {type(exc).__name__}: {exc}", False)
    print("RESULT", "PASS" if not FAILS else f"FAIL {len(FAILS)}: {FAILS}")
    sys.exit(1 if FAILS else 0)
