"""Tests for the liars_dice visible-bid reader (the default label) and its hero-vs-pool generator.

The eval prompt shows our dice, the latest bid and the seat, so the label must be a deterministic,
always-legal function of exactly that. Sections:
  * reader unit checks (stdlib);
  * policy-table fingerprint against an independent reference implementation of the same rule;
  * backstop modes (sampled EV = the previous label, EV argmax) and the blueprint gate;
  * parse parity on eval-formatted states and the pool generator (need pyspiel);
  * prewarm/worker-init gate in generate_trajectories (needs `datasets`);
  * teacher-level gate: seeded, seat-swapped games on common deals vs the field styles.

Run from scripts/envs/:  python test_liars_reader.py
"""
import functools
import hashlib
import json
import math
import os
import random
import sys
import types
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.liar_dice_expert as ld  # noqa: E402

TOTAL = 10
LIAR = ld.liar_id(TOTAL)
HANDS = ld._hands(5)
PRIOR = [ld._hand_probability(h) for h in HANDS]
_ENV_KEYS = ("LIARS_TEACHER", "LIARS_DICE_FORCE_EV", "LIARS_GEN")
FAILS = []

# sha256 of the (current bid 0..59, then no bid) x 252 sorted hands -> action table, produced by an
# independent implementation of the same rules (see _table_digest for the exact serialisation).
REFERENCE_DIGESTS = {
    "reader": "476d03073f94a0e518385e0159e7f9f0edb92cb85c9a9d846887fe9c05773c09",
    "reader_u0_lo": "4e468c1c8645f209a845229c8bc133762b88f7be9aa3d483d729ad4918898448",
    "reader_u0_hi": "7234dd85d99f313a92fdec0e3b2823d5afa901a7d0f9201dd0845d4abc77184f",
}


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def clear_env() -> None:
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def bid_label(aid: int) -> str:
    if aid == LIAR:
        return "Liar"
    q, f = ld.decode_bid(aid)
    return f"{q}-{f}"


def obs(dice, cur, player: int = 0) -> str:
    """Teacher envelope around an eval-formatted liars state (= pvp_selfplay._teacher_obs(
    _format_liars_dice(...)))."""
    lines = [
        f"Your dice: {list(dice)} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {len(dice)}",
        f"Total dice in game: {2 * len(dice)}",
        "Players: 2",
        f"Current player: Player {player}",
    ]
    if cur is None:
        lines.append("No bid yet - you must make the first bid")
    else:
        q, f = ld.decode_bid(cur)
        lines.append(f'\nCurrent bid: "{q}-{f}" (at least {q} dice showing {f} across all players)')
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    legal = ld.legal_action_ids_for(cur, 2 * len(dice))
    actions = "\n".join(f"{a} -> {bid_label(a)}" for a in legal)
    return (f"Current State:\n" + "\n".join(lines) + f"\nYou are Player {player}.\n"
            f"Legal Actions:\n{actions}\nYour choice (ID only):")


def random_state(rng: random.Random):
    dice = [rng.randint(1, 6) for _ in range(5)]
    cur = None if rng.random() < 0.15 else rng.randrange(LIAR)
    return dice, cur, ld.legal_action_ids_for(cur, TOTAL)


def own_match(dice, face: int) -> int:
    return ld._matching_count(tuple(dice), face)


def _table_digest(action_fn) -> str:
    rows = []
    for cur in list(range(LIAR)) + [None]:
        legal = ld.legal_action_ids_for(cur, TOTAL)
        rows.append(",".join(str(action_fn(list(h), cur, legal)) for h in HANDS))
    return hashlib.sha256(";".join(rows).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------
# Reader unit checks
# ---------------------------------------------------------------------------------------------------

def test_reader_units() -> None:
    clear_env()
    check("default teacher is the visible-bid reader", ld._teacher_mode() == "reader")
    os.environ["LIARS_TEACHER"] = "bogus"
    check("unknown LIARS_TEACHER falls back to the reader", ld._teacher_mode() == "reader")
    clear_env()

    rng = random.Random(123)
    states = [random_state(rng) for _ in range(2000)]
    same = all(ld.choose_action(obs(d, c), 0, lg, rng=random.Random(1))
               == ld.choose_action(obs(d, c), 0, lg, rng=random.Random(99))
               == ld.choose_action(obs(d, c), 0, lg) for d, c, lg in states[:300])
    check("same observation, different rngs -> identical action", same)

    acts = [ld.choose_action(obs(d, c), 0, lg, rng=random.Random(5)) for d, c, lg in states]
    check("always legal over 2000 random (hand, bid) states",
          all(a in lg for a, (_, _, lg) in zip(acts, states)))
    covered = [(a, c) for a, (d, c, _) in zip(acts, states)
               if c is not None and own_match(d, ld.decode_bid(c)[1]) >= ld.decode_bid(c)[0]]
    check(f"never Liar on a bid our own hand satisfies ({len(covered)} states)",
          bool(covered) and all(a != LIAR for a, _ in covered))
    openings = [ld.choose_action(obs(list(h), None), 0, list(range(LIAR))) for h in HANDS]
    check("opening is always a bid (all 252 hands)", all(0 <= a < LIAR for a in openings))

    bid_10_5 = ld.encode_bid(10, 5)
    check("impossible bid: [1,1,2,2,3] vs '10-5' -> Liar",
          ld.choose_action(obs([1, 1, 2, 2, 3], bid_10_5), 1, ld.legal_action_ids_for(bid_10_5, TOTAL)) == LIAR)

    prior = ld._opp_match_dist(None, TOTAL, 5)
    post = ld._opp_match_dist(ld.encode_bid(3, 4), TOTAL, 5)
    check("posterior after '3-4' raises P(opponent holds >=1 / >=2 of face 4) above the prior",
          sum(post[4][1:]) > sum(prior[4][1:]) and sum(post[4][2:]) > sum(prior[4][2:]))
    check("posterior rows are distributions",
          all(abs(sum(post[f]) - 1.0) < 1e-9 for f in range(1, 7)))

    certain = [a for a in range(LIAR) if ld._p_true_post(*ld.decode_bid(a), [2, 2, 2, 2, 2], prior) >= 1.0]
    opening = ld.choose_action(obs([2, 2, 2, 2, 2], None), 0, list(range(LIAR)))
    check(f"tie-high: several P=1 openings {certain} -> the highest ('5-2' = id 25)",
          len(certain) >= 2 and opening == max(certain) == 25)

    check("unparsable observation -> still a legal action",
          ld.choose_action("garbage", 0, [3, 4, 60]) in (3, 4, 60))
    mismatch = obs([1, 2, 3, 4, 5], 7).replace("Total dice in game: 10", "Total dice in game: 12")
    check("total dice != 2 x own dice -> EV fallback, legal",
          ld.choose_action(mismatch, 0, ld.legal_action_ids_for(7, TOTAL)) in ld.legal_action_ids_for(7, TOTAL))


def test_reference_tables() -> None:
    clear_env()
    for kind, (mix, high) in {"reader": (ld.LIARS_PRIOR_MIX, True),
                              "reader_u0_lo": (0.0, False), "reader_u0_hi": (0.0, True)}.items():
        digest = _table_digest(lambda d, c, lg: ld._reader_action(d, TOTAL, c, lg, prior_mix=mix, tie_high=high))
        check(f"{kind} policy table == reference implementation ({digest[:12]})", digest == REFERENCE_DIGESTS[kind])
    digest = _table_digest(lambda d, c, lg: ld.choose_action(obs(d, c), 0, lg))
    check("choose_action (default) table == reference reader table", digest == REFERENCE_DIGESTS["reader"])


def test_backstops_and_blueprint_gate() -> None:
    clear_env()
    rng = random.Random(77)
    states = [random_state(rng) for _ in range(400)]
    os.environ["LIARS_TEACHER"] = "ev"
    same_as_head = True
    for i, (d, c, lg) in enumerate(states):
        dist = ld._heuristic_distribution(d, TOTAL, c, lg)
        ids = list(dist)
        expected = random.Random(i).choices(ids, weights=[dist[a] for a in ids], k=1)[0]
        same_as_head &= ld.choose_action(obs(d, c), 0, lg, rng=random.Random(i)) == expected
    check("LIARS_TEACHER=ev is the previous sampled flat-EV label", same_as_head)

    os.environ["LIARS_TEACHER"] = "ev_argmax"
    acts = [ld.choose_action(obs(d, c), 0, lg, rng=random.Random(i)) for i, (d, c, lg) in enumerate(states)]
    again = [ld.choose_action(obs(d, c), 0, lg, rng=random.Random(i + 1)) for i, (d, c, lg) in enumerate(states)]
    check("LIARS_TEACHER=ev_argmax is deterministic and legal",
          acts == again and all(a in lg for a, (_, _, lg) in zip(acts, states)))
    check("ev_argmax ties go to the highest id ([2,2,2,2,2] opens '5-2')",
          ld.choose_action(obs([2, 2, 2, 2, 2], None), 0, list(range(LIAR))) == 25)

    clear_env()
    check("blueprint not needed by default", not ld.blueprint_needed())
    os.environ["LIARS_DICE_FORCE_EV"] = "0"
    check("blueprint not needed by the reader even with LIARS_DICE_FORCE_EV=0", not ld.blueprint_needed())
    os.environ["LIARS_TEACHER"] = "ev"
    check("blueprint needed only for LIARS_TEACHER=ev + LIARS_DICE_FORCE_EV=0", ld.blueprint_needed())
    os.environ["LIARS_DICE_FORCE_EV"] = "1"
    check("blueprint not needed for LIARS_TEACHER=ev with the EV default", not ld.blueprint_needed())
    clear_env()


# ---------------------------------------------------------------------------------------------------
# pyspiel: parse parity + pool generator
# ---------------------------------------------------------------------------------------------------

def _parse_user_prompt(user: str):
    body = user[len("Current state:\n"):]
    state_desc, _, rest = body.partition("\n\nYou are Player ")
    pid, _, after = rest.partition(".\nLegal actions:\n")
    legal = []
    for line in after.splitlines():
        if "->" in line:
            aid, _, lab = line.partition("->")
            legal.append((int(aid.strip()), lab.strip()))
    return state_desc, int(pid), legal


def test_pyspiel_paths() -> None:
    try:
        import pyspiel
    except ImportError as exc:
        print(f"SKIP pyspiel checks ({exc})")
        return
    import envs.liar_dice_pool as pool
    import envs.pvp_selfplay as sp
    clear_env()
    game = pyspiel.load_game("liars_dice", {"players": 2, "numdice": 5})
    rng = random.Random(3)
    parsed_ok, n = True, 0
    for _ in range(60):
        state, last_bid = game.new_initial_state(), None
        while not state.is_terminal():
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            p = state.current_player()
            legal = state.legal_actions(p)
            labels = [(a, state.action_to_string(p, a)) for a in legal]
            teacher_obs = sp._teacher_obs(sp._format_liars_dice(state, p), p, labels)
            dice = [int(c) for c in state.information_state_string(p).split()[0] if c.isdigit()]
            parsed_ok &= ld._parse_observation(teacher_obs) == (dice, TOTAL, last_bid)
            parsed_ok &= ld.choose_action(teacher_obs, p, legal) in legal
            n += 1
            a = rng.choice(legal)
            last_bid = a if a != LIAR else last_bid
            state.apply_action(a)
    check(f"parse parity on {n} eval-formatted states (dice, total, latest bid)", parsed_ok)

    kinds = pool.OPPONENT_POOL
    check(f">= 50% of pool games are reader self-play ({kinds.count('reader')}/{len(kinds)})",
          kinds.count("reader") * 2 >= len(kinds))
    check("pool opponents are known styles", set(kinds) <= set(ld.OPPONENT_KINDS))

    base = 150_000_000
    labels_ok, seats_ok, rows, games = True, True, 0, 400
    for seed in range(base, base + games):
        views = pool.play_liars_pool_game(seed, max_turn=30)
        kind, hero = pool.opponent_for(seed)
        seats = set()
        for examples, score in views:
            labels_ok &= 0.0 <= score <= 1.0
            for ex in examples:
                msgs = ex["messages"]
                state_desc, pid, legal = _parse_user_prompt(msgs[1]["content"])
                chosen = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["action_id"]
                labels_ok &= chosen == ld.choose_action(sp._teacher_obs(state_desc, pid, legal), pid,
                                                        [a for a, _ in legal])
                seats.add(pid)
                rows += 1
        seats_ok &= seats == ({0, 1} if kind == "reader" else {hero})
    check("every pool row is labelled by the reader (label == choose_action(obs))", labels_ok)
    check("both seats recorded vs the reader, only the hero seat otherwise", seats_ok)
    mean_rows = rows / games
    check(f"rows per game {mean_rows:.2f} in [2, 12]", 2.0 <= mean_rows <= 12.0)
    check("hero seat alternates within every opponent style",
          all({pool.opponent_for(s)[1] for s in range(base, base + 64) if pool.opponent_for(s)[0] == k} == {0, 1}
              for k in set(kinds)))
    check("pool game is deterministic per seed",
          pool.play_liars_pool_game(base + 7, 30) == pool.play_liars_pool_game(base + 7, 30))

    gen = sp.make_selfplay_generator("liars_dice")
    check("make_selfplay_generator routes liars_dice to the pool generator by default",
          all(gen(s, None, 30) == pool.play_liars_pool_game(s, 30, random.Random(s)) for s in range(base, base + 16)))
    os.environ["LIARS_GEN"] = "selfplay"
    gen = sp.make_selfplay_generator("liars_dice")
    check("LIARS_GEN=selfplay restores mirror self-play",
          gen(base + 3, None, 30) == sp.play_selfplay_game("liars_dice", base + 3, 30, random.Random(base + 3)))
    clear_env()


# ---------------------------------------------------------------------------------------------------
# generate_trajectories: no MCCFR blueprint build unless the MCCFR teacher is selected
# ---------------------------------------------------------------------------------------------------

def test_prewarm_gate() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        print(f"SKIP prewarm gate ({exc})")
        return
    saved = {k: sys.modules.get(k) for k in ("trl_compat", "envs.shared_env", "envs.sft_env_configs")}
    sys.modules["trl_compat"] = types.ModuleType("trl_compat")
    shared = types.ModuleType("envs.shared_env")
    shared.GAMES_TO_TASK_ID_RANGE, shared.init_env_pool, shared._log = {}, (lambda *a, **k: None), (lambda *a, **k: None)
    shared.recipe_log = lambda *a, **k: None
    shared.generation_id_bounds = lambda env_name: (0, 1)
    sys.modules["envs.shared_env"] = shared
    cfg = types.ModuleType("envs.sft_env_configs")
    cfg._OFFLINE_ENVS, cfg._SELFPLAY_ENVS = {"intercode"}, {"liars_dice"}
    cfg.get_sft_trajectory_generator = lambda *a, **k: None
    sys.modules["envs.sft_env_configs"] = cfg
    real_get = ld._get_blueprint
    calls = []

    def _no_build(total_dice):
        calls.append(total_dice)
        raise RuntimeError("blueprint build must not run")

    try:
        import envs.generate_trajectories as gt
        ld._get_blueprint = _no_build
        clear_env()
        gt._prewarm_env("liars_dice")
        gt._worker_init("liars_dice")
        check("default: _prewarm_env / _worker_init never build the MCCFR blueprint", calls == [])
        os.environ["LIARS_TEACHER"], os.environ["LIARS_DICE_FORCE_EV"] = "ev", "0"
        gt._prewarm_env("liars_dice")
        gt._worker_init("liars_dice")
        check("opt-in MCCFR teacher still warms its blueprint", calls == [10, 10])
    finally:
        ld._get_blueprint = real_get
        clear_env()
        for k, mod in saved.items():
            if mod is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = mod


# ---------------------------------------------------------------------------------------------------
# Teacher-level gate: seeded deals, both seatings of each deal, vs the field styles
# ---------------------------------------------------------------------------------------------------

_TEMPS = (0.01, 0.05, 0.1, 0.3)
_MATCH = [[ld._matching_count(h, f) for f in range(7)] for h in HANDS]


@functools.lru_cache(maxsize=None)
def _move_likelihood(ctx):
    """lik[h][i]: P(a player holding HANDS[h] plays legal(ctx)[i]) under a softmax of its own P(claim
    true) averaged over _TEMPS (an opponent model for the full-history reader below)."""
    legal = ld.legal_action_ids_for(ctx, TOTAL)
    out = []
    for h in HANDS:
        scores = [1.0 - ld._bid_true_prob(list(h), ld.decode_bid(ctx), 5) if a == LIAR
                  else ld._bid_true_prob(list(h), ld.decode_bid(a), 5) for a in legal]
        m = max(scores)
        acc = [0.0] * len(legal)
        for t in _TEMPS:
            w = [math.exp((s - m) / t) for s in scores]
            z = sum(w)
            acc = [x + y / z for x, y in zip(acc, w)]
        out.append([x / len(_TEMPS) for x in acc])
    return out


def _full_history_action(dice, seat: int, hist, visible_only: bool = False) -> int:
    """A reader that sees the WHOLE bid history (the eval prompt never shows it): every opponent bid
    multiplies in the likelihood of raising to it from the bid before it; argmax P(claim true), ties to
    the lowest id, Liar only when strictly better. visible_only uses the latest bid as an opening."""
    cur = hist[-1] if hist else None
    w = list(PRIOR)
    seen = [(None, cur)] if (visible_only and cur is not None) else [
        (hist[i - 1] if i else None, b) for i, b in enumerate(hist) if i % 2 != seat]
    for ctx, b in seen:
        lik = _move_likelihood(ctx)
        idx = b if ctx is None else b - ctx - 1
        w = [x * lik[h][idx] for h, x in enumerate(w)]
    z = sum(w) or 1.0
    md = [[0.0] * 6 for _ in range(7)]
    for h, x in enumerate(w):
        for f in range(1, 7):
            md[f][_MATCH[h][f]] += x / z
    best, best_s = None, -1.0
    for a in ld.legal_action_ids_for(cur, TOTAL):
        q, f = ld.decode_bid(cur if a == LIAR else a)
        s = ld._p_true_post(q, f, dice, md)
        s = 1.0 - s if a == LIAR else s
        if s > best_s + 1e-12:
            best, best_s = a, s
    return best


def _deal(g: int, seed: int):
    r = random.Random(seed * 1_000_003 + g)
    return [[r.randint(1, 6) for _ in range(5)] for _ in range(2)], r


def _play(pols, hands, rng) -> int:
    hist, p = [], 0
    while True:
        a = pols[p](hands[p], p, hist, rng)
        if a == LIAR:
            q, f = ld.decode_bid(hist[-1])
            true = own_match(hands[0], f) + own_match(hands[1], f) >= q
            return (1 - p) if true else p
        hist.append(a)
        p = 1 - p


def head_to_head(pa, pb, deals: int, seed: int) -> float:
    wins = 0
    for g in range(deals):
        hands, r = _deal(g, seed)
        wins += _play((pa, pb), hands, r) == 0
        hands, r = _deal(g, seed)
        wins += _play((pb, pa), hands, r) == 1
    return wins / (2 * deals)


def _labelled(dice, seat, hist):
    cur = hist[-1] if hist else None
    return obs(dice, cur, seat), ld.legal_action_ids_for(cur, TOTAL)


def pol_reader(dice, seat, hist, rng):
    o, legal = _labelled(dice, seat, hist)
    return ld.choose_action(o, seat, legal, rng=rng)


def opponent(kind):
    def pol(dice, seat, hist, rng):
        o, legal = _labelled(dice, seat, hist)
        return ld.opponent_action(kind, o, legal, rng)
    return pol


def pol_random(dice, seat, hist, rng):
    return rng.choice(ld.legal_action_ids_for(hist[-1] if hist else None, TOTAL))


def visible_student(games: int, seed: int):
    """Imitation of the full-history reader by a visible-state student: its most frequent label per
    (hand, latest bid) over self-play games, else the same rule fed only the latest bid."""
    table = {}
    rng = random.Random(seed)
    for _ in range(games):
        hands = [[rng.randint(1, 6) for _ in range(5)] for _ in range(2)]
        hist, p = [], 0
        while True:
            a = _full_history_action(hands[p], p, hist)
            table.setdefault((tuple(sorted(hands[p])), hist[-1] if hist else None), Counter())[a] += 1
            if a == LIAR:
                break
            hist.append(a)
            p = 1 - p
    mode = {k: c.most_common(1)[0][0] for k, c in table.items()}

    def pol(dice, seat, hist, rng):
        cur = hist[-1] if hist else None
        a = mode.get((tuple(sorted(dice)), cur))
        if a is None or a not in ld.legal_action_ids_for(cur, TOTAL):
            return _full_history_action(dice, seat, hist, visible_only=True)
        return a
    return pol


def test_teacher_gate() -> None:
    clear_env()
    deals = int(os.getenv("LIARS_GATE_DEALS", "2000"))
    bars = [("the previous sampled EV label", opponent("ev_sampled"), 0.62),
            ("a lowest-tie visible reader", opponent("reader_u0_lo"), 0.56),
            ("a highest-tie visible reader", opponent("reader_u0_hi"), 0.56),
            ("a visible-state student of a full-history reader", visible_student(3000, 5), 0.55),
            ("uniform random", pol_random, 0.95)]
    for name, pol, bar in bars:
        rate = head_to_head(pol_reader, pol, deals, seed=21)
        check(f"reader vs {name}: {rate:.3f} >= {bar} ({2 * deals} seat-swapped games)", rate >= bar)


if __name__ == "__main__":
    test_reader_units()
    test_reference_tables()
    test_backstops_and_blueprint_gate()
    test_pyspiel_paths()
    test_prewarm_gate()
    test_teacher_gate()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
