"""Goofspiel teacher-level meta check: our label vs the rule field on the EVAL rules.

Plays seat-swapped games (every deal twice, seats swapped, like the validator's matchups) between
goofspiel_expert policies on the eval variant: imp_info, random prize order, win/loss, 5/8/10/13 cards.
Policies read the same observation envelope the generator builds, so the real labels are measured.
Both seats can carry student-style noise: with probability ``eps`` a bid becomes an adjacent card in
hand ('adj', an off-by-one slip) or any other legal card ('uni'). The MCTS bot never gets noise.

Engines:
  sim      pure python (default, fast); renders the pyspiel observation text byte-for-byte
  pyspiel  the real turn-based game via pvp_selfplay; adds 'mcts' = the validator's baseline bot
           (OpenSpiel MCTS, 50 simulations, 1 random rollout, uct_c 2). It searches the true state, so
           as the second mover it reads the pending bid (every rule scores ~0 in that seat); it is the
           baseline opponent, not a tournament one, and never a generation opponent here.

Run from scripts/envs:
  python verify_goofspiel_meta.py                                   # sim, 400 games/pair, eps 0.1 adj
  python verify_goofspiel_meta.py --engine pyspiel --eps 0 --per-deck \\
      --field one-up,two-up,three-up,matching,mcts,random
  python verify_goofspiel_meta.py --gate                            # regression floors, exit 1 on a miss
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
if "envs" not in sys.modules:
    sys.path.insert(0, os.path.dirname(_HERE))
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

import envs.goofspiel_expert as ge  # noqa: E402

DECKS = (5, 8, 10, 13)
TIE = -3  # pyspiel's win-sequence entry for a tied round
_POLICY_SALT = 0x5EA7  # policies and noise draw from their own stream, so a deal replays across seats
_GATE_FLOOR = 0.60
_GATE_FIELD = ("one-up", "two-up", "three-up", "matching", "random",
               "strict-two-up", "one-up-tie", "overbid-deck")
_GATE_PREVIOUS = ("two-up", "up2abs3")  # the 13759ee label and skim-up v1
_GATE_HEAD_FLOOR = 0.55  # vs each previous label under the harsher uniform-noise model


def policies() -> dict:
    """Name -> policy(obs, player_id, legal_ids, rng). 'default' is choose_action (GOOFSPIEL_TEACHER aware)."""
    return {"default": ge.choose_action, **dict(ge.OPPONENT_POOL)}


def render_state(pid: int, prize: int, future: list[int], points: tuple[int, int], hand: list[int],
                 seq: list[int]) -> str:
    """The eval state text (pvp_selfplay._format_goofspiel of the imp_info observation) for seat ``pid``."""
    return (f"You are Player {pid} (P{pid}).\nCurrent player: {pid}\nCurrent point card: {prize}\n"
            f"Remaining Point Cards: {''.join(map(str, sorted(future)))}\n"
            f"Points: {points[0]} {points[1]} \nP{pid} hand: {' '.join(map(str, sorted(hand)))} \n"
            f"Win sequence: {''.join(f'{w} ' for w in seq)}\n")


def _envelope(state_desc: str, pid: int, legal: list[tuple[int, str]]) -> str:
    """Same envelope as pvp_selfplay._teacher_obs (kept here so the sim engine needs no pyspiel)."""
    lines = "\n".join(f"{aid} -> {label}" for aid, label in legal)
    return f"Current State:\n{state_desc}\nYou are Player {pid}.\nLegal Actions:\n{lines}\nYour choice (ID only):"


def _slip(aid: int, legal: list[int], eps: float, noise: str, rng: random.Random) -> int:
    if eps <= 0.0 or len(legal) < 2 or rng.random() >= eps:
        return aid
    if noise == "adj":
        order = sorted(legal)  # action id = card - 1, so id order is card order
        i = order.index(aid)
        return rng.choice([order[j] for j in (i - 1, i + 1) if 0 <= j < len(order)])
    return rng.choice([a for a in legal if a != aid])


def _outcome(points: "list[float]") -> float:
    return 1.0 if points[0] > points[1] else 0.0 if points[0] < points[1] else 0.5


def play_sim(n: int, seats: tuple, deal_seed: int, eps: tuple, noise: str) -> float:
    """One game on the pure-python engine; returns P0's score (1 / 0.5 / 0)."""
    order = list(range(1, n + 1))
    random.Random(deal_seed).shuffle(order)
    rng = random.Random(deal_seed ^ _POLICY_SALT)
    hands = [set(range(1, n + 1)), set(range(1, n + 1))]
    points, seq = [0, 0], []
    for rnd, prize in enumerate(order):
        bids = []
        for pid in (0, 1):
            hand = sorted(hands[pid])
            legal = [c - 1 for c in hand]
            desc = render_state(pid, prize, order[rnd + 1:], (points[0], points[1]), hand, seq)
            obs = _envelope(desc, pid, [(c - 1, f"[P{pid}]Bid: {c}") for c in hand])
            aid = seats[pid](obs, pid, legal, rng)
            bids.append(_slip(aid if aid in legal else legal[0], legal, eps[pid], noise, rng) + 1)
        winner = 0 if bids[0] > bids[1] else 1 if bids[1] > bids[0] else TIE
        if winner != TIE:
            points[winner] += prize
        seq.append(winner)
        hands[0].discard(bids[0])
        hands[1].discard(bids[1])
    return _outcome(points)


def play_pyspiel(n: int, seats: tuple, deal_seed: int, eps: tuple, noise: str) -> float:
    """One game on the real eval game (pvp_selfplay._load_game); a seat may be the string 'mcts'."""
    from envs.pvp_selfplay import _FORMATTERS, _load_game, _make_mcts_bot, _teacher_obs

    game = _load_game("goofspiel", DECKS.index(n))
    chance_rng, rng = random.Random(deal_seed), random.Random(deal_seed ^ _POLICY_SALT)
    bots = {p: _make_mcts_bot(game, deal_seed + p, 50) for p in (0, 1) if seats[p] == "mcts"}
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(chance_rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        pid = state.current_player()
        legal = state.legal_actions(pid)
        if pid in bots:
            state.apply_action(bots[pid].step(state))
            continue
        labels = [(a, state.action_to_string(pid, a)) for a in legal]
        aid = seats[pid](_teacher_obs(_FORMATTERS["goofspiel"](state, pid), pid, labels), pid, legal, rng)
        state.apply_action(_slip(aid if aid in legal else legal[0], legal, eps[pid], noise, rng))
    return _outcome(state.returns())


def rate(a, b, games: int = 400, eps: float = 0.1, noise: str = "adj", decks=DECKS,
         engine: str = "sim", seed: int = 1) -> float:
    """A's mean score vs B: games/2 deals cycling over ``decks``, each played with A in both seats."""
    play = play_pyspiel if engine == "pyspiel" else play_sim
    total = 0.0
    for g in range(games):
        deal, a_seat = g // 2, g % 2
        n = decks[deal % len(decks)]
        seats = (a, b) if a_seat == 0 else (b, a)
        p0 = play(n, seats, seed * 1_000_003 + deal, (eps, eps), noise)
        total += p0 if a_seat == 0 else 1.0 - p0
    return total / games


def gate_checks(games: int = 400, engine: str = "sim") -> list[tuple[str, float, float]]:
    """(opponent, score, floor) for the default label: >= 0.60 vs each field rule at eps 0.1 off-by-one
    noise on both seats, and >= 0.55 vs the previous labels (two-up, up2abs3) with eps 0.1 uniform noise."""
    pol = policies()
    out = [(name, rate(pol["default"], pol[name], games, 0.1, "adj", engine=engine), _GATE_FLOOR)
           for name in _GATE_FIELD]
    for prev in _GATE_PREVIOUS:
        out.append((f"{prev} (uniform noise)",
                    rate(pol["default"], pol[prev], games, 0.1, "uni", engine=engine), _GATE_HEAD_FLOOR))
    return out


def _matrix(cands, field, games, eps, noise, decks, engine) -> None:
    pol = {**policies(), "mcts": "mcts"}
    print("candidate".ljust(12) + "".join(f[:9].rjust(10) for f in field) + "     mean  worst")
    for c in cands:
        rs = [rate(pol[c], pol[f], games, eps, noise, decks, engine) for f in field]
        print(c.ljust(12) + "".join(f"{x:10.3f}" for x in rs) + f"  {sum(rs) / len(rs):7.3f}  {min(rs):.3f}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=("sim", "pyspiel"), default="sim")
    ap.add_argument("--games", type=int, default=400, help="games per pair (per deck with --per-deck)")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--noise", choices=("adj", "uni"), default="adj")
    ap.add_argument("--cands", default="default,up2abs3,two-up")
    ap.add_argument("--field", default=",".join(_GATE_FIELD))
    ap.add_argument("--per-deck", action="store_true", help="one matrix per deck size instead of pooled")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    if args.gate:
        fails = 0
        for name, score, floor in gate_checks(args.games, args.engine):
            ok = score >= floor
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} default vs {name}: {score:.3f} (floor {floor:.2f})")
        print("RESULT", "PASS" if not fails else f"FAIL ({fails})")
        return 1 if fails else 0

    cands, field = args.cands.split(","), args.field.split(",")
    print(f"engine={args.engine} games/pair={args.games} eps={args.eps} noise={args.noise} "
          f"teacher={ge._teacher_mode()}")
    for decks in ([(n,) for n in DECKS] if args.per_deck else [DECKS]):
        print(f"-- decks {'/'.join(map(str, decks))}")
        _matrix(cands, field, args.games, args.eps, args.noise, decks, args.engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
