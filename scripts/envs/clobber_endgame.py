"""Exact endgame solver for Clobber, used as a STRONG move-selection teacher during
offline SFT data generation (a drop-in upgrade over the greedy heuristic in
clobber_expert.py, which only looks one ply ahead — hence its ~coin-flip win-rate).

Why an exact solve is cheap here: Clobber is normal-play (a player with no legal move
loses), zero-sum, and DRAW-FREE, and every move deletes exactly one stone — so the game
tree shrinks by one piece per ply and the branching factor is small. In a {-1,+1}
draw-free game α-β degenerates to a plain AND/OR question — "does the side to move have a
move that leaves the opponent unable to win?" — which a memoised negamax answers exactly
once the board is small. Above a piece threshold (or if the node/time budget is spent) we
defer to the fast greedy heuristic, so generation never stalls on a deep midgame.

This runs ONLY while generating training data, to distill stronger move targets; at eval
the model plays alone with no search (fully within the rules). Every failure path falls
back to the heuristic, so the solver can neither crash nor hang trajectory generation.
"""
from __future__ import annotations

import math
import os
import random
import time

from envs.clobber_expert import (
    _parse_board,
    _extract_action_labels,
    _label_to_indices,
    choose_action as _greedy_choose,
)

# LIVE ROUND-2 LESSON (clobber lost 46-66 / 33-53): the endgame is solved exactly, but a
# 5x6 board starts at 30 stones, so the whole OPENING+MIDGAME (30 → ~16 stones, ~14 plies)
# was played by the 1-ply greedy heuristic — a coin-flip. Fix: a depth-limited negamax
# (mobility leaf eval) gives real midgame lookahead, and any subtree that drops to the exact
# threshold is resolved perfectly. Below the threshold we still solve to the end.
_SOLVE_MAX_PIECES = int(os.getenv("CLOBBER_SOLVE_MAX_PIECES", "16"))  # stones at/below which we solve EXACTLY; env-tunable (was hardcoded 16) so offline gen can solve DEEPER (perfect midgame) — falls back to the heuristic on any position it can't solve within the time/node budget, so raising it never hangs generation
# Negamax lookahead plies above the exact threshold. Default 4; env-tunable because the
# open-sourced R1-winner's clobber teacher runs a DEEP negamax (14-16 ply) + mobility eval
# (alpha-beta prunes hard on this draw-free game), so CLOBBER_MIDGAME_DEPTH lets us A/B a
# deep negamax vs the MCTS teacher. Default unchanged -> gen behaviour identical unless opted in.
_MIDGAME_DEPTH = int(os.getenv("CLOBBER_MIDGAME_DEPTH", "4"))
_NODE_BUDGET = int(os.getenv("CLOBBER_NODE_BUDGET", "2000000"))   # search node cap per decision (raise for deep search)
# Wall-clock seconds per decision before falling back (was 0.1). Env-tunable so the VM
# head-to-head can measure MCTS at full 400-sim strength (e.g. CLOBBER_MOVE_TIME=2.0) as well
# as at the 0.3s deployment budget. DEFAULT UNCHANGED (0.3) -> gen behaviour is identical.
_MOVE_TIME_BUDGET = float(os.getenv("CLOBBER_MOVE_TIME", "0.3"))
# Exact win/loss must dominate any mobility score (mobility ∈ roughly [-30, 30]).
_WIN_SCORE = 1_000_000.0
_INF = float("inf")

# LEAF-EVAL MODE. The 2026-07-20 head-to-head measured each mode's TEACHER strength (win-rate vs
# the 1-ply greedy) and picked "greedy". CAVEAT (2026-08-13, from the open-sourced R1 WINNER):
# teacher-vs-teacher win-rate is the WRONG metric — what wins is the DISTILLED STUDENT's play vs
# the eval MCTS, and that hinges on LABEL LEARNABILITY (consistent, smooth, no smallest-id bias),
# NOT teacher head-to-head strength. So re-measure by TRAINING a student per mode and evaluating
# it vs the eval opponent, not by teacher self-play.
#   * "greedy"   (DEFAULT, proven — placed 3rd in R1, ~1% off 1st): greedy midgame + EXACT endgame.
#   * "mobility" (opt-in, UPGRADED 2026-08-13 to the winner's recipe): depth-BY-PIECE-COUNT negamax
#                 (was flat depth-4) + SEMANTIC tie-break (was smallest-id) + our exact endgame.
#                 The old flat-4/smallest-id variant measured 45% teacher-vs-greedy; this upgrade
#                 targets the winner's learnability fixes — RE-MEASURE AS A STUDENT, not a teacher.
#   * "rollout"  (opt-in): flat Monte-Carlo → 35% teacher-vs-greedy (noisy; reference only).
def _resolve_leaf_eval() -> str:
    """RANK-9 gate. CLOBBER_DEEP_MOBILITY=1 promotes the DEEP-mobility negamax + exact-endgame teacher
    to the default leaf eval (replacing the 1-ply greedy coin-flip) WITHOUT touching the committed
    default. Precedence: an explicit CLOBBER_LEAF_EVAL always wins (so A/B still works); only when it
    is UNSET does CLOBBER_DEEP_MOBILITY select 'mobility'. Flag off + CLOBBER_LEAF_EVAL unset =>
    'greedy' (byte-identical prior default, kept as the A/B backstop). Never raises."""
    explicit = os.getenv("CLOBBER_LEAF_EVAL")
    if explicit not in (None, ""):
        return explicit.strip().lower()
    deep = (os.getenv("CLOBBER_DEEP_MOBILITY") or "").strip().lower()
    if deep not in ("", "0", "false", "no", "off"):
        return "mobility"
    return "greedy"


_LEAF_EVAL = _resolve_leaf_eval()  # greedy | mcts | mobility | rollout
_LAST_REASON = ""  # RANK-9 debuggability: the last decision's eval mode + chosen move


def _set_reason(msg: str) -> None:
    global _LAST_REASON
    _LAST_REASON = msg


def last_reason() -> str:
    """Human-readable note on the last clobber decision (debug only)."""
    return _LAST_REASON
_ROLLOUTS = int(os.getenv("CLOBBER_ROLLOUTS", "24"))  # random playouts per root move (rollout mode)

# CLOBBER_LEAF_EVAL=mcts — UCT Monte-Carlo Tree Search with EXACT-solver leaves. Grounded in
# the Clobber literature (Maastricht "CGT + alpha-beta", Claessen "CGT-value DB + MCTS"): a
# strong Clobber engine needs SEARCH backed by exact endgame values, NOT a 1-ply heuristic.
# We approximate the CGT-value database with our exact `_wins` solver at small nodes, and let
# UCT do the lookahead the 30->16 opening/midgame needs (where greedy/mobility/rollout tie or
# lose). More sims = stronger; the offline prebake path can afford a high sim count.
_MCTS_SIMS = int(os.getenv("CLOBBER_MCTS_SIMS", "400"))   # UCT simulations per decision
_MCTS_C = float(os.getenv("CLOBBER_MCTS_C", "1.4"))       # UCB1 exploration constant

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _BudgetExceeded(Exception):
    """Node or time budget exhausted mid-search — abandon the solve, use the heuristic."""


def _to_board(grid: "list[list[str]]") -> "tuple[str, ...]":
    """Immutable, hashable board — one string per row, cheap to copy and to use as a
    transposition-table key."""
    return tuple("".join(row) for row in grid)


def _piece_count(board: "tuple[str, ...]") -> int:
    return sum(row.count("x") + row.count("o") for row in board)


def _legal_moves(board: "tuple[str, ...]", me: str, opp: str) -> "list[tuple[int, int, int, int]]":
    """Every (r, c, nr, nc): my stone at (r,c) clobbering an orthogonally adjacent enemy
    stone at (nr,nc)."""
    rows, cols = len(board), len(board[0])
    moves: "list[tuple[int, int, int, int]]" = []
    for r in range(rows):
        row = board[r]
        for c in range(cols):
            if row[c] == me:
                for dr, dc in _NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == opp:
                        moves.append((r, c, nr, nc))
    return moves


def _apply(board: "tuple[str, ...]", r: int, c: int, nr: int, nc: int, me: str) -> "tuple[str, ...]":
    """Board after my stone moves (r,c) -> (nr,nc): source emptied, destination becomes
    mine (the clobbered enemy stone is gone). Handles nr==r correctly."""
    rows = list(board)
    rows[r] = rows[r][:c] + "." + rows[r][c + 1:]
    rows[nr] = rows[nr][:nc] + me + rows[nr][nc + 1:]  # re-reads rows[nr] after the edit above
    return tuple(rows)


def _wins(board: "tuple[str, ...]", me: str, opp: str,
          tt: dict, budget: "list[int]", deadline: float) -> bool:
    """True iff the side to move (`me`) can FORCE a win. Draw-free {-1,+1} game, so this
    is a memoised AND/OR search: no legal move -> I lose; otherwise I win iff some move
    leaves the opponent unable to force a win."""
    budget[0] -= 1
    if budget[0] <= 0 or time.monotonic() > deadline:
        raise _BudgetExceeded
    key = (board, me)
    cached = tt.get(key)
    if cached is not None:
        return cached
    result = False
    for (r, c, nr, nc) in _legal_moves(board, me, opp):
        if not _wins(_apply(board, r, c, nr, nc, me), opp, me, tt, budget, deadline):
            result = True  # opponent cannot win after this move -> it is a winning move
            break
    tt[key] = result
    return result


def _mobility(board: "tuple[str, ...]", me: str, opp: str) -> int:
    """Midgame leaf heuristic: my legal moves minus the opponent's. Normal-play Clobber
    is lost by the side that runs OUT of moves, so keeping more moves than the opponent
    (and starving them) is the whole game — the same signal the greedy heuristic uses,
    now looked at several plies deep instead of one."""
    return len(_legal_moves(board, me, opp)) - len(_legal_moves(board, opp, me))


def _parse_depth_ladder(spec: str) -> "tuple[tuple[int, int], ...]":
    """Parse a 'pieces:depth,...' ladder into an ascending tuple; malformed parts are skipped so a
    bad CLOBBER_MOBILITY_LADDER override can never crash trajectory generation."""
    pairs: "list[tuple[int, int]]" = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            p_str, d_str = entry.split(":")
            pairs.append((int(p_str), int(d_str)))
        except Exception:
            continue
    return tuple(sorted(pairs))


# Mobility-mode depth ladder — TUNED FOR OUR ENGINE (NOT the winner's 12:10,16:8,22:6). The idea is
# the winner's (deeper as the board empties, shallow opening), but the numbers exploit an edge the
# winner's exact-solver-less engine LACKS: our EXACT solver fires at <= _SOLVE_MAX_PIECES (16) inside
# _negamax, so a negamax of depth (pieces-16) from a P-piece board bottoms out entirely in EXACTLY-
# solved leaves = a PERFECT solve of that position. So we set depth ~= pieces-16 to exact-solve the
# whole game up to ~24 pieces (the mid+endgame on every board size), and leave the dense 25-30-piece
# opening at the fast _MIDGAME_DEPTH (deep opening search is low-ROI and starves the time-boxed gen).
# "pieces:depth,..." ascending; first threshold >= pieces wins, else _MIDGAME_DEPTH. Env-tunable
# (CLOBBER_MOBILITY_LADDER) so the VM A/B can sweep exact-solve depth vs generation throughput.
_DEPTH_LADDER = _parse_depth_ladder(os.getenv("CLOBBER_MOBILITY_LADDER", "20:4,22:6,24:8"))
# Root moves within this negamax margin of the best score are treated as TIES and broken
# SEMANTICALLY (most-reduces-opponent-mobility, then board geometry) instead of by smallest action
# id — the winner's fix for the unlearnable top-left label bias that collapses the distilled student.
_TIE_MARGIN = float(os.getenv("CLOBBER_TIE_MARGIN", "0.5"))


def _mobility_depth(pieces: int) -> int:
    """Mobility-negamax search depth at this piece count (the winner's depth-by-piece-count).
    First ladder threshold >= pieces wins; above the last threshold use the flat _MIDGAME_DEPTH."""
    for p, d in _DEPTH_LADDER:
        if pieces <= p:
            return d
    return _MIDGAME_DEPTH


# mobility_PURE ladder: the winner's OWN depth-by-piece-count (12:10,16:8,22:6 — deeper as the board
# empties) for the no-exact-solver mode, where each number is a genuine SEARCH depth (not our "reach
# the exact leaves" tuning). Env-tunable via CLOBBER_MOBILITY_PURE_LADDER.
_DEPTH_LADDER_PURE = _parse_depth_ladder(os.getenv("CLOBBER_MOBILITY_PURE_LADDER", "12:10,16:8,22:6"))


def _mobility_pure_depth(pieces: int) -> int:
    for p, d in _DEPTH_LADDER_PURE:
        if pieces <= p:
            return d
    return _MIDGAME_DEPTH


def _negamax(board: "tuple[str, ...]", me: str, opp: str, depth: int,
             alpha: float, beta: float, budget: "list[int]", deadline: float,
             tt_exact: dict) -> float:
    """Negamax (`me`-to-move perspective) with fail-soft alpha-beta. Any node small enough
    is resolved EXACTLY via `_wins` (±_WIN_SCORE); otherwise recurse to `depth`, then fall
    back to the mobility heuristic. Raises _BudgetExceeded when the node/time cap is hit.

    No transposition table here on purpose: an alpha-beta score can be a bound (not the
    exact value) when a cutoff fires, so caching it under (board, depth) and reusing it as
    exact would silently corrupt the midgame eval. Depth is shallow and pruning is enough,
    so we skip the TT and stay correct. (The exact `_wins` solver DOES memoize — its AND/OR
    verdict is window-independent, so that cache is sound.)"""
    budget[0] -= 1
    if budget[0] <= 0 or time.monotonic() > deadline:
        raise _BudgetExceeded
    if _piece_count(board) <= _SOLVE_MAX_PIECES:
        return _WIN_SCORE if _wins(board, me, opp, tt_exact, budget, deadline) else -_WIN_SCORE
    moves = _legal_moves(board, me, opp)
    if not moves:
        return -_WIN_SCORE  # no legal move -> the side to move LOSES (normal play)
    if depth <= 0:
        return float(_mobility(board, me, opp))
    best = -_INF
    for (r, c, nr, nc) in moves:
        score = -_negamax(_apply(board, r, c, nr, nc, me), opp, me, depth - 1,
                          -beta, -alpha, budget, deadline, tt_exact)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # this line is already too good for the opponent to allow
    return best


def _negamax_mobility(board: "tuple[str, ...]", me: str, opp: str, depth: int,
                      alpha: float, beta: float, budget: "list[int]", deadline: float,
                      tt_exact: dict) -> float:
    """Like `_negamax` but with NO exact-solve short-circuit — leaves are ALWAYS the smooth mobility
    gradient (my_legal − opp_legal), never a ±_WIN_SCORE cliff. The winner-teardown (2026-08-29)
    showed the ±cliff / exact-terminal label is UNLEARNABLE for the SFT student (it reverts to the
    base prior and forfeits = the 0-10 loss the winner's ablation pins); a continuous gradient distils.
    Only a genuine no-legal-move node is a real terminal (a loss). `tt_exact` is unused (signature
    parity for the shared root loop)."""
    budget[0] -= 1
    if budget[0] <= 0 or time.monotonic() > deadline:
        raise _BudgetExceeded
    moves = _legal_moves(board, me, opp)
    if not moves:
        return -_WIN_SCORE  # no legal move -> the side to move LOSES (a genuine terminal, not a cliff)
    if depth <= 0:
        return float(_mobility(board, me, opp))
    best = -_INF
    for (r, c, nr, nc) in moves:
        score = -_negamax_mobility(_apply(board, r, c, nr, nc, me), opp, me, depth - 1,
                                   -beta, -alpha, budget, deadline, tt_exact)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _rollout_once(board: "tuple[str, ...]", me: str, opp: str, rng) -> float:
    """One uniform-random playout from `board` with `me` to move. Normal-play Clobber:
    the side to move with NO legal move loses. Returns +1 if `me` wins the playout, -1 if
    `me` loses. Bounded — each ply deletes a stone, so it ends in <= piece_count plies."""
    cur, other, me_to_move = me, opp, True
    b = board
    while True:
        moves = _legal_moves(b, cur, other)
        if not moves:
            return -1.0 if me_to_move else 1.0  # `cur` (to move) is stuck and loses
        r, c, nr, nc = moves[rng.randrange(len(moves))]
        b = _apply(b, r, c, nr, nc, cur)
        cur, other, me_to_move = other, cur, not me_to_move


def _rollout_value(board: "tuple[str, ...]", me: str, opp: str, rng, n: int) -> float:
    """Mean terminal outcome in [-1, 1] for `me` over `n` random playouts (parity-aware —
    it plays to the real end, unlike the systematically-wrong mobility heuristic)."""
    if n <= 0:
        return 0.0
    return sum(_rollout_once(board, me, opp, rng) for _ in range(n)) / n


def _choose_by_rollout(board: "tuple[str, ...]", grid_rows: int, labels: dict,
                       legal_action_ids: "list[int]", me: str, opp: str, rng,
                       budget: "list[int]", deadline: float, tt_exact: dict) -> "int | None":
    """Flat Monte-Carlo root policy: score each legal move by the EXACT solver when its
    child is small enough, else by the mean of `_ROLLOUTS` random playouts. Returns the
    best action id, or None if nothing was scorable (caller falls back to greedy)."""
    best_aid, best_val = None, -_INF
    for aid in legal_action_ids:
        if best_aid is not None and time.monotonic() > deadline:
            break  # out of wall-clock budget -> keep the best move scored so far
        label = labels.get(aid)
        if label is None:
            continue
        try:
            (r, c), (nr, nc) = _label_to_indices(label, grid_rows)
        except Exception:
            continue
        child = _apply(board, r, c, nr, nc, me)  # opponent to move at `child`
        if _piece_count(child) <= _SOLVE_MAX_PIECES:
            try:  # I win iff the opponent cannot force a win from `child`.
                val = _WIN_SCORE if not _wins(child, opp, me, tt_exact, budget, deadline) else -_WIN_SCORE
            except _BudgetExceeded:
                val = -_rollout_value(child, opp, me, rng, _ROLLOUTS)  # solver too slow -> estimate
        else:
            val = -_rollout_value(child, opp, me, rng, _ROLLOUTS)  # child is opp-to-move -> negate
        if val > best_val:
            best_val, best_aid = val, aid
    return best_aid


class _MCTSNode:
    """UCT node. `value` accumulates reward from the ROOT player's perspective (single-
    perspective MCTS), so a node where the opponent is to move MINIMIZES it during selection."""
    __slots__ = ("board", "player", "opp", "move", "children", "untried", "visits", "value")

    def __init__(self, board, player, opp, move):
        self.board = board
        self.player = player          # side to move at this node
        self.opp = opp
        self.move = move              # the (r,c,nr,nc) move (from the parent) that led here
        self.children: list = []
        self.untried = _legal_moves(board, player, opp)
        self.visits = 0
        self.value = 0.0              # cumulative ROOT-perspective reward


def _mcts_leaf_value(board, player, opp, root_player, rng, budget, deadline, tt_exact) -> float:
    """Leaf value in {-1,+1} from ROOT's perspective: EXACT when the node is small enough
    (our stand-in for the CGT-value database the literature uses), else one random playout.
    Draw-free normal play, so both are well-defined."""
    if not _legal_moves(board, player, opp):
        return 1.0 if player != root_player else -1.0          # `player` is stuck -> loses
    if _piece_count(board) <= _SOLVE_MAX_PIECES:
        try:
            player_wins = _wins(board, player, opp, tt_exact, budget, deadline)
        except _BudgetExceeded:
            player_wins = _rollout_once(board, player, opp, rng) > 0.0
    else:
        player_wins = _rollout_once(board, player, opp, rng) > 0.0
    root_wins = player_wins if player == root_player else not player_wins
    return 1.0 if root_wins else -1.0


def _uct_child(node, root_player):
    """UCB1 select. Root player MAXIMIZES root-reward; the opponent MINIMIZES it (sign flip)."""
    sign = 1.0 if node.player == root_player else -1.0
    log_n = math.log(node.visits)
    best, best_score = None, -_INF
    for c in node.children:
        score = sign * (c.value / c.visits) + _MCTS_C * math.sqrt(log_n / c.visits)
        if score > best_score:
            best_score, best = score, c
    return best


def _mcts_choose(board, grid_rows, labels, legal_action_ids, me, opp, rng,
                 budget, deadline, tt_exact) -> "int | None":
    """UCT search rooted at `board` (me to move); returns the most-visited root move's action
    id (mapped via `labels`), or None if nothing was searchable (caller falls back to greedy)."""
    root = _MCTSNode(board, me, opp, None)
    if not root.untried:
        return None
    while root.visits < _MCTS_SIMS:
        if time.monotonic() > deadline or budget[0] <= 0:
            break
        budget[0] -= 1
        node, path = root, [root]
        while not node.untried and node.children:                     # SELECT
            node = _uct_child(node, me)
            path.append(node)
        if node.untried:                                              # EXPAND
            r, c, nr, nc = node.untried.pop()
            child = _MCTSNode(_apply(node.board, r, c, nr, nc, node.player),
                              node.opp, node.player, (r, c, nr, nc))
            node.children.append(child)
            node = child
            path.append(child)
        result = _mcts_leaf_value(node.board, node.player, node.opp, me,   # SIMULATE
                                  rng, budget, deadline, tt_exact)
        for n in path:                                                # BACKPROP (single-perspective)
            n.visits += 1
            n.value += result
    best = max(root.children, key=lambda c: c.visits, default=None)
    if best is None or best.move is None:
        return None
    tgt = best.move
    for aid in legal_action_ids:                                     # map (r,c,nr,nc) -> action id
        label = labels.get(aid)
        if label is None:
            continue
        try:
            (r, c), (nr, nc) = _label_to_indices(label, grid_rows)
        except Exception:
            continue
        if (r, c, nr, nc) == tgt:
            return aid
    return None


def choose_action(observation: str, player_id: int, legal_action_ids: "list[int]", rng=None) -> int:
    """Pick a Clobber move. DEFAULT (CLOBBER_LEAF_EVAL=greedy) is the previously-measured-best
    teacher: 1-ply greedy midgame + EXACT solver for the <=_SOLVE_MAX_PIECES endgame. Opt-in
    modes via CLOBBER_LEAF_EVAL:
      * 'mcts'     — NEW CANDIDATE (post-R1-loss): UCT search + exact-solver leaves, grounded
                     in the Clobber literature (search backed by exact endgame values, the CGT
                     approach). The intended replacement once a head-to-head vs the R1 WINNER'S
                     model confirms it — verify against a STRONG EXTERNAL baseline, not our own
                     greedy (which gave false confidence and we placed LAST in R1).
      * 'mobility' — UPGRADED to the winner's recipe: depth-by-piece-count negamax + SEMANTIC
                     tie-break + our exact endgame. Re-measure as a distilled STUDENT vs the eval
                     MCTS (the old flat-4/smallest-id 45% was a misleading teacher-vs-greedy number).
      * 'rollout'  — flat Monte-Carlo (measured ~35% — random clobber play is too noisy).
    Never raises: any parse/budget error falls back to greedy."""
    try:
        grid, grid_rows, _ = _parse_board(observation)
        board = _to_board(grid)
        labels = _extract_action_labels(observation)
        # OpenSpiel clobber: player 0 moves the "o" stones (validator ClobberAgent: P0 = "o (White)").
        me = "o" if player_id == 0 else "x"
        opp = "x" if player_id == 0 else "o"
        budget = [_NODE_BUDGET]
        deadline = time.monotonic() + _MOVE_TIME_BUDGET
        tt_exact: dict = {}
        _set_reason(f"leaf_eval={_LEAF_EVAL}")  # RANK-9 debug: refined below for the deep-mobility path

        if _LEAF_EVAL == "rollout":
            aid = _choose_by_rollout(board, grid_rows, labels, legal_action_ids,
                                     me, opp, rng if rng is not None else random,
                                     budget, deadline, tt_exact)
            return aid if aid is not None else _greedy_choose(observation, player_id, legal_action_ids, rng)

        if _LEAF_EVAL == "mcts":
            aid = _mcts_choose(board, grid_rows, labels, legal_action_ids, me, opp,
                               rng if rng is not None else random, budget, deadline, tt_exact)
            return aid if aid is not None else _greedy_choose(observation, player_id, legal_action_ids, rng)

        if _LEAF_EVAL in ("mobility", "mobility_pure"):
            # Winner-recipe iterative-deepening mobility negamax. TWO fixes over the old flat-depth-4
            # variant, both cited by the open-sourced R1 winner as what turned a 0-10 distilled
            # student into a winning one:
            #   (1) DEPTH BY PIECE COUNT (_mobility_depth) — deeper as the board empties, shallow in
            #       the dense opening where deep search just starves the time-boxed generator.
            #   (2) SEMANTIC deterministic tie-break — among moves within _TIE_MARGIN of the best
            #       score, pick the one that most reduces the opponent's mobility, then board
            #       geometry. The old strict-'>' loop kept the FIRST (smallest-id) move on a tie,
            #       baking an unlearnable top-left tell that collapses the student to the base prior.
            # Each root move is searched with a FULL window (no cross-sibling alpha narrowing) so
            # every score is exact and comparable — required to detect ties accurately for (2). Our
            # exact endgame solver (<=_SOLVE_MAX_PIECES, inside _negamax) still plays the late game
            # perfectly — an edge the winner's pure-negamax teacher lacks.
            # mobility_pure (winner-teardown): NO exact-solver short-circuit, so the label is a SMOOTH
            # mobility gradient the SFT student can distil (the exact ±cliff is what makes our current
            # clobber label unlearnable = the mythical "46% ceiling"). Uses the winner's deeper ladder.
            _nmax = _negamax_mobility if _LEAF_EVAL == "mobility_pure" else _negamax
            _depth_of = _mobility_pure_depth if _LEAF_EVAL == "mobility_pure" else _mobility_depth
            root_moves: "list[tuple[int, tuple[str, ...], tuple[int, int, int, int]]]" = []
            for aid in legal_action_ids:
                label = labels.get(aid)
                if label is None:
                    continue
                try:
                    (r, c), (nr, nc) = _label_to_indices(label, grid_rows)
                except Exception:
                    continue
                root_moves.append((aid, _apply(board, r, c, nr, nc, me), (r, c, nr, nc)))
            scores: "dict[int, float]" = {}
            for depth in range(1, _depth_of(_piece_count(board)) + 1):
                try:
                    depth_scores = {
                        aid: -_nmax(child, opp, me, depth - 1, -_INF, _INF,
                                    budget, deadline, tt_exact)
                        for aid, child, _mv in root_moves
                    }
                    scores = depth_scores         # this depth completed fully -> trust it
                    if scores and max(scores.values()) >= _WIN_SCORE:
                        break                     # a forced win is proven; deeper search is moot
                except _BudgetExceeded:
                    break                         # time/nodes out -> keep the last full depth's scores
            best_aid = None
            if scores:
                best_val = max(scores.values())
                child_by_aid = {aid: child for aid, child, _mv in root_moves}
                move_by_aid = {aid: mv for aid, _child, mv in root_moves}
                tied = [aid for aid, s in scores.items() if best_val - s <= _TIE_MARGIN]
                best_aid = min(tied, key=lambda a: (len(_legal_moves(child_by_aid[a], opp, me)),
                                                    move_by_aid[a]))
            if best_aid is not None:
                _set_reason(f"deep-mobility: pieces={_piece_count(board)} "
                            f"depth<={_depth_of(_piece_count(board))} best_aid={best_aid} "
                            f"score={scores.get(best_aid)}")
                return best_aid
            _set_reason("deep-mobility: no scorable root move -> greedy fallback")
            return _greedy_choose(observation, player_id, legal_action_ids, rng)

        # DEFAULT "greedy": measured-best — 1-ply greedy midgame + EXACT endgame. When the
        # board is small enough, return a PROVEN winning move (I win iff the opponent cannot
        # force a win after it); otherwise defer to the greedy heuristic. This is the pre-
        # negamax R2 logic, restored after deep-search variants lost the head-to-head to it.
        if _piece_count(board) <= _SOLVE_MAX_PIECES:
            try:
                for aid in legal_action_ids:
                    label = labels.get(aid)
                    if label is None:
                        continue
                    (r, c), (nr, nc) = _label_to_indices(label, grid_rows)
                    if not _wins(_apply(board, r, c, nr, nc, me), opp, me, tt_exact, budget, deadline):
                        return aid
            except _BudgetExceeded:
                pass  # exact solve too slow -> fall through to the greedy heuristic
        return _greedy_choose(observation, player_id, legal_action_ids, rng)
    except Exception:
        return _greedy_choose(observation, player_id, legal_action_ids, rng)
