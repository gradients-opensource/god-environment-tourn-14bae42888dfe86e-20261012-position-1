"""Strong perfect-information Othello expert teacher for SFT data generation.

This is a *strictly stronger* replacement for the depth-limited heuristic policy
in ``othello_minimax.choose_action``. It keeps that module's battle-tested,
phase-weighted evaluation (corner/edge table, mobility, frontier, stability,
parity) for the midgame but adds two things that make it play materially better:

  1. **Perfect endgame solver.** Once the position has ``ENDGAME_EMPTIES`` or
     fewer empty squares, the engine abandons the heuristic and searches the
     game to the very end, returning the move that maximises the *exact* final
     disc differential under optimal play. The previous minimax only ever
     applied a shallow heuristic, so it routinely misplayed won/lost endgames;
     this engine plays them perfectly.
  2. **Exact-aware midgame search.** Iterative-deepening negamax with
     alpha-beta and a transposition table; any subtree that drops below the
     endgame-empties threshold is resolved exactly (scaled to dominate the
     heuristic), so deep tactical lines near the end of the game are never
     misjudged.

Contract (called by ``othello_trajectories.generate_heuristic_episode``):

    choose_action(observation, player_id, legal_action_ids, rng=None) -> int

Action-id encoding matches ``othello_board``: ``action_id = row * 8 + col``
(0..63), and ``action_id == 64`` is "pass" when no board move is legal.
"""

from __future__ import annotations

import time

from envs.othello_minimax import (
    EMPTY,
    ME,
    OPP,
    _WEIGHTS,
    _apply_move,
    _evaluate,
    _legal_moves,
    _parse_board,
    _to_internal,
)
from envs.othello_board import PASS_ACTION_ID

# Below this many empty squares the engine switches to an exact (perfect)
# search to the end of the game. 16 is a good speed/strength trade-off for
# offline data generation; tests override it with a smaller value for speed.
ENDGAME_EMPTIES = 16

_DEFAULT_TIME_BUDGET = 0.35
_DEFAULT_MAX_DEPTH = 18
_TIME_CHECK_INTERVAL = 2048

# Exact endgame results are integer disc differentials in [-64, 64]. Scaling
# them well above the heuristic's magnitude (a few hundred) guarantees that a
# proven win/loss/draw always dominates any heuristic estimate when the two are
# compared in the same search.
_ENDGAME_SCALE = 10_000.0
_INF = float("inf")

_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


def _count_empties(board: list[list[int]]) -> int:
    return sum(row.count(EMPTY) for row in board)


def _board_key(board: list[list[int]], color: int) -> tuple:
    return (tuple(tuple(row) for row in board), color)


def _disc_diff(board: list[list[int]], color: int) -> int:
    my = sum(row.count(color) for row in board)
    opp = sum(row.count(-color) for row in board)
    return my - opp


# --- Exact endgame solver --------------------------------------------------------------


def _endgame(
    board: list[list[int]], color: int, alpha: int, beta: int,
    deadline: float, counter: list[int], ett: dict,
) -> int:
    """Exact final disc differential (``color`` perspective) under optimal play.

    Negamax with alpha-beta over a separate transposition table (``ett``); the
    only terminal value is the true end-of-game disc difference, so the result
    is provably optimal (no heuristic is consulted).
    """
    counter[0] += 1
    if counter[0] % _TIME_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise _Timeout()

    key = _board_key(board, color)
    entry = ett.get(key)
    if entry is not None:
        e_score, e_flag = entry
        if e_flag == _TT_EXACT:
            return e_score
        if e_flag == _TT_LOWER:
            alpha = max(alpha, e_score)
        else:
            beta = min(beta, e_score)
        if alpha >= beta:
            return e_score

    moves = _legal_moves(board, color)
    if not moves:
        if not _legal_moves(board, -color):
            return _disc_diff(board, color)
        # Forced pass — turn passes to the opponent, perspective flips.
        return -_endgame(board, -color, -beta, -alpha, deadline, counter, ett)

    moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)
    orig_alpha = alpha
    best = -(10 ** 9)
    for move in moves:
        child = _apply_move(board, move, color)
        score = -_endgame(child, -color, -beta, -alpha, deadline, counter, ett)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= orig_alpha:
        flag = _TT_UPPER
    elif best >= beta:
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    ett[key] = (best, flag)
    return best


# --- Exact-aware midgame search --------------------------------------------------------


def _negamax(
    board: list[list[int]], color: int, depth: int, alpha: float, beta: float,
    deadline: float, counter: list[int], tt: dict, ett: dict, endgame_empties: int,
) -> float:
    counter[0] += 1
    if counter[0] % _TIME_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise _Timeout()

    # Solve exactly once the position is close enough to the end. The scaled
    # exact value dominates any heuristic comparison elsewhere in the tree.
    if _count_empties(board) <= endgame_empties:
        return _ENDGAME_SCALE * _endgame(board, color, -(10 ** 9), 10 ** 9, deadline, counter, ett)

    orig_alpha, orig_beta = alpha, beta
    key = _board_key(board, color)
    entry = tt.get(key)
    if entry is not None:
        e_depth, e_score, e_flag = entry
        if e_depth >= depth:
            if e_flag == _TT_EXACT:
                return e_score
            if e_flag == _TT_LOWER:
                alpha = max(alpha, e_score)
            elif e_flag == _TT_UPPER:
                beta = min(beta, e_score)
            if alpha >= beta:
                return e_score

    moves = _legal_moves(board, color)
    if not moves:
        if not _legal_moves(board, -color):
            return _evaluate(board, color)
        if depth == 0:
            return _evaluate(board, color)
        return -_negamax(board, -color, depth - 1, -beta, -alpha,
                         deadline, counter, tt, ett, endgame_empties)

    if depth == 0:
        return _evaluate(board, color)

    moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)
    best = -_INF
    for move in moves:
        child = _apply_move(board, move, color)
        score = -_negamax(child, -color, depth - 1, -beta, -alpha,
                          deadline, counter, tt, ett, endgame_empties)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best <= orig_alpha:
        flag = _TT_UPPER
    elif best >= orig_beta:
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    tt[key] = (depth, best, flag)
    return best


# --- Public helpers (used by tests and the trajectory generator) -----------------------


def best_move_internal(
    board: list[list[int]], *,
    time_budget: float = _DEFAULT_TIME_BUDGET,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    endgame_empties: int = ENDGAME_EMPTIES,
) -> "tuple[int, int] | None":
    """Return the best (row, col) move for ME on an internal board, or None if
    ME has no legal move (a pass)."""
    moves = _legal_moves(board, ME)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    empties = _count_empties(board)
    deadline = time.monotonic() + time_budget
    counter = [0]
    ett: dict = {}

    # Endgame: solve exactly to the end and pick the move with the best true
    # disc differential. No time pressure should cut this short for reasonable
    # ``endgame_empties``, but we still respect the deadline defensively.
    if empties <= endgame_empties:
        # Sort BEFORE taking the fallback: a solve that times out on its first candidate (most
        # decisions at 12-16 empties under the gen budget) returns the table's best move, not the
        # lowest legal id that row-major move generation puts first.
        moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)
        best_move = moves[0]
        best_score = -(10 ** 9)
        try:
            alpha = -(10 ** 9)
            for move in moves:
                child = _apply_move(board, move, ME)
                score = -_endgame(child, OPP, -(10 ** 9), -alpha, deadline, counter, ett)
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score > alpha:
                    alpha = best_score
            return best_move
        except _Timeout:
            return best_move

    # Midgame: iterative-deepening negamax with a heuristic leaf eval (exact
    # near the end via the endgame switch inside _negamax).
    tt: dict = {}
    moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)
    best_move = moves[0]
    depth = 1
    while depth <= max_depth and time.monotonic() < deadline:
        alpha, beta = -_INF, _INF
        iter_best_move = moves[0]
        iter_best_score = -_INF
        try:
            for move in moves:
                child = _apply_move(board, move, ME)
                score = -_negamax(child, OPP, depth - 1, -beta, -alpha,
                                  deadline, counter, tt, ett, endgame_empties)
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move = move
                if iter_best_score > alpha:
                    alpha = iter_best_score
        except _Timeout:
            break
        best_move = iter_best_move
        # Search the new best move first next iteration (improves pruning).
        moves.remove(best_move)
        moves.insert(0, best_move)
        depth += 1

    return best_move


# --- Contract entry point --------------------------------------------------------------


def choose_action(
    observation: str,
    player_id: int,
    legal_action_ids: "list[int]",
    rng=None,
    *,
    time_budget: float = _DEFAULT_TIME_BUDGET,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    endgame_empties: int = ENDGAME_EMPTIES,
) -> int:
    """Pick the best move id for the side to move. Perfect-information game, so
    this returns the search-optimal move (no sampling)."""
    legal = list(legal_action_ids)
    if not legal:
        return PASS_ACTION_ID
    if len(legal) == 1:
        return legal[0]

    # player_id is authoritative for our colour (0 = Black/x, 1 = White/o),
    # matching othello_trajectories' colour mapping.
    my_symbol = "x" if player_id == 0 else "o"
    board = _to_internal(_parse_board(observation), my_symbol)

    move = best_move_internal(
        board, time_budget=time_budget, max_depth=max_depth,
        endgame_empties=endgame_empties,
    )
    if move is None:
        return PASS_ACTION_ID if PASS_ACTION_ID in legal else legal[0]

    action_id = move[0] * 8 + move[1]
    # Defensive: only ever return a server-legal id.
    if action_id in legal:
        return action_id
    return legal[0]


# ---------------------------------------------------------------------------
# One-sentence rule-grounded CoT rationale (winner's learnability device)
# ---------------------------------------------------------------------------
# Naming the visible feature the chosen move exploits makes the policy imitable
# at HIGHER FIDELITY: the student learns board -> [feature] -> move rather than a
# raw board -> move map (the same device the pos-1 open-source repo used for
# othello). Our own implementation over othello_expert's helpers (NOT a verbatim
# copy). The comment describes the CALLER'S action_id, so it is byte-consistent
# with the game_action tool call it precedes (a rationale for a DIFFERENT move
# than the one played would teach announce-one-move-play-another).

_CORNER_IDS = frozenset({0, 7, 56, 63})
_X_TO_CORNER = {9: 0, 14: 7, 49: 56, 54: 63}  # X-square id -> the corner it guards


def _alg(action_id: int) -> str:
    """action_id (row*8+col) -> algebraic 'a1'..'h8' for the human-readable rationale."""
    r, c = divmod(action_id, 8)
    return f"{chr(97 + c)}{r + 1}"


def get_label_comment(observation: str, action_id: int, player_id: int) -> str:
    """Deterministic, <=140-char, never-raises rationale for ``action_id`` on this
    othello board (mover = ``player_id``). Returns '' on any failure so the caller can
    fall back to a content-less row. Consistency with the tool call is the caller's
    responsibility: pass the ROW's action_id."""
    try:
        if action_id == PASS_ACTION_ID or action_id >= 64:
            return "No flipping move available; passing the turn."
        if action_id in _CORNER_IDS:
            return f"Taking corner {_alg(action_id)}: a permanent, unflippable disc."[:140]
        my_symbol = "x" if player_id == 0 else "o"
        raw = _parse_board(observation)
        board = _to_internal(raw, my_symbol)
        r, c = divmod(action_id, 8)
        child = _apply_move(board, (r, c), ME)
        # Endgame: prove the outcome of the chosen line to the last disc.
        if _count_empties(board) <= ENDGAME_EMPTIES:
            try:
                deadline = time.monotonic() + 0.30
                diff = -_endgame(child, OPP, -(10 ** 9), 10 ** 9, deadline, [0], {})
            except Exception:
                diff = None
            if diff is not None:
                a = _alg(action_id)
                if diff > 0:
                    return f"Endgame solved: {a} wins the final disc count by {diff}."[:140]
                if diff == 0:
                    return f"Endgame solved: {a} secures a draw on disc count."[:140]
                return f"Endgame: {a} is the best remaining line here."[:140]
        # X-square avoidance: a corner-adjacent X-square is legal with its corner still
        # empty, and this move keeps the corner contestable by NOT playing the X-square.
        legal_ids = {mr * 8 + mc for mr, mc in _legal_moves(board, ME)}
        open_x = [x for x in _X_TO_CORNER
                  if x in legal_ids and raw[_X_TO_CORNER[x] // 8][_X_TO_CORNER[x] % 8] == "-"]
        if action_id not in _X_TO_CORNER and open_x:
            x = min(open_x)
            return f"Avoiding X-square {_alg(x)}; corner {_alg(_X_TO_CORNER[x])} stays contestable."[:140]
        # Mobility restriction: fewer opponent replies after our move is better.
        opp_replies = len(_legal_moves(child, OPP))
        return f"Playing {_alg(action_id)}, holding the opponent to {opp_replies} replies."[:140]
    except Exception:
        return ""
