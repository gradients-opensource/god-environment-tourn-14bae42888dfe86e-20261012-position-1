"""Othello alpha-beta search strategy for SFT trajectory generation.

Ported from G.O.D's scripts/mcts_strategies/othello_minimax.py: parses the
ASCII board out of `state_desc`, then runs iterative-deepening negamax with
alpha-beta pruning over a phase-weighted evaluation function combining (per
othello_bot_research.md): a corner/edge/X-square piece-square table, actual
mobility, frontier (potential-mobility) exposure, stable-disc count, and coin
parity (weighted in only as the board fills up).

`time_budget`/`max_depth` are exposed as parameters (rather than fixed module
constants) so the trajectory generator can vary them per game for a mix of
"compute more, smarter move" vs "compute less, quicker but dumber move".
"""

import os
import time

ME, OPP, EMPTY = 1, -1, 0

# Default piece-square table — VALIDATED (this teacher scores 58% vs the othello champ;
# corner:center ratio is already 100:1, a sharp anchor).
_WEIGHTS_DEFAULT = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [5, -2, 1, 1, 1, 1, -2, 5],
    [5, -2, 1, 1, 1, 1, -2, 5],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

# Corner-ANCHOR table — the runner-up's "corner-anchor" teacher IDEA (SFT_OTHELLO_ANCHOR),
# ported in OUR magnitude, NOT their raw integers (corner364744/ring45593): those ~4-orders-larger
# values would swamp mobility/parity/stability AND the perfect-endgame solver's ×10000 scale,
# breaking eval balance. This variant just SHARPENS the corner/edge emphasis (corner 100→160,
# edge 10→14) and the deadly X/C-square penalties (X -50→-70, C -20→-30) in the same range.
# OFF BY DEFAULT (flag unset) so the validated teacher is byte-identical. Enable ONLY after a VM
# A/B (verify_othello_teacher.py anchor-vs-default AND verify_teacher_vs_mcts.py --mcts-sims 25,
# the real eval opponent) confirms it wins more — the champion did NOT use an anchor table,
# so this is a lower-confidence lever than the othello DATA FLOOD.
_WEIGHTS_ANCHOR = [
    [160, -30, 14, 6, 6, 14, -30, 160],
    [-30, -70, -3, -3, -3, -3, -70, -30],
    [14, -3, 6, 1, 1, 6, -3, 14],
    [6, -3, 1, 1, 1, 1, -3, 6],
    [6, -3, 1, 1, 1, 1, -3, 6],
    [14, -3, 6, 1, 1, 6, -3, 14],
    [-30, -70, -3, -3, -3, -3, -70, -30],
    [160, -30, 14, 6, 6, 14, -30, 160],
]

# Bind at module import (BEFORE othello_expert imports _WEIGHTS by name) so the flag takes effect.
# Explicit "1" check (not bare truthiness) so SFT_OTHELLO_ANCHOR=0 / =false does NOT enable it.
_WEIGHTS = _WEIGHTS_ANCHOR if os.getenv("SFT_OTHELLO_ANCHOR") == "1" else _WEIGHTS_DEFAULT

_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

_CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# Defaults, mirrors the originating module. Trajectory generator samples
# per-game values around these.
_TIME_BUDGET = 0.15
_MAX_DEPTH = 14
_TIME_CHECK_INTERVAL = 2048

_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


# --- Board parsing -------------------------------------------------------------------


def _parse_board(state_desc: str) -> list[list[str]]:
    """8x8 grid of '-'/'x'/'o' parsed from the rendered board rows."""
    board = [["-"] * 8 for _ in range(8)]
    for line in state_desc.splitlines():
        tokens = line.split()
        if len(tokens) == 10 and tokens[0] == tokens[-1] and tokens[0].isdigit():
            row = int(tokens[0]) - 1
            if 0 <= row < 8:
                board[row] = tokens[1:9]
    return board


def _my_symbol(state_desc: str) -> str:
    if "Black (x) to play" in state_desc:
        return "x"
    if "White (o) to play" in state_desc:
        return "o"
    return "x"


def _to_internal(raw_board: list[list[str]], my_symbol: str) -> list[list[int]]:
    opp_symbol = "o" if my_symbol == "x" else "x"
    board = [[EMPTY] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            cell = raw_board[r][c]
            if cell == my_symbol:
                board[r][c] = ME
            elif cell == opp_symbol:
                board[r][c] = OPP
    return board


# --- Move generation ------------------------------------------------------------------


def _flips(board: list[list[int]], r: int, c: int, color: int) -> list[tuple[int, int]]:
    if board[r][c] != EMPTY:
        return []
    captured: list[tuple[int, int]] = []
    for dr, dc in _DIRECTIONS:
        line: list[tuple[int, int]] = []
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == -color:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == color:
            captured.extend(line)
    return captured


def _legal_moves(board: list[list[int]], color: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(8)
        for c in range(8)
        if board[r][c] == EMPTY and _flips(board, r, c, color)
    ]


def _apply_move(board: list[list[int]], move: tuple[int, int], color: int) -> list[list[int]]:
    new_board = [row[:] for row in board]
    r, c = move
    new_board[r][c] = color
    for fr, fc in _flips(board, r, c, color):
        new_board[fr][fc] = color
    return new_board


# --- Evaluation ------------------------------------------------------------------------


def _frontier_diff(board: list[list[int]], color: int) -> int:
    """opponent_frontier - my_frontier (we prefer to minimize our own frontier)."""
    my_frontier = 0
    opp_frontier = 0
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == EMPTY:
                continue
            for dr, dc in _DIRECTIONS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == EMPTY:
                    if piece == color:
                        my_frontier += 1
                    else:
                        opp_frontier += 1
                    break
    return opp_frontier - my_frontier


def _stable_mask(board: list[list[int]]) -> list[list[bool]]:
    """Discs that can never be flipped for the rest of the game.

    Covers: occupied corners, same-color edge runs anchored to an owned
    corner (classic edge stability), and discs whose row/column/both
    diagonals are completely full (no empty cell remains on any of their
    four lines, so nothing can ever flip them).
    """
    stable = [[False] * 8 for _ in range(8)]

    row_full = [EMPTY not in board[r] for r in range(8)]
    col_full = [all(board[r][c] != EMPTY for r in range(8)) for c in range(8)]
    diag1_full = {k: all(board[r][r - k] != EMPTY for r in range(max(0, k), min(8, 8 + k))) for k in range(-7, 8)}
    diag2_full = {k: all(board[r][k - r] != EMPTY for r in range(max(0, k - 7), min(8, k + 1))) for k in range(0, 15)}

    for r in range(8):
        for c in range(8):
            if board[r][c] == EMPTY:
                continue
            if row_full[r] and col_full[c] and diag1_full[r - c] and diag2_full[r + c]:
                stable[r][c] = True

    for r, c in _CORNERS:
        if board[r][c] != EMPTY:
            stable[r][c] = True

    edges = [
        ((0, 0), (0, 1)), ((0, 7), (0, -1)),
        ((7, 0), (0, 1)), ((7, 7), (0, -1)),
        ((0, 0), (1, 0)), ((7, 0), (-1, 0)),
        ((0, 7), (1, 0)), ((7, 7), (-1, 0)),
    ]
    for (cr, cc), (dr, dc) in edges:
        if board[cr][cc] == EMPTY:
            continue
        color = board[cr][cc]
        r, c = cr + dr, cc + dc
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == color:
            stable[r][c] = True
            r += dr
            c += dc

    return stable


def _stability_diff(board: list[list[int]], color: int) -> int:
    stable = _stable_mask(board)
    my = sum(1 for r in range(8) for c in range(8) if stable[r][c] and board[r][c] == color)
    opp = sum(1 for r in range(8) for c in range(8) if stable[r][c] and board[r][c] == -color)
    return my - opp


def _evaluate(board: list[list[int]], color: int) -> float:
    """Static evaluation of `board` from `color`'s perspective."""
    empties = sum(row.count(EMPTY) for row in board)
    phase = (64 - empties) / 64.0

    position_score = sum(_WEIGHTS[r][c] * board[r][c] for r in range(8) for c in range(8)) * color

    my_moves = len(_legal_moves(board, color))
    opp_moves = len(_legal_moves(board, -color))
    mobility = my_moves - opp_moves

    my_count = sum(row.count(color) for row in board)
    opp_count = sum(row.count(-color) for row in board)
    total = my_count + opp_count
    parity = 100.0 * (my_count - opp_count) / total if total else 0.0

    frontier = _frontier_diff(board, color)
    stability = _stability_diff(board, color)

    w_mobility = 15.0 * (1 - phase) + 2.0 * phase
    w_parity = 25.0 * phase
    w_frontier = -2.0 * (1 - phase)
    w_stability = 12.0

    return (
        position_score
        + w_mobility * mobility
        + w_parity * parity
        + w_frontier * frontier
        + w_stability * stability
    )


# --- Search ------------------------------------------------------------------------------


def _board_key(board: list[list[int]], color: int) -> tuple:
    return (tuple(tuple(row) for row in board), color)


def _negamax(
    board: list[list[int]], color: int, depth: int, alpha: float, beta: float,
    deadline: float, counter: list[int], tt: dict, max_depth: int,
) -> float:
    counter[0] += 1
    if counter[0] % _TIME_CHECK_INTERVAL == 0 and time.monotonic() > deadline:
        raise _Timeout()

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
            score = _evaluate(board, color)
            tt[key] = (max_depth, score, _TT_EXACT)
            return score
        if depth == 0:
            score = _evaluate(board, color)
            tt[key] = (depth, score, _TT_EXACT)
            return score
        score = -_negamax(board, -color, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
        tt[key] = (depth, score, _TT_EXACT)
        return score

    if depth == 0:
        score = _evaluate(board, color)
        tt[key] = (depth, score, _TT_EXACT)
        return score

    moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)
    best = -float("inf")
    for move in moves:
        child = _apply_move(board, move, color)
        score = -_negamax(child, -color, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
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


# --- Entry point -----------------------------------------------------------------------


def choose_action(
    state_desc: str,
    player_id: int,
    legal_actions: list[tuple[int, str]],
    time_budget: float = _TIME_BUDGET,
    max_depth: int = _MAX_DEPTH,
) -> int:
    if len(legal_actions) == 1:
        return legal_actions[0][0]

    raw_board = _parse_board(state_desc)
    board = _to_internal(raw_board, _my_symbol(state_desc))

    label_to_cell: dict[tuple[int, int], int] = {}
    for action_id, label in legal_actions:
        label = label.strip().lower()
        if label == "pass" or len(label) != 2:
            continue
        col = ord(label[0]) - ord("a")
        row = int(label[1]) - 1
        label_to_cell[(row, col)] = action_id

    moves = [m for m in _legal_moves(board, ME) if m in label_to_cell]
    if not moves:
        return legal_actions[0][0]

    moves.sort(key=lambda m: _WEIGHTS[m[0]][m[1]], reverse=True)

    best_move = moves[0]
    deadline = time.monotonic() + time_budget
    counter = [0]
    tt: dict = {}
    depth = 1
    while depth <= max_depth and time.monotonic() < deadline:
        alpha, beta = -float("inf"), float("inf")
        iter_best_move = moves[0]
        iter_best_score = -float("inf")
        try:
            for move in moves:
                child = _apply_move(board, move, ME)
                score = -_negamax(child, OPP, depth - 1, -beta, -alpha, deadline, counter, tt, max_depth)
                if score > iter_best_score:
                    iter_best_score = score
                    iter_best_move = move
                if iter_best_score > alpha:
                    alpha = iter_best_score
        except _Timeout:
            break

        best_move = iter_best_move
        moves.remove(best_move)
        moves.insert(0, best_move)

        if len(moves) <= 1:
            break
        depth += 1

    return label_to_cell[best_move]
