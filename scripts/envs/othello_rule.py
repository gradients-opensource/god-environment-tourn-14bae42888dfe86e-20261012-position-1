"""Deterministic corner-aware othello label (the default othello teacher).

A row's label is a pure function of the visible board and the legal set, so the student imitates
one consistent, learnable policy: the same position gives the same move on any machine, under any
CPU load, with no clock anywhere. The rule, in order:

  1. take a legal corner (lowest id first);
  2. veto the X squares, then the C squares, beside a corner that is still EMPTY, as long as a
     safer move remains (a square beside a corner either colour already holds is not vetoed);
  3. play the highest-weight square of the IAGO positional table (othello_minimax's default
     table), ties to the lowest id.

Without step 2's corner check this is the plain board-blind positional rule (``positional_label``),
which is also the fallback when the board cannot be read. The search-based expert it replaces
(othello_expert, kept behind OTHELLO_TEACHER=expert) labels by a wall-clock search whose endgame
solve times out at 12-16 empties on most decisions, and loses to this rule head-to-head.
Runs only while generating training data; at eval the model plays alone.
"""
from __future__ import annotations

from envs.othello_board import PASS_ACTION_ID
from envs.othello_minimax import _WEIGHTS_DEFAULT, _parse_board

CORNERS = (0, 7, 56, 63)
# Square id -> the corner it touches: X squares (diagonal) and C squares (edge).
X_SQUARES = {9: 0, 14: 7, 49: 56, 54: 63}
C_SQUARES = {1: 0, 8: 0, 6: 7, 15: 7, 48: 56, 57: 56, 55: 63, 62: 63}
_WEIGHT = tuple(_WEIGHTS_DEFAULT[i // 8][i % 8] for i in range(64))
_NO_CORNERS: frozenset = frozenset()


def _board_ids(legal_ids) -> "list[int]":
    """Sorted, de-duplicated board-move ids (0..63) of the legal set; anything else is dropped."""
    ids = set()
    for a in legal_ids or ():
        try:
            a = int(a)
        except (TypeError, ValueError):
            continue
        if 0 <= a < 64:
            ids.add(a)
    return sorted(ids)


def occupied_corners(board: "list[list[str]]") -> frozenset:
    """Ids of the corners holding a disc of either colour on a parsed ('-'/'x'/'o') board."""
    return frozenset(c for c in CORNERS if board[c // 8][c % 8] in ("x", "o"))


def _pick(ids: "list[int]", taken: frozenset) -> "tuple[int, str]":
    corners = [a for a in ids if a in CORNERS]
    if corners:
        return corners[0], "corner"
    x_free = [a for a in ids if X_SQUARES.get(a) is None or X_SQUARES[a] in taken]
    safe = [a for a in x_free if C_SQUARES.get(a) is None or C_SQUARES[a] in taken]
    pool = safe or x_free or ids
    # max() keeps the first of equal weights, and ids are ascending: ties go to the lowest id.
    best = max(pool, key=lambda a: _WEIGHT[a])
    if len(pool) == len(ids):
        reason = "table"
    elif len(x_free) < len(ids):
        reason = "avoid_x"
    else:
        reason = "avoid_c"
    return best, reason


def positional_label(legal_ids) -> int:
    """The board-blind positional rule: corner, X then C veto, IAGO table, lowest id on ties."""
    ids = _board_ids(legal_ids)
    if not ids:
        return PASS_ACTION_ID
    return _pick(ids, _NO_CORNERS)[0]


def rule_label(board: "list[list[str]]", me: str, legal_ids) -> "tuple[int, str]":
    """(action id, reason) for the side ``me`` ('x' or 'o') on a parsed board. The reason is one of
    pass / corner / avoid_x / avoid_c / table. The label depends on the board only through which
    corners are occupied, so ``me`` does not change it. Never raises; an unreadable board labels
    with the plain positional rule."""
    ids = _board_ids(legal_ids)
    if not ids:
        return PASS_ACTION_ID, "pass"
    try:
        taken = occupied_corners(board)
    except Exception:  # noqa: BLE001 — a malformed board degrades to the board-blind rule
        taken = _NO_CORNERS
    return _pick(ids, taken)


def choose_action(observation: str, player_id: int, legal_action_ids, rng=None) -> int:
    """Teacher contract used by pvp_selfplay: the rule label of the board in ``observation``."""
    try:
        board = _parse_board(observation)
    except Exception:  # noqa: BLE001
        board = None
    me = "x" if player_id == 0 else "o"
    if board is None:
        return positional_label(legal_action_ids)
    return rule_label(board, me, legal_action_ids)[0]
