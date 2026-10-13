"""Board-text <-> internal-grid helpers for Othello's "double view" SFT data.

Used to reconstruct the MCTS opponent's per-turn (state, legal_actions) from
"our" turn's state_desc + chosen action_id + ``info.last_opponent_action``,
so the opponent's move can also become a training example (see
othello_trajectories.py).

Action-id encoding (matches the mcts-api env server / OpenSpiel Othello):
``action_id = row*8 + col`` (0-indexed row/col, 0-63), e.g. 44 -> "e6"
(row=5, col=4). ``action_id == 64`` is "pass" (no legal board moves).
"""

from envs.othello_minimax import ME, OPP
from envs.othello_minimax import _apply_move, _legal_moves, _to_internal

PASS_ACTION_ID = 64


def _action_to_cell(action_id: int) -> tuple[int, int]:
    return action_id // 8, action_id % 8


def _cell_to_label(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{row + 1}"


def apply_action_symbols(raw_board: list[list[str]], action_id: int, symbol: str) -> list[list[str]]:
    """Apply `symbol`'s move (action_id, possibly PASS) to a symbol board ('x'/'o'/'-')."""
    if action_id == PASS_ACTION_ID:
        return [row[:] for row in raw_board]

    opp_symbol = "o" if symbol == "x" else "x"
    board = _to_internal(raw_board, symbol)
    row, col = _action_to_cell(action_id)
    new_board = _apply_move(board, (row, col), ME)

    new_raw = [["-"] * 8 for _ in range(8)]
    for r in range(8):
        for c in range(8):
            if new_board[r][c] == ME:
                new_raw[r][c] = symbol
            elif new_board[r][c] == OPP:
                new_raw[r][c] = opp_symbol
    return new_raw


def legal_actions_symbols(raw_board: list[list[str]], symbol: str) -> list[tuple[int, str]]:
    """Legal actions for `symbol` on a symbol board, as [(action_id, label), ...]."""
    board = _to_internal(raw_board, symbol)
    moves = _legal_moves(board, ME)
    if not moves:
        return [(PASS_ACTION_ID, "pass")]
    return [(r * 8 + c, _cell_to_label(r, c)) for r, c in moves]


def render_board(raw_board: list[list[str]], to_play_symbol: str) -> str:
    """Render a symbol board back into the env's "Current State" text block."""
    colour = "Black" if to_play_symbol == "x" else "White"
    header = "  a b c d e f g h  "
    lines = [f"{colour} ({to_play_symbol}) to play:", header]
    for r in range(8):
        row_cells = " ".join(raw_board[r])
        lines.append(f"{r + 1} {row_cells} {r + 1}")
    lines.append(header)
    return "\n".join(lines)
