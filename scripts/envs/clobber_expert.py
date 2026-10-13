"""Clobber teacher: greedy mobility-maximizer for pvp_selfplay.py.

Strategy: for each legal move, simulate the result and pick the action that
MINIMIZES the opponent's resulting move count (maximum pressure). Ties are broken
by maximising our own move count, then by taking the piece closest to the center
(more future options).

Falls back to random if board parsing fails (e.g. unexpected observation format).
"""
from __future__ import annotations

import copy
import re


# ---------------------------------------------------------------------------
# Board parsing
# ---------------------------------------------------------------------------

def _parse_board(obs: str) -> tuple[list[list[str]], int, int]:
    """Extract grid from observation text. Returns (grid, rows, cols).

    Handles two common OpenSpiel clobber observation formats:
      - compact: 'x.x.\\n.x.x\\n...'     (no spaces)
      - spaced:  'x . x .\\n. x . x\\n...'
    """
    lines = obs.strip().split("\n")
    grid: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # pyspiel clobber rows are row-number-prefixed, e.g. "5ooox.x". Strip the
        # leading row number before reading cells; the column-label line
        # (" abcdef") and any prose ("You play x.") have no .xo cells and drop out.
        body = stripped.lstrip("0123456789")
        # Compact: only '.', 'x', 'o' chars
        if body and all(c in ".xo" for c in body):
            grid.append(list(body))
        # Spaced: chars separated by spaces — EVERY token must be a cell, so prose that merely
        # contains a lone 'o'/'x' token (the eval colour line "You play o (White).") is not a row.
        else:
            cells = body.split()
            if len(cells) >= 2 and all(c in (".", "x", "o") for c in cells):
                grid.append(cells)
    if not grid:
        raise ValueError("Cannot parse board from observation")
    rows = len(grid)
    cols = len(grid[0])
    return grid, rows, cols


def _parse_action_label(label: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse chess-like action label into (from_row, from_col), (to_row, to_col).

    Accepts 'a1-b2' (with dash) and 'a1b2' (without dash).
    Column is the letter (a=0, b=1, …), row is the digit (1-based from bottom).
    """
    label = label.strip()
    m = re.match(r"([a-z])(\d+)-?([a-z])(\d+)", label)
    if not m:
        raise ValueError(f"Unparseable action label: {label!r}")
    c1, r1, c2, r2 = m.groups()
    # rows: digit 1 = last grid row (bottom), so row_idx = grid_rows - int(r)
    # We defer the inversion to the caller which knows grid_rows.
    return (int(r1), ord(c1) - ord("a")), (int(r2), ord(c2) - ord("a"))


def _label_to_indices(label: str, grid_rows: int) -> tuple[tuple[int,int], tuple[int,int]]:
    (r1, c1), (r2, c2) = _parse_action_label(label)
    # OpenSpiel row 1 = bottom → index (grid_rows - r)
    return (grid_rows - r1, c1), (grid_rows - r2, c2)


def _apply_move(grid: list[list[str]], fr: tuple[int,int], to: tuple[int,int]) -> list[list[str]]:
    g = copy.deepcopy(grid)
    g[to[0]][to[1]] = g[fr[0]][fr[1]]
    g[fr[0]][fr[1]] = "."
    return g


def _count_moves(grid: list[list[str]], my_sym: str, opp_sym: str) -> int:
    rows, cols = len(grid), len(grid[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == my_sym:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == opp_sym:
                        count += 1
    return count


def _centrality_bonus(pos: tuple[int,int], rows: int, cols: int) -> float:
    """Pieces closer to center have higher value (more potential future moves)."""
    cr, cc = (rows - 1) / 2, (cols - 1) / 2
    dist = abs(pos[0] - cr) + abs(pos[1] - cc)
    max_dist = cr + cc
    return 1.0 - dist / (max_dist + 1e-9)


# ---------------------------------------------------------------------------
# Action label extraction from teacher-obs envelope
# ---------------------------------------------------------------------------

def _extract_action_labels(obs: str) -> dict[int, str]:
    """Parse 'N -> label' lines from _teacher_obs envelope."""
    labels: dict[int, str] = {}
    in_actions = False
    for line in obs.split("\n"):
        if "Legal Actions:" in line:
            in_actions = True
            continue
        if in_actions:
            m = re.match(r"\s*(\d+)\s*->\s*(.+)", line.strip())
            if m:
                labels[int(m.group(1))] = m.group(2).strip()
    return labels


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def choose_action(obs: str, player_id: int, legal_action_ids: list[int],
                  rng=None) -> int:
    """Greedy mobility teacher. Falls back to random on parse error."""
    try:
        return _greedy_choose(obs, player_id, legal_action_ids)
    except Exception:
        if rng is not None:
            return rng.choice(legal_action_ids)
        return legal_action_ids[0]


def _greedy_choose(obs: str, player_id: int, legal_action_ids: list[int]) -> int:
    action_labels = _extract_action_labels(obs)
    grid, grid_rows, grid_cols = _parse_board(obs)

    # OpenSpiel clobber: player 0 moves the "o" stones (validator ClobberAgent: P0 = "o (White)").
    my_sym  = "o" if player_id == 0 else "x"
    opp_sym = "x" if player_id == 0 else "o"

    best_action = legal_action_ids[0]
    best_score  = float("-inf")

    for aid in legal_action_ids:
        label = action_labels.get(aid)
        if label is None:
            continue
        try:
            fr, to = _label_to_indices(label, grid_rows)
        except Exception:
            continue

        new_grid = _apply_move(grid, fr, to)

        # Immediate winning move?
        opp_after = _count_moves(new_grid, opp_sym, my_sym)
        if opp_after == 0:
            return aid  # instant win, take it

        my_after = _count_moves(new_grid, my_sym, opp_sym)
        # Primary: minimise opponent moves; secondary: maximise own; tertiary: centrality of dest
        score = (-2 * opp_after) + my_after + 0.1 * _centrality_bonus(to, grid_rows, grid_cols)
        if score > best_score:
            best_score  = score
            best_action = aid

    return best_action
