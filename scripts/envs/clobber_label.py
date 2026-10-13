"""Deterministic clobber label for SFT data generation (the default clobber teacher).

A row's label must be a FUNCTION of the position: the same board and side to move give the
same move on any machine and under any load, so the student imitates one consistent policy
instead of seed noise. Hence the search is bounded by a NODE count, never a clock, and ties
are broken by board features, never by action id:

  * <= EXACT_MAX_STONES stones: memoised AND/OR solve of the root moves in tie-break order;
    the first proven win is the label. In a lost position the label is the swindle move, the
    one that leaves the opponent the fewest winning replies.
  * above that: iterative-deepening alpha-beta with exact leaves (<= EXACT_MAX_STONES) and a
    mobility leaf, depth capped by a stone-count ladder; the deepest depth that finished
    inside the node budget decides.
  * tie-break: fewest opponent replies, then most own moves, then board coordinates.

Clobber is normal-play, zero-sum and draw-free, and every move removes one stone, so a solved
position is an exact win/loss and the tree shrinks by one stone per ply. Boards are bitboards
(one int per colour, bit r*cols + c) so the node budget buys real depth in pure Python.
Runs only while generating training data; at eval the model plays alone.
"""
from __future__ import annotations

from envs.clobber_expert import _extract_action_labels, _label_to_indices, _parse_board
from envs.clobber_expert import choose_action as _greedy_choose

EXACT_MAX_STONES = 17
# Nodes per label for the midgame search (calibrated on CPU against the R1/R2 gen slices) and for
# the exact solve at <= EXACT_MAX_STONES. The budget is the ONLY bound on the search.
NODE_BUDGET = 20_000
EXACT_NODE_BUDGET = 60_000
# Midgame depth cap by stone count (first threshold >= stones wins, denser boards get _DEPTH_DENSE);
# the 5x6 board searches one ply deeper.
_DEPTH_LADDER = ((20, 6), (24, 5))
_DEPTH_DENSE = 4
_WIDE_BOARD_CELLS = 30
_WIN = 1_000_000


class _OutOfNodes(Exception):
    """Node budget spent: abandon the current search iteration."""


def _directions(rows: int, cols: int) -> tuple:
    """Per direction (source mask, shift, target-is-higher-bit): a stone on a source-mask cell
    has a neighbour `shift` bits away in that direction."""
    cells = [(r, c) for r in range(rows) for c in range(cols)]

    def mask(pred) -> int:
        return sum(1 << (r * cols + c) for r, c in cells if pred(r, c))

    return (
        (mask(lambda r, c: c < cols - 1), 1, True),      # east
        (mask(lambda r, c: c > 0), 1, False),            # west
        (mask(lambda r, c: r < rows - 1), cols, True),   # south
        (mask(lambda r, c: r > 0), cols, False),         # north
    )


_DIRECTIONS: dict = {}


def _dirs_for(rows: int, cols: int) -> tuple:
    dirs = _DIRECTIONS.get((rows, cols))
    if dirs is None:
        dirs = _DIRECTIONS[(rows, cols)] = _directions(rows, cols)
    return dirs


def _moves(me: int, opp: int, dirs: tuple) -> "list[tuple[int, int]]":
    """Every (source bit, target bit): my stone clobbering an orthogonally adjacent enemy stone."""
    out = []
    for srcmask, s, higher in dirs:
        if higher:
            srcs = me & srcmask & (opp >> s)
            while srcs:
                b = srcs & -srcs
                srcs ^= b
                out.append((b, b << s))
        else:
            srcs = me & srcmask & (opp << s)
            while srcs:
                b = srcs & -srcs
                srcs ^= b
                out.append((b, b >> s))
    return out


def _count(me: int, opp: int, dirs: tuple) -> int:
    """Number of legal moves for `me` (mobility)."""
    n = 0
    for srcmask, s, higher in dirs:
        n += (me & srcmask & ((opp >> s) if higher else (opp << s))).bit_count()
    return n


class _Search:
    """One label's search state: node counter, budget and the exact-result memo."""

    def __init__(self, dirs: tuple, budget: int, exact_max: int):
        self.dirs = dirs
        self.budget = budget
        self.exact_max = exact_max
        self.nodes = 0
        self.solved: dict = {}
        self.best_move: dict = {}

    def _spend(self) -> None:
        if self.nodes >= self.budget:
            raise _OutOfNodes
        self.nodes += 1

    def wins(self, me: int, opp: int) -> bool:
        """True iff the side to move (`me`) can force a win. Memoised AND/OR: I win iff some
        move leaves the opponent unable to win; no legal move loses."""
        key = (me, opp)
        hit = self.solved.get(key)
        if hit is not None:
            return hit
        self._spend()
        result = False
        for sb, db in _moves(me, opp, self.dirs):
            if not self.wins(opp ^ db, me ^ sb ^ db):
                result = True
                break
        self.solved[key] = result
        return result

    def negamax(self, me: int, opp: int, depth: int, alpha: int, beta: int) -> int:
        """Side-to-move score: +-_WIN when proven, else the mobility difference at the horizon."""
        if (me | opp).bit_count() <= self.exact_max:
            return _WIN if self.wins(me, opp) else -_WIN
        self._spend()
        if depth <= 0:
            mobility = _count(me, opp, self.dirs)
            return mobility - _count(opp, me, self.dirs) if mobility else -_WIN
        moves = _moves(me, opp, self.dirs)
        if not moves:
            return -_WIN
        # Move ordering (cutoffs come earlier, so the same nodes buy more depth): the previous
        # iteration's best move here first, then the most-restricting replies.
        if depth >= 2:
            moves.sort(key=lambda m: _count(opp ^ m[1], me ^ m[0] ^ m[1], self.dirs))
        key = (me, opp)
        hint = self.best_move.get(key)
        if hint is not None and hint in moves:
            moves.remove(hint)
            moves.insert(0, hint)
        best, best_mv = -_WIN - 1, None
        for sb, db in moves:
            score = -self.negamax(opp ^ db, me ^ sb ^ db, depth - 1, -beta, -alpha)
            if score > best:
                best, best_mv = score, (sb, db)
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        self.best_move[key] = best_mv
        return best


def _depth_cap(stones: int, cells: int) -> int:
    depth = _DEPTH_DENSE
    for threshold, d in _DEPTH_LADDER:
        if stones <= threshold:
            depth = d
            break
    return depth + (1 if cells >= _WIDE_BOARD_CELLS else 0)


def _to_bits(board: "tuple[str, ...] | list", sym: str) -> int:
    cols = len(board[0])
    return sum(1 << (r * cols + c) for r, row in enumerate(board) for c, cell in enumerate(row) if cell == sym)


def _coords(bit: int, cols: int) -> "tuple[int, int]":
    i = bit.bit_length() - 1
    return divmod(i, cols)


def _exact_label(search: _Search, root: list, me: int, opp: int):
    """First proven win in tie-break order; if every move loses, the swindle move. Raises
    _OutOfNodes when the exact budget is spent."""
    for key, sb, db in root:
        if not search.wins(opp ^ db, me ^ sb ^ db):
            return key
    swindle = []
    for key, sb, db in root:
        child_me, child_opp = opp ^ db, me ^ sb ^ db
        winning_replies = sum(1 for rs, rd in _moves(child_me, child_opp, search.dirs)
                              if not search.wins(child_opp ^ rd, child_me ^ rs ^ rd))
        swindle.append((winning_replies, key))
    return min(swindle)[1]


def _midgame_label(search: _Search, root: list, me: int, opp: int, max_depth: int):
    """Deepest fully searched iterative-deepening result; the first root move (tie-break order)
    with the best score wins ties. None when not even depth 1 fits the budget."""
    chosen = None
    for depth in range(1, max_depth + 1):
        try:
            alpha, best = -_WIN - 1, None
            for key, sb, db in root:
                score = -search.negamax(opp ^ db, me ^ sb ^ db, depth - 1, -_WIN - 1, -alpha)
                if score > alpha:
                    alpha, best = score, key
        except _OutOfNodes:
            break
        chosen = best
        if alpha >= _WIN:
            break  # a proven win at this depth cannot be improved by searching deeper
    return chosen


def _root_moves(mine: int, theirs: int, dirs: tuple, cols: int) -> list:
    """Root moves as (tie-break key, source bit, target bit), sorted by the key: fewest opponent
    replies, then most own moves, then board coordinates (r, c, nr, nc) — the key's last item."""
    root = []
    for sb, db in _moves(mine, theirs, dirs):
        child_me, child_opp = theirs ^ db, mine ^ sb ^ db
        coords = _coords(sb, cols) + _coords(db, cols)
        key = (_count(child_me, child_opp, dirs), -_count(child_opp, child_me, dirs), coords)
        root.append((key, sb, db))
    root.sort()
    return root


def greedy_move(board: "tuple[str, ...] | list", me: str, opp: str) -> "tuple[int, int, int, int] | None":
    """The 1-ply move (no search): first move in tie-break order. None when `me` cannot move."""
    rows, cols = len(board), len(board[0])
    root = _root_moves(_to_bits(board, me), _to_bits(board, opp), _dirs_for(rows, cols), cols)
    return root[0][0][2] if root else None


def label_move(board: "tuple[str, ...] | list", me: str, opp: str,
               node_budget: int = NODE_BUDGET,
               exact_node_budget: int = EXACT_NODE_BUDGET) -> "tuple[int, int, int, int] | None":
    """The label move (r, c, nr, nc) for `me` to move on `board` (rows of '.', 'x', 'o', top row
    first). None when `me` has no legal move. Deterministic: no clock, no rng, no action ids."""
    rows, cols = len(board), len(board[0])
    dirs = _dirs_for(rows, cols)
    mine, theirs = _to_bits(board, me), _to_bits(board, opp)
    root = _root_moves(mine, theirs, dirs, cols)
    if not root:
        return None
    if len(root) == 1:
        return root[0][0][2]
    stones = (mine | theirs).bit_count()
    if stones <= EXACT_MAX_STONES:
        try:
            return _exact_label(_Search(dirs, exact_node_budget, EXACT_MAX_STONES),
                                root, mine, theirs)[2]
        except _OutOfNodes:
            pass  # too dense to prove inside the budget: search it like a midgame (no exact leaves)
        exact_max = -1
    else:
        exact_max = EXACT_MAX_STONES
    chosen = _midgame_label(_Search(dirs, node_budget, exact_max), root, mine, theirs,
                            _depth_cap(stones, rows * cols))
    return (chosen if chosen is not None else root[0][0])[2]


def choose_action(observation: str, player_id: int, legal_action_ids: "list[int]", rng=None) -> int:
    """pvp_selfplay teacher interface: parse the observation (P0 moves the 'o' stones) and map
    the label move back to its action id. Never raises: falls back to the greedy teacher."""
    try:
        grid, grid_rows, _ = _parse_board(observation)
        labels = _extract_action_labels(observation)
        me, opp = ("o", "x") if player_id == 0 else ("x", "o")
        move = label_move(grid, me, opp)
        for aid in legal_action_ids:
            label = labels.get(aid)
            if label is not None:
                (r, c), (nr, nc) = _label_to_indices(label, grid_rows)
                if (r, c, nr, nc) == move:
                    return aid
    except Exception:
        pass
    return _greedy_choose(observation, player_id, legal_action_ids, rng)
