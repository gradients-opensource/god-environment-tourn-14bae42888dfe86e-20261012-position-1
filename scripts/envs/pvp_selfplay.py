"""Local pyspiel teacher self-play data generation.

Replaces the env-server (mcts-api) trajectory loop. Instead of POSTing one
controlled player's actions to a remote server that auto-plays a server-side
MCTS opponent, we load the OpenSpiel game in-process and have the per-game
TEACHER play BOTH seats. This:

  * removes ALL HTTP round-trips and server-side MCTS — the dominant cost on the
    long-game envs (gin_rummy/othello), which the env-server path generated at
    well under 1 game/s;
  * harvests EVERY decision node (both seats are teacher-quality), ~2x the
    examples per game and a state distribution drawn from strong-vs-strong play;
  * produces prompts byte-identical to the PvP eval, because the rows are built
    from the exact eval-side pieces: ``agent.format_state(state, player)`` for
    the state text and ``state.action_to_string(player, a)`` for the legal-action
    labels, fed through the same ``build_user_prompt`` the eval harness mirrors.

Eval (validator/evaluation/pvp/game_runner.py) drives games with local pyspiel,
NOT the mcts-api server — so self-play here is the same game engine the model is
ultimately scored on.

Each game returns a list of (examples, score) "views" — one per seat — matching
the multi-view shape generate_trajectories already understands (cf. othello's
double view), so per-seat score filtering (wins-only / sample-by-score) works
unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import random
import re

import pyspiel

from envs.pvp_format import (
    build_full_system_prompt,
    build_pvp_tools,
    build_user_prompt,
    tools_to_openai,
)

# --- per-env pyspiel game spec + eval-faithful state formatting --------------
# Ported from core/pvp/agents.py (the eval-side formatter). Keep format_state in
# lock-step with that file: the model is trained and scored on these exact
# strings.

# task-id ranges mirror core/constants.py ENVIRONMENT_CONFIGS, used only to map a
# game seed -> game-variant config id the same way eval does
# (game_eval.config_id_for_seed).
_TASK_ID_RANGE = {
    "goofspiel": (0, 99_999_999),
    "liars_dice": (100_000_000, 199_999_999),
    "leduc_poker": (200_000_000, 299_999_999),
    "gin_rummy": (300_000_000, 399_999_999),
    "othello": (400_000_000, 499_999_999),
    "clobber": (700_000_000, 799_999_999),
}
_PVP_CONFIG_ID_DIVISOR = 100_000_000  # core/pvp/constants.py PVP_CONFIG_ID_DIVISOR


def _config_id_for_seed(seed: int, env_name: str) -> int:
    lo, hi = _TASK_ID_RANGE[env_name]
    r = random.Random(seed)
    return r.randint(lo, hi) % _PVP_CONFIG_ID_DIVISOR


def _card_name(card_id: int) -> str:
    ranks = ["J", "Q", "K", "A"]
    suits = ["♠", "♥"]
    rank_idx, suit_idx = card_id // 2, card_id % 2
    return f"{ranks[rank_idx]}{suits[suit_idx]}" if rank_idx < len(ranks) else f"Card_{card_id}"


def _parse_betting(seq: str) -> str:
    if not seq or not seq.strip():
        return "(none)"
    amap = {0: "Fold", 1: "Call", 2: "Raise"}
    nums = [int(x) for x in seq.split() if x.isdigit()]
    return ", ".join(amap.get(a, f"Action{a}") for a in nums) if nums else "(none)"


def _re_extract(info_str: str, pattern: str) -> str:
    import re

    m = re.search(pattern, info_str)
    return m.group(1) if m else ""


def _format_liars_dice(state: pyspiel.State, player_id: int) -> str:
    info_str = state.information_state_string(player_id)
    if not info_str:
        return str(state)
    parts = info_str.split()
    dice = [int(d) for d in parts[0] if d.isdigit()]
    bid_parts = [p for p in parts[1:] if "-" in p]
    num_dice = len(dice)
    lines = [
        f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
        f"Dice per player: {num_dice}",
        f"Total dice in game: {num_dice * state.num_players()}",
        f"Players: {state.num_players()}",
        f"Current player: Player {state.current_player()}",
    ]
    if bid_parts:
        q, f = bid_parts[-1].split("-")
        lines.append(
            f'\nCurrent bid: "{q}-{f}" (at least {q} dice showing {f} across all players)'
        )
        lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
    else:
        lines.append("No bid yet - you must make the first bid")
    return "\n".join(lines)


def _format_leduc(state: pyspiel.State, player_id: int) -> str:
    info_str = state.information_state_string(player_id)
    private_card = _re_extract(info_str, r"\[Private: (-?\d+)\]")
    round_num = _re_extract(info_str, r"\[Round (\d+)\]")
    pot = _re_extract(info_str, r"\[Pot: (\d+)\]")
    money = _re_extract(info_str, r"\[Money: ([\d ]+)\]")
    public_card = _re_extract(info_str, r"\[Public: (-?\d+)\]")
    r1 = _re_extract(info_str, r"\[Round1: ([^\]]*)\]")
    r2 = _re_extract(info_str, r"\[Round2: ([^\]]*)\]")
    lines: list[str] = []
    if private_card and private_card != "-10000":
        lines.append(f"Your card: {_card_name(int(private_card))}")
    else:
        lines.append("Your card: (not dealt yet)")
    if public_card and public_card != "-10000":
        lines.append(f"Public card: {_card_name(int(public_card))}")
        if private_card and private_card != "-10000" and int(private_card) // 2 == int(public_card) // 2:
            lines.append("Hand: PAIR")
    lines.append(f"Round: {round_num}/2")
    lines.append(f"Pot: {pot} chips")
    if money:
        chips = money.split()
        if len(chips) >= 2:
            lines.append(f"Your chips: {chips[player_id]}")
            lines.append(f"Opponent chips: {chips[1 - player_id]}")
    if r1:
        lines.append(f"Round 1 actions: {_parse_betting(r1)}")
    if r2:
        lines.append(f"Round 2 actions: {_parse_betting(r2)}")
    return "\n".join(lines)


def _format_gin_rummy(state: pyspiel.State, player_id: int) -> str:
    return state.observation_string(player_id)


def _format_othello(state: pyspiel.State, player_id: int) -> str:
    colour = "x (Black)" if player_id == 0 else "o (White)"
    return f"You play {colour}.\n{state.observation_string(player_id)}"


def _format_goofspiel(state: pyspiel.State, player_id: int) -> str:
    # = GoofspielAgent.format_state: imp_info observation (own hand only, "Current player", point card,
    # remaining cards, points, win sequence) behind the eval's "You are Player N (PN)." header.
    return f"You are Player {player_id} (P{player_id}).\n{state.observation_string(player_id)}"


def _format_clobber(state: pyspiel.State, player_id: int) -> str:
    # = ClobberAgent.format_state. OpenSpiel clobber: player 0 moves the 'o' stones (verified: P0's
    # first legal move a5b5 moves the a5 'o'); the old "You play x." for P0 named the opponent's colour.
    colour = "o (White)" if player_id == 0 else "x (Black)"
    return f"You play {colour}.\n{state.observation_string(player_id)}"


# Eval game variants (core/pvp/agents.py), picked by the eval config id exactly like generate_params:
# goofspiel num_cards (5, 8, 10, 13), imp_info, random prize order, win/loss, wrapped turn-based;
# clobber boards (4,5)/(5,5)/(5,6) with 2-6 seeded random opening plies. GOOFSPIEL_NUM_CARDS /
# CLOBBER_ROWS+CLOBBER_COLS pin one variant (diagnostics only).
_GOOFSPIEL_NUM_CARDS_CHOICES = (5, 8, 10, 13)
_CLOBBER_BOARD_SIZES = ((4, 5), (5, 5), (5, 6))
_CLOBBER_OPENING_PLIES = (2, 6)


_OTHELLO_OPENING_PLIES = (2, 6)


def _load_game(env_name: str, config_id: int) -> pyspiel.Game:
    if env_name == "liars_dice":
        return pyspiel.load_game("liars_dice", {"players": 2, "numdice": 5})
    if env_name == "leduc_poker":
        return pyspiel.load_game("leduc_poker", {"players": 2})
    if env_name == "gin_rummy":
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return pyspiel.load_game(
            "gin_rummy", {"hand_size": 7 + hand_var, "knock_card": 10 - knock_var}
        )
    if env_name == "othello":
        return pyspiel.load_game("othello")
    if env_name == "goofspiel":
        pinned = os.getenv("GOOFSPIEL_NUM_CARDS", "").strip()
        num_cards = int(pinned) if pinned else _GOOFSPIEL_NUM_CARDS_CHOICES[config_id % len(_GOOFSPIEL_NUM_CARDS_CHOICES)]
        return pyspiel.convert_to_turn_based(pyspiel.load_game(
            "goofspiel",
            {
                "num_cards": num_cards,
                "players": 2,
                "imp_info": True,
                "points_order": "random",
                "returns_type": "win_loss",
            },
        ))
    if env_name == "clobber":
        rows, cols = _CLOBBER_BOARD_SIZES[config_id % len(_CLOBBER_BOARD_SIZES)]
        if os.getenv("CLOBBER_ROWS", "").strip() and os.getenv("CLOBBER_COLS", "").strip():
            rows, cols = int(os.environ["CLOBBER_ROWS"]), int(os.environ["CLOBBER_COLS"])
        return pyspiel.load_game("clobber", {"rows": rows, "columns": cols})
    raise ValueError(f"No pyspiel self-play spec for env {env_name!r}")


def _setup_initial_state(env_name: str, state: pyspiel.State, seed: int) -> None:
    """Mirror agents.py setup_initial_state: the deterministic board games (othello, clobber) get
    seeded random opening plies for variety; chance-node games deal via _resolve_chance."""
    if env_name == "othello":
        plies = _OTHELLO_OPENING_PLIES
    elif env_name == "clobber":
        plies = _CLOBBER_OPENING_PLIES
    else:
        return
    rng = random.Random(seed)
    for _ in range(rng.randint(*plies)):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(rng.choice(legal))


_FORMATTERS = {
    "liars_dice": _format_liars_dice,
    "leduc_poker": _format_leduc,
    "gin_rummy": _format_gin_rummy,
    "othello": _format_othello,
    "goofspiel": _format_goofspiel,
    "clobber": _format_clobber,
}


# Othello's teacher is an iterative-deepening minimax + PERFECT endgame solver.
# LIVE ROUND-2 LESSON (othello lost 0-25 / 2-32 — our worst env by far): format is
# byte-identical to eval (ruled out), so the collapse was a WEAK TEACHER. The old
# gen-speed overrides crippled it — a 0.04s search is shallow, and an endgame solver
# that only wakes at 8 empties hands the DECISIVE last ~16 plies (where othello is won,
# via disc flips) back to the weak heuristic. Othello is the ONLY perfect-info env
# WITHOUT a full solver (games use CFR/alpha-beta solvers), so a shallow teacher hits
# it hardest. Restore near-eval strength for gen:
#   - budget 0.04 -> 0.20s: deeper midgame. ~5x slower/game, but othello's _GEN_TIME_WEIGHT
#     already allocates it the most gen time, and the balance floor backfills the row count.
#   - endgame 8 -> 16 (the module's own gen default): perfect play across the decisive endgame.
#   - depth 8 -> 12.
# All still env-tunable (OTHELLO_TEACHER_BUDGET/DEPTH/ENDGAME) to sweep the speed/strength trade.
# MEASURED 2026-07-20 (verify_othello_teacher, 40 games): 0.20s ≈ 0.04s in STRENGTH (47%,
# a statistical tie) — the engine is already near-optimal at a small budget, so extra search
# time buys ~nothing. Since the R2 othello collapse (0-25) was NOT a strength problem, the
# likely lever is DATA DIVERSITY: a faster teacher plays ~2.5x more games in the time-boxed
# gen → ~2.5x more distinct othello positions → better coverage. So drop the budget back to
# 0.08 (fast, more data) while KEEPING the strong endgame solver (16 empties) + depth cap 12.
_OTHELLO_TEACHER_BUDGET = float(os.getenv("OTHELLO_TEACHER_BUDGET", "0.08"))
_OTHELLO_TEACHER_DEPTH = int(os.getenv("OTHELLO_TEACHER_DEPTH", "12"))
_OTHELLO_TEACHER_ENDGAME = int(os.getenv("OTHELLO_TEACHER_ENDGAME", "16"))
# The othello label (2026-10-01) is the deterministic corner-aware rule (othello_rule): no clock, the
# same move for the same board on any machine. On CPU it beats the expert above head-to-head, and the
# expert's wall-clock endgame times out at 12-16 empties on most decisions. "expert" restores the
# 13759ee search label (with its timeout fallback fixed); OTHELLO_TEACHER=expert selects it per run, and
# OTHELLO_TEACHER=mcts still routes to _openspiel_mcts_teacher.
_OTHELLO_TEACHER_DEFAULT = "rule"


def _othello_teacher_mode() -> str:
    """'rule' or 'expert': OTHELLO_TEACHER when it names one of them, else _OTHELLO_TEACHER_DEFAULT."""
    mode = os.getenv("OTHELLO_TEACHER", "").strip().lower()
    return mode if mode in ("rule", "expert") else _OTHELLO_TEACHER_DEFAULT


def _teacher(env_name: str):
    """Lazy import the per-env teacher's choose_action(observation, player_id,
    legal_action_ids, rng) -> int."""
    if env_name == "liars_dice":
        from envs.liar_dice_expert import choose_action
    elif env_name == "gin_rummy":
        from envs.gin_rummy_expert import choose_action
    elif env_name == "leduc_poker":
        from envs.leduc_poker_expert import choose_action
    elif env_name == "othello" and _othello_teacher_mode() == "rule":
        from envs.othello_rule import choose_action
    elif env_name == "othello":
        from envs.othello_expert import choose_action as _oth

        def choose_action(observation, player_id, legal_action_ids, rng=None):
            return _oth(observation, player_id, legal_action_ids, rng=rng,
                        time_budget=_OTHELLO_TEACHER_BUDGET, max_depth=_OTHELLO_TEACHER_DEPTH,
                        endgame_empties=_OTHELLO_TEACHER_ENDGAME)
    elif env_name == "goofspiel":
        from envs.goofspiel_expert import choose_action
    elif env_name == "clobber":
        # Deterministic node-budgeted label (clobber_label): exact solve at <=17 stones, iterative-
        # deepening alpha-beta above, ties by board features — same position, same move, any machine.
        from envs.clobber_label import choose_action
    else:
        raise ValueError(f"No teacher for env {env_name!r}")
    return choose_action


# Envs whose DEFAULT teacher is the OpenSpiel MCTS bot. othello was REMOVED 2026-08-02:
# its random-rollout MCTS is weak vs the real othello champ, while the hand-written EXPERT
# engine (negamax + perfect endgame, othello_expert.py) is 46% vs the Boss AND fast — so othello left
# for _teacher('othello') (since 2026-10-01 the othello_rule label). Override with OTHELLO_TEACHER=mcts.
# clobber left this list for its deterministic label (_teacher): the MCTS-100 label with one random
# rollout gave a different move per seed on the same position. CLOBBER_TEACHER=mcts still selects it
# (self-play and hero-seat paths); ("clobber",) restores the 13759ee default.
_MCTS_TEACHER_DEFAULT_ENVS = ()
_MCTS_WARNED: set = set()  # envs we've already logged an MCTS-unavailable fallback for (avoid log spam)


def _make_mcts_bot(game: "pyspiel.Game", seed: int, sims: int,
                   rollouts: int = 1, uct_c: float = 2.0):
    """Build an OpenSpiel MCTSBot (RandomRolloutEvaluator) on `game`. Returns None if open_spiel
    is unavailable so callers can fall back and never crash generation. Shared by the MCTS teacher
    (_openspiel_mcts_teacher) and the vs-eval-opponent generator (play_vs_eval_game)."""
    try:
        import numpy as np
        from open_spiel.python.algorithms import mcts
    except Exception:
        return None
    np_rng = np.random.RandomState(seed & 0x7FFFFFFF)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=rollouts, random_state=np_rng)
    return mcts.MCTSBot(game, uct_c=uct_c, max_simulations=sims, evaluator=evaluator,
                        random_state=np_rng)


def _openspiel_mcts_teacher(env_name: str, game: "pyspiel.Game", seed: int):
    """STRONG teacher: OpenSpiel's own MCTSBot playing on the live pyspiel state — the exact
    bot FAMILY the validator scores against. ON BY DEFAULT for the perfect-info differentiators
    (othello, clobber); every other env stays on its hand-written teacher. Override per env with
    <ENV>_TEACHER=mcts / =off, and set the strength with <ENV>_MCTS_SIMS.

    WHY (measured 2026-07-26, verify_teacher_vs_mcts.py, recalibrated to the REAL ~25-sim eval
    opponent): our hand-written othello/clobber teachers lose to even a WEAK MCTS (greedy clobber
    20%, and a deep mobility-negamax is worse still — mobility is the wrong proxy for clobber),
    while an MCTS teacher at 100 sims BEATS that opponent 70% and is fast (~1s/game). So for these
    envs we distil MCTS play into the model: training rows keep the identical obs/label format,
    only the CHOSEN move gets stronger. Returns None (fall back to the hand-written teacher) on
    any import/config failure, so this path can never crash generation."""
    default_mode = "mcts" if env_name in _MCTS_TEACHER_DEFAULT_ENVS else ""
    if os.getenv(f"{env_name.upper()}_TEACHER", default_mode).strip().lower() != "mcts":
        return None
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        return None  # MCTSBot can't drive simultaneous games (goofspiel); keep the text teacher
    sims = int(os.getenv(f"{env_name.upper()}_MCTS_SIMS", "100"))  # 100 beats the ~25-sim eval opponent 70%, and is fast
    # Evaluator-strength knobs (env-tunable per env; defaults = what got us to R1 3rd, so gen is
    # UNCHANGED unless set). n_rollouts averages more random playouts per MCTS leaf → a less noisy
    # value estimate; othello's random playouts are very noisy (its MCTS teacher plateaued at parity
    # and lost the live othello 2-14), so ROLLOUTS is the lever we test to strengthen othello.
    rollouts = int(os.getenv(f"{env_name.upper()}_MCTS_ROLLOUTS", "1"))
    uct_c = float(os.getenv(f"{env_name.upper()}_MCTS_UCT_C", "2.0"))
    bot = _make_mcts_bot(game, seed, sims, rollouts, uct_c)
    if bot is None and env_name in _MCTS_TEACHER_DEFAULT_ENVS and env_name not in _MCTS_WARNED:
        # A DEFAULT-on env fell back to the weak hand-written teacher — make it LOUD in the logs
        # (a silent fallback = shipping the weak teacher and losing again).
        _MCTS_WARNED.add(env_name)
        print(f"[pvp_selfplay] WARNING: '{env_name}' MCTS teacher unavailable — open_spiel.mcts "
              f"import FAILED, falling back to the WEAK hand-written teacher. Ensure open_spiel "
              f"is installed in the training image.", flush=True)
    return bot


# --- example construction (byte-identical to eval; see build_user_prompt) -----

_SYSTEM_PROMPT_CACHE: dict[str, str] = {}


def _system_prompt(env_name: str) -> str:
    sp = _SYSTEM_PROMPT_CACHE.get(env_name)
    if sp is None:
        sp = build_full_system_prompt(env_name)
        _SYSTEM_PROMPT_CACHE[env_name] = sp
    return sp


# One-sentence rule-grounded CoT rationale (winner's learnability device). Gated by
# PVP_LABEL_COT=1; othello only for now (its expert exposes get_label_comment). The
# comment goes in the assistant `content`, rendered by apply_chat_template BEFORE the
# game_action tool call, so the student learns board -> [feature] -> move. OFF by default:
# it changes the SFT row format and must be probe-validated end-to-end (emit + parse under
# the 512-tok eval budget) before default-on. The comment describes the ROW's action_id, so
# it is always consistent with the tool call.
_LABEL_COT = os.getenv("PVP_LABEL_COT", "0").strip().lower() in ("1", "true", "yes", "on")
_LABEL_COMMENT_FNS: dict = {}

# --- action-id anchor + dual-model emit surface (format_emit_dual_model) ------------
# PVP_ACTION_ANCHOR=1 appends an OUR-worded "Committing action_id {N}." to the assistant
# `content` for the BARE-INTEGER games — the ones whose printed legal-action label IS the
# action_id integer, so the student can mint an illegal move by echoing a printed number
# (the illegal-copy forfeit: heavy on gin/liars/goofspiel). The anchor states the committed
# integer explicitly so the tool-call arg is anchored to a stated id, not a copied label.
# gin/liars/goofspiel always; leduc/othello opt-in via PVP_ANCHOR_ENVS; clobber (coordinate
# labels, not bare ints) NEVER. Default OFF -> row format byte-identical to today.
_ACTION_ANCHOR = os.getenv("PVP_ACTION_ANCHOR", "0").strip().lower() in ("1", "true", "yes", "on")
_ANCHOR_ENVS = {"gin_rummy", "liars_dice", "goofspiel"}
_ANCHOR_ENVS |= {e.strip() for e in os.getenv("PVP_ANCHOR_ENVS", "").split(",") if e.strip()}
_ANCHOR_ENVS.discard("clobber")  # coordinate labels — anchor never applies
# Drawn base family (threaded from sft_env_config's generate_cmd). Qwen2.5 renders assistant
# `content` inline BESIDE a structured tool_call; Llama-3.1 DROPS it, so a Llama/unknown draw
# must text-encode the turn so the anchor/rationale reach the (assistant-only) loss. Unset ->
# text-encode, which reproduces the pre-existing PVP_LABEL_COT surface byte-for-byte.
_BASE_FAMILY = os.getenv("PVP_BASE_FAMILY", "").strip().lower()


def _emit_text_encoded() -> bool:
    """True when the assistant turn must be TEXT-ENCODED (rationale/anchor + tool call as the
    native llama3 JSON surface in `content`) rather than emitted as a structured tool_call with
    inline content. Only a confirmed Qwen draw uses the structured+content path; Llama or an
    unknown/unset family text-encodes (safe: renders verbatim in loss on both templates)."""
    return _BASE_FAMILY != "qwen"


def _anchor_sentence(env_name: str, action_id: int) -> str:
    """OUR-worded action-id commitment, or '' when disabled or the env is not bare-integer."""
    if not _ACTION_ANCHOR or env_name not in _ANCHOR_ENVS:
        return ""
    return f"Committing action_id {action_id}."


def _label_comment(env_name: str, state_desc: str, player_id: int, action_id: int) -> str:
    """Deterministic <=140-char rationale for the row's action, or '' when disabled/unavailable.

    Consumes any ``envs.{env}_expert.get_label_comment(observation, action_id, player_id)`` the
    per-game dimensions expose (othello today; a gin/liars/goofspiel helper is picked up the
    same way if a sibling dimension adds one). Null-object fallback keeps '' when none exists.
    Gated by PVP_LABEL_COT (default OFF -> always '', behavior unchanged)."""
    if not _LABEL_COT:
        return ""
    fn = _LABEL_COMMENT_FNS.get(env_name)
    if fn is None:
        try:
            import importlib
            mod = importlib.import_module(f"envs.{env_name}_expert")
            fn = getattr(mod, "get_label_comment")
        except Exception:
            fn = lambda *a, **k: ""  # noqa: E731 - null-object: env has no rationale helper
        _LABEL_COMMENT_FNS[env_name] = fn
    try:
        return (fn(state_desc, action_id, player_id) or "").strip()[:140]
    except Exception:
        return ""


def _build_row(env_name: str, state_desc: str, player_id: int,
               legal_actions: list[tuple[int, str]], action_id: int) -> dict:
    tools = build_pvp_tools([aid for aid, _ in legal_actions])
    comment = _label_comment(env_name, state_desc, player_id, action_id)
    anchor = _anchor_sentence(env_name, action_id)
    content_text = "\n".join(p for p in (comment, anchor) if p)  # rationale then anchor, non-empty only
    structured_call = {"type": "function", "function": {
        "name": "game_action", "arguments": json.dumps({"action_id": action_id})}}
    if content_text and _emit_text_encoded():
        # Llama / unknown base. Llama-3.1's chat template DROPS assistant `content` whenever a
        # structured `tool_calls` is present (verified), so any rationale/anchor beside the tool
        # call never reaches the (assistant-only) loss. TEXT-ENCODE the whole turn as `content`:
        # the rationale+anchor, then the game_action call as the native llama3 JSON surface.
        # apply_chat_template renders `content` verbatim, so it IS in the loss; at eval the model
        # emits the text + the same JSON, which SGLang's llama3 parser extracts.
        # ⚠️ DEPLOY CAVEAT: this emit surface is NOT yet SGLang-verified — the render-probe in
        # train_sft_env is a by-shape guard; confirm on a live serve before default-on.
        tool_text = json.dumps({"name": "game_action", "parameters": {"action_id": action_id}})
        assistant = {"role": "assistant", "content": f"{content_text}\n{tool_text}"}
    elif content_text:
        # Confirmed Qwen base. Its template renders assistant `content` INLINE alongside a
        # structured tool_call, so the rationale+anchor reach loss AND the call is emitted as the
        # native <tool_call> surface the Qwen parser reads — no text-encoding needed.
        assistant = {"role": "assistant", "content": content_text, "tool_calls": [structured_call]}
    else:
        # No content to carry (anchor + CoT both off, or non-anchor env): unchanged default row.
        assistant = {"role": "assistant", "content": None, "tool_calls": [structured_call]}
    return {
        "messages": [
            {"role": "system", "content": _system_prompt(env_name)},
            {"role": "user", "content": build_user_prompt(state_desc, player_id, legal_actions)},
            assistant,
        ],
        "tools": json.dumps(tools_to_openai(tools)),
    }


# Rare-decision oversampling, DEFAULT k=4. The Qwen R1 student matched its PIMC teacher on 92% of
# decisions but on only 1 of 25 "draw the upcard" decisions in the Draw phase (cand#2): ~90% of Draw
# rows are "draw stock", so it learned the majority prior and almost never took the upcard (the move
# that completes a meld). Each teacher "draw upcard" row is written k times (capped at 8).
# VM A/B through the real entrypoint (2026-09-25, cand#5 k=4 vs cand#4 k=1, both final checkpoints):
# upcard agreement 47% -> 80% with no extra false upcards (1/224), student vs teacher 16-24 -> 20-20,
# gin h2h vs the R1 winners (the champion + a breakout rival) 54.1% -> 59.6% over 200 games.
# GIN_UPCARD_OVERSAMPLE=1 restores the old row stream. Bases under 4B default to k=2 (sft_env_config threads
# GIN_UPCARD_BASE_K from budget_controller.gin_generate_prefix): at k=4 the 3B students took 28-38 false
# upcards. An explicit GIN_UPCARD_OVERSAMPLE still wins.
_UPCARD_OVERSAMPLE_DEFAULT = "4"
_UPCARD_OVERSAMPLE_CAP = 8
# Terminal-ply oversampling: a Wall / Knock / Layoff row with a real choice (>= 2 legal ids) is written
# _GIN_TERMINAL_K times. These few rows per hand settle the score and now carry the exact labels of
# gin_rummy_expert._choose_terminal (Wall knock, layoff only when it lowers deadwood, partition melds), which
# the student must learn against the old first-meld habit. _GIN_TERMINAL_K = 1 restores the old row stream.
_GIN_TERMINAL_K = 2
_GIN_TERMINAL_PHASES = ("Wall", "Knock", "Layoff")
_RE_GIN_PHASE = re.compile(r"Phase: (\w+)")


def _row_copies(env_name: str, legal_actions: list[tuple[int, str]], action_id: int,
                state_desc: str = "") -> int:
    if env_name != "gin_rummy":
        return 1
    phase = _RE_GIN_PHASE.search(state_desc or "")
    if phase and phase.group(1) in _GIN_TERMINAL_PHASES:
        return min(max(1, _GIN_TERMINAL_K), _UPCARD_OVERSAMPLE_CAP) if len(legal_actions) >= 2 else 1
    default_k = os.getenv("GIN_UPCARD_BASE_K", "").strip() or _UPCARD_OVERSAMPLE_DEFAULT
    try:
        k = int(os.getenv("GIN_UPCARD_OVERSAMPLE", default_k).strip() or 1)
    except ValueError:
        return 1
    if k <= 1:
        return 1
    label = dict(legal_actions).get(action_id, "").lower()
    return min(k, _UPCARD_OVERSAMPLE_CAP) if ("draw" in label and "upcard" in label) else 1


def _emit_rows(rows: list[dict], env_name: str, state_desc: str, player_id: int,
               legal_actions: list[tuple[int, str]], action_id: int) -> None:
    """Append the labeled row for this decision (``_row_copies`` independent copies)."""
    row = _build_row(env_name, state_desc, player_id, legal_actions, action_id)
    rows.append(row)
    copies = _row_copies(env_name, legal_actions, action_id, state_desc)
    rows.extend(copy.deepcopy(row) for _ in range(copies - 1))


def _teacher_obs(state_desc: str, player_id: int,
                 legal_actions: list[tuple[int, str]]) -> str:
    """Normalized envelope the per-env teachers' parsers expect:
    'Current State:\\n<state>\\nYou are Player N.\\nLegal Actions:\\n<id> -> <label>...'."""
    action_lines = "\n".join(f"{aid} -> {label}" for aid, label in legal_actions)
    return (
        f"Current State:\n{state_desc}\n"
        f"You are Player {player_id}.\n"
        f"Legal Actions:\n{action_lines}\n"
        f"Your choice (ID only):"
    )


def _score(returns: list[float], player_id: int) -> float:
    """Per-seat outcome score in [0,1]: win=1.0, draw=0.5, loss=0.0."""
    me, opp = returns[player_id], returns[1 - player_id]
    if me > opp:
        return 1.0
    if me < opp:
        return 0.0
    return 0.5


# DART noise injection (Laskey et al., CoRL 2017 — "DART: Noise Injection for Robust Imitation
# Learning"), OPT-IN per env. Plain self-play only records states the TEACHER reaches, but the
# trained student drifts into states the teacher never visits after its own mistakes and then has
# no idea how to recover (covariate shift / compounding error). DART keeps every row labeled with
# the TEACHER's action but, with probability eps, continues the game from a different random legal
# move, so the recorded trajectories also cover post-mistake states — each labeled with the
# teacher's recovery move. It needs no model inference, so it runs at data-gen time inside the
# budget (fits the no-bundled-data rule), unlike DAgger's student rollouts.
# <ENV>_DART_EPS overrides PVP_DART_EPS (clamped to [0, 0.5]); unset/0 = unchanged behavior.
def _dart_eps(env_name: str) -> float:
    raw = (os.getenv(f"{env_name.upper()}_DART_EPS", "").strip()
           or os.getenv("PVP_DART_EPS", "").strip())
    if not raw:
        return 0.0
    try:
        return min(max(float(raw), 0.0), 0.5)
    except ValueError:
        return 0.0


def _dart_executed_action(label_action: int, legal_ids: list[int], eps: float,
                          rng: "random.Random") -> int:
    """The move actually played: the teacher's label, or with probability ``eps`` another legal
    move. ``eps`` is checked first so the default (0) never touches ``rng`` — the rng stream (and
    the teacher's own sampling that shares it) stays byte-identical when DART is off."""
    if eps > 0.0 and len(legal_ids) > 1 and rng.random() < eps:
        return rng.choice([a for a in legal_ids if a != label_action])
    return label_action


def play_selfplay_game(env_name: str, seed: int, max_turn: int,
                       rng: "random.Random | None" = None):
    """Play one teacher-vs-teacher game locally via pyspiel.

    Returns a list of (examples, score) views — one per seat — where examples is
    that seat's per-turn training rows and score is its terminal outcome in
    [0,1]. On an unfinished game (max_turn hit) scores default to 0.5 (draw) so
    score filters neither strongly keep nor drop it.
    """
    if rng is None:
        rng = random.Random(seed)
    formatter = _FORMATTERS[env_name]
    config_id = _config_id_for_seed(seed, env_name)
    game = _load_game(env_name, config_id)
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, seed)
    # Env-gated: <ENV>_TEACHER=mcts distils OpenSpiel MCTS play into the perfect-info envs
    # (othello/clobber) where our hand-written teachers lose ~90% to MCTS. None = unchanged.
    strong_bot = _openspiel_mcts_teacher(env_name, game, seed)
    # Import the hand-written teacher only when we actually need it — strong_bot picks moves
    # itself, so the MCTS-teacher path won't pull in unused per-env deps.
    choose_action = _teacher(env_name) if strong_bot is None else None
    dart_eps = _dart_eps(env_name)

    per_seat: dict[int, list[dict]] = {0: [], 1: []}
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue

        if state.is_simultaneous_node():
            # Both players choose at once (e.g. goofspiel sealed bids). Harvest a
            # training row for EACH seat from its own view, then apply the joint
            # action. One simultaneous round counts as one turn.
            joint: list[int] = []
            for p in range(game.num_players()):
                p_legal = state.legal_actions(p)
                if not p_legal:
                    # A seat with no legal action takes pyspiel's no-op (0); no row.
                    joint.append(0)
                    continue
                p_labels = [(a, state.action_to_string(p, a)) for a in p_legal]
                p_state_desc = formatter(state, p)
                try:
                    p_action = choose_action(
                        _teacher_obs(p_state_desc, p, p_labels), p, p_legal, rng=rng)
                except Exception:
                    p_action = rng.choice(p_legal)
                if p_action not in p_legal:
                    p_action = rng.choice(p_legal)
                per_seat[p].append(_build_row(env_name, p_state_desc, p, p_labels, p_action))
                joint.append(_dart_executed_action(p_action, p_legal, dart_eps, rng))
            state.apply_actions(joint)
            turns += 1
            continue

        cp = state.current_player()
        if cp < 0:  # invalid (e.g. terminal/unexpected) — these games are sequential
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
        state_desc = formatter(state, cp)
        try:
            if strong_bot is not None:
                action_id = int(strong_bot.step(state))  # OpenSpiel MCTS move (state-aware)
            else:
                action_id = choose_action(_teacher_obs(state_desc, cp, labels), cp, legal_ids, rng=rng)
        except Exception:
            action_id = rng.choice(legal_ids)
        if action_id not in legal_ids:
            action_id = rng.choice(legal_ids)
        _emit_rows(per_seat[cp], env_name, state_desc, cp, labels, action_id)
        state.apply_action(_dart_executed_action(action_id, legal_ids, dart_eps, rng))
        turns += 1

    if state.is_terminal():
        returns = state.returns()
        scores = {0: _score(returns, 0), 1: _score(returns, 1)}
    else:
        scores = {0: 0.5, 1: 0.5}
    return [(per_seat[p], scores[p]) for p in (0, 1) if per_seat[p]]


# RANK-10 eval-opponent diversification (all default OFF => committed behaviour unchanged). A per-game
# opponent-strength sample + a handful of random opening plies make the eval-aligned data GENERALISE
# across the eval's sim range / opening distribution instead of overfitting one fixed depth.
_OPP_OPENING_PLIES_CAP = int(os.getenv("PVP_OPP_OPENING_PLIES_CAP", "6"))  # hard cap on random opening plies


def _sample_opp_sims(env_name: str, seed: int) -> int:
    """Per-game eval-opponent MCTS sim count. When <ENV>_OPP_MCTS_SIMS_MIN and _MAX are both set
    (MAX>=MIN>0) sample uniformly in [MIN, MAX] with a seed-derived rng (deterministic per game);
    otherwise return the single <ENV>_OPP_MCTS_SIMS (default 25) = byte-identical prior behaviour.
    Never raises."""
    up = env_name.upper()
    try:
        fixed = int(os.getenv(f"{up}_OPP_MCTS_SIMS", "25"))
    except ValueError:
        fixed = 25
    lo_raw = os.getenv(f"{up}_OPP_MCTS_SIMS_MIN")
    hi_raw = os.getenv(f"{up}_OPP_MCTS_SIMS_MAX")
    if lo_raw in (None, "") or hi_raw in (None, ""):
        return fixed
    try:
        lo, hi = int(lo_raw), int(hi_raw)
    except ValueError:
        return fixed
    if lo <= 0 or hi < lo:
        return fixed
    return random.Random((seed ^ 0x51357) & 0x7FFFFFFF).randint(lo, hi)


def _apply_random_opening_plies(env_name: str, state: "pyspiel.State", seed: int) -> None:
    """Apply up to <ENV>_OPP_OPENING_PLIES seed-derived random legal plies (hard-capped at
    _OPP_OPENING_PLIES_CAP) to ``state`` in place before labeling. Chance nodes are resolved by their
    outcome distribution; a terminal/simultaneous/no-legal state stops early. Unset/0 => no-op (prior
    behaviour, and othello's own _setup_initial_state opening plies are untouched). Never raises — any
    failure leaves the state as-is."""
    up = env_name.upper()
    try:
        n = int(os.getenv(f"{up}_OPP_OPENING_PLIES", "0"))
    except ValueError:
        n = 0
    if n <= 0:
        return
    n = min(n, _OPP_OPENING_PLIES_CAP)
    rng = random.Random((seed ^ 0x0DDBA11) & 0x7FFFFFFF)
    try:
        for _ in range(n):
            if state.is_terminal():
                break
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
                continue
            if state.is_simultaneous_node():
                break  # simultaneous games take the self-play fallback earlier anyway
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(rng.choice(legal))
    except Exception:
        return


# Goofspiel records our teacher on one seat vs a seeded rule-opponent pool (envs/goofspiel_hero.py), the
# default route for its generator. False = the 13759ee mirror self-play (one-line rollback).
_GOOFSPIEL_HERO_POOL = True


def play_vs_eval_game(env_name: str, seed: int, max_turn: int,
                      rng: "random.Random | None" = None):
    """ASYMMETRIC gen: OUR teacher plays ONE seat (recorded) — the STRONG OpenSpiel MCTS teacher for
    the perfect-info differentiators (clobber; via <ENV>_TEACHER), else the hand teacher — while a
    separate OpenSpiel MCTSBot at the eval's ~25 sims (the actual eval-opponent FAMILY) plays the other
    seat (NOT recorded). So the student is distilled to BEAT the real eval opponent with a strong move
    on every recorded state, not to mirror itself (the champion self-played; the eval is vs MCTS ~25). The recorded seat alternates by seed (mirrors the eval's seat-swap) for
    balance. Same ``[(examples, score)]`` return shape as play_selfplay_game, but ONE view (our
    seat). Opponent strength via <ENV>_OPP_MCTS_SIMS (default 25 = the eval opponent) /
    <ENV>_OPP_MCTS_ROLLOUTS (default 1). Falls back to self-play if open_spiel is unavailable or the
    game is simultaneous, so it can never crash generation. OPT-IN via <ENV>_OPP_MCTS=1 — SMOKE-GATE
    it: training vs one opponent can help (targets the metric) or overfit; A/B student vs MCTS-25."""
    if rng is None:
        rng = random.Random(seed)
    if env_name == "goofspiel" and _GOOFSPIEL_HERO_POOL:
        from envs.goofspiel_hero import play_goofspiel_hero_game  # lazy: that module imports this one
        return play_goofspiel_hero_game(seed, max_turn, rng)
    config_id = _config_id_for_seed(seed, env_name)
    game = _load_game(env_name, config_id)
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS or env_name == "goofspiel":
        # MCTSBot can't drive simultaneous games, and eval goofspiel is a turn-based imperfect-info wrap
        # of one (an MCTS opponent would read the hidden bid) — asymmetric mode is undefined; self-play.
        return play_selfplay_game(env_name, seed, max_turn, rng)
    # RANK-10 (default OFF): sample the eval-opponent strength per game from [MIN, MAX] (seed-derived,
    # deterministic) instead of the fixed 25, so the distilled data generalises across the eval's sim
    # range (e.g. othello 25-75, clobber 50-100) rather than overfitting one depth. Unset MIN/MAX =>
    # the single <ENV>_OPP_MCTS_SIMS value (default 25) = byte-identical prior behaviour.
    opp_sims = _sample_opp_sims(env_name, seed)
    opp_rollouts = int(os.getenv(f"{env_name.upper()}_OPP_MCTS_ROLLOUTS", "1"))
    opp_bot = _make_mcts_bot(game, seed ^ 0x5EED, opp_sims, opp_rollouts)
    if opp_bot is None:
        return play_selfplay_game(env_name, seed, max_turn, rng)  # open_spiel missing → self-play

    formatter = _FORMATTERS[env_name]
    # RECORDED seat = the STRONG teacher (OpenSpiel MCTS for the perfect-info differentiators like
    # clobber, via <ENV>_TEACHER; mirrors play_selfplay_game:418-421). This path previously hard-routed
    # the recorded seat to the WEAK hand teacher, so <ENV>_OPP_MCTS=1 distilled a weak policy AGAINST the
    # MCTS-25 opponent — DOWNGRADING clobber instead of upgrading it. Building strong_bot here combines
    # champion levers 3+4 (a strong search teacher ON the recorded seat, playing vs the MCTS-25 eval
    # opponent) for the first time. strong_bot is None (→ hand teacher) when <ENV>_TEACHER!=mcts or the
    # game is simultaneous, so non-MCTS envs (e.g. leduc/liars) keep their hand teacher unchanged.
    strong_bot = _openspiel_mcts_teacher(env_name, game, seed)
    choose_action = _teacher(env_name) if strong_bot is None else None
    our_seat = seed % 2  # alternate the recorded (our-teacher) seat by game, like the eval seat-swap
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, seed)
    # RANK-10 (default OFF): apply a handful of seed-derived RANDOM opening plies before the teacher
    # starts labeling, so recorded states are drawn from a wider, eval-like opening distribution
    # (capped, so we never random-play the whole game away). Unset/0 => no extra plies (unchanged).
    _apply_random_opening_plies(env_name, state, seed)

    our_rows: list[dict] = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        if cp == our_seat:  # OUR teacher moves — record the row
            labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
            state_desc = formatter(state, cp)
            try:
                if strong_bot is not None:
                    action_id = int(strong_bot.step(state))  # strong MCTS teacher (state-aware), mirrors play_selfplay_game:468
                else:
                    action_id = choose_action(_teacher_obs(state_desc, cp, labels), cp, legal_ids, rng=rng)
            except Exception:
                action_id = rng.choice(legal_ids)
            if action_id not in legal_ids:
                action_id = rng.choice(legal_ids)
            _emit_rows(our_rows, env_name, state_desc, cp, labels, action_id)
        else:  # eval-opponent MCTS moves — NOT recorded
            try:
                action_id = int(opp_bot.step(state))
            except Exception:
                action_id = rng.choice(legal_ids)
            if action_id not in legal_ids:
                action_id = rng.choice(legal_ids)
        state.apply_action(action_id)
        turns += 1

    score = _score(state.returns(), our_seat) if state.is_terminal() else 0.5
    return [(our_rows, score)] if our_rows else []


# leduc_poker hero-seat vs a DIVERSE opponent pool (leduc v2, 2026-09-30). Two never-folders
# (teacher-vs-teacher mirror self-play) score an uninformative exact 50% against each other under
# WIN/LOSS scoring (see leduc_poker_expert._aggro_abstract's docstring) -- rank separation only
# comes from extracting folds out of a FOLDING opponent, so the recorded seat needs a genuinely
# different opponent, not a mirror of itself. Weights are RELATIVE and env-tunable
# (LEDUC_POOL_WEIGHTS, "kind:weight,..."); 'mcts' silently redistributes to 'wl_cfr' when
# open_spiel's MCTS submodule is unavailable (never crashes generation).
_LEDUC_POOL_DEFAULT_WEIGHTS: "dict[str, float]" = {
    "wl_cfr": 0.35, "chips_cfr": 0.35, "mcts": 0.15, "random": 0.15,
}
_LEDUC_POOL_KNOWN_KINDS = ("wl_cfr", "chips_cfr", "mcts", "random")


def _leduc_pool_weights() -> "dict[str, float]":
    """Parse LEDUC_POOL_WEIGHTS ("kind:weight,..."). Falls back to the defaults on any parse
    error, an unknown kind, a negative weight, or a total of zero (all of which would otherwise
    make random.choices raise inside every game) instead of crashing generation."""
    raw = os.getenv("LEDUC_POOL_WEIGHTS", "").strip()
    if not raw:
        return _LEDUC_POOL_DEFAULT_WEIGHTS
    try:
        out = {}
        for part in raw.split(","):
            k, v = part.split(":")
            k = k.strip()
            w = float(v)
            if k not in _LEDUC_POOL_KNOWN_KINDS or w < 0.0:
                return _LEDUC_POOL_DEFAULT_WEIGHTS
            out[k] = w
        return out if out and sum(out.values()) > 0.0 else _LEDUC_POOL_DEFAULT_WEIGHTS
    except Exception:
        return _LEDUC_POOL_DEFAULT_WEIGHTS


def _leduc_pool_kind(seed: int) -> str:
    weights = _leduc_pool_weights()
    kinds = list(weights.keys())
    r = random.Random((seed ^ 0x1EDC0) & 0x7FFFFFFF)
    return r.choices(kinds, weights=[weights[k] for k in kinds], k=1)[0]


def play_leduc_hero_game(seed: int, max_turn: int, rng: "random.Random | None" = None):
    """HERO-SEAT generation for leduc_poker (leduc v2): our teacher (LEDUC_TEACHER, default
    'aggro' -- deterministic never-fold, raise-whenever-legal; see leduc_poker_expert.py) plays ONE
    seat (recorded) against a seed-derived DIVERSE opponent pool -- a W/L-objective CFR+ sample, a
    chips-objective CFR+ sample (a genuine chip-optimal folder), an OpenSpiel MCTS-50 bot when
    open_spiel's mcts submodule is importable, or uniform-random legal -- instead of the symmetric
    teacher-mirror self-play play_selfplay_game falls back to (which draws two never-folders to an
    uninformative exact 50%). Hero seat alternates by seed (mirrors the eval seat-swap). Only the
    hero's rows are recorded. Same ``[(examples, score)]`` return shape as play_selfplay_game /
    play_vs_eval_game. Falls back to play_selfplay_game on any setup failure, so generation can
    never crash. ROLLBACK: LEDUC_HERO_POOL=0 restores the prior symmetric self-play / vs-eval-MCTS
    dispatch in make_selfplay_generator -- byte-identical to before leduc v2 only when LEDUC_TEACHER is
    also reverted to 'cfr' (LEDUC_HERO_POOL alone still self-plays the new default 'aggro' label,
    just via the mirrored teacher-vs-teacher path instead of the pool)."""
    if rng is None:
        rng = random.Random(seed)
    env_name = "leduc_poker"
    try:
        from envs.leduc_poker_expert import choose_action as hero_choose, pool_action
    except Exception:
        return play_selfplay_game(env_name, seed, max_turn, rng)

    config_id = _config_id_for_seed(seed, env_name)
    game = _load_game(env_name, config_id)
    formatter = _FORMATTERS[env_name]
    state = game.new_initial_state()
    _setup_initial_state(env_name, state, seed)

    hero_seat = seed % 2  # alternate the recorded seat by game, like the eval seat-swap
    kind = _leduc_pool_kind(seed)
    opp_bot = None
    if kind == "mcts":
        try:  # a bad value must not raise in every game
            sims = max(1, int(os.getenv("LEDUC_POOL_MCTS_SIMS", "50")))
        except ValueError:
            sims = 50
        opp_bot = _make_mcts_bot(game, seed ^ 0x5EED, sims)
        if opp_bot is None:
            kind = "wl_cfr"  # open_spiel.mcts unavailable -> redistribute within the pool, never crash

    hero_rows: list[dict] = []
    turns = 0
    while not state.is_terminal() and turns < max_turn:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
            continue
        cp = state.current_player()
        if cp < 0:
            break
        legal_ids = state.legal_actions(cp)
        if not legal_ids:
            break
        labels = [(a, state.action_to_string(cp, a)) for a in legal_ids]
        state_desc = formatter(state, cp)
        if cp == hero_seat:  # OUR teacher moves -- record the row
            try:
                action_id = hero_choose(_teacher_obs(state_desc, cp, labels), cp, legal_ids, rng=rng)
            except Exception:
                action_id = rng.choice(legal_ids)
            if action_id not in legal_ids:
                action_id = rng.choice(legal_ids)
            _emit_rows(hero_rows, env_name, state_desc, cp, labels, action_id)
        else:  # pool opponent moves -- NOT recorded
            try:
                if kind == "mcts" and opp_bot is not None:
                    action_id = int(opp_bot.step(state))
                else:
                    action_id = pool_action(kind, _teacher_obs(state_desc, cp, labels), cp, legal_ids, rng=rng)
            except Exception:
                action_id = rng.choice(legal_ids)
            if action_id not in legal_ids:
                action_id = rng.choice(legal_ids)
        state.apply_action(action_id)
        turns += 1

    score = _score(state.returns(), hero_seat) if state.is_terminal() else 0.5
    return [(hero_rows, score)] if hero_rows else []


# ROLLBACK: LEDUC_HERO_POOL=0 (or flip _LEDUC_HERO_POOL_DEFAULT back to False) restores the prior
# symmetric self-play / vs-eval-MCTS dispatch in make_selfplay_generator, byte-identical to before
# leduc v2 -- the same one-line-constant rollback shape as leduc_poker_expert._LEDUC_TEACHER_DEFAULT.
_LEDUC_HERO_POOL_DEFAULT = True


def _leduc_hero_pool_on() -> bool:
    v = os.getenv("LEDUC_HERO_POOL")
    if v is None or v.strip() == "":
        return _LEDUC_HERO_POOL_DEFAULT
    return v.strip().lower() not in ("0", "false", "no", "off")


# Teacher-vs-eval-MCTS generation for every game env (2026-09-21: the top miners' public training logs
# all generate that way, not via self-play).
# PVP_ALL_OPP_MCTS=1 flips the DEFAULT for every game to the eval-aligned generator
# (play_vs_eval_game) in one switch; a per-env <ENV>_OPP_MCTS still wins ("1" force-on, "0"
# force-off; goofspiel takes play_vs_eval_game's hero-pool route by default, see
# _GOOFSPIEL_HERO_POOL). PVP_ALL_OPP_MCTS itself is OFF by default; clobber, othello, leduc_poker,
# liars_dice and goofspiel take their own default hero-seat / opponent-pool generators (make_selfplay_generator).
# ⚠️ Validate on Qwen2.5-7B before flipping the default — this + volume flood + LR ~4.3e-5 + 2GPU
# batch-128 is the confirmed winning recipe we did NOT run (we only ever wired OTHELLO_OPP_MCTS).
_ALL_OPP_MCTS = os.getenv("PVP_ALL_OPP_MCTS", "").strip() == "1"
# Envs generated as hero-seat rows vs an opponent pool with both outcomes kept (clobber_hero.py,
# othello_hero.py); dropping an env from the tuple restores its 13759ee symmetric self-play path below
# (othello's 13759ee label also needs _OTHELLO_TEACHER_DEFAULT = "expert").
_HERO_SEAT_DEFAULT_ENVS = ("clobber", "othello")


def make_selfplay_generator(env_name: str):
    """Return a generator with the SFT generator signature
    ``fn(game_id, env_endpoint, max_turn) -> views``. ``env_endpoint`` is ignored
    (self-play needs no env server); kept for drop-in compatibility with the
    env-server generators in _SFT_REGISTRY.

    Teacher-vs-eval-MCTS (play_vs_eval_game) vs symmetric self-play (play_selfplay_game) is
    chosen per env: <ENV>_OPP_MCTS=1 forces it on, =0 forces it off, and when unset it follows
    the global PVP_ALL_OPP_MCTS (default OFF = self-play). Five games default to their own
    hero-seat / opponent-pool generators instead: clobber (clobber_hero), othello (othello_hero),
    leduc_poker (below), liars_dice (liar_dice_pool) and goofspiel (play_vs_eval_game's hero-pool route).

    leduc_poker takes its own DEFAULT-ON path (leduc v2): hero-seat vs a diverse opponent pool
    (play_leduc_hero_game) instead of this vs_eval/self-play choice. LEDUC_HERO_POOL=0 opts back
    into the dispatch below, unchanged."""
    if env_name in _HERO_SEAT_DEFAULT_ENVS:
        if env_name == "othello":
            from envs.othello_hero import play_hero_game
        else:
            from envs.clobber_hero import play_hero_game

        def _hero_gen(game_id: int, env_endpoint: str | None, max_turn: int):
            return play_hero_game(game_id, max_turn, rng=random.Random(game_id))

        return _hero_gen
    if env_name == "leduc_poker" and _leduc_hero_pool_on():
        def _leduc_gen(game_id: int, env_endpoint: str | None, max_turn: int):
            return play_leduc_hero_game(seed=game_id, max_turn=max_turn, rng=random.Random(game_id))

        return _leduc_gen

    per_env = os.getenv(f"{env_name.upper()}_OPP_MCTS", "").strip()
    vs_eval = per_env == "1" or (per_env != "0" and _ALL_OPP_MCTS)
    if env_name == "liars_dice" and not vs_eval:
        from envs import liar_dice_pool  # hero vs a field of styles, see that module

        if liar_dice_pool.pool_on():
            return liar_dice_pool.generate
    if env_name == "goofspiel" and _GOOFSPIEL_HERO_POOL and per_env != "0":
        vs_eval = True  # play_vs_eval_game hands goofspiel to the hero-seat rule-pool game

    def _gen(game_id: int, env_endpoint: str | None, max_turn: int):
        play = play_vs_eval_game if vs_eval else play_selfplay_game
        return play(env_name, seed=game_id, max_turn=max_turn, rng=random.Random(game_id))

    return _gen
