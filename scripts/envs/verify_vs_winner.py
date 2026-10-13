"""GOLD-STANDARD measurement: OUR R1 model vs the R1 WINNER's model, head-to-head in
clobber/othello, via local pyspiel — proxy-free, unlike verify_teacher_vs_mcts.

The tournament EVAL is itself PvP over local pyspiel (see pvp_selfplay's docstring:
"Eval drives games with local pyspiel, NOT the mcts-api server"), so playing our uploaded
model vs the winner's uploaded model here REPLICATES the real eval directly. Each move uses
the EVAL-IDENTICAL prompt (pvp_selfplay._system_prompt + pvp_format.build_user_prompt +
build_pvp_tools), so both models see exactly what they saw at eval time.

Needs a GPU + torch/transformers/peft (loads two 8B models — fits on an 80GB H100). It PEEKS
the first 2 model interactions (prompt tail + raw output + parsed action) so you can confirm the
models are really generating game tool-calls (not random) before trusting the win-rate.

Run on the VM from scripts/ (install deps first: `uv pip install torch transformers peft accelerate`):
    python envs/verify_vs_winner.py --env clobber --games 20
    python envs/verify_vs_winner.py --env othello --games 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import types

import pyspiel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # silence per-generate max_new_tokens/max_length + HF_TOKEN spam
# max_new_tokens=64 already governs generation; drop the config's 131072 max_length so it can't warn.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Make envs.* importable when run as a file (same bootstrap as the other verify scripts).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg

from envs.pvp_selfplay import _system_prompt, _FORMATTERS, _load_game, _teacher, _teacher_obs
from envs.pvp_format import build_pvp_tools, build_user_prompt, sglang_template_tools, tools_to_openai


def _make_mcts_bot(game, sims: int, seed: int, rollouts: int = 1, uct_c: float = 2.0):
    """OpenSpiel MCTSBot — used with --our-mcts to play OUR side as the raw MCTS TEACHER (not our
    trained model) vs the winner's model. If the teacher beats #2 but our model doesn't, the gap
    is DISTILLATION (train harder); if the teacher also loses, the TEACHER is the ceiling."""
    import numpy as np
    from open_spiel.python.algorithms import mcts as _mcts
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    evaluator = _mcts.RandomRolloutEvaluator(n_rollouts=rollouts, random_state=rng)
    return _mcts.MCTSBot(game, uct_c=uct_c, max_simulations=sims, evaluator=evaluator, random_state=rng)

_BASE = "unsloth/Meta-Llama-3.1-8B-Instruct"
_TOURNEY = ("gradients-io-tournaments/tournament-tourn_c386abdef131e120_20260727-"
            "723a4832-dbe4-450a-b453-5dd385bfae25")
_OURS_DEFAULT = f"{_TOURNEY}-5D7iEJm5"     # our R1 upload (full fine-tune -> full model)
_WINNER_DEFAULT = f"{_TOURNEY}-5EEaxgnm"   # the R1 winner's upload (LoRA adapter on the base)

# Per-game CHAMPION opponents, measured from tournament task 5ccbf224 (Group-Stage 2, SAME tournament,
# 4 envs: intercode+goofspiel+othello+clobber). The R1 meta-lesson = validate vs the STRONGEST opponent,
# never a weak one. The task's head-to-heads prove the champ DIFFERS by game: at OTHELLO the R2 Boss
# (5GU4Xkd3) BEAT 5EEa 13-7, so 5EEa is NOT the othello champ — the Boss is; at CLOBBER 5EEa IS the champ
# (beat the Boss 15-9). Default --winner-model per env to the measured champ so an othello run is validated
# against the real othello king, not a model that itself loses othello.
_TASK_5CCBF224 = ("gradients-io-tournaments/tournament-tourn_c386abdef131e120_20260727-"
                  "5ccbf224-23a0-4d12-adeb-7023e07ed06e")
# Current tournament (Qwen2.5-7B, 2026-08-03): 5EEaxgnm won R1 (leduc+liars) AND R2 (liars+goofspiel+
# leduc+intercode) — a 3-peat champ. Its R2 model trained on all 4 R2 envs, so it's the leduc/liars opponent.
_TASK_R2_QWEN = ("gradients-io-tournaments/tournament-tourn_590e311a35f6a234_20260803-"
                 "72d1502c-b5d0-4fc0-a98a-07d438ae7a3e-5EEaxgnm")
_WINNER_BY_ENV = {
    "othello": f"{_TASK_5CCBF224}-5GU4Xkd3",  # R2 Boss = the measured OTHELLO champ (beat 5EEa 13-7)
    "clobber": _WINNER_DEFAULT,               # 5EEa = the measured CLOBBER champ (beat the Boss 15-9)
    "goofspiel": f"{_TASK_5CCBF224}-5FjRh74Y",  # 5FjR = the measured GOOFSPIEL champ (beat 5EEa 16-4)
    "leduc_poker": _TASK_R2_QWEN,   # 5EEaxgnm R2 (Qwen) — current-tourney champ, trained on leduc
    "liars_dice": _TASK_R2_QWEN,    # 5EEaxgnm R2 (Qwen) — current-tourney champ, trained on liars
}

_PEEK = [2]  # print this many model interactions for sanity, then go quiet


def _is_lora_repo(repo: str) -> bool:
    """True if the model is a PEFT/LoRA adapter (has adapter_config.json). Handles a LOCAL dir (a
    freshly-trained checkpoint — `--our-model outputs/.../repo`) as well as an HF repo: both our R1
    model and the winner's are LoRA adapters on the same base, and env-SFT saves LoRA locally too."""
    if os.path.isdir(repo):
        return os.path.exists(os.path.join(repo, "adapter_config.json"))
    try:
        from huggingface_hub import HfApi
        return any(s.rfilename == "adapter_config.json" for s in HfApi().model_info(repo).siblings)
    except Exception:
        return False


def _load(repo: str, base: str):
    """Load a tournament model — auto-detects LoRA adapter (base + merged adapter) vs full model."""
    is_lora = _is_lora_repo(repo)
    tok = AutoTokenizer.from_pretrained(base if is_lora else repo)
    if is_lora:
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, repo).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok, ("LoRA" if is_lora else "full")


# STRICT format check (default ON; --loose restores the old lenient scoring). The validator serves the
# model with SGLang's family tool-call parser and a player that does not commit a legal game_action this
# turn loses the game by forfeit (core/pvp game_eval: "did not commit a legal move this turn").
# The lenient path below still plays SOME legal move (any legal integer, else random), which hides
# format regressions. _strict_action mirrors the detectors (llama3 = output starts with '{' or carries
# <|python_tag|>, then ';'-separated JSON calls; qwen25 = exact '<tool_call>\n{...}\n</tool_call>' blocks)
# and, like the validator, commits the FIRST LEGAL game_action among the calls (memory calls may precede it).
_STRICT = [os.getenv("VERIFY_STRICT", "1").strip() != "0"]
_SGLANG_TOOLS = os.getenv("EVAL_SGLANG_TOOLS", "1").strip() != "0"
_LAST_STRICT = [None]      # strict parse of the most recent _choose_action call (None = would forfeit)
_FORMAT_FAILS: dict = {}   # side -> [strict failures, calls]
_END_TOKENS = ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>")


class _Forfeit(Exception):
    def __init__(self, side: str):
        super().__init__(side)
        self.side = side


def _family(tok) -> str:
    tmpl = getattr(tok, "chat_template", "") or ""
    return "qwen" if "<|im_start|>" in tmpl else "llama"


def _tool_calls(text: str, family: str) -> list:
    """Every tool call SGLang's detector would extract, in order (unparseable ones skipped).
    llama3 (Llama32Detector): only when the text starts with '{' or carries <|python_tag|>; then
    consecutive JSON objects separated by ';' / whitespace. qwen25 (Qwen25Detector): every exact
    '<tool_call>\\n{...}\\n</tool_call>' block."""
    calls = []
    if family == "llama":
        if "<|python_tag|>" in text:
            text = text.split("<|python_tag|>", 1)[1]
        elif not text.startswith("{"):
            return calls
        dec, i = json.JSONDecoder(), 0
        while i < len(text):
            while i < len(text) and text[i] in " \t\r\n;":
                i += 1
            if i >= len(text) or text[i] != "{":
                break
            try:
                obj, i = dec.raw_decode(text, i)
            except ValueError:
                break
            if isinstance(obj, dict):
                calls.append(obj)
    else:
        for block in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.S):
            try:
                obj = json.loads(block.strip())
            except ValueError:
                continue
            if isinstance(obj, dict):
                calls.append(obj)
    return calls


def _strict_action(raw: str, family: str, legal=None) -> "int | None":
    """First LEGAL game_action among the parsed calls (the validator runs memory calls and commits the
    first legal game_action); None = the turn would forfeit."""
    text = raw
    for t in _END_TOKENS:
        text = text.replace(t, "")
    for obj in _tool_calls(text, family):
        if obj.get("name") != "game_action":
            continue
        args = obj.get("parameters") or obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                continue
        aid = args.get("action_id") if isinstance(args, dict) else None
        if isinstance(aid, bool) or not isinstance(aid, (int, str)):
            continue
        try:
            aid = int(aid)
        except ValueError:
            continue
        if legal is None or aid in legal:
            return aid
    return None


def _choose_action(model, tok, env_name: str, state, player_id: int, rng: random.Random) -> int:
    """Ask the model for a legal move via the eval-identical tool-call prompt; parse action_id.
    Falls back to a random legal action if the output is unparseable (game keeps going)."""
    legal = state.legal_actions(player_id)
    labels = [(a, state.action_to_string(player_id, a)) for a in legal]
    state_desc = _FORMATTERS[env_name](state, player_id)
    tools = tools_to_openai(build_pvp_tools([aid for aid, _ in labels]))
    if _SGLANG_TOOLS:  # the eval server (SGLang v0.5.14) renders tools as Tool.model_dump()
        tools = sglang_template_tools(tools)
    messages = [
        {"role": "system", "content": _system_prompt(env_name)},
        {"role": "user", "content": build_user_prompt(state_desc, player_id, labels)},
    ]
    inputs = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # 64 is enough for a bare game_action tool call; a CoT model (PVP_LABEL_COT rows)
        # emits a one-sentence rationale first, so raise to the real eval budget
        # (PVP_TURN_MAX_TOKENS=512) via EVAL_MAX_NEW_TOKENS so the tool call is not truncated.
        out = model.generate(**inputs, max_new_tokens=int(os.getenv("EVAL_MAX_NEW_TOKENS", "512")),
                             do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    strict = _strict_action(tok.decode(out[0][prompt_len:], skip_special_tokens=False), _family(tok), legal)
    _LAST_STRICT[0] = strict if strict in legal else None

    action = None
    m = re.search(r'action_id"?\s*[:=]\s*"?(-?\d+)', text)
    if m and int(m.group(1)) in legal:
        action = int(m.group(1))
    if action is None:
        for n in re.findall(r"-?\d+", text):
            if int(n) in legal:
                action = int(n)
                break
    if action is None:
        action = rng.choice(legal)

    if _PEEK[0] > 0:
        _PEEK[0] -= 1
        prompt_tail = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)[-350:]
        print(f"\n[peek] {env_name} player {player_id}  legal={legal}\n"
              f"  PROMPT(tail): ...{prompt_tail}\n  OUTPUT: {text!r}\n  -> action {action}\n", flush=True)
    return action


def _random_opening(state, plies: int, rng: random.Random) -> None:
    """Play `plies` seeded-random legal moves so each game starts from a DIFFERENT position.
    Clobber/othello are deterministic from the fixed start, so without this two greedy models
    replay the same game every time and the SEAT (not skill) decides the winner — the same
    position-variety trick verify_clobber_teacher.py uses to make the measurement meaningful."""
    for _ in range(plies):
        if state.is_terminal():
            return
        legal = state.legal_actions()
        if not legal:
            return
        state.apply_action(rng.choice(legal))


def _is_simultaneous(game) -> bool:
    return game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS


def _play_one(env_name, game, ours, winner, our_first: bool, seed: int, opening: int = 6,
              our_mcts_sims: int = 0, our_mcts_rollouts: int = 1, our_uct: float = 2.0,
              our_teacher=None,
              winner_mcts_sims: int = 0, winner_mcts_rollouts: int = 1, winner_uct: float = 2.0) -> str:
    """One full game. OUR side = our model, OR the raw OpenSpiel MCTS TEACHER (if our_mcts_sims>0),
    OR the hand-written EXPERT teacher (if our_teacher given); the winner's side is always the
    winner's model. Handles sequential, CHANCE (goofspiel prize order), and SIMULTANEOUS
    (goofspiel sealed bids) nodes. Returns 'ours'/'winner'/'draw'."""
    state = game.new_initial_state()
    rng = random.Random(seed)
    # Random opening ONLY for DETERMINISTIC perfect-info games (othello/clobber) — they need position
    # variety. Chance games (leduc/liars — the deal randomizes for free) and simultaneous games
    # (goofspiel) skip it: a random opening there would corrupt the deal / play random bids.
    _gt = game.get_type()
    if (opening > 0
            and _gt.dynamics != pyspiel.GameType.Dynamics.SIMULTANEOUS
            and _gt.chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC):
        _random_opening(state, opening, rng)
    our_bot = (_make_mcts_bot(game, our_mcts_sims, seed + 555, our_mcts_rollouts, our_uct)
               if our_mcts_sims > 0 else None)
    # opponent = OpenSpiel MCTS (the tournament's scoring opponent family) when winner_mcts_sims>0,
    # else the winner's model. Used to eval OUR trained model vs MCTS without a (base-incompatible)
    # champ model — e.g. leduc/liars whose champ is a Qwen adapter that can't load on a Llama base.
    winner_bot = (_make_mcts_bot(game, winner_mcts_sims, seed + 777, winner_mcts_rollouts, winner_uct)
                  if winner_mcts_sims > 0 else None)
    our_seat = 0 if our_first else 1

    def pick(p: int) -> int:
        """Choose seat p's action: our teacher/MCTS/model on our seat, else the winner's model/MCTS."""
        legal = state.legal_actions(p)
        if not legal:
            return 0
        if p == our_seat and our_teacher is not None:
            labels = [(a, state.action_to_string(p, a)) for a in legal]
            obs = _teacher_obs(_FORMATTERS[env_name](state, p), p, labels)
            a = our_teacher(obs, p, legal, rng=rng)
        elif p == our_seat and our_bot is not None:
            a = int(our_bot.step(state))
        elif p != our_seat and winner_bot is not None:
            a = int(winner_bot.step(state))
        else:
            model, tok = ours if p == our_seat else winner
            a = _choose_action(model, tok, env_name, state, p, rng)
            side = "ours" if p == our_seat else "winner"
            tally = _FORMAT_FAILS.setdefault(side, [0, 0])
            tally[1] += 1
            if _LAST_STRICT[0] is None:
                tally[0] += 1
                if _STRICT[0]:
                    raise _Forfeit(side)
            else:
                a = _LAST_STRICT[0]
        return a if a in legal else rng.choice(legal)

    try:
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(rng.choices(list(actions), weights=list(probs), k=1)[0])
                continue
            if state.is_simultaneous_node():
                state.apply_actions([pick(p) for p in range(game.num_players())])
                continue
            cp = state.current_player()
            if cp < 0:
                break
            state.apply_action(pick(cp))
    except _Forfeit as f:  # validator rule: no legal committed move this turn = the game is lost
        return "winner" if f.side == "ours" else "ours"
    returns = state.returns()
    if returns[our_seat] > returns[1 - our_seat]:
        return "ours"
    if returns[our_seat] < returns[1 - our_seat]:
        return "winner"
    return "draw"


def main() -> None:
    p = argparse.ArgumentParser(description="Our R1 model vs the R1 winner's model (gold-standard PvP).")
    p.add_argument("--env", required=True,
                   choices=["clobber", "othello", "goofspiel", "leduc_poker", "liars_dice", "gin_rummy"])
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--our-model", default=_OURS_DEFAULT)
    p.add_argument("--winner-model", default=None,
                   help="opponent model (default = the per-game CHAMP measured in task 5ccbf224: "
                        "othello->R2 Boss 5GU4Xkd3, clobber->5EEa; both the strongest for that game)")
    p.add_argument("--base-model", default=_BASE)
    p.add_argument("--opening", type=int, default=6,
                   help="seeded random opening plies before the models take over (position variety; "
                        "without it deterministic play makes SEAT, not skill, decide clobber/othello)")
    p.add_argument("--our-mcts", type=int, default=0,
                   help="if >0, OUR side is the raw OpenSpiel MCTS TEACHER at this sim count (not our "
                        "trained model) vs the winner's model — diagnoses TEACHER-ceiling vs distillation")
    p.add_argument("--our-mcts-rollouts", type=int, default=1, help="n_rollouts for the --our-mcts teacher")
    p.add_argument("--our-uct-c", type=float, default=2.0, help="uct_c for the --our-mcts teacher")
    p.add_argument("--winner-mcts", type=int, default=0,
                   help="if >0, the OPPONENT is raw OpenSpiel MCTS at this sim count (the tournament's scoring "
                        "opponent family) instead of a champ model — eval OUR trained model vs MCTS when the "
                        "champ is base-incompatible (leduc/liars champ is a Qwen adapter, unloadable on Llama).")
    p.add_argument("--winner-mcts-rollouts", type=int, default=1, help="n_rollouts for the --winner-mcts opponent")
    p.add_argument("--winner-uct-c", type=float, default=2.0, help="uct_c for the --winner-mcts opponent")
    p.add_argument("--our-expert", action="store_true",
                   help="OUR side is the hand-written EXPERT teacher (othello_expert negamax+perfect-endgame "
                        "/ clobber_endgame) vs the winner's model — measures the DEPLOYED expert's absolute "
                        "strength (the thing verify_othello_teacher can't: it only compares expert-vs-expert). "
                        "Sweep othello strength/speed via OTHELLO_TEACHER_BUDGET / _DEPTH / _ENDGAME env vars.")
    p.add_argument("--config-id", type=int, default=0)
    p.add_argument("--loose", action="store_true",
                   help="old lenient scoring: an unparseable/illegal model output plays some legal move "
                        "instead of forfeiting the game (default = STRICT, like the validator)")
    args = p.parse_args()
    if args.loose:
        _STRICT[0] = False
    if args.winner_model is None:
        args.winner_model = _WINNER_BY_ENV.get(args.env, _WINNER_DEFAULT)

    our_teacher = None
    if args.our_expert:
        our_label = f"OUR expert teacher ({args.env})"
        print(f"OUR side: {our_label}  (hand-written teacher, NOT our model)", flush=True)
        ours = None
        our_teacher = _teacher(args.env)
    elif args.our_mcts > 0:
        our_label = f"OUR MCTS teacher ({args.our_mcts} sims, {args.our_mcts_rollouts} rollouts)"
        print(f"OUR side: {our_label}  (raw teacher, NOT our model)", flush=True)
        ours = None
    else:
        our_label = "OUR R1 model"
        print(f"Loading OUR model:    {args.our_model}", flush=True)
        our_model, our_tok, our_kind = _load(args.our_model, args.base_model)
        print(f"  -> loaded as {our_kind}", flush=True)
        ours = (our_model, our_tok)
    if args.winner_mcts > 0:
        winner = None
        print(f"OPPONENT: OpenSpiel MCTS ({args.winner_mcts} sims, {args.winner_mcts_rollouts} rollouts) "
              f"— the tournament scoring opponent, not a champ model", flush=True)
    else:
        print(f"Loading WINNER model: {args.winner_model}", flush=True)
        win_model, win_tok, win_kind = _load(args.winner_model, args.base_model)
        print(f"  -> loaded as {win_kind}", flush=True)
        winner = (win_model, win_tok)

    game = _load_game(args.env, args.config_id)
    tally = {"ours": 0, "winner": 0, "draw": 0}
    for i in range(args.games):
        res = _play_one(args.env, game, ours, winner, our_first=(i % 2 == 0),
                        seed=7000 + i, opening=args.opening, our_mcts_sims=args.our_mcts,
                        our_mcts_rollouts=args.our_mcts_rollouts, our_uct=args.our_uct_c,
                        our_teacher=our_teacher,
                        winner_mcts_sims=args.winner_mcts,
                        winner_mcts_rollouts=args.winner_mcts_rollouts, winner_uct=args.winner_uct_c)
        tally[res] += 1
        print(f"  game {i + 1}/{args.games}: {res}   "
              f"(ours {tally['ours']} / winner {tally['winner']} / draw {tally['draw']})", flush=True)

    decisive = tally["ours"] + tally["winner"]
    wr = 100 * tally["ours"] / decisive if decisive else 0.0
    opp_tag = args.winner_model.rsplit("-", 1)[-1]  # ACTUAL opponent hotkey (per-env champ, not always 5EEa)
    print(f"\n{args.env}: {our_label} vs opponent {opp_tag} — {args.games} games "
          f"(random opening {args.opening} plies)")
    print(f"  ours: {tally['ours']}   winner: {tally['winner']}   draws: {tally['draw']}")
    print(f"  OUR win-rate of decisive games: {wr:.0f}%")
    fmt = ", ".join(f"{s} {f}/{n} turns" for s, (f, n) in sorted(_FORMAT_FAILS.items()))
    print(f"  format failures (SGLang-parser view; {'STRICT: each one forfeits its game' if _STRICT[0] else 'loose'}): "
          f"{fmt or 'none'}")
    if wr >= 55:
        print("\n[AHEAD] we out-play the winner on this env. ✅")
    elif wr >= 45:
        print("\n[EVEN] ~coin-flip vs the winner. 🟡")
    else:
        print("\n[BEHIND] the winner out-plays us here — this is the gap to close. 🔴")


if __name__ == "__main__":
    main()
