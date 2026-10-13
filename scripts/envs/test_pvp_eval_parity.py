"""Train/eval PARITY for every PvP game: our SFT prompts must be byte-identical to what the validator's
LLMBot sends at eval, on the same game variant and the same state.

For each game, config ids and seeds, it compares (1) the config id our generator derives from a seed with
core.pvp.game_eval.config_id_for_seed, (2) the pyspiel game (params) from our _load_game with the
validator agent's load_game(generate_params(config_id)), (3) the seeded opening, then along random play
(4) the system prompt, (5) the user prompt and (6) the tools list. Goofspiel's hero-seat generation rows
from games against the field-style pool bidders get the same byte check on the states they came from.

Needs the validator checkout (GOD_ROOT, default /app as in the godpvp image) plus open_spiel/openai/pydantic:
    docker run ... -v <our scripts>:/workspace/scripts godpvp:<sha> python /workspace/scripts/envs/test_pvp_eval_parity.py
Skips (exit 0) when the validator code is not importable.
"""
import os
import random
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
# The validator's top-level `core` package must win over our scripts/core, so import it first with the
# validator checkout at the front of sys.path; our envs modules are loaded as a path-registered package.
GOD_ROOT = os.getenv("GOD_ROOT", "/app")
sys.path = [GOD_ROOT] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(_HERE)]
if "envs" not in sys.modules:
    _pkg = types.ModuleType("envs")
    _pkg.__path__ = [_HERE]
    sys.modules["envs"] = _pkg
try:
    from core.constants.environments import ENVIRONMENT_CONFIGS, EnvironmentName
    from core.models.pvp_models import ChatCompletionConfig, GameOutcome, GameScoringContext, MemoryArea
    from core.pvp import constants as v_pvp_cst
    from core.pvp import tools as v_tools
    from core.pvp.bot import LLMBot
    from core.pvp.game_eval import _AGENT_REGISTRY, config_id_for_seed
    from core.pvp.scoring import determine_outcome
except Exception as exc:  # noqa: BLE001
    print(f"SKIP: validator code not importable from {GOD_ROOT} ({exc})")
    sys.exit(0)

import json  # noqa: E402

import pyspiel  # noqa: E402

from envs import pvp_matchup_memory as mm  # noqa: E402
from envs.pvp_format import build_pvp_tools, build_user_prompt, tools_to_openai  # noqa: E402
from envs.pvp_selfplay import (_FORMATTERS, _build_row, _config_id_for_seed, _load_game,  # noqa: E402
                               _setup_initial_state, _system_prompt)

GAMES = ["liars_dice", "leduc_poker", "gin_rummy", "othello", "clobber", "goofspiel"]
FAILS: list[str] = []
_CFG = ChatCompletionConfig(inference_model="parity", base_url="http://127.0.0.1:1/v1")


def fail(msg: str) -> None:
    if len(FAILS) < 40:
        print("FAIL " + msg, flush=True)
    FAILS.append(msg)


def _first_diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i}: ours={a[max(0, i - 40):i + 60]!r} | eval={b[max(0, i - 40):i + 60]!r}"


def check_game(env: str, n_seeds: int, turns_per_game: int) -> int:
    name = EnvironmentName(env)
    agent = _AGENT_REGISTRY[name]()
    compared = 0
    for s in range(n_seeds):
        seed = 7919 * (s + 1)
        cid_ours = _config_id_for_seed(seed, env)
        cid_eval = config_id_for_seed(seed, ENVIRONMENT_CONFIGS[name])
        if cid_ours != cid_eval:
            fail(f"{env} seed {seed}: config id ours={cid_ours} eval={cid_eval}")
            continue
        g_ours = _load_game(env, cid_ours)
        g_eval = agent.load_game(agent.generate_params(cid_eval))
        if str(g_ours) != str(g_eval):
            fail(f"{env} cid {cid_ours}: game ours={g_ours} eval={g_eval}")
            continue
        st_ours, st_eval = g_ours.new_initial_state(), g_eval.new_initial_state()
        _setup_initial_state(env, st_ours, seed)
        agent.setup_initial_state(st_eval, seed)
        if str(st_ours) != str(st_eval):
            fail(f"{env} seed {seed}: seeded opening differs")
            continue
        rng = random.Random(seed)
        state = st_eval
        done = 0
        while not state.is_terminal() and done < turns_per_game:
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            if state.is_simultaneous_node():
                fail(f"{env}: eval state is simultaneous (should be turn-based)")
                break
            p = state.current_player()
            legal = state.legal_actions(p)
            bot = LLMBot(g_eval, p, chat_fn=None, config=_CFG, agent=agent)
            sys_eval = bot._system_prompt()
            user_eval = bot._user_prompt(state, legal)
            tools_eval = [t.to_openai() for t in bot._memory_tools
                          + [v_tools.build_game_action_tool(bot._legal_hint(legal), legal)]]
            labels = [(a, state.action_to_string(p, a)) for a in legal]
            sys_ours = _system_prompt(env)
            user_ours = build_user_prompt(_FORMATTERS[env](state, p), p, labels)
            tools_ours = tools_to_openai(build_pvp_tools(legal))
            if sys_ours != sys_eval:
                fail(f"{env} system prompt {_first_diff(sys_ours, sys_eval)}")
                return compared
            if user_ours != user_eval:
                fail(f"{env} cid {cid_eval} user prompt {_first_diff(user_ours, user_eval)}")
            if tools_ours != tools_eval:
                fail(f"{env} tools differ: ours={tools_ours[-1]} eval={tools_eval[-1]}")
            compared += 1
            done += 1
            state.apply_action(rng.choice(legal))
    return compared


def _captured_hero_game(gh, seed: int) -> list[tuple]:
    """Play one goofspiel hero-seat generation game and return (state, player, legal ids, row) for every
    emitted row. The generator's own formatter and row emitter are wrapped (restored afterwards), so the
    rows are exactly what training sees and each is paired with the pyspiel state it was rendered from."""
    formatters, emit = gh._FORMATTERS, gh._emit_rows
    last: list = []
    out: list[tuple] = []

    def fmt(state, pid):
        last[:] = [state.clone()]
        return formatters["goofspiel"](state, pid)

    def spy_emit(rows, env, state_desc, pid, labels, action_id):
        emit(rows, env, state_desc, pid, labels, action_id)
        out.append((last[0], pid, [a for a, _ in labels], rows[-1]))

    gh._FORMATTERS, gh._emit_rows = {**formatters, "goofspiel": fmt}, spy_emit
    try:
        gh.play_goofspiel_hero_game(seed, max_turn=30)
    finally:
        gh._FORMATTERS, gh._emit_rows = formatters, emit
    return out


def check_goofspiel_hero_rows(games_per_opponent: int) -> int:
    """Hero-seat goofspiel rows from games against each field-style pool bidder == what LLMBot sends on the
    same state (system prompt, user prompt, tools), on the validator's own game for that seed."""
    from envs import goofspiel_hero as gh
    name = EnvironmentName("goofspiel")
    agent = _AGENT_REGISTRY[name]()
    compared = 0
    for opponent in ("strict-two-up", "one-up-tie", "overbid-deck"):
        seeds = [s for s in range(50_000, 60_000) if gh.pick_opponent(s)[0] == opponent][:games_per_opponent]
        if not seeds:
            fail(f"goofspiel hero: no seed draws the pool opponent {opponent!r}")
        for seed in seeds:
            game = agent.load_game(agent.generate_params(config_id_for_seed(seed, ENVIRONMENT_CONFIGS[name])))
            for state, pid, legal, row in _captured_hero_game(gh, seed):
                if str(state.get_game()) != str(game):
                    fail(f"goofspiel hero seed {seed}: game ours={state.get_game()} eval={game}")
                    return compared
                bot = LLMBot(game, pid, chat_fn=None, config=_CFG, agent=agent)
                tools_eval = [t.to_openai() for t in bot._memory_tools
                              + [v_tools.build_game_action_tool(bot._legal_hint(legal), legal)]]
                sys_ours, user_ours = row["messages"][0]["content"], row["messages"][1]["content"]
                sys_eval, user_eval = bot._system_prompt(), bot._user_prompt(state, legal)
                where = f"goofspiel hero vs {opponent} seed {seed}"
                if sys_ours != sys_eval:
                    fail(f"{where} system prompt {_first_diff(sys_ours, sys_eval)}")
                if user_ours != user_eval:
                    fail(f"{where} user prompt {_first_diff(user_ours, user_eval)}")
                if json.loads(row["tools"]) != tools_eval:
                    fail(f"{where}: tools differ")
                compared += 1
    return compared


def _write_long_term(bot: LLMBot, slots: dict[int, str]) -> None:
    """Fill the validator bot's LONG_TERM memory with ITS OWN ops: a multi-line slot is the free-form
    'appended' case, so it is built by successive append calls; everything else by rewrite."""
    mem = bot._memories[MemoryArea.LONG_TERM]
    for slot, text in slots.items():
        for part in (text.split("\n") if "\n" in text else [text]):
            result = mem.append(slot, part) if "\n" in text else mem.rewrite(slot, part)
            if not result.startswith("ok") or "truncated" in result or "dropped" in result:
                fail(f"validator memory op on slot {slot} -> {result!r}")


def check_memory_rows(env: str, n_seeds: int, turns_per_game: int) -> int:
    """Game rows re-rendered with LONG_TERM notes == LLMBot._system_prompt with those notes in its memory."""
    name = EnvironmentName(env)
    agent = _AGENT_REGISTRY[name]()
    rng = random.Random(f"memory-parity:{env}")
    compared = 0
    for s in range(n_seeds):
        seed = 104729 * (s + 1)
        game = agent.load_game(agent.generate_params(config_id_for_seed(seed, ENVIRONMENT_CONFIGS[name])))
        state = game.new_initial_state()
        agent.setup_initial_state(state, seed)
        done = 0
        while not state.is_terminal() and done < turns_per_game:
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choices(list(acts), weights=list(probs), k=1)[0])
                continue
            p = state.current_player()
            legal = state.legal_actions(p)
            labels = [(a, state.action_to_string(p, a)) for a in legal]
            row = _build_row(env, _FORMATTERS[env](state, p), p, labels, legal[0])
            slots = mm.game_row_long_term(rng, env)
            ours = mm.with_long_term_memory(row, env, slots)
            bot = LLMBot(game, p, chat_fn=None, config=_CFG, agent=agent)
            _write_long_term(bot, slots)
            if ours is None:
                fail(f"{env}: standard game row not recognised for memory rendering")
                return compared
            if ours["messages"][0]["content"] != bot._system_prompt():
                fail(f"{env} memory system prompt {_first_diff(ours['messages'][0]['content'], bot._system_prompt())}")
                return compared
            if ours["messages"][1:] != row["messages"][1:] or ours["tools"] != row["tools"]:
                fail(f"{env}: memory rendering changed the user prompt, label or tools")
            compared += 1
            done += 1
            state.apply_action(rng.choice(legal))
    return compared


def check_reflection_rows(env: str, n_rows: int) -> int:
    """Reflection rows == LLMBot._reflection_system_prompt / _reflection_user_prompt / _memory_tools for the
    same final state and prior notes; the outcome is the validator's own determine_outcome; replaying the
    taught call through the validator's execute_memory_tool stores the note in slot 1 untruncated. The
    reflection games are seeded above the validator's own eval seed span."""
    if mm.reflection_seed_bounds(env)[0] <= v_pvp_cst.PVP_SEED_RANGE_MAX:
        fail(f"{env} reflection seeds start at {mm.reflection_seed_bounds(env)[0]}, inside the eval seed span "
             f"1..{v_pvp_cst.PVP_SEED_RANGE_MAX}")
    name = EnvironmentName(env)
    agent = _AGENT_REGISTRY[name]()
    for i in range(n_rows):
        ex = mm.build_reflection_example(env, game_seed=mm.reflection_seed_bounds(env)[0] + 15485863 * i % 999_983)
        game = ex.state.get_game()
        bot = LLMBot(game, ex.player_id, chat_fn=None, config=_CFG, agent=agent)
        _write_long_term(bot, ex.prior_slots)
        sys_ours, user_ours, call = (ex.row["messages"][0]["content"], ex.row["messages"][1]["content"],
                                     ex.row["messages"][2]["tool_calls"][0]["function"])
        outcome = determine_outcome(GameScoringContext(
            returns=ex.state.returns(), player_id=ex.player_id,
            is_zero_sum=game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM,
            min_utility=game.min_utility(), max_utility=game.max_utility()))
        if outcome.value != ex.outcome:
            fail(f"{env} reflection outcome ours={ex.outcome} eval={outcome.value}")
        if sys_ours != bot._reflection_system_prompt():
            fail(f"{env} reflection system prompt {_first_diff(sys_ours, bot._reflection_system_prompt())}")
            return i
        user_eval = bot._reflection_user_prompt(ex.state, outcome)
        if user_ours != user_eval:
            fail(f"{env} reflection user prompt {_first_diff(user_ours, user_eval)}")
            return i
        if json.loads(ex.row["tools"]) != [t.to_openai() for t in bot._memory_tools]:
            fail(f"{env} reflection tools differ from LLMBot._memory_tools")
        result = v_tools.execute_memory_tool(bot._memories, call["name"], json.loads(call["arguments"]))
        stored = bot._memories[MemoryArea.LONG_TERM].read(1)
        if not result.startswith("ok") or "truncated" in result or stored != json.loads(call["arguments"])["content"]:
            fail(f"{env} reflection call through execute_memory_tool -> {result!r}, slot 1 = {stored!r}")
    return n_rows


if __name__ == "__main__":
    n_seeds = int(os.getenv("PARITY_SEEDS", "24"))
    turns = int(os.getenv("PARITY_TURNS", "12"))
    for env in GAMES:
        before = len(FAILS)
        n = check_game(env, n_seeds, turns)
        print(f"{'PASS' if len(FAILS) == before else 'FAIL'} {env}: {n} states compared "
              f"(system + user + tools byte-identical, game/config/opening parity)", flush=True)
        before = len(FAILS)
        n = check_memory_rows(env, max(2, n_seeds // 3), turns)
        print(f"{'PASS' if len(FAILS) == before else 'FAIL'} {env}: {n} game rows with LONG_TERM notes "
              f"byte-identical to LLMBot._system_prompt (validator rewrite/append ops)", flush=True)
        before = len(FAILS)
        n = check_reflection_rows(env, max(4, n_seeds))
        print(f"{'PASS' if len(FAILS) == before else 'FAIL'} {env}: {n} reflection rows byte-identical to "
              f"LLMBot reflection prompts + tools; outcome + stored note via the validator's own code", flush=True)
    before = len(FAILS)
    n = check_goofspiel_hero_rows(max(2, n_seeds // 4))
    print(f"{'PASS' if len(FAILS) == before else 'FAIL'} goofspiel: {n} hero-seat generation rows vs the "
          f"field-style pool bidders byte-identical to LLMBot (system + user + tools)", flush=True)
    print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)} mismatches)")
    sys.exit(1 if FAILS else 0)
