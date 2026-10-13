"""Offline expert-trajectory generator using OpenSpiel's own game engine plus
its built-in solvers (CFR / MCTS / ISMCTS), instead of reimplementing game
rules from scratch (risk: rules could diverge from the validator's) or
relying on rejection-sampling from an untrained model (Bagian 2 — quality
capped by how often the base model wins by chance).

The environment names gin_rummy/liars_dice/leduc_poker/othello/goofspiel are
identical to OpenSpiel's own game names, and the validator's built-in
opponent ("opponent": "mcts") matches OpenSpiel's own bot family — strong
evidence the validator is built on OpenSpiel. Using OpenSpiel's real game
logic here means the rules are the standard, verifiable implementation, not
a guess.

CPU-only, no GPU/torch needed — run this anywhere:
    pip install -r scripts/envs/openspiel_requirements.txt
    python3 scripts/envs/generate_openspiel_experts.py --env-name leduc_poker --num-games 200

Output format matches scripts/envs/generate_trajectories.py exactly
(data.json with {"messages": [...]} rows) and is written straight into
scripts/envs/prebaked/{env_name}/ for fetch_prebaked_dataset.py to pick up —
no changes needed anywhere else in the pipeline (Bagian 2's sft_env_config.py
already prefers a prebaked dataset over live generation when one exists).

Progress is saved periodically (same cadence as the progress log lines), and
on Ctrl+C the rows collected so far are saved before exiting — long runs
(gin_rummy/othello can take a while) don't need to run to completion to keep
their results.

Algorithm per game (see plan for full rationale/confidence level):
  leduc_poker, liars_dice, goofspiel -> CFR (open_spiel.python.algorithms.cfr)
  othello                            -> MCTS (open_spiel.python.algorithms.mcts)
  gin_rummy                          -> ISMCTS (open_spiel.python.algorithms.ismcts) — lowest
                                         confidence here: ISMCTSBot's exact kwargs are less
                                         commonly documented than vanilla MCTSBot. Verify this
                                         runs cleanly against your installed open_spiel version
                                         before trusting the output.

NOTE: state.observation_string()/information_state_string() text format is
OpenSpiel's own representation, not necessarily identical to what the
validator's environment server actually sends over /reset and /step. Good
enough to teach the model good *patterns* via SFT warm-start; re-verify once
real ENVIRONMENT_SERVER_URLS access is available.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, ismcts, mcts


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFR_GAMES = {"leduc_poker", "liars_dice", "goofspiel"}
MCTS_GAMES = {"othello"}
ISMCTS_GAMES = {"gin_rummy"}

DEFAULT_CFR_ITERATIONS = {
    "leduc_poker": 1000,
    "liars_dice": 1000,
    "goofspiel": 500,
}
# goofspiel needs a small num_cards for CFR to converge in reasonable time,
# and vanilla CFRSolver requires sequential games — wrap it with
# turn_based_simultaneous_game so it presents alternating turns to CFR
# instead of true simultaneous moves (confirmed needed: CFRSolver raises
# "CFR requires sequential games" on the raw simultaneous-move goofspiel).
DEFAULT_GAME_STRING = {
    "goofspiel": (
        "turn_based_simultaneous_game(game=goofspiel("
        "imp_info=True,num_cards=5,points_order=descending))"
    ),
}


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPTS_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_output(
    rows: list[dict],
    output_dir: str,
    env_name: str,
    algorithm_note: str,
    num_games_requested: int,
    num_games_done: int,
) -> None:
    """Write data.json + metadata. Called periodically during generation (not
    just at the end) so Ctrl+C never loses more than a few games of progress."""
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "data.json")
    with open(data_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False)

    metadata_path = os.path.join(output_dir, "_GENERATED_FROM_COMMIT.txt")
    with open(metadata_path, "w") as f:
        f.write(f"commit: {get_commit_hash()}\n")
        f.write(f"env_name: {env_name}\n")
        f.write(algorithm_note)
        f.write(f"num_games_requested: {num_games_requested}\n")
        f.write(f"num_games_done: {num_games_done}\n")
        f.write(f"num_games_kept: {len(rows)}\n")
        f.write(f"generated_at: {datetime.now(timezone.utc).isoformat()}\n")


def _state_text(state, player: int) -> str:
    """Best-effort human-readable text for a state from one player's view."""
    for method in ("observation_string", "information_state_string"):
        try:
            return getattr(state, method)(player)
        except (pyspiel.SpielError, AttributeError):
            continue
    return str(state)


def _advance_chance_nodes(state, rng: random.Random) -> None:
    while state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(outcomes, weights=probs)[0])


def _sample_action(probs_dict: dict, rng: random.Random) -> int:
    actions, probs = zip(*probs_dict.items())
    return rng.choices(actions, weights=probs)[0]


def _play_one_game_cfr(game, policy, rng: random.Random):
    """Self-play one game sampling both players from the same CFR average
    policy. Records only player 0's turns as the SFT trajectory."""
    state = game.new_initial_state()
    messages = []

    while not state.is_terminal():
        _advance_chance_nodes(state, rng)
        if state.is_terminal():
            break

        if state.current_player() == pyspiel.PlayerId.SIMULTANEOUS:
            # Goofspiel-style: both players choose at once. Risk: vanilla CFR's
            # support for simultaneous-move games depends on the open_spiel
            # version's internal handling — verify this branch is actually
            # reached and produces sane actions before trusting goofspiel output.
            actions = []
            for player in range(game.num_players()):
                probs_dict = policy.action_probabilities(state, player)
                actions.append(_sample_action(probs_dict, rng))
            obs_text = _state_text(state, 0)
            messages.append({"role": "user", "content": obs_text})
            messages.append({"role": "assistant", "content": f"Action: {actions[0]}"})
            state.apply_actions(actions)
        else:
            player = state.current_player()
            probs_dict = policy.action_probabilities(state)
            action = _sample_action(probs_dict, rng)
            if player == 0:
                obs_text = _state_text(state, 0)
                messages.append({"role": "user", "content": obs_text})
                messages.append({"role": "assistant", "content": f"Action: {action}"})
            state.apply_action(action)

    return messages, state.returns()[0]


def _play_one_game_bot(game, bot_factory, rng: random.Random):
    """Self-play one game using a search bot (MCTS/ISMCTS) for both players —
    a fresh bot instance per player avoids any shared-tree state issues."""
    state = game.new_initial_state()
    bots = [bot_factory(), bot_factory()]
    messages = []

    while not state.is_terminal():
        _advance_chance_nodes(state, rng)
        if state.is_terminal():
            break

        player = state.current_player()
        action = bots[player].step(state)
        if player == 0:
            obs_text = _state_text(state, 0)
            messages.append({"role": "user", "content": obs_text})
            messages.append({"role": "assistant", "content": f"Action: {action}"})
        state.apply_action(action)

    return messages, state.returns()[0]


def _run_self_play_loop(env_name, num_games, output_dir, algorithm_note, play_one_game, *play_args):
    """Shared self-play loop for all 3 algorithms: play games, keep winners,
    save progress periodically and on Ctrl+C so partial runs aren't lost."""
    rows = []
    wins = 0
    save_every = max(1, num_games // 20)  # ~20 checkpoints over the full run

    try:
        for i in range(num_games):
            messages, player0_return = play_one_game(*play_args)
            if player0_return > 0:
                wins += 1
                rows.append({"messages": messages})
            if (i + 1) % save_every == 0 or (i + 1) == num_games:
                write_output(rows, output_dir, env_name, algorithm_note, num_games, i + 1)
                print(f"[generate_openspiel_experts] {env_name}: {i + 1}/{num_games} games, {wins} wins kept", flush=True)
    except KeyboardInterrupt:
        write_output(rows, output_dir, env_name, algorithm_note, num_games, i + 1)
        print(
            f"\n[generate_openspiel_experts] interrupted at {i + 1}/{num_games} — "
            f"saved {wins} wins kept so far to {output_dir}/data.json",
            flush=True,
        )
        sys.exit(130)

    return rows


def generate_with_cfr(env_name: str, num_games: int, iterations: int, output_dir: str, algorithm_note: str, seed: int = 42) -> list[dict]:
    game_string = DEFAULT_GAME_STRING.get(env_name, env_name)
    game = pyspiel.load_game(game_string)

    print(f"[generate_openspiel_experts] training CFR on {game_string} for {iterations} iterations...", flush=True)
    solver = cfr.CFRSolver(game)
    for i in range(iterations):
        solver.evaluate_and_update_policy()
        if (i + 1) % max(1, iterations // 10) == 0:
            print(f"[generate_openspiel_experts] CFR iteration {i + 1}/{iterations}", flush=True)
    policy = solver.average_policy()

    rng = random.Random(seed)
    return _run_self_play_loop(env_name, num_games, output_dir, algorithm_note, _play_one_game_cfr, game, policy, rng)


def generate_with_mcts(env_name: str, num_games: int, simulations: int, output_dir: str, algorithm_note: str, seed: int = 42) -> list[dict]:
    game = pyspiel.load_game(env_name)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)  # OpenSpiel bots expect numpy's RandomState, not stdlib random

    def bot_factory():
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np_rng)
        return mcts.MCTSBot(game, uct_c=2.0, max_simulations=simulations, evaluator=evaluator, random_state=np_rng)

    return _run_self_play_loop(env_name, num_games, output_dir, algorithm_note, _play_one_game_bot, game, bot_factory, rng)


def generate_with_ismcts(env_name: str, num_games: int, simulations: int, output_dir: str, algorithm_note: str, seed: int = 42) -> list[dict]:
    # Lowest-confidence path in this script (see module docstring). Fall back
    # to rejection-sampling (Bagian 2, generate_trajectories.py) for gin_rummy
    # if this needs more API tuning than expected.
    game = pyspiel.load_game(env_name)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    def bot_factory():
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=np_rng)
        return ismcts.ISMCTSBot(game=game, evaluator=evaluator, uct_c=2.0, max_simulations=simulations, random_state=np_rng)

    return _run_self_play_loop(env_name, num_games, output_dir, algorithm_note, _play_one_game_bot, game, bot_factory, rng)


def main():
    parser = argparse.ArgumentParser(description="OpenSpiel-backed expert trajectory generator for EnvTask SFT warm-start")
    parser.add_argument("--env-name", required=True, choices=sorted(CFR_GAMES | MCTS_GAMES | ISMCTS_GAMES))
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--cfr-iterations", type=int, default=None, help="Defaults vary per game, see DEFAULT_CFR_ITERATIONS")
    parser.add_argument("--mcts-simulations", type=int, default=400)
    parser.add_argument("--output-dir", default=None, help="Defaults to scripts/envs/prebaked/{env_name}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPTS_DIR, "envs", "prebaked", args.env_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.env_name in CFR_GAMES:
        iterations = args.cfr_iterations or DEFAULT_CFR_ITERATIONS[args.env_name]
        algorithm_note = f"algorithm: cfr\ncfr_iterations: {iterations}\n"
        rows = generate_with_cfr(args.env_name, args.num_games, iterations, output_dir, algorithm_note, args.seed)
    elif args.env_name in MCTS_GAMES:
        algorithm_note = f"algorithm: mcts\nmcts_simulations: {args.mcts_simulations}\n"
        rows = generate_with_mcts(args.env_name, args.num_games, args.mcts_simulations, output_dir, algorithm_note, args.seed)
    else:
        algorithm_note = f"algorithm: ismcts\nmcts_simulations: {args.mcts_simulations}\n"
        rows = generate_with_ismcts(args.env_name, args.num_games, args.mcts_simulations, output_dir, algorithm_note, args.seed)

    if not rows:
        sys.exit(
            f"No winning games out of {args.num_games} for {args.env_name} — algorithm may need more "
            "iterations/simulations, or check the implementation against your open_spiel version."
        )

    print(f"[generate_openspiel_experts] done: wrote {len(rows)} rows to {output_dir}/data.json", flush=True)


if __name__ == "__main__":
    main()
