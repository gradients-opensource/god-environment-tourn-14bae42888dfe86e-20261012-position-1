"""Lightweight registry mapping env names to their SFT trajectory generator.

Generators take ``(game_id, env_endpoint, max_turn)`` and return per-turn
training rows. A generator may return ``list[dict]`` (messages only),
``tuple[list[dict], float]`` (messages + final reward score), or a list of such
(rows, score) "views" — one per seat. The score(s) drive optional score-based
sampling in generate_trajectories.py.

All four game envs now use LOCAL pyspiel teacher self-play
(envs/pvp_selfplay.py): the teacher plays both seats in-process, with no
mcts-api env server. This removes all HTTP + server-side MCTS, harvests both
seats (~2x rows/game), and produces eval-identical prompts. ``env_endpoint`` is
accepted but ignored by these generators. See _SELFPLAY_ENVS.

"intercode" is the one remaining OFFLINE env: it reads from the
validator-mounted miner dataset (envs/intercode_dataset.py), not self-play.
"""

from typing import Callable

from envs.pvp_selfplay import make_selfplay_generator

# Game envs: local pyspiel teacher self-play (no env server).
_SELFPLAY_ENVS: frozenset[str] = frozenset(
    {"liars_dice", "gin_rummy", "leduc_poker", "othello", "goofspiel", "clobber"}
)

_SFT_REGISTRY: dict[str, Callable] = {
    name: make_selfplay_generator(name) for name in _SELFPLAY_ENVS
}

# envs that have their own generate path (not via generate_trajectories.py + env server):
# each reads a validator-mounted whitelisted dataset. "swe_infinite" reads SWE-ZERO
# trajectories (plain-text MiniSWE format) — see envs/swe_infinite_dataset.py.
_OFFLINE_ENVS: frozenset[str] = frozenset({"intercode", "swe_infinite"})


def supports_sft(env_name: str) -> bool:
    return env_name in _SFT_REGISTRY or env_name in _OFFLINE_ENVS


def get_sft_trajectory_generator(env_name: str) -> Callable:
    if env_name not in _SFT_REGISTRY:
        raise ValueError(f"No SFT trajectory generator for env: {env_name!r}")
    return _SFT_REGISTRY[env_name]
