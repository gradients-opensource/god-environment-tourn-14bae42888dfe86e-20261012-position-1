import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore

import requests

_VERBOSE = os.getenv("SFT_ENV_VERBOSE", "0") == "1"


def _log(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Quiet recipe logging (default ON). A planning/generation entrypoint can
# otherwise print its whole recipe straight to stdout -> the public training
# Loki (run command, intercode prefix, per-env row targets, teacher modes,
# LR/LoRA). ``recipe_log`` keeps the full line local and puts only a short
# decodable tag on stdout. Errors and tracebacks are never routed through
# this helper — callers print those directly so failures stay visible.
# ---------------------------------------------------------------------------

QUIET_RECIPE_LOGS = os.getenv("SFT_QUIET_RECIPE_LOGS", "1") == "1"  # backstop constant; default ON


def _recipe_log_path() -> str:
    out_dir = os.getenv("SFT_RECIPE_LOG_DIR") or os.getenv("OUTPUT_DIR") or "."
    return os.path.join(out_dir, "recipe.log")


def recipe_log(detail: str, tag: str = "") -> None:
    """Record one recipe-revealing line.

    The full ``detail`` always goes to a local ``recipe.log`` inside the
    container (never uploaded). stdout gets ``tag`` (or nothing) while
    QUIET_RECIPE_LOGS is on; set SFT_QUIET_RECIPE_LOGS=0 to also print
    ``detail`` on stdout, for local debugging.
    """
    try:
        path = _recipe_log_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(detail.rstrip("\n") + "\n")
    except OSError:
        pass  # local-file logging is best-effort; never block training on it
    if QUIET_RECIPE_LOGS:
        if tag:
            print(tag, flush=True)
    else:
        print(detail, flush=True)


# ---------------------------------------------------------------------------
# Game ID ranges (single source of truth — also imported by train_grpo_env.py)
# ---------------------------------------------------------------------------

GAMES_TO_TASK_ID_RANGE: dict[str, tuple[int, int]] = {
    "goofspiel":   (0,         99_999_999),
    "liars_dice":  (100000000, 199_999_999),
    "leduc_poker": (200000000, 299_999_999),
    "gin_rummy":   (300000000, 399_999_999),
    "othello":     (400000000, 499_999_999),
    "backgammon":  (500000000, 599_999_999),
    "hex":         (600000000, 699_999_999),
    "clobber":     (700000000, 799_999_999),
}

# Backstop: the validator draws its own eval seeds from 1..VALIDATOR_EVAL_ID_CEILING
# (inclusive). Training-data GENERATION must never sample a game id inside that span,
# or a generated row can collide with an id the validator itself draws for scoring.
# Today only goofspiel's raw range (it starts at 0) overlaps it.
VALIDATOR_EVAL_ID_CEILING = 1_000_000


def generation_id_bounds(env_name: str) -> tuple[int, int]:
    """Return the ``[low, high)`` game-id span to sample from when GENERATING
    training data for ``env_name``.

    Identical to ``GAMES_TO_TASK_ID_RANGE[env_name]`` (offset by one on the
    low end, since id ``task_id_min`` itself is reserved as the server
    warm-up/sentinel id) except the low end is also pushed above
    ``VALIDATOR_EVAL_ID_CEILING`` whenever the raw range would otherwise
    overlap the validator's eval seed span. ``GAMES_TO_TASK_ID_RANGE`` itself
    is left untouched — other callers (server warm-up task ids, the GRPO
    task-id index in train_grpo_env.py) keep reading the raw table.
    """
    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[env_name]
    low = max(task_id_min + 1, VALIDATOR_EVAL_ID_CEILING + 1)
    return low, task_id_max


# ---------------------------------------------------------------------------
# Curriculum scheduler (base class — subclass to extend per-game logic)
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Manages curriculum learning parameters throughout training.

    Subclass and override ``get_max_turn`` or ``get_hint_prob`` to implement
    custom scheduling logic for a specific game environment.
    """

    def __init__(
        self,
        initial_max_turn: int,
        final_max_turn: int,
        rollouts_per_stage: int,
        initial_hint_prob: float,
        final_hint_prob: float,
        warmup_rollouts: int,
    ) -> None:
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        """Return current max turns. Override for non-linear schedules."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted = self.total_rollouts - self.warmup_rollouts
        stage = adjusted // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        """Return current hint probability. Override for custom decay."""
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = self.final_max_turn - self.initial_max_turn
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted / total_decay_rollouts, 1.0) if total_decay_rollouts > 0 else 1.0
        current = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        """Advance internal rollout counter. Override to add custom logic."""
        self.total_rollouts += num_rollouts

    def get_status(self) -> dict:
        return {
            "total_rollouts": self.total_rollouts,
            "max_turn": self.get_max_turn(),
            "hint_prob": self.get_hint_prob(),
        }


# ---------------------------------------------------------------------------
# Environment pool initialisation
# ---------------------------------------------------------------------------

def init_env_pool(
    reset_payload: dict,
    reset_endpoint: str = "reset",
    lock_per_server: bool = False,
) -> tuple[int, list[dict], int, ThreadPoolExecutor, Semaphore]:
    """
    Initialise the environment server pool once per process.

    Reads ``LOCAL_RANK`` and ``ENVIRONMENT_SERVER_URLS`` from the environment.
    Sends a warm-up request to each server to verify it is reachable.

    Args:
        reset_payload:    JSON body for the warm-up POST request.
        reset_endpoint:   API endpoint to POST (``"reset"`` or ``"create"``).
        lock_per_server:  If True, attach a ``threading.Lock`` to each pool
                          entry (required by AlfWorld's per-server serialisation).

    Returns:
        ``(rank, env_pool, num_servers, thread_pool, generation_semaphore)``
        Each ``env_pool`` entry is a dict with at least ``"base_url"``.
        When ``lock_per_server=True`` it also has ``"env_id"`` and ``"lock"``.
    """
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

    if not server_urls:
        import warnings
        warnings.warn(
            "ENVIRONMENT_SERVER_URLS is empty — no env servers available. "
            "All episodes will return fallback reward=0. "
            "This is expected during local testing; set ENVIRONMENT_SERVER_URLS in production.",
            RuntimeWarning,
            stacklevel=2,
        )
        return rank, [], 0, ThreadPoolExecutor(max_workers=1), Semaphore(1)

    env_pool: list[dict] = []
    for idx, base_url in enumerate(server_urls):
        try:
            _log(f"[INIT] Connecting to server {idx}: {base_url}")
            res = requests.post(f"{base_url}/{reset_endpoint}", json=reset_payload, timeout=300)
            res.raise_for_status()
            entry: dict = {"base_url": base_url}
            if lock_per_server:
                entry["env_id"] = res.json()["id"]
                entry["lock"] = Lock()
            env_pool.append(entry)
            _log(f"[INIT] Server {idx} ready")
        except Exception as exc:
            raise RuntimeError(f"Failed to init server {base_url}: {exc}") from exc

    thread_pool = ThreadPoolExecutor(max_workers=len(env_pool))
    generation_semaphore = Semaphore(1)
    return rank, env_pool, len(env_pool), thread_pool, generation_semaphore


# ---------------------------------------------------------------------------
# Generic reward passthrough (identical across all game environments)
# ---------------------------------------------------------------------------

def rollout_reward_func(completions, **kwargs):
    """Generic reward passthrough used by all game environments."""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
