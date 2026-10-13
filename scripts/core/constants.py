import os

from dotenv import load_dotenv


load_dotenv()

VERSION_KEY = 61_000
# Default NETUID if not set in environment
DEFAULT_NETUID = 56

try:
    NETUID = int(os.getenv("NETUID", DEFAULT_NETUID))
except (TypeError, ValueError):
    NETUID = DEFAULT_NETUID

IS_PROD_ENV = NETUID == DEFAULT_NETUID

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
MINER_DOCKER_IMAGE_DIFFUSION = "diagonalge/miner-diffusion-flux:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"
VALIDATOR_DOCKER_IMAGE_DIFFUSION = "diagonalge/tuning_validator_diffusion:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "core/config/"
OUTPUT_DIR = "core/outputs/"
CACHE_DIR = "~/.cache/huggingface"
CACHE_DIR_HUB = os.path.expanduser("~/.cache/huggingface/hub")
DIFFUSION_DATASET_DIR = "core/dataset/images"
CONTAINER_FLUX_PATH = "/app/flux/unet"

DIFFUSION_SDXL_REPEATS = 10
DIFFUSION_FLUX_REPEATS = 1
DIFFUSION_DEFAULT_INSTANCE_PROMPT = "lora"
DIFFUSION_DEFAULT_CLASS_PROMPT = "style"

MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50

CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL = CONFIG_DIR + "base_diffusion_sdxl.toml"
CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX = CONFIG_DIR + "base_diffusion_flux.toml"


CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
CONFIG_TEMPLATE_PATH_GRPO = CONFIG_DIR + "base_grpo.yml"

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
RAYONLABS_HF_USERNAME = "rayonlabs"

CUSTOM_DATASET_TYPE = "custom"

# DPO default dataset type
DPO_DEFAULT_DATASET_TYPE = "chatml.default" 
# Field names must match exactly what Axolotl's formatter expects
DPO_DEFAULT_FIELD_PROMPT = "question"  # chatml.intel expects 'question'
DPO_DEFAULT_FIELD_SYSTEM = "system"
DPO_DEFAULT_FIELD_CHOSEN = "chosen"
DPO_DEFAULT_FIELD_REJECTED = "rejected"

GRPO_DEFAULT_FIELD_PROMPT = "prompt"

CONFIG_TEMPLATE_PATH_ENV = CONFIG_DIR + "base_grpo.yml"  # EnvTask reuses the GRPO trainer path

ENVIRONMENT_DEFAULT_FIELD_PROMPT = "prompt"
ENVIRONMENT_EXTRA_DATA_COLUMN = "extra_data"
ENVIRONMENT_SERVER_URLS_ENV_VAR = "ENVIRONMENT_SERVER_URLS"
ENVIRONMENT_DEFAULT_MAX_TURNS = 30

# task_id ranges, seed pool size, and baseline episode counts per environment,
# mirrored from gradients-ai/G.O.D core/constants.py so seeds stay validator-compatible
ENVIRONMENT_TASK_ID_RANGES = {
    "goofspiel": {"task_id_min": 0, "task_id_max": 99_999_999, "n_seeds": 10_000, "baseline_episodes": 25},
    "liars_dice": {"task_id_min": 100_000_000, "task_id_max": 199_999_999, "n_seeds": 10_000, "baseline_episodes": 50},
    "leduc_poker": {"task_id_min": 200_000_000, "task_id_max": 299_999_999, "n_seeds": 2_000, "baseline_episodes": 50},
    "gin_rummy": {"task_id_min": 300_000_000, "task_id_max": 399_999_999, "n_seeds": 1_000, "baseline_episodes": 25},
    "othello": {"task_id_min": 400_000_000, "task_id_max": 499_999_999, "n_seeds": 10_000, "baseline_episodes": 25},
    "intercode": {"task_id_min": 1, "task_id_max": 200, "n_seeds": 20, "baseline_episodes": 7},
}
