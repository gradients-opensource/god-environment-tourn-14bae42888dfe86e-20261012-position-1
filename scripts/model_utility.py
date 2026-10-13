DPO = "dpo"
GRPO = "grpo"
INSTRUCT = "instruct"
import re
import json
import glob
import os
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


MODEL_CONFIG = {
    "facebook/opt-1.3b": {"model_size": 1_300_000_000},
    "facebook/opt-3b": {"model_size": 3_000_000_000},
    "facebook/opt-6.7b": {"model_size": 6_700_000_000},
    "facebook/opt-13b": {"model_size": 13_000_000_000},
    "EleutherAI/gpt-neo-1.3B": {"model_size": 1_300_000_000},
    "EleutherAI/gpt-neo-125m": {"model_size": 125_000_000},
    "bigscience/bloom-560m": {"model_size": 560_000_000},
    "TinyLlama/TinyLlama_v1.1": {"model_size": 1_100_000_000},
}


def is_reasoning_tokenizer(tokenizer: AutoTokenizer) -> bool:
    try:
        vocab = tokenizer.get_vocab()
        pairs = [
            ('<think>', '</think>'),
            ('<thinking>', '</thinking>'),
            ('<reasoning>', '</reasoning>'),
            ('<thought>', '</thought>'),
            ('<reflection>', '</reflection>'),
        ]
        return any(open_tag in vocab and close_tag in vocab for open_tag, close_tag in pairs)
    except Exception:
        return False


def get_model_architecture(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    if architecture.lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]:
        return "True"
    return "False"


def count_params_from_safetensors(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        return None
    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        tensors = load_file(shard_path)
        total_params += sum(v.numel() for v in tensors.values())
    return total_params


def count_params_from_bin(model_dir):
    total_params = 0
    shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if not shards:
        return None
    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            total_params += sum(v.numel() for v in state_dict.values())
        except Exception as e:
            print(f"cannot load {shard_path}: {e}")
            continue
    return total_params


def get_model_size_from_local_path(model_path: str) -> int:
    size = count_params_from_safetensors(model_path)
    if size is not None and size > 1000:
        print(f"Model size from safetensors: {size}")
        return size
    size = count_params_from_bin(model_path)
    if size is not None and size > 1000:
        print(f"Model size from bin: {size}")
        return size
    return None


def get_gpu_count():
    return torch.cuda.device_count()


def get_model_num_params(model_id_or_path: str, model_path: str = None) -> int:
    """Accepts single-arg (model_path only) or two-arg (model_id, model_path) calling conventions."""
    # Two-arg: (model_id, model_path) — legacy callers
    if model_path is not None:
        model_id = model_id_or_path
        if model_id in MODEL_CONFIG:
            return MODEL_CONFIG[model_id]["model_size"]
        path = model_path
    else:
        # Single-arg: model_id_or_path is the local model path
        model_id = model_id_or_path
        path = model_id_or_path

    try:
        size = get_model_size_from_local_path(path)
        if size is not None:
            return size
        raise Exception(f"Cannot get model size from {path}")
    except Exception as e:
        print(f"Error getting model size from safetensors: {e}")
        try:
            match = re.search(r"(\d+)(?=[bB])", model_id)
            model_size = int(match.group(1)) * 1_000_000_000 if match else None
            print(f"Model size from regex: {model_size}")
            return model_size
        except Exception as e2:
            print(f"Error getting model size from regex: {e2}")
            return None


def disable_flash_attention(architecture: str, model: str = "") -> str:
    """Accepts single-arg (architecture) or two-arg (architecture, model) calling conventions."""
    model = model or ""
    if model in ["microsoft/phi-2"]:
        return "True"
    if "falcon-rw" in model.lower():
        return "True"
    if architecture.strip().lower() in [
        "gptneoforcausallm",
        "bloomforcausallm",
        "gptossforcausallm",
        "phiforcausallm",
        "falconforcausallm",
    ]:
        return "True"
    return "False"


def disable_action_mask(model_path: str) -> str:
    """Returns 'True' for tokenizers with known BPE byte-level quirks that break action mask alignment."""
    try:
        tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        tokenizer_class = tokenizer_config.get("tokenizer_class", "")
        if tokenizer_class in ("LlamaTokenizer", "CodeLlamaTokenizer"):
            return "True"
        if tokenizer_class == "LlamaTokenizerFast" and tokenizer_config.get("legacy") is True:
            return "True"
    except Exception as e:
        print(f"Could not read tokenizer_config.json from {model_path}: {e}")
    return "False"


def get_use_vllm(architecture: str, model: str = "") -> str:
    """Accepts single-arg (architecture) or two-arg (architecture, model) calling conventions."""
    model = model or ""
    if model in [
        "Eurdem/Defne_llama3_2x8B",
        "heegyu/WizardVicuna-open-llama-3b-v2",
        "openlm-research/open_llama_3b",
        "TitanML/tiny-mixtral",
        "dunzhang/stella_en_1.5B_v5",
        "oopsung/llama2-7b-n-ox-test-v1",
        "microsoft/phi-2",
        "databricks/dolly-v2-3b",
    ]:
        return False
    if "falcon-rw" in model.lower():
        return False
    if architecture.strip().lower() in [
        "gptneoforcausallm",
        "bloomforcausallm",
        "falconforcausallm",
        "phiforcausallm",
        "gptjforcausallm",
    ]:
        return False
    return True


def get_gradient_checkpointing(arch_or_model: str, _unused: str = "") -> str:
    """Accepts architecture or model path; disables for architectures that don't support it."""
    val = arch_or_model.strip().lower()
    if "falcon-rw" in val or val == "falconforcausallm":
        return "False"
    return "True"


def get_data_size(data_path: str) -> int:
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)
