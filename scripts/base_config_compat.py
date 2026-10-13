"""Make a transformers-v5-saved base model config readable by our transformers 4.51 stack.

Continuation rounds (env tournament R2+, boss task 1, and boss task 3 when the previous winner is an
adapter) hand us the previous round's LoRA MERGED by the validator's trainer downloader, which runs
transformers 5.x and re-saves config.json in the v5 layout: rope_theta / rope_scaling move into a
``rope_parameters`` dict and torch_dtype becomes ``dtype``. transformers 4.51 does not know
rope_parameters, so it silently falls back to rope_theta=10000 and no rope_scaling (Qwen2.5 uses 1e6,
Qwen3-4B-2507 5e6, Llama-3.x 5e5 plus llama3 scaling). The model then trains on positional encodings the
eval server (which rebuilds the base with its own v5 stack) never uses. Verified 2026-09-26 by diffing
v5 saves of all whitelisted foundations against their original configs.

The base dir is mounted read-only, so the fix is a shadow dir: symlinks to every file plus a patched
config.json. Foundation configs (R1) carry no rope_parameters, so this is a no-op there.
"""
import json
import os

SHADOW_ROOT = "/workspace/base_model"


def v4_compatible_config(cfg: dict) -> "dict | None":
    """Return a patched copy of a v5-layout config, or None when no patch is needed. Pure."""
    rope = cfg.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("rope_theta") is None:
        return None
    patched = dict(cfg)
    changed = False
    if cfg.get("rope_theta") != rope["rope_theta"]:
        patched["rope_theta"] = rope["rope_theta"]
        changed = True
    rope_type = rope.get("rope_type") or rope.get("type") or "default"
    if rope_type != "default" and not cfg.get("rope_scaling"):
        scaling = {k: v for k, v in rope.items() if k not in ("rope_theta", "type")}
        scaling["rope_type"] = rope_type
        patched["rope_scaling"] = scaling
        changed = True
    if not cfg.get("torch_dtype") and cfg.get("dtype"):
        patched["torch_dtype"] = cfg["dtype"]
        changed = True
    return patched if changed else None


def normalize_base_config(model_path: str, shadow_root: str = SHADOW_ROOT) -> str:
    """Return a path whose config.json our transformers reads correctly (model_path itself if already fine)."""
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[base-config] could not read {cfg_path} ({exc}); using the base as-is", flush=True)
        return model_path
    patched = v4_compatible_config(cfg)
    if patched is None:
        return model_path
    shadow = os.path.join(shadow_root, os.path.basename(os.path.normpath(model_path)))
    try:
        os.makedirs(shadow, exist_ok=True)
        for name in os.listdir(model_path):
            if name == "config.json":
                continue
            dst = os.path.join(shadow, name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(os.path.abspath(model_path), name), dst)
        with open(os.path.join(shadow, "config.json"), "w") as f:
            json.dump(patched, f, indent=2)
    except OSError as exc:
        print(f"[base-config] could not build shadow dir {shadow} ({exc}); using the base as-is", flush=True)
        return model_path
    print(f"[base-config] v5-layout config -> v4: rope_theta {cfg.get('rope_theta')} -> {patched.get('rope_theta')}, "
          f"rope_scaling {cfg.get('rope_scaling')} -> {patched.get('rope_scaling')}; training from {shadow}", flush=True)
    return shadow
