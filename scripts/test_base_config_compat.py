"""Tests for base_config_compat (v5-layout continuation configs -> our transformers 4.51).

Run from scripts/:  python test_base_config_compat.py
Optional integration (in the training image, HF cache mounted): set V5_CFG_ROOT to a dir of configs saved
by transformers 5 (one subdir per base, named as in BASES below) and the test loads each one with our
transformers before/after the patch and compares against the original foundation config.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_config_compat import normalize_base_config, v4_compatible_config  # noqa: E402

FAILS = []
BASES = {"llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct", "llama3b": "unsloth/Llama-3.2-3B-Instruct",
         "qwen7b": "Qwen/Qwen2.5-7B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
         "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507"}


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def test_pure() -> None:
    v4 = {"rope_theta": 1000000.0, "torch_dtype": "bfloat16"}
    check("v4 foundation config -> no patch", v4_compatible_config(v4) is None)
    llama_v5 = {"rope_parameters": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                    "original_max_position_embeddings": 8192, "rope_theta": 500000.0,
                                    "rope_type": "llama3"}, "rope_scaling": None, "rope_theta": None,
                "dtype": "bfloat16"}
    p = v4_compatible_config(llama_v5)
    check("llama v5 -> rope_theta restored", p["rope_theta"] == 500000.0)
    check("llama v5 -> rope_scaling restored (llama3, factor 8, no theta inside)",
          p["rope_scaling"] == {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                "original_max_position_embeddings": 8192, "rope_type": "llama3"})
    check("llama v5 -> torch_dtype from dtype", p["torch_dtype"] == "bfloat16")
    check("input not mutated", llama_v5["rope_theta"] is None and llama_v5["rope_scaling"] is None)
    qwen_v5 = {"rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}, "dtype": "float16"}
    q = v4_compatible_config(qwen_v5)
    check("qwen v5 -> rope_theta 1e6, no rope_scaling", q["rope_theta"] == 1000000.0 and not q.get("rope_scaling"))
    check("already-normalized config -> no patch", v4_compatible_config(q) is None)


def test_shadow_dir() -> None:
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as root:
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"rope_parameters": {"rope_theta": 5000000, "rope_type": "default"}, "dtype": "bfloat16"}, f)
        for name in ("model.safetensors", "tokenizer.json", "chat_template.jinja"):
            with open(os.path.join(src, name), "w") as f:
                f.write(name)
        out = normalize_base_config(src, shadow_root=root)
        check("shadow dir returned", out != src and out.startswith(root))
        check("shadow config patched", json.load(open(os.path.join(out, "config.json")))["rope_theta"] == 5000000)
        check("original config untouched", "rope_theta" not in json.load(open(os.path.join(src, "config.json"))))
        check("other files symlinked", all(os.path.islink(os.path.join(out, n)) and open(os.path.join(out, n)).read() == n
                                           for n in ("model.safetensors", "tokenizer.json", "chat_template.jinja")))
        check("idempotent rerun", normalize_base_config(src, shadow_root=root) == out)
    with tempfile.TemporaryDirectory() as src:
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"rope_theta": 1000000.0}, f)
        check("foundation dir returned as-is", normalize_base_config(src, shadow_root=src + "_x") == src)
    check("missing config -> path as-is", normalize_base_config("/nonexistent/dir") == "/nonexistent/dir")


def test_integration() -> None:
    root = os.getenv("V5_CFG_ROOT")
    if not root:
        print("SKIP integration (V5_CFG_ROOT unset)")
        return
    from transformers import AutoConfig
    for name, repo in BASES.items():
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        orig = AutoConfig.from_pretrained(repo)
        raw = AutoConfig.from_pretrained(d)
        with tempfile.TemporaryDirectory() as sh:
            fixed = AutoConfig.from_pretrained(normalize_base_config(d, shadow_root=sh))
        print(f"  {name}: orig theta={orig.rope_theta} scaling={orig.rope_scaling} | v5-raw theta={raw.rope_theta} "
              f"scaling={raw.rope_scaling} | fixed theta={fixed.rope_theta} scaling={fixed.rope_scaling}")
        check(f"{name}: fixed rope == foundation rope",
              fixed.rope_theta == orig.rope_theta and fixed.rope_scaling == orig.rope_scaling)


if __name__ == "__main__":
    test_pure()
    test_shadow_dir()
    test_integration()
    print("RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
    sys.exit(1 if FAILS else 0)
