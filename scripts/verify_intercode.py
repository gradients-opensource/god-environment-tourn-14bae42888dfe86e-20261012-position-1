"""Verify intercode SFT data generation via the REAL pipeline path.

Downloads the whitelisted dataset's raw files with huggingface_hub (NOT
datasets.load_dataset, whose strict json builder chokes on the repo's extra
*.meta.json sidecar), mounts them under a temp MINER_DATASETS_DIR exactly like
the validator does, then runs build_intercode_sft_dataset() — the same code path
generate_trajectories uses. Confirms the .meta.json is tolerated and dumps the
tool-call format + conversion stats.

Run inside the training image (no rebuild). Add -e HF_TOKEN=hf_xxx if gated:

  docker run --rm --entrypoint python -w /workspace/scripts \
    -v "$(pwd)/scripts:/workspace/scripts" \
    standalone-text-trainer -u verify_intercode.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import trl_compat  # noqa: F401

from envs.intercode_dataset import INTERCODE_HF_REPO


def main() -> None:
    from huggingface_hub import snapshot_download

    dirname = INTERCODE_HF_REPO.replace("/", "--")
    tmp = tempfile.mkdtemp()
    local = os.path.join(tmp, dirname)

    print(f"Downloading raw files of {INTERCODE_HF_REPO} ...")
    try:
        snapshot_download(repo_id=INTERCODE_HF_REPO, repo_type="dataset", local_dir=local)
    except Exception as e:
        print(f"\nDOWNLOAD FAILED: {e!r}")
        print("If gated, re-run with  -e HF_TOKEN=hf_xxx  (needs gradients-io-tournaments access).")
        print("In the real tournament the validator mounts this at MINER_DATASETS_DIR anyway.")
        return

    print(f"Files in repo: {sorted(os.listdir(local))}")

    # Mount exactly like the validator: MINER_DATASETS_DIR/<org--repo>/, listed in MINER_DATASETS.
    os.environ["MINER_DATASETS_DIR"] = tmp
    os.environ["MINER_DATASETS"] = dirname

    from envs.intercode_dataset import build_intercode_sft_dataset

    dd = build_intercode_sft_dataset()
    if dd is None:
        print("\nbuild_intercode_sft_dataset() returned None — the real loader could NOT read")
        print("this dataset layout. That would mean intercode is skipped in the tournament.")
        return

    train = dd["train"]
    print(f"\n=== SUCCESS: real pipeline built {len(train)} intercode SFT examples ===")
    print("(=> the .meta.json sidecar is tolerated; intercode will generate in the tournament)")

    print("\n--- SAMPLE TRAINING ROW (what the model learns) ---")
    ex = train[0]
    for m in ex["messages"]:
        role = m["role"]
        if m.get("tool_calls"):
            print(f"[{role}] tool_calls:")
            print("    " + json.dumps(m["tool_calls"], indent=2).replace("\n", "\n    ")[:700])
        else:
            print(f"[{role}] {str(m.get('content'))[:600]}")
    print("\ntools field (first 500 chars):")
    print("  " + str(ex.get("tools"))[:500])


if __name__ == "__main__":
    main()
