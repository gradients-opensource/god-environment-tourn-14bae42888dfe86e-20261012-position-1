"""Combine one or more trajectory data.json files (prebaked and/or freshly
generated) into a single file for tokenize_sft_env.py to split/tokenize."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    combined = []
    for path in args.inputs:
        with open(path, "r") as f:
            combined.extend(json.load(f))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(combined, f, ensure_ascii=False)

    print(
        f"[combine_trajectories] combined {len(args.inputs)} file(s) into {len(combined)} rows -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
