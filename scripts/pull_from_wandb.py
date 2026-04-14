#!/usr/bin/env python3
"""Pull evaluation results from W&B projects and save as a local parquet.

Reads the "pull" key from configs/wandb.json and downloads full checkpoint
histories for the specified model families.

Usage:
    python scripts/pull_from_wandb.py
    python scripts/pull_from_wandb.py --output results/wandb_data.parquet
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.data import load_wandb_projects

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def main():
    parser = argparse.ArgumentParser(description="Pull eval results from W&B")
    parser.add_argument(
        "--output", type=str, default="results/wandb_data.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    with open(CONFIGS_DIR / "wandb.json") as f:
        cfg = json.load(f)

    pull = cfg["pull"]
    projects = pull["projects"]
    families = pull.get("model_families")

    print(f"Pulling from {len(projects)} W&B projects")
    if families:
        print(f"  Filtering to families: {families}")

    df = load_wandb_projects(projects, model_families=families)

    if df.empty:
        print("No results found.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\nSaved {len(df)} rows to {output_path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Tasks: {df['task'].nunique()}")
    print(f"  Checkpoints per model:")
    for m in sorted(df["model"].unique()):
        n = df[df["model"] == m]["checkpoint"].nunique()
        print(f"    {m}: {n}")


if __name__ == "__main__":
    main()
