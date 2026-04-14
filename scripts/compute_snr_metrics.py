#!/usr/bin/env python3
"""Compute signal, noise, SNR, and decision accuracy from pulled W&B data.

Usage:
    python scripts/compute_snr_metrics.py
    python scripts/compute_snr_metrics.py --input results/wandb_data.parquet --tasks hellaswag
    python scripts/compute_snr_metrics.py --task-filter hellaswag
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute SNR metrics")
    parser.add_argument(
        "--input", type=str, default="results/wandb_data.parquet",
        help="Input parquet from pull_from_wandb.py",
    )
    parser.add_argument("--output", type=str, default="results/snr/snr.csv")
    parser.add_argument("--tasks", nargs="+", help="Exact task names")
    parser.add_argument("--task-filter", type=str, help="Only tasks containing this string")
    parser.add_argument("--last-n", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, {df['task'].nunique()} tasks")

    # Resolve tasks
    tasks = args.tasks
    if args.task_filter:
        all_tasks = sorted(df["task"].unique())
        tasks = [t for t in all_tasks if args.task_filter.lower() in t.lower()]
        print(f"Filtered to {len(tasks)} tasks matching '{args.task_filter}': {tasks}")

    results = compute_all_metrics(df, tasks=tasks, last_n_checkpoints=args.last_n)

    if results.empty:
        print("No results.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSaved {len(results)} rows to {output_path}")

    # Summary per (category, size_group)
    for (cat, sg), g in results.groupby(["category", "size_group"]):
        da_cols = [c for c in g.columns if c.startswith("da_")]
        da_info = ""
        for dc in da_cols:
            valid = g[dc].dropna()
            if not valid.empty:
                da_info += f"  {dc}: mean={valid.mean():.2f} ({len(valid)} tasks)"
        print(f"  [{cat}/{sg}] {len(g)} tasks, SNR median={g['snr'].median():.2f}{da_info}")


if __name__ == "__main__":
    main()
