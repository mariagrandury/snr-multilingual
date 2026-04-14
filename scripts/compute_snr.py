#!/usr/bin/env python3
"""Compute SNR metrics from evaluation results.

Usage:
    # From local results directory
    python scripts/compute_snr.py --results-dir results/ --output results/snr/snr_custom.csv

    # From W&B
    python scripts/compute_snr.py --wandb-project snr-experiments --output results/snr/snr_wandb.csv

    # Specify models and tasks
    python scripts/compute_snr.py --results-dir results/ --models SmolLM2-135M Qwen3-0.6B --tasks hellaswag piqa

    # With decision accuracy (requires --target-models)
    python scripts/compute_snr.py --results-dir results/ --target-models SmolLM3-3B SmolLM3-3B-Base
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import compute_all_metrics
from src.snr.data import load_results_dir, load_wandb_results

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SNR metrics")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results-dir", type=str, help="Local results directory"
    )
    source.add_argument(
        "--wandb-project", type=str, help="W&B project name"
    )

    parser.add_argument("--output", type=str, default="results/snr/snr.csv", help="Output CSV path")
    parser.add_argument("--models", nargs="+", help="Model names for SNR (default: all)")
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to evaluate (default: all). Can also be a stage name from configs/tasks.json.",
    )
    parser.add_argument(
        "--target-models", nargs="+",
        help="Target model names for decision accuracy (same order as --models)",
    )
    parser.add_argument(
        "--last-n", type=int, default=5,
        help="Number of final checkpoints for noise estimation (default: 5)",
    )
    parser.add_argument("--wandb-tags", nargs="+", help="W&B run tags to filter")

    return parser.parse_args()


def resolve_tasks(task_args: list[str] | None) -> list[str] | None:
    """Resolve task arguments, expanding stage names from configs/tasks.json."""
    if task_args is None:
        return None

    tasks_config = CONFIGS_DIR / "tasks.json"
    if tasks_config.exists():
        with open(tasks_config) as f:
            all_tasks = json.load(f)

        expanded = []
        for t in task_args:
            if t in all_tasks:
                expanded.extend(all_tasks[t])
            else:
                expanded.append(t)
        return expanded

    return task_args


def main():
    args = parse_args()

    # Load data
    if args.results_dir:
        print(f"Loading results from {args.results_dir}")
        df = load_results_dir(args.results_dir)
    else:
        with open(CONFIGS_DIR / "wandb.json") as f:
            wandb_config = json.load(f)
        entity = wandb_config["entity"]
        print(f"Loading results from W&B: {entity}/{args.wandb_project}")
        df = load_wandb_results(entity, args.wandb_project, tags=args.wandb_tags)

    if df.empty:
        print("No results found.")
        sys.exit(1)

    print(f"Loaded {len(df)} result rows")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Tasks: {sorted(df['task'].unique())}")

    # Resolve arguments
    tasks = resolve_tasks(args.tasks)

    # Compute metrics
    results = compute_all_metrics(
        df,
        tasks=tasks,
        models=args.models,
        target_models=args.target_models,
        last_n_checkpoints=args.last_n,
    )

    if results.empty:
        print("No results computed (insufficient data for any task).")
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSaved {len(results)} rows to {output_path}")
    print(f"\nSummary:")
    print(f"  Tasks with results: {len(results)}")
    print(f"  SNR range: {results['snr'].min():.2f} - {results['snr'].max():.2f}")
    print(f"  Mean signal: {results['signal'].mean():.4f}")
    print(f"  Mean noise: {results['noise'].mean():.4f}")
    if "decision_accuracy" in results.columns:
        valid_da = results["decision_accuracy"].dropna()
        print(f"  Mean decision accuracy: {valid_da.mean():.2%}")

    print(f"\nTop 5 tasks by SNR:")
    print(results.nlargest(5, "snr")[["task", "signal", "noise", "snr"]].to_string(index=False))


if __name__ == "__main__":
    main()
