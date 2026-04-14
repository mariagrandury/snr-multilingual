#!/usr/bin/env python3
"""Compute SNR metrics from evaluation results.

Usage:
    # From local results directory
    python scripts/compute_snr.py --results-dir results/

    # From all W&B projects in configs/wandb.json "pull" key
    python scripts/compute_snr.py --pull

    # From a single W&B project
    python scripts/compute_snr.py --wandb-project ist/SwissAI-QAT-evals

    # Specify models and tasks
    python scripts/compute_snr.py --pull --models Apertus-0.6B-from8B Qwen3-0.6B --tasks hellaswag piqa

    # With decision accuracy
    python scripts/compute_snr.py --pull --target-models SmolLM3-3B SmolLM3-3B-Base
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import compute_all_metrics
from src.snr.data import load_results_dir, load_wandb_projects, load_wandb_results

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SNR metrics")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=str, help="Local results directory")
    source.add_argument("--wandb-project", type=str, help="Single W&B project (entity/project)")
    source.add_argument(
        "--pull", action="store_true",
        help="Pull from all W&B projects listed in configs/wandb.json 'pull' key",
    )

    parser.add_argument("--output", type=str, default="results/snr/snr.csv", help="Output CSV path")
    parser.add_argument("--models", nargs="+", help="Model names for SNR (default: all)")
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to evaluate (default: all). Can be a stage name from configs/tasks.json.",
    )
    parser.add_argument(
        "--target-models", nargs="+",
        help="Target model names for decision accuracy (same order as --models)",
    )
    parser.add_argument(
        "--last-n", type=int, default=5,
        help="Number of final checkpoints for noise estimation (default: 5)",
    )

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
    elif args.wandb_project:
        print(f"Loading results from W&B: {args.wandb_project}")
        df = load_wandb_results(args.wandb_project)
    else:
        with open(CONFIGS_DIR / "wandb.json") as f:
            wandb_config = json.load(f)
        projects = wandb_config["pull"]
        print(f"Pulling from {len(projects)} W&B projects:")
        df = load_wandb_projects(projects)

    if df.empty:
        print("No results found.")
        sys.exit(1)

    print(f"\nLoaded {len(df)} rows total")
    print(f"  Models ({df['model'].nunique()}): {sorted(df['model'].unique())}")
    print(f"  Tasks: {df['task'].nunique()}")
    print(f"  Checkpoints per model: "
          f"{df.groupby('model')['checkpoint'].nunique().to_dict()}")

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
        if not valid_da.empty:
            print(f"  Mean decision accuracy: {valid_da.mean():.2%}")

    print(f"\nTop 10 tasks by SNR:")
    cols = ["task", "signal", "noise", "snr"]
    if "decision_accuracy" in results.columns:
        cols.append("decision_accuracy")
    print(results.nlargest(10, "snr")[cols].to_string(index=False))

    print(f"\nBottom 10 tasks by SNR:")
    print(results.nsmallest(10, "snr")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
