#!/usr/bin/env python3
"""Compute SNR metrics from evaluation results.

Usage:
    # From all W&B projects in configs/wandb.json
    python scripts/compute_snr.py --pull --target-size-group "~8B"

    # From local results directory
    python scripts/compute_snr.py --results-dir results/ --target-size-group "~8B"

    # From a single W&B project
    python scripts/compute_snr.py --wandb-project ist/SwissAI-QAT-evals
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
    source.add_argument("--pull", action="store_true", help="Pull from configs/wandb.json")

    parser.add_argument("--output", type=str, default="results/snr/snr.csv")
    parser.add_argument("--tasks", nargs="+", help="Tasks (or stage name from configs/tasks.json)")
    parser.add_argument("--target-size-group", type=str, help='Target size for DA (e.g., "~8B")')
    parser.add_argument("--last-n", type=int, default=5, help="Final checkpoints for noise")
    parser.add_argument("--min-models", type=int, default=2, help="Min models per size group")

    return parser.parse_args()


def resolve_tasks(task_args):
    if task_args is None:
        return None
    tasks_config = CONFIGS_DIR / "tasks.json"
    if tasks_config.exists():
        with open(tasks_config) as f:
            all_tasks = json.load(f)
        expanded = []
        for t in task_args:
            expanded.extend(all_tasks[t]) if t in all_tasks else expanded.append(t)
        return expanded
    return task_args


def main():
    args = parse_args()

    if args.results_dir:
        print(f"Loading from {args.results_dir}")
        df = load_results_dir(args.results_dir)
    elif args.wandb_project:
        print(f"Loading from W&B: {args.wandb_project}")
        df = load_wandb_results(args.wandb_project)
    else:
        with open(CONFIGS_DIR / "wandb.json") as f:
            cfg = json.load(f)
        pull = cfg["pull"]
        projects = pull["projects"]
        families = pull.get("model_families")
        print(f"Pulling from {len(projects)} W&B projects (families: {families})")
        df = load_wandb_projects(projects, model_families=families)

    if df.empty:
        print("No results found.")
        sys.exit(1)

    print(f"\n{len(df)} rows, {df['model'].nunique()} models, {df['task'].nunique()} tasks")

    results = compute_all_metrics(
        df,
        tasks=resolve_tasks(args.tasks),
        target_size_group=args.target_size_group,
        last_n_checkpoints=args.last_n,
        min_models=args.min_models,
    )

    if results.empty:
        print("No results computed.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSaved {len(results)} rows to {output_path}")
    print(f"  Size groups: {sorted(results['size_group'].unique())}")
    print(f"  SNR range: {results['snr'].min():.2f} – {results['snr'].max():.2f}")
    print(f"  Mean signal: {results['signal'].mean():.4f}")
    print(f"  Mean noise: {results['noise'].mean():.4f}")

    if "decision_accuracy" in results.columns:
        valid_da = results["decision_accuracy"].dropna()
        if not valid_da.empty:
            print(f"  Mean DA: {valid_da.mean():.2%} ({len(valid_da)} tasks)")

    for sg in sorted(results["size_group"].unique()):
        g = results[results["size_group"] == sg]
        print(f"\n  [{sg}] {len(g)} tasks, SNR median={g['snr'].median():.2f}")


if __name__ == "__main__":
    main()
