#!/usr/bin/env python3
"""Compute SNR metrics from evaluation results.

Usage:
    # From all W&B projects in configs/wandb.json
    python scripts/compute_snr.py --pull

    # From a single W&B project
    python scripts/compute_snr.py --wandb-project ist/SwissAI-QAT-evals

    # From local lm-eval result JSONs
    python scripts/compute_snr.py --results-dir results/

    # From a parquet previously saved by scripts/pull_from_wandb.py
    python scripts/compute_snr.py --input results/wandb_data.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import compute_all_metrics
from src.snr.data import load_results_dir, load_wandb_projects, load_wandb_results

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SNR metrics")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-dir", type=str, help="Local lm-eval results directory")
    source.add_argument("--wandb-project", type=str, help="Single W&B project (entity/project)")
    source.add_argument("--pull", action="store_true", help="Pull from configs/wandb.json")
    source.add_argument("--input", type=str, help="Parquet from scripts/pull_from_wandb.py")

    parser.add_argument("--output", type=str, default="results/snr/snr.csv")
    parser.add_argument(
        "--tasks", nargs="+",
        help="Exact task names, or stage keys from configs/tasks.json (e.g. 'pretraining'). "
             "Matching against task substrings is supported via --task-filter.",
    )
    parser.add_argument("--task-filter", type=str, help="Only tasks containing this substring")
    parser.add_argument("--last-n", type=int, default=5, help="Final checkpoints for noise")
    parser.add_argument("--min-models", type=int, default=2, help="Min models per size group")

    return parser.parse_args()


def resolve_tasks(task_args, available_tasks=None):
    """Expand stage names from configs/tasks.json into concrete task lists."""
    if task_args is None:
        return None
    tasks_config = CONFIGS_DIR / "tasks.json"
    stage_map = {}
    if tasks_config.exists():
        with open(tasks_config) as f:
            stage_map = json.load(f)
    expanded = []
    for t in task_args:
        if t in stage_map:
            expanded.extend(stage_map[t])
        else:
            expanded.append(t)
    return expanded


def load_source(args) -> pd.DataFrame:
    if args.results_dir:
        print(f"Loading from {args.results_dir}")
        return load_results_dir(args.results_dir)
    if args.wandb_project:
        print(f"Loading from W&B: {args.wandb_project}")
        return load_wandb_results(args.wandb_project)
    if args.input:
        print(f"Loading parquet: {args.input}")
        return pd.read_parquet(args.input)
    with open(CONFIGS_DIR / "wandb.json") as f:
        cfg = json.load(f)
    pull = cfg["pull"]
    projects = pull["projects"]
    families = pull.get("model_families")
    print(f"Pulling from {len(projects)} W&B projects (families: {families})")
    return load_wandb_projects(projects, model_families=families)


def main():
    args = parse_args()

    df = load_source(args)
    if df.empty:
        print("No results found.")
        sys.exit(1)

    print(f"\n{len(df)} rows, {df['model'].nunique()} models, {df['task'].nunique()} tasks")

    tasks = resolve_tasks(args.tasks)
    if args.task_filter:
        all_tasks = sorted(df["task"].unique())
        matched = [t for t in all_tasks if args.task_filter.lower() in t.lower()]
        tasks = sorted(set((tasks or []) + matched)) if tasks else matched
        print(f"Filtered to {len(matched)} tasks containing '{args.task_filter}'")

    results = compute_all_metrics(
        df,
        tasks=tasks,
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

    da_cols = [c for c in results.columns if c.startswith("da_")]
    for da_col in da_cols:
        valid_da = results[da_col].dropna()
        if not valid_da.empty:
            print(f"  Mean {da_col}: {valid_da.mean():.2%} ({len(valid_da)} tasks)")

    for (cat, sg), g in results.groupby(["category", "size_group"]):
        print(f"  [{cat}/{sg}] {len(g)} tasks, SNR median={g['snr'].median():.2f}")


if __name__ == "__main__":
    main()
