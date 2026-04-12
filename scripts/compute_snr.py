#!/usr/bin/env python3
"""Compute SNR metrics from W&B or local evaluation results.

Usage:
    # From W&B (default)
    python scripts/compute_snr.py --stage pretraining

    # From local results
    python scripts/compute_snr.py --stage pretraining --local

    # With benchmark noise (requires sample data)
    python scripts/compute_snr.py --stage pretraining --noise benchmark

    # Custom sizes
    python scripts/compute_snr.py --stage pretraining --small-sizes 175M 350M --large-sizes 1B

    # Save results without W&B
    python scripts/compute_snr.py --stage pretraining --no-wandb --output results/snr/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import (
    compute_all_metrics,
    compute_fold_scores_from_samples,
    log_results_to_wandb,
)
from src.snr.data import load_local_results, load_samples, load_wandb_results

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SNR metrics")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pretraining", "midtraining", "posttraining"],
        help="Training stage (determines task list)",
    )
    parser.add_argument(
        "--local", action="store_true", help="Load from local results/ instead of W&B"
    )
    parser.add_argument(
        "--noise",
        default="checkpoint",
        choices=["checkpoint", "benchmark"],
        help="Noise computation method (default: checkpoint)",
    )
    parser.add_argument(
        "--small-sizes", nargs="+", help="Proxy model sizes (e.g., 175M 350M)"
    )
    parser.add_argument(
        "--large-sizes", nargs="+", help="Target model sizes (e.g., 1B)"
    )
    parser.add_argument(
        "--n-last", type=int, default=5, help="Last N checkpoints for signal/noise"
    )
    parser.add_argument(
        "--k-folds", type=int, default=5, help="Number of folds for benchmark noise"
    )
    parser.add_argument("--tags", nargs="+", help="W&B tags to filter runs")
    parser.add_argument(
        "--no-wandb", action="store_true", help="Don't log results to W&B"
    )
    parser.add_argument("--output", type=str, help="Save results to this directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load task list for this stage
    with open(CONFIGS_DIR / "tasks.json") as f:
        all_tasks = json.load(f)
    tasks = all_tasks[args.stage]
    print(f"Stage: {args.stage}, Tasks: {len(tasks)}")

    # Load W&B config
    with open(CONFIGS_DIR / "wandb.json") as f:
        wandb_config = json.load(f)
    entity = wandb_config["entity"]
    project = wandb_config["project"]["evals"]

    # Load evaluation data
    if args.local:
        df = load_local_results()
    else:
        df = load_wandb_results(entity, project, tags=args.tags)

    if df.empty:
        print("No evaluation data found. Run evaluations first.")
        sys.exit(1)

    # Compute fold scores for benchmark noise
    fold_scores = None
    if args.noise == "benchmark":
        print("Computing benchmark noise from sample data...")
        results_dir = Path(__file__).resolve().parent.parent / "results"
        samples_map = {}
        for task in tasks:
            task_samples = {}
            for model_dir in results_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for rev_dir in model_dir.iterdir():
                    if not rev_dir.is_dir():
                        continue
                    loaded = load_samples(rev_dir)
                    if task in loaded:
                        model_key = f"{model_dir.name}/{rev_dir.name}"
                        task_samples[model_key] = loaded[task]
            if task_samples:
                samples_map[task] = task_samples

        if samples_map:
            fold_scores = compute_fold_scores_from_samples(samples_map, k=args.k_folds)
            print(f"Computed fold scores for {len(fold_scores)} tasks")
        else:
            print("No sample data found. Falling back to checkpoint noise.")

    # Compute metrics
    results_df = compute_all_metrics(
        df,
        tasks,
        small_sizes=args.small_sizes,
        large_sizes=args.large_sizes,
        noise_type=args.noise if fold_scores else "checkpoint",
        fold_scores=fold_scores,
        n_last_checkpoints=args.n_last,
    )

    if results_df.empty:
        print("No results computed. Check that evaluation data matches the task list.")
        sys.exit(1)

    # Display results
    print(f"\n{'='*80}")
    print(f"SNR Results for {args.stage} ({len(results_df)} task-size combinations)")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

    # Save locally
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"snr_{args.stage}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    # Log to W&B
    if not args.no_wandb:
        log_results_to_wandb(results_df, entity, project, stage=args.stage)


if __name__ == "__main__":
    main()
