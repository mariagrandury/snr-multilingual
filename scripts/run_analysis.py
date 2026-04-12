#!/usr/bin/env python3
"""Generate analysis plots and summary tables from SNR results.

Usage:
    # From saved CSV results
    python scripts/run_analysis.py --input results/snr/ --output results/figures/

    # From W&B
    python scripts/run_analysis.py --stage pretraining --output results/figures/

    # All stages
    python scripts/run_analysis.py --all-stages --output results/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.plots import (
    benchmark_ranking,
    correlation_matrix,
    signal_noise_scatter,
    snr_vs_decision_accuracy,
    stage_comparison,
)
from src.analysis.summary import (
    recommend_benchmarks,
    results_table,
    stage_recommendations,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
STAGES = ["pretraining", "midtraining", "posttraining"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SNR analysis plots and tables"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=str, help="Directory with snr_<stage>.csv files"
    )
    source.add_argument("--stage", choices=STAGES, help="Load from W&B for this stage")
    source.add_argument(
        "--all-stages", action="store_true", help="Load from W&B for all stages"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot file format",
    )
    parser.add_argument(
        "--no-recommend", action="store_true", help="Skip recommendation tables"
    )
    return parser.parse_args()


def load_results(args) -> dict[str, pd.DataFrame]:
    """Load SNR results from CSV files or compute from W&B."""
    if args.input:
        input_dir = Path(args.input)
        results = {}
        for stage in STAGES:
            csv_file = input_dir / f"snr_{stage}.csv"
            if csv_file.exists():
                results[stage] = pd.read_csv(csv_file)
                print(f"Loaded {stage}: {len(results[stage])} rows from {csv_file}")
        return results

    # Load from W&B via compute_snr
    with open(CONFIGS_DIR / "wandb.json") as f:
        wandb_config = json.load(f)

    from src.snr.compute import compute_all_metrics
    from src.snr.data import load_wandb_results

    entity = wandb_config["entity"]
    project = wandb_config["project"]["evals"]
    df = load_wandb_results(entity, project)

    with open(CONFIGS_DIR / "tasks.json") as f:
        all_tasks = json.load(f)

    stages = [args.stage] if args.stage else STAGES
    results = {}
    for stage in stages:
        if stage in all_tasks:
            results[stage] = compute_all_metrics(df, all_tasks[stage])
            print(f"Computed {stage}: {len(results[stage])} rows")

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    results = load_results(args)
    if not results:
        print("No results found.")
        sys.exit(1)

    # Per-stage plots
    for stage, df in results.items():
        print(f"\n--- {stage} ---")

        # SNR vs Decision Accuracy
        fig = snr_vs_decision_accuracy(df, title=f"SNR vs Decision Accuracy ({stage})")
        fig.savefig(
            output_dir / f"snr_vs_da_{stage}.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"  Saved snr_vs_da_{stage}.{fmt}")

        # Signal vs Noise
        fig = signal_noise_scatter(df, title=f"Signal vs Noise ({stage})")
        fig.savefig(
            output_dir / f"signal_noise_{stage}.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"  Saved signal_noise_{stage}.{fmt}")

        # Benchmark ranking
        fig = benchmark_ranking(df, title=f"Benchmark Ranking by SNR ({stage})")
        fig.savefig(
            output_dir / f"ranking_snr_{stage}.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"  Saved ranking_snr_{stage}.{fmt}")

        # Decision accuracy ranking
        fig = benchmark_ranking(
            df,
            metric="decision_accuracy",
            title=f"Benchmark Ranking by Decision Accuracy ({stage})",
        )
        fig.savefig(
            output_dir / f"ranking_da_{stage}.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"  Saved ranking_da_{stage}.{fmt}")

        # Correlation matrix
        fig = correlation_matrix(df)
        fig.savefig(
            output_dir / f"correlations_{stage}.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"  Saved correlations_{stage}.{fmt}")

        # Summary table
        table = results_table(df)
        table.to_csv(output_dir / f"summary_{stage}.csv", index=False)
        print(f"  Saved summary_{stage}.csv")
        print(table.to_string(index=False))

        # Recommendations
        if not args.no_recommend:
            rec = recommend_benchmarks(df)
            rec.to_csv(output_dir / f"recommendations_{stage}.csv", index=False)
            print(f"\n  Recommended benchmarks for {stage}:")
            print(rec.to_string(index=False))

    # Cross-stage comparison
    if len(results) > 1:
        fig = stage_comparison(results)
        fig.savefig(
            output_dir / f"stage_comparison.{fmt}", dpi=150, bbox_inches="tight"
        )
        plt_close(fig)
        print(f"\nSaved stage_comparison.{fmt}")

        if not args.no_recommend:
            recs = stage_recommendations(results)
            for stage, rec in recs.items():
                print(f"\n  {stage} recommendations:")
                print(rec.to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}/")


def plt_close(fig):
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
