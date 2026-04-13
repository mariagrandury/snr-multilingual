#!/usr/bin/env python3
"""Compute SNR and generate visuals for imported evaluation data.

Thin wrapper that loads AllenAI and/or QAT data via src.snr.data,
computes SNR via src.snr.compute, and generates plots via src.analysis.

Usage:
    # Full analysis (both datasets)
    python scripts/analyze_imported.py

    # Only AllenAI
    python scripts/analyze_imported.py --allenai-only

    # Only QAT
    python scripts/analyze_imported.py --qat-only

    # Custom output directory / format
    python scripts/analyze_imported.py --output results/figures/ --format pdf
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snr.compute import compute_all_metrics
from src.snr.data import load_csv_results, load_imported_results
from src.analysis.plots import (
    benchmark_ranking,
    correlation_matrix,
    cross_dataset_scatter,
    side_by_side_ranking,
    side_by_side_scatter,
    signal_noise_scatter,
    snr_vs_decision_accuracy,
)
from src.analysis.summary import (
    dataset_metadata,
    recommend_benchmarks,
    results_table,
    save_metadata,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze imported evaluation data")
    parser.add_argument("--output", default="results/figures/", help="Output directory")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--allenai-only", action="store_true")
    group.add_argument("--qat-only", action="store_true")
    return parser.parse_args()


def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _generate_plots(results: pd.DataFrame, output_dir: Path, prefix: str, fmt: str):
    """Generate standard SNR plots for a single dataset — reuses src.analysis.plots."""
    agg = results.groupby("task").agg({
        "signal": "mean", "noise": "mean", "snr": "mean",
        "decision_accuracy": "mean", "n_models_small": "first",
    }).reset_index()

    _save_fig(
        snr_vs_decision_accuracy(agg, title=f"SNR vs Decision Accuracy ({prefix.upper()})"),
        output_dir / f"snr_vs_da_{prefix}.{fmt}",
    )
    _save_fig(
        signal_noise_scatter(agg, title=f"Signal vs Noise ({prefix.upper()})"),
        output_dir / f"signal_noise_{prefix}.{fmt}",
    )
    for m in ["snr", "decision_accuracy"]:
        _save_fig(
            benchmark_ranking(agg, metric=m, title=f"Ranking by {m.upper()} ({prefix.upper()})", top_n=25),
            output_dir / f"ranking_{m}_{prefix}.{fmt}",
        )
    _save_fig(correlation_matrix(agg), output_dir / f"correlations_{prefix}.{fmt}")

    # Summary + recommendations — need small_size/large_size columns for results_table
    agg["small_size"] = results["small_size"].mode().iloc[0] if "small_size" in results else "proxy"
    agg["large_size"] = results["large_size"].mode().iloc[0] if "large_size" in results else "target"
    agg["n_models_small"] = agg.get("n_models_small", 0)

    table = results_table(agg)
    table.to_csv(output_dir / f"summary_{prefix}.csv", index=False)
    print(f"  Saved summary_{prefix}.csv")
    print(table.head(15).to_string(index=False))

    rec = recommend_benchmarks(agg)
    rec.to_csv(output_dir / f"recommendations_{prefix}.csv", index=False)
    print(f"\n  Top recommended benchmarks ({prefix}):")
    print(rec.to_string(index=False))


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    # ── Load data ──
    allenai_df = pd.DataFrame()
    allenai_meta = {}
    qat_df = pd.DataFrame()
    qat_meta = {}

    if not args.qat_only:
        csv_path = RESULTS_DIR / "imported" / "preliminary" / "core.csv"
        allenai_df = load_csv_results(csv_path)

    if not args.allenai_only:
        qat_df, qat_meta = load_imported_results(tag="QAT")

    if allenai_df.empty and qat_df.empty:
        print("No data found. Run import scripts first.")
        sys.exit(1)

    # ── Compute SNR ──
    allenai_results = pd.DataFrame()
    qat_results = pd.DataFrame()

    if not allenai_df.empty:
        tasks = sorted(allenai_df["task"].unique())
        print(f"\nComputing AllenAI SNR for {len(tasks)} tasks...")
        allenai_results = compute_all_metrics(
            allenai_df, tasks,
            small_sizes=["150M", "300M"], large_sizes=["750M", "1B"],
            noise_type="checkpoint", n_last_checkpoints=5,
        )
        allenai_results = allenai_results.dropna(subset=["snr"])
        allenai_results = allenai_results[np.isfinite(allenai_results["snr"])]

    if not qat_df.empty:
        tasks = sorted(qat_df["task"].unique())
        print(f"\nComputing QAT SNR for {len(tasks)} tasks...")
        qat_results = compute_all_metrics(
            qat_df, tasks,
            small_sizes=["0.6B", "1.7B"], large_sizes=["3B", "8B"],
            noise_type="checkpoint", n_last_checkpoints=5,
        )
        qat_results = qat_results.dropna(subset=["snr"])
        qat_results = qat_results[np.isfinite(qat_results["snr"])]

    # ── Save SNR CSVs + metadata ──
    snr_dir = RESULTS_DIR / "snr"
    snr_dir.mkdir(parents=True, exist_ok=True)

    if not allenai_results.empty:
        allenai_results.to_csv(snr_dir / "snr_allenai.csv", index=False)
        meta = dataset_metadata(allenai_df, label="AllenAI DataDecide")
        meta["snr_summary"] = dataset_metadata(allenai_results, label="AllenAI SNR results")
        save_metadata(meta, snr_dir / "metadata_allenai.json")

    if not qat_results.empty:
        qat_results.to_csv(snr_dir / "snr_qat.csv", index=False)
        meta = dataset_metadata(qat_df, model_catalog=qat_meta, label="QAT Apertus")
        meta["snr_summary"] = dataset_metadata(qat_results, label="QAT SNR results")
        save_metadata(meta, snr_dir / "metadata_qat.json")

    # ── Generate plots ──
    if not allenai_results.empty:
        print("\n=== AllenAI Plots ===")
        _generate_plots(allenai_results, output_dir, "allenai", fmt)

    if not qat_results.empty:
        print("\n=== QAT Plots ===")
        _generate_plots(qat_results, output_dir, "qat", fmt)

    # ── Comparative plots (reusing src.analysis.plots) ──
    if not allenai_results.empty and not qat_results.empty:
        print("\n=== Comparative Plots ===")
        _save_fig(
            side_by_side_scatter(allenai_results, qat_results,
                                 label_a="AllenAI (DataDecide)", label_b="QAT (Apertus)"),
            output_dir / f"comparison_snr_vs_da.{fmt}",
        )
        _save_fig(
            cross_dataset_scatter(allenai_results, qat_results,
                                   label_a="AllenAI DataDecide", label_b="QAT Apertus"),
            output_dir / f"snr_correlation_allenai_vs_qat.{fmt}",
        )
        _save_fig(
            side_by_side_ranking(allenai_results, qat_results,
                                  label_a="AllenAI", label_b="QAT"),
            output_dir / f"comparison_top_benchmarks.{fmt}",
        )

    # Copy to documents/public for slides
    public_dir = Path(__file__).resolve().parent.parent / "documents" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob(f"*.{fmt}"):
        (public_dir / f.name).write_bytes(f.read_bytes())
    print(f"\nCopied figures to {public_dir}/")
    print(f"Done! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
