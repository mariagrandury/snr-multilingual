#!/usr/bin/env python3
"""Compute SNR and generate visuals for imported evaluation data.

Loads both the AllenAI preliminary data and QAT imported results,
computes SNR metrics, and produces comparative visualizations.

Usage:
    python scripts/analyze_imported.py --output results/figures/
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

from src.snr import metrics
from src.snr.data import load_local_results, get_primary_metric
from src.snr.compute import compute_all_metrics
from src.analysis.plots import (
    snr_vs_decision_accuracy,
    signal_noise_scatter,
    benchmark_ranking,
    stage_comparison,
    correlation_matrix,
)
from src.analysis.summary import results_table, recommend_benchmarks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze imported evaluation data")
    parser.add_argument("--output", default="results/figures/", help="Output directory")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return parser.parse_args()


# ─── AllenAI Data Loading ──────────────────────────────────────────────────


def load_allenai_data() -> pd.DataFrame:
    """Load AllenAI signal-and-noise preliminary data from saved CSV."""
    csv_path = RESULTS_DIR / "imported" / "preliminary" / "core.csv"
    if not csv_path.exists():
        print(f"AllenAI data not found at {csv_path}. Run import_hf_to_wandb.py first.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    print(f"AllenAI: {len(df)} rows, {df['task'].nunique()} tasks, sizes={sorted(df['model_size'].dropna().unique())}")
    return df


# ─── QAT Data Loading ─────────────────────────────────────────────────────


def load_qat_data() -> pd.DataFrame:
    """Load QAT imported results from local JSON files into canonical format."""
    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        imported_dir = model_dir / "imported"
        if not imported_dir.is_dir():
            continue

        for results_file in sorted(imported_dir.glob("results_*.json")):
            with open(results_file) as f:
                data = json.load(f)

            if data.get("import_tag") != "QAT":
                continue

            model_name = data.get("source_run_name", model_dir.name)
            model_size = _parse_qat_size(model_name)
            # Use training variant as "data_mix" for signal computation
            data_mix = _parse_qat_variant(model_name)

            for ckpt_idx, checkpoint in enumerate(data.get("checkpoints", [])):
                for task_name, task_results in checkpoint.get("results", {}).items():
                    for metric_key, value in task_results.items():
                        if metric_key == "alias" or "stderr" in metric_key:
                            continue
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue
                        rows.append({
                            "model_id": model_name,
                            "revision": str(ckpt_idx),
                            "checkpoint_index": ckpt_idx,
                            "task": task_name,
                            "metric": metric_key,
                            "score": value,
                            "model_size": model_size,
                            "data_mix": data_mix,
                            "seed": None,
                            "run_id": None,
                            "run_name": model_name,
                        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"QAT: {len(df)} rows, {df['task'].nunique()} tasks, "
              f"sizes={sorted(df['model_size'].dropna().unique())}, "
              f"models={df['model_id'].nunique()}")
    return df


def _parse_qat_size(name: str) -> str:
    """Extract size from QAT model name like 'Apertus-1.7B-from8B-long'."""
    import re
    # First number with B/M suffix is the model size
    m = re.search(r"(\d+\.?\d*[BM])", name)
    return m.group(1).upper() if m else "unknown"


def _parse_qat_variant(name: str) -> str:
    """Extract training variant as 'data_mix' proxy."""
    name_lower = name.lower()
    if "sft" in name_lower:
        return "sft"
    if "longctx" in name_lower:
        return "longctx"
    if "long" in name_lower:
        return "long"
    if "instruct" in name_lower:
        return "instruct"
    if "base" in name_lower:
        return "base"
    return "base"


# ─── SNR Computation ──────────────────────────────────────────────────────


def compute_qat_snr(df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """Compute SNR for QAT data.

    For QAT, signal measures separation across training variants (base/SFT/long/etc.)
    at the same model size. Decision accuracy compares small vs large Apertus models.
    """
    return compute_all_metrics(
        df,
        tasks,
        small_sizes=["0.6B", "1.7B"],
        large_sizes=["3B", "8B"],
        noise_type="checkpoint",
        n_last_checkpoints=5,
    )


def compute_allenai_snr(df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    """Compute SNR for AllenAI DataDecide data.

    Standard SNR: signal from data mix variation, noise from checkpoint variation.
    """
    return compute_all_metrics(
        df,
        tasks,
        small_sizes=["150M", "300M"],
        large_sizes=["750M", "1B"],
        noise_type="checkpoint",
        n_last_checkpoints=5,
    )


# ─── Visualization ────────────────────────────────────────────────────────


def generate_visuals(
    allenai_results: pd.DataFrame,
    qat_results: pd.DataFrame,
    output_dir: Path,
    fmt: str,
):
    """Generate all analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── AllenAI plots ──
    if not allenai_results.empty:
        print("\n=== AllenAI (DataDecide) Plots ===")
        _generate_dataset_plots(allenai_results, output_dir, "allenai", fmt)

    # ── QAT plots ──
    if not qat_results.empty:
        print("\n=== QAT Plots ===")
        _generate_dataset_plots(qat_results, output_dir, "qat", fmt)

    # ── Comparative plots ──
    if not allenai_results.empty and not qat_results.empty:
        print("\n=== Comparative Plots ===")
        _generate_comparative_plots(allenai_results, qat_results, output_dir, fmt)


def _generate_dataset_plots(results: pd.DataFrame, output_dir: Path, prefix: str, fmt: str):
    """Generate standard SNR plots for a single dataset."""
    # Aggregate: one row per task (mean across sizes)
    agg = results.groupby("task").agg({
        "signal": "mean",
        "noise": "mean",
        "snr": "mean",
        "decision_accuracy": "mean",
        "n_models_small": "first",
        "n_mixes": "first",
    }).reset_index()

    # SNR vs Decision Accuracy
    fig = snr_vs_decision_accuracy(agg, title=f"SNR vs Decision Accuracy ({prefix.upper()})")
    fig.savefig(output_dir / f"snr_vs_da_{prefix}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved snr_vs_da_{prefix}.{fmt}")

    # Signal vs Noise
    fig = signal_noise_scatter(agg, title=f"Signal vs Noise ({prefix.upper()})")
    fig.savefig(output_dir / f"signal_noise_{prefix}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved signal_noise_{prefix}.{fmt}")

    # Rankings
    for metric_name in ["snr", "decision_accuracy"]:
        fig = benchmark_ranking(agg, metric=metric_name,
                                title=f"Benchmark Ranking by {metric_name.upper()} ({prefix.upper()})",
                                top_n=25)
        fig.savefig(output_dir / f"ranking_{metric_name}_{prefix}.{fmt}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved ranking_{metric_name}_{prefix}.{fmt}")

    # Correlation matrix
    fig = correlation_matrix(agg)
    fig.savefig(output_dir / f"correlations_{prefix}.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved correlations_{prefix}.{fmt}")

    # Summary table
    # Build a summary-compatible df
    summary_df = agg.copy()
    summary_df["small_size"] = results["small_size"].mode().iloc[0] if "small_size" in results else "proxy"
    summary_df["large_size"] = results["large_size"].mode().iloc[0] if "large_size" in results else "target"
    summary_df["n_models_small"] = summary_df.get("n_models_small", 0)
    table = results_table(summary_df)
    table.to_csv(output_dir / f"summary_{prefix}.csv", index=False)
    print(f"  Saved summary_{prefix}.csv")
    print(table.head(15).to_string(index=False))

    # Recommendations
    rec = recommend_benchmarks(summary_df)
    rec.to_csv(output_dir / f"recommendations_{prefix}.csv", index=False)
    print(f"\n  Top recommended benchmarks ({prefix}):")
    print(rec.to_string(index=False))


def _generate_comparative_plots(
    allenai_df: pd.DataFrame,
    qat_df: pd.DataFrame,
    output_dir: Path,
    fmt: str,
):
    """Generate plots comparing AllenAI and QAT results side by side."""
    # Aggregate both
    agg_a = allenai_df.groupby("task").agg({"snr": "mean", "decision_accuracy": "mean"}).reset_index()
    agg_a["source"] = "AllenAI (DataDecide)"
    agg_q = qat_df.groupby("task").agg({"snr": "mean", "decision_accuracy": "mean"}).reset_index()
    agg_q["source"] = "QAT (Apertus)"

    # Find common tasks
    common_tasks = set(agg_a["task"]) & set(agg_q["task"])
    print(f"  Common tasks between AllenAI and QAT: {len(common_tasks)}")

    if len(common_tasks) < 3:
        print("  Not enough common tasks for comparative analysis.")
        return

    agg_a_common = agg_a[agg_a["task"].isin(common_tasks)]
    agg_q_common = agg_q[agg_q["task"].isin(common_tasks)]

    # Side-by-side SNR vs DA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, agg, title in [
        (ax1, agg_a_common, "AllenAI (DataDecide)"),
        (ax2, agg_q_common, "QAT (Apertus)"),
    ]:
        valid = agg.dropna(subset=["snr", "decision_accuracy"])
        valid = valid[np.isfinite(valid["snr"])]
        ax.scatter(valid["snr"], valid["decision_accuracy"], s=50, alpha=0.7)
        for _, row in valid.iterrows():
            label = row["task"][:20]
            ax.annotate(label, (row["snr"], row["decision_accuracy"]),
                       fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("SNR")
        ax.set_ylabel("Decision Accuracy")
        ax.set_title(title)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SNR vs Decision Accuracy Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_snr_vs_da.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison_snr_vs_da.{fmt}")

    # SNR correlation between datasets
    merged = agg_a_common[["task", "snr"]].merge(
        agg_q_common[["task", "snr"]], on="task", suffixes=("_allenai", "_qat")
    )
    merged = merged.dropna()
    merged = merged[np.isfinite(merged["snr_allenai"]) & np.isfinite(merged["snr_qat"])]

    if len(merged) >= 3:
        from scipy import stats
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(merged["snr_allenai"], merged["snr_qat"], s=50, alpha=0.7)
        for _, row in merged.iterrows():
            ax.annotate(row["task"][:18], (row["snr_allenai"], row["snr_qat"]),
                       fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

        r, p = stats.pearsonr(merged["snr_allenai"], merged["snr_qat"])
        ax.set_xlabel("SNR (AllenAI DataDecide)")
        ax.set_ylabel("SNR (QAT Apertus)")
        ax.set_title(f"SNR Correlation: AllenAI vs QAT (R={r:.3f})")
        # Diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"snr_correlation_allenai_vs_qat.{fmt}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved snr_correlation_allenai_vs_qat.{fmt} (R={r:.3f})")

    # Top benchmarks comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    for ax, agg, title in [
        (ax1, agg_a_common.nlargest(20, "snr").sort_values("snr"), "AllenAI Top 20"),
        (ax2, agg_q_common.nlargest(20, "snr").sort_values("snr"), "QAT Top 20"),
    ]:
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(agg)))
        ax.barh([t[:22] for t in agg["task"]], agg["snr"], color=colors)
        ax.set_xlabel("SNR")
        ax.set_title(title)

    fig.suptitle("Top Benchmarks by SNR", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_top_benchmarks.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison_top_benchmarks.{fmt}")


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    output_dir = Path(args.output)

    # Load data
    print("Loading datasets...")
    allenai_df = load_allenai_data()
    qat_df = load_qat_data()

    if allenai_df.empty and qat_df.empty:
        print("No data found. Run import scripts first.")
        sys.exit(1)

    # Select tasks: use overlap between datasets + our config
    with open(CONFIGS_DIR / "tasks.json") as f:
        our_tasks = set()
        for stage_tasks in json.load(f).values():
            our_tasks.update(stage_tasks)

    # Compute SNR
    allenai_results = pd.DataFrame()
    qat_results = pd.DataFrame()

    if not allenai_df.empty:
        allenai_tasks = sorted(allenai_df["task"].unique())
        print(f"\nComputing AllenAI SNR for {len(allenai_tasks)} tasks...")
        allenai_results = compute_allenai_snr(allenai_df, allenai_tasks)
        allenai_results = allenai_results.dropna(subset=["snr"])
        allenai_results = allenai_results[np.isfinite(allenai_results["snr"])]
        print(f"AllenAI: {len(allenai_results)} valid task-size results")

    if not qat_df.empty:
        qat_tasks = sorted(qat_df["task"].unique())
        print(f"\nComputing QAT SNR for {len(qat_tasks)} tasks...")
        qat_results = compute_qat_snr(qat_df, qat_tasks)
        qat_results = qat_results.dropna(subset=["snr"])
        qat_results = qat_results[np.isfinite(qat_results["snr"])]
        print(f"QAT: {len(qat_results)} valid task-size results")

    # Save SNR CSVs
    snr_dir = RESULTS_DIR / "snr"
    snr_dir.mkdir(parents=True, exist_ok=True)
    if not allenai_results.empty:
        allenai_results.to_csv(snr_dir / "snr_allenai.csv", index=False)
        print(f"Saved {snr_dir / 'snr_allenai.csv'}")
    if not qat_results.empty:
        qat_results.to_csv(snr_dir / "snr_qat.csv", index=False)
        print(f"Saved {snr_dir / 'snr_qat.csv'}")

    # Generate visuals
    generate_visuals(allenai_results, qat_results, output_dir, args.format)

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
