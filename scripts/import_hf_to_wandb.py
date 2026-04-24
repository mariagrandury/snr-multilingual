#!/usr/bin/env python3
"""Import evaluation results from HuggingFace datasets and push to W&B.

Fetches the Allen AI signal-and-noise dataset (preliminary analysis data)
and logs it to our W&B project for unified analysis.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_hf_dataset(
    repo_id: str,
    split: str,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load a HF parquet dataset (e.g. allenai/signal-and-noise) into our schema.

    Returned columns: model_id, run_name, task, metric, score, checkpoint_index,
    model_size, data_mix.
    """
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=cache_dir,
    )

    data_dir = Path(local_path) / "data"
    parquet_files = sorted(data_dir.glob(f"{split}*.parquet"))
    if not parquet_files:
        parquet_files = sorted(Path(local_path).glob(f"*{split}*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' in {local_path}"
        )

    raw = pd.concat(
        [pd.read_parquet(f, engine="fastparquet") for f in parquet_files],
        ignore_index=True,
    )

    raw = raw.dropna(subset=["primary_score"])

    def _col(name, default=None):
        return raw[name] if name in raw.columns else pd.Series([default] * len(raw))

    size = _col("size").where(_col("size").notna() & (_col("size") != ""), None)
    return pd.DataFrame({
        "model_id": _col("model_path").where(_col("model_path").notna(), _col("model")).astype(str),
        "run_name": _col("model").fillna("").astype(str),
        "task": raw["task"].astype(str),
        "metric": "primary_score",
        "score": raw["primary_score"].astype(float),
        "checkpoint_index": _col("step", 0).fillna(0).astype(int),
        "model_size": size,
        "data_mix": _col("mix").where(_col("mix").notna(), None),
    })

# DataDecide sizes closest to our custom model sizes (175M/350M/600M/1B)
MATCHING_SIZES = ["150M", "300M", "750M", "1B"]


def parse_args():
    parser = argparse.ArgumentParser(description="Import HuggingFace eval data to W&B")
    parser.add_argument(
        "--repo",
        default="allenai/signal-and-noise",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument("--split", default="core", help="Dataset split name")
    parser.add_argument("--tag", default="preliminary", help="Tag for imported runs")

    # Filtering options
    filter_group = parser.add_argument_group("filtering")
    filter_group.add_argument(
        "--match-sizes",
        action="store_true",
        help=f"Filter to sizes matching our models: {MATCHING_SIZES}",
    )
    filter_group.add_argument(
        "--match-tasks",
        action="store_true",
        help="Filter to tasks matching configs/tasks.json (all stages combined)",
    )
    filter_group.add_argument(
        "--all", action="store_true", help="Import everything (no filters)"
    )
    filter_group.add_argument(
        "--sizes", nargs="+", help="Explicit list of sizes to include"
    )
    filter_group.add_argument(
        "--tasks", nargs="+", help="Explicit list of tasks to include"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Load and display stats only"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Save locally only")
    parser.add_argument("--cache-dir", help="Local cache for HF download")
    return parser.parse_args()


def _load_our_tasks() -> set[str]:
    """Load all task names from configs/tasks.json across all stages."""
    with open(CONFIGS_DIR / "tasks.json") as f:
        all_tasks = json.load(f)
    tasks = set()
    for stage_tasks in all_tasks.values():
        tasks.update(stage_tasks)
    return tasks


def main():
    args = parse_args()

    # Load W&B config
    with open(CONFIGS_DIR / "wandb.json") as f:
        wandb_config = json.load(f)
    entity = wandb_config["entity"]
    project = wandb_config["project"]["evals"]

    # Load data from HuggingFace
    print(f"Loading {args.repo} (split={args.split})...")
    df = load_hf_dataset(args.repo, args.split, cache_dir=args.cache_dir)
    print(
        f"Total: {len(df)} rows, {df['task'].nunique()} tasks, {df['model_size'].nunique()} sizes"
    )

    # Apply filters
    if not args.all:
        # Size filtering
        sizes = args.sizes or (MATCHING_SIZES if args.match_sizes else None)
        if sizes:
            df = df[df["model_size"].isin(sizes)]
            print(f"Filtered to sizes {sizes}: {len(df)} rows")

        # Task filtering
        if args.match_tasks:
            our_tasks = _load_our_tasks()
            # Match both exact and partial (some HF tasks have :mc/:rc suffixes)
            mask = df["task"].isin(our_tasks)
            for task in our_tasks:
                mask |= df["task"].str.startswith(task + "_")
            df = df[mask]
            print(
                f"Filtered to matching tasks ({len(our_tasks)} patterns): {len(df)} rows"
            )
        elif args.tasks:
            df = df[df["task"].isin(args.tasks)]
            print(f"Filtered to tasks {args.tasks}: {len(df)} rows")

    # Display stats
    print(f"\nDataset summary:")
    print(f"  Rows: {len(df)}")
    print(
        f"  Tasks ({df['task'].nunique()}): {sorted(df['task'].unique())[:20]}{'...' if df['task'].nunique() > 20 else ''}"
    )
    print(f"  Sizes: {sorted(df['model_size'].dropna().unique())}")
    print(f"  Mixes: {sorted(df['data_mix'].dropna().unique())}")
    print(f"  Models: {df['model_id'].nunique()}")

    if args.dry_run:
        print("\nDry run — not saving or pushing.")
        return

    # Save locally
    output_dir = RESULTS_DIR / "imported" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.split}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} rows to {output_file}")

    # Push to W&B
    if not args.no_wandb:
        _push_to_wandb(df, entity, project, tag=args.tag, split=args.split)


def _push_to_wandb(df, entity, project, *, tag, split):
    """Push evaluation data to W&B as one run per model, with task metrics as summary."""
    import wandb

    # Group by unique model runs
    group_cols = ["run_name", "model_size", "data_mix"]
    available_cols = [c for c in group_cols if c in df.columns and df[c].notna().any()]
    grouped = df.groupby(available_cols)

    n_runs = 0
    total_groups = len(grouped)
    for group_key, group_df in grouped:
        if isinstance(group_key, str):
            group_key = (group_key,)

        model_id = group_df["model_id"].iloc[0]
        model_name = (
            group_df["run_name"].iloc[0] if "run_name" in group_df else model_id
        )
        model_size = group_df["model_size"].iloc[0]
        data_mix = group_df["data_mix"].iloc[0]
        run_name = f"{tag}_{model_name}"

        # Get unique checkpoints for this model
        checkpoints = sorted(group_df["checkpoint_index"].unique())

        wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            group=str(model_name),
            job_type="import",
            tags=[tag, split] + ([str(model_size)] if model_size else []),
            config={
                "model_id": model_id,
                "model_size": model_size,
                "data_mix": data_mix,
                "source_repo": f"hf://{tag}",
                "split": split,
            },
        )

        # Log each checkpoint as a step
        for ckpt_idx in checkpoints:
            ckpt_df = group_df[group_df["checkpoint_index"] == ckpt_idx]
            log_dict = {"checkpoint_index": int(ckpt_idx)}
            for _, row in ckpt_df.iterrows():
                log_dict[f"{row['task']}/{row['metric']}"] = row["score"]
            wandb.log(log_dict)

        # Also set summary with latest checkpoint
        latest_df = group_df[group_df["checkpoint_index"] == checkpoints[-1]]
        for _, row in latest_df.iterrows():
            wandb.summary[f"{row['task']}/{row['metric']}"] = row["score"]
        wandb.summary["checkpoint_index"] = int(checkpoints[-1])

        wandb.finish()
        n_runs += 1
        if n_runs % 10 == 0:
            print(f"  Pushed {n_runs}/{total_groups} runs...")

    print(f"Pushed {n_runs} runs to {entity}/{project} (tag={tag})")


if __name__ == "__main__":
    main()
