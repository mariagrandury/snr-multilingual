#!/usr/bin/env python3
"""Import evaluation results from an existing W&B project into our project.

Reads runs from a source project (including full history for multi-checkpoint runs),
saves results locally in our format, and re-logs them to our W&B project.

Usage:
    python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT
    python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT --dry-run
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import RESULTS_DIR, WANDB_ENTITY, WANDB_PROJECT


def parse_args():
    parser = argparse.ArgumentParser(description="Import W&B runs into our project")
    parser.add_argument("--source", required=True, help="Source W&B project (entity/project)")
    parser.add_argument("--tag", required=True, help="Tag to add to imported runs")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be imported without doing it")
    parser.add_argument("--no-wandb", action="store_true", help="Save locally only, don't push to W&B")
    return parser.parse_args()


def _split_metrics(row: dict) -> tuple[dict, dict]:
    """Split a row into task metrics (keys with /) and metadata (rest)."""
    task_metrics = {}
    metadata = {}
    for k, v in row.items():
        if k.startswith("_") or (isinstance(v, float) and math.isnan(v)):
            continue
        if "/" in k:
            task_metrics[k] = v
        else:
            metadata[k] = v
    return task_metrics, metadata


def import_runs(source_project: str, tag: str, dry_run: bool = False, log_wandb: bool = True):
    import wandb

    api = wandb.Api()
    runs = api.runs(source_project)
    print(f"Found {len(runs)} runs in {source_project}\n")

    for run in runs:
        model_name = run.name

        # Get full history (each row = one checkpoint evaluation)
        history = run.history(samples=10000, pandas=True)
        n_rows = len(history)

        tasks_sample = run.summary._json_dict
        n_tasks = len(set(k.split("/")[0] for k in tasks_sample if "/" in k and not k.startswith("_")))

        print(f"{'[DRY RUN] ' if dry_run else ''}Importing: {model_name} ({n_tasks} tasks, {n_rows} checkpoints)")

        if dry_run:
            continue

        # Save each checkpoint row locally
        output_dir = RESULTS_DIR / model_name / "imported"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        all_rows = []
        for _, row in history.iterrows():
            row_dict = row.to_dict()
            task_metrics, metadata = _split_metrics(row_dict)
            if not task_metrics:
                continue

            results_by_task = {}
            for key, value in task_metrics.items():
                task, metric = key.split("/", 1)
                if task not in results_by_task:
                    results_by_task[task] = {"alias": task}
                results_by_task[task][metric] = value

            all_rows.append({
                "results": results_by_task,
                "metadata": metadata,
                "task_metrics": task_metrics,
            })

        # Save all checkpoints in one file
        results_to_save = {
            "source_project": source_project,
            "source_run_name": model_name,
            "imported_at": datetime.now().isoformat(),
            "import_tag": tag,
            "checkpoints": [
                {"results": r["results"], **r["metadata"]}
                for r in all_rows
            ],
        }
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"  Saved {len(all_rows)} checkpoints to {results_file}")

        # Push to our W&B: one run per source run, log each checkpoint as a step
        if log_wandb:
            wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=model_name,
                group=model_name,
                job_type="import",
                tags=[tag, model_name],
                config={
                    "model_name": model_name,
                    "source_project": source_project,
                },
            )

            for idx, row_data in enumerate(all_rows):
                log_dict = {**row_data["task_metrics"], **row_data["metadata"]}
                log_dict["checkpoint_index"] = idx
                wandb.log(log_dict)

            wandb.finish()
            print(f"  Logged {len(all_rows)} steps to W&B: {WANDB_ENTITY}/{WANDB_PROJECT}/{model_name}")


def main():
    args = parse_args()
    import_runs(
        source_project=args.source,
        tag=args.tag,
        dry_run=args.dry_run,
        log_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
