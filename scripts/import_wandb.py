#!/usr/bin/env python3
"""Import evaluation results from an existing W&B project into our project.

Reads runs from a source project, saves results locally in our format,
and re-logs them to our W&B project with configurable tags.

Usage:
    python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT
    python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT --dry-run
"""

import argparse
import json
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


def import_runs(source_project: str, tag: str, dry_run: bool = False, log_wandb: bool = True):
    import wandb

    api = wandb.Api()
    runs = api.runs(source_project)
    print(f"Found {len(runs)} runs in {source_project}\n")

    for run in runs:
        summary = run.summary._json_dict
        model_name = run.name

        # Separate task metrics from metadata
        task_metrics = {}
        metadata = {}
        for k, v in summary.items():
            if k.startswith("_"):
                continue
            if "/" in k:
                task_metrics[k] = v
            else:
                metadata[k] = v

        tasks = sorted(set(k.split("/")[0] for k in task_metrics))
        print(f"{'[DRY RUN] ' if dry_run else ''}Importing: {model_name} ({len(tasks)} tasks)")

        if dry_run:
            continue

        # Build results dict in our format
        results_by_task = {}
        for key, value in task_metrics.items():
            task, metric = key.split("/", 1)
            if task not in results_by_task:
                results_by_task[task] = {"alias": task}
            results_by_task[task][metric] = value

        results_to_save = {
            "results": results_by_task,
            "source_project": source_project,
            "source_run_name": model_name,
            "imported_at": datetime.now().isoformat(),
            "import_tag": tag,
            **metadata,
        }

        # Save locally
        output_dir = RESULTS_DIR / model_name / "imported"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"  Saved to {results_file}")

        # Push to our W&B
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
                    "checkpoint_index": 0,
                    **metadata,
                },
            )
            # Include metadata (ConsumedTokens, OptStep) so they can be used as x-axis
            wandb.log({**task_metrics, **metadata})
            wandb.finish()
            print(f"  Logged to W&B: {WANDB_ENTITY}/{WANDB_PROJECT}/{model_name}")


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
