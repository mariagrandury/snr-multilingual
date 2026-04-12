#!/usr/bin/env python3
"""Local evaluation runner (CPU or single GPU). Thin wrapper around src/."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoints import resolve_checkpoints
from src.config import load_models, load_tasks
from src.evaluate import run_all


def parse_args():
    parser = argparse.ArgumentParser(description="Run lm-eval locally")
    parser.add_argument("--tasks", required=True, help="Key from configs/tasks.json")
    parser.add_argument("--models", required=True, help="Key from configs/models.json")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--last", type=int, help="Last N checkpoints (alphabetical)")
    ckpt_group.add_argument("--total", type=int, help="T evenly spaced checkpoints")
    ckpt_group.add_argument("--names", action="store_true", help="Use checkpoint names from models.json")

    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for testing)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument("--batch-size", default="auto", help="Batch size (default: auto)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks)
    models = load_models(args.models)

    checkpoints_per_model = {}
    for model_entry in models:
        model_id = model_entry["id"]
        if args.names:
            if "checkpoints" not in model_entry:
                raise ValueError(f"--names requires 'checkpoints' in models.json for {model_id}")
            checkpoints_per_model[model_id] = resolve_checkpoints(model_id, names=model_entry["checkpoints"])
        elif args.last is not None:
            checkpoints_per_model[model_id] = resolve_checkpoints(model_id, last=args.last)
        else:
            checkpoints_per_model[model_id] = resolve_checkpoints(model_id, total=args.total)

    print(f"Tasks: {tasks}")
    for model_id, ckpts in checkpoints_per_model.items():
        print(f"Model: {model_id} -> checkpoints: {ckpts}")

    run_all(
        models,
        checkpoints_per_model,
        tasks,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        log_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
