#!/usr/bin/env python3
"""
Merge all W&B runs for each size-mix-seed combination into a single run.

Finds runs matching the naming convention:
    apertus-{SIZE}-fwEdu{EDU}-fw2{FW2}-seed{SEED}-{JOB_ID}

Then calls merge_wandb_runs.py to create:
    apertus-{SIZE}-fwEdu{EDU}-fw2{FW2}-seed{SEED}-merged

Any argument not specified is iterated over all values.

Usage:
    # Merge all combinations
    python merge_wandb_experiment.py

    # Merge all sizes and mixtures for seed 28
    python merge_wandb_experiment.py --seed 28

    # Merge all seeds for 175M with edu=90
    python merge_wandb_experiment.py --size 175M --edu 90

    # Single combination
    python merge_wandb_experiment.py --size 175M --edu 90 --seed 28

    # Dry run: list matching runs without merging
    python merge_wandb_experiment.py --seed 28 --dry-run
"""

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import wandb

ENTITY = "mariagrandury-epflnlp"
PROJECT = "data-mix-small"
MERGE_SCRIPT = Path(__file__).parent.parent / "scripts/merge_wandb_runs.py"

SIZES = ["175M", "350M", "600M", "1B"]
DATA_RATIOS = [("30", "70"), ("60", "40"), ("90", "10")]
SEEDS = ["28", "64", "1797"]


def merge_combination(api, all_runs, size, edu, fw2, seed, dry_run):
    prefix = f"apertus-{size}-fwEdu{edu}-fw2{fw2}-seed{seed}-"
    merged_name = f"apertus-{size}-fwEdu{edu}-fw2{fw2}-seed{seed}-merged"

    matching = [
        r
        for r in all_runs
        if r.name.startswith(prefix) and not r.name.endswith("-merged")
    ]
    matching.sort(key=lambda r: r.summary.get("_step", 0) - (r.lastHistoryStep or 0))

    if not matching:
        print(f"  No runs found for {prefix}*, skipping.\n")
        return

    if len(matching) == 1:
        print(f"  Only 1 run for {prefix}*, nothing to merge.\n")
        return

    print(f"  Found {len(matching)} runs:")
    for r in matching:
        step_range = f"steps 0-{r.lastHistoryStep}" if r.lastHistoryStep else "no steps"
        print(f"    {r.name} (id={r.id}, {step_range}, state={r.state})")

    if dry_run:
        print(f"  Would merge into: {merged_name}\n")
        return

    run_ids = [r.id for r in matching]
    cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--entity", ENTITY,
        "--source_project", PROJECT,
        "--runs", *run_ids,
        "--name", merged_name,
    ]
    print(f"  Merging into: {merged_name}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: merge failed for {merged_name}\n", file=sys.stderr)
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", help="Model size, e.g. 175M, 350M, 600M, 1B")
    parser.add_argument("--edu", help="FineWeb-Edu ratio, e.g. 30, 60, 90")
    parser.add_argument("--seed", help="Seed, e.g. 28, 64, 1797")
    parser.add_argument("--dry-run", action="store_true", help="List matching runs without merging")
    args = parser.parse_args()

    sizes = [args.size] if args.size else SIZES
    ratios = [(args.edu, str(100 - int(args.edu)))] if args.edu else DATA_RATIOS
    seeds = [args.seed] if args.seed else SEEDS

    combos = list(product(sizes, ratios, seeds))
    print(f"Merging {len(combos)} combination(s){'  (dry run)' if args.dry_run else ''}\n")

    api = wandb.Api()
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    for size, (edu, fw2), seed in combos:
        print(f"=== {size} | fwEdu{edu}-fw2{fw2} | seed{seed} ===")
        merge_combination(api, all_runs, size, edu, fw2, seed, args.dry_run)


if __name__ == "__main__":
    main()
