#!/usr/bin/env python3
"""Print which tasks still need to run for a given checkpoint NAME.

Used by hf_base_runner.sh (skip whole-ckpt submission) and _run_per_task.sh
(skip per-task within a running job) to make eval launches idempotent.

Reads (NAME, ENTITY, PROJECT) and a list of expected tasks; outputs the
*remaining* tasks (one per line) by scanning, in
$LOGS_ROOT/$ENTITY/$PROJECT/$NAME/harness/, every prior eval_*/ run for:
  * `per_task/<task>/` directories that contain at least one file (saved
    by killed runs that didn't reach the merge step), AND
  * `results_*.json` files whose top-level `.results` dict has the task as
    a key (saved by runs that finished cleanly and merged).

Exit code is 0 if at least one task is remaining, 1 if everything is
already done — handy for `if ! _eval_status ...; then continue; fi`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_LOGS_ROOT = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
DEFAULT_ENTITY = "mariagrandury-epflnlp"
DEFAULT_PROJECT = "snr-experiments"


def completed_tasks(name, entity, project, logs_root):
    base = Path(logs_root) / entity / project / name / "harness"
    completed = set()
    if not base.is_dir():
        return completed

    # per_task/<task>/ subdirs from killed runs (only count if non-empty)
    for d in base.glob("eval_*/per_task/*"):
        if d.is_dir():
            try:
                if any(d.iterdir()):
                    completed.add(d.name)
            except OSError:
                pass

    # results_*.json keys from clean merged runs
    for f in base.glob("eval_*/results_*.json"):
        try:
            data = json.loads(f.read_text())
            completed.update((data.get("results") or {}).keys())
        except Exception:
            pass

    return completed


def parse_tasks_input(tasks_arg):
    """tasks_arg is either a path to a file (one task per line, # comments) or a
    comma-separated string. Comma-separated lists from the inner runner can
    exceed Linux's NAME_MAX (255 bytes), so guard against the OSError that
    Path.is_file() raises in that case (errno 36 / ENAMETOOLONG)."""
    is_file = False
    if "," not in tasks_arg and len(tasks_arg) < 4000:
        try:
            is_file = Path(tasks_arg).is_file()
        except OSError:
            is_file = False
    if is_file:
        return [
            line.strip()
            for line in Path(tasks_arg).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [t.strip() for t in tasks_arg.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Harness NAME (model-ckpt)")
    ap.add_argument("--tasks", required=True, help="File path or comma-list of task names")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    ap.add_argument("--logs-root", default=os.environ.get("LOGS_ROOT", DEFAULT_LOGS_ROOT))
    args = ap.parse_args()

    expected = parse_tasks_input(args.tasks)
    done = completed_tasks(args.name, args.entity, args.project, args.logs_root)
    remaining = [t for t in expected if t not in done]

    for t in remaining:
        print(t)
    sys.exit(0 if remaining else 1)


if __name__ == "__main__":
    main()
