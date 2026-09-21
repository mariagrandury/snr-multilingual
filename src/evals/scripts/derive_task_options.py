#!/usr/bin/env python3
"""Record each task's answer-option count and item count in configs/tasks.json.

`n_options` is what turns a raw accuracy into "above chance": a 0.31 means
something different on 3-way xnli (chance 0.333, i.e. BELOW it) than on 4-way
arc (chance 0.25). Nothing else on disk carries it — lm_eval's results files
record the score but not the format — so without this every "is it learning?"
check has to hardcode a table and drift from the task list.

It is DERIVED, not asserted: each samples_<task>_*.jsonl record holds one
`arguments` entry per candidate continuation, so the option count is read off
a real evaluated document. Multiple-choice tasks only; generative ones
(exact_match) have no fixed option count and are left without the field.

`n_items` is the number of scored examples, read from the harness results
files (`n-samples.effective`, the count after the task's own filtering): with
it a reported accuracy is a count of correct answers again, which is what a
binomial confidence bound on "above chance" needs. One results file per task
is enough (every checkpoint scores the same test set), so the walk over
eval_logs stops as soon as every listed task has a count.

Idempotent — re-run after adding benchmarks; existing values are overwritten
only when the samples disagree, and that disagreement is printed.

    python3.11 scripts/derive_task_options.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS_JSON = REPO / "configs" / "tasks.json"
EVAL_LOGS = Path("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/"
                 "Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr")
SAMPLE_RE = re.compile(r"samples_(.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")


def observed_options() -> dict[str, Counter]:
    """task -> Counter of option counts seen across sample files.

    A Counter rather than a single value so a task whose format is not
    constant shows up instead of being silently reduced to whichever file was
    read last. One SAMPLE per file is enough (a file's records share one
    format), but every file per task must be read or the Counter can never
    hold a second value and the non-constant case it exists for is invisible.
    """
    seen: dict[str, Counter] = {}
    for f in EVAL_LOGS.glob("*/harness/eval_*/samples_*.jsonl"):
        m = SAMPLE_RE.search(f.name)
        if not m:
            continue
        task = m.group(1)
        try:
            with open(f) as fh:
                rec = json.loads(fh.readline())
        except (OSError, json.JSONDecodeError):
            continue
        args = rec.get("arguments")
        if isinstance(args, (list, dict)) and len(args) > 1:
            seen.setdefault(task, Counter())[len(args)] += 1
    return seen


def observed_items(listed: dict) -> dict[str, int]:
    """task -> number of scored items, from the newest results file of each
    task (per_task/<task>/<model>/results_*.json, or the older one-file-per-job
    layout). Stops once every listed task is covered."""
    seen: dict[str, int] = {}
    want = set(listed)
    runs = sorted(EVAL_LOGS.glob("*/harness/eval_*"), key=lambda d: d.stat().st_mtime, reverse=True)
    for run in runs:
        for f in list(run.glob("per_task/*/*/results_*.json")) + list(run.glob("results_*.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            # A group task (global_mmlu_full_ar) reports n-samples for its
            # subject facets only; its own count is their sum within the file.
            parts: dict[str, int] = {}
            for task, n in r.get("n-samples", {}).items():
                if not n.get("effective"):
                    continue
                if task in want:
                    seen.setdefault(task, int(n["effective"]))
                else:
                    parent = max((k for k in want if task.startswith(k + "_")), key=len, default=None)
                    if parent is not None and parent not in r["n-samples"]:
                        parts[parent] = parts.get(parent, 0) + int(n["effective"])
            for parent, n in parts.items():
                seen.setdefault(parent, n)
        if want <= set(seen):
            break
    return seen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = json.loads(TASKS_JSON.read_text())
    seen = observed_options()
    added = changed = 0
    listed = tasks["tasks"]
    for task, counts in sorted(seen.items()):
        n = counts.most_common(1)[0][0]
        entry = listed.get(task)
        if entry is None:
            # lm_eval reports MMLU-style subtopics (global_mmlu_full_en_anatomy)
            # that tasks.json only lists by their parent (global_mmlu_full_en).
            # The subtopics share the parent's answer format, so attribute the
            # observation to the longest listed prefix instead of dropping it —
            # without this the big multi-subject benchmarks, which are most of
            # the rows, get no chance level at all.
            parent = max((k for k in listed if task.startswith(k + "_")),
                         key=len, default=None)
            if parent is None:
                continue
            entry = listed[parent]
        old = entry.get("n_options")
        if old == n:
            continue
        if old is None:
            added += 1
        else:
            changed += 1
            print(f"  {task}: n_options {old} -> {n} (samples say {dict(counts)})")
        entry["n_options"] = n

    print(f"{len(seen)} tasks have samples; +{added} new, ~{changed} changed")
    items = observed_items(listed)
    added = changed = 0
    for task, n in sorted(items.items()):
        old = listed[task].get("n_items")
        if old == n:
            continue
        added += old is None
        changed += old is not None
        if old is not None:
            print(f"  {task}: n_items {old} -> {n}")
        listed[task]["n_items"] = n
    print(f"{len(items)} tasks have results; n_items +{added} new, ~{changed} changed; "
          f"{len([t for t in listed if 'n_items' not in listed[t]])} listed tasks without")
    by_n = Counter(e["n_options"] for e in tasks["tasks"].values()
                   if "n_options" in e)
    print("distribution:", dict(sorted(by_n.items())))
    if args.dry_run:
        print("(dry-run: tasks.json not written)")
        return
    TASKS_JSON.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
