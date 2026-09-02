#!/usr/bin/env python3
"""Record each task's answer-option count in configs/tasks.json.

`n_options` is what turns a raw accuracy into "above chance": a 0.31 means
something different on 3-way xnli (chance 0.333, i.e. BELOW it) than on 4-way
arc (chance 0.25). Nothing else on disk carries it — lm_eval's results files
record the score but not the format — so without this every "is it learning?"
check has to hardcode a table and drift from the task list.

It is DERIVED, not asserted: each samples_<task>_*.jsonl record holds one
`arguments` entry per candidate continuation, so the option count is read off
a real evaluated document. Multiple-choice tasks only; generative ones
(exact_match) have no fixed option count and are left without the field.

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
