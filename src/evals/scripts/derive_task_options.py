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
only when the samples disagree, and that disagreement is printed. It is also
expensive: the samples live one file per (task, run), 4.0M of them on Lustre
as of 2026-09-23, and a full pass opens every one, so it costs about an hour.
`--only-missing` opens only the files of tasks that have no `n_options` yet
and looks for items only where `n_items` is still blank, which is what adding
a benchmark needs; run the full pass when a task's format may have changed
under it (that is how cultural_bench_hard was caught declaring four options
for a True/False task).

    python3.11 scripts/derive_task_options.py [--only-missing] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from utils.configs import read_tasks_json, write_tasks_json

REPO = Path(__file__).resolve().parents[3]
TASKS_JSON = REPO / "configs" / "tasks.json"
EVAL_LOGS = Path("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/"
                 "Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr")
SAMPLE_RE = re.compile(r"samples_(.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")


def observed_options(listed: dict, only_missing: bool
                     ) -> tuple[dict[str, Counter], set[str]]:
    """(task -> Counter of option counts seen across sample files, tasks that
    have samples at all).

    A Counter rather than a single value so a task whose format is not
    constant shows up instead of being silently reduced to whichever file was
    read last. One SAMPLE per file is enough (a file's records share one
    format), but every file per task must be read or the Counter can never
    hold a second value and the non-constant case it exists for is invisible.

    lm_eval reports MMLU-style subtopics (global_mmlu_full_en_anatomy) that
    tasks.json only lists by their parent (global_mmlu_full_en); the subtopics
    share the parent's answer format, so an observation is attributed to the
    longest listed prefix rather than dropped — without this the big
    multi-subject benchmarks, which are most of the rows, get no chance level
    at all. The resolution is memoised per sample-file task name: there are
    thousands of those and millions of files.

    The second return value costs nothing (it is read off the file names) and
    is what lets the item walk stop: a task with no samples anywhere has no
    results either, so waiting for one keeps the walk going to the last run.
    """
    seen: dict[str, Counter] = {}
    have: set[str] = set()
    resolved: dict[str, str | None] = {}
    for f in EVAL_LOGS.glob("*/harness/eval_*/samples_*.jsonl"):
        m = SAMPLE_RE.search(f.name)
        if not m:
            continue
        task = m.group(1)
        if task not in resolved:
            resolved[task] = task if task in listed else max(
                (k for k in listed if task.startswith(k + "_")), key=len, default=None)
        entry = resolved[task]
        if entry is None:
            continue
        have.add(entry)
        if only_missing and "n_options" in listed[entry]:
            continue
        try:
            with open(f) as fh:
                rec = json.loads(fh.readline())
        except (OSError, json.JSONDecodeError):
            continue
        args = rec.get("arguments")
        if isinstance(args, (list, dict)) and len(args) > 1:
            seen.setdefault(entry, Counter())[len(args)] += 1
    return seen, have


def observed_items(want: set[str]) -> dict[str, int]:
    """task -> number of scored items, from the newest results file of each
    task (per_task/<task>/<model>/results_*.json, or the older one-file-per-job
    layout). Stops once every wanted task is covered."""
    seen: dict[str, int] = {}
    want = set(want)
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


# --examples: the cells whose evals hold every pretraining task the site
# shows (L50 = the 50 trained languages, L1 = English), newest saves first.
EXAMPLE_RUNS = ("lm-1.7B-L50-*/harness/eval_*", "lm-1.7B-L1-*/harness/eval_*")
PROMPT_CHARS, OPTION_CHARS = 600, 240


def _clip(text: str, n: int, tail: bool = False) -> str:
    """At most n characters; a prompt keeps its END (the question and the
    answer cue come last, after any passage or few-shot block)."""
    text = text.strip("\n")
    if len(text) <= n:
        return text
    return "…" + text[-n:] if tail else text[:n] + "…"


def _example(rec: dict) -> dict | None:
    """One scored document as the model saw it: the prompt, the candidate
    continuations, and the index of the correct one (None when the target is
    not one of them, e.g. a generative task)."""
    args = [a for _, a in sorted((rec.get("arguments") or {}).items())]
    if not args or not isinstance(args[0].get("arg_0"), str):
        return None
    ctx = [a["arg_0"] for a in args]
    conts = [a.get("arg_1") if isinstance(a.get("arg_1"), str) else None for a in args]
    if len(set(ctx)) == 1:                 # one prompt, one continuation per option
        prompt, options = ctx[0], [c for c in conts if c is not None]
    else:                                  # the option sits in the prompt (partial scoring)
        prompt, options = None, [c + (k or "") for c, k in zip(ctx, conts)]
    target, gold = str(rec.get("target", "")), None
    if target.isdigit() and int(target) < len(options):
        gold = int(target)
    else:
        hits = [i for i, o in enumerate(options) if o.strip() == target.strip()]
        gold = hits[0] if len(hits) == 1 else None
    return {"prompt": prompt and _clip(prompt, PROMPT_CHARS, tail=True),
            "options": [_clip(o, OPTION_CHARS) for o in options], "gold": gold}


def _source(run: Path, sample_task: str) -> str | None:
    """`dataset_path (dataset_name)` from the results file that scored the task."""
    files = list(run.glob(f"per_task/{sample_task}/*/results_*.json")) + list(run.glob("results_*.json"))
    for f in files:
        try:
            cfg = json.loads(f.read_text()).get("configs", {}).get(sample_task)
        except (OSError, json.JSONDecodeError):
            continue
        if cfg and cfg.get("dataset_path") and cfg["dataset_path"] not in ("json", "csv", "parquet", "arrow"):
            name = cfg.get("dataset_name")
            return cfg["dataset_path"] + (f" ({name})" if name else "")
    return None


def observed_examples(listed: dict, want: set[str]) -> dict[str, dict]:
    """task -> {"example": first scored document, "source": its dataset}, read
    from ONE samples file per task (first record; a group task takes its first
    subtopic in name order). Walks only EXAMPLE_RUNS and stops once every
    wanted task is covered."""
    out: dict[str, dict] = {}
    resolved: dict[str, str | None] = {}
    runs = sorted((r for pat in EXAMPLE_RUNS for r in EVAL_LOGS.glob(pat)), key=lambda d: d.name, reverse=True)
    for run in runs:
        try:
            names = sorted(e.name for e in os.scandir(run) if e.name.startswith("samples_"))
        except OSError:
            continue
        for fname in names:
            m = SAMPLE_RE.search(fname)
            if not m:
                continue
            task = m.group(1)
            if task not in resolved:
                resolved[task] = task if task in listed else max(
                    (k for k in listed if task.startswith(k + "_")), key=len, default=None)
            entry = resolved[task]
            if entry is None or entry not in want or entry in out:
                continue
            try:
                with open(run / fname) as fh:
                    ex = _example(json.loads(fh.readline()))
            except (OSError, json.JSONDecodeError):
                continue
            if ex:
                out[entry] = {"example": ex, "source": _source(run, task)}
        if want <= set(out):
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only-missing", action="store_true",
                   help="derive only what tasks.json does not already carry "
                        "(minutes instead of an hour; see the module docstring)")
    p.add_argument("--examples", action="store_true",
                   help="instead: record one scored example and the dataset source of every "
                        "pretraining task (for the website's benchmark cards); with "
                        "--only-missing, only tasks that have no example yet")
    args = p.parse_args()

    # an hour of reading samples: another generator may write tasks.json meanwhile
    tasks, before = read_tasks_json(TASKS_JSON)
    listed = tasks["tasks"]
    if args.examples:
        want = {t for t, e in listed.items() if "pretraining" in e.get("stages", [])
                and not (args.only_missing and "example" in e)}
        found = observed_examples(listed, want)
        for task, f in found.items():
            listed[task]["example"] = f["example"]
            if f["source"]:
                listed[task]["source"] = f["source"]
        print(f"examples: {len(found)} of {len(want)} wanted tasks; "
              f"{len(want) - len(found)} without samples in {', '.join(EXAMPLE_RUNS)}")
        if not args.dry_run:
            write_tasks_json(tasks, before, TASKS_JSON)
            print(f"wrote {TASKS_JSON}")
        return
    seen, have_samples = observed_options(listed, args.only_missing)
    added = changed = 0
    for task, counts in sorted(seen.items()):
        n = counts.most_common(1)[0][0]
        entry = listed[task]
        old = entry.get("n_options")
        if old == n:
            continue
        if old is None:
            added += 1
        else:
            changed += 1
            print(f"  {task}: n_options {old} -> {n} (samples say {dict(counts)})")
        entry["n_options"] = n

    print(f"{len(seen)} tasks read from samples; +{added} new, ~{changed} changed")
    want = {t for t in have_samples
            if not args.only_missing or "n_items" not in listed[t]}
    items = observed_items(want)
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
    write_tasks_json(tasks, before, TASKS_JSON)
    print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
