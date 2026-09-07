#!/usr/bin/env python3.11
"""What eval jobs actually cost, and how much of a killed one survives.

Two questions the sweep keeps asking, answered from disk + sacct:

  1. minutes per task — the number `auto_evals_cscs.MIN_PER_TASK` is fitted
     from and every walltime request depends on. Split by pipeline, because
     the 2026-09-04 worker pool changed it: the old runner made one lm_eval
     call for every task in one process on one GPU; the new one runs
     EVAL_WORKERS workers, each with its own model copy on its own GPU.
  2. what a killed job saved — the whole point of the change. The batched
     runner wrote everything in one burst at the end, so a TIMEOUT kept
     nothing; the worker pool publishes each task as it finishes.

A job is `worker` if its eval dir has the job.json evaluate.sbatch writes,
`batched` otherwise — the file did not exist before the worker pool, so the
classification is exact rather than a guess from timestamps.

Wall-clock comes from sacct (one bulk call), not from the harness's own
`total_evaluation_time_seconds`, so both generations are measured the same
way and the container start and pip install are included in both.

    python3.11 scripts/eval_timing.py                 # the comparison
    python3.11 scripts/eval_timing.py --detail        # one row per job
    python3.11 scripts/eval_timing.py --name lm-600M  # NAME substring filter
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

DEFAULT_LOGS_ROOT = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
DEFAULT_ENTITY = "mariagrandury-epflnlp"
DEFAULT_PROJECT = "msnr"
SIZE_RE = re.compile(r"-(\d+\.?\d*[MB])-")
KILLED = {"TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}


def sacct(job_ids: list[str]) -> dict[str, tuple[str, int]]:
    """job id -> (state, elapsed seconds). Empty when sacct is unavailable
    (a laptop) or the jobs have aged out of the accounting DB."""
    if not job_ids:
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-o", "JobID,State,ElapsedRaw",
             "--jobs", ",".join(job_ids)],
            capture_output=True, text=True, timeout=120).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    jobs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[2].isdigit():
            jobs[parts[0]] = (parts[1].split()[0], int(parts[2]))
    return jobs


def tasks_in(eval_dir: Path, meta: dict) -> int:
    """Tasks this run finished, as QUEUED tasks — one per name the job was
    given, which is what the walltime is priced in.

    job.json's count first, then the published per_task/ dirs. The results
    file is the last resort (a batched run has nothing else) and it OVERCOUNTS:
    lm_eval lists a group's subtasks individually, so one queued
    `global_mmlu_full_es` becomes dozens of rows. That inflates the old
    pipeline's task count and so understates its minutes per task, which
    makes the batched-vs-worker comparison conservative rather than flattering.
    """
    if isinstance(meta.get("tasks_done"), int):
        return meta["tasks_done"]
    done = {d.name for d in eval_dir.glob("per_task/*") if d.is_dir()}
    if done:
        return len(done)
    for f in eval_dir.glob("results_*.json"):
        try:
            done |= set((json.loads(f.read_text()).get("results") or {}))
        except (OSError, json.JSONDecodeError):
            pass
    return len(done)


def scan(project_dir: Path, name_filter: str | None) -> list[dict]:
    runs = []
    for name_dir in sorted(project_dir.iterdir()) if project_dir.is_dir() else []:
        if name_filter and name_filter not in name_dir.name:
            continue
        for eval_dir in sorted(name_dir.glob("harness/eval_*")):
            if not eval_dir.is_dir():
                continue
            job = (eval_dir / "job.json")
            meta = {}
            if job.is_file():
                try:
                    meta = json.loads(job.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            size = SIZE_RE.search(name_dir.name)
            runs.append({
                "name": name_dir.name,
                "size": size.group(1) if size else "?",
                # eval_<date>_<time>_<jobid>; job.json carries it too, but the
                # dir name is the only source for the pre-worker-pool runs.
                "job_id": eval_dir.name.rsplit("_", 1)[-1],
                "pipeline": "worker" if job.is_file() else "batched",
                "workers": meta.get("workers"),
                "tasks": tasks_in(eval_dir, meta),
                "dir": eval_dir,
            })
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--name", help="only NAMEs containing this substring")
    p.add_argument("--detail", action="store_true", help="one row per job")
    args = p.parse_args()

    runs = scan(Path(args.logs_root) / args.entity / args.project, args.name)
    if not runs:
        raise SystemExit(f"no eval dirs under {args.logs_root}/{args.entity}/{args.project}")
    jobs = sacct([r["job_id"] for r in runs])
    for r in runs:
        r["state"], r["elapsed"] = jobs.get(r["job_id"], ("?", 0))
        r["min_per_task"] = (r["elapsed"] / 60 / r["tasks"]) if r["tasks"] else None
    if not jobs:
        print("(sacct unavailable or the jobs aged out — timings will be blank)\n")

    if args.detail:
        print(f"{'pipeline':<9} {'size':<5} {'job':<9} {'state':<10} "
              f"{'tasks':>5} {'elapsed':>8} {'min/task':>9}  name")
        for r in sorted(runs, key=lambda r: (r["pipeline"], r["size"], r["job_id"])):
            mpt = f"{r['min_per_task']:.2f}" if r["min_per_task"] else "-"
            print(f"{r['pipeline']:<9} {r['size']:<5} {r['job_id']:<9} {r['state']:<10} "
                  f"{r['tasks']:>5} {r['elapsed'] / 60:>7.1f}m {mpt:>9}  {r['name']}")
        print()

    print("== minutes per task (wall-clock / tasks finished, sacct elapsed) ==")
    print(f"{'pipeline':<9} {'size':<5} {'jobs':>5} {'tasks':>7} "
          f"{'median':>8} {'p10':>7} {'p90':>7}")
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        if r["min_per_task"] and r["state"] == "COMPLETED":
            cells[(r["pipeline"], r["size"])].append(r)
    for (pipeline, size), rs in sorted(cells.items()):
        vals = sorted(r["min_per_task"] for r in rs)
        q = st.quantiles(vals, n=10) if len(vals) > 1 else None
        print(f"{pipeline:<9} {size:<5} {len(rs):>5} {sum(r['tasks'] for r in rs):>7} "
              f"{st.median(vals):>8.2f} "
              f"{q[0] if q else vals[0]:>7.2f} {q[-1] if q else vals[0]:>7.2f}")
    if len(cells) and len({p for p, _ in cells}) == 2:
        for size in sorted({s for _, s in cells}):
            b, w = cells.get(("batched", size)), cells.get(("worker", size))
            if b and w:
                mb = st.median([r["min_per_task"] for r in b])
                mw = st.median([r["min_per_task"] for r in w])
                print(f"  {size}: {mb:.2f} -> {mw:.2f} min/task  ({mb / mw:.1f}x)")
    print("Only COMPLETED jobs; a killed one's elapsed is its walltime, not its cost.")

    print("\n== what a killed job kept ==")
    print(f"{'pipeline':<9} {'jobs':>5} {'kept 0':>7} {'kept >0':>8} {'tasks saved':>12}")
    for pipeline in ("batched", "worker"):
        rs = [r for r in runs if r["pipeline"] == pipeline and r["state"] in KILLED]
        if not rs:
            continue
        saved = [r["tasks"] for r in rs]
        print(f"{pipeline:<9} {len(rs):>5} {sum(1 for s in saved if not s):>7} "
              f"{sum(1 for s in saved if s):>8} {sum(saved):>12}")
    print("A batched job wrote everything at the end, so every kill kept 0 "
          "(evals CLAUDE.md bug 13).")


if __name__ == "__main__":
    main()
