#!/usr/bin/env python3.11
"""What the predictivity sweep has actually cost, by task, size and user.

plan/compute-budget.md estimates what the sweep SHOULD cost; this measures what
it DID and writes it to plan/compute-costs.md. Every allocation comes from
sacct (node-hours = elapsed x nodes, 4 GPUs per node) and is attributed from
what it left on disk, whatever state Slurm ended it in — a 1B job that hit
NODE_FAIL after eight hours had saved seven checkpoints first:

  pretrain  grid cells only, pro-rated by the iterations that ended in a
            checkpoint save — a job killed 40 min after its last save lost
            those 40 min. Read from the training log, never the Slurm state:
            a crashed Megatron step still reports COMPLETED (CLAUDE.md #3).
  eval      pro-rated by the tasks the job published that are in its cell's
            CURRENT `auto` list, over the tasks it attempted (job.json). A
            task published by several jobs counts for the newest one only.
  bpb       pro-rated by checkpoints written over checkpoints attempted, from
            the job's log.
  convert   grid cells or a models.json batch, unless the job failed.
  data      mixture builds and staging, unless the job failed — TIMEOUT is how
            a self-chaining build ends a link, not a failure.

What is not kept lands in one of the other BUCKETS. `other` is these users'
non-sweep jobs and stays out of every total. Azure runs are not in sacct and
are not included.

    python3.11 compute_cost.py                                # since 2026-08-01
    python3.11 compute_cost.py --since 2026-09-01 --users mariagrandury
    python3.11 compute_cost.py --out /tmp/costs.md            # not the plan doc
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from launch_trainings import (  # noqa: E402
    DATA_SCHEMES, LADDER, exp_name, predictivity_cells)
from ladder_report import EVAL_LOGS  # noqa: E402
from pretrain_progress import TRAIN_LOGS  # noqa: E402
from auto_evals_cscs import (  # noqa: E402
    ALL_LANGUAGES_RUNS, EVAL_JOB_LOGS, auto_benchmarks, eval_languages)
sys.path.insert(0, str(SCRIPT_DIR.parent))
from evals.scripts.eval_timing import SIZE_RE, task_names  # noqa: E402
from evals.scripts.utils.configs import tasks_for_benchmarks  # noqa: E402

GPUS_PER_NODE = 4
PLAN_DOC = SCRIPT_DIR.parent.parent / "plan" / "compute-costs.md"
SWEEP_TASKS = ("pretrain", "eval", "bpb", "convert", "data")
# Column order of every table, and the legend written under them.
BUCKETS = {
    "kept": "pretrain iterations up to the job's last checkpoint save, eval "
            "tasks in the cell's `auto` list that the job was the newest to "
            "publish, BPB checkpoints written, converts and data builds that "
            "did not fail",
    "failed": "what a job Slurm ended FAILED, NODE_FAIL, OUT_OF_MEMORY, "
              "BOOT_FAIL or DEADLINE did not keep",
    "not_in_grid": "`diag-` runs, the `apertus-*` training runs from before "
                   "the 2026-08-21 rename, and anything else no grid cell is "
                   "named after",
    "not_auto": "eval tasks published but no longer in the cell's `auto` list",
    "superseded": "eval tasks a later job published again",
    "wasted": "what a job that ended any other way (COMPLETED, TIMEOUT, "
              "CANCELLED) did not keep",
    "in_flight": "pending or running",
    "no_record": "the job's log or eval dir is missing or unreadable — a "
                 "collaborator's Slurm logs live in their own scratch and need "
                 "read access",
}
FAILED = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE"}
LIVE = {"RUNNING", "PENDING", "REQUEUED", "SUSPENDED"}
ITER_RE = re.compile(r"iteration\s+(\d+)/")
SAVED_RE = re.compile(r"successfully saved checkpoint from iteration\s+(\d+)")


def allocations(users: list[str], since: str) -> list[dict]:
    """Every allocation the users ran since `since`. -X gives one row per job,
    not per step, so nothing is charged twice."""
    out = subprocess.run(
        ["sacct", "-u", ",".join(users), "-S", since, "-X", "-n", "-P",
         "-o", "JobID,JobName%200,State,ElapsedRaw,NNodes,Start,User"],
        capture_output=True, text=True, check=True, timeout=900,
        env=dict(os.environ, SLURM_TIME_FORMAT="%s")).stdout
    jobs = []
    for line in out.splitlines():
        jid, name, state, elapsed, nodes, start, user = line.split("|")
        if elapsed.isdigit() and nodes.isdigit():
            jobs.append({"id": jid, "name": name, "state": state.split()[0],
                         "nh": int(elapsed) * int(nodes) / 3600, "user": user,
                         "start": int(start) if start.isdigit() else 0})
    return jobs


def kind_of(name: str) -> tuple[str, str | None]:
    """(task, cell) from the names the submitters give: job_name() for
    pretrain/eval, `bpb-<cell>` (launch_bpb.sh), `convert-snr-<cell>`
    (auto_evals_cscs.convert_job_name) or `convert-snr[-models]` for a
    models.json batch. Training jobs were named after the cell itself,
    `apertus-*`, until the 2026-08-21 rename."""
    if name.startswith("pretrain-"):
        return "pretrain", "lm-" + name.removeprefix("pretrain-")
    if name.startswith("apertus-"):
        return "pretrain", None
    if name.startswith("eval-"):
        return "eval", "lm-" + re.sub(r"-iter\d+$", "", name.removeprefix("eval-"))
    if name.startswith("bpb-"):
        return "bpb", name.removeprefix("bpb-")
    if name in ("convert-snr", "convert-snr-models"):
        return "convert", "models"
    if name.startswith("convert-snr-"):
        return "convert", name.removeprefix("convert-snr-")
    if name.startswith(("build-", "stage-", "submit_build_one")):
        return "data", None
    return "other", None


def readable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def pretrain_kept(log: Path) -> float:
    """Share of a job's iterations that ended in a checkpoint save."""
    first = last = saved = None
    with log.open(errors="ignore") as fh:
        for line in fh:
            if m := SAVED_RE.search(line):
                saved = max(saved or 0, int(m[1]))
            elif m := ITER_RE.search(line):
                last = int(m[1])
                first = last if first is None else first
    if first is None or saved is None or saved < first:
        return 0.0
    return min(1.0, (saved - first + 1) / (last - first + 1))


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |",
             "| :--- |" + " ---: |" * (len(header) - 1)]
            + ["| " + " | ".join(row) + " |" for row in rows])


def fmt(nh: float) -> str:
    return f"{nh:,.0f}"


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "—"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="2026-08-01",
                   help="sacct start date (the sweep's first builds ran 2026-08-14)")
    p.add_argument("--users", default="mariagrandury,aromanou",
                   help="comma-separated; everyone who submits sweep jobs")
    p.add_argument("--out", type=Path, default=PLAN_DOC,
                   help=f"markdown report (default {PLAN_DOC})")
    args = p.parse_args()
    users = args.users.split(",")

    grid = {exp_name(c["size"], c["L"], arch, c["seed"], c["scheme"]): {**c, "arch": arch}
            for c in predictivity_cells() for arch in DATA_SCHEMES[c["scheme"]]["arches"]}
    # evaluate.sbatch writes every user's results into ONE tree, but each
    # user's Slurm logs go under their own scratch (`%u` in --output).
    log_dirs = [Path(str(TRAIN_LOGS).replace("/mariagrandury/", f"/{u}/")) for u in users]
    benchmarks, auto_cache = auto_benchmarks(), {}

    def auto_tasks(cell: str) -> set[str]:
        # The languages the watcher evaluates this cell in: its trained ones,
        # or every language for ALL_LANGUAGES_RUNS, whose extra tasks are
        # deliberate work, not work outside the auto list.
        g = grid[cell]
        key = (g["L"], g["scheme"], (g["scheme"], g["arch"], g["seed"]) == ALL_LANGUAGES_RUNS)
        if key not in auto_cache:
            auto_cache[key] = set(tasks_for_benchmarks(benchmarks, eval_languages(*key)))
        return auto_cache[key]

    # evaluate.sbatch names each run's dir eval_<date>_<time>_<jobid>.
    eval_dirs = {d.name.rsplit("_", 1)[-1]: d for d in EVAL_LOGS.glob("*/harness/eval_*")}
    cost = defaultdict(lambda: defaultdict(float))                        # task -> bucket
    by_size = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # size -> task -> bucket
    by_user = defaultdict(lambda: defaultdict(float))                     # user -> bucket
    other = defaultdict(float)                                            # job name -> node-h
    n_jobs = defaultdict(int)
    newest = {}      # (cell-iter, task) -> (start, job id) of its newest publisher
    evals = []       # attributed after the pass, once every publisher is known

    def book(job, task, **share):
        n_jobs[task] += 1
        for bucket, frac in share.items():
            nh = job["nh"] * frac
            cost[task][bucket] += nh
            if task == "other":
                other[job["name"]] += nh
            else:
                by_size[job["size"]][task][bucket] += nh
                by_user[job["user"]][bucket] += nh

    for job in allocations(users, args.since):
        task, cell = kind_of(job["name"])
        size = grid[cell]["size"] if cell in grid else None
        # Off-grid work still has a rung, which the by-size table shows.
        m = SIZE_RE.search(job["name"])
        job["size"] = size or (m[1] if m and m[1] in LADDER else None)
        lost = "failed" if job["state"] in FAILED else "wasted"
        if task == "other":
            book(job, task, not_in_grid=1)
        elif job["state"] in LIVE:
            book(job, task, in_flight=1)
        elif task in ("data", "convert") and lost == "failed":
            book(job, task, failed=1)
        elif task == "data":
            book(job, task, kept=1)
        elif task == "convert":
            book(job, task, **({"kept": 1} if size or cell == "models"
                               else {"not_in_grid": 1}))
        elif size is None:
            book(job, task, not_in_grid=1)
        elif task == "pretrain":
            log = next((f for d in log_dirs
                        if readable(f := d / f"{job['name']}-{job['id']}.out")), None)
            if log is None:
                book(job, task, no_record=1)
            else:
                kept = pretrain_kept(log)
                book(job, task, kept=kept, **{lost: 1 - kept})
        elif task == "bpb":
            log = EVAL_JOB_LOGS / f"{job['name']}_{job['id']}.out"
            if not readable(log):
                book(job, task, no_record=1)
                continue
            text = log.read_text(errors="ignore")
            tried = len(re.findall(r"== scoring iter_", text))
            kept = len(re.findall(r"wrote \S+/bpb\.json", text)) / tried if tried else 0.0
            book(job, task, kept=kept, **{lost: 1 - kept})
        else:
            d = eval_dirs.get(job["id"])
            if d is None:
                # Unlike a training log, the results tree is shared and
                # readable: no eval dir means the job died before publishing
                # anything (all 14,091 of aromanou's FAILED evals to 09-10).
                book(job, task, **{lost: 1})
                continue
            try:
                published = task_names(d)
                meta = json.loads((d / "job.json").read_text()) if (d / "job.json").is_file() else {}
            except (OSError, json.JSONDecodeError):
                book(job, task, no_record=1)
                continue
            attempted = max(len(published), (meta.get("tasks_requested") or 0)
                            - (meta.get("tasks_skipped") or 0))
            if not attempted:
                book(job, task, **{lost: 1})
                continue
            auto = published & auto_tasks(cell)
            ckpt = d.parent.parent.name
            for t in auto:
                if job["start"] >= newest.get((ckpt, t), (-1, ""))[0]:
                    newest[(ckpt, t)] = (job["start"], job["id"])
            evals.append((job, ckpt, auto, len(published) - len(auto), attempted, lost))

    for job, ckpt, auto, n_not_auto, attempted, lost in evals:
        mine = sum(1 for t in auto if newest[(ckpt, t)][1] == job["id"])
        book(job, "eval", kept=mine / attempted,
             superseded=(len(auto) - mine) / attempted, not_auto=n_not_auto / attempted,
             **{lost: 1 - (len(auto) + n_not_auto) / attempted})

    losses = [b for b in BUCKETS if b != "kept"]
    label = {b: b.replace("_", " ") for b in BUCKETS}
    total = defaultdict(float)
    for task in SWEEP_TASKS:
        for b, nh in cost[task].items():
            total[b] += nh
    charged, kept = sum(total.values()), total["kept"]

    lines = [
        "# Compute costs",
        "",
        "<!-- Generated by src/pretrain/compute_cost.py. Do not edit: re-run it. -->",
        "",
        f"What the predictivity sweep has spent on CSCS as of {datetime.date.today()}: "
        f"every Slurm allocation {', '.join(users)} ran since {args.since}, "
        f"attributed from what it left on disk. What the sweep *should* cost is "
        f"[compute-budget.md](compute-budget.md); how each task is attributed is "
        f"the docstring of [compute_cost.py](../src/pretrain/compute_cost.py). "
        f"Node-hours = elapsed × nodes, 1 node = {GPUS_PER_NODE} GPUs. Azure runs "
        f"are not included.",
        "",
        f"**Kept {fmt(kept)} of {fmt(charged)} node-hours charged ({pct(kept, charged)}), "
        f"{fmt(kept * GPUS_PER_NODE)} GPU-hours.**",
        "",
        "## By task",
        "",
    ]
    rows = [[task, fmt(n_jobs[task]), fmt(sum(cost[task].values())), fmt(cost[task]["kept"]),
             pct(cost[task]["kept"], sum(cost[task].values())),
             *(fmt(cost[task][b]) for b in losses)] for task in SWEEP_TASKS]
    rows.append(["**total**", fmt(sum(n_jobs[t] for t in SWEEP_TASKS)), f"**{fmt(charged)}**",
                 f"**{fmt(kept)}**", pct(kept, charged), *(fmt(total[b]) for b in losses)])
    lines += md_table(["task", "jobs", "charged", "kept", "kept %", *(label[b] for b in losses)], rows)
    top = sorted(other.items(), key=lambda kv: -kv[1])[:3]
    lines += ["", f"Not in the total: {fmt(sum(other.values()))} node-hours in "
              f"{n_jobs['other']} non-sweep jobs, the largest "
              + ", ".join(f"`{name}` {nh:,.1f}" for name, nh in top) + "."]

    lines += ["", "## By size", "",
              "Per task, kept / charged node-hours. Jobs whose name carries no ladder "
              "size (data builds, models.json convert batches) are under *no size*.", ""]
    rows = []
    for size in [*LADDER, None]:
        if size not in by_size:
            continue
        t = by_size[size]
        s_charged = sum(sum(bk.values()) for bk in t.values())
        s_kept = sum(bk["kept"] for bk in t.values())
        worst = max(losses, key=lambda b: sum(bk[b] for bk in t.values()))
        worst_nh = sum(bk[worst] for bk in t.values())
        rows.append([size or "no size", fmt(s_charged), fmt(s_kept), pct(s_kept, s_charged),
                     *(f"{fmt(t[k]['kept'])} / {fmt(sum(t[k].values()))}" if k in t else "—"
                       for k in SWEEP_TASKS),
                     f"{label[worst]} {fmt(worst_nh)}" if worst_nh >= 0.5 else "—"])
    lines += md_table(["size", "charged", "kept", "kept %", *SWEEP_TASKS, "largest loss"], rows)

    lines += ["", "## By user", ""]
    rows = [[user, fmt(sum(c.values())), fmt(c["kept"]), pct(c["kept"], sum(c.values())),
             *(fmt(c[b]) for b in losses)] for user, c in sorted(by_user.items())]
    lines += md_table(["user", "charged", "kept", "kept %", *(label[b] for b in losses)], rows)

    lines += ["", "## Buckets", ""] + [f"- **{label[b]}**: {text}." for b, text in BUCKETS.items()]

    report = "\n".join(lines) + "\n"
    print(report)
    args.out.write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
