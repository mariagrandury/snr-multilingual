#!/usr/bin/env python3
"""SNR evaluation progress dashboard.

Enumerates the target (model, checkpoint) tuples declared in
configs/models.json (filtered by --pool or --filter), cross-references
the eval_logs directory for completed tasks, and queries `squeue` for
pending/running jobs. Prints a per-checkpoint summary by default, and
per-task detail with --details.

Examples:
    # Per-ckpt summary for the recommended pool (all 3 seeds + externals)
    python scripts/snr_progress.py --pool seeds_28_1797_1904

    # Restrict to one Apertus seed pool
    python scripts/snr_progress.py --pool seeds_1904

    # Per-task breakdown for a specific checkpoint
    python scripts/snr_progress.py --pool seeds_28_1797_1904 \
        --details --filter apertus-350M-fwEdu30-fw270-seed1904-iter6000

    # Show only ckpts with no submitted jobs
    python scripts/snr_progress.py --pool seeds_28_1797_1904 \
        --status not_submitted
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "utils"))
from configs import (  # noqa: E402
    expand_pool, get_model, iters_for, load_pools, load_tasks,
    tasks_for_group,
)

LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
DEFAULT_ENTITY = "mariagrandury-epflnlp"
DEFAULT_PROJECT = "snr-experiments"


@dataclass
class Target:
    """One (model, checkpoint) cell, with its expected harness NAME."""

    model_name: str  # e.g. apertus-350M-fwEdu30-fw270-seed1904
    ckpt_id: str  # e.g. iter2000  OR  stage1-step1413814
    name: str  # full NAME used by evaluate.sbatch (model-ckpt)
    completed: set[str] = field(default_factory=set)
    pending_jobs: list[tuple[str, str, str]] = field(default_factory=list)  # (jobid, jobname, state)


def enumerate_targets_from_json(model_names: list[str]) -> list[Target]:
    """Return one Target per (model, ckpt) declared in configs/models.json.

    For each model in ``model_names``, expands its ``checkpoints.full_eval``
    list. The harness NAME convention is preserved verbatim from the
    historical pipeline:

      - Megatron iter checkpoints: ``<model>-iter<N>``
      - HF branch checkpoints:     ``<model>-<branch>``
    """
    targets: list[Target] = []
    for name in model_names:
        entry = get_model(name)
        ckpts = iters_for(name, subset="full_eval")
        kind = entry["checkpoint_kind"]
        for c in ckpts:
            if kind == "megatron_iter":
                ckpt_id = f"iter{c}"
            else:  # hf_branch / hf_local
                ckpt_id = c["branch"] if isinstance(c, dict) else str(c)
            targets.append(
                Target(model_name=name, ckpt_id=ckpt_id, name=f"{name}-{ckpt_id}")
            )
    return targets


def scan_completed_tasks(name: str, entity: str, project: str) -> set[str]:
    """Tasks with results = per_task/<task>/ subdirs ∪ keys in any results_*.json."""
    base = LOGS_BASE / entity / project / name / "harness"
    completed: set[str] = set()
    if not base.is_dir():
        return completed
    # per_task/ subdirs from killed runs
    for d in base.glob("eval_*/per_task/*"):
        if d.is_dir():
            completed.add(d.name)
    # results_*.json keys from clean (merged) runs
    for f in base.glob("eval_*/results_*.json"):
        try:
            data = json.loads(f.read_text())
            completed.update((data.get("results") or {}).keys())
        except Exception:
            pass
    return completed


def squeue_jobs() -> list[dict]:
    """All jobs visible to me (running + pending). Returns [] on hosts
    without `squeue` (e.g. local Mac) so the script still produces a
    CSV without job-pending info."""
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "-o", "%i|%j|%t|%P|%M|%L"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        jobid, jobname, state, partition, time_used, time_left = line.split("|")
        rows.append(
            {"jobid": jobid, "jobname": jobname, "state": state,
             "partition": partition, "time": time_used, "left": time_left}
        )
    return rows


def attach_pending_jobs(targets: list[Target], jobs: list[dict]) -> None:
    """Match jobs by name pattern eval-<NAME>{,-b,-suffix}."""
    by_name = defaultdict(list)
    for t in targets:
        by_name[t.name].append(t)
    for j in jobs:
        jn = j["jobname"]
        if not jn.startswith("eval-"):
            continue
        # Strip "eval-" prefix and any optional "-suffix" we add (-b, -srun-debug, etc.)
        candidate = jn[len("eval-") :]
        # Match the longest target name that's a prefix of candidate
        best = None
        for name in by_name:
            if candidate == name or candidate.startswith(name + "-"):
                if best is None or len(name) > len(best):
                    best = name
        if best:
            for t in by_name[best]:
                t.pending_jobs.append((j["jobid"], jn, j["state"]))


def render_bar(done: int, total: int, width: int = 25) -> str:
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Heatmap: per-(task, checkpoint) status from snr_progress.csv
#   rows    — tasks, ORDER preserved from the task group (--tasks-group).
#   cols    — every canonical-sweep ckpt (4 sizes × 9 cells × 9 iters/cell),
#             sorted by (size, cell name alphabetical, iter ascending).
#             Per-cell iter list comes from configs/models.json
#             `checkpoints.full_eval`.
#   cell    — green: task done · orange: pending · white: ckpt not in CSV (we
#             don't intend to eval that ckpt-benchmark combination)
#   x-axis  — no per-ckpt labels, just black vertical lines between size groups
# ---------------------------------------------------------------------------

HEATMAP_SIZES = ["175M", "350M", "600M", "1B"]
HEATMAP_MIXES = [(30, 70), (60, 40), (90, 10)]
HEATMAP_SEEDS = [28, 1797, 1904]


def make_eval_progress_heatmap(
    csv_path: Path, tasks: list[str], out_path: Path
) -> None:
    """Render a tasks × checkpoints heatmap from `csv_path` to `out_path`.

    See the module-level comment block above for the cell-color semantics
    and the column ordering rule.
    """
    import csv as _csv
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # CSV: name -> set of pending tasks. Missing key  ==>  white column.
    csv_rows: dict[str, set[str]] = {}
    with csv_path.open() as fh:
        for r in _csv.DictReader(fh):
            rem = (r.get("remaining") or "").strip()
            csv_rows[r["name"]] = set(rem.split(",")) if rem else set()

    # Build the column list and capture size-group boundaries. Per-cell
    # iter list comes from configs/models.json `checkpoints.full_eval`.
    columns: list[str] = []
    size_boundaries: list[int] = []  # column indices where a new size starts (excl. 0)
    for s_idx, size in enumerate(HEATMAP_SIZES):
        if s_idx > 0:
            size_boundaries.append(len(columns))
        cells = sorted(
            f"apertus-{size}-fwEdu{e}-fw2{w}-seed{s}"
            for (e, w) in HEATMAP_MIXES
            for s in HEATMAP_SEEDS
        )
        for cell in cells:
            for it in iters_for(cell, subset="full_eval"):
                columns.append(f"{cell}-iter{it}")

    n_rows = len(tasks)
    n_cols = len(columns)
    # 0 = white (no row in CSV), 1 = orange (pending), 2 = green (done)
    matrix = [[0] * n_cols for _ in range(n_rows)]
    for j, name in enumerate(columns):
        rem = csv_rows.get(name)
        if rem is None:
            continue  # white
        for i, task in enumerate(tasks):
            matrix[i][j] = 1 if task in rem else 2

    cmap = ListedColormap(["#ffffff", "#ff9933", "#90ee90"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(
        figsize=(max(12, n_cols * 0.04), max(8, n_rows * 0.10 + 1.5))
    )
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(tasks, fontsize=5)
    ax.set_xticks([])

    for b in size_boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=1.2)

    # Size labels centered above each block.
    block_starts = [0, *size_boundaries, n_cols]
    for i_block, size in enumerate(HEATMAP_SIZES):
        left, right = block_starts[i_block], block_starts[i_block + 1]
        ax.text(
            (left + right - 1) / 2, -1.5, size,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.5)
    n_iters_each = len(iters_for(
        f"apertus-{HEATMAP_SIZES[0]}-fwEdu{HEATMAP_MIXES[0][0]}-fw2{HEATMAP_MIXES[0][1]}-seed{HEATMAP_SEEDS[-1]}",
        subset="full_eval",
    ))
    ax.set_xlabel(
        f"checkpoints (per size: 9 cells × {n_iters_each} iters, "
        "sorted alphabetical × iter ascending; per-seed iter sets in "
        "configs/models.json `checkpoints.full_eval`)"
    )

    # Tally the legend for context.
    n_done = sum(1 for row in matrix for v in row if v == 2)
    n_pending = sum(1 for row in matrix for v in row if v == 1)
    n_white = n_rows * n_cols - n_done - n_pending
    legend_handles = [
        Patch(facecolor="#ffffff", edgecolor="black",
              label=f"not in pipeline ({n_white})"),
        Patch(color="#ff9933", label=f"pending ({n_pending})"),
        Patch(color="#90ee90", label=f"done ({n_done})"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
    )

    fig.suptitle(
        f"Eval progress: {n_rows} tasks × {n_cols} checkpoints "
        f"({100 * n_done / max(n_rows * n_cols, 1):.1f}% done)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pool",
        required=True,
        help="Pool name from configs/models.json. Enumerates the pool's "
             "models × their checkpoints.full_eval lists.",
    )
    p.add_argument(
        "--tasks-group",
        default="pretraining_full",
        help="Task group name from configs/tasks.json (default: pretraining_full).",
    )
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument(
        "--filter",
        default=None,
        help="Substring filter on the harness NAME (model-ckpt).",
    )
    p.add_argument(
        "--status",
        choices=["all", "completed", "in_progress", "pending", "not_submitted"],
        default="all",
        help="Filter rows by overall ckpt status.",
    )
    p.add_argument(
        "--details",
        action="store_true",
        help="Show per-task status for each ckpt (verbose).",
    )
    p.add_argument(
        "--plot",
        metavar="PATH",
        help=(
            "After writing the CSV snapshot, render a tasks × checkpoints "
            "heatmap to PATH. Rows = tasks in --tasks-group order. "
            "Columns = pool members × full_eval iters. Cell color: green "
            "= task done, orange = pending, white = ckpt not in CSV."
        ),
    )
    args = p.parse_args()

    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")

    csv_path = REPO / "snr_progress.csv"

    # Enumerate targets directly from configs/models.json via the pool.
    model_names = expand_pool(args.pool)
    targets = enumerate_targets_from_json(model_names)

    if args.filter:
        targets = [t for t in targets if args.filter in t.name]

    # Tasks come from configs/tasks.json groups.
    all_tasks = sorted(set(tasks_for_group(args.tasks_group)))
    total_tasks = len(all_tasks)

    # Scan completion + jobs
    for t in targets:
        t.completed = scan_completed_tasks(t.name, args.entity, args.project)
    attach_pending_jobs(targets, squeue_jobs())

    def status_for(t: Target) -> str:
        done = len(t.completed & set(all_tasks))
        if done == total_tasks:
            return "completed"
        if t.pending_jobs:
            return "in_progress" if any(j[2] == "R" for j in t.pending_jobs) else "pending"
        if done > 0:
            return "in_progress"  # has partial results, no active job — partial leftover
        return "not_submitted"

    # Always-on CSV snapshot: schema name,status,done,total,remaining,active_jobids.
    # The launchers (launch_pretraining_{hf,megatron}.sh) read this directly to size
    # jobs and decide what to submit, without re-enumerating per cell. Written
    # BEFORE --status / --filter narrows `targets`, so the on-disk file always
    # describes the FULL matrix that the snapshot was taken over.
    import csv as _csv
    all_tasks_set = set(all_tasks)
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["name", "status", "done", "total", "remaining", "active_jobids"])
        for t in sorted(targets, key=lambda x: x.name):
            done = len(t.completed & all_tasks_set)
            remaining = ",".join(task for task in all_tasks if task not in t.completed)
            jobids = ",".join(j[0] for j in t.pending_jobs)
            w.writerow([t.name, status_for(t), done, total_tasks, remaining, jobids])

    if args.plot:
        make_eval_progress_heatmap(csv_path, all_tasks, Path(args.plot))

    if args.status != "all":
        targets = [t for t in targets if status_for(t) == args.status]

    if args.details:
        # Per-(ckpt, task) breakdown
        for t in targets:
            done_set = t.completed & set(all_tasks)
            print(f"\n=== {t.name} — {len(done_set)}/{total_tasks} done ===")
            if t.pending_jobs:
                print(f"  Pending jobs: {', '.join(j[0]+'('+j[2]+')' for j in t.pending_jobs)}")
            for task in all_tasks:
                mark = "✓" if task in done_set else "·"
                print(f"  {mark} {task}")
        return

    # Per-ckpt summary table
    counts = defaultdict(int)
    print(f"=== SNR progress: {len(targets)} ckpts × {total_tasks} tasks ({args.entity}/{args.project}) ===\n")
    for t in sorted(targets, key=lambda x: x.name):
        done = len(t.completed & set(all_tasks))
        bar = render_bar(done, total_tasks)
        st = status_for(t)
        counts[st] += 1
        pj = ""
        if t.pending_jobs:
            ids = ",".join(f"{j[0]}({j[2]})" for j in t.pending_jobs)
            pj = f"  jobs={ids}"
        print(f"  {bar} {done:>3}/{total_tasks}  {t.name:<60}  [{st}]{pj}")

    cell_total = len(targets) * total_tasks
    cell_done = sum(len(t.completed & set(all_tasks)) for t in targets)
    print(
        f"\nSummary: {cell_done}/{cell_total} (model, ckpt, task) cells completed"
        f" ({100 * cell_done / max(cell_total,1):.1f}%)"
    )
    for k in ("completed", "in_progress", "pending", "not_submitted"):
        print(f"  {k:>14}: {counts.get(k, 0)} ckpts")


if __name__ == "__main__":
    main()
