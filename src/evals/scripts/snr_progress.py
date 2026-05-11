#!/usr/bin/env python3
"""SNR evaluation progress dashboard.

Enumerates the target (model, checkpoint) tuples implied by the
configs/signal_to_ratio/models_pretraining_*.txt files (10 ckpts for
models <3B, last ckpt for >=3B), cross-references the eval_logs
directory for completed tasks, and queries `squeue` for pending/running
jobs. Prints a per-checkpoint summary by default, and per-task detail
with --details.

Examples:
    # Per-ckpt summary across all models_pretraining_*.txt files
    python scripts/snr_progress.py

    # Restrict to one models file
    python scripts/snr_progress.py --models configs/signal_to_ratio/models_pretraining_custom.txt

    # Per-task breakdown for a specific checkpoint
    python scripts/snr_progress.py --details --filter apertus-350M-fwEdu30-fw270-seed1904-iter2000

    # Show only ckpts with no submitted jobs
    python scripts/snr_progress.py --status not_submitted
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
DEFAULT_ENTITY = "mariagrandury-epflnlp"
DEFAULT_PROJECT = "snr-experiments"
SMALL_MODEL_THRESHOLD_B = 3.0  # in billions


@dataclass
class Target:
    """One (model, checkpoint) cell, with its expected harness NAME."""

    model_name: str  # e.g. apertus-350M-fwEdu30-fw270-seed1904
    ckpt_id: str  # e.g. iter2000  OR  stage1-step1413814
    name: str  # full NAME used by evaluate.sbatch (model-ckpt)
    completed: set[str] = field(default_factory=set)
    pending_jobs: list[tuple[str, str, str]] = field(default_factory=list)  # (jobid, jobname, state)


def parse_size_b(model_name: str) -> float | None:
    """Extract model size in billions from a name like apertus-350M-..., apertus-1B-..., -3b-, -7B-."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([MmBb])", model_name)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return n / 1000 if unit == "m" else n


def derive_base_name(spec: str) -> str:
    """Mirror scripts/generate_snr_runner.sh:derive_name."""
    m = spec
    if m.startswith("https://huggingface.co/"):
        m = m[len("https://huggingface.co/") :]
        return m.rstrip("/").split("/")[-1]
    m = m.rstrip("/")
    if m.endswith("/checkpoints"):
        m = m[: -len("/checkpoints")]
    return os.path.basename(m)


def _list_cmd(spec: str, total: int | None, last: int | None,
              dense_tail: int | None, tail_pct: int | None) -> list[str]:
    cmd = [str(REPO / "scripts" / "list_checkpoints.sh"), spec]
    cmd += ["--total", str(total)] if total else ["--last", str(last)]
    if dense_tail:
        cmd += ["--dense-tail", str(dense_tail)]
    if tail_pct:
        cmd += ["--tail-pct", str(tail_pct)]
    return cmd


def list_megatron_iters(ckpt_dir: str, total: int | None, last: int | None,
                        dense_tail: int | None = None,
                        tail_pct: int | None = None) -> list[int]:
    """Run scripts/list_checkpoints.sh to get the same enumeration the generator uses."""
    try:
        out = subprocess.check_output(
            _list_cmd(ckpt_dir, total, last, dense_tail, tail_pct),
            stderr=subprocess.DEVNULL, text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def list_hf_branches(repo_url: str, total: int | None, last: int | None,
                     dense_tail: int | None = None,
                     tail_pct: int | None = None) -> list[str]:
    """Same logic for HF repos via list_checkpoints.sh."""
    try:
        out = subprocess.check_output(
            _list_cmd(repo_url, total, last, dense_tail, tail_pct),
            stderr=subprocess.DEVNULL, text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [b for b in out.splitlines() if b.strip()]


def parse_seed_iters(specs: list[str] | None) -> dict[str, set[int]]:
    """Parse repeated --seed-iters flags like 'seed28=6000,28000,42000'.

    Returns a {seed_str: {iter_int}} mapping. Cells whose seed is in the map
    are restricted to the listed iters; other seeds keep the canonical set.
    """
    out: dict[str, set[int]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--seed-iters expects seed=N,N,N (got '{spec}')")
        seed, csv = spec.split("=", 1)
        out[seed.strip()] = {int(x) for x in csv.split(",") if x.strip()}
    return out


def cell_seed(model_name: str) -> str | None:
    """Extract seed token from a name like apertus-...-seed1904."""
    m = re.search(r"-seed(\d+)$", model_name)
    return f"seed{m.group(1)}" if m else None


def enumerate_targets_from_models_file(
    models_file: Path,
    dense_tail: int | None = 5,
    tail_pct: int | None = 10,
    seed_iters: dict[str, set[int]] | None = None,
) -> list[Target]:
    """Return one Target per (model, ckpt) selected per the size-based rule.

    For <3B models we pick 10 evenly spaced ckpts plus up to ``dense_tail`` more
    from the last ``tail_pct`` percent of training. To keep the curve points
    comparable across runs of different lengths (some models in the file are
    still mid-resume and don't have iter 50000 on disk yet), we derive the
    canonical iter set from the longest fully-trained reference model in the
    same file and apply it to ALL small-size Megatron models. Half-trained
    models will list iters they don't have on disk yet — those show up as
    ``not_submitted`` until the resume training fills them in.

    For >=3B we just take the last 1 ckpt (sufficient for HF reference models).
    """
    # First pass: classify entries and collect on-disk iters for small Megatron.
    entries: list[tuple[str, str, str]] = []  # (kind, spec, base)
    small_meg_iters: dict[str, list[int]] = {}  # base -> on-disk iters
    for line in models_file.read_text().splitlines():
        spec = line.strip()
        if not spec or spec.startswith("#"):
            continue
        base = derive_base_name(spec)
        size_b = parse_size_b(base)
        small = size_b is None or size_b < SMALL_MODEL_THRESHOLD_B
        if spec.startswith(("/iopsstor", "/capstor")):
            kind = "meg_small" if small else "meg_large"
            entries.append((kind, spec, base))
            if small:
                small_meg_iters[base] = list_megatron_iters(
                    spec, 10, None, dense_tail, tail_pct
                )
        elif spec.startswith("https://huggingface.co/"):
            entries.append(("hf_small" if small else "hf_large", spec, base))
        else:
            print(f"# WARNING: unrecognized format: {spec}", file=sys.stderr)

    # Canonical iter set = the longest list among small Megatron models. Ties
    # broken by max iter so a fully-trained model wins over a half-trained one
    # of the same length.
    canonical_iters: list[int] | None = None
    if small_meg_iters:
        canonical_iters = max(
            small_meg_iters.values(),
            key=lambda its: (len(its), max(its) if its else 0),
        )

    # Second pass: emit Targets.
    targets: list[Target] = []
    for kind, spec, base in entries:
        if kind == "meg_small":
            iters = canonical_iters or small_meg_iters.get(base, [])
            # Restrict iters for cells whose seed has an explicit policy.
            # Use the user-provided list verbatim (sorted) — don't intersect
            # with `canonical_iters`, otherwise non-canonical iters like
            # 10000 / 20000 / 30000 would be silently dropped.
            seed = cell_seed(base)
            if seed_iters and seed in seed_iters:
                iters = sorted(seed_iters[seed])
            for it in iters:
                targets.append(
                    Target(model_name=base, ckpt_id=f"iter{it}", name=f"{base}-iter{it}")
                )
        elif kind == "meg_large":
            iters = list_megatron_iters(spec, None, 1)
            for it in iters:
                targets.append(
                    Target(model_name=base, ckpt_id=f"iter{it}", name=f"{base}-iter{it}")
                )
        elif kind == "hf_small":
            branches = list_hf_branches(spec, 10, None, dense_tail, tail_pct)
            for br in branches:
                targets.append(
                    Target(model_name=base, ckpt_id=br, name=f"{base}-{br}")
                )
        elif kind == "hf_large":
            branches = list_hf_branches(spec, None, 1)
            for br in branches:
                targets.append(
                    Target(model_name=base, ckpt_id=br, name=f"{base}-{br}")
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
    """All jobs visible to me (running + pending)."""
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "-o", "%i|%j|%t|%P|%M|%L"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
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
#   rows    — tasks, ORDER preserved from tasks_file (e.g. tasks_pretraining_full.txt)
#   cols    — every canonical-sweep ckpt (4 sizes × 9 cells × 13 iters = 468),
#             sorted by (size, cell name alphabetical, iter ascending)
#   cell    — green: task done · orange: pending · white: ckpt not in CSV (we
#             don't intend to eval that ckpt-benchmark combination)
#   x-axis  — no per-ckpt labels, just black vertical lines between size groups
# ---------------------------------------------------------------------------

HEATMAP_SIZES = ["175M", "350M", "600M", "1B"]
HEATMAP_MIXES = [(30, 70), (60, 40), (90, 10)]
HEATMAP_SEEDS = [28, 1797, 1904]
# Per-seed iter sets we actually want to evaluate (2026-05-10). Used as the
# default `--seed-iters` policy in `main()`, which means: the CSV snapshot,
# the heatmap, and every launcher that calls `snr_progress.py` will all
# restrict the canonical sweep to these (cell × iter) combinations unless
# the caller passes `--seed-iters` explicitly.
#   seed1904 → 9 picks from the canonical 13-iter set
#   seed28 / seed1797 → 10000-stepped grid (10k/20k/30k iters NOT canonical)
ITERS_SEED1904 = [6000, 12000, 22000, 28000, 42000, 44000, 46000, 48000, 50000]
ITERS_OTHER   = [6000, 10000, 20000, 30000, 42000, 44000, 46000, 48000, 50000]


def default_seed_iters() -> dict[str, set[int]]:
    """Default per-seed iter policy. Applied when no --seed-iters is passed."""
    return {
        "seed1904": set(ITERS_SEED1904),
        "seed1797": set(ITERS_OTHER),
        "seed28":   set(ITERS_OTHER),
    }


def make_eval_progress_heatmap(
    csv_path: Path, tasks_file: Path, out_path: Path
) -> None:
    """Render a tasks × checkpoints heatmap from `csv_path` to `out_path`.

    See the module-level comment block above for the cell-color semantics
    and the column ordering rule.
    """
    import csv as _csv
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    tasks = [
        line.strip()
        for line in tasks_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    # CSV: name -> set of pending tasks. Missing key  ==>  white column.
    csv_rows: dict[str, set[str]] = {}
    with csv_path.open() as fh:
        for r in _csv.DictReader(fh):
            rem = (r.get("remaining") or "").strip()
            csv_rows[r["name"]] = set(rem.split(",")) if rem else set()

    # Build the column list and capture size-group boundaries.
    # Per-seed iter set: seed1904 uses ITERS_SEED1904; others use
    # ITERS_OTHER. (The two sets are the same length, so each cell
    # contributes the same number of columns.)
    columns: list[str] = []
    size_boundaries: list[int] = []  # column indices where a new size starts (excl. 0)
    for s_idx, size in enumerate(HEATMAP_SIZES):
        if s_idx > 0:
            size_boundaries.append(len(columns))
        cells = sorted(
            (f"apertus-{size}-fwEdu{e}-fw2{w}-seed{s}", s)
            for (e, w) in HEATMAP_MIXES
            for s in HEATMAP_SEEDS
        )
        for cell, seed in cells:
            iters = ITERS_SEED1904 if seed == 1904 else ITERS_OTHER
            for it in iters:
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
    n_iters_each = len(ITERS_SEED1904)  # same length as ITERS_OTHER
    ax.set_xlabel(
        f"checkpoints (per size: 9 cells × {n_iters_each} iters, "
        "sorted alphabetical × iter ascending; "
        "seed1904 iters differ from seed28 / seed1797 — see HEATMAP_ITERS_* in script)"
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
        "--models",
        action="append",
        default=None,
        help="Path to a models_pretraining_*.txt file (repeatable). Default: all matching files.",
    )
    p.add_argument(
        "--tasks-file",
        default=str(REPO / "configs" / "signal_to_ratio" / "tasks_pretraining_full.txt"),
        help="Task list file used to compute total tasks per ckpt.",
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
        "--seed-iters",
        action="append",
        default=None,
        help=(
            "Restrict iters for a specific seed. Format: seed28=6000,28000,42000 "
            "(repeatable). Cells whose seed is listed only emit the listed iters; "
            "other seeds keep the canonical 13-iter set."
        ),
    )
    p.add_argument(
        "--plot",
        metavar="PATH",
        help=(
            "After writing the CSV snapshot, render a tasks × checkpoints heatmap "
            "to PATH. Rows = tasks in --tasks-file order. Columns = the canonical "
            "4×9×13 sweep (sorted by size, then alphabetical). Cell color: green = "
            "task done, orange = pending, white = ckpt not in CSV."
        ),
    )
    args = p.parse_args()
    # Per-seed iter policy: caller's --seed-iters wins; otherwise apply the
    # project default (ITERS_SEED1904 / ITERS_OTHER above).
    seed_iters = parse_seed_iters(args.seed_iters) if args.seed_iters else default_seed_iters()
    # Always written next to this script. Launchers read it directly so they
    # don't re-enumerate the matrix per cell. Schema:
    #   name,status,done,total,remaining,active_jobids
    csv_path = REPO / "snr_progress.csv"

    if args.models is None:
        args.models = sorted(
            glob.glob(str(REPO / "configs" / "signal_to_ratio" / "models_pretraining_*.txt"))
        )

    # Enumerate targets
    targets: list[Target] = []
    for mf in args.models:
        targets.extend(
            enumerate_targets_from_models_file(Path(mf), seed_iters=seed_iters)
        )

    if args.filter:
        targets = [t for t in targets if args.filter in t.name]

    # Tasks
    tasks_path = Path(args.tasks_file)
    all_tasks = sorted(
        {
            line.strip()
            for line in tasks_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    )
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
        make_eval_progress_heatmap(csv_path, tasks_path, Path(args.plot))

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
