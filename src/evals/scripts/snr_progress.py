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
    expand_pool, get_model, iters_for, load_hf_wandb_config, load_models,
    load_pools, load_tasks, stages_of, tasks_for_group,
)

LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
DEFAULT_ENTITY = "mariagrandury-epflnlp"
DEFAULT_PROJECT = "snr-experiments"

# Splits in the published HF dataset (completion truth — disk logs get wiped).
_DATASET_SPLITS = (
    "pretraining_custom", "pretraining_a06", "reference_hf",
    "posttraining", "distillation",
)
# snr conda env python (has huggingface_hub/pandas/pyarrow); system
# python3.11 — which the launchers use — does not.
_SNR_ENV_PYTHON = os.environ.get(
    "SNR_ENV_PYTHON", "/users/mariagrandury/miniconda3/envs/snr/bin/python"
)


# ---------------------------------------------------------------------------
# Completion truth = the published HF dataset (∪ disk scan). Scratch
# auto-cleans eval files after ~30 days, so disk alone undercounts what has
# actually been evaluated; the hub dataset is the authoritative record.
# ---------------------------------------------------------------------------

# The inline loader run under the snr env when this process can't import the
# HF/parquet stack itself (system python3.11). Prints {name: [tasks]} JSON.
_DATASET_LOADER_SRC = """
import json, sys
from collections import defaultdict
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

repo_id, splits = sys.argv[1], sys.argv[2].split(",")
m = defaultdict(set)
for split in splits:
    try:
        p = hf_hub_download(
            repo_id, "data/%s-00000-of-00001.parquet" % split,
            repo_type="dataset",
        )
        t = pq.read_table(p, columns=["name", "task"])
        for n, tk in zip(t.column("name").to_pylist(),
                         t.column("task").to_pylist()):
            m[n].add(tk)
    except Exception as e:
        print("WARN split %s: %r" % (split, e), file=sys.stderr)
print(json.dumps({k: sorted(v) for k, v in m.items()}))
"""


def _load_dataset_direct() -> dict[str, set[str]]:
    """Load the hub dataset in-process (works under the snr env)."""
    from collections import defaultdict

    import pyarrow.parquet as pq  # noqa: F401  (ImportError ⇒ caller bridges)
    from huggingface_hub import hf_hub_download

    repo_id = load_hf_wandb_config()["repo_id"]
    out: dict[str, set[str]] = defaultdict(set)
    for split in _DATASET_SPLITS:
        try:
            path = hf_hub_download(
                repo_id, f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
            )
            table = pq.read_table(path, columns=["name", "task"])
            for n, tk in zip(table.column("name").to_pylist(),
                             table.column("task").to_pylist()):
                out[n].add(tk)
        except Exception as e:  # one split missing/unreachable ≠ fatal
            print(f"[snr_progress] warn: dataset split {split!r}: {e!r}",
                  file=sys.stderr)
    return dict(out)


def _load_dataset_via_subprocess() -> dict[str, set[str]]:
    """Bridge to the snr-env python (system python3.11 lacks the HF stack)."""
    repo_id = load_hf_wandb_config()["repo_id"]
    out = subprocess.check_output(
        [_SNR_ENV_PYTHON, "-c", _DATASET_LOADER_SRC,
         repo_id, ",".join(_DATASET_SPLITS)],
        text=True,
    )
    return {k: set(v) for k, v in json.loads(out).items()}


_DATASET_CACHE: dict[str, set[str]] | None = None


def _dataset_completion() -> dict[str, set[str]]:
    """`dict[name] -> {tasks}` of every (name, task) row in the published
    HF dataset. Cached in-process. Graceful fallback to ``{}`` (disk-only)
    on any failure — never crashes."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    try:
        try:
            _DATASET_CACHE = _load_dataset_direct()
        except ImportError:
            _DATASET_CACHE = _load_dataset_via_subprocess()
    except Exception as e:
        print(f"[snr_progress] warn: could not load HF dataset completion "
              f"({e!r}); falling back to disk scan only.", file=sys.stderr)
        _DATASET_CACHE = {}
    return _DATASET_CACHE


def completed_tasks(name: str, ckpt_id: str, entity: str,
                    project: str) -> set[str]:
    """Union of dataset-recorded ∪ on-disk completed tasks for a NAME.

    For ``main`` checkpoints the eval NAME on disk / in the dataset is
    sometimes the bare ``<model>`` (no ``-main`` suffix) — e.g.
    ``gemma-3-1b-it`` vs the matrix NAME ``gemma-3-1b-it-main`` — so we
    also fold in the bare-model key when ckpt is ``main``."""
    ds = _dataset_completion()
    done = set(ds.get(name, set()))
    done |= scan_completed_tasks(name, entity, project)
    if ckpt_id == "main" and name.endswith("-main"):
        bare = name[: -len("-main")]
        done |= set(ds.get(bare, set()))
        done |= scan_completed_tasks(bare, entity, project)
    return done


@dataclass
class Target:
    """One (model, checkpoint) cell, with its expected harness NAME."""

    model_name: str  # e.g. apertus-350M-fwEdu30-fw270-seed1904
    ckpt_id: str  # e.g. iter2000  OR  stage1-step1413814
    name: str  # full NAME used by evaluate.sbatch (model-ckpt)
    completed: set[str] = field(default_factory=set)
    pending_jobs: list[tuple[str, str, str]] = field(default_factory=list)  # (jobid, jobname, state)


def enumerate_targets_from_json(model_names: list[str],
                                stage: str | None = None) -> list[Target]:
    """Return one Target per (model, ckpt) declared in configs/models.json.

    For each model in ``model_names``, expands its ``checkpoints.full_eval``
    list. The harness NAME convention is preserved verbatim from the
    historical pipeline:

      - Megatron iter checkpoints: ``<model>-iter<N>``
      - HF branch checkpoints:     ``<model>-<branch>``

    When ``stage`` is given (the pool's declared phase, e.g. ``posttraining``),
    only that stage's checkpoints are enumerated — a model that has no such
    stage contributes nothing. This is what keeps a posttraining pool from
    listing the *base* checkpoints of its base-only members (Qwen3-*-Base,
    gemma-3-*-pt, …).
    """
    targets: list[Target] = []
    for name in model_names:
        entry = get_model(name)
        if stage is not None and stage not in stages_of(name):
            continue
        ckpts = iters_for(name, subset="full_eval", stage=stage)
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


# ---------------------------------------------------------------------------
# Matrix mode: the full (model, ckpt, eval_group) set declared via stage-level
# `eval_groups` in configs/models.json. A model is in scope iff some stage has
# `eval_groups`; the cells are that stage's `checkpoints.full_eval` × each
# group in `eval_groups`.
# ---------------------------------------------------------------------------

# Ordered size tokens: small→large. Anything not here (30B-A3B, E2B, …) sorts
# last (rank = len(list)), keeping non-standard tags at the bottom.
_SIZE_ORDER = [
    "270M", "175M", "350M", "600M", "0.6B", "750M", "1B", "1.7B", "3B",
    "4B", "7B", "8B", "12B", "13B", "14B", "24B", "27B", "32B", "70B",
]


def _size_key(size: str | None) -> tuple[int, int]:
    """Numeric sort key: known sizes small→large by `_SIZE_ORDER`,
    non-standard tags (30B-A3B, E2B, …) sort last."""
    try:
        return (0, _SIZE_ORDER.index(size))
    except ValueError:
        return (1, 0)


@dataclass
class MatrixCell:
    model: str
    name: str          # f"{model}-{ckpt_id}"
    ckpt_id: str
    eval_group: str
    family: str
    size: str | None
    done: int = 0
    total: int = 0
    remaining: list[str] = field(default_factory=list)
    active_jobids: list[str] = field(default_factory=list)


def enumerate_matrix() -> list[MatrixCell]:
    """Every (model, ckpt, eval_group) cell from stage-level `eval_groups`."""
    cells: list[MatrixCell] = []
    for model, entry in load_models().items():
        kind = entry.get("checkpoint_kind")
        family = entry.get("family", model)
        size = entry.get("size")
        for stage, sdata in entry.get("stages", {}).items():
            groups = sdata.get("eval_groups")
            if not groups:
                continue
            for c in iters_for(model, subset="full_eval", stage=stage):
                ckpt_id = f"iter{c}" if kind == "megatron_iter" else (
                    c["branch"] if isinstance(c, dict) else str(c)
                )
                for group in groups:
                    cells.append(MatrixCell(
                        model=model, name=f"{model}-{ckpt_id}",
                        ckpt_id=ckpt_id, eval_group=group,
                        family=family, size=size,
                    ))
    return cells


def matrix_status(cell: MatrixCell) -> str:
    """Same semantics as the per-pool status_for()."""
    if cell.total and cell.done == cell.total:
        return "completed"
    if cell.active_jobids:
        return "in_progress"
    if cell.done > 0:
        return "in_progress"  # partial leftover, no active job
    return "not_submitted"


def run_matrix(entity: str, project: str) -> None:
    cells = enumerate_matrix()

    # Completion (dataset ∪ disk) + active jobs.
    group_tasks: dict[str, set[str]] = {}
    for cell in cells:
        tasks = group_tasks.setdefault(
            cell.eval_group, set(tasks_for_group(cell.eval_group)))
        done_set = completed_tasks(cell.name, cell.ckpt_id, entity, project)
        cell.total = len(tasks)
        cell.done = len(done_set & tasks)
        cell.remaining = sorted(t for t in tasks if t not in done_set)

    # Attach squeue active jobs by eval-<NAME> prefix (reuse the matcher).
    targets = [Target(model_name=c.model, ckpt_id=c.ckpt_id, name=c.name)
               for c in cells]
    attach_pending_jobs(targets, squeue_jobs())
    jobs_by_name: dict[str, list[str]] = defaultdict(list)
    for t in targets:
        jobs_by_name[t.name].extend(j[0] for j in t.pending_jobs)
    for cell in cells:
        # dedupe preserving order
        seen: set[str] = set()
        cell.active_jobids = [j for j in jobs_by_name.get(cell.name, [])
                              if not (j in seen or seen.add(j))]

    # Order: family A→Z, size small→large, ckpt, eval_group.
    cells.sort(key=lambda c: (c.family.lower(), _size_key(c.size),
                              c.ckpt_id, c.eval_group))

    # --- CSV ---------------------------------------------------------------
    import csv as _csv
    csv_path = REPO / "snr_progress_matrix.csv"
    with csv_path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["model", "name", "eval_group", "status", "done",
                    "total", "remaining", "active_jobids"])
        for c in cells:
            w.writerow([c.model, c.name, c.eval_group, matrix_status(c),
                        c.done, c.total, ",".join(c.remaining),
                        ",".join(c.active_jobids)])

    # --- Summary by eval_group --------------------------------------------
    print(f"=== SNR matrix: {len(cells)} (model, ckpt, eval_group) cells "
          f"({entity}/{project}) ===\n")
    by_group: dict[str, list[MatrixCell]] = defaultdict(list)
    for c in cells:
        by_group[c.eval_group].append(c)
    for group in sorted(by_group):
        gc = by_group[group]
        cell_done = sum(c.done for c in gc)
        cell_total = sum(c.total for c in gc)
        st = defaultdict(int)
        for c in gc:
            st[matrix_status(c)] += 1
        print(f"[{group}] {len(gc)} cells · {cell_done}/{cell_total} "
              f"(model,ckpt,task) done ({100 * cell_done / max(cell_total, 1):.1f}%)")
        for k in ("completed", "in_progress", "not_submitted"):
            print(f"      {k:>14}: {st.get(k, 0)} cells")

    # --- Per-model fully-done tally ---------------------------------------
    by_model: dict[str, list[MatrixCell]] = defaultdict(list)
    for c in cells:
        by_model[c.model].append(c)
    fully_done = [m for m, mc in by_model.items()
                  if all(matrix_status(c) == "completed" for c in mc)]
    not_done = [m for m in by_model if m not in fully_done]
    print(f"\nPer-model: {len(fully_done)}/{len(by_model)} models fully done; "
          f"{len(not_done)} not done.")
    print(f"  fully done: {sorted(fully_done)}")
    print(f"  not done  : {sorted(not_done)}")
    print(f"\nWrote {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--matrix",
        action="store_true",
        help="Full (model, ckpt, eval_group) matrix from stage-level "
             "eval_groups in models.json. Writes snr_progress_matrix.csv.",
    )
    p.add_argument(
        "--pool",
        required=False,
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

    if args.matrix:
        run_matrix(args.entity, args.project)
        return

    if not args.pool:
        p.error("--pool is required (or pass --matrix for the full "
                "(model, ckpt, eval_group) matrix).")
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")

    csv_path = REPO / "snr_progress.csv"

    # Enumerate targets directly from configs/models.json via the pool.
    # Checkpoints are selected from the pool's declared stage so a
    # posttraining pool lists only post-trained ckpts (base-only members
    # are skipped).
    model_names = expand_pool(args.pool)
    pool_stage = load_pools()[args.pool].get("stage")
    targets = enumerate_targets_from_json(model_names, stage=pool_stage)

    if args.filter:
        targets = [t for t in targets if args.filter in t.name]

    # Tasks come from configs/tasks.json groups.
    all_tasks = sorted(set(tasks_for_group(args.tasks_group)))
    total_tasks = len(all_tasks)

    # Completion = HF dataset (truth; disk logs get wiped) ∪ disk scan.
    for t in targets:
        t.completed = completed_tasks(t.name, t.ckpt_id, args.entity, args.project)
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
