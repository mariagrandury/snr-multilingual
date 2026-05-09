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


def enumerate_targets_from_models_file(
    models_file: Path,
    dense_tail: int | None = 5,
    tail_pct: int | None = 10,
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
    args = p.parse_args()

    if args.models is None:
        args.models = sorted(
            glob.glob(str(REPO / "configs" / "signal_to_ratio" / "models_pretraining_*.txt"))
        )

    # Enumerate targets
    targets: list[Target] = []
    for mf in args.models:
        targets.extend(enumerate_targets_from_models_file(Path(mf)))

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
