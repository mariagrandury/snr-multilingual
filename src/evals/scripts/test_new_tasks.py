#!/usr/bin/env python3
"""Verify task names in a tasks-list file against the swiss-ai/lm-evaluation-
harness registry, then launch a small eval (`--limit 2`, per-task mode) so
each task is exercised end-to-end on a tiny sample.

Why this shape: cluster eval jobs install the swiss-ai fork inside the eval
container. To know whether a task name is valid (and thus worth submitting a
full job for) we check it against the same fork's YAML task definitions. We
keep a local clone at /iopsstor/scratch/cscs/$USER/lm-eval-swiss-ai (auto-
created + `git pull`'d each run); verification is a pure string lookup over
the ~15k YAML files, no dataset I/O — finishes in a few seconds.

After verification, we submit ONE sbatch through evaluate.sbatch with:
  - LM_EVAL_BACKEND=vllm, BATCH_TASKS=0   per-task isolation (a bad task
                                          only kills itself; lm_eval's
                                          per-task error handler logs and
                                          continues)
  - HARNESS_LIMIT=2                       2 examples per task (enough to
                                          flush dataset cache + verify
                                          generation works)
  - NAME suffixed `-tasktest`             keeps results out of the main
                                          NAME's eval_logs/W&B run
  - WANDB_PROJECT=snr-experiments-tasktest separates the test history

Side effect: every task that succeeds also caches its dataset under
HF_HOME — subsequent full-walltime jobs read offline.

Usage:
    python3.11 scripts/eval_new_tasks.py [--tasks-file PATH]
                                         [--cell apertus-175M-fwEdu30-fw270-seed1904-iter50000]
                                         [--limit 2] [--time 04:00:00] [--no-launch]

Exit codes:
    0   verification passed; sbatch submitted (or --no-launch)
    1   one or more tasks unregistered in swiss-ai harness — nothing submitted
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_FILE = REPO / "configs" / "signal_to_ratio" / "tasks_pretraining_full.txt"
DEFAULT_CELL = "apertus-175M-fwEdu30-fw270-seed1904-iter50000"
HF_BASE = Path("/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints")
SWISS_FORK = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'mariagrandury')}/lm-eval-swiss-ai")
SWISS_FORK_URL = "https://github.com/swiss-ai/lm-evaluation-harness.git"


def ensure_swiss_fork() -> None:
    """Clone or pull the swiss-ai harness fork."""
    if SWISS_FORK.exists():
        subprocess.run(["git", "-C", str(SWISS_FORK), "pull", "--quiet"],
                       check=False, capture_output=True)
        return
    print(f"[eval_new_tasks] cloning swiss-ai fork into {SWISS_FORK} ...", file=sys.stderr)
    SWISS_FORK.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", SWISS_FORK_URL, str(SWISS_FORK)],
        check=True,
    )


def harness_task_names() -> set[str]:
    """All `task:` and `group:` names declared in any YAML under
    lm_eval/tasks/. Single string-scan over ~15k files; no YAML parse needed.
    """
    names: set[str] = set()
    for yp in (SWISS_FORK / "lm_eval" / "tasks").rglob("*.yaml"):
        try:
            for line in yp.read_text().splitlines():
                s = line.strip()
                if s.startswith("task:") or s.startswith("group:"):
                    rest = s.split(":", 1)[1].strip()
                    # Skip lists like `task: [a, b]` or `task:` followed by `- a` lines
                    if rest and not rest.startswith("[") and not rest.startswith("-"):
                        names.add(rest.strip("'\""))
        except Exception:
            continue
    return names


def load_tasks_file(path: Path) -> list[str]:
    return [s for s in (l.strip() for l in path.read_text().splitlines())
            if s and not s.startswith("#")]


def find_iter_dir(cell: str) -> Path:
    """Resolve `apertus-{size}-...-iter{N}` -> staged HF iter_NNNNNNN dir."""
    if "-iter" not in cell:
        raise ValueError(f"cell name must end in -iter<N>: {cell}")
    cell_root, iter_num = cell.rsplit("-iter", 1)
    iter_dir = HF_BASE / cell_root / f"iter_{int(iter_num):07d}"
    if not (iter_dir / "config.json").is_file():
        raise FileNotFoundError(f"HF ckpt not found: {iter_dir}")
    return iter_dir


def tp_pp_for(cell: str) -> tuple[int, int]:
    """Per-size TP/PP per CLAUDE.md bug 14 (vLLM kv_heads constraint)."""
    sizes = {"175M": (4, 1), "350M": (1, 4), "600M": (2, 2), "1B": (1, 4)}
    for sz, tp_pp in sizes.items():
        if f"-{sz}-" in cell:
            return tp_pp
    raise ValueError(f"unknown size in cell: {cell}")


def submit_sbatch(tasks_csv: str, cell: str, iter_dir: Path,
                  limit: int, walltime: str) -> str:
    """Returns sbatch jobid."""
    name = f"{cell}-tasktest"
    tp, pp = tp_pp_for(cell)
    env = {
        **os.environ,
        "LM_EVAL_BACKEND": "vllm",
        "TOKENIZER": "alehc/swissai-tokenizer",
        "BOS": "true",
        "APPLY_CHAT_TEMPLATE": "false",
        "BATCH_TASKS": "0",
        "TP": str(tp),
        "PP": str(pp),
        "HARNESS_LIMIT": str(limit),
        "WANDB_ENTITY": "mariagrandury-epflnlp",
        "WANDB_PROJECT": "snr-experiments-tasktest",
        "TASKS": tasks_csv,
        # Override the production default (HF_DATASETS_OFFLINE=1 in
        # evaluate.sbatch) so the tasktest can DOWNLOAD any dataset that
        # isn't cached yet. With BATCH_TASKS=0 the per-task loop runs one
        # lm_eval call at a time → exactly one concurrent download → no
        # rate-limit risk. Side effect: this run populates the cache for
        # future production jobs.
        "HF_DATASETS_OFFLINE": "0",
    }
    cmd = [
        "sbatch", "--parsable",
        "--job-name", f"eval-{name}",
        "--partition", "normal",
        "--time", walltime,
        "scripts/evaluate.sbatch", str(iter_dir), name,
    ]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    p.add_argument("--cell", default=DEFAULT_CELL,
                   help=f"Cell + iter to evaluate (default: {DEFAULT_CELL})")
    p.add_argument("--limit", type=int, default=2,
                   help="Per-task example limit (default: 2)")
    p.add_argument("--time", default="04:00:00",
                   help="sbatch --time (default: 04:00:00)")
    p.add_argument("--no-launch", action="store_true",
                   help="Verify only; don't submit sbatch.")
    args = p.parse_args()

    if not args.tasks_file.exists():
        print(f"ERROR: tasks file not found: {args.tasks_file}", file=sys.stderr)
        return 1

    ensure_swiss_fork()
    requested = load_tasks_file(args.tasks_file)
    print(f"[eval_new_tasks] checking {len(requested)} tasks from {args.tasks_file}")
    registry = harness_task_names()
    print(f"  swiss-ai harness registry: {len(registry)} task/group names")

    unregistered = [t for t in requested if t not in registry]
    if unregistered:
        print()
        print(f"NOT IN swiss-ai harness ({len(unregistered)}):")
        for t in unregistered:
            print(f"  - {t}")
        print()
        print("Fix the tasks file before re-running. Nothing submitted.")
        return 1

    print(f"  registered: {len(requested)}/{len(requested)} ✓")

    if args.no_launch:
        return 0

    iter_dir = find_iter_dir(args.cell)
    tp, pp = tp_pp_for(args.cell)
    tasks_csv = ",".join(requested)
    print()
    print(f"[eval_new_tasks] submitting tasktest eval:")
    print(f"  cell:  {args.cell}")
    print(f"  ckpt:  {iter_dir}")
    print(f"  tasks: {len(requested)}  TP={tp} PP={pp}  --limit={args.limit}  walltime={args.time}")
    print(f"  NAME suffix: -tasktest    WANDB_PROJECT: snr-experiments-tasktest")
    jid = submit_sbatch(tasks_csv, args.cell, iter_dir, args.limit, args.time)
    print(f"  submitted jobid: {jid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
