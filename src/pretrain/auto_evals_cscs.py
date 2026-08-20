#!/usr/bin/env python3
"""
CSCS auto-eval watcher — the cluster twin of auto_evals_azure.py.

Every N saved checkpoints (default 2) plus each run's final checkpoint,
evaluate on the "auto" benchmark group (configs/tasks.json) and push to W&B
`mariagrandury-epflnlp/msnr` — the same project the training loss logs to.

Idempotent, safe to run alongside the trainings (login node, tmux):

    cd src/pretrain
    python3.11 auto_evals_cscs.py --watch 600     # one pass every 10 min

The eval job pushes to W&B with the key from your environment or, as
everywhere else in the cluster pipeline, from the fallback file
src/evals/scripts/wandb_api_key.txt — nothing to export here.

Each pass, per cell of the selected variant (--arch/--scheme, same flags as
the launcher), for each due checkpoint on disk:

  1. results already under LOGS_ROOT/<entity>/msnr/<cell>-iter<N>/  -> skip
  2. eval job for it already in squeue (any user)                    -> skip
  3. HF snapshot staged at <staging>/<cell>/iter_<N>  -> submit ONE eval job
       (src/evals/scripts/evaluate.sbatch, vLLM, BOS, no chat template,
        TP=1 — the ladder's KV-head counts only divide 1 — BATCH_TASKS=1)
  4. otherwise -> convert first: one conversion/convert-snr.sh --models job
       per cell with the missing iters (models.json-driven). Conversion jobs
       share one Slurm name, so while any is running no new ones are
       submitted; the next pass picks up whatever is still missing.

The convert -> eval sequencing resolves across passes, exactly like the
Azure watcher. configs/models.json is kept in sync with the grid
automatically (sync_models_json.sync() runs at the start of every pass);
the only manual precondition is the tokenizer pre-warmed into the offline
HF cache (see README "Before the first CSCS run").
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from launch_trainings import (  # noqa: E402
    HYPERPARAMS, SCHEME_B_LANGS, TOKENIZER_MODEL, cell_languages,
    exp_name, job_name, predictivity_cells, save_interval, schedule_for)
from pretrain_progress import CKPT_ROOT, ITER_RE, is_valid_iter_dir  # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR.parent))
from evals.scripts.utils.configs import tasks_for_benchmarks  # noqa: E402

EVALS_DIR = SCRIPT_DIR.parent / "evals"
CONVERT_SNR = SCRIPT_DIR / "conversion" / "convert-snr.sh"
TASKS_JSON = SCRIPT_DIR.parent.parent / "configs" / "tasks.json"

WANDB_ENTITY = "mariagrandury-epflnlp"
PROJECT_NAME = json.loads(
    (SCRIPT_DIR.parent.parent / "configs" / "hf_wandb.json").read_text()
)["wandb"]["project"]

# Converted HF checkpoints are the durable copy — persist them on capstor store
# (push-snr.py mirrors this tree to the public msnr Hub org). Not iopsstor
# scratch, which is auto-purged.
DEFAULT_STAGING = "/capstor/store/cscs/swissai/infra01/msnr-hf-models"
DEFAULT_LOGS_ROOT = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"

CONVERT_JOB_NAME = "convert-snr-models"  # fixed by convert-snr.sh's launcher
# The auto group spans 9 tasks (L=1) to 290 (L=100), so a fixed walltime
# can't fit both — and the whole batched run executes on 1 of the node's 4
# GPUs (the ladder's KV-head counts force TP=1 and vLLM clamps dense DP to
# 1). Per-task minutes extrapolated from the 36-sweep's per-task-mode
# timings (launch_ckpts_in_progress.sh: 4-10 min/task on the full node),
# ~/10 for batched mode, ~x2 for the single GPU. A walltime kill mid-batch
# writes NO results_*.json, so an undersized job would just be resubmitted
# and re-killed forever — err generous.
MIN_PER_TASK = {"90M": 0.6, "175M": 0.8, "350M": 1.2, "600M": 1.6,
                "1B": 2.0, "1.7B": 2.8}


def eval_walltime(size: str, n_tasks: int) -> str:
    """~1h overhead (container + vLLM cold start + W&B push) + per-task
    budget, rounded up to 15 min, capped at the normal queue's 11:59:59
    limit (launch_trainings.TIME_MAX_SEC) — an over-cap request is rejected
    at submission and would crash the watch loop."""
    minutes = math.ceil((60 + n_tasks * MIN_PER_TASK.get(size, 2.8)) / 15) * 15
    minutes = min(minutes, 719)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def auto_benchmarks() -> list[str]:
    """The `auto` group in configs/tasks.json — BENCHMARK names; each cell
    is evaluated on every benchmark's tasks in the languages it trains on
    (tasks_for_benchmarks x cell_languages)."""
    return json.loads(TASKS_JSON.read_text())["groups"]["auto"]


def saved_valid_iters(cell: str, root: Path) -> list[int]:
    """Sorted iters with a loadable checkpoint on disk for one cell."""
    ckpt_dir = root / cell / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(
        int(m.group(1))
        for e in ckpt_dir.iterdir()
        if (m := ITER_RE.match(e.name)) and is_valid_iter_dir(e)
    )


def active_jobs() -> set[str]:
    """All queued/running Slurm job names, ANY user — collaborators share the
    trees, so their in-flight converts/evals count as ours."""
    try:
        out = subprocess.run(["squeue", "-h", "--format=%j"],
                             capture_output=True, text=True, timeout=30)
        return set(out.stdout.split()) if out.returncode == 0 else set()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def evaluated(name: str, logs_root: Path) -> bool:
    """Results on disk for NAME (the same layout every eval path writes)."""
    base = logs_root / WANDB_ENTITY / PROJECT_NAME / name / "harness"
    return base.is_dir() and any(base.glob("eval_*/results_*.json"))


def hf_staged(cell: str, it: int, staging: Path) -> bool:
    """Converted AND complete: convert-snr.sh touches .hf_complete as its
    last step (config.json + a weights glob alone can match a half-written
    save_pretrained; its skip branch backfills the marker for snapshots
    converted before the marker existed)."""
    return (staging / cell / f"iter_{it:07d}" / ".hf_complete").is_file()


def submit_eval(cell: str, it: int, staging: Path, logs_root: Path,
                tasks: str, size: str, dry_run: bool) -> None:
    name = f"{cell}-iter{it}"
    hf_dir = staging / cell / f"iter_{it:07d}"
    n_tasks = tasks.count(",") + 1
    # Prefix-export via the process env rather than --export=ALL,K=V,...:
    # sbatch's --export uses commas as separators BETWEEN vars, so the
    # comma-joined TASKS list would be truncated at its first comma and the
    # job would silently evaluate a single task (the trap the retired
    # launch_pretraining_*.sh launchers documented). --export=ALL snapshots
    # the submission env intact.
    env = {**os.environ,
           "LM_EVAL_BACKEND": "vllm",
           "TOKENIZER": TOKENIZER_MODEL,
           "BOS": "true",
           "APPLY_CHAT_TEMPLATE": "false",
           "BATCH_TASKS": "1",
           "TP": "1", "PP": "1",
           "WANDB_ENTITY": WANDB_ENTITY,
           "WANDB_PROJECT": PROJECT_NAME,
           "LOGS_ROOT": str(logs_root),
           "TASKS": tasks}
    cmd = ["sbatch", f"--job-name={job_name('eval', name)}",
           f"--time={eval_walltime(size, n_tasks)}",
           "--export=ALL", "scripts/evaluate.sbatch", str(hf_dir), name]
    print(f"  submit: {job_name('eval', name)}")
    if dry_run:
        print(f"    (cd {EVALS_DIR} && TASKS=<{n_tasks} tasks> ... {' '.join(cmd)})")
    else:
        subprocess.run(cmd, cwd=EVALS_DIR, env=env, check=True)


def submit_convert(cell: str, iters: list[int], staging: Path,
                   dry_run: bool) -> None:
    cmd = ["bash", str(CONVERT_SNR), "--models", cell,
           "--iters", ",".join(str(i) for i in iters), "--submit"]
    env = {**os.environ,
           "HF_TOKENIZER": TOKENIZER_MODEL,   # forwarded into the container
           "STAGING_BASE": str(staging),      # final HF -> capstor (durable)
           # keep the large intermediate torch checkpoint on scratch (fast,
           # transient) instead of churning it on capstor.
           "TMP_TORCH_BASE": f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'mariagrandury')}/snr-hf-checkpoints/_tmp_torch"}
    print(f"  submit: convert {cell} iters {iters}")
    if dry_run:
        print(f"    (HF_TOKENIZER={TOKENIZER_MODEL} STAGING_BASE={staging} "
              f"{' '.join(cmd)})")
    else:
        subprocess.run(cmd, env=env, check=True)


def one_pass(args, root: Path, staging: Path, logs_root: Path,
             benchmarks: list[str]) -> None:
    # Keep configs/models.json following the grid — conversion and the W&B
    # push resolve cells through it. No-op when already in sync.
    from sync_models_json import sync
    added, updated = sync(args.arch, args.scheme)
    if added or updated:
        print(f"(models.json synced: +{len(added)} ~{len(updated)} cells "
              f"— commit the diff)")

    configs = json.loads(HYPERPARAMS[args.arch].read_text())["configs"]
    running = active_jobs() if not args.dry_run else set()
    convert_busy = CONVERT_JOB_NAME in running

    for c in predictivity_cells():
        scheme = args.scheme if c["L"] in SCHEME_B_LANGS else "A"
        cell = exp_name(c["size"], c["L"], args.arch, c["seed"], scheme)
        if args.name and cell != args.name:
            continue
        target = schedule_for(configs[c["size"]])[0]
        saved = saved_valid_iters(cell, root)
        if not saved:
            continue
        # Every Nth saved checkpoint on the cell's per-size save grid, plus
        # the run's final one whatever its number — same rule as Azure.
        due = [i for i in saved
               if i % (args.every * save_interval(target)) == 0 or i == target]
        # The cell's task list: every auto benchmark, in the languages
        # this cell trains on (e.g. L2 -> hellaswag + hellaswag_ru + ...).
        tasks = ",".join(tasks_for_benchmarks(
            benchmarks, cell_languages(c["L"], scheme)))
        # Convert EVERY saved checkpoint (persist all of them to capstor), but
        # evaluate only the due ones — conversion is the durability step, eval
        # is the expensive one we sample at 1/N.
        to_convert = [it for it in saved if not hf_staged(cell, it, staging)]
        print(f"{cell}: {len(saved)} saved | convert {to_convert or '-'} | eval due {due}")

        if to_convert and not convert_busy:
            submit_convert(cell, to_convert, staging, args.dry_run)
            # One conversion sbatch at a time (they share a Slurm name, so
            # dedupe is coarse); the next pass converts the remaining cells.
            convert_busy = True
        for it in due:
            name = f"{cell}-iter{it}"
            if evaluated(name, logs_root):
                continue
            if hf_staged(cell, it, staging) and job_name("eval", name) not in running:
                submit_eval(cell, it, staging, logs_root, tasks, c["size"],
                            args.dry_run)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=["deep", "shallow"], default="deep")
    p.add_argument("--scheme", choices=["A", "B"], default="A")
    p.add_argument("--name", help="watch a single cell (its full name)")
    p.add_argument("--every", type=int, default=2,
                   help="evaluate every N saved checkpoints (the final "
                        "checkpoint is always evaluated on top)")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="keep running, one pass every SECONDS")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", default=str(CKPT_ROOT),
                   help=f"Megatron run root (default: {CKPT_ROOT})")
    p.add_argument("--staging", default=DEFAULT_STAGING,
                   help=f"Converted-HF staging dir (default: {DEFAULT_STAGING})")
    p.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT,
                   help=f"Eval results root (default: {DEFAULT_LOGS_ROOT})")
    args = p.parse_args()

    benchmarks = auto_benchmarks()
    while True:
        one_pass(args, Path(args.root), Path(args.staging),
                 Path(args.logs_root), benchmarks)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
