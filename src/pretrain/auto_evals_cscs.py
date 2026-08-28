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

  1. every task of the cell's list already has a result under
     LOGS_ROOT/<entity>/msnr/<cell>-iter<N>/                        -> skip
     (task-level, so adding a benchmark reaches evaluated checkpoints)
  2. eval job for it already in squeue (any user)                    -> skip
  2b. --max-attempts eval runs already wrote nothing at all -> diagnose
     rather than resubmit. A dataset missing from the offline cache is
     downloaded and the checkpoint retried immediately; every other cause
     is held back and recorded in <logs-root>/auto_eval_errors.json, so a
     checkpoint that cannot succeed stops costing a job per pass.
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
import tempfile
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
from evals.scripts._eval_status import completed_tasks  # noqa: E402

EVALS_DIR = SCRIPT_DIR.parent / "evals"
CONVERT_SNR = SCRIPT_DIR / "conversion" / "convert-snr.sh"
TASKS_JSON = SCRIPT_DIR.parent.parent / "configs" / "tasks.json"
EVAL_JOB_LOGS = EVALS_DIR / "logs"          # evaluate.sbatch: --output=logs/%x_%j
DATASET_MANIFEST = EVALS_DIR / "configs" / "eval_datasets.txt"
DOWNLOAD_DATASETS = EVALS_DIR / "scripts" / "download_eval_datasets.py"
ERRORS_JSON = "auto_eval_errors.json"       # written under --logs-root each pass

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
# The auto group spans 9 tasks (L=1) to 290 (L=100), so a fixed walltime can't
# fit both — and the whole batched run executes on 1 of the node's 4 GPUs (the
# ladder's KV-head counts force TP=1 and vLLM clamps dense DP to 1).
#
# Elapsed time is very close to linear in the task count. Fitted on 69
# completed eval jobs (2026-08-21..27), median elapsed per (size, n_tasks):
#
#     size   9 tasks  18 tasks  60 tasks    overhead   per task
#     90M      7.9       13.9         -      1.95 min   0.667 min
#     175M     8.3       13.9         -      2.70 min   0.622 min
#     350M     9.1       15.3      47.4      2.34 min   0.751 min
#
# The 350M fit (taken on 9 -> 60) predicts 47.4 min at 60 tasks against 47.4
# measured, so extrapolating it to the larger language settings is sound.
#
# Model size barely moves the per-task cost at this scale — these are small
# models and the run is dominated by dataset load and tokenization, not by the
# forward pass. That will stop holding as the forward pass grows, and 600M+
# have no eval run yet, so those keep their older conservative estimates
# rather than an extrapolation dressed up as a measurement. The step from
# 350M's measured 0.75 to 600M's 1.6 is that ignorance, not a cliff.
#
# A walltime kill mid-batch writes NO results_*.json (BATCH_TASKS=1 is a single
# lm_eval call), so an undersized job is pure loss: hence SAFETY below, and the
# generous fixed overhead relative to the ~2.5 min measured — container pull and
# vLLM cold start are both much slower on a loaded node.
MIN_PER_TASK = {"90M": 0.67, "175M": 0.62, "350M": 0.75,   # measured
                "600M": 1.6, "1B": 2.0, "1.7B": 2.8}       # not measured
OVERHEAD_MIN = 15   # measured 2-2.7; the rest is cold-start headroom
SAFETY = 1.5        # on the per-task term only


def eval_walltime(size: str, n_tasks: int) -> str:
    """Fixed overhead + per-task budget x SAFETY, rounded up to 15 min, capped
    at the normal queue's 11:59:59 limit (launch_trainings.TIME_MAX_SEC) — an
    over-cap request is rejected at submission and would crash the watch loop."""
    minutes = OVERHEAD_MIN + n_tasks * MIN_PER_TASK.get(size, 2.8) * SAFETY
    minutes = min(math.ceil(minutes / 15) * 15, 719)
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


def evaluated(name: str, logs_root: Path, tasks: list[str]) -> bool:
    """True when every task in TASKS already has a result for NAME.

    Task-level, not checkpoint-level: `any results_*.json` would mean that
    adding a benchmark to the `auto` group never reaches checkpoints already
    evaluated on the old list — the sweep would silently carry two different
    task sets. The inner runner is already per-task idempotent
    (_run_per_task.sh filters through the same _eval_status.completed_tasks),
    so a resubmitted job runs ONLY the new tasks and merges them in.
    """
    return not set(tasks) - completed_tasks(
        name, WANDB_ENTITY, PROJECT_NAME, str(logs_root))


def barren_attempts(name: str, logs_root: Path) -> int:
    """Eval runs for NAME that produced nothing at all.

    The task gate asks "is this done?", never "can this ever finish?", so a
    checkpoint whose eval fails outright is resubmitted once per pass forever
    — 196 such jobs on the L50 cells before this existed (one uncached dataset
    aborts the whole BATCH_TASKS=1 call, so not one result lands). An
    `eval_*/` with neither a results file nor a non-empty per_task/ is a total
    loss; a run that saved anything counts as progress and doesn't.
    """
    base = logs_root / WANDB_ENTITY / PROJECT_NAME / name / "harness"
    return sum(1 for d in sorted(base.glob("eval_*")) if d.is_dir()
               and not any(d.glob("results_*.json"))
               and not any(f for t in d.glob("per_task/*") for f in t.iterdir()))


# lm_eval instantiates every task in the batch up front, so ONE dataset that
# isn't in the offline cache aborts the entire call — the compute nodes have
# no internet. The watcher runs on the login node, which does, so this is the
# one failure it can repair itself.
MISSING_DATASET_RE = re.compile(
    r"Couldn't reach '([^']+)' on the Hub \(OfflineModeIsEnabled\)")
ERROR_LINE_RE = re.compile(r"^(?:\d+: )?(\w*(?:Error|Exception)): (.+)$", re.M)
_LOG_TAIL = 400_000     # bytes; the traceback is at the end of a ~14 MB log


def eval_error(name: str) -> tuple[str, str]:
    """Why NAME's most recent eval job wrote nothing.

    ("dataset", <hf repo>) — fixable here, see fix_missing_dataset.
    ("other", <last Error line>) / ("unknown", ...) — needs a human; recorded
    in the errors file rather than retried into the ground.
    """
    logs = sorted(EVAL_JOB_LOGS.glob(f"{job_name('eval', name)}_*.err"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs[:2]:                     # the two most recent attempts
        for path in (log, log.with_suffix(".out")):   # srun splits the two
            try:
                with open(path, errors="ignore") as f:
                    f.seek(max(0, path.stat().st_size - _LOG_TAIL))
                    text = f.read()
            except OSError:
                continue
            hit = MISSING_DATASET_RE.search(text)
            if hit:
                return "dataset", hit.group(1)
            lines = ERROR_LINE_RE.findall(text)
            if lines:
                return "other", f"{lines[-1][0]}: {lines[-1][1][:160]}"
    return "unknown", "no eval job log found"


_DATASET_TRIED: set[str] = set()


def fix_missing_dataset(repo: str, dry_run: bool) -> bool:
    """Add REPO to the eval-dataset manifest and build it into the offline
    cache, so the next submission gets past it. Once per repo per process:
    if the build fails, the checkpoint lands in the errors file instead of
    re-downloading every pass."""
    if repo in _DATASET_TRIED:
        return False
    _DATASET_TRIED.add(repo)
    listed = repo in DATASET_MANIFEST.read_text().split()
    print(f"  missing offline dataset {repo}"
          f"{'' if listed else ' (also absent from eval_datasets.txt)'}"
          f" — building it into the cache")
    if dry_run:
        print(f"    (would append to {DATASET_MANIFEST.name} and run "
              f"{DOWNLOAD_DATASETS.name})")
        return False
    if not listed:
        with open(DATASET_MANIFEST, "a") as f:
            f.write(f"{repo}\n")
        print(f"  ({DATASET_MANIFEST.name} updated — commit the diff)")
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as manifest:
        manifest.write(f"{repo}\n")
        manifest.flush()
        out = subprocess.run([sys.executable, str(DOWNLOAD_DATASETS),
                              manifest.name], capture_output=True, text=True)
    ok = out.returncode == 0 and "0 failed" in out.stdout
    print(f"  {'built ' + repo + ' — retrying' if ok else 'could NOT build ' + repo}")
    if not ok:
        print("   ", (out.stdout or out.stderr).strip().splitlines()[-1:])
    return ok


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
    errors: dict[str, dict] = {}   # checkpoints held back, written out below

    for c in predictivity_cells():
        scheme = args.scheme if c["L"] in SCHEME_B_LANGS else "A"
        cell = exp_name(c["size"], c["L"], args.arch, c["seed"], scheme)
        if args.name and cell != args.name:
            continue
        # capstor intermittently faults a read outright (Errno 5 / 108 — the
        # same blips data_progress.py works around, hit here on a .hf_complete
        # probe). The watcher now runs unattended behind every launch, so one
        # blip must cost one cell for one pass, not kill the whole loop.
        try:
            convert_busy = one_cell(args, c, cell, scheme, configs, root,
                                    staging, logs_root, benchmarks, running,
                                    convert_busy, errors)
        except OSError as e:
            print(f"{cell}: skipped this pass — {e.strerror or e}",
                  file=sys.stderr)

    # One place to look for what is stuck and why. A snapshot, not a log: a
    # checkpoint drops out of it as soon as an eval writes results, so an
    # empty file means nothing is held back.
    path = logs_root / ERRORS_JSON
    if not args.dry_run:
        path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
    if errors:
        print(f"\n{len(errors)} checkpoint(s) held back after "
              f"{args.max_attempts} failed evals — details in {path}")


def one_cell(args, c: dict, cell: str, scheme: str, configs: dict, root: Path,
             staging: Path, logs_root: Path, benchmarks: list[str],
             running: set[str], convert_busy: bool, errors: dict) -> bool:
    """One cell of a pass: convert what is missing, evaluate what is due.

    Returns convert_busy — only one conversion sbatch may be in flight at a
    time (they share a Slurm name), so the flag carries across cells.
    """
    target = schedule_for(configs[c["size"]])[0]
    saved = saved_valid_iters(cell, root)
    if not saved:
        return convert_busy
    # Every Nth saved checkpoint on the cell's per-size save grid, plus
    # the run's final one whatever its number — same rule as Azure.
    due = [i for i in saved
           if i % (args.every * save_interval(target)) == 0 or i == target]
    # The cell's task list: every auto benchmark, in the languages
    # this cell trains on (e.g. L2 -> hellaswag + hellaswag_ru + ...).
    task_list = tasks_for_benchmarks(benchmarks, cell_languages(c["L"], scheme))
    tasks = ",".join(task_list)
    # Convert EVERY saved checkpoint (persist all of them to capstor), but
    # evaluate only the due ones — conversion is the durability step, eval
    # is the expensive one we sample at 1/N.
    to_convert = [it for it in saved if not hf_staged(cell, it, staging)]
    # Report what's still OUTSTANDING, not what's due: a due checkpoint
    # whose results are already on disk needs no action, and printing it
    # every pass reads as work the watcher is failing to submit.
    pending = [it for it in due
               if not evaluated(f"{cell}-iter{it}", logs_root, task_list)]
    # Checkpoints whose evals have only ever failed. A missing offline
    # dataset is repaired in place and the checkpoint goes straight back
    # into pending; anything else is recorded for a human and held back,
    # so the watcher stops burning a job per pass on it.
    broken: dict[int, str] = {}
    for it in list(pending):
        name = f"{cell}-iter{it}"
        if not args.max_attempts:
            continue
        if barren_attempts(name, logs_root) < args.max_attempts:
            continue
        kind, detail = eval_error(name)
        if kind == "dataset" and fix_missing_dataset(detail, args.dry_run):
            continue                     # cache repaired — retry this pass
        broken[it] = f"{kind}: {detail}"
        pending.remove(it)
        errors[name] = {"attempts": barren_attempts(name, logs_root),
                        "kind": kind, "detail": detail}
    # "DONE" only when nothing is outstanding for any reason — a cell whose
    # whole eval column is erroring has not finished, it has stopped.
    status = pending or ("-" if broken else f"DONE ({len(due)}/{len(due)})")
    print(f"{cell}: {len(saved)} saved | convert {to_convert or '-'} | "
          f"eval {status}"
          + (f" | ERRORS on {sorted(broken)}" if broken else ""))
    for reason in sorted(set(broken.values())):
        print(f"    {reason}")

    if to_convert and not convert_busy:
        submit_convert(cell, to_convert, staging, args.dry_run)
        # One conversion sbatch at a time (they share a Slurm name, so
        # dedupe is coarse); the next pass converts the remaining cells.
        convert_busy = True
    for it in pending:
        name = f"{cell}-iter{it}"
        if hf_staged(cell, it, staging) and job_name("eval", name) not in running:
            submit_eval(cell, it, staging, logs_root, tasks, c["size"],
                        args.dry_run)

    return convert_busy


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
    p.add_argument("--max-attempts", type=int, default=3,
                   help="after N eval runs that wrote no results at all, "
                        "diagnose instead of resubmitting: a dataset missing "
                        "from the offline cache is downloaded and retried "
                        "automatically, anything else is held back and "
                        "recorded in <logs-root>/" + ERRORS_JSON +
                        " (0 = always resubmit, never diagnose)")
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
