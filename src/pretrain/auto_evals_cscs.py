#!/usr/bin/env python3
"""
CSCS auto-eval watcher — the cluster twin of auto_evals_azure.py.

Every N saved checkpoints (default 2) plus each run's final checkpoint,
evaluate on the "auto" benchmark group (configs/tasks.json) and push to W&B
`mariagrandury-epflnlp/msnr` — the same project the training loss logs to.

Idempotent, safe to run alongside the trainings (login node, tmux):

    cd src/pretrain
    python3.11 auto_evals_cscs.py --watch 600     # one pass every 10 min
    python3.11 auto_evals_cscs.py --name lm-175M-L1-deep-seed1904 --max-submit 2
    python3.11 auto_evals_cscs.py --convert-only

The eval job pushes to W&B with the key from your environment or, as
everywhere else in the cluster pipeline, from the fallback file
src/evals/scripts/wandb_api_key.txt — nothing to export here.

Each pass covers EVERY variant — both architectures and both data schemes,
so the shallow ladder and the scheme-B cells cannot fall behind a watcher
someone forgot to start. --arch/--scheme narrow it. For each due checkpoint
of each cell:

  1. every task of the cell's list already has a result under
     LOGS_ROOT/<entity>/msnr/<cell>-iter<N>/                        -> skip
     (task-level, so adding a benchmark reaches evaluated checkpoints)
  2. eval job for it already in squeue (any user)                    -> skip
  2b. a task that failed in --max-attempts consecutive eval runs (or runs
     that wrote nothing at all) is diagnosed rather than resubmitted. A
     dataset missing from the offline cache is downloaded and the task
     retried immediately; every other cause is held back and recorded in
     <logs-root>/auto_eval_errors.json, so a task that cannot succeed stops
     costing a job per pass while the checkpoint's other tasks still run.
  3. HF snapshot staged at <staging>/<cell>/iter_<N>  -> submit ONE eval job
       (src/evals/scripts/evaluate.sbatch, vLLM, BOS, no chat template,
        TP=1 — the ladder's KV-head counts only divide 1 — so one worker per
        GPU, each task's results written the moment it finishes; a walltime
        kill keeps them and the next pass resubmits only what is missing)
  4. otherwise -> convert first: one conversion/convert-snr.sh --models job
       per cell with the missing iters (models.json-driven). Each job is
       named convert-snr-<cell> and sized to its checkpoint count, so cells
       convert in PARALLEL and each dedupes only against itself.

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

def convert_job_name(cell: str) -> str:
    """The Slurm name convert-snr.sh gives a single-cell --models submission.
    Keep the two in step: this string is the only dedupe against submitting a
    second conversion for a cell that already has one in flight."""
    return f"convert-snr-{cell}"
# The auto group spans 9 tasks (L=1) to 290 (L=100), so a fixed walltime can't
# fit both. The ladder's KV-head counts force TP=1, so evaluate.sbatch runs
# EVAL_WORKERS independent workers per job — one per GPU of the node, each
# with its own model copy — sharing the task queue (../evals/scripts/
# _run_per_task.sh); the estimate below divides the per-task term by that.
#
# Elapsed time is very close to linear in the task count. Fitted on 69
# completed eval jobs (2026-08-21..27) of the previous single-process
# pipeline, median elapsed per (size, n_tasks):
#
#     size   9 tasks  18 tasks  60 tasks  100  164  219    per task
#     90M      7.9       13.9         -    -    -    -     0.667 min
#     175M     8.3       13.9         -    -    -    -     0.622 min
#     350M     9.1       15.3      47.4    -    -    -     0.751 min
#     600M     9.2       16.1      48.7  89.8  145  187    0.846 min
#
# 600M is measured on 60 jobs out to 219 tasks (2026-09-02) and lands at 0.846,
# in the same 0.62-0.85 band as every smaller rung: over a 6.7x size range the
# per-task cost barely moves, because the run is dominated by dataset load and
# tokenization rather than the forward pass. 1B and 1.7B still have no eval
# run, so they keep conservative estimates rather than an extrapolation dressed
# up as a measurement. The per-task figures are per WORKER; re-fit them on the
# first jobs of the worker-pool pipeline (job.json in each eval dir has the
# elapsed time and the task counts).
#
# Every finished task is on disk before the next starts, so a walltime kill
# costs only the tasks in flight and the next pass resubmits the rest: the
# cap is a resume point, not a loss. SAFETY and the generous fixed overhead
# (container pull and vLLM cold start are much slower on a loaded node) only
# buy fewer resubmissions.
MIN_PER_TASK = {"90M": 0.67, "175M": 0.62, "350M": 0.75, "600M": 0.85,  # measured
                "1B": 2.0, "1.7B": 2.8}                                # not measured
OVERHEAD_MIN = 15   # measured 2-2.7; the rest is cold-start headroom
SAFETY = 1.5        # on the per-task term only
EVAL_WORKERS = 4    # one per GPU at TP=PP=1; passed to evaluate.sbatch


WALLTIME_CAP_MIN = 719   # the normal queue's 11:59:59 (launch_trainings.TIME_MAX_SEC)


def eval_minutes(size: str, n_tasks: int) -> int:
    """Fixed overhead + per-task budget x SAFETY over EVAL_WORKERS, rounded
    up to 15 min."""
    per_worker = math.ceil(n_tasks / EVAL_WORKERS)
    minutes = OVERHEAD_MIN + per_worker * MIN_PER_TASK.get(size, 2.8) * SAFETY
    return math.ceil(minutes / 15) * 15


def eval_walltime(size: str, n_tasks: int) -> str:
    """Slurm walltime, capped at the queue limit: an over-cap request would be
    rejected at submission and crash the watch loop, while a capped job
    simply resumes on the next pass with the tasks it did not reach."""
    minutes = min(eval_minutes(size, n_tasks), WALLTIME_CAP_MIN)
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


def remaining_tasks(name: str, logs_root: Path, tasks: list[str]) -> list[str]:
    """The tasks in TASKS without a result for NAME yet (empty = evaluated).

    Task-level, not checkpoint-level: `any results_*.json` would mean that
    adding a benchmark to the `auto` group never reaches checkpoints already
    evaluated on the old list — the sweep would silently carry two different
    task sets. The inner runner is already per-task idempotent
    (_run_per_task.sh filters through the same _eval_status.completed_tasks),
    so a resubmitted job runs ONLY the new tasks and merges them in.
    """
    done = completed_tasks(name, WANDB_ENTITY, PROJECT_NAME, str(logs_root))
    return [t for t in tasks if t not in done]


def eval_runs(name: str, logs_root: Path) -> list[Path]:
    """NAME's eval_*/ dirs, newest first."""
    base = logs_root / WANDB_ENTITY / PROJECT_NAME / name / "harness"
    return [d for d in sorted(base.glob("eval_*"), reverse=True) if d.is_dir()]


def wrote_results(run: Path) -> bool:
    """A run that saved anything: a results file, or a published per-task dir."""
    return (any(run.glob("results_*.json"))
            or any(f for t in run.glob("per_task/*") if t.is_dir()
                   for f in t.iterdir()))


def failed_in(run: Path) -> dict[str, str]:
    """task -> reason from the run's failed_tasks.log (eval_worker.py writes
    one `<task>\\t<Error>: <message>` line per failure)."""
    try:
        lines = (run / "failed_tasks.log").read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        if line.strip():
            task, _, reason = line.partition("\t")
            out[task] = reason
    return out


def task_attempts(name: str, logs_root: Path,
                  tasks: list[str]) -> dict[str, tuple[int, str]]:
    """Per task, how many of NAME's most recent eval runs IN A ROW failed it,
    with the first reason recorded: task -> (attempts, reason).

    The task gate asks "is this done?", never "can this ever finish?", so a
    task whose eval fails outright would be resubmitted once per pass forever
    — 196 such jobs on the L50 cells before this existed. Per task rather
    than per run because the workers isolate failures: one task with a
    broken dataset no longer takes the others down with it.

    Walking newest-first, a run counts against a task when it lists it in
    failed_tasks.log, or when it wrote nothing at all (a crash before any
    task could land, reason unknown — eval_error() reads the job log for
    those). A run that made progress without failing the task is no evidence
    either way and ends its streak, so a task that failed, was repaired, and
    has been running since is not held back by its past.
    """
    streak = {t: 0 for t in tasks}
    reason: dict[str, str] = {}
    open_ = set(tasks)
    for run in eval_runs(name, logs_root):
        failed, barren = failed_in(run), not wrote_results(run)
        for t in list(open_):
            if t in failed:
                streak[t] += 1
                reason.setdefault(t, failed[t])
            elif barren:
                streak[t] += 1
            else:
                open_.discard(t)
        if not open_:
            break
    return {t: (n, reason.get(t, "")) for t, n in streak.items() if n}


# lm_eval instantiates every task in the batch up front, so ONE dataset that
# isn't in the offline cache aborts the entire call — the compute nodes have
# no internet. The watcher runs on the login node, which does, so this is the
# one failure it can repair itself.
MISSING_DATASET_RE = re.compile(
    r"Couldn't reach '([^']+)' on the Hub \(OfflineModeIsEnabled\)")
ERROR_LINE_RE = re.compile(r"^(?:\d+: )?(\w*(?:Error|Exception)): (.+)$", re.M)
_LOG_TAIL = 400_000     # bytes; the traceback is at the end of a ~14 MB log


def classify(reason: str) -> tuple[str, str]:
    """A failed_tasks.log reason -> (kind, detail), the kinds of eval_error()."""
    hit = MISSING_DATASET_RE.search(reason)
    return ("dataset", hit.group(1)) if hit else ("other", reason[:200])


def eval_error(name: str) -> tuple[str, str]:
    """Why NAME's most recent eval job wrote nothing — for runs that died
    before any task could be recorded in failed_tasks.log.

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


_DATASET_FIXED: dict[str, bool] = {}


def fix_missing_dataset(repo: str, dry_run: bool) -> bool:
    """Add REPO to the eval-dataset manifest and build it into the offline
    cache, so the next submission gets past it. Built once per repo per
    process, memoising the RESULT: many checkpoints block on the same repo,
    and after the first one repairs the cache the rest must be retried too,
    not held back because the repo was "already tried". Only a failed build
    parks its checkpoints in the errors file."""
    if repo in _DATASET_FIXED:
        return _DATASET_FIXED[repo]
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
    ok = _DATASET_FIXED[repo] = (out.returncode == 0 and "0 failed" in out.stdout)
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
                task_list: list[str], size: str, dry_run: bool,
                exclude: set[str] = frozenset()) -> None:
    name = f"{cell}-iter{it}"
    hf_dir = staging / cell / f"iter_{it:07d}"
    # Size the request on what is LEFT to run, not the full list — the inner
    # runner skips completed tasks anyway (debug_loop.sh's narrowing trick),
    # and pricing a 3-tasks-missing checkpoint at the full 463 would oversize
    # the walltime. Held-back tasks (`exclude`) leave the list entirely.
    remaining = [t for t in remaining_tasks(name, logs_root, task_list)
                 if t not in exclude]
    tasks = ",".join(remaining)
    n_tasks = len(remaining)
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
           "EVAL_WORKERS": str(EVAL_WORKERS),
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
    added, updated = [], []
    for arch in args.archs:
        for scheme in args.schemes:
            a, u = sync(arch, scheme)
            added += a
            updated += u
    if added or updated:
        print(f"(models.json synced: +{len(added)} ~{len(updated)} cells "
              f"— commit the diff)")

    running = active_jobs() if not args.dry_run else set()
    errors: dict[str, dict] = {}   # checkpoints held back, written out below
    submitted = {"evals": 0}       # against --max-submit, across all cells

    # EVERY variant in one pass, not one watcher per arch. A watcher covering
    # a single --arch/--scheme means the shallow ladder and the scheme-B cells
    # only progress while someone remembers to run their own watcher, and they
    # fall behind silently — the checkpoints pile up, nothing complains.
    # Cells are deduped by name because scheme B collapses onto A wherever the
    # two language sets agree (SCHEME_B_LANGS), so both schemes name the same
    # cell at those settings.
    seen: set[str] = set()
    for arch in args.archs:
        configs = json.loads(HYPERPARAMS[arch].read_text())["configs"]
        for scheme_arg in args.schemes:
            for c in predictivity_cells():
                scheme = scheme_arg if c["L"] in SCHEME_B_LANGS else "A"
                cell = exp_name(c["size"], c["L"], arch, c["seed"], scheme)
                if args.name and cell != args.name:
                    continue
                if cell in seen:
                    continue
                seen.add(cell)
                # capstor intermittently faults a read outright (Errno 5 /
                # 108 — the same blips data_progress.py works around, hit
                # here on a .hf_complete probe). The watcher runs unattended
                # behind every launch, so one blip must cost one cell for one
                # pass, not kill the whole loop.
                try:
                    one_cell(args, c, cell, scheme, configs, root, staging,
                             logs_root, benchmarks, running, errors, submitted)
                except OSError as e:
                    print(f"{cell}: skipped this pass — {e.strerror or e}",
                          file=sys.stderr)

    # One place to look for what is stuck and why. A snapshot, not a log: a
    # checkpoint drops out of it as soon as an eval writes results, so an
    # empty file means nothing is held back.
    # The `normal` queue is the bottleneck, and the convert jobs this pass just
    # submitted are the gate on every eval downstream of them. --ensure starts a
    # drainer only if none is running, so this is safe to call every pass; the
    # drainer exits on its own once nothing movable is left.
    if not args.dry_run:
        subprocess.run(["bash", str(EVALS_DIR / "scripts" / "debug_drain.sh"),
                        "--ensure"], check=False)

    path = logs_root / ERRORS_JSON
    if not args.dry_run:
        path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
    if errors:
        print(f"\n{len(errors)} checkpoint(s) with tasks held back after "
              f"{args.max_attempts} failed evals — "
              + ("(dry-run: not written)" if args.dry_run else f"details in {path}"))


def one_cell(args, c: dict, cell: str, scheme: str, configs: dict, root: Path,
             staging: Path, logs_root: Path, benchmarks: list[str],
             running: set[str], errors: dict, submitted: dict) -> None:
    """One cell of a pass: convert what is missing, evaluate what is due."""
    target = schedule_for(configs[c["size"]])[0]
    saved = saved_valid_iters(cell, root)
    if not saved:
        return
    # Every Nth saved checkpoint on the cell's per-size save grid, plus
    # the run's final one whatever its number — same rule as Azure.
    due = [i for i in saved
           if i % (args.every * save_interval(target)) == 0 or i == target]
    # The cell's task list: every auto benchmark, in the languages
    # this cell trains on (e.g. L2 -> hellaswag + hellaswag_ru + ...).
    task_list = tasks_for_benchmarks(benchmarks, cell_languages(c["L"], scheme))
    # Convert EVERY saved checkpoint (persist all of them to capstor), but
    # evaluate only the due ones — conversion is the durability step, eval
    # is the expensive one we sample at 1/N.
    to_convert = [it for it in saved if not hf_staged(cell, it, staging)]
    # Report what's still OUTSTANDING, not what's due: a due checkpoint
    # whose results are already on disk needs no action, and printing it
    # every pass reads as work the watcher is failing to submit.
    pending = [it for it in due
               if remaining_tasks(f"{cell}-iter{it}", logs_root, task_list)]
    # Tasks whose evals have only ever failed. A missing offline dataset is
    # repaired in place and the task retried this pass; anything else is
    # recorded for a human and held back, so the watcher stops burning a job
    # per pass on it — while the checkpoint's other tasks still run.
    held: dict[int, dict[str, dict]] = {}   # iter -> task -> attempts/kind/detail
    for it in list(pending):
        name = f"{cell}-iter{it}"
        # An in-flight job's freshly-mkdir'd eval_* dir looks barren until it
        # writes something — don't let the running attempt itself push the
        # checkpoint over the threshold.
        if not args.max_attempts or job_name("eval", name) in running:
            continue
        remaining = remaining_tasks(name, logs_root, task_list)
        for t, (n, why) in task_attempts(name, logs_root, remaining).items():
            if n < args.max_attempts:
                continue
            kind, detail = classify(why) if why else eval_error(name)
            if kind == "dataset" and fix_missing_dataset(detail, args.dry_run):
                continue                 # cache repaired — retry this pass
            held.setdefault(it, {})[t] = {"attempts": n, "kind": kind,
                                          "detail": detail}
        if it in held:
            errors[name] = held[it]
            if len(held[it]) == len(remaining):
                pending.remove(it)       # nothing submittable is left
    # "DONE" only when nothing is outstanding for any reason — a cell whose
    # whole eval column is erroring has not finished, it has stopped.
    status = pending or ("-" if held else f"DONE ({len(due)}/{len(due)})")
    print(f"{cell}: {len(saved)} saved | convert {to_convert or '-'} | "
          f"eval {status}"
          + (f" | HELD BACK on {sorted(held)}" if held else ""))
    for it, tasks in sorted(held.items()):
        reasons = sorted({f"{v['kind']}: {v['detail']}" for v in tasks.values()})
        print(f"    iter {it}: {len(tasks)} task(s) held back — "
              + "; ".join(reasons[:3])
              + (f" (+{len(reasons) - 3} more)" if len(reasons) > 3 else ""))

    # Conversions run one per cell, in parallel across cells: they gate every
    # eval downstream, so serializing them cluster-wide (one shared job name,
    # which is what this used to do) throttled the whole pipeline to a single
    # cell at a time. The per-cell name is the dedupe.
    if to_convert and convert_job_name(cell) not in running:
        submit_convert(cell, to_convert, staging, args.dry_run)
    if args.convert_only:
        return
    for it in pending:
        if args.max_submit is not None and submitted["evals"] >= args.max_submit:
            return
        name = f"{cell}-iter{it}"
        if hf_staged(cell, it, staging) and job_name("eval", name) not in running:
            submit_eval(cell, it, staging, logs_root, task_list, c["size"],
                        args.dry_run, exclude=set(held.get(it, {})))
            submitted["evals"] += 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Default: EVERY variant. One watcher covers the whole grid, so the
    # shallow ladder and the scheme-B cells cannot quietly fall behind while
    # a deep/A-only watcher runs. The flags narrow it for a targeted pass.
    p.add_argument("--arch", choices=["deep", "shallow"], default=None,
                   help="only this architecture (default: both)")
    p.add_argument("--scheme", choices=["A", "B"], default=None,
                   help="only this data scheme (default: both)")
    p.add_argument("--max-submit", type=int, metavar="N",
                   help="submit at most N eval jobs this pass — a throttle for "
                        "the burst an expanded task list creates, and what "
                        "makes a single-job integration test possible")
    p.add_argument("--name", help="watch a single cell (its full name)")
    p.add_argument("--every", type=int, default=2,
                   help="evaluate every N saved checkpoints (the final "
                        "checkpoint is always evaluated on top)")
    p.add_argument("--convert-only", action="store_true",
                   help="submit conversions but no eval jobs — for driving the "
                        "convert half forward while the eval half is blocked "
                        "(a broken task list, a missing dataset). Conversion "
                        "is what every eval waits on, so it is always worth "
                        "keeping ahead.")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="after N eval runs in a row that failed a task (or "
                        "wrote nothing at all), diagnose it instead of "
                        "resubmitting: a dataset missing from the offline "
                        "cache is downloaded and the task retried "
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
    # The pass iterates over these; a flag narrows the default "everything".
    args.archs = [args.arch] if args.arch else list(HYPERPARAMS)
    args.schemes = [args.scheme] if args.scheme else ["A", "B"]

    benchmarks = auto_benchmarks()
    while True:
        one_pass(args, Path(args.root), Path(args.staging),
                 Path(args.logs_root), benchmarks)
        # eval_progress.png is a view of exactly the state this pass just
        # changed, so refresh it here rather than on launch (the launcher only
        # redraws the training-side figures). Best-effort: a plotting problem
        # must never stop the watch loop.
        if not args.dry_run:
            try:
                from pretrain_progress import eval_progress
                eval_progress(root=Path(args.root), logs_root=Path(args.logs_root))
            except Exception as e:
                print(f"(eval progress plot not refreshed: {e})", file=sys.stderr)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
