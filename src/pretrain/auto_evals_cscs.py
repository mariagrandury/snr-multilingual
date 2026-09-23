#!/usr/bin/env python3
"""
CSCS auto-eval watcher — the cluster twin of auto_evals_azure.py.

Twelve checkpoints per run — the ten tenths of training plus 85 % and 95 %,
which is exactly the grid the analysis reads — read on the grid the run
actually saved at, so every size lands on the same fractions of training
whether it saved 20, 40 or 60 times (launch_trainings.due_iters). Evaluate
them on the "auto" benchmark group (configs/tasks.json) and push to W&B
`mariagrandury-epflnlp/msnr` — the same project the training loss logs to.

Idempotent, safe to run alongside the trainings (login node, tmux):

    cd src
    python3.11 pretrain/auto_evals_cscs.py --dry-run
    python3.11 pretrain/auto_evals_cscs.py --watch 600   # one pass every 10 min
    python3.11 pretrain/auto_evals_cscs.py --retry-held
    python3.11 pretrain/auto_evals_cscs.py --name lm-175M-L1-deep-seed1904 --max-submit 2
    python3.11 pretrain/auto_evals_cscs.py --convert-only
    python3.11 pretrain/auto_evals_cscs.py --arch deep --scheme A --seed 1904 --all-languages

The eval job pushes to W&B with the key from your environment or, as
everywhere else in the cluster pipeline, from the fallback file
src/evals/scripts/wandb_api_key.txt — nothing to export here.

Each pass covers EVERY variant — both architectures and every data scheme,
so the shallow ladder and the non-baseline mixtures cannot fall behind a
watcher someone forgot to start. --arch/--scheme narrow it. For each due
checkpoint
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
     A partial hold heals itself — the next run that progresses without
     failing a held task ends its streak — but a checkpoint whose tasks are
     ALL held is never submitted again, so nothing can ever reset it.
     `--retry-held` is the way back after fixing the cause.
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
    DATA_SCHEMES, EVAL_SIZES, HYPERPARAMS, LADDER, TOKENIZER_MODEL, cell_languages,
    arches_for, due_iters, exp_name, job_name, predictivity_cells, schedule_for)
from pretrain_progress import CKPT_ROOT, ITER_RE, is_valid_iter_dir  # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR.parent))
from evals.scripts.utils.configs import load_tasks, tasks_for_benchmarks  # noqa: E402
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
DEFAULT_STAGING = "/capstor/store/cscs/swissai/infra01/msnr/msnr-hf-models"
DEFAULT_LOGS_ROOT = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"

def convert_job_name(cell: str) -> str:
    """The Slurm name convert-snr.sh gives a single-cell --models submission.
    Keep the two in step: this string is the only dedupe against submitting a
    second conversion for a cell that already has one in flight."""
    return f"convert-snr-{cell}"
# The auto group spans 13 tasks (L=1) to 329 (L=50), so a fixed walltime can't
# fit both. The ladder's KV-head counts force TP=1, so evaluate.sbatch runs
# EVAL_WORKERS independent workers per job — one per GPU of the node, each
# with its own model copy — sharing the task queue (../evals/scripts/
# _run_per_task.sh); the estimate below divides the per-task term by that.
#
# Elapsed time is very close to linear in the tasks each worker runs.
# Re-fitted 2026-09-10 by least squares (elapsed = a + b x ceil(tasks /
# EVAL_WORKERS)) over the 583 COMPLETED worker-pool jobs on disk (job.json
# present, sacct elapsed), 4 to 329 tasks per job
# (../evals/scripts/eval_timing.py is where the per-job rows come from):
#
#     size    jobs   task range   overhead a   per worker-task b   R^2   1 process
#     90M       66       4-233        0.3 min          0.387 min    0.77     0.747
#     175M     146       4-329        1.6              0.479        0.97     0.717
#     350M     108       5-239        0.3              0.537        0.82     0.835
#     600M     163       4-329        1.4              0.554        0.94     0.868
#     1B        40      14-329       -2.0              0.594        0.92     0.902
#     1.7B      60       4-239        1.9              0.679        0.98     0.968
#
# The last column is the 2026-09-07 fit over 364 single-process jobs. Four
# workers buy 1.4-1.9x, not 4x: a run is dominated by dataset load and
# tokenization, CPU and IO the workers share. So MIN_PER_TASK is b itself —
# the measured cost of one task in one worker's queue — not a single-process
# cost divided by the worker count, which is what the constants used to be.
#
# OVERHEAD_MIN and SAFETY come from coverage. The largest fitted intercept is
# 1.9 min, so 10 min is mostly cold-start headroom; at 10 min the worst of the
# 583 jobs needed SAFETY 0.91, so 1.15 is a 26% margin. Replayed over 594
# clean jobs, these constants with 5-min rounding request 269 node-hours
# against 526 for the old ones (135 actually burned), and undersize none.
#
# Every finished task is on disk before the next starts, so a walltime kill
# costs only the tasks in flight and the next pass resubmits the rest: the cap
# is a resume point, not a loss. SAFETY and the fixed overhead only buy fewer
# resubmissions.
MIN_PER_TASK = {"90M": 0.39, "175M": 0.48, "350M": 0.54,   # per worker-task, 583 jobs
                "600M": 0.55, "1B": 0.59, "1.7B": 0.68,
                "3B": 0.85}   # not fitted: 1.7B x 1.25, deliberately above
                              # the measured 1B->1.7B step (0.59->0.68 = 1.15)
                              # because nothing at 3B has been timed yet
OVERHEAD_MIN = 10   # max fitted intercept 1.9; the rest is cold-start headroom
SAFETY = 1.15       # worst observed requirement 0.91 -> 26% margin
# Must match what evaluate.sbatch derives (GPUS_PER_NODE / (TP x PP), forced
# to 1 off the vLLM backend). They agree only because submit_eval below pins
# TP=PP=1 and vllm; change either and this estimate is silently 4x too small.
EVAL_WORKERS = 4


WALLTIME_CAP_MIN = 719   # the normal queue's 11:59:59 (launch_trainings.TIME_MAX_SEC)


def eval_minutes(size: str, n_tasks: int) -> int:
    """Fixed overhead + each worker's share of the tasks x MIN_PER_TASK x
    SAFETY, rounded up to 5 min.

    MIN_PER_TASK is fitted per worker-task, so the worker pool's sub-linear
    speedup is already inside it. Re-fit when the worker count, the backend or
    the task mix changes — the constants are only as good as the jobs they
    were fitted on."""
    per_worker = math.ceil(n_tasks / EVAL_WORKERS)
    minutes = OVERHEAD_MIN + per_worker * MIN_PER_TASK.get(size, 2.8) * SAFETY
    return math.ceil(minutes / 5) * 5


def eval_walltime(size: str, n_tasks: int) -> str:
    """Slurm walltime, capped at the queue limit: an over-cap request would be
    rejected at submission and crash the watch loop, while a capped job
    simply resumes on the next pass with the tasks it did not reach."""
    minutes = min(eval_minutes(size, n_tasks), WALLTIME_CAP_MIN)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def auto_benchmarks(group: str = "auto") -> list[str]:
    """The `auto` group in configs/tasks.json — BENCHMARK names; each cell
    is evaluated on every benchmark's tasks in the languages it trains on
    (tasks_for_benchmarks x cell_languages). `auto_rf` / `auto_rfgm` are the
    reformulated sets: the letter-format families rewritten as cloze tasks
    / as Gemini statements (../evals/scripts/make_rf_tasks.py), distinct task
    names, so all sets coexist on disk and in W&B."""
    groups = json.loads(TASKS_JSON.read_text())["groups"]
    if group not in groups:
        # `auto_rf` / `auto_rfgm` were retired on 2026-09-23: the twins live in
        # `auto`, the candidates in `auto_probe`, and nothing else is a group.
        raise SystemExit(f"no group {group!r} in configs/tasks.json; it has "
                         f"{', '.join(sorted(groups))}")
    return groups[group]


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


def active_jobs() -> set[str] | None:
    """All queued/running Slurm job names and job ids, ANY user — collaborators
    share the trees, so their in-flight converts/evals count as ours. None when
    squeue fails: an unreachable controller must not read as an empty queue,
    or the pass resubmits every eval and convert that is already running."""
    try:
        out = subprocess.run(["squeue", "-h", "--format=%j %i"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return set(out.stdout.split()) if out.returncode == 0 else None


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


def job_facts(run: Path) -> dict:
    """evaluate.sbatch's job.json for one run, {} when absent or unreadable.

    It carries the final counts only if the wrapper got to rewrite it, so a
    walltime-killed run has the header (status "started") and no counts —
    which is why every caller falls back to the filesystem rather than
    reading absence as zero.
    """
    try:
        return json.loads((run / "job.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def attempted_nothing(run: Path) -> bool:
    """True when this run finished and ran no task at all — every task it was
    given was already done elsewhere. Only job.json can tell that apart from a
    crash, since both leave an empty directory; a run without the final counts
    (killed, or from before job.json existed) is not claimed to be either."""
    j = job_facts(run)
    return (j.get("status") == "finished"
            and not j.get("tasks_done") and not j.get("tasks_failed"))


_JOB_STATE: dict[str, str] = {}


def interrupted(run: Path) -> bool:
    """True when this run's job was cancelled, preempted or lost its node before
    it saved anything. Its job.json still says `started` and the directory is
    empty, exactly what a crash leaves, but it says nothing about the tasks:
    counted as failures, two quick scancels plus one real error held back a
    whole checkpoint. sacct is asked only about such barren runs, once per job
    per pass."""
    if job_facts(run).get("status") != "started" or wrote_results(run):
        return False
    job = run.name.rsplit("_", 1)[-1]
    if job not in _JOB_STATE:
        try:
            out = subprocess.run(["sacct", "-j", job, "-X", "-n", "-o", "State"],
                                 capture_output=True, text=True, timeout=30)
            _JOB_STATE[job] = out.stdout.strip() if out.returncode == 0 else ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _JOB_STATE[job] = ""
    return _JOB_STATE[job].startswith(("CANCELLED", "PREEMPTED", "NODE_FAIL"))


def eval_runs(name: str, logs_root: Path) -> list[Path]:
    """NAME's eval_*/ dirs, newest first."""
    base = logs_root / WANDB_ENTITY / PROJECT_NAME / name / "harness"
    return [d for d in sorted(base.glob("eval_*"), reverse=True) if d.is_dir()]


def wrote_results(run: Path) -> bool:
    """A run that saved anything: a results file, or a published per-task dir.

    job.json's own count wins when the wrapper recorded it — it is the run's
    own report, and it stays right even after the eval tree is mirrored or
    tidied. Without it (a killed run), the files are the only evidence.
    """
    j = job_facts(run)
    if "tasks_done" in j:
        return bool(j["tasks_done"])
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

    A run that attempted NOTHING is skipped entirely — neither a strike nor a
    reset. evaluate.sbatch creates the eval dir (for job.json) before the
    idempotency filter runs, so a duplicate submission whose sibling already
    did the work leaves a directory that looks identical to a crash. Counting
    those would let two of them plus one real failure park a task at the
    default --max-attempts 3.
    """
    streak = {t: 0 for t in tasks}
    reason: dict[str, str] = {}
    open_ = set(tasks)
    for run in eval_runs(name, logs_root):
        # Neither a strike nor a reset: a no-op duplicate (above), or a job
        # someone cancelled before it saved anything (interrupted()).
        if attempted_nothing(run) or interrupted(run):
            continue
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


_EVAL_ERROR: dict[tuple[str, str], tuple[str, str]] = {}


def eval_error(name: str, logs_root: Path, suffix: str = "") -> tuple[str, str]:
    """Why NAME's most recent eval job wrote nothing — for runs that died
    before any task could be recorded in failed_tasks.log, and as the second
    opinion when a per-task reason is not self-explanatory.

    ("dataset", <hf repo>) — fixable here, see fix_missing_dataset.
    ("other", <last Error line>) / ("unknown", ...) — needs a human; recorded
    in the errors file rather than retried into the ground.

    Memoised for the pass: the answer is a property of NAME's job logs, and
    an L50 checkpoint asks it once per held-back task — up to ~290 scans of
    the same 400 KB log tails.

    The logs read are those of the runs being explained: NAME's two newest
    runs that count as attempts (task_attempts' rule), found by the job id
    their eval_* dir ends in, in THIS watcher's job family (SUFFIX, e.g. -rf).
    Until 2026-09-19 it read the two newest `eval-<cell>-iter<N>_*.err` by
    mtime instead — the plain family only, so for -rf jobs it read another
    family's logs, and a Sep 11 xstorycloze_gl line relabelled a week of
    lock-file PermissionErrors as that dataset missing (308 tasks held on 22
    checkpoints behind a "repair" of a dataset that was already cached).
    """
    key = (name, suffix)
    if key not in _EVAL_ERROR:
        _EVAL_ERROR[key] = _eval_error_uncached(name, logs_root, suffix)
    return _EVAL_ERROR[key]


def _eval_error_uncached(name: str, logs_root: Path, suffix: str) -> tuple[str, str]:
    job = job_name("eval", name) + suffix
    # The two newest attempts FIRST, then their logs: filtering on "has a log
    # in this family" before slicing would skip past a newer run of the other
    # family and reach back to a stale log — the very bug this replaced.
    runs = [r for r in eval_runs(name, logs_root)
            if not (attempted_nothing(r) or interrupted(r))][:2]
    logs = [EVAL_JOB_LOGS / f"{job}_{r.name.rsplit('_', 1)[-1]}.err" for r in runs]
    logs = [p for p in logs if p.exists() or p.with_suffix(".out").exists()]
    for log in logs:
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
    return "unknown", ("no error line in the job log" if logs
                       else "no eval job log found")


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
    # The manifest may pin a revision (`repo@commit`, see eval_datasets.txt):
    # build THAT entry, never the bare repo — a pin exists because the tip no
    # longer loads, and appending an unpinned duplicate would rebuild from it.
    entry = next((e for e in DATASET_MANIFEST.read_text().split()
                  if e.partition("@")[0] == repo), None)
    listed = entry is not None
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
        manifest.write(f"{entry or repo}\n")
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
                exclude: set[str] = frozenset(), suffix: str = "") -> None:
    name = f"{cell}-iter{it}"
    job = job_name("eval", name) + suffix
    hf_dir = staging / cell / f"iter_{it:07d}"
    # Size the request on what is LEFT to run, not the full list — the inner
    # runner skips completed tasks anyway (debug_loop.sh's narrowing trick),
    # and pricing a 3-tasks-missing checkpoint at the full 463 would oversize
    # the walltime. Held-back tasks (`exclude`) leave the list entirely.
    remaining = [t for t in remaining_tasks(name, logs_root, task_list)
                 if t not in exclude]
    tasks = ",".join(remaining)
    # The rf / rfgm Global-MMLU twins are one task over the whole 14k-row split
    # with four answer strings to score per item: 2.7-3.3 min per worker-task
    # on the 2026-09-18 pilots against the ~0.5 the fit assumes, so each
    # counts as six tasks in the walltime. `bbq` (auto_probe) is 58,492 items
    # x 12 choices = 702k requests, ~17 min at 600M's ~700 req/s plus ~4 min
    # of context building (job 3485222, 2026-09-23): 36 task-minutes that no
    # worker can share, so x EVAL_WORKERS, because eval_minutes divides the
    # count across the workers. Without this the solo top-up job the watcher
    # gives it is 15 min, the kill is not a recorded failure, and it is
    # resubmitted every pass forever.
    TASK_WEIGHT = {"bbq": 36 * EVAL_WORKERS}
    n_tasks = sum(TASK_WEIGHT.get(t, 6 if t.startswith(("rf_global_mmlu_full", "rfgm_global_mmlu_full")) else 1)
                  for t in remaining)
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
           "HARNESS_INCLUDE_PATH": str(EVALS_DIR / "tasks"),   # the rf_* / rfgm_* YAMLs
           "TASKS": tasks}
    cmd = ["sbatch", f"--job-name={job}",
           f"--time={eval_walltime(size, n_tasks)}",
           "--export=ALL", "scripts/evaluate.sbatch", str(hf_dir), name]
    print(f"  submit: {job}")
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


def merge_unmerged(name: str, logs_root: Path, running: set[str],
                   dry_run: bool) -> None:
    """Fold a run's per_task/<task>/ results into one results_*.json when the
    run never did so itself: a walltime-killed job dies before
    _run_per_task.sh reaches its merge step, leaving its results only under
    per_task/. The done-test and every reader cope with that layout, but the
    merged file is what downstream tools and people look for. Same call the
    job makes; a run whose job is still in the queue is left to do it."""
    for run in eval_runs(name, logs_root):
        if run.name.rsplit("_", 1)[-1] in running or any(run.glob("results_*.json")):
            continue
        splits = sorted(d for d in run.glob("per_task/*") if d.is_dir())
        if not splits:
            continue
        print(f"  merge: {len(splits)} per-task result(s) into {name}/{run.name}")
        if dry_run:
            continue
        out = subprocess.run(
            [sys.executable, "-m", "scripts.alignment.merge_split_results",
             "--split_dirs", *map(str, splits), "--output_dir", str(run),
             "--move-samples"], cwd=EVALS_DIR, capture_output=True, text=True)
        if out.returncode:
            print(f"    merge failed: {(out.stderr or out.stdout).strip()[-300:]}",
                  file=sys.stderr)


# The shared offline dataset cache (evaluate.sbatch hard-sets HF_HOME to it for
# every user). `datasets` takes a FileLock next to each dataset it loads and
# creates the lock file 0644, so the directory's default ACL (rwx for the
# named collaborators) is masked down to r-- on it — and the OTHER user's
# jobs then die on that dataset with PermissionError at the lock, every task,
# every job (2026-09-15..18: 22k of aromanou's task attempts failed on locks
# I created, 800 of mine on hers). eval_worker.py now evaluates under umask
# 002, so new locks are 0664 (mask rw-); this pass repairs the ones that are
# still closed. A lock file's mask is its owner's to raise, so each watcher
# widens the locks its user owns; the other user's watcher does the same.
DATASETS_CACHE = Path("/iopsstor/scratch/cscs/mariagrandury/hf_home/datasets")


def share_dataset_locks() -> None:
    # Create the lock of every cached config up front (the path is the
    # config dir with "/" -> "_"): a lock that already exists with an open
    # mask is what everyone's jobs then take, and nobody creates a closed one.
    for cfg in DATASETS_CACHE.glob("*/*/*/*"):     # <ns>___<name>/<config>/<version>/<hash>
        lock = DATASETS_CACHE / (str(cfg).replace("/", "_") + ".lock")
        if cfg.is_dir() and not lock.exists():
            lock.touch()
    mine = [p for p in DATASETS_CACHE.glob("*.lock")
            if (st := p.stat()).st_uid == os.getuid() and st.st_mode & 0o070 != 0o070]
    if mine:
        subprocess.run(["setfacl", "-m", "m::rwx", *map(str, mine)], check=False)
        print(f"({len(mine)} dataset lock file(s) opened to the collaborators)")


def one_pass(args, root: Path, staging: Path, logs_root: Path,
             benchmarks: list[str]) -> None:
    # The memos live for ONE pass. Under --watch a single process runs every
    # pass, and a diagnosis cached from an old job log kept answering for good:
    # a task misread as a missing dataset was retried every pass, never held.
    for memo in (_EVAL_ERROR, _DATASET_FIXED, _JOB_STATE):
        memo.clear()
    if not args.dry_run:
        share_dataset_locks()
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
    if running is None:
        # Without the queue there is no dedupe: every running eval and convert
        # would be submitted again, and their fresh, empty eval dirs counted as
        # strikes. Skip the pass; the errors file keeps its last snapshot.
        print("squeue failed — skipping this pass, nothing submitted", file=sys.stderr)
        return
    errors: dict[str, dict] = {}   # checkpoints held back, written out below
    submitted = {"evals": 0}       # against --max-submit, across all cells

    # EVERY variant in one pass, not one watcher per arch. A watcher covering
    # a single --arch/--scheme means the shallow ladder and the non-baseline
    # data schemes only progress while someone remembers to run their own
    # watcher, and they fall behind silently — the checkpoints pile up and
    # nothing complains. Each scheme now names a distinct cell, so unlike the
    # old two-scheme loop there are no duplicates to dedupe.
    for arch in args.archs:
        configs = json.loads(HYPERPARAMS[arch].read_text())["configs"]
        for c in predictivity_cells(args.schemes, arch):
            scheme = c["scheme"]
            if arch not in arches_for(scheme, c["size"], c["L"]):
                continue
            cell = exp_name(c["size"], c["L"], arch, c["seed"], scheme)
            if args.name:
                if cell != args.name:
                    continue
            elif c["size"] not in args.sizes or (args.seed and c["seed"] != args.seed):
                continue
            # capstor intermittently faults a read outright (Errno 5 / 108 —
            # the same blips data_progress.py works around, hit here on a
            # .hf_complete probe), and sbatch or convert-snr.sh can fail on a
            # controller timeout. The watcher runs unattended behind every
            # launch, so one blip must cost one cell for one pass, not kill
            # the whole loop.
            try:
                one_cell(args, {**c, "arch": arch}, cell, scheme, configs, root, staging,
                         logs_root, benchmarks, running, errors, submitted)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"{cell}: skipped this pass — {getattr(e, 'strerror', None) or e}",
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


# (scheme, arch, seed) of the runs evaluated in EVERY language, flag or not:
# one full size x L ladder showing how each benchmark behaves in languages
# the model never trained on (eval_progress_all_languages.png).
ALL_LANGUAGES_RUNS = ("A", "deep", 1904)


def eval_languages(L: int, scheme: str, all_languages: bool = False):
    """The languages a cell is evaluated in: the ones it trains on, or every
    language any task is tagged with ("multi"/"??" stay out, as
    tasks_for_benchmarks promises)."""
    if all_languages:
        return {e.get("language") for e in load_tasks().values()} - {"multi", "??"}
    return cell_languages(L, scheme)


def one_cell(args, c: dict, cell: str, scheme: str, configs: dict, root: Path,
             staging: Path, logs_root: Path, benchmarks: list[str],
             running: set[str], errors: dict, submitted: dict) -> None:
    """One cell of a pass: convert what is missing, evaluate what is due."""
    target = schedule_for(configs[c["size"]])[0]
    saved = saved_valid_iters(cell, root)
    if not saved:
        return
    # The ten tenths of training, read on the grid the run actually saved at
    # (a 20-save run yields every 2nd save, a 40-save one every 4th, a
    # 60-save one every 6th — the same fractions), plus the 85 % / 95 %
    # noise points and its final one — same rule as Azure. 12 per run.
    due = due_iters(saved, target, args.every)
    if args.final_only:
        # A screening pass over a new benchmark group: the gate and the signal
        # at each size, without the noise window. `--every` cannot express it
        # (its noise-window clause ignores `every`, so the floor is 5) and
        # --max-submit takes the EARLIEST outstanding checkpoint, not the last.
        due = due[-1:]
    # The cell's task list: every auto benchmark, in the languages this cell
    # trains on (e.g. L2 -> hellaswag + hellaswag_ru + ...), or in all of them
    # under --all-languages and for the ALL_LANGUAGES_RUNS.
    langs = eval_languages(c["L"], scheme, args.all_languages
                           or (scheme, c["arch"], c["seed"]) == ALL_LANGUAGES_RUNS)
    task_list = tasks_for_benchmarks(benchmarks, langs)
    # Convert EVERY saved checkpoint (persist all of them to capstor), but
    # evaluate only the due ones — conversion is the durability step, eval
    # is the expensive one we sample at 1/N.
    to_convert = [it for it in saved if not hf_staged(cell, it, staging)]
    # A killed job's per-task results, merged the way its own last step would
    # have. Nothing is re-run: the done-test already counts them.
    for it in due:
        merge_unmerged(f"{cell}-iter{it}", logs_root, running, args.dry_run)
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
        if not args.max_attempts or args.retry_held \
                or job_name("eval", name) + args.job_suffix in running:
            continue
        remaining = remaining_tasks(name, logs_root, task_list)
        for t, (n, why) in task_attempts(name, logs_root, remaining).items():
            if n < args.max_attempts:
                continue
            kind, detail = (classify(why) if why
                            else eval_error(name, logs_root, args.job_suffix))
            if kind == "other":
                # The per-task reason is one truncated line; the job log may
                # still name a missing dataset, and that is the one cause
                # this watcher can repair itself. Memoised, so at most one
                # log scan per checkpoint per pass.
                log_kind, log_detail = eval_error(name, logs_root, args.job_suffix)
                if log_kind == "dataset":
                    kind, detail = log_kind, log_detail
            # A repair is trusted for a bounded number of runs: a task whose
            # "missing dataset" was built and that still fails is failing for
            # another reason (the log can name ANOTHER task's dataset, or the
            # build can land in a cache the job never reads), so past twice
            # the threshold it is held like any other failure.
            if (kind == "dataset" and n < 2 * args.max_attempts
                    and fix_missing_dataset(detail, args.dry_run)):
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
        if hf_staged(cell, it, staging) \
                and job_name("eval", name) + args.job_suffix not in running:
            submit_eval(cell, it, staging, logs_root, task_list, c["size"],
                        args.dry_run, exclude=set(held.get(it, {})),
                        suffix=args.job_suffix)
            submitted["evals"] += 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Default: every arch and every scheme. One watcher covers the whole
    # grid, so the shallow ladder and the non-A schemes cannot quietly fall
    # behind while a deep/A-only watcher runs. The flags narrow it for a
    # targeted pass.
    p.add_argument("--arch", choices=["deep", "shallow"], default=None,
                   help="only this architecture (default: both)")
    p.add_argument("--scheme", choices=list(DATA_SCHEMES), default=None,
                   help="only this data scheme (default: all of them)")
    p.add_argument("--max-submit", type=int, metavar="N",
                   help="submit at most N eval jobs this pass — a throttle for "
                        "the burst an expanded task list creates, and what "
                        "makes a single-job integration test possible")
    p.add_argument("--name", help="watch a single cell (its full name)")
    p.add_argument("--seed", type=int, help="only cells with this seed")
    p.add_argument("--size", metavar="SIZES",
                   help="only these sizes, comma-separated (e.g. '600M' or "
                        f"'1B,1.7B'); default {','.join(EVAL_SIZES)} — 90M is "
                        "off the ladder and not evaluated unless named")
    p.add_argument("--all-languages", action="store_true",
                   help="evaluate every auto benchmark in every language, not "
                        "only the languages the cell trains on")
    p.add_argument("--reformulated", nargs="?", const="rf", choices=["rf", "rfgm"],
                   help="evaluate a reformulated group instead of `auto`: the "
                        "letter-format families (belebele, global_mmlu_full, "
                        "include_base_44) as `rf` cloze tasks scored on the "
                        "answer strings (the default when no value is given, "
                        "group auto_rf) or as the `rfgm` Gemini-rewritten "
                        "statements (group auto_rfgm) — prefixed task names, "
                        "so nothing already evaluated is touched")
    p.add_argument("--every", type=int, default=1,
                   help="coarsen the evaluated grid: every Nth of the ten "
                        "tenths of training. The default 1 is the grid the "
                        "analysis reads (12 checkpoints/run: the tenths plus "
                        "85%% and 95%%); raise it only for one-off passes")
    p.add_argument("--final-only", action="store_true",
                   help="evaluate only each cell's last due checkpoint, not "
                        "all twelve — a screening pass over a new benchmark "
                        "group, where the noise window is not yet worth its "
                        "node-hours. No SNR and no DA-ckpt come out of it")
    p.add_argument("--group", default="auto", metavar="NAME",
                   help="the configs/tasks.json benchmark group to evaluate "
                        "(default auto). A probe group keeps candidate "
                        "benchmarks out of the watchers' way, which read auto "
                        "every pass; the jobs take the group's name (minus "
                        "auto_) as a suffix so neither reads the other's job "
                        "as its own")
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
    p.add_argument("--retry-held", action="store_true",
                   help="give every held-back task ONE more chance, then go "
                        "back to normal. Run this after fixing a root cause "
                        "(a task dropped from the group, a rebuilt wheel, a "
                        "repaired dataset): a checkpoint whose tasks are ALL "
                        "held is dropped from `pending`, so no job is ever "
                        "submitted for it again and its streak can never "
                        "reset — it stays parked long after the cause is "
                        "gone. Only the FIRST pass skips the gate, so under "
                        "--watch the watcher protects itself again from the "
                        "second pass on.")
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
    args.schemes = [args.scheme] if args.scheme else list(DATA_SCHEMES)
    args.sizes = args.size.split(",") if args.size else EVAL_SIZES
    if bad := set(args.sizes) - set(LADDER):
        p.error(f"unknown size(s) {sorted(bad)}; the ladder is {LADDER}")

    if args.reformulated and args.group != "auto":
        p.error("--group and --reformulated both choose the benchmark group; pass one")
    group = f"auto_{args.reformulated}" if args.reformulated else args.group
    benchmarks = auto_benchmarks(group)
    # The reformulated evals get their own job name (`eval-<cell>-iter<N>-rf`
    # / `-rfgm`): the original and each reformulated set of one checkpoint are
    # different work, so no watcher may read another's job as its own and skip it.
    args.job_suffix = f"-{args.reformulated}" if args.reformulated else (
        "" if group == "auto" else f"-{group.removeprefix('auto_')}")
    if args.retry_held:
        print("--retry-held: the failure gate is off for this pass only\n")
    while True:
        try:
            one_pass(args, Path(args.root), Path(args.staging),
                     Path(args.logs_root), benchmarks)
        except Exception:
            # Nothing restarts the watcher, so under --watch an unexpected
            # error must cost one pass, not stop the sweep's evals until
            # someone notices. A one-shot run still fails loudly.
            if not args.watch:
                raise
            import traceback
            traceback.print_exc()
        # One shot, whatever --watch says: the point is to let a fixed root
        # cause prove itself once, not to disable the gate for the session.
        args.retry_held = False
        # eval_progress.png is a view of exactly the state this pass just
        # changed, so refresh it here rather than on launch (the launcher only
        # redraws the training-side figures). Best-effort: a plotting problem
        # must never stop the watch loop.
        if not args.dry_run:
            try:
                from pretrain_progress import eval_progress
                eval_progress(root=Path(args.root), logs_root=Path(args.logs_root))
                eval_progress(root=Path(args.root), logs_root=Path(args.logs_root),
                              all_languages=True)
            except Exception as e:
                print(f"(eval progress plot not refreshed: {e})", file=sys.stderr)
            # The 1B/1.7B status table rides along: its job states go stale
            # within a pass, and nothing else redraws it between launches.
            try:
                from pretrain_progress import large_rung_status
                large_rung_status(root=Path(args.root))
            except Exception as e:
                print(f"(1B/1.7B status not refreshed: {e})", file=sys.stderr)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
