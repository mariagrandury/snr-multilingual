#!/usr/bin/env python3
"""
Auto-eval watcher: every N saved checkpoints (default 2) plus the run's final
checkpoint, evaluate on the "auto" benchmark group (configs/tasks.json) and
push to W&B (msnr) — progress signal beyond the loss curve while a training
runs on Azure.

Idempotent, like the cluster's eval launchers: each pass lists the blob
storage, finds due checkpoints (saved iters at multiples of N x save-interval),
and per due iter submits at most one missing step — jobs/convert.yml when
the HF snapshot doesn't exist yet, else jobs/eval.yml (TASKS=auto) when
results don't exist yet. Anything already done or in flight is skipped, so the
convert -> eval sequencing simply resolves across successive passes. Run it
alongside training:

    source azure/env.sh          # + WANDB_API_KEY in your shell profile (laptop)
    python auto_evals_azure.py --watch 600            # one pass every 10 min

Same filters as the other launchers (--size/--seed/--name); default = every
cell with checkpoints in the workspace's blob storage. --dry-run prints
instead of submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import (  # noqa: E402
    filter_models, get_model, load_hf_wandb_config, stages_of,
    tasks_for_benchmarks)
from launch_trainings import cell_languages  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "azure"))
from launch_evals import az_args, resolve_tasks, submit as submit_eval  # noqa: E402

SOURCES = ["snr-pretraining-predictivity"]
SCRIPT_DIR = Path(__file__).parent
DATASTORE = "azureml://datastores/workspaceblobstore/paths"
AUTO_BENCHMARKS = json.loads(
    (Path(__file__).resolve().parents[2] / "configs" /
     "tasks.json").read_text())["groups"]["auto"]
ACTIVE_STATES = {"NotStarted", "Queued", "Starting", "Preparing", "Running", "Finalizing"}
# Eval results land under eval_logs/<entity>/<project>/ (azure/eval.sh) — the
# project is msnr (configs/hf_wandb.json), the same one training logs to.
WANDB = load_hf_wandb_config()["wandb"]


def az_json(*cmd: str):
    out = subprocess.run(["az", *cmd, "--output", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def storage_auth() -> list[str]:
    """workspaceblobstore's account/container + key (no data-plane RBAC needed)."""
    ds = az_json("ml", "datastore", "show", "--name", "workspaceblobstore", *az_args())
    keys = az_json("storage", "account", "keys", "list",
                   "--account-name", ds["account_name"],
                   "--resource-group", os.environ["AZ_RG"])
    return ["--account-name", ds["account_name"], "--container-name", ds["container_name"],
            "--account-key", keys[0]["value"]]


def list_blobs(auth: list[str], prefix: str) -> list[str]:
    return az_json("storage", "blob", "list", *auth,
                   "--prefix", prefix, "--query", "[].name", "--num-results", "5000")


def saved_iters(auth: list[str], name: str) -> list[int]:
    """Valid saved checkpoints (dir has .metadata + >=1 .distcp shard, the same
    validity check as pretrain_progress.is_valid_iter_dir)."""
    blobs = list_blobs(auth, f"runs/{name}/checkpoints/iter_")
    meta, shard = set(), set()
    for b in blobs:
        m = re.search(r"iter_(\d+)/([^/]+)$", b)
        if not m:
            continue
        it = int(m.group(1))
        if m.group(2) == ".metadata":
            meta.add(it)
        elif m.group(2).endswith(".distcp"):
            shard.add(it)
    return sorted(meta & shard)


def active_jobs() -> set[str]:
    jobs = az_json("ml", "job", "list", "--max-results", "200", *az_args(),
                   "--query", "[].{d:display_name,s:status}")
    return {j["d"] for j in jobs if j["s"] in ACTIVE_STATES}


def submit_convert(name: str, it: int, dry_run: bool) -> None:
    overrides = {
        "display_name": f"convert-{name}-iter{it}",
        "inputs.checkpoints.path": f"{DATASTORE}/runs/{name}/checkpoints",
        "environment_variables.CKPT_STEP": it,
        "outputs.hf_model.path": f"{DATASTORE}/models/{name}/iter_{it:07d}",
    }
    cmd = ["az", "ml", "job", "create",
           "--file", str(SCRIPT_DIR / "azure" / "jobs" / "convert.yml"), *az_args()]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    print(f"  submit: convert-{name}-iter{it}")
    if dry_run:
        print("  " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def one_pass(names: list[str], auth: list[str], every: int,
             tasks: str | None, dry_run: bool) -> None:
    """`tasks` None = the auto group: per cell, every auto benchmark in
    the languages that cell trains on (models.json carries L/scheme)."""
    # Keep configs/models.json following the grid (see sync_models_json).
    from sync_models_json import sync
    added, updated = sync()
    if added or updated:
        print(f"(models.json synced: +{len(added)} ~{len(updated)} cells — commit the diff)")
    running = active_jobs() if not dry_run else set()
    for name in names:
        iters = saved_iters(auth, name)
        if not iters:
            continue
        # Every Nth saved checkpoint (default every 2nd: iters on the
        # every*save-interval grid), PLUS the run's final checkpoint whatever
        # its number — predictivity targets end off-grid (e.g. 4500, 81000).
        ck = stages_of(name)["pretraining"]["checkpoints"]
        step = ck["all"][1] - ck["all"][0] if len(ck["all"]) > 1 else 2000
        due = [i for i in iters if i % (every * step) == 0 or i == ck["final"]]
        m = get_model(name)
        cell_tasks = tasks or ",".join(tasks_for_benchmarks(
            AUTO_BENCHMARKS, cell_languages(m["L"], m["scheme"])))
        converted = {b.split("/")[2] for b in list_blobs(auth, f"models/{name}/")
                     if b.endswith("config.json")}
        evaluated = {m.group(1)
                     for b in list_blobs(
                         auth,
                         f"eval_logs/{WANDB['entity']}/{WANDB['project']}/{name}-iter")
                     if "results_" in b
                     for m in [re.search(rf"/({re.escape(name)}-iter\d+)/harness/", b)] if m}
        to_convert = [it for it in iters if f"iter_{it:07d}" not in converted]
        print(f"{name}: {len(iters)} saved | convert {to_convert or '-'} | eval due {due}")
        # Convert EVERY saved checkpoint (persist all HF snapshots to blob) — the
        # converted models are the durable copy; eval only samples 1/N of them.
        for it in to_convert:
            if f"convert-{name}-iter{it}" not in running:
                submit_convert(name, it, dry_run)
        for it in due:
            if f"{name}-iter{it}" in evaluated:
                continue
            if f"iter_{it:07d}" in converted and f"eval-{name}-iter{it}" not in running:
                submit_eval(name, it, cell_tasks, dry_run)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--size")
    p.add_argument("--seed", type=int)
    p.add_argument("--name", help="watch a single cell (a configs/models.json key)")
    p.add_argument("--every", type=int, default=2,
                   help="evaluate every N saved checkpoints (the final "
                        "checkpoint is always evaluated on top)")
    p.add_argument("--tasks", default="auto")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="keep running, one pass every SECONDS")
    args = p.parse_args()

    names = [args.name] if args.name else filter_models(
        source=SOURCES, size=args.size,
        seeds=[args.seed] if args.seed is not None else None)
    # "auto" resolves per cell (benchmarks x trained languages); anything
    # else is a fixed task list/group as before.
    tasks = None if args.tasks == "auto" else resolve_tasks(args.tasks)
    auth = storage_auth()
    while True:
        one_pass(names, auth, args.every, tasks, args.dry_run)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
