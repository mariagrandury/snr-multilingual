#!/usr/bin/env python3
"""
Auto-eval watcher: every N saved checkpoints (default 2) plus the run's final
checkpoint, evaluate on the "auto" benchmark group (configs/tasks.json) and
push to W&B (msnr) — progress signal beyond the loss curve while a training
runs on Azure.

Idempotent, like the cluster's eval launchers: each pass lists the blob
storage, finds due checkpoints (saved iters at multiples of N x save-interval),
and per due iter submits at most one missing step — jobs/convert.yml when
the HF snapshot doesn't exist yet, else jobs/eval.yml (TASKS=auto) while any
of the cell's tasks is still missing a result (task-level, so adding a
benchmark reaches already-evaluated checkpoints instead of leaving two task
sets in the sweep). Anything already done or in flight is skipped, so the
convert -> eval sequencing simply resolves across successive passes. Run it
alongside training:

    source azure/env.sh          # + WANDB_API_KEY in your shell profile (laptop)
    python auto_evals_azure.py --watch 600                 # Spain: sizes <= 600M
    python auto_evals_azure.py --workspace ca --watch 600  # Canada: 1B and 1.7B

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
import tempfile
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import evals.scripts.utils.configs as configs  # noqa: E402
from evals.scripts.utils.configs import (  # noqa: E402
    filter_models, get_model, load_hf_wandb_config, stages_of,
    tasks_for_benchmarks)
from launch_trainings import (  # noqa: E402
    TOKENIZER_MODEL, ND_SIZES, cell_languages, job_name)
from sync_models_json import sync  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "azure"))
from launch_evals import az_args, resolve_tasks, submit as submit_eval  # noqa: E402

SOURCES = ["snr-pretraining-predictivity"]
SCRIPT_DIR = Path(__file__).parent
DATASTORE = "azureml://datastores/workspaceblobstore/paths"
# Training outputs are pinned under predictivity/ (launch_trainings.DATASTORE);
# converted models and eval logs live at the blob root (convert/eval.yml).
RUNS_PREFIX = "predictivity/runs"
# jobs/{convert,eval}.yml default to the Spain compute; the Canada Central
# workspace only has the ND96 pool, so its submissions override it.
# NOTE 2026-08-26: gpu-nd96-spot cannot currently allocate (H100 is refused
# for low-priority and dedicated H100 quota is 0). Point this at an A100
# cluster (gpu-nc96-a100-lp) if evals need to run before H100 quota lands.
ND_COMPUTE = "azureml:gpu-nd96-spot"
AUTO_BENCHMARKS = json.loads(
    (Path(__file__).resolve().parents[2] / "configs" /
     "tasks.json").read_text())["groups"]["auto"]
ACTIVE_STATES = {"NotStarted", "Queued", "Starting", "Preparing", "Running", "Finalizing"}
# Eval results land under eval_logs/<entity>/<project>/ (azure/eval.sh) — the
# project is msnr (configs/hf_wandb.json), the same one training logs to.
WANDB = load_hf_wandb_config()["wandb"]


def az_json(*cmd: str):
    # timeout: a throttled or hung az call must fail the pass, not the
    # multi-day watcher (main() retries the pass); launch_trainings.active_jobs
    # takes the same precaution.
    out = subprocess.run(["az", *cmd, "--output", "json"], check=True,
                         capture_output=True, text=True, timeout=120).stdout
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
    """All blob names under prefix — paginated: a single call silently
    truncates at --num-results, and an L=100 cell's eval_logs (one samples
    file per task per eval) can exceed 5000 blobs."""
    names, marker = [], None
    while True:
        cmd = ["storage", "blob", "list", *auth, "--prefix", prefix,
               "--num-results", "5000", "--show-next-marker",
               "--query", "[].{name:name,nextMarker:nextMarker}"]
        page = az_json(*(cmd + (["--marker", marker] if marker else [])))
        names += [b["name"] for b in page if b.get("name")]
        # --show-next-marker appends one {"nextMarker": ...} element at the end
        marker = page[-1].get("nextMarker") if page else None
        if not marker:
            return names


_BLOB_JSON: dict[str, dict] = {}


def blob_json(auth: list[str], name: str) -> dict:
    """One blob parsed as JSON. `az storage blob download` writes the content
    to --file and its own property dict to stdout, so a temp file is the only
    unambiguous way to get just the content. Returns {} if the blob is
    unreadable or not JSON — a half-uploaded results file must not kill the
    watch loop.

    Memoised for the life of the watcher: a results blob's name carries the
    eval's timestamp, so its content never changes once written, and without
    this every pass would re-download every finished checkpoint's results.
    """
    if name in _BLOB_JSON:
        return _BLOB_JSON[name]
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "blob.json"
        try:
            subprocess.run(["az", "storage", "blob", "download", *auth,
                            "--name", name, "--file", str(dest),
                            "--only-show-errors"],
                           check=True, capture_output=True, text=True)
            # Cache only on success: a transient download failure must be
            # retried next pass, not remembered as "no tasks evaluated".
            _BLOB_JSON[name] = json.loads(dest.read_text())
            return _BLOB_JSON[name]
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
            print(f"  warn: could not read {name}", file=sys.stderr)
            return {}


def saved_iters(auth: list[str], name: str) -> list[int]:
    """Valid saved checkpoints (dir has .metadata + >=1 .distcp shard, the same
    validity check as pretrain_progress.is_valid_iter_dir)."""
    blobs = list_blobs(auth, f"{RUNS_PREFIX}/{name}/checkpoints/iter_")
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
    # --all-results: the default newest-200 window includes completed jobs,
    # so a long-queued job falls out of it and gets resubmitted as a
    # duplicate. The --query projection keeps the full listing small.
    jobs = az_json("ml", "job", "list", "--all-results", "true", *az_args(),
                   "--query", "[].{d:display_name,s:status}")
    return {j["d"] for j in jobs if j["s"] in ACTIVE_STATES}


def submit_convert(name: str, it: int, dry_run: bool,
                   compute: str | None = None) -> None:
    overrides = {
        "display_name": job_name("convert", f"{name}-iter{it}"),
        "inputs.checkpoints.path": f"{DATASTORE}/{RUNS_PREFIX}/{name}/checkpoints",
        "environment_variables.CKPT_STEP": it,
        # Predictivity cells train with the Apertus V1 tokenizer, not
        # convert.sh's alehc/swissai-tokenizer default (the 36-sweep's).
        "environment_variables.HF_TOKENIZER": TOKENIZER_MODEL,
        "outputs.hf_model.path": f"{DATASTORE}/models/{name}/iter_{it:07d}",
    }
    if compute:
        overrides["compute"] = compute
    cmd = ["az", "ml", "job", "create",
           "--file", str(SCRIPT_DIR / "azure" / "jobs" / "convert.yml"), *az_args()]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]
    print(f"  submit: {job_name('convert', f'{name}-iter{it}')}")
    if dry_run:
        print("  " + " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def one_pass(names: list[str], auth: list[str], every: int,
             tasks: str | None, dry_run: bool,
             compute: str | None = None) -> None:
    """`tasks` None = the auto group: per cell, every auto benchmark in
    the languages that cell trains on (models.json carries L/scheme).
    `names` comes from resolve_names(), which re-syncs models.json first."""
    running = active_jobs() if not dry_run else set()
    for name in names:
        iters = saved_iters(auth, name)
        if not iters:
            continue
        # Every Nth saved checkpoint counted from the run's first save, PLUS
        # the run's final checkpoint whatever its number — the CSCS rule
        # (auto_evals_cscs.one_cell): counting saves keeps cells on an older
        # save grid due, where `iter % (N x interval)` would find nothing.
        ck = stages_of(name)["pretraining"]["checkpoints"]
        due = [i for k, i in enumerate(sorted(iters))
               if (k + 1) % every == 0 or i == ck["final"]]
        m = get_model(name)
        cell_tasks = tasks or ",".join(tasks_for_benchmarks(
            AUTO_BENCHMARKS, cell_languages(m["L"], m["scheme"])))
        # Converted = the .hf_complete marker azure/convert.sh writes LAST —
        # config.json lands on the rw_mount while save_pretrained is still
        # uploading shards, so its presence alone would let a preempted
        # convert job poison the iter forever (never re-converted, evaluated
        # against partial weights). Marker-less snapshots from before the
        # marker existed are simply re-converted (idempotent, same bytes).
        converted = {b.split("/")[2] for b in list_blobs(auth, f"models/{name}/")
                     if b.endswith("/.hf_complete")}
        # Results blobs per checkpoint. Gate on TASKS, not on "any results
        # exist": adding a benchmark to the auto group has to reach
        # checkpoints already evaluated on the old list, or the sweep
        # silently carries two different task sets. The CSCS watcher gets
        # this from _eval_status.completed_tasks against local disk; here
        # list_blobs returns names only, so the results files must be read.
        # Only for checkpoints that HAVE results, and only when due (below),
        # so a pass costs one download per completed eval rather than per
        # checkpoint.
        results_blobs: dict[str, list[str]] = {}
        for b in list_blobs(
                auth,
                f"eval_logs/{WANDB['entity']}/{WANDB['project']}/{name}-iter"):
            hit = re.search(rf"/({re.escape(name)}-iter\d+)/harness/", b)
            if hit and "/results_" in b and b.endswith(".json"):
                results_blobs.setdefault(hit.group(1), []).append(b)

        def evaluated_tasks(ckpt: str) -> set[str]:
            done: set[str] = set()
            for b in results_blobs.get(ckpt, ()):
                done |= set((blob_json(auth, b).get("results") or {}).keys())
            return done

        to_convert = [it for it in iters if f"iter_{it:07d}" not in converted]
        print(f"{name}: {len(iters)} saved | convert {to_convert or '-'} | eval due {due}")
        # Convert EVERY saved checkpoint (persist all HF snapshots to blob) — the
        # converted models are the durable copy; eval only samples 1/N of them.
        for it in to_convert:
            if job_name("convert", f"{name}-iter{it}") not in running:
                submit_convert(name, it, dry_run, compute)
        # No twin of the CSCS watcher's --max-attempts guard here: a totally
        # failed eval leaves an EMPTY HARNESS_EVAL_DIR (azure/eval.sh), and an
        # empty directory uploads no blob at all, so the listing cannot count
        # attempts. Watch the AML job list for a run of failures instead.
        want = set(cell_tasks.split(","))
        for it in due:
            if not want - evaluated_tasks(f"{name}-iter{it}"):
                continue
            if (f"iter_{it:07d}" in converted
                    and job_name("convert", f"{name}-iter{it}") not in running
                    and job_name("eval", f"{name}-iter{it}") not in running):
                # Same tokenizer note as submit_convert: override eval.sh's
                # 36-sweep default with the tokenizer this sweep trains with.
                submit_eval(name, it, cell_tasks, dry_run, compute,
                            env={"TOKENIZER": TOKENIZER_MODEL})


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
    p.add_argument("--workspace", choices=["es", "ca"], default="es",
                   help="which workspace to watch: es (sizes up to 600M) or "
                        "ca (1B/1.7B, Canada Central ND pool) — run one "
                        "watcher per workspace")
    args = p.parse_args()

    # The launcher splits the grid across two workspaces (ND_SIZES train on
    # the Canada Central ND pool, the rest in Spain Central); each workspace
    # has its own blob store and compute, so point az at the right one and
    # keep only the cells that live there.
    nd = args.workspace == "ca"
    if nd:
        for var in ("AZ_RG", "AZ_WS"):
            os.environ[var] = os.environ.get(f"AZ_CA_{var[3:]}") or sys.exit(
                f"AZ_CA_{var[3:]} not set — run `source azure/env.sh` first.")
    compute = ND_COMPUTE if nd else None

    def resolve_names() -> list[str]:
        """The cells this watcher covers, re-read EVERY pass: sync() upserts
        cells launched since the last pass into models.json, but the configs
        loaders are lru_cached, so without clearing them the running process
        would keep the list it read at startup forever."""
        added, updated = sync()
        if added or updated:
            print(f"(models.json synced: +{len(added)} ~{len(updated)} cells "
                  f"— commit the diff)")
            for f in (configs.load_models, configs.load_pools,
                      configs.load_sources, configs._bucket_map):
                f.cache_clear()
        names = [args.name] if args.name else filter_models(
            source=SOURCES, size=args.size,
            seeds=[args.seed] if args.seed is not None else None)
        return [n for n in names if (get_model(n)["size"] in ND_SIZES) == nd]

    names = resolve_names()
    if not names:
        sys.exit(f"no cells in the {args.workspace} workspace match the "
                 f"filters (1B/1.7B live in ca, the rest in es)")
    # "auto" resolves per cell (benchmarks x trained languages); anything
    # else is a fixed task list/group as before.
    tasks = None if args.tasks == "auto" else resolve_tasks(args.tasks)
    auth = storage_auth()
    while True:
        # A watcher meant to run for days must outlive one throttled az call
        # or one half-written blob: log the pass and try again next time.
        try:
            one_pass(names, auth, args.every, tasks, args.dry_run, compute)
        except Exception as e:
            if not args.watch:
                raise
            print(f"pass failed ({e!r}) — retrying in {args.watch}s",
                  file=sys.stderr)
        if not args.watch:
            break
        names = resolve_names() or names
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
