#!/usr/bin/env python3
"""
Launch predictivity-sweep training jobs on CSCS (sbatch) or Azure ML (az ml).

The grid (see plan/small-to-large-predictivity-training-plan.md):

  * size — the 6-rung ladder (90M..1.7B) shared by the reviewed hyperparams
           files; --arch picks deep (hyperparams/hyperparams_deep.json, the
           baseline) or shallow (hyperparams/hyperparams_shallow.json, the
           model-depth intervention level).
  * L    — language setting in {1, 2, 8, 15, 30, 50, 100}: English + L-1
           FineWeb-2 languages. 1.7B trains only at L in {1, 8, 30, 100}.
  * seed — 1904 by default; three seeds (28, 1797, 1904) on the cells the
           plan marks x3 (the 175M and 1B columns at L in {1, 30, 100}).

Each run trains its size's own budget D(N) = 5 x Chinchilla = 100 x N on the
fixed 50/50 English (DCLM) + FineWeb-2 mix (L=1 is 100% English), blended at
training time from the pre-built datasets (data/build_data_mixtures.py).

Both platforms run the SAME training logic: this script builds one env-var
dict per cell and hands it to megatron_args.sh through a thin platform
wrapper — launch_pretraining_cscs.sh (sbatch --export) or
launch_pretraining_azure.sh (az ml job create --set, via
azure/jobs/pretrain.yml). Azure placement is by size: <=600M on the Spain
low-priority pool, 1B/1.7B on the UK Spot pool.

IDEMPOTENT — re-running is always safe, there is no separate resume script:
  * a cell whose target checkpoint is already on disk is skipped ("done");
  * a cell with a queued/running job is skipped;
  * a partially-trained cell is resubmitted as a RESUME (--save/--load point
    at the same dir): on CSCS the remaining iters size the walltime and a
    stale latest_checkpointed_iteration.txt marker is rewound to the last
    valid checkpoint first; on Azure resubmitting simply continues.
  * a corrupt cell (iter dirs on disk but none loadable) is skipped with a
    warning — cleanup is always a human decision.
(Azure done-detection would need a blob listing per cell, so only the
active-job check runs there; resubmitting a finished cell is a no-op run —
Megatron loads the final checkpoint and exits at --train-iters.)

Usage:
    python launch_trainings.py cscs  [--data_dir DIR] [--time HH:MM:SS]
                                     [--account ACCT] [--dependency DEP]
                                     [--training-steps N] [--test] [filters]
    python launch_trainings.py azure [filters]        # `source azure/env.sh` first

Filters (both platforms): --arch {deep,shallow}, --scheme {A,B},
--size 350M[,175M,...], --langs L, --seed N, --dry-run.

Examples:
    python launch_trainings.py cscs --dry-run              # whole sweep (52 jobs)
    python launch_trainings.py cscs --size 350M,175M       # two sizes, all L
    python launch_trainings.py azure --size 1.7B --langs 30
    python launch_trainings.py azure --langs 1             # monolingual anchors
    python launch_trainings.py cscs --arch shallow --dry-run  # depth variant
    python launch_trainings.py cscs --scheme B --langs 8   # scheme-B data variant
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent

# The two reviewed architecture families cover the same six non-embedding
# sizes; "shallow vs deep" (width/depth 128 vs 64) is the model-depth level of
# the intervention axis. Deep is the baseline.
HYPERPARAMS = {
    "deep": SCRIPT_DIR / "hyperparams" / "hyperparams_deep.json",
    "shallow": SCRIPT_DIR / "hyperparams" / "hyperparams_shallow.json",
}

# W&B project for this sweep — single source of truth is configs/hf_wandb.json.
# The entity is the constant "mariagrandury-epflnlp", hardcoded in
# megatron_args.sh (the shared training-argument file).
PROJECT_NAME = json.loads(
    (SCRIPT_DIR.parent.parent / "configs" / "hf_wandb.json").read_text()
)["wandb"]["project"]

# Tokenizer that produced the .bin token IDs (data/build_data_mixtures.py
# default). Must match at train time or the vocab/EOD ids won't line up.
TOKENIZER_MODEL = "swiss-ai/Apertus-70B-2509"

# --- Platform endpoints ------------------------------------------------------

CSCS_SUBMIT_SCRIPT = SCRIPT_DIR / "launch_pretraining_cscs.sh"
# Where data/build_data_mixtures.py wrote english_dclm.* and fineweb_L*.* on
# the cluster — override with --data_dir.
CSCS_DEFAULT_DATA_DIR = (
    "/capstor/store/cscs/swissai/infra01/users/mariagrandury/predictivity-data"
)

AZURE_JOB_YML = SCRIPT_DIR / "azure" / "jobs" / "pretrain.yml"
DATASTORE = "azureml://datastores/workspaceblobstore/paths/predictivity"
UK_SIZES = {"1B", "1.7B"}  # everything else runs on the Spain economy pool

# --- Grid definition (edit these to change the sweep) ------------------------

LANG_SETTINGS = [1, 2, 8, 15, 30, 50, 100]
EN_SHARE = 50  # fixed English share for the multilingual (L >= 2) settings

# Which language settings each size trains at. Every size covers all settings
# except 1.7B, the top rung, which the plan trains at L in {1, 2, 8, 30, 100}.
SIZE_LANG_SETTINGS = {
    "90M": LANG_SETTINGS,
    "175M": LANG_SETTINGS,
    "350M": LANG_SETTINGS,
    "600M": LANG_SETTINGS,
    "1B": LANG_SETTINGS,
    "1.7B": [1, 2, 8, 30, 100],
}

# Cells trained with three seeds (else one). The plan marks the 175M and 1B
# columns x3 at L in {1, 30, 100}.
SEED_SINGLE = [1904]
SEED_TRIPLE = [28, 1797, 1904]
TRIPLE_SIZES = {"175M", "1B"}
TRIPLE_LANGS = {1, 30, 100}


def _scheme_b_langs() -> set[int]:
    """The language settings where scheme B actually differs from scheme A,
    derived from the language-set JSONs so it can't drift: {8, 15, 30}.
    Everywhere else the two schemes define identical data, so those cells
    always run (and are named) as the scheme-A baseline."""
    sets = {
        s: json.loads((SCRIPT_DIR / "data" /
                       f"language_sets_scheme{s}.json").read_text())["sets"]
        for s in ("A", "B")
    }
    return {L for L in LANG_SETTINGS
            if f"FW_L{L}" in sets["A"]
            and sets["A"][f"FW_L{L}"] != sets["B"][f"FW_L{L}"]}


SCHEME_B_LANGS = _scheme_b_langs()

# CSCS smoke test: smallest size, one mid setting, 50 steps.
TEST_SIZE = "90M"
TEST_LANGS = 8
TEST_SEED = 1904
TEST_STEPS = 50
TEST_WARMUP = 10
TEST_DECAY = 20


def seeds_for(size: str, L: int) -> list[int]:
    """Three seeds on the x3 cells, one otherwise."""
    return SEED_TRIPLE if (size in TRIPLE_SIZES and L in TRIPLE_LANGS) else SEED_SINGLE


def predictivity_cells() -> list[dict]:
    """Every (size, L, seed) run in size -> L -> seed order."""
    cells = []
    for size, settings in SIZE_LANG_SETTINGS.items():
        for L in settings:
            for seed in seeds_for(size, L):
                cells.append({"size": size, "L": L, "seed": seed})
    return cells


def schedule_for(cfg: dict) -> tuple[int, int, int]:
    """Per-size predictivity schedule from the config's own "predictivity"
    block (D = 100 x N tokens at 504 x 4096 tokens/iter, ~4% warmup, ~20% WSD
    decay — see "predictivity_schedule" in the file's global section). The
    top-level train_iters belongs to the fixed-token SNR sweeps and is
    ignored here."""
    p = cfg["predictivity"]
    return p["train_iters"], p["lr_warmup_iters"], p["lr_wsd_decay_iters"]


def mix_label(L: int, arch: str = "deep", scheme: str = "A") -> str:
    """Short label for EXP_NAME: `L8` (50/50), `L1` (100% English). Non-default
    variants are marked with suffixes, e.g. `L8-schemeB`, `L8-shallow`,
    `L8-schemeB-shallow` (scheme A / deep are the unmarked baselines)."""
    return (f"L{L}"
            + ("-schemeB" if scheme == "B" else "")
            + ("-shallow" if arch == "shallow" else ""))


def exp_name(size: str, L: int, arch: str, seed: int, scheme: str = "A") -> str:
    """Canonical run name — also the Slurm job name, the Azure display name,
    the checkpoint dir under Meg-Runs/<project>/, and the W&B run prefix.
    pretrain_progress.py parses this format."""
    return f"apertus-{size}-{mix_label(L, arch, scheme)}-seed{seed}"


def data_blend(english: str, fineweb: str, L: int) -> str:
    """Megatron --data-path value blending English (DCLM) and FineWeb-2.

    L = 1: English only (weight 1.0). L >= 2: English and the setting's
    FineWeb-2 each at the fixed 50% share. The prefixes are the .bin/.idx
    prefixes written by data/build_data_mixtures.py — cluster paths on CSCS,
    AML input mounts on Azure.
    """
    if L == 1:
        return f"1.0 {english}"
    return f"{EN_SHARE / 100:.2f} {english} {(100 - EN_SHARE) / 100:.2f} {fineweb}"


# Width-scaled init anchor: 1/sqrt(hidden_size) scaling that keeps the
# reviewed 0.008944 exactly at the 1B width (d=1792), so the init is
# consistent across the 768..3072 ladder instead of one fixed value.
INIT_STD_ANCHOR = 0.008944
INIT_STD_ANCHOR_WIDTH = 1792


def init_std(hidden_size: int) -> float:
    return round(INIT_STD_ANCHOR * (INIT_STD_ANCHOR_WIDTH / hidden_size) ** 0.5, 6)


def cell_env(
    cfg: dict,
    size: str,
    seed: int,
    exp: str,
    blend: str,
    training_steps: Optional[int] = None,
    lr_warmup_iters: Optional[int] = None,
    lr_wsd_decay_iters: Optional[int] = None,
    mbs: Optional[int] = None,
) -> dict:
    """The env-var dict megatron_args.sh consumes — the platform-independent
    description of one run. Identical on CSCS and Azure by construction."""
    iters, warmup, decay = schedule_for(cfg)
    return {
        "MODEL_SIZE": size,
        "NUM_LAYERS": cfg["n_layers"],
        "HIDDEN_SIZE": cfg["hidden_size"],
        "FFN_HIDDEN_SIZE": cfg["ffn_hidden_size"],
        "NUM_ATTENTION_HEADS": cfg["num_attention_heads"],
        "NUM_QUERY_GROUPS": cfg["num_query_groups"],
        "MBS": mbs if mbs is not None else cfg["micro_batch_size"],
        "TRAINING_STEPS": training_steps if training_steps is not None else iters,
        "LR": cfg["lr"],
        "LR_WARMUP_ITERS": lr_warmup_iters if lr_warmup_iters is not None else warmup,
        "LR_WSD_DECAY_ITERS": (
            lr_wsd_decay_iters if lr_wsd_decay_iters is not None else decay
        ),
        # AdEMAMix alpha/beta3 warm up over the cell's FULL schedule — always
        # the target iters, never a capped resume's --training-steps, so every
        # (re)submission runs the identical optimizer schedule.
        "ADEMAMIX_WARMUP": iters,
        "INIT_STD": init_std(cfg["hidden_size"]),
        "SEED": seed,
        "EXP_NAME": exp,
        "DATA_BLEND": blend,
        "TOKENIZER_MODEL": TOKENIZER_MODEL,
        "PROJECT_NAME": PROJECT_NAME,
    }


# --- CSCS (sbatch) -----------------------------------------------------------

GBS = 504  # global batch size (megatron_args.sh) — fixed across the sweep

# Cluster scale (nodes) lives in the deep hyperparams file; the shallow ladder
# shares the same non-embedding sizes, so it uses the same node counts.
NODES_BY_SIZE = {
    size: cfg["nodes"]
    for size, cfg in json.loads(HYPERPARAMS["deep"].read_text())["configs"].items()
}


def cscs_mbs(nodes: int, mbs: int) -> int:
    """Largest micro-batch <= the memory-tuned value that divides the cluster
    layout (DP = 4 GPUs x nodes; Megatron requires GBS % (DP x MBS) == 0) —
    the same resolution launch_pretraining_azure.sh does against its GPU
    count. A no-op for the deep ladder (its values are already valid); the
    shallow ladder's generator-suggested MBS (24/14/8/4/...) needs it."""
    dp = 4 * nodes
    while GBS % (dp * mbs) != 0:
        mbs -= 1
    return mbs


# Per-size steady-state iter time (ms), for walltime sizing. 175M..1B sampled
# from the 36-sweep training logs (same architectures/node counts); 90M and
# 1.7B are estimates — tighten them from the first real runs.
ITER_MS = {"90M": 800, "175M": 800, "350M": 565, "600M": 520, "1B": 715, "1.7B": 1200}
TIME_MARGIN_SEC = 9000   # 2h30m: 1h SIGUSR2 grace + cold-start + buffer
TIME_MIN_SEC = 5400      # 1h30m
TIME_MAX_SEC = 43199     # 11:59:59 (slurm normal queue cap)


def auto_time(size: str, remaining_iters: int) -> str:
    """Walltime for a run with `remaining_iters` to go, rounded up to 15 min."""
    total = remaining_iters * ITER_MS.get(size, 800) // 1000 + TIME_MARGIN_SEC
    total = (total + 899) // 900 * 900
    total = min(max(total, TIME_MIN_SEC), TIME_MAX_SEC)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def active_slurm_jobs() -> set[str]:
    """Names of this user's queued/running jobs (empty off-cluster)."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "--format=%j"],
                             capture_output=True, text=True, timeout=30)
        return set(out.stdout.split()) if out.returncode == 0 else set()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def rewind_marker(ckpt_dir: Path, want: int, dry_run: bool) -> bool:
    """Point latest_checkpointed_iteration.txt at `want` (the last VALID
    checkpoint) when it is ahead of it — e.g. after an async-save left a
    shell iter dir the marker already references. Refuses (False) if `want`
    isn't a loadable checkpoint; True when the marker is correct/rewound."""
    marker_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    try:
        current = int(marker_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return True  # no marker (fresh) or unreadable — Megatron will complain
    if current <= want:
        return True
    if dry_run:
        print(f"    (dry-run) would rewind marker: {current} -> {want}")
        return True
    from pretrain_progress import is_valid_iter_dir  # lazy: avoids import cycle
    if not is_valid_iter_dir(ckpt_dir / f"iter_{want:07d}"):
        print(f"    !! refusing to rewind marker: iter_{want:07d} is not a "
              f"valid checkpoint", file=sys.stderr)
        return False
    marker_file.write_text(f"{want}\n")
    print(f"    rewound marker: {current} -> {want}")
    return True


def submit_cscs(env: dict, dry_run: bool, nodes: Optional[int] = None,
                time: Optional[str] = None, account: Optional[str] = None,
                dependency: Optional[str] = None) -> None:
    export_vars = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", f"--job-name={env['EXP_NAME']}", f"--export=ALL,{export_vars}"]
    if nodes is not None:
        cmd.append(f"--nodes={nodes}")
    if time is not None:
        cmd.append(f"--time={time}")
    if account is not None:
        cmd.append(f"--account={account}")
    if dependency is not None:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(CSCS_SUBMIT_SCRIPT))

    print(f"  job:    {env['EXP_NAME']}" + (f"  (nodes: {nodes})" if nodes else ""))
    print(f"  export: {export_vars}")
    if dry_run:
        print("  - skipped (dry-run)\n")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n{result.stdout.strip()}\n")


# --- Azure ML (az ml job create) ---------------------------------------------

def az_args(size: str) -> tuple[str, list[str]]:
    var = "AZ_ML_ARGS_UK" if size in UK_SIZES else "AZ_ML_ARGS_ES"
    try:
        return var, os.environ[var].split()
    except KeyError:
        sys.exit(f"{var} not set — run `source azure/env.sh` first.")


def active_azure_jobs(ws_args: list[str]) -> set[str]:
    """Display names of queued/running jobs in one workspace (empty on any
    az failure — the check is best-effort)."""
    states = {"NotStarted", "Queued", "Starting", "Preparing", "Running", "Finalizing"}
    try:
        out = subprocess.run(
            ["az", "ml", "job", "list", "--max-results", "200", *ws_args,
             "--query", "[].{d:display_name,s:status}", "--output", "json"],
            capture_output=True, text=True, timeout=120)
        jobs = json.loads(out.stdout) if out.returncode == 0 else []
        return {j["d"] for j in jobs if j["s"] in states}
    except Exception:
        return set()


def submit_azure(env: dict, cell: dict, dry_run: bool,
                 data_root: Optional[str] = None) -> None:
    size, L, exp = cell["size"], cell["L"], env["EXP_NAME"]
    ws_var, ws_args = az_args(size)
    data_root = data_root or f"{DATASTORE}/data"
    overrides = {
        "display_name": exp,
        "compute": ("azureml:gpu-nd96-spot" if size in UK_SIZES
                    else "azureml:gpu-nc80-lp"),
        "inputs.fineweb.path":
            f"{data_root}/{'english_dclm' if L == 1 else f'fineweb_L{L}'}",
        **{f"outputs.{o}.path": f"{DATASTORE}/runs/{exp}/{o}"
           for o in ("checkpoints", "logs", "cache")},
        **{f"environment_variables.{k}": v for k, v in env.items()},
    }
    if os.environ.get("WANDB_API_KEY"):
        overrides["environment_variables.WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    cmd = ["az", "ml", "job", "create", "--file", str(AZURE_JOB_YML), *ws_args]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]

    print(f"  job: {exp}  [{ws_var}]  ({env['TRAINING_STEPS']} iters)")
    if dry_run:
        print("  " + " ".join(c if "WANDB_API_KEY" not in c else
                              "environment_variables.WANDB_API_KEY=***" for c in cmd) + "\n")
    else:
        subprocess.run(cmd, check=True)


# --- Driver ------------------------------------------------------------------

def run_test(data: dict, arch: str, data_dir: str, dry_run: bool) -> None:
    cfg = data["configs"][TEST_SIZE]
    exp = f"apertus-test-{TEST_SIZE.lower()}-{mix_label(TEST_LANGS, arch)}"
    blend = data_blend(f"{data_dir}/english_dclm",
                       f"{data_dir}/fineweb_L{TEST_LANGS}", TEST_LANGS)
    print(f"=== Test run: {TEST_SIZE} | {mix_label(TEST_LANGS, arch)} | "
          f"seed {TEST_SEED} | {TEST_STEPS} steps ===\n")
    submit_cscs(
        cell_env(cfg, TEST_SIZE, TEST_SEED, exp, blend,
                 training_steps=TEST_STEPS,
                 lr_warmup_iters=TEST_WARMUP, lr_wsd_decay_iters=TEST_DECAY),
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("platform", choices=["cscs", "azure"],
                        help="Where to submit: cscs (sbatch) or azure (az ml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print submit commands without running them")
    parser.add_argument("--arch", choices=["deep", "shallow"], default="deep",
                        help="Architecture family: deep (baseline) or shallow "
                             "(the model-depth intervention level)")
    parser.add_argument("--scheme", choices=["A", "B"], default="A",
                        help="Language-set scheme: A (resource-ranked, "
                             "baseline) or B (diversity-first; data under "
                             "<data_dir>/schemeB). B differs from A only at "
                             "L in {8, 15, 30} — other settings always run "
                             "as scheme A")
    parser.add_argument("--size", metavar="SIZES",
                        help="Filter by size — one or a comma-separated list "
                             "(e.g. '600M' or '350M,175M'). Default: all sizes.")
    parser.add_argument("--langs", metavar="L", type=int,
                        help=f"Filter by language setting (one of {LANG_SETTINGS})")
    parser.add_argument("--seed", metavar="SEED", type=int, help="Filter by seed")
    # CSCS-only knobs
    parser.add_argument("--data_dir", default=CSCS_DEFAULT_DATA_DIR,
                        help="CSCS only: dir holding english_dclm.* and "
                             f"fineweb_L*.* (default: {CSCS_DEFAULT_DATA_DIR})")
    parser.add_argument("--time", metavar="HH:MM:SS",
                        help="CSCS only: override sbatch --time")
    parser.add_argument("--account", metavar="ACCOUNT",
                        help="CSCS only: override sbatch --account (e.g. a139)")
    parser.add_argument("--dependency", metavar="DEP",
                        help="CSCS only: pass-through to sbatch --dependency")
    parser.add_argument("--training-steps", metavar="N", type=int,
                        help="CSCS only: manually cap --train-iters (the "
                             "idempotent resume logic sets this itself)")
    parser.add_argument("--test", action="store_true",
                        help=f"CSCS only: smoke test ({TEST_SIZE}, L{TEST_LANGS}, "
                             f"{TEST_STEPS} steps)")
    args = parser.parse_args()

    if args.platform != "cscs":
        for flag in ("time", "account", "dependency", "training_steps", "test"):
            if getattr(args, flag):
                parser.error(f"--{flag.replace('_', '-')} is CSCS-only")

    size_filter = args.size.split(",") if args.size else None
    valid_sizes = list(SIZE_LANG_SETTINGS)
    bad_sizes = [s for s in (size_filter or []) if s not in valid_sizes]
    if bad_sizes:
        parser.error(f"--size {bad_sizes} not valid. Choose from: {valid_sizes}")
    if args.langs and args.langs not in LANG_SETTINGS:
        parser.error(f"--langs '{args.langs}' not valid. Choose from: {LANG_SETTINGS}")

    config = HYPERPARAMS[args.arch]
    data = json.loads(config.read_text())
    print(f"Platform: {args.platform} | Config: {config.name} (arch: {args.arch})")
    if args.dry_run:
        print("(dry-run — submit commands will be printed but not executed)")
    print()

    if args.test:
        run_test(data, args.arch, args.data_dir, args.dry_run)
        return

    cells = [
        c for c in predictivity_cells()
        if (size_filter is None or c["size"] in size_filter)
        and (args.langs is None or c["L"] == args.langs)
        and (args.seed is None or c["seed"] == args.seed)
    ]
    if not cells:
        print("No cells match the given filters.")
        return
    print(f"=== {len(cells)} matching cells ===\n")

    # Idempotency state, computed once per invocation.
    if args.platform == "cscs":
        from pretrain_progress import CKPT_ROOT, cell_action  # lazy: no cycle
        active = active_slurm_jobs()
    else:
        active = set() if args.dry_run else set().union(
            *(active_azure_jobs(az_args(s)[1])
              for s in {c["size"] for c in cells}))

    # Scheme B data lives in its own subdir of the same layout (the english
    # build and validation manifest are symlinked in — see data/launch_builds.sh).
    cscs_dir = args.data_dir + ("/schemeB" if args.scheme == "B" else "")
    az_data = f"{DATASTORE}/data" + ("/schemeB" if args.scheme == "B" else "")

    for c in cells:
        cfg = data["configs"][c["size"]]
        # Scheme B only defines different data at SCHEME_B_LANGS ({8, 15, 30});
        # everywhere else (L=1 English-only, and the settings whose language
        # lists are identical across schemes) the cell is the scheme-A one — no
        # duplicate runs of identical data, and no pointing at schemeB data
        # paths that only hold the differing builds. A `--scheme B` sweep thus
        # submits B-cells where B differs and A-cells elsewhere (which the
        # idempotency check dedupes against an earlier scheme-A sweep).
        scheme = args.scheme if c["L"] in SCHEME_B_LANGS else "A"
        exp = exp_name(c["size"], c["L"], args.arch, c["seed"], scheme)
        target = schedule_for(cfg)[0]

        if exp in active:
            print(f"  skip [active]: {exp} already queued/running")
            continue

        if args.platform == "cscs":
            action, a, b = cell_action(CKPT_ROOT / exp, target)
            if action == "done":
                print(f"  skip [done]: {exp} (valid checkpoint at target {target})")
                continue
            if action == "corrupt":
                print(f"  *** corrupt: {exp} — {a} iter dir(s) on disk but none "
                      f"valid; SKIPPING (manual review)")
                continue
            load_iter, tgt = (0, a) if action == "fresh" else (a, b)
            if action == "resume" and not rewind_marker(
                    CKPT_ROOT / exp / "checkpoints", load_iter, args.dry_run):
                continue
            cell_dir = args.data_dir if scheme == "A" else cscs_dir
            blend = data_blend(f"{cell_dir}/english_dclm",
                               f"{cell_dir}/fineweb_L{c['L']}", c["L"])
            print(f"  [{action}] {exp}: iters {load_iter} -> {tgt}")
            nodes = cfg.get("nodes", NODES_BY_SIZE[c["size"]])
            submit_cscs(
                cell_env(cfg, c["size"], c["seed"], exp, blend,
                         training_steps=args.training_steps or tgt,
                         mbs=cscs_mbs(nodes, cfg["micro_batch_size"])),
                dry_run=args.dry_run, nodes=nodes,
                time=args.time or auto_time(c["size"], tgt - load_iter),
                account=args.account, dependency=args.dependency,
            )
        else:
            blend = data_blend("${{inputs.english}}/english_dclm",
                               f"${{{{inputs.fineweb}}}}/fineweb_L{c['L']}", c["L"])
            submit_azure(
                cell_env(cfg, c["size"], c["seed"], exp, blend),
                cell=c, dry_run=args.dry_run,
                data_root=az_data if scheme == "B" else None,
            )

    # Refresh the progress heatmaps on every CSCS launch so
    # pretrain_progress_{simple,detailed}.png are always up to date.
    # Best-effort: a plotting problem must never fail a submission.
    if args.platform == "cscs" and not args.dry_run:
        try:
            from pretrain_progress import update_plots
            update_plots()
        except Exception as e:  # e.g. matplotlib missing off-cluster
            print(f"(progress plots not refreshed: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
