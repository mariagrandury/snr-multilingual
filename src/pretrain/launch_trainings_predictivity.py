#!/usr/bin/env python3
"""
Launch Slurm training jobs for the small-to-large predictivity sweep.

The grid (see
.claude-shared/plans/small-to-large-predictivity-training-plan.md):

  * size            — the 6-rung ladder (90M..1.7B) shared by the reviewed
                       hyperparams files; --arch picks deep (hyperparams_deep
                       .json, baseline) or shallow (hyperparams_shallow.json,
                       the model-depth intervention level)
  * language setting — L in {1, 2, 8, 15, 30, 50, 100} (English + L-1
                       FineWeb-2 languages); not every size trains at every L.
  * seed            — one seed by default; three on the cells the plan marks
                       x3 (the 175M and 1B columns at L in {1, 30, 100}).

Each run trains its size's own budget D(N) = 5 x Chinchilla = 100 x N on the
fixed 50/50 English (DCLM) + FineWeb-2 data mix (L=1 is 100% English), composed
from the pre-built datasets via the Megatron data loader's blend weights. The
datasets are built once by build_data_mixtures.py; this launcher only composes
the blend and submits.

The design-choice intervention (tokenizer / depth / temperature; plan open
question 4) is not yet wired: this launches one run per (size, L, seed). Once
the intervention is chosen, add it as a third axis here and fold its level into
DATA_MIX_LABEL / the blend.

It reuses submit-apertus-data-mix.sh unchanged except for three env hooks that
script already honours: DATA_BLEND (the pre-built --data-path value),
TOKENIZER_MODEL (match the build tokenizer), and PROJECT_NAME.

Usage:
    python launch_trainings_predictivity.py [--dry-run] [--test]
        [--size SIZE] [--langs L] [--seed SEED]
        [--data_dir DIR] [--time HH:MM:SS] [--account ACCT] [--dependency DEP]

Examples:
    python launch_trainings_predictivity.py --dry-run            # whole sweep
    python launch_trainings_predictivity.py --size 1B --langs 30 # one (size,L)
    python launch_trainings_predictivity.py --langs 1            # monolingual anchors
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
SUBMIT_SCRIPT = SCRIPT_DIR / "submit-apertus-data-mix.sh"

# The two reviewed architecture families cover the same six non-embedding
# sizes; "shallow vs deep" (width/depth 128 vs 64) is the model-depth level of
# the intervention axis. Deep is the baseline.
HYPERPARAMS = {
    "deep": SCRIPT_DIR / "hyperparams" / "hyperparams_deep.json",
    "shallow": SCRIPT_DIR / "hyperparams" / "hyperparams_shallow.json",
}

# W&B project for this sweep — single source of truth is configs/hf_wandb.json.
# The entity is always "mariagrandury-epflnlp"; it is hardcoded in the training
# entrypoints (submit-apertus-data-mix.sh / azure/train.sh), not threaded here.
PROJECT_NAME = json.loads(
    (SCRIPT_DIR.parent.parent / "configs" / "hf_wandb.json").read_text()
)["wandb"]["project"]

# Tokenizer that produced the .bin token IDs (build_data_mixtures.py default).
# Must match at train time or the vocab/EOD ids won't line up.
TOKENIZER_MODEL = "swiss-ai/Apertus-70B-2509"

# Where build_data_mixtures.py wrote english_dclm.* and fineweb_L*.* — override
# with --data_dir. Placeholder cluster path; adjust to the actual build output.
DEFAULT_DATA_DIR = (
    "/capstor/store/cscs/swissai/infra01/users/mariagrandury/predictivity-data"
)

# --- Grid definition (edit these to change the sweep) -----------------------

LANG_SETTINGS = [1, 2, 8, 15, 30, 50, 100]
EN_SHARE = 50  # fixed English share for the multilingual (L >= 2) settings

# Which language settings each size trains at. Every size covers all settings
# except 1.7B, the top rung, which the plan trains only at L in {1,8,30,100}.
# (The 200-language setting was dropped 2026-08-13 for budget + deadline.)
SIZE_LANG_SETTINGS = {
    "90M": LANG_SETTINGS,
    "175M": LANG_SETTINGS,
    "350M": LANG_SETTINGS,
    "600M": LANG_SETTINGS,
    "1B": LANG_SETTINGS,
    "1.7B": [1, 8, 30, 100],
}

# Cells trained with three seeds (else one). The plan marks the 175M and 1B
# columns x3 at L in {1, 30, 100}.
SEED_SINGLE = [1904]
SEED_TRIPLE = [28, 1797, 1904]
TRIPLE_SIZES = {"175M", "1B"}
TRIPLE_LANGS = {1, 30, 100}

# Test run: smallest size, one mid setting, 50 steps.
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
    decay — see "predictivity_schedule" in the file's global section; the
    generators keep the block in sync with the architecture). The top-level
    train_iters belongs to the fixed-token SNR sweeps and is ignored here."""
    p = cfg["predictivity"]
    return p["train_iters"], p["lr_warmup_iters"], p["lr_wsd_decay_iters"]


def data_blend(data_dir: str, L: int) -> str:
    """Megatron --data-path value blending English (DCLM) and FineWeb-2.

    L = 1: English only (weight 1.0). L >= 2: English and the setting's
    FineWeb-2 each at the fixed 50% share. Prefixes are the .bin/.idx prefixes
    written by build_data_mixtures.py.
    """
    english = f"{data_dir}/english_dclm"
    if L == 1:
        return f"1.0 {english}"
    fineweb = f"{data_dir}/fineweb_L{L}"
    return f"{EN_SHARE / 100:.2f} {english} {(100 - EN_SHARE) / 100:.2f} {fineweb}"


def mix_label(L: int, arch: str = "deep") -> str:
    """Short label for EXP_NAME / job name: `L8` (50/50), `L1` (100% English);
    the shallow depth-variant is marked, e.g. `L8-shallow`."""
    return f"L{L}" + ("-shallow" if arch == "shallow" else "")


def build_export_vars(
    cfg: dict,
    size: str,
    data_blend_str: str,
    mix: str,
    seed: int,
    *,
    training_steps: Optional[int] = None,
    lr_warmup_iters: Optional[int] = None,
    lr_wsd_decay_iters: Optional[int] = None,
    mbs: Optional[int] = None,
) -> str:
    """Comma-separated KEY=VALUE string for sbatch --export.

    Architecture, LR, and the predictivity schedule come from the selected
    reviewed hyperparams file (deep or shallow); the data blend, tokenizer,
    and project name are the predictivity-specific env hooks the submit
    script honours.
    """
    iters, warmup, decay = schedule_for(cfg)
    vars_dict = {
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
        "SEED": seed,
        # Predictivity-specific env hooks (see submit-apertus-data-mix.sh)
        "DATA_BLEND": data_blend_str,
        "DATA_MIX_LABEL": mix,
        "TOKENIZER_MODEL": TOKENIZER_MODEL,
        "PROJECT_NAME": PROJECT_NAME,
    }
    return ",".join(f"{k}={v}" for k, v in vars_dict.items())


def submit(
    job_name: str,
    export_vars: str,
    dry_run: bool,
    nodes: Optional[int] = None,
    time: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
) -> None:
    cmd = ["sbatch", f"--job-name={job_name}", f"--export=ALL,{export_vars}"]
    if nodes is not None:
        cmd.append(f"--nodes={nodes}")
    if time is not None:
        cmd.append(f"--time={time}")
    if account is not None:
        cmd.append(f"--account={account}")
    if dependency is not None:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(SUBMIT_SCRIPT))

    print(f"  job:    {job_name}")
    if nodes is not None:
        print(f"  nodes:  {nodes}")
    print(f"  export: {export_vars}")
    if dry_run:
        print("  - skipped (dry-run)\n")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n{result.stdout.strip()}\n")


def job_name_for(size: str, mix: str, seed: int) -> str:
    """Slurm job name (lowercased size), e.g. apertus-1b-L30-en30-seed28."""
    return f"apertus-{size.lower()}-{mix}-seed{seed}"


def run_test(data: dict, data_dir: str, dry_run: bool, arch: str = "deep") -> None:
    cfg = data["configs"][TEST_SIZE]
    mix = mix_label(TEST_LANGS, arch)
    print(
        f"=== Test run: {TEST_SIZE} | {mix} | seed {TEST_SEED} | {TEST_STEPS} steps ===\n"
    )
    submit(
        job_name=f"apertus-test-{TEST_SIZE.lower()}-{mix}",
        export_vars=build_export_vars(
            cfg,
            TEST_SIZE,
            data_blend(data_dir, TEST_LANGS),
            mix,
            TEST_SEED,
            training_steps=TEST_STEPS,
            lr_warmup_iters=TEST_WARMUP,
            lr_wsd_decay_iters=TEST_DECAY,
        ),
        dry_run=dry_run,
    )


def run_filtered(
    data: dict,
    data_dir: str,
    dry_run: bool,
    arch: str = "deep",
    size_filter: Optional[list[str]] = None,
    langs_filter: Optional[int] = None,
    seed_filter: Optional[int] = None,
    time: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
) -> None:
    cells = [
        c
        for c in predictivity_cells()
        if (size_filter is None or c["size"] in size_filter)
        and (langs_filter is None or c["L"] == langs_filter)
        and (seed_filter is None or c["seed"] == seed_filter)
    ]
    if not cells:
        print("No cells match the given filters.")
        return
    print(f"=== Launching {len(cells)} jobs ===\n")
    for c in cells:
        cfg = data["configs"][c["size"]]
        mix = mix_label(c["L"], arch)
        submit(
            job_name=job_name_for(c["size"], mix, c["seed"]),
            export_vars=build_export_vars(
                cfg,
                c["size"],
                data_blend(data_dir, c["L"]),
                mix,
                c["seed"],
            ),
            dry_run=dry_run,
            nodes=cfg.get("nodes"),
            time=time,
            account=account,
            dependency=dependency,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Smoke-test: {TEST_SIZE}, {mix_label(TEST_LANGS)}, "
        f"seed {TEST_SEED}, {TEST_STEPS} steps",
    )
    parser.add_argument(
        "--size",
        metavar="SIZES",
        help="Filter by size — one or a comma-separated list of "
        "the ladder's size keys (e.g. "
        "'600M' or '175M,350M'). Default: all sizes.",
    )
    parser.add_argument(
        "--langs",
        metavar="L",
        type=int,
        help=f"Filter by language setting (one of {LANG_SETTINGS})",
    )
    parser.add_argument("--seed", metavar="SEED", type=int, help="Filter by seed")
    parser.add_argument(
        "--arch",
        choices=["deep", "shallow"],
        default="deep",
        help="Architecture family: deep (baseline, width/depth 64) or shallow "
        "(width/depth 128 — the model-depth intervention level).",
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory holding english_dclm.* and fineweb_L*.* "
        f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--time",
        metavar="HH:MM:SS",
        help="Override sbatch --time (default from the submit script)",
    )
    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        help="Override sbatch --account (e.g. a139 if infra01 is depleted)",
    )
    parser.add_argument(
        "--dependency",
        metavar="DEPENDENCY",
        help="Pass-through to sbatch --dependency (e.g. 'singleton')",
    )
    args = parser.parse_args()

    if not SUBMIT_SCRIPT.exists():
        sys.exit(f"Error: submit script not found: {SUBMIT_SCRIPT}")
    if not HYPERPARAMS[args.arch].exists():
        sys.exit(f"Error: hyperparams file not found: {HYPERPARAMS[args.arch]}")

    valid_sizes = list(SIZE_LANG_SETTINGS)
    size_filter = args.size.split(",") if args.size else None
    bad_sizes = [s for s in (size_filter or []) if s not in valid_sizes]
    if bad_sizes:
        sys.exit(f"Error: --size {bad_sizes} not valid. Choose from: {valid_sizes}")
    if args.langs and args.langs not in LANG_SETTINGS:
        sys.exit(
            f"Error: --langs '{args.langs}' not valid. Choose from: {LANG_SETTINGS}"
        )

    print(f"Config:   {HYPERPARAMS[args.arch]} (arch: {args.arch})")
    print(f"Script:   {SUBMIT_SCRIPT}")
    print(f"Data dir: {args.data_dir}")
    if args.dry_run:
        print("(dry-run — sbatch commands will be printed but not executed)")
    print()

    data = json.loads(HYPERPARAMS[args.arch].read_text())

    if args.test:
        run_test(data, args.data_dir, args.dry_run, arch=args.arch)
    else:
        run_filtered(
            data,
            args.data_dir,
            args.dry_run,
            arch=args.arch,
            size_filter=size_filter,
            langs_filter=args.langs,
            seed_filter=args.seed,
            time=args.time,
            account=args.account,
            dependency=args.dependency,
        )
