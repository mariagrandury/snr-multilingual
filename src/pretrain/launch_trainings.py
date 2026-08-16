#!/usr/bin/env python3
"""
Launch Slurm training jobs for (model size × data ratio × seed) combinations.

Usage:
    python launch_trainings.py [--dry-run] [--test]
                               [--size SIZE] [--mix_en MIX_EN] [--seed SEED]

Flags:
    --dry-run        Print sbatch commands without running them.
    --test           Submit a smoke-test: 175M, mix_en=60, seed=28, 50 steps,
                     lr_warmup_iters=10, lr_wsd_decay_iters=20.
    --size SIZE      Filter by model size (must be a key in the JSON, e.g. 175M, 350M, 600M, 1B).
    --mix_en MIX_EN  Filter by FW_EDU_RATIO — one of 30, 60, or 90.
    --seed SEED      Filter by seed (integer).

Filters can be combined. With no filters, all (sizes × mixtures × seeds) are launched.

Examples:
    # All combinations
    python launch_trainings.py

    # All mixtures × all seeds for a single size
    python launch_trainings.py --size 175M

    # All sizes × all seeds for a single mixture
    python launch_trainings.py --mix_en 90

    # All sizes × all mixtures for a single seed
    python launch_trainings.py --seed 28

    # Specific size + mixture, all seeds
    python launch_trainings.py --size 175M --mix_en 90

    # Single job
    python launch_trainings.py --size 1B --mix_en 30 --seed 28

    # One command
    cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/ && git pull && conda activate && python launch_trainings.py --size 175M --mix_en 60 --seed 28

    # See training logs
    cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/ && cat apertus-175m-edu60-fw240-seed28-1827322.out

    # See checkpoints
    ls /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed28/checkpoints/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Shared configs loader (src/evals/scripts/utils/configs.py). One-shot `src/`
# on sys.path so `from evals.scripts.utils.configs import …` resolves via
# Python's implicit namespace packages — same content cluster or local.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import filter_models, get_model  # noqa: E402

# The custom-pretrain sweep (4 sizes × 3 mixes × 3 seeds = 36 cells) is
# enumerated from configs/models.json — every cell with
# source="snr-pretraining-custom". Architecture configs still come from
# hyperparams_deep.json, keyed by each model's `hyperparams_key`.
CUSTOM_SOURCE = "snr-pretraining-custom"

# Test run configuration
TEST_SIZE = "175M"
TEST_RATIO = ("90", "10")
TEST_SEED = 28
TEST_STEPS = 50
TEST_LR_WARMUP_ITERS = 10
TEST_LR_WSD_DECAY_ITERS = 20
TEST_MBS = 1

SCRIPT_DIR = Path(__file__).parent
SUBMIT_SCRIPT = SCRIPT_DIR / "submit-apertus-data-mix.sh"
CONFIG_FILE = SCRIPT_DIR / "hyperparams" / "hyperparams_deep.json"


def custom_cells() -> list[dict]:
    """The 36 canonical custom-pretrain model entries from configs/models.json,
    in size → mix → seed order (build_configs.py's insertion order)."""
    return [get_model(name) for name in filter_models(source=CUSTOM_SOURCE)]


def build_export_vars(
    model_size: str,
    cfg: dict,
    fw_edu: Optional[str | int] = None,
    fw2: Optional[str | int] = None,
    training_steps: Optional[int] = None,
    seed: Optional[int] = None,
    lr_warmup_iters: Optional[int] = None,
    lr_wsd_decay_iters: Optional[int] = None,
    mbs: Optional[int] = None,
) -> str:
    """Return a comma-separated KEY=VALUE string suitable for sbatch --export."""
    vars_dict = {
        "MODEL_SIZE": model_size,
        "NUM_LAYERS": cfg["n_layers"],
        "HIDDEN_SIZE": cfg["hidden_size"],
        "FFN_HIDDEN_SIZE": cfg["ffn_hidden_size"],
        "NUM_ATTENTION_HEADS": cfg["num_attention_heads"],
        "NUM_QUERY_GROUPS": cfg["num_query_groups"],
        "MBS": mbs if mbs is not None else cfg["micro_batch_size"],
        "TRAINING_STEPS": (
            training_steps if training_steps is not None else cfg["train_iters"]
        ),
        "LR": cfg["lr"],
    }
    if fw_edu is not None:
        vars_dict["FW_EDU_RATIO"] = fw_edu
    if fw2 is not None:
        vars_dict["FW2_RATIO"] = fw2
    if seed is not None:
        vars_dict["SEED"] = seed
    if lr_warmup_iters is not None:
        vars_dict["LR_WARMUP_ITERS"] = lr_warmup_iters
    if lr_wsd_decay_iters is not None:
        vars_dict["LR_WSD_DECAY_ITERS"] = lr_wsd_decay_iters
    return ",".join(f"{k}={v}" for k, v in vars_dict.items())


def submit(
    job_name: str,
    export_vars: str,
    dry_run: bool,
    nodes: Optional[int] = None,
    time: Optional[str] = None,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
) -> None:
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export=ALL,{export_vars}",
    ]
    if nodes is not None:
        cmd.append(f"--nodes={nodes}")
    if time is not None:
        cmd.append(f"--time={time}")
    if partition is not None:
        cmd.append(f"--partition={partition}")
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
        print("  - skipped (dry-run)")
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(result.returncode)

        msg = result.stdout.strip()
        job_id = msg.split(" ")[-1]
        print(f"\n{msg}")
        print(
            f"\n/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/{job_name}-{job_id}.out or .err"
        )
    print()


def run_test(data: dict, dry_run: bool) -> None:
    if TEST_SIZE not in data["configs"]:
        print(
            f"Error: '{TEST_SIZE}' config not found in JSON — cannot run test.",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg = data["configs"][TEST_SIZE]
    fw_edu, fw2 = TEST_RATIO
    print(
        f"=== Test run: {TEST_SIZE} | edu={fw_edu} fw2={fw2} | seed={TEST_SEED} | {TEST_STEPS} steps ==="
        f" | lr_warmup={TEST_LR_WARMUP_ITERS} | lr_wsd_decay={TEST_LR_WSD_DECAY_ITERS} | mbs={TEST_MBS} ===\n"
    )
    submit(
        job_name=f"apertus-test-{TEST_SIZE.lower()}",
        export_vars=build_export_vars(
            TEST_SIZE,
            cfg,
            fw_edu=fw_edu,
            fw2=fw2,
            seed=TEST_SEED,
            training_steps=TEST_STEPS,
            lr_warmup_iters=TEST_LR_WARMUP_ITERS,
            lr_wsd_decay_iters=TEST_LR_WSD_DECAY_ITERS,
            mbs=TEST_MBS,
        ),
        dry_run=dry_run,
    )


def run_filtered(
    data: dict,
    dry_run: bool,
    size_filter: Optional[str] = None,
    mix_en_filter: Optional[str] = None,
    seed_filter: Optional[int] = None,
    time: Optional[str] = None,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    dependency: Optional[str] = None,
    training_steps: Optional[int] = None,
) -> None:
    cells = [
        e for e in custom_cells()
        if (size_filter is None or e["size"] == size_filter)
        and (mix_en_filter is None or str(e["mix_en"]) == mix_en_filter)
        and (seed_filter is None or e["seed"] == seed_filter)
    ]
    if not cells:
        print("No cells match the given filters.")
        return
    print(f"=== Launching {len(cells)} jobs ===\n")
    for e in cells:
        model_size = e["size"]
        cfg = data["configs"][e["hyperparams_key"]]
        submit(
            job_name=(
                f"apertus-{model_size.lower()}-edu{e['mix_en']}"
                f"-fw2{e['mix_fw2']}-seed{e['seed']}"
            ),
            export_vars=build_export_vars(
                model_size, cfg,
                fw_edu=e["mix_en"], fw2=e["mix_fw2"], seed=e["seed"],
                training_steps=training_steps,
            ),
            dry_run=dry_run,
            nodes=cfg.get("nodes"),
            time=time,
            partition=partition,
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
        help="Print commands without submitting",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            f"Smoke-test: {TEST_SIZE}, mix_en={TEST_RATIO[0]}, seed={TEST_SEED},"
            f" {TEST_STEPS} steps, lr_warmup={TEST_LR_WARMUP_ITERS},"
            f" lr_wsd_decay={TEST_LR_WSD_DECAY_ITERS}"
        ),
    )
    parser.add_argument(
        "--size",
        metavar="SIZE",
        help="Filter by model size — must match a key in the JSON (e.g. 175M)",
    )
    parser.add_argument(
        "--mix_en",
        metavar="MIX_EN",
        help="Filter by FW_EDU_RATIO: 30, 60, or 90",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        help="Filter by seed",
    )
    parser.add_argument(
        "--time",
        metavar="HH:MM:SS",
        help="Override sbatch --time (default: 11:59:59 from the script). "
             "Must be > the SIGUSR2 grace (1h) so training has time to run "
             "before the signal-then-checkpoint-then-exit dance starts.",
    )
    parser.add_argument(
        "--partition",
        metavar="PARTITION",
        help="Override sbatch --partition (default: cluster default).",
    )
    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        help="Override sbatch --account (default: infra01 from the script). "
             "Use 'a139' if your fairshare on infra01 is depleted.",
    )
    parser.add_argument(
        "--dependency",
        metavar="DEPENDENCY",
        help="Pass-through to sbatch --dependency (e.g. 'singleton' to ensure "
             "only one job with the same name runs at a time).",
    )
    parser.add_argument(
        "--training-steps",
        metavar="N",
        type=int,
        help="Override TRAINING_STEPS (default: 50000 from the submit script). "
             "Used by launch_resumes.sh to cap training at a specific "
             "canonical iter when filling a mid-training gap.",
    )
    args = parser.parse_args()

    if not SUBMIT_SCRIPT.exists():
        print(f"Error: submit script not found: {SUBMIT_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    print(f"Config:  {CONFIG_FILE}")
    print(f"Script:  {SUBMIT_SCRIPT}")
    if args.dry_run:
        print("(dry-run — sbatch commands will be printed but not executed)")
    print()

    with open(CONFIG_FILE) as f:
        data = json.load(f)

    # Validate --size / --mix_en against the canonical custom-cell grid
    # (dict.fromkeys dedups while keeping the JSON's size→mix→seed order).
    cells = custom_cells()
    valid_sizes = list(dict.fromkeys(e["size"] for e in cells))
    valid_mix_en = list(dict.fromkeys(str(e["mix_en"]) for e in cells))
    if args.size and args.size not in valid_sizes:
        print(
            f"Error: --size '{args.size}' not valid. Choose from: {valid_sizes}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.mix_en and args.mix_en not in valid_mix_en:
        print(
            f"Error: --mix_en '{args.mix_en}' not valid. Choose from: {valid_mix_en}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.test:
        run_test(data, args.dry_run)
    else:
        run_filtered(
            data,
            args.dry_run,
            size_filter=args.size,
            mix_en_filter=args.mix_en,
            seed_filter=args.seed,
            time=args.time,
            partition=args.partition,
            account=args.account,
            dependency=args.dependency,
            training_steps=args.training_steps,
        )
