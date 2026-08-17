#!/usr/bin/env python3
"""
Launch Azure ML jobs for the small-to-large predictivity sweep — the Azure
counterpart of ../launch_trainings_predictivity.py (sbatch -> `az ml job
create`), sharing its grid definition (sizes x language settings x seeds,
x3-seed cells) by importing it.

Placement is by size: <=600M runs go to the Spain Central workspace
(gpu-nc80-lp, 2x H100, fixed low-priority pricing); 1B and 1.7B go to the UK
South workspace (gpu-nd96-spot, 8x H100 + InfiniBand, Spot). --arch picks the
reviewed architecture family (deep baseline or the shallow depth-intervention
variant); the schedule comes from each config's "predictivity" block. Requires
`source env.sh` (AZ_ML_ARGS_ES / AZ_ML_ARGS_UK) and the pre-built datasets
uploaded to each workspace's blob store under
predictivity/data/{english_dclm,fineweb_L<L>}/ (see the README's predictivity
section). Resubmitting a cell resumes it from its latest checkpoint.

Usage:
    python launch_azure_predictivity.py [--dry-run] [--arch {deep,shallow}]
                                        [--size SIZE] [--langs L] [--seed SEED]

Examples:
    python launch_azure_predictivity.py --dry-run              # whole sweep (51 jobs)
    python launch_azure_predictivity.py --size 1.7B --langs 30
    python launch_azure_predictivity.py --langs 1              # monolingual anchors
    python launch_azure_predictivity.py --arch shallow --dry-run  # depth variant
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from launch_trainings_predictivity import (  # noqa: E402
    EN_SHARE, HYPERPARAMS, PROJECT_NAME, TOKENIZER_MODEL, WANDB_ENTITY,
    mix_label, predictivity_cells, schedule_for)

DATASTORE = "azureml://datastores/workspaceblobstore/paths/predictivity"
UK_SIZES = {"1B", "1.7B"}  # everything else runs on the Spain economy pool


def az_args(size: str) -> tuple[str, list[str]]:
    var = "AZ_ML_ARGS_UK" if size in UK_SIZES else "AZ_ML_ARGS_ES"
    try:
        return var, os.environ[var].split()
    except KeyError:
        sys.exit(f"{var} not set — run `source env.sh` first.")


def data_blend(L: int) -> str:
    """Megatron --data-path over the job's mounted inputs (resolved by AML)."""
    english = "${{inputs.english}}/english_dclm"
    if L == 1:
        return f"1.0 {english}"
    return (f"{EN_SHARE / 100:.2f} {english} "
            f"{(100 - EN_SHARE) / 100:.2f} ${{{{inputs.fineweb}}}}/fineweb_L{L}")


def submit(cell: dict, cfg: dict, arch: str, dry_run: bool) -> None:
    size, L, seed = cell["size"], cell["L"], cell["seed"]
    mix = mix_label(L, arch)
    exp = f"apertus-{size}-{mix}-seed{seed}"
    ws_var, ws_args = az_args(size)
    iters, warmup, decay = schedule_for(cfg)
    overrides = {
        "display_name": exp,
        "compute": ("azureml:gpu-nd96-spot" if size in UK_SIZES
                    else "azureml:gpu-nc80-lp"),
        "inputs.fineweb.path":
            f"{DATASTORE}/data/{'english_dclm' if L == 1 else f'fineweb_L{L}'}",
        **{f"outputs.{o}.path": f"{DATASTORE}/runs/{exp}/{o}"
           for o in ("checkpoints", "logs", "cache")},
        "environment_variables.MODEL_SIZE": size,
        "environment_variables.NUM_LAYERS": cfg["n_layers"],
        "environment_variables.HIDDEN_SIZE": cfg["hidden_size"],
        "environment_variables.FFN_HIDDEN_SIZE": cfg["ffn_hidden_size"],
        "environment_variables.NUM_ATTENTION_HEADS": cfg["num_attention_heads"],
        "environment_variables.NUM_QUERY_GROUPS": cfg["num_query_groups"],
        "environment_variables.MBS": cfg["micro_batch_size"],
        "environment_variables.LR": cfg["lr"],
        "environment_variables.TRAINING_STEPS": iters,
        "environment_variables.LR_WARMUP_ITERS": warmup,
        "environment_variables.LR_WSD_DECAY_ITERS": decay,
        "environment_variables.SEED": seed,
        "environment_variables.DATA_MIX_LABEL": mix,
        "environment_variables.DATA_BLEND": data_blend(L),
        # W&B entity/project from configs/hf_wandb.json (via the shared
        # launcher constants) — the job container can't read the config itself
        # (code context is src/pretrain/), so pass them in here.
        "environment_variables.WANDB_ENTITY": WANDB_ENTITY,
        "environment_variables.PROJECT_NAME": PROJECT_NAME,
    }
    if os.environ.get("WANDB_API_KEY"):
        overrides["environment_variables.WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    cmd = ["az", "ml", "job", "create",
           "--file", str(SCRIPT_DIR / "jobs" / "train-predictivity.yml"), *ws_args]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]

    print(f"  job: {exp}  [{ws_var}]  ({iters} iters, "
          f"{cfg['predictivity']['train_tokens'] / 1e9:.1f}B tokens)")
    if dry_run:
        print("  " + " ".join(c if "WANDB_API_KEY" not in c else
                              "environment_variables.WANDB_API_KEY=***" for c in cmd) + "\n")
    else:
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arch", choices=["deep", "shallow"], default="deep",
                   help="Architecture family: deep (baseline) or shallow "
                        "(the model-depth intervention level).")
    p.add_argument("--size")
    p.add_argument("--langs", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    data = json.loads(HYPERPARAMS[args.arch].read_text())
    launched = 0
    for cell in predictivity_cells():
        if args.size and cell["size"] != args.size:
            continue
        if args.langs is not None and cell["L"] != args.langs:
            continue
        if args.seed is not None and cell["seed"] != args.seed:
            continue
        submit(cell, data["configs"][cell["size"]], args.arch, args.dry_run)
        launched += 1
    print(f"{launched} job(s) {'printed' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
