#!/usr/bin/env python3
"""
Launch Azure ML eval jobs for (cell × checkpoint) combinations — the Azure
counterpart of the SLURM runner chain (hf_base_runner.sh → evaluate.sbatch).

Each selected checkpoint submits jobs/eval.yml pointed at the converted
HF snapshot under models/<cell>/iter_<N> (produce those with
jobs/convert.yml first) with NAME=<cell>-iter<N>, the id
push_all_results.py resolves.

Usage:
    python azure/launch_evals.py [--dry-run]
                                 [--size SIZE] [--mix_en MIX_EN] [--seed SEED]
                                 [--ckpts final|full_eval|dense_tail|10_ckpts|da_ckpts|all]
                                 [--tasks TASKS]

--tasks accepts comma-separated lm-eval task names (default: hellaswag) or a
group name from configs/tasks.json (e.g. pretraining_full).

Examples:
    python azure/launch_evals.py --size 175M --mix_en 30 --seed 28            # hellaswag, final ckpt
    python azure/launch_evals.py --seed 28 --ckpts full_eval --tasks pretraining_full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import (  # noqa: E402
    filter_models, get_model, iters_for, stages_of, tasks_for_group)

SOURCES = ["snr-pretraining-custom", "snr-pretraining-bilingual"]
SCRIPT_DIR = Path(__file__).parent
DATASTORE = "azureml://datastores/workspaceblobstore/paths"


def az_args() -> list[str]:
    try:
        return ["--resource-group", os.environ["AZ_RG"],
                "--workspace-name", os.environ["AZ_WS"]]
    except KeyError:
        sys.exit("AZ_RG/AZ_WS not set — run `source azure/env.sh` first.")


def resolve_tasks(spec: str) -> str:
    try:
        return ",".join(tasks_for_group(spec))  # a configs/tasks.json group name
    except KeyError:
        return spec  # already a comma-separated lm-eval task list


def resolve_iters(name: str, subset: str) -> list[int]:
    if subset == "final":
        return [stages_of(name)["pretraining"]["checkpoints"]["final"]]
    return iters_for(name, subset=subset, stage="pretraining")


def submit(name: str, it: int, tasks: str, dry_run: bool) -> None:
    eval_name = f"{name}-iter{it}"
    overrides = {
        "display_name": f"eval-{eval_name}",
        "inputs.hf_model.path": f"{DATASTORE}/models/{name}/iter_{it:07d}",
        "environment_variables.NAME": eval_name,
        "environment_variables.TASKS": tasks,
    }
    if os.environ.get("WANDB_API_KEY"):
        overrides["environment_variables.WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    cmd = ["az", "ml", "job", "create",
           "--file", str(SCRIPT_DIR / "jobs" / "eval.yml"), *az_args()]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]

    print(f"  job: eval-{eval_name}  tasks={tasks}")
    if dry_run:
        print("  " + " ".join(c if "WANDB_API_KEY" not in c else
                              "environment_variables.WANDB_API_KEY=***" for c in cmd) + "\n")
    else:
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--size")
    p.add_argument("--mix_en", type=int, choices=[30, 60, 90])
    p.add_argument("--seed", type=int)
    p.add_argument("--ckpts", default="final",
                   choices=["final", "all", "dense_tail", "10_ckpts", "da_ckpts",
                            "full_eval", "auto"])
    p.add_argument("--tasks", default="hellaswag")
    args = p.parse_args()

    tasks = resolve_tasks(args.tasks)
    names = filter_models(source=SOURCES, size=args.size,
                          seeds=[args.seed] if args.seed is not None else None)
    launched = 0
    for name in names:
        if args.mix_en is not None and get_model(name)["mix_en"] != args.mix_en:
            continue
        for it in resolve_iters(name, args.ckpts):
            submit(name, it, tasks, args.dry_run)
            launched += 1
    print(f"{launched} job(s) {'printed' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
