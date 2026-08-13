#!/usr/bin/env python3
"""
Launch Azure ML training jobs for (model size × data ratio × seed) cells —
the Azure counterpart of ../launch_trainings.py (sbatch → `az ml job create`).

Each selected cell submits jobs/train-full.yml with the cell's architecture
env vars (from ../hyperparams_deep.json) and its own data/checkpoint paths.
Requires `source env.sh` (AZ_RG/AZ_WS) and the setup from the README; the
mixture the cell needs must already exist under tokenized/mix_<edu>_<fw2>/full
(prepare_data.py --edu-ratio 0.3/0.6/0.9).

Usage:
    python launch_azure_trainings.py [--dry-run]
                                     [--size SIZE] [--mix_en MIX_EN] [--seed SEED]

Examples:
    python launch_azure_trainings.py --size 175M --mix_en 30 --seed 28
    python launch_azure_trainings.py --seed 1904 --dry-run   # one seed, all 12 cells
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import filter_models, get_model  # noqa: E402

SOURCES = ["snr-pretraining-custom", "snr-pretraining-bilingual"]
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR.parent / "hyperparams_deep.json"
DATASTORE = "azureml://datastores/workspaceblobstore/paths"


def mix_label(name: str, entry: dict) -> str:
    """DATA_MIX_LABEL from the models.json key: apertus-<size>-<label>-seed<seed>."""
    return (name.removeprefix(f"apertus-{entry['size']}-")
                .removesuffix(f"-seed{entry['seed']}"))


def mix_dir(entry: dict) -> str:
    prefix = "mix_enru" if entry["source"] == "snr-pretraining-bilingual" else "mix"
    return f"{prefix}_{entry['mix_en']}_{entry['mix_fw2']}"


def az_args() -> list[str]:
    try:
        return ["--resource-group", os.environ["AZ_RG"],
                "--workspace-name", os.environ["AZ_WS"]]
    except KeyError:
        sys.exit("AZ_RG/AZ_WS not set — run `source env.sh` first.")


def submit(name: str, cfg: dict, entry: dict, dry_run: bool) -> None:
    overrides = {
        "display_name": name,
        "inputs.data.path": f"{DATASTORE}/tokenized/{mix_dir(entry)}/full",
        **{f"outputs.{o}.path": f"{DATASTORE}/runs/{name}/{o}"
           for o in ("checkpoints", "logs", "cache")},
        "environment_variables.MODEL_SIZE": entry["size"],
        "environment_variables.NUM_LAYERS": cfg["n_layers"],
        "environment_variables.HIDDEN_SIZE": cfg["hidden_size"],
        "environment_variables.FFN_HIDDEN_SIZE": cfg["ffn_hidden_size"],
        "environment_variables.NUM_ATTENTION_HEADS": cfg["num_attention_heads"],
        "environment_variables.NUM_QUERY_GROUPS": cfg["num_query_groups"],
        "environment_variables.MBS": cfg["micro_batch_size"],
        "environment_variables.LR": cfg["lr"],
        "environment_variables.TRAINING_STEPS": cfg["train_iters"],
        "environment_variables.FW_EDU_RATIO": entry["mix_en"],
        "environment_variables.FW2_RATIO": entry["mix_fw2"],
        "environment_variables.SEED": entry["seed"],
        "environment_variables.DATA_MIX_LABEL": mix_label(name, entry),
    }
    if os.environ.get("WANDB_API_KEY"):
        overrides["environment_variables.WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    cmd = ["az", "ml", "job", "create", "--file", str(SCRIPT_DIR / "jobs" / "train-full.yml"),
           *az_args()]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]

    print(f"  job: {name}  ({cfg['train_iters']} iters, mbs {cfg['micro_batch_size']})")
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
    args = p.parse_args()

    data = json.loads(CONFIG_FILE.read_text())
    names = filter_models(source=SOURCES, size=args.size,
                          seeds=[args.seed] if args.seed is not None else None)
    launched = 0
    for name in names:
        entry = get_model(name)
        if args.mix_en is not None and entry["mix_en"] != args.mix_en:
            continue
        submit(name, data["configs"][entry["hyperparams_key"]], entry, args.dry_run)
        launched += 1
    print(f"{launched} job(s) {'printed' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
