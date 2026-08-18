#!/usr/bin/env python3
"""
Sync configs/models.json with the predictivity grid — one entry per cell.

The eval chain resolves everything through configs/models.json:
conversion/convert-snr.sh --models reads `backends.megatron`, and
push_all_results.py refuses a NAME (`<cell>-iter<N>`) whose cell has no
entry — it needs the `stages` to compute tokens/FLOPs. The entries are
derived from the same grid + hyperparams the launcher uses, so the two can
never disagree.

Normally you never run this by hand: **both auto-eval watchers call sync()
at the start of every pass**, so the registry follows the grid
automatically. The CLI exists for explicit use after editing the grid
(commit the resulting models.json diff). Idempotent; entries of other
variants (e.g. an earlier --arch shallow run) are left untouched.

Usage:
    python sync_models_json.py                 # the 52 baseline cells
    python sync_models_json.py --arch shallow  # + the shallow variant's cells
    python sync_models_json.py --dry-run       # show what would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from launch_trainings import (  # noqa: E402
    HYPERPARAMS, SCHEME_B_LANGS, exp_name, mix_label, predictivity_cells,
    schedule_for)

MODELS_JSON = SCRIPT_DIR.parent.parent / "configs" / "models.json"

SOURCE = "snr-pretraining-predictivity"
TOKENS_PER_ITER = 504 * 4096
VOCAB_SIZE = 131072
SAVE_INTERVAL = 2000

# Where the CSCS artifacts live (Azure cells keep the same entry shape; their
# checkpoints are watched in blob storage by auto_evals_azure.py instead).
MEG_BASE = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr"
HF_LOCAL_BASE = "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints"


def save_points(target: int) -> list[int]:
    """Every iter Megatron writes on the way to `target` (SAVE_INTERVAL grid
    plus the final step, which ends off-grid for every predictivity size)."""
    pts = list(range(SAVE_INTERVAL, target + 1, SAVE_INTERVAL))
    if not pts or pts[-1] != target:
        pts.append(target)
    return pts


def cell_entry(cfg: dict, c: dict, arch: str, scheme: str) -> tuple[str, dict]:
    name = exp_name(c["size"], c["L"], arch, c["seed"], scheme)
    target = schedule_for(cfg)[0]
    return name, {
        "source": SOURCE,
        # Cross-size identity (the size token is what varies along the ladder).
        "family": f"apertus-{mix_label(c['L'], arch, scheme)}-seed{c['seed']}",
        "size": c["size"],
        # Total parameters = non-embedding + the tied embedding matrix.
        "params": int(cfg["n_non_emb_params"] + VOCAB_SIZE * cfg["hidden_size"]),
        "hyperparams_key": c["size"],
        "L": c["L"],
        "arch": arch,
        "scheme": scheme,
        "seed": c["seed"],
        "checkpoint_kind": "megatron_iter",
        "backends": {
            "megatron": f"{MEG_BASE}/{name}/checkpoints/",
            "hf_local": f"{HF_LOCAL_BASE}/{name}/",
        },
        "stages": {
            "pretraining": {
                "tokens": target * TOKENS_PER_ITER,
                "num_iters": target,
                "tokens_per_iter": TOKENS_PER_ITER,
                "checkpoints": {"final": target, "all": save_points(target)},
            },
        },
    }


def sync(arch: str = "deep", scheme: str = "A",
         write: bool = True) -> tuple[list[str], list[str]]:
    """Upsert the variant's cell entries; returns (added, updated) names.
    A no-op (and no write) when models.json already matches the grid — the
    watchers call this at the start of every pass."""
    data = json.loads(MODELS_JSON.read_text())
    configs = json.loads(HYPERPARAMS[arch].read_text())["configs"]

    added, updated = [], []
    for c in predictivity_cells():
        # Scheme normalization mirrors the launcher: scheme B cells only exist
        # where B differs from A.
        cell_scheme = scheme if c["L"] in SCHEME_B_LANGS else "A"
        name, entry = cell_entry(configs[c["size"]], c, arch, cell_scheme)
        old = data["models"].get(name)
        if old == entry:
            continue
        (updated if old is not None else added).append(name)
        data["models"][name] = entry

    sources_missing = SOURCE not in data.get("sources", {})
    data.setdefault("sources", {}).setdefault(SOURCE, {"split": None})

    if write and (added or updated or sources_missing):
        MODELS_JSON.write_text(json.dumps(data, indent=2) + "\n")
    return added, updated


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", choices=["deep", "shallow"], default="deep")
    p.add_argument("--scheme", choices=["A", "B"], default="A")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    added, updated = sync(args.arch, args.scheme, write=not args.dry_run)
    print(f"added {len(added)}, updated {len(updated)} "
          f"(of {len(predictivity_cells())} cells, arch={args.arch} "
          f"scheme={args.scheme})")
    for n in added:
        print(f"  + {n}")
    for n in updated:
        print(f"  ~ {n}")
    if args.dry_run:
        print("(dry-run — models.json not written)")
    elif added or updated:
        print(f"wrote {MODELS_JSON} — remember to commit the diff")


if __name__ == "__main__":
    main()
