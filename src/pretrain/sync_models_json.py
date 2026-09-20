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
(commit the resulting models.json diff). Idempotent; entries outside the
--arch/--scheme filter (e.g. the shallow cells under --arch deep) are left
untouched.

Usage:
    python sync_models_json.py                 # every deep cell, all schemes
    python sync_models_json.py --arch shallow  # + the shallow ladder's cells
    python sync_models_json.py --dry-run       # show what would change
    python sync_models_json.py --prune         # drop entries the grid lost

sync() only upserts, so a grid edit (seeds, x3 placement) leaves the old
cells' entries behind — and the Azure watcher enumerates cells FROM
models.json (filter_models), so stale entries are scanned and reported as
real cells every pass. --prune removes predictivity entries no scheme of
the current grid defines; it is a manual step because an entry may belong
to a run that already produced checkpoints or eval results under the old
grid — check before pruning, and keep such entries by reverting the diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from launch_trainings import (  # noqa: E402
    DATA_SCHEMES, HYPERPARAMS, arches_for, exp_name, mix_label, predictivity_cells,
    save_interval, schedule_for)

MODELS_JSON = SCRIPT_DIR.parent.parent / "configs" / "models.json"

SOURCE = "snr-pretraining-predictivity"
TOKENS_PER_ITER = 504 * 4096
VOCAB_SIZE = 131072

# Where the CSCS artifacts live (Azure cells keep the same entry shape; their
# checkpoints are watched in blob storage by auto_evals_azure.py instead).
MEG_BASE = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr"
# Where convert-snr.sh actually lands the HF snapshots (<cell>/iter_<NNNNNNN>/):
# auto_evals_cscs.DEFAULT_STAGING. snr-hf-checkpoints on iopsstor was the
# 36-sweep's root and is never written for lm-* cells.
HF_LOCAL_BASE = "/capstor/store/cscs/swissai/infra01/msnr-hf-models"


def save_points(target: int) -> list[int]:
    """Every iter Megatron writes on the way to `target` (the cell's per-size
    save-interval grid — launch_trainings.save_interval — plus the final
    step, which ends off-grid for the 2000-capped sizes)."""
    step = save_interval(target)
    pts = list(range(step, target + 1, step))
    if not pts or pts[-1] != target:
        pts.append(target)
    return pts


def cell_entry(cfg: dict, c: dict, arch: str, scheme: str) -> tuple[str, dict]:
    name = exp_name(c["size"], c["L"], arch, c["seed"], scheme)
    target = schedule_for(cfg)[0]
    return name, {
        "source": SOURCE,
        # Cross-size identity (the size token is what varies along the ladder).
        "family": f"lm-{mix_label(c['L'], arch, scheme)}-seed{c['seed']}",
        "size": c["size"],
        # Total parameters = non-embedding + the tied embedding matrix — which
        # is exactly the count the FLOPs convention wants
        # (configs.flops_params: 6 x (N_non_emb + d_model x V) x D). The three
        # shape fields put the cell explicitly on that basis.
        "params": int(cfg["n_non_emb_params"] + VOCAB_SIZE * cfg["hidden_size"]),
        "n_non_emb": int(cfg["n_non_emb_params"]),
        "d_model": int(cfg["hidden_size"]),
        "vocab_size": VOCAB_SIZE,
        "hyperparams_key": c["size"],
        "L": c["L"],
        "arch": arch,
        # Kept under the historical key `scheme`, which now holds the data
        # SCHEME (A / AT3 / B / ZH / ES). `temperature` is broken out beside
        # it because AT3 differs from A by allocation alone — its language list
        # is byte-identical, so nothing else in the entry would reveal that two
        # cells were trained on different distributions.
        "scheme": scheme,
        "temperature": DATA_SCHEMES[scheme]["temp"],
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


def grid_names() -> set[str]:
    """Every cell name the current grid can produce, over every data scheme
    and the architectures each is trained in — the keep-set for --prune. A
    scheme absent from this set gets its models.json entries deleted, so it
    must enumerate the whole grid, not one slice of it."""
    return {exp_name(c["size"], c["L"], arch, c["seed"], c["scheme"])
            for c in predictivity_cells()
            for arch in arches_for(c["scheme"], c["size"], c["L"])}


def prune(write: bool = True) -> list[str]:
    """Remove predictivity entries no scheme of the grid defines; returns
    the removed names. Manual (CLI --prune), not part of sync() — see the
    module docstring."""
    data = json.loads(MODELS_JSON.read_text())
    keep = grid_names()
    stale = [n for n, e in data["models"].items()
             if e.get("source") == SOURCE and n not in keep]
    for n in stale:
        del data["models"][n]
    if write and stale:
        MODELS_JSON.write_text(json.dumps(data, indent=2) + "\n")
    return stale


def sync(arch: str = "deep", scheme: str | None = None,
         write: bool = True) -> tuple[list[str], list[str]]:
    """Upsert cell entries for one architecture; returns (added, updated)
    names. `scheme` narrows to a single data scheme, otherwise every scheme
    trained in this arch is upserted — the watchers want all of them, and
    entries that do not exist are cells the conversion and W&B push cannot
    resolve. A no-op (and no write) when models.json already matches."""
    data = json.loads(MODELS_JSON.read_text())
    configs = json.loads(HYPERPARAMS[arch].read_text())["configs"]

    added, updated = [], []
    for c in predictivity_cells([scheme] if scheme else None):
        if arch not in arches_for(c["scheme"], c["size"], c["L"]):
            continue
        name, entry = cell_entry(configs[c["size"]], c, arch, c["scheme"])
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
    p.add_argument("--scheme", choices=list(DATA_SCHEMES), default=None,
                   help="Only this data scheme (default: every scheme "
                        "trained in --arch)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prune", action="store_true",
                   help="also remove predictivity entries the grid no longer "
                        "defines (any arch/scheme) — check they have no "
                        "artifacts first")
    args = p.parse_args()

    if args.prune:
        for n in prune(write=not args.dry_run):
            print(f"  - {n} (not in the current grid)")
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
