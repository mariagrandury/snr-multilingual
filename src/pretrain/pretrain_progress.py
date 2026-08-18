#!/usr/bin/env python3
"""Predictivity-sweep pretraining progress on CSCS: per-cell status + plots.

Prints one tab-separated status line per cell of the selected variant
(`--arch`/`--scheme`). Each cell's target is its size's own 5xC budget (the
"predictivity" block in hyperparams/hyperparams_{deep,shallow}.json):

    <model>\tdone
    <model>\tfresh\t<target>
    <model>\tcorrupt\t<n_iters>
    <model>\tresume\t<load_iter>\t<target>

A cell is as far along as its latest valid checkpoint: `done` when that
checkpoint has reached the target, `resume` from it otherwise. The same
decision function (`cell_action`) drives launch_trainings.py's idempotent
submission (which also refreshes the plots below on every CSCS launch), so
this output is exactly what a (re-)launch would do.

With --plot, renders two heatmaps over the sweep grid (x = model size,
y = number of languages), aggregated over EVERY run found on disk regardless
of variant:

  pretrain_progress_simple.png    cell value = how many finished models exist
                                  at (size, L) across all variants (seeds,
                                  deep/shallow, scheme A/B, tokenizers).
  pretrain_progress_detailed.png  one row of binary heatmaps per
                                  transformation — SEED (28/1797/1904),
                                  ARCH (deep/shallow), SCHEME (A/B),
                                  TOKENIZER (v1) — yellow 0 / blue 1.

Azure cells are not visible here (their checkpoints live in blob storage —
auto_evals.py watches those); this tool covers the CSCS half of the sweep.

Examples:
    python3.11 pretrain_progress.py
    python3.11 pretrain_progress.py --filter seed1904
    python3.11 pretrain_progress.py --arch shallow --plot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from launch_trainings import (  # noqa: E402
    HYPERPARAMS, LANG_SETTINGS, SCHEME_B_LANGS, SIZE_LANG_SETTINGS, exp_name,
    predictivity_cells, schedule_for)

# Megatron writes checkpoints under Meg-Runs/<PROJECT_NAME>/<EXP_NAME>/
# (launch_pretraining_cscs.sh); PROJECT_NAME for the predictivity sweep is
# msnr (configs/hf_wandb.json).
CKPT_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr"
)
ITER_RE = re.compile(r"^iter_(\d+)$")
# The canonical cell name (see launch_trainings.exp_name).
NAME_RE = re.compile(
    r"^apertus-(?P<size>90M|175M|350M|600M|1B|1\.7B)-L(?P<L>\d+)"
    r"(?P<scheme>-schemeB)?(?P<arch>-shallow)?-seed(?P<seed>\d+)$"
)

SIZES = list(SIZE_LANG_SETTINGS)  # 90M .. 1.7B, grid order


def is_valid_iter_dir(iter_dir: Path) -> bool:
    """The single source of truth for "is this iter dir loadable?".

    Shell call sites (conversion/convert-snr.sh) invoke this via the
    ``--is-valid <iter_dir>`` CLI at the bottom of this file; Python callers
    (launch_trainings.py) import it directly. Centralising the check means
    future tightening (e.g. byte-level shard-name parsing of .metadata) only
    needs to land here.

    KNOWN LIMITATION (tracked, do-not-fix-here): the loose ".metadata +
    ≥1 .distcp" check passes some dirs that Megatron's load_checkpoint
    can't actually load — hit on iter_0002000 of a seed1904 cell
    (2026-05-14): 41 shards on disk, same set as a known-good iter, yet
    the resume crashed in dist_checkpointing.load. Tighter byte-level
    .metadata parses over-rejected good iters, so this stays loose.
    """
    if not iter_dir.is_dir():
        return False
    if not (iter_dir / ".metadata").is_file():
        return False
    for p in iter_dir.iterdir():
        if p.suffix == ".distcp":
            return True
    return False


def model_progress(model_dir: Path) -> tuple[int, int | None]:
    """Return (n_iter_dirs, max_valid_iter) for a cell.

    `max_valid_iter` is the largest iter_X dir whose contents pass
    `is_valid_iter_dir` — i.e. the latest checkpoint that can actually be
    resumed (None when no iter dir on disk is loadable).
    """
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return 0, None

    n_iters = 0
    max_valid = None
    for entry in ckpt_dir.iterdir():
        m = ITER_RE.match(entry.name)
        if m and entry.is_dir():
            n_iters += 1
            n = int(m.group(1))
            if (max_valid is None or n > max_valid) and is_valid_iter_dir(entry):
                max_valid = n
    return n_iters, max_valid


# ---------------------------------------------------------------------------
# The per-cell decision — shared by this tool's status output and by
# launch_trainings.py's idempotent submission. A cell is simply as far along
# as its latest valid checkpoint:
#
#   • latest valid checkpoint >= target     → ("done", 0, 0)
#   • no iter dirs on disk                  → ("fresh", target, 0)
#   • iter dirs exist but none loadable     → ("corrupt", n_iters, 0) — we
#     never wipe; corrupt cells are left for manual review
#   • otherwise                             → ("resume", max_valid, target)
# ---------------------------------------------------------------------------

def cell_action(model_dir: Path, target: int) -> tuple[str, int, int]:
    n_iters, max_valid = model_progress(model_dir)
    if max_valid is None:
        return ("corrupt", n_iters, 0) if n_iters else ("fresh", target, 0)
    if max_valid >= target:
        return ("done", 0, 0)
    return ("resume", max_valid, target)


def sweep_cells(arch: str, scheme: str = "A") -> list[tuple[str, int]]:
    """(exp_name, target_iters) for every cell of one variant, in grid order.
    Scheme B only exists at SCHEME_B_LANGS — other settings are always the
    scheme-A cell (same normalization as the launcher)."""
    configs = json.loads(HYPERPARAMS[arch].read_text())["configs"]
    return [
        (exp_name(c["size"], c["L"], arch, c["seed"],
                  scheme if c["L"] in SCHEME_B_LANGS else "A"),
         schedule_for(configs[c["size"]])[0])
        for c in predictivity_cells()
    ]


def emit_actions(arch: str, scheme: str, root: Path = CKPT_ROOT,
                 filter_substr: str | None = None) -> None:
    for name, target in sweep_cells(arch, scheme):
        if filter_substr and filter_substr not in name:
            continue
        action, a, b = cell_action(root / name, target)
        if action == "done":
            print(f"{name}\tdone")
        elif action == "fresh":
            print(f"{name}\tfresh\t{a}")
        elif action == "corrupt":
            print(f"{name}\tcorrupt\t{a}")
        else:
            print(f"{name}\tresume\t{a}\t{b}")


# ---------------------------------------------------------------------------
# Plots: the sweep grid (x = size, y = number of languages), aggregated over
# every run found on disk — any seed, arch, scheme, tokenizer.
# ---------------------------------------------------------------------------

def _targets() -> dict[tuple[str, str], int]:
    """(arch, size) -> target iters, from both reviewed hyperparams files."""
    out = {}
    for arch, path in HYPERPARAMS.items():
        configs = json.loads(path.read_text())["configs"]
        for size, cfg in configs.items():
            out[(arch, size)] = schedule_for(cfg)[0]
    return out


def scan_runs(root: Path) -> list[dict]:
    """Every canonical run dir on disk, parsed into its variant coordinates,
    with `done` = its latest valid checkpoint reached its size's target."""
    if not root.is_dir():
        return []
    targets = _targets()
    runs = []
    for entry in sorted(root.iterdir()):
        m = NAME_RE.match(entry.name)
        if not m:
            continue
        arch = "shallow" if m["arch"] else "deep"
        target = targets[(arch, m["size"])]
        _, max_valid = model_progress(entry)
        runs.append({
            "name": entry.name,
            "size": m["size"],
            "L": int(m["L"]),
            "seed": int(m["seed"]),
            "arch": arch,
            "scheme": "B" if m["scheme"] else "A",
            "tokenizer": "v1",
            "done": (max_valid or 0) >= target,
        })
    return runs


def _grid_matrix(runs: list[dict], predicate) -> "list[list[float]]":
    """len(LANG_SETTINGS) x len(SIZES) matrix: count of runs matching
    `predicate` at each (L, size); NaN where (size, L) is not in the grid."""
    matrix = []
    for L in LANG_SETTINGS:
        row = []
        for size in SIZES:
            if L not in SIZE_LANG_SETTINGS[size]:
                row.append(float("nan"))
            else:
                row.append(sum(1 for r in runs
                               if r["size"] == size and r["L"] == L
                               and predicate(r)))
        matrix.append(row)
    return matrix


def _draw_grid(ax, matrix, cmap, vmax, annotate=True):
    import numpy as np
    arr = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(arr)
    cmap.set_bad("#d9d9d9")  # (size, L) not in the sweep grid
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES, fontsize=8)
    ax.set_yticks(range(len(LANG_SETTINGS)))
    ax.set_yticklabels(LANG_SETTINGS, fontsize=8)
    ax.set_xticks([x - 0.5 for x in range(1, len(SIZES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(LANG_SETTINGS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for i in range(len(LANG_SETTINGS)):
            for j in range(len(SIZES)):
                if arr[i][j] == arr[i][j]:  # not NaN
                    ax.text(j, i, int(arr[i][j]), ha="center", va="center",
                            fontsize=8,
                            color="white" if arr[i][j] > vmax / 2 else "black")


def update_plots(root: Path = CKPT_ROOT, out_dir: Path = SCRIPT_DIR) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    runs = scan_runs(root)
    done = [r for r in runs if r["done"]]

    # --- simple: count of finished models per (size, L), all variants -------
    matrix = _grid_matrix(done, lambda r: True)
    vmax = max(3, max((v for row in matrix for v in row if v == v), default=0))
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_grid(ax, matrix, plt.get_cmap("Blues").copy(), vmax)
    ax.set_xlabel("model size (non-embedding)")
    ax.set_ylabel("number of languages")
    ax.set_title(f"Finished models per grid cell — all variants "
                 f"({len(done)}/{len(runs)} runs done)")
    fig.tight_layout()
    simple_path = out_dir / "pretrain_progress_simple.png"
    fig.savefig(simple_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {simple_path}", file=sys.stderr)

    # --- detailed: one row of binary heatmaps per transformation ------------
    rows = [
        ("SEED", "seed", [28, 1797, 1904]),
        ("ARCH", "arch", ["deep", "shallow"]),
        ("SCHEME", "scheme", ["A", "B"]),
        ("TOKENIZER", "tokenizer", ["v1"]),
    ]
    ncols = max(len(values) for _, _, values in rows)
    binary_cmap = ListedColormap(["#f4d03f", "#3b6fb6"])  # 0 yellow, 1 blue

    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(4.2 * ncols, 3.6 * len(rows)))
    for i, (label, key, values) in enumerate(rows):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(values):
                ax.axis("off")
                continue
            value = values[j]
            matrix = _grid_matrix(done, lambda r, k=key, v=value: r[k] == v)
            # Binary: 1 if any finished run with this factor value.
            matrix = [[min(v, 1) if v == v else v for v in row] for row in matrix]
            if key == "scheme" and value == "B":
                # Scheme B only exists where its language sets differ from A
                # ({8, 15, 30}) — grey the rest out like off-grid cells.
                matrix = [[v if L in SCHEME_B_LANGS else float("nan")
                           for v in row]
                          for L, row in zip(LANG_SETTINGS, matrix)]
            _draw_grid(ax, matrix, binary_cmap.copy(), 1, annotate=False)
            ax.set_title(f"{label} = {value}", fontsize=10)
            if j == 0:
                ax.set_ylabel("number of languages", fontsize=8)
            ax.set_xlabel("model size", fontsize=8)
    fig.suptitle("Finished models by transformation (yellow 0 / blue 1; "
                 "grey = not in grid)", y=1.0)
    fig.tight_layout()
    detailed_path = out_dir / "pretrain_progress_detailed.png"
    fig.savefig(detailed_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {detailed_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default=str(CKPT_ROOT),
                   help=f"Megatron run root (default: {CKPT_ROOT})")
    p.add_argument("--arch", choices=["deep", "shallow"], default="deep",
                   help="Which architecture family's cells to report")
    p.add_argument("--scheme", choices=["A", "B"], default="A",
                   help="Which language-scheme variant's cells to report")
    p.add_argument("--filter", default=None, help="Substring filter on cell name.")
    p.add_argument("--plot", action="store_true",
                   help="Also render pretrain_progress_{simple,detailed}.png "
                        "(these aggregate over ALL variants, not just "
                        "--arch/--scheme)")
    args = p.parse_args()

    emit_actions(args.arch, args.scheme, root=Path(args.root),
                 filter_substr=args.filter)
    if args.plot:
        update_plots(root=Path(args.root))


if __name__ == "__main__":
    # CLI subcommand for shell scripts that need the canonical validity check:
    #   python3.11 pretrain_progress.py --is-valid <iter_dir>
    # Returns exit 0 if the dir is a loadable torch_dist checkpoint, 1 otherwise.
    if len(sys.argv) >= 2 and sys.argv[1] == "--is-valid":
        if len(sys.argv) != 3:
            print("usage: pretrain_progress.py --is-valid <iter_dir>", file=sys.stderr)
            sys.exit(2)
        sys.exit(0 if is_valid_iter_dir(Path(sys.argv[2])) else 1)
    main()
