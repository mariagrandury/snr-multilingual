#!/usr/bin/env python3
"""Pretraining progress dashboard.

Scans the Megatron checkpoints tree and prints, for each model, the latest
iteration saved, the number of iter_* dirs on disk, and a progress bar
toward --target (default 50000). The companion to scripts/snr_progress.py
(which tracks evaluation progress).

Examples:
    python3.11 scripts/pretrain_progress.py
    python3.11 scripts/pretrain_progress.py --filter seed1904
    python3.11 scripts/pretrain_progress.py --all                # include non-canonical exp dirs
    python3.11 scripts/pretrain_progress.py --target 50000
    python3.11 scripts/pretrain_progress.py --plot progress.png   # heatmap across the 4 stages
    python3.11 scripts/pretrain_progress.py --plot progress.png --no-hub  # skip Hub query
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

CKPT_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small"
)
HF_STAGING_ROOT = Path("/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints")
# Per-seed org (storage quota is per-account, so we shard the 36 cells across
# three accounts: snr-models-1904, snr-models-1797, snr-models-28).
def hf_org_for_seed(seed: int) -> str:
    return f"snr-models-{seed}"
CANONICAL_RE = re.compile(r"^apertus-(175M|350M|600M|1B)-fwEdu\d+-fw2\d+-seed\d+$")
ITER_RE = re.compile(r"^iter_(\d+)$")

# The 13 canonical eval iters used everywhere in this project (convert-snr.sh,
# the SNR runner, etc.). Plot uses these as columns.
CANONICAL_ITERS = [
    2000, 6000, 12000, 18000, 22000, 28000, 34000,
    38000, 42000, 44000, 46000, 48000, 50000,
]
SIZES = ["175M", "350M", "600M", "1B"]
MIXES = [(30, 70), (60, 40), (90, 10)]
SEEDS = [1904, 1797, 28]


def is_valid_iter_dir(iter_dir: Path) -> bool:
    """A torch_dist checkpoint is loadable only when the dir contains both
    a .metadata file AND at least one .distcp shard. Empty dirs and "shell"
    dirs (.metadata + common.pt + metadata.json with no shards) are not."""
    if not iter_dir.is_dir():
        return False
    if not (iter_dir / ".metadata").is_file():
        return False
    for p in iter_dir.iterdir():
        if p.suffix == ".distcp":
            return True
    return False


def model_progress(
    model_dir: Path,
) -> tuple[int | None, int, int | None, int | None]:
    """Return (marker, n_iter_dirs, max_iter_dir, max_valid_iter_dir).

    `max_valid_iter_dir` is the largest iter_X dir whose contents pass
    `is_valid_iter_dir` — i.e. the latest checkpoint that can actually be
    resumed. If `marker > max_valid_iter_dir`, the marker file points at a
    corrupt/incomplete dir and the model is not resumable from `marker`.
    """
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, 0, None, None

    iters: list[int] = []
    valid_iters: list[int] = []
    for entry in ckpt_dir.iterdir():
        m = ITER_RE.match(entry.name)
        if m and entry.is_dir():
            n = int(m.group(1))
            iters.append(n)
            if is_valid_iter_dir(entry):
                valid_iters.append(n)

    marker_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    marker = None
    if marker_file.is_file():
        try:
            marker = int(marker_file.read_text().strip())
        except ValueError:
            marker = None

    max_iter = max(iters) if iters else None
    max_valid = max(valid_iters) if valid_iters else None
    return marker, len(iters), max_iter, max_valid


def render_bar(done: int, total: int, width: int = 25) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = max(0, min(width, int(round(width * done / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Plot: 3-panel heatmap over (size×mix) × canonical_iter, one panel per seed.
# Color encodes the highest pipeline stage reached:
#   0 (red)         missing in local megatron
#   1 (orange)      megatron ckpt valid on disk
#   2 (yellow)      converted HF ckpt staged at snr-hf-checkpoints/
#   3 (light green) pushed as a stage1-step-N branch under snr-models/
# ---------------------------------------------------------------------------

def _has_hf_staged(cell: str, step: int, hf_root: Path = HF_STAGING_ROOT) -> bool:
    d = hf_root / cell / f"iter_{step:07d}"
    return (d / "config.json").is_file() and any(d.glob("model.safetensors*"))


def _hub_branches_for_cell(api, cell: str, seed: int,
                           max_attempts: int = 6) -> set[str]:
    """Return the set of branch names on snr-models-<seed>/<cell>. Empty set on
    repo-not-found (404) or unrecoverable errors. 429-aware retry."""
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    repo_id = f"{hf_org_for_seed(seed)}/{cell}"
    delay = 10.0
    for attempt in range(max_attempts):
        try:
            refs = api.list_repo_refs(repo_id, repo_type="model")
            return {b.name for b in refs.branches}
        except RepositoryNotFoundError:
            return set()
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None) if e.response else None
            if status == 404:
                return set()
            if status == 429:
                wait = None
                if e.response is not None:
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = float(ra)
                        except ValueError:
                            pass
                if wait is None:
                    wait = delay
                    delay *= 2
                wait += 5
                print(f"[plot] 429 on {repo_id}; sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            print(f"[plot] giving up on {repo_id}: {e}")
            return set()
    return set()


def _build_status_matrix(seed: int, hub_branches: dict[str, set[str]],
                        ckpt_root: Path = CKPT_ROOT) -> "list[list[int]]":
    """Return a 12 × len(CANONICAL_ITERS) matrix of status codes for one seed."""
    matrix: list[list[int]] = []
    for size in SIZES:
        for fw_edu, fw2 in MIXES:
            cell = f"apertus-{size}-fwEdu{fw_edu}-fw2{fw2}-seed{seed}"
            row: list[int] = []
            for step in CANONICAL_ITERS:
                meg_ok = is_valid_iter_dir(
                    ckpt_root / cell / "checkpoints" / f"iter_{step:07d}"
                )
                if not meg_ok:
                    row.append(0)
                    continue
                hf_ok = _has_hf_staged(cell, step)
                if not hf_ok:
                    row.append(1)
                    continue
                branch = f"stage1-step-{step:05d}"
                if branch in hub_branches.get(cell, set()):
                    row.append(3)
                else:
                    row.append(2)
            matrix.append(row)
    return matrix


def make_plot(out_path: Path, query_hub: bool = True,
              ckpt_root: Path = CKPT_ROOT) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Pre-fetch hub branches for all 36 cells (or skip if --no-hub).
    hub_branches: dict[str, set[str]] = {}
    if query_hub:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        cells = [
            (f"apertus-{size}-fwEdu{e}-fw2{w}-seed{seed}", seed)
            for size in SIZES
            for e, w in MIXES
            for seed in SEEDS
        ]
        print(f"[plot] querying {len(cells)} repos across {sorted({hf_org_for_seed(s) for _, s in cells})}...")
        for cell, seed in cells:
            hub_branches[cell] = _hub_branches_for_cell(api, cell, seed)
    else:
        print("[plot] --no-hub: skipping Hub query (all branches treated as missing)")

    cmap = ListedColormap(["#cc4040", "#ff9933", "#f4d03f", "#90ee90"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    row_labels = [f"{s}-fwEdu{e}/{w}" for s in SIZES for e, w in MIXES]
    col_labels = [f"{i // 1000}k" for i in CANONICAL_ITERS]

    fig, axes = plt.subplots(
        1, len(SEEDS),
        figsize=(5 * len(SEEDS) + 1, 5.5),
        sharey=True,
    )
    if len(SEEDS) == 1:
        axes = [axes]

    for ax, seed in zip(axes, SEEDS):
        matrix = _build_status_matrix(seed, hub_branches, ckpt_root)
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(CANONICAL_ITERS)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("canonical iter")
        # Light grid between cells.
        ax.set_xticks([x - 0.5 for x in range(1, len(CANONICAL_ITERS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    legend_handles = [
        Patch(color="#cc4040", label="missing"),
        Patch(color="#ff9933", label="megatron (local)"),
        Patch(color="#f4d03f", label="HF format (local)"),
        Patch(color="#90ee90", label="pushed to hub"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Pretraining progress: megatron → HF → snr-models hub", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--root",
        default=str(CKPT_ROOT),
        help=f"Megatron run root (default: {CKPT_ROOT})",
    )
    p.add_argument("--filter", default=None, help="Substring filter on model dir name.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Include non-canonical experiment directories (default: only "
             "apertus-{175M,350M,600M,1B}-fwEdu*-fw2*-seed* dirs).",
    )
    p.add_argument("--target", type=int, default=50000, help="Target iteration for progress bar.")
    p.add_argument(
        "--plot",
        metavar="PATH",
        help="Write a 3-panel heatmap (one per seed) showing per-cell stage "
             "status (missing → megatron → HF → hub) to PATH and exit.",
    )
    p.add_argument(
        "--no-hub",
        action="store_true",
        help="With --plot, skip the HF Hub branch lookup (treat all branches "
             "as missing). Useful when the Hub API is rate-limited.",
    )
    args = p.parse_args()

    if args.plot:
        make_plot(Path(args.plot), query_hub=not args.no_hub, ckpt_root=Path(args.root))
        return

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    models = sorted(d for d in root.iterdir() if d.is_dir())
    if not args.all:
        models = [d for d in models if CANONICAL_RE.match(d.name)]
    if args.filter:
        models = [d for d in models if args.filter in d.name]

    if not models:
        print("No matching model directories.")
        return

    name_w = max(len(d.name) for d in models)
    done_count = 0
    corrupt_count = 0
    for d in models:
        marker, n_iters, max_iter, max_valid = model_progress(d)
        latest = marker if marker is not None else max_iter
        # corrupt: marker (or fallback max_iter) points at an iter that is
        # not actually loadable. Either no valid iter exists at all, or the
        # latest valid iter is older than what marker claims.
        corrupt = (latest is not None) and (
            max_valid is None or max_valid < latest
        )
        bar = render_bar((max_valid if corrupt else latest) or 0, args.target)
        latest_s = f"{latest}" if latest is not None else "-"
        if latest is not None and latest >= args.target and not corrupt:
            done_count += 1
            tag = "[done]"
        elif latest is None:
            tag = "[no_ckpts]"
        elif corrupt:
            corrupt_count += 1
            valid_s = f"{max_valid}" if max_valid is not None else "none"
            tag = f"[corrupt] (latest valid: {valid_s})"
        else:
            tag = "[in_progress]"
        print(
            f"  {bar} {latest_s:>6} / {args.target}   "
            f"saved={n_iters:>3}  {d.name:<{name_w}}  {tag}"
        )

    msg = (
        f"\nSummary: {done_count}/{len(models)} models reached iter {args.target} "
        f"({100 * done_count / len(models):.0f}%)"
    )
    if corrupt_count:
        msg += f" — {corrupt_count} corrupt"
    print(msg)


if __name__ == "__main__":
    main()
