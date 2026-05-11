#!/usr/bin/env python3
"""Pretraining progress dashboard.

Scans the Megatron checkpoints tree and prints, for each model, the latest
iteration saved, the number of iter_* dirs on disk, and a progress bar
toward --target (default 50000). The companion to scripts/snr_progress.py
(which tracks evaluation progress).

Examples:
    python3.11 pretrain_progress.py
    python3.11 pretrain_progress.py --filter seed1904
    python3.11 pretrain_progress.py --all                # include non-canonical exp dirs
    python3.11 pretrain_progress.py --target 50000
    python3.11 pretrain_progress.py --plot progress.png   # writes progress.png + progress_all.png
    python3.11 pretrain_progress.py --plot progress.png --no-hub  # skip Hub query
    python3.11 pretrain_progress.py --actions             # machine-readable per-model action
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

# Per-seed iter policy (single source of truth for the whole pipeline:
# pretraining, conversion, eval). Mirrors evals/scripts/snr_progress.py's
# ITERS_SEED1904 / ITERS_OTHER — keep both files in sync when this changes.
ITERS_SEED1904 = [6000, 12000, 22000, 28000, 42000, 44000, 46000, 48000, 50000]
ITERS_OTHER    = [6000, 10000, 20000, 30000, 42000, 44000, 46000, 48000, 50000]

def canonical_iters_for_seed(seed: int) -> list[int]:
    return ITERS_SEED1904 if seed == 1904 else ITERS_OTHER

# Union of all per-seed iters — used by the canonical-stage heatmap (columns).
CANONICAL_ITERS = sorted(set(ITERS_SEED1904) | set(ITERS_OTHER))
# All training save points: Megatron writes every 2000 iters (CHECKPOINT_STEPS).
CHECKPOINT_INTERVAL = 2000
TARGET_ITER = 50000
ALL_ITERS = list(range(CHECKPOINT_INTERVAL, TARGET_ITER + 1, CHECKPOINT_INTERVAL))
SIZES = ["175M", "350M", "600M", "1B"]
MIXES = [(30, 70), (60, 40), (90, 10)]
SEEDS = [1904, 1797, 28]


def _seed_of(cell_name: str) -> int:
    """Extract the seed int from a canonical cell name."""
    m = re.search(r"-seed(\d+)$", cell_name)
    return int(m.group(1)) if m else 0


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
) -> tuple[int | None, int, int | None, int | None, set[int]]:
    """Return (marker, n_iter_dirs, max_iter_dir, max_valid_iter_dir, valid_iters).

    `max_valid_iter_dir` is the largest iter_X dir whose contents pass
    `is_valid_iter_dir` — i.e. the latest checkpoint that can actually be
    resumed. If `marker > max_valid_iter_dir`, the marker file points at a
    corrupt/incomplete dir and the model is not resumable from `marker`.
    `valid_iters` is the set of all iter numbers on disk that are valid.
    """
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None, 0, None, None, set()

    iters: list[int] = []
    valid_iters: set[int] = set()
    for entry in ckpt_dir.iterdir():
        m = ITER_RE.match(entry.name)
        if m and entry.is_dir():
            n = int(m.group(1))
            iters.append(n)
            if is_valid_iter_dir(entry):
                valid_iters.add(n)

    marker_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    marker = None
    if marker_file.is_file():
        try:
            marker = int(marker_file.read_text().strip())
        except ValueError:
            marker = None

    max_iter = max(iters) if iters else None
    max_valid = max(valid_iters) if valid_iters else None
    return marker, len(iters), max_iter, max_valid, valid_iters


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
#
# A second plot covers ALL 2000-step iters (megatron presence only) — this is
# the operational view the resume launcher uses to spot gaps.
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


def _build_all_iters_matrix(seed: int, ckpt_root: Path = CKPT_ROOT) -> "list[list[int]]":
    """Return a 12 × len(ALL_ITERS) matrix of megatron presence (0 missing, 1 valid)."""
    matrix: list[list[int]] = []
    for size in SIZES:
        for fw_edu, fw2 in MIXES:
            cell = f"apertus-{size}-fwEdu{fw_edu}-fw2{fw2}-seed{seed}"
            row: list[int] = []
            for step in ALL_ITERS:
                meg_ok = is_valid_iter_dir(
                    ckpt_root / cell / "checkpoints" / f"iter_{step:07d}"
                )
                row.append(1 if meg_ok else 0)
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
    plt.close(fig)
    print(f"[plot] saved {out_path}")

    # Companion plot covering every 2000-step ckpt — the operational view.
    all_path = out_path.with_name(out_path.stem + "_all" + out_path.suffix)
    _make_all_iters_plot(all_path, ckpt_root)


def _make_all_iters_plot(out_path: Path, ckpt_root: Path = CKPT_ROOT) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(["#cc4040", "#ff9933"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    row_labels = [f"{s}-fwEdu{e}/{w}" for s in SIZES for e, w in MIXES]
    canonical_set = set(CANONICAL_ITERS)
    col_labels = [f"{i // 1000}k" for i in ALL_ITERS]

    fig, axes = plt.subplots(
        1, len(SEEDS),
        figsize=(7 * len(SEEDS) + 1, 5.5),
        sharey=True,
    )
    if len(SEEDS) == 1:
        axes = [axes]

    for ax, seed in zip(axes, SEEDS):
        matrix = _build_all_iters_matrix(seed, ckpt_root)
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(ALL_ITERS)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
        # Bold + slightly larger font for canonical iter labels so they pop.
        for lbl, it in zip(ax.get_xticklabels(), ALL_ITERS):
            if it in canonical_set:
                lbl.set_fontweight("bold")
                lbl.set_fontsize(7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("iter (every 2000; canonical iters in bold)")
        ax.set_xticks([x - 0.5 for x in range(1, len(ALL_ITERS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)

    legend_handles = [
        Patch(color="#cc4040", label="missing"),
        Patch(color="#ff9933", label="megatron (local)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Pretraining progress: every 2000-step checkpoint", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Machine-readable per-model action mode (consumed by launch_resumes.sh).
#
# Output (one line per model, tab-separated):
#   <model>\tdone
#   <model>\tfresh\t<target>                  # no iter dirs on disk → fresh start
#   <model>\tcorrupt\t<n_iters>               # iter dirs exist but none valid → skip + warn
#   <model>\tresume\t<load_iter>\t<target>    # resume from <load_iter>, train until <target>
#
# We never emit a "wipe" action — the launcher must not auto-delete checkpoint
# directories. A corrupt model is reported and left alone for manual review.
#
# Decision logic per model:
#   • All canonical iters ≤ target valid    → done
#   • No iter dirs on disk                  → fresh
#   • Iter dirs exist but no valid ones     → corrupt (skip)
#   • Any canonical iter > max_existing_canonical missing (end-gap)
#       → resume from max_valid until <target>
#   • Otherwise (mid-gap only): earliest missing canonical
#       → resume from max valid iter strictly less than that canonical, until canonical
#     If a model has both end- and mid-gaps, end wins (per spec).
#     Re-running picks up the next mid-gap after the previous job finishes.
# ---------------------------------------------------------------------------

def _canonical_models(root: Path) -> list[Path]:
    """Return sorted canonical model dirs under root (those that match
    apertus-{175M,350M,600M,1B}-fwEdu*-fw2*-seed*) — even if the dir is missing
    on disk (we still want to emit a `fresh` action for those)."""
    out: list[Path] = []
    for size in SIZES:
        for fw_edu, fw2 in MIXES:
            for seed in SEEDS:
                out.append(root / f"apertus-{size}-fwEdu{fw_edu}-fw2{fw2}-seed{seed}")
    return out


def emit_actions(target: int, root: Path = CKPT_ROOT,
                 filter_substr: str | None = None) -> None:
    """Print per-model action lines for the resume launcher (see above).

    The canonical iter set is picked per-cell from `canonical_iters_for_seed`:
    seed1904 uses ITERS_SEED1904; seed28 / seed1797 use ITERS_OTHER. This
    matches the rest of the pipeline (eval CSV, conversion plan files).
    """
    for d in _canonical_models(root):
        if filter_substr and filter_substr not in d.name:
            continue
        marker, n_iters, max_iter, max_valid, valid_iters = model_progress(d)

        canonical_target_iters = [
            c for c in canonical_iters_for_seed(_seed_of(d.name)) if c <= target
        ]
        existing_canonical = [c for c in canonical_target_iters if c in valid_iters]
        missing_canonical = [c for c in canonical_target_iters if c not in valid_iters]

        if not missing_canonical:
            print(f"{d.name}\tdone")
            continue

        max_existing = max(existing_canonical) if existing_canonical else 0
        end_missing = [c for c in missing_canonical if c > max_existing]
        mid_missing = [c for c in missing_canonical if c < max_existing]

        if end_missing:
            # Drive to the final target. Spec: end wins over mid.
            if max_valid is None:
                if n_iters > 0:
                    # Iter dirs exist but none valid — manual review required.
                    print(f"{d.name}\tcorrupt\t{n_iters}")
                else:
                    print(f"{d.name}\tfresh\t{target}")
            else:
                print(f"{d.name}\tresume\t{max_valid}\t{target}")
        else:
            # Mid-only: target the earliest missing canonical.
            target_iter = mid_missing[0]
            valid_before = [v for v in valid_iters if v < target_iter]
            load_iter = max(valid_before) if valid_before else 0
            if load_iter == 0:
                # Edge case: no valid iter strictly before this mid canonical
                # (e.g. only later iters are valid). Skip and let the user look.
                if n_iters > 0:
                    print(f"{d.name}\tcorrupt\t{n_iters}")
                else:
                    print(f"{d.name}\tfresh\t{target_iter}")
            else:
                print(f"{d.name}\tresume\t{load_iter}\t{target_iter}")


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
    p.add_argument("--target", type=int, default=TARGET_ITER, help="Target iteration for progress bar.")
    p.add_argument(
        "--plot",
        metavar="PATH",
        help="Write a 3-panel heatmap (one per seed) showing per-cell stage "
             "status (missing → megatron → HF → hub) to PATH, plus a companion "
             "all-2000-step-iters plot at PATH_all.png, and exit.",
    )
    p.add_argument(
        "--no-hub",
        action="store_true",
        help="With --plot, skip the HF Hub branch lookup (treat all branches "
             "as missing). Useful when the Hub API is rate-limited.",
    )
    p.add_argument(
        "--actions",
        action="store_true",
        help="Print one machine-readable line per canonical model: "
             "<model>\\tdone | fresh\\t<target> | corrupt\\t<n_iters> | "
             "resume\\t<load_iter>\\t<target>. Consumed by launch_resumes.sh.",
    )
    args = p.parse_args()

    if args.plot:
        make_plot(Path(args.plot), query_hub=not args.no_hub, ckpt_root=Path(args.root))
        return

    if args.actions:
        emit_actions(args.target, root=Path(args.root), filter_substr=args.filter)
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
        marker, n_iters, max_iter, max_valid, _ = model_progress(d)
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
