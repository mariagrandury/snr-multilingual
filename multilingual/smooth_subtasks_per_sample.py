"""Per-sample SNR subset search (Option D from PROPOSALS.md).

For each language-specific multilingual benchmark (arc_es, xnli_de,
belebele_zh, ...), find the subset of samples (lm-eval doc_ids) that
maximizes SNR.

Method (Option D):
  1. Walk Apertus eval_logs and parse every ``samples_<task>_*.jsonl``
     for the requested tasks, keeping only ``doc_id`` and ``acc``.
  2. Per task × size, build an (n_ckpts × n_samples) acc matrix plus
     per-ckpt (mix, step) metadata.
  3. Variance prefilter: drop "dead" samples (per-mix mean accuracy
     constant across the 3 mixes — they carry no signal).
  4. Per-sample SNR using the same primitive as the rest of the repo
     (signal = range of per-mix means / mean; noise = std of per-mix
     last-5-ckpt scores pooled / mean), vectorised over samples.
  5. Sort surviving samples by SNR; sweep cumulative subset 1..N and
     pick argmax. Also a random-order baseline for sanity.
  6. Write per-(lang, benchmark) outputs under
     ``results/smooth_subtasks/per_sample/<lang>/<benchmark>_<lang>/``.

Other proposed search strategies (forward greedy, IRT, random) are
documented in the same dir's ``PROPOSALS.md``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multilingual.analyze_snr_variants import assign_language, benchmark_family
from multilingual.smooth_subtasks import collect_multilingual_families
from snr.constants import PLOT_DIR
from snr.download.apertus import (
    DEFAULT_EVAL_ROOT,
    _MODEL_RE,
    load_apertus_eval_results,
)

ALL_SIZES = ["175M", "350M", "600M", "1B"]
LAST_N = 5
OUT_ROOT = PLOT_DIR / "smooth_subtasks" / "per_sample"

_SAMPLES_FNAME_RE = re.compile(r"^samples_(?P<task>.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")


### Phase 1: walk + parse samples_*.jsonl ###


def _parse_ckpt_id(name: str):
    m = _MODEL_RE.match(name)
    if not m:
        return None
    return {
        "size": m["size"],
        "mix": f"fwEdu{m['edu']}",
        "seed": int(m["seed"]),
        "step": int(m["iter"]),
    }


def _load_one_samples_file(path: Path) -> dict[int, float]:
    """{doc_id: acc} from one samples_*.jsonl. ``acc`` is binary 0/1
    float; unparseable lines are skipped."""
    out: dict[int, float] = {}
    with open(path) as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "doc_id" not in obj or "acc" not in obj:
                continue
            try:
                out[int(obj["doc_id"])] = float(obj["acc"])
            except Exception:
                continue
    return out


def load_samples(
    tasks: set[str], eval_root: Path = DEFAULT_EVAL_ROOT
) -> dict[str, dict[tuple[str, str, int], dict[int, float]]]:
    """{task: {(size, mix, step): {doc_id: acc}}} for the requested tasks.

    Multiple samples files for the same (ckpt, task) can exist if the
    eval was rerun; the lexicographically last filename (which encodes
    a UTC timestamp) wins.
    """
    eval_root = Path(eval_root)
    # Collect candidate files per (ckpt_dir, task), keep lexicographically
    # latest filename per (ckpt, task).
    plan: dict[tuple[Path, str], Path] = {}
    ckpt_meta: dict[Path, dict] = {}
    ckpt_dirs = [d for d in eval_root.iterdir() if d.is_dir()]
    for ckpt_dir in tqdm(ckpt_dirs, desc="scanning ckpt dirs"):
        meta = _parse_ckpt_id(ckpt_dir.name)
        if meta is None:
            continue
        ckpt_meta[ckpt_dir] = meta
        for f in ckpt_dir.rglob("samples_*.jsonl"):
            m = _SAMPLES_FNAME_RE.match(f.name)
            if m is None:
                continue
            task = m["task"]
            if task not in tasks:
                continue
            key = (ckpt_dir, task)
            if key not in plan or f.name > plan[key].name:
                plan[key] = f

    by_task: dict[str, dict[tuple[str, str, int], dict[int, float]]] = defaultdict(dict)
    for (ckpt_dir, task), f in tqdm(plan.items(), desc="parsing samples"):
        meta = ckpt_meta[ckpt_dir]
        accs = _load_one_samples_file(f)
        if not accs:
            continue
        by_task[task][(meta["size"], meta["mix"], meta["step"])] = accs
    return by_task


### Phase 2: per-task matrices + SNR ###


def _build_matrix(
    by_ckpt: dict[tuple[str, str, int], dict[int, float]],
    size: str,
) -> tuple[np.ndarray, list[tuple[str, str, int]], list[int]] | None:
    """Return (A, ckpts, doc_ids) or None if data is insufficient.
    A has shape (n_ckpts, n_samples), filled with NaN where a doc is
    missing from a particular ckpt (rare; we drop docs with any NaN
    after this)."""
    ckpts = [k for k in by_ckpt if k[0] == size]
    if not ckpts:
        return None
    ckpts.sort(key=lambda k: (k[1], k[2]))  # by mix, then step
    all_docs = sorted({d for k in ckpts for d in by_ckpt[k]})
    if not all_docs:
        return None
    doc_to_col = {d: i for i, d in enumerate(all_docs)}
    A = np.full((len(ckpts), len(all_docs)), np.nan, dtype=np.float32)
    for r, k in enumerate(ckpts):
        for d, v in by_ckpt[k].items():
            A[r, doc_to_col[d]] = v
    keep_cols = ~np.isnan(A).any(axis=0)
    if not keep_cols.any():
        return None
    A = A[:, keep_cols]
    doc_ids = [all_docs[i] for i, k in enumerate(keep_cols) if k]
    return A, ckpts, doc_ids


def _last_n_rows_per_mix(
    ckpts: list[tuple[str, str, int]], last_n: int = LAST_N
) -> dict[str, list[int]]:
    """Row indices into A corresponding to each mix's last-n ckpts (by step)."""
    by_mix: dict[str, list[int]] = defaultdict(list)
    for r, (_, mix, step) in enumerate(ckpts):
        by_mix[mix].append((step, r))
    return {
        mix: [r for _, r in sorted(rows)[-last_n:]]
        for mix, rows in by_mix.items()
    }


def _signal_noise(combined: np.ndarray, mix_rows: dict[str, list[int]]
                  ) -> tuple[float, float, float]:
    """signal_to_noise_ratio replicated for a 1-D combined-score vector
    (length = n_ckpts). Returns (signal, noise, snr) with NaN-safe fallback.
    """
    arrs = [combined[rows] for rows in mix_rows.values()]
    arrs = [a for a in arrs if a.size >= 2]
    if len(arrs) < 2:
        return float("nan"), float("nan"), float("nan")
    mix_means = np.array([a.mean() for a in arrs])
    pooled = np.concatenate(arrs)
    if pooled.size == 0 or pooled.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    dispersion = mix_means.max() - mix_means.min()
    signal = dispersion / mix_means.mean() if mix_means.mean() else float("nan")
    noise = pooled.std() / pooled.mean() if pooled.mean() else float("nan")
    if not (np.isfinite(signal) and np.isfinite(noise)) or noise == 0:
        return signal, noise, float("nan")
    return float(signal), float(noise), float(signal / noise)


def _drop_sparse_mixes(mix_rows: dict[str, list[int]],
                       min_ckpts: int = 2) -> dict[str, list[int]]:
    """Drop mixes with fewer than ``min_ckpts`` ckpts — same gate as the
    aggregate-level ``_signal_noise``: a mix with one ckpt contributes
    no within-mix step variance so its inclusion biases pooled.std()."""
    return {m: rs for m, rs in mix_rows.items() if len(rs) >= min_ckpts}


def _per_sample_snr(A: np.ndarray, mix_rows: dict[str, list[int]]) -> np.ndarray:
    """Vectorised per-sample SNR. Shape (n_samples,)."""
    mix_rows = _drop_sparse_mixes(mix_rows)
    mixes = list(mix_rows.keys())
    if len(mixes) < 2:
        return np.full(A.shape[1], np.nan, dtype=np.float64)
    # mix_means shape (n_mixes, n_samples)
    mix_means = np.stack([A[mix_rows[m], :].mean(axis=0) for m in mixes], axis=0)
    pooled_rows = [r for m in mixes for r in mix_rows[m]]
    pooled = A[pooled_rows, :]  # shape (n_pooled, n_samples)

    overall_mix_mean = mix_means.mean(axis=0)
    pooled_mean = pooled.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        signal = (mix_means.max(axis=0) - mix_means.min(axis=0)) / overall_mix_mean
        noise = pooled.std(axis=0) / pooled_mean
        snr = signal / noise

    snr = np.where(np.isfinite(snr), snr, np.nan)
    return snr


def _variance_prefilter_mask(A: np.ndarray, mix_rows: dict[str, list[int]]
                             ) -> np.ndarray:
    """True for samples whose per-mix mean varies across mixes (not dead).

    Mixes with <2 ckpts are dropped first (matches ``_per_sample_snr``);
    a sample whose surviving mix-means are all equal is "dead" and
    excluded from the SNR sweep.
    """
    mix_rows = _drop_sparse_mixes(mix_rows)
    mixes = list(mix_rows.keys())
    if len(mixes) < 2:
        return np.zeros(A.shape[1], dtype=bool)
    mix_means = np.stack([A[mix_rows[m], :].mean(axis=0) for m in mixes], axis=0)
    return mix_means.std(axis=0) > 0


def _cumulative_subset_snrs(A: np.ndarray, ordered_cols: np.ndarray,
                            mix_rows: dict[str, list[int]]) -> list[float]:
    """For each N=1..len(ordered_cols), SNR of A[:, ordered_cols[:N]].mean(axis=1)."""
    if ordered_cols.size == 0:
        return []
    cumsum = A[:, ordered_cols].cumsum(axis=1)  # (n_ckpts, n_ordered)
    out = []
    for n in range(1, ordered_cols.size + 1):
        combined = cumsum[:, n - 1] / n
        _, _, snr = _signal_noise(combined, mix_rows)
        out.append(snr)
    return out


def _argmax_safe(values: list[float]) -> int:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return -1
    arr = np.where(np.isfinite(arr), arr, -np.inf)
    return int(np.argmax(arr))


### Phase 3: per-task driver + outputs ###


def _run_one_size(
    A: np.ndarray, ckpts: list[tuple[str, str, int]], doc_ids: list[int],
    rng: np.random.Generator,
) -> dict | None:
    mix_rows = _last_n_rows_per_mix(ckpts)
    if len(mix_rows) < 2:
        return None
    keep = _variance_prefilter_mask(A, mix_rows)
    n_total = A.shape[1]
    if not keep.any():
        return {"n_total": n_total, "n_after_prefilter": 0}

    survivor_idx = np.flatnonzero(keep)
    snrs = _per_sample_snr(A, mix_rows)
    survivor_snrs = snrs[survivor_idx]

    # Sort survivors by SNR descending; NaNs to the end.
    finite = np.isfinite(survivor_snrs)
    order_finite = survivor_idx[finite][np.argsort(-survivor_snrs[finite])]
    order_nan = survivor_idx[~finite]
    ordered = np.concatenate([order_finite, order_nan])

    cumulative = _cumulative_subset_snrs(A, ordered, mix_rows)
    rand_order = ordered.copy()
    rng.shuffle(rand_order)
    rand_cumulative = _cumulative_subset_snrs(A, rand_order, mix_rows)

    full_idx = np.arange(A.shape[1])
    _, _, full_set_snr = _signal_noise(A[:, full_idx].mean(axis=1), mix_rows)

    best_idx = _argmax_safe(cumulative)
    return {
        "n_total": n_total,
        "n_after_prefilter": int(keep.sum()),
        "ordered": ordered,
        "ordered_doc_ids": [doc_ids[i] for i in ordered],
        "cumulative_snrs": cumulative,
        "random_cumulative_snrs": rand_cumulative,
        "best_n": best_idx + 1 if best_idx >= 0 else 0,
        "best_snr": cumulative[best_idx] if best_idx >= 0 else float("nan"),
        "full_set_snr": full_set_snr,
        "per_sample_snrs": snrs,  # length n_total
    }


def _plot_cumulative(task: str, per_size: dict[str, dict], save_path: Path):
    sizes = [s for s in ALL_SIZES if per_size.get(s) and "cumulative_snrs" in per_size[s]]
    if not sizes:
        return
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8, 2.4 * len(sizes)),
                             sharex=False, squeeze=False)
    for i, size in enumerate(sizes):
        ax = axes[i][0]
        r = per_size[size]
        x = np.arange(1, len(r["cumulative_snrs"]) + 1)
        ax.plot(x, r["cumulative_snrs"], linewidth=0.9, label="sorted by SNR")
        ax.plot(x, r["random_cumulative_snrs"], color="r", linewidth=0.7,
                alpha=0.7, label="random order")
        if r["best_n"] > 0:
            ax.axvline(r["best_n"], color="grey", linestyle="--", linewidth=0.6)
        ax.set_title(
            f"{task} — {size}  (N={r['n_total']}, after prefilter "
            f"{r['n_after_prefilter']}; best top-{r['best_n']}, "
            f"best SNR {r['best_snr']:.3f}, full {r['full_set_snr']:.3f})",
            fontsize=9,
        )
        ax.set_ylabel("Combined SNR")
        ax.grid(True, linestyle="-", alpha=0.2)
    axes[-1][0].set_xlabel("Subset size (samples added in SNR order)")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def _write_outputs(task: str, per_size: dict[str, dict], doc_ids_per_size: dict,
                   out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.csv: one row per size
    sum_rows = []
    for size in ALL_SIZES:
        r = per_size.get(size)
        if not r:
            sum_rows.append({"size": size, "status": "no_data"})
            continue
        if "cumulative_snrs" not in r:
            sum_rows.append({"size": size, "status": "all_dead",
                             "n_total": r["n_total"],
                             "n_after_prefilter": r["n_after_prefilter"]})
            continue
        sum_rows.append({
            "size": size, "status": "ok",
            "n_total": r["n_total"],
            "n_after_prefilter": r["n_after_prefilter"],
            "best_n": r["best_n"],
            "best_snr": r["best_snr"],
            "full_set_snr": r["full_set_snr"],
            "snr_gain": (r["best_snr"] - r["full_set_snr"])
                if np.isfinite(r["best_snr"]) and np.isfinite(r["full_set_snr"])
                else float("nan"),
        })
    pd.DataFrame(sum_rows).to_csv(out_dir / "summary.csv", index=False)

    # ranked_samples.csv: one row per (task-level) sample, with per-size SNR
    # plus a flag for whether it landed in the best subset for each size.
    all_doc_ids = sorted({d for ds in doc_ids_per_size.values() for d in ds})
    rows = []
    by_size_index = {
        size: {d: i for i, d in enumerate(doc_ids_per_size.get(size, []))}
        for size in ALL_SIZES
    }
    for d in all_doc_ids:
        row = {"doc_id": d}
        for size in ALL_SIZES:
            r = per_size.get(size)
            if not r or "per_sample_snrs" not in r:
                row[f"snr_{size}"] = float("nan")
                row[f"in_best_{size}"] = False
                continue
            idx = by_size_index[size].get(d)
            if idx is None:
                row[f"snr_{size}"] = float("nan")
                row[f"in_best_{size}"] = False
                continue
            row[f"snr_{size}"] = float(r["per_sample_snrs"][idx])
            best_set = set(r["ordered"][: r["best_n"]].tolist()) if r["best_n"] > 0 else set()
            row[f"in_best_{size}"] = idx in best_set
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "ranked_samples.csv", index=False)

    # best_subset_<size>.txt: doc_ids only.
    for size in ALL_SIZES:
        r = per_size.get(size)
        if not r or "ordered_doc_ids" not in r or r["best_n"] <= 0:
            continue
        best_doc_ids = r["ordered_doc_ids"][: r["best_n"]]
        (out_dir / f"best_subset_{size}.txt").write_text(
            "\n".join(str(d) for d in best_doc_ids) + "\n"
        )

    _plot_cumulative(task, per_size, out_dir / "cumulative_snr.png")


def run_one_task(task: str, by_ckpt: dict, out_root: Path,
                 rng: np.random.Generator):
    lang = assign_language(task)
    if lang == "??":
        return None
    out_dir = out_root / lang / task

    per_size = {}
    doc_ids_per_size: dict[str, list[int]] = {}
    for size in ALL_SIZES:
        built = _build_matrix(by_ckpt, size)
        if built is None:
            per_size[size] = None
            continue
        A, ckpts, doc_ids = built
        doc_ids_per_size[size] = doc_ids
        result = _run_one_size(A, ckpts, doc_ids, rng)
        per_size[size] = result

    _write_outputs(task, per_size, doc_ids_per_size, out_dir)
    return out_dir


def main():
    df_meta = load_apertus_eval_results()
    families = collect_multilingual_families(df_meta)
    tasks = sorted({t for ts in families.values() for t in ts})
    print(f"Targeting {len(tasks)} multilingual language-tasks "
          f"({len(families)} families).")

    samples = load_samples(set(tasks))
    print(f"Parsed samples for {len(samples)} tasks.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    written = 0
    for task in tqdm(tasks, desc="tasks"):
        if task not in samples:
            continue
        out = run_one_task(task, samples[task], OUT_ROOT, rng)
        if out is not None:
            written += 1

    # Roll-up: aggregate every task's summary.csv into one master file so
    # users don't have to grep across 98 dirs.
    master_rows = []
    for csv in OUT_ROOT.rglob("summary.csv"):
        if csv.parent == OUT_ROOT:
            continue
        df = pd.read_csv(csv)
        df["task"] = csv.parent.name
        df["language"] = csv.parent.parent.name
        master_rows.append(df)
    if master_rows:
        master = pd.concat(master_rows, ignore_index=True)
        cols = ["language", "task"] + [c for c in master.columns
                                       if c not in ("language", "task")]
        master = master[cols].sort_values(["language", "task", "size"])
        master_path = OUT_ROOT / "summary_all.csv"
        master.to_csv(master_path, index=False)
        print(f"Wrote roll-up → {master_path}")

    print(f"Wrote {written} per-task output dirs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
