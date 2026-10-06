"""rq08: keep only the items the reference model gets right, then read DA and SNR.

Does a benchmark rank the designs better, or is its SNR a better guide to its
DA, once the items the 1.7B models fail are dropped? Read on the per-item
store's final checkpoints (`build_per_item_store.py --finals-only`), the
grid-seed design variants of `predictivity`:

  solved      an item is solved when at least SOLVED of the TARGET_SIZE runs
              in the selecting set answer it right (the task's metric, acc or
              acc_norm, per item)
  DA-size     per proxy size, the proxy's final mean over the items (all, or the
              solved ones) against the reference's final on the FULL task, over
              the multi-axis pairs (`pair_agreement`, NaN below MIN_PAIRS, rule
              5). The truth stays the full task: on the solved items the
              reference itself scores close to 1 and would rank nothing.
  SNR         at the proxy's final: relative dispersion of the design means
              ((max - min) / mean) over the relative k-fold benchmark noise
              (`catalogue.kfold_noise`, closed form on the item count), the one
              noise a single checkpoint carries.

The selection reads the reference's answers, and DA is scored against the
reference: chosen on every 1.7B run, the subset has seen the truth (rule 11).
So the subset is chosen on the 1.7B runs of half of the families (stratified
over L, arch, scheme, T, as per_item_ladder.heldout_da) and scored on the other
half's pairs, both ways round, against the full set on the same held-out pairs;
the in-sample reading (every 1.7B run selects) is kept beside it as an upper
bound and labelled so. Rule 1: a task counts at a size when it is above chance
there and at the reference (`passes_gate`, the `predictivity` mask).

Outputs under pretraining/<pool>/: reference_solved_da_size_multi_axes.csv
(task x proxy size), reference_solved_summary_da_size_multi_axes.csv (per size:
mean DAs, the paired test, and the SNR-DA Spearman of both sets), the figure
reference_solved_da_size_multi_axes.png with its CSV, and the README's
`reference-solved` block.

    python analysis/rq08_subset_selection/reference_solved.py --pool predictivity [--store predictivity_schemes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, wilcoxon

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import metric_for  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SUBSET_SELECTION  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq04_surrogates.catalogue import kfold_noise  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import STORE, readable_parts, store_gap  # noqa: E402
from analysis.utils import (BBPB, MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, benchmark_family,  # noqa: E402
                            design_axes, finals, ladder_frame, pair_agreement, pair_sets, passes_gate)

DOC_POOL = "predictivity"            # the pool the store is built on; only it writes the README
GATE_POOL = "predictivity"           # the gate every rq02 reader uses
SOLVED = 0.5                         # share of the selecting 1.7B runs that must answer an item right
NAME = "reference_solved_da_size_multi_axes"
mpl.rcParams.update(S.RC)


def snr(x: np.ndarray, n_items: int) -> float:
    """Relative dispersion of the design means over their relative k-fold noise."""
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.mean() <= 0:
        return np.nan
    noise = kfold_noise(x, n_items)["kfold_rel"]
    return float((x.max() - x.min()) / x.mean() / noise) if np.isfinite(noise) and noise > 0 else np.nan


def finite_mean(v: list) -> float:
    """Mean of the halves that gave a value; NaN when neither did."""
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def task_rows(task: str, M: pd.DataFrame, cols: pd.DataFrame, truth: dict, pairs: list, halves: tuple) -> list[dict]:
    """Per proxy size: n items, n solved, DA of the full set and of the solved
    subset (in sample and held out), and both SNRs. `M` = items x runs."""
    A = M.to_numpy(dtype=float)
    fam, size = cols["family"].to_numpy(), cols["size"].to_numpy()
    ref = size == TARGET_SIZE
    solved_all = A[:, ref].mean(1) >= SOLVED
    rows = []
    for s in SMALL_SIZES:
        at = size == s
        if not at.any():
            continue
        P = dict(zip(fam[at], A[:, at].T))
        mean = lambda keep, fams: {f: P[f][keep].mean() for f in fams if f in P}
        da_full, n = pair_agreement(mean(slice(None), P), truth, pairs)
        da_in = pair_agreement(mean(solved_all, P), truth, pairs)[0] if solved_all.any() else np.nan
        ho_full, ho_sub, ho_pairs = [], [], 0
        for sel, held in (halves, halves[::-1]):
            by_sel = ref & np.isin(fam, list(sel))             # np.isin reads a set as one object
            keep = A[:, by_sel].mean(1) >= SOLVED if by_sel.any() else None
            hp = [(a, b) for a, b in pairs if a in held and b in held]
            if keep is None or not keep.any() or len(hp) < MIN_PAIRS:               # rule 5
                continue
            ho_full.append(pair_agreement(mean(slice(None), held), truth, hp)[0])
            d, k = pair_agreement(mean(keep, held), truth, hp)
            ho_sub.append(d); ho_pairs += k
        x_full = np.array([P[f].mean() for f in P])
        x_sub = np.array([P[f][solved_all].mean() for f in P]) if solved_all.any() else np.array([])
        rows.append({"task": task, "family": benchmark_family(task), "size": s, "axes": "multi-axis",
                     "n_items": len(A), "n_solved": int(solved_all.sum()), "solved_share": float(solved_all.mean()),
                     "n_pairs": n, "n_pairs_heldout": ho_pairs,
                     "da_full": da_full, "da_solved_in_sample": da_in,
                     "da_full_heldout": finite_mean(ho_full), "da_solved_heldout": finite_mean(ho_sub),
                     "snr_full": snr(x_full, len(A)), "snr_solved": snr(x_sub, int(solved_all.sum()))})
    return rows


def summarise(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in out.groupby("size"):
        both = g.dropna(subset=["da_full_heldout", "da_solved_heldout"])
        d = both["da_solved_heldout"] - both["da_full_heldout"]
        r = {"size": s, "axes": "multi-axis", "n_tasks": len(g), "solved_share": g["solved_share"].median()}
        for c in ("da_full", "da_solved_in_sample", "da_full_heldout", "da_solved_heldout"):
            r[c] = g[c].mean()
        r |= {"n_paired": len(d), "gain_heldout": d.mean(), "p_wilcoxon": wilcoxon(d).pvalue if (d != 0).any() else np.nan}
        for tag, x, y in (("full", "snr_full", "da_full_heldout"), ("solved", "snr_solved", "da_solved_heldout")):
            h = g[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = spearmanr(h[x], h[y]) if len(h) >= 3 else None
            r |= {f"rho_snr_da_{tag}": rho.statistic if rho else np.nan, f"p_rho_{tag}": rho.pvalue if rho else np.nan,
                  f"n_rho_{tag}": len(h)}
        rows.append(r)
    return pd.DataFrame(rows).set_index("size").reindex([s for s in SMALL_SIZES if s in set(out["size"])]).reset_index()


def figure(summ: pd.DataFrame, out_dir: Path, note: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = range(len(summ))
    labels = [f"{s}\n({int(n)} tasks)" for s, n in zip(summ["size"], summ["n_paired"])]
    rows = []
    for col, lab, c, ls in (("da_full_heldout", "all items", S.INK, "-"),
                            ("da_solved_heldout", f"items the {TARGET_SIZE} solves (chosen on the other half)", S.SERIES[0], "-"),
                            ("da_solved_in_sample", "same, chosen on every 1.7B run (sees the truth)", S.SERIES[0], ":")):
        axes[0].plot(x, summ[col], ls, marker="o", ms=4, lw=1.4, color=c, label=lab)
        rows += [{"panel": "DA-size", "row": lab, "col": s, "value": v} for s, v in zip(summ["size"], summ[col])]
    axes[0].set_title(f"DA-size against the {TARGET_SIZE} final of the full task", loc="left", fontsize=8.5)
    axes[0].set_ylabel("decision accuracy (mean over tasks)"); axes[0].set_ylim(0.3, 1.0)
    for col, lab, c in (("rho_snr_da_full", "all items", S.INK), ("rho_snr_da_solved", f"items the {TARGET_SIZE} solves", S.SERIES[0])):
        axes[1].plot(x, summ[col], "-o", ms=4, lw=1.4, color=c, label=lab)
        rows += [{"panel": "Spearman rho(SNR, DA)", "row": lab, "col": s, "value": v} for s, v in zip(summ["size"], summ[col])]
    axes[1].axhline(0, color=S.MUTED, lw=.8)
    axes[1].set_title("Does SNR track DA? Spearman ρ over tasks (held-out DA)", loc="left", fontsize=8.5)
    axes[1].set_ylabel("Spearman ρ(SNR, DA-size)"); axes[1].set_ylim(-1, 1)
    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_xlabel("proxy size")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].axhline(0.5, color=S.MUTED, lw=.8, ls=":")
    G.save_highlights(fig, out_dir, "Items the reference solves: do they rank the designs better?", note,
                      [pd.DataFrame(rows)], name=NAME)


def readme(summ: pd.DataFrame, pool: str, n_tasks: int, store: str, missing: list[str]) -> None:
    rows = [[r.size, int(r.n_paired), fmt(r.solved_share), fmt(r.da_full_heldout), fmt(r.da_solved_heldout),
             f"{r.gain_heldout:+.3f}" if np.isfinite(r.gain_heldout) else "", fmt(r.p_wilcoxon, 3),
             fmt(r.da_solved_in_sample), fmt(r.rho_snr_da_full), fmt(r.rho_snr_da_solved)] for r in summ.itertuples()]
    body = "\n\n".join([
        "## Items the reference solves",
        f"DA-size, final checkpoints, multi-axis pairs of `{pool}` (grid seed), gate `{GATE_POOL}` at the proxy and the "
        f"reference, {n_tasks} tasks with per-item outputs. An item is solved when at least {SOLVED:g} of the {TARGET_SIZE} "
        f"runs of the selecting half answer it right; the subset's proxy mean is scored against the {TARGET_SIZE} final of "
        "the full task on the other half's pairs, both ways round (held out, rule 11), beside the full set on the same "
        "pairs; the in-sample column selects with every 1.7B run and has seen the truth. SNR = relative dispersion of the "
        "design means over the relative k-fold noise, read at the proxy alone; the solved set's SNR is on the items at least "
        f"half of all the {TARGET_SIZE} runs solve, while the DA it is correlated with is the held-out one. Regenerate with "
        f"`python analysis/rq08_subset_selection/reference_solved.py --pool {pool}"
        f"{'' if store == pool else ' --store ' + store}`."
        + (f" The store `{store}` lacks {len(missing)} of the pool's models ({', '.join(missing)}); "
           "they are left out until the store is rebuilt for the pool." if missing else ""),
        md_table(["proxy", "tasks", "solved share", "DA all (held out)", "DA solved (held out)", "Δ", "Wilcoxon p",
                  "DA solved (in sample)", "ρ(SNR, DA) all", "ρ(SNR, DA) solved"], rows),
        f"![Items the reference solves](pretraining/{pool}/{NAME}.png)"])
    replace_block(SUBSET_SELECTION / "README.md", "reference-solved", body, f"reference_solved.py --pool {pool}")


def main(pool: str, store: str | None = None) -> None:
    out_dir = SUBSET_SELECTION / "pretraining" / pool
    store = store or pool
    parts = sorted((STORE / store).glob("*.parquet"))
    if not parts:
        print(f"no per-item store at {STORE / store}: nothing written (build_per_item_store.sbatch builds it)")
        return
    df = ladder_frame(pool)
    fin = finals(df[(df["kind"] == "benchmark") & ~df["task"].str.startswith(BBPB)])
    keys = fin[["model", "step", "task", "size", "family"]]
    missing = store_gap(keys["model"], STORE / store)
    last = keys.groupby("model")["step"].max()                     # the store's finals are these steps
    truth = {t: dict(zip(g["family"], g["primary_score"])) for t, g in fin[fin["size"] == TARGET_SIZE].groupby("task")}
    attrs = design_axes(df)
    pairs = pair_sets(attrs)["multi-axis"]
    fams = attrs.loc[sorted(set(fin.loc[fin["size"] == TARGET_SIZE, "family"]))].sort_values(["L", "arch", "scheme", "T"]).index
    halves = (set(fams[::2]), set(fams[1::2]))              # alternate along the sorted axes: stratified
    mask = load_mask(GATE_POOL)
    print(f"{pool}: {len(parts)} store families, {keys['task'].nunique()} tasks, {len(pairs)} multi-axis pairs, "
          f"halves of {len(halves[0])} / {len(halves[1])} families at {TARGET_SIZE}")
    rows = []
    for part in parts:
        # dictionary-encoded strings: the largest family is ~10^7 items x runs
        files = readable_parts(part)
        if not files:
            continue
        s = pq.read_table(files, columns=["model", "step", "task", "doc_id", "acc", "acc_norm"],
                          read_dictionary=["model", "task"]).to_pandas()
        s = s[s["step"].to_numpy() == s["model"].map(last).astype(float).to_numpy()]
        for task, g in s.groupby("task", observed=True):
            cols = keys[keys["task"] == task].set_index("model")    # the (model, task) the pool keeps: rules 2 and 6
            g = g[g["model"].isin(cols.index)]
            M = g.pivot_table(index="doc_id", columns="model", values=metric_for(task) or "acc", observed=True).dropna()
            cols = cols.loc[M.columns]
            if M.empty or not (cols["size"] == TARGET_SIZE).any():
                continue
            rows += task_rows(task, M, cols, truth.get(task, {}), pairs, halves)
    out = pd.DataFrame(rows)
    ok = [bool(passes_gate(mask, [t], s, TARGET_SIZE).iloc[0]) for t, s in zip(out["task"], out["size"])]
    out["gated"] = ~np.asarray(ok)
    cols = [c for c in out.columns if c.startswith(("da_", "snr_"))]
    out.loc[out["gated"], cols] = np.nan                     # rule 1: at chance, kept and blanked
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{NAME}.csv", index=False)
    summ = summarise(out[~out["gated"]])
    summ.to_csv(out_dir / "reference_solved_summary_da_size_multi_axes.csv", index=False)
    print(summ.round(3).to_string())
    note = (f"DA-size, finals, multi-axis pairs of {pool}, gate {GATE_POOL} at the proxy and {TARGET_SIZE}; solved = "
            f">= {SOLVED:g} of the selecting {TARGET_SIZE} runs right. Held out: chosen on half of the families, scored on "
            "the other half's pairs against the full task's reference final, both ways; dotted = chosen on every "
            f"{TARGET_SIZE} run (sees the truth, rule 11). ρ over tasks, SNR = relative dispersion / relative k-fold noise.")
    figure(summ, out_dir, note)
    if pool == DOC_POOL:
        readme(summ, pool, out.loc[~out["gated"], "task"].nunique(), store, missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=DOC_POOL)
    ap.add_argument("--store", default=None, help="the per_item_store/<store> folder to read (default: the pool's)")
    a = ap.parse_args()
    main(a.pool, a.store)
