"""Above-chance items: how much do decision accuracy and SNR rise when every
benchmark-language task keeps only the items its 1.7B runs answer above chance?

Selection. An item is above chance at the reference when the mean of its
per-item metric (the task's own, acc or acc_norm, `metric_for`, as the subset
selection's `reference_solved.py` reads it) over the pool's TARGET_SIZE final
runs that train the task's language (the loader's frame, rule 2: from 2 runs
for a language only the widest mixtures train to every run for English)
exceeds the task's chance level (`above_random.task_chance`, the level the
above-random gate tests). Each task (one benchmark in one language; parents
only, rule 6) is reduced on its own, from the per-item store
(`build_per_item_store.py`): a
run's sub-benchmark score is its mean over the kept items, its item count the
number kept and its chance level the task's. The store carries no per-item
option count, so an item of a variable-option family is held to the family's
E[1/n] (TruthfulQA mc1) or mean true-option share (mc2), not to its own.

Rule 11 is waived by request: the 1.7B runs of the pool select and DA is
still scored against the 1.7B reference, so the selection reads the reference
and the numbers measure how much that inflates DA and SNR. No held-out half.

Three orderings, same pool, same families, same sizes:

  full              the full benchmarks under the committed gate (the reference point)
  gate, then items  the cells the committed gate passes on the full benchmark,
                    read on their above-chance items
  items, then gate  every task reduced first; the gate is then recomputed on
                    the sub-benchmark scores by its own code (`scores_and_mask`
                    with the kept item counts) and decides which cells count

Per ordering: the gate's pass share per size; DA-size over the multi-axis and
mono-axis pairs (decision accuracy's decision rows and kernel,
`scale_convergence.decisions`; MIN_PAIRS; rule 1 at the proxy and the
reference), per task and pooled with the leave-one-family-out jackknife
(`jackknife_ratio`); the SNR at the final checkpoint, noise and SNR's
`rel_std` signal over the relative k-fold benchmark noise of the surrogate
catalogue (`kfold_noise`, the one noise a single checkpoint carries), and its
Spearman ρ with DA-size over the tasks; the log-N fit of scaling
predictability (`fit_table`). With checkpoints in the store, also DA-ckpt
(rule 1 at the proxy) and the checkpoint SNR of noise and SNR
(`per_model_inputs`, the window of rule 4); a finals-only store skips both and
the run says so. The surrogates are not computed: the catalogue reads a task's
item count by its name (`task_n_items`) and its truths from the decision
accuracy table on disk, so it cannot take a substituted score frame. Nothing is
injected into the shared loaders, so no other analysis moves.

Outputs under pretraining/<pool>/, each name prefixed `above_chance_items_`:
  selection.csv                  per task: items, kept items, reference runs, both gates at the reference
  survival.csv                   per ordering: tasks and items left after each step
  gate_pass_share.png/.csv       per ordering and size: share of the tasks the gate passes
  da_size_per_task_both_axes.csv per ordering, task, size, pair set: DA-size, pairs, gated
  da_size_both_axes.png/.csv     per ordering, pair set, size: pooled DA-size and its jackknife band,
                                 the mean over tasks and the paired gain over the full benchmark
  da_size_by_benchmark_multi_axes.png/.csv, ..._mono_axis.png/.csv
                                 per benchmark and proxy: mean DA-size per ordering and the paired gain
  snr_per_task.csv               per ordering, task, size: signal, noise and SNR (final; checkpoint if stored)
  snr.png/.csv                   per ordering and size: median SNR, paired ratio over full, ρ(SNR, DA-size)
  snr_by_benchmark.png/.csv      per benchmark and size: median SNR per ordering and the median paired
                                 log2(SNR kept / SNR full)
  scaling_fits.csv               per ordering, task, L: the log-N fit of the final scores
  da_ckpt_per_task_both_axes.csv, da_ckpt_both_axes.csv   only when the store holds checkpoints

    python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity [--store-pool predictivity]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, metric_for, size_bucket  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402
from snr.snr_variants import rel_std_snr  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import ABOVE_CHANCE_ITEMS  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, scores_and_mask, task_chance  # noqa: E402
from analysis.rq01_scaling_predictability.analyze import fit_table  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import decisions, grid_frame, reliability  # noqa: E402
from analysis.rq03_noise_and_snr.run_apertus_snr_variants import per_model_inputs, variant_signal_noise_snr  # noqa: E402
from analysis.rq04_surrogates.catalogue import kfold_noise  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import BBPB_POOL, STORE, readable_parts, store_gap  # noqa: E402
from analysis.utils import (  # noqa: E402
    ANALYSIS_SIZES, AXES_SUFFIX, BBPB, CKPT_DA_EARLY_FRACS, FRAC_TOL, MIN_PAIRS, PAIR_AXES, SMALL_SIZES, TARGET_SIZE,
    assign_language, benchmark_family, design_axes, finals, jackknife_ratio, ladder_frame, languages_only,
    on_shared_grid, pair_sets)

NAME = "above_chance_items"
ORDERINGS = {"full": "full benchmark", "gate_then_items": "gate, then items", "items_then_gate": "items, then gate"}
COLOUR = {"full": S.INK, "gate_then_items": S.SERIES[0], "items_then_gate": S.SERIES[1]}
AXES = PAIR_AXES[:2]                       # the design pair sets; the seed null decides nothing
KEYS = ["model", "step", "task"]
ROW_BUDGET = 50_000_000                     # store rows read at a time (~1 GB): a full store is every checkpoint
KEPT_BINS, KEPT_LABELS = [0, 30, 100, 300, 1000, np.inf], ["1-30", "31-100", "101-300", "301-1000", "> 1000"]
IN_SAMPLE = (f"the items are chosen on the pool's {TARGET_SIZE} runs that train the task's language and DA is scored against the "
             f"{TARGET_SIZE} reference: the selection reads the reference by design, and the numbers measure "
             "how much that inflates DA and SNR")
mpl.rcParams.update(S.RC)


def select_and_score(g: pd.DataFrame, metric: str, ref: pd.MultiIndex, chance: float) -> tuple[pd.Index, pd.DataFrame]:
    """One task's per-item rows `g` (model, step, doc_id, `metric`): the items
    whose mean over the reference runs `ref` ((model, step) pairs) is above
    `chance`, and per (model, step) the mean over every item (`full`) and over
    the kept ones (`sub`), with the item counts behind each."""
    at_ref = pd.MultiIndex.from_arrays([g["model"].astype(str), g["step"].astype(int)]).isin(ref)
    item = g.loc[at_ref].groupby("doc_id")[metric].mean()
    keep = item.index[item > chance]
    out = (g.assign(sub=g[metric].where(g["doc_id"].isin(keep)))
           .groupby(["model", "step"], observed=True)
           .agg(full=(metric, "mean"), sub=("sub", "mean"), n_full=(metric, "count"), n_sub=("sub", "count"))
           .reset_index())
    return keep, out


def store_scores(store: Path, fin: pd.DataFrame, models: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every task of the pool's frame that the store holds, family by family and
    in batches of tasks whose rows (the manifest's record counts) fit ROW_BUDGET:
    its selection (one row per task) and its full and sub-benchmark scores (one
    row per (model, step, task))."""
    size = pd.read_csv(store / "manifest.csv", usecols=["task", "n_records"]).groupby("task")["n_records"].sum()
    ref = {t: pd.MultiIndex.from_arrays([g["model"].astype(str), g["step"].astype(int)])
           for t, g in fin[fin["size"] == TARGET_SIZE].groupby("task")}
    by_family = fin.groupby(fin["task"].map(benchmark_family))["task"].unique()
    sel, scores = [], []
    for part in sorted(store.glob("*.parquet")):
        tasks = sorted(by_family.get(part.name.removesuffix(".parquet"), []))
        files = readable_parts(part) if tasks else []      # a truncated part is skipped with a warning
        batch = (np.cumsum([size.get(t, 0) for t in tasks]) // ROW_BUDGET) if files else []
        for b in np.unique(batch):
            s = pq.read_table(files, columns=["model", "step", "task", "doc_id", "acc", "acc_norm"],
                              filters=[("task", "in", [t for t, k in zip(tasks, batch) if k == b]),
                                       ("model", "in", models)]).to_pandas()
            for task, g in s.groupby("task", observed=True):
                metric, chance = metric_for(task) or "acc", task_chance(task)
                row = {"task": task, "metric": metric, "chance": chance, "n_items": g["doc_id"].nunique(),
                       "n_reference_runs": len(ref.get(task, []))}
                if not np.isfinite(chance) or task not in ref or g[metric].isna().all():
                    sel.append(row | {"n_kept": np.nan})          # no chance level or no reference run: nothing to select on
                    continue
                keep, sc = select_and_score(g, metric, ref[task], chance)
                sel.append(row | {"n_kept": len(keep)})
                scores.append(sc.assign(task=task))
    out = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(columns=KEYS)
    return pd.DataFrame(sel), out.astype({"model": str, "step": int})


def final_snr(x: np.ndarray, n_items: float) -> tuple[float, float, float]:
    """(signal, noise, SNR) of one (task, size) at the final checkpoint: the
    `rel_std` aggregator's signal on the finals (it reads no window) over the
    relative k-fold benchmark noise on `n_items` items, the catalogue's
    `snr__rel_std__kfold_rel`."""
    signal = variant_signal_noise_snr((np.full_like(x, np.nan), x, np.full_like(x, np.std(x)), x), rel_std_snr)[0]
    noise = kfold_noise(x, n_items)["kfold_rel"]
    return signal, noise, signal / noise if noise > 0 else np.nan


def da_cells(dec: pd.DataFrame, mask: pd.DataFrame, pool: str, ref: str | None, keys: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (task, pair set, size[, frac]) DA of decision rows `dec` (rule 5, then
    rule 1 at the size and at `ref`, gated cells kept blank), and the same pooled
    per `keys` with the leave-one-family-out jackknife band, beside the mean over tasks."""
    cells = reliability(dec).rename(columns={"group": "axes", "n_comparable": "n_pairs"}).drop(columns="compute")
    cells.loc[cells["n_pairs"] < MIN_PAIRS, "da"] = np.nan                 # rule 5
    cells = G.mark_gated(cells, pool, "size", "da", ref, mask=mask)        # rule 1
    kept = cells.dropna(subset=["da"])
    d = (dec.astype({c: str for c in ("task", "group", "size", "family_a", "family_b")}).rename(columns={"group": "axes"})
         .merge(kept[["task", "axes", "size", "frac"]], on=["task", "axes", "size", "frac"]))
    if d.empty:
        return cells, pd.DataFrame(columns=keys)
    pooled = jackknife_ratio(d, keys).merge(
        kept.groupby(keys).agg(da_macro=("da", "mean"), n_tasks=("task", "nunique")).reset_index(), on=keys)
    return cells, pooled


def rule5(cells: pd.DataFrame, what: str) -> pd.DataFrame:
    """Rule 5's report: the cells left NaN for fewer than MIN_PAIRS pairs, per ordering and pair set."""
    few = cells[cells["n_pairs"] < MIN_PAIRS]
    t = few.groupby(["ordering", "axes"]).agg(n_cells=("task", "size"), n_tasks=("task", "nunique")).reset_index()
    if len(few):
        print(f"!!! RULE 5: {len(few)} {what} cells over {few['task'].nunique()} tasks had fewer than {MIN_PAIRS} pairs "
              f"and are NaN (their pair counts are in the per-task table); per ordering and pair set:\n{t.to_string(index=False)}")
    return t


def gain(cells: dict, value: str, keys: list, ratio: bool = False) -> pd.DataFrame:
    """Each sub-benchmark ordering against the full benchmark on the cells both
    have a value for: the difference (or the ratio) of `value` per cell."""
    out = []
    for o in ("gate_then_items", "items_then_gate"):
        m = cells[o].merge(cells["full"], on=keys, suffixes=("", "_full")).dropna(subset=[value, f"{value}_full"])
        if ratio:
            m = m[(m[value] > 0) & (m[f"{value}_full"] > 0)]
        out.append(m.assign(ordering=o, delta=m[value] / m[f"{value}_full"] if ratio else m[value] - m[f"{value}_full"]))
    return pd.concat(out, ignore_index=True)


def admitted(cells: dict, keys: list, value: str) -> pd.DataFrame:
    """The cells items, then gate reads that the committed gate blanks on the full
    benchmark: what reducing the items first lets in."""
    m = cells["items_then_gate"].merge(cells["full"][[*keys, "gated"]], on=keys, how="left", suffixes=("", "_full"))
    return m[m[value].notna() & m["gated_full"].fillna(True).astype(bool)]


# --- figures --------------------------------------------------------------------------------------------

def save(fig, out_dir: Path, stem: str, title: str, note: str, table: pd.DataFrame) -> None:
    """`<stem>.png` with its title and caption line, `<stem>.csv` the values it draws (rule 12)."""
    top = G._header(fig, title, note)
    fig.tight_layout(rect=(0, 0, 1, top))
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    S.save(fig, out_dir / f"{stem}.png", dpi=150)


def lines(ax, t: pd.DataFrame, y: str, sizes: list, lo: str | None = None, hi: str | None = None,
          count: str | None = None, orderings=tuple(ORDERINGS), counted=("full", "items_then_gate")) -> None:
    """One line per ordering along `sizes`, with its band and its count next to the
    points (gate, then items reads the full benchmark's cells: not counted twice),
    above the higher of the counted points at a size and below the lower one."""
    x = {s: i for i, s in enumerate(sizes)}
    top = t[t["ordering"].isin(counted)].dropna(subset=[y]).sort_values(y).groupby("size")["ordering"].last()
    for o in orderings:
        g = t[t["ordering"] == o].set_index("size").reindex(sizes)
        ax.plot(range(len(sizes)), g[y], "-o", ms=3.5, lw=1.4, color=COLOUR[o], label=ORDERINGS[o])
        if lo:
            ax.fill_between(range(len(sizes)), g[lo], g[hi], color=COLOUR[o], alpha=.12, lw=0)
        if count and o in counted:
            for s, r in g.dropna(subset=[y]).iterrows():
                ax.annotate(f"{int(r[count])}", (x[s], r[y]), textcoords="offset points", ha="center", fontsize=5.5,
                            xytext=(0, 5 if top.get(s) == o else -9), color=COLOUR[o],
                            bbox=dict(boxstyle="square,pad=0.05", fc=S.SURFACE, ec="none", alpha=.85))
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_xlabel("model size")
    ax.grid(color=S.GRID, lw=.6); S.clean(ax)


GAIN = "gate, then items − full (paired)"


def per_benchmark(cells: dict, paired: pd.DataFrame, value: str, stat: str, keys: list) -> pd.DataFrame:
    """Per benchmark and `keys`, one block per heat-map panel: each ordering's
    `stat` of `value` over its tasks, then the paired gain of gate, then items
    over the full benchmark (`delta`); `n_tasks` behind each value, `all_gated`
    where every task of the benchmark is gated there (grey)."""
    bench = lambda t: t.assign(benchmark=t["task"].map(benchmark_family))
    out = {o: bench(c).groupby(["benchmark", *keys]).agg(value=(value, stat), n_tasks=(value, "count"),
                                                         all_gated=("gated", "all")).reset_index().assign(panel=ORDERINGS[o])
           for o, c in cells.items()}
    p = bench(paired[paired["ordering"] == "gate_then_items"])
    gain_ = (p.groupby(["benchmark", *keys]).agg(value=("delta", stat), n_tasks=("delta", "count")).reset_index()
             .merge(out["gate_then_items"][["benchmark", *keys, "all_gated"]], on=["benchmark", *keys], how="right")
             .assign(panel=GAIN))          # gate, then items reads the full benchmark's gate: its grey is the gain's
    t = pd.concat([*out.values(), gain_], ignore_index=True)
    return t.assign(n_tasks=t["n_tasks"].fillna(0).astype(int))


def heat(t: pd.DataFrame, sizes: list, styles: dict, out_dir: Path, stem: str, title: str, note: str) -> None:
    """Benchmark x size heat maps, one per panel of `t` (`styles`: panel -> cmap,
    vmin, vmax, centre, format); the task count under the value, grey where every
    task of the benchmark is gated there, white where there is no value (rule 12)."""
    rows = sorted(t["benchmark"].unique(), key=G.display)
    fig, axes = plt.subplots(1, len(styles), figsize=(2.6 * len(styles) + 1.8, 0.22 * len(rows) + 1.8), squeeze=False)
    for k, (ax, (panel, (cmap, lo, hi, centre, f))) in enumerate(zip(axes[0], styles.items())):
        piv = lambda c: (t[t["panel"] == panel].pivot_table(index="benchmark", columns="size", values=c, aggfunc="first")
                         .reindex(index=rows, columns=sizes))
        G.matrix_ax(ax, piv("value"), panel, cnt=piv("n_tasks"), vmin=lo, vmax=hi, cmap=cmap, center=centre, fmt=f,
                    xlabel="model size", gated=piv("all_gated").astype(float).fillna(0).astype(bool))
        if k:
            ax.set_yticklabels([])
    save(fig, out_dir, stem, title, note, t[t["size"].isin(sizes)])


# --- the run ---------------------------------------------------------------------------------------------

def main(pool: str, store_pool: str, out_dir: Path, readme: bool) -> None:
    store = STORE / store_pool
    if not any(store.glob("*.parquet")):
        print(f"no per-item store at {store}: nothing written (build_per_item_store.sbatch builds it)")
        return
    mask0 = load_mask(pool) if load_mask(pool) is not None else load_mask(CANONICAL_POOL)
    if mask0 is None:
        print("no committed above-random mask: nothing written (the gate's pass writes it first)")
        return
    df = ladder_frame(pool)
    bench = df[(df["kind"] == "benchmark") & ~df["task"].str.startswith(BBPB)]
    fin_keys = finals(bench)[KEYS]
    store_gap(bench["model"], store)                                       # printed: the pool models the store lacks
    sel, sc = store_scores(store, bench.merge(fin_keys), sorted(set(bench["model"])))
    S_ = bench.merge(sc, on=KEYS)
    fin = S_.merge(fin_keys)                                                # the frame's finals, where the store has them
    have = sel[sel["n_kept"] > 0]["task"]
    fams = fin[fin["task"].isin(have)].groupby(["task", "size"])["family"].nunique()
    usable = fams[fams.index.get_level_values("size").isin(SMALL_SIZES) & (fams * (fams - 1) // 2 >= MIN_PAIRS)]
    if usable.empty:
        print(f"{store}: no task has above-chance items at {TARGET_SIZE} and >= {MIN_PAIRS} pairs at a proxy for "
              f"the pool {pool}: nothing written (the store does not cover the pool)")
        return
    ckpts = bool((S_["frac"] < 1 - FRAC_TOL).any())
    print(f"!!! RULE 11: {IN_SAMPLE}")
    print(f"{pool}: {S_['model'].nunique()} models, store {store_pool}: {len(sel)} tasks of the pool's frame, "
          f"{sel['n_kept'].notna().sum()} with a chance level and a {TARGET_SIZE} run, {len(have)} with an item above chance; "
          + ("checkpoints in the store" if ckpts else "finals only"))
    if not ckpts:
        print("!!! skipped (the store holds finals only): DA-ckpt and the checkpoint SNR; they fill in once "
              "build_per_item_store.sbatch has built the store over every checkpoint")
    print("!!! skipped: the surrogates (the catalogue reads the item count by task name and its truths from the "
          "decision-accuracy table on disk, so it cannot take a substituted score frame)")
    d = (fin["full"] - fin["primary_score"]).abs()
    checks = f"the store's full-benchmark score equals the ladder report's within 1e-3 on {(d < 1e-3).mean():.1%} of {len(d)} finals"

    # the three orderings: scores, gate, item counts
    sel = sel.assign(benchmark=sel["task"].map(benchmark_family), language=sel["task"].map(assign_language),
                     kept_share=sel["n_kept"] / sel["n_items"])
    base_tasks = sel.loc[sel["n_kept"].notna(), "task"]
    n_full, n_kept = sel.set_index("task")["n_items"], sel.set_index("task")["n_kept"]
    sub_frame = S_[S_["task"].isin(have)].assign(primary_score=lambda x: x["sub"])
    mask_sub = scores_and_mask(sub_frame, n_items=n_kept.to_dict())[1]
    mask_re = scores_and_mask(S_[S_["task"].isin(base_tasks)].assign(primary_score=lambda x: x["full"]), n_items=n_full.to_dict())[1]
    lv = ["task", "size"]                     # the committed mask's columns carry no name: align before joining
    both = (mask_re.stack().rename_axis(lv).to_frame("re")
            .join(mask0.stack().rename_axis(lv).rename("committed"), how="inner").dropna())
    checks += (f"; the gate recomputed on them by the same code agrees with the committed mask on "
               f"{(both['re'] == both['committed']).sum()} of {len(both)} (task, size) cells")
    print(f"  checks: {checks}")
    orders = {"full": (S_[S_["task"].isin(base_tasks)].assign(primary_score=lambda x: x["full"]), mask0, n_full),
              "gate_then_items": (sub_frame, mask0, n_kept),
              "items_then_gate": (sub_frame, mask_sub, n_kept)}
    ref_gate = lambda m: sel["task"].map(m[TARGET_SIZE]) if TARGET_SIZE in m.columns else np.nan
    sel = sel.assign(gate_full_at_reference=ref_gate(mask0), gate_items_at_reference=ref_gate(mask_sub))

    attrs = design_axes(df)
    psets = pair_sets(attrs)
    groups = {a: psets[a] for a in AXES if psets[a]}
    sizes = [s for s in ANALYSIS_SIZES if s in set(fin["size"])]
    da, da_pool, snr, fits, ck, ck_pool, share = {}, [], {}, {}, {}, [], []
    for o, (frame, mask, n_items) in orders.items():
        f_o = fin.merge(frame[KEYS + ["primary_score"]].rename(columns={"primary_score": "score"}), on=KEYS)
        f_o = f_o.assign(primary_score=f_o["score"]).drop(columns="score")
        tasks = sorted(set(f_o["task"]))
        for s in sizes:                                                    # the gate's pass share
            col = mask[s].reindex(tasks) if s in mask.columns else pd.Series(np.nan, index=tasks)
            refc = mask[TARGET_SIZE].reindex(tasks)
            share.append({"ordering": o, "size": s, "n_tasks": len(base_tasks), "n_pass": int((col == 1).sum()),
                          "n_pass_with_reference": int(((col == 1) & (refc == 1)).sum())})
        cells, pooled = da_cells(decisions(f_o.assign(frac=1.0), groups, SMALL_SIZES, f_o), mask, pool, TARGET_SIZE,
                                 ["axes", "size"])
        flat = f_o.groupby(["task", "size"])["primary_score"].nunique() <= 1      # every variant scores alike
        flat = set(flat.index[flat])
        cells["tied"] = [(t, s) in flat or (t, TARGET_SIZE) in flat for t, s in zip(cells["task"], cells["size"])]
        da[o] = cells.drop(columns="frac").assign(ordering=o)
        da_pool.append(pooled.assign(ordering=o))
        rows = []
        for (t, s), g in f_o.groupby(["task", "size"]):
            if g["primary_score"].notna().sum() >= 2:
                sig, noi, r = final_snr(g["primary_score"].dropna().to_numpy(float), float(n_items[t]))
                rows.append({"task": t, "size": s, "n_items": int(n_items[t]), "signal": sig, "noise_kfold": noi, "snr": r})
        t_snr = pd.DataFrame(rows)
        if ckpts:                                                          # noise and SNR's own: the checkpoint window
            fr = frame.assign(bucket=frame["size"].map(size_bucket))
            by_task = {t: g for t, g in fr.groupby("task")}
            got = [variant_signal_noise_snr(per_model_inputs(by_task[t], t, s), rel_std_snr) for t, s in zip(t_snr["task"], t_snr["size"])]
            t_snr = t_snr.join(pd.DataFrame(got, columns=["signal_ckpt", "noise_ckpt", "snr_ckpt"], index=t_snr.index))
            grid = frame[on_shared_grid(frame)]
            dec = pd.concat([decisions(grid_frame(grid[grid["size"] == b], CKPT_DA_EARLY_FRACS), groups, [b], f_o,
                                       CKPT_DA_EARLY_FRACS, ref=b) for b in sizes], ignore_index=True)
            c, p = da_cells(dec, mask, pool, None, ["axes", "size", "frac"])   # DA-ckpt reads its own size: rule 1 there alone
            ck[o] = c.assign(ordering=o)
            ck_pool.append(p.assign(ordering=o))
        t_snr = G.mark_gated(t_snr, pool, "size", "snr", mask=mask)        # rule 1 at the size
        t_snr.loc[t_snr["gated"], [c for c in t_snr.columns if c.startswith(("signal", "noise", "snr"))]] = np.nan
        snr[o] = t_snr.assign(ordering=o)
        f = fit_table(f_o, pool, mask=mask)[0]
        fits[o] = f[f["kind"] == "benchmark"].assign(ordering=o)

    # tables
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = lambda t: languages_only(t.assign(benchmark=t["task"].map(benchmark_family), language=t["task"].map(assign_language)))
    sel = languages_only(sel)
    sel.to_csv(out_dir / f"{NAME}_selection.csv", index=False)
    surv = survival(sel)
    surv.to_csv(out_dir / f"{NAME}_survival.csv", index=False)
    share = pd.DataFrame(share).assign(share=lambda x: x["n_pass"] / x["n_tasks"],
                                       share_with_reference=lambda x: x["n_pass_with_reference"] / x["n_tasks"])
    cells = meta(pd.concat(da.values(), ignore_index=True))
    cells[["ordering", "task", "benchmark", "language", "axes", "size", "da", "n_pairs", "n_matching", "gated", "tied"]].to_csv(
        out_dir / f"{NAME}_da_size_per_task_both_axes.csv", index=False)
    r5 = rule5(cells, "DA-size")
    adm = admitted(da, ["task", "axes", "size"], "da")
    adm_bins = (adm.merge(sel[["task", "n_kept"]], on="task")
                .assign(kept=lambda x: pd.cut(x["n_kept"], KEPT_BINS, labels=KEPT_LABELS))
                .groupby(["axes", "kept"], observed=True).agg(da_mean=("da", "mean"), n_cells=("da", "count")).reset_index())
    tied = (cells.dropna(subset=["da"]).groupby(["ordering", "axes"], sort=False)
            .agg(n_cells=("da", "count"), n_tied=("tied", "sum"), da_tied=("da", lambda v: v[cells.loc[v.index, "tied"]].mean()))
            .reset_index())
    g_da = gain(da, "da", ["task", "axes", "size"])
    paired = g_da.groupby(["ordering", "axes", "size"]).agg(gain_paired=("delta", "mean"), n_paired=("delta", "count")).reset_index()
    da_pool = (pd.concat(da_pool, ignore_index=True).merge(paired, on=["ordering", "axes", "size"], how="left")
               .merge(adm.groupby(["axes", "size"]).agg(da_macro_admitted=("da", "mean"), n_admitted=("task", "nunique"))
                      .reset_index().assign(ordering="items_then_gate"), on=["ordering", "axes", "size"], how="left"))
    snr_all = meta(pd.concat(snr.values(), ignore_index=True))
    snr_all.to_csv(out_dir / f"{NAME}_snr_per_task.csv", index=False)
    g_snr = gain(snr, "snr", ["task", "size"], ratio=True)
    g_sig = gain(snr, "signal", ["task", "size"], ratio=True)
    g_noi = gain(snr, "noise_kfold", ["task", "size"], ratio=True)
    rho = []
    for o in ORDERINGS:
        m = snr[o].merge(da[o][da[o]["axes"] == AXES[0]], on=["task", "size"]).dropna(subset=["snr", "da"])
        for s, g in m.groupby("size"):
            r = spearmanr(g["snr"], g["da"]) if len(g) >= 3 else None
            rho.append({"ordering": o, "size": s, "rho_snr_da_size": r.statistic if r else np.nan,
                        "p": r.pvalue if r else np.nan, "n_tasks_rho": len(g)})
    snr_summary = (snr_all.groupby(["ordering", "size"]).agg(snr_median=("snr", "median"), n_tasks=("snr", "count")).reset_index()
                   .merge(pd.concat([g.groupby(["ordering", "size"])["delta"].median().rename(n) for g, n in
                                     ((g_snr, "snr_ratio_median"), (g_sig, "signal_ratio_median"), (g_noi, "noise_ratio_median"))],
                                    axis=1).reset_index(), on=["ordering", "size"], how="left")
                   .merge(pd.DataFrame(rho), on=["ordering", "size"], how="left")
                   .merge(admitted(snr, ["task", "size"], "snr").groupby("size").agg(
                       snr_median_admitted=("snr", "median"), n_admitted=("snr", "count")).reset_index()
                       .assign(ordering="items_then_gate"), on=["ordering", "size"], how="left"))
    fits = pd.concat(fits.values(), ignore_index=True)
    fits.to_csv(out_dir / f"{NAME}_scaling_fits.csv", index=False)
    g_fit = gain({o: f for o, f in fits.groupby("ordering")}, "r2", ["task", "L"])
    scaling = (fits.groupby("ordering").agg(n_fits=("r2", "count"), r2_median=("r2", "median"), rho_median=("rho", "median"))
               .join(g_fit.groupby("ordering").agg(r2_gain_median=("delta", "median"), n_paired=("delta", "count")))
               .reindex(list(ORDERINGS)).reset_index())
    if ckpts:
        ck_cells = pd.concat(ck.values(), ignore_index=True).pipe(meta)
        ck_cells.to_csv(out_dir / f"{NAME}_da_ckpt_per_task_both_axes.csv", index=False)
        rule5(ck_cells, "DA-ckpt")
        pd.concat(ck_pool, ignore_index=True).to_csv(out_dir / f"{NAME}_da_ckpt_both_axes.csv", index=False)
    print(surv.to_string(index=False))
    print(da_pool[["ordering", "axes", "size", "reliability", "lo", "hi", "da_macro", "n_tasks", "gain_paired"]].round(3).to_string(index=False))
    print(snr_summary.round(3).to_string(index=False))
    print(scaling.round(3).to_string(index=False))

    # figures
    n_fam = fin.groupby("size")["family"].nunique()
    n_fam = f"{n_fam.min()}" if n_fam.min() == n_fam.max() else f"{n_fam.min()}-{n_fam.max()}"
    cells_note = (f"pool {pool}: {n_fam} design variants per size, {', '.join(sizes)}; per-item store "
                  f"{store_pool} ({'checkpoints' if ckpts else 'finals only'}); items above chance = mean over the "
                  f"{TARGET_SIZE} runs > the task's chance level. RULE 11 waived: {IN_SAMPLE}.")
    gate_note = ("gate: the committed mask (full; gate, then items, which reads the full benchmark's cells and coincides "
                 "with it) or the gate recomputed on the kept items (items, then gate)")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, (y, lab) in zip(axes, (("share", "above chance at the size"), ("share_with_reference", f"... and at {TARGET_SIZE}"))):
        lines(ax, share, y, sizes, count=y.replace("share", "n_pass"))
        ax.set_title(lab, loc="left", fontsize=8.5); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(f"share of the {len(base_tasks)} tasks"); axes[0].legend(fontsize=7, frameon=False)
    save(fig, out_dir, f"{NAME}_gate_pass_share", "Above-chance items: the above-random gate's pass share",
         f"point = share of the tasks with a chance level whose cell the gate passes (count next to the point); {gate_note}. "
         + cells_note, share)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 3.8), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], groups):
        lines(ax, da_pool[da_pool["axes"] == a], "reliability", SMALL_SIZES, "lo", "hi", "n_tasks")
        ax.axhline(.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_title(f"{a} pairs ({len(groups[a])})", loc="left", fontsize=8.5); ax.set_xlabel("proxy size")
    axes[0][0].set_ylabel(f"DA-size against the {TARGET_SIZE} final (pooled)"); axes[0][0].legend(fontsize=7, frameon=False)
    save(fig, out_dir, f"{NAME}_da_size_both_axes", "Above-chance items: decision accuracy",
         f"DA-size = share of the design-variant pairs a proxy's final orders like the {TARGET_SIZE} final, pooled over the "
         f"tasks (count next to the point) with >= {MIN_PAIRS} pairs and above chance at the proxy and at {TARGET_SIZE}; "
         f"band = leave-one-family-out jackknife, 90 %; {gate_note}. " + cells_note, da_pool)
    t = per_benchmark(da, g_da, "da", "mean", ["axes", "size"])
    styles = {ORDERINGS[o]: (S.DIV, 0.0, 1.0, 0.5, "{:.2f}") for o in ORDERINGS} | {GAIN: (S.DIV, -0.3, 0.3, 0.0, "{:+.2f}")}
    for a in groups:
        heat(t[t["axes"] == a], SMALL_SIZES, styles, out_dir, f"{NAME}_da_size_by_benchmark{AXES_SUFFIX[a]}",
             f"Above-chance items: DA-size per benchmark, {a} pairs",
             "cell = mean DA-size over the benchmark's languages (count under the value) on the full benchmark, on its "
             "above-chance items under each ordering, and the paired gain of gate, then items over the full benchmark "
             f"on the same cells; grey = every task gated (at the proxy or at {TARGET_SIZE}); {gate_note}. " + cells_note)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    lines(axes[0], snr_summary, "snr_median", sizes, count="n_tasks")
    axes[0].set_ylabel("median SNR over the tasks"); axes[0].legend(fontsize=7, frameon=False)
    lines(axes[1], snr_summary, "snr_ratio_median", sizes, orderings=("gate_then_items",))
    g = snr_summary[snr_summary["ordering"] == "gate_then_items"].set_index("size").reindex(sizes)
    for c, ls, lab in (("signal_ratio_median", "--", "gate, then items: signal"), ("noise_ratio_median", ":", "gate, then items: k-fold noise")):
        axes[1].plot(range(len(sizes)), g[c], ls, lw=1.2, color=COLOUR["gate_then_items"], label=lab)
    axes[1].axhline(1, color=S.MUTED, lw=.8); axes[1].set_ylabel("median ratio, kept / full (paired cells)")
    axes[1].set_ylim(0.3, None); axes[1].legend(fontsize=6.5, frameon=False, loc="lower left")
    lines(axes[2], snr_summary[snr_summary["size"].isin(SMALL_SIZES)], "rho_snr_da_size", SMALL_SIZES, count="n_tasks_rho")
    axes[2].axhline(0, color=S.MUTED, lw=.8); axes[2].set_ylim(-1, 1); axes[2].set_ylabel("Spearman ρ(SNR, DA-size multi-axis)")
    for ax, lab in zip(axes, ("SNR", "gain over the full benchmark", "does SNR track DA-size?")):
        ax.set_title(lab, loc="left", fontsize=8.5)
    save(fig, out_dir, f"{NAME}_snr", "Above-chance items: signal-to-noise ratio",
         "SNR at the final checkpoint = the spread of the design variants' finals (rel_std signal, std / mean) over the "
         "relative k-fold benchmark noise on the task's (kept) items; cells above chance at the size (count next to the "
         "point); ratios over the cells both readings have (items, then gate reads the same sub-scores there, so its "
         f"ratio is in the CSV only); ρ over the tasks with a DA-size value at the proxy; {gate_note}. "
         + cells_note, snr_summary)
    t = per_benchmark(snr, g_snr.assign(delta=np.log2(g_snr["delta"])), "snr", "median", ["size"])
    hi = float(np.nanpercentile(t.loc[t["panel"] != GAIN, "value"], 95))
    heat(t, sizes, {ORDERINGS[o]: (S.SEQ, 0.0, hi, None, "{:.2f}") for o in ORDERINGS} | {GAIN: (S.DIV, -2.0, 2.0, 0.0, "{:+.2f}")},
         out_dir, f"{NAME}_snr_by_benchmark", "Above-chance items: SNR per benchmark",
         "cell = median final-checkpoint SNR over the benchmark's languages (count under the value) on the full "
         "benchmark and on its above-chance items under each ordering; last panel: median log2(SNR kept / SNR full) "
         f"over the cells both have; grey = every task gated at the size; {gate_note}. " + cells_note)
    if readme:
        write_readme(pool, store_pool, ckpts, checks, sel, surv, share, da_pool, g_da, r5, adm_bins, tied, snr_summary,
                     scaling, sizes)


def survival(sel: pd.DataFrame) -> pd.DataFrame:
    """Per ordering, the tasks and items left after each of its steps."""
    base = sel[sel["n_kept"].notna()]
    any_kept = base[base["n_kept"] > 0]
    rows = [("full", "tasks with a chance level and a reference run", base),
            ("full", f"... the gate passes at {TARGET_SIZE}", base[base["gate_full_at_reference"] == 1]),
            ("gate_then_items", f"tasks the gate passes at {TARGET_SIZE}", base[base["gate_full_at_reference"] == 1]),
            ("gate_then_items", f"... with an item above chance at {TARGET_SIZE}",
             base[(base["gate_full_at_reference"] == 1) & (base["n_kept"] > 0)]),
            ("items_then_gate", f"tasks with an item above chance at {TARGET_SIZE}", any_kept),
            ("items_then_gate", f"... the recomputed gate passes at {TARGET_SIZE}", any_kept[any_kept["gate_items_at_reference"] == 1])]
    return pd.DataFrame([{"ordering": o, "step": st, "n_tasks": len(t), "n_items": int(t["n_items"].sum()),
                          "n_kept": int(t["n_kept"].sum()) if o != "full" else int(t["n_items"].sum())} for o, st, t in rows])


def write_readme(pool: str, store_pool: str, ckpts: bool, checks: str, sel: pd.DataFrame, surv: pd.DataFrame,
                 share: pd.DataFrame, da_pool: pd.DataFrame, g_da: pd.DataFrame, r5: pd.DataFrame, adm_bins: pd.DataFrame,
                 tied: pd.DataFrame, snr: pd.DataFrame, scaling: pd.DataFrame, sizes: list) -> None:
    """The `results` block of this folder's README: every number the prose quotes, from the tables just written."""
    snap = datetime.fromtimestamp((ladder_dir() / "ladder_report.csv").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    sign = lambda v: f"{v:+.3f}" if np.isfinite(v) else ""
    with_n = lambda v, n, f="{:.3f}": f"{f.format(v)} ({int(n)})" if np.isfinite(v) else ""
    row = lambda t, o, cols: t[t["ordering"] == o].set_index("size").reindex(cols)
    tab_surv = md_table(["ordering", "step", "tasks", "items", "items kept"],
                        [[ORDERINGS[r.ordering], r.step, r.n_tasks, r.n_items,
                          f"{r.n_kept} ({r.n_kept / r.n_items:.0%})" if r.n_items else ""] for r in surv.itertuples()])
    tab_share = md_table(["ordering", *sizes], [[ORDERINGS[o], *[f"{r.share:.2f} ({r.n_pass})" for r in row(share, o, sizes).itertuples()]]
                                                for o in ORDERINGS])
    da_rows, gain_rows = [], []
    for a in [x for x in AXES if x in set(da_pool["axes"])]:
        t = da_pool[da_pool["axes"] == a]
        for o in ORDERINGS:
            da_rows.append([a, ORDERINGS[o], *[f"{r.reliability:.3f} [{r.lo:.2f}, {r.hi:.2f}] ({int(r.n_tasks)})"
                                               if np.isfinite(r.reliability) else "" for r in row(t, o, SMALL_SIZES).itertuples()]])
        f, g, i = (row(t, o, SMALL_SIZES) for o in ORDERINGS)
        gain_rows += [[a, "gate, then items − full: mean paired Δ over the tasks (cells)",
                       *[with_n(v, n, "{:+.3f}") for v, n in zip(g["gain_paired"], g["n_paired"])]],
                      [a, "gate, then items − full: Δ of the pooled DA-size (the figure's lines)",
                       *[sign(v) for v in g["reliability"] - f["reliability"]]],
                      [a, "items, then gate: mean DA on the admitted cells (cells)",
                       *[with_n(v, n) for v, n in zip(i["da_macro_admitted"], i["n_admitted"].fillna(0))]]]
    tab_da = md_table(["pairs", "ordering", *SMALL_SIZES], da_rows)
    tab_gain = md_table(["pairs", "reading", *SMALL_SIZES], gain_rows)
    k = ["task", "axes", "size"]
    j = (g_da[g_da["ordering"] == "gate_then_items"].set_index(k)["delta"].to_frame("g")
         .join(g_da[g_da["ordering"] == "items_then_gate"].set_index(k)["delta"].rename("i"), how="outer"))
    same = f"{int(((j['g'] - j['i']).abs() < 1e-12).sum())} of {len(j)}"
    r = r5[["n_cells", "n_tasks"]].drop_duplicates()
    emptied = ("no cell has fewer" if r5.empty else
               f"{r.n_cells.iloc[0]} cells over {r.n_tasks.iloc[0]} tasks have fewer in each ordering and pair set" if len(r) == 1 else
               "; ".join(f"{ORDERINGS[x.ordering]} {x.axes}: {x.n_cells} cells over {x.n_tasks} tasks" for x in r5.itertuples())
               + " have fewer")
    nr = sel.loc[sel["n_kept"].notna(), "n_reference_runs"]
    runs = (f"The runs that select a task's items are its {TARGET_SIZE} finals in the cells that train its language "
            f"(rule 2): {int(nr.min())}–{int(nr.max())} per task, {int((nr == nr.min()).sum())} of {len(nr)} tasks on "
            f"{int(nr.min())} (`n_reference_runs` in the selection table).")
    tab_bins = md_table(["pairs", *[f"{k} kept items" for k in KEPT_LABELS]],
                        [[a, *[with_n(r.da_mean, r.n_cells) if np.isfinite(r.da_mean) else "" for r in
                               g.set_index("kept").reindex(KEPT_LABELS).itertuples()]]
                         for a in AXES for g in [adm_bins[adm_bins["axes"] == a]] if len(g)])
    tab_tied = md_table(["pairs", "ordering", "cells with a DA-size value", "of which every variant ties", "their mean DA-size"],
                        [[r.axes, ORDERINGS[r.ordering], r.n_cells, int(r.n_tied), fmt(r.da_tied)]
                         for a in AXES for r in tied[tied["axes"] == a].itertuples()])
    g, i = row(snr, "gate_then_items", sizes), row(snr, "items_then_gate", sizes)
    tab_snr = md_table(["reading", *sizes],
                       [[f"{ORDERINGS[o]}: median SNR (tasks)", *[with_n(v, n) for v, n in zip(row(snr, o, sizes)["snr_median"], row(snr, o, sizes)["n_tasks"])]]
                        for o in ORDERINGS]
                       + [[f"gate, then items / full: median paired {lab}", *[fmt(v) for v in g[c]]]
                          for c, lab in (("snr_ratio_median", "SNR ratio"), ("signal_ratio_median", "signal ratio"),
                                         ("noise_ratio_median", "k-fold noise ratio"))]
                       + [["items, then gate: median SNR on the admitted cells (cells)",
                           *[with_n(v, n) for v, n in zip(i["snr_median_admitted"], i["n_admitted"].fillna(0))]]])
    tab_rho = md_table(["ordering", *SMALL_SIZES], [[ORDERINGS[o], *[with_n(r.rho_snr_da_size, r.n_tasks_rho, "{:.2f}")
                                                                    for r in row(snr, o, SMALL_SIZES).itertuples()]] for o in ORDERINGS])
    tab_fit = md_table(["ordering", "fits", "median R²", "median ρ with size", "median ΔR² (paired)", "paired fits"],
                       [[ORDERINGS[r.ordering], r.n_fits, fmt(r.r2_median), fmt(r.rho_median), sign(r.r2_gain_median),
                         "" if pd.isna(r.n_paired) else int(r.n_paired)] for r in scaling.itertuples()])
    skipped = ("DA-ckpt and the checkpoint SNR are in `above_chance_items_da_ckpt_*` and the `*_ckpt` columns of the SNR table."
               if ckpts else "**Skipped: DA-ckpt and the checkpoint SNR** — the store holds each run's final checkpoint only; "
               "they fill in, without a code change, once the store is built over every checkpoint.")
    body = "\n\n".join([
        "## Results",
        f"Pool `{pool}` ({', '.join(sizes)}), per-item store `{store_pool}`, ladder report of {snap}. "
        f"**Rule 11 is waived by design:** {IN_SAMPLE}. {skipped} The surrogates are not computed (the catalogue "
        "reads a task's item count by its name and its truths from the decision-accuracy table on disk). "
        f"Checks: {checks}. Regenerate with "
        f"`python analysis/rq12_above_chance_items/above_chance_items.py --pool {pool} --store-pool {store_pool}`.",
        runs,
        "**What survives each step** (items kept = items above chance at the reference, over the tasks of the row):", tab_surv,
        "**The gate's pass share** per size (tasks passing, over every task with a chance level and a reference run; "
        "gate, then items reads the committed gate and coincides with the full benchmark):", tab_share,
        f"**DA-size** against the {TARGET_SIZE} final, pooled over the tasks, its 90 % leave-one-family-out jackknife band "
        f"and the task count; rule 1 at the proxy and at {TARGET_SIZE}, ≥ {MIN_PAIRS} pairs (rule 5: {emptied} and are "
        "NaN); the task count moves along a row and between the orderings (rule 13):", tab_da,
        "**The DA-size gain**: the paired difference over the full benchmark averaged over the (task, proxy) cells both "
        "readings have, the difference of the pooled lines (which weight a task by its pairs), and the cells items, then "
        "gate admits that the committed gate blanks on the full benchmark. Items, then gate − full equals gate, then "
        f"items − full on {same} paired (task, pair set, proxy) cells (the same sub-scores wherever both have a value), "
        f"so only the latter is shown; both are in `{NAME}_da_size_both_axes.csv`:", tab_gain,
        "The admitted cells by the number of items their task keeps (mean DA-size over every proxy, cells):", tab_bins,
        "**Tied sub-benchmarks.** Where the models answer a two-option task by a constant bias, the items above chance are "
        "the ones whose gold matches it, and every design variant scores alike on them; a pair tied at the proxy and at the "
        f"reference counts as agreeing (decision accuracy's tie convention), so such a cell reads DA-size 1. Cells where every "
        f"variant ties at the proxy or at {TARGET_SIZE} (`tied` in the per-task table):", tab_tied,
        "**SNR** at the final checkpoint (rel_std signal over the relative k-fold noise of the task's items; tasks above "
        "chance at the size) and its gain on the paired cells:", tab_snr,
        "**Does SNR track DA-size?** Spearman ρ(SNR, DA-size multi-axis) over the tasks at each proxy (tasks):", tab_rho,
        "**Scaling predictability**: the log-N fit of the final scores per (task, L) on the deep data-A cells at the grid "
        "seed, over the rungs the ordering's gate passes:", tab_fit])
    replace_block(ABOVE_CHANCE_ITEMS / "README.md", "results", body,
                  f"above_chance_items.py --pool {pool} --store-pool {store_pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="the analysis pool: its models, families and gate")
    ap.add_argument("--store-pool", default=BBPB_POOL,
                    help="the per-item store folder to read (rows filtered to the pool's models and checkpoints)")
    ap.add_argument("--out-dir", type=Path, default=None, help="write here instead of pretraining/<pool>/ (no README)")
    a = ap.parse_args()
    default = ABOVE_CHANCE_ITEMS / load_pools()[a.pool].get("stage", "pretraining") / a.pool
    main(a.pool, a.store_pool, a.out_dir or default, readme=a.out_dir is None and a.pool == CANONICAL_POOL)
