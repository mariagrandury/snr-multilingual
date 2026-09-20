"""rq04 per benchmark and per language: which SNR definition, and which cheap
statistic, tracks decision accuracy where.

    snr_definition_by_language.png   Pearson r of log10 SNR with DA, SNR definition x language (DA-size = proxy → 1.7B,
                                     DA-ckpt = the proxy sizes' own early checkpoints, pooled); a language needs
                                     MIN_LANG_TASKS tasks with an SNR and a DA (rule 8)
    surrogates_by_benchmark.png      Spearman rho of each statistic with DA-size, statistic x proxy size, per benchmark
    surrogates_by_language.png       the same per language
    min_level_by_L.png               smallest proxy size (DA-size) / earliest checkpoint of the reference (DA-ckpt)
                                     at which each measurement reads the decision at DA >= SAFE_DA, per language count
    min_level_by_L_lines.png         the same as lines, one per measurement
    snr_variant_min_size_by_L.png    smallest proxy size at which an SNR definition's Spearman rho with DA (across the
                                     L's benchmark tasks in the languages every variant at the L trains on; BPB is a
                                     separate population, see the CSV's `kind`) reaches RHO_MIN, per language count,
                                     DA-size and DA-ckpt
    snr_variant_min_size_by_L_lines.png   the same as lines, one per definition
    min_level_by_L_b.png             version B, every pair pooled (the headline pool): per measurement and proxy size,
                                     the earliest checkpoint (0.5C-5C) at DA >= SAFE_DA against the reference, and the
                                     DA at 1C itself; min_level_by_L_lines_b.png the same as lines
    snr_variant_min_size_by_L_b.png  version B, every pair pooled: Spearman rho of each definition with DA-size and
                                     with DA-ckpt per proxy size; snr_variant_min_size_by_L_lines_b.png as lines
    min_level_by_L_flops.png         version per FLOPs, every pair pooled: DA against the reference of every (proxy size,
                                     checkpoint) cell at its training compute, one line per measurement
    snr_variant_min_size_by_L_flops.png   the same for the SNR definitions: rho between log10 SNR at the size and the
                                     DA of each (size, checkpoint) cell, at the cell's compute

The first reads `snr_variant_ranking.csv` (the per-language pooled r that
`analyze_snr_variants.py` writes). The second reruns `analyze.surrogates` on
one benchmark's, or one language's, tasks; a subplot needs MIN_TASKS tasks
at a proxy size, so most languages stay blank, and a cell the gate emptied
(tasks at chance at the proxy or at the reference) is grey (rule 1).
The per-L figures read rq02's `da_by_L_per_task.csv` (pairs of design
variants that share the L): a measurement is `train_loss`, `bpb_macro`, the
per-language BPB (mean) or a benchmark family (mean over its gated tasks);
a level is safe when it holds at every larger level with a value. Rule 9:
the L2 ZH/ES settings stop at 1B, so L2's reference would be 1B; this pool
excludes ZH/ES and L2 has too few pairs against 1.7B, so it is blank in the
DA-size panel — a 1B reference is not implemented, and the script says so.
Version B and the FLOPs version read `da_pooled_per_task.csv`: every pair
of the pool, on the tasks of the languages each cell trains (rule 2) and
parent tasks only (rule 6); the benchmark mean's task count per size is in
the caption and on the DA-at-1C panel (rule 13).

    python analysis/rq04_surrogates/panels.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, GATE_AND_CURVES, NOISE_AND_SNR, SURROGATES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq04_surrogates.analyze import CANONICAL, KINDS, MIN_TASKS, proxy_fit_r2, surrogates  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import MIN_PAIRS, SAFE_DA  # noqa: E402
from analysis.utils import MIN_LANG_TASKS, SMALL_SIZES, TARGET_SIZE, benchmark_family, passes_gate, trained_bpb_tasks  # noqa: E402
from pretrain.ladder_report import _trained_tasks  # noqa: E402
from pretrain.launch_trainings import DATA_SCHEMES  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT_ROOT = SURROGATES
ALL_L = (1, 2, 8, 15, 30, 50)   # every language count of the grid, blank until it has pairs (L100 was dropped)
RHO_MIN = 0.3                   # an SNR definition "tracks" DA at a size once its Spearman rho over the L's tasks reaches this
RULE9_NOTE = (f"L2's reference would be 1B (the ZH/ES L2 settings stop at 1B, rule 9); this pool excludes ZH/ES and L2 has "
              f"fewer than {MIN_PAIRS} pairs against {TARGET_SIZE}, so L2 is blank in the DA-size panel; a 1B reference is not implemented")
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    rk = pd.read_csv(out_dir / "snr_variant_ranking.csv")
    # a language's r is pooled over its (task, size pair) points; `all` has one r per size pair, read as their mean
    rk = rk[((rk["column"] == "pooled") | ((rk["scope"] == "all") & (rk["column"] == "mean")))
            & rk["da_def"].isin(["da_size", "da_ckpt/da_ckpt_mix"])].copy()
    rk["da"] = rk["da_def"].map({"da_size": "DA-size", "da_ckpt/da_ckpt_mix": "DA-ckpt (proxy sizes pooled)"})
    order = rk.groupby("variant")["pearson_r"].mean().sort_values(ascending=False).index.tolist()
    # a per-language r needs MIN_LANG_TASKS distinct tasks with an SNR and a DA (rule 8): analyze_snr_variants writes NaN
    # below that, so the languages shown are the ones with a finite r, not the ones with that many CSV rows
    keep = ["all"] + sorted(rk.loc[(rk["column"] == "pooled") & rk["pearson_r"].notna(), "scope"].unique())
    rk = rk[rk["scope"].isin(keep)]
    G.panel_grid(rk, out_dir / "snr_definition_by_language.png", by="da", row="variant", col="scope", value="pearson_r",
                 row_order=order, col_order=keep, ncols=1, vmin=-1, vmax=1, center=0.0, cmap=S.DIV, fmt="{:+.1f}",
                 counts=False, cell_w=0.3,
                 order=["DA-size", "DA-ckpt (proxy sizes pooled)"], cbar="Pearson r of log10 SNR with DA",
                 xlabel=f"language (≥ {MIN_LANG_TASKS} tasks with an SNR and a DA; `all` = pooled)", ylabel="SNR definition",
                 title="Which SNR definition tracks decision accuracy, per language",
                 note="cell = Pearson r, over the language's tasks, between log10 SNR under that definition and the task's decision "
                      f"accuracy (pooled over the proxy sizes {', '.join(SMALL_SIZES)}; DA-size = proxy final → {TARGET_SIZE} final, "
                      "DA-ckpt = the proxy's early checkpoints → its final); `all` = every task, mean of the r over the columns; "
                      f"a language with fewer than {MIN_LANG_TASKS} tasks with a point is blank; `multi` is not a language")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    tables = []
    for ax, da in zip(axes, ["DA-size", "DA-ckpt (proxy sizes pooled)"]):
        r = rk[(rk["da"] == da) & (rk["scope"] == "all")].set_index("variant")["pearson_r"]
        tables.append(G.rank_ax(ax, r, f"SNR definitions by r with {da}, all tasks", k=6, xlabel="Pearson r", ref=0.0))
    per_lang = rk[(rk["da"] == "DA-size") & (rk["scope"] != "all")].dropna(subset=["pearson_r"])
    wins = per_lang.loc[per_lang.groupby("scope")["pearson_r"].idxmax(), "variant"].value_counts().astype(float)
    tables.append(G.rank_ax(axes[2], wins, f"Languages in which a definition is the best (DA-size, {per_lang['scope'].nunique()} languages)",
                            k=len(wins), xlabel="languages", fmt="{:.0f}"))
    G.save_highlights(fig, out_dir, "rq04 in one figure: which SNR definition predicts decision accuracy?",
                      f"Pearson r between log10 SNR and DA over tasks (DA-size = proxy final → {TARGET_SIZE} final, DA-ckpt = the "
                      f"proxy sizes' early checkpoints → their final, {', '.join(SMALL_SIZES)} pooled); a language counts with "
                      f"≥ {MIN_LANG_TASKS} tasks with an SNR and a DA", tables)

    v = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv", index_col=0).rename_axis("task").reset_index()
    scores = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_scores.csv")
    fits_r2 = proxy_fit_r2(SMALL_SIZES, pool)            # the R² of the fit over the rungs up to the proxy (rule 11)
    gate = load_mask(pool)
    v = G.add_meta(v)                                     # languages only (rule 7)
    proxies = [s for s in SMALL_SIZES if f"decision_acc_size_{s}" in v.columns]
    both = []
    for by, ncols in (("family", 4), ("language", 6)):
        cells, gated = [], []
        for key, g in v.groupby(by):
            t = surrogates(g.reset_index(drop=True), scores, fits_r2, gate)
            cells.append(t.assign(**{by: key}))
            # the (subplot, proxy) cells the gate emptied: enough tasks, too few above chance at the proxy and at the
            # reference; grey where they have no value (rules 1, 12)
            gated += [{by: key, "proxy": s, "gated": True} for s in proxies
                      if len(g) >= MIN_TASKS > passes_gate(gate, g["task"], s, TARGET_SIZE).sum()]
        cells = pd.concat(cells) if cells else pd.DataFrame()
        if cells.empty:
            continue
        cells = pd.concat([cells.assign(gated=False)] + [pd.DataFrame(gated).merge(pd.DataFrame({"metric": cells["metric"].unique()}), how="cross")]
                          if gated else [cells.assign(gated=False)], ignore_index=True)
        both.append(cells.rename(columns={by: "key"}).assign(unit="benchmark" if by == "family" else "language"))
        metrics = cells.groupby("metric")["rho"].mean().sort_values(ascending=False).index.tolist()
        G.panel_grid(cells, out_dir / f"surrogates_by_{'benchmark' if by == 'family' else 'language'}.png", by=by,
                     row="metric", col="proxy", value="rho", row_order=metrics, csv=False,
                     col_order=[s for s in SMALL_SIZES if s in set(cells["proxy"])], ncols=ncols, vmin=-1, vmax=1,
                     center=0.0, cmap=S.DIV, fmt="{:+.2f}", counts=False, cbar="Spearman ρ with DA-size", xlabel="proxy size",
                     note="cell = Spearman ρ, over the subplot's tasks above chance at the proxy and at "
                          f"{TARGET_SIZE}, between the statistic read at the proxy size and the task's DA-size (needs {MIN_TASKS} "
                          "tasks with a value; the R² is fitted on the rungs up to the proxy only)",
                     title="Which statistic predicts DA-size, per " + ("benchmark" if by == "family" else "language"))
    if both:                        # the two figures draw different cells: one table, `unit` says whose
        pd.concat(both).to_csv(out_dir / "surrogates.csv", index=False)
    by_L(pool, out_dir, stage)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The rankings above, without the aggregation (`{pool}` pool); a surrogate subplot needs 8 tasks at a proxy size. Regenerate with `python analysis/rq04_surrogates/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq04 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('SNR definition per language', 'snr_definition_by_language.png'), ('Surrogates per benchmark', 'surrogates_by_benchmark.png'), ('Surrogates per language', 'surrogates_by_language.png')]]
        + [f"**Per language count** (rq02's `da_by_L_per_task.csv`: pairs of design variants sharing the L, on the {TRAINED_NOTE}; a level counts when it holds at every larger level with a value; DA ≥ {SAFE_DA}, an SNR definition tracks DA at ρ ≥ {RHO_MIN}). Rule 9: {RULE9_NOTE}:"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Smallest safe level per measurement and L', 'min_level_by_L.png'), ('the same as lines', 'min_level_by_L_lines.png'), ('Smallest size at which an SNR definition tracks DA, per L', 'snr_variant_min_size_by_L.png'), ('the same as lines', 'snr_variant_min_size_by_L_lines.png')]]
        + [f"**Version B — every pair pooled, the size axis instead of the language count** (`da_pooled_per_task.csv`, ten checkpoints; the population is every design-variant pair of the pool on the tasks of the languages each cell trains (rule 2), parent tasks only (rule 6), gated at the proxy and at {TARGET_SIZE}; the benchmark mean's task count per size differs with the gate (rule 13) and is in each figure's caption and on the DA-at-1C panel):"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Earliest checkpoint per proxy size and DA at 1C', 'min_level_by_L_b.png'), ('the same as lines', 'min_level_by_L_lines_b.png'), ('Spearman rho of each SNR definition with DA per proxy size', 'snr_variant_min_size_by_L_b.png'), ('the same as lines', 'snr_variant_min_size_by_L_lines_b.png')]]
        + ["**Version per FLOPs** — every (proxy size, checkpoint) cell at its training compute, the same population as version B:"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('DA of every cell against compute', 'min_level_by_L_flops.png'), ('rho of each SNR definition against compute', 'snr_variant_min_size_by_L_flops.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


# --- per language count ------------------------------------------------------

def _measurement(task: pd.Series) -> pd.Series:
    fam = task.map(benchmark_family)
    return pd.Series(np.select([task == "train_loss", task == "bpb_macro", fam == "bpb"],
                               ["train_loss", "bpb_macro", "bpb (per language)"], fam), index=task.index)


BENCH_MEAN = "benchmarks (mean)"
FIRST_ROWS = ("train_loss", "bpb_macro", "bpb (per language)", BENCH_MEAN)


def _with_mean_benchmarks(t: pd.DataFrame) -> pd.DataFrame:
    """The benchmark tasks once more under one label: the mean over all of them."""
    bench = ~t["measurement"].isin(FIRST_ROWS[:3])
    return pd.concat([t, t[bench].assign(measurement=BENCH_MEAN)], ignore_index=True)


def _kind(task: pd.Series) -> pd.Series:
    """`KINDS[1]` for the per-language BPB, `KINDS[0]` for the rest (the two
    populations analyze.surrogates keeps apart; the aggregates go with the
    benchmarks as one task each)."""
    return pd.Series(np.where(task.str.startswith("bpb_"), KINDS[1], KINDS[0]), index=task.index)


def _min_level(cells: pd.DataFrame, row: str, level: str, levels: list, value: str) -> pd.DataFrame:
    """row x L: index in `levels` of the smallest safe level of the mean `value`."""
    m = cells.groupby([row, "L", level])[value].mean().reset_index()
    wide = m.pivot_table(index=[row, "L"], columns=level, values=value).reindex(columns=levels)
    ok = wide.ge(SAFE_DA).where(wide.notna())
    lev = (G.smallest_safe(ok).rename("level").rename_axis([row, "L"]).reset_index()
           .pivot(index=row, columns="L", values="level"))
    return G.with_gated(lev, set(zip(cells.loc[cells["gated"], row], cells.loc[cells["gated"], "L"])))


def _two_panels(out_dir: Path, name: str, size_map: pd.DataFrame, ckpt_map: pd.DataFrame, ckpt_levels: list, ckpt_label,
                title: str, note: str, ylabel: str, row_order: list, sizes: list, size_title: str, ckpt_title: str) -> None:
    Ls = sorted(set(size_map.columns) | set(ckpt_map.columns) | set(ALL_L))
    size_map = size_map.reindex(index=row_order, columns=Ls); ckpt_map = ckpt_map.reindex(index=row_order, columns=Ls)
    cols = [f"L{L}" for L in Ls]
    size_map.columns = cols; ckpt_map.columns = cols
    for kind, fn in (("", G.level_ax), ("_lines", G.level_lines_ax)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 0.3 * len(row_order) + 2.4) if not kind else (15, 4.6))
        xl, yl = ("language count", ylabel) if not kind else ("", "language count")
        tables = [fn(axes[0], size_map, size_title, levels=sizes, xlabel=xl or "proxy size", ylabel=yl),
                  fn(axes[1], ckpt_map, ckpt_title, levels=ckpt_levels, level_label=ckpt_label,
                     xlabel=xl or "reference's checkpoint")]
        G.save_highlights(fig, out_dir, title, note, tables, name=f"{name}{kind}")


def trained_at(L: int) -> set[str]:
    """The tasks (benchmarks and per-language BPB) in the languages EVERY
    design variant at L trains on: the intersection of the lists of the
    schemes the registry (`DATA_SCHEMES`) defines at L, English always — the
    registry, not the schemes with a cell in the table, so a scheme that has
    not trained at L yet already narrows the set. A benchmark in a language
    only one scheme's list carries measures that list, not the decision."""
    schemes = [s for s, d in DATA_SCHEMES.items() if L in d["langs"]]
    return set.intersection(*[set(_trained_tasks(L, s)) | (trained_bpb_tasks(L, s) or set()) for s in schemes])


TRAINED_NOTE = "tasks in the languages every variant at the L trains on (the intersection of the L's lists, English always)"


def by_L(pool: str, out_dir: Path, stage: str) -> None:
    src = DECISION_ACCURACY / stage / pool / "da_by_L_per_task.csv"
    if not src.is_file():
        return
    t = pd.read_csv(src)
    trained = {L: trained_at(L) for L in t["L"].unique()}
    t = t[t["task"].isin(("train_loss", "bpb_macro")) | np.array([task in trained[L] for task, L in zip(t["task"], t["L"])], dtype=bool)]
    t["measurement"] = _measurement(t["task"])
    t = _with_mean_benchmarks(t)
    sizes = [s for s in SMALL_SIZES if s in set(t["proxy_size"])]        # before the pair filter: the pooled maps keep 1B
    # a cell needs MIN_PAIRS pairs (rq02's rule); the L's this leaves out are named under the title
    has_pairs = set(t.loc[t["da_ref"].notna(), "L"])
    t = t[(t["n_pairs_ref"] >= MIN_PAIRS) | (t["n_pairs_own"] >= MIN_PAIRS)]
    t = t.assign(da_ref=t["da_ref"].where(t["n_pairs_ref"] >= MIN_PAIRS), da_own=t["da_own"].where(t["n_pairs_own"] >= MIN_PAIRS))
    few = sorted(has_pairs - set(t.loc[t["da_ref"].notna(), "L"]))
    few_note = (f"; left out of the DA-size panel for having fewer than {MIN_PAIRS} pairs against {TARGET_SIZE}: "
                f"L{', L'.join(map(str, few))}" if few else "")
    if 2 not in set(t.loc[t["da_ref"].notna(), "L"]):
        print(f"!!! RULE 9: {RULE9_NOTE}")
        few_note += f"; rule 9: {RULE9_NOTE}"
    fracs = sorted(f for f in t["frac"].unique() if f < 1.0)
    # the measurements: DA-size at every proxy's final checkpoint, DA-ckpt within the reference's own run
    ref = G.mark_gated(t[t["frac"] == 1.0].dropna(subset=["da_ref"]), pool, "proxy_size", "da_ref", TARGET_SIZE)
    own = G.mark_gated(t[(t["proxy_size"] == TARGET_SIZE) & (t["frac"] < 1.0)].dropna(subset=["da_own"]), pool, "proxy_size", "da_own")
    rows = G.panel_order(t["measurement"].unique(), first=FIRST_ROWS)
    _two_panels(out_dir, "min_level_by_L", _min_level(ref, "measurement", "proxy_size", sizes, "da_ref"),
                _min_level(own, "measurement", "frac", fracs, "da_own"), fracs, G.chinchilla,
                "How small, and how early, each measurement reads the decision, per language count",
                f"cell = smallest proxy size whose final ranking agrees with the {TARGET_SIZE} final ranking at DA ≥ {SAFE_DA} "
                f"(left), and the earliest checkpoint of the {TARGET_SIZE} run that agrees with its own final ranking (right); "
                f"DA = mean over the measurement's gated tasks (≥ {MIN_PAIRS} pairs; {TRAINED_NOTE}) of the share of design-variant pairs sharing the L ordered alike; "
                "a level counts when it holds at every larger level with a value" + few_note,
                "measurement", rows, sizes, f"DA-size: smallest proxy size at DA ≥ {SAFE_DA}",
                f"DA-ckpt: earliest {TARGET_SIZE} checkpoint at DA ≥ {SAFE_DA}")
    # the SNR definitions: does log10 SNR at a size rank the L's tasks like their DA?
    v = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv", index_col=0).rename_axis("task")
    variants = sorted({c[len("snr_"):].rsplit("_", 1)[0] for c in v.columns if c.startswith("snr_")})
    t1 = t[t["measurement"] != BENCH_MEAN]                 # one row per task again
    own_s = t1[t1["frac"] < 1.0].groupby(["task", "L", "proxy_size"])["da_own"].mean().rename("da_own").reset_index()
    rows_v = []
    for da_kind, d, col in (("size", ref[ref["measurement"] != BENCH_MEAN], "da_ref"), ("ckpt", own_s, "da_own")):
        d = d.assign(kind=_kind(d["task"]))
        for (L, s, kind), g in d.groupby(["L", "proxy_size", "kind"]):
            g = g.set_index("task")[col].dropna()
            for var in variants:
                c = f"snr_{var}_{s}"
                if c not in v.columns:
                    continue
                x = np.log10(v[c].reindex(g.index).where(lambda x: x > 0)).dropna()
                if len(x) < MIN_TASKS:
                    continue
                rho = spearmanr(x, g.loc[x.index]).statistic
                rows_v.append({"da": da_kind, "kind": kind, "variant": var, "L": L, "proxy_size": s, "rho": rho, "n": len(x)})
    rv = pd.DataFrame(rows_v)
    if rv.empty:
        return
    rv.to_csv(out_dir / "snr_variant_rho_by_L.csv", index=False)
    rv = rv[rv["kind"] == KINDS[0]]                # the maps show the benchmark tasks; BPB stays in the CSV

    def _min_size(d):
        wide = d.pivot_table(index=["variant", "L"], columns="proxy_size", values="rho").reindex(columns=sizes)
        return (G.smallest_safe(wide.ge(RHO_MIN).where(wide.notna())).rename("level").rename_axis(["variant", "L"])
                .reset_index().pivot(index="variant", columns="L", values="level"))
    order = rv[rv["da"] == "size"].groupby("variant")["rho"].mean().sort_values(ascending=False).index.tolist()
    order += [x for x in variants if x not in order]
    _two_panels(out_dir, "snr_variant_min_size_by_L", _min_size(rv[rv["da"] == "size"]), _min_size(rv[rv["da"] == "ckpt"]),
                sizes, str, "Smallest size at which an SNR definition tracks decision accuracy, per language count",
                f"cell = smallest proxy size at which the Spearman ρ, over the L's benchmark {TRAINED_NOTE} (≥ {MIN_TASKS}; BPB in the CSV), between log10 SNR at that "
                f"size and the task's DA reaches {RHO_MIN} and stays there at every larger size with a value; left: DA-size "
                f"(proxy final → {TARGET_SIZE} final), right: DA-ckpt (the size's own early checkpoints → its final, mean over "
                f"the nine checkpoints before the last); cells need ≥ {MIN_PAIRS} pairs; definitions ordered by their mean ρ with DA-size" + few_note,
                "SNR definition", order, sizes, f"DA-size: smallest size at ρ ≥ {RHO_MIN}", f"DA-ckpt: smallest size at ρ ≥ {RHO_MIN}")
    pooled_b(pool, out_dir, stage, v, variants, sizes, rows)


def pooled_b(pool: str, out_dir: Path, stage: str, v: pd.DataFrame, variants: list, sizes: list, rows: list) -> None:
    """Version B of the per-L figures: every pair pooled (the headline pool),
    the size axis instead of the language count. Measurements: the earliest
    checkpoint of each proxy size that reads the reference's ranking at
    SAFE_DA, and the DA at 1C itself. SNR definitions: their Spearman rho with
    DA-size and DA-ckpt at each proxy size."""
    e = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_pooled_per_task.csv")
    e["measurement"] = _measurement(e["task"])
    e = _with_mean_benchmarks(G.mark_gated(e, pool, "proxy_size", "da_ref", TARGET_SIZE))
    own_pooled = e[(e["frac"] < 1.0) & (e["measurement"] != BENCH_MEAN)].groupby(["task", "proxy_size"])["da_own"].mean()
    e = e[e["proxy_size"].isin(sizes)].rename(columns={"da_ref": "da"})
    fracs = sorted(e["frac"].unique())
    m = e.groupby(["measurement", "proxy_size", "frac"])["da"].mean().reset_index()
    wide = m.pivot_table(index=["measurement", "proxy_size"], columns="frac", values="da").reindex(columns=fracs)
    lev = (G.smallest_safe(wide.ge(SAFE_DA).where(wide.notna())).rename("level").rename_axis(["measurement", "proxy_size"])
           .reset_index().pivot(index="measurement", columns="proxy_size", values="level"))
    gated_pairs = set(zip(e.loc[e["gated"], "measurement"], e.loc[e["gated"], "proxy_size"]))
    lev = G.with_gated(lev, gated_pairs).reindex(index=rows, columns=sizes)
    at_1c = wide[0.2].unstack("proxy_size").reindex(index=rows, columns=sizes)          # 1C = 20 % of the run
    cnt_1c = (e[e["frac"] == 0.2].groupby(["measurement", "proxy_size"])["da"].count().unstack("proxy_size")
              .reindex(index=rows, columns=sizes))                                     # tasks behind each cell (rule 13)
    gated_1c = pd.DataFrame([[(r, c) in gated_pairs for c in sizes] for r in rows], index=rows, columns=sizes)
    bench_n = e[(e["measurement"] == BENCH_MEAN) & (e["frac"] == 1.0)].groupby("proxy_size")["da"].count().reindex(sizes)
    population = (f"population = every design-variant pair of the pool on the tasks of the languages each cell trains (rule 2), "
                  f"parent tasks only, above chance at the proxy and at {TARGET_SIZE}; {BENCH_MEAN} pools "
                  + ", ".join(f"{int(n) if np.isfinite(n) else 0} tasks at {s}" for s, n in bench_n.items()))
    title = f"How early each proxy size reads the {TARGET_SIZE} ranking, every pair pooled ({pool} pool)"
    note = (f"left: cell = earliest checkpoint (Chinchilla multiples, 5C = the proxy's final) at which the proxy's ranking of the "
            f"design variants agrees with the {TARGET_SIZE} final ranking at DA ≥ {SAFE_DA}, holding at every later checkpoint; "
            f"right: the DA at 1C itself, the small number = tasks behind the cell; DA = mean over the measurement's gated tasks "
            f"of the share of pairs ordered alike; grey = the gate emptied the cell; {population}")
    for kind, fn in (("", G.level_ax), ("_lines", G.level_lines_ax)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.3 * len(rows) + 2.4) if not kind else (14, 4.6),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        t1 = fn(axes[0], lev, f"earliest checkpoint at DA ≥ {SAFE_DA}", levels=fracs, level_label=G.chinchilla,
                xlabel="proxy size" if not kind else "checkpoint of the proxy", ylabel="measurement" if not kind else "proxy size")
        if kind:
            for name, r in at_1c.iterrows():          # the DA at 1C as lines over the sizes
                axes[1].plot(range(len(sizes)), r.to_numpy(dtype=float), marker="o", ms=3.5, lw=1.2, label=str(name))
            axes[1].set_xticks(range(len(sizes))); axes[1].set_xticklabels(sizes); axes[1].axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
            axes[1].set_ylim(0.3, 1.0); axes[1].set_xlabel("proxy size"); axes[1].set_ylabel(f"DA vs {TARGET_SIZE} final at 1C")
            axes[1].set_title("DA at 1C", loc="left", fontsize=8.5); axes[1].grid(color=S.GRID, lw=.6); S.clean(axes[1])
            t2 = at_1c.rename_axis(index="row", columns="col").stack().dropna().rename("value").reset_index().assign(panel="DA at 1C")
        else:
            t2 = G.matrix_ax(axes[1], at_1c, "DA at 1C (20 % of the proxy's run)", cnt=cnt_1c, gated=gated_1c, vmin=0.5, vmax=1.0,
                             xlabel="proxy size")
        G.save_highlights(fig, out_dir, title, note, [t1, t2], name=f"min_level_by_L{kind}_b")
    # SNR definitions against the pooled DA, per proxy size
    rho = []
    kind = _kind(pd.Series(v.index, index=v.index))
    for s_ in sizes:
        da_size = v.get(f"decision_acc_size_{s_}")
        da_ckpt = own_pooled.xs(s_, level="proxy_size").reindex(v.index) if s_ in own_pooled.index.get_level_values(1) else None
        for var in variants:
            c = f"snr_{var}_{s_}"
            if c not in v.columns:
                continue
            x = np.log10(v[c].where(v[c] > 0))
            for da_kind, d in (("size", da_size), ("ckpt", da_ckpt)):
                if d is None:
                    continue
                for kd in KINDS:
                    ok = x.notna() & d.notna() & (kind == kd)
                    if ok.sum() >= MIN_TASKS:
                        rho.append({"da": da_kind, "kind": kd, "variant": var, "proxy_size": s_,
                                    "rho": spearmanr(x[ok], d[ok]).statistic, "n": int(ok.sum())})
    rho = pd.DataFrame(rho)
    if rho.empty:
        return
    rho.to_csv(out_dir / "snr_variant_rho_pooled.csv", index=False)
    order = (rho[(rho["da"] == "size") & (rho["kind"] == KINDS[0])].groupby("variant")["rho"].mean()
             .sort_values(ascending=False).index.tolist())
    order += [x for x in variants if x not in order]
    mats = {(k, kd): rho[(rho["da"] == k) & (rho["kind"] == kd)].pivot(index="variant", columns="proxy_size", values="rho")
            .reindex(index=order, columns=sizes) for k in ("size", "ckpt") for kd in KINDS}
    title = f"Which SNR definition tracks decision accuracy at each proxy size, every pair pooled ({pool} pool)"
    note = (f"cell = Spearman ρ over the gated tasks of the population (≥ {MIN_TASKS}) between log10 SNR at the proxy size and the "
            f"task's DA; DA-size = proxy final → {TARGET_SIZE} final, DA-ckpt = the size's own early checkpoints → its final, mean "
            f"over the nine checkpoints before the last; definitions ordered by their mean ρ with the benchmarks' DA-size; {population}")
    fig, axes = plt.subplots(2, 2, figsize=(11, 0.6 * len(order) + 4))
    tables = [G.matrix_ax(axes[j][i], mats[(k, kd)], f"ρ with DA-{k}, {kd}", vmin=-1, vmax=1, center=0.0, cmap=S.DIV, fmt="{:+.2f}",
                          xlabel="proxy size", ylabel="SNR definition" if i == 0 else "")
              for j, kd in enumerate(KINDS) for i, k in enumerate(("size", "ckpt"))]
    G.save_highlights(fig, out_dir, title, note, tables, name="snr_variant_min_size_by_L_b")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for j, kd in enumerate(KINDS):
        for i, k in enumerate(("size", "ckpt")):
            ax = axes[j][i]
            for n, var in enumerate(order):
                ax.plot(range(len(sizes)), mats[(k, kd)].loc[var].to_numpy(dtype=float), marker="o", ms=3, lw=1.1,
                        color=plt.cm.tab20(n % 20), label=var)
            ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.axhline(0, color=S.MUTED, lw=.8, ls=":")
            ax.axhline(RHO_MIN, color=S.MUTED, lw=.6, ls="--"); ax.set_xlabel("proxy size")
            ax.set_title(f"ρ with DA-{k}, {kd}", loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        axes[j][0].set_ylabel("Spearman ρ")
    axes[0][1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    G.save_highlights(fig, out_dir, title, note + f"; dashed line = {RHO_MIN}", tables, name="snr_variant_min_size_by_L_lines_b")
    flops_version(pool, out_dir, e, v, variants, order, rows, population)


def _flops_ax(ax, curves: dict, *, colours: dict, styles: dict | None = None, ylabel: str, ref: float | None = None,
              label: dict | None = None) -> pd.DataFrame:
    """One line per key of `curves` (a frame with `compute_share` and `y`,
    sorted by compute, every (size, checkpoint) cell a point), x on a log
    scale as a share of the reference run's compute."""
    tables = []
    for key, c in curves.items():
        c = c.sort_values("compute_share")
        ax.plot(c["compute_share"], c["y"], marker="o", ms=2.8, lw=1.1, color=colours[key],
                ls=(styles or {}).get(key, "-"), label=(label or {}).get(key, key))
        tables.append(c.assign(row=str(key)).rename(columns={"compute_share": "col", "y": "value"})[["row", "col", "value"]])
    ax.set_xscale("log"); ax.set_xlabel(f"training compute of the (size, checkpoint) cell, share of the {TARGET_SIZE} run")
    if ref is not None:
        ax.axhline(ref, color=S.MUTED, lw=.8, ls=":")
    ax.set_ylabel(ylabel); ax.grid(color=S.GRID, lw=.6, which="both"); S.clean(ax)
    return pd.concat(tables) if tables else pd.DataFrame(columns=["row", "col", "value"])


def flops_version(pool: str, out_dir: Path, e: pd.DataFrame, v: pd.DataFrame, variants: list, order: list, rows: list,
                  population: str) -> None:
    """Every (proxy size, checkpoint) cell at its training compute, on version
    B's population (`population` names it and the benchmark mean's task count
    per size for the caption)."""
    cells = e.groupby(["measurement", "proxy_size", "frac"]).agg(y=("da", "mean"), compute_share=("compute_share", "first")).reset_index()
    colours = {m: (plt.cm.tab20(i % 20) if m not in FIRST_ROWS else [S.RAMP[3], S.RAMP[1], S.SERIES[1], S.SERIES[2]][FIRST_ROWS.index(m)])
               for i, m in enumerate(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    t1 = _flops_ax(axes[0], {m: cells[cells["measurement"] == m] for m in FIRST_ROWS if m in set(cells["measurement"])},
                   colours=colours, ylabel=f"DA vs {TARGET_SIZE} final", ref=SAFE_DA)
    t2 = _flops_ax(axes[1], {m: cells[cells["measurement"] == m] for m in rows if m not in FIRST_ROWS}, colours=colours,
                   ylabel="", ref=SAFE_DA)
    axes[0].set_title("the aggregates", loc="left", fontsize=8.5); axes[1].set_title("the benchmark families", loc="left", fontsize=8.5)
    for ax in axes:
        ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0) if ax is axes[0] else (1.01, 1.0))
    G.save_highlights(fig, out_dir, f"How much compute reads the {TARGET_SIZE} ranking, every pair pooled ({pool} pool)",
                      f"point = one (proxy size, checkpoint) cell at the compute spent up to that checkpoint; y = mean over the "
                      f"measurement's gated tasks of the share of design-variant pairs the cell orders like the {TARGET_SIZE} final "
                      f"checkpoint; dotted line = {SAFE_DA}; {population}", [t1.assign(panel="aggregates"), t2.assign(panel="families")],
                      name="min_level_by_L_flops")
    # the SNR definitions: rho between log10 SNR at the size and the DA of each (size, checkpoint) cell
    e1 = e[e["measurement"] != BENCH_MEAN].assign(kind=lambda d: _kind(d["task"]))
    rho = []
    for (s_, f, kd), g in e1.groupby(["proxy_size", "frac", "kind"]):
        d = g.set_index("task")["da"].dropna()
        for var in variants:
            c = f"snr_{var}_{s_}"
            if c not in v.columns:
                continue
            x = np.log10(v[c].reindex(d.index).where(lambda x: x > 0)).dropna()
            if len(x) >= MIN_TASKS:
                rho.append({"kind": kd, "variant": var, "proxy_size": s_, "frac": f, "compute_share": g["compute_share"].iloc[0],
                            "y": spearmanr(x, d.loc[x.index]).statistic, "n": len(x)})
    rho = pd.DataFrame(rho)
    if rho.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    tables = []
    for ax, kd in zip(axes, KINDS):
        r = rho[rho["kind"] == kd]
        tables.append(_flops_ax(ax, {var: r[r["variant"] == var] for var in order if var in set(r["variant"])},
                                colours={var: plt.cm.tab20(i % 20) for i, var in enumerate(order)},
                                ylabel="Spearman ρ with the cell's DA" if ax is axes[0] else "", ref=0.0).assign(panel=kd))
        ax.axhline(RHO_MIN, color=S.MUTED, lw=.6, ls="--"); ax.set_title(kd, loc="left", fontsize=8.5)
    axes[1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, out_dir, f"Which SNR definition tracks the decision at every compute, every pair pooled ({pool} pool)",
                      f"point = one (proxy size, checkpoint) cell; y = Spearman ρ over the population's gated tasks (≥ {MIN_TASKS}) "
                      f"between log10 SNR at the size and the task's DA at that checkpoint against the {TARGET_SIZE} final ranking; "
                      f"SNR is a property of the size (its final window), so every size restarts the curve; dashed line = {RHO_MIN}; "
                      f"{population}",
                      tables, name="snr_variant_min_size_by_L_flops")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
