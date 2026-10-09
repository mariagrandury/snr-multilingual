"""rq00 per benchmark and per language: the gate as cell grids, and the score
curves of every language.

    highlights.png / .csv                   rq00 on one page
    gate_margin.csv                         the cells behind the two gate_margin figures
    gate_margin_by_benchmark.png            score minus chance, language x size, one subplot per benchmark
    gate_margin_by_language.png             score minus chance, benchmark x size, one subplot per language
    first_size_above_random.png / .csv      language x benchmark: smallest size from which the score stays above chance
    first_size_above_random_paper.png/.csv  the same for the paper: languages down the side in the scheme-A resource
                                            order, benchmarks across it from the most to the fewest languages above the
                                            gate, the trained languages of the L50 list only (rule 2), no header
    first_size_share_paper.png/.csv    the paper map condensed: per benchmark the share of its languages at each
                                            smallest size (highlights' right panel, in the same order and population)
    first_size_share_paper_horizontal.png/.csv  the same bars standing along the x axis, for a full-width figure
    gate_pass_counts_all_considered.csv/.tex  the "all considered" variant (`considered_panels`): the analysed tasks
    first_size_share_all_considered_paper.png/.csv  beside the screened and discarded ones (600M, 1B and 1.7B, and 3B
                                            once the 3B probe results are in the probe gate)
    gate_screened_verdicts_all_considered.csv  per screened task its verdict per size and `passes_3B` (written once a 3B
                                            verdict exists): the tasks at chance at 1B and 1.7B that clear it at 3B
    score_curves.csv                        the curves below
    score_curves/<language>.png             score vs training tokens (Chinchilla multiples, 5C = the full run),
                                            one line per size, one subplot per benchmark

The margin reads `above_random_scores.csv` (the gate's bucket means: a
benchmark averaged over the cells that trained its language). The curves
average, per size, the cells that train the language, on the shared
checkpoint grid; the dotted line is chance.

    python analysis/rq00_gate_and_curves/panels.py --pool predictivity
    python analysis/rq00_gate_and_curves/panels.py --paper      # the two paper figures alone, from the tables on disk
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

from evals.scripts.utils.configs import bucket_order, load_languages, load_pools  # noqa: E402
from pretrain.ladder_report import _trained_tasks  # noqa: E402
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import ALPHA, MIN_SHARE, md_table, task_chance, wilson_lcb  # noqa: E402
from analysis.utils import FORMATS, LADDER_SIZES, ladder_frame  # noqa: E402

OUT_ROOT = GATE_AND_CURVES
mpl.rcParams.update(S.RC)


def gate_panels(out_dir: Path) -> None:
    scores = pd.read_csv(out_dir / "above_random_scores.csv")
    sizes = [b for b in bucket_order() if b in scores.columns]
    long = scores.melt(id_vars=["task", "family", "language", "random_baseline"], value_vars=sizes,
                       var_name="size", value_name="score").dropna(subset=["score", "random_baseline"])
    long = long[~long["language"].isin(["??", "multi"])]
    long["margin"] = long["score"] - long["random_baseline"]
    mask = pd.read_csv(out_dir / "above_random_mask.csv").melt(id_vars=["task"], value_vars=sizes, var_name="size", value_name="above")
    long = long.merge(mask, on=["task", "size"], how="left")
    # rule 2: a cell no run of the size trains is gated on untrained runs (a transfer reading)
    pop = pd.read_csv(out_dir / "above_random_share.csv").melt(
        id_vars=["task"], value_vars=[f"{s}__population" for s in sizes], var_name="size", value_name="population")
    pop["size"] = pop["size"].str.removesuffix("__population")
    long = long.merge(pop, on=["task", "size"], how="left")
    long = long[long["population"] == "trained"].drop(columns="population")
    gate = (f"the gate keeps a cell when at least {MIN_SHARE:.0%} of the size's runs are confidently above chance "
            f"(one-sided {1 - ALPHA / 2:.0%} Wilson lower bound over the task's items > chance)")
    note = ("cell = mean final-checkpoint score of the size's models that trained the language (cells no model of the size trains are left out, rule 2), minus chance (1 / number of "
            f"options); {gate}")
    kw = dict(value="margin", vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV, fmt="{:+.2f}", note=note,
              cbar="score − chance (neutral colour = chance)")
    G.panel_grid(long, out_dir / "gate_margin_by_benchmark.png", by="family", row="size", row_order=sizes,
                 col="language", ncols=1, cell_w=0.3, counts=False, xlabel="language", ylabel="model size",
                 title="Margin above chance per benchmark", **kw)
    G.panel_grid(long, out_dir / "gate_margin_by_language.png", by="language", row="family", ylabel="benchmark",
                 col="size", col_order=sizes, xlabel="model size", ncols=6, csv=False,
                 title="Margin above chance per language", **kw)
    ok = long.groupby(["family", "language", "size"])["above"].mean().unstack("size").reindex(columns=sizes)
    ok = ok.ge(MIN_SHARE).where(ok.notna())
    level = G.smallest_safe(ok).rename("level").rename_axis(["family", "language"]).reset_index() \
             .pivot(index="family", columns="language", values="level")
    G.level_heatmap(level, out_dir / "first_size_above_random.png", levels=sizes, cbar="smallest size above chance",
                    title="Smallest model size from which the score stays above chance",
                    note=f"cell = smallest size that clears the gate ({gate}), and so does every larger size with a value")
    highlights(out_dir, long, level, sizes, gate)
    threshold_panels(out_dir)
    paper_figures(out_dir)
    if out_dir.name == CANONICAL_POOL:
        considered_panels(out_dir)


def paper_figures(out_dir: Path) -> None:
    """The paper's rq00 figures, read from `first_size_above_random.csv`: the
    level map with languages down the side in the scheme-A resource order
    (English first) and benchmarks across it from the most to the fewest
    languages above the gate at some size, and the same map condensed to one
    bar per benchmark. Both cover the trained languages of the L50 list (rule 2;
    the map on disk is cut to trained cells too) and carry no header."""
    t = pd.read_csv(out_dir / "first_size_above_random.csv")
    sizes = [b for b in bucket_order() if b in pd.read_csv(out_dir / "above_random_mask.csv", nrows=0).columns]
    level = t.pivot(index="family", columns="language", values="level_index")
    iso2 = load_languages()["fineweb_iso2"]
    langs = [l for l in ["en"] + [iso2[s.split("_")[0]] for s in cell_fineweb_subsets(50, "A")] if l in level.columns]
    level = level[langs]
    order = list((level >= 0).sum(axis=1).sort_values(ascending=False, kind="stable").index)
    G.level_heatmap(level.T, out_dir / "first_size_above_random_paper.png", levels=sizes, title="",
                    cbar="Smallest size above chance", xlabel="", ylabel="", rows=langs, cols=order,
                    names=("language", "benchmark"), name=G.paper_name, cell_text=False, cell_w=0.18)
    height = 0.17 * len(order) + 0.9
    fig, ax = plt.subplots(figsize=(5.4, height))
    tab = G.stack_ax(ax, level, "", levels=sizes, rows=order, name=G.paper_name,
                     xlabel="Share of languages above the chance threshold")
    ax.set_ylim(len(order) - 0.5, -0.5)                  # no blank rows above the first bar or below the last
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.3 / height, 1, 1))       # the legend right under the axis label
    tab = tab.drop(columns="panel").rename(columns={"row": "benchmark", "col": "level", "value": "share"})
    tab.to_csv(out_dir / "first_size_share_paper.csv", index=False)
    S.save_paper(fig, out_dir / "first_size_share_paper")
    # the same bars standing along the x axis, for a full-width figure (legend on top)
    height = 3.4
    fig, ax = plt.subplots(figsize=(0.125 * len(order) + 0.9, height))
    G.stack_ax(ax, level, "", levels=sizes, rows=order, name=G.paper_name, upright=True,
               xlabel="Share of languages\nabove the chance threshold")
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / height))   # the legend right above the bars
    tab.to_csv(out_dir / "first_size_share_paper_horizontal.csv", index=False)
    S.save_paper(fig, out_dir / "first_size_share_paper_horizontal")


CONSIDERED = {"auto": "Analysed", "probe": "Screened, not analysed", "discarded": "Discarded", "all": "All considered"}
FORMAT_NAMES = {"original": "original", "rf": "RF", "rfgm": "LLM-RF", "all": "all"}
GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves"


def considered_panels(out_dir: Path) -> None:
    """The gate over every benchmark the study considered (`above_random.considered_gate`):
    the analysed `auto` tasks at every size beside the screened (`probe`) and
    `discarded` tasks, which were evaluated at 600M, 1B and 1.7B alone, and at
    3B once the 3B probe jobs land in the probe gate (`utils.PROBE_SIZES`; the
    sizes are read off the gate, so without a 3B run every output is the
    600M-1.7B one). An `auto` benchmark whose ladder evals have not landed
    (`utils.pending_benchmarks`) counts as screened. A variant next to the
    canonical figures, which stay on `auto`; no pool reads it.

        gate_pass_counts_all_considered.csv / .tex      per group and format: tasks above chance / tasks with a verdict, per size
        first_size_share_all_considered_paper.png/.csv  per benchmark, its tasks by the smallest size from which the gate
                                                        holds: analysed (left) and screened (right, the probe sizes)
        gate_screened_verdicts_all_considered.csv       per screened task its verdicts, `discard_rule` (at chance at
                                                        1B and at 1.7B, the `discarded` definition, unchanged),
                                                        `fails_600M` and `passes_3B`
    """
    from analysis.rq00_gate_and_curves.above_random import considered_gate
    from analysis.utils import ANALYSIS_SIZES, DISCARD_SIZES, PROBE_GATE_DIR, pending_benchmarks
    if not PROBE_GATE_DIR.is_dir():          # the probe gate lives on the cluster scratch only
        print(f"!!! no probe gate at {PROBE_GATE_DIR}: the all-considered variant is not redrawn")
        return
    g = considered_gate(out_dir.name)
    pending = pending_benchmarks()
    sizes = [s for s in bucket_order() if s in set(g["size"])]
    screened = [s for s in sizes if s in set(g.loc[g["group"] != "auto", "size"])]
    # the union is one population only where every group was read: 600M, 1B and 1.7B (the canonical gate stops at 1.7B)
    shared = [s for s in screened if s in set(g.loc[g["group"] == "auto", "size"])]
    above_ref = [s for s in screened if s not in ANALYSIS_SIZES]         # 3B, once its probe results are in
    say = lambda xs: " and ".join([", ".join(xs[:-1]), xs[-1]]) if len(xs) > 1 else "".join(xs)
    combos = []
    for grp in ("auto", "probe", "discarded"):
        fmts = [f for f in FORMATS if f in set(g.loc[g["group"] == grp, "format"])]
        combos += [(grp, f) for f in fmts] + ([(grp, "all")] if len(fmts) > 1 else [])
    rows = []
    for grp, fmt in combos + [("all", "all")]:
        d = g if grp == "all" else g[g["group"] == grp]
        d = d if fmt == "all" else d[d["format"] == fmt]
        for s in sizes:
            c = d[d["size"] == s]
            if grp == "all" and s not in shared:
                c = c.iloc[:0]
            rows.append({"group": grp, "format": fmt, "size": s, "benchmarks": d["family"].nunique(),
                         "tasks": d["task"].nunique(), "languages": d["language"].nunique(),
                         "passing": int(c["above"].sum()) if len(c) else np.nan, "gated": len(c) or np.nan})
    t = pd.DataFrame(rows)
    t["share"] = t["passing"] / t["gated"]
    stem = out_dir / "gate_pass_counts_all_considered"
    t.to_csv(stem.with_suffix(".csv"), index=False)
    wide = t.pivot_table(index=["group", "format", "benchmarks", "tasks"], columns="size", values=["passing", "gated"],
                         sort=False)
    lines = ["\\begin{tabular}{llrr" + "r" * len(sizes) + "}", "\\toprule",
             f" & & & & \\multicolumn{{{len(sizes)}}}{{c}}{{Tasks above chance / tasks with a verdict}} \\\\",
             f"\\cmidrule(lr){{5-{4 + len(sizes)}}}",
             "Group & Format & Bench. & Tasks & " + " & ".join(sizes) + " \\\\", "\\midrule"]
    last = None
    for (grp, fmt, nb, nt), r in wide.iterrows():
        if last is not None and grp != last:
            lines.append("\\midrule")
        num = lambda v: f"{int(v):,}".replace(",", "{,}")
        cells = [f"{num(r[('passing', s)])}/{num(r[('gated', s)])}" if pd.notna(r.get(("gated", s))) else "--" for s in sizes]
        lines.append(" & ".join([CONSIDERED[grp] if grp != last else "", FORMAT_NAMES[fmt], str(nb), num(nt)]
                                + cells) + " \\\\")
        last = grp
    stem.with_suffix(".tex").write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""]))
    # per task, the smallest size from which it stays above chance; analysed and screened side by side
    ok = g.pivot_table(index=["group", "family", "task"], columns="size", values="above").reindex(columns=sizes)
    level = G.smallest_safe(ok.astype(float).where(ok.notna()))
    level.index = ok.index
    late = None
    if screened:        # a screened task's verdict per size: 600M below the discard rule, 3B above it
        v = ok.drop("auto", level="group", errors="ignore")[screened]
        meta = g.drop_duplicates("task").set_index("task")[["language", "format"]]
        v = v.reset_index().join(meta, on="task")
        v["discard_rule"] = v[list(DISCARD_SIZES)].eq(0).all(axis=1)       # NaN at either size is no verdict, not a fail
        v["fails_600M"] = v["600M"].eq(0) if "600M" in v else False
        v["passes_3B"] = v["3B"].eq(1) if "3B" in v else False
        v = v.rename(columns={s: f"above_{s}" for s in screened})
        v.to_csv(out_dir / "gate_screened_verdicts_all_considered.csv", index=False)
        late = v[v["discard_rule"] & v["passes_3B"]]
        print(f"all considered: {int(v['discard_rule'].sum())} screened tasks at chance at {say(list(DISCARD_SIZES))}, "
              f"{int((v['discard_rule'] & v['fails_600M']).sum())} of them at 600M too; {len(late)} pass at 3B "
              f"({late['family'].nunique()} benchmarks)")
    panels = {"Analysed benchmarks, 90M to 1.7B": level.loc[["auto"]],
              f"Screened benchmarks, {say(screened)} only": level.drop("auto", level="group", errors="ignore")}
    per = {k: v.droplevel("group").reset_index("task", drop=True).groupby(level=0).apply(list) for k, v in panels.items()}
    n = max(len(v) for v in per.values())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.125 * n + 1.0))
    tabs = []
    for ax, (title, v) in zip(axes, per.items()):
        mat = pd.DataFrame(v.tolist(), index=v.index)          # one column per task of the benchmark
        key = pd.DataFrame({"share": (mat >= 0).sum(axis=1) / mat.notna().sum(axis=1), "level": mat.where(mat >= 0).mean(axis=1)})
        order = list(key.sort_values(["share", "level"], ascending=[False, True], kind="stable").index)   # most passing, earliest first
        tab = G.stack_ax(ax, mat, title, levels=sizes, rows=order, name=G.paper_name,
                         xlabel="Share of tasks above the chance threshold")
        ax.set_ylim(n - 0.5, -0.5)
        ax.get_legend().remove()
        tabs.append(tab.assign(panel=title))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.35 / (0.125 * n + 1.0), 1, 1))
    pd.concat(tabs).rename(columns={"row": "benchmark", "col": "level", "value": "share"}).to_csv(
        out_dir / "first_size_share_all_considered_paper.csv", index=False)
    S.save_paper(fig, out_dir / "first_size_share_all_considered_paper")
    if out_dir.name != CANONICAL_POOL:
        return
    rel = f"{out_dir.parent.name}/{out_dir.name}"
    cell = t.assign(c=[f"{p:.0f} / {n:.0f}" if n == n else "" for p, n in zip(t["passing"], t["gated"])])
    md = [[CONSIDERED[grp], FORMAT_NAMES[fmt], str(r["benchmarks"].iloc[0]), str(r["tasks"].iloc[0])] + list(r["c"])
          for (grp, fmt), r in cell.groupby(["group", "format"], sort=False)]
    body = "\n\n".join([
        "## Every considered benchmark, the screened ones included",
        f"**Pool `{out_dir.name}` (seed 1904), the same gate; a cell is read on the runs that train the task's language.** "
        f"Analysed = the `auto` group at every size (the canonical tables above), less the benchmarks whose ladder "
        f"evals have not landed (`utils.pending_benchmarks`: {', '.join(sorted(pending)) or 'none'}), which count as "
        f"screened. Screened = the `auto_probe` benchmarks outside the analysed group, and discarded = the `discarded` "
        f"group: both evaluated on the final checkpoint of "
        f"the {say(screened)} cells only (`rq00_task_reformulation/probe.sh`, gate tables in "
        + (f"`utils.PROBE_GATE_DIR`), so their columns below {screened[0]} are empty and the all-considered row is read "
           f"at {say(shared)}. " if not above_ref else
           f"`utils.PROBE_GATE_DIR`; a 3B cell without a probe result has no verdict), so their columns below "
           f"{screened[0]} are empty, the analysed group has no {say(above_ref)} column (the canonical gate stops at the 1.7B reference, "
           f"rule 10) and the all-considered row is read at {say(shared)} alone. `discarded` stays defined at "
           f"{say(list(DISCARD_SIZES))}: {len(late)} screened tasks of {late['family'].nunique()} benchmarks are at "
           f"chance at both and above chance at 3B"
           + (f" ({', '.join(sorted(late['family'].unique()))})" if len(late) else "")
           + "; `gate_screened_verdicts_all_considered.csv` lists every screened task's verdicts (`discard_rule`, "
           "`passes_3B`). ")
        + "Tasks without a chance level have no verdict. No pool reads this variant: the DA and SNR populations "
        "stay on `auto`. Regenerate with `python analysis/rq00_gate_and_curves/panels.py --paper`.",
        md_table(["group", "format", "benchmarks", "tasks"] + sizes, md),
        f"![Every considered benchmark against the gate]({rel}/first_size_share_all_considered_paper.png)",
        f"Files: [`gate_pass_counts_all_considered.csv`]({GITHUB}/{rel}/gate_pass_counts_all_considered.csv) / "
        f"[`.tex`]({GITHUB}/{rel}/gate_pass_counts_all_considered.tex), "
        f"[`first_size_share_all_considered_paper.png`]({GITHUB}/{rel}/first_size_share_all_considered_paper.png) / "
        f"[`.csv`]({GITHUB}/{rel}/first_size_share_all_considered_paper.csv)"
        + (f", [`gate_screened_verdicts_all_considered.csv`]({GITHUB}/{rel}/gate_screened_verdicts_all_considered.csv)."
           if screened else ".")])
    replace_block(OUT_ROOT / "README.md", "considered", body, f"panels.py --pool {out_dir.name}")


def threshold_panels(out_dir: Path) -> None:
    """above_random_thresholds.png: what the Wilson gate asks of each benchmark
    and how many (task, size) cells it keeps, against fixed margins."""
    runs = pd.read_csv(out_dir / "above_random_runs.csv")
    runs = runs[runs["trained"]] if "trained" in runs else runs
    runs = runs[runs["n_items"].notna()]   # no item count, no Wilson bound: nothing to draw (every registered task has one; a new one until derive_task_options runs)
    scores = pd.read_csv(out_dir / "above_random_scores.csv")
    sizes = [b for b in bucket_order() if b in scores.columns]
    scores = scores[scores["language"].isin(["??", "multi"]) == False]
    runs = runs.merge(scores[["task", "family", "random_baseline"]], on="task")
    # the smallest accuracy whose lower bound clears chance, per task: the margin the test implies
    tasks = runs.drop_duplicates("task").set_index("task")
    thr = {}
    for t, r in tasks.iterrows():
        n = int(r["n_items"]); ks = np.arange(0, n + 1)
        ok = wilson_lcb(ks / n, np.full(n + 1, n)) > r["random_baseline"]
        thr[t] = ks[ok][0] / n - r["random_baseline"] if ok.any() else np.nan
    tasks["threshold"] = pd.Series(thr)
    fam = G.panel_order(tasks["family"].unique())
    # cells above random under each rule
    cell = runs.groupby(["task", "bucket"]).agg(share=("above", "mean"), lcb=("lcb", "median"), n=("score", "size")).reset_index()
    cell = cell.merge(scores.melt(id_vars=["task", "random_baseline", "family"], value_vars=sizes, var_name="bucket", value_name="score"),
                      on=["task", "bucket"])
    cell["margin"] = cell["score"] - cell["random_baseline"]
    cell["share_002"] = runs.assign(a=(runs["lcb"] > runs["random_baseline"] + 0.02).astype(float)).groupby(["task", "bucket"])["a"].mean() \
                            .reindex(pd.MultiIndex.from_frame(cell[["task", "bucket"]])).to_numpy()
    rules = {"Wilson LCB > chance": cell["share"] >= MIN_SHARE, "Wilson LCB > chance + 0.02": cell["share_002"] >= MIN_SHARE,
             "mean > chance + 0.01": cell["margin"] > 0.01, "mean > chance + 0.03": cell["margin"] > 0.03,
             "mean > chance + 0.05": cell["margin"] > 0.05}
    counts = pd.DataFrame({k: v.groupby(cell["family"]).sum() for k, v in rules.items()}).reindex(fam)
    total = cell.groupby("family").size().reindex(fam)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    med = tasks.groupby("family")["threshold"].median().reindex(fam)
    ni = tasks.groupby("family")["n_items"].median().reindex(fam)
    tables = [G.rank_ax(axes[0], med.rename(index=lambda f: "%s  (n = %s)" % (f, format(int(ni[f]), ","))),
                        "Margin over chance the Wilson gate implies per benchmark (median task)", k=len(fam),
                        xlabel="smallest (accuracy − chance) with LCB > chance", fmt="{:+.3f}")]
    tables.append(G.matrix_ax(axes[1], counts.div(total, axis=0), "Share of a benchmark's (task, size) cells above random, per rule",
                              cnt=counts, xlabel="rule", ylabel="benchmark"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(20); lab.set_ha("right")
    G.save_highlights(fig, out_dir, "What the Wilson gate asks, and what each rule keeps",
                      f"left: per task the smallest accuracy whose Wilson {1 - ALPHA:.0%} lower bound over its n items clears chance, "
                      "minus chance (median over the benchmark's tasks; n = median items); right: (task, size) cells above random "
                      f"under the gate (≥ {MIN_SHARE:.0%} of the size's runs), the gate with a +0.02 floor, and fixed margins on the "
                      "mean score; small number = cells; runs of the size that trained the language", tables,
                      name="above_random_thresholds")


def highlights(out_dir: Path, long: pd.DataFrame, level: pd.DataFrame, sizes: list, gate: str) -> None:
    """rq00 on one page: which benchmarks clear chance, at which size, and in how many languages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    fam = G.panel_order(long["family"].unique())
    share = long.groupby(["family", "size"])["above"].mean().unstack().reindex(index=fam, columns=sizes)
    n = long.groupby(["family", "size"])["task"].nunique().unstack().reindex(index=fam, columns=sizes)
    margin = long.groupby(["family", "size"])["margin"].median().unstack().reindex(index=fam, columns=sizes)
    tables = [G.matrix_ax(axes[0], share, "Share of a benchmark's languages above the gate", cnt=n, xlabel="model size"),
              G.matrix_ax(axes[1], margin, "Median margin above chance", vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV,
                          fmt="{:+.2f}", xlabel="model size"),
              G.stack_ax(axes[2], level, "Smallest size that stays above chance, share of languages", levels=sizes)]
    G.save_highlights(fig, out_dir, "rq00 in one figure: which benchmarks carry information, and from which size?",
                      f"{gate}; small number = language tasks behind the cell; "
                      "right: per benchmark, how its languages split by the smallest size from which the gate holds", tables)


def score_curves(pool: str, out_dir: Path) -> int:
    df = ladder_frame(pool)
    df = df[df["kind"] == "benchmark"]
    df = G.add_meta(df[[t in _trained_tasks(L, d) for t, L, d in zip(df["task"], df["L"], df["data"])]])
    df["chance"] = df["task"].map(task_chance)
    df["chinchilla"] = df["frac"] * G.CHINCHILLA_AT_FULL
    curve_dir = out_dir / "score_curves"
    curves = df.groupby(["language", "family", "size", "chinchilla"]).agg(
        score=("primary_score", "mean"), chance=("chance", "mean"), tasks=("task", "nunique"), models=("model", "nunique")).reset_index()
    curves.to_csv(out_dir / "score_curves.csv", index=False)
    n = 0
    for lang, g in curves.groupby("language"):
        fams = G.panel_order(g["family"].unique())
        cols = min(5, len(fams)); rows = (len(fams) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows + 0.5), squeeze=False)
        flat = [a for r in axes for a in r]
        for ax in flat[len(fams):]:
            ax.axis("off")
        for ax, fam in zip(flat, fams):
            gf = g[g["family"] == fam]
            for size in [s for s in LADDER_SIZES if s in set(gf["size"])]:
                m = gf[gf["size"] == size].sort_values("chinchilla")
                ax.plot(m["chinchilla"], m["score"], color=S.SIZE_COLOR.get(size, S.MUTED), lw=1.4, marker="o", ms=2.5,
                        label=size)
            if gf["chance"].notna().any():
                ax.axhline(gf["chance"].mean(), color=S.ALERT, lw=.9, ls=":")
            nt = int(gf["tasks"].max())
            ax.set_title(f"{fam} ({nt} task{'s' if nt > 1 else ''})", loc="left", fontsize=8)
            ax.set_xlabel("training tokens (× Chinchilla)"); ax.set_ylabel("score"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        handles = [plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in LADDER_SIZES if s in set(g["size"])]
        fig.legend(handles=handles, ncol=len(handles), loc="lower center", frameon=False)
        fig.suptitle(f"{lang}: benchmark score along the run, mean over the cells that train it "
                     "(colour = size, dotted = chance; 5C = the full run)", x=0.01, ha="left", y=1.0, fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        curve_dir.mkdir(parents=True, exist_ok=True)
        S.save(fig, curve_dir / f"{lang}.png", dpi=110)
        n += 1
    print(f"Wrote {n} per-language score-curve figures → {curve_dir}")
    return n


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    gate_panels(out_dir)
    n = score_curves(pool, out_dir)
    if pool != CANONICAL_POOL:
        return
    rel = f"{stage}/{pool}"
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The gate and the curves without the aggregation (`{pool}` pool). Regenerate with "
        f"`python analysis/rq00_gate_and_curves/panels.py --pool {pool}`. In every grid white is \"no value\" and grey "
        "\"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq00 in one figure]({rel}/highlights.png)",
        f"![Smallest size above chance]({rel}/first_size_above_random.png)",
        f"![Margin above chance per benchmark]({rel}/gate_margin_by_benchmark.png)",
        f"![Margin above chance per language]({rel}/gate_margin_by_language.png)",
        f"![What the gate asks, and what each rule keeps]({rel}/above_random_thresholds.png)",
        f"Score along the run, one figure per language ({n} languages, one subplot per benchmark, one line per "
        f"size): `{rel}/score_curves/<language>.png`, e.g.",
        f"![Score curves, German]({rel}/score_curves/de.png)"])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--paper", action="store_true", help="only the paper figures, from the tables on disk")
    args = p.parse_args()
    if args.paper:
        out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
        paper_figures(out)
        if args.pool == CANONICAL_POOL:
            considered_panels(out)
    else:
        main(args.pool)
