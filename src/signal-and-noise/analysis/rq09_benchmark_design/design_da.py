"""Benchmark design against DA-size: do the characteristics rq09 reads on SNR
also order the benchmarks by how well a small proxy decides like the reference?

The SNR analysis (`analyze.py`) groups the benchmark families by the
characteristics of `FAMILY_META` (curation, source origin, task format,
option count, reading passage, domain) and the length features (`length_features.csv`).
This reads the same characteristics against DA-size instead of SNR:

  population  rq02's per-task DA-size (`reliable_tasks.long_da`: the proxy's final
              checkpoint against the 1.7B final, gated at the proxy and at 1.7B, rule 1,
              >= MIN_PAIRS pairs, rule 5) on the mono-axis pairs, over the tasks of the
              paper's rq2 figure (`above_66_either`: median DA-size or median DA-ckpt
              >= 0.66, on the mono-axis reliability, rule 15) and, beside it (CLAUDE.md
              #17), over every gated task (`unfiltered`); per-language parent tasks of the
              families with a `FAMILY_META` entry, as the SNR analysis keeps them.
  benchmark   per proxy size, the median DA-size over a family's tasks (rq09 takes the
              median over a family's tasks for the SNR); the benchmark's DA-size is the
              mean of those medians over the proxies 90M-1B that have one.
  statistic   per characteristic, Spearman rho over the benchmarks against the
              characteristic's centred code (`encode`: an ordinal characteristic ranked, a
              binary one -1/+1; a length as is). The characteristics with more than two
              unordered levels are binarised (the user's call, 2026-10-08): curation as
              translated (human, MT, MT + post-edit, method unknown) vs native (written in the
              language, English originals included, or generated from treebanks or planning
              problems), source as translated from English vs not, task format as continuation
              (the options scored as continuations: completion, minimal pair, classification
              labels, cloze, LLM-RF statement, LAMBADA's last word) vs lettered MCQ (question
              or passage with lettered options), and the domain, nominal with eleven levels, as
              its most populous level against the rest (knowledge vs every other domain; the
              level is chosen by its count, not by its DA). A characteristic FAMILY_META leaves None
              (no source settles it) drops that benchmark from that row only. 95 % percentile bootstrap over
              the benchmarks; the Kruskal-Wallis H and p of every two-group split beside.
              The same statistics on rq09's SNR (median `snr_mpd_1.7B`) sit beside.
  per task    the same statistics with each (benchmark, language) task as one observation
              (`task:<population>` readings): its DA-size is the mean over the proxies 90M-1B
              that have one, and the bootstrap resamples whole benchmarks (a benchmark's tasks
              share its characteristics, so they are not independent observations).

    benchmark_characteristics.csv / .tex        one row per benchmark of either reading, one column per
                                                characteristic (the LaTeX table the paper copies)
    design_da_size_per_family_mono_axis.csv     per (population, family, proxy) the median DA-size and its tasks
    design_da_size_correlation_above_66_either_mono_axis.png/.csv (+ _paper)
                                                left: one value per characteristic on the mean DA-size over
                                                90M-1B with its bootstrap CI, against the unfiltered reading
                                                and the SNR; right: one line per proxy size
    design_da_size_by_level_above_66_either_mono_axis.png/.csv (+ _paper)
                                                a row per characteristic level, one dot per benchmark at its
                                                DA-size - 0.5 (0 = chance agreement)
    design_da_size_quadrant_mono_axis.png/.csv (+ _paper)
                                                one panel per characteristic: x = DA-size - 0.5 on every gated
                                                task, y = the centred code, the benchmarks counted per quadrant
    design_da_size_quadrant_above_66_either_mono_axis.png/.csv (+ _paper)
                                                the same on the above_66_either tasks (most benchmarks land
                                                right of chance there)
    design_da_size_per_task_mono_axis.csv       per (population, task, proxy) the DA-size the figures below read
    design_da_size_by_level_per_task_above_66_either_mono_axis.png/.csv (+ _paper)
                                                the by-level figure with one dot per task (its mean over the
                                                proxies), the level read as the median and IQR over its tasks
    design_da_size_quadrant_per_task_mono_axis.png/.csv (+ _paper)
                                                the quadrant figure with one dot per (task, proxy), coloured by
                                                proxy size, on every gated task
    design_da_size_quadrant_per_task_above_66_either_mono_axis.png/.csv (+ _paper)
                                                the same on the above_66_either tasks
    item_count_per_task_da_all_mono_axis.csv   per task: item count, DA-size, DA-ckpt, log10 SNR and R² (`task_table`)
    item_count_da_all_mono_axis.csv / .tex     per metric (DA-size, DA-ckpt, SNR, R²): the family-level rho of item count
                                                and of skill, the slope on log10 items within benchmarks (family fixed
                                                effect, clustered by family) and a family-level fit on both
    wave_robustness_da_size_mono_axis.csv / .tex
                                                the original benchmarks (auto on 2026-09-07) against each later wave
                                                (`analyze.WAVES`): family medians with a bootstrap CI over families,
                                                a Mann-Whitney test and the later-wave coefficient after log10 items
    Item count is the median `n_items` (configs/tasks.json) over a benchmark's per-language tasks, a continuous
    characteristic; skill is the group of its domain (`analyze.DOMAINS`), LM-like (commonsense, grammaticality,
    language modelling) against the rest, None for a suite whose subtasks fall in both.
    The quadrant figures colour a dot by its proxy size (the per-benchmark one draws one dot per
    benchmark and proxy, at the median over its tasks). The by-level figures group the levels under
    a header row per characteristic.

    python analysis/rq09_benchmark_design/design_da.py --pool predictivity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable, long_da  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import task_n_items  # noqa: E402
from analysis.rq09_benchmark_design.analyze import (  # noqa: E402
    DOMAINS, FAMILY_META, KNOWLEDGE_ETC, LM_LIKE, ORIGINAL, SUBTASK_DOMAIN, WAVE, WAVES, family_skill, untagged)
from analysis.utils import SMALL_SIZES, _is_language_aggregate, auto_tasks, benchmark_family, size_order  # noqa: E402

HERE = Path(__file__).resolve().parent
AXES = "mono-axis"
FILTER = "above_66_either"
N_BOOT = 1000
CHANCE = 0.5
# characteristic -> (label, kind, the FAMILY_META / length column, the level order). Ordinal and
# binary levels are listed low to high: the code is the level's rank, centred at 0.
CHARACTERISTICS = {
    "items": ("Item count", "continuous", None),
    "skill": ("Skill", "binary", [KNOWLEDGE_ETC, LM_LIKE]),
    "curation": ("Curation", "binary", ["translated", "native"]),
    "source": ("Source", "binary", ["english_translated", "original"]),
    "format_type": ("Task format", "binary", ["continuation", "lettered MCQ"]),
    "n_options": ("Answer options", "ordinal", [2, 3, 4, 5]),
    "passage": ("Reading passage", "binary", [False, True]),
    "domain_group": ("Domain", "binary", ["other", "knowledge"]),
    "context_len_chars_median": ("Context length", "continuous", None),
    "option_len_chars_median": ("Option length", "continuous", None),
}
# the binarised characteristics: FAMILY_META level -> binary level
NATIVE_CURATION = {"originally_multilingual", "template_generated", "english_original"}
TRANSLATED_CURATION = {"human_translation", "mt_post_edited", "machine_translation", "translation_unknown"}
LETTERED_FORMATS = {"mcq_question_only", "mrc_passage"}
LEVEL_NAMES = {"translated": "translated", "native": "native", "continuation": "continuation",
               "original": "not translated", "english_original": "English original",
               "translation_unknown": "translated, method unknown", "cloze_completion": "cloze",
               "last_word": "last word", 5: "5 options",
               "lettered MCQ": "lettered MCQ","originally_multilingual": "originally multilingual", "human_translation": "human translation",
               "template_generated": "template generated", "mt_post_edited": "MT post-edited",
               "machine_translation": "machine translation", "english_translated": "translated from English",
               "minimal_pair": "minimal pair", "completion": "completion", "classification": "classification",
               "mcq_question_only": "question with options", "mrc_passage": "passage with options",
               "statement_continuation": "statement (LLM-RF)",
               2: "2 options", 3: "3 options", 4: "4 options", False: "no passage", True: "passage",
               "knowledge": "knowledge", "other": "other domains", LM_LIKE: LM_LIKE, KNOWLEDGE_ETC: KNOWLEDGE_ETC}
LENGTH_CSV = HERE / "length_features.csv"
mpl.rcParams.update(S.RC)


def encode(col: str, v):
    """The centred numeric code of a level: its rank among the levels present, centred at 0
    (an ordinal of three levels reads -1, 0, 1, a binary one -1, +1)."""
    levels = CHARACTERISTICS[col][2]
    return 2 * levels.index(v) / (len(levels) - 1) - 1


def features() -> pd.DataFrame:
    meta = pd.DataFrame.from_dict(FAMILY_META, orient="index")
    meta.index.name = "family"
    cat = meta["curation_category"]
    # an unknown method (BBH) is still untranslated when the benchmark is an English original
    meta["curation"] = np.select([cat.isin(NATIVE_CURATION), cat.isin(TRANSLATED_CURATION),
                                  meta["source_origin"] == "english_original"], ["native", "translated", "native"], None)
    # a suite's mixed characteristic (None) stays unknown in its binary code too
    meta["source"] = np.where(meta["source_origin"] == "english_translated", "english_translated", "original")
    meta["format_type"] = np.where(meta["format"].isin(LETTERED_FORMATS), "lettered MCQ", "continuation")
    meta["domain_group"] = np.where(meta["domain"] == "knowledge", "knowledge", "other")
    for col, src in (("source", "source_origin"), ("format_type", "format"), ("domain_group", "domain")):
        meta[col] = meta[col].where(meta[src].notna(), None)
    meta["skill"] = [family_skill(f) for f in meta.index]
    meta["n_items"] = item_counts()
    meta["items"] = np.log10(meta["n_items"])
    lf = pd.read_csv(LENGTH_CSV).set_index("family")
    return meta.join(lf[["context_len_chars_median", "option_len_chars_median"]])


def item_counts() -> pd.Series:
    """A benchmark's item count: the median `n_items` (configs/tasks.json) over its per-language tasks in the
    trained set."""
    t = pd.DataFrame({"task": sorted(auto_tasks())})
    t["family"] = t["task"].map(benchmark_family)
    t = t[[f in FAMILY_META and _is_language_aggregate(a, f) for a, f in zip(t["task"], t["family"])]]
    return t.assign(n=t["task"].map(task_n_items)).groupby("family")["n"].median()


def per_task(out_dir: Path, pool: str) -> tuple[pd.DataFrame, pd.Series]:
    """(population, family, task, proxy) -> the task's DA-size at that proxy, and per task its DA-ckpt: the
    median over its DA-ckpt cells (every early checkpoint of every size, gated at that size), the reduction
    rq02's reliability filter reads."""
    l = long_da(out_dir, pool, axes=AXES)
    untagged(l[(l["kind"] == "size") & l["size"].isin(SMALL_SIZES)], "da")
    l = l[[f in FAMILY_META and _is_language_aggregate(t, f) for t, f in zip(l["task"], l["family"])]]
    ckpt = l[l["kind"] == "ckpt"].groupby("task")["da"].median().rename("da_ckpt")
    l = l[(l["kind"] == "size") & l["size"].isin(SMALL_SIZES)]
    rel = load_reliable(out_dir, FILTER, axes=AXES)
    parts = [l.assign(population="unfiltered"), l[l["task"].isin(set(rel["task"]))].assign(population=FILTER)]
    return pd.concat(parts)[["population", "family", "task", "language", "size", "da", "n_pairs"]], ckpt


def per_family(tasks: pd.DataFrame) -> pd.DataFrame:
    """(population, family, proxy) -> median DA-size over the family's tasks and their count."""
    return (tasks.groupby(["population", "family", "size"])
            .agg(da_size=("da", "median"), n_tasks=("task", "nunique"), n_pairs_min=("n_pairs", "min")).reset_index())


def task_means(tasks: pd.DataFrame) -> pd.DataFrame:
    """(population, family, task) -> the task's DA-size, the mean over the proxies that have one."""
    return (tasks.groupby(["population", "family", "task", "language"])
            .agg(da_size=("da", "mean"), n_proxies=("size", "nunique")).reset_index())


def task_table(tm: pd.DataFrame, ckpt: pd.Series, out_dir: Path, stage: str) -> pd.DataFrame:
    """One row per task of a tagged family with any value: its item count (configs/tasks.json), DA-size (the
    mean over the proxies, every gated task), DA-ckpt (`per_task`), log10 SNR at 1.7B (`per_task_snr.csv`,
    the tasks of the SNR ranking) and the R² of rq01's fit of score against size (`scaling_regimes.csv`, pool
    predictivity_seeds, the fit over the sizes above chance)."""
    snr = pd.read_csv(out_dir / "per_task_snr.csv")
    col = next(c for c in snr.columns if c.startswith("snr_"))
    r2 = pd.read_csv(SCALING_PREDICTABILITY / stage / "predictivity_seeds" / "scaling_regimes.csv")
    t = (tm[tm["population"] == "unfiltered"].set_index("task")["da_size"].to_frame()
         .join([ckpt, np.log10(snr.set_index("task")[col].where(lambda v: v > 0)).rename("log_snr"),
                r2.set_index("task")["r2_size"].rename("r2")], how="outer"))
    t = t.rename_axis("task").reset_index()
    t["family"] = t["task"].map(benchmark_family)
    t = t[[f in FAMILY_META and _is_language_aggregate(a, f) for a, f in zip(t["task"], t["family"])]]
    t["n_items"] = t["task"].map(task_n_items)
    return t.assign(log_items=np.log10(t["n_items"]))


def benchmark_values(fam: pd.DataFrame, snr: pd.DataFrame, tm: pd.DataFrame, tt: pd.DataFrame) -> dict[str, pd.Series]:
    """Each reading's value per benchmark: mean DA-size over the proxies (both populations), each
    proxy's DA-size (`size:<s>`, filtered population), the SNR median and the median DA-ckpt and R² over
    its tasks (`task_table`); then per task (`task:<pop>`, indexed by the task's benchmark, so the index
    repeats) the task's mean DA-size over the proxies."""
    out = {}
    for pop in (FILTER, "unfiltered"):
        out[pop] = fam[fam["population"] == pop].groupby("family")["da_size"].mean()
    f = fam[fam["population"] == FILTER]
    for s in size_order(f["size"].unique()):
        out[f"size:{s}"] = f[f["size"] == s].set_index("family")["da_size"]
    out["snr"] = snr.set_index("family")["snr_median"]
    for m in ("da_ckpt", "r2"):                      # the family median over its tasks
        out[m] = tt.dropna(subset=[m]).groupby("family")[m].median()
    for pop in (FILTER, "unfiltered"):
        out[f"task:{pop}"] = tm[tm["population"] == pop].set_index("family")["da_size"]
    return out


def statistic(col: str, values: pd.Series, feat: pd.DataFrame) -> float:
    """Spearman rho against the centred code (ordinal, binary) or the length, over the benchmarks of
    `values` (its index may repeat: a bootstrap draw); NaN for a binary characteristic one of whose
    levels holds fewer than two distinct benchmarks."""
    x = feat.loc[values.index, col]
    ok = x.notna().to_numpy() & values.notna().to_numpy()
    x, y = x[ok], values[ok].to_numpy(float)
    kind = CHARACTERISTICS[col][1]
    if kind == "binary" and x[~x.index.duplicated()].value_counts().min() < 2:
        return np.nan                 # a level held by one benchmark: its rank is that benchmark's, not the level's
    xs = x.map(lambda v: encode(col, v)).to_numpy(float) if kind != "continuous" else x.to_numpy(float)
    if len(y) < 3 or len(np.unique(xs)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(stats.spearmanr(xs, y).statistic)


def correlations(values: dict[str, pd.Series], feat: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for reading, v in values.items():
        v = v.dropna()
        fams = v.index.unique()           # resample whole benchmarks: a task reading carries all of a benchmark's tasks
        draws = [v.loc[fams[rng.integers(0, len(fams), len(fams))]] for _ in range(N_BOOT)] \
            if not reading.startswith("size:") else []
        for col, (label, kind, _) in CHARACTERISTICS.items():
            ok = feat.loc[v.index, col].notna()
            row = {"reading": reading, "characteristic": col, "label": label, "kind": kind,
                   "statistic": "spearman_rho", "value": statistic(col, v, feat),
                   "n_benchmarks": int(feat.loc[fams, col].notna().sum()), "n_tasks": int(ok.sum()),
                   "ci_low": np.nan, "ci_high": np.nan, "kruskal_H": np.nan, "kruskal_p": np.nan}
            if draws:
                b = np.array([statistic(col, d, feat) for d in draws])
                if np.isfinite(b).sum() >= N_BOOT // 2:
                    row["ci_low"], row["ci_high"] = np.nanpercentile(b, [2.5, 97.5])
            if kind != "continuous":
                x = feat.loc[v.index, col]
                groups = [v[(x == lv).to_numpy()].to_numpy(float) for lv in pd.unique(x.dropna())]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) >= 2:
                    row["kruskal_H"], row["kruskal_p"] = stats.kruskal(*groups)
            rows.append(row)
    return pd.DataFrame(rows)


def _row_label(r) -> str:
    return r.label


def fig_correlation(c: pd.DataFrame, path: Path, paper: bool) -> None:
    order = list(CHARACTERISTICS)
    y = {k: i for i, k in enumerate(order)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    series = [(FILTER, "DA-size, tasks reliable at 0.66", S.SERIES[0], "o", "full"),
              ("unfiltered", "DA-size, every task above chance", S.SERIES[0], "o", "none"),
              ("snr", "SNR at 1.7B", S.SERIES[1], "D", "full")]
    for k, (reading, label, colour, marker, fill) in enumerate(series):
        g = c[c["reading"] == reading].set_index("characteristic").reindex(order)
        yy = np.array([y[o] for o in order]) + (k - 1) * 0.22
        a.errorbar(g["value"], yy, xerr=[(g["value"] - g["ci_low"]).clip(lower=0), (g["ci_high"] - g["value"]).clip(lower=0)], fmt=marker,
                   color=colour, mfc=colour if fill == "full" else "white", ms=4, lw=.8, capsize=1.5, label=label)
    sizes = [r.split(":")[1] for r in dict.fromkeys(c["reading"]) if r.startswith("size:")]
    for s in sizes:
        g = c[c["reading"] == f"size:{s}"].set_index("characteristic").reindex(order)
        b.plot(g["value"], [y[o] for o in order], "-o", color=S.SIZE_COLOR[s], ms=3.5, lw=1, label=s)
    for ax in (a, b):
        ax.axvline(0, color=S.MUTED, lw=.6, ls=":"); ax.set_xlim(-1.05, 1.05)
        ax.grid(axis="x", color=S.GRID, lw=.5); S.clean(ax)
    lab = c.drop_duplicates("characteristic").set_index("characteristic").reindex(order)
    a.set_yticks(range(len(order))); a.set_yticklabels([_row_label(r) for r in lab.itertuples()])
    a.invert_yaxis()
    a.set_xlabel("Rank correlation, mean over 90M to 1B")
    b.set_xlabel("Rank correlation with DA-size per proxy")
    a.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    b.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, title="Proxy size",
             title_fontsize=7)
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    top = G._header(fig, "Benchmark characteristics against DA-size (and SNR)",
                    f"rank correlation over the benchmark families: Spearman ρ against the centred code of an ordinal or binary "
                    f"characteristic (options 2 → 5, no passage → passage, translated → native, translated from English → "
                    f"not, continuation → lettered MCQ) or a length; DA-size = rq02's mono-axis DA-size against 1.7B, median over a "
                    f"family's tasks per proxy; left: its mean over 90M–1B with a 95 % bootstrap CI over the families, on the "
                    f"{FILTER} tasks (filled), on every gated task (hollow) and rq09's SNR at 1.7B (diamond); right: one line "
                    f"per proxy ({FILTER})")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def level_table(units: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """One row per (population, characteristic, unit): its level, centred code and DA-size - 0.5.
    `units` has a row per benchmark, task or (task, proxy), with its `population`, `family` and
    `da_size`; a level no source settles (FAMILY_META None) drops that unit from that row."""
    rows = []
    for pop in (FILTER, "unfiltered"):
        u = units[(units["population"] == pop) & units["da_size"].notna()]
        for col, (label, kind, levels) in CHARACTERISTICS.items():
            if kind == "continuous":
                continue
            lv = u["family"].map(feat[col])
            g = u[lv.notna()]
            lv = lv[lv.notna()]
            rows.append(pd.DataFrame({
                "population": pop, "characteristic": col, "label": label, "kind": kind,
                "level": lv.map(LEVEL_NAMES), "level_rank": lv.map(levels.index), "code": lv.map(lambda v: encode(col, v)),
                "family": g["family"], "name": g["family"].map(G.paper_name),
                **{c: g[c] for c in ("task", "language", "size") if c in g},
                "da_size": g["da_size"], "x": g["da_size"] - CHANCE}))
    return pd.concat(rows, ignore_index=True)


def example(t: pd.DataFrame) -> str:
    """The highlighted benchmark: the highest DA-size of the filtered reading."""
    f = t[t["population"] == FILTER]
    return f.loc[f["da_size"].idxmax(), "family"]


def _group_labels(ax, rows: list, counts: dict, colours: dict) -> None:
    """The y labels of a grouped level figure, left-aligned: a bold header row per characteristic
    in its colour, then its levels indented under it with their counts."""
    labels = [CHARACTERISTICS[c][0] if lv is None else f"    {lv[0].upper() + lv[1:]} ({counts[(c, lv)]})"
              for c, lv in rows]
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    for lab, (c, lv) in zip(ax.get_yticklabels(), rows):
        lab.set_color(colours[c] if lv is None else S.INK)
        lab.set_fontweight("bold" if lv is None else "normal")
    fig = ax.figure
    fig.canvas.draw()
    width = max(l.get_window_extent().width for l in ax.get_yticklabels()) * 72 / fig.dpi
    ax.tick_params(axis="y", pad=width + 2)
    for lab in ax.get_yticklabels():
        lab.set_horizontalalignment("left")


def fig_levels(t: pd.DataFrame, path: Path, paper: bool) -> None:
    """A header row per characteristic, then a row per level, coloured by characteristic; x = DA-size
    - 0.5 (the filtered population). Per benchmark (no `task` column): a dot per benchmark and the
    level's median as a bar, the highest benchmark named once per characteristic. Per task: a dot per
    task and the level's median (bar) and interquartile range (line) over its tasks."""
    per_task = "task" in t
    top_fam = example(t)
    t = t[t["population"] == FILTER]
    t = t.sort_values(["characteristic", "level_rank"], key=lambda s: s.map(list(CHARACTERISTICS).index)
                      if s.name == "characteristic" else s)
    rows = []
    for c, lv in dict.fromkeys(zip(t["characteristic"], t["level"])):
        rows += [(c, None)] * (not rows or rows[-1][0] != c) + [(c, lv)]
    pos = {r: i for i, r in enumerate(rows)}
    counts = t.groupby(["characteristic", "level"]).size().to_dict()
    colours = dict(zip(dict.fromkeys(t["characteristic"]), plt.cm.tab10.colors))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.6, 0.17 * len(rows) + 0.9))
    for (col, lv), g in t.groupby(["characteristic", "level"], sort=False):
        y = pos[(col, lv)] + rng.uniform(-0.25, 0.25, len(g))
        if per_task:
            ax.scatter(g["x"], y, s=5, color=colours[col], alpha=.55, lw=0, zorder=2)
            q1, q3 = g["x"].quantile([.25, .75])
            ax.plot([q1, q3], [pos[(col, lv)]] * 2, color=S.INK, lw=1.6, solid_capstyle="butt", zorder=3)
        else:
            top = (g["family"] == top_fam).to_numpy()
            ax.scatter(g["x"], y, s=12, color=colours[col], lw=.3, edgecolor="white", zorder=2)
            # the highest benchmark is the rightmost dot of its row, so its name to the right never covers a dot
            for x0, y0 in zip(g["x"][top], y[top]):
                ax.annotate(G.paper_name(top_fam), (x0, y0), xytext=(2.5, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=5.5, color=S.INK)
        ax.plot([g["x"].median()] * 2, [pos[(col, lv)] - .35, pos[(col, lv)] + .35], color=S.INK, lw=1, zorder=3)
    for i, (col, lv) in enumerate(rows):
        if i and lv is None:
            ax.axhline(i - .5, color=S.GRID, lw=.6)
    ax.axvline(0, color=S.MUTED, lw=.7, ls=":")
    lo, hi = t["x"].min(), t["x"].max()
    ax.set_xlim(lo - .05 * (hi - lo), hi + (.05 if per_task else .22) * (hi - lo))     # room for the name inside the frame
    ax.set_ylim(len(rows) - .5, -.5)
    ax.grid(axis="x", color=S.GRID, lw=.5); S.clean(ax)
    _group_labels(ax, rows, counts, colours)
    ax.set_xlabel("DA-size minus 0.5")
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    if per_task:
        top = G._header(fig, "Task DA-size by characteristic level",
                        f"dot = one (benchmark, language) task (jittered): its DA-size − 0.5 (0 = chance agreement), the mean "
                        f"over the proxies 90M–1B that have one (mono-axis, against 1.7B), on the {FILTER} tasks; bar = the "
                        "level's median, line = its interquartile range over the tasks; count = tasks")
    else:
        top = G._header(fig, "Benchmark DA-size by characteristic level",
                        f"dot = one benchmark family (jittered): its DA-size − 0.5 (0 = chance agreement), the mean over 90M–1B of "
                        f"the median over its {FILTER} tasks (mono-axis, against 1.7B); bar = the level's median; count = benchmarks; "
                        f"{G.paper_name(top_fam)}, the highest, named once per characteristic")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def fig_quadrant(t: pd.DataFrame, pop: str, path: Path, paper: bool) -> pd.DataFrame:
    """One panel per characteristic: x = DA-size - 0.5 of population `pop` at one proxy, y = the
    centred code, one dot per benchmark (or task) and proxy, coloured by the proxy's size and drawn in
    a band of its own inside the level's row (90M lowest, jittered within the band), with the band's
    median as a red tick; the dots counted per quadrant."""
    per_task = "task" in t
    t = t[t["population"] == pop]
    cols = list(dict.fromkeys(t["characteristic"]))
    sizes = size_order(t["size"].unique())
    rng = np.random.default_rng(0)
    ncol = 3
    nrow = -(-len(cols) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.2 * nrow + 0.5), sharex=True, squeeze=False)
    counts = []
    lim = max(0.3, t["x"].abs().max() + 0.05)
    for ax, col in zip(axes.flat, cols):
        g = t[t["characteristic"] == col]
        levels = CHARACTERISTICS[col][2]
        band = min(0.4, 0.42 * 2 / (len(levels) - 1)) / len(sizes)          # one sub-band per proxy size
        ax.axhspan(0, 1.7, xmin=0.5, xmax=1, color=S.RAMP[0], alpha=.12, lw=0)
        ax.axhspan(-1.7, 0, xmin=0, xmax=0.5, color=S.RAMP[0], alpha=.12, lw=0)
        ax.axvline(0, color=S.MUTED, lw=.7); ax.axhline(0, color=S.MUTED, lw=.7)
        for k, s in enumerate(sizes):
            h = g[g["size"] == s]
            off = (2 * k - len(sizes) + 1) * band
            ax.scatter(h["x"], h["code"] + off + rng.uniform(-band * .8, band * .8, len(h)), s=3 if per_task else 10,
                       color=S.SIZE_COLOR[s], alpha=.8 if per_task else 1, lw=0, zorder=2)
            for c0, m in h.groupby("code")["x"].median().items():          # the band's median
                ax.plot([m, m], [c0 + off - band, c0 + off + band], color=S.ALERT, lw=1, zorder=3)
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            n = int(((np.sign(g["x"]) == sx) & (np.sign(g["code"]) == sy)).sum())
            counts.append({"characteristic": col, "x_side": sx, "code_side": sy, "n_points": n})
            ax.text(sx * lim * .95, sy * 1.55, str(n), ha="right" if sx > 0 else "left", va="center", fontsize=7,
                    color=S.INK, zorder=4)
        ax.set_yticks([encode(col, lv) for lv in levels]); ax.set_yticklabels([LEVEL_NAMES[lv] for lv in levels], fontsize=7)
        ax.set_ylim(-1.7, 1.7); ax.set_xlim(-lim, lim); ax.set_title(CHARACTERISTICS[col][0], fontsize=8, loc="left")
        step = 0.5 if lim > 0.5 else 0.2
        ax.set_xticks([-step, 0, step]); ax.set_xticklabels([f"{v:g}".replace("-", "−") for v in (-step, 0, step)])
        S.clean(ax)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    axes[-1, ncol // 2].set_xlabel("DA-size minus 0.5")
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color=S.SIZE_COLOR[s], label=s) for s in sizes]
    legend = lambda y: fig.legend(handles=handles, title="Proxy size", title_fontsize=7, fontsize=7, frameon=False,
                                  ncol=len(sizes), loc="upper center", bbox_to_anchor=(0.5, y))
    if paper:
        legend(1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        S.save_paper(fig, path.with_suffix(""))
    else:
        what = f"its {FILTER} tasks" if pop == FILTER else "every task above chance at the proxy and at 1.7B"
        unit = (f"one (benchmark, language) task at one proxy; x = its DA-size − 0.5 (0 = chance agreement; on {what}, "
                "mono-axis, against 1.7B)") if per_task else (
                f"one benchmark family at one proxy; x = its DA-size − 0.5 (0 = chance agreement; the median over {what}, "
                "mono-axis, against 1.7B)")
        top_ = G._header(fig, f"{'Task' if per_task else 'Benchmark'} DA-size against the coded characteristics",
                         f"dot = {unit}, coloured by the proxy's size; y = the characteristic's centred code (ordinal "
                         "levels ranked, binary ±1), each proxy in a band of its own (90M lowest), jittered; red tick = the band's median; number = dots per "
                         "quadrant (a dot on a zero line is in none); "
                         "shaded = the quadrants a positive correlation fills")
        legend(top_)
        fig.tight_layout(rect=(0, 0, 1, top_ - 0.07))
        S.save(fig, path)
    return pd.DataFrame(counts)


# the readings the item-count and wave tables report: (reading in `values`, task_table column, label)
METRICS = (("unfiltered", "da_size", "DA-size"), ("da_ckpt", "da_ckpt", "DA-ckpt"),
           ("snr", "log_snr", "SNR at 1.7B"), ("r2", "r2", "R² over size"))


def _ols(y: pd.Series, X: pd.DataFrame, groups=None):
    """OLS of y on X with a constant: cluster-robust by `groups`, else HC1."""
    import statsmodels.api as sm
    m = sm.OLS(y.to_numpy(float), sm.add_constant(X.astype(float)))
    return m.fit(cov_type="cluster", cov_kwds={"groups": groups}) if groups is not None else m.fit(cov_type="HC1")


def item_count(values: dict, tt: pd.DataFrame, c: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Per metric: the family-level Spearman rho of item count and of skill (`correlations`), the slope
    against log10 items within families (task-level OLS with a family fixed effect, standard errors
    clustered by family, so only the families whose tasks differ in item count inform it), and a
    family-level fit on both (HC1), the skill coded 1 for LM-like."""
    rows = []
    for reading, col, label in METRICS:
        g = c[c["reading"] == reading].set_index("characteristic")
        d = tt.dropna(subset=[col, "log_items"])
        varies = d.groupby("family")["log_items"].transform("nunique") > 1
        fe = pd.get_dummies(d["family"], drop_first=True, dtype=float)
        w = _ols(d[col], pd.concat([d[["log_items"]], fe], axis=1), pd.factorize(d["family"])[0])
        y = values[reading].dropna()
        if reading == "snr":
            y = np.log10(y)
        f = feat.loc[y.index, ["items", "skill"]].dropna()
        j = _ols(y[f.index], f.assign(skill=(f["skill"] == LM_LIKE).astype(float)))
        ci, cj = w.conf_int(), j.conf_int()
        rows.append({
            "metric": label, "reading": reading, "n_benchmarks": int(g.loc["items", "n_benchmarks"]),
            **{f"rho_{k}{s}": g.loc[k, v] for k in ("items", "skill") for s, v in (("", "value"), ("_low", "ci_low"),
                                                                                 ("_high", "ci_high"))},
            "n_benchmarks_skill": int(g.loc["skill", "n_benchmarks"]),
            "within_slope": w.params["log_items"], "within_low": ci.loc["log_items", 0],
            "within_high": ci.loc["log_items", 1], "within_p": w.pvalues["log_items"], "within_n_tasks": len(d),
            "within_n_varying": int(d.loc[varies, "family"].nunique()),
            "joint_items": j.params["items"], "joint_items_low": cj.loc["items", 0], "joint_items_high": cj.loc["items", 1],
            "joint_skill": j.params["skill"], "joint_skill_low": cj.loc["skill", 0], "joint_skill_high": cj.loc["skill", 1],
            "joint_n": len(f)})
    return pd.DataFrame(rows)


def waves(values: dict, feat: pd.DataFrame) -> pd.DataFrame:
    """The robustness split: per metric and wave (analyze.WAVES), the median over the benchmark families with a
    value (each family counts once) with a 95 % bootstrap interval over the families; the original benchmarks
    against every later one by a two-sided Mann-Whitney test and by the later-wave coefficient of a
    family-level fit on log10 items (HC1)."""
    rng = np.random.default_rng(0)
    rows = []
    for reading, _, label in METRICS[:1] + METRICS[2:]:
        v = values[reading].dropna()
        wave = v.index.map(WAVE)
        later = v[wave != ORIGINAL]
        f = feat.loc[v.index, ["items"]].assign(later=(wave != ORIGINAL).astype(float)).dropna()
        y = np.log10(v[f.index]) if reading == "snr" else v[f.index]
        fit = _ols(y, f)
        for group, x in [(ORIGINAL, v[wave == ORIGINAL]), ("later, all", later)] + [
                (w, v[wave == w]) for w in WAVES if w != ORIGINAL]:
            boot = [np.median(rng.choice(x.to_numpy(), len(x))) for _ in range(N_BOOT)] if len(x) > 1 else [np.nan]
            rows.append({"metric": label, "reading": reading, "group": group, "n_benchmarks": len(x),
                         "median": x.median() if len(x) else np.nan, "ci_low": np.nanpercentile(boot, 2.5),
                         "ci_high": np.nanpercentile(boot, 97.5),
                         "median_items": feat.loc[x.index, "n_items"].median() if len(x) else np.nan,
                         "mannwhitney_p": stats.mannwhitneyu(v[wave == ORIGINAL], later).pvalue
                         if group == "later, all" else np.nan,
                         "items_adjusted_later": fit.params["later"] if group == "later, all" else np.nan,
                         "items_adjusted_p": fit.pvalues["later"] if group == "later, all" else np.nan})
    return pd.DataFrame(rows)


def write_tex(rows: list[list[str]], head: list[str], align: str, path: Path) -> None:
    """A booktabs tabular the paper inputs (\\midrule where a row is None)."""
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    lines += ["\\midrule" if r is None else " & ".join(r) + " \\\\" for r in rows]
    path.write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""]))


def _ci(v, lo, hi, f="+.2f"):
    return "--" if not np.isfinite(v) else f"${v:{f}}$" + (f" [${lo:{f}}$, ${hi:{f}}$]" if np.isfinite(lo) else "")


def item_count_tex(t: pd.DataFrame, path: Path) -> None:
    rows = [[r.metric.replace("²", "$^2$"), str(r.n_benchmarks), _ci(r.rho_items, r.rho_items_low, r.rho_items_high),
             _ci(r.within_slope, r.within_low, r.within_high) + f" ({r.within_n_tasks}, {r.within_n_varying})",
             _ci(r.rho_skill, r.rho_skill_low, r.rho_skill_high),
             _ci(r.joint_items, r.joint_items_low, r.joint_items_high),
             _ci(r.joint_skill, r.joint_skill_low, r.joint_skill_high)] for r in t.itertuples()]
    write_tex(rows, ["Metric", "Benchmarks", "$\\rho$ items", "Within benchmarks", "$\\rho$ LM-like",
                     "Joint: items", "Joint: LM-like"], "lrlllll", path)


def waves_tex(t: pd.DataFrame, path: Path) -> None:
    rows = []
    for m, g in t.groupby("metric", sort=False):
        rows += [None] if rows else []
        for r in g.itertuples():
            sub = r.group not in (ORIGINAL, "later, all")
            rows.append([m.replace("²", "$^2$") if r.group == ORIGINAL else "",
                         ("\\quad " if sub else "") + r.group[0].upper() + r.group[1:], str(r.n_benchmarks),
                         "--" if not r.n_benchmarks else f"{r.median_items:.0f}",
                         _ci(r.median, r.ci_low, r.ci_high, ".2f"),
                         f"{r.mannwhitney_p:.2f}" if np.isfinite(r.mannwhitney_p) else "",
                         f"${r.items_adjusted_later:+.2f}$ ({r.items_adjusted_p:.2f})" if np.isfinite(r.items_adjusted_p) else ""])
    write_tex(rows, ["Metric", "Benchmarks added", "Benchmarks", "Items", "Median [95\\% CI]", "Rank test $p$",
                     "After items ($p$)"], "llrrlrl", path)


# the table's short level names (the figures use LEVEL_NAMES)
TABLE_NAMES = {"originally_multilingual": "native", "human_translation": "human transl.", "template_generated": "template",
               "mt_post_edited": "MT + post-edit", "machine_translation": "MT", "english_translated": "translated",
               "minimal_pair": "minimal pair", "completion": "completion", "classification": "classification",
               "mcq_question_only": "MCQ", "mrc_passage": "MCQ + passage", "cloze_completion": "cloze",
               "statement_continuation": "statement (LLM-RF)", "last_word": "last word",
               "english_original": "English", "translation_unknown": "transl. (unknown)", False: "no", True: "yes",
               KNOWLEDGE_ETC: "knowledge etc."}
TABLE_HEAD = {"domain": "Domain", "skill": "Skill", "curation_category": "Curation", "source_origin": "Source", "format": "Format", "n_options": "Options",
              "passage": "Passage", "n_items": "Items", "context_len_chars_median": "Context",
              "option_len_chars_median": "Option"}
LENGTHS = ("n_items", "context_len_chars_median", "option_len_chars_median")


def characteristics_table(families: list, feat: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """One row per benchmark, one column per characteristic: the CSV (with each annotation's source),
    and the tabular the paper prints (lengths are the median characters of 100 sampled items, --
    where none were sampled; options -- where none are scored, LAMBADA, or the range where the
    subtasks differ; `unknown` where no source settles a characteristic)."""
    t = feat.loc[families, list(TABLE_HEAD) + ["curation", "source", "format_type", "n_options_range",
                                                "data_source", "curation_process", "mixed"]].copy()
    t.insert(0, "benchmark", [G.paper_name(f) for f in families])
    t = t.sort_values("benchmark")
    t.to_csv(out_dir / "benchmark_characteristics.csv", index_label="family")
    def cell(c, v, r):
        if c == "n_options":
            return f"{v:.0f}" if pd.notna(v) else r["n_options_range"] if pd.notna(r["n_options_range"]) else "--"
        if pd.isna(v):
            return "--" if c in LENGTHS else "mixed" if r["mixed"] == True else "unknown"
        return f"{v:.0f}" if c in LENGTHS else TABLE_NAMES.get(v, v)
    lines = ["\\begin{tabular}{llllllllrrr}", "\\toprule",
             " & ".join(["Benchmark"] + list(TABLE_HEAD.values())) + " \\\\", "\\midrule"]
    lines += [" & ".join([r["benchmark"]] + [cell(c, r[c], r) for c in TABLE_HEAD]) + " \\\\" for _, r in t.iterrows()]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / "benchmark_characteristics.tex").write_text("\n".join(lines))
    return t


def generate_readme(pool: str, c: pd.DataFrame, fam: pd.DataFrame, tm: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = fam[fam["population"] == FILTER]
    n_fam, n_tasks = rel["family"].nunique(), int(rel.groupby("family")["n_tasks"].max().sum())
    below = (rel.groupby("family")["da_size"].mean() < CHANCE).sum()

    def line(reading):
        g = c[c["reading"] == reading].set_index("characteristic")
        task = reading.startswith("task:")
        return ", ".join(f"{r.label} ρ = {r.value:+.2f}"
                         + (f" [{r.ci_low:+.2f}, {r.ci_high:+.2f}]" if np.isfinite(r.ci_low) else "")
                         + (f" (n = {r.n_tasks} tasks of {r.n_benchmarks}"
                            + (f", Kruskal–Wallis p = {r.kruskal_p:.3f}" if np.isfinite(r.kruskal_p) else "") + ")" if task
                            else f" (n = {r.n_benchmarks})") for r in g.itertuples() if np.isfinite(r.value))

    def spread(pop):
        """How far from chance the benchmarks and their tasks sit (DA-size - 0.5), the evidence for the paper's view."""
        b = fam[fam["population"] == pop].groupby("family")["da_size"].mean() - CHANCE
        t = tm.loc[tm["population"] == pop, "da_size"] - CHANCE
        f = lambda x, w: (f"{w} median {x.median():+.2f} (IQR {x.quantile(.25):+.2f} to {x.quantile(.75):+.2f}, range "
                          f"{x.min():+.2f} to {x.max():+.2f}), {(x.abs() < .05).mean():.0%} within 0.05 of chance and "
                          f"{(x >= .16).mean():.0%} at DA-size ≥ 0.66")
        return f"{f(b, f'{len(b)} benchmarks')}. {f(t, f'{len(t)} tasks')}"
    sizes = [r for r in dict.fromkeys(c["reading"]) if r.startswith("size:")]
    opt = c[(c["characteristic"] == "n_options") & c["reading"].isin(sizes)].set_index("reading")["value"]
    meta = features()
    dom = (tm.assign(domain=tm["family"].map(meta["domain"]))
           .groupby(["population", "domain"]).agg(n_fam=("family", "nunique"), n=("task", "size"), da=("da_size", "median")))
    snr = pd.read_csv(HERE / stage / pool / "per_family_snr.csv").assign(domain=lambda d: d["family"].map(meta["domain"]))
    snr_dom = snr.groupby("domain")["snr_median"].agg(["median", "size"]).sort_values("median", ascending=False)
    files = (f"design_da_size_correlation_{FILTER}_mono_axis", f"design_da_size_by_level_{FILTER}_mono_axis",
             f"design_da_size_by_level_per_task_{FILTER}_mono_axis", "design_da_size_quadrant_mono_axis",
             "design_da_size_quadrant_per_task_mono_axis", f"design_da_size_quadrant_{FILTER}_mono_axis",
             f"design_da_size_quadrant_per_task_{FILTER}_mono_axis")
    body = "\n\n".join([
        "## Benchmark characteristics against DA-size",
        f"Pool `{pool}`: rq02's per-task DA-size (mono-axis pairs, each proxy's final checkpoint against the 1.7B final, gated "
        f"at the proxy and at 1.7B, ≥ 3 pairs) over the paper's `{FILTER}` tasks ({n_fam} families, up to {n_tasks} tasks), "
        "with the unfiltered reading (every gated task) beside it. A benchmark's DA-size is the median over its tasks at a "
        "proxy, averaged over 90M–1B, and a task's DA-size is its mean over the proxies that have one. Statistic: "
        "Spearman ρ against the centred code of an ordinal or "
        "binary characteristic (options 2 → 5, no passage → passage) or against a continuous one (the item count, the median "
        "`n_items` over a benchmark's languages, and the two lengths). Skill is knowledge, reading, reasoning → LM-like "
        "(`analyze.DOMAINS`), None for a suite whose subtasks fall in both groups. Curation, source, task format and "
        "domain are binarised: translated → native (written in the language, English originals included, or generated from "
        "treebanks or planning problems), translated from English → not, continuation → lettered MCQ, and every other "
        "domain → knowledge (the most populous of the eleven domains, chosen by its count). A "
        "characteristic that no source settles (`FAMILY_META` None: BBH's option count and passage, LAMBADA's option "
        "count) leaves that benchmark out of that row. BBH's curation method is unknown too, but its items were written "
        "in English, so the binary curation codes it native (`features`). The interval is a 95 % bootstrap over the families, "
        "which for the per-task readings resamples each benchmark with all of its tasks. The by-level figures group each "
        "characteristic's levels under a header row. The quadrant figures draw one dot per proxy, coloured by its size, "
        "with each size in a band of its own and its median marked. They read every task above chance, so "
        "benchmarks fall on both sides of chance agreement. Their "
        f"`{FILTER}` twins, where {n_fam - below} of the {n_fam} benchmarks sit right of 0.5, stay as analysis figures. "
        f"Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool {pool}`.",
        f"![Characteristics against DA-size]({stage}/{pool}/{files[0]}.png)",
        f"![DA-size by characteristic level, one dot per benchmark]({stage}/{pool}/{files[1]}.png)",
        f"![DA-size by characteristic level, one dot per task]({stage}/{pool}/{files[2]}.png)",
        f"![DA-size against the coded characteristics, one dot per benchmark and proxy, every task above chance]"
        f"({stage}/{pool}/{files[3]}.png)",
        f"![DA-size against the coded characteristics, one dot per task and proxy, every task above chance]"
        f"({stage}/{pool}/{files[4]}.png)",
        "Key findings (generated):",
        "\n".join([f"- **DA-size, `{FILTER}`** (mean over 90M–1B): {line(FILTER)}.",
                   f"- **DA-size, every gated task**: {line('unfiltered')}.",
                   f"- **SNR at 1.7B** (the same statistics on rq09's family medians): {line('snr')}.",
                   f"- **DA-size per task, `{FILTER}`** (tasks as observations, the Kruskal–Wallis p treats them as "
                   f"independent, the interval does not): {line('task:' + FILTER)}.",
                   f"- **DA-size per task, every gated task**: {line('task:unfiltered')}.",
                   "- **Option count per proxy** (`" + FILTER + "`): "
                   + ", ".join(f"{s.split(':')[1]} ρ = {v:+.2f}" for s, v in opt.items() if np.isfinite(v)) + ".",
                   f"- **Aggregating pulls the benchmarks toward chance** (DA-size − 0.5). Every gated task: {spread('unfiltered')}. "
                   f"`{FILTER}`: {spread(FILTER)}.",
                   "- **Domain levels** (SNR median over the families at 1.7B, then the median task DA-size on every gated "
                   "task): " + ", ".join(
                       f"{d} {r['median']:.2f} ({int(r['size'])} {'family' if r['size'] == 1 else 'families'}"
                       + (f", task DA-size {dom.loc[('unfiltered', d), 'da']:.2f} over {dom.loc[('unfiltered', d), 'n']} "
                          f"task{'' if dom.loc[('unfiltered', d), 'n'] == 1 else 's'})"
                          if ("unfiltered", d) in dom.index else ", no gated task)")
                       for d, r in snr_dom.iterrows()) + "."]),
        "GitHub: " + " · ".join(
            f"[{n}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/"
            f"rq09_benchmark_design/{stage}/{pool}/{n})"
            for n in [f"{f}.{e}" for f in files for e in ("png", "csv")]
            + ["design_da_size_per_family_mono_axis.csv", "design_da_size_per_task_mono_axis.csv", "benchmark_characteristics.csv"]),
        "Table: `benchmark_characteristics.csv` (`.tex` for the paper) lists each benchmark's characteristics."])
    replace_block(HERE / "README.md", "design-da", body, f"design_da.py --pool {pool}")


def readme_items(pool: str, items: pd.DataFrame, w: pd.DataFrame, feat: pd.DataFrame, tt: pd.DataFrame) -> None:
    """The README block of the item-count and skill axes and of the wave split, and the taxonomy table
    (analyze.DOMAINS, the one place it is defined)."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    fams = sorted(set(tt["family"]))
    by_domain = {d: [] for d in DOMAINS}
    for f in fams:                                   # a suite is listed under each of its subtasks' domains
        tasks = tt.loc[tt["family"] == f, "task"]
        if FAMILY_META[f].get("domain"):
            by_domain[FAMILY_META[f]["domain"]].append(G.paper_name(f))
        for p, d in SUBTASK_DOMAIN.items():
            if FAMILY_META[f].get("domain") is None and tasks.str.startswith(p).any():
                by_domain[d].append(f"{G.paper_name(f)} `{p}`")
    table = "\n".join(["| domain | skill | benchmarks with a value | basis |", "|---|---|---|---|"]
                       + [f"| {d} | {sk} | {', '.join(by_domain[d]) or 'none'} | {basis} |"
                          for d, (sk, basis) in DOMAINS.items()])
    ci = lambda v, lo, hi: f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    lines = [f"- **{r.metric}** ({r.n_benchmarks} benchmarks): items ρ = {ci(r.rho_items, r.rho_items_low, r.rho_items_high)}, "
             f"LM-like ρ = {ci(r.rho_skill, r.rho_skill_low, r.rho_skill_high)} ({r.n_benchmarks_skill} benchmarks with one "
             f"skill). Within benchmarks {r.within_slope:+.3f} per tenfold items [{r.within_low:+.3f}, {r.within_high:+.3f}], "
             f"p = {r.within_p:.3g} ({r.within_n_tasks} tasks, {r.within_n_varying} benchmarks whose tasks differ in item "
             f"count). Joint fit over {r.joint_n} benchmarks: items {ci(r.joint_items, r.joint_items_low, r.joint_items_high)}, "
             f"LM-like {ci(r.joint_skill, r.joint_skill_low, r.joint_skill_high)}." for r in items.itertuples()]
    wl = []
    for m, g in w.groupby("metric", sort=False):
        g = g.set_index("group")
        a = g.loc["later, all"]
        wl.append(f"- **{m}**: " + ", ".join(f"{k} {r['median']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] "
                                            f"({int(r['n_benchmarks'])} benchmarks, median {r['median_items']:.0f} items)"
                                            for k, r in g.iterrows() if r["n_benchmarks"])
                  + f". Original against later: Mann–Whitney p = {a['mannwhitney_p']:.2f}. After log10 items the later "
                  f"benchmarks differ by {a['items_adjusted_later']:+.2f} (p = {a['items_adjusted_p']:.2f}).")
    names = ["item_count_da_all_mono_axis", "wave_robustness_da_size_mono_axis"]
    body = "\n\n".join([
        "## Item count and skill",
        f"Pool `{pool}`. A benchmark's item count is the median `n_items` (configs/tasks.json) over its per-language "
        "tasks, and its skill is the group of its domain (the table below): LM-like for commonsense, grammaticality and "
        "language modelling, against every other domain. A suite whose subtasks fall in both groups (KoBEST, NorEval) "
        "has no skill. Per metric, a benchmark's value is its DA-size on every gated task (as in the hollow dots above), "
        "the median over its tasks of DA-ckpt (each task's median over its DA-ckpt cells) and of the R² of rq01's "
        "power-law fit over size (pool `predictivity_seeds`), and its median SNR at 1.7B (log10 in the fits). The ρ "
        "are the Spearman correlations of the figure above with their bootstrap intervals over the benchmarks. "
        "Within benchmarks: an OLS of the task values on log10 items with a fixed effect per benchmark, the interval "
        "clustered by benchmark, so only the benchmarks whose tasks differ in item count inform it. Joint fit: a "
        "family-level OLS on log10 items and LM-like (HC1 interval). Regenerate with "
        f"`python analysis/rq09_benchmark_design/design_da.py --pool {pool}`.",
        table,
        "Item count and skill (generated, `item_count_da_all_mono_axis.csv`):",
        "\n".join(lines),
        "Original against later-added benchmarks (generated, `wave_robustness_da_size_mono_axis.csv`; a benchmark "
        "counts once, its value as above; `analyze.WAVES` dates each family from configs/tasks.json):",
        "\n".join(wl),
        "GitHub: " + " · ".join(
            f"[{n}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/"
            f"rq09_benchmark_design/{stage}/{pool}/{n})"
            for n in [f"{f}.{e}" for f in names for e in ("csv", "tex")] + ["item_count_per_task_da_all_mono_axis.csv"])])
    replace_block(HERE / "README.md", "item-count", body, f"design_da.py --pool {pool}")


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = HERE / stage / pool
    tasks, ckpt = per_task(DECISION_ACCURACY / stage / pool, pool)
    tasks.to_csv(out_dir / "design_da_size_per_task_mono_axis.csv", index=False)
    fam = per_family(tasks)
    fam.to_csv(out_dir / "design_da_size_per_family_mono_axis.csv", index=False)
    tm = task_means(tasks)
    snr = pd.read_csv(out_dir / "per_family_snr.csv")
    feat = features()
    tt = task_table(tm, ckpt, out_dir, stage)
    tt.to_csv(out_dir / "item_count_per_task_da_all_mono_axis.csv", index=False)
    values = benchmark_values(fam, snr, tm, tt)
    c = correlations(values, feat)
    stem = f"design_da_size_correlation_{FILTER}_mono_axis"
    for paper in (False, True):
        name = stem + ("_paper" if paper else "")
        c.to_csv(out_dir / f"{name}.csv", index=False)
        fig_correlation(c, out_dir / f"{name}.png", paper)
    bench = pd.concat([values[p].dropna().rename("da_size").rename_axis("family").reset_index().assign(population=p)
                       for p in (FILTER, "unfiltered")])
    # the by-level figures read a benchmark's (task's) mean over the proxies, the quadrants one dot per proxy
    levels = {"": level_table(bench, feat), "_per_task": level_table(tm, feat)}
    quadrants = {"": level_table(fam, feat), "_per_task": level_table(tasks.rename(columns={"da": "da_size"}), feat)}
    for paper in (False, True):
        sfx = "_paper" if paper else ""
        for unit, lv in levels.items():
            name = f"design_da_size_by_level{unit}_{FILTER}_mono_axis{sfx}"
            lv[lv["population"] == FILTER].to_csv(out_dir / f"{name}.csv", index=False)
            fig_levels(lv, out_dir / f"{name}.png", paper)
        for unit, lv in quadrants.items():
            for pop, name in (("unfiltered", f"design_da_size_quadrant{unit}_mono_axis"),
                              (FILTER, f"design_da_size_quadrant{unit}_{FILTER}_mono_axis")):
                q = out_dir / f"{name}{sfx}"
                counts = fig_quadrant(lv, pop, q.with_suffix(".png"), paper)
                pd.concat([lv[lv["population"] == pop].assign(row="task" if unit else "benchmark"),
                           counts.assign(row="quadrant count")]).to_csv(q.with_suffix(".csv"), index=False)
    families = sorted(set(snr["family"]) | set(fam["family"]))
    characteristics_table(families, feat, out_dir)
    items = item_count(values, tt, c, feat)
    items.to_csv(out_dir / "item_count_da_all_mono_axis.csv", index=False)
    item_count_tex(items, out_dir / "item_count_da_all_mono_axis.tex")
    w = waves(values, feat)
    w.to_csv(out_dir / "wave_robustness_da_size_mono_axis.csv", index=False)
    waves_tex(w, out_dir / "wave_robustness_da_size_mono_axis.tex")
    print(items.round(3).to_string(index=False)); print(w.round(3).to_string(index=False))
    print(c[c["reading"].isin([FILTER, "unfiltered", "snr", f"task:{FILTER}", "task:unfiltered"])].round(3).to_string(index=False))
    generate_readme(pool, c, fam, tm)
    readme_items(pool, items, w, feat, tt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
