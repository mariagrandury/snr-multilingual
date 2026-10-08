"""Benchmark design against DA-size: do the characteristics rq09 reads on SNR
also order the benchmarks by how well a small proxy decides like the reference?

The SNR analysis (`analyze.py`) groups the benchmark families by the
characteristics of `FAMILY_META` (curation, source origin, task format,
option count, reading passage) and the length features (`length_features.csv`).
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
              binary one -1/+1; a length as is). The two characteristics with more than two
              unordered levels are binarised (the user's call, 2026-10-08): curation as
              translated (human, MT, MT + post-edit) vs native (written in the language or
              generated from its treebanks), task format as continuation (the options scored
              as continuations: completion, minimal pair, classification labels, RF cloze,
              LLM-RF statement) vs lettered MCQ (question or passage with lettered options).
              On these families native curation and an originally multilingual source pick
              the same benchmarks, so those two rows coincide. 95 % percentile bootstrap over
              the benchmarks; the Kruskal-Wallis H and p of every two-group split beside.
              The same statistics on rq09's SNR (median `snr_mpd_1.7B`) sit beside.

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
                                                the same on the above_66_either tasks (every benchmark lands
                                                right of chance there)

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
from analysis.rq09_benchmark_design.analyze import FAMILY_META  # noqa: E402
from analysis.utils import SMALL_SIZES, _is_language_aggregate, size_order  # noqa: E402

HERE = Path(__file__).resolve().parent
AXES = "mono-axis"
FILTER = "above_66_either"
N_BOOT = 1000
CHANCE = 0.5
# characteristic -> (label, kind, the FAMILY_META / length column, the level order). Ordinal and
# binary levels are listed low to high: the code is the level's rank, centred at 0.
CHARACTERISTICS = {
    "curation": ("Curation", "binary", ["translated", "native"]),
    "source_origin": ("Source", "binary", ["english_translated", "originally_multilingual"]),
    "format_type": ("Task format", "binary", ["continuation", "lettered MCQ"]),
    "n_options": ("Answer options", "ordinal", [2, 3, 4]),
    "passage": ("Reading passage", "binary", [False, True]),
    "context_len_chars_median": ("Context length", "continuous", None),
    "option_len_chars_median": ("Option length", "continuous", None),
}
# the binarised characteristics: FAMILY_META level -> binary level
NATIVE_CURATION = {"originally_multilingual", "template_generated"}
LETTERED_FORMATS = {"mcq_question_only", "mrc_passage"}
LEVEL_NAMES = {"translated": "translated", "native": "native", "continuation": "continuation",
               "lettered MCQ": "lettered MCQ","originally_multilingual": "originally multilingual", "human_translation": "human translation",
               "template_generated": "template generated", "mt_post_edited": "MT post-edited",
               "machine_translation": "machine translation", "english_translated": "translated from English",
               "minimal_pair": "minimal pair", "completion": "completion", "classification": "classification",
               "mcq_question_only": "question with options", "mrc_passage": "passage with options",
               "cloze_completion": "cloze (RF)", "statement_continuation": "statement (LLM-RF)",
               2: "2 options", 3: "3 options", 4: "4 options", False: "no passage", True: "passage"}
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
    meta["curation"] = np.where(meta["curation_category"].isin(NATIVE_CURATION), "native", "translated")
    meta["format_type"] = np.where(meta["format"].isin(LETTERED_FORMATS), "lettered MCQ", "continuation")
    lf = pd.read_csv(LENGTH_CSV).set_index("family")
    return meta.join(lf[["context_len_chars_median", "option_len_chars_median"]])


def per_family(out_dir: Path, pool: str) -> pd.DataFrame:
    """(population, family, proxy) -> median DA-size over the family's tasks and their count."""
    l = long_da(out_dir, pool, axes=AXES)
    l = l[(l["kind"] == "size") & l["size"].isin(SMALL_SIZES)]
    l = l[[f in FAMILY_META and _is_language_aggregate(t, f) for t, f in zip(l["task"], l["family"])]]
    rel = load_reliable(out_dir, FILTER, axes=AXES)
    parts = [l.assign(population="unfiltered"), l[l["task"].isin(set(rel["task"]))].assign(population=FILTER)]
    d = pd.concat(parts)
    return (d.groupby(["population", "family", "size"])
            .agg(da_size=("da", "median"), n_tasks=("task", "nunique"), n_pairs_min=("n_pairs", "min")).reset_index())


def benchmark_values(fam: pd.DataFrame, snr: pd.DataFrame) -> dict[str, pd.Series]:
    """Each reading's value per benchmark: mean DA-size over the proxies (both populations), each
    proxy's DA-size (`size:<s>`, filtered population) and the SNR median."""
    out = {}
    for pop in (FILTER, "unfiltered"):
        out[pop] = fam[fam["population"] == pop].groupby("family")["da_size"].mean()
    f = fam[fam["population"] == FILTER]
    for s in size_order(f["size"].unique()):
        out[f"size:{s}"] = f[f["size"] == s].set_index("family")["da_size"]
    out["snr"] = snr.set_index("family")["snr_median"]
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
        draws = [v.iloc[rng.integers(0, len(v), len(v))] for _ in range(N_BOOT)] if not reading.startswith("size:") else []
        for col, (label, kind, _) in CHARACTERISTICS.items():
            n = int(feat.loc[v.index, col].notna().sum())
            row = {"reading": reading, "characteristic": col, "label": label, "kind": kind,
                   "statistic": "spearman_rho", "value": statistic(col, v, feat),
                   "n_benchmarks": n, "ci_low": np.nan, "ci_high": np.nan, "kruskal_H": np.nan, "kruskal_p": np.nan}
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
                    f"characteristic (options 2 → 4, no passage → passage, translated → originally multilingual or native, "
                    f"continuation → lettered MCQ) or a length; DA-size = rq02's mono-axis DA-size against 1.7B, median over a "
                    f"family's tasks per proxy; left: its mean over 90M–1B with a 95 % bootstrap CI over the families, on the "
                    f"{FILTER} tasks (filled), on every gated task (hollow) and rq09's SNR at 1.7B (diamond); right: one line "
                    f"per proxy ({FILTER})")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def level_table(values: dict[str, pd.Series], feat: pd.DataFrame) -> pd.DataFrame:
    """One row per (population, characteristic, benchmark): its level, centred code and DA-size - 0.5."""
    rows = []
    for pop in (FILTER, "unfiltered"):
        for col, (label, kind, levels) in CHARACTERISTICS.items():
            if kind == "continuous":
                continue
            for fam, v in values[pop].dropna().items():
                lv = feat.at[fam, col]
                rows.append({"population": pop, "characteristic": col, "label": label, "kind": kind,
                             "level": LEVEL_NAMES[lv], "level_rank": levels.index(lv),
                             "code": encode(col, lv),
                             "family": fam, "name": G.paper_name(fam), "da_size": v, "x": v - CHANCE})
    return pd.DataFrame(rows)


def example(t: pd.DataFrame) -> str:
    """The highlighted benchmark: the highest DA-size of the filtered reading."""
    f = t[t["population"] == FILTER]
    return f.loc[f["da_size"].idxmax(), "family"]


def fig_levels(t: pd.DataFrame, path: Path, paper: bool) -> None:
    """A row per characteristic level, grouped and coloured by characteristic; x = DA-size - 0.5
    (the filtered population), the highest benchmark outlined and named."""
    top_fam = example(t)
    t = t[t["population"] == FILTER]
    t = t.sort_values(["characteristic", "level_rank"], key=lambda s: s.map(list(CHARACTERISTICS).index)
                      if s.name == "characteristic" else s)
    rows = list(dict.fromkeys(zip(t["characteristic"], t["level"])))
    pos = {r: i for i, r in enumerate(rows)}
    colours = dict(zip(dict.fromkeys(t["characteristic"]), plt.cm.tab10.colors))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.6, 0.19 * len(rows) + 0.9))
    for (col, lv), g in t.groupby(["characteristic", "level"], sort=False):
        y = pos[(col, lv)] + rng.uniform(-0.22, 0.22, len(g))
        top = (g["family"] == top_fam).to_numpy()
        ax.scatter(g["x"], y, s=np.where(top, 20, 12), color=colours[col], lw=np.where(top, .8, .3),
                   edgecolor=np.where(top, S.INK, "white"), zorder=2)
        for x0, y0 in zip(g["x"][top], y[top]):
            ax.annotate(G.paper_name(top_fam), (x0, y0), xytext=(-3, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=6, color=S.INK)
        ax.plot([g["x"].median()] * 2, [pos[(col, lv)] - .35, pos[(col, lv)] + .35], color=S.INK, lw=1, zorder=3)
    for i, (col, lv) in enumerate(rows):
        if i and rows[i - 1][0] != col:
            ax.axhline(i - .5, color=S.GRID, lw=.6)
    ax.axvline(0, color=S.MUTED, lw=.7, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{CHARACTERISTICS[c][0]}: {lv} ({int(((t['characteristic'] == c) & (t['level'] == lv)).sum())})"
                        for c, lv in rows], fontsize=6.5)
    for lab, (c, _) in zip(ax.get_yticklabels(), rows):
        lab.set_color(colours[c])
    ax.invert_yaxis(); ax.grid(axis="x", color=S.GRID, lw=.5); S.clean(ax)
    ax.set_xlabel("DA-size minus 0.5")
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    top = G._header(fig, "Benchmark DA-size by characteristic level",
                    f"dot = one benchmark family (jittered): its DA-size − 0.5 (0 = chance agreement), the mean over 90M–1B of "
                    f"the median over its {FILTER} tasks (mono-axis, against 1.7B); bar = the level's median; count = benchmarks")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def fig_quadrant(t: pd.DataFrame, pop: str, path: Path, paper: bool) -> pd.DataFrame:
    """One panel per characteristic: x = DA-size - 0.5 of population `pop`, y = the centred code
    (jittered); the benchmarks counted per quadrant, the highest DA-size outlined."""
    top_fam = example(t)
    t = t[t["population"] == pop]
    cols = list(dict.fromkeys(t["characteristic"]))
    rng = np.random.default_rng(0)
    ncol = 3
    nrow = -(-len(cols) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.2 * nrow + 0.3), sharex=True, squeeze=False)
    counts = []
    lim = max(0.3, t["x"].abs().max() * 1.15)
    for ax, col in zip(axes.flat, cols):
        g = t[t["characteristic"] == col]
        top = (g["family"] == top_fam).to_numpy()
        ax.axhspan(0, 1.4, xmin=0.5, xmax=1, color=S.RAMP[0], alpha=.12, lw=0)
        ax.axhspan(-1.4, 0, xmin=0, xmax=0.5, color=S.RAMP[0], alpha=.12, lw=0)
        ax.axvline(0, color=S.MUTED, lw=.7); ax.axhline(0, color=S.MUTED, lw=.7)
        ax.scatter(g["x"], g["code"] + rng.uniform(-0.15, 0.15, len(g)), s=np.where(top, 20, 14), color=S.SERIES[0],
                   lw=np.where(top, .8, .3), edgecolor=np.where(top, S.INK, "white"), zorder=2)
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            n = int(((np.sign(g["x"]) == sx) & (np.sign(g["code"]) == sy)).sum())
            counts.append({"characteristic": col, "x_side": sx, "code_side": sy, "n_benchmarks": n})
            ax.text(sx * lim * .92, sy * 1.3, str(n), ha="right" if sx > 0 else "left", va="center", fontsize=7,
                    color=S.INK)
        levels = CHARACTERISTICS[col][2]
        ax.set_yticks([encode(col, lv) for lv in levels]); ax.set_yticklabels([LEVEL_NAMES[lv] for lv in levels], fontsize=7)
        ax.set_ylim(-1.45, 1.45); ax.set_xlim(-lim, lim); ax.set_title(CHARACTERISTICS[col][0], fontsize=8, loc="left")
        S.clean(ax)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    axes[-1, 1].set_xlabel("DA-size minus 0.5")
    for ax in axes.flat:
        ax.tick_params(labelbottom=True)
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
    else:
        what = f"its {FILTER} tasks" if pop == FILTER else "every task above chance at the proxy and at 1.7B"
        top_ = G._header(fig, "Benchmark DA-size against the coded characteristics",
                         f"dot = one benchmark family; x = DA-size − 0.5 (0 = chance agreement; mean over 90M–1B of the median "
                         f"over {what}, mono-axis, against 1.7B); y = the characteristic's centred code (ordinal levels "
                         "ranked, binary ±1), jittered; number = benchmarks per quadrant (a benchmark on a zero line is in "
                         f"none); shaded = the quadrants a positive correlation fills; outlined = {G.paper_name(top_fam)}")
        fig.tight_layout(rect=(0, 0, 1, top_))
        S.save(fig, path)
    return pd.DataFrame(counts)


# the table's short level names (the figures use LEVEL_NAMES)
TABLE_NAMES = {"originally_multilingual": "native", "human_translation": "human transl.", "template_generated": "template",
               "mt_post_edited": "MT + post-edit", "machine_translation": "MT", "english_translated": "translated",
               "minimal_pair": "minimal pair", "completion": "completion", "classification": "classification",
               "mcq_question_only": "MCQ", "mrc_passage": "MCQ + passage", "cloze_completion": "cloze (RF)",
               "statement_continuation": "statement (LLM-RF)", False: "no", True: "yes"}
TABLE_HEAD = {"curation_category": "Curation", "source_origin": "Source", "format": "Format", "n_options": "Options",
              "passage": "Passage", "context_len_chars_median": "Context", "option_len_chars_median": "Option"}
LENGTHS = ("context_len_chars_median", "option_len_chars_median")


def characteristics_table(families: list, feat: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """One row per benchmark, one column per characteristic: the CSV, and the tabular the paper prints
    (lengths are the median characters of 100 sampled items; -- where none were sampled)."""
    t = feat.loc[families, list(TABLE_HEAD) + ["curation", "format_type"]].copy()
    t.insert(0, "benchmark", [G.paper_name(f) for f in families])
    t = t.sort_values("benchmark")
    t.to_csv(out_dir / "benchmark_characteristics.csv", index_label="family")
    cell = lambda c, v: ("--" if pd.isna(v) else f"{v:.0f}" if c in LENGTHS
                         else str(v) if c == "n_options" else TABLE_NAMES[v])
    lines = ["\\begin{tabular}{lllllrrr}", "\\toprule",
             " & ".join(["Benchmark"] + list(TABLE_HEAD.values())) + " \\\\", "\\midrule"]
    lines += [" & ".join([r["benchmark"]] + [cell(c, r[c]) for c in TABLE_HEAD]) + " \\\\" for _, r in t.iterrows()]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / "benchmark_characteristics.tex").write_text("\n".join(lines))
    return t


def generate_readme(pool: str, c: pd.DataFrame, fam: pd.DataFrame, quad: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = fam[fam["population"] == FILTER]
    n_fam, n_tasks = rel["family"].nunique(), int(rel.groupby("family")["n_tasks"].max().sum())

    def line(reading):
        g = c[c["reading"] == reading].set_index("characteristic")
        return "; ".join(f"{r.label} ρ = {r.value:+.2f}"
                         + (f" [{r.ci_low:+.2f}, {r.ci_high:+.2f}]" if np.isfinite(r.ci_low) else "")
                         + f" (n = {r.n_benchmarks})" for r in g.itertuples() if np.isfinite(r.value))
    sizes = [r for r in dict.fromkeys(c["reading"]) if r.startswith("size:")]
    opt = c[(c["characteristic"] == "n_options") & c["reading"].isin(sizes)].set_index("reading")["value"]
    body = "\n\n".join([
        "## Benchmark characteristics against DA-size",
        f"Pool `{pool}`: rq02's per-task DA-size (mono-axis pairs, each proxy's final checkpoint against the 1.7B final, gated "
        f"at the proxy and at 1.7B, ≥ 3 pairs) over the paper's `{FILTER}` tasks ({n_fam} families, up to {n_tasks} tasks), "
        "with the unfiltered reading (every gated task) beside it; a benchmark's DA-size is the median over its tasks at a "
        "proxy and the mean of those medians over 90M–1B. Statistic: Spearman ρ against the centred code of an ordinal or "
        "binary characteristic (options 2 → 4, no passage → passage, translated → originally multilingual) or a length; "
        "curation and task format are binarised, translated → native (written in the language or generated from its "
        "treebanks) and continuation → lettered MCQ, and on these families native curation picks the same benchmarks as "
        "an originally multilingual source, so those two rows coincide; 95 % bootstrap over the families. The quadrant "
        "figure reads every task above chance, so benchmarks fall on both sides of chance agreement; its "
        f"`{FILTER}` twin, where every benchmark sits right of 0.5, stays as an analysis figure. Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool {pool}`.",
        f"![Characteristics against DA-size]({stage}/{pool}/design_da_size_correlation_above_66_either_mono_axis.png)",
        f"![DA-size by characteristic level]({stage}/{pool}/design_da_size_by_level_above_66_either_mono_axis.png)",
        f"![DA-size against the coded characteristics, every task above chance]({stage}/{pool}/design_da_size_quadrant_mono_axis.png)",
        "Key findings (generated):",
        "\n".join([f"- **DA-size, `{FILTER}`** (mean over 90M–1B): {line(FILTER)}.",
                   f"- **DA-size, every gated task**: {line('unfiltered')}.",
                   f"- **SNR at 1.7B** (the same statistics on rq09's family medians): {line('snr')}.",
                   "- **Option count per proxy** (`" + FILTER + "`): "
                   + ", ".join(f"{s.split(':')[1]} ρ = {v:+.2f}" for s, v in opt.items() if np.isfinite(v)) + "."]),
        "GitHub: " + " · ".join(
            f"[{n}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/"
            f"rq09_benchmark_design/{stage}/{pool}/{n})"
            for n in (f"design_da_size_correlation_{FILTER}_mono_axis.png", f"design_da_size_correlation_{FILTER}_mono_axis.csv",
                      f"design_da_size_by_level_{FILTER}_mono_axis.png", f"design_da_size_by_level_{FILTER}_mono_axis.csv",
                      "design_da_size_quadrant_mono_axis.png", "design_da_size_quadrant_mono_axis.csv",
                      f"design_da_size_quadrant_{FILTER}_mono_axis.png", f"design_da_size_quadrant_{FILTER}_mono_axis.csv",
                      "design_da_size_per_family_mono_axis.csv", "benchmark_characteristics.csv")),
        "Table: `benchmark_characteristics.csv` (`.tex` for the paper) lists each benchmark's characteristics."])
    replace_block(HERE / "README.md", "design-da", body, f"design_da.py --pool {pool}")


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = HERE / stage / pool
    fam = per_family(DECISION_ACCURACY / stage / pool, pool)
    fam.to_csv(out_dir / "design_da_size_per_family_mono_axis.csv", index=False)
    snr = pd.read_csv(out_dir / "per_family_snr.csv")
    feat = features()
    values = benchmark_values(fam, snr)
    c = correlations(values, feat)
    stem = f"design_da_size_correlation_{FILTER}_mono_axis"
    for paper in (False, True):
        name = stem + ("_paper" if paper else "")
        c.to_csv(out_dir / f"{name}.csv", index=False)
        fig_correlation(c, out_dir / f"{name}.png", paper)
    lv = level_table(values, feat)
    quad = None
    for paper in (False, True):
        sfx = "_paper" if paper else ""
        lv[lv["population"] == FILTER].to_csv(out_dir / f"design_da_size_by_level_{FILTER}_mono_axis{sfx}.csv", index=False)
        fig_levels(lv, out_dir / f"design_da_size_by_level_{FILTER}_mono_axis{sfx}.png", paper)
        for pop, name in (("unfiltered", "design_da_size_quadrant_mono_axis"),
                          (FILTER, f"design_da_size_quadrant_{FILTER}_mono_axis")):
            q = out_dir / f"{name}{sfx}"
            counts = fig_quadrant(lv, pop, q.with_suffix(".png"), paper)
            quad = counts if pop == "unfiltered" else quad
            pd.concat([lv[lv["population"] == pop].assign(row="benchmark"), counts.assign(row="quadrant count")]) \
                .to_csv(q.with_suffix(".csv"), index=False)
    families = sorted(set(snr["family"]) | set(fam["family"]))
    characteristics_table(families, feat, out_dir)
    print(c[c["reading"].isin([FILTER, "unfiltered", "snr"])].round(3).to_string(index=False))
    generate_readme(pool, c, fam, quad)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
