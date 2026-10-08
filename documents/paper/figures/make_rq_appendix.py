"""Write one appendix page per analysis folder from its README.

For every src/signal-and-noise/analysis/rqNN_<name>/ this writes
documents/paper/sections/app_rqNN_<name>.tex, one page: \\clearpage, then a
full-width block at the top of the page (\\twocolumn[...], so it cannot drift
to another page as a figure* would): the subsection title (the first page
also opens their parent section, PARENT), the research question
and the setup in the left column, the key findings in the right one, and the
main figure (the paper copy make_rq_figures.py puts in this folder) across
both columns under them. A main figure taller than wide goes on a float page
of its own instead. The question and the findings are read from the README on
every run, so the pages follow each refresh; the title, the setup, the figure
and its description are fixed below (they name no counts that a refresh
moves). Every caption starts with the figure's takeaway in bold: the lead of
the first finding for the main figure, a fixed sentence for the extras.

    question     the README's **Question.** paragraph or the first paragraph of
                 its "Research question" / "Question" section (or, failing that,
                 the first paragraph), cut after its last "?"; without one, the
                 question in the title line
    findings     an rqfinding box (defined in main.tex) of at most three
                 numbered findings ("1. " inside the bold lead), separated by two
                 \\\\ lines, each a bold lead sentence and its elaboration: the opening key finding, i.e. the **Key finding.**
                 paragraph if the README has one, else the bold lead and first
                 sentence of a bullet after the figure's image in the README
                 (the folder's first image when the figure is not drawn there),
                 then the next Key-findings bullets (the rest of that list, or
                 the first "Key findings" list after the paragraph), each cut
                 to its bold lead and first sentence
    extras       optional sixth PAGES element: further figures after the main
                 one, each (paper figure stem, caption in LaTeX starting with
                 its bold takeaway, label), on a full-width float page of their own;
                 the caption may be a function of the figure's CSV, so its
                 numbers follow each refresh

A FIGURE_PAGES entry writes sections/<stem>.tex, one more figure on a float page
of its own, with a caption built from the figure's CSV on every run:
app_rq02_bbpb, the main-text rq2 figure split into an accuracy and a bBPB row.
FIXED writes sections/app_fixed_populations.tex, the fixed-task twins (one task
set per line, rule 13) of the paper figures whose task set moves along x, each
on a float page with a caption read from its CSV. A page whose main figure only
the cluster run draws (make_rq_figures.PENDING) is skipped until it is drawn.

The text is double-blind: links, file names and rule numbers are dropped.
plain() gives every written sentence the appendix punctuation: no dash and no
";" outside math and code. A numeric range keeps its en dash (90M--1.7B).
The paper writes the number of languages as K where the analysis writes L
(L8, "(task, L)"); to_tex renames it outside code spans, the README keeps L.

    python make_rq_appendix.py
"""
import re
import struct
import sys
from pathlib import Path

import pandas as pd

from make_rq_figures import ANALYSIS, FIGURES, HERE

SECTIONS = HERE.parent / "sections"
# the appendix section the pages are subsections of, opened by the first page
PARENT = ("Detailed Analyses", "app:analyses")
# rq12 scores DA against both 1.7B truths once its per-item store run has written the full-truth columns
_RQ12_CSV = ANALYSIS / "rq12_above_chance_items" / "pretraining" / "predictivity" / "above_chance_items_snr_paper.csv"
RQ12_TWO_TRUTHS = _RQ12_CSV.is_file() and "rho_snr_da_size_full_truth" in _RQ12_CSV.read_text().split("\n", 1)[0].split(",")
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"]


def readings(d, solid, dashed, value="da", **where):
    """A paper figure's CSV `d` with two readings of each line, one row per
    point: the two values and their difference (solid - dashed)."""
    for k, v in where.items():
        d = d[d[k] == v]
    keys = [c for c in ("panel", "population", "line", "x") if c in d and c not in where]
    w = d.pivot_table(index=keys, columns="reading", values=value).dropna(subset=[solid, dashed])
    return w.assign(diff=w[solid] - w[dashed]).reset_index()


def rq02_crossfit_caption(d):
    """The caption of app_rq02_rq2_crossfit, its numbers read from the figure's CSV `d`."""
    rq2 = readings(d, "cross-fitted", "in-sample (half)", panel="DA-size", line="all pairs")
    worst = rq2.loc[rq2["diff"].abs().idxmax()]
    return (
        f"\\textbf{{Chosen out of sample, the reliable tasks move DA-size over every single-axis pair by at most "
        f"{abs(worst['diff']):.2f} (at {worst['x']}).}} Figure~\\ref{{fig:rq2}} with the reliable tasks chosen out of "
        "sample. The verdict (DA above 0.66 on the size or the checkpoint axis) is decided on one half of the design "
        "pairs and the panels are read on the other half, over 20 random splits stratified by the axes a pair moves, in "
        "both directions (solid, with the 5--95\\% range over splits). The dashed lines choose and read the tasks on the "
        "same half, so the gap between the two is the selection effect alone. Lines with too few pairs per half are "
        "not drawn.")


def rq02_da_crossfit_caption(d):
    """The caption of app_rq02_decision_accuracy_crossfit, its numbers read from the figure's CSV `d`."""
    twin = readings(d, "cross-fitted", "in-sample (half)", population="all benchmarks")
    return (
        f"\\textbf{{On the multi-axis pairs, the reliable benchmark tasks chosen out of sample and in sample give DA-size "
        f"within {twin['diff'].abs().max():.2f} of each other.}} Figure~\\ref{{fig:app_rq02_decision_accuracy}} with the "
        "reliable benchmark tasks, chosen out of sample (solid) and in sample (dashed) on the same half of the pairs, "
        "over the splits of Figure~\\ref{fig:app_rq02_rq2_crossfit}.")


def fixed_impact(d, line, what, fmt=".2f", value="value", unit="tasks", **where):
    """What one task set per line does to a line of a paper figure, from the twin's
    CSV `d`: (a bold lead with the largest shift (fixed - moving) along the line
    and the verdict, a sentence with its fixed and moving values at the first and
    last point and the task counts)."""
    for k, v in dict(where, line=line).items():
        d = d[d[k] == v]
    w = d.pivot_table(index="x", columns="reading", values=[value, "n_tasks"])
    w = w.loc[sorted(w.index, key=lambda x: SIZES.index(x) if x in SIZES else float(x))].dropna()
    f, m, n = w[(value, "fixed")], w[(value, "moving")], w[("n_tasks", "moving")]
    diff = f - m
    x = diff.abs().idxmax()
    first, last = w.index[0], w.index[-1]
    num = lambda v, sign="": f"{v:{sign}{fmt}}".replace("-", "−")      # a minus sign, not a hyphen
    verdict = ("so part of the line's movement comes from its changing tasks" if abs(diff[x]) > (m.max() - m.min()) / 4
               else "small against the line's own range, so its shape is not a population effect")
    return (f"Holding the tasks fixed moves {what} by at most {num(diff[x], '+')} (at {x}), {verdict}.",
            f"The {int(w[('n_tasks', 'fixed')].iloc[0])} {unit} it has at every point read {num(f.iloc[0])} at {first} "
            f"and {num(f.iloc[-1])} at {last}, against {num(m.iloc[0])} and {num(m.iloc[-1])} on the {unit} of each "
            f"point ({int(n.min())}--{int(n.max())}).")


# fixed-task twin stem -> (the label of the figure it is a twin of, what the twin changes, fixed_impact's arguments)
FIXED = {
    "app_rq02_fixed_rq2": ("fig:rq2", "the reliable tasks of each panel",
                           dict(line="all pairs", what="the DA-size line over every single-axis pair", value="da",
                                panel="DA-size")),
    "app_rq02_fixed_decision_accuracy": ("fig:app_rq02_decision_accuracy", "the tasks above chance",
                                         dict(line="all benchmarks", what="the benchmark line", value="da")),
    "app_rq05_fixed_design_decisions": ("fig:app_rq05_design_decisions", "the (task, language setting) cells of each line",
                                        dict(line="temperature / benchmark",
                                             what="the benchmark line of the temperature decision", unit="cells",
                                             panel="da_size")),
    "app_rq10_fixed_size_generalisation": ("fig:app_rq10_size_generalisation",
                                           "the tasks of each channel (the left panel is not redrawn)",
                                           dict(line="benchmarks to 3B", what="the benchmark line against the 3B reference",
                                                panel="3B")),
    "app_rq11_fixed_evaluation_recipe": ("fig:app_rq11_evaluation_recipe", "the tasks of each variant",
                                         dict(line="All variants", what="the pooled line")),
    "app_rq12_fixed_above_chance_items": ("fig:app_rq12_above_chance_items", "the tasks of each ordering",
                                          dict(line="items_then_gate", what="the median SNR of the items-then-gate ordering",
                                               panel="snr_median")),
    "app_rq13_fixed_english_only": ("fig:app_rq13_english_only", "the English tasks of each comparison",
                                    dict(line="every L > 1",
                                         what="the deep cells' mean gap in points to every other language setting",
                                         fmt=".1f", panel="deep: mean gap")),
}


def fixed_caption(ref, tasks, kw):
    """The caption of a fixed-task twin, a function of its CSV."""
    def caption(d):
        lead, rest = fixed_impact(d, **kw)
        return (f"\\textbf{{{escape(lead)}}} Figure~\\ref{{{ref}}} on one task set per line, {tasks} that have a value "
                f"at every point (solid), against the original (dashed). {escape(rest)}")
    return caption


def fixed_populations():
    """The appendix of the fixed-task twins: one float page per paper figure whose task set moves along x."""
    return "\n".join([
        "\\clearpage", "\\section{Fixed task sets}", "\\label{app:fixed_populations}", "",
        plain("A line of a mean over tasks against model size averages, at each size, the tasks that have a value "
              "there. The above-chance gate keeps more tasks at larger sizes, so a line can rise or fall because "
              "its tasks changed. Each figure below redraws a paper figure on one task set per line, the tasks "
              "that have a value at every point of that line (solid), with the original line dashed behind it."),
        "", *(float_page(stem, fixed_caption(ref, tasks, kw), f"fig:{stem}") for stem, (ref, tasks, kw) in FIXED.items()
              if FIGURES[stem][0].with_suffix(".csv").is_file())])


def rq11_benchmarks_caption(d):
    """The caption of app_rq11_recipe_by_benchmark, its numbers read from the figure's CSV `d`."""
    drawn = d[d["drawn"] & (d["panel"] != "ranking")]
    size = drawn[drawn["panel"] == "DA-size"].pivot_table(index=["row", "x"], columns="line", values="da")
    rows = [r for r in dict.fromkeys(drawn["row"]) if r != "Overall"]
    acc = [c for c in ("original", "rf", "rfgm") if c in size]
    small = size.xs("90M", level="x")
    both = small.dropna(subset=["bbpb_acc"])[small.dropna(subset=["bbpb_acc"])[acc].notna().any(axis=1)]
    wins = int((both["bbpb_acc"] > both[acc].max(axis=1)).sum())
    ov = size.xs("Overall", level="row")
    no_twin = [r for r in rows if not drawn[(drawn["row"] == r) & drawn["line"].str.startswith("bbpb")].shape[0]]
    n = len(rows)
    return (
        f"\\textbf{{At the 90M proxy, a benchmark's bBPB read against the 1.7B accuracy ranks the design pairs more like "
        f"the reference than its best accuracy format on {wins} of the {len(both)} top benchmarks where both have a "
        f"value.}} Overall, DA-size on accuracy goes from {ov.at['90M', 'acc']:.2f} at 90M to {ov.at['1B', 'acc']:.2f} at "
        f"1B, against {ov['bbpb_bbpb'].min():.2f}--{ov['bbpb_bbpb'].max():.2f} for the bBPB twins (read against the 1.7B "
        f"bBPB) and {ov['bpb'].min():.2f}--{ov['bpb'].max():.2f} for BPB. The other rows are the {n} benchmarks with the "
        "highest mean DA-size over 90M--1B in their original or RF format (at least five tasks), ranked, with that "
        "format and value after the name. "
        "Columns: DA-size against the 1.7B final checkpoint, DA-ckpt against the proxy's own final checkpoint and "
        "DA-goal against the 1.7B final checkpoint, the last two averaged over the proxies 90M--1B. All use the "
        "multi-axis pairs of the seed-1904 runs, and accuracy is gated above chance. bBPB uses the twins of the "
        "ranking format, read against the 1.7B accuracy of the same items or against the 1.7B bBPB. The reading "
        "against accuracy has no DA-ckpt, as its reference would be another score. "
        + (f"{' and '.join(no_twin)} {'has' if len(no_twin) == 1 else 'have'} no twin. " if no_twin else "")
        + "The bands are 95\\% bootstrap intervals over tasks. A point on fewer than five tasks is not drawn. A hollow "
        "point is 1.0 by self comparison.")


def _rq02_maps(d):
    """What the per-language maps share: the drawn cells and the setup sentences (the filter read from the CSV)."""
    rest = d[~d["drawn"]].drop_duplicates("benchmark")
    single, empty = (sorted(rest.loc[rest["n_languages"] < 2, "benchmark"]),
                     sorted(rest.loc[rest["n_languages"] >= 2, "benchmark"]))
    filt = d["filter"].iloc[0]
    cut = ("Only the tasks whose median DA-size or median DA-ckpt is at least 0.66 are kept, as there. This "
           "filter selects on DA, so the maps are conditional on it." if filt != "none" else "There is no reliability "
           "filter.")
    setup = (
        "A cell is the mean over a benchmark's original accuracy tasks (no RF, LLM-RF or bBPB twin) in a trained "
        f"language, on the mono-axis pairs of Figure~\\ref{{fig:rq2}}. {cut} Grey cells are gated, white cells have "
        "no value, and the scale is centred at 0.5 (a coin flip). Languages follow resource rank, with lines where "
        f"the lists of $K$ = 1, 2, 8, 15, 30 and 50 languages end. Not drawn: {len(single)} single-language "
        "benchmarks" + (f", and {', '.join(empty)}, which have no value." if empty else "."))
    lead = "On the tasks that rank reliably, {}" if filt != "none" else "{}"
    return d[d["drawn"]], setup, lambda t: lead.format(t if filt != "none" else t[0].upper() + t[1:])


def rq02_language_caption(d):
    """The caption of app_rq02_da_by_language, its numbers read from the figure's CSV `d`."""
    drawn, setup, lead = _rq02_maps(d)
    clear = {k: (g.drop_duplicates("family")["benchmark_lo"].gt(0.5).sum(),
                 g.drop_duplicates("family")["benchmark_mean"].notna().sum(),
                 g.drop_duplicates("language")["language_lo"].gt(0.5).sum(),
                 g.drop_duplicates("language")["language_mean"].notna().sum(), g["da"].mean())
             for k, g in drawn.groupby("kind")}
    (sb, nb, sl, nl, sm), (cb, cnb, cl, cnl, cm), (gb, gnb, gl, gnl, gm) = (clear[k] for k in ("DA-size", "DA-ckpt", "DA-goal"))
    return (
        "\\textbf{" + lead(f"averaged over the proxies 90M--1B, DA-size is above 0.5 (95\\% interval) on {sb} of "
                            f"{nb} benchmarks and {sl} of {nl} languages.") + "} "
        f"DA-goal clears 0.5 on {gb} of {gnb} benchmarks and {gl} of {gnl} languages, DA-ckpt on {cb} of {cnb} and {cl} of {cnl}. Cell means: {sm:.2f}, {cm:.2f} and {gm:.2f}. Top: "
        "DA-size of each proxy's final checkpoint against the 1.7B final. Middle: DA-ckpt of the nine earlier tenths "
        "of each run against its own final, over the proxies and the 1.7B run, where DA-ckpt is defined too. Bottom: "
        "DA-goal of the ten tenths of each proxy against the 1.7B final. DA-size and DA-goal stop at 1B because 1.7B "
        "is their reference. Each task is first averaged over these readings. Side panels: row and column means with "
        "95\\% bootstrap intervals over the cells. " + setup)


def rq02_language_per_proxy_caption(d):
    """The caption of app_rq02_da_size_by_language_per_proxy, its numbers read from the figure's CSV `d` and,
    for the contrast, from the same figure without the filter."""
    drawn, setup, lead = _rq02_maps(d)

    def per(x):
        return x.groupby("proxy_size").agg(cells=("da", "count"), mean=("da", "mean"), above=("da", lambda v: v.gt(0.5).mean()))
    p = per(drawn)
    hl = drawn[drawn["highlight"]].iloc[0]
    out = ("\\textbf{" + lead(
        f"a larger proxy decides more like the 1.7B reference: {p.at['90M', 'above']:.0%} of the cells have DA-size "
        f"above 0.5 at 90M and {p.at['1B', 'above']:.0%} at 1B, and {hl['benchmark']} (outlined) is above 0.5 at every "
        f"proxy in {int(hl['steady_languages'])} of its {int(hl['complete_languages'])} languages.") + "} "
           f"The mean goes from {p.at['90M', 'mean']:.2f} to "
           f"{p.at['1B', 'mean']:.2f}. ").replace("%", "\\%")
    if d["filter"].iloc[0] != "none":
        src = FIGURES["app_rq02_da_size_by_language_per_proxy"][0]
        u = per(pd.read_csv(src.with_name(src.name.replace(f"_{d['filter'].iloc[0]}", "")).with_suffix(".csv"))
                .query("drawn"))
        out += f"Without the filter it goes only from {u.at['90M', 'mean']:.2f} to {u.at['1B', 'mean']:.2f}. "
    return out + ("DA-size of each proxy's final checkpoint against the 1.7B final, one map per proxy, the row means "
                  "last. Only benchmarks with a DA-size at some proxy are drawn. " + setup)


def rq13_regimes_caption(d):
    """The caption of app_rq13_english_scaling_regimes, its numbers read from the figure's CSV `d`."""
    pct = lambda v: f"{100 * v:.0f}\\%"
    d = d.assign(b=d["regime"] == "predictable across both", acc=~d["family"].str.startswith(("bbpb", "bpb")))
    both = d.groupby("corpus")["b"].agg(["mean", "size"])
    acc = d[d["acc"]].groupby("corpus")["b"].agg(["mean", "size"])
    dclm = both.loc[["DCLM-Edu", "DCLM"], "mean"]
    back = "every K (rq01)"
    return (
        f"\\textbf{{At $K$ = 1 the FineWeb cell puts {pct(both.at['FineWeb', 'mean'])} of the English tasks in the "
        f"regime predictable across both size and training, against {pct(dclm.min())}--{pct(dclm.max())} for the two "
        "DCLM corpora.} The scaling regimes of Figure~\\ref{fig:rq1} drawn for the monolingual English cells, one "
        "column per English corpus: DCLM-Edu (the English of every other cell), DCLM without the educational filter, "
        f"and FineWeb. Each is one deep cell at seed 1904, 90M--1.7B, on the same {int(both.at['FineWeb', 'size'])} "
        "English tasks (originals, RF, LLM-RF, bBPB twins and the English bits per byte). The lead is in the bBPB "
        f"twins: on the {int(acc.at['FineWeb', 'size'])} accuracy tasks alone the three corpora reach "
        + ", ".join(f"{pct(acc.at[c, 'mean'])} ({c})" for c in ("DCLM-Edu", "DCLM", "FineWeb")) + ". "
        "The fits and the above-chance gate are those of Figure~\\ref{fig:rq1}. With one $K$, a task has one "
        "model-size fit instead of a median over the six $K$, and the trajectory $R^2$ is the median over at least two "
        "sizes. Top: the $R^2$ of the model-size fit against its Spearman $\\rho$. Bottom: the same $R^2$ against the "
        "median $R^2$ of the training-trajectory fits, shaded by regime at 0.5. A black edge marks a task whose score "
        f"declines with size. The grey points are the {int(both.at[back, 'size'])} tasks of Figure~\\ref{{fig:rq1}} "
        f"(medians over $K$, {pct(both.at[back, 'mean'])} predictable across both), so they sit higher than a single "
        "fit would.")


def rq06_cross_task_caption(d):
    """The caption of app_rq06_cross_task_hellaswag, its numbers read from the figure's CSV `d`."""
    lang = lambda t: "en" if t == "hellaswag" else t.rsplit("_", 1)[-1]
    d = d.assign(pl=d["proxy"].map(lang), tl=d["target"].map(lang))
    off, diag = d[d["pl"] != d["tl"]], d[d["pl"] == d["tl"]].set_index("pl")["da_size"]
    eu4 = ["de", "es", "fr", "it"]
    block = off[off["pl"].isin(eu4) & off["tl"].isin(eu4)]["da_size"].mean()
    return (
        f"\\textbf{{On HellaSwag, German, Spanish, French and Italian, which the $K$ = 8 list trains together, predict "
        f"each other's 1.7B ranking at {block:.2f}, as well as each predicts its own ({diag[eu4].mean():.2f}), while English "
        f"ranks the other languages at {off[off['pl'] == 'en']['da_size'].mean():.2f}.}} Each cell is the DA-size of the "
        "ranking of the design variants on the HellaSwag task of one language at a proxy size (rows) against the ranking "
        "on the task of another language at the 1.7B final checkpoint (columns), averaged over the proxies 90M--1B. The "
        "diagonal is the task's own DA-size. The pairs are the mono-axis design pairs of the seed-1904 runs, at least "
        "three per size, each task gated above chance at the size it is read at. A model's score on a language it does "
        "not train counts here: the scheme-A runs are evaluated on every language, so a language that no list trains, "
        f"such as Basque (its column averages {off[off['tl'] == 'eu']['da_size'].mean():.2f}), is read on those runs "
        "alone. A cell averages the pairs both tasks have, so two cells may average different decisions. The colour is "
        "centred at 0.5, the agreement of a coin flip. Languages follow the resource order of the training lists.")


def rq06_family_caption(d):
    """The caption of app_rq06_family_transfer, its numbers read from the figure's CSV `d`."""
    s = d[(d["cell"] == "L8 A") & (d["level"] == "sub")].pivot(index="size", columns="group", values="lift")
    best = s["same family"].idxmax()
    return (
        f"\\textbf{{The $K$ = 8 cell clears chance on more of its untrained same-subfamily tasks than English only does on "
        f"the same tasks, by up to {s.at[best, 'same family']:+.2f} at {best}, and "
        + ("on the other subfamilies it does not" if s["other family"].max() < 0.005 else
           f"on the other subfamilies by no more than {s['other family'].max():+.2f}") + ".} Each panel is one cell (rows) and one family level (columns). A line is "
        "the share of the cell's accuracy tasks in untrained languages whose final run clears chance, split by whether "
        "the language shares the family of a trained non-English language (left: the top-level family such as "
        "Indo-European, right: the subfamily such as Slavic). The dotted lines are the English-only $K$ = 1 cell on "
        "exactly the same tasks, so the gap between a solid line and its dotted twin is what training the cell's "
        "languages adds. Deep runs at seed 1904. Only the scheme-A cells and, at 350M--1B, the $K$ = 8 cell of scheme B "
        "are evaluated on untrained languages, and the $K$ = 1 cell has no 350M evaluation, so its dotted lines break "
        "there. Languages without a family in our language table are left out.")


def rq06_lift_caption(d):
    """The caption of app_rq06_family_lift, its numbers read from the figure's CSV `d`."""
    r = d[(d["cell"] == "L8 A") & (d["family_sub"] == "Indo-European, Romance")].iloc[0]
    rb = d[(d["cell"] == "L8 B") & (d["family_sub"] == "Indo-European, Romance")].iloc[0]
    trained = d[d["same_sub"]].sort_values("lift", ascending=False)
    return (
        f"\\textbf{{Training Spanish, French and Italian lifts the untrained Romance languages most: the $K$ = 8 scheme-A "
        f"cell puts {100 * r['share_above']:.0f}\\% of their tasks above chance against {100 * r['baseline_share_above']:.0f}\\% "
        f"for English only, while the scheme-B list, which has no Romance language, gains {rb['lift']:+.2f}.}} Each row is "
        "one subfamily of the untrained languages. A dot is the share of a cell's tasks in those languages above chance "
        "minus the share of the English-only $K$ = 1 cell on the same tasks and sizes, pooled over the sizes both have "
        "(every size but 350M for scheme A, 600M and 1B for scheme B). A filled dot marks a subfamily of one of the "
        "cell's trained languages: " + ", ".join(
            f"{f} for $K$ = {c.split()[0][1:]} scheme {c.split()[1]} ({v:+.2f})"
            for c, f, v in zip(trained["cell"], trained["family_sub"].str.replace("Indo-European, ", ""), trained["lift"]))
        + ". "
        "Rows follow their mean over the cells.")


RQ09_POP = ("the per-task DA-size of Figure~\\ref{fig:rq2}: the final checkpoint of each proxy against the 1.7B final, "
            "on the mono-axis design pairs of the seed-1904 runs, on the tasks above chance at the proxy and at 1.7B. A "
            "benchmark's DA-size is the median over its tasks at a proxy, averaged over the proxies 90M--1B")


def rq09_correlation_caption(d):
    """The caption of app_rq09_design_da_correlation, its numbers read from the figure's CSV `d`."""
    v = d.set_index(["reading", "characteristic"])
    signed = d[d["statistic"] == "spearman_rho"]
    crosses = lambda r: bool(((signed["reading"] == r) & (signed["ci_low"] <= 0) & (signed["ci_high"] >= 0)).sum()
                             == (signed["reading"] == r).sum())
    snr, rel, unf = (v.loc[(r, "n_options")] for r in ("snr", "above_66_either", "unfiltered"))
    return (
        "\\textbf{No characteristic orders the benchmarks by DA-size"
        + (": every signed rank correlation has a 95\\% interval that contains zero" if crosses("above_66_either")
           and crosses("unfiltered") else "")
        + f".}} The answer options, which correlate with SNR at $\\rho = {snr['value']:.2f}$ "
        f"(${snr['ci_low']:.2f}$ to ${snr['ci_high']:.2f}$), reach ${rel['value']:.2f}$ on the reliable tasks and "
        f"${unf['value']:.2f}$ on every task above chance. A row is one characteristic of the benchmark families (Table "
        "\\ref{tab:app_rq09_benchmark_characteristics}). For the answer options, the reading passage, the source and the "
        "two lengths the value is a Spearman $\\rho$ over the benchmarks against the characteristic's code (the options "
        "ranked 2 to 4, no passage to passage, translated to native). For curation and task format, which have no order, "
        "it is the rank correlation ratio $\\eta$ from the Kruskal-Wallis $H$, which has no sign and is always positive. "
        f"DA-size is {RQ09_POP}. Left: that mean with a 95\\% bootstrap interval over the "
        f"{int(rel['n_benchmarks'])} benchmarks whose tasks pass the reliability filter of Figure~\\ref{{fig:rq2}} "
        f"(filled), over the {int(unf['n_benchmarks'])} benchmarks with any task above chance (hollow), and the same "
        f"statistics on the median SNR at 1.7B of the {int(snr['n_benchmarks'])} benchmarks (diamonds). Right: one line per "
        "proxy size, on the reliable tasks. The lengths exist for the original benchmarks with sampled items only.")


def rq09_levels_caption(d):
    """The caption of app_rq09_design_da_by_level, its numbers read from the figure's CSV `d`."""
    top = d.loc[d["da_size"].idxmax()]
    n = d["family"].nunique()
    return (
        f"\\textbf{{{top['name']} has the highest DA-size of the {n} benchmarks ({top['da_size']:.2f}), so the levels "
        f"it carries ({', '.join(d[d['family'] == top['family']]['level'])}) are lifted by one benchmark.}} "
        "Each row is one level of one characteristic, coloured by characteristic, with the number of benchmarks at that "
        "level. A dot is one benchmark at its DA-size minus 0.5, so the dotted line is chance agreement and every "
        f"benchmark sits to its right. The bar is the level's median. DA-size is {RQ09_POP}, on the tasks that pass "
        f"the reliability filter of Figure~\\ref{{fig:rq2}}. {top['name']} is outlined.")


def rq09_quadrant_caption(d):
    """The caption of app_rq09_design_da_quadrant, its numbers read from the figure's CSV `d`."""
    dots, cnt = d[(d["row"] == "benchmark") & (d["population"] == "above_66_either")], d[d["row"] == "quadrant count"]
    left = cnt[cnt["x_side"] < 0].groupby("characteristic")[["n_benchmarks", "n_benchmarks_unfiltered"]].sum()
    return (
        f"\\textbf{{On the reliable tasks all {dots['family'].nunique()} benchmarks decide above chance agreement, so "
        "the left quadrants stay empty.} On every task above chance "
        f"{int(left['n_benchmarks_unfiltered'].max())} benchmarks fall below 0.5. "
        "One panel per characteristic with an order: x is the DA-size minus 0.5 and y the characteristic's code centred "
        "at 0 (the options ranked 2, 3, 4 as $-1$, 0, $+1$, a binary characteristic as $-1$ and $+1$). Filled dots read "
        "the tasks that pass the reliability filter of Figure~\\ref{fig:rq2}, hollow dots every task above chance "
        "(drawn a little lower). The numbers count the benchmarks per quadrant, reliable (every task). A benchmark on a "
        "zero line, such as the three-option XNLI, is in no quadrant. The shaded quadrants are the ones a positive "
        f"correlation fills. DA-size is {RQ09_POP}.")


def rq09_table_caption(d):
    """The caption of app_rq09_benchmark_characteristics, its numbers read from the table's CSV `d`."""
    return (
        f"\\textbf{{The {len(d)} benchmark families of the benchmark design analysis and the characteristics it "
        f"groups them by.}} {int((d['source_origin'] == 'english_translated').sum())} are translated from English and "
        f"{int((d['n_options'] == 4).sum())} have four answer options. Curation: how the items in each language were made "
        "(native: written in the language, human transl.: translated by people, MT: machine translated, MT + post-edit: "
        "machine translated and corrected, template: generated from treebank templates). Format: MCQ is a question with "
        "lettered options, cloze (RF) scores each option as a continuation, statement (LLM-RF) is the LLM rewrite into a "
        "statement with four continuations. Context and Option: the median characters of the context and of an option "
        "over 100 sampled English items.")


# folder -> title, paper figure stem, setup, (README image the finding follows, which bullet)[, extras]
PAGES = {
    "rq00_gate_and_curves": (
        "The above-chance gate", "app_rq00_chance_share_horizontal",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. A run clears chance on a "
        "task when the one-sided 95\\% Wilson lower bound of its accuracy over the task's items is above the chance "
        "level. A (task, size) cell is above chance when at least half of the runs of that size that train the task's "
        "language clear it. Tasks without a chance level (bits per byte) are not gated.",
        None, 0),
    "rq00_task_reformulation": (
        "Task reformulation", "app_rq00_chance_reformulation",
        "We compare the letter-format multiple-choice benchmarks (Belebele, Global-MMLU, INCLUDE) with two "
        "reformulated twins of every task. The first twin (RF) drops the answer letters and scores each option as a "
        "continuation. The second twin (LLM-RF) uses an LLM to rewrite the items as cloze statements. We use the same "
        "above-chance gate and the same runs as for the gate itself. We pair each original task with its twin and "
        "compare the two with a McNemar test.",
        None, 1),
    "rq00_chance_vs_train_tokens": (
        "Above chance against the tokens of the language seen", "app_rq00_chance_vs_train_tokens",
        "A cell is one (task, size, language setting) read at each of the ten evaluated tenths of the run. We place "
        "it at the number of tokens of the task's language that the checkpoint had seen. This number is the "
        "language's share of the mixture $\\times$ the size's budget $\\times$ the tenth. We use seed-1904 runs of "
        "every data build and ladder, 90M--1.7B, at $K \\in \\{1, 2, 8, 15, 30, 50\\}$. The above-chance gate "
        "is applied per run.",
        None, 0),
    "rq01_scaling_predictability": (
        "Scaling predictability", "rq1",
        "We fit the final score of each task against model size with a log-linear fit, one series per task and "
        "language setting. We fit only on the sizes where the task is above chance, and we need at least three of "
        "them. We use the deep cells with the baseline data at seed 1904, 90M--1.7B. Each task is one point, "
        "the median over its language settings.",
        None, 0,
        (("app_rq00_benchmark_size_curves",
          "\\textbf{The reformulated twins rise with model size while their letter-format originals stay at "
          "chance up to 1.7B.} Benchmark accuracy against model size. For each benchmark family, we plot the final-checkpoint accuracy "
          "of every design against non-embedding parameters. This accuracy is the mean over the tasks in the "
          "languages the design trains on. There is one line per (number of languages, ladder, data build, seed), "
          "90M--1.7B, for every seed and data build. Colour gives the number of trained languages, line "
          "width the ladder and line style the data build. The dotted red line is chance. The curves are not "
          "gated, so a family at chance stays visible.",
          "fig:app_size_scaling"),
         ("app_rq00_benchmark_curves",
          "\\textbf{Along training the letter-format originals stay at chance for the whole run while their "
          "reformulated twins climb steadily.} Benchmark accuracy along training. For each benchmark family, we plot the accuracy of every run at the "
          "ten evaluated tenths of the run. This accuracy is the mean over the tasks in the languages the run trains "
          "on. Training progress is given in Chinchilla multiples of the run's token budget. The figure covers "
          "every seed and data build, 90M--1.7B. Colour gives the model size, line width the ladder and "
          "line style the data build. The dotted red line is chance. The curves are not gated.",
          "fig:app_training_scaling"))),
    "rq02_decision_accuracy": (
        "Decision accuracy across sizes", "app_rq02_decision_accuracy",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. DA-size is the share of "
        "design pairs that the final checkpoint of a proxy orders in the same way as the final checkpoint of the "
        "1.7B reference. Every task is above chance at the proxy and at 1.7B. A pair may differ on any number of "
        "design axes.",
        None, 0,
        (("app_rq02_da_by_language", rq02_language_caption, "fig:app_rq02_da_by_language"),
         ("app_rq02_da_size_by_language_per_proxy", rq02_language_per_proxy_caption,
          "fig:app_rq02_da_size_by_language_per_proxy"),
         ("app_rq02_rq2_crossfit", rq02_crossfit_caption, "fig:app_rq02_rq2_crossfit"),
         ("app_rq02_decision_accuracy_crossfit", rq02_da_crossfit_caption, "fig:app_rq02_decision_accuracy_crossfit"))),
    "rq02_permutation_null": (
        "Decision accuracy against a no-signal null", "app_rq02_permutation_null",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B, on pairs that may differ on any "
        "number of design axes. For every (task, proxy size) cell with at least three comparable pairs, we shuffle "
        "the proxy's final scores across the design variants 1000 times and score each shuffle against the unchanged "
        "1.7B ranking with the same sign rule, ties included. A cell's p-value is the share of shuffles that match at "
        "least as many pairs, corrected over the cells with the Benjamini-Hochberg procedure. The lines pool the "
        "matching pairs over the comparable pairs of the tasks above chance at the proxy and at 1.7B, and the band is "
        "the 95\\% range of the pooled shuffles. The benchmark line includes the bBPB variants at the sizes where "
        "they were computed (175M, 350M and 1B).",
        None, 0),
    "rq02_decisive_pairs": (
        "Decision accuracy on decisive pairs", "app_rq02_decisive_pairs",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B, on pairs that may differ on any "
        "number of design axes. A pair is decisive for a task when its gap at 1.7B exceeds $k \\sqrt{2}$ times the "
        "task's seed standard deviation, for $k \\in \\{1, 2\\}$. The seed standard deviation is the median, over the "
        "baseline cells trained with three seeds (175M, 600M and 1B), of the standard deviation of the final score. "
        "There are no replicates at 1.7B, so we assume that seed noise does not grow with size. Every line reads "
        "the same accuracy tasks: those with a seed standard deviation and at least three pairs above the larger "
        "threshold, above chance at the proxy and at 1.7B.",
        None, 0),
    "rq02_da_vs_train_tokens": (
        "Decision accuracy against the tokens of the language seen", "app_rq02_da_goal_multi_axes_bpb",
        "We use the bits per byte of 50 languages, one task each. Design pairs are formed among the variants that "
        "train the language (at least three per language). We use every data build at seed 1904, the proxies from "
        "90M--1B and the early checkpoints of the 1.7B run. A point is the mean over languages of DA-goal (a "
        "checkpoint against the 1.7B final checkpoint) at one size and one tenth of the run. We place it at the "
        "number of tokens of the language seen.",
        None, 0),
    "rq03_noise_and_snr": (
        "Noise and SNR", "app_rq03_noise_and_snr",
        "We use the deep cells with the baseline data that were trained with three seeds (175M and 600M at "
        "$K \\in \\{1, 2, 50\\}$, 1B at $K \\in \\{1, 2, 30\\}$). For each (size, $K$, task) cell, we divide the "
        "absolute effect of each design decision on the final score by a noise. The seed noise is the sample "
        "standard deviation across seeds. The checkpoint noise is the detrended standard deviation over the last "
        "20\\% of the run. Cells at chance are left out.",
        None, 0),
    "rq04_surrogates": (
        "SNR as a surrogate for decision accuracy", "app_rq04_surrogates",
        "Seed-1904 runs of every cell, 90M--1.7B, on the tasks above chance. For each of 22 SNR definitions, "
        "we compute the Pearson $r$ between $\\log_{10}$ SNR and decision accuracy over the tasks of one language. "
        "We keep the languages with at least three tasks and average over the 50 trained languages. We do this for "
        "DA-size (proxy final against 1.7B final) and for DA-ckpt (the early checkpoints of a proxy against its "
        "final checkpoint).",
        "highlights", 0),
    "rq05_design_decisions": (
        "Design decisions", "app_rq05_design_decisions",
        "We study four single-axis interventions against the baseline (deep, scheme A, $T = 1$). They are depth "
        "(deep vs shallow at equal non-embedding size), data scheme A vs B, data scheme A vs C, and sampling "
        "temperature $T = 1$ vs $T = 3$. We use every seed. We report the DA-size of the proxies 90M--1B and "
        "the DA-ckpt of the earlier checkpoints of the 1.7B run, both against the 1.7B final checkpoint. We read "
        "them on the bits per byte of the languages that both levels train and on the benchmarks above chance.",
        None, 0),
    "rq06_language_transfer": (
        "Language transfer", "app_rq06_language_transfer",
        "We read the language-list decision (scheme A vs B at $K \\in \\{8, 15, 30\\}$, single-axis pairs, every "
        "seed) on the bits per byte of 100 evaluation languages. We group the languages by whether both lists, one "
        "list or neither list trains them. When neither list trains a language, we also check whether they train "
        "its script. We report the DA-size of the proxies 90M--1B and the DA-ckpt of the checkpoints of the "
        "1.7B run, against the 1.7B final checkpoint.",
        None, 0,
        (("app_rq06_cross_task_hellaswag", rq06_cross_task_caption, "fig:app_rq06_cross_task_hellaswag"),
         ("app_rq06_family_transfer", rq06_family_caption, "fig:app_rq06_family_transfer"),
         ("app_rq06_family_lift", rq06_lift_caption, "fig:app_rq06_family_lift"))),
    "rq07_external_frameworks": (
        "Agreement with DataDecide", "app_rq07_external_frameworks",
        "We compare our 1B rung (seed 1904, every cell and data build) with the 1B rung of DataDecide (25 data "
        "recipes). We use the English tasks that both evaluate and that clear the above-chance gate on our side. "
        "SNR is the average absolute deviation over noise, and we compare it on a log scale.",
        None, 0),
    "rq08_subset_selection": (
        "Subset selection", "app_rq08_subset_selection",
        "Seed-1904 runs of every cell, 90M--1.7B. For each multilingual benchmark and size, we take the "
        "language subset with the highest SNR, that is, the best prefix of the languages ranked by SNR. We compare "
        "it with the 95th percentile of 100 random subsets of the same size. Each per-language task is gated above "
        "chance.",
        None, 0),
    "rq09_benchmark_design": (
        "Benchmark design", "app_rq09_benchmark_design",
        "Seed-1904 runs of every cell. SNR is the mean pairwise distance over noise at the 1.7B reference. For each "
        "family, we take the median over its per-language tasks. We keep the benchmark families that clear the "
        "above-chance gate. We group the families by curation, source, task format, answer-option count and "
        "passage use, and we test the groups with a Kruskal-Wallis test.",
        None, 0,
        (("app_rq09_benchmark_characteristics", rq09_table_caption, "tab:app_rq09_benchmark_characteristics"),
         ("app_rq09_design_da_correlation", rq09_correlation_caption, "fig:app_rq09_design_da_correlation"),
         ("app_rq09_design_da_by_level", rq09_levels_caption, "fig:app_rq09_design_da_by_level"),
         ("app_rq09_design_da_quadrant", rq09_quadrant_caption, "fig:app_rq09_design_da_quadrant"))),
    "rq10_size_generalisation": (
        "Size generalisation to 3B", "app_rq10_size_generalisation",
        "The deep cells trained at 3B (seed 1904) and the same families at 1.7B. Each (family, task) is scored at "
        "both rungs. Left: for each benchmark, the share of its tasks above chance at 1.7B and at 3B. We show the "
        "benchmarks with at least five tasks whose share moves. Right: DA-size from the final checkpoints of 90M--1B "
        "to the 3B final checkpoint and to the 1.7B final checkpoint, on the same single-axis decisions. "
        "Benchmark accuracy (tasks above chance at the proxy, at 1.7B and at 3B) and per-language bits per byte "
        "are pooled separately. The bands are 90\\% leave-one-family-out jackknife bands.",
        None, 0),
    "rq11_evaluation_recipe": (
        "Evaluation recipe", "app_rq11_evaluation_recipe",
        "Seed-1904 runs of every cell. We report the DA-size of the proxies 90M--1B against the 1.7B final "
        "checkpoint, on multi-axis pairs. Each benchmark is read in up to six ways. The items are used as "
        "published, as RF or as LLM-RF, and each version is scored by accuracy or by the bits per byte of the gold "
        "answer (bBPB). The accuracy variants count on the tasks above chance at the proxy and at the reference. A "
        "bBPB variant has no chance level and is read against its original's 1.7B accuracy, so it counts where that "
        "accuracy is above chance at the reference. The pooled line holds both kinds.",
        None, 0,
        (("app_rq11_recipe_by_benchmark", rq11_benchmarks_caption, "fig:app_rq11_recipe_by_benchmark"),)),
    "rq12_above_chance_items": (
        "Above-chance items", "app_rq12_above_chance_items",
        "Seed-1904 runs of every cell, 90M--1.7B, final checkpoints. Every benchmark-language task keeps only "
        "the items that its 1.7B runs answer above chance, after the above-chance gate. We compare it with the full "
        "task. The selection reads the reference by design, so the gains are an upper bound, not a held-out "
        "estimate." + (" Decision accuracy is scored against two truths, the 1.7B ranking on the full task and on "
                       "the kept items, and the gap between them is the part of the gain that comes from re-scoring "
                       "the reference." if RQ12_TWO_TRUTHS else ""),
        None, 0),
    "rq13_english_only": (
        "English-only models", "app_rq13_english_only",
        "We compare the monolingual English cells ($K = 1$) with the baseline cells of the same depth at every "
        "other language setting. These other cells give half of their tokens to English. We use seed 1904, final "
        "checkpoints and sizes 90M--1.7B, on the English accuracy tasks above chance at each size.",
        None, 0,
        (("app_rq13_english_scaling_regimes", rq13_regimes_caption, "fig:app_rq13_english_scaling_regimes"),)),
    "rq14_proxy_item_selection": (
        "Item selection from the proxies alone", "app_rq14_proxy_item_selection",
        "Seed-1904 runs of every cell, final checkpoints, on pairs that may differ on any number of design axes. "
        "For each benchmark-language task, we split the design families into two halves. On one half, we rank the "
        "items by their discrimination at 600M and 1B: the correlation of an item's correctness with the run's task "
        "score, both centred within each size. We keep the top half of the items and score the DA-size of the "
        "proxies 90M--1B on the other half's pairs, against the 1.7B ranking on the full task. We then swap the "
        "halves and average. Random subsets of the same size, drawn from the same items, are the baseline. The "
        "selection never reads the 1.7B runs, so its DA is a held-out estimate that is available before the "
        "reference is trained. Tasks are above chance at the proxy and at 1.7B.",
        None, 0),
}

# what each main figure draws, after its bold takeaway (the README's alt text when a folder has none)
CAPTIONS = {
    "rq00_gate_and_curves":
        "One bar per benchmark gives the share of its language tasks that clear the above-chance gate, split by the "
        "smallest model size from which the gate holds (a darker blue is a larger size). Red marks the tasks that stay "
        "at chance up to 1.7B. The number in parentheses is the benchmark's number of language tasks. The benchmarks "
        "are ordered from the most to the fewest tasks above chance.",
    "rq00_task_reformulation":
        "For each letter-format benchmark (its number of tasks in parentheses), the share of tasks above the chance "
        "threshold for the original items (grey bars), for the RF version at each model size (blue bars, a darker blue "
        "is a larger size) and for the LLM-RF version (orange diamonds). A star marks a size at which the McNemar test "
        "between the original and its twin gives p < 0.05.",
    "rq00_chance_vs_train_tokens":
        "One panel per benchmark (its number of tasks in parentheses). Each point is the share of (language, K, "
        "checkpoint) cells above chance, binned by the training tokens of the task's language that the checkpoint had "
        "seen (log scale). There is one line per model size, a darker blue for a larger size.",
    "rq01_scaling_predictability":
        "Each point is one benchmark-language task. Left: the median Spearman ρ of the score with model size against "
        "the median R² of the log-linear fits across model size. Right: the median R² of the fits along training "
        "against the same x axis. The shading splits each axis at 0.5. Bold labels name each benchmark family at its "
        "median point (its number of tasks in parentheses) and small labels name the outlying tasks.",
    "rq02_decision_accuracy":
        "DA-size of the final checkpoint of each proxy size against the final checkpoint of the 1.7B reference, on "
        "multi-axis design pairs. Left: the benchmark tasks above chance. Right: the per-language bits per byte. The "
        "band is the jackknife interval, the dotted line marks 0.9 and the circle marks the smallest proxy above it. "
        "The dashed segment joins the 1B proxy to the reference itself.",
    "rq02_da_vs_train_tokens":
        "Mean DA-goal over the bits per byte of the 50 trained languages against the training tokens of the language "
        "that the checkpoint had seen (log scale). There is one line per proxy size, a darker blue for a larger size, "
        "and one point per evaluated tenth of the run. The bars give the uncertainty over languages and the dotted "
        "line marks 0.75.",
    "rq02_permutation_null":
        "DA-size of the final checkpoint of each proxy size against the 1.7B final checkpoint, on multi-axis design "
        "pairs, for the benchmark tasks (left) and the per-language bits per byte (middle). The black line is the "
        "observed DA-size, the dashed grey line the mean of the shuffled null and the grey band its 95% range. The "
        "dotted line marks 0.5. Right: the share of (task, proxy size) cells above their own null at q < 0.05.",
    "rq02_decisive_pairs":
        "DA-size of the final checkpoint of each proxy size against the 1.7B final checkpoint, on multi-axis design "
        "pairs, for the benchmark tasks (left) and the per-language bits per byte (middle). One line per threshold: "
        "every comparable pair (black) and the pairs whose 1.7B gap exceeds one or two seed standard deviations of a "
        "difference (dark and light blue). The dotted line marks 0.5. Right: the share of the comparable pairs each threshold keeps, "
        "solid for the bits per byte and dashed for the benchmarks.",
    "rq03_noise_and_snr":
        "Median absolute effect of each design decision over the noise, per model size, on the benchmarks, the "
        "per-language bits per byte, the macro average of the bits per byte and the training loss. Top row: over the "
        "seed noise. Bottom row: over the checkpoint noise. There is one column per decision. The dashed line at 1 is "
        "where the effect equals the noise (log scale).",
    "rq04_surrogates":
        "For each of the 22 SNR definitions, the mean over languages of the Pearson r between log₁₀ SNR and decision "
        "accuracy across the tasks of one language, for DA-size (blue) and DA-ckpt (orange). The definitions are "
        "ordered by their DA-size correlation.",
    "rq05_design_decisions":
        "Decision accuracy (mean over K) of each single-axis design decision. Left: the final checkpoint of each proxy "
        "size against the 1.7B final checkpoint (DA-size). Right: the earlier checkpoints of the 1.7B run against its "
        "final checkpoint (DA-ckpt), in Chinchilla multiples. Solid lines read the decision on the bits per byte of the "
        "trained languages and dashed lines on the benchmark tasks above chance. The dotted line marks 0.75.",
    "rq06_language_transfer":
        "Decision accuracy (mean over K) of the language-list decision (scheme A vs B), read on the bits per byte of "
        "four groups of languages: trained by both lists, trained by one list, not trained but written in a trained "
        "script, and not trained in an untrained script. Left: DA-size by proxy size. Right: DA-ckpt along the 1.7B "
        "run, in Chinchilla multiples. The dotted line marks 0.75.",
    "rq07_external_frameworks":
        "Each point is one English task that both our ladder and DataDecide evaluate and that clears the above-chance "
        "gate on our side. The x axis gives the SNR of our 1B rung and the y axis the SNR of the DataDecide 1B rung, "
        "both on a log10 scale.",
    "rq08_subset_selection":
        "For each benchmark (rows, accuracy and bBPB versions) and model size (columns), the SNR of the best language "
        "subset minus the 95th percentile of 100 random subsets of the same size. Blue means that the chosen subset "
        "beats the null. White cells have no value.",
    "rq09_benchmark_design":
        "Median SNR at 1.7B over the languages of each benchmark family that clears the above-chance gate, ranked. "
        "The colour gives the number of answer options.",
    "rq10_size_generalisation":
        "Left: for each benchmark with at least five tasks whose share moves, the share of its tasks above chance at "
        "1.7B (open circles) and at 3B (filled circles). Right: DA-size of the proxies 90M--1B against the 3B final "
        "checkpoint (solid) and against the 1.7B final checkpoint (dashed), on benchmark accuracy (black) and on "
        "per-language bits per byte (orange). The bands are 90% leave-one-family-out jackknife bands and the dotted "
        "line marks 0.5.",
    "rq11_evaluation_recipe":
        "Mean DA-size against the 1.7B accuracy for each way of reading a benchmark, by proxy size. The colour gives "
        "the item format (original, RF, LLM-RF) and the line style the score (solid for accuracy, dotted for bBPB). "
        "The dashed black line pools every variant and the dotted line marks 0.75.",
    "rq12_above_chance_items":
        "Left: median SNR over the tasks of the full benchmark (black), of the above-chance items selected after the "
        "gate (blue) and of the gate applied after the selection (orange), per model size. The numbers give the tasks "
        "behind each point. Middle: the median ratio of the selected items to the full benchmark for the SNR, the "
        "signal and the k-fold noise. Right: Spearman ρ between SNR and DA-size across tasks.",
    "rq14_proxy_item_selection":
        "Left: DA-size of the proxies against the 1.7B ranking on the full task, on multi-axis design pairs, for the "
        "full task (black), for random items of the same count (orange) and for the top half of the items chosen on "
        "the proxies (blue). "
        "Right: the gain of the chosen items over the random ones, one line per share of the items kept.",
    "rq13_english_only":
        "(a) and (b): the English-only cells (K = 1) minus the cells with K > 1 of the same size, in accuracy points "
        "above chance averaged over the English tasks above chance, for the deep and the shallow ladder. There is one "
        "line per K, the thick line pools every K > 1 and the dotted line is their median. (c): the share of (task, K) "
        "comparisons in which the English-only cell is ahead.",
}

UNICODE = {"≥": "$\\geq$", "≤": "$\\leq$", "×": "$\\times$", "−": "$-$", "→": "$\\rightarrow$",
           "↔": "$\\leftrightarrow$", "ρ": "$\\rho$", "τ": "$\\tau$", "α": "$\\alpha$", "Δ": "$\\Delta$",
           "σ": "$\\sigma$", "²": "$^2$", "±": "$\\pm$", "≈": "$\\approx$", "₁₀": "$_{10}$", "∈": "$\\in$",
           "·": "$\\cdot$", "–": "--", "—": "---", "…": "\\ldots{}", "’": "'", "“": "``", "”": "''",
           "⚠️": "", "⚠": "", " ": " "}
ESCAPE = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_", "$": "\\$",
          "{": "\\{", "}": "\\}", "<": "$<$", ">": "$>$", "|": "$|$",
          "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}


def escape(s):
    s = "".join(ESCAPE.get(c, c) for c in s)
    for k, v in UNICODE.items():
        s = s.replace(k, v)
    return s


def clean_md(s):
    """Drop what is internal or identifying: comments, links, file names, rule numbers."""
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"\[([^\]]*)\]\(([^)]*)\)",
               lambda m: "" if m[2].startswith(("http", "#")) or "`" in m[1] or "/" in m[1] else m[1], s)
    s = re.sub(r",?\s*`[^`]*\.(csv|py|png|md|json|tex)`", "", s)
    s = re.sub(r"\s*\((?:[^()]*\b(?:rules?\s+\d|RULES|TARGET_SIZE|the paper's)[^()]*)\)", "", s)
    s = re.sub(r";\s*the figure's[^()]*(?=\))", "", s)        # a README figure's numbers, not the paper's
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)
    s = re.sub(r"[,;]\s*\)", ")", s)
    return re.sub(r"\s+", " ", s).strip()


def to_tex(s):
    """Markdown inline (code, bold, italics, 10^k) to LaTeX, everything else escaped."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", clean_md(s))):
        if i % 2:
            out.append("\\texttt{" + escape(part) + "}")
            continue
        part = re.sub(r"\bL(?=\d+\b)|\bL\b(?![-'’])", "K", part)          # the language count is K in the paper
        part = re.sub(r"(\d+)\^(\d+)", lambda m: f"\x00{m[1]}\x01{m[2]}\x02", part)
        part = re.sub(r'"([^"]*)"', "\x03\\1\x04", part)
        part = escape(part).replace("\x03", "``").replace("\x04", "''")
        part = part.replace("\x00", "$").replace("\x01", "^{").replace("\x02", "}$")
        part = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", part)
        out.append(part)
    return "".join(out)


# math, code and cross-references keep their characters
PROTECT = re.compile(r"(?<!\\)\$[^$]*(?<!\\)\$|\\texttt\{[^}]*\}|\\(?:ref|label|cite[pt]?)\{[^}]*\}")
NUM = re.compile("[0-9\x10]")                                # a range end: a number, size, math or ref
LOWER_KEEP = {"p", "r", "n", "k", "rf", "rfgm", "vs"}       # names that stay lower case at a sentence start


def plain(tex):
    """The appendix punctuation: no dash and no ';' outside math and code. A numeric
    range a--b keeps its en dash, a word--word pair keeps one hyphen, a dash aside becomes a
    parenthesis, a single dash a comma, and ';' ends the sentence (inside parentheses it becomes a comma)."""
    kept = iter(PROTECT.findall(tex))
    s = PROTECT.sub("\x10", tex)
    s = re.sub(r"\s*---\s*([^.;]+?)\s*---\s*|\s+--\s+([^.;]+?)\s+--\s+",
               lambda m: f" ({m[1] or m[2]}) ", s)                 # a dash aside reads as a parenthesis
    s = re.sub(r"\s*---\s*|\s+--?\s+", ", ", s)
    s = re.sub(r"([^\s(]+)--([^\s,;.:)]+)",
               lambda m: f"{m[1]}--{m[2]}" if NUM.search(m[1]) and NUM.search(m[2]) else f"{m[1]}-{m[2]}", s)
    out, depth = [], 0
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        out.append(ch if ch != ";" else "," if depth > 0 else ".\x11")
    # the sentence a ';' used to join starts with a capital (names such as p or rf stay as they are)
    s = re.sub(r"\x11(\s*)([a-z]+)\b",
               lambda m: m[1] + (m[2] if m[2] in LOWER_KEEP else m[2].capitalize()), "".join(out))
    s = s.replace("\x11", "")
    s = re.sub(r",\s*([,.:)])", r"\1", s)
    s = re.sub(r"\(\s*,\s*", "(", s)
    s = re.sub(r"  +", " ", s)
    s = re.sub(r" ([,.])(?=\s|$)", r"\1", s)
    return re.sub("\x10", lambda m: next(kept), s)


def paragraphs(lines):
    """Blank-line separated paragraphs, blockquote markers stripped."""
    text = "\n".join(re.sub(r"^>\s?", "", l) for l in lines)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def question(lines):
    title = lines[0].lstrip("# ").strip()
    para = None
    for i, l in enumerate(lines):
        if l.startswith("**Question.**"):
            para = paragraphs(lines[i:])[0].replace("**Question.**", "")
            break
        if re.match(r"##\s+(Research question|Question)\s*$", l):
            para = paragraphs(lines[i + 1:])[0]
            break
    if para is None:
        para = paragraphs(lines[1:])[0]
    para = clean_md(para)
    if "?" not in para:                       # the question is in the title line
        para = re.split(r"\s+[—:-]\s+", title, maxsplit=1)[-1]
        para = para[0].upper() + para[1:]
    return para[: para.rfind("?") + 1]


def bullet_list(lines, start):
    """The first Markdown bullet list after line `start`, one string per bullet."""
    bullets, cur = [], None
    for l in lines[start + 1:]:
        if l.startswith("- "):
            cur = [l[2:]]
            bullets.append(cur)
        elif cur is not None and l.startswith("  ") and l.strip():
            cur.append(l.strip())
        elif l.startswith("## ") and bullets:
            break
        else:
            cur = None
    return [" ".join(b) for b in bullets]


def short(text):
    """A bullet's bold lead and first sentence."""
    m = re.match(r"(\*\*.+?\*\*)\s*(.*)", text)
    lead, rest = (m[1], m[2]) if m else ("", text)
    first = re.split(r"(?<=[\w%)\]])[.;]\s+(?=[A-Z])", rest, maxsplit=1)[0].rstrip(".")
    # a closing clause that points at the README's own tables, figures or bullets
    first = re.sub(r";[^;]*\b(?:above(?!\s+chance)|below|bullet)\b[^;]*$", "", first)
    return f"{lead} {first}." if first else lead


def key_findings(lines, image, n=3):
    """The opening key finding and the Key-findings bullets after it, at most n."""
    for i, l in enumerate(lines):
        m = re.match(r"\*\*Key finding(?: \([^)]*\))?\.\*\*\s*", l)
        if m:
            nxt = next((j for j in range(i + 1, len(lines))
                        if re.match(r"(#+\s*)?(\*\*)?Key findings\b", lines[j])), None)
            rest = bullet_list(lines, nxt) if nxt is not None else []
            return [paragraphs([l[m.end():]] + lines[i + 1:])[0]] + [short(b) for b in rest[: n - 1]]
    images = [i for i, l in enumerate(lines) if l.startswith("![")]
    start = next((i for i in images if re.search(rf"/{re.escape(image[0])}\.png\)", lines[i])), images[0])
    bullets = bullet_list(lines, start)
    return [short(b) for b in bullets[image[1]: image[1] + n]]


def sentence(s):
    """`s` as a sentence: a capital first word (names such as p or rf stay as they
    are) and a full stop."""
    s = re.sub(r"^([a-z]+)\b", lambda m: m[1] if m[1] in LOWER_KEEP else m[1].capitalize(), s.strip())
    return s if not s or s.endswith(".") else s.rstrip(":") + "."


def bold_lead(tex):
    """A finding as (its bold lead, the finding): the lead is the bold opening, or the
    first sentence when the finding opens unbolded, cut to one sentence. A README
    bullet bold from end to end keeps as its lead its first sentence, else what comes
    before its colon, else (when that is still long) what comes before its ", but"; the
    rest of it becomes the elaboration."""
    if tex.startswith("\\textbf{"):
        depth = 0
        for end in range(len("\\textbf"), len(tex)):
            depth += (tex[end] == "{") - (tex[end] == "}")
            if depth == 0:
                break
        lead, rest = tex[len("\\textbf{"):end], tex[end + 1:]
    else:
        m = re.search(r"(?<=[\w%)$])\.\s+(?=[A-Z])", tex)
        lead, rest = (tex[:m.start()], tex[m.end():]) if m else (tex, "")
    lead, extra = lead.strip().rstrip(":.").strip(), ""
    if m := re.search(r"(?<=[\w%)$])\.\s+(?=[A-Z])", lead):
        lead, extra = lead[:m.start()], lead[m.end():]
    elif (m := re.search(r":\s", lead)) and len(lead[:m.start()].split()) >= 6:
        lead, extra = lead[:m.start()], lead[m.end():]
    if len(lead.split()) > 30 and ", but " in lead:
        lead, but = lead.split(", but ", 1)
        extra = f"{sentence('but ' + but)} {extra}"
    lead = sentence(lead)
    rest = f"{sentence(extra)} {sentence(rest)}" if extra.strip() else sentence(rest)
    return lead, f"\\textbf{{{lead}}} {rest}".strip()


def rq02_bbpb_caption(d):
    """The caption of app_rq02_bbpb, its numbers read from the figure's CSV `d`."""
    size = d[(d["panel"] == "DA-size") & (d["group"] == "all pairs") & (d["size"] != "1.7B")]
    acc, bb = (size[size["row"] == r].set_index("size") for r in ("Accuracy", "bBPB"))
    n_acc, n_bb = acc["n_tasks"].astype(int), bb["n_tasks"].astype(int)
    lo, hi = bb["reliability_macro"].min(), bb["reliability_macro"].max()
    tasks_bb = f"the same {n_bb.iloc[0]} tasks" if n_bb.nunique() == 1 else f"{n_bb.min()}--{n_bb.max()} tasks"
    return (
        "\\textbf{On accuracy alone a larger proxy decides more like the 1.7B reference, while on the bBPB twins it "
        "does not.} This figure splits the panels of Figure~\\ref{fig:rq2} by how a task is scored. Top row: accuracy (the original "
        "items and their RF and LLM-RF versions). Bottom row: the bBPB twins, the bits per byte of the gold answer. A "
        "twin is ranked against its own bBPB (bBPB $\\rightarrow$ 1.7B bBPB): at the 1.7B final checkpoint for DA-size "
        "and DA-goal, and at the proxy's own final checkpoint for DA-ckpt. It has no chance level, so the above-chance "
        "gate keeps it at every size. Both rows use the mono-axis design pairs of the seed-1904 runs and the task filter "
        "of Figure~\\ref{fig:rq2} (median DA-size or median DA-ckpt at least 0.66), and together they hold exactly its "
        f"tasks. On accuracy, DA-size over all pairs goes from {acc.at['90M', 'reliability_macro']:.2f} at 90M to "
        f"{acc.at['1B', 'reliability_macro']:.2f} at 1B over {n_acc.min()}--{n_acc.max()} tasks (the gate keeps more "
        f"tasks at larger sizes). On bBPB it stays at {lo:.2f}--{hi:.2f} over {tasks_bb}. Left: DA-size, one black "
        "line over all pairs and one green line per design axis. Middle: DA-ckpt. Right: DA-goal, one line per proxy "
        "size. A hollow point is 1.0 by comparing a ranking with itself.")


# paper figure stem -> (its caption from its CSV, label): one float page each, sections/<stem>.tex
FIGURE_PAGES = {"app_rq02_bbpb": (rq02_bbpb_caption, "fig:app_rq02_bbpb")}


def float_page(stem, caption, label):
    """A full-width figure on a float page of its own. The figure takes at most
    0.8 of the text height, which leaves room for a caption of ten lines: a float
    taller than the page runs into the bottom margin, which ACL's format check
    rejects. The page block caps its figure at 0.36 for the same reason. A stem
    copied as TeX is a generated table, set in a table* float instead. A callable
    `caption` is built from the figure's CSV, so its numbers follow each refresh."""
    caption = plain(caption(pd.read_csv(FIGURES[stem][0].with_suffix(".csv"))) if callable(caption) else caption)
    if FIGURES[stem][1] == ("tex",):
        return "\n".join(["\\begin{table*}[p]", "\\centering", "\\small", f"\\caption{{{caption}}}",
                          f"\\label{{{label}}}", f"\\input{{figures/{stem}}}",
                          f"% source: {FIGURES[stem][0].relative_to(ANALYSIS.parents[2])}.tex", "\\end{table*}", ""])
    return "\n".join([
        "\\begin{figure*}[p]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{{figures/{stem}.png}}",
        f"% source: {FIGURES[stem][0].relative_to(ANALYSIS.parents[2])}.png",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{figure*}",
        "",
    ])


def tall(stem):
    """A paper figure taller than wide (the PNG header's width and height)."""
    w, h = struct.unpack(">II", (HERE / f"{stem}.png").read_bytes()[16:24])
    return h > w


def page(folder, title, stem, setup, kf_image, kf_bullet, extras=()):
    lines = (ANALYSIS / folder / "README.md").read_text().splitlines()
    alt = next(re.match(r"!\[([^\]]*)\]", l)[1] for l in lines if l.startswith("!["))
    src = FIGURES[stem][0]
    # a `_paper` twin the README does not show is read through the figure it is a twin of
    shown = (src.name if any(f"/{src.name}.png)" in l for l in lines)
             else src.name.removesuffix("_horizontal").removesuffix("_paper"))
    image = (kf_image or shown, kf_bullet)
    caption = CAPTIONS.get(folder) or next((re.match(r"!\[([^\]]*)\]", l)[1] for l in lines
                    if l.startswith("![") and f"/{shown}.png)" in l), None) or alt
    caption = re.sub(r",\s*(the )?paper (figure|copy)$", "", caption.rstrip("."))
    label = f"fig:app_{folder}"
    leads, findings = zip(*(bold_lead(plain(to_tex(k))) for k in key_findings(lines, image)))
    findings = [findings[0] + f" (Figure~\\ref{{{label}}}.)", *findings[1:]]
    findings = [f.replace("\\textbf{", f"\\textbf{{{i}. ", 1) for i, f in enumerate(findings, 1)]
    caption = f"\\textbf{{{leads[0]}}} {plain(to_tex(caption))}."
    main = [] if tall(stem) else [
        "",
        "\\vspace{\\baselineskip}",
        "\\noindent\\begin{minipage}{\\textwidth}",
        "\\centering",
        f"\\includegraphics[width=\\textwidth,height=0.36\\textheight,keepaspectratio]{{figures/{stem}.png}}",
        f"% source: {src.relative_to(ANALYSIS.parents[2])}.png",
        f"\\captionof{{figure}}{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{minipage}",
    ]
    parent = [f"\\section{{{PARENT[0]}}}", f"\\label{{{PARENT[1]}}}", ""] if folder == next(iter(PAGES)) else []
    return "\n".join([
        "\\clearpage",
        "\\twocolumn[{%",
        *parent,
        f"\\subsection{{{title}}}",
        f"\\label{{app:{folder}}}",
        "",
        "\\noindent\\begin{minipage}[t]{0.485\\textwidth}",
        "\\vspace{0pt}",
        f"\\paragraph{{Research Question.}} \\emph{{{plain(to_tex(question(lines)))}}}",
        "",
        f"\\paragraph{{Setup.}} {plain(setup)}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.485\\textwidth}",
        "\\vspace{0pt}",
        "\\begin{rqfinding}",
        "\n\\\\\n\\\\\n".join(findings),
        "\\end{rqfinding}",
        "\\end{minipage}",
        *main,
        "\\vspace{\\baselineskip}",
        "}]",
        "",
    ] + ([float_page(stem, caption, label)] if tall(stem) else [])
      + [float_page(*e) for e in extras])


def main():
    folders = sorted(p.name for p in ANALYSIS.glob("rq*") if (p / "README.md").is_file())
    missing = [f for f in folders if f not in PAGES]
    for folder in [f for f in PAGES if f in folders]:
        if not (HERE / f"{PAGES[folder][1]}.png").is_file():      # a figure only the cluster run draws (PENDING)
            print(f"  PENDING app_{folder}: its figure is not drawn yet")
            continue
        tex = page(folder, *PAGES[folder])
        out = SECTIONS / f"app_{folder}.tex"
        if not out.is_file() or out.read_text() != tex:
            out.write_text(tex)
        print(f"\\input{{sections/app_{folder}}}")
    for stem, (caption, label) in FIGURE_PAGES.items():
        tex = float_page(stem, caption, label)
        out = SECTIONS / f"{stem}.tex"
        if not out.is_file() or out.read_text() != tex:
            out.write_text(tex)
        print(f"\\input{{sections/{stem}}}")
    tex, out = fixed_populations(), SECTIONS / "app_fixed_populations.tex"
    if not out.is_file() or out.read_text() != tex:
        out.write_text(tex)
    print("\\input{sections/app_fixed_populations}")
    for f in missing:
        print(f"  NO PAGE for {f}: add it to PAGES")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
