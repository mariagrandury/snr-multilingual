"""Write one appendix page per analysis folder from its README.

For every src/signal-and-noise/analysis/rqNN_<name>/ this writes
documents/paper/sections/rqs/app_rqNN_<name>.tex, one page: \\clearpage, then a
full-width block at the top of the page (\\twocolumn[...], so it cannot drift
to another page as a figure* would): the subsection title (after \\rqparent,
the section main.tex opens for the group the page starts), the research question,
the setup in the left column, the key findings in the right one, and the
main figure (the paper copy make_rq_figures.py puts in this folder) across
both columns under them. A main figure taller than wide goes on a float page
of its own instead. The question, the setup and the findings are read from the README on
every run, so the pages follow each refresh; the title, the figure
and its description are fixed below (they name no counts that a refresh
moves), and so is the setup of a README without one. Every caption starts with the
figure's takeaway in bold: the lead of the first finding for the main figure, a fixed
sentence for the extras. Every caption sits below its figure or table.

    question     the README's **Question.** paragraph or the first paragraph of
                 its "Research question" / "Question" section (or, failing that,
                 the first paragraph), cut after its last "?"; without one, the
                 question in the title line
    setup        the bullet list right under a `**Setup.**` line of the README (what
                 is measured, on which population and pool, and how), joined into
                 one paragraph without the bullets' bold labels; without one, the
                 setup string of PAGES
    findings   an rqfinding box (defined in main.tex) of at most three
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

The setup bullet or finding of the README that an extra supports cites it by its paper stem:
"(figure: app_rq02_bbpb_acc_gate)", "(table: rq00_thresholds)", or several stems in one
token, "(figures: a, b)" or "(figure: a, table: b)". The README shows the token as written,
and to_tex turns it into "(Figure~\\ref{...})" with the float's label. A page warns about
every extra that its setup and findings leave uncited, so each float is referred to in the text.

A FIGURE_PAGES entry writes sections/rqs/<stem>.tex, one more figure on a float page
of its own, with a caption built from the figure's CSV on every run:
app_rq02_bbpb, the main-text rq2 figure split into an accuracy and a bBPB row.
FIXED writes sections/rqs/app_fixed_populations.tex, the fixed-task twins (one task
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

# every page this writes goes to sections/rqs/, the generated part of the paper
RQS = HERE.parent / "sections" / "rqs"
# Every page opens with \rqparent, the appendix section its group of pages is a subsection of. main.tex
# sets it with \rqgroup{title}{label} before the group's first page, which typesets it inside its
# full-width block and empties it, so main.tex alone decides the order and grouping of the pages.
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
        f"\\textbf{{Choosing the reliable tasks out of sample moves DA-size over all single-axis pairs by at most "
        f"{abs(worst['diff']):.2f} (at {worst['x']}).}} Figure~\\ref{{fig:rq2}} redrawn with the reliable tasks chosen out "
        "of sample. We decide which tasks are reliable (DA above 0.66 on the size or the checkpoint axis) on one half of "
        "the design pairs and read the panels on the other half. We repeat this over 20 random splits, stratified by the "
        "axes a pair moves, and in both directions (solid lines, with the 5--95\\% range over splits). The dashed lines "
        "choose and read the tasks on the same half, so the gap between solid and dashed lines is the selection effect "
        "alone. Lines with too few pairs per half are not drawn.")


def rq02_da_crossfit_caption(d):
    """The caption of app_rq02_decision_accuracy_crossfit, its numbers read from the figure's CSV `d`."""
    twin = readings(d, "cross-fitted", "in-sample (half)", population="all benchmarks")
    return (
        f"\\textbf{{On the multi-axis pairs, the reliable benchmark tasks chosen out of sample and in sample give DA-size "
        f"within {twin['diff'].abs().max():.2f} of each other.}} Figure~\\ref{{fig:app_rq02_decision_accuracy}} redrawn "
        "with the reliable benchmark tasks chosen out of sample (solid) or in sample on the same half of the pairs "
        "(dashed), over the splits of Figure~\\ref{fig:app_rq02_rq2_crossfit}.")


def fixed_impact(d, line, what, fmt=".2f", value="value", unit="tasks", **where):
    """What one task set per line does to a line of a paper figure, from the twin's
    CSV `d`: (a bold lead with the largest shift (fixed - moving) along the line
    and the verdict, a sentence with its fixed and moving values at the first and
    last point and the task counts, and the largest shift of any line of the same
    panel when another line moves more)."""
    panel = d
    for k, v in where.items():
        panel = panel[panel[k] == v]
    d = panel[panel["line"] == line]
    w = d.pivot_table(index="x", columns="reading", values=[value, "n_tasks"])
    w = w.loc[sorted(w.index, key=lambda x: SIZES.index(x) if x in SIZES else float(x))].dropna()
    f, m, n = w[(value, "fixed")], w[(value, "moving")], w[("n_tasks", "moving")]
    diff = f - m
    x = diff.abs().idxmax()
    first, last = w.index[0], w.index[-1]
    num = lambda v, sign="": f"{v:{sign}{fmt}}".replace("-", "−")      # a minus sign, not a hyphen
    step = 10 ** -int(fmt.strip(".f"))                                  # the smallest shift `fmt` can show
    span, flip = m.max() - m.min(), f[x] * m[x] < 0
    verdict = (f"{'which changes its sign, ' if flip else ''}so the {x} point is a population effect"
               if flip or abs(diff[x]) >= span else
               "so part of the line's movement comes from its changing tasks" if abs(diff[x]) > span / 4
               else "small against the line's own range, so its shape is not a population effect")
    shift = f"by at most {num(diff[x], '+')} (at {x})" if abs(diff[x]) >= step / 2 else f"by less than {step:{fmt}}"
    o = panel.pivot_table(index=["line", "x"], columns="reading", values=value).dropna()
    o = o["fixed"] - o["moving"]
    i = o.abs().idxmax()
    name = re.sub(r"\bL(?=\d+\b)", "K", i[0])                       # the language count is K in the paper
    other = (f" Over every line of the panel the largest shift is {num(o[i], '+')} ({name}, at {i[1]})."
             if abs(o[i]) >= abs(diff[x]) + step / 2 else "")
    return (f"Holding the tasks fixed moves {what} {shift}, {verdict}.",
            f"On the {int(w[('n_tasks', 'fixed')].iloc[0])} {unit} with a value at every point, the line reads "
            f"{num(f.iloc[0])} at {first} and {num(f.iloc[-1])} at {last}, against {num(m.iloc[0])} and "
            f"{num(m.iloc[-1])} on the {unit} of each point ({int(n.min())}--{int(n.max())}).{other}")


# fixed-task twin stem -> (the label of the figure it is a twin of, what the twin changes, fixed_impact's arguments)
FIXED = {
    "app_rq02_fixed_rq2": ("fig:rq2", "the reliable tasks of each line",
                           dict(line="all pairs", what="the DA-size line over every single-axis pair", value="da",
                                panel="DA-size")),
    "app_rq02_fixed_decision_accuracy": ("fig:app_rq02_decision_accuracy", "the tasks above chance",
                                         dict(line="all benchmarks", what="the benchmark line", value="da")),
    "app_rq05_fixed_design_decisions": ("fig:app_rq05_design_decisions", "the (task, language setting) cells of each line",
                                        dict(line="scheme_B:DCLMP / benchmark",      # the K = 1 line Key finding 1 cites
                                             what="the benchmark line of the A vs DCLMP decision", unit="cells",
                                             panel="da_size")),
    "app_rq10_fixed_size_generalisation": ("fig:app_rq10_size_generalisation",
                                           "the tasks of each channel (the left panel is not redrawn)",
                                           dict(line="benchmarks to 3B", what="the benchmark line against the 3B reference",
                                                panel="3B",
                                                also=(dict(line="benchmarks to 1.7B", panel="1.7B",
                                                           what="the benchmark line against the 1.7B reference"),))),
    "app_rq11_fixed_evaluation_recipe": ("fig:app_rq11_evaluation_recipe", "the tasks of each variant",
                                         dict(line="All variants", what="the pooled line",
                                              also=(dict(line="Original accuracy", what="the original accuracy line"),))),
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
        lead, rest = fixed_impact(d, **{k: v for k, v in kw.items() if k != "also"})
        # the other lines the figure's claim reads (`also`): their largest shift, so the lead's line is not the only one
        more = "".join(f" {escape(fixed_impact(d, **a)[0])}" for a in kw.get("also", ()))
        return (f"\\textbf{{{escape(lead)}}} Figure~\\ref{{{ref}}} redrawn with one task set per line, {tasks} that "
                f"have a value at every point (solid), against the original (dashed). {escape(rest)}{more}")
    return caption


def fixed_populations():
    """The appendix of the fixed-task twins: one float page per paper figure whose task set moves along x."""
    drawn = {stem: ref for stem, (ref, _, _) in FIXED.items() if FIGURES[stem][0].with_suffix(".csv").is_file()}
    pairs = [f"Figure~\\ref{{fig:{stem}}} {'redraws ' if i == 0 else ''}Figure~\\ref{{{ref}}}"
             for i, (stem, ref) in enumerate(drawn.items())]
    which = pairs[0] if len(pairs) < 2 else ", ".join(pairs[:-1]) + f", and {pairs[-1]}"
    return "\n".join([
        "\\clearpage", "\\section{Fixed task sets}", "\\label{app:fixed_populations}", "",
        plain("A line that summarizes tasks (a mean, a median or a pooled agreement) against model size or training "
              "reads, at each point, the tasks that have a value there. The above-chance gate keeps more tasks at larger "
              "sizes and the three-pair minimum empties some cells, so a line can rise or fall because its tasks change. Each figure below redraws a paper figure with one task set per line, the tasks "
              f"that have a value at every point of that line (solid), and draws the original line dashed behind it. {which}."),
        "", *(float_page(stem, fixed_caption(ref, tasks, kw), f"fig:{stem}") for stem, (ref, tasks, kw) in FIXED.items()
              if stem in drawn)])


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
        f"the reference than its best accuracy format on "
        f"{'both' if wins == len(both) == 2 else f'{wins} of the {len(both)}'} top benchmarks where both have a "
        f"value.}} Overall, DA-size on accuracy goes from {ov.at['90M', 'acc']:.2f} at 90M to {ov.at['1B', 'acc']:.2f} at "
        f"1B, against {ov['bbpb_bbpb'].min():.2f}--{ov['bbpb_bbpb'].max():.2f} for the bBPB twins (read against the 1.7B "
        f"bBPB) and {ov['bpb'].min():.2f}--{ov['bpb'].max():.2f} for BPB. The other rows are the {n} benchmarks with the "
        "highest mean DA-size over 90M--1B in their original or RF format (at least five tasks), ranked, with that "
        "format and value after the name. "
        "Columns: DA-size against the 1.7B final checkpoint, DA-ckpt against the proxy's own final checkpoint and "
        "DA-goal against the 1.7B final checkpoint, the last two averaged over the proxies 90M--1B. All use the "
        "multi-axis pairs of the seed-1904 runs, and accuracy is gated above chance. bBPB uses the twins of the "
        "ranking format, read against the 1.7B accuracy of the same items or against the 1.7B bBPB. The reading "
        "against accuracy has no DA-ckpt, because its reference would be a different score. "
        + (f"{', '.join(no_twin[:-1]) + ' and ' if len(no_twin) > 1 else ''}{no_twin[-1]} "
           f"{'has' if len(no_twin) == 1 else 'have'} no twin. " if no_twin else "")
        + "The bands are 95\\% bootstrap intervals over tasks. A point on fewer than five tasks is not drawn. A hollow "
        "point is 1.0 because it compares a ranking with itself.")


def _rq02_maps(d):
    """What the per-language maps share: the drawn cells and the setup sentences (the filter read from the CSV)."""
    rest = d[~d["drawn"]].drop_duplicates("benchmark")
    single, empty = (sorted(rest.loc[rest["n_languages"] < 2, "benchmark"]),
                     sorted(rest.loc[rest["n_languages"] >= 2, "benchmark"]))
    filt = d["filter"].iloc[0]
    cut = ("As in that figure, only the tasks whose median DA-size or median DA-ckpt is at least 0.66 are kept. This "
           "filter selects on DA, so the maps are conditional on it." if filt != "none" else "There is no reliability "
           "filter.")
    setup = (
        "A cell is the mean over a benchmark's original accuracy tasks (no RF, LLM-RF or bBPB twin) in a trained "
        f"language, on the mono-axis pairs of Figure~\\ref{{fig:rq2}}. {cut} Grey cells are gated, white cells have "
        "no value, and the scale is centred at 0.5 (chance agreement). Languages follow resource rank, with lines where "
        f"the lists of $K$ = 1, 2, 8, 15, 30 and 50 languages end. Not drawn are {len(single)} single-language "
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
        f"DA-goal clears 0.5 on {gb} of {gnb} benchmarks and {gl} of {gnl} languages, DA-ckpt on {cb} of {cnb} and {cl} of {cnl}. The mean cell is {sm:.2f} for DA-size, {cm:.2f} for DA-ckpt and {gm:.2f} for DA-goal. Top: "
        "DA-size of each proxy's final checkpoint against the 1.7B final. Middle: DA-ckpt of the nine earlier tenths "
        "of each run against its own final, over the proxies and the 1.7B run, where DA-ckpt is defined too. Bottom: "
        "DA-goal of the ten tenths of each proxy against the 1.7B final. DA-size and DA-goal stop at 1B because 1.7B "
        "is their reference. Each task is first averaged over these readings. Side panels: row and column means with "
        "95\\% bootstrap intervals over the cells. " + setup)


def rq02_bbpb_acc_gate_caption(d):
    """The caption of app_rq02_bbpb_acc_gate, its numbers read from the figure's CSV `d`."""
    mid = d[d["population"] == "acc at chance at the proxy"]
    span = lambda c: f"{mid[c].min():.2f}--{mid[c].max():.2f}"
    return (
        f"\\textbf{{Where accuracy is at chance at the proxy, bBPB still ranks the design variants like the 1.7B bBPB "
        f"at {span('bbpb_bbpb')}, above accuracy on the same tasks ({span('acc_acc')}) but far below the bits per byte "
        f"of the training languages ({d['bpb_macro'].min():.2f}--{d['bpb_macro'].max():.2f}).}} DA-size of the final "
        "checkpoint of each proxy against the 1.7B final checkpoint, on multi-axis design pairs, over the benchmark "
        "tasks split by their accuracy: above chance at the proxy (left), at chance at the proxy (middle) and, a subset of "
        "the middle, at chance at every size (right). The pairs are the 406 of the 528 multi-axis pairs whose two families the per-item store holds. Blue: accuracy against the 1.7B accuracy, without the above-chance filter. Orange: the "
        "bBPB of the proxy against the 1.7B accuracy, defined where the 1.7B accuracy is above chance "
        f"({span('bbpb_acc')} in the middle panel). Green: bBPB against the 1.7B bBPB, which has no chance level. "
        f"The middle panel holds {mid['bbpb_bbpb_n'].max()} tasks at 90M and {mid['bbpb_bbpb_n'].min()} at 1B. The black "
        "tick is the DA-size of the macro average of the per-language bits per byte on the same pairs, and the dashed "
        "line marks 0.5.")


def rq04_by_scoring_caption(d):
    """The caption of app_rq04_surrogates_by_scoring, its numbers read from the figure's CSV `d`."""
    snr, early = "SNR, relative std", "early-checkpoint agreement (10 %)"
    acc, bb, bpb = (d[d["kind"] == k] for k in ("benchmark accuracy", "benchmark bBPB", "per-language bits per byte"))
    span = lambda x, m: f"{x.loc[x['metric'] == m, 'rho'].min():.2f}--{x.loc[x['metric'] == m, 'rho'].max():.2f}"
    leader = acc.loc[acc.groupby("proxy")["rho"].idxmax()].set_index("proxy")["metric"]
    leads = [s for s in SIZES if leader.get(s) == snr]
    at = f"{leads[0]}--{leads[-1]}" if SIZES.index(leads[-1]) - SIZES.index(leads[0]) == len(leads) - 1 else ", ".join(leads)
    lead_acc = acc[acc["metric"] == snr].set_index("proxy").loc[leads, "rho"]
    neg = bpb[(bpb["proxy"] == "600M") & (bpb["rho"] < 0)]
    return (
        f"\\textbf{{The bBPB twins carry most of the pooled association of Figure~\\ref{{fig:rq3}} and the lead of "
        f"early-checkpoint agreement, while on accuracy alone the relative-standard-deviation SNR leads at {at} "
        f"($\\rho$ {lead_acc.min():.2f}--{lead_acc.max():.2f}).}} Spearman $\\rho$ between each surrogate statistic and DA-size against the 1.7B reference, at each "
        "proxy size, for the benchmark tasks above chance scored by accuracy (left), their bBPB twins (middle) and the "
        "per-language bits per byte of the 50 trained languages (right). The number next to each point is the number "
        f"of tasks. On the twins early-checkpoint agreement reads {span(bb, early)} and the SNR {span(bb, snr)}, "
        f"against {span(acc, early)} and {span(acc, snr)} on accuracy. On the per-language bits per byte {len(neg)} of "
        f"the {bpb.loc[bpb['proxy'] == '600M', 'metric'].nunique()} statistics turn negative at 600M. SNR discrepancy "
        "and the margin above chance are defined on the accuracy tasks only, and the scaling-fit $R^2$ needs three "
        "model sizes, so it starts at 350M.")


def rq04_mono_axis_caption(d):
    """The caption of app_rq04_surrogates_mono_axis, its numbers read from the figure's CSV `d` (the rq3 heatmap
    over the single-axis pairs) and from the main-text rq3's CSV (every pair)."""
    snr, early = "SNR, relative std", "early-checkpoint agreement (10 %)"
    name = {snr: "the relative-standard-deviation SNR", early: "early-checkpoint agreement"}
    multi = pd.read_csv(FIGURES["rq3"][0].with_suffix(".csv"))
    span = lambda x, m: f"{x.loc[x['metric'] == m, 'rho'].min():.2f}--{x.loc[x['metric'] == m, 'rho'].max():.2f}"
    leader = d.loc[d.groupby("proxy")["rho"].idxmax()].set_index("proxy")["metric"]
    top = leader.value_counts().idxmax()
    leads = [s for s in SIZES if leader.get(s) == top]
    at = ("every proxy size" if len(leads) == leader.size else
          f"{leads[0]}--{leads[-1]}" if SIZES.index(leads[-1]) - SIZES.index(leads[0]) == len(leads) - 1 else ", ".join(leads))
    both = d.merge(multi, on=["metric", "proxy"], suffixes=("", "_all"))
    weaker = "more weakly" if both["rho"].mean() < both["rho_all"].mean() else "about as strongly"
    return (
        f"\\textbf{{On the single-axis pairs the surrogates correlate {weaker} with DA-size, and "
        f"{name.get(top, top)} leads at {at} ($\\rho$ {span(d, top)}).}} Figure~\\ref{{fig:rq3}} redrawn "
        "with DA-size and early-checkpoint agreement taken over the single-axis pairs of Figure~\\ref{fig:rq2} instead "
        "of every pair of design variants, with the same gate (above chance at the proxy and at 1.7B) and the same "
        f"bBPB twins. Early-checkpoint agreement reads {span(d, early)} against {span(multi, early)} over every pair, "
        f"and the relative-standard-deviation SNR {span(d, snr)} against {span(multi, snr)}. Each cell gives $\\rho$ "
        "and the number of tasks $n$. The scaling-fit $R^2$ needs three model sizes, so it starts at 350M.")


def rq02_language_per_proxy_caption(d):
    """The caption of app_rq02_da_size_by_language_per_proxy, its numbers read from the figure's CSV `d` and,
    for the contrast, from the same figure without the filter."""
    drawn, setup, lead = _rq02_maps(d)

    def per(x):
        return x.groupby("proxy_size").agg(cells=("da", "count"), mean=("da", "mean"), above=("da", lambda v: v.dropna().gt(0.5).mean()))
    p = per(drawn)
    hl = drawn[drawn["highlight"]].iloc[0]
    out = ("\\textbf{" + lead(
        f"a larger proxy decides more like the 1.7B reference: {p.at['90M', 'above']:.0%} of the cells with a value have DA-size "
        f"above 0.5 at 90M and {p.at['1B', 'above']:.0%} at 1B, and {hl['benchmark']} (outlined) is above 0.5 at every "
        f"proxy in {int(hl['steady_languages'])} of its {int(hl['complete_languages'])} languages.") + "} "
           f"The mean goes from {p.at['90M', 'mean']:.2f} to "
           f"{p.at['1B', 'mean']:.2f}. ").replace("%", "\\%")
    if d["filter"].iloc[0] != "none":
        src = FIGURES["app_rq02_da_size_by_language_per_proxy"][0]
        u = per(pd.read_csv(src.with_name(src.name.replace(f"_{d['filter'].iloc[0]}", "")).with_suffix(".csv"))
                .query("drawn"))
        out += f"Without the filter it goes only from {u.at['90M', 'mean']:.2f} to {u.at['1B', 'mean']:.2f}. "
    return out + ("DA-size of each proxy's final checkpoint against the 1.7B final, one map per proxy, with the row "
                  "means in the last map. Only benchmarks with a DA-size at some proxy are drawn. " + setup)


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
        "English tasks (originals, RF, LLM-RF, bBPB twins and the English bits per byte). The FineWeb lead comes from "
        f"the bBPB twins. On the {int(acc.at['FineWeb', 'size'])} accuracy tasks alone, the three corpora reach "
        + ", ".join(f"{pct(acc.at[c, 'mean'])} ({c})" for c in ("DCLM-Edu", "DCLM", "FineWeb")) + ". "
        "The fits and the above-chance gate are those of Figure~\\ref{fig:rq1}. With one $K$, a task has one "
        "model-size fit instead of a median over the six $K$, and the trajectory $R^2$ is the median over at least two "
        "sizes. Top: the $R^2$ of the model-size fit against its Spearman $\\rho$. Bottom: the same $R^2$ against the "
        "median $R^2$ of the training-trajectory fits, shaded by regime at 0.5. A black edge marks a task whose score "
        f"declines with size. The grey points are the {int(both.at[back, 'size'])} tasks of Figure~\\ref{{fig:rq1}} "
        f"(medians over $K$, {pct(both.at[back, 'mean'])} predictable across both), so they sit higher than a single "
        "fit would.")


def rq13_regimes_l1_caption(d):
    """The caption of app_rq13_english_scaling_regimes_l1_accuracy, its numbers read from the figure's CSV `d`."""
    pct = lambda v: f"{100 * v:.0f}\\%"
    span = lambda s: pct(s.min()) if round(100 * s.min()) == round(100 * s.max()) else f"{pct(s.min())}--{pct(s.max())}"
    d = d.assign(b=d["regime"] == "predictable across both")
    both = d.groupby("corpus")["b"].mean()
    fmt = d.groupby(["format", "corpus"])["b"].mean()
    n = d[d["corpus"] == "FineWeb"]["format"].value_counts()
    same = "the same " if round(100 * both.min()) == round(100 * both.max()) else ""
    return (
        f"\\textbf{{On the English benchmarks alone, the three English corpora put {same}{span(both)} of the tasks in the "
        "regime predictable across both size and training, so the FineWeb lead of "
        "Figure~\\ref{fig:app_rq13_english_scaling_regimes} comes from the bBPB twins.} The same scaling regimes "
        "restricted to the monolingual English cells ($K$ = 1) and the "
        "English benchmarks. Each column is one deep cell at seed 1904, 90M--1.7B, on the same "
        f"{int(n.sum())} English accuracy tasks ({n.get('original', 0)} originals, {n.get('rf', 0)} RF and "
        f"{n.get('rfgm', 0)} LLM-RF). The bBPB twins, the English bits per byte and every cell with $K > 1$ are left out, "
        "so no point comes from a multilingual model. The fits and the above-chance gate are those of "
        f"Figure~\\ref{{fig:rq1}}. Per corpus, {span(fmt['original'])} of the originals and {span(fmt['rf'])} of the RF "
        "twins are predictable across both. Top: the $R^2$ of the model-size fit against its Spearman $\\rho$. Bottom: the "
        "same $R^2$ against the median $R^2$ of the training-trajectory fits, shaded by regime at 0.5. A black edge marks "
        "a task whose score declines with size.")


def rq13_regimes_tagged_caption(d):
    """The caption of app_rq13_english_scaling_regimes_l1_accuracy_tagged, its benchmarks read from the figure's CSV `d`."""
    wide = d.pivot(index="task", columns="corpus", values="regime")
    name = d.drop_duplicates("task").set_index("task")["name"]
    pop = d.loc[d["popular"], "task"].unique()
    every = lambda r: [name[k] for k in pop if (wide.loc[k] == r).all()]
    listed = lambda xs: ", ".join(xs[:-1]) + f" and {xs[-1]}" if len(xs) > 1 else "".join(xs)
    weak = [k for k in pop if not wide.loc[k].isin(["predictable across both", "filtered"]).any()]
    n_filtered = d[d["regime"] == "filtered"].groupby("corpus").size()
    return (
        "\\textbf{The well-known English benchmarks are the predictable ones, and the popular tasks that drop out stay "
        "at chance rather than scale badly.} " + listed(every("predictable across both")) + " are predictable across both "
        "size and training in all three corpora. " + listed(every("filtered")) + " never pass the above-chance filter "
        "in their original format" + (", and " + listed([name[k] for k in weak]) + " is the one above chance that does "
        "not scale" if weak else "") + ". The bottom row of Figure~\\ref{fig:app_rq13_english_scaling_regimes_l1_accuracy}, "
        "one panel per English corpus ($K$ = 1), with every task named beside its panel, on the left when the $R^2$ of its "
        "model-size fit is below 0.5 and on the right otherwise. A name takes the colour of its regime (blue predictable "
        "across both, green during training only, slate across size only, brown weak in both, black declines with size). "
        f"The {n_filtered.max()} English accuracy tasks that the filter removes have no point.")


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
        "not train counts here. The scheme-A runs are evaluated on every language, so a language that no list trains, "
        f"such as Basque (its column averages {off[off['tl'] == 'eu']['da_size'].mean():.2f}), is read on those runs "
        "alone. A cell averages the pairs both tasks have, so two cells may average different decisions. The colour is "
        "centred at 0.5, the agreement expected by chance. Languages follow the resource order of the training lists.")


# A family-transfer cell ("L2 ZH") as the paper names it: its K and the scheme letter its build implements
# there (DATA_SCHEMES `letter`; at K = 2 Russian is A, Chinese B, Spanish C).
RQ06_LETTER = {"A": "A", "B": "B", "ZH": "B", "ES": "C"}
rq06_cell = lambda c: f"$K$ = {c.split()[0][1:]} scheme-{RQ06_LETTER[c.split()[1]]}"


def rq06_family_caption(d):
    """The caption of app_rq06_family_transfer, its numbers read from the figure's CSV `d`."""
    s = d[(d["cell"] == "L8 A") & (d["level"] == "sub")].pivot(index="size", columns="group", values="lift")
    best = s["same family"].idxmax()
    sizes = ["90M", "175M", "350M", "600M", "1B", "1.7B"]
    has = d.groupby("cell")["size"].agg(lambda x: [z for z in sizes if z in set(x)])
    cover = [f"the {rq06_cell(c)} cell " + ("at every size" if len(v) == len(sizes) else f"at {v[0]}--{v[-1]}")
             for c, v in sorted(has.items(), key=lambda kv: (int(kv[0].split()[0][1:]), RQ06_LETTER[kv[0].split()[1]]))]
    cover = ", ".join(cover[:-1]) + " and " + cover[-1]
    return (
        f"\\textbf{{The $K$ = 8 scheme-A cell clears chance on more of its untrained same-subfamily tasks than the English-only cell does on "
        f"the same tasks, by up to {s.at[best, 'same family']:+.2f} at {best}, and "
        + ("on the other subfamilies it does not" if s["other family"].max() < 0.005 else
           f"on the other subfamilies by no more than {s['other family'].max():+.2f}") + ".} Each panel is one cell (rows) "
        "and one family level (columns). A line is the share of the cell's accuracy tasks in untrained languages whose "
        "final run clears chance (on the benchmarks every cell is evaluated on, not on the candidates still being "
        "screened), split by whether the language shares the family of a trained non-English language "
        "(left: the top-level family such as Indo-European, right: the subfamily such as Slavic, from a Glottolog-style "
        "classification of every evaluated language). The dotted lines are the English-only $K$ = 1 cell on exactly the "
        "same tasks, so the gap between a solid line and its dotted twin is what training the cell's languages adds. "
        f"Deep runs at seed 1904. Untrained languages are evaluated for {cover}, some of them on part of the tasks so "
        "far. The $K$ = 1 cell has no 350M evaluation, so its dotted lines break there. Chinese has no subfamily of its "
        "own in the classification, so the $K$ = 2 scheme-B cell has no same-subfamily line.")


def rq06_lift_caption(d):
    """The caption of app_rq06_family_lift, its numbers read from the figure's CSV `d`."""
    d = d[d["n_verdicts"] >= 20]
    trained, other = d[d["same_sub"]], d[~d["same_sub"]]
    w = lambda g: (g["lift"] * g["n_verdicts"]).sum() / g["n_verdicts"].sum()
    r = d[(d["cell"] == "L8 A") & (d["family_sub"] == "Indo-European, Romance")].iloc[0]
    up, down = trained[trained["lift"] > 0], trained[trained["lift"] <= 0]
    lead = (f"All {len(trained)} subfamilies that a cell trains" if len(up) == len(trained)
            else f"Of the {len(trained)} subfamilies that a cell trains, {len(up)}")
    exceptions = " and ".join(f"{x.family_sub.split(', ')[-1]} under the {rq06_cell(x.cell)} cell ({x.lift:+.2f})"
                              for x in down.itertuples())
    return (
        f"\\textbf{{{lead} clear chance more often than under English-only training "
        f"({up['lift'].min():+.2f} to {up['lift'].max():+.2f}), while the {len(other)} others centre on zero "
        f"(median {other['lift'].median():+.2f}).}} Romance under the $K$ = 8 scheme-A cell, "
        f"which trains Spanish, French and Italian, puts {100 * r['share_above']:.0f}\\% of the Catalan, Galician, Portuguese "
        f"and Romanian tasks above chance against {100 * r['baseline_share_above']:.0f}\\% for the English-only cell. "
        + (f"The exception{'s are' if len(down) > 1 else ' is'} {exceptions}. " if len(down) else "")
        + "Each row is one subfamily of the untrained languages, read on the benchmarks every cell is evaluated on, not "
        "on the candidates still being screened. A dot is the share of a cell's tasks in those languages above chance minus "
        "the share of the English-only $K$ = 1 cell on the same tasks and sizes, pooled over the sizes both have (the "
        "$K$ = 1 cell has no 350M evaluation), drawn where it rests on at least 20 verdicts. A filled "
        "dot marks a subfamily of one of the cell's trained languages. Weighted by verdicts, the trained subfamilies gain "
        f"{w(trained):+.2f} and the others {w(other):+.2f}. Rows follow their mean over the cells.")


RQ09_TASK = ("the per-task DA-size of Figure~\\ref{fig:rq2}: the final checkpoint of each proxy against the 1.7B final, "
             "on the mono-axis design pairs of the seed-1904 runs, on the tasks above chance at the proxy and at 1.7B")
RQ09_POP = (RQ09_TASK + ". A benchmark's DA-size is the median over its tasks at a proxy, averaged over the proxies "
            "90M--1B at which it has a task")


def rq09_correlation_caption(d):
    """The caption of app_rq09_design_da_correlation, its numbers read from the figure's CSV `d`."""
    v = d.set_index(["reading", "characteristic"])
    ok = d[d["ci_low"].notna()]
    crosses = lambda r: bool(((ok["reading"] == r) & (ok["ci_low"] <= 0) & (ok["ci_high"] >= 0)).sum()
                             == (ok["reading"] == r).sum())
    snr, rel, unf = (v.loc[(r, "n_options")] for r in ("snr", "above_66_either", "unfiltered"))
    n = d.groupby("reading")["n_benchmarks"].max()          # a benchmark with no option count still counts
    per_proxy = n[n.index.str.startswith("size:")]           # the right panel: fewer benchmarks at the small proxies
    items, skill = ({r: v.loc[(r, k)] for r in ("unfiltered", "snr", "above_66_either")} for k in ("items", "skill"))
    excl = lambda r, k: bool(v.loc[(r, k), "ci_low"] > 0 or v.loc[(r, k), "ci_high"] < 0)
    other = [v.loc[("snr", k), "label"].lower() for k in d["characteristic"].unique() if k not in ("items", "skill")
             and excl("above_66_either", k) and excl("unfiltered", k)]
    return (
        f"\\textbf{{Item count orders the benchmarks by DA-size on every task above chance "
        f"($\\rho = {items['unfiltered']['value']:.2f}$) but not by SNR ({items['snr']['value']:.2f}), and the LM-like "
        f"skill correlates with both ({skill['unfiltered']['value']:.2f} and {skill['snr']['value']:.2f}).}} "
        + (f"No other characteristic has a DA-size interval that excludes zero in both readings. " if not other else
           f"Of the others, only {' and '.join(other)} excludes zero in both DA-size readings. ")
        + f"The answer options, which correlate with SNR at $\\rho = {snr['value']:.2f}$ "
        f"(${snr['ci_low']:.2f}$ to ${snr['ci_high']:.2f}$), reach ${rel['value']:.2f}$ on the reliable tasks and "
        f"${unf['value']:.2f}$ on every task above chance. A row is one characteristic of the benchmark families (Table "
        "\\ref{tab:app_rq09_benchmark_characteristics}). The value is a Spearman $\\rho$ over the benchmarks against the "
        "characteristic's code: the options ranked 2 to 5, no passage to passage, and the item count (the median "
        "number of items over a benchmark's languages) and the two lengths as they are. "
        "Skill, curation, source, task format and domain are each split in two and coded from the first group to the "
        "second: knowledge, reading and reasoning to LM-like (commonsense, grammaticality and language modelling), "
        "translated to native (written in the language, English originals included, or generated from treebanks or "
        "planning problems), translated from English to not, continuation to lettered options, and the other domains "
        "to knowledge, the most common domain. A split whose smaller side holds one benchmark has no value, so task format "
        "has none on DA-size, where INCLUDE is the only benchmark with lettered options. A characteristic marked unknown "
        "in the table leaves that benchmark out of its row, except the curation of BBH, which was written in English and "
        "so counts as native. "
        f"DA-size is {RQ09_POP}. Left: that mean with a 95\\% bootstrap interval over the "
        f"{n['above_66_either']} benchmarks whose tasks pass the reliability filter of Figure~\\ref{{fig:rq2}} "
        f"(filled), over the {n['unfiltered']} benchmarks with any task above chance (hollow), and the same "
        f"statistics on the median SNR at 1.7B of the {n['snr']} benchmarks (diamonds). Right: one line per "
        "proxy size, on the reliable tasks, " f"over {per_proxy.min()} to {per_proxy.max()} benchmarks per proxy. The lengths exist for the original benchmarks with sampled items only.")


def rq09_levels_caption(d):
    """The caption of app_rq09_design_da_by_level, its numbers read from the figure's CSV `d` (one row per
    characteristic and task of the reliable tasks)."""
    t = d.drop_duplicates("task")
    full = d.groupby(["characteristic", "level"])["da_size"].agg(["median", "size"])
    lv = full[full["size"] >= 5]                              # a level of one task has no distribution to read
    small = full[full["size"] < 5].reset_index()
    one = d.groupby("label")["level"].nunique()
    return (
        f"\\textbf{{Read per task, the {len(t)} reliable tasks sit above chance agreement (median DA-size "
        f"{t['da_size'].median():.2f}), but the levels overlap: the medians of the levels with at least five tasks lie "
        f"between {lv['median'].min():.2f} and {lv['median'].max():.2f}.}} "
        + ("".join(f"{r.level[0].upper() + r.level[1:]} holds {r.size} task{'s' if r.size > 1 else ''} "
                   f"({r.median:.2f}). " for r in small.itertuples()))
        + ("".join(f"{lab} has one level on these tasks. " for lab in one[one == 1].index))
        + "Each block is one characteristic, a header row and "
        "then one row per level with the number of tasks at that level. A dot is one task (one benchmark in one "
        "language) at its DA-size minus 0.5, so the dotted line is chance agreement, and "
        f"{(t['x'] > 0).sum()} of the {len(t)} tasks sit to its right. The bar is the level's median and the "
        f"horizontal line its interquartile range over the tasks. DA-size is {RQ09_TASK}, averaged over the proxies "
        f"90M--1B that have one, on the tasks of {t['family'].nunique()} benchmarks that pass the reliability filter of "
        "Figure~\\ref{fig:rq2}. Figure~\\ref{fig:app_rq09_design_da_correlation} gives the same characteristics as "
        "one correlation per benchmark.")


def rq09_quadrant_caption(d):
    """The caption of app_rq09_design_da_quadrant, its numbers read from the figure's CSV `d` (one row per
    characteristic, task and proxy of every task above chance, then the quadrant counts)."""
    dots = d[d["row"] == "task"]
    pts = dots.drop_duplicates(["task", "size"])
    by_size = pts.groupby("size")["da_size"].median()
    band = dots.groupby(["characteristic", "level", "size"])["da_size"].median()
    return (
        f"\\textbf{{On every task above chance, the median task decides at chance agreement at every proxy size "
        f"(DA-size {by_size.min():.2f} to {by_size.max():.2f}), on every level of every characteristic.}} One panel "
        "per characteristic. A dot is one task (one benchmark in one language) at one proxy, coloured by the proxy's "
        "size: x is its DA-size minus 0.5, so the vertical line is chance agreement, and y the characteristic's code "
        "centred at 0 (the options ranked 2 to 5 as $-1$, $-1/3$, $+1/3$, $+1$, every other characteristic split in "
        "two as $-1$ and $+1$), each proxy in a band of its own (90M lowest), jittered within it. The red tick is the "
        f"band's median, and every band's median lies between {band.min():.2f} and {band.max():.2f}. The numbers count "
        f"the dots per quadrant, and a dot at exactly 0.5 ({(pts['x'] == 0).mean() * 100:.0f}\\% of them) is in none. The "
        f"shaded quadrants are the ones a positive correlation fills. DA-size is {RQ09_TASK}, for the {len(pts)} "
        f"pairs of a task and a proxy over {pts['task'].nunique()} tasks of {pts['family'].nunique()} benchmarks.")


def rq09_table_caption(d):
    """The caption of app_rq09_benchmark_characteristics, its numbers read from the table's CSV `d`."""
    return (
        f"\\textbf{{The {len(d)} benchmark families of the benchmark design analysis and the characteristics it "
        f"groups them by.}} {int((d['source_origin'] == 'english_translated').sum())} are translated from English, "
        f"{int((d['source_origin'] == 'english_original').sum())} are English originals evaluated in English, and "
        f"{int((d['n_options'] == 4).sum())} have four answer options. Domain: what the benchmark tests, as its own "
        f"paper describes it ({int((d['domain'] == 'knowledge').sum())} test knowledge, the domain the analysis "
        "contrasts with the rest). Curation: how the items in each language were made "
        "(native: written in the language, English: written in English, human transl.: translated by people, MT: machine "
        "translated, MT + post-edit: machine translated and corrected, transl. (unknown): translated by a method the "
        "dataset does not document, template: generated from treebank or planning templates). Source: whether the "
        "benchmark was translated from English, written natively in several languages or written in English. Format "
        "(as the evaluation prompts it): MCQ is a question with lettered options, cloze scores each option as a "
        "continuation of the question, completion scores continuations of a context, statement (LLM-RF) is the LLM "
        "rewrite into a statement with four continuations, last word scores the final word of a passage. Options gives "
        "the range where subtasks differ. Unknown marks a characteristic no source settles, which the analysis leaves "
        "out, except the curation of BBH, which counts as native because its items were written in English. Mixed marks "
        "a suite whose subtasks differ, which the analysis also leaves out. Skill groups the domains: LM-like for "
        f"commonsense, grammaticality and language modelling ({int((d['skill'] == 'LM-like').sum())} benchmarks), "
        "knowledge etc. for the rest. Items: the median number of items over the benchmark's languages. Context and "
        "Option: the median characters of the context and of an option over 100 sampled English items.")


def rq09_item_count_caption(d):
    """The caption of app_rq09_item_count, its numbers read from the table's CSV `d` (one row per metric)."""
    r = d.set_index("reading")
    a, c = r.loc["unfiltered"], r.loc["da_ckpt"]
    return (
        f"\\textbf{{Benchmarks with more items decide more like the reference: DA-size rises by "
        f"{a['within_slope']:.2f} per tenfold more items within a benchmark, across its languages and subtasks.}} "
        "One row per metric. A benchmark's value is its DA-size on every task above chance (the hollow dots of "
        "Figure~\\ref{fig:app_rq09_design_da_correlation}), the median over its tasks of DA-ckpt (each task's median "
        "over its checkpoints and sizes) and of the R$^2$ of the scaling fit of Figure~\\ref{fig:rq1}, and its median SNR "
        "at 1.7B (log10 in the fits). $\\rho$ items and $\\rho$ LM-like: the Spearman correlation over the benchmarks "
        "with the item count (the median over its languages) and with the LM-like skill (commonsense, grammaticality, "
        "language modelling, against the rest, so KoBEST and NorEval, which mix both, are left out), with a 95\\% "
        "bootstrap interval over the benchmarks. Within benchmarks: the slope of the task values on log10 items with "
        "a fixed effect per benchmark and an interval clustered by benchmark, over the tasks and (in brackets) the "
        f"number of tasks and of the {int(a['within_n_varying'])} benchmarks whose tasks differ in item count, the "
        "only ones that inform it. Joint: a fit of the benchmark values on log10 items and LM-like together. DA-ckpt "
        f"follows item count as DA-size does ($\\rho = {c['rho_items']:.2f}$), SNR does not.")


def rq09_waves_caption(d):
    """The caption of app_rq09_wave_robustness, its numbers read from the table's CSV `d` (per metric and group)."""
    g = d.set_index(["reading", "group"])
    later = {r: g.loc[(r, "later, all")] for r in ("unfiltered", "snr", "r2")}
    names = {"unfiltered": "DA-size", "snr": "SNR", "r2": "the R$^2$ of the scaling fit"}
    same = [names[r] for r, x in later.items() if x["mannwhitney_p"] >= 0.05]
    diff = [f"{'lower' if x['items_adjusted_later'] < 0 else 'higher'} {names[r]} ($p = {x['mannwhitney_p']:.2f}$)"
            for r, x in later.items() if x["mannwhitney_p"] < 0.05]
    s = later["snr"]
    return (
        "\\textbf{The benchmarks added after the sweep began match the original ones on "
        + " and on ".join(same) + (f", but have a {' and a '.join(diff)}" if diff else "") + ".} "
        "The original benchmarks are the ones the sweep evaluated from its start (7 September 2026). The later ones "
        "entered in three waves: twenty screened benchmarks promoted on 23 September, the reformulated twins (22 "
        "September to 9 October), and the probes promoted on 9 October. A benchmark counts once, so the medians are "
        "over benchmarks, not tasks, and a benchmark with no task above chance is absent. Its value is its DA-size on "
        "every task above chance, its median SNR at 1.7B or its median R$^2$ of the scaling fit "
        "(Table~\\ref{tab:app_rq09_item_count}). Items: the median of the benchmarks' item counts. The interval is a 95\\% "
        "bootstrap over the benchmarks. Rank test: a two-sided Mann-Whitney test of the original benchmarks against "
        "all later ones. After items: the difference of the later benchmarks in a fit on log10 items (SNR in log10), "
        f"with its $p$ (SNR: ${s['items_adjusted_later']:+.2f}$, $p = {s['items_adjusted_p']:.2f}$).")


def rq00_screened_3b(d):
    """Whether rq00's "all considered" outputs carry a 3B verdict for the
    screened tasks (the 3B probe jobs landed in the probe gate): `d` is either
    of their CSVs, gate_pass_counts_all_considered (a `size` column) or
    first_size_share_all_considered_paper (the screened `panel` title names
    its sizes). Until then every rq00 "considered" caption is the 1B/1.7B one."""
    if "size" in d:
        return bool(((d["size"] == "3B") & (d["group"] != "auto") & d["gated"].notna()).any())
    return bool(d["panel"].str.startswith("Screened").any() and d["panel"].str.contains("3B").any())


def rq00_pending():
    """The `auto` benchmarks still counted as screened (analysis.utils.pending_benchmarks)."""
    sys.path.insert(0, str(ANALYSIS.parent))
    from analysis.utils import pending_benchmarks
    return pending_benchmarks()


def rq00_screened_sizes(d):
    """The sizes the screened tasks have a verdict at, as prose ("600M, 1B and 1.7B"), from either CSV of
    `rq00_screened_3b`: the probe groups' gated sizes, or the screened panel's title."""
    if "size" in d:
        xs = list(dict.fromkeys(d.loc[(d["group"].isin(["probe", "discarded"])) & d["gated"].notna(), "size"]))
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    t = d.loc[d["panel"].str.startswith("Screened"), "panel"].iloc[0]
    return t.removeprefix("Screened benchmarks, ").removesuffix(" only")


def rq00_considered_caption(d):
    """The caption of app_rq00_pass_counts_considered, its numbers read from the table's CSV `d`."""
    at = lambda g, f="all", z="1.7B": d[(d["group"] == g) & (d["format"] == f) & (d["size"] == z)].iloc[0]
    a, s, x = at("all"), at("probe"), at("discarded", "original")
    has3 = rq00_screened_3b(d)
    s3 = at("probe", z="3B") if has3 else None
    sz = rq00_screened_sizes(d)
    return (
        f"\\textbf{{Over the {a['benchmarks']} benchmarks with a chance level that we considered, {a['passing']:,.0f} of {a['gated']:,.0f} tasks are "
        f"above chance at 1.7B, and {s['passing']:,.0f} of the {s['gated']:,.0f} screened tasks pass although we do not "
        f"analyse them.}} Each cell gives the tasks above chance over the tasks with a verdict, per group, answer "
        f"format and model size, on the trained languages. Analysed is the group of benchmarks that every other "
        f"analysis uses. Screened, not analysed are the "
        f"benchmarks we evaluated only on the final checkpoints of the {sz} models"
        + (", including those that joined the analysed group after the screen and whose evaluation throughout "
           "pretraining has not finished" if rq00_pending() else "")
        + (f". Discarded are the "
           f"{x['gated']:.0f} screened tasks at chance at both 1B and 1.7B. " if not has3 else
           f". At 3B, "
           f"{s3['passing']:,.0f} of the {s3['gated']:,.0f} screened tasks with a 3B verdict pass. Analysed tasks have "
           f"no 3B verdict, since every other analysis stops at the 1.7B reference, so the all-considered row stops "
           f"at 1.7B. Discarded are the {x['gated']:.0f} screened tasks at chance at both 1B and 1.7B. ")
        + f"Original is the benchmark as published, RF scores the "
        f"answer strings instead of the option letters, and LLM-RF is the LLM rewrite into statements. A cell is above "
        f"chance when at least half of the seed-1904 runs of that size that train the task's language clear chance. "
        f"Tasks without a chance level have no verdict.")


def rq00_chance_considered_caption(d):
    """The caption of app_rq00_chance_considered: the 600M-1.7B text until the
    figure's CSV `d` has a 3B screened panel, then the one with 3B."""
    if not rq00_screened_3b(d):
        sz = rq00_screened_sizes(d)
        return ("\\textbf{Most of the benchmarks we screened and do not analyse clear chance by 1.7B.} Left: the "
                "analysed benchmarks of Figure~\\ref{fig:app_rq00_gate_and_curves}, here split by task. Right: the "
                f"benchmarks we evaluated only on the final checkpoints of the {sz} models and do not analyse, with "
                "the discarded tasks. Each bar splits a benchmark's tasks by the smallest model size from which they stay "
                f"above chance (a darker blue is a larger size). On the right only {sz} are evaluated, so {sz.split(',')[0]} means "
                "above chance at every evaluated size. Red marks the tasks at chance at 1.7B. The number in parentheses is the "
                "benchmark's number of tasks with a chance level. We read the seed-1904 runs that train the task's language.")
    sz = rq00_screened_sizes(d)
    return ("\\textbf{Most of the benchmarks we screened and do not analyse clear chance by 1.7B.} Left: the "
            "analysed benchmarks of Figure~\\ref{fig:app_rq00_gate_and_curves}, here split by task, from 90M to the "
            f"1.7B reference. Right: the benchmarks we evaluated only on the final checkpoints of the {sz} "
            "models and do not analyse, with the discarded tasks. Each bar splits a benchmark's tasks by the smallest "
            f"model size from which they stay above chance (a darker blue is a larger size). On the right only {sz} "
            f"are evaluated, so {sz.split(',')[0]} means above chance at every evaluated size, and 3B means at chance at 1.7B and "
            "above it at 3B. Red marks the tasks at chance at the largest size with a verdict. The number in "
            "parentheses is the benchmark's number of tasks with a chance level. We read the seed-1904 runs that train "
            "the task's language.")


# folder -> title, paper figure stem, setup, (README image the finding follows, which bullet)[, extras]
PAGES = {
    "rq00_gate_and_curves": (
        "The above-chance gate", "app_rq00_chance_share_horizontal",
        None,
        "highlights", 0,
        (("app_rq00_early_passes",
          "\\textbf{The tasks that pass at very small token counts answer with one option.} The ten tasks above chance "
          "in at least half of their cells below $10^{8}$ tokens of their language and in fewer than half of their cells "
          "at $10^{8}$--$10^{10}$ tokens. We read the deep runs with the baseline data at seed 1904, at every evaluated "
          "tenth, from their stored option log-likelihoods. Share of runs above chance: trained runs below $10^{8}$ "
          "tokens of the language (Low), trained runs at $10^{8}$--$10^{10}$ tokens (Mid), and runs that never trained "
          "the language nor any language of its subfamily, at the sizes and checkpoints of the Low runs (Untrained). Over "
          "the Low runs above chance: their number, the median share of items answered with the option the run picks "
          "most often (One option), and the share of these runs in which that option alone would clear chance "
          "(Constant passes). Longest option: the median share of picks on the single longest option and the share of "
          "gold answers on it. CulturalBench-easy asks in English about a country and is placed at the tokens of the "
          "country's language.",
          "tab:app_rq00_early_passes"),
         ("app_rq00_chance_considered", rq00_chance_considered_caption, "fig:app_rq00_chance_considered"),
         ("app_rq00_pass_counts_considered", rq00_considered_caption, "tab:app_rq00_pass_counts_considered"),)),
    "rq00_task_reformulation": (
        "Task reformulation", "app_rq00_chance_reformulation",
        None,
        None, 1),
    "rq00_chance_vs_train_tokens": (
        "Above chance against the training tokens of the language", "app_rq00_chance_vs_train_tokens",
        None,
        None, 0,
        (("rq00_thresholds",
          "\\textbf{Training tokens of the language from which a benchmark is above chance.} For each multilingual "
          "benchmark (trained in at least two languages) and answer format, the lower edge of the token bin from which "
          "the share of (language, $K$, checkpoint) cells above chance is at least 50\\% or 90\\% in that bin and in "
          "every larger bin, pooled over the model sizes 90M--1.7B and for the 90M and 1.7B models alone. Bins are three "
          "per decade of the training tokens of the language that the checkpoint had seen, and a bin counts only with at "
          "least 20 cells. $\\leq$ marks a level already met in the first bin, never a level that is not reached. Lang. is "
          "the number of trained languages. Margin is the median accuracy above chance that a task needs for its Wilson "
          "bound to clear chance, and Items its median item count. Seed-1904 runs of every data build and ladder, every "
          "evaluated tenth of a run. The pooled bins mix sizes, and from $10^{10}$ tokens on, most of their cells are "
          "English.",
          "tab:rq0_thresholds"),
         ("app_rq00_token_exclusion",
          "\\textbf{Leaving out the cells below the token threshold lowers the decision accuracy of an aggregate "
          "score.} For each benchmark and each seed-1904 run at its final checkpoint, the aggregate is the mean of the "
          "benchmark's tasks over the languages the run trains. We compute it on every cell, on the cells at or above "
          "the benchmark's pooled 50\\% token threshold (Table~\\ref{tab:rq0_thresholds}), and on the cells above chance "
          "at the run's size and at 1.7B. SE is the median over runs of the standard deviation of the per-language "
          "values divided by the square root of their number. DA-size compares the aggregate of each proxy with the "
          "1.7B aggregate under the same rule, on single-axis pairs, pooled over the 18 benchmarks that keep an "
          "aggregate under every rule. The nine benchmarks that never reach their threshold lose every cell.",
          "tab:app_rq00_token_exclusion"))),
    "rq01_scaling_predictability": (
        "Scaling predictability", "rq1",
        None,
        None, 0,
        (("app_rq00_benchmark_size_curves",
          "\\textbf{The reformulated twins rise with model size while their letter-format originals stay at "
          "chance up to 1.7B.} Benchmark accuracy against model size. For each benchmark family, we plot the final-checkpoint accuracy "
          "of every design against non-embedding parameters. This accuracy is the mean over the tasks in the "
          "languages the design trains on. There is one line per (number of languages, ladder, data build, seed), "
          "over 90M--1.7B. Colour gives the number of trained languages, line "
          "width the ladder and line style the data build. The dotted red line is chance. The curves are not "
          "gated, so a family at chance stays visible.",
          "fig:app_size_scaling"),
         ("app_rq00_benchmark_curves",
          "\\textbf{Along training, the letter-format originals stay at chance for the whole run while their "
          "reformulated twins climb steadily.} Benchmark accuracy along training. For each benchmark family, we plot the accuracy of every run at the "
          "ten evaluated tenths of the run. This accuracy is the mean over the tasks in the languages the run trains "
          "on. Training progress is given in Chinchilla multiples of the run's token budget. The figure covers "
          "every seed and data build, 90M--1.7B. Colour gives the model size, line width the ladder and "
          "line style the data build. The dotted red line is chance. The curves are not gated.",
          "fig:app_training_scaling"))),
    "rq02_decision_accuracy": (
        "Decision accuracy across sizes", "app_rq02_decision_accuracy",
        None,
        None, 0,
        (("app_rq02_da_by_language", rq02_language_caption, "fig:app_rq02_da_by_language"),
         ("app_rq02_da_size_by_language_per_proxy", rq02_language_per_proxy_caption,
          "fig:app_rq02_da_size_by_language_per_proxy"),
         ("app_rq02_bbpb_acc_gate", rq02_bbpb_acc_gate_caption, "fig:app_rq02_bbpb_acc_gate"),
         ("app_rq02_rq2_crossfit", rq02_crossfit_caption, "fig:app_rq02_rq2_crossfit"),
         ("app_rq02_decision_accuracy_crossfit", rq02_da_crossfit_caption, "fig:app_rq02_decision_accuracy_crossfit"),
         ("app_rq02_definition_sensitivity",
          "\\textbf{No definition of decision accuracy lifts benchmark accuracy far above chance.} DA-size of the "
          "final checkpoint of each proxy against the 1.7B final checkpoint, on multi-axis pairs, under the definition "
          "of Eq.~\\ref{eq:decision-accuracy} and its variants: a tied pair (on one side or both) counts one half, the pairs the "
          "reference ties are dropped, or only the pairs whose 1.7B difference exceeds one or two checkpoint standard "
          "deviations of the reference (over the last 20\\% of its run, detrended) or one or two seed standard "
          "deviations of a difference are kept. A value is the mean over the tasks above chance at the proxy and at "
          "1.7B, for benchmark accuracy, for accuracy and the bBPB twins together, and for the per-language bits per "
          "byte, which never tie. Pairs kept is the share of the comparable pairs a variant keeps, pooled over the "
          "proxy sizes. The task set moves across rows, since a cell needs three kept pairs: on accuracy it holds "
          "359--553 tasks per proxy under Eq.~\\ref{eq:decision-accuracy} and 243--387 under the strictest filter. A value is a mean over tasks, so it can differ from the pooled lines of Figures~\\ref{fig:app_rq02_decision_accuracy} and \\ref{fig:app_rq02_decisive_pairs}, which weigh a task by its pairs.",
          "tab:app_rq02_definition_sensitivity"),
         ("app_rq02_seed_uncertainty",
          "\\textbf{Two seeds of one design agree with each other about as often as a proxy agrees with the "
          "reference.} Left and middle: the replicated cells only, English at 175M, 600M and 1B and Russian at 1B, the deep "
          "cells with the baseline data trained with three seeds, pooled over tasks. Left: DA-size of the proxy at each of its three seeds against "
          "the 1.7B final checkpoint (filled: seed 1904). Middle: the agreement between two seeds' rankings of the "
          "same designs at one size. Right: DA-ckpt of the pairs formed by two seeds of one design, which have nothing to decide "
          "(seed null), against the real design pairs, over the evaluated checkpoints of each size. Every task above "
          "chance, bBPB twins included.",
          "fig:app_rq02_seed_uncertainty"))),
    "rq02_permutation_null": (
        "Decision accuracy against a no-signal null", "app_rq02_permutation_null",
        None,
        None, 0),
    "rq02_decisive_pairs": (
        "Decision accuracy on decisive pairs", "app_rq02_decisive_pairs",
        None,
        None, 0),
    "rq02_da_vs_train_tokens": (
        "Decision accuracy against the training tokens of the language", "app_rq02_da_goal_multi_axes_bpb",
        None,
        None, 0),
    "rq03_noise_and_snr": (
        "Noise and SNR", "app_rq03_noise_and_snr",
        None,
        None, 0,
        (("app_rq03_flip_attribution",
          "\\textbf{At 600M, per-language BPB misses the 1.7B decision mostly on the depth pairs, which are not "
          "size-matched.} One row per design axis (single-axis pairs) and for every design pair (the multi-axis set, "
          "single-axis pairs included), split by whether the pair moves depth. A comparison is one design pair on one of the 50 per-language BPB tasks, between the "
          "final checkpoints of seed 1904 at 600M and at 1.7B. (a): the 600M difference exceeds one 600M seed standard "
          "deviation of the task, the median over the deep three-seed cells of the baseline data (the K = 50 cell alone for "
          "48 of the 50 languages). (b): its sign differs "
          "from the sign of the 1.7B difference. (b) without (a) means that the 600M comparison cannot distinguish the "
          "variants, (a) with (b) that the design choice has opposite measured effects at the two sizes. The last "
          "column counts the (a), (b) comparisons whose 1.7B difference also exceeds that deviation. A difference of two "
          "runs has a seed standard deviation of 1.41 times that of one run, so (a) admits about half of the pure "
          "seed differences. At that 1.41 threshold the last column falls from 64 to 13 for depth. The optimizer "
          "axis has no per-language BPB pair at both sizes.",
          "tab:app_rq03_flip_attribution"),)),
    "rq04_surrogates": (
        "SNR as a surrogate for decision accuracy", "app_rq04_surrogates",
        None,
        "highlights", 0,
        (("app_rq04_surrogates_by_scoring", rq04_by_scoring_caption, "fig:app_rq04_surrogates_by_scoring"),
         ("app_rq04_surrogates_mono_axis", rq04_mono_axis_caption, "fig:app_rq04_surrogates_mono_axis"))),
    "rq05_design_decisions": (
        "Design decisions", "app_rq05_design_decisions",
        None,
        None, 0),
    "rq06_language_transfer": (
        "Language transfer", "app_rq06_language_transfer",
        None,
        None, 0,
        (("app_rq06_cross_task_hellaswag", rq06_cross_task_caption, "fig:app_rq06_cross_task_hellaswag"),
         ("app_rq06_family_transfer", rq06_family_caption, "fig:app_rq06_family_transfer"),
         ("app_rq06_family_lift", rq06_lift_caption, "fig:app_rq06_family_lift"))),
    "rq07_external_frameworks": (
        "Agreement with DataDecide", "app_rq07_external_frameworks",
        None,
        None, 0),
    "rq08_subset_selection": (
        "Subset selection", "app_rq08_subset_selection",
        None,
        None, 0),
    "rq09_benchmark_design": (
        "Benchmark design", "app_rq09_benchmark_design",
        None,
        None, 0,
        (("app_rq09_benchmark_characteristics", rq09_table_caption, "tab:app_rq09_benchmark_characteristics"),
         ("app_rq09_design_da_correlation", rq09_correlation_caption, "fig:app_rq09_design_da_correlation"),
         ("app_rq09_design_da_by_level", rq09_levels_caption, "fig:app_rq09_design_da_by_level"),
         ("app_rq09_design_da_quadrant", rq09_quadrant_caption, "fig:app_rq09_design_da_quadrant"),
         ("app_rq09_item_count", rq09_item_count_caption, "tab:app_rq09_item_count"),
         ("app_rq09_wave_robustness", rq09_waves_caption, "tab:app_rq09_wave_robustness"))),
    "rq10_size_generalisation": (
        "Size generalisation to 3B", "app_rq10_size_generalisation",
        None,
        None, 0,
        (("app_rq10_reference_choice",
          "\\textbf{Neither the seed of the reference nor a 3B reference at $K=30$ lifts benchmark accuracy far above "
          "chance.} DA-size of benchmark accuracy from the final checkpoint of each proxy, pooled over decisions (top block: three quarters "
          "of the tasks are bBPB twins of English tasks, since $K=1$ trains English only; bottom block: no bBPB twins, "
          "which the 3B model at $K=30$ lacks). "
          "Top: against the 1B final checkpoint of each of its three seeds, on the four designs trained with three "
          "seeds at 1B (deep, $K \\in \\{1, 2, 30\\}$ with scheme A and $K=30$ with scheme B), and against the 1.7B "
          "final checkpoint on the same designs. Bottom: against the 3B final checkpoint, on the decisions between "
          "the one 3B model at $K=30$ and the five other 3B models, then the same decisions against 1.7B, and every "
          "decision among the six 3B models. Within a block a task counts only where every column scores it. A task "
          "is above chance at the proxy and at the reference and has at least three pairs.",
          "tab:app_rq10_reference_choice"),)),
    "rq11_evaluation_recipe": (
        "Evaluation recipe", "app_rq11_evaluation_recipe",
        None,
        None, 0,
        (("app_rq11_recipe_by_benchmark", rq11_benchmarks_caption, "fig:app_rq11_recipe_by_benchmark"),)),
    "rq12_above_chance_items": (
        "Above-chance items", "app_rq12_above_chance_items",
        None,
        None, 0),
    "rq13_english_only": (
        "English-only models", "app_rq13_english_only",
        None,
        None, 0,
        (("app_rq13_english_scaling_regimes", rq13_regimes_caption, "fig:app_rq13_english_scaling_regimes"),
         ("app_rq13_english_scaling_regimes_l1_accuracy", rq13_regimes_l1_caption,
          "fig:app_rq13_english_scaling_regimes_l1_accuracy"),
         ("app_rq13_english_scaling_regimes_l1_accuracy_tagged", rq13_regimes_tagged_caption,
          "fig:app_rq13_english_scaling_regimes_l1_accuracy_tagged"))),
    "rq14_proxy_item_selection": (
        "Item selection from the proxies alone", "app_rq14_proxy_item_selection",
        None,
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
        "multi-axis design pairs, pooled over decisions (matching over comparable pairs, summed over tasks). Left: the "
        "benchmark tasks above chance and their bBPB twins. Right: the per-language bits per byte. The "
        "band is the 90% leave-one-variant-out jackknife interval, the dotted line marks 0.9 and the circle marks the smallest proxy above it. "
        "The dashed segment joins the 1B proxy to the reference itself.",
    "rq02_da_vs_train_tokens":
        "Mean DA-goal over the bits per byte of the 50 trained languages against the training tokens of the language "
        "that the checkpoint had seen (log scale). There is one line per proxy size, a darker blue for a larger size, "
        "and one point per evaluated tenth of the run. The x position is the geometric mean over languages, the bars give the standard error over languages and the "
        "dotted line marks 0.75.",
    "rq02_permutation_null":
        "DA-size of the final checkpoint of each proxy size against the 1.7B final checkpoint, on multi-axis design "
        "pairs, pooled over decisions, for the benchmark tasks with their bBPB twins (left) and the per-language bits "
        "per byte (middle). The black line is the observed DA-size, the dashed grey line the mean of the shuffled null and the grey band its 95% range. The "
        "dotted line marks 0.5. Right: the share of (task, proxy size) cells above their own null at q < 0.05.",
    "rq02_decisive_pairs":
        "DA-size of the final checkpoint of each proxy size against the 1.7B final checkpoint, on multi-axis design "
        "pairs, pooled over decisions, for the benchmark tasks with their bBPB twins (left) and the per-language bits per "
        "byte (middle). One line per threshold: "
        "every comparable pair (black) and the pairs whose 1.7B gap exceeds one or two seed standard deviations of a "
        "difference (dark and light blue). The dotted line marks 0.5. Right: the share of the comparable pairs each threshold keeps, "
        "solid for the bits per byte and dashed for the benchmarks.",
    "rq03_noise_and_snr":
        "Median over the (K, task) cells of a size of the absolute effect of each design decision (seed 1904, final "
        "checkpoint) over the noise, on the benchmarks (accuracy tasks with their rf and bBPB twins), the "
        "per-language bits per byte, the macro average of the bits per byte and the training loss. Top row: over the "
        "seed noise, only at the nine replicated cells (175M, 600M, 1B), so activation has only 1B (K = 30) and the "
        "optimizer a single cell (the training loss at 1B). Bottom row: over the checkpoint noise of the baseline "
        "run, 90M--1.7B. Muon has only the training loss at 1B and 1.7B. There is one column per decision. The dashed "
        "line at 1 is where the effect equals the noise (log scale). A pure seed re-roll would reach a median of "
        "1.15 in the top row.",
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
        "Median SNR at 1.7B over the per-language tasks of each benchmark family that clears the above-chance gate, "
        "ranked, on a log scale. The number above a bar counts its tasks. The colour gives the number of answer options "
        "(grey: LAMBADA scores a last word and BBH mixes two to eight).",
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


# a README's citation of paper floats by stem: "(figure: a)", "(tables: a, b)", "(figure: a, table: b)"
CITE = re.compile(r"\((?:figures?|tables?):\s*([^()]*)\)")


def cite(inner):
    """The LaTeX of a CITE token's stems: Figure~\\ref{..}, Figures~\\ref{..} and~\\ref{..}, Table~\\ref{..}."""
    groups = []
    for stem in re.split(r",\s*(?:(?:figures?|tables?):\s*)?|\s+and\s+", inner.strip()):
        if stem not in LABELS:
            print(f"  WARNING: ({inner}) names {stem}, which is no float of the paper")
            continue
        kind = "Table" if LABELS[stem].startswith("tab:") else "Figure"
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(LABELS[stem])
        else:
            groups.append((kind, [LABELS[stem]]))
    refs = lambda ls: ", ".join(f"\\ref{{{l}}}" for l in ls[:-1]) + f" and~\\ref{{{ls[-1]}}}"
    return " and ".join(f"{k}~\\ref{{{ls[0]}}}" if len(ls) == 1 else f"{k}s~{refs(ls)}" for k, ls in groups)


def to_tex(s):
    """Markdown inline (code, bold, italics, 10^k, CITE tokens) to LaTeX, everything else escaped."""
    out, held = [], []
    for i, part in enumerate(re.split(r"`([^`]*)`", clean_md(s))):
        if i % 2:
            out.append("\\texttt{" + escape(part) + "}")
            continue
        part = CITE.sub(lambda m: held.append(cite(m[1])) or f"\x05{len(held) - 1}\x06", part)
        part = re.sub(r"\bL(?=\d+\b)|\bL\b(?![-'’])", "K", part)          # the language count is K in the paper
        part = re.sub(r"(\d+)\^(\d+)", lambda m: f"\x00{m[1]}\x01{m[2]}\x02", part)
        part = re.sub(r'"([^"]*)"', "\x03\\1\x04", part)
        part = escape(part).replace("\x03", "``").replace("\x04", "''")
        part = part.replace("\x00", "$").replace("\x01", "^{").replace("\x02", "}$")
        part = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", part)
        out.append(re.sub("\x05(\\d+)\x06", lambda m: f"({held[int(m[1])]})" if held[int(m[1])] else "", part))
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
    """The first Markdown bullet list after line `start`, one string per bullet; it
    ends at the next "## " heading or "**Follow-ups**" line."""
    bullets, cur = [], None
    for l in lines[start + 1:]:
        if l.startswith("- "):
            cur = [l[2:]]
            bullets.append(cur)
        elif cur is not None and l.startswith("  ") and l.strip():
            cur.append(l.strip())
        elif (l.startswith("## ") or l.startswith("**Follow-ups**")) and bullets:
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
        "size. A hollow point is 1.0 because it compares a ranking with itself.")


def rq00_majority_caption(d):
    """The caption of app_rq00_majority_chance (rq00 majority_chance.py), its
    numbers read from the figure's CSV `d`: panel a per task, panel b the
    trained tasks above chance per size under each chance level."""
    t, p = d[d["panel"] == "a"], d[d["panel"] == "b"].set_index(["chance", "size"])
    n = int(p["n_tasks"].max())
    flip = t["flipped_sizes"].notna()
    fixed = t["fixed_options"].astype(str) == "True"
    c = lambda chance, size: int(p.at[(chance, size), "n_pass"])
    return (
        "\\textbf{A majority-class chance level removes passes mainly through answer order, which log-likelihood scoring "
        "cannot exploit.} The majority baseline of a task is the accuracy of always choosing its most frequent gold "
        "option position, read from the evaluated items. (a) Uniform chance against the majority baseline for the "
        f"{len(t)} trained tasks with a chance level. Orange marks the {int(flip.sum())} tasks above chance at some size "
        "under the uniform level and at chance there under the larger of the two, with a benchmark label where a "
        "benchmark has at least five such tasks. Circles are tasks whose answer positions carry the same label on every item (a letter or a yes or no "
        "answer), where a model with a constant preference can reach the majority baseline. Triangles are tasks whose "
        "answers change per item. (b) Trained tasks above chance per size under the uniform level, under the majority "
        "level, and under the majority level applied to the labelled tasks only. Of the "
        f"{n} trained tasks, {c('uniform', '90M')} are above chance at 90M and {c('uniform', '1.7B')} at 1.7B under the "
        f"uniform level, {c('majority', '90M')} and {c('majority', '1.7B')} under the majority level, and "
        f"{c('majority_fixed', '90M')} and {c('majority_fixed', '1.7B')} when only the labelled tasks change. "
        f"Of the flipped tasks, {int((flip & fixed).sum())} are labelled. Seed-1904 runs of every language setting, "
        "ladder and data build, with the Wilson rule of Appendix~\\ref{app:chance-threshold}.")


# paper figure stem -> (its caption from its CSV, label): one float page each, sections/rqs/<stem>.tex
FIGURE_PAGES = {"app_rq02_bbpb": (rq02_bbpb_caption, "fig:app_rq02_bbpb"),
                "app_rq00_majority_chance": (rq00_majority_caption, "fig:app_rq00_majority_chance")}

# paper stem -> label of every float a README can cite: the main figures, the extras, FIGURE_PAGES, FIXED
LABELS = {**{p[1]: f"fig:app_{f}" for f, p in PAGES.items()},
          **{stem: label for p in PAGES.values() for stem, _, label in (p[5] if len(p) > 5 else ())},
          **{stem: label for stem, (_, label) in FIGURE_PAGES.items()}, **{stem: f"fig:{stem}" for stem in FIXED}}


def float_page(stem, caption, label):
    """A full-width figure on a float page of its own. The figure takes at most
    0.8 of the text height, which leaves room for a caption of ten lines: a float
    taller than the page runs into the bottom margin, which ACL's format check
    rejects. The page block caps its figure at 0.36 for the same reason. A stem
    copied as TeX is a generated table (in ../tables/), set in a table* float instead and scaled down
    to the text width when wider. Every caption sits below its figure or table. A callable
    `caption` is built from the figure's CSV, so its numbers follow each refresh."""
    caption = plain(caption(pd.read_csv(FIGURES[stem][0].with_suffix(".csv"))) if callable(caption) else caption)
    if FIGURES[stem][1] == ("tex",):
        return "\n".join(["\\begin{table*}[p]", "\\centering", "\\small",
                          "\\begin{adjustbox}{max width=\\textwidth}", f"\\input{{tables/{stem}}}", "\\end{adjustbox}",
                          f"% source: {FIGURES[stem][0].relative_to(ANALYSIS.parents[2])}.tex",
                          f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}", ""])
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


def readme_setup(lines):
    """The README's setup: the bullet list right under a line `**Setup.**` as one paragraph, each
    bullet's bold label dropped, or None when the README has none (then PAGES gives it)."""
    i = next((i for i, l in enumerate(lines) if l.strip() == "**Setup.**"), None)
    if i is None:
        return None
    bullets = []
    for l in lines[i + 1:]:
        if l.startswith("- "):
            bullets.append(l[2:])
        elif l.startswith("  ") and l.strip() and bullets:
            bullets[-1] += " " + l.strip()
        elif bullets or l.strip():
            break
    return plain(to_tex(" ".join(re.sub(r"^\*\*[^*]+\*\*\s*", "", b) for b in bullets)))


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
    setup = readme_setup(lines) or plain(setup)
    for _, _, ref in extras:
        if f"\\ref{{{ref}}}" not in setup + "".join(findings):
            print(f"  WARNING app_{folder}: neither the setup nor the findings cite {ref}, add a (figure: stem) or "
                  "(table: stem) to the README bullet it supports")
    return "\n".join([
        "\\clearpage",
        "\\twocolumn[{%",
        "\\rqparent",
        f"\\subsection{{{title}}}",
        f"\\label{{app:{folder}}}",
        "",
        "\\noindent\\begin{minipage}[t]{0.485\\textwidth}",
        "\\vspace{0pt}",
        f"\\paragraph{{Research Question.}} \\emph{{{plain(to_tex(question(lines)))}}}",
        "",
        f"\\paragraph{{Setup.}} {setup}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.485\\textwidth}",
        "\\vspace{0pt}",
        "\\begin{rqfinding}",
        "\n\\\\\n\\\\\n".join(findings),
        "\\end{rqfinding}",
        "\\end{minipage}",
        *main,
        *([] if main else ["\\vspace{\\baselineskip}"]),     # under the figure it would only push the page over
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
        out = RQS / f"app_{folder}.tex"
        if not out.is_file() or out.read_text() != tex:
            out.write_text(tex)
        print(f"\\input{{sections/rqs/app_{folder}}}")
    for stem, (caption, label) in FIGURE_PAGES.items():
        tex = float_page(stem, caption, label)
        out = RQS / f"{stem}.tex"
        if not out.is_file() or out.read_text() != tex:
            out.write_text(tex)
        print(f"\\input{{sections/rqs/{stem}}}")
    tex, out = fixed_populations(), RQS / "app_fixed_populations.tex"
    if not out.is_file() or out.read_text() != tex:
        out.write_text(tex)
    print("\\input{sections/rqs/app_fixed_populations}")
    for f in missing:
        print(f"  NO PAGE for {f}: add it to PAGES")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
