"""Write one appendix page per analysis folder from its README.

For every src/signal-and-noise/analysis/rqNN_<name>/ this writes
documents/paper/sections/app_rqNN_<name>.tex: \\clearpage, a section title, the
question, the setup, the key finding and the main figure (the paper copy
make_rq_figures.py puts in this folder). The question and the key finding are
read from the README on every run, so the pages follow each refresh; the title,
the setup and the figure are fixed below (the setup names no counts that a
refresh moves).

    question     the README's **Question.** paragraph or the first paragraph of
                 its "Research question" / "Question" section (or, failing that,
                 the first paragraph), cut after its last "?"; without one, the
                 question in the title line
    findings     an rqfinding box (defined in main.tex) of at most three
                 findings: the opening key finding, i.e. the **Key finding.**
                 paragraph if the README has one, else the bold lead and first
                 sentence of a bullet after the figure's image in the README
                 (the folder's first image when the figure is not drawn there),
                 then the next Key-findings bullets (the rest of that list, or
                 the first "Key findings" list after the paragraph), each cut
                 to its bold lead and first sentence
    extras       optional sixth PAGES element: further figures after the main
                 one, each (paper figure stem, caption in LaTeX, label), on a
                 float page of their own

The text is double-blind: links, file names and rule numbers are dropped.
plain() gives every written sentence the appendix punctuation: no dash and no
";" outside math and code. A numeric range keeps its en dash (90M--1.7B).
The paper writes the number of languages as K where the analysis writes L
(L8, "(task, L)"); to_tex renames it outside code spans, the README keeps L.

    python make_rq_appendix.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

from make_rq_figures import ANALYSIS, FIGURES, HERE

SECTIONS = HERE.parent / "sections"
# rq12 scores DA against both 1.7B truths once its per-item store run has written the full-truth columns
_RQ12_CSV = ANALYSIS / "rq12_above_chance_items" / "pretraining" / "predictivity" / "above_chance_items_snr_paper.csv"
RQ12_TWO_TRUTHS = _RQ12_CSV.is_file() and "rho_snr_da_size_full_truth" in _RQ12_CSV.read_text().split("\n", 1)[0].split(",")



def readings(stem, solid, dashed, value="da", **where):
    """The plotted CSV of a paper figure with two readings of each line, one
    row per point: the two values and their difference (solid - dashed)."""
    t = pd.read_csv(FIGURES[stem][0].with_suffix(".csv"))
    for k, v in where.items():
        t = t[t[k] == v]
    keys = [c for c in ("panel", "population", "line", "x") if c in t and c not in where]
    w = t.pivot_table(index=keys, columns="reading", values=value).dropna(subset=[solid, dashed])
    return w.assign(diff=w[solid] - w[dashed]).reset_index()


SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"]


def fixed_impact(stem, line, what, fmt=".2f", value="value", unit="tasks", **where):
    """One sentence on what one task set per line does to a line of a paper
    figure: its fixed and moving values at the first and last point, the task
    counts, and the largest shift (fixed - moving) along it."""
    t = pd.read_csv(FIGURES[stem][0].with_suffix(".csv"))
    for k, v in dict(where, line=line).items():
        t = t[t[k] == v]
    w = t.pivot_table(index="x", columns="reading", values=[value, "n_tasks"])
    w = w.loc[sorted(w.index, key=lambda x: SIZES.index(x) if x in SIZES else float(x))].dropna()
    f, m, n = w[(value, "fixed")], w[(value, "moving")], w[("n_tasks", "moving")]
    d = f - m
    x = d.abs().idxmax()
    first, last = w.index[0], w.index[-1]
    num = lambda v, sign="": f"{v:{sign}{fmt}}".replace("-", "−")      # a minus sign, not a hyphen
    verdict = ("so part of the line's movement comes from its changing tasks" if abs(d[x]) > (m.max() - m.min()) / 4
               else "small against the line's own range, so its shape is not a population effect")
    return (f"On {what}, the {int(w[('n_tasks', 'fixed')].iloc[0])} {unit} it has at every point read "
            f"{num(f.iloc[0])} at {first} and {num(f.iloc[-1])} at {last}, against {num(m.iloc[0])} and "
            f"{num(m.iloc[-1])} on the {unit} of each point ({int(n.min())}--{int(n.max())}). The largest "
            f"shift is {num(d[x], '+')} at {x}, {verdict}.")


# paper figure -> (its fixed-task twin's stem, the label of the figure it is a twin of, what the twin
# changes, the impact sentence's arguments)
FIXED = {
    "rq2": ("app_fixed_rq2", "fig:rq2", "the reliable tasks of each panel",
            dict(line="all pairs", what="the DA-size line over every single-axis pair", value="da", panel="DA-size")),
    "app_decision_accuracy": ("app_fixed_decision_accuracy", "fig:app_rq02_decision_accuracy", "the tasks above chance",
                              dict(line="all benchmarks", what="the benchmark line", value="da")),
    "app_design_decisions": ("app_fixed_design_decisions", "fig:app_rq05_design_decisions",
                             "the (task, language setting) cells of each line",
                             dict(line="temperature / benchmark", what="the benchmark line of the temperature decision",
                                  unit="cells", panel="da_size")),
    "app_size_generalisation": ("app_fixed_size_generalisation", "fig:app_rq10_size_generalisation",
                                "the tasks of each channel (the left panel is not redrawn)",
                                dict(line="benchmarks to 3B", what="the benchmark line against the 3B reference",
                                     panel="3B")),
    "app_evaluation_recipe": ("app_fixed_evaluation_recipe", "fig:app_rq11_evaluation_recipe",
                              "the tasks of each variant", dict(line="All variants", what="the pooled line")),
    "app_above_chance_items": ("app_fixed_above_chance_items", "fig:app_rq12_above_chance_items",
                               "the tasks of each ordering",
                               dict(line="items_then_gate", what="the median SNR of the items-then-gate ordering",
                                    panel="snr_median")),
    "app_english_only": ("app_fixed_english_only", "fig:app_rq13_english_only", "the English tasks of each comparison",
                         dict(line="every L > 1", what="the deep cells' mean gap in points to every other language setting",
                              fmt=".1f", panel="deep: mean gap")),
}


def fixed_populations():
    """The appendix of the fixed-task twins: one figure per paper figure whose task set moves along x."""
    out = ["% Generated by documents/paper/figures/make_rq_appendix.py from the fixed-task CSVs; do not edit.",
           "\\clearpage", "\\section{Fixed task sets}", "\\label{app:fixed_populations}", "",
           plain("A line of a mean over tasks against model size averages, at each size, the tasks that have a value "
                 "there. The above-chance gate keeps more tasks at larger sizes, so a line can rise or fall because "
                 "its tasks changed. Each figure below redraws a paper figure on one task set per line, the tasks "
                 "that have a value at every point of that line (solid), with the original line dashed behind it."),
           ""]
    for orig, (stem, ref, tasks, kw) in FIXED.items():
        if not FIGURES[stem][0].with_suffix(".csv").is_file():
            continue
        caption = (f"Figure~\\ref{{{ref}}} on one task set per line, {tasks} that have a value at every point (solid), "
                   "against the original (dashed). " + escape(fixed_impact(stem, **kw)))
        out += ["\\begin{figure}[p]", "\\centering",
                f"\\includegraphics[width=\\textwidth,height=0.9\\textheight,keepaspectratio]{{figures/{stem}.png}}",
                f"% source: {FIGURES[stem][0].relative_to(ANALYSIS.parents[2])}.png",
                f"\\caption{{{plain(caption)}}}",
                f"\\label{{fig:{stem}}}", "\\end{figure}", ""]
    return "\n".join(out)


def crossfit_captions():
    rq2 = readings("app_rq2_crossfit", "cross-fitted", "in-sample (half)", panel="DA-size", line="all pairs")
    twin = readings("app_decision_accuracy_crossfit", "cross-fitted", "in-sample (half)", population="all benchmarks")
    worst = rq2.loc[rq2["diff"].abs().idxmax()]
    shift = f"{worst['diff']:+.2f}".replace("-", "$-$")
    return (
        ("app_rq2_crossfit",
         "Figure~\\ref{fig:rq2} with the reliable tasks chosen out of sample. The verdict (DA above 0.66 on the size "
         "or the checkpoint axis) is decided on one half of the design pairs and the panels are read on the other half, "
         "over 20 random splits stratified by the axes a pair moves, in both directions (solid, with the 5--95\\% range "
         "over splits). The dashed lines choose and read the tasks on the same half, so the gap between the two is "
         "the selection effect alone. Out of sample, DA-size over every single-axis pair changes by up to "
         f"{shift} (at {worst['x']}). Lines with too few pairs per half are not drawn.",
         "fig:app_rq2_crossfit"),
        ("app_decision_accuracy_crossfit",
         "Figure~\\ref{fig:app_rq02_decision_accuracy} with the reliable benchmark tasks, chosen out of sample and in "
         "sample on the same half of the pairs (the same splits). The two readings differ by at most "
         f"{twin['diff'].abs().max():.2f} on the multi-axis pairs.",
         "fig:app_decision_accuracy_crossfit"))


# folder -> title, paper figure stem, setup, (README image the finding follows, which bullet)[, extras]
PAGES = {
    "rq00_gate_and_curves": (
        "The above-chance gate", "app_chance_share",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. A run clears chance on a "
        "task when the one-sided 95\\% Wilson lower bound of its accuracy over the task's items is above the chance "
        "level. A (task, size) cell is above chance when at least half of the runs of that size that train the task's "
        "language clear it. Tasks without a chance level (bits per byte) are not gated.",
        None, 0),
    "rq00_task_reformulation": (
        "Task reformulation", "app_chance_reformulation",
        "We compare the letter-format multiple-choice benchmarks (Belebele, Global-MMLU, INCLUDE) with two "
        "reformulated twins of every task. The first twin (RF) drops the answer letters and scores each option as a "
        "continuation. The second twin (LLM-RF) uses an LLM to rewrite the items as cloze statements. We use the same "
        "above-chance gate and the same runs as for the gate itself. We pair each original task with its twin and "
        "compare the two with a McNemar test.",
        None, 1),
    "rq00_chance_vs_train_tokens": (
        "Above chance against the tokens of the language seen", "app_chance_vs_train_tokens",
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
        (("app_benchmark_size_curves",
          "Benchmark accuracy against model size. For each benchmark family, we plot the final-checkpoint accuracy "
          "of every design against non-embedding parameters. This accuracy is the mean over the tasks in the "
          "languages the design trains on. There is one line per (number of languages, ladder, data build, seed), "
          "90M--1.7B, for every seed and data build. Colour gives the number of trained languages, line "
          "width the ladder and line style the data build. The dotted red line is chance. The curves are not "
          "gated, so a family at chance stays visible.",
          "fig:app_size_scaling"),
         ("app_benchmark_curves",
          "Benchmark accuracy along training. For each benchmark family, we plot the accuracy of every run at the "
          "ten evaluated tenths of the run. This accuracy is the mean over the tasks in the languages the run trains "
          "on. Training progress is given in Chinchilla multiples of the run's token budget. The figure covers "
          "every seed and data build, 90M--1.7B. Colour gives the model size, line width the ladder and "
          "line style the data build. The dotted red line is chance. The curves are not gated.",
          "fig:app_training_scaling"))),
    "rq02_decision_accuracy": (
        "Decision accuracy across sizes", "app_decision_accuracy",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. DA-size is the share of "
        "design pairs that the final checkpoint of a proxy orders in the same way as the final checkpoint of the "
        "1.7B reference. Every task is above chance at the proxy and at 1.7B. A pair may differ on any number of "
        "design axes.",
        None, 0, crossfit_captions()),
    "rq02_permutation_null": (
        "Decision accuracy against a no-signal null", "app_permutation_null",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B, on pairs that may differ on any "
        "number of design axes. For every (task, proxy size) cell with at least three comparable pairs, we shuffle "
        "the proxy's final scores across the design variants 1000 times and score each shuffle against the unchanged "
        "1.7B ranking with the same sign rule, ties included. A cell's p-value is the share of shuffles that match at "
        "least as many pairs, corrected over the cells with the Benjamini--Hochberg procedure. The lines pool the "
        "matching pairs over the comparable pairs of the tasks above chance at the proxy and at 1.7B, and the band is "
        "the 95\\% range of the pooled shuffles. The benchmark line includes the bBPB variants at the sizes where "
        "they were computed (175M, 350M and 1B).",
        None, 0),
    "rq02_decisive_pairs": (
        "Decision accuracy on decisive pairs", "app_decisive_pairs",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B, on pairs that may differ on any "
        "number of design axes. A pair is decisive for a task when its gap at 1.7B exceeds $k \\sqrt{2}$ times the "
        "task's seed standard deviation, for $k \\in \\{1, 2\\}$. The seed standard deviation is the median, over the "
        "baseline cells trained with three seeds (175M, 600M and 1B), of the standard deviation of the final score. "
        "There are no replicates at 1.7B, so we assume that seed noise does not grow with size. Every line reads "
        "the same accuracy tasks: those with a seed standard deviation and at least three pairs above the larger "
        "threshold, above chance at the proxy and at 1.7B.",
        None, 0),
    "rq02_da_vs_train_tokens": (
        "Decision accuracy against the tokens of the language seen", "app_da_goal_multi_axes_bpb",
        "We use the bits per byte of 50 languages, one task each. Design pairs are formed among the variants that "
        "train the language (at least three per language). We use every data build at seed 1904, the proxies from "
        "90M--1B and the early checkpoints of the 1.7B run. A point is the mean over languages of DA-goal (a "
        "checkpoint against the 1.7B final checkpoint) at one size and one tenth of the run. We place it at the "
        "number of tokens of the language seen.",
        None, 0),
    "rq03_noise_and_snr": (
        "Noise and SNR", "app_noise_and_snr",
        "We use the deep cells with the baseline data that were trained with three seeds (175M and 600M at "
        "$K \\in \\{1, 2, 50\\}$, 1B at $K \\in \\{1, 2, 30\\}$). For each (size, $K$, task) cell, we divide the "
        "absolute effect of each design decision on the final score by a noise. The seed noise is the sample "
        "standard deviation across seeds. The checkpoint noise is the detrended standard deviation over the last "
        "20\\% of the run. Cells at chance are left out.",
        None, 0),
    "rq04_surrogates": (
        "SNR as a surrogate for decision accuracy", "app_surrogates",
        "Seed-1904 runs of every cell, 90M--1.7B, on the tasks above chance. For each of 22 SNR definitions, "
        "we compute the Pearson $r$ between $\\log_{10}$ SNR and decision accuracy over the tasks of one language. "
        "We keep the languages with at least three tasks and average over the 50 trained languages. We do this for "
        "DA-size (proxy final against 1.7B final) and for DA-ckpt (the early checkpoints of a proxy against its "
        "final checkpoint).",
        "highlights", 0),
    "rq05_design_decisions": (
        "Design decisions", "app_design_decisions",
        "We study four single-axis interventions against the baseline (deep, scheme A, $T = 1$). They are depth "
        "(deep vs shallow at equal non-embedding size), data scheme A vs B, data scheme A vs C, and sampling "
        "temperature $T = 1$ vs $T = 3$. We use every seed. We report the DA-size of the proxies 90M--1B and "
        "the DA-ckpt of the earlier checkpoints of the 1.7B run, both against the 1.7B final checkpoint. We read "
        "them on the bits per byte of the languages that both levels train and on the benchmarks above chance.",
        None, 0),
    "rq06_language_transfer": (
        "Language transfer", "app_language_transfer",
        "We read the language-list decision (scheme A vs B at $K \\in \\{8, 15, 30\\}$, single-axis pairs, every "
        "seed) on the bits per byte of 100 evaluation languages. We group the languages by whether both lists, one "
        "list or neither list trains them. When neither list trains a language, we also check whether they train "
        "its script. We report the DA-size of the proxies 90M--1B and the DA-ckpt of the checkpoints of the "
        "1.7B run, against the 1.7B final checkpoint.",
        None, 0),
    "rq07_external_frameworks": (
        "Agreement with DataDecide", "app_external_frameworks",
        "We compare our 1B rung (seed 1904, every cell and data build) with the 1B rung of DataDecide (25 data "
        "recipes). We use the English tasks that both evaluate and that clear the above-chance gate on our side. "
        "SNR is the average absolute deviation over noise, and we compare it on a log scale.",
        None, 0),
    "rq08_subset_selection": (
        "Subset selection", "app_subset_selection",
        "Seed-1904 runs of every cell, 90M--1.7B. For each multilingual benchmark and size, we take the "
        "language subset with the highest SNR, that is, the best prefix of the languages ranked by SNR. We compare "
        "it with the 95th percentile of 100 random subsets of the same size. Each per-language task is gated above "
        "chance.",
        None, 0),
    "rq09_benchmark_design": (
        "Benchmark design", "app_benchmark_design",
        "Seed-1904 runs of every cell. SNR is the mean pairwise distance over noise at the 1.7B reference. For each "
        "family, we take the median over its per-language tasks. We keep the benchmark families that clear the "
        "above-chance gate. We group the families by curation, source, task format, answer-option count and "
        "passage use, and we test the groups with a Kruskal-Wallis test.",
        None, 0),
    "rq10_size_generalisation": (
        "Size generalisation to 3B", "app_size_generalisation",
        "The deep cells trained at 3B (seed 1904) and the same families at 1.7B. Each (family, task) is scored at "
        "both rungs. Left: for each benchmark, the share of its tasks above chance at 1.7B and at 3B. We show the "
        "benchmarks with at least five tasks whose share moves. Right: DA-size from the final checkpoints of 90M--1B "
        "to the 3B final checkpoint and to the 1.7B final checkpoint, on the same single-axis decisions. "
        "Benchmark accuracy (tasks above chance at the proxy, at 1.7B and at 3B) and per-language bits per byte "
        "are pooled separately. The bands are 90\\% leave-one-family-out jackknife bands.",
        None, 0),
    "rq11_evaluation_recipe": (
        "Evaluation recipe", "app_evaluation_recipe",
        "Seed-1904 runs of every cell. We report the DA-size of the proxies 90M--1B against the 1.7B final "
        "checkpoint, on multi-axis pairs. Each benchmark is read in up to six ways. The items are used as "
        "published, as RF or as LLM-RF, and each version is scored by accuracy or by the bits per byte of the gold "
        "answer (bBPB). The accuracy variants count on the tasks above chance at the proxy and at the reference. A "
        "bBPB variant has no chance level and is read against its original's 1.7B accuracy, so it counts where that "
        "accuracy is above chance at the reference. The pooled line holds both kinds.",
        None, 0),
    "rq12_above_chance_items": (
        "Above-chance items", "app_above_chance_items",
        "Seed-1904 runs of every cell, 90M--1.7B, final checkpoints. Every benchmark-language task keeps only "
        "the items that its 1.7B runs answer above chance, after the above-chance gate. We compare it with the full "
        "task. The selection reads the reference by design, so the gains are an upper bound, not a held-out "
        "estimate." + (" Decision accuracy is scored against two truths, the 1.7B ranking on the full task and on "
                       "the kept items, and the gap between them is the part of the gain that comes from re-scoring "
                       "the reference." if RQ12_TWO_TRUTHS else ""),
        None, 0),
    "rq14_proxy_item_selection": (
        "Item selection from the proxies alone", "app_proxy_item_selection",
        "Seed-1904 runs of every cell, final checkpoints, on pairs that may differ on any number of design axes. "
        "For each benchmark-language task, we split the design families into two halves. On one half, we rank the "
        "items by their discrimination at 600M and 1B: the correlation of an item's correctness with the run's task "
        "score, both centred within each size. We keep the top half of the items and score the DA-size of the "
        "proxies 90M--1B on the other half's pairs, against the 1.7B ranking on the full task. We then swap the "
        "halves and average. Random subsets of the same size, drawn from the same items, are the baseline. The "
        "selection never reads the 1.7B runs, so its DA is a held-out estimate that is available before the "
        "reference is trained. Tasks are above chance at the proxy and at 1.7B.",
        None, 0),
    "rq13_english_only": (
        "English-only models", "app_english_only",
        "We compare the monolingual English cells ($K = 1$) with the baseline cells of the same depth at every "
        "other language setting. These other cells give half of their tokens to English. We use seed 1904, final "
        "checkpoints and sizes 90M--1.7B, on the English accuracy tasks above chance at each size.",
        None, 0),
}

# a caption for a figure the README does not draw (otherwise the README's alt text)
CAPTIONS = {"rq00_gate_and_curves": "Share of each benchmark's language tasks above chance, by the smallest size "
                                    "from which the gate holds",
            "rq03_noise_and_snr": "Median absolute effect of each design decision over the seed noise and over the "
                                  "checkpoint noise, per size (1 = the effect equals the noise)",
            "rq13_english_only": "English benchmark scores of the English-only cells against the multilingual cells "
                                 "of the same size"}

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


def extra_figure(stem, caption, label):
    return "\n".join([
        "\\begin{figure}[p]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth,height=0.9\\textheight,keepaspectratio]{{figures/{stem}.png}}",
        f"% source: {FIGURES[stem][0].relative_to(ANALYSIS.parents[2])}.png",
        f"\\caption{{{plain(caption)}}}",
        f"\\label{{{label}}}",
        "\\end{figure}",
        "",
    ])


def page(folder, title, stem, setup, kf_image, kf_bullet, extras=()):
    lines = (ANALYSIS / folder / "README.md").read_text().splitlines()
    alt = next(re.match(r"!\[([^\]]*)\]", l)[1] for l in lines if l.startswith("!["))
    src = FIGURES[stem][0]
    # a `_paper` twin the README does not show is read through the figure it is a twin of
    shown = src.name if any(f"/{src.name}.png)" in l for l in lines) else src.name.removesuffix("_paper")
    image = (kf_image or shown, kf_bullet)
    caption = CAPTIONS.get(folder) or next((re.match(r"!\[([^\]]*)\]", l)[1] for l in lines
                    if l.startswith("![") and f"/{shown}.png)" in l), None) or alt
    caption = re.sub(r",\s*(the )?paper (figure|copy)$", "", caption.rstrip("."))
    label = f"fig:app_{folder}"
    findings = [plain(to_tex(k)) for k in key_findings(lines, image)]
    findings[0] += f" (Figure~\\ref{{{label}}}.)"
    return "\n".join([
        f"% Generated by documents/paper/figures/make_rq_appendix.py from the {folder} README; do not edit.",
        "\\clearpage",
        f"\\section{{{title}}}",
        f"\\label{{app:{folder}}}",
        "",
        "\\begin{rqfinding}",
        "\n\n".join(f"{i}. {k}" for i, k in enumerate(findings, 1)),
        "\\end{rqfinding}",
        "",
        f"\\paragraph{{Question.}} {plain(to_tex(question(lines)))}",
        "",
        f"\\paragraph{{Setup.}} {plain(setup)}",
        "",
        "\\begin{figure}[h]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{figures/{stem}.png}}",
        f"% source: {src.relative_to(ANALYSIS.parents[2])}.png",
        f"\\caption{{{plain(to_tex(caption))}.}}",
        f"\\label{{{label}}}",
        "\\end{figure}",
        "",
    ] + [extra_figure(*e) for e in extras])


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
    tex, out = fixed_populations(), SECTIONS / "app_fixed_populations.tex"
    if not out.is_file() or out.read_text() != tex:
        out.write_text(tex)
    print("\\input{sections/app_fixed_populations}")
    for f in missing:
        print(f"  NO PAGE for {f}: add it to PAGES")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
