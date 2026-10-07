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

The text is double-blind: links, file names and rule numbers are dropped.
The paper writes the number of languages as K where the analysis writes L
(L8, "(task, L)"); to_tex renames it outside code spans, the README keeps L.

    python make_rq_appendix.py
"""
import re
import sys
from pathlib import Path

from make_rq_figures import ANALYSIS, FIGURES, HERE

SECTIONS = HERE.parent / "sections"

# folder -> title, paper figure stem, setup, (README image the finding follows, which bullet)
PAGES = {
    "rq00_gate_and_curves": (
        "The above-chance gate", "app_chance_share",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. A run clears chance on a task "
        "when the one-sided 95\\% Wilson lower bound of its accuracy over the task's items exceeds the chance level; "
        "a (task, size) cell is above chance when at least half of the size's runs that train the task's language "
        "clear it. Tasks without a chance level (bits per byte) are not gated.",
        None, 0),
    "rq00_task_reformulation": (
        "Task reformulation", "app_chance_reformulation",
        "The letter-format multiple-choice benchmarks (Belebele, Global-MMLU, INCLUDE) against two reformulated "
        "twins of every task: the answer letters dropped and each option scored as a continuation (RF), and the items "
        "rewritten as cloze statements by an LLM (LLM-RF). The same above-chance gate and runs as the gate itself; "
        "original and twin are paired on the task and compared with a McNemar test.",
        None, 1),
    "rq00_chance_vs_train_tokens": (
        "Above chance against the tokens of the language seen", "app_chance_vs_train_tokens",
        "A cell is one (task, size, language setting) read at each of the ten evaluated tenths of the run, placed at "
        "the tokens of the task's language the checkpoint had seen (the language's share of the mixture $\\times$ the "
        "size's budget $\\times$ the tenth). Seed-1904 runs of every data build and ladder, 90M--1.7B, "
        "$K \\in \\{1, 2, 8, 15, 30, 50\\}$; the above-chance gate per run.",
        None, 0),
    "rq01_scaling_predictability": (
        "Scaling predictability", "rq1",
        "Log-linear fits of each task's final score against model size, one series per task and language setting, on "
        "the sizes where the task is above chance (at least three); deep baseline-data cells at seed 1904, 90M--1.7B. "
        "Each task is one point, the median over its language settings.",
        None, 0),
    "rq02_decision_accuracy": (
        "Decision accuracy across sizes", "app_decision_accuracy",
        "Seed-1904 runs of every language setting, ladder and data build, 90M--1.7B. DA-size is the share of design "
        "pairs that a proxy's final checkpoint orders like the 1.7B reference's final checkpoint; every task is above "
        "chance at the proxy and at 1.7B, and a pair may differ on any number of design axes.",
        None, 0),
    "rq02_da_vs_train_tokens": (
        "Decision accuracy against the tokens of the language seen", "app_da_goal_multi_axes_bpb",
        "The bits per byte of 50 languages, one task each; design pairs among the variants that train the language "
        "(at least three per language), every data build at seed 1904, proxies 90M--1B and the 1.7B run's own early "
        "checkpoints. A point is the mean over languages of DA-goal (a checkpoint against the 1.7B final) at one size "
        "and tenth of the run, placed at the tokens of the language seen.",
        None, 0),
    "rq03_noise_and_snr": (
        "Noise and SNR", "app_noise_and_snr",
        "The deep baseline-data cells trained with three seeds (175M and 600M at $K \\in \\{1, 2, 50\\}$, 1B at "
        "$K \\in \\{1, 2, 30\\}$). Per (size, $K$, task) cell, the absolute effect of each design decision on the final "
        "score over the seed noise (sample standard deviation across seeds) or the checkpoint noise (detrended standard "
        "deviation over the last 20\\% of the run); cells at chance are left out.",
        None, 0),
    "rq04_surrogates": (
        "SNR as a surrogate for decision accuracy", "app_surrogates",
        "Seed-1904 runs of every cell, 90M--1.7B, tasks above chance. For each of 22 SNR definitions, the Pearson $r$ "
        "between $\\log_{10}$ SNR and decision accuracy over a language's tasks (languages with at least three tasks), "
        "averaged over the 50 trained languages, for DA-size (proxy final against 1.7B final) and DA-ckpt (a proxy's "
        "early checkpoints against its final).",
        "highlights", 0),
    "rq05_design_decisions": (
        "Design decisions", "app_design_decisions",
        "Four single-axis interventions against the deep, scheme-A, $T = 1$ baseline: depth (deep vs shallow at equal "
        "non-embedding size), data scheme A vs B and A vs C, and sampling temperature $T = 1$ vs $T = 3$. Every seed; "
        "DA-size of the 90M--1B proxies and DA-ckpt of the 1.7B run's earlier checkpoints, both against the 1.7B final, "
        "read on the bits per byte of the languages both levels train and on the benchmarks above chance.",
        None, 0),
    "rq06_language_transfer": (
        "Language transfer", "app_language_transfer",
        "The language-list decision (scheme A vs B at $K \\in \\{8, 15, 30\\}$, single-axis pairs, every seed) read on "
        "the bits per byte of 100 evaluation languages, grouped by whether both lists, one list or neither train the "
        "language and, if neither, whether they train its script. DA-size of the 90M--1B proxies and DA-ckpt of the "
        "1.7B run's checkpoints, against the 1.7B final.",
        None, 0),
    "rq07_external_frameworks": (
        "Agreement with DataDecide", "app_external_frameworks",
        "Our 1B rung (seed 1904, every cell and data build) against DataDecide's 1B rung (25 data recipes), on the "
        "English tasks both evaluate that clear the above-chance gate on our side; SNR as the average absolute "
        "deviation over noise, compared on a log scale.",
        None, 0),
    "rq08_subset_selection": (
        "Subset selection", "app_subset_selection",
        "Seed-1904 runs of every cell, 90M--1.7B; per multilingual benchmark and size, the language subset with the "
        "highest SNR (best prefix of languages ranked by SNR) against the 95th percentile of 100 random subsets of the "
        "same size; each per-language task gated above chance.",
        None, 0),
    "rq09_benchmark_design": (
        "Benchmark design", "app_benchmark_design",
        "Seed-1904 runs of every cell; SNR as mean pairwise distance over noise at the 1.7B reference, the median over "
        "each family's per-language tasks, for the benchmark families that clear the above-chance gate; families "
        "grouped by curation, source, task format, answer-option count and passage use, tested with Kruskal--Wallis.",
        None, 0),
    "rq10_size_generalisation": (
        "Size generalisation to 3B", "app_size_generalisation",
        "The deep cells trained at 3B (seed 1904) and the same families at 1.7B, each (family, task) scored at both "
        "rungs. Left: per benchmark, the share of its tasks above chance at 1.7B and at 3B, for the benchmarks with at "
        "least five tasks whose share moves. Right: DA-size from the final checkpoints of 90M--1B to the 3B final and to "
        "the 1.7B final on the same single-axis decisions, benchmark accuracy (tasks above chance at the proxy, 1.7B and "
        "3B) and per-language bits per byte pooled separately, with 90\\% leave-one-family-out jackknife bands.",
        None, 0),
    "rq11_evaluation_recipe": (
        "Evaluation recipe", "app_evaluation_recipe",
        "Seed-1904 runs of every cell; DA-size of the 90M--1B proxies against the 1.7B final, multi-axis pairs, tasks "
        "above chance at the proxy and the reference. Each benchmark is read in up to six ways: as published, RF or "
        "LLM-RF, each scored by accuracy or by the gold answer's bits per byte (bBPB).",
        None, 0),
    "rq12_above_chance_items": (
        "Above-chance items", "app_above_chance_items",
        "Seed-1904 runs of every cell, 90M--1.7B, final checkpoints. Every benchmark-language task keeps only the "
        "items its 1.7B runs answer above chance, after the above-chance gate, and is compared with the full task. The "
        "selection reads the reference by design, so the gains are an upper bound, not a held-out estimate.",
        None, 0),
    "rq13_english_only": (
        "English-only models", "app_english_only",
        "The monolingual-English cells ($K = 1$) against the same-depth baseline cells of every other language setting, "
        "which give English half of their tokens; seed 1904, final checkpoints, 90M--1.7B, on the English accuracy "
        "tasks above chance at each size.",
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


def page(folder, title, stem, setup, kf_image, kf_bullet):
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
    findings = [to_tex(k) for k in key_findings(lines, image)]
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
        f"\\paragraph{{Question.}} {to_tex(question(lines))}",
        "",
        f"\\paragraph{{Setup.}} {setup}",
        "",
        "\\begin{figure}[h]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{figures/{stem}.png}}",
        f"% source: {src.relative_to(ANALYSIS.parents[2])}.png",
        f"\\caption{{{to_tex(caption)}.}}",
        f"\\label{{{label}}}",
        "\\end{figure}",
        "",
    ])


def main():
    folders = sorted(p.name for p in ANALYSIS.glob("rq*") if (p / "README.md").is_file())
    missing = [f for f in folders if f not in PAGES]
    for folder in [f for f in PAGES if f in folders]:
        tex = page(folder, *PAGES[folder])
        out = SECTIONS / f"app_{folder}.tex"
        if not out.is_file() or out.read_text() != tex:
            out.write_text(tex)
        print(f"\\input{{sections/app_{folder}}}")
    for f in missing:
        print(f"  NO PAGE for {f}: add it to PAGES")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
