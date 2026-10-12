"""Generated appendix tables: everything the setup appendices tabulate that moves
when the sweep or the evaluation suite does. Each table is read from the
configs the launcher and the eval watcher read, so it describes the grid that
is trained, not the one that was planned when the text was written.

    languages   sections/rqs/app_02_table_languages.tex (whole file, \\input by
                app_02_data_mixtures.tex): the FineWeb-2 languages of the
                validation set (VALIDATION_MANIFEST), the smallest trained
                setting of each scheme that contains them, and the benchmark
                families the harness offers for them
    grid        block in app_01_model_ladder.tex: the data builds x ladders,
                read from launch_trainings.DATA_SCHEMES / LADDERS / seeds_for
    seeds       block in app_01_model_ladder.tex: seeds per (size, L) cell
    ladder      block in app_01_model_ladder.tex: the per-ladder, per-size
                hyperparameters (hyperparams_*.json, at each rung's own batch)
    benchmarks  block in app_03_evaluation.tex: the evaluated benchmarks
                (configs/tasks.json `auto` group, `analysed_benchmarks`) with
                their language and task counts on the sweep's trained
                languages, the `discarded` tasks left out
    screened    block in app_03_evaluation.tex: the benchmarks screened at 600M, 1B
                and 1.7B and not analysed (`auto_probe` outside `auto`, and the
                `auto` benchmarks whose ladder evals have not landed) and those of
                the `discarded` tasks, per benchmark: trained languages, candidate
                and discarded tasks
    discarded   block in app_03_evaluation.tex: the screened probe tasks
                dropped at the gate (configs/tasks.json `discarded` group),
                per benchmark family
    nbenchmarks block in main.tex: the setup counts the prose quotes, written
                in the same run as the tables they count:
                \\nbenchmarks   rows of the benchmarks table's multilingual block
                \\nruns         grid runs, every cell with 3B and replicate
                               seeds included (the grid table's Total)
                \\nrunsfinished grid runs at their target iteration in the
                               ladder report
                \\nbenchanalysed, \\ntasksanalysed  the benchmarks table's rows and Total
                \\nbenchscreened, \\ntasksscreened, \\ntasksdiscarded  the screened table's
                               benchmarks that are not analysed, its tasks and
                               the discarded ones among them
                \\nbenchconsidered, \\ntasksconsidered  the two together
                \\ntasksKone ... \\ntasksKfifty  tasks a scheme-A cell is
                               evaluated on at K = 1, 2, 8, 15, 30, 50 (one
                               macro per LANG_SETTINGS entry, K spelled out)

A block sits between `% BEGIN generated: KEY (make_appendix_tables.py)` and
`% END generated: KEY`; text outside it is hand-written and kept.

    python make_appendix_tables.py
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SECTIONS = HERE.parent / "sections"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "pretrain"))
from evals.scripts.utils.configs import _TASK_LANG_ALIASES, load_tasks  # noqa: E402
import launch_trainings as lt  # noqa: E402

DATA = REPO / "src" / "pretrain" / "data"
LADDER_REPORT = REPO / "src" / "signal-and-noise" / "data" / "ladder-report" / "ladder_report.csv"
SCRIPT = Path(__file__).name
OUT_LANGUAGES = SECTIONS / "rqs" / "app_02_table_languages.tex"
# The per-language validation set score_bpb.py scores (its --manifest default): the languages
# the val column lists. It was built from the original 100-language list, before the K = 100
# swap-ins of language_sets_schemeA.json's FW_L100, so it is not that list.
VALIDATION_MANIFEST = Path("/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/"
                           "predictivity-data/validation.manifest.json")

# What each data build is, for a reader. A build missing here is printed under
# its own name, so a new scheme shows up in the table instead of vanishing.
BUILD_DESC = {
    "A": "resource-ranked lists",
    "AT3": "scheme-A lists, flattened",
    "B": "diversity-first lists",
    "ZH": "Chinese as second language",
    "ES": "Spanish as second language",
    "DCLMP": "DCLM without the edu filter",
    "FWEB": r"FineWeb (crawls $\le$ 2022)",
}
LADDER_DESC = {"deep": "deep", "shallow": "shallow", "swiglu": "deep + SwiGLU", "muon": "deep + Muon"}

# Benchmarks of the `auto` group: (name, capability, format, construction).
# Counts are computed; only these descriptive columns are curated. A group
# entry missing here is printed under its key.
BENCH_META = {
    "belebele": ("Belebele", "reading comprehension", "4-way MC", "HT (FLORES)"),
    "global_piqa": ("Global PIQA", "physical commonsense", "2-way completion, 4-way (parallel)",
                    "native, MT + post-edit (parallel)"),
    "multiblimp": ("MultiBLiMP", "grammatical acceptability", "minimal pairs", "auto (UD, UniMorph)"),
    "include_base_44": ("INCLUDE", "regional knowledge", "4-way MC", "native exams"),
    "include_v2_og": ("INCLUDE v2 RF", "regional knowledge", "4-way completion", "native exams"),
    "include_v2_en": ("INCLUDE v2 RF (English)", "regional knowledge", "4-way completion", "native, translated to en"),
    "global_mmlu": ("Global-MMLU", "world knowledge", "4-way MC", "HT + community"),
    "hellaswag": ("HellaSwag (Okapi)", "activity commonsense", "4-way completion", "MT"),
    "arc": ("ARC (Okapi)", "science QA", "4-way completion", "MT"),
    "arc_mt": ("ARC-Challenge MT", "science QA", "4-way completion", "MT"),
    "xnli": ("XNLI", "natural language inference", "3-way", "HT"),
    "xstorycloze": ("XStoryCloze", "narrative commonsense", "2-way ending", "HT"),
    "xcopa": ("XCOPA", "causal commonsense", "2-way", "HT"),
    "paws": ("PAWS-X", "paraphrase identification", "2-way", "HT"),
    "xwinograd": ("XWinograd", "coreference", "Winograd schemas", "native"),
    "lambada_openai_mt": ("LAMBADA multilingual", "last-word prediction", "completion", "MT"),
    "truthfulqa-multi_mc1": ("TruthfulQA-Multi (mc1)", "truthfulness", "MC", "HT"),
    "truthfulqa_mc2": ("TruthfulQA (mc2)", "truthfulness", "MC, probability mass", "native + MT"),
    "cultural_bench_easy": ("CulturalBench (easy)", "cultural knowledge", "4-way MC", "native (English)"),
    "cultural_bench_hard": ("CulturalBench (hard)", "cultural knowledge", "2-way (true/false)", "native (English)"),
    "blend_sample": ("BLEnD (sample)", "everyday cultural knowledge", "MC", "native"),
    "mmlu": ("MMLU", "world knowledge", "4-way MC", "native"),
    "commonsense_qa": ("CommonsenseQA", "commonsense", "5-way MC", "native"),
    "openbookqa": ("OpenBookQA", "science QA", "4-way completion", "native"),
    "mathqa": ("MathQA", "math word problems", "5-way completion", "native"),
    "toxigen": ("ToxiGen", "toxicity detection", "2-way", "auto (LLM-generated)"),
    "bbh_mcq": ("BBH (letter subtasks)", "reasoning", "letter MC", "native"),
    "bbh_cloze": ("BBH (two-way subtasks)", "reasoning", "2-way completion", "native"),
    "acp_bench_mcq": ("ACP-Bench (MC)", "planning", "4-way letter MC", "auto (PDDL)"),
    "acp_bench_cloze": ("ACP-Bench (boolean)", "planning", "yes/no completion", "auto (PDDL)"),
    "social_iqa": ("Social IQa", "social commonsense", "3-way completion", "crowd-sourced (English)"),
    "boolq": ("BoolQ", "reading comprehension", "yes/no", "native"),
    "piqa": ("PIQA", "physical commonsense", "2-way completion", "native"),
    "winogrande": ("WinoGrande", "coreference", "Winograd schemas", "native"),
    "multiblimp-extra": ("MultiBLiMP (extra languages)", "grammatical acceptability", "minimal pairs",
                         "auto (UD, UniMorph)"),
    "kobest": ("KoBEST", "Korean understanding", "2- and 4-way", "native"),
    "noreval": ("NorEval", "Norwegian commonsense, reading, grammar", "2- to 5-way", "native, HT"),
    "noreval_extra": ("NorEval (extra)", "Norwegian science QA, quiz, truthfulness, sentiment", "2- and 4-way",
                      "native, HT"),
}
TWIN_FORMAT = {"rf_": "cloze completion", "rfgm_": "statement continuation"}
# Tagged with several languages in the harness, but the text is English (the
# country or the source item's language is the tag), so not a multilingual row.
ENGLISH_TEXT = {"cultural_bench_easy", "cultural_bench_hard", "include_v2_en"}
SCREENED = frozenset(json.loads((REPO / "configs" / "tasks.json").read_text())["groups"]["auto_probe"])


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def replace_block(tex: Path, key: str, body: str) -> None:
    """Rewrite the generated block KEY of `tex`; the markers must exist."""
    begin = f"% BEGIN generated: {key} ({SCRIPT})"
    end = f"% END generated: {key}"
    text = tex.read_text()
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < i:
        raise SystemExit(f"{tex.name}: no `{begin}` ... `{end}` block")
    tex.write_text(text[: i + len(begin)] + "\n" + body.rstrip("\n") + "\n" + text[j:])
    print(f"wrote {tex.name}: {key}")


def fmt_list(xs) -> str:
    return ", ".join(str(x) for x in xs)


def and_list(xs, last=" and ") -> str:
    """a, b and c"""
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + last + xs[-1]


# --- the grid ---------------------------------------------------------------

def grid_runs() -> list[dict]:
    """Every run the grid trains: {build, ladder, size, L, seed}."""
    runs = []
    for ladder in lt.HYPERPARAMS:
        for c in lt.predictivity_cells(ladder=ladder):
            if ladder in lt.ladders_for(c["scheme"], c["size"], c["L"]):
                runs.append({"build": c["scheme"], "ladder": ladder, "size": c["size"],
                             "L": c["L"], "seed": c["seed"]})
    return runs


def finished(runs: list[dict]) -> tuple[int, list[str]]:
    """How many grid runs reached their target iteration in the ladder report."""
    if not LADDER_REPORT.exists():
        return -1, []
    last = {}
    with open(LADDER_REPORT) as f:
        for r in csv.DictReader(f):
            last[r["cell"]] = max(last.get(r["cell"], 0), int(float(r["iter"])))
    hp = {lad: json.loads(p.read_text())["configs"] for lad, p in lt.HYPERPARAMS.items()}
    done, missing = 0, []
    for r in runs:
        name = lt.exp_name(r["size"], r["L"], r["ladder"], r["seed"], r["build"])
        name = name if name.startswith("lm-") else "lm-" + name
        target = lt.cell_schedule(hp[r["ladder"]][r["size"]], r["size"])[0]
        if last.get(name, 0) >= target:
            done += 1
        else:
            missing.append(name.removeprefix("lm-"))
    return done, missing


def size_span(sizes) -> str:
    sizes = [s for s in lt.LADDER if s in sizes]
    contiguous = lt.LADDER[lt.LADDER.index(sizes[0]): lt.LADDER.index(sizes[-1]) + 1] == sizes
    return f"{sizes[0]}--{sizes[-1]}" if contiguous and len(sizes) > 2 else fmt_list(sizes)


def grid_table(runs: list[dict], done: int, missing: list[str]) -> str:
    order = list(lt.DATA_SCHEMES)
    rows, total, total_rep = [], 0, 0
    for build in order:
        for ladder in lt.HYPERPARAMS:
            rs = [r for r in runs if r["build"] == build and r["ladder"] == ladder]
            if not rs:
                continue
            cfg = lt.DATA_SCHEMES[build]
            Ls = sorted({r["L"] for r in rs})
            by_L = {L: {r["size"] for r in rs if r["L"] == L} for L in Ls}
            common = set.intersection(*by_L.values())
            extra = {s for v in by_L.values() for s in v} - common
            sizes = size_span(common)
            for s in [s for s in lt.LADDER if s in extra]:
                sizes += f" and {s} at $K \\in \\{{{fmt_list(L for L in Ls if s in by_L[L])}\\}}$"
            rep = sum(r["seed"] != 1904 for r in rs)
            total, total_rep = total + len(rs), total_rep + rep
            rows.append(" & ".join([esc(build), BUILD_DESC.get(build, esc(build)), cfg["letter"],
                                    f"{cfg['temp']:g}", LADDER_DESC.get(ladder, ladder),
                                    fmt_list(Ls), sizes, str(len(rs)), str(rep or "--")]) + r" \\")
    status = ("" if done < 0 else
              f" All {total} runs had finished at the ladder-report snapshot." if not missing else
              f" At the ladder-report snapshot, {done} of the {total} runs had finished. "
              f"The unfinished {'run is' if len(missing) == 1 else 'runs are'} "
              # the run names say L<k> for the language setting; the paper calls it K
              f"{esc(and_list([re.sub(r'(?<=-)L(?=[0-9])', 'K', m) for m in missing]))}.")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llcclllrr}", r"\toprule",
        r"Build & Data & $M$ & $T$ & Ladder & $K$ & Sizes & Runs & Repl. \\", r"\midrule",
        *rows, r"\midrule",
        f"Total & & & & & & & {total} & {total_rep} \\\\", r"\bottomrule", r"\end{tabular}}",
        r"\caption{\textbf{Training grid.} One row per data build and ladder. $M$ is the data scheme of the "
        r"build at its $K$: A is the baseline recipe, and B and C are the alternatives at that $K$. $T$ is the "
        r"sampling temperature of the build. $M$ and $T$ are the two design axes that a build sets. The ladders are "
        r"deep (the baseline), shallow (about twice the width-to-depth ratio), deep + SwiGLU (the activation "
        r"axis) and deep + Muon (the optimizer axis). The Muon ladder trains with Muon and the other three with "
        r"AdEMAMix. Runs counts the runs of every seed, and Repl.\ the replicate-seed "
        r"runs among them (Table~\ref{tab:grid})." + status + "}",
        r"\label{tab:schemes}", r"\end{table*}"])


def seeds_table(runs: list[dict]) -> str:
    sizes = list(lt.LADDER)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\toprule",
             "$K$ & " + " & ".join(sizes) + r" \\", r"\midrule"]
    for L in lt.LANG_SETTINGS:
        cells = []
        for s in sizes:
            n = sum(r["build"] == "A" and r["ladder"] == "deep" and r["size"] == s and r["L"] == L for r in runs)
            cells.append(str(n) if n else "--")
        lines.append(f"{L} & " + " & ".join(cells) + r" \\")
    by_triple = {}
    for s, (seeds, _) in lt.SEED_TRIPLES.items():
        by_triple.setdefault(tuple(seeds), []).append(s)
    seeds_txt = and_list([f"{and_list(seeds)} at {' and '.join(ss)}" for seeds, ss in by_triple.items()], ", and ")
    n_a = sum(r["build"] == "A" and r["ladder"] == "deep" for r in runs)
    # the authors' shorter caption (Overleaf, 2026-10-09): the ladders and builds that
    # train seed 1904 alone, and the scheme-B replicates, are told in the app_01 prose
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{\textbf{Random seeds.} Number of seeds per $(N, K)$ cell of the deep scheme-A grid "
              rf"({n_a} runs). The replicate seeds are "
              rf"{seeds_txt}. Each seed changes both the initialization and the data order.}}",
              r"\label{tab:grid}", r"\end{table}"]
    return "\n".join(lines)


def ladder_table(runs: list[dict]) -> str:
    trained = {(r["ladder"], r["size"]) for r in runs}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{",
             r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
             r"Ladder & Size & Layers & $d_{\mathrm{model}}$ & $d_{\mathrm{model}}/n_{\mathrm{layers}}$ & FFN & "
             r"FFN$/d_{\mathrm{model}}$ & Heads & $N$ (M) & $D$ (B) & Batch & Steps & LR ($10^{-3}$) & Ckpts \\",
             r"\midrule"]
    rows = {}
    for ladder, path in lt.HYPERPARAMS.items():
        cfgs = json.loads(path.read_text())["configs"]
        rows[ladder] = {}
        for size in [s for s in lt.LADDER if (ladder, s) in trained]:
            c = cfgs[size]
            iters = lt.cell_schedule(c, size)[0]
            rows[ladder][size] = [
                str(c["n_layers"]), f"{c['hidden_size']:,}",
                f"{c['hidden_size'] / c['n_layers']:.0f}", f"{c['ffn_hidden_size']:,}",
                f"{c['ffn_hidden_size'] / c['hidden_size']:.2f}",
                f"{c['num_attention_heads']}/{c['num_query_groups']}",
                f"{c['n_non_emb_params'] / 1e6:,.1f}", f"{c['predictivity']['train_tokens'] / 1e9:,.1f}",
                str(lt.cell_gbs(size)), f"{iters:,}", f"{c['lr'] * 1e3:.2f}", str(lt.n_checkpoints(iters))]
    # a ladder whose every row equals deep's (muon: same shapes and schedule, only the
    # optimizer differs) is named in the caption instead of repeating deep's rows
    same = [lad for lad in rows if lad != "deep"
            and all(rows["deep"].get(s) == r for s, r in rows[lad].items())]
    for i, ladder in enumerate(lad for lad in rows if lad not in same):
        if i:
            lines.append(r"\midrule")
        lines += [" & ".join([LADDER_DESC.get(ladder, ladder), size, *r]) + r" \\"
                  for size, r in rows[ladder].items()]
    same_txt = "".join(f" The {LADDER_DESC[lad].removeprefix('deep + ')} ladder has the same values as the deep one."
                       for lad in same)
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{\textbf{Pretraining hyperparameters per ladder and model size}, for the rungs that each "
              r"ladder trains. $N$ counts non-embedding parameters. The tied embedding adds "
              r"$d_{\mathrm{model}} \times 131{,}072$. $D = 100N$ is the token budget. Heads gives the query and "
              r"key-value head counts (grouped-query attention, four query heads per key-value head, head dimension "
              r"64). FFN is the feed-forward hidden dimension. The SwiGLU ladder keeps the deep shape and narrows the "
              r"FFN so that its three projections match the deep non-embedding count." + same_txt
              + r" Batch is the global batch in sequences of 4,096 tokens. The two smallest rungs train at their own batch and take proportionally "
              r"more steps for the same $D$. LR is the peak learning rate from the compute-based law at the run's "
              r"own budget. Ckpts is the number of saved checkpoints.}",
              r"\label{tab:ladder}", r"\end{table*}"]
    return "\n".join(lines)


# --- benchmarks -------------------------------------------------------------

def pending() -> frozenset[str]:
    """`auto` benchmarks whose ladder evals have not landed (analysis.utils.pending_benchmarks):
    counted with the screened ones, never as analysed, until they have."""
    sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
    from analysis.utils import pending_benchmarks
    return pending_benchmarks()


def analysed_benchmarks() -> list[str]:
    """The `auto` benchmarks less the `pending` ones, plus the screened benchmarks
    one of them pulls in by prefix (`tasks_for_benchmarks` matches `<benchmark>_…`):
    the watcher evaluates `noreval_extra` along every run with `noreval`, and the
    analysis reads it (`utils.considered_tasks`), so it is analysed, in a row of its own."""
    groups = json.loads((REPO / "configs" / "tasks.json").read_text())["groups"]
    auto = [b for b in groups["auto"] if b not in pending()]
    return auto + [b for b in groups["auto_probe"] if b not in auto and any(b.startswith(g + "_") for g in auto)]


def task_languages() -> tuple[dict, list[str]]:
    """The registered tasks less the `discarded` ones, which `tasks_for_benchmarks`
    skips, so no run is evaluated on them, and the analysed benchmarks."""
    disc = set(json.loads((REPO / "configs" / "tasks.json").read_text())["groups"]["discarded"])
    return {n: e for n, e in load_tasks().items() if n not in disc}, analysed_benchmarks()


def owner(bench: str, auto: list[str]):
    """The `auto` entry a task's benchmark belongs to: the longest matching key,
    so `arc_mt` is its own row rather than part of `arc`. A benchmark screened
    under its own name (`auto_probe`) is never folded into a prefix of it:
    `noreval_extra` is not part of `noreval` (`analysed_benchmarks` lists it)."""
    if bench not in auto and bench in SCREENED:
        return None
    keys = [g for g in auto if bench == g or bench.startswith(g + "_")]
    return max(keys, key=len) if keys else None


def canon(lang):
    return _TASK_LANG_ALIASES.get(lang, lang)


def trained_languages() -> set[str]:
    return set().union(*[lt.cell_languages(L, s) for s in lt.DATA_SCHEMES for L in lt.DATA_SCHEMES[s]["langs"]])


def benchmark_table() -> tuple[str, dict]:
    tasks, auto = task_languages()
    trained = trained_languages()
    langs, n_tasks, langs_tr = {}, Counter(), {}
    for name, e in tasks.items():
        if "pretraining" not in e.get("stages", []):
            continue
        g = owner(e.get("benchmark", ""), auto)
        lang = canon(e.get("language"))
        if g is None or lang in (None, "multi", "??"):
            continue
        langs.setdefault(g, set()).add(lang)
        if lang in trained:
            n_tasks[g] += 1
            langs_tr.setdefault(g, set()).add(lang)

    def row(g, name, task, form, cons):
        return " & ".join([esc(name), task, form, cons, str(len(langs.get(g, ()))),
                           str(len(langs_tr.get(g, ()))), str(n_tasks[g])]) + r" \\"

    originals = [g for g in auto if not g.startswith(tuple(TWIN_FORMAT))]
    multi = [g for g in originals if len(langs_tr.get(g, ())) > 1 and g not in ENGLISH_TEXT]
    mono = [g for g in originals if g not in multi]
    english = [g for g in mono if g in ENGLISH_TEXT or (langs_tr.get(g) or langs.get(g, set())) <= {"en"}]
    twins = [g for g in auto if g not in originals]
    order = list(BENCH_META)
    key = lambda g: (order.index(g) if g in order else len(order), g)  # noqa: E731
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{llllrrr}", r"\toprule",
             r"\textbf{Benchmark} & \textbf{Task} & \textbf{Format} & \textbf{Construction} & "
             r"\textbf{Langs} & \textbf{Trained} & \textbf{Tasks} \\", r"\midrule"]
    for block, gs in (("multilingual", multi), ("English text", english),
                      ("one language other than English", [g for g in mono if g not in english])):
        if not gs:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{block}}}}} \\")
        for g in sorted(gs, key=key):
            lines.append(row(g, *BENCH_META.get(g, (g, "", "", ""))))
        lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{reformulated variants}} \\")
    for g in sorted(twins, key=lambda g: (key(owner(re.sub(r"^rf(gm)?_", "", g), originals) or g), g)):
        prefix = "rfgm_" if g.startswith("rfgm_") else "rf_"
        base = g[len(prefix):]
        base_key = owner(base, originals) or base
        bname, task, _, cons = BENCH_META.get(base_key, (base, "", "", ""))
        label = f"{bname} ({'LLM-RF' if prefix == 'rfgm_' else 'RF'})"
        lines.append(row(g, label, task, TWIN_FORMAT[prefix], cons))
    total = sum(n_tasks.values())
    lines += [r"\midrule", f"Total & & & & & {len(trained)} & {total} \\\\", r"\bottomrule", r"\end{tabular}", "}",
              r"\caption{\textbf{The benchmarks we analyse}, evaluated throughout pretraining. Table~\ref{tab:screened-benchmarks} "
              r"lists the benchmarks we screened and do not analyse. Langs is the number of languages that the harness "
              r"registers for the benchmark. Trained is the number of them that the sweep trains (English and the "
              r"FineWeb-2 languages of every trained build). Tasks is the number of language tasks evaluated on "
              r"those languages. A model is evaluated only on the tasks of the languages in its training "
              r"mixture. The reformulated variants pose the same items as their original. RF scores the answer "
              r"strings instead of the option letters. LLM-RF rewrites each item into a statement stem with "
              r"short continuations. MC = multiple choice, QA = question answering, UD = Universal "
              r"Dependencies. Construction: HT = human or professional translation, MT = machine translation, "
              r"native = written in the language, auto = generated automatically.}",
              r"\label{tab:benchmark-families}", r"\end{table*}"]
    stats = {"multilingual": len(multi), "single": len(mono), "twins": len(twins), "tasks": total}
    return "\n".join(lines), stats


def discarded_table() -> str:
    """The `discarded` group per family: probe tasks not above chance at 1B nor
    at 1.7B (the rule-1 gate on the cells that trained the language), so they
    left `auto_probe` (plan/todos/probe-candidates-2026-10-08.md)."""
    cfg = json.loads((REPO / "configs" / "tasks.json").read_text())
    tasks, by = cfg["tasks"], {}
    for t in cfg["groups"]["discarded"]:
        by.setdefault(tasks[t]["benchmark"], []).append(canon(tasks[t]["language"]))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{llrl}", r"\toprule",
             r"\textbf{Benchmark} & \textbf{Languages} & \textbf{Tasks} & \textbf{Reason} \\", r"\midrule"]
    for b, langs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        name = re.sub(r" \(.*\)$", "", cfg["benchmarks"].get(b, {}).get("name", b))   # "C-Eval (Chinese)"
        lines.append(f"{esc(name)} & {fmt_list(sorted(set(langs)))} & {len(langs)} & not above chance at 1B and 1.7B \\\\")
    lines += [r"\midrule", f"Total & & {sum(map(len, by.values()))} & \\\\", r"\bottomrule", r"\end{tabular}", "}",
              r"\caption{\textbf{Screened benchmark tasks that we discarded.} A task is discarded when, both at 1B and "
              r"at 1.7B, fewer than half of the runs that trained its language have a one-sided 95\% Wilson lower "
              r"bound above chance. Tasks counts the discarded tasks of the benchmark, whose other tasks stay in the "
              r"screen. Languages are ISO 639-1 codes.}",
              r"\label{tab:discarded-benchmarks}", r"\end{table}"]
    return "\n".join(lines)


CONSIDERED_COUNTS = (REPO / "src" / "signal-and-noise" / "analysis" / "rq00_gate_and_curves" / "pretraining"
                     / "predictivity" / "gate_pass_counts_all_considered.csv")


def screened_sizes() -> str:
    """The sizes the screened benchmarks were evaluated at, as the caption says
    them ("600M, 1B and 1.7B", with 3B once the 3B probe jobs land), read off rq00's
    "all considered" table (panels.considered_panels): the sizes with a verdict for a screened task."""
    rows = list(csv.DictReader(CONSIDERED_COUNTS.open())) if CONSIDERED_COUNTS.is_file() else []
    sizes = list(dict.fromkeys(r["size"] for r in rows if r["group"] in ("probe", "discarded") and r["gated"]))
    return and_list(sizes or ["600M", "1B", "1.7B"])


def screened_table() -> tuple[str, dict]:
    """The benchmarks we screened and did not adopt (`auto_probe` outside `auto`)
    and those of the `discarded` tasks, on the sweep's trained languages: per
    benchmark its trained languages, its candidate tasks and its discarded ones.
    The gate verdicts are rq00's "all considered" variant (Figure
    app_rq00_chance_considered); this table reads the configs, and that
    variant's table only for the sizes the caption names (`screened_sizes`)."""
    sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
    from analysis.grids import paper_name
    groups = json.loads((REPO / "configs" / "tasks.json").read_text())["groups"]
    analysed = analysed_benchmarks()   # a pending `auto` benchmark is still screened
    probe, disc, trained = [b for b in groups["auto_probe"] if b not in analysed], set(groups["discarded"]), trained_languages()
    langs, n_tasks, n_disc = {}, Counter(), Counter()
    for name, e in load_tasks().items():
        lang, b = canon(e.get("language")), e.get("benchmark", "")
        g = owner(b, probe) or (b if name in disc else None)
        if g is None or lang not in trained or "pretraining" not in e.get("stages", []):
            continue
        langs.setdefault(g, set()).add(lang)
        (n_disc if name in disc else n_tasks)[g] += 1
    rows = sorted(langs, key=lambda g: paper_name(g).lower())
    def tab(part):
        return "\n".join([r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}[t]{lrrr}", r"\toprule",
                          r"\textbf{Benchmark} & \textbf{Langs} & \textbf{Tasks} & \textbf{Discarded} \\", r"\midrule",
                          *(f"{esc(paper_name(g))} & {len(langs[g])} & {n_tasks[g] or '--'} & {n_disc[g] or '--'} \\\\" for g in part),
                          r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    half = (len(rows) + 1) // 2
    stats = {"benchmarks": len(rows), "candidates": sum(n_tasks.values()), "discarded": sum(n_disc.values()),
             "languages": len(set().union(*langs.values())),
             # KoBEST and NorEval are analysed; only their discarded tasks are listed here
             "not_analysed": sum(g not in analysed for g in rows)}
    lines = [r"\begin{table*}[p]", r"\centering", r"\footnotesize",
             r"\begin{minipage}[t]{0.49\textwidth}\centering", tab(rows[:half]), r"\end{minipage}\hfill",
             r"\begin{minipage}[t]{0.49\textwidth}\centering", tab(rows[half:]), r"\end{minipage}",
             rf"\caption{{\textbf{{Benchmarks we screened and do not analyse}} ({stats['benchmarks']} benchmarks, "
             rf"{stats['candidates'] + stats['discarded']:,} tasks in {stats['languages']} trained languages). We evaluated "
             rf"them only on the final checkpoints of the {screened_sizes()} models. "
             + (rf"{and_list(sorted(esc(paper_name(g)) for g in rows if g in pending()))} joined the analysed benchmarks after "
                r"the screen and count here until their evaluation throughout pretraining is complete. "
                if any(g in pending() for g in rows) else "")
             + r"Langs is the number of trained languages a "
             r"benchmark covers. Tasks counts its language tasks on those languages that stayed candidates. Discarded "
             r"counts the tasks we dropped because, both at 1B and at 1.7B, fewer than half of the runs that trained the "
             r"language had a one-sided 95\% Wilson lower bound above chance (Table~\ref{tab:discarded-benchmarks}). "
             r"Figure~\ref{fig:app_rq00_chance_considered} gives the gate verdicts of every task. RF marks a reformulated "
             r"variant that scores the answer strings instead of the option letters.}",
             r"\label{tab:screened-benchmarks}", r"\end{table*}"]
    return "\n".join(lines), stats


# --- languages --------------------------------------------------------------

def languages_table() -> None:
    meta = {r["subset"]: r for r in csv.DictReader(open(DATA / "fineweb2-language-distribution.csv"))
            if r["split"] == "train"}
    # every validation language (the FW_L100 list was swapped after the validation set was built)
    full = [k.removeprefix("fineweb_") for k in json.loads(VALIDATION_MANIFEST.read_text()) if k != "dclm"]
    iso2 = json.loads((REPO / "configs" / "languages.json").read_text())["fineweb_iso2"]
    tasks, auto = task_languages()
    originals = [g for g in auto if not g.startswith(tuple(TWIN_FORMAT))]

    def first_L(letter, subset):
        """Smallest trained setting of a build with this scheme letter whose
        list contains the subset, or None."""
        Ls = [L for b, cfg in lt.DATA_SCHEMES.items() if cfg["letter"] == letter
              for L in cfg["langs"] if L >= 2 and subset in lt.cell_fineweb_subsets(L, b)]
        return min(Ls) if Ls else None

    def n_families(code):
        fams = set()
        for name, e in tasks.items():
            if "pretraining" in e.get("stages", []) and canon(e.get("language")) == code:
                g = owner(e.get("benchmark", ""), originals)
                if g:
                    fams.add(g)
        return len(fams)

    rows = []
    for subset in full:
        m = meta[subset]
        fam = m["family"].split(",")[0].split("(")[0].strip()
        la, lb = first_L("A", subset), first_L("B", subset)
        code = iso2.get(subset.split("_")[0])
        rows.append([subset, re.sub(r"\((\d+)-\)", r"(\1 onward)", m["name"]), m["script"], fam,
                     la if la else "val", lb or "--",
                     n_families(code) if code else 0, la or 1000])
    rows.append(["dclm", "English", "Latn", "Indo-European", 1, 1, n_families("en"), 0])
    rows.sort(key=lambda r: (r[7], full.index(r[0]) if r[0] in full else -1))
    rows = [r[:7] for r in rows]

    def tab(part):
        # max width: a long language name must not push the right half into the margin;
        # [t] puts the baseline on the top row, so the two halves align at their tops
        lines = [r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}[t]{llllrrr}", r"\toprule",
                 r"Subset & Language & Script & Family & $K_A$ & $K_B$ & Fam. \\", r"\midrule"]
        for r in part:
            lines.append(" & ".join(esc(x) for x in r) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
        return "\n".join(lines)

    c_builds = "".join(
        f" The scheme-C build at $K={L}$ trains {', '.join(meta[s]['name'] for s in lt.cell_fineweb_subsets(L, bld))}."
        for bld, cfg in lt.DATA_SCHEMES.items() if cfg["letter"] == "C" for L in sorted(cfg["langs"]) if L >= 2)
    half = (len(rows) + 1) // 2
    n_val = sum(r[4] == "val" for r in rows)
    body = "\n".join([
        r"\begin{table*}[p]", r"\centering", r"\tiny", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", tab(rows[:half]), r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", tab(rows[half:]), r"\end{minipage}",
        r"\caption{\textbf{The languages of the sweep.} $K_A$ gives the smallest language setting whose scheme-A "
        r"(resource-ranked) list contains the subset. $K_B$ gives the smallest trained setting of a scheme-B "
        r"build that contains it (the Chinese swap at $K=2$, the diversity-first lists at $K \in \{8, 15, 30\}$)."
        + c_builds + r" The lists are "
        r"nested, so a language is also trained at every larger setting of its scheme. "
        rf"The {n_val} languages marked val are in no training mixture and enter only the shared validation "
        r"set. Fam.\ counts the benchmark families of the pretraining suite (reformulated variants excluded) "
        r"that the harness offers for the language. English is the DCLM half of every mixture. Family names "
        r"are shortened to their top-level group.}",
        r"\label{tab:app-languages}",
        r"\end{table*}", ""])
    OUT_LANGUAGES.write_text(body)
    print(f"wrote {OUT_LANGUAGES.name} ({len(rows)} languages, {n_val} validation-only)")
    trained = [r for r in rows if r[4] != "val"]
    cov = Counter(r[6] for r in trained)
    print("families per trained language:", dict(sorted(cov.items())))


# LaTeX macro names take letters only: \ntasksK<K spelled out>.
K_WORDS = {1: "one", 2: "two", 8: "eight", 15: "fifteen", 30: "thirty", 50: "fifty"}


def tasks_per_K() -> dict[int, int]:
    """Tasks a scheme-A cell at each K is evaluated on (its trained languages)."""
    tasks, auto = task_languages()
    out = {}
    for L in lt.LANG_SETTINGS:
        langs = lt.cell_languages(L, "A")
        out[L] = sum(1 for e in tasks.values() if "pretraining" in e.get("stages", [])
                     and owner(e.get("benchmark", ""), auto) and canon(e.get("language")) in langs)
    return out


def main():
    runs = grid_runs()
    done, missing = finished(runs)
    replace_block(SECTIONS / "app_01_model_ladder.tex", "grid", grid_table(runs, done, missing))
    replace_block(SECTIONS / "app_01_model_ladder.tex", "seeds", seeds_table(runs))
    replace_block(SECTIONS / "app_01_model_ladder.tex", "ladder", ladder_table(runs))
    body, stats = benchmark_table()
    replace_block(SECTIONS / "app_03_evaluation.tex", "benchmarks", body)
    screened, sstats = screened_table()
    replace_block(SECTIONS / "app_03_evaluation.tex", "screened", screened)
    replace_block(SECTIONS / "app_03_evaluation.tex", "discarded", discarded_table())
    # written in the same run as the tables they count, so text and tables cannot disagree
    per_K = tasks_per_K()
    if unnamed := sorted(set(per_K) - set(K_WORDS)):
        raise SystemExit(f"K_WORDS has no name for K = {unnamed}")
    replace_block(SECTIONS / "main.tex", "nbenchmarks", "\n".join([
        f"\\newcommand{{\\nbenchmarks}}{{{stats['multilingual']} }}",
        f"\\newcommand{{\\nruns}}{{{len(runs)} }}",
        f"\\newcommand{{\\nrunsfinished}}{{{done if done >= 0 else '??'} }}",  # ?? = no ladder report
        *(f"\\newcommand{{\\ntasksK{K_WORDS[L]}}}{{{n:,} }}" for L, n in per_K.items()),
        # every benchmark considered: the analysed `auto` group, plus the screened ones and the discarded tasks
        f"\\newcommand{{\\nbenchanalysed}}{{{stats['multilingual'] + stats['single'] + stats['twins']} }}",
        f"\\newcommand{{\\ntasksanalysed}}{{{stats['tasks']:,} }}",
        f"\\newcommand{{\\nbenchscreened}}{{{sstats['not_analysed']} }}",
        f"\\newcommand{{\\ntasksscreened}}{{{sstats['candidates'] + sstats['discarded']:,} }}",
        f"\\newcommand{{\\ntasksdiscarded}}{{{sstats['discarded']:,} }}",
        f"\\newcommand{{\\nbenchconsidered}}{{{stats['multilingual'] + stats['single'] + stats['twins'] + sstats['not_analysed']} }}",
        f"\\newcommand{{\\ntasksconsidered}}{{{stats['tasks'] + sstats['candidates'] + sstats['discarded']:,} }}"]))
    languages_table()
    by = Counter((r["build"], r["ladder"]) for r in runs)
    print(f"grid: {len(runs)} runs ({sum(r['seed'] == 1904 for r in runs)} at seed 1904, "
          f"{sum(r['size'] != '3B' for r in runs)} at 90M-1.7B); "
          + ", ".join(f"{b}/{lad} {n}" for (b, lad), n in by.items()))
    print(f"benchmarks: {stats['multilingual']} multilingual, {stats['single']} single-language, "
          f"{stats['twins']} reformulated variants, {stats['tasks']} tasks on trained languages")
    print(f"runs: {done} of {len(runs)} finished")
    for L, n in per_K.items():
        print(f"  scheme-A L{L}: {n} tasks")


if __name__ == "__main__":
    main()
