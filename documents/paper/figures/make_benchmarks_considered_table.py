"""One row per benchmark the study considered (analysed, screened, discarded):
Langs, Trained, Tasks and Formats, for the paper appendix.

    cd src/signal-and-noise && PYTHONPATH=$PWD:$PWD/../../src SNR_LADDER_DIR=$PWD/data/ladder-report \
        python ../../documents/paper/figures/make_benchmarks_considered_table.py

Writes documents/paper/tables/app_benchmarks_considered.tex. Reads the cells that
make_considered_figures.py writes (src/signal-and-noise/data/considered/).

The task population is `analysis.utils.considered_tasks()` (the one behind the
"all considered" figures) on the trained languages; a row is a base benchmark,
its rf_/rfgm_ twins folded in. Langs/Trained reuse the computation of
make_appendix_tables.benchmark_table (registered pretraining tasks of the row,
`canon` languages, `trained_languages()`), discarded tasks included.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

WT = Path(__file__).resolve().parents[3]
OUT = WT / "documents" / "paper" / "tables" / "app_benchmarks_considered.tex"
# the considered figures' cells (task, random_baseline): a task whose chance level is not
# 1/K has a number of options that varies across items (tasks.json's n_options is one item's)
CELLS = WT / "src" / "signal-and-noise" / "data" / "considered" / "considered_cells.csv"
sys.path.insert(0, str(WT / "documents" / "paper" / "figures"))
import make_appendix_tables as mat  # noqa: E402
from analysis.grids import base, paper_name  # noqa: E402
from analysis.utils import FAMILY_MERGE, considered_tasks  # noqa: E402
from evals.scripts.utils.configs import load_languages  # noqa: E402
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402

CFG = json.loads((WT / "configs" / "tasks.json").read_text())
TASKS = CFG["tasks"]
REGISTERED = mat.load_tasks()
_chance = pd.read_csv(CELLS, usecols=["task", "random_baseline"]).drop_duplicates("task").set_index("task")["random_baseline"]
VARIABLE_K = {t for t, c in _chance.items() if TASKS[t].get("n_options") and abs(c - 1 / TASKS[t]["n_options"]) > 0.005}

# paper_name falls back to the key for these
NAMES = {"global_piqa": "Global PIQA"}

# Formats of tasks without an `example` in tasks.json, resolved from the harness
NO_EXAMPLE = {
    **dict.fromkeys(["boolq", "copal_id", "headqa", "kobest", "m_truthfulqa_mc1", "m_truthfulqa_mc2", "oall_acva",
                     "oall_alghafa", "oall_mt", "piqa", "ibero_copa", "ibero_nli", "ibero_paraphrase",
                     "ibero_truthfulqa", "truthfulqa_mc1", "truthfulqa-multi_mc2"], "cloze"),
    # WinoGrande scores the continuation after each filled-in option, as XWinograd does; MultiBLiMP (extra)
    # uses MultiBLiMP's minimal-pair template: both completion, like their siblings that have an example
    **dict.fromkeys(["winogrande", "multiblimp-extra"], "completion"),
    **dict.fromkeys(["arabicmmlu", "careqa", "m_mmlu", "mmmlu", "oall_arabic_mmlu", "oall_exams"], "letter"),
}


def no_example_format(task: str, bench: str) -> str | None:
    """The mixed-format benchmarks, per prompt template (harness YAMLs):
    EVALITA-LLM prompts 3-4 score the option letters, 1-2 and 5-6 the answer
    texts; NorOpenBookQA / NRK-Quiz-QA p2-p3 the letters, p0/p1/p4 the texts;
    NorTruthfulQA-MC (mc1_targets.choices) and NoReC (negativ/positiv) the texts."""
    if bench == "evalita_llm_extra":
        return "letter" if re.search(r"prompt-[34]$", task) else "cloze"
    if bench == "french_bench_extra":      # fquadv2_bool lists A/B (its RF twin removes them); the rest score words
        return "letter" if task == "french_bench_fquadv2_bool" else "cloze"
    if bench == "noreval_extra":
        return "letter" if re.match(r"(noropenbookqa|nrk_quiz_qa)_.*_p[23]$", task) else "cloze"
    return NO_EXAMPLE.get(bench)


def task_format(task: str) -> str | None:
    """letter: the options are the labels (A, B, ... or 1, 2, ...) listed in the
    prompt; cloze: the options are answer texts continuing a shared prompt;
    completion: no shared prompt (each option a whole sentence) or a single /
    generated continuation."""
    e = TASKS[task]
    ex = e.get("example")
    if e.get("n_options") is None or e.get("metric") == "exact_match":
        return "completion"
    if not ex:
        return no_example_format(task, e["benchmark"])
    opts = {re.sub(r"[().]", "", o).strip() for o in ex["options"]}
    k = len(ex["options"])
    if opts in ({chr(65 + i) for i in range(k)}, {str(i + 1) for i in range(k)}):
        return "letter"
    return "completion" if not (ex.get("prompt") or "").strip() else "cloze"


def k_text(ks) -> str:
    """4, 2--10: the range of the option counts; `var.` only when no task has a fixed one."""
    ks = sorted(k for k in ks if k != "var")
    return "var." if not ks else str(ks[0]) if len(ks) == 1 else f"{ks[0]}--{ks[-1]}"


def strip_twin(name: str) -> str:
    return re.sub(r"^rf(gm)?_", "", name)


# --- rows -------------------------------------------------------------------
groups = CFG["groups"]
trained = mat.trained_languages()
considered = {t: g for t, g in considered_tasks().items()
              if mat.canon(TASKS[t].get("language")) in trained and "pretraining" in TASKS[t].get("stages", [])}
# a benchmark registered in two parts (FAMILY_MERGE) is one row
merged = lambda b: FAMILY_MERGE.get(b, b)
KEYS = {merged(base(b)) for b in groups["auto"] + groups["auto_probe"]} | {merged(base(TASKS[t]["benchmark"])) for t in groups["discarded"]}


def row_of(bench: str):
    """The row of a benchmark field: its twin prefix stripped, then the longest
    row key it equals or extends (`owner` of make_appendix_tables)."""
    b = merged(base(bench))
    keys = [k for k in KEYS if b == k or b.startswith(k + "_")]
    return max(keys, key=len) if keys else None


rows = defaultdict(lambda: {"groups": set(), "tasks": set(), "all": 0, "fmt": defaultdict(set), "twins": set(),
                            "langs": set(), "undetermined": []})
for t, g in considered.items():
    b = TASKS[t]["benchmark"]
    r = rows[row_of(b)]
    r["groups"].add(g)
    r["all"] += 1
    r["tasks"].add(strip_twin(t))
    if m := re.match(r"^(rfgm|rf)_", b):
        r["twins"].add(m.group(1))
        continue
    f = task_format(t)
    if f is None:
        r["undetermined"].append(t)
    else:
        r["fmt"][f].add("var" if t in VARIABLE_K else TASKS[t].get("n_options"))
assert None not in rows, "a considered task without a row"
for name, e in REGISTERED.items():      # Langs: the registered pretraining tasks of the row
    k = row_of(e.get("benchmark", ""))
    lang = mat.canon(e.get("language"))
    if k in rows and "pretraining" in e.get("stages", []) and lang not in (None, "multi", "??"):
        rows[k]["langs"].add(lang)

# --- citations ----------------------------------------------------------------
OKAPI, ARC, TQA, OALL = "TODO-okapi", "TODO-arc", "TODO-truthfulqa", "TODO-oall"
CITE = {
    "belebele": ["bandarkar_belebele_2024"], "global_mmlu_full": ["singh_global_2024"],
    "include_base_44": ["romanou_include_2024"], "include_v2_en": ["romanou_include_2024"],
    "include_v2_og": ["romanou_include_2024"], "hellaswag": [OKAPI, "zellers_hellaswag_2019"],
    "arc": [OKAPI, ARC], "arc_mt": ["TODO-arc_mt", ARC], "xnli": ["conneau_xnli_2018"],
    "xcopa": ["ponti_xcopa_2020"], "paws": ["yang_paws-x_2019"], "cultural_bench_easy": ["chiu_culturalbench_2025"],
    "cultural_bench_hard": ["chiu_culturalbench_2025"], "blend_sample": ["myung_blend_2025"],
    "mmlu": ["hendrycks_measuring_2021"], "winogrande": ["sakaguchi_winogrande_2019"],
    "m_mmlu": [OKAPI, "hendrycks_measuring_2021"], "mmmlu": ["TODO-mmmlu", "hendrycks_measuring_2021"],
    "m_truthfulqa_mc1": [OKAPI, TQA], "m_truthfulqa_mc2": [OKAPI, TQA], "truthfulqa_mc1": [TQA],
    "truthfulqa_mc2": [TQA], "truthfulqa-multi_mc1": ["TODO-truthfulqa_multi"],
    "truthfulqa-multi_mc2": ["TODO-truthfulqa_multi"], "multiblimp-extra": ["TODO-multiblimp"],
    "bbh_mcq": ["TODO-bbh"], "bbh_cloze": ["TODO-bbh"], "acp_bench_mcq": ["TODO-acpbench"],
    "acp_bench_cloze": ["TODO-acpbench"], "noreval_extra": ["TODO-noreval"], "toksuite_math": ["TODO-toksuite"],
    "toksuite_stem": ["TODO-toksuite"], "evalita_llm_extra": ["TODO-evalita_llm"],
    "french_bench_extra": ["TODO-french_bench"], "oall_exams": [OALL, "hardalov_exams_2020"],
    "oall_arabic_mmlu": [OALL, "hendrycks_measuring_2021"], "oall_alghafa": [OALL, "TODO-alghafa"],
    "oall_mt": [OALL, "TODO-alghafa"], "oall_acva": [OALL, "TODO-acva"],
    "ibero_cola": ["TODO-iberobench"], "ibero_copa": ["TODO-iberobench"], "ibero_nli": ["TODO-iberobench"],
    "ibero_openbookqa": ["TODO-iberobench"], "ibero_paraphrase": ["TODO-iberobench"],
    "ibero_piqa": ["TODO-iberobench"], "ibero_siqa": ["TODO-iberobench"], "ibero_truthfulqa": ["TODO-iberobench"],
    "ibero_arc": ["TODO-iberobench"], "ibero": ["TODO-iberobench"],
}


def cite(k: str) -> str:
    """One citation per benchmark: its first key in the bib, or a citation-needed mark."""
    key = next((c for c in CITE.get(k, []) if not c.startswith("TODO")), None)
    return f"\\citep{{{key}}}" if key else "\\citeneeded{}"


# --- table --------------------------------------------------------------------
def formats(r) -> str:
    lc = [f for f in ("letter", "cloze") if r["fmt"][f]]          # letter and cloze share one option range
    parts = [f"{k_text(set().union(*(r['fmt'][f] for f in lc)))}-option {'/'.join(lc)}"] if lc else []
    parts += ["completion"] if r["fmt"]["completion"] else []
    parts += [lab for tw, lab in (("rf", "RF"), ("rfgm", "LLM-RF")) if tw in r["twins"]]
    return ", ".join(parts) or "??"


def name(k: str) -> str:
    return NAMES.get(k, paper_name(k))


# the capability each benchmark targets (the Task column), one coarse label per row
_TASK_OF = {
    "Knowledge": "global_mmlu_full mmlu m_mmlu mmmlu include_base_44 include_v2_en include_v2_og arc arc_mt openbookqa "
                 "arabicmmlu oall_arabic_mmlu oall_exams ceval kmmlu turkishmmlu careqa headqa haerae",
    "Cultural knowledge": "cultural_bench_easy cultural_bench_hard blend_sample oall_acva",
    "Commonsense": "hellaswag piqa global_piqa xcopa xstorycloze commonsense_qa copal_id xwinograd winogrande",
    "Reasoning": "bbh_mcq bbh_cloze acp_bench_mcq acp_bench_cloze mathqa",
    "Reading": "belebele boolq xquad",               # reading comprehension
    "Inference": "xnli paws",
    "Grammar": "multiblimp blimp_nl turblimp zhoblimp mela",
    "Truthfulness": "truthfulqa_mc1 truthfulqa_mc2 truthfulqa-multi_mc1 truthfulqa-multi_mc2 m_truthfulqa_mc1 m_truthfulqa_mc2",
    "Toxicity": "toxigen",
    "Language modelling": "lambada_openai_mt",
    "Tokenization": "toksuite toksuite_math toksuite_stem",
    "Mixed suite": "noreval evalita_llm french_bench kobest bangla ibero oall_alghafa oall_mt",
}
TASK = {k: t for t, ks in _TASK_OF.items() for k in ks.split()}

# the L50 list by resource rank (English first, then scheme A's FineWeb-2 order); the low-resource
# examples are its ranks 30-50, the least-resourced first
_iso2 = load_languages()["fineweb_iso2"]
L50 = ["en"] + [_iso2[s.split("_")[0]] for s in cell_fineweb_subsets(50, "A")]
LOW_RESOURCE = L50[29:][::-1]


def line(k: str) -> str:
    r = rows[k]
    low = " ".join(lang for lang in LOW_RESOURCE if lang in r["langs"]) or "--"
    return (f"{mat.esc(name(k))} {cite(k)} & {len(r['langs'])} & "
            f"{len(r['langs'] & trained)} & {TASK[k]} & {formats(r)} & {low} \\\\")


analysed = sorted((k for k in rows if "auto" in rows[k]["groups"]), key=lambda k: name(k).lower())
screened = sorted((k for k in rows if "auto" not in rows[k]["groups"]), key=lambda k: name(k).lower())
n_tasks = {g: sum(len(rows[k]["tasks"]) for k in ks) for g, ks in (("a", analysed), ("s", screened))}
n_all = {g: sum(rows[k]["all"] for k in ks) for g, ks in (("a", analysed), ("s", screened))}

body = "\n".join([
    r"\begin{table*}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
    r"\renewcommand{\arraystretch}{0.8}",
    r"\begin{tabular}{@{}lrrll>{\raggedright\arraybackslash}p{2.8cm}@{}}", r"\toprule",
    r"\textbf{Benchmark} & \textbf{Langs} & \textbf{Trained} & \textbf{Task} & \textbf{Formats} & "
    r"\textbf{Low-Resource Language Examples} \\", r"\midrule",
    rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Analysed}} ({len(analysed)} benchmarks, {n_tasks['a']:,} tasks)}} \\",
    *map(line, analysed), r"\midrule",
    rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Screened, not analysed}} ({len(screened)} benchmarks, "
    rf"{n_tasks['s']:,} tasks)}} \\",
    *map(line, screened), r"\bottomrule", r"\end{tabular}",
    r"\caption{\textbf{Every benchmark we considered}, its reformulated variants folded into the row of their "
    r"original. Analysed benchmarks are evaluated throughout pretraining, and "
    r"a benchmark with any analysed task is listed with them. Screened benchmarks were evaluated only on the final "
    r"checkpoints of the 600M, 1B, 1.7B and 3B models. Langs is the number of languages that the harness registers for the "
    r"benchmark. Trained is the number of them that the sweep trains (English and the FineWeb-2 languages of "
    r"every trained build). Task is the capability the benchmark targets, and a mixed suite is a national "
    r"benchmark collection that covers several. Formats lists the scoring formats "
    r"of the original tasks with $K$ answer options, a range when $K$ differs across tasks (var.\ when it only varies across items). Letter scores the "
    r"labels (A, B, \ldots) of options listed in the prompt. Cloze scores each answer text as a continuation of "
    r"the prompt. Completion has no shared prompt (minimal pairs, sentence pairs, Winograd schemas) or a single "
    r"target continuation (LAMBADA, XQuAD). RF and LLM-RF mark the reformulated variants (\\S\\ref{subsec:rq1_factors}). "
    r"Low-Resource Language Examples lists the benchmark's languages among "
    r"ranks 30 to 50 of our 50-language list, which orders English and the FineWeb-2 languages by resources, "
    r"from the least resourced (rank 50) up (ISO 639-1 codes).}",
    r"\label{tab:app_benchmarks_considered}", r"\end{table*}", ""])
OUT.write_text(body)

# --- report -------------------------------------------------------------------
print(f"wrote {OUT}: {len(analysed)} analysed + {len(screened)} screened rows")
print(f"tasks (distinct items-sets): analysed {n_tasks['a']}, screened {n_tasks['s']}; "
      f"counting each format: analysed {n_all['a']}, screened {n_all['s']}, total {sum(n_all.values())}")
for k in analysed + screened:
    r = rows[k]
    if r["undetermined"]:
        print("UNDETERMINED format:", k, r["undetermined"])
    print(f"  {k:24s} {sorted(r['groups'])} all={r['all']} tasks={len(r['tasks'])} {formats(r)}")
print("citation needed:", sorted(k for k in rows if cite(k) == "\\citeneeded{}"))
