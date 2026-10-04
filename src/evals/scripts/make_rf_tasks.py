#!/usr/bin/env python3
"""Generate the reformulated ("rf") harness tasks: the letter-format
multiple-choice families rewritten as cloze tasks, scored on the answer
strings themselves instead of on "A"/"B"/"C"/"D".

Why. Three auto families — belebele, global_mmlu_full, include_base_44 — put
the options in the prompt and ask the model for a letter. Base models at
90M-1.7B never learn that convention, and the ladder report shows the
families sitting at chance up to 1.7B (21 of 252 four-option tasks clear
the gate). The FineWeb-edu ablations solved the same problem for MMLU by
dropping the letters and scoring the full answer as the continuation
(src/signal-and-noise/analysis/rq00_task_reformulation/README.md); this
does that inside lm-eval-harness. The
other eleven auto families already score answer strings and are untouched.
The lettered probe families (`auto_probe`, 2026-09-23: mmlu, commonsense_qa,
cultural_bench_easy) get twins the same way, but through an absolute
`include:` of the original YAML rather than copied keys -- see
_INCLUDE_ORIGINAL below for why the copy cannot work for them.

What it writes, idempotently:
  src/evals/tasks/rf/<family>/rf_<task>.yaml   one self-contained harness
        YAML per original task (same dataset, config and split, read from
        the pinned harness checkout), reachable through eval_worker.py's
        --include_path (evaluate.sbatch passes $HARNESS_INCLUDE_PATH)
  configs/tasks.json                            one entry per rf task
        (language, benchmark rf_<family>, n_options 4 -- 5 for commonsense_qa --
        metric acc_norm),
        the `auto_rf` group and the `benchmarks` metadata

Task names carry an `rf_` PREFIX on purpose: tasks_for_benchmarks matches
`<benchmark>_…`, so a `_rf` suffix would be swept into the original family.

`--set rfgm` writes the Gemini-rewritten twins instead (Tier 2): the same
originals, read from the JSONLs rewrite_items_gemini.py left in RFGM_DATA
(tasks without a file are skipped and counted), through one `dataset_path:
json` YAML per task under src/evals/tasks/rfgm/, registered as `rfgm_<task>`
/ `rfgm_<family>` / group `auto_rfgm`.

Usage:
    python3.11 src/evals/scripts/make_rf_tasks.py [--set rf|rfgm] [--harness DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import yaml

from utils.configs import read_tasks_json, write_tasks_json

ROOT = Path(__file__).resolve().parents[3]
TASKS_JSON = ROOT / "configs" / "tasks.json"
OUT_DIR = ROOT / "src" / "evals" / "tasks" / "rf"
HARNESS = Path("/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/"
               "lm-evaluation-harness/lm_eval/tasks")

# family -> (doc_to_text, doc_to_choice, doc_to_target); the prompt keeps the
# original context and question, drops the lettered option list, and the
# harness scores each answer string as the continuation after "Answer:".
TEMPLATES = {
    "belebele": (
        "{{flores_passage}}\nQuestion: {{question.strip()}}\nAnswer:",
        "{{[mc_answer1, mc_answer2, mc_answer3, mc_answer4]}}",
        "{{['1', '2', '3', '4'].index(correct_answer_num)}}",
    ),
    "global_mmlu_full": (
        "The following are questions about {{subject.replace('_', ' ')}}.\n"
        "Question: {{question.strip()}}\nAnswer:",
        "{{[option_a, option_b, option_c, option_d]}}",
        "{{['A', 'B', 'C', 'D'].index(answer)}}",
    ),
    "include_base_44": (
        "Question: {{question.strip()}}\nAnswer:",
        "{{[option_a, option_b, option_c, option_d]}}",
        "answer",                       # already a 0-based int in the dataset
    ),
    # The probe candidates that also ask for a letter (auto_probe, 2026-09-22).
    # openbookqa, toxigen, mathqa, bbq and truthfulqa_mc2 already score answer
    # strings, so they have no twin: they are their own cloze form.
    "mmlu": (
        "{{question.strip()}}\nAnswer:",
        "{{choices}}",
        "answer",                       # 0-based int
    ),
    "commonsense_qa": (
        "Question: {{question.strip()}}\nAnswer:",
        "{{choices.text}}",             # five options, not four
        "{{choices.label.index(answerKey.lstrip())}}",
    ),
    # cultural_bench_easy's own process_docs has already turned `answer` into a
    # 0-based index by the time the template runs (utils.process_docs_by_country
    # maps "ABCD".index), so the target is the column, not a letter lookup.
    # cultural_bench_HARD has no twin: the harness scores it over
    # ["False", "True"] and its rows carry a single `prompt_option`, so it is
    # already a cloze task and there is nothing to reformulate.
    "cultural_bench_easy": (
        "{{prompt_question.strip()}}\nAnswer:",
        "{{[prompt_option_a, prompt_option_b, prompt_option_c, prompt_option_d]}}",
        "{{answer}}",
    ),
}
# n_options is 4 unless the family says otherwise.
N_OPTIONS = {"commonsense_qa": 5}

FORMAT = {
    "mmlu": "cloze reformulation of mmlu: question only, the four options scored as continuations",
    "commonsense_qa": "cloze reformulation of commonsense_qa: question only, the five options scored as continuations",
    "cultural_bench_easy": "cloze reformulation of cultural_bench_easy: question only, the four options scored as continuations",
    "belebele": "cloze reformulation of belebele: passage + question, the four answers scored as continuations",
    "global_mmlu_full": "cloze reformulation of global_mmlu_full: question only, the four options scored as continuations",
    "include_base_44": "cloze reformulation of include_base_44: question only, the four options scored as continuations",
}
# Tier 2: the items rewritten by Gemini (rewrite_items_gemini.py) into a
# statement stem with four short continuations, one JSONL per task with
# `text` / `choices` / `gold` columns, so one YAML template covers every family.
RFGM_DATA = Path("/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/rf-data/rfgm")
RFGM_DIR = ROOT / "src" / "evals" / "tasks" / "rfgm"
FORMAT_RFGM = {f: f"Gemini statement rewrite of {f}: stem + four short continuations, scored as continuations "
                  "(rf-data/rfgm/<task>.jsonl)" for f in TEMPLATES}


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!function", lambda loader, node: node.value)


# The three original families are found by a hand-written path rule below, and
# their twins copy a few source keys out of the original. The probe families
# cannot be reduced to copied keys, and trying cost 38 broken tasks:
# cultural_bench's per-country `process_docs` is a `!function`, so copied it
# became the literal string "utils.process_docs_by_country" ("TypeError: 'str'
# object is not callable"), and commonsense_qa is scored on `validation_split`,
# which the copied set never carried ("must have valid or test docs!"). Neither
# twin could fail before the eval job ran it.
#
# So these `include:` the original leaf YAML by absolute path instead. The
# harness's load_yaml accepts an absolute include, and builds its loader per
# FILE, so a `!function` inside the included file resolves against the
# harness's own directory even though the including file is ours. The twin
# inherits the dataset, the split and the filter exactly as the original has
# them, and overrides only the three prompt keys.
#
# The three original families deliberately keep the copied-keys form: their
# YAMLs are already evaluated on a thousand-odd checkpoints, and rewriting them
# would change a measurement that is mid-flight.
_INCLUDE_ORIGINAL = ("commonsense_qa", "cultural_bench_easy")
# `mmlu` is a group of 57 subject tasks, so no leaf YAML declares `task: mmlu`
# and there is nothing to include -- that is the "no harness YAML declares"
# exit. The twin reads the dataset's own `all` config, the same items the group
# aggregates over, as one task.
_EXPLICIT_SRC = {"mmlu": {"dataset_path": "cais/mmlu", "dataset_name": "all",
                          "test_split": "test"}}


@lru_cache(maxsize=1)
def _task_index(harness: Path) -> dict[str, Path]:
    """task name -> its leaf YAML, over the whole harness. Built once and
    cached: the tree holds thousands of YAMLs, and a family like
    cultural_bench asks for nineteen of them. The cheap substring test comes
    first so only the handful of candidate files are parsed."""
    idx: dict[str, Path] = {}
    for f in harness.rglob("*.yaml"):
        try:
            text = f.read_text()
        except OSError:
            continue
        if "task:" not in text:
            continue
        try:
            head = yaml.load(text, _Loader) or {}
        except Exception:
            continue
        if isinstance(head, dict) and isinstance(head.get("task"), str):
            idx.setdefault(head["task"], f)
    return idx


def source_config(name: str, family: str, harness: Path) -> dict:
    """dataset_path / dataset_name / test_split of the original task, from
    its harness YAML: the leaf yaml for belebele (galician_bench overrides
    the repo and split), the per-language template for the two group
    families (global_mmlu_full_<lang> and include_base_44_<lang> are groups
    of subject subtasks over one split)."""
    if family in _EXPLICIT_SRC:
        return dict(_EXPLICIT_SRC[family])
    if family in _INCLUDE_ORIGINAL:
        f = _task_index(harness).get(name)
        if f is None:
            raise SystemExit(f"no harness YAML declares `task: {name}`")
        return {"include": str(f)}
    if family == "belebele":
        path = next(harness.rglob(f"{name}.yaml"))
    else:
        lang = name[len(family) + 1:]
        sub = "global_mmlu/full" if family == "global_mmlu_full" else "include/default"
        path = next((harness / sub).glob(f"*/_{lang}_template_yaml"))
    cfg = {"dataset_path": "facebook/belebele", "test_split": "test"}
    cfg.update({k: v for k, v in yaml.load(path.read_text(), _Loader).items()
                if k in ("dataset_path", "dataset_name", "test_split")})
    return cfg


# Rows with a missing option: 37 of the 519k Global-MMLU items and one INCLUDE
# item have a None or blank option (e.g. Global-MMLU vi row 1588). Behind
# letters that was harmless; as a choice, None crashes the task and an empty
# string is a zero-cost continuation the model "prefers" every time. Each
# family dir gets this utils.py and the YAMLs a `process_docs` that drops
# them (the harness resolves `!function utils.x` against the YAML's dir).
UTILS_PY = '''"""Drop rows whose prompt fields are missing (make_rf_tasks.py writes this)."""
FIELDS = ("question", "option_a", "option_b", "option_c", "option_d")


def process_docs(dataset):
    return dataset.filter(lambda r: all(isinstance(r[k], str) and r[k].strip()
                                        for k in FIELDS))
'''
FILTERED = ("global_mmlu_full", "include_base_44")

# One row of CulturalBench-Easy (Spain) carries prompt_option_d = None. Behind
# letters that is harmless -- the original scores "A".."D" -- but as a choice
# None crashes the task, the same failure the filter above exists for. The twin
# cannot just add a `process_docs`: the `include:`d original already uses one,
# both to split the 19 countries into 19 tasks and to turn `answer` into an
# index, and overriding it would silently merge every country into one task. So
# this reimplements both steps and drops the unusable rows as well. Countries
# match on a normalised name, so the task suffix is the only thing the
# generator has to get right.
CB_UTILS = '''"""cultural_bench_easy twins: per-country split, answer to index, blank rows
dropped (make_rf_tasks.py writes this; it stands in for the harness's own
utils.process_<country>, which the twin inherits but which keeps None options)."""
FIELDS = ("prompt_question", "prompt_option_a", "prompt_option_b",
          "prompt_option_c", "prompt_option_d")


def _country(slug):
    def process(dataset):
        ds = dataset.filter(
            lambda r: r["country"].lower().replace(" ", "_") == slug
            and all(isinstance(r[k], str) and r[k].strip() for k in FIELDS))
        return ds.map(lambda r: {**r, "answer": "ABCD".index(r["answer"])
                                 if r["answer"] in ("A", "B", "C", "D") else r["answer"]})
    return process


%s'''


def rf_yaml(name: str, family: str, src: dict) -> str:
    text, choice, target = TEMPLATES[family]
    doc = {"task": f"rf_{name}", **src, "output_type": "multiple_choice",
           "num_fewshot": 0, "doc_to_text": text, "doc_to_choice": choice,
           "doc_to_target": target,
           "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True}
                           for m in ("acc", "acc_norm")],
           "metadata": {"version": 0.0}}
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)
    if family in FILTERED:   # a YAML tag safe_dump cannot emit
        body += "process_docs: !function utils.process_docs\n"
    if family == "cultural_bench_easy":
        body += f"process_docs: !function utils.process_{name[len(family) + 1:]}\n"
    return body


def rfgm_yaml(name: str) -> str:
    """The Gemini twin: a local JSONL through the `json` loader (no network;
    the harness splats dataset_kwargs into datasets.load_dataset), the three
    columns read by name — the harness returns a column's raw value, so
    `choices` is the list and `gold` the index as stored."""
    doc = {"task": f"rfgm_{name}", "dataset_path": "json",
           "dataset_kwargs": {"data_files": {"test": str(RFGM_DATA / f"{name}.jsonl")}},
           "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
           "doc_to_text": "text", "doc_to_choice": "choices", "doc_to_target": "gold",
           "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True}
                           for m in ("acc", "acc_norm")],
           "metadata": {"version": 0.0}}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", choices=["rf", "rfgm"], default="rf",
                   help="rf: the cloze twins (default); rfgm: the Gemini-rewritten twins")
    p.add_argument("--family", choices=list(TEMPLATES),
                   help="register only this family; for rfgm, whose families are rewritten one at a "
                        "time, so a family still mid-rewrite is not registered with a partial item set")
    p.add_argument("--harness", type=Path, default=HARNESS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    pre, out_dir = args.set, OUT_DIR if args.set == "rf" else RFGM_DIR

    data, before = read_tasks_json(TASKS_JSON)
    tasks = data["tasks"]
    families = [args.family] if args.family else list(TEMPLATES)
    originals = [(n, e) for n, e in tasks.items()
                 if e["benchmark"] in families and "pretraining" in e["stages"]
                 and e["language"] not in ("multi", "??")]
    written, missing = 0, []
    for name, e in originals:
        fam = e["benchmark"]
        if pre == "rfgm" and not (RFGM_DATA / f"{name}.jsonl").exists():
            missing.append(name)
            continue
        twin = f"{pre}_{name}"
        out = out_dir / fam / f"{twin}.yaml"
        body = (rf_yaml(name, fam, source_config(name, fam, args.harness)) if pre == "rf"
                else rfgm_yaml(name))
        prev = tasks.get(twin, {})
        tasks[twin] = {"language": e["language"], "benchmark": f"{pre}_{fam}",
                       "stages": ["pretraining"], "n_options": N_OPTIONS.get(fam, 4),
                       "metric": "acc_norm"}
        # `n_items` is derived from the harness results by derive_task_options.py,
        # and the above-random gate needs it: rewriting the entry without it drops
        # the task out of the gate silently, as if it had stopped clearing chance.
        if "n_items" in prev:
            tasks[twin]["n_items"] = prev["n_items"]
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            if pre == "rf" and fam in FILTERED:
                (out.parent / "utils.py").write_text(UTILS_PY)
            if pre == "rf" and fam == "cultural_bench_easy":
                slugs = sorted(n[len(fam) + 1:] for n, e2 in originals
                               if e2["benchmark"] == fam)
                (out.parent / "utils.py").write_text(CB_UTILS % "\n".join(
                    f'process_{g} = _country("{g}")' for g in slugs))
            if not out.exists() or out.read_text() != body:
                out.write_text(body)
                written += 1
    # the group lists the families that actually have registered tasks, so a
    # family still being rewritten is not advertised to the watcher
    have = sorted({f"{pre}_{e['benchmark']}" for n, e in originals
                   if f"{pre}_{n}" in tasks} | set(data["groups"].get(f"auto_{pre}", [])))
    data["groups"][f"auto_{pre}"] = have
    fmt = FORMAT if pre == "rf" else FORMAT_RFGM
    for fam in families:
        # The `benchmarks` block documents the multilingual families only, so a
        # probe family has nothing to inherit: give the twin a bare entry
        # rather than failing after its YAMLs are already on disk.
        data["benchmarks"][f"{pre}_{fam}"] = {**data["benchmarks"].get(fam, {"name": fam}),
                                              "format": fmt[fam]}
    print(f"{len(originals) - len(missing)} {pre} tasks ({written} yaml files written) under {out_dir}; "
          f"auto_{pre} = {data['groups'][f'auto_{pre}']}"
          + (f"; {len(missing)} tasks have no JSONL in {RFGM_DATA} yet" if missing else ""))
    if not args.dry_run:
        write_tasks_json(data, before, TASKS_JSON)
        print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
