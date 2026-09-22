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

What it writes, idempotently:
  src/evals/tasks/rf/<family>/rf_<task>.yaml   one self-contained harness
        YAML per original task (same dataset, config and split, read from
        the pinned harness checkout), reachable through eval_worker.py's
        --include_path (evaluate.sbatch passes $HARNESS_INCLUDE_PATH)
  configs/tasks.json                            one entry per rf task
        (language, benchmark rf_<family>, n_options 4, metric acc_norm),
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
import json
import re
from pathlib import Path

import yaml

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
}

FORMAT = {
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


def source_config(name: str, family: str, harness: Path) -> dict:
    """dataset_path / dataset_name / test_split of the original task, from
    its harness YAML: the leaf yaml for belebele (galician_bench overrides
    the repo and split), the per-language template for the two group
    families (global_mmlu_full_<lang> and include_base_44_<lang> are groups
    of subject subtasks over one split)."""
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

    data = json.loads(TASKS_JSON.read_text())
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
                       "stages": ["pretraining"], "n_options": 4, "metric": "acc_norm"}
        # `n_items` is derived from the harness results by derive_task_options.py,
        # and the above-random gate needs it: rewriting the entry without it drops
        # the task out of the gate silently, as if it had stopped clearing chance.
        if "n_items" in prev:
            tasks[twin]["n_items"] = prev["n_items"]
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            if pre == "rf" and fam in FILTERED:
                (out.parent / "utils.py").write_text(UTILS_PY)
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
        data["benchmarks"][f"{pre}_{fam}"] = {**data["benchmarks"][fam], "format": fmt[fam]}
    print(f"{len(originals) - len(missing)} {pre} tasks ({written} yaml files written) under {out_dir}; "
          f"auto_{pre} = {data['groups'][f'auto_{pre}']}"
          + (f"; {len(missing)} tasks have no JSONL in {RFGM_DATA} yet" if missing else ""))
    if not args.dry_run:
        TASKS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
