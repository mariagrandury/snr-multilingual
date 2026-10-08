#!/usr/bin/env python3
"""Generate logprob (cloze) harness tasks for the two *mixed* probe families,
BBH and ACP-Bench, in both formulations so the reformulation effect is
measurable on them.

Why. Both ship as `generate_until` + `exact_match` behind a chain of thought,
which a 600M-1.7B base model cannot produce -- so neither carries any signal
on the ladder as published. But most of their subtasks are closed-form
already: the candidate answers are printed in the item, only the *scoring* is
generative. This rewrites those subtasks as `multiple_choice`, which is a
different measurement from the published one (it drops the chain of thought
BBH's headline numbers depend on, and moves chance from ~0 to 1/n) and must be
reported as such.

Mixed families need more than one benchmark name. A subtask that prints
"(A) ... (B) ..." and asks for a letter has a reformulation; a yes/no subtask
does not -- scoring "Yes" against "No" IS the cloze form, so its twin would be
the identical task. results.json carries task names and nothing else, so the
arms have to be separable by name; `tasks_for_benchmarks` and
`benchmark_family` both read the `benchmark` field of configs/tasks.json, so
one benchmark per arm is what makes the filter possible at all:

    bbh                 unchanged: the published 27-subtask generate_until
                        benchmark, posttraining only, not written here
    bbh_mcq             17 lettered subtasks, published formulation, scored
                        over "(A)".."(R)"                      <-- pairs with
    rf_bbh_mcq          the same 17, stem without the option block, scored
                        over the option strings                <-- this
    bbh_cloze           6 two-way subtasks (True/False, Yes/No, valid/invalid),
                        scored over the answer strings; no twin exists

    acp_bench           unchanged: the published generate_until benchmark
    acp_bench_mcq       7 subtasks, question with its "A. B. C. D." list,
                        scored over the letters                <-- pairs with
    rf_acp_bench_mcq    the same 7, the bare `query` field, scored over
                        `choices.text`                         <-- this
    acp_bench_cloze     7 yes/no subtasks; no twin exists

Because the `benchmark` field is matched by prefix, a group listing plain
`bbh` at the pretraining stage selects `bbh_mcq` and `bbh_cloze` and not the
posttraining original -- which is the intended reading. Groups here list the
arms explicitly anyway.

The four BBH subtasks with no candidate set (dyck_languages,
multistep_arithmetic_two, object_counting, word_sorting) and ACP's `_gen`
configs are free-form and are not written: ranking needs candidates.

Option counts are read off the data, not assumed. Several BBH subtasks vary
the count per item (reasoning_about_colored_objects runs 2 to 18), so
`n_options` is the item-weighted effective count, round(1 / mean(1/n_i)) --
the integer whose 1/n is the task's true chance level. `n_items` is the
surviving item count, so the above-random gate has both of its inputs without
waiting for derive_task_options.py.

What it writes, idempotently:
  src/evals/tasks/cloze/<benchmark>/<task>.yaml   one self-contained harness
        YAML per task, reached through eval_worker.py's --include_path
        (auto_evals_cscs sets HARNESS_INCLUDE_PATH to src/evals/tasks)
  src/evals/tasks/cloze/<benchmark>/utils.py      the BBH option-block parser
  configs/tasks.json                              one entry per task, plus the
        six benchmarks and their membership in `auto_probe`

Usage:
    python3.11 src/evals/scripts/make_cloze_tasks.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, OrderedDict
from pathlib import Path

import yaml

from utils.configs import read_tasks_json, write_tasks_json

ROOT = Path(__file__).resolve().parents[3]
TASKS_JSON = ROOT / "configs" / "tasks.json"
OUT_DIR = ROOT / "src" / "evals" / "tasks" / "cloze"

# The eval jobs run offline against these caches; evaluate.sbatch hard-sets
# them. Reading the data here (on the login node) is what makes the option and
# item counts measured rather than assumed.
os.environ.setdefault("HF_HOME", "/iopsstor/scratch/cscs/mariagrandury/hf_home")
os.environ.setdefault("HF_HUB_CACHE",
                      "/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

BBH_REPO = "SaylorTwift/bbh"
ACP_REPO = "ibm-research/acp_bench"
METRICS = [{"metric": m, "aggregation": "mean", "higher_is_better": True}
           for m in ("acc", "acc_norm")]

# The option block BBH prints under "Options:". Letters run past (D): geometric
# shapes offers eleven, reasoning_about_colored_objects eighteen.
OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")

# One parser, shared by the lettered arm and its twin: both need the stem and
# the option list split out of `input`, they differ only in what they score.
# The harness resolves `!function utils.x` against the YAML's own directory,
# so this is written into each of the two directories.
BBH_UTILS = '''"""Split BBH's `input` into stem + options (make_cloze_tasks.py writes this)."""
import re

OPT_RE = re.compile(r"^\\(([A-Z])\\)\\s*(.*)$")


def split(text):
    """(stem, labels, texts) of an item, or None when it has no usable
    option block -- two of BBH's items print no list at all, and one of
    snarks' prints a single option, which is not a choice."""
    stem, sep, body = text.partition("\\nOptions:")
    if not sep:
        return None
    lines = [l.strip() for l in body.strip().split("\\n") if l.strip()]
    ms = [OPT_RE.match(l) for l in lines]
    if len(lines) < 2 or not all(ms):
        return None
    texts = [m.group(2).strip() for m in ms]
    if not all(texts):
        return None
    return stem.strip(), ["(%s)" % m.group(1) for m in ms], texts


def usable(row):
    got = split(row["input"])
    return got is not None and row["target"].strip() in got[1]


def process_docs(dataset):
    def add(row):
        stem, labels, texts = split(row["input"])
        return {"stem": stem, "labels": labels, "texts": texts,
                "gold": labels.index(row["target"].strip())}
    return dataset.filter(usable).map(add)
'''


def _load(repo: str, config: str):
    from datasets import load_dataset
    return load_dataset(repo, config, split="test")


def _effective_options(counts: Counter) -> int:
    """The integer whose 1/n is the item-weighted chance level. A task whose
    items offer different numbers of options has no single option count; the
    gate takes one, and this is the one that keeps `1 / n_options` equal to
    the probability of a random guess."""
    n = sum(counts.values())
    return round(n / sum(c / k for k, c in counts.items()))


def bbh_families(configs: list[str]) -> tuple[dict, dict]:
    """(lettered, two_way) subtask -> stats, classified from the data.

    lettered: every usable item parses into >=2 labelled options whose gold is
    one of the labels. two_way: no option list to parse, but the whole subtask
    has exactly two distinct targets, which are then the two choices."""
    lettered, two_way = {}, {}
    for cfg in configs:
        ds = _load(BBH_REPO, cfg)
        targets = Counter(ds["target"])
        counts, kept = Counter(), 0
        for row in ds:
            stem, sep, body = row["input"].partition("\nOptions:")
            if not sep:
                continue
            lines = [l.strip() for l in body.strip().split("\n") if l.strip()]
            ms = [OPT_RE.match(l) for l in lines]
            if len(lines) < 2 or not all(ms) or not all(m.group(2).strip() for m in ms):
                continue
            if row["target"].strip() not in ["(%s)" % m.group(1) for m in ms]:
                continue
            counts[len(lines)] += 1
            kept += 1
        if kept >= 0.9 * len(ds):
            lettered[cfg] = {"n_items": kept, "n_options": _effective_options(counts),
                             "spread": dict(sorted(counts.items())), "dropped": len(ds) - kept}
        elif len(targets) == 2:
            two_way[cfg] = {"n_items": len(ds), "n_options": 2,
                            "choices": sorted(targets)}
    return lettered, two_way


def bbh_yaml(cfg: str, arm: str) -> str:
    """arm: `bbh_mcq` scores the letters with the option block left in the
    prompt (the published formulation); `rf_bbh_mcq` scores the option strings
    after the stem alone (the reformulation)."""
    lettered = arm == "bbh_mcq"
    return yaml.safe_dump(
        {"task": f"{arm}_{cfg}", "dataset_path": BBH_REPO, "dataset_name": cfg,
         "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
         "doc_to_text": "{{input}}\nAnswer:" if lettered else "{{stem}}\nAnswer:",
         "doc_to_choice": "{{labels}}" if lettered else "{{texts}}",
         "doc_to_target": "gold",
         "metric_list": METRICS, "metadata": {"version": 0.0}},
        sort_keys=False, allow_unicode=True, width=1000
    ) + "process_docs: !function utils.process_docs\n"


def bbh_two_way_yaml(cfg: str, choices: list[str]) -> str:
    """The published item verbatim -- for three of these six the prompt already
    carries an `Options:\n- Yes\n- No` block -- with the two answer strings
    scored as continuations. There is nothing to reformulate: the strings the
    model ranks are the answers themselves."""
    lit = json.dumps(choices)
    return yaml.safe_dump(
        {"task": f"bbh_cloze_{cfg}", "dataset_path": BBH_REPO, "dataset_name": cfg,
         "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
         "doc_to_text": "{{input}}\nAnswer:", "doc_to_choice": choices,
         "doc_to_target": "{{%s.index(target.strip())}}" % lit,
         "metric_list": METRICS, "metadata": {"version": 0.0}},
        sort_keys=False, allow_unicode=True, width=1000)


def acp_yaml(cfg: str, arm: str) -> str:
    """ACP needs no parser: the dataset ships `choices.label` / `choices.text`
    beside `question` (which embeds the "A. B. C. D." list) and `query` (which
    does not), so the two arms are two templates over the same columns."""
    if arm == "acp_bench_cloze":
        doc = {"doc_to_text": "{{context}} {{question}}\nAnswer:",
               "doc_to_choice": ["yes", "no"],
               "doc_to_target": '{{["yes", "no"].index(answer.strip().lower())}}'}
    elif arm == "acp_bench_mcq":
        doc = {"doc_to_text": "{{context}} {{question}}\nAnswer:",
               "doc_to_choice": "{{choices.label}}",
               "doc_to_target": "{{choices.label.index(answer.strip())}}"}
    else:
        doc = {"doc_to_text": "{{context}} {{query}}\nAnswer:",
               "doc_to_choice": "{{choices.text}}",
               "doc_to_target": "{{choices.label.index(answer.strip())}}"}
    short = cfg.removeprefix("acp_").removesuffix("_bool").removesuffix("_mcq")
    return yaml.safe_dump(
        {"task": f"{arm}_{short}", "dataset_path": ACP_REPO, "dataset_name": cfg,
         "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
         **doc, "metric_list": METRICS, "metadata": {"version": 0.0}},
        sort_keys=False, allow_unicode=True, width=1000)


FORMAT = {
    "bbh_mcq": "BBH's closed-form subtasks as logprob: the published item incl. its "
               "(A)-(R) option block, the letters scored as continuations (no chain of thought)",
    "rf_bbh_mcq": "cloze reformulation of bbh_mcq: the stem without the option block, "
                  "the option strings scored as continuations",
    "bbh_cloze": "BBH's two-way subtasks as logprob: the published item, the two answer "
                 "strings scored as continuations (already cloze; no reformulated twin)",
    "acp_bench_mcq": "ACP-Bench's multiple-choice subtasks as logprob: context + the question "
                     "incl. its A.-D. list, the letters scored as continuations",
    "rf_acp_bench_mcq": "cloze reformulation of acp_bench_mcq: context + the bare query, "
                        "the four action strings scored as continuations",
    "acp_bench_cloze": "ACP-Bench's boolean subtasks as logprob: context + question, "
                       "yes/no scored as continuations (already cloze; no reformulated twin)",
}
NAME = {"bbh": "BIG-Bench Hard (Suzgun et al. 2023)",
        "acp_bench": "ACP-Bench (Kokel et al. 2025)"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import datasets
    datasets.disable_progress_bars()

    hub = Path(os.environ["HF_HUB_CACHE"])
    def cached_configs(repo: str) -> list[str]:
        snaps = sorted((hub / f"datasets--{repo.replace('/', '--')}" / "snapshots").glob("*"))
        if not snaps:
            raise SystemExit(f"{repo} is not cached under {hub}; download it on the login node")
        return sorted(d.name for d in snaps[-1].iterdir() if d.is_dir() and d.name != "data")

    bbh_configs = cached_configs(BBH_REPO)
    lettered, two_way = bbh_families(bbh_configs)
    # The subtasks that fit neither shape are the free-form ones; naming them
    # is what makes a mis-parsed subtask, or a partial snapshot, visible.
    free_form = sorted(set(bbh_configs) - set(lettered) - set(two_way))
    acp = {c: _load(ACP_REPO, c).num_rows for c in cached_configs(ACP_REPO)
           if c.endswith(("_bool", "_mcq"))}

    data, before = read_tasks_json(TASKS_JSON)
    written, entries = 0, Counter()

    def emit(arm: str, task: str, body: str, n_options: int, n_items: int,
             utils: str | None = None) -> None:
        nonlocal written
        out = OUT_DIR / arm / f"{task}.yaml"
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            if utils:
                (out.parent / "utils.py").write_text(utils)
            if not out.exists() or out.read_text() != body:
                out.write_text(body)
                written += 1
        # `n_items` is what the gate's Wilson bound is taken over and `n_options`
        # its chance level; a task missing either is dropped from the gate in
        # silence, so both are written now rather than derived from results later.
        data["tasks"][task] = OrderedDict(
            [("language", "en"), ("benchmark", arm), ("stages", ["pretraining"]),
             ("n_options", n_options), ("n_items", n_items), ("metric", "acc_norm")])
        entries[arm] += 1

    for cfg, st in lettered.items():
        for arm in ("bbh_mcq", "rf_bbh_mcq"):
            emit(arm, f"{arm}_{cfg}", bbh_yaml(cfg, arm), st["n_options"], st["n_items"],
                 utils=BBH_UTILS)
    for cfg, st in two_way.items():
        emit("bbh_cloze", f"bbh_cloze_{cfg}", bbh_two_way_yaml(cfg, st["choices"]), 2, st["n_items"])
    for cfg, n in acp.items():
        short = cfg.removeprefix("acp_").removesuffix("_bool").removesuffix("_mcq")
        if cfg.endswith("_bool"):
            emit("acp_bench_cloze", f"acp_bench_cloze_{short}",
                 acp_yaml(cfg, "acp_bench_cloze"), 2, n)
        else:
            for arm in ("acp_bench_mcq", "rf_acp_bench_mcq"):
                emit(arm, f"{arm}_{short}", acp_yaml(cfg, arm), 4, n)

    for arm, fmt in FORMAT.items():
        data["benchmarks"][arm] = OrderedDict(
            [("name", NAME["bbh" if "bbh" in arm else "acp_bench"]),
             ("languages", 1), ("format", fmt)])
    data["groups"]["auto_probe"] = sorted(set(data["groups"]["auto_probe"]) | set(FORMAT))

    for arm in FORMAT:
        print(f"{arm:20s} {entries[arm]:3d} tasks")
    print(f"\nBBH: {len(lettered)} lettered + {len(two_way)} two-way of "
          f"{len(bbh_configs)} subtasks; not written (free-form): {', '.join(free_form)}")
    for cfg, st in sorted(lettered.items()):
        if len(st["spread"]) > 1 or st["dropped"]:
            print(f"  {cfg}: options {st['spread']} -> n_options {st['n_options']} "
                  f"(chance {1 / st['n_options']:.4f} vs exact "
                  f"{sum(c / k for k, c in st['spread'].items()) / st['n_items']:.4f}), "
                  f"{st['dropped']} item(s) dropped")
    print(f"{written} yaml files written under {OUT_DIR}")
    if not args.dry_run:
        write_tasks_json(data, before, TASKS_JSON)
        print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
