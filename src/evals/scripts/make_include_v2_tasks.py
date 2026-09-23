#!/usr/bin/env python3
"""Generate the INCLUDE v2 harness tasks as logprob (cloze) evaluations.

Why. INCLUDE v2 (`include-results/include-128`, 128 language-country pairs)
ships with a CoT generator in the include-private repo
(`include-tasks/generate_include_tasks.py`), which writes `generate_until`
tasks scored by `exact_match` after an `A. B. C. D. Answer:` prompt. A base
model at 175M-1.7B cannot answer in that format — the same finding that made
the `rf_` twins necessary (analysis/rq00_task_reformulation) — so this writes
the cloze form instead: the stem alone, the four options scored as
continuations. v1 (`include_base_44`) stays registered and untouched.

Two variants, from the two language columns the dataset carries:

    og   `question` / `choices`        the item in its own language
    en   `question_en` / `choices_en`  the same item translated to English

Scored side by side on the same models, they separate "does the model know
this" from "can the model read this language", which one variant alone
cannot.

What it writes, idempotently:
  src/evals/tasks/include_v2/<variant>/include_v2_<variant>_<pair>.yaml
        one self-contained harness YAML per pair, reached through
        eval_worker.py's --include_path (evaluate.sbatch passes
        $HARNESS_INCLUDE_PATH) -- not the pinned wheel, which would have to
        be rebuilt
  configs/tasks.json
        one entry per task (language, benchmark include_v2_<variant>,
        n_options 4, metric acc_norm) and their membership in `auto_probe`

Only the pairs whose language the ladder trains (`languages.json` ->
groups.trained, the L50 set) are written: rule 2 drops every other row from
the analysis, so evaluating them buys nothing.

Usage:
    python3.11 src/evals/scripts/make_include_v2_tasks.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import yaml

from utils.configs import read_tasks_json, write_tasks_json

ROOT = Path(__file__).resolve().parents[3]
TASKS_JSON = ROOT / "configs" / "tasks.json"
LANGUAGES_JSON = ROOT / "configs" / "languages.json"
OUT_DIR = ROOT / "src" / "evals" / "tasks" / "include_v2"
HF_REPO = "include-results/include-128"
# The hub cache the eval jobs read (evaluate.sbatch hard-sets it). The config
# list is taken from disk rather than the Hub: compute nodes are offline and
# the login node should not need the network to regenerate the task set.
HUB_CACHE = Path("/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models")
SNAPSHOT = HUB_CACHE / f"datasets--{HF_REPO.replace('/', '--')}" / "snapshots"

# variant -> (doc_to_text, doc_to_choice). `answer` is a letter in the data.
VARIANTS = {
    "og": ("{{question.strip()}}\nAnswer:", "{{choices}}"),
    "en": ("{{question_en.strip()}}\nAnswer:", "{{choices_en}}"),
}
TARGET = "{{['A', 'B', 'C', 'D'].index(answer)}}"
FORMAT = {
    "og": "cloze INCLUDE v2: question only in the item's own language, the four options scored as continuations",
    "en": "cloze INCLUDE v2: the English translation of the same items, the four options scored as continuations",
}


# v1 covers 44 languages; these six are in the L50 set, have a v2 pair, and
# are named by no v1 task, so they would be silently dropped.
EXTRA_ISO = {"czech": "cs", "danish": "da", "dutch": "nl",
             "marathi": "mr", "slovak": "sk", "swedish": "sv"}
# The task registry spells two languages differently from languages.json;
# tasks_for_benchmarks resolves them the same way when it matches a cell.
ALIASES = {"jp": "ja", "cn": "zh"}


def language_iso() -> dict[str, str]:
    """English language name -> ISO code, read off the v1 tasks already
    registered (`include_base_44_<name>`), so the two INCLUDE versions agree
    on what a language is called without a second table to keep in sync."""
    tasks = json.loads(TASKS_JSON.read_text())["tasks"]
    return {**{t[len("include_base_44_"):]: e["language"]
               for t, e in tasks.items()
               if e.get("benchmark") == "include_base_44" and e["language"] != "multi"},
            **EXTRA_ISO}


@lru_cache(maxsize=1)
def snapshot() -> Path:
    """The most complete cached revision, newest first on a tie.

    Four revisions are cached here and two of them are partial (one holds four
    configs, one is empty), so picking by name or by date alone can pin the
    task set to a snapshot that is missing most of the data -- and the failure
    would be a task that quietly scores four languages instead of 77."""
    snaps = [(len(list(d.glob("*/*.parquet"))), d.stat().st_mtime, d)
             for d in SNAPSHOT.glob("*") if d.is_dir()]
    if not snaps or not max(snaps)[0]:
        raise SystemExit(f"{HF_REPO} is not in {SNAPSHOT}; download it on the login node first")
    best = max(snaps)
    print(f"reading revision {best[2].name} ({best[0]} parquet files)")
    return best[2]


def configs() -> list[str]:
    """The dataset's config names, from the cached snapshot."""
    return sorted(d.name for d in snapshot().iterdir() if d.is_dir())


def resolve(pair: str, iso: dict[str, str]) -> str | None:
    """The ISO code of a `<language>_<country>` config, by longest language
    prefix — country names carry underscores too (`chinese_hong_kong`), so
    splitting on the first one is wrong."""
    for name in sorted(iso, key=len, reverse=True):
        if pair == name or pair.startswith(name + "_"):
            return iso[name]
    return None


# 26 of the 443,536 items carry fewer than four options, and in 63 of the 128
# pairs the English and original lists differ in length — so an `answer` of "D"
# can index past the end of the variant being scored. Behind letters that was
# harmless; as a choice it is an IndexError, or worse a silent mis-scoring.
# Each variant dir gets this filter (the harness resolves `!function utils.x`
# against the YAML's own directory).
UTILS_PY = '''"""Drop items without four usable FIELD (make_include_v2_tasks.py writes this)."""


def process_docs(dataset):
    return dataset.filter(
        lambda r: r["answer"] in ("A", "B", "C", "D")
        and len(r["FIELD"]) == 4
        and all(isinstance(c, str) and c.strip() for c in r["FIELD"])
        and isinstance(r["STEM"], str) and r["STEM"].strip())
'''
FIELDS = {"og": ("choices", "question"), "en": ("choices_en", "question_en")}


def parquet_files(pair: str) -> list[str]:
    """The pair's parquet shards inside the cached hub snapshot.

    The tasks do NOT go through `dataset_path: include-results/include-128`.
    That resolves to whatever revision the hub snapshot is at (7ba8acc) and then
    looks for a PREPARED dataset under that revision; the prepared copy on this
    cluster is under an older one (d0d7aa), so every one of the 154 tasks died
    offline with `FileNotFoundError: .../7ba8acc.../dataset_info.json`, and
    passing `revision=` does not help because the offline path ignores it.
    Reading the snapshot's parquet directly skips hub resolution entirely, works
    on the offline compute nodes, and pins the data to one revision -- the same
    thing the rfgm tasks do with their JSONLs."""
    files = sorted(str(f) for f in (snapshot() / pair).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet for {pair} under {snapshot()}")
    return files


def task_yaml(variant: str, pair: str) -> str:
    text, choice = VARIANTS[variant]
    doc = {"task": f"include_v2_{variant}_{pair}", "dataset_path": "parquet",
           "dataset_kwargs": {"data_files": {"test": parquet_files(pair)}},
           "test_split": "test",
           "output_type": "multiple_choice", "num_fewshot": 0,
           "doc_to_text": text, "doc_to_choice": choice, "doc_to_target": TARGET,
           "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True}
                           for m in ("acc", "acc_norm")],
           "metadata": {"version": 0.0}}
    # a YAML tag safe_dump cannot emit
    return (yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)
            + "process_docs: !function utils.process_docs\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    iso = language_iso()
    trained = set(json.loads(LANGUAGES_JSON.read_text())["groups"]["trained"])
    data, before = read_tasks_json(TASKS_JSON)

    kept, skipped, written = [], [], 0
    for pair in configs():
        code = resolve(pair, iso)
        if code is None or ALIASES.get(code, code) not in trained:
            skipped.append(pair)
            continue
        kept.append((pair, code))
        for variant in VARIANTS:
            name = f"include_v2_{variant}_{pair}"
            out = OUT_DIR / variant / f"{name}.yaml"
            body = task_yaml(variant, pair)
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                field, stem = FIELDS[variant]
                (out.parent / "utils.py").write_text(
                    UTILS_PY.replace("FIELD", field).replace("STEM", stem))
                if not out.exists() or out.read_text() != body:
                    out.write_text(body)
                    written += 1
            data["tasks"][name] = OrderedDict(
                [("language", code), ("benchmark", f"include_v2_{variant}"),
                 ("stages", ["pretraining"]), ("n_options", 4), ("metric", "acc_norm")])

    # One probe group, not two: a second group is a second job per cell, each
    # paying the fixed overhead. `auto_include_v2` (2026-09-22) is retired.
    data["groups"]["auto_probe"] = sorted(set(data["groups"].get("auto_probe", []))
                                          | {f"include_v2_{v}" for v in VARIANTS})
    data["groups"].pop("auto_include_v2", None)
    for variant in VARIANTS:
        data["benchmarks"][f"include_v2_{variant}"] = OrderedDict(
            [("name", "INCLUDE v2 (Romanou et al.)"), ("languages", len(kept)),
             ("format", FORMAT[variant])])

    print(f"{len(kept)} of {len(kept) + len(skipped)} pairs are in the L50 set -> "
          f"{len(kept) * len(VARIANTS)} tasks ({written} yaml files written) under {OUT_DIR}")
    print(f"  skipped (untrained language): {len(skipped)}")
    if not args.dry_run:
        write_tasks_json(data, before, TASKS_JSON)
        print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
