#!/usr/bin/env python3
"""Register tasks the harness ALREADY ships into configs/tasks.json.

The other generators write YAMLs (make_rf_tasks, make_cloze_tasks,
make_include_v2_tasks); this one writes nothing but registrations, for
benchmarks the pinned wheel already defines and that only need a project
entry — language, benchmark, stage, and the two numbers the above-random gate
needs. Each task is loaded through a real `TaskManager` first, so a name that
does not resolve, or whose dataset is not in the offline cache, fails here on
the login node instead of inside an eval job (`n_options` and `n_items` are
MEASURED off the loaded docs, never asserted: `_effective_options` is the
integer whose 1/n is the item-weighted chance level, for the tasks whose
items differ in option count).

A group name (toksuite_math) registers as one entry the same way `mmlu` does:
the harness expands it, results.json carries the children, and
derive_task_options attributes them to the longest listed prefix.

Everything lands in `groups.auto_probe` — these are candidates, screened at
the last checkpoint before anything joins `auto`.

    python3.11 src/evals/scripts/add_harness_tasks.py [--set ibero|toksuite]
                                                      [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.configs import read_tasks_json, write_tasks_json  # noqa: E402
from make_cloze_tasks import _effective_options  # noqa: E402

HARNESS = Path("/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/"
               "lm-evaluation-harness")

# (task, language, benchmark). The benchmark is the unit the gate and every
# per-benchmark figure aggregate over, so it groups the same task type across
# languages — an acceptability judgement and ARC do not belong in one mean.
# `ibero_` prefixes keep them out of the families already in `auto`: these are
# other groups' translations, and mixing them into `arc` would widen a
# benchmark the study is already reporting.
#
# Left out on purpose: wnli_{es,ca,eu} are 71 items over two options, so the
# Wilson bound needs 60 % accuracy before the gate can call them anything but
# random -- they cannot inform the decision. galcola (the repo serves no data
# file) and qnlieu (BasqueGLUE is a script dataset, unsupported since
# datasets v3) do not load at all.
IBERO = [
    ("arc_ca_challenge", "ca", "ibero_arc"),
    ("arc_ca_easy", "ca", "ibero_arc"),
    ("arc_eu_challenge", "eu", "ibero_arc"),
    ("arc_eu_easy", "eu", "ibero_arc"),
    ("openbookqa_es", "es", "ibero_openbookqa"),
    ("openbookqa_ca", "ca", "ibero_openbookqa"),
    ("openbookqa_gl", "gl", "ibero_openbookqa"),
    ("piqa_ca", "ca", "ibero_piqa"),
    ("piqa_eu", "eu", "ibero_piqa"),
    ("siqa_ca", "ca", "ibero_siqa"),
    ("escola", "es", "ibero_cola"),
    ("catcola", "ca", "ibero_cola"),
    ("truthfulqa_gl_mc1", "gl", "ibero_truthfulqa"),
]

# TokSuite measures tokenisation robustness: one canonical question set per
# language plus perturbations of it. The five canonical tasks are the parallel
# set (the same 40 sentences in five languages); math and stem are groups over
# their own perturbations, English content, registered whole because a single
# perturbation is 16-44 items and could never clear the gate.
TOKSUITE = [
    ("toksuite_english_canonical", "en", "toksuite"),
    ("toksuite_farsi_canonical", "fa", "toksuite"),
    ("toksuite_italian_canonical", "it", "toksuite"),
    ("toksuite_chinese_canonical", "zh", "toksuite"),
    ("toksuite_turkish_canonical", "tr", "toksuite"),
    ("toksuite_math", "en", "toksuite_math"),
    ("toksuite_stem", "en", "toksuite_stem"),
]

# Whole harness directories, scanned for their multiple-choice tasks: the
# language-specific leaderboards the survey of 2026-09-23 turned up that this
# project evaluates nothing from. `lang` is the language every task in the dir
# is in, or None to read it off the task name's last underscore field (arc_mt,
# mela are one task per language). cmmlu and basqueGLUE are absent on purpose:
# both are script datasets, which `datasets` v3 refuses (evals/CLAUDE.md #15).
# BertaQA is absent for a different reason: Basque is in no training mixture
# (`languages.json` groups.trained), so every one of its tasks would be
# selected by no cell and dropped by rule 2 if it were.
SCAN = [
    ("ceval", "zh", "ceval"),
    ("zhoblimp", "zh", "zhoblimp"),
    ("kmmlu", "ko", "kmmlu"),
    ("haerae", "ko", "haerae"),
    ("turkishmmlu", "tr", "turkishmmlu"),
    ("turblimp", "tr", "turblimp"),
    ("evalita_llm", "it", "evalita_llm"),
    ("blimp_nl", "nl", "blimp_nl"),
    ("noreval", "no", "noreval"),
    ("bangla", "bn", "bangla"),
    ("french_bench", "fr", "french_bench"),
    ("arc_mt", None, "arc_mt"),      # one task per language, read off the name
    ("mela", None, "mela"),
]
LANG_ALIAS = {"nb": "no", "nn": "no"}      # the project's code for Norwegian


def scan(directory: str, lang: str | None, benchmark: str) -> list[tuple[str, str, str]]:
    """The multiple-choice tasks of one harness directory, as (task, language,
    benchmark). output_type is resolved through `include:` because most task
    files carry only the dataset name and inherit the rest from a base."""
    import yaml

    def read(path: Path) -> dict:
        """safe_load, but a harness YAML may carry `!function utils.x`, which
        safe_load has no constructor for — the tag is irrelevant here, so it
        resolves to None rather than killing the scan."""
        loader = yaml.SafeLoader
        loader.add_multi_constructor("!function", lambda *_: None)
        try:
            return yaml.load(path.read_text(), Loader=loader) or {}
        except Exception:                                # a template, not a task
            return {}

    out = []
    for f in sorted((HARNESS / "lm_eval" / "tasks" / directory).rglob("*.yaml")):
        y = read(f)
        task = y.get("task")
        if not isinstance(task, str):
            continue
        ot = y.get("output_type")
        if ot is None and isinstance(y.get("include"), str):
            base = f.parent / y["include"]
            if base.is_file():
                ot = read(base).get("output_type")
        if ot != "multiple_choice":
            continue
        code = lang or task.rsplit("_", 1)[-1]
        out.append((task, LANG_ALIAS.get(code, code), benchmark))
    return out


SETS = {"ibero": IBERO, "toksuite": TOKSUITE,
        "leaderboards": [t for d, l, b in SCAN for t in scan(d, l, b)]}

BENCHMARK_META = {
    "ibero_arc": ("ARC (IberoBench: projecte-aina, HiTZ)", "https://huggingface.co/collections/BSC-LT"),
    "ibero_openbookqa": ("OpenBookQA (IberoBench)", "https://huggingface.co/collections/BSC-LT"),
    "ibero_piqa": ("PIQA (IberoBench)", "https://huggingface.co/collections/BSC-LT"),
    "ibero_siqa": ("SIQA (IberoBench)", "https://huggingface.co/collections/BSC-LT"),
    "ibero_cola": ("CoLA acceptability (EsCoLA / CatCoLA / GalCoLA)", "https://huggingface.co/nbel"),
    "ibero_truthfulqa": ("TruthfulQA mc1 (Galician)", "https://huggingface.co/proxectonos"),
    "toksuite": ("TokSuite parallel set (Altintas et al. 2025)", "https://arxiv.org/abs/2512.20757"),
    "toksuite_math": ("TokSuite math", "https://arxiv.org/abs/2512.20757"),
    "toksuite_stem": ("TokSuite STEM", "https://arxiv.org/abs/2512.20757"),
    "ceval": ("C-Eval (Chinese)", "https://cevalbenchmark.com"),
    "zhoblimp": ("ZhoBLiMP (Chinese minimal pairs)", "https://huggingface.co/datasets/Junrui1202/zhoblimp"),
    "kmmlu": ("KMMLU (Korean)", "https://huggingface.co/HAERAE-HUB"),
    "haerae": ("HAE-RAE Bench (Korean)", "https://huggingface.co/HAERAE-HUB"),
    "turkishmmlu": ("TurkishMMLU", "https://huggingface.co/AYueksel/TurkishMMLU"),
    "turblimp": ("TurBLiMP (Turkish minimal pairs)", "https://huggingface.co/juletxara/turblimp"),
    "evalita_llm": ("EVALITA-LLM (Italian)", "https://huggingface.co/evalitahf"),
    "blimp_nl": ("BLiMP-NL (Dutch minimal pairs)", "https://huggingface.co/jmichaelov/blimp_nl"),
    "noreval": ("NorEval (Norwegian)", "https://huggingface.co/ltg"),
    "bangla": ("Bangla benchmarks (hishab)", "https://huggingface.co/hishab"),
    "french_bench": ("FrenchBench", "https://huggingface.co/manu"),
    "arc_mt": ("ARC-Challenge translated (LumiOpen)", "https://huggingface.co/LumiOpen/arc_challenge_mt"),
    "mela": ("MELA (multilingual acceptability)", "https://huggingface.co/Geralt-Targaryen/MELA"),
}


def measure(names: list[str]) -> dict[str, dict]:
    """task -> {n_items, n_options}, read off the harness's own documents.

    A group resolves to its children; its counts are theirs summed, which is
    what the harness reports for the group row and what the gate then reads.
    """
    os.environ.setdefault("HF_HOME", "/iopsstor/scratch/cscs/mariagrandury/hf_home")
    os.environ.setdefault("HF_HUB_CACHE", "/capstor/store/cscs/swissai/infra01/"
                                          "users/mariagrandury/hf_models")
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(HARNESS))
    import datasets                                    # noqa: E402
    datasets.disable_progress_bars()
    from lm_eval.tasks import TaskManager              # noqa: E402

    tm = TaskManager(include_path=str(ROOT / "src" / "evals" / "tasks"))
    out: dict[str, dict] = {}
    for name in names:
        try:
            loaded = tm.load_task_or_group([name])
        except Exception as e:                          # noqa: BLE001 - reported, not raised
            print(f"  !! {name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        counts: Counter = Counter()
        items = 0
        for obj in loaded.values():
            for t in (obj.values() if isinstance(obj, dict) else [obj]):
                if not hasattr(t, "eval_docs"):
                    continue
                for d in t.eval_docs:
                    counts[len(t.doc_to_choice(d))] += 1
                    items += 1
        if not items:
            print(f"  !! {name}: no documents")
            continue
        out[name] = {"n_items": items, "n_options": _effective_options(counts),
                     "spread": dict(sorted(counts.items()))}
        print(f"  {name:32s} {items:7d} items, {out[name]['n_options']} options "
              f"{'(mixed: ' + str(out[name]['spread']) + ')' if len(counts) > 1 else ''}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", choices=sorted(SETS) + ["all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    wanted = [t for s in (SETS if args.set == "all" else [args.set]) for t in SETS[s]]
    print(f"measuring {len(wanted)} tasks through the harness")
    stats = measure([t for t, _, _ in wanted])

    data, before = read_tasks_json()
    added = skipped = 0
    for task, lang, bench in wanted:
        if task not in stats:
            skipped += 1
            continue
        entry = data["tasks"].setdefault(task, {})
        entry.update({"language": lang, "benchmark": bench, "stages": ["pretraining"],
                      "n_options": stats[task]["n_options"],
                      "n_items": stats[task]["n_items"], "metric": "acc_norm"})
        added += 1
    benches = sorted({b for t, _, b in wanted if t in stats})
    data["groups"]["auto_probe"] = sorted(set(data["groups"].get("auto_probe", [])) | set(benches))
    for b in benches:
        name, url = BENCHMARK_META[b]
        data.setdefault("benchmarks", {}).setdefault(b, {})
        data["benchmarks"][b].update({"name": name, "url": url,
                                      "languages": len({l for t, l, bb in wanted
                                                        if bb == b and t in stats})})
    print(f"\n{added} tasks registered, {skipped} unavailable; "
          f"auto_probe now {len(data['groups']['auto_probe'])} benchmarks")
    if args.dry_run:
        print("(dry-run: tasks.json not written)")
        return
    write_tasks_json(data, before)
    print("wrote configs/tasks.json")


if __name__ == "__main__":
    main()
