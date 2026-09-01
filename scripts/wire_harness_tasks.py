#!/usr/bin/env python3
"""
Wire every per-language task the installed lm-eval harness offers for the
FineWeb-2 language pool into configs/tasks.json — the "evaluate on all the
benchmarks available for the training languages" policy (team decision
2026-08-21, plan/team-discussion-2026-08-21.md point 4B), and the benchmark
availability that src/pretrain/data/generate_language_sets.py reads.

For each multilingual family below, the harness's task names are scanned
(the `task:` field of every yaml under lm_eval/tasks), the language code is
parsed out of the name, mapped to the FineWeb-2 subset it evaluates, and an
entry {language, benchmark, stages} is added to tasks.json under the project's
canonical language tag (configs/languages.json `fineweb_iso2`: iso639-1 where
one exists, else the FineWeb iso3; Arabic dialects fold into `ar`). Existing
entries are left untouched; the `benchmarks` section records the paper/venue
per family (see plan/benchmark_selection.md); every pretraining-stage family
is listed in the `auto` group. Idempotent.

`--report` needs no harness: it reads the wired state back and regenerates the
per-language coverage table in plan/benchmark_selection.md (what every trained
language is actually evaluated on, and where each language enters the ladder),
so the doc cannot drift from configs/tasks.json.

Usage:
    python scripts/wire_harness_tasks.py [--harness DIR] [--top N] [--dry-run]
    python scripts/wire_harness_tasks.py --report [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_JSON = ROOT / "configs" / "tasks.json"
LANGUAGES_JSON = ROOT / "configs" / "languages.json"
FINEWEB_CSV = ROOT / "src" / "pretrain" / "data" / "fineweb2-language-distribution.csv"

# FineWeb-2 subset -> the code a benchmark uses when it differs from the
# FineWeb iso3_Script (FLORES-200 conventions: zho/pes/est/tgl, Hant for Yue).
FLORES_ALIAS = {"cmn_Hani": "zho_Hans", "fas_Arab": "pes_Arab", "ekk_Latn": "est_Latn",
                "fil_Latn": "tgl_Latn", "yue_Hani": "yue_Hant"}
ISO3_ALIAS = {"cmn": "zho", "fas": "pes", "ekk": "est", "fil": "tgl"}

# family -> (task-name regex with a `code` group, code system, stage, paper)
# code systems: flores (xxx_Xxxx), flores_lc (xxx_xxxx[_region]), iso3, iso2
FAMILIES = {
    "belebele":      (r"^belebele_(?P<code>[a-z]{3}_[A-Z][a-z]{3})$", "flores", "pretraining"),
    "global_piqa_completions": (r"^global_piqa_completions_(?P<code>[a-z]{3}_[a-z]{4})(?:_[a-z]+)?$", "flores_lc", "pretraining"),
    "global_mmlu_full": (r"^global_mmlu_full_(?P<code>[a-z]{2,3})$", "iso2", "pretraining"),
    "multiblimp":    (r"^multiblimp_(?P<code>[a-z]{3})$", "iso3", "pretraining"),
    "hellaswag":     (r"^hellaswag_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "arc":           (r"^arc_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "xnli":          (r"^xnli_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "xstorycloze":   (r"^xstorycloze_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "xcopa":         (r"^xcopa_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "xwinograd":     (r"^xwinograd_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "paws":          (r"^paws_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "lambada_openai_mt": (r"^lambada_openai_mt_(?P<code>[a-z]{2})$", "iso2", "pretraining"),
    "afrixnli":      (r"^afrixnli_(?P<code>[a-z]{3})_prompt_1$", "iso3", "pretraining"),
    "afrimmlu":      (r"^afrimmlu_direct_(?P<code>[a-z]{3})_prompt_1$", "iso3", "pretraining"),
    "afrimgsm":      (r"^afrimgsm_(?P<code>[a-z]{3})$", "iso3", "midtraining"),
    "mgsm_direct":   (r"^mgsm_direct_(?P<code>[a-z]{2})$", "iso2", "midtraining"),
}

# Paper / venue per benchmark family (plan/benchmark_selection.md has the
# assessment; this is the machine-readable pointer the user asked for).
BENCHMARKS = {
    "belebele": {"paper": "https://aclanthology.org/2024.acl-long.44/", "venue": "ACL 2024", "name": "Belebele (Bandarkar et al.)", "languages": 122, "format": "MC reading comprehension, 900 items/lang, human-translated"},
    "global_piqa_completions": {"paper": "https://arxiv.org/abs/2510.24081", "venue": "MRL workshop @ EMNLP 2025 shared task", "name": "Global PIQA (Chang et al.)", "languages": 116, "format": "2-way physical-commonsense completion, hand-written per language by native speakers"},
    "global_mmlu_full": {"paper": "https://aclanthology.org/2025.acl-long.919/", "venue": "ACL 2025", "name": "Global-MMLU (Singh et al.)", "languages": 42, "format": "MC knowledge QA (MMLU), professional + community translations"},
    "include_base_44": {"paper": "https://openreview.net/forum?id=k3gCieTXeY", "venue": "ICLR 2025", "name": "INCLUDE (Romanou et al.)", "languages": 44, "format": "MC regional exam questions, natively sourced"},
    "multiblimp": {"paper": "https://aclanthology.org/2026.tacl-1.10/", "venue": "TACL 2026", "name": "MultiBLiMP 1.0 (Jumelet et al.)", "languages": 101, "format": "grammatical minimal pairs (subject-verb agreement), auto-generated from UD/UniMorph"},
    "hellaswag": {"paper": "https://aclanthology.org/2023.emnlp-demo.28/", "venue": "EMNLP 2023 (demo); English original Zellers et al. ACL 2019", "name": "Okapi m_hellaswag (Lai et al.)", "languages": 31, "format": "4-way sentence completion, machine-translated (ChatGPT)"},
    "arc": {"paper": "https://aclanthology.org/2023.emnlp-demo.28/", "venue": "EMNLP 2023 (demo); English original Clark et al. 2018", "name": "Okapi m_arc (Lai et al.)", "languages": 31, "format": "MC science QA, machine-translated (ChatGPT)"},
    "xnli": {"paper": "https://aclanthology.org/D18-1269/", "venue": "EMNLP 2018", "name": "XNLI (Conneau et al.)", "languages": 15, "format": "3-way NLI, professionally translated"},
    "xstorycloze": {"paper": "https://aclanthology.org/2022.emnlp-main.616/", "venue": "EMNLP 2022 (XGLM)", "name": "XStoryCloze (Lin et al.)", "languages": 11, "format": "2-way story ending, professionally translated"},
    "xcopa": {"paper": "https://aclanthology.org/2020.emnlp-main.185/", "venue": "EMNLP 2020", "name": "XCOPA (Ponti et al.)", "languages": 11, "format": "2-way causal commonsense, professionally translated"},
    "xwinograd": {"paper": "https://aclanthology.org/2021.findings-acl.310/", "venue": "Findings of ACL 2021", "name": "XWinograd (Tikhonov & Ryabinin)", "languages": 6, "format": "Winograd schema coreference"},
    "paws": {"paper": "https://aclanthology.org/D19-1382/", "venue": "EMNLP 2019", "name": "PAWS-X (Yang et al.)", "languages": 7, "format": "paraphrase identification, professionally translated"},
    "lambada_openai_mt": {"paper": "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/lambada_multilingual", "venue": "no paper (EleutherAI machine translation of LAMBADA, Paperno et al. ACL 2016)", "name": "LAMBADA multilingual", "languages": 5, "format": "last-word prediction, machine-translated"},
    "afrixnli": {"paper": "https://aclanthology.org/2025.naacl-long.139/", "venue": "NAACL 2025", "name": "IrokoBench AfriXNLI (Adelani et al.)", "languages": 17, "format": "3-way NLI, human-translated (XNLI subset)"},
    "afrimmlu": {"paper": "https://aclanthology.org/2025.naacl-long.139/", "venue": "NAACL 2025", "name": "IrokoBench AfriMMLU (Adelani et al.)", "languages": 17, "format": "MC knowledge QA, human-translated (MMLU subset, 5 subjects)"},
    "afrimgsm": {"paper": "https://aclanthology.org/2025.naacl-long.139/", "venue": "NAACL 2025", "name": "IrokoBench AfriMGSM (Adelani et al.)", "languages": 17, "format": "grade-school math generation, human-translated (MGSM)"},
    "mgsm_direct": {"paper": "https://openreview.net/forum?id=fR3wGCk-IXp", "venue": "ICLR 2023", "name": "MGSM (Shi et al.)", "languages": 10, "format": "grade-school math generation, human-translated"},
    "truthfulqa-multi_mc1": {"paper": "https://arxiv.org/abs/2502.09387", "venue": "arXiv 2025", "name": "TruthfulQA-Multi (Calvo Figueras et al.)", "languages": 5, "format": "MC truthfulness, professionally translated"},
}


# Tasks whose language was recorded as "??" (underivable from the name) or
# with a wrong/missing tag — fixed here so the whole tasks.json state is
# reproducible from this script (2026-08-21 review, finding 4a).
LANGUAGE_TAG_FIXES = {
    "include_base_44": "multi", "ceval-valid": "zh",
    "include_base_44_albanian": "sq", "include_base_44_armenian": "hy",
    "include_base_44_azerbaijani": "az", "include_base_44_belarusian": "be",
    "include_base_44_bengali": "bn", "include_base_44_bulgarian": "bg",
    "include_base_44_croatian": "hr", "include_base_44_dutch": "nl",
    "include_base_44_finnish": "fi", "include_base_44_georgian": "ka",
    "include_base_44_greek": "el", "include_base_44_hebrew": "he",
    "include_base_44_hungarian": "hu", "include_base_44_indonesian": "id",
    "include_base_44_kazakh": "kk", "include_base_44_lithuanian": "lt",
    "include_base_44_malay": "ms", "include_base_44_malayalam": "ml",
    "include_base_44_nepali": "ne", "include_base_44_north_macedonian": "mk",
    "include_base_44_persian": "fa", "include_base_44_polish": "pl",
    "include_base_44_serbian": "sr", "include_base_44_tagalog": "tl",
    "include_base_44_tamil": "ta", "include_base_44_uzbek": "uz",
}


def harness_task_names(harness: Path) -> set[str]:
    names = set()
    for y in harness.rglob("*.yaml"):
        for line in y.read_text(errors="ignore").splitlines():
            m = re.match(r"^task:\s*([A-Za-z0-9_.\-]+)\s*$", line)
            if m:
                names.add(m.group(1))
    return names


def language_pool(top: int) -> list[str]:
    rows = {}
    with open(FINEWEB_CSV) as f:
        for r in csv.DictReader(f):
            s = r["subset"]
            if r["split"] == "train" and not s.endswith("_removed") and not s.startswith("und_") and s != "eng_Latn":
                rows[s] = int(r["utf8_bytes"])
    return sorted(rows, key=lambda k: -rows[k])[:top]


DOC = ROOT / "plan" / "benchmark_selection.md"
DOC_BEGIN = "<!-- BEGIN generated: scripts/wire_harness_tasks.py --report -->"
DOC_END = "<!-- END generated -->"


def coverage_table() -> str:
    """The auto set per trained language: what `tasks_for_benchmarks` x
    `cell_languages` actually selects, which is what the watchers run."""
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "pretrain")]
    from launch_trainings import LANG_SETTINGS, cell_languages
    from evals.scripts.utils.configs import tasks_for_benchmarks

    data = json.loads(TASKS_JSON.read_text())
    auto, tasks = data["groups"]["auto"], data["tasks"]
    names = {}                      # language tag -> FineWeb display name
    with open(FINEWEB_CSV) as f:
        iso2 = json.loads(LANGUAGES_JSON.read_text())["fineweb_iso2"]
        for r in csv.DictReader(f):
            tag = iso2.get(r["subset"].split("_")[0])
            if r["split"] == "train" and tag:
                names.setdefault(tag, r["name"])
    names["en"] = "English"

    enters, per_setting = {}, {}
    for L in LANG_SETTINGS:
        langs = cell_languages(L)
        per_setting[L] = len(tasks_for_benchmarks(auto, langs))
        for lang in sorted(langs):
            enters.setdefault(lang, L)

    fams: dict[str, dict[str, int]] = {}
    for name in tasks_for_benchmarks(auto, set(enters)):
        e = tasks[name]
        fams.setdefault(e["language"], {}).setdefault(e["benchmark"], 0)
        fams[e["language"]][e["benchmark"]] += 1

    out = [DOC_BEGIN, "",
           "Tasks per cell (auto group x trained languages): "
           + " · ".join(f"L{L} {n}" for L, n in per_setting.items()) + ".", "",
           "| Enters at | Language | Families | Tasks | Benchmark families |",
           "|---|---|---|---:|---|"]
    for lang, L in sorted(enters.items(), key=lambda kv: (kv[1], -len(fams.get(kv[0], {})), kv[0])):
        f = fams.get(lang, {})
        out.append(f"| L{L} | `{lang}` {names.get(lang, '')} | {len(f)} | "
                   f"{sum(f.values())} | {', '.join(sorted(f)) or '**none**'} |")
    counts = sorted(Counter(len(fams.get(l, {})) for l in enters).items())
    out += ["", "Languages by number of families: "
            + " · ".join(f"{n}→{k}" for n, k in counts) + ".", "", DOC_END]
    return "\n".join(out)


def write_report(dry_run: bool) -> None:
    table = coverage_table()
    print(table if dry_run else
          f"{sum(l.startswith('| L') for l in table.splitlines())} languages tabulated")
    if dry_run:
        return
    text = DOC.read_text()
    if DOC_BEGIN not in text or DOC_END not in text:
        sys.exit(f"no {DOC_BEGIN} / {DOC_END} markers in {DOC.name}")
    head, rest = text.split(DOC_BEGIN, 1)
    DOC.write_text(head + table + rest.split(DOC_END, 1)[1])
    print(f"wrote {DOC}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", type=Path, default=None, help="lm_eval/tasks dir (default: the installed lm_eval)")
    p.add_argument("--top", type=int, default=150, help="FineWeb-2 pool: top-N subsets by bytes (default 150)")
    p.add_argument("--report", action="store_true",
                   help="regenerate the coverage table in plan/benchmark_selection.md and exit")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.report:
        return write_report(args.dry_run)

    harness = args.harness
    if harness is None:
        import lm_eval
        harness = Path(lm_eval.__file__).parent / "tasks"
    names = harness_task_names(harness)

    iso2 = json.loads(LANGUAGES_JSON.read_text())["fineweb_iso2"]
    pool = language_pool(args.top) + ["eng_Latn"]
    # every code system -> FineWeb subset(s) it denotes
    by_code = {"flores": {}, "flores_lc": {}, "iso3": {}, "iso2": {}}
    for subset in pool:
        iso3 = subset.split("_")[0]
        tag = "en" if subset == "eng_Latn" else iso2.get(iso3)
        if tag is None:
            continue
        by_code["flores"][FLORES_ALIAS.get(subset, subset)] = tag
        by_code["flores_lc"][FLORES_ALIAS.get(subset, subset).lower()] = tag
        by_code["iso3"][ISO3_ALIAS.get(iso3, iso3)] = tag
        by_code["iso2"][tag] = tag
    by_code["iso2"]["jp"] = "ja"  # xwinograd_jp

    data = json.loads(TASKS_JSON.read_text())
    tasks = data["tasks"]
    retagged = [t for t, lang in LANGUAGE_TAG_FIXES.items()
                if t in tasks and tasks[t]["language"] != lang]
    for t in retagged:
        tasks[t]["language"] = LANGUAGE_TAG_FIXES[t]
    added = {}
    for fam, (rx, system, stage) in FAMILIES.items():
        for name in sorted(names):
            m = re.match(rx, name)
            if not m:
                continue
            tag = by_code[system].get(m.group("code"))
            if tag is None or name in tasks:
                continue
            tasks[name] = {"language": tag, "benchmark": fam, "stages": [stage]}
            added.setdefault(fam, []).append(name)

    data["benchmarks"] = {**BENCHMARKS, **data.get("benchmarks", {})}
    auto = data["groups"]["auto"]
    for fam in ("global_piqa", "afrixnli", "afrimmlu", "paws", "lambada_openai_mt", "truthfulqa-multi_mc1"):
        if fam not in auto:
            auto.append(fam)
    auto.sort()

    n = sum(len(v) for v in added.values())
    for fam, v in added.items():
        print(f"{fam:24} +{len(v):3}  e.g. {', '.join(v[:4])}")
    print(f"{len(retagged)} tasks retagged; {n} tasks added; auto group: {auto}")
    if args.dry_run:
        return
    # Existing order is preserved (new tasks append, sorted) so re-runs and
    # cluster-side edits (e.g. derive_task_options.py adding n_options) diff
    # minimally. n_options for the new tasks is derived on the cluster from
    # real eval samples once they have run.
    TASKS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {TASKS_JSON}")


if __name__ == "__main__":
    main()
