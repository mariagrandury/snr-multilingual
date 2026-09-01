#!/usr/bin/env python3
"""
Generate language_sets_schemeA.json and language_sets_schemeB.json from the
FineWeb-2 language distribution and the benchmark availability in
configs/tasks.json — so the lists are reproducible from stated rules instead
of hand-edited (team decision 2026-08-21, plan/team-discussion-2026-08-21.md
point 4A).

Inputs
  fineweb2-language-distribution.csv   per-subset train-split UTF-8 bytes, script, family
  configs/tasks.json                   which benchmark families exist per language
                                       (pretraining-stage tasks, tagged with the
                                       project's language codes)
  configs/languages.json               FineWeb iso3 -> project language code

Rules (English is the DCLM half and is never listed; FW_L<L> has L-1 entries)
  Pool     FineWeb-2 train subsets ranked by UTF-8 bytes, excluding `und_*`
           (unidentified language) and the subsets absent from the swiss-ai
           filtered dataset dir (UNAVAILABLE below).
  Scheme A FW_L2..FW_L50: the top-(L-1) of the pool. FW_L100: the top-99, minus
           any subset with no benchmark family in tasks.json, the vacated
           slots filled by the next subsets by bytes that have at least two
           families (fill-ins are low-resource, so they must be evaluable by
           more than one benchmark) and are not a script variant of a language
           already in the list. Sets are nested by construction.
  Scheme B FW_L8/FW_L15/FW_L30 are diversity-first within the top-49: the first
           subset (by bytes) of each not-yet-covered script, then of each
           not-yet-covered family, then the remaining top-49 by bytes, until
           L-1 entries. FW_L50 and FW_L100 are scheme A's.

Usage
  python generate_language_sets.py            # rewrite both JSONs
  python generate_language_sets.py --check    # exit 1 if the JSONs are stale
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CSV = HERE / "fineweb2-language-distribution.csv"
TASKS_JSON = ROOT / "configs" / "tasks.json"
LANGUAGES_JSON = ROOT / "configs" / "languages.json"

SETTINGS = [2, 8, 15, 30, 50, 100]
DIVERSITY_SETTINGS = [8, 15, 30]   # scheme B differs from A only here
DIVERSITY_POOL = 49                # ... and picks within scheme A's FW_L50
# Subsets missing from the swiss-ai fineweb-2 filtered dir the builds read
# (create_data_mixture.py raises "Languages not found" for them; hau_Latn hit
# that on 2026-08-16, commit f57a4b7).
UNAVAILABLE = {"hau_Latn": "absent from the swiss-ai filtered dataset dir"}
# L100 benchmark conditions: keep a top-99 subset if it has >= MIN_KEEP
# families; fill a vacated slot only with a subset having >= MIN_FILL.
MIN_KEEP, MIN_FILL = 1, 2


def ranked_pool() -> list[dict]:
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            s = r["subset"]
            if (r["split"] == "train" and not s.endswith("_removed")
                    and not s.startswith("und_") and s != "eng_Latn" and s not in UNAVAILABLE):
                rows.append({"code": s, "bytes": int(r["utf8_bytes"]),
                             "script": r["script"], "family": r["family"]})
    return sorted(rows, key=lambda r: -r["bytes"])


def benchmark_families() -> dict[str, int]:
    """project language code -> number of distinct pretraining-stage benchmark
    families in tasks.json."""
    tasks = json.loads(TASKS_JSON.read_text())["tasks"]
    fams: dict[str, set] = {}
    for t in tasks.values():
        if "pretraining" in t.get("stages", []):
            fams.setdefault(t["language"], set()).add(t["benchmark"])
    return {k: len(v) for k, v in fams.items()}


def scheme_a(pool: list[dict], n_fam) -> dict[str, list[str]]:
    sets = {}
    for L in SETTINGS:
        k = L - 1
        if L < 100:
            sets[f"FW_L{L}"] = [r["code"] for r in pool[:k]]
            continue
        kept = [r for r in pool[:k] if n_fam(r) >= MIN_KEEP]
        # Fill-ins must add a NEW language: a script variant of a language
        # already kept (e.g. romanised hin_Latn next to hin_Deva) inherits
        # that language's benchmarks without being evaluable by them.
        langs = {r["code"].split("_")[0] for r in kept}
        fill = [r for r in pool[k:]
                if n_fam(r) >= MIN_FILL and r["code"].split("_")[0] not in langs][: k - len(kept)]
        sets[f"FW_L{L}"] = [r["code"] for r in sorted(kept + fill, key=lambda r: -r["bytes"])]
    return sets


def scheme_b(pool: list[dict], a: dict[str, list[str]]) -> dict[str, list[str]]:
    top = pool[:DIVERSITY_POOL]
    ordered = []  # diversity order: new scripts, then new families, then bytes
    for key in ("script", "family"):
        seen = {r[key] for r in top if r["code"] in ordered}
        for r in top:
            if r["code"] not in ordered and r[key] not in seen:
                ordered.append(r["code"]); seen.add(r[key])
    ordered += [r["code"] for r in top if r["code"] not in ordered]
    sets = dict(a)
    for L in DIVERSITY_SETTINGS:
        sets[f"FW_L{L}"] = ordered[: L - 1]
    return sets


def describe(scheme: str) -> str:
    base = ("Generated by src/pretrain/data/generate_language_sets.py from "
            "fineweb2-language-distribution.csv (train-split UTF-8 bytes) and "
            "configs/tasks.json (benchmark availability) — do not edit by hand. "
            "Pool: FineWeb-2 subsets ranked by bytes, excluding und_* and "
            + ", ".join(f"{k} ({v})" for k, v in UNAVAILABLE.items()) + ". ")
    a = (f"Scheme A: FW_L2..FW_L50 = top-(L-1) of the pool; FW_L100 = top-99 minus "
         f"subsets with < {MIN_KEEP} benchmark family in tasks.json, vacated slots "
         f"filled by the next subsets by bytes with >= {MIN_FILL} families that are not a script variant of a kept language. Nested. "
         "English is supplied separately as the DCLM 50% and is not listed.")
    b = (f"Scheme B: FW_L8/L15/L30 are diversity-first within the top-{DIVERSITY_POOL}: "
         "first subset by bytes of each uncovered script, then of each uncovered "
         "family, then by bytes; FW_L50/FW_L100 are scheme A's. " + a.split("Nested. ")[1])
    return base + (a if scheme == "A" else b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify the JSONs match; don't write")
    args = p.parse_args()

    pool = ranked_pool()
    iso2 = json.loads(LANGUAGES_JSON.read_text())["fineweb_iso2"]
    fam_count = benchmark_families()
    n_fam = lambda r: fam_count.get(iso2.get(r["code"].split("_")[0], ""), 0)

    a = scheme_a(pool, n_fam)
    b = scheme_b(pool, a)
    for sets in (a, b):
        for small, big in zip(SETTINGS, SETTINGS[1:]):
            assert set(sets[f"FW_L{small}"]) <= set(sets[f"FW_L{big}"]), (small, big)

    stale = False
    for scheme, sets in (("A", a), ("B", b)):
        path = HERE / f"language_sets_scheme{scheme}.json"
        out = {"scheme": scheme, "description": describe(scheme), "sets": sets}
        if args.check:
            cur = json.loads(path.read_text()) if path.exists() else None
            if cur != out:
                stale = True
                print(f"STALE: {path.name}")
        else:
            path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {path.name}: " + ", ".join(f"L{L}={len(sets[f'FW_L{L}'])}" for L in SETTINGS))
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
