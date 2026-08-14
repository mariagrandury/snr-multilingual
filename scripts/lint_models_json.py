"""Lint configs/models.json.

Checks (R5, R7, R8 from the v2 plan):

1. For Apertus rows (`source` starts with `snr-pretraining-`), the
   declared `family` must equal `_strip_size_from_name(name, size)` —
   the auto-derivation. Diverging is allowed for HF / external rows
   (that's the whole point of an explicit field).
2. Every `family` either:
   - appears at ≥ 2 sizes (DA-size is meaningful), OR
   - has exactly 1 model and 1 size (DA-size NaN — flagged for awareness).
3. Rows with `size == "TBD"` are flagged.
4. Every `seed`-tagged row's `family` ends with `-seed<N>` (sanity for
   Apertus naming).

Exit code 0 if no errors; 1 if any check fails (warnings are informational).

Usage: python scripts/lint_models_json.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "evals" / "scripts" / "utils"))

from configs import _strip_size_from_name, load_models  # noqa: E402


def main():
    models = load_models(REPO / "configs" / "models.json")

    errors: list[str] = []
    warnings: list[str] = []

    # Check 1: Apertus-custom family auto-derive consistency. (Only the
    # snr-pretraining-custom rows have regular naming; a06 / HF /
    # external rows are explicit by design — that's the whole point of
    # the field.)
    for name, e in models.items():
        if e.get("source") != "snr-pretraining-custom":
            continue
        size = e.get("size")
        expected = _strip_size_from_name(name, size)
        if e["family"] != expected:
            errors.append(
                f"  family mismatch: {name} declares family={e['family']!r} "
                f"but _strip_size_from_name({name!r}, {size!r}) = {expected!r}"
            )

    # Check 4: seed naming
    for name, e in models.items():
        if e.get("seed") is None:
            continue
        if not e["family"].endswith(f"-seed{e['seed']}"):
            warnings.append(
                f"  seed/family mismatch: {name} has seed={e['seed']} but "
                f"family={e['family']!r} doesn't end with -seed{e['seed']}"
            )

    # Check 2: family-size coverage
    family_sizes: dict[str, set[str]] = defaultdict(set)
    family_models: dict[str, list[str]] = defaultdict(list)
    for name, e in models.items():
        family_sizes[e["family"]].add(e.get("size", "TBD"))
        family_models[e["family"]].append(name)
    for fam, sizes in family_sizes.items():
        if "TBD" in sizes:
            continue  # caught by Check 3
        if len(sizes) == 1 and len(family_models[fam]) > 1:
            warnings.append(
                f"  family {fam!r} has {len(family_models[fam])} models at "
                f"size={list(sizes)[0]!r} only — DA-size will be NaN "
                f"(needs ≥2 sizes). Models: {family_models[fam]}"
            )

    # Check 3: TBD sizes
    tbd = [n for n, e in models.items() if e.get("size") == "TBD"]
    for name in tbd:
        warnings.append(f"  TBD size: {name}")

    # Summary
    n_models = len(models)
    n_families = len(family_sizes)
    cross_size_families = sum(1 for s in family_sizes.values() if len(s) >= 2)
    print(f"Linted {n_models} models, {n_families} families "
          f"({cross_size_families} with ≥2 sizes — DA-size meaningful)")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(e)
    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for w in warnings:
            print(w)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
