"""Check the analysis outputs on disk against analysis/RULES.md.

Reads every CSV under analysis/rq*/pretraining/ and reports each violation it
can detect from the tables alone; exits non-zero when it finds one. What it
checks, by rule:

  10  no size outside ANALYSIS_SIZES anywhere — neither 90M (below the
      ladder) nor a size ABOVE the reference (3B): not a value in a
      size-like column, not a column name
   7  no `multi` / `??` row in a table that has a language column
   5  no finite decision accuracy where the pair count is below MIN_PAIRS
      (long tables with an n_pairs column; the wide da_per_task against its
      da_n_pairs_per_task twin)
   6  every task name is a per-language parent, except under rq08
   3  a table with a `frac` column covers the ten tenths
  12  a CSV of the same name next to every PNG
   2  a table with `task`, `L` and `scheme` columns holds no untrained
      (task, L, scheme) row, except under rq06 and the rq00 gate

Rules 1, 4, 8, 9, 11 and 13 need the code, not the tables; the review skill
reads them off the diff.

    python analysis/check_rules.py [--root analysis] [--quiet]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.utils import (ANALYSIS_SIZES, EVAL_SIZES, FRAC_TOL, LANGUAGE_AGGREGATES,  # noqa: E402
                            MIN_PAIRS, NOISE_WINDOW, SHARED_FRACS,
                            _is_parent_task, is_trained)

# Rule 10 has two halves and the checker has to test both. Derived from the
# constants, never spelled out, so that moving TARGET_SIZE moves the check:
#   below  90M, which is trained but never evaluated (not in EVAL_SIZES)
#   above  every evaluated size past the reference — 3B today. The loader
#          drops it at utils.py:249 unless a caller passes above_reference,
#          which only the size-generalization RQ may do.
FORBIDDEN_SIZES = tuple(["90M"] + [s for s in EVAL_SIZES if s not in ANALYSIS_SIZES])

SIZE_COLS = ("size", "proxy_size", "bucket", "reference", "reference_size", "small", "target")
# folders allowed to break a rule, by rule number, each with its reason:
#  6  rq08 reads the sub-benchmarks by design; rq07 compares against AllenAI's own task names
#  2  rq06 is the untrained-language question; the rq00 gate must cover every task
#  7  the rq00 gate tables list every task, the aggregates included (they are not per-language tables)
# 12  rq00's score-curve viewers (~200 grids), rq07, rq08 and rq09 predate the convention and are
#     not yet converted (listed in RULES.md as open work); the 36-sweep pools are frozen (CLAUDE.md)
# 10  the EXTERNAL model pools have their own size axis (270M, 4B, 7-9B, ... 70B): their "3B"
#     is somebody else's model, not this ladder's rung, and the rule is about this ladder.
#     The size-generalization RQ joins this list when it lands — it is the one analysis of
#     OUR sweep that may pass above_reference=True.
EXEMPT = {10: ("custom_swissai_hf", "external"),
          6: ("rq08_subset_selection", "rq07_external_frameworks"),
          2: ("rq06_language_transfer", "rq00_gate_and_curves"),
          7: ("rq00_gate_and_curves",),
          12: ("rq07_external_frameworks", "rq08_subset_selection", "rq09_benchmark_design",
               "per_language", "score_curves", "per_benchmark",
               "seeds_1904", "seeds_28_1797", "seeds_28_1797_1904", "custom_swissai_hf", "external", "all")}


def _exempt(path: Path, rule: int) -> bool:
    return any(part in path.parts for part in EXEMPT.get(rule, ()))


LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def check_csv(path: Path) -> list[str]:
    # An unsmudged LFS pointer is three short lines that pd.read_csv parses
    # happily as a one-column table, so it raises nothing and every rule below
    # passes on a file whose contents were never read. Sniff the header instead
    # of relying on a parse error.
    try:
        with path.open("rb") as fh:
            if fh.read(len(LFS_POINTER)) == LFS_POINTER:
                return ["unreadable (git-lfs pointer, not smudged); run git lfs pull"]
    except OSError as e:
        return [f"unreadable ({type(e).__name__}: {e})"]
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:             # a table with no rows is not a violation
        return []
    except Exception as e:
        # An unreadable table is a finding, not silence: without git-lfs smudge
        # EVERY csv here is a pointer file, and returning [] would let the whole
        # check pass on a tree it never read.
        return [f"unreadable ({type(e).__name__}: {e}); is git-lfs smudged?"]
    out = []
    # rule 10 — both halves: below the ladder and above the reference
    if not _exempt(path, 10):
        for size in FORBIDDEN_SIZES:
            pat = re.compile(rf"(^|_){re.escape(size)}(_|$)")
            if any(pat.search(c) for c in df.columns):
                out.append(f"rule 10: a {size} column")
            for c in SIZE_COLS:
                if c in df.columns and (df[c].astype(str) == size).any():
                    out.append(f"rule 10: {size} in column {c}")
    # rule 7
    for c in ("language", "lang"):
        if c in df.columns and not _exempt(path, 7):
            bad = df[c].astype(str).isin(LANGUAGE_AGGREGATES).sum()
            if bad:
                out.append(f"rule 7: {bad} rows with language in {LANGUAGE_AGGREGATES}")
    # rule 5, long form
    if "n_pairs" in df.columns and "da" in df.columns:
        bad = (df["da"].notna() & ~(df["n_pairs"] >= MIN_PAIRS)).sum()   # NaN < MIN_PAIRS is False: an absent pair count would pass
        if bad:
            out.append(f"rule 5: {bad} finite DA cells with fewer than {MIN_PAIRS} pairs")
    # rule 6
    if "task" in df.columns and not _exempt(path, 6):
        tasks = df["task"].dropna().astype(str).unique()
        facets = [t for t in tasks if not _is_parent_task(t)]
        if facets:
            out.append(f"rule 6: {len(facets)} non-parent tasks, e.g. {facets[:3]}")
    # rule 3
    if "frac" in df.columns:
        fr = df["frac"].dropna().astype(float)
        have = set((fr * 10).round().astype(int) / 10)
        missing = [f for f in SHARED_FRACS if f not in have]
        if missing and len(have) > 1:
            out.append(f"rule 3: frac axis lacks {missing}")
        # the hazard rule 3 names: a checkpoint axis drawn on the twentieths.
        # Rounding to tenths hides it, so test the distance to the tenths, and
        # allow the noise window, where the k/20 points are the rule (rule 4).
        off = fr[((fr * 10).round() / 10 - fr).abs() > FRAC_TOL]
        off = off[off < 1 - NOISE_WINDOW - FRAC_TOL]
        # `*_curves` tables are single-measurement viewers (BPB on its own k/20
        # save grid, the training loss on its logging interval). Rule 3 governs
        # the axis a quantity is READ on and where kinds are compared, not how
        # densely one kind may be drawn against itself.
        if len(off) and not path.stem.endswith("_curves"):
            out.append(f"rule 3: {len(off)} rows off the tenths outside the noise "
                       f"window, e.g. {sorted(set(off.round(3)))[:3]}")
    # rule 2
    if {"task", "L", "scheme"} <= set(df.columns) and not _exempt(path, 2):
        sub = df[["task", "L", "scheme"]].dropna().drop_duplicates()
        bad = [(t, L, s) for t, L, s in zip(sub["task"], sub["L"], sub["scheme"]) if not is_trained(str(t), int(L), str(s))]
        if bad:
            out.append(f"rule 2: {len(bad)} untrained (task, L, scheme) rows, e.g. {bad[:2]}")
    return out


def check_wide_pairs(folder: Path) -> list[str]:
    """rule 5 on the wide rq02 table: da_per_task against da_n_pairs_per_task."""
    out = []
    for da_path in folder.rglob("da_per_task.csv"):
        n_path = da_path.with_name("da_n_pairs_per_task.csv")
        if not n_path.is_file():
            continue
        da = pd.read_csv(da_path, index_col="task"); n = pd.read_csv(n_path, index_col="task")
        common = [c for c in da.columns if c in n.columns]
        n_al = n[common].reindex(da.index)
        # NaN < MIN_PAIRS is False, so an absent pair count would pass silently
        bad = int((da[common].notna() & ~(n_al >= MIN_PAIRS)).sum().sum())
        if bad:
            out.append(f"{da_path.relative_to(folder)}: rule 5: {bad} finite DA cells with fewer than {MIN_PAIRS} pairs")
    return out


def check_png_csv(folder: Path) -> list[str]:
    out = []
    for png in folder.rglob("*.png"):
        if _exempt(png, 12):
            continue
        # The CSV of the same name first. Only if there is none, allow the
        # shared table grids._csv_path writes for a _by_benchmark/_by_language
        # pair — stripping first meant a panel could point at a table computed
        # for the OTHER facet and still pass (rq05's _early panels did).
        stem = png.stem
        for suffix in ("_by_benchmark", "_by_language"):
            stem = stem.removesuffix(suffix)
        if not (png.with_suffix(".csv").is_file() or png.with_name(stem + ".csv").is_file()):
            out.append(f"{png.relative_to(folder)}: rule 12: no CSV of the same name")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    root = Path(a.root)
    findings = []
    # The analysis tables, plus the paper-local ones. `documents/paper/` was
    # outside every glob until 2026-09-21, so a table the paper cites could
    # break any rule in silence — which is how a re-derived audit table came
    # to carry pair counts rule 2 forbids. The paper quotes these numbers, so
    # they are held to the same rules as the tables they come from.
    # analysis/ -> src/signal-and-noise -> src -> the repo root
    repo = root.resolve().parents[2]
    paper = repo / "documents" / "paper"
    checked = sorted(root.glob("rq*/pretraining/**/*.csv"))
    checked += sorted(paper.rglob("*.csv")) if paper.is_dir() else []
    for csv in checked:
        base = root if root.resolve() in csv.resolve().parents else repo
        for f in check_csv(csv):
            findings.append(f"{csv.relative_to(base)}: {f}")
    findings += check_wide_pairs(root) + check_png_csv(root)
    if not a.quiet:
        for f in findings:
            print(f)
    by_rule = pd.Series([re.search(r"rule (\d+)", f).group(1) for f in findings if re.search(r"rule (\d+)", f)]).value_counts()
    print(f"check_rules: {len(findings)} findings" + (f" (by rule: {by_rule.to_dict()})" if len(findings) else ""))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
