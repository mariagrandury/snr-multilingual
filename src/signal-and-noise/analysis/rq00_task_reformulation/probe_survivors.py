#!/usr/bin/env python3
"""Which probe benchmarks survive the above-random gate, per language.

The decision the probe exists for: of the candidates in `groups.auto_probe`,
which carry signal at each size, in which languages, and does the `rf_` twin
(where one exists) survive where the original does not? Everything here is
read off the committed gate mask (rule 1, `above_random.load_mask`) — no
score is recomputed — so it agrees with every other RQ by construction. Run
`above_random.py --only predictivity` first, with SNR_TRAINED_GROUPS naming
the probe group so the multilingual candidates are gated on the cells that
trained their language (probe.sh does both).

Writes, next to this file:
    probe_survivors.csv   one row per (task, size): benchmark, language and
                          the gate's verdict (1 / 0 / NA)
    README.md             the `probe-survivors` block: benchmark x size
                          (languages surviving, original | rf) and
                          language x size (the benchmarks surviving)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src" / "signal-and-noise"))
from analysis.autodoc import replace_block  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.utils import ANALYSIS_SIZES, languages_only  # noqa: E402

TASKS_JSON = ROOT / "configs" / "tasks.json"
POOL = "predictivity"
GROUP = "auto_probe"


def table(rows: pd.DataFrame) -> pd.DataFrame:
    """(benchmark, language, size) -> above_orig, above_rf, from the long rows.

    A benchmark with several tasks in one language (blend has two Spanish
    ones) survives there if any of them does; a cell the gate left NA stays
    NA -- pivot_table's default drops those rows outright, and an all-NA
    probe then reads as an empty table instead of an ungated one."""
    rows = rows.assign(set=rows["benchmark"].str.startswith("rf_").map({True: "rf", False: "orig"}),
                       base=rows["benchmark"].str.replace(r"^rf_", "", regex=True),
                       above=rows["above"].astype(float))
    w = rows.pivot_table(index=["base", "language", "size"], columns="set", values="above",
                         aggfunc="max", dropna=False)
    return w.reindex(columns=["orig", "rf"]).rename(columns={"orig": "above_orig", "rf": "above_rf"}).reset_index()


def main() -> None:
    cfg = json.loads(TASKS_JSON.read_text())
    probe, tasks = set(cfg["groups"][GROUP]), cfg["tasks"]
    mask = load_mask(POOL)
    if mask is None:
        sys.exit(f"no gate mask for pool {POOL}: run above_random.py --only {POOL} first")
    sizes = [s for s in ANALYSIS_SIZES if s in mask.columns]
    rows = [{"benchmark": e["benchmark"], "task": t, "language": e["language"], "size": s,
             "above": mask.at[t, s]}
            for t, e in tasks.items()
            if e.get("benchmark") in probe and "pretraining" in e.get("stages", []) and t in mask.index
            for s in sizes]
    if not rows:
        sys.exit(f"the mask holds no task of {GROUP}: the probe evals are not in the ladder report yet")
    long = languages_only(pd.DataFrame(rows))          # rule 7
    long["above"] = long["above"].astype("Int64")
    if long["above"].isna().all():
        sys.exit(f"every {GROUP} task in the mask is ungated (no score at any size): nothing to report yet")
    long.to_csv(HERE / "probe_survivors.csv", index=False)   # task level, the gate's own granularity
    tab = table(long)

    # benchmark x size: languages surviving, "orig k/n · rf k/n" (n = languages gated)
    def count(g: pd.DataFrame, col: str) -> str:
        v = g[col].dropna()
        return f"{int(v.sum())}/{len(v)}" if len(v) else "—"
    bench = ["| benchmark | " + " | ".join(sizes) + " |", "|---|" + "---:|" * len(sizes)]
    for b, g in tab.groupby("base", sort=True):
        cells = []
        for s in sizes:
            gs = g[g["size"] == s]
            c = f"orig {count(gs, 'above_orig')}"
            if gs["above_rf"].notna().any():
                c += f" · rf {count(gs, 'above_rf')}"
            cells.append(c)
        bench.append(f"| {b} | " + " | ".join(cells) + " |")
    # language x size: the benchmarks surviving, twin marked -rf
    lang = ["| language | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]
    for l, g in tab.groupby("language", sort=True):
        cells = []
        for s in sizes:
            gs = g[g["size"] == s]
            names = sorted(set(gs.loc[gs["above_orig"] == 1, "base"])
                           | {f"{b}-rf" for b in gs.loc[gs["above_rf"] == 1, "base"]})
            cells.append(", ".join(names) if names else "—")
        lang.append(f"| {l} | " + " | ".join(cells) + " |")
    n_b, n_l = tab["base"].nunique(), tab["language"].nunique()
    lines = [f"Probe survivors: which of the {n_b} candidate benchmarks in `auto_probe` clear the above-random gate "
             f"(rule 1, read from the committed `{POOL}` mask, cells that trained the language), over {n_l} languages. "
             "A cell counts languages (original | rf twin where one exists) — the population differs per cell "
             "(rule 13) — and the second table names the surviving benchmarks per language, the twin as `-rf`. "
             "Same items as `rf_gate_probe` (`compare.py --tag probe`), which measures how far above chance; this "
             "table is only the gate.", ""] + bench + [""] + lang
    replace_block(HERE / "README.md", "probe-survivors", "\n".join(lines),
                  "analysis/rq00_task_reformulation/probe_survivors.py")
    print("\n".join(lines))
    print(f"\nwrote {HERE / 'probe_survivors.csv'} and the probe-survivors block of {HERE / 'README.md'}")


if __name__ == "__main__":
    main()
