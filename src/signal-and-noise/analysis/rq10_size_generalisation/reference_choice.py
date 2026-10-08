"""Two other references on the decisions they can score: the 1B rung at each
of its three seeds, and the one 3B model at K = 30.

Every other reading takes the 1.7B final of seed 1904 as the reference (rule
9), which has no replicate, and rq10's `above_reference.py` pools the six 3B
families. This asks what DA-size and DA-goal read when the reference is

  1B seeds   the 1B final of one seed (28, 1797 or 1904) of the four designs the
             grid trains at three seeds at 1B (deep L1, L2, L30-A, L30-B), the
             proxies the seed-1904 runs of the same designs at 90M–600M; and the
             same proxies read to the 1.7B final, for comparison. The spread over
             the three reference seeds is the reference-side seed noise of one
             proxy reading, which the 1.7B reference cannot show.
  3B K=30    the 3B final, over the decisions that involve the one K = 30 family
             at 3B (A-L30: B-L30 is not in the report), i.e. its pairs with the
             five other 3B families; proxies 90M–1.7B. The same pairs read to the
             1.7B final, and every 3B pair (rq10's pooled reading) beside it.

Pairs at the grid seed (`utils.pair_sets`, multi-axis and mono-axis, rule 15);
a (task, pair set, size, checkpoint) cell needs MIN_PAIRS pairs (rule 5) and
the task above chance at the proxy and at the reference (rule 1: `predictivity`'s
mask, and at 3B the Wilson rule on the 3B runs, `above_reference.gate_mask`).
Within a block, a cell is kept only where every reading of the block that has
that (pair set, size, checkpoint) scores it, so the readings differ in the
reference alone (rule 13; the `all 3B pairs` reading has more pairs per task).
Benchmark accuracy (the bBPB twins included) and per-language BPB are pooled
apart, matching pairs over pairs. The 3B K = 30 cell has no BPB final yet.

!!! RULE 9: the 1B rows read a reference below 1.7B on purpose: the 1B rung is
the only one above 600M with seed replicates. !!! RULE 10: the 3B rows read the
rung above the reference, which this RQ alone may do (`above_reference=True`).

    reference_choice_da_all_both_axes.csv        every reading, DA-size (frac 1) and DA-goal (the nine earlier tenths)
    reference_choice_da_size_both_axes.csv/.tex  the paper table: DA-size of benchmark accuracy, both pair sets

    python analysis/rq10_size_generalisation/reference_choice.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import SIZE_GENERALISATION  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import FRACS, decisions, grid_frame, reliability  # noqa: E402
from analysis.rq10_size_generalisation.above_reference import GITHUB, REFERENCE, gate_mask  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, MIN_PAIRS, NON_EMB, PAIR_AXES, TARGET_SIZE, benchmark_family, design_axes, finals, ladder_frame,
    on_shared_grid, pair_sets, passes_gate, size_order)

OUT_ROOT = SIZE_GENERALISATION
POOL = "predictivity_seeds"            # every seed: the 1B replicates live here
SEED_SIZE = "1B"                       # the one rung above 600M trained at three seeds
K30 = 30
STEM = "reference_choice_da"


def read(df, fin, fams, ref_size, ref_fin, mask, keep=lambda pair: True) -> pd.DataFrame:
    """Per (task, pair set, proxy size, frac): DA against `ref_fin`'s finals at
    `ref_size`, over the pairs of `fams` that `keep` admits; rules 5 and 1."""
    attrs = design_axes(fin[fin["family"].isin(fams)])
    groups = {k: [p for p in v if keep(p)] for k, v in pair_sets(attrs).items() if k in PAIR_AXES[:2]}
    proxies = [s for s in size_order(df["size"].unique()) if NON_EMB[s] < NON_EMB[ref_size]]
    frame = grid_frame(df[df["family"].isin(fams) & df["size"].isin(proxies)], FRACS)
    c = reliability(decisions(frame, groups, proxies, ref_fin, fracs=FRACS, ref=ref_size))
    c = c[c["n_comparable"] >= MIN_PAIRS]                                               # rule 5
    keep_gate = pd.Series(False, index=c.index)
    for size, g in c.groupby("size"):
        keep_gate[g.index] = passes_gate(mask, g["task"], size, ref_size).to_numpy()      # rule 1
    c = c[keep_gate]
    return c.assign(channel=np.where(c["task"].map(benchmark_family) == "bpb", "bpb", "benchmarks"))


def seed_block(df, fin, mask) -> pd.DataFrame:
    f = fin[fin["size"] == SEED_SIZE].assign(design=lambda d: d["family"].str.replace(r"-seed\d+$", "", regex=True))
    n = f.groupby("design")["seed"].nunique()
    designs = sorted(n[n >= 3].index)
    fams = [f"{d}-seed{GRID_SEED}" for d in designs]
    out = []
    for s in sorted(f["seed"].unique()):
        ref = f[f["design"].isin(designs) & (f["seed"] == s)].assign(family=lambda d: d["design"] + f"-seed{GRID_SEED}")
        out.append(read(df, fin, fams, SEED_SIZE, ref, mask).assign(reading=f"{SEED_SIZE} seed {s}"))
    to17 = read(df, fin, fams, TARGET_SIZE, fin, mask)
    out.append(to17[to17["size"].map(NON_EMB) < NON_EMB[SEED_SIZE]].assign(reading=f"{TARGET_SIZE} seed {GRID_SEED}"))
    print(f"1B seed block: {len(designs)} designs ({', '.join(designs)})")
    return pd.concat(out, ignore_index=True).assign(block=f"{SEED_SIZE} seeds")


def k30_block(df, fin, mask) -> pd.DataFrame:
    fams = sorted(set(fin.loc[(fin["size"] == REFERENCE) & (fin["kind"] != "loss"), "family"]))
    k30 = {f for f in fams if (fin.loc[fin["family"] == f, "L"] == K30).all()}
    involves = lambda pair: bool(k30 & set(pair))
    print(f"3B K={K30} block: {sorted(k30)} against {len(fams) - len(k30)} other 3B families")
    to17 = read(df, fin, fams, TARGET_SIZE, fin, mask, involves)
    return pd.concat([read(df, fin, fams, REFERENCE, fin, mask, involves).assign(reading=f"{REFERENCE}, K={K30} pairs"),
                      to17[to17["size"] != TARGET_SIZE].assign(reading=f"{TARGET_SIZE}, K={K30} pairs"),
                      read(df, fin, fams, REFERENCE, fin, mask).assign(reading=f"{REFERENCE}, all pairs")],
                     ignore_index=True).assign(block=f"{REFERENCE} K={K30}")


def common(cells: pd.DataFrame) -> pd.DataFrame:
    """Rule 13: per (block, channel, pair set, size, frac), the tasks every reading
    that scores that point has."""
    keys = ["block", "channel", "group", "size", "frac"]
    n_read = cells.groupby(keys)["reading"].transform("nunique")
    n_task = cells.groupby(keys + ["task"])["reading"].transform("nunique")
    return cells[n_task == n_read]


def pooled(cells: pd.DataFrame) -> pd.DataFrame:
    out = (cells.groupby(["block", "reading", "channel", "group", "size", "frac"], sort=False)
           .agg(n_matching=("n_matching", "sum"), n_pairs=("n_comparable", "sum"), n_tasks=("task", "nunique"))
           .reset_index().rename(columns={"group": "axes"}))
    out["da"] = out["n_matching"] / out["n_pairs"]
    return out.sort_values(["frac"], kind="stable").sort_values("size", key=lambda s: s.map(NON_EMB), kind="stable")


def write_tex(t: pd.DataFrame, path: Path) -> None:
    """DA-size of benchmark accuracy: a block per reference, a column per reading."""
    lines = ["\\begin{tabular}{llcccc}", "\\toprule"]
    for block, g in t.groupby("block", sort=False):
        readings = list(dict.fromkeys(g["reading"]))
        lines += [f"\\multicolumn{{2}}{{l}}{{\\textbf{{Reference {block}}}}} & "
                  + " & ".join(r.replace(" seed", ", seed").replace("K=", "$K$=") for r in readings)
                  + " & " * (4 - len(readings)) + " \\\\", "\\midrule"]
        for axes, h in g.groupby("axes", sort=False):
            for i, size in enumerate(size_order(h["size"].unique())):
                v = [h[(h["reading"] == r) & (h["size"] == size)]["da"] for r in readings]
                lines.append(" & ".join([("Single-axis" if axes == "mono-axis" else "All pairs") if i == 0 else "", size]
                                        + [f"{x.iloc[0]:.2f}" if len(x) else "--" for x in v]
                                        + ["--"] * (4 - len(readings))) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    path.with_suffix(".tex").write_text("\n".join(lines + ["\\end{tabular}", ""]))


def generate_readme(pool: str, t: pd.DataFrame, allp: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    blocks = []
    for (block, channel), g in allp[allp["frac"] == 1.0].groupby(["block", "channel"], sort=False):
        readings = list(dict.fromkeys(g["reading"]))
        rows = []
        for (axes, size), h in g.sort_values("axes", kind="stable").groupby(["axes", "size"], sort=False):
            h = h.set_index("reading")
            rows.append([axes, size] + [f"{h.at[r, 'da']:.2f} ({int(h.at[r, 'n_tasks'])})" if r in h.index else "—"
                                        for r in readings])
        blocks += [f"**{block}, {channel}** — DA-size (tasks)", md_table(["axes", "proxy"] + readings, rows)]
    goal = allp[(allp["frac"] < 1.0) & (allp["channel"] == "benchmarks")]
    goal_rows = [[b, r, a, f"{g['da'].min():.2f}–{g['da'].max():.2f}"]
                 for (b, r, a), g in goal.groupby(["block", "reading", "axes"], sort=False)]
    body = "\n\n".join([
        "## Other references: the 1B rung at three seeds, and the 3B model at K = 30",
        f"**DA-size and DA-goal · pool `{POOL}` read at the grid seed (the replicates enter as references only) · "
        f"multi-axis and mono-axis pairs · gate `{pool}` at the proxy and at the reference (Wilson rule on the 3B runs) · "
        f"no filter · ≥ {MIN_PAIRS} pairs.** The {SEED_SIZE} readings take a reference below {TARGET_SIZE} on purpose "
        f"(rule 9: the only rung above 600M with seed replicates). Within a block a cell counts only where every reading scores it, so the "
        f"readings differ in the reference alone (`all pairs` reads more pairs per task). Regenerate with "
        f"`python analysis/rq10_size_generalisation/reference_choice.py --pool {pool}`.",
        *blocks,
        "DA-goal (the nine tenths before the final, benchmark accuracy), range over the proxy sizes and checkpoints:",
        md_table(["block", "reading", "axes", "DA-goal"], goal_rows),
        f"Files: [`{STEM}_all_both_axes.csv`]({GITHUB}/rq10_size_generalisation/{rel}/{STEM}_all_both_axes.csv), "
        f"[`{STEM}_size_both_axes.csv`]({GITHUB}/rq10_size_generalisation/{rel}/{STEM}_size_both_axes.csv) / "
        f"[`.tex`]({GITHUB}/rq10_size_generalisation/{rel}/{STEM}_size_both_axes.tex) (the paper table)."])
    replace_block(OUT_ROOT / "README.md", "reference-choice", body, f"reference_choice.py --pool {pool}")


def main(pool: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"!!! RULE 9: the {SEED_SIZE} block reads a reference below {TARGET_SIZE}, on purpose (seed replicates)")
    df = ladder_frame(POOL, above_reference=True)
    df = df[on_shared_grid(df)]
    fin = finals(df)
    mask = gate_mask(df, pool, REFERENCE)
    cells = common(pd.concat([seed_block(df, fin, mask), k30_block(df, fin, mask)], ignore_index=True))
    allp = pooled(cells)
    allp.to_csv(out_dir / f"{STEM}_all_both_axes.csv", index=False)
    t = allp[(allp["frac"] == 1.0) & (allp["channel"] == "benchmarks")]
    t.to_csv(out_dir / f"{STEM}_size_both_axes.csv", index=False)
    write_tex(t, out_dir / f"{STEM}_size_both_axes")
    print(allp[allp["frac"] == 1.0].pivot_table(index=["block", "channel", "axes", "size"], columns="reading",
                                                 values="da").round(3).to_string())
    if out_dir == OUT_ROOT / load_pools()[pool].get("stage", "pretraining") / pool:
        generate_readme(pool, t, allp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="the gate's pool and the output folder")
    ap.add_argument("--out-dir", default=None, help="write elsewhere (a test run)")
    a = ap.parse_args()
    main(a.pool, Path(a.out_dir) if a.out_dir else OUT_ROOT / load_pools()[a.pool].get("stage", "pretraining") / a.pool)
