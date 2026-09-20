"""rq05: the grid's transformations side by side — language count, sampling
temperature, model depth and language lists — read as decisions a proxy size
makes for the reference, on ONE item set, so their predictability can be
compared. analyze.py reads each intervention on its own items; pooled over
different task mixes those numbers are not comparable (a pair whose 1.7B has
no reformulated evals yet pools only the predictable families).

    transformation_da.csv   per transformation, pair (the two cells' L / level), proxy size and population:
                            DA on the pair's own items the reference decides (`decision_acc_own`, `n_own`)
                            and on the items every transformation decides somewhere (`decision_acc_shared`,
                            `n_shared`); benchmarks are gated by rq00's above-random mask at the proxy and
                            at the reference; `mean_abs_delta_ref` is the reference's effect on the own items
    transformation_da.png   mean over a transformation's pairs of the shared-item DA vs proxy size;
                            solid benchmarks, dashed per-language BPB (all 100 validation languages)

    python analysis/rq05_design_decisions/transformations.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq05_design_decisions.analyze import CANONICAL, COLOUR, MIN_ITEMS, OUT_ROOT  # noqa: E402
from analysis.utils import GRID_SEED, finals, ladder_frame, size_order  # noqa: E402
from pretrain.launch_trainings import DATA_SCHEMES  # noqa: E402

mpl.rcParams.update(S.RC)
# key -> (label, pairs); a pair is two cells as (L, scheme, arch), first = baseline
_A = sorted(DATA_SCHEMES["A"]["langs"])
TRANSFORMATIONS = {
    "langs":       ("language count (L vs next L)",
                    [((a, "A", "deep"), (b, "A", "deep")) for a, b in zip(_A, _A[1:])]),
    "temperature": ("temperature (T=1 vs T=3)",
                    [((L, "A", "deep"), (L, "AT3", "deep")) for L in sorted(DATA_SCHEMES["AT3"]["langs"])]),
    "arch":        ("depth (deep vs shallow)",
                    [((L, "A", "deep"), (L, "A", "shallow")) for L in _A]),
    "scheme":      ("language lists (A vs B)",
                    [((L, "A", "deep"), (L, "B", "deep")) for L in sorted(DATA_SCHEMES["B"]["langs"])]),
}
COLOUR = {**COLOUR, "langs": S.RAMP[2]}
# BPB is read on ALL validation languages, not on the languages both cells
# train: consecutive language-count pairs share only the smaller setting's
# list (L1 vs L2 shares English alone), which leaves too few items to
# compare transformations on. Every cell is validated on the same 100
# languages, so bpb_all is the one item set they all share.
POPULATIONS = ("benchmark", "bpb_all")
# rq00 computes the above-random gate on the grid-seed pool; the all-seeds
# pool has no mask of its own, so fall back to that one rather than leave the
# benchmarks ungated (a task at chance in both cells decides nothing).
GATE_POOL = "predictivity"


def _cell(size: str, c: tuple) -> str:
    L, scheme, arch = c
    return f"lm-{size}-L{L}{DATA_SCHEMES[scheme]['label']}-{arch}-seed{GRID_SEED}"


def _items(fin: dict, kind: dict, size: str, x: tuple, y: tuple, pop: str) -> pd.Series | None:
    """first − second score per item, for one pair at one size, or None when a cell is missing."""
    a, b = fin.get(_cell(size, x)), fin.get(_cell(size, y))
    if a is None or b is None:
        return None
    d = (a - b).dropna()
    if pop == "benchmark":
        return d[d.index.map(lambda t: kind.get(t) == "benchmark")]
    return d[d.index.map(lambda t: kind.get(t) == "bpb" and t != "bpb_macro")]


def transformation_da(df: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    grid = finals(df[df["seed"] == GRID_SEED])
    kind = dict(zip(grid["task"], grid["kind"]))
    fin = {m: g.set_index("task")["primary_score"] for m, g in grid.groupby("model")}
    gate = mask   # already indexed by task, one Int64 0/1 column per size
    # pass 1: every pair's decided items at its reference and at each proxy
    rows = []
    for key, (label, pairs) in TRANSFORMATIONS.items():
        for x, y in pairs:
            sizes = [s for s in size_order(grid["size"].unique()) if _cell(s, x) in fin and _cell(s, y) in fin]
            if len(sizes) < 2:
                continue
            ref = sizes[-1]
            for pop in POPULATIONS:
                d_ref = _items(fin, kind, ref, x, y, pop)
                d_ref = d_ref[d_ref != 0]
                for s in sizes[:-1]:
                    d = _items(fin, kind, s, x, y, pop)
                    items = d.index.intersection(d_ref.index)
                    if pop == "benchmark" and gate is not None and {s, ref} <= set(gate.columns):
                        items = items.intersection(gate.index[(gate[s] == 1) & (gate[ref] == 1)])
                    rows.append({"transformation": key, "label": label, "population": pop,
                                 "pair": f"L{x[0]}{DATA_SCHEMES[x[1]]['label']}-{x[2]} vs L{y[0]}{DATA_SCHEMES[y[1]]['label']}-{y[2]}",
                                 "proxy_size": s, "reference_size": ref,
                                 "_agree": (np.sign(d[items]) == np.sign(d_ref[items])),
                                 "mean_abs_delta_ref": float(d_ref[items].abs().mean()) if len(items) else np.nan})
    t = pd.DataFrame(rows)
    # pass 2: the shared set per (population, proxy) = the items every
    # transformation that HAS data there decides in some pair. Intersecting
    # over all four instead would empty the set whenever one of them has no
    # data at all (the AT3 cells have no BPB scored, which would blank the
    # BPB read for every transformation); `n_transformations` records how
    # many went into the intersection so a thin one is visible.
    out = []
    for (pop, s), g in t.groupby(["population", "proxy_size"]):
        per_tr = {k: set().union(*[set(a.index) for a in gg["_agree"]]) for k, gg in g.groupby("transformation")}
        per_tr = {k: v for k, v in per_tr.items() if v}
        shared = set.intersection(*per_tr.values()) if per_tr else set()
        for _, r in g.iterrows():
            a = r["_agree"]; sh = a[a.index.isin(shared)]
            out.append({**{k: v for k, v in r.items() if k != "_agree"},
                        "n_own": int(len(a)), "decision_acc_own": float(a.mean()) if len(a) >= MIN_ITEMS else np.nan,
                        "n_shared": int(len(sh)), "decision_acc_shared": float(sh.mean()) if len(sh) >= MIN_ITEMS else np.nan,
                        "n_transformations": len(per_tr)})
    cols = ["transformation", "label", "population", "pair", "proxy_size", "reference_size",
            "n_own", "decision_acc_own", "n_shared", "decision_acc_shared", "n_transformations",
            "mean_abs_delta_ref"]
    return pd.DataFrame(out)[cols].sort_values(["population", "transformation", "pair", "proxy_size"])


def summary(da: pd.DataFrame) -> pd.DataFrame:
    """Mean over a transformation's pairs, per population and proxy size."""
    return (da.groupby(["transformation", "label", "population", "proxy_size"])
            .agg(decision_acc_shared=("decision_acc_shared", "mean"), decision_acc_own=("decision_acc_own", "mean"),
                 pairs=("pair", "nunique"), pairs_with_data=("decision_acc_shared", "count"),
                 n_shared=("n_shared", "max")).reset_index())


def plot(sm: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    order = size_order(sm["proxy_size"].unique())
    for key, (label, _) in TRANSFORMATIONS.items():
        for pop, ls, mk in (("benchmark", "-", "s"), ("bpb_all", "--", "o")):
            g = sm[(sm["transformation"] == key) & (sm["population"] == pop)].set_index("proxy_size").reindex(order)
            if g["decision_acc_shared"].notna().any():
                ax.plot(range(len(order)), g["decision_acc_shared"], ls=ls, marker=mk, ms=4.5, lw=1.6, color=COLOUR[key])
    ax.axhline(.5, color=S.MUTED, ls=":", lw=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_ylim(0, 1.05)
    ax.set_xlabel("proxy size"); ax.set_ylabel("agreement with the reference (mean over pairs)")
    ax.set_title("transformations on one item set", loc="left")
    h = [Line2D([], [], color=COLOUR[k], lw=1.6, label=v[0]) for k, v in TRANSFORMATIONS.items()]
    h += [Line2D([], [], color=S.INK, ls="-", marker="s", ms=4, label="benchmark tasks (gated)"),
          Line2D([], [], color=S.INK, ls="--", marker="o", ms=4, label="per-language BPB")]
    ax.legend(handles=h, frameon=False, loc="lower right", fontsize=6.8)
    ax.grid(color=S.GRID, lw=.6); ax.set_axisbelow(True); S.clean(ax)
    S.save_figure(fig, out_dir, "transformation_da")


def generate_readme(pool: str, out_dir: Path, sm: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    rel = out_dir.relative_to(OUT_ROOT)
    blocks = ["## Transformations on one item set",
              f"Mean decision accuracy over each transformation's pairs, on the items every transformation "
              f"decides somewhere (benchmarks gated by rq00's above-random mask at the proxy and the reference); "
              f"`transformation_da.csv` has every pair. Regenerate with "
              f"`python analysis/rq05_design_decisions/transformations.py --pool {pool}`."]
    for pop, title in (("benchmark", "benchmarks"), ("bpb_all", "per-language BPB (100 validation languages)")):
        g = sm[sm["population"] == pop]
        if g.empty:
            continue
        grid = g.pivot_table(index="label", columns="proxy_size", values="decision_acc_shared")
        cols = size_order(grid.columns)
        pairs = g.groupby("label")["pairs"].max()
        withdata = g.groupby("label")["pairs_with_data"].max()
        items = g.groupby("label")["n_shared"].max()
        blocks += [f"**{title}** (rows: transformation; columns: proxy size; mean over pairs, shared items):",
                   md_table(["transformation", "pairs (with data)", "items"] + cols,
                            [[lab, f"{int(pairs[lab])} ({int(withdata[lab])})", int(items[lab])]
                             + [fmt(grid.loc[lab, c]) for c in cols] for lab in grid.index])]
    blocks.append(f"![Transformations]({rel}/transformation_da.png)")
    replace_block(OUT_ROOT / "README.md", "transformations", "\n\n".join(blocks), f"transformations.py --pool {pool}")


def main(pool: str) -> None:
    out_dir = OUT_ROOT / "pretraining" / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = load_mask(pool)
    if mask is None:
        mask = load_mask(GATE_POOL)
    da = transformation_da(ladder_frame(pool), mask)
    da.to_csv(out_dir / "transformation_da.csv", index=False)
    sm = summary(da)
    print(sm[sm["population"] == "benchmark"].pivot_table(index="label", columns="proxy_size", values="decision_acc_shared").round(2).to_string())
    plot(sm, out_dir)
    generate_readme(pool, out_dir, sm)
    print(f"Wrote → {out_dir / 'transformation_da.csv'} ({len(da)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
