"""How early, and how small, can we make the right decision? (paper RQ2)

Reads the intervention decision table `analyze.py` writes next to it (one
row per intervention, L, population, proxy size, fraction of the proxy's run)
and answers the paper's question for the two planned decisions, depth and
language lists: at which proxy size, and how early in that proxy's run, does
the decision match the reference's final one? The heat map is the mean over
language settings of the per-setting agreement, so a setting with thousands
of benchmark tasks does not outweigh one with hundreds.

    rq2_decisions.csv     the decision-table rows of the two planned decisions
    rq2_early_small.csv   agreement per (decision, population, proxy size, fraction)
    rq2_early_small.png/.pdf
    early_decision_facts.json

    python analysis/rq05_design_decisions/early_decision.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DESIGN_DECISIONS  # noqa: E402
from analysis.utils import size_order  # noqa: E402

OUT_ROOT = DESIGN_DECISIONS
CANONICAL = "predictivity_all"
DECISIONS = ["arch", "scheme"]                       # the two planned axes (analyze.py keys)
POPULATIONS = [("bpb_trained", "per-language bits per byte"), ("benchmark", "benchmark tasks")]
mpl.rcParams.update(S.RC)


def early_small(dt: pd.DataFrame) -> pd.DataFrame:
    core = dt[dt["intervention"].isin(DECISIONS)]
    return (core.groupby(["intervention", "label", "population", "proxy_size", "frac"])
            .agg(da=("decision_acc", "mean"), cells=("decision_acc", "size"),
                 items=("n_items", "sum"), refs=("reference_size", lambda s: "/".join(sorted(set(s)))))
            .reset_index())


def plot(agg: pd.DataFrame, out_dir: Path) -> None:
    fracs = sorted(agg["frac"].unique())
    fig, axes = plt.subplots(len(DECISIONS), len(POPULATIONS), figsize=(9.2, 6.4), sharex=True, squeeze=False)
    im = None
    for i, dec in enumerate(DECISIONS):
        for j, (pop, title) in enumerate(POPULATIONS):
            ax = axes[i][j]
            g = agg[(agg["intervention"] == dec) & (agg["population"] == pop)]
            sizes = size_order(g["proxy_size"])
            mat = np.full((len(sizes), len(fracs)), np.nan)
            cnt = np.zeros_like(mat)
            for _, r in g.iterrows():
                mat[sizes.index(r["proxy_size"]), fracs.index(r["frac"])] = r["da"]
                cnt[sizes.index(r["proxy_size"]), fracs.index(r["frac"])] = r["cells"]
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=S.SEQ, aspect="auto")
            for a in range(len(sizes)):
                for b in range(len(fracs)):
                    if np.isfinite(mat[a, b]):
                        ax.text(b, a, f"{mat[a, b]:.2f}\n({int(cnt[a, b])})", ha="center", va="center",
                                fontsize=6.8, color="white" if mat[a, b] > 0.7 else S.INK)
            ax.set_xticks(range(len(fracs))); ax.set_xticklabels([G.chinchilla(f) for f in fracs])
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
            if i == 0:
                ax.set_title(title, loc="left")
            if j == 0:
                label = g["label"].iloc[0] if not g.empty else dec
                ax.set_ylabel(f"{label}\nproxy size")
            if i == len(DECISIONS) - 1:
                ax.set_xlabel("proxy's training tokens (C = Chinchilla-optimal; 5C = the full run)")
            S.clean(ax, spines=()); ax.tick_params(length=0)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02)
        cb.set_label("agreement with the reference's final decision (mean over L; settings in brackets)")
        cb.outline.set_visible(False)
    S.save_figure(fig, out_dir, "rq2_early_small")


def generate_readme(pool: str, out_dir: Path, agg: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    bullets, blocks = [], []
    for dec in DECISIONS:
        for pop, title in POPULATIONS:
            g = agg[(agg["intervention"] == dec) & (agg["population"] == pop)]
            if g.empty:
                continue
            label = g["label"].iloc[0]
            final = g[g["frac"] == 1.0].set_index("proxy_size")["da"]
            sizes = size_order(final.index)
            first = next((s for s in sizes if final[s] >= 0.75), None)
            early = g[(g["proxy_size"] == (first or sizes[-1]))].sort_values("frac")
            e_first = next((G.chinchilla(r.frac) for r in early.itertuples() if r.da >= 0.75), "never")
            bullets.append(
                f"- **{label}, {title}** — final-checkpoint agreement by proxy: "
                + ", ".join(f"{s} {fmt(final[s])}" for s in sizes)
                + (f"; smallest proxy at ≥ 0.75: **{first}**, which reaches it at {e_first} of training (5C = the full run)."
                   if first else "; no proxy reaches 0.75."))
            piv = g.pivot_table(index="proxy_size", columns="frac", values="da")
            piv = piv.reindex(size_order(piv.index))
            blocks += [f"**{label} — {title}** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; "
                       "mean over L of the per-L agreement):",
                       md_table(["proxy"] + [G.chinchilla(f) for f in piv.columns],
                                [[s] + [fmt(piv.loc[s, f]) for f in piv.columns] for s in piv.index])]
    blocks.append(f"![Early and small]({stage}/{pool}/rq2_early_small.png)")
    readme = OUT_ROOT / "README.md"
    body = "\n\n".join([
        "## How small, and how early (paper RQ2)",
        f"Numbers from the `{pool}` decision table above. Regenerate with "
        f"`python analysis/rq05_design_decisions/early_decision.py --pool {pool}`.",
        "\n".join(bullets)] + blocks)
    replace_block(readme, "early-decision", body, f"early_decision.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    src = DESIGN_DECISIONS / stage / pool / "intervention_da.csv"
    if not src.is_file():
        sys.exit(f"missing {src} — run analysis/rq05_design_decisions/analyze.py --pool {pool} first")
    dt = pd.read_csv(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    core = dt[dt["intervention"].isin(DECISIONS)]
    core.to_csv(out_dir / "rq2_decisions.csv", index=False)
    agg = early_small(dt)
    agg.to_csv(out_dir / "rq2_early_small.csv", index=False)
    print(f"Wrote → {out_dir / 'rq2_early_small.csv'} ({len(agg)} cells from {len(core)} rows)")
    if not agg.empty:
        plot(agg, out_dir)
    refs = {f"{d}|{p}|L{int(L)}": r for (d, p, L), r in
            core[core["frac"] == 1.0].groupby(["intervention", "population", "L"])["reference_size"].first().items()}
    (out_dir / "early_decision_facts.json").write_text(json.dumps(
        {"rq2": {"table": agg.round(3).to_dict("records"), "references": refs}}, indent=1, default=str))
    generate_readme(pool, out_dir, agg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool whose decision table to read (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
