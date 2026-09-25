"""Decision accuracy on the public model lines: does a 1B model of one lab
rank the labs' recipes the way their 13B models do?

The ladder's reference stops at 1.7B. The external tier holds three public
model LINES with a base model both around 1B and around 13B — gemma-3
(1B → 12B), Qwen3-Base (1.7B → 14B) and OLMo-2 (1B → 13B) — evaluated on
the same tasks and gated with the same rule (`all/external/`). Treating a
line as a design variant (a lab's whole recipe: data, tokenizer, schedule,
tokens) and the size buckets as the ladder's rungs gives DA-size one rung
above anything the ladder can see: per task, the share of the three line
pairs the 1B–1.7B models order the way the 12–14B models do.

What it is and is not. Three lines are three pairs, the minimum rule 5
allows, so a task's DA sits on {0, ⅓, ⅔, 1} and only the pooled reading
carries information; the "decision" is between recipes that differ in
everything at once, which is the developer's situation when comparing
labs, not the ladder's controlled one-axis decisions; and the token
budgets differ by line and by size, so the rung is a model size, not a
size at fixed Chinchilla multiple. The comparison the figure draws is with
the ladder's own DA-size at the 1B proxy on the same tasks, which is the
number a reader would otherwise extrapolate.

    public_ladders.png / .csv   (a) per task, DA of the 1B–1.7B models against the 12–14B
                                models over the three line pairs, beside the ladder's
                                1B → 1.7B DA-size on the same task; (b) per line pair, the
                                share of gated tasks where the small-scale order holds at
                                12–14B; per (task) rows in the CSV
    python analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, GATE_AND_CURVES  # noqa: E402
from analysis.utils import MIN_PAIRS, benchmark_family, build_snr_pool  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
EXTERNAL_MASK = GATE_AND_CURVES / "all" / "external" / "above_random_mask.csv"
# line -> (proxy model, reference model): the base releases around 1B and around 13B
LINES = {"gemma-3": ("gemma-3-1b-pt", "gemma-3-12b-pt"),
         "Qwen3": ("Qwen3-1.7B-Base", "Qwen3-14B-Base"),
         "OLMo-2": ("OLMo-2-0425-1B", "OLMo-2-1124-13B")}
PROXY_BUCKETS, REF_BUCKET = ("1B", "1.7B"), "12-14B"
LADDER_PROXY = "1B"
mpl.rcParams.update(S.RC)


def finals(pool: str = "external") -> pd.DataFrame:
    df = build_snr_pool(pool)
    fin = df.loc[df.groupby(["model", "task"])["step"].idxmax()]
    return fin[fin["model"].isin({m for pair in LINES.values() for m in pair})]


def decisions(fin: pd.DataFrame) -> pd.DataFrame:
    """Per task and line pair: the sign of the proxy difference against the reference's."""
    mask = pd.read_csv(EXTERNAL_MASK).set_index("task")
    score = fin.pivot_table(index="task", columns="model", values="primary_score")
    rows = []
    for task, r in score.iterrows():
        if task not in mask.index or not all(mask.loc[task, b] == 1 for b in PROXY_BUCKETS + (REF_BUCKET,)):
            continue                                                       # rule 1 at the proxy and at the reference
        for (la, (pa, ra)), (lb, (pb, rb)) in combinations(LINES.items(), 2):
            if any(pd.isna(r.get(m)) for m in (pa, ra, pb, rb)):
                continue
            rows.append({"task": task, "family": benchmark_family(task), "pair": f"{la} vs {lb}",
                         "match": int(np.sign(r[pa] - r[pb]) == np.sign(r[ra] - r[rb])),
                         "ref_sign": int(np.sign(r[ra] - r[rb])), "proxy_sign": int(np.sign(r[pa] - r[pb]))})
    return pd.DataFrame(rows)


def per_pair(d: pd.DataFrame) -> pd.DataFrame:
    """Per line pair: DA, and the two baselines that expose a level difference.
    `majority` is the share of tasks on which the reference's order is the
    common one (a proxy that always says "line A" scores this without
    reading anything); `da_minority` is DA on the tasks where the reference
    order is the uncommon one — the only tasks on which the proxy can beat
    the majority rule. `informative` = DA − majority."""
    rows = []
    for pair, g in d.groupby("pair"):
        maj = g["ref_sign"].value_counts().idxmax()
        minority = g[g["ref_sign"] != maj]
        rows.append({"pair": pair, "tasks": len(g), "da": g["match"].mean(), "majority": (g["ref_sign"] == maj).mean(),
                     "minority_tasks": len(minority), "da_minority": minority["match"].mean() if len(minority) else np.nan,
                     "informative": g["match"].mean() - (g["ref_sign"] == maj).mean()})
    return pd.DataFrame(rows)


def per_task(d: pd.DataFrame, pool: str) -> pd.DataFrame:
    """DA per task over the line pairs (≥ MIN_PAIRS), beside the ladder's DA-size at the 1B proxy."""
    t = d.groupby(["task", "family"]).agg(da_public=("match", "mean"), pairs=("match", "size")).reset_index()
    t = t[t["pairs"] >= MIN_PAIRS]
    stage = load_pools()[pool].get("stage", "pretraining")
    da = pd.read_csv(OUT_ROOT / stage / pool / "da_per_task.csv")
    da = da[da["axes"] == "multi-axis"] if "axes" in da.columns else da
    return t.merge(da[["task", f"decision_acc_size_{LADDER_PROXY}"]].rename(columns={f"decision_acc_size_{LADDER_PROXY}": "da_ladder"}),
                   on="task", how="left")


def figure(t: pd.DataFrame, d: pd.DataFrame, path: Path, pool: str) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": (1.3, 1)})
    both = t.dropna(subset=["da_ladder"])
    rng = np.random.default_rng(0)
    a.scatter(both["da_ladder"], both["da_public"] + rng.uniform(-.02, .02, len(both)), s=14, color=S.RAMP[2], alpha=.7, lw=0)
    a.axhline(0.5, color=S.MUTED, lw=.8, ls=":"); a.axvline(0.5, color=S.MUTED, lw=.8, ls=":")
    a.set_xlabel(f"ladder: DA-size of the {LADDER_PROXY} proxy → 1.7B (multi-axis pairs)")
    a.set_ylabel("public lines: DA, 1B–1.7B → 12–14B (3 pairs)")
    a.set_xlim(0, 1.02); a.set_ylim(-0.05, 1.05); a.set_yticks([0, 1 / 3, 2 / 3, 1]); a.set_yticklabels(["0", "⅓", "⅔", "1"])
    a.text(0.03, 0.97, f"pooled DA, public lines: {t['da_public'].mean():.2f} ({len(t)} tasks)\n"
                       f"ladder on the same tasks: {both['da_ladder'].mean():.2f} ({len(both)} tasks)\n"
                       f"Spearman across tasks: {both['da_ladder'].corr(both['da_public'], method='spearman'):.2f}",
           transform=a.transAxes, va="top", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE, ec=S.GRID, lw=.6))
    a.set_title("(a) per task: the public lines' agreement against the ladder's", loc="left", fontsize=8.5)
    a.grid(color=S.GRID, lw=.6); S.clean(a)
    p = per_pair(d)
    y = np.arange(len(p))
    b.barh(y + .2, p["da"], .38, color=S.RAMP[2], label="DA: the 1B–1.7B order holds at 12–14B")
    b.barh(y - .2, p["majority"], .38, color=S.GRID, edgecolor=S.MUTED, lw=.5,
           label="majority baseline: always name the line that usually wins at 12–14B")
    for i, r in p.iterrows():
        b.text(r["da"] + .01, i + .2, f"{r['da']:.2f}", va="center", fontsize=7)
        b.text(r["majority"] + .01, i - .2, f"{r['majority']:.2f}  (minority tasks: {int(r['minority_tasks'])}, DA there "
                                            f"{r['da_minority']:.2f})" if r["minority_tasks"] else f"{r['majority']:.2f}",
               va="center", fontsize=6.5, color=S.MUTED)
    b.axvline(0.5, color=S.MUTED, lw=.8, ls=":")
    b.set_yticks(y); b.set_yticklabels(p["pair"], fontsize=8); b.set_xlim(0, 1.6)
    b.set_xlabel("share of the gated tasks")
    b.set_title("(b) per line pair, against the level-difference baseline", loc="left", fontsize=8.5)
    b.legend(fontsize=6.5, frameon=False, loc="lower right")
    b.grid(color=S.GRID, lw=.6, axis="x"); S.clean(b)
    top = G._header(fig, "Decision accuracy one rung above the ladder: the public model lines at 1B–1.7B against 12–14B",
                    f"Three public lines with a base model at both sizes — {', '.join(f'{k} ({a_} → {b_})' for k, (a_, b_) in LINES.items())} — "
                    f"on the tasks above chance at the 1B, 1.7B and 12–14B buckets of the external gate (`all/external/`). "
                    f"A decision is a line pair; DA is the share of pairs the small models order the way the large "
                    f"ones do. Three pairs put a task's DA on {{0, ⅓, ⅔, 1}}, so read the pooled numbers. The lines "
                    f"differ in data, tokenizer, schedule and tokens at once, so this is the between-lab decision, not "
                    f"the ladder's one-axis one; the rung is a model size, not a size at 5× Chinchilla. (a) against the "
                    f"ladder's own DA-size at {LADDER_PROXY} on the same tasks; (b) per pair, DA beside the majority baseline — the share "
                    f"a proxy scores by always naming the line that usually wins at 12–14B; DA above it is what the small "
                    f"models add, and the DA on the minority tasks is where they can add it. "
                    f"Gate: external mask; ladder DA from `{pool}`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame, d: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    p = per_pair(d)
    fam = t.groupby("family").agg(tasks=("task", "size"), da_public=("da_public", "mean"), da_ladder=("da_ladder", "mean")).reset_index()
    fam = fam[fam["tasks"] >= 3].sort_values("tasks", ascending=False)
    body = "\n\n".join([
        "## Decision accuracy one rung above the ladder: the public lines",
        f"gemma-3, Qwen3-Base and OLMo-2 each have a base model near 1B and near 13B in the external tier. Per task "
        f"above chance at both buckets, DA is the share of the three line pairs the small models order like the large "
        f"ones: pooled {t['da_public'].mean():.2f} over {len(t)} tasks, against the ladder's own {LADDER_PROXY} → 1.7B "
        f"DA-size of {t['da_ladder'].mean():.2f} on the same tasks. Per line pair: "
        + "; ".join(f"{r['pair']} {r['da']:.2f} against a majority baseline of {r['majority']:.2f} "
                    f"({int(r['minority_tasks'])} minority tasks, DA there {r['da_minority']:.2f})" for _, r in p.iterrows()) + ". "
        f"The majority baseline is what a proxy scores by always naming the line that usually wins at 12–14B, so only DA "
        f"above it is information the small models add. Between-lab decisions on public sizes, not the ladder's one-axis ones; three pairs, so per-task values are a "
        f"lattice. Regenerate with `python analysis/rq02_decision_accuracy/public_ladders.py --pool {pool}`.",
        md_table(["family", "tasks", "DA public lines (1B–1.7B → 12–14B)", f"DA ladder ({LADDER_PROXY} → 1.7B)"],
                 [[r["family"], int(r["tasks"]), f"{r['da_public']:.2f}", f"{r['da_ladder']:.2f}" if r["da_ladder"] == r["da_ladder"] else "—"]
                  for _, r in fam.iterrows()]),
        f"![Public ladders]({stage}/{pool}/public_ladders.png)"])
    replace_block(OUT_ROOT / "README.md", "public-ladders", body, f"public_ladders.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the ladder pool whose DA-size is drawn beside the public lines'")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    d = decisions(finals())
    t = per_task(d, args.pool)
    t.to_csv(out_dir / "public_ladders.csv", index=False)
    print(f"{len(t)} gated tasks; pooled DA public lines {t['da_public'].mean():.3f}; ladder {LADDER_PROXY} on the same "
          f"tasks {t['da_ladder'].mean():.3f} ({t['da_ladder'].notna().sum()} tasks)")
    pp = per_pair(d); pp.to_csv(out_dir / "public_ladders_pairs.csv", index=False); print(pp.round(3).to_string())
    print(t.groupby("family").agg(tasks=("task", "size"), da_public=("da_public", "mean"), da_ladder=("da_ladder", "mean")).round(2).to_string())
    figure(t, d, out_dir / "public_ladders.png", args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, t, d)
