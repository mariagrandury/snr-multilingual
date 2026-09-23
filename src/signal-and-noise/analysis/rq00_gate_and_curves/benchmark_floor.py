"""Is a gated benchmark a size problem or a benchmark problem? The floor of
each task on the ladder against its floor on the public models.

The above-random gate removes half of the ladder's benchmark-language tasks
at 1.7B. Two readings are possible: the task is unreadable below some
capability the ladder never reaches (a SIZE floor — a larger or better-
trained model would read it), or the task is at chance for any model of the
sizes a developer decides at (a BENCHMARK floor). The external tier —
public base models from 270M to 70B, evaluated on the same tasks and gated
with the same rule (`above_random.py --only external`, mask in
`all/external/`) — separates the two: for every task both tiers score, the
smallest ladder size from which the gate holds (`grids.smallest_safe`, as in
`first_size_above_random`) is set beside the smallest external bucket.

Reading: a task gated on the whole ladder but readable at a 600M public
model is a training-recipe/size floor the ladder's 5×-Chinchilla runs do not
clear (public 600M models train on 10–36 T tokens); a task readable only at
27B or never is a benchmark or language-resource floor. Caveats the figure
states: the overlap is the 86 tasks of the old task list (no `rf_` twins, no
`auto` probe families); some external buckets hold one or two models, so the
gate's "half of the runs" is one run; external token counts are not
comparable, so the floor is a MODEL floor, not a parameters-at-5C floor.

    benchmark_floor.csv   one row per shared task: ladder floor, external floor, family, language
    benchmark_floor.png   (a) ladder floor × external floor, one cell per task count;
                          (b) per family, the tasks gated on the ladder split by where
                          the public models first read them
    python analysis/rq00_gate_and_curves/benchmark_floor.py --pool predictivity
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402

EXTERNAL = GATE_AND_CURVES / "all" / "external" / "above_random_mask.csv"
META = ["task", "family", "language", "n_options", "random_baseline", "options_exact", "n_items"]
NEVER = "never"
# where the public models first read a task, in the bins the paper can use
BINS = [("≤ 600M", {"270M", "600M"}), ("1B–1.7B", {"1B", "1.7B"}), ("3B–4B", {"3B", "4B"}),
        ("7B–14B", {"7-9B", "12-14B"}), ("≥ 27B", {"27-32B", "70B"}), (NEVER, {NEVER})]
mpl.rcParams.update(S.RC)


def floor_of(mask: pd.DataFrame) -> pd.Series:
    """task -> the smallest level from which the gate holds at every larger
    informed level, `never` when the largest informed level fails, NaN when
    no level has information."""
    levels = [c for c in mask.columns if c not in META]
    m = mask.set_index("task")[levels]
    idx = G.smallest_safe(m.astype("float"))
    return idx.map(lambda i: np.nan if i != i else (NEVER if i < 0 else levels[int(i)]))


def floors(pool: str) -> pd.DataFrame:
    stage = load_pools()[pool].get("stage", "pretraining")
    ladder = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_mask.csv")
    ext = pd.read_csv(EXTERNAL)
    t = ladder[["task", "family", "language", "n_options"]].merge(ext[["task"]], on="task")
    t["ladder_floor"] = t["task"].map(floor_of(ladder))
    t["external_floor"] = t["task"].map(floor_of(ext))
    t["external_bin"] = t["external_floor"].map({lvl: name for name, lv in BINS for lvl in lv})
    t["ladder_gated_at_ref"] = t["ladder_floor"].eq(NEVER)
    return t.dropna(subset=["ladder_floor", "external_floor"])


def external_models() -> pd.DataFrame:
    """The public models behind the external buckets: one row per model with
    its family (the model line), its parameter count and whether it is a base
    or a post-trained release — so the reader knows what "1B–1.7B" holds."""
    from analysis.utils import build_snr_pool
    df = build_snr_pool("external")
    m = df.groupby("model").agg(size=("size", "first"), family=("family", "first")).reset_index()

    def params(size: str) -> float:
        head = size.split("-")[0]                              # 30B-A3B counts its total
        return float(head[:-1]) * (1e9 if head.endswith("B") else 1e6)
    m["params"] = m["size"].map(params)
    m["line"] = m["family"].str.replace(r"-(it|pt|base|Base|Instruct.*|Think.*|SFT|DPO|checkpoints|2509)$", "", regex=True)
    m["post_trained"] = ~m["family"].str.contains(r"-pt$|-base$|-Base$|Base|checkpoints|a06|from8b|-2509$", regex=True) \
        | m["family"].str.contains("Instruct|Think|-it$|-SFT|-DPO", regex=True)
    return m.sort_values(["line", "params"])


def figure(t: pd.DataFrame, path: Path, pool: str, ladder_levels: list, ext_levels: list) -> None:
    fig, (m_ax, a, b) = plt.subplots(1, 3, figsize=(15.6, 4.9), gridspec_kw={"width_ratios": (0.95, 0.9, 1.05)})
    # (0) the public models behind the buckets
    m = external_models()
    lines = list(dict.fromkeys(m["line"]))
    for i, line in enumerate(lines):
        g = m[m["line"] == line]
        base, post = g[~g["post_trained"]], g[g["post_trained"]]
        m_ax.scatter(base["params"], [i] * len(base), s=26, color=S.RAMP[2], zorder=3, label="base / pretrained" if i == 0 else None)
        m_ax.scatter(post["params"], [i] * len(post), s=26, facecolor=S.SURFACE, edgecolor=S.RAMP[2], lw=.9, zorder=2,
                     label="post-trained (SFT / DPO / instruct / think)" if i == 0 else None)
    m_ax.set_yticks(range(len(lines))); m_ax.set_yticklabels(lines, fontsize=7)
    m_ax.set_xscale("log"); m_ax.set_xlabel("parameters (log)")
    for lvl in ext_levels:                                   # "7-9B": the bucket's lower edge
        num = re.findall(r"[\d.]+", lvl)[0]
        m_ax.axvline(float(num) * (1e9 if lvl.endswith("B") else 1e6), color=S.GRID, lw=.6, zorder=1)
    m_ax.set_title(f"(0) the {len(m)} public models behind the buckets", loc="left", fontsize=8.5)
    m_ax.legend(fontsize=6, frameon=False, loc="upper left"); S.clean(m_ax)
    rows, cols = ladder_levels + [NEVER], ext_levels + [NEVER]
    ct = pd.crosstab(t["ladder_floor"], t["external_floor"]).reindex(index=rows, columns=cols, fill_value=0)
    im = a.imshow(ct.to_numpy(), cmap="Blues", aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = ct.iloc[i, j]
            if v:
                a.text(j, i, str(v), ha="center", va="center", fontsize=7, color=S.SURFACE if v > ct.values.max() / 2 else S.INK)
    a.set_xticks(range(len(cols))); a.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    a.set_yticks(range(len(rows))); a.set_yticklabels(rows, fontsize=7)
    a.set_xlabel("smallest public-model bucket that reads the task"); a.set_ylabel("smallest ladder size that reads the task")
    a.set_title(f"(a) {len(t)} tasks scored by both tiers", loc="left", fontsize=8.5)
    a.tick_params(axis="x", labelsize=6.5)
    # (b) the tasks gated on the ladder, per family, by where the public models read them
    g = t[t["ladder_gated_at_ref"]]
    order = [n for n, _ in BINS]
    tab = pd.crosstab(g["family"], g["external_bin"]).reindex(columns=order, fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    left = np.zeros(len(tab))
    colours = S.RAMP + [S.SERIES[1], S.MUTED]
    for k, (name, _) in enumerate(BINS):
        b.barh(range(len(tab))[::-1], tab[name], left=left, color=colours[k], label=name)
        left += tab[name].to_numpy()
    b.set_yticks(range(len(tab))[::-1]); b.set_yticklabels(tab.index, fontsize=7)
    b.set_xlabel("tasks gated at every ladder size"); b.legend(fontsize=6.5, frameon=False, title="public models first read it at", title_fontsize=6.5)
    b.set_title(f"(b) the {len(g)} tasks the ladder never reads: size floor or benchmark floor?", loc="left", fontsize=8.5)
    b.grid(color=S.GRID, lw=.6, axis="x"); S.clean(b)
    n_size = int(g["external_bin"].isin(["≤ 600M", "1B–1.7B"]).sum())
    top = G._header(fig, "Benchmark floors: what the ladder cannot read, the public models can — mostly",
                    f"Every task scored by both the ladder (`{pool}`, gate = one-sided 95 % Wilson lower bound over "
                    f"chance for at least half of the size's runs, final checkpoint) and the external tier (public base "
                    f"models 270M–70B, same task, same gate, `all/external/`; panel (0) lists them, grid lines at the bucket sizes). A floor "
                    f"is the smallest level from which "
                    f"the gate holds at every larger informed level. (a) the joint distribution; (b) the tasks the "
                    f"ladder never reads, by the public bucket that first reads them: {n_size} of {len(g)} are readable "
                    f"at ≤ 1.7B by a public model — a size/recipe floor the 5×-Chinchilla ladder does not clear — and the "
                    f"rest need ≥ 3B or are never read, a benchmark or language floor. Overlap is the 86-task list of "
                    f"the 36-model sweep (no cloze twins, no probe families); some external buckets hold one or two "
                    f"models; external token budgets are not the ladder's, so this is a model floor, not a size-at-5C floor.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    g = t[t["ladder_gated_at_ref"]]
    counts = g["external_bin"].value_counts().reindex([n for n, _ in BINS], fill_value=0)
    fam = pd.crosstab(g["family"], g["external_bin"]).reindex(columns=[n for n, _ in BINS], fill_value=0)
    fam = fam.loc[fam.sum(axis=1).sort_values(ascending=False).index].reset_index()
    body = "\n\n".join([
        "## Benchmark floors: the ladder's gate against the public models'",
        f"Of the {len(t)} tasks both tiers score, {len(g)} are at chance at every ladder size. Where the public base "
        f"models (270M–70B, same gate) first read them: " + ", ".join(f"{n} {c}" for n, c in counts.items()) + ". "
        f"A task readable at ≤ 1.7B by a public model is a size/recipe floor (the 5×-Chinchilla ladder does not reach "
        f"it; public models of that size train on 10–36 T tokens); a task that needs ≥ 3B or is never read is a "
        f"benchmark or language-resource floor. Overlap is the 36-sweep's 86-task list only. Regenerate with "
        f"`python analysis/rq00_gate_and_curves/benchmark_floor.py --pool {pool}`.",
        md_table(list(fam.columns), fam.values.tolist()),
        f"![Benchmark floors]({stage}/{pool}/benchmark_floor.png)"])
    replace_block(GATE_AND_CURVES / "README.md", "benchmark-floor", body, f"benchmark_floor.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = GATE_AND_CURVES / stage / args.pool
    t = floors(args.pool)
    t.to_csv(out_dir / "benchmark_floor.csv", index=False)
    ladder_levels = [c for c in pd.read_csv(out_dir / "above_random_mask.csv", nrows=0).columns if c not in META]
    ext_levels = [c for c in pd.read_csv(EXTERNAL, nrows=0).columns if c not in META]
    print(pd.crosstab(t["ladder_floor"], t["external_bin"]).to_string())
    figure(t, out_dir / "benchmark_floor.png", args.pool, ladder_levels, ext_levels)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, t)
