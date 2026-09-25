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

"Never reads" is the gate's verdict, not a run's: a task the ladder never
reads is one where at EVERY ladder size fewer than half of that size's runs
that train the task's language clear the one-sided 95 % Wilson bound over
chance (`above_random.MIN_SHARE`). Single runs may clear it — `mmlu` is never
read although 8 of 18 runs do at 1.7B (2026-09-23 snapshot), the L1
English-only cells among them; panel (d) carries the count per task.

    above_random_external.csv   one row per shared task: ladder floor, external floor, family,
                          language; for the ladder-gated tasks the best single run at the
                          reference (margin over chance, its LCB margin) and the best public
                          base model ≤ 1.7B on the same task (panel d)
    above_random_external_lines.csv   one row per public base model of the six lines and per
                          ladder size: the share of the shared tasks above the gate (panel c)
    above_random_external.png   (0) the public models; (a) ladder floor × external floor, one cell
                          per task count; (b) per family, the tasks gated on the ladder split by
                          where the public models first read them; (c) share of the shared tasks
                          a model reads against its parameters, public lines vs the ladder;
                          (d) the ladder-gated tasks: best ladder run at the reference vs the
                          best public model ≤ 1.7B, a near-miss view
    python analysis/rq00_gate_and_curves/above_random_external.py --pool predictivity
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
from analysis.rq00_gate_and_curves.above_random import MIN_SHARE, scores_and_mask  # noqa: E402
from analysis.utils import TARGET_SIZE  # noqa: E402

EXTERNAL = GATE_AND_CURVES / "all" / "external" / "above_random_mask.csv"
LINES = ["gemma-3", "Qwen3", "OLMo-2", "Olmo-3", "Apertus", "apertus3-a06"]   # the public base lines of (c) and (d)
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
    t = ladder[["task", "family", "language", "n_options", "random_baseline"]].merge(ext[["task"]], on="task")
    t["ladder_floor"] = t["task"].map(floor_of(ladder))
    t["external_floor"] = t["task"].map(floor_of(ext))
    t["external_bin"] = t["external_floor"].map({lvl: name for name, lv in BINS for lvl in lv})
    t["ladder_gated_at_ref"] = t["ladder_floor"].eq(NEVER)
    return t.dropna(subset=["ladder_floor", "external_floor"])


def params(size: str) -> float:
    head = size.split("-")[0]                              # 30B-A3B counts its total
    return float(head[:-1]) * (1e9 if head.endswith("B") else 1e6)


def external_models(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The public models behind the external buckets: one row per model with
    its family (the model line), its parameter count and whether it is a base
    or a post-trained release — so the reader knows what "1B–1.7B" holds."""
    from analysis.utils import build_snr_pool
    df = build_snr_pool("external") if df is None else df
    m = df.groupby("model").agg(size=("size", "first"), family=("family", "first")).reset_index()
    m["params"] = m["size"].map(params)
    m["line"] = m["family"].str.replace(r"-(it|pt|base|Base|Instruct.*|Think.*|SFT|DPO|checkpoints|2509)$", "", regex=True)
    m["post_trained"] = ~m["family"].str.contains(r"-pt$|-base$|-Base$|Base|checkpoints|a06|from8b|-2509$", regex=True) \
        | m["family"].str.contains("Instruct|Think|-it$|-SFT|-DPO", regex=True)
    return m.sort_values(["line", "params"])


def external_lines(t: pd.DataFrame, pool: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(models, lines, near): the public models; per public base model of `LINES` and
    per ladder size the share of the shared tasks above the gate (a public model's
    own run clears the Wilson bound; a ladder size clears the mask, half of its
    runs); and, for the ladder-gated tasks, the best single ladder run at the
    reference against the best public base model ≤ 1.7B, both as margins over chance."""
    from analysis.utils import build_snr_pool
    stage = load_pools()[pool].get("stage", "pretraining")
    df = build_snr_pool("external", untrained=True)
    m = external_models(df)
    runs = scores_and_mask(df, runs=True)[3]
    base = m[~m["post_trained"] & m["line"].isin(LINES)]
    runs = runs[runs["task"].isin(t["task"]) & runs["model"].isin(base["model"])].merge(base[["model", "line", "params"]], on="model")
    lines = runs.groupby(["line", "model", "params"])["above"].agg(share="mean", n_tasks="count").reset_index()
    ladder = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_mask.csv").set_index("task").reindex(t["task"])
    levels = [c for c in ladder.columns if c not in META]
    lines = pd.concat([lines, pd.DataFrame({"line": "ladder", "model": levels, "params": [params(s) for s in levels],
                                            "share": [ladder[s].eq(1).mean() for s in levels],
                                            "n_tasks": [int(ladder[s].notna().sum()) for s in levels]})], ignore_index=True)
    # (d): the ladder's best single run at the reference, and the best public base model ≤ 1.7B
    lr = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_runs.csv")
    lr = lr[(lr["bucket"] == TARGET_SIZE) & lr["trained"] & lr["task"].isin(t.loc[t["ladder_gated_at_ref"], "task"])]
    chance = t.set_index("task")["random_baseline"]      # the gate's own chance level (task_chance)
    best = lr.loc[lr.groupby("task")["score"].idxmax()].set_index("task")
    near = pd.DataFrame({"ladder_best_run": best["model"], "ladder_best_margin": best["score"] - chance,
                         "ladder_best_lcb_margin": best["lcb"] - chance,
                         "ladder_runs_clear_ref": lr.groupby("task")["above"].sum().astype(int),   # single runs that clear the bound
                         "ladder_runs_ref": lr.groupby("task")["above"].count()})
    small = runs[runs["params"] <= params(TARGET_SIZE)]
    pb = small.loc[small.groupby("task")["score"].idxmax()].set_index("task")
    near["public_best_model"] = pb["model"]
    near["public_best_margin"] = pb["score"] - chance
    return m, lines, near.dropna(subset=["ladder_best_margin"]).reset_index()


def figure(t: pd.DataFrame, m: pd.DataFrame, lines_tab: pd.DataFrame, near: pd.DataFrame, path: Path, pool: str,
           ladder_levels: list, ext_levels: list) -> None:
    fig = plt.figure(figsize=(15.6, 11.2))
    gs = fig.add_gridspec(2, 3, width_ratios=(0.95, 0.9, 1.05), height_ratios=(1, 1.25))
    m_ax, a, b = (fig.add_subplot(gs[0, k]) for k in range(3))
    c, d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    # (0) the public models behind the buckets
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
    b.set_title(f"(b) the {len(g)} tasks the ladder never reads (< ½ of the runs at every size): size floor or benchmark floor?",
                loc="left", fontsize=8.5)
    b.grid(color=S.GRID, lw=.6, axis="x"); S.clean(b)
    n_size = int(g["external_bin"].isin(["≤ 600M", "1B–1.7B"]).sum())
    # (c) how many of the shared tasks a model of size N reads: the public lines against the ladder
    palette = dict(zip(LINES, S.SERIES + S.RAMP[1:]))
    for line, h in lines_tab.groupby("line"):
        h = h.sort_values("params")
        if line == "ladder":
            c.plot(h["params"], h["share"], color=S.INK, marker="s", ms=4, lw=1.6, label=f"the ladder (`{pool}`, gate: ≥ ½ of the runs)", zorder=4)
        else:
            c.plot(h["params"], h["share"], color=palette[line], marker="o", ms=4, lw=1.2, label=f"{line} (base)", zorder=3)
    c.set_xscale("log"); c.set_xlabel("parameters (log)"); c.set_ylabel(f"share of the {len(t)} shared tasks above the gate")
    c.set_ylim(0, 1); c.axvline(params(TARGET_SIZE), color=S.GRID, lw=.8, ls="--")
    c.set_title("(c) how many of the shared tasks a model of size N reads", loc="left", fontsize=8.5)
    c.legend(fontsize=6, frameon=False, loc="lower right"); c.grid(color=S.GRID, lw=.6); S.clean(c)
    # (d) the near-miss view of the ladder-gated tasks at the reference
    n_ = near.sort_values("ladder_best_lcb_margin", ascending=True).reset_index(drop=True)
    y = np.arange(len(n_))
    d.hlines(y, n_["ladder_best_lcb_margin"], n_["ladder_best_margin"], color=S.RAMP[1], lw=1.2, zorder=2)
    d.scatter(n_["ladder_best_margin"], y, s=22, color=S.RAMP[2], zorder=3, label=f"best ladder run at {TARGET_SIZE}: score − chance")
    d.scatter(n_["ladder_best_lcb_margin"], y, s=22, facecolor=S.SURFACE, edgecolor=S.RAMP[2], lw=.9, zorder=3, label="its Wilson 95 % lower bound − chance")
    d.scatter(n_["public_best_margin"], y, s=22, marker="D", color=S.SERIES[1], zorder=3, label=f"best public base model ≤ {TARGET_SIZE}: score − chance")
    d.axvline(0, color=S.INK, lw=.8, ls=":")
    d.set_yticks(y); d.set_yticklabels([f"{r.task}  ({str(r.ladder_best_run).replace('lm-' + TARGET_SIZE + '-', '').replace('-seed1904', '')}, "
                                        f"{int(r.ladder_runs_clear_ref)}/{int(r.ladder_runs_ref)} runs clear)" for r in n_.itertuples()], fontsize=6)
    d.set_xlabel("margin over chance (score − 1/n_options)"); d.set_ylim(-0.8, len(n_) - 0.2)
    n_pass = int((n_["ladder_best_lcb_margin"] > 0).sum())
    top = n_.sort_values("ladder_runs_clear_ref").iloc[-1]
    no_ref = ", ".join(sorted(set(g.loc[~g["task"].isin(near["task"]), "language"])))   # no 1.7B cell trains these
    d.set_title(f"(d) {len(n_)} of the {len(g)} ladder-gated tasks (those with a trained run at {TARGET_SIZE}; none for {no_ref}): "
                f"best single run vs the best public model ≤ {TARGET_SIZE} ({n_pass} single runs clear the bound)", loc="left", fontsize=8.5)
    d.legend(fontsize=6.5, frameon=False, loc="lower right"); d.grid(color=S.GRID, lw=.6, axis="x"); S.clean(d)
    top = G._header(fig, "Benchmark floors: what the ladder cannot read, the public models can — mostly",
                    f"Every task scored by both the ladder (`{pool}`, gate = one-sided 95 % Wilson lower bound over "
                    f"chance for at least half of the size's runs, final checkpoint) and the external tier (public base "
                    f"models 270M–70B, same task, same gate, `all/external/`; panel (0) lists them, grid lines at the bucket sizes). A floor "
                    f"is the smallest level from which "
                    f"the gate holds at every larger informed level. (a) the joint distribution; (b) the tasks the "
                    f"ladder never reads — at EVERY ladder size fewer than {MIN_SHARE:.0%} of the size's runs that train the "
                    f"task's language clear the Wilson bound, so single runs may (at most {top.task}: {int(top.ladder_runs_clear_ref)} of "
                    f"{int(top.ladder_runs_ref)} at {TARGET_SIZE}) — by the public bucket that first reads them: {n_size} of {len(g)} are readable "
                    f"at ≤ 1.7B by a public model — a size/recipe floor the 5×-Chinchilla ladder does not clear — and the "
                    f"rest need ≥ 3B or are never read, a benchmark or language floor. (c) per public base model of the six "
                    f"lines, the share of the shared tasks whose own run clears the bound, against parameters, with the ladder's "
                    f"share per size (its mask: half of the runs); (d) for the ladder-gated tasks with a trained run at {TARGET_SIZE} "
                    f"({len(near)} of {len(g)}: no {TARGET_SIZE} cell trains {no_ref}, rule 2), the best single trained run at "
                    f"{TARGET_SIZE} (score − chance, with its lower bound) beside the best public base model ≤ {TARGET_SIZE} on the "
                    f"same task. Overlap is the 86-task list of "
                    f"the 36-model sweep (no cloze twins, no probe families); some external buckets hold one or two "
                    f"models; external token budgets are not the ladder's, so this is a model floor, not a size-at-5C floor.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame, lines_tab: pd.DataFrame, near: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    gh = f"https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/{stage}/{pool}"
    g = t[t["ladder_gated_at_ref"]]
    lad = lines_tab[lines_tab["line"] == "ladder"].set_index("model")["share"]
    small = lines_tab[(lines_tab["line"] != "ladder") & (lines_tab["params"] <= params(TARGET_SIZE))].sort_values("share", ascending=False)
    n_pass = int((near["ladder_best_lcb_margin"] > 0).sum())
    n_public = int((near["public_best_margin"] > near["ladder_best_margin"]).sum())
    no_ref = ", ".join(sorted(set(g.loc[~g["task"].isin(near["task"]), "language"])))
    counts = g["external_bin"].value_counts().reindex([n for n, _ in BINS], fill_value=0)
    fam = pd.crosstab(g["family"], g["external_bin"]).reindex(columns=[n for n, _ in BINS], fill_value=0)
    fam = fam.loc[fam.sum(axis=1).sort_values(ascending=False).index].reset_index()
    body = "\n\n".join([
        "## Benchmark floors: the ladder's gate against the public models'",
        f"Of the {len(t)} tasks both tiers score, {len(g)} are at chance at every ladder size. Where the public base "
        f"models (270M–70B, same gate) first read them: " + ", ".join(f"{n} {c}" for n, c in counts.items()) + ". "
        f"A task readable at ≤ 1.7B by a public model is a size/recipe floor (the 5×-Chinchilla ladder does not reach "
        f"it; public models of that size train on 10–36 T tokens); a task that needs ≥ 3B or is never read is a "
        f"benchmark or language-resource floor. \"Never reads\" is the gate's verdict: at every ladder size fewer than "
        f"{MIN_SHARE:.0%} of the size's runs that train the task's language clear the one-sided 95 % Wilson bound over chance; "
        f"single runs may clear it. Overlap is the 36-sweep's 86-task list only. Regenerate with "
        f"`python analysis/rq00_gate_and_curves/above_random_external.py --pool {pool}`.",
        md_table(list(fam.columns), fam.values.tolist()),
        f"![The gate on the public models]({stage}/{pool}/above_random_external.png)",
        "Population: the `predictivity` pool (seed 1904, 175M–1.7B, final checkpoint, trained languages) against the "
        "external tier's base models (`all/external`, same gate; panels (c) and (d) use the six public lines "
        + ", ".join(LINES) + "); no task filter beyond the 84-task overlap.",
        "Key findings:\n" + "\n".join([
            f"- {len(g)} of the {len(t)} shared tasks are gated at every ladder size; {int(counts['≤ 600M'] + counts['1B–1.7B'])} "
            f"of them are read by a public base model ≤ 1.7B, {int(counts['never'])} by none up to 70B.",
            f"- (c) the ladder reads {lad.iloc[0]:.0%} of the shared tasks at {lad.index[0]} and {lad.iloc[-1]:.0%} at "
            f"{lad.index[-1]}; the public base models ≤ 1.7B read " + ", ".join(f"{r.model} {r.share:.0%}" for r in small.itertuples())
            + " (a model's own run clears the bound; the ladder's share is its mask, half of the runs).",
            f"- (d) {len(near)} of the {len(g)} ladder-gated tasks have a trained run at {TARGET_SIZE} (no {TARGET_SIZE} cell trains "
            f"{no_ref}, rule 2); {n_pass} of them have a single run whose lower bound clears chance (near misses: the gate wants "
            f"half of the runs), and the best public base model ≤ {TARGET_SIZE} beats the ladder's best run on {n_public}."]),
        "Follow-ups:\n" + "\n".join([
            "- (c) with the post-trained releases as open markers: whether instruction tuning moves a task over the gate at the same size.",
            "- (d) at every ladder size, not only the reference: the size at which each near miss first has a passing run.",
            "- (c)/(d) on the `rf_` cloze twins once the external tier is evaluated on them: the twins pass the gate on the ladder, so the overlap would stop being the old 86-task list."]),
        f"Figure: {gh}/above_random_external.png · tables: {gh}/above_random_external.csv, {gh}/above_random_external_lines.csv"])
    replace_block(GATE_AND_CURVES / "README.md", "above-random-external", body, f"above_random_external.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = GATE_AND_CURVES / stage / args.pool
    t = floors(args.pool)
    m, lines_tab, near = external_lines(t, args.pool)
    t.merge(near, on="task", how="left").to_csv(out_dir / "above_random_external.csv", index=False)
    lines_tab.to_csv(out_dir / "above_random_external_lines.csv", index=False)
    ladder_levels = [c for c in pd.read_csv(out_dir / "above_random_mask.csv", nrows=0).columns if c not in META]
    ext_levels = [c for c in pd.read_csv(EXTERNAL, nrows=0).columns if c not in META]
    print(pd.crosstab(t["ladder_floor"], t["external_bin"]).to_string())
    print(lines_tab.round(3).to_string(index=False))
    print(near.round(3).sort_values("ladder_best_lcb_margin", ascending=False).to_string(index=False))
    figure(t, m, lines_tab, near, out_dir / "above_random_external.png", args.pool, ladder_levels, ext_levels)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, t, lines_tab, near)
