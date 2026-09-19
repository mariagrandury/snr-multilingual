"""RQ5 — Which proxy sizes rank a design decision like the reference, and how does that depend on the number of languages?

The plan's question (plan/small-to-large-predictivity-training-plan.md,
"Analysis"): at a given number of languages L, does a proxy size rank a design
choice the way the reference size — the largest model trained at that L —
does? rq00–rq04 ask which *benchmarks* carry reliable signal; this RQ asks
which *model sizes* do.

  intervention DA   — per (intervention, L, population, proxy size, fraction of
                      the proxy's run): the share of population items on which
                      the proxy and the reference (final checkpoint) prefer the
                      same level. Five two-level interventions, each read with
                      the other axes at their baseline: depth (deep vs shallow),
                      language lists (A vs B), temperature (A vs AT3), the
                      second language of L2 (ru vs zh, ru vs es). Populations:
                      per-language BPB on the languages both levels train
                      (`bpb_trained`), on the languages neither trains
                      (`bpb_untrained`), on all 100 (`bpb_all`), the benchmark
                      tasks (`benchmark`), and the two single-item decisions, the
                      macro BPB (`bpb_macro`) and the training loss (`loss`). With two models per item this is
                      `snr.metrics.decision_acc_fast` per item — sign
                      agreement, items the reference ties dropped.
                      `decision_acc_decided` is the same on the items whose
                      reference |Δ| is at least DECIDED seed standard
                      deviations: where the reference's own preference is
                      inside seed noise there is no decision to agree with.
  effect at the reference — per intervention, the |Δ| in seed standard
                      deviations (the paper's "is there a decision to make?").

`early_decision.py` next to this script reads the decision table for "how
small and how early" (paper RQ2); rq06 reads it for the never-trained
languages; rq01's `scaling_law_error.py` and rq03's `effect_vs_noise.py` hold
the two other reads this folder used to carry. The paper's RQ4 figure
(`rq4_interventions`) is drawn here.

    python analysis/rq05_design_decisions/analyze.py --pool predictivity_all
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
from matplotlib.lines import Line2D

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
from analysis.utils import (  # noqa: E402
    GRID_SEED, at_fraction, finals, ladder_frame, size_order, trained_bpb_tasks)

OUT_ROOT = DESIGN_DECISIONS
CANONICAL = "predictivity_all"        # every cell: all seeds and schemes
MIN_ITEMS = 3                         # fewest population items for a DA cell
DECIDED = 2.0                         # reference |Δ| in seed sds for an item to count as decided
FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]     # where the proxy is read, as a share of its run
# key -> (label, axis, levels, (held axis, its baseline level)). The first
# level is the baseline; the reference at each L is the largest size trained
# at both levels.
INTERVENTIONS = {
    "arch":        ("depth (deep vs shallow)",  "arch",   ("deep", "shallow"), ("scheme", "A")),
    "scheme":      ("language lists (A vs B)",  "scheme", ("A", "B"),          ("arch", "deep")),
    "temperature": ("temperature (T=1 vs T=3)", "scheme", ("A", "AT3"),        ("arch", "deep")),
    "zh":          ("2nd language (ru vs zh)",  "scheme", ("A", "ZH"),         ("arch", "deep")),
    "es":          ("2nd language (ru vs es)",  "scheme", ("A", "ES"),         ("arch", "deep")),
}
POPULATIONS = ("bpb_trained", "bpb_untrained", "bpb_all", "benchmark", "bpb_macro", "loss")
SINGLE = {"bpb_macro": "bpb_macro", "loss": "train_loss"}   # one-task populations: the aggregates
CELL_POPULATIONS = ("bpb_trained", "benchmark")   # the items behind the per-benchmark / per-language tables
COLOUR = dict(zip(INTERVENTIONS, [S.RAMP[3], S.RAMP[1], S.SERIES[2], S.SERIES[1], "#8c1d18"]))
mpl.rcParams.update(S.RC)


def _population(sub: pd.DataFrame, name: str, L: int, levels: tuple, axis: str) -> pd.DataFrame:
    """Rows of `sub` (already at one L) belonging to one population."""
    if name == "benchmark":
        return sub[sub["kind"] == "benchmark"]
    if name in SINGLE:
        return sub[sub["task"] == SINGLE[name]]
    bpb = sub[(sub["kind"] == "bpb") & (sub["task"] != "bpb_macro")]
    if name == "bpb_all":
        return bpb
    schemes = levels if axis == "scheme" else ("A",)
    tr = [trained_bpb_tasks(L, s) for s in schemes]
    if any(t is None for t in tr):              # a level defines no list at this L
        return bpb.iloc[0:0]
    if name == "bpb_trained":
        return bpb[bpb["task"].isin(set.intersection(*tr))]
    return bpb[~bpb["task"].isin(set.union(*tr))]   # bpb_untrained


def _pivot(rows: pd.DataFrame, axis: str, levels: tuple) -> pd.DataFrame | None:
    piv = rows.pivot_table(index=["size", "task"], columns=axis, values="primary_score")
    if not set(levels) <= set(piv.columns):
        return None
    return piv.dropna(subset=list(levels))


# --- 1. intervention decision accuracy ---------------------------------------

def seed_sd(fin: pd.DataFrame) -> pd.Series:
    """Per task, the seed standard deviation of the final score: the median
    over the baseline (deep, scheme A) (size, L) cells with replicates."""
    base = fin[(fin["arch"] == "deep") & (fin["scheme"] == "A")]
    sd = base.groupby(["size", "L", "task"])["primary_score"].agg(["std", "count"])
    return sd[sd["count"] >= 2]["std"].groupby("task").median()


def intervention_da(df: pd.DataFrame, fracs: list = FRACS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(the decision table, the per-item agreement behind it, the same for
    every language's BPB with its `group`). The second frame has one row per
    (intervention, L, proxy size, fraction, task) of CELL_POPULATIONS with
    `agree` in {0, 1}: what the per-benchmark and per-language tables
    aggregate; the third is what rq06's transfer lines aggregate."""
    sd = seed_sd(finals(df))
    grid = df[df["seed"] == GRID_SEED]
    fin = finals(grid)
    at_f = {f: at_fraction(grid, f) for f in fracs}
    rows, items_rows, group_rows = [], [], []
    for key, (label, axis, levels, (hcol, hval)) in INTERVENTIONS.items():
        sub_fin = fin[fin[hcol] == hval]
        for L in sorted(sub_fin["L"].unique()):
            for pop in POPULATIONS:
                min_items = 1 if pop in SINGLE else MIN_ITEMS
                ref_piv = _pivot(_population(sub_fin[sub_fin["L"] == L], pop, int(L), levels, axis), axis, levels)
                if ref_piv is None:
                    continue
                sizes = size_order(ref_piv.index.get_level_values("size"))
                if len(sizes) < 2:
                    continue
                ref = sizes[-1]
                r = ref_piv.xs(ref, level="size")
                d_ref = r[levels[0]] - r[levels[1]]
                ref_sign = np.sign(d_ref)[np.sign(d_ref) != 0]      # items the reference decides
                over_sd = d_ref.abs() / d_ref.index.map(sd).to_numpy(float)   # NaN where no seed replicates
                decided = ref_sign.index[over_sd.reindex(ref_sign.index) >= DECIDED]
                for f in fracs:
                    sub = at_f[f]
                    sub = sub[(sub[hcol] == hval) & (sub["L"] == L)]
                    piv = _pivot(_population(sub, pop, int(L), levels, axis), axis, levels)
                    if piv is None:
                        continue
                    for s in sizes:
                        if (s == ref and f == 1.0) or s not in piv.index.get_level_values("size"):
                            continue
                        p = piv.xs(s, level="size")
                        items = p.index.intersection(ref_sign.index)
                        if len(items) < min_items:
                            continue
                        d_proxy = p.loc[items, levels[0]] - p.loc[items, levels[1]]
                        agree = (np.sign(d_proxy) == ref_sign.loc[items]).to_numpy(float)
                        if pop in CELL_POPULATIONS:
                            items_rows.append(pd.DataFrame({
                                "intervention": key, "label": label, "L": int(L), "proxy_size": s, "frac": f,
                                "task": items, "agree": agree}))
                        if pop == "bpb_all":            # every language, grouped by what the cell's lists train
                            group_rows.append(pd.DataFrame({
                                "intervention": key, "label": label, "L": int(L), "proxy_size": s, "frac": f, "reference_size": ref,
                                "group": language_group(items, int(L), levels if axis == "scheme" else ("A",)), "agree": agree}))
                        dec = items.intersection(decided)
                        row = {"intervention": key, "label": label, "population": pop, "L": int(L),
                               "proxy_size": s, "frac": f, "reference_size": ref, "n_items": int(len(items)),
                               "decision_acc": float(agree.mean()), "n_decided": int(len(dec)),
                               "decision_acc_decided": float((np.sign(d_proxy.loc[dec]) == ref_sign.loc[dec]).mean())
                               if len(dec) >= min_items else np.nan}
                        if f == 1.0:
                            # the first level wins an item when its score is
                            # higher on a benchmark, lower on BPB
                            first = (d_ref.loc[items] > 0) if pop == "benchmark" else (d_ref.loc[items] < 0)
                            row.update({"mean_abs_delta_proxy": float(d_proxy.abs().mean()),
                                        "mean_abs_delta_ref": float(d_ref.loc[items].abs().mean()),
                                        "reference_prefers": levels[0] if first.mean() > 0.5 else levels[1]})
                        rows.append(row)
    return (pd.DataFrame(rows), pd.concat(items_rows, ignore_index=True) if items_rows else pd.DataFrame(),
            pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame())


LANGUAGE_GROUPS = ("trained by both levels", "trained by one level", "script trained", "script not trained")


def language_group(tasks, L: int, schemes: tuple) -> list[str]:
    """Per `bpb_<subset>` task, what the lists of the two levels (`schemes`
    at this L) do with the language: both train it, only one does (then the
    decision is mostly "prefer the model that saw it"), neither does but a
    list trains its script, or neither trains even the script. The last two
    are the transfer test proper."""
    lists = [trained_bpb_tasks(L, s) or set() for s in schemes]
    both, either = set.intersection(*lists), set.union(*lists)
    scripts = {t.rsplit("_", 1)[-1] for t in either} | {"Latn"}          # bpb_dclm is English
    return [LANGUAGE_GROUPS[0] if t in both else LANGUAGE_GROUPS[1] if t in either
            else LANGUAGE_GROUPS[2] if t.rsplit("_", 1)[-1] in scripts else LANGUAGE_GROUPS[3] for t in tasks]


def effect_at_reference(fin: pd.DataFrame) -> pd.DataFrame:
    """Per (intervention, L, population): at the reference size, the median
    |Δ| in per-task seed standard deviations (the seed sd of the baseline
    cells, median over the (size, L) cells with replicates)."""
    sd = seed_sd(fin)
    grid = fin[fin["seed"] == GRID_SEED]
    rows = []
    for key, (label, axis, levels, (hcol, hval)) in INTERVENTIONS.items():
        sub = grid[grid[hcol] == hval]
        for L, g in sub.groupby("L"):
            piv = _pivot(g, axis, levels)
            if piv is None:
                continue
            tasks = piv.index.get_level_values("task")
            is_bpb = tasks.str.startswith("bpb_")
            for pop, mask in (("bits per byte", is_bpb & (tasks != "bpb_macro")), ("benchmarks", ~is_bpb & (tasks != "train_loss"))):
                pp = piv[mask]
                counts = pp.groupby(level="size").size()
                sizes = size_order(counts[counts >= MIN_ITEMS].index)
                if not sizes:
                    continue
                ref = sizes[-1]
                p = pp.xs(ref, level="size")
                ratio = ((p[levels[0]] - p[levels[1]]).abs() / p.index.map(sd)).replace(np.inf, np.nan).dropna()
                if len(ratio) >= MIN_ITEMS:
                    rows.append({"intervention": key, "label": label, "L": int(L), "reference_size": ref,
                                 "population": pop, "median_effect_over_seed_sd": float(ratio.median()),
                                 "share_above_2": float((ratio > 2).mean()), "n": int(len(ratio))})
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------

def plot_da_grid(da: pd.DataFrame, path: Path) -> None:
    da = da[da["frac"] == 1.0]
    pops = [p for p in ("bpb_trained", "benchmark", "bpb_all") if p in set(da["population"])]
    keys = [k for k in INTERVENTIONS if k in set(da["intervention"])]
    if not pops or not keys:
        return
    fig, axes = plt.subplots(len(keys), len(pops), figsize=(4.2 * len(pops), 3.0 * len(keys)), squeeze=False)
    im = None
    for i, k in enumerate(keys):
        for j, pop in enumerate(pops):
            ax = axes[i][j]
            sub = da[(da["intervention"] == k) & (da["population"] == pop)]
            if sub.empty:
                ax.set_visible(False)
                continue
            Ls = sorted(sub["L"].unique())
            sizes = size_order(sub["proxy_size"])
            mat = np.full((len(sizes), len(Ls)), np.nan)
            for _, r in sub.iterrows():
                mat[sizes.index(r["proxy_size"]), Ls.index(r["L"])] = r["decision_acc"]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap=S.SEQ, aspect="auto")
            for a in range(len(sizes)):
                for b in range(len(Ls)):
                    if np.isfinite(mat[a, b]):
                        n = int(sub[(sub["proxy_size"] == sizes[a]) & (sub["L"] == Ls[b])]["n_items"].iloc[0])
                        ax.text(b, a, f"{mat[a, b]:.2f}\n(n={n})", ha="center", va="center",
                                fontsize=6, color="white" if mat[a, b] > 0.7 else S.INK)
            ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L{L}" for L in Ls], fontsize=7)
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes, fontsize=7)
            ax.set_title(f"{INTERVENTIONS[k][0]} — {pop}", loc="left", fontsize=8)
            ax.set_xlabel("languages"); ax.set_ylabel("proxy size"); S.clean(ax, spines=()); ax.tick_params(length=0)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="decision accuracy (proxy vs reference, final checkpoints)",
                     fraction=0.02)
    fig.suptitle("Does a proxy size rank the intervention like the reference size at that L?", y=1.0)
    S.save(fig, path, dpi=140)


def plot_interventions(ev: pd.DataFrame, dag: pd.DataFrame, out_dir: Path) -> None:
    """The paper's RQ4 figure: (a) |effect| at the reference in seed sds per
    intervention; (b) final-checkpoint agreement by proxy size per intervention."""
    keys = [k for k in INTERVENTIONS if k in set(ev["intervention"]) | set(dag["intervention"])]
    names = [INTERVENTIONS[k][0] for k in keys]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(keys))
    for j, (pop, col) in enumerate((("bits per byte", S.RAMP[3]), ("benchmarks", S.SERIES[1]))):
        g = (ev[ev["population"] == pop].groupby("intervention")["median_effect_over_seed_sd"].median()
             .reindex(keys))
        a0.barh(y + (j - .5) * .36, g.values, .34, color=col, label=pop)
        for yi, val in zip(y, g.values):
            if np.isfinite(val):
                a0.annotate(f"{val:.1f}×", (val, yi + (j - .5) * .36), xytext=(4, 0), va="center",
                            textcoords="offset points", fontsize=7, color=S.INK)
    a0.axvline(1, color=S.MUTED, ls="--", lw=1); a0.axvline(2, color=S.GRID, lw=.8)
    a0.set_yticks(y); a0.set_yticklabels(names, fontsize=7.5); a0.invert_yaxis()
    a0.set_xscale("log"); a0.set_xlabel("|effect| at the reference, in seed standard deviations (median)")
    a0.set_title("(a) is there a decision to make?", loc="left")
    a0.legend(frameon=False, loc="lower right"); a0.grid(axis="x", color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0); a0.tick_params(length=0)
    styles = {"bpb_trained": ("-", "o"), "benchmark": ("--", "s")}
    order = size_order(dag["proxy_size"])
    for k in keys:
        for pop, (ls, mk) in styles.items():
            g = dag[(dag["intervention"] == k) & (dag["population"] == pop)]
            if g.empty:
                continue
            g = g.set_index("proxy_size").reindex(order)
            a1.plot(range(len(order)), g["decision_acc"], ls=ls, marker=mk, ms=4.5, lw=1.6, color=COLOUR[k])
    a1.axhline(.5, color=S.MUTED, ls=":", lw=1)
    a1.set_xticks(range(len(order))); a1.set_xticklabels(order)
    a1.set_ylim(0, 1.05); a1.set_xlabel("proxy size"); a1.set_ylabel("agreement with the reference (mean over L)")
    a1.set_title("(b) does the proxy agree, per intervention", loc="left")
    h = [Line2D([], [], color=COLOUR[k], lw=1.6, label=INTERVENTIONS[k][0]) for k in keys]
    h += [Line2D([], [], color=S.INK, ls="-", marker="o", ms=4, label="per-language bits per byte"),
          Line2D([], [], color=S.INK, ls="--", marker="s", ms=4, label="benchmark tasks")]
    a1.legend(handles=h, frameon=False, loc="lower right", ncol=1, fontsize=6.8)
    a1.grid(color=S.GRID, lw=.6); a1.set_axisbelow(True); S.clean(a1)
    fig.subplots_adjust(wspace=.5)
    S.save_figure(fig, out_dir, "rq4_interventions")


# --- README ------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, da: pd.DataFrame, ev: pd.DataFrame,
                    dag: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    bullets, blocks = [], []
    fin = da[da["frac"] == 1.0] if not da.empty else da
    if not fin.empty:
        for pop, title in (("bpb_trained", "per-language BPB"), ("benchmark", "benchmarks")):
            for k in INTERVENTIONS:
                core = fin[(fin["intervention"] == k) & (fin["population"] == pop)]
                if core.empty:
                    continue
                grid = core.pivot_table(index="proxy_size", columns="L", values="decision_acc")
                grid = grid.reindex(size_order(grid.index))
                first = {L: next((s for s in grid.index if grid.loc[s, L] >= 0.75), "—") for L in grid.columns}
                if pop == "bpb_trained":
                    bullets.append(f"- **{INTERVENTIONS[k][0]} on {title}** — smallest proxy reaching DA ≥ 0.75 "
                                   "against the reference: " + ", ".join(f"L{L}: {s}" for L, s in first.items()) + ".")
                rows = [[s] + [fmt(grid.loc[s, L]) for L in grid.columns] for s in grid.index]
                refs = ", ".join(f"L{L} → {r}" for L, r in core.groupby("L")["reference_size"].first().items())
                blocks += [f"**{INTERVENTIONS[k][0]}, {title}** (rows: proxy size; columns: L; reference {refs}):",
                           md_table(["proxy"] + [f"L{L}" for L in grid.columns], rows)]
        bench = fin[(fin["intervention"] == "arch") & (fin["population"] == "benchmark")]
        if not bench.empty:
            m = bench.groupby("proxy_size")["decision_acc"].mean()
            bullets.append("- **Depth decision on benchmarks** — mean DA over L by proxy: "
                           + ", ".join(f"{s} {fmt(m[s])}" for s in size_order(m.index)) + ".")
        blocks.append(f"![Intervention DA grid]({rel}/intervention_da.png)")
    if not ev.empty:
        med = ev.groupby(["intervention", "population"])["median_effect_over_seed_sd"].median().unstack("population")
        bullets.append("- **Is there a decision to make?** median |Δ| at the reference in seed sds — "
                       + "; ".join(f"{INTERVENTIONS[k][0]}: " + ", ".join(f"{p} {fmt(v, 1)}×" for p, v in r.items() if np.isfinite(v))
                                   for k, r in med.iterrows()) + ".")
        blocks += ["**Effect at the reference in seed standard deviations** (median over items and L):",
                   md_table(["intervention"] + list(med.columns),
                            [[INTERVENTIONS[k][0]] + [fmt(med.loc[k, c], 1) for c in med.columns] for k in med.index]),
                   f"![Interventions]({rel}/rq4_interventions.png)"]
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq05_design_decisions/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- driver -------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, {fin['task'].nunique()} tasks, "
          f"seeds {sorted(df['seed'].unique())}, schemes {sorted(df['scheme'].unique())}")

    da, items, groups = intervention_da(df)
    da.to_csv(out_dir / "intervention_da.csv", index=False)
    if not items.empty:
        # the same agreement, per benchmark and per language (panels.py draws them);
        # add_meta drops the items with no single language (aggregates, subject facets)
        items = G.add_meta(items)
        keys = ["intervention", "label", "L", "proxy_size", "frac"]
        for by, name in (("family", "benchmark"), ("language", "language")):
            (items.groupby(keys + [by]).agg(decision_acc=("agree", "mean"), n_items=("agree", "size")).reset_index()
             .to_csv(out_dir / f"intervention_da_by_{name}.csv", index=False))
    if not groups.empty:
        (groups.groupby(["intervention", "label", "L", "proxy_size", "frac", "reference_size", "group"])
         .agg(decision_acc=("agree", "mean"), n_items=("agree", "size")).reset_index()
         .to_csv(out_dir / "intervention_da_by_group.csv", index=False))
    print(f"Wrote → {out_dir / 'intervention_da.csv'} ({len(da)} cells)")
    dag = pd.DataFrame()
    if not da.empty:
        plot_da_grid(da, out_dir / "intervention_da.png")
        dag = (da[da["frac"] == 1.0].groupby(["intervention", "label", "population", "proxy_size"])
               .agg(decision_acc=("decision_acc", "mean"), cells=("decision_acc", "size"),
                    refs=("reference_size", lambda s: ",".join(sorted(set(s))))).reset_index())
        dag.to_csv(out_dir / "rq4_da_by_intervention.csv", index=False)

    ev = effect_at_reference(fin)
    ev.to_csv(out_dir / "rq4_effect_vs_seed.csv", index=False)
    if not ev.empty or not dag.empty:
        plot_interventions(ev, dag, out_dir)

    (out_dir / "facts.json").write_text(json.dumps(
        {"rq4": {"effect": ev.round(3).to_dict("records"), "da": dag.round(3).to_dict("records")}},
        indent=1, default=str))
    generate_readme(pool, out_dir, da, ev, dag)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL}; every seed and "
                        "scheme is needed for the five interventions and the seed-noise column).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
