"""Cross-task transfer: does ranking the design variants on task y at a small
size predict their ranking on task x at the 1.7B reference?

The decision-accuracy analysis's cross-task maps (rq02 `cross_task.py`) put every
parent task against every other one and draw the smallest proxy size from
which the transfer is safe (a level). This reads the same pair signs as a value
and turns the map the way the language-transfer question reads it: the PROXY
task on the y axis (its ranking at a proxy size), the TARGET task on the x axis
(its final ranking at 1.7B). A cell is DA-size, the share of design pairs the
two rankings order alike, averaged over the proxies 90M–1B that have a value;
the diagonal is rq02's own-task DA-size. A two-language cell of one benchmark
says whether a small model's ranking on one language predicts the reference's
ranking on the other.

Population: pool `predictivity` (seed 1904, every data build), the mono-axis
design pairs (rule 15: the decision a practitioner makes), >= MIN_PAIRS pairs
per (task pair, size) (rule 5), each side gated where it is at chance (rule 1:
the proxy task at the proxy size, the target task at 1.7B), accuracy-scored
parent benchmark tasks (no BPB, no bBPB twin). Ties follow
`snr.metrics.decision_acc_fast`. Two language populations:

  trained         the loader's default (rule 2): a family's score on a task counts only
                  where it trains the task's language;
  all_languages   rule 2's rq06 exception (`untrained=True`): the scheme-A deep seed-1904
                  cells (`auto_evals_cscs.ALL_LANGUAGES_RUNS`) are also scored on the
                  languages they do not train, and so are, at 350M-1B, the scheme-B L8 deep
                  cells; a task of a language no family trains is read on those cells
                  alone (the six L of scheme A, so its pairs move L only).

Filters (the user's ask: a nice plot, so several are drawn and the clearest is
the paper's):

  above_chance     every task above chance at 1.7B and at >= 1 proxy size
  above_66_either  the paper's rq2 tasks (median DA-size or DA-ckpt >= 0.66, mono-axis)
  L8               the tasks in the eight L8 languages, above chance
  hellaswag        one benchmark, its languages against each other (HellaSwag: the
                   benchmark with the highest DA-size, rq09)

Every map is drawn per task (rows and columns grouped by benchmark) and, for the
multi-benchmark filters, aggregated per benchmark (the mean over the task pairs
of two benchmarks with a value) and per language (the same benchmark in two
languages, mean over the benchmarks).

    cross_task_da_size_by_<task|benchmark|language>_<filter>[_all_languages]_mono_axis.png/.csv
    cross_task_da_size_by_language_hellaswag_mono_axis_paper.png/.csv   the paper's map
    (a per-task map's CSV is the wide matrix, proxy task rows and target task columns, "gated" where grey)
    cross_task_da_size_summary_mono_axis.csv   per (population, filter): tasks, cells, the mean
                                               DA-size on and off the diagonal

    python analysis/rq06_language_transfer/cross_task_transfer.py --pool predictivity
"""
from __future__ import annotations

import argparse
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

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.cross_task import (  # noqa: E402
    _at_chance, cross_da, pair_signs, resource_order, scores_at)
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable  # noqa: E402
from analysis.utils import (MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, _is_parent_task, assign_language,  # noqa: E402
                            benchmark_family, build_snr_pool, design_axes, languages_only, pair_sets)
from pretrain.launch_trainings import cell_languages  # noqa: E402

HERE = Path(__file__).resolve().parent
AXES = "mono-axis"
SUFFIX = "_mono_axis"
POPULATIONS = {"trained": False, "all_languages": True}
FILTERS = ("above_chance", "above_66_either", "L8", "hellaswag")
PAPER = ("all_languages", "hellaswag")      # the paper's map: (population, filter)
mpl.rcParams.update(S.RC)


def accuracy_task(t: str) -> bool:
    f = benchmark_family(t)
    return f not in ("bpb", "loss") and not f.startswith("bbpb")


def maps(pool: str, untrained: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(proxy task x target task) mean DA-size over the proxies, the number of proxies behind it, and
    whether the gate emptied a cell that has pairs (grey)."""
    df = build_snr_pool(pool, untrained=untrained)
    df = df[df["task"].map(_is_parent_task) & df["task"].map(accuracy_task)].copy()
    df["bucket"] = df["size"].map(size_bucket)
    tasks = sorted(df["task"].unique())
    mono = set(pair_sets(design_axes(df))[AXES])
    mask = load_mask(pool) if load_mask(pool) is not None else load_mask(CANONICAL_POOL)
    final = scores_at(df, 1.0)
    ref = final[TARGET_SIZE].reindex(tasks)
    das, gated, informed = [], [], []
    for b in [s for s in SMALL_SIZES if s in final.columns.get_level_values(0)]:
        fams = sorted(set(final[b].columns) & set(ref.columns))
        i, j = np.triu_indices(len(fams), 1)
        keep = np.array([(fams[a], fams[c]) in mono for a, c in zip(i, j)], dtype=bool)
        if keep.sum() < MIN_PAIRS:
            continue
        sp, vp, ip = (x[:, keep] for x in pair_signs(final[b][fams].reindex(tasks), _at_chance(mask, b, tasks)))
        sr, vr, ir = (x[:, keep] for x in pair_signs(ref[fams], _at_chance(mask, TARGET_SIZE, tasks)))
        da, n = cross_da(sp, vp, sr, vr)          # target x proxy
        _, n_all = cross_da(sp, ip, sr, ir)
        das.append(da.T); informed.append((n_all >= MIN_PAIRS).T); gated.append(((n_all >= MIN_PAIRS) & (n < MIN_PAIRS)).T)
        print(f"  {b}: {len(fams)} families, {int(keep.sum())} mono-axis pairs")
    stack = np.stack(das)
    idx = dict(index=pd.Index(tasks, name="proxy_task"), columns=pd.Index(tasks, name="target_task"))
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    n_sizes = np.isfinite(stack).sum(axis=0)
    grey = np.stack(gated).any(axis=0) & (n_sizes == 0)
    return pd.DataFrame(mean, **idx), pd.DataFrame(n_sizes, **idx), pd.DataFrame(grey, **idx)


def select(filt: str, tasks: list, pool: str) -> list:
    """The tasks a filter keeps: above chance at 1.7B and at >= 1 proxy, then the filter's own cut."""
    mask = load_mask(pool) if load_mask(pool) is not None else load_mask(CANONICAL_POOL)
    proxies = [s for s in SMALL_SIZES if s in mask.columns]
    ok = (mask[TARGET_SIZE] == 1).fillna(False) & (mask[proxies] == 1).fillna(False).any(axis=1)
    keep = [t for t in tasks if ok.get(t, False)]
    if filt == "above_66_either":
        rel = set(load_reliable(DECISION_ACCURACY / "pretraining" / pool, filt, axes=AXES)["task"])
        keep = [t for t in keep if t in rel]
    elif filt == "L8":
        keep = [t for t in keep if assign_language(t) in cell_languages(8, "A")]
    elif filt == "hellaswag":
        keep = [t for t in keep if benchmark_family(t) == "hellaswag"]
    return keep


def order(tasks: list) -> list:
    """Benchmarks alphabetically, each one's languages in the resource order of the scheme-A lists."""
    rank = {l: i for i, l in enumerate(resource_order({assign_language(t) for t in tasks}))}
    return sorted(tasks, key=lambda t: (G.display(benchmark_family(t)), rank[assign_language(t)], t))


def long_table(da, n, grey, tasks) -> pd.DataFrame:
    sub = lambda m: m.loc[tasks, tasks]
    t = sub(da).stack(future_stack=True).rename("da_size").reset_index()
    t["n_sizes"] = sub(n).stack(future_stack=True).to_numpy()
    t["gated"] = sub(grey).stack(future_stack=True).to_numpy()
    for side in ("proxy", "target"):
        t[f"{side}_benchmark"] = t[f"{side}_task"].map(benchmark_family)
        t[f"{side}_language"] = t[f"{side}_task"].map(assign_language)
    return t


def aggregate(t: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean DA-size over the task pairs of two benchmarks (by=benchmark) or over the same-benchmark
    pairs of two languages (by=language), with the cell count; grey where every pair is gated."""
    if by == "language":
        t = t[t["proxy_benchmark"] == t["target_benchmark"]]
        t = languages_only(languages_only(t, "proxy_language"), "target_language")      # rule 7
    g = t.groupby([f"proxy_{by}", f"target_{by}"])
    return g.agg(da_size=("da_size", "mean"), n_cells=("da_size", "count"),
                 gated=("gated", "all")).reset_index().rename(columns={f"proxy_{by}": "proxy", f"target_{by}": "target"})


def heatmap(cells: pd.DataFrame, keys: list, path: Path, *, label, ticks: str, title: str, note: str) -> None:
    """y = proxy, x = target, colour = DA-size centred at 0.5 (S.DIV); white = no value, grey = gated."""
    mat = cells.pivot(index="proxy", columns="target", values="da_size").reindex(index=keys, columns=keys)
    grey = cells.pivot(index="proxy", columns="target", values="gated").reindex(index=keys, columns=keys)
    grey = grey.fillna(False).astype(bool).to_numpy() & ~np.isfinite(mat.to_numpy(float))
    n = len(keys)
    paper = path.stem.endswith("_paper")
    side = min(13.0, max(4.5, 0.22 * n + 2.5)) if ticks == "each" else 12.0
    fig, ax = plt.subplots(figsize=(side + 1.2, side))
    ax.imshow(np.where(grey, 1.0, 0.0), cmap=mpl.colors.ListedColormap([S.SURFACE, S.NODATA]), vmin=0, vmax=1,
              interpolation="nearest")
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), cmap=S.DIV, vmin=0, vmax=1, interpolation="nearest")
    ax.plot([-.5, n - .5], [-.5, n - .5], color=S.MUTED, lw=.4, alpha=.6)
    if ticks == "each":
        ax.set_xticks(range(n)); ax.set_xticklabels([label(k) for k in keys], rotation=90, fontsize=6.5)
        ax.set_yticks(range(n)); ax.set_yticklabels([label(k) for k in keys], fontsize=6.5)
    else:                                       # task maps: one tick per benchmark block
        fams = pd.Series([benchmark_family(k) for k in keys])
        edges = np.flatnonzero(fams.ne(fams.shift()).to_numpy())
        bounds = np.r_[edges, n]
        centres = (bounds[:-1] + bounds[1:] - 1) / 2
        names = [f"{G.paper_name(fams[e])} ({m})" for e, m in zip(edges, np.diff(bounds))]
        for e in edges[1:]:
            ax.axhline(e - .5, color=S.INK, lw=.35); ax.axvline(e - .5, color=S.INK, lw=.35)
        ax.set_xticks(centres); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(centres); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Target task, its ranking at 1.7B" if paper else "target x: its final ranking at 1.7B", fontsize=8)
    ax.set_ylabel("Proxy task, its ranking at 90M to 1B" if paper else "proxy y: its final ranking at a proxy size", fontsize=8)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("DA-size, mean over 90M to 1B", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    if ticks == "each":
        cells.to_csv(path.with_suffix(".csv"), index=False)
    else:       # a task map is wide (proxy task rows, target task columns): the long form runs to 50 MB
        mat.astype(object).where(~grey, "gated").to_csv(path.with_suffix(".csv"))
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    top = G._header(fig, title, note)
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def summary_row(t: pd.DataFrame, pop: str, filt: str) -> dict:
    v = t.dropna(subset=["da_size"])
    diag = v["proxy_task"] == v["target_task"]
    same_b = v["proxy_benchmark"] == v["target_benchmark"]
    same_l = v["proxy_language"] == v["target_language"]
    return {"population": pop, "filter": filt, "n_tasks": t["proxy_task"].nunique(), "n_cells": len(v),
            "n_gated": int(t["gated"].sum()), "da_diagonal": v.loc[diag, "da_size"].mean(),
            "da_same_benchmark_other_language": v.loc[same_b & ~same_l, "da_size"].mean(),
            "da_other_benchmark_same_language": v.loc[~same_b & same_l, "da_size"].mean(),
            "da_other_benchmark_other_language": v.loc[~same_b & ~same_l, "da_size"].mean(),
            "share_above_075": (v.loc[~diag, "da_size"] >= 0.75).mean()}


def generate_readme(pool: str, summ: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    f = lambda v: "–" if pd.isna(v) else f"{v:.2f}"
    rows = ["| population | filter | tasks | cells | diagonal | same benchmark, other language | other benchmark, same language "
            "| other benchmark and language | off-diagonal share ≥ 0.75 |", "|---|---|---|---|---|---|---|---|---|"]
    rows += [f"| {r.population} | `{r.filter}` | {r.n_tasks} | {r.n_cells} | {f(r.da_diagonal)} | "
             f"{f(r.da_same_benchmark_other_language)} | {f(r.da_other_benchmark_same_language)} | "
             f"{f(r.da_other_benchmark_other_language)} | {f(r.share_above_075)} |" for r in summ.itertuples()]
    p, flt = PAPER
    sfx = "_all_languages" if p == "all_languages" else ""
    body = "\n\n".join([
        "## Cross-task transfer of the ranking",
        f"Pool `{pool}` (seed 1904, every data build), mono-axis pairs, ≥ {MIN_PAIRS} pairs per (task pair, size), each "
        f"side gated (proxy task at the proxy size, target task at {TARGET_SIZE}), accuracy-scored parent tasks. A cell: "
        f"DA-size of the proxy task's final ranking at a proxy size (y) against the target task's final ranking at "
        f"{TARGET_SIZE} (x), the mean over 90M–1B. `trained` reads each family only on the languages it trains (rule 2); "
        "`all_languages` adds the scheme-A deep seed-1904 cells' scores on every other language (and the scheme-B L8 "
        "deep cells' at 350M–1B), rq06's exception to rule 2. Regenerate with "
        f"`python analysis/rq06_language_transfer/cross_task_transfer.py --pool {pool}`.",
        "\n".join(rows),
        f"![HellaSwag, language against language]({stage}/{pool}/cross_task_da_size_by_language_{flt}{sfx}{SUFFIX}.png)",
        f"![By benchmark, the rq2 tasks]({stage}/{pool}/cross_task_da_size_by_benchmark_above_66_either{SUFFIX}.png)",
        f"![By language, the rq2 tasks]({stage}/{pool}/cross_task_da_size_by_language_above_66_either{SUFFIX}.png)",
        "The other maps (every filter, per task, per benchmark and per language, both populations) sit beside them as "
        f"`cross_task_da_size_by_<task|benchmark|language>_<filter>[_all_languages]{SUFFIX}.png`."])
    replace_block(HERE / "README.md", "cross-task-transfer", body, f"cross_task_transfer.py --pool {pool}")


def main(pool: str) -> None:
    out_dir = HERE / load_pools()[pool].get("stage", "pretraining") / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    summ = []
    for pop, untrained in POPULATIONS.items():
        print(f"{pop}:")
        da, n, grey = maps(pool, untrained)
        sfx = ("_all_languages" if untrained else "") + SUFFIX
        for filt in FILTERS:
            tasks = order(select(filt, list(da.index), pool))
            if len(tasks) < 2:
                print(f"  {filt}: {len(tasks)} tasks, nothing to draw")
                continue
            t = long_table(da, n, grey, tasks)
            summ.append(summary_row(t, pop, filt))
            what = f"{pop.replace('_', ' ')}, {filt.replace('_', ' ')}"
            note = (f"cell = DA-size of the proxy task's final ranking (y) at a proxy size against the target task's final "
                    f"ranking at {TARGET_SIZE} (x), mean over 90M–1B; mono-axis pairs, ≥ {MIN_PAIRS} per size; white = no "
                    f"value, grey = the gate emptied every size; diagonal = the task's own DA-size; {len(tasks)} tasks")
            if filt == "hellaswag":
                t2 = t.rename(columns={"proxy_task": "proxy", "target_task": "target"})
                keys = list(dict.fromkeys(tasks))
                lab = lambda k: assign_language(k)
                for name in [f"cross_task_da_size_by_language_{filt}{sfx}"] + (
                        [f"cross_task_da_size_by_language_{filt}{sfx}_paper"] if (pop, filt) == PAPER else []):
                    heatmap(t2, keys, out_dir / f"{name}.png", label=lab, ticks="each",
                            title=f"Cross-task DA-size, HellaSwag language against language ({pop.replace('_', ' ')})",
                            note=note)
                continue
            heatmap(t.rename(columns={"proxy_task": "proxy", "target_task": "target"}), tasks,
                    out_dir / f"cross_task_da_size_by_task_{filt}{sfx}.png", label=str, ticks="blocks",
                    title=f"Cross-task DA-size per task ({what})", note=note)
            b = aggregate(t, "benchmark")
            heatmap(b, sorted(set(b["proxy"]), key=G.display), out_dir / f"cross_task_da_size_by_benchmark_{filt}{sfx}.png",
                    label=G.paper_name, ticks="each", title=f"Cross-task DA-size per benchmark ({what})",
                    note=note + "; a cell = the mean over the task pairs of the two benchmarks")
            lg = aggregate(t, "language")
            heatmap(lg, resource_order(set(lg["proxy"])), out_dir / f"cross_task_da_size_by_language_{filt}{sfx}.png",
                    label=str, ticks="each", title=f"Cross-task DA-size per language ({what})",
                    note=note + "; a cell = the mean over the benchmarks of the same benchmark in the two languages")
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / f"cross_task_da_size_summary{SUFFIX}.csv", index=False)
    print(summ.round(3).to_string(index=False))
    generate_readme(pool, summ)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
