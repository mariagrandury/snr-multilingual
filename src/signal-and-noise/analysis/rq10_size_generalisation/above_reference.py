"""RQ10 — Size generalisation: does a ranking that holds at the 1.7B reference
still hold one rung ABOVE it, at 3B?

Every other RQ stops at the reference (rule 10). This is the one analysis that
opts in with `above_reference=True`: it takes every family with a final at the
`--reference` rung (3B: deep, L ∈ {8, 15}, schemes A and B — four families, six
multi-axis and four mono-axis pairs) and reads, on those families alone,

  (a) DA-size from every smaller rung to the reference, pooled over the gated
      tasks — the scale-convergence line one rung further out — beside the
      SAME families read to 1.7B, so the two lines differ only in the
      reference (a family that 1.7B orders like 3B keeps its 1.7B line);
  (b) DA-goal: every evaluated checkpoint of every smaller rung against the
      reference final, one line per proxy size (the early-and-small reading
      at the new reference);
  (c) per task, DA-size 1.7B → 3B against DA-size 1B → 1.7B on the same
      families: whether the tasks whose ranking converged by 1.7B are the
      ones that keep it at 3B (the question the 3B rung was trained for);
  (d) per benchmark family, DA-size 1.7B → 3B with the pair count.

Multi-axis and mono-axis pair sets both (rule 15, `axes` column); the gate is
`predictivity`'s mask at the proxy and, at the reference rung, the same Wilson
rule computed here on the reference's own runs (rule 1), since the committed
mask stops at 1.7B.

Until the 3B evaluations land the report holds no 3B row: the script then
writes the tables with their headers and a figure that says so, and the driver
runs it every pass so the figure fills in by itself. `--reference 1.7B
--design 3B` is the preview available today — the same four families read to
the current reference — and `--reference 1.7B --check` is the known-answer
check: its DA-size per task and pair set equals rq02's
`predictivity_schemes/da_per_task.csv` (`decision_acc_size_<proxy>`) on every
cell (verified exact on 3,312 cells, 2026-09-23).

    above_reference_<ref>[_design<d>].png / .csv       the pooled lines of (a), (b) and (d)
    above_reference_<ref>[_design<d>]_per_task.csv     one row per (task, axes, proxy size, frac)

    python analysis/rq10_size_generalisation/above_reference.py --pool predictivity            # reference 3B
    python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B
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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, SIZE_GENERALISATION  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, scores_and_mask  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, EVAL_SIZES, MIN_PAIRS, NON_EMB, PAIR_AXES, TARGET_SIZE, at_fraction,
    benchmark_family, design_axes, finals, ladder_frame, on_shared_grid, one_axes,
    pair_sets, size_order)

GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis"
OUT_ROOT = SIZE_GENERALISATION
POOL = "predictivity_all"                   # every trained cell; the pairs are held at the grid seed by pair_sets
REFERENCE = next((s for s in EVAL_SIZES if NON_EMB[s] > NON_EMB[TARGET_SIZE]), TARGET_SIZE)   # 3B today
FRACS = [*CKPT_DA_EARLY_FRACS, 1.0]
# The design set of a rung: which families it was planned with. `3B` is the
# four deep L8/L15 A/B cells (plan/3b_models.md); `all` is every family.
DESIGNS = {"all": lambda a: pd.Series(True, index=a.index),
           "3B": lambda a: (a["arch"] == "deep") & a["L"].isin([8, 15]) & a["scheme"].isin(["A", "B"])}
mpl.rcParams.update(S.RC)


def stem_for(reference: str, design: str) -> str:
    return f"above_reference_{reference}" + (f"_design{design}" if design != "all" else "")


def gate_mask(df: pd.DataFrame, pool: str, reference: str) -> pd.DataFrame:
    """`predictivity`'s committed mask, with the reference rung's column added
    from the same rule on the reference's own runs when the mask has none."""
    mask = load_mask(pool)
    if reference not in mask.columns:
        ref_rows = df[(df["size"] == reference)]
        if len(ref_rows):
            _, m_ref, _ = scores_and_mask(ref_rows[ref_rows["frac"] >= .99], sizes=[reference])
            mask = mask.join(m_ref, how="outer")
        else:
            mask[reference] = pd.array([pd.NA] * len(mask), dtype="Int64")
    return mask


def at_chance(mask: pd.DataFrame, task: str, size: str) -> bool:
    """Rule 1's reading of a mask cell: 0 rejects, 1 and NA (no chance level,
    or no column) pass."""
    if task not in mask.index or size not in mask.columns:
        return False
    v = mask.at[task, size]
    return (not pd.isna(v)) and int(v) == 0


def per_task(df: pd.DataFrame, reference: str, families: list, mask: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, axes, proxy size, frac): the DA of the proxy at that
    checkpoint against the reference final over the pair set, the pair count
    and the gate at both sides."""
    fin = finals(df)
    fin = fin[fin["family"].isin(families)]
    attrs = design_axes(fin)
    sets = pair_sets(attrs)
    ref = fin[fin["size"] == reference]
    ref_by_task = {t: dict(zip(g["family"], g["primary_score"])) for t, g in ref.groupby("task")}
    proxies = [s for s in size_order(df["size"].unique()) if NON_EMB[s] < NON_EMB[reference]]
    rows = []
    for fr in FRACS:
        at = at_fraction(df[df["family"].isin(families) & df["size"].isin(proxies)], fr)
        for (task, size), g in at.groupby(["task", "size"]):
            R = ref_by_task.get(task)
            if not R:
                continue
            P = dict(zip(g["family"], g["primary_score"]))
            for axes in PAIR_AXES[:2]:
                pl = [(a, b) for a, b in sets[axes] if a in P and b in P and a in R and b in R]
                n = len(pl)
                da = np.mean([np.sign(P[a] - P[b]) == np.sign(R[a] - R[b]) for a, b in pl]) if n >= MIN_PAIRS else np.nan
                gated = at_chance(mask, task, size) or at_chance(mask, task, reference)
                rows.append({"task": task, "axes": axes, "size": size, "frac": fr, "reference": reference,
                             "da": da, "n_pairs": n, "n_matching": int(round(da * n)) if n >= MIN_PAIRS else np.nan,
                             "gated": gated, "family": benchmark_family(task)})
    out = pd.DataFrame(rows, columns=["task", "axes", "size", "frac", "reference", "da", "n_pairs", "n_matching", "gated", "family"])
    return out


def pooled(pt: pd.DataFrame) -> pd.DataFrame:
    """The gated, rule-5 cells pooled per (axes, size, frac): the ratio the
    figure draws, with the number of tasks behind it."""
    k = pt[~pt["gated"] & pt["da"].notna() & (pt["family"] != "bpb")]
    out = (k.groupby(["axes", "size", "frac"]).agg(n_matching=("n_matching", "sum"), n_pairs=("n_pairs", "sum"),
                                                   n_tasks=("task", "nunique"), da_macro=("da", "mean")).reset_index())
    out["da"] = out["n_matching"] / out["n_pairs"]
    out["non_emb"] = out["size"].map(NON_EMB)
    return out.sort_values(["axes", "frac", "non_emb"])


def figure(tables: dict, path: Path, pool: str, reference: str, design: str, fams: list, n_ref_runs: int) -> None:
    pt, pool_ref, pt17, pool_17 = tables["per_task"], tables["pooled"], tables["per_task_1.7B"], tables["pooled_1.7B"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    sizes = size_order(pt["size"].unique()) if len(pt) else []
    empty = not len(pool_ref)

    # (a) DA-size to the reference vs to 1.7B, same families
    same_ref = reference == TARGET_SIZE
    for out, ref, ls in ((pool_ref, reference, "-"),) + (() if same_ref else ((pool_17, TARGET_SIZE, (0, (3, 2))),)):
        for axes_, c in (("multi-axis", S.INK), ("mono-axis", S.RAMP[1])):
            g = out[(out["axes"] == axes_) & (out["frac"] == 1.0)]
            if not len(g):
                continue
            ax_a.plot(g["non_emb"], g["da"], color=c, ls=ls, marker="o", ms=4, lw=1.6 if ref == reference else 1.1,
                      label=f"→ {ref} final, {axes_} ({int(g['n_tasks'].median())} tasks)")
            for _, r in g.iterrows():
                ax_a.annotate(f"{int(r['n_tasks'])}", (r["non_emb"], r["da"]), textcoords="offset points", xytext=(0, -9),
                              ha="center", va="top", fontsize=5.5, color=c)
    ax_a.axhline(.5, color=S.MUTED, lw=.8, ls=":")
    ax_a.set_xscale("log"); ax_a.set_xticks([NON_EMB[s] for s in EVAL_SIZES if s in set(sizes) | {TARGET_SIZE}])
    ax_a.set_xticklabels([s for s in EVAL_SIZES if s in set(sizes) | {TARGET_SIZE}]); ax_a.minorticks_off()
    ax_a.set_ylim(0, 1); ax_a.set_xlabel("proxy size (non-embedding parameters)"); ax_a.set_ylabel("DA-size (pooled over gated tasks)")
    ax_a.set_title(f"(a) DA-size to the {reference} final" + ("" if same_ref else f" and to the {TARGET_SIZE} final") + f", {len(fams)} families",
                   loc="left", fontsize=8.5)
    ax_a.legend(fontsize=6.3, frameon=False, loc="lower right"); ax_a.grid(color=S.GRID, lw=.6); S.clean(ax_a)

    # (b) DA-goal along the run, per proxy size, to the reference final
    for s_ in sizes:
        g = pool_ref[(pool_ref["axes"] == "multi-axis") & (pool_ref["size"] == s_)].sort_values("frac")
        if len(g):
            ax_b.plot(g["frac"] * G.CHINCHILLA_AT_FULL, g["da"], color=S.SIZE_COLOR.get(s_, S.INK), marker="o", ms=3, label=s_)
    ax_b.axhline(.5, color=S.MUTED, lw=.8, ls=":")
    ax_b.set_xticks([1, 2, 3, 4, 5]); ax_b.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)]); ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("proxy checkpoint (× Chinchilla)"); ax_b.set_ylabel(f"DA-goal vs the {reference} final (multi-axis)")
    ax_b.set_title(f"(b) early and small, read to {reference}: every checkpoint of every proxy", loc="left", fontsize=8.5)
    if sizes:
        ax_b.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax_b.grid(color=S.GRID, lw=.6); S.clean(ax_b)

    # (c) per task: DA-size 1.7B -> reference vs DA-size 1B -> 1.7B, same families
    a = pt[(pt["axes"] == "multi-axis") & (pt["frac"] == 1.0) & (pt["size"] == TARGET_SIZE) & ~pt["gated"]].set_index("task")["da"]
    b = pt17[(pt17["axes"] == "multi-axis") & (pt17["frac"] == 1.0) & (pt17["size"] == "1B") & ~pt17["gated"]].set_index("task")["da"]
    j = pd.concat([a.rename("to_ref"), b.rename("to_17")], axis=1).dropna()
    if same_ref:
        ax_c.text(.5, .5, f"needs a reference above {TARGET_SIZE}:\nwith --reference {TARGET_SIZE} there is nothing to compare",
                  transform=ax_c.transAxes, ha="center", va="center", fontsize=8, color=S.MUTED)
    elif len(j):
        ax_c.scatter(j["to_17"], j["to_ref"], s=14, color=S.RAMP[2], alpha=.7)
        r = np.corrcoef(j["to_17"], j["to_ref"])[0, 1] if len(j) > 2 else np.nan
        ax_c.text(.02, .97, f"{len(j)} tasks, Pearson r = {r:.2f}", transform=ax_c.transAxes, va="top", fontsize=7)
    ax_c.plot([0, 1], [0, 1], color=S.MUTED, lw=.8, ls=":")
    ax_c.set_xlim(-.03, 1.03); ax_c.set_ylim(-.03, 1.03)
    ax_c.set_xlabel(f"DA-size 1B → {TARGET_SIZE} (same families)"); ax_c.set_ylabel(f"DA-size {TARGET_SIZE} → {reference}")
    ax_c.set_title("(c) per task: does a ranking that converged by 1.7B hold one rung above it?", loc="left", fontsize=8.5)
    ax_c.grid(color=S.GRID, lw=.6); S.clean(ax_c)

    # (d) per benchmark family, DA-size from the largest proxy to the reference
    top = size_order(sizes)[-1] if sizes else None
    d = pt[(pt["axes"] == "multi-axis") & (pt["frac"] == 1.0) & (pt["size"] == top) & ~pt["gated"] & pt["da"].notna()] if top else pt.iloc[:0]
    if len(d):
        fam = d.groupby("family").agg(da=("da", "mean"), n=("task", "nunique")).sort_values("da")
        ax_d.barh(fam.index, fam["da"], color=S.RAMP[1])
        for i, (k, r) in enumerate(fam.iterrows()):
            ax_d.text(r["da"] + .01, i, f"{r['da']:.2f} ({int(r['n'])})", va="center", fontsize=6)
        ax_d.axvline(.5, color=S.MUTED, lw=.8, ls=":")
    ax_d.set_xlim(0, 1.15); ax_d.set_xlabel(f"mean DA-size {top or '—'} → {reference} over the family's gated tasks (n tasks)")
    ax_d.set_title(f"(d) per benchmark family: {top or '—'} → {reference}", loc="left", fontsize=8.5)
    ax_d.grid(color=S.GRID, lw=.6, axis="x"); S.clean(ax_d)

    if empty:
        for ax in (ax_a, ax_b, ax_c, ax_d):
            ax.text(.5, .5, f"no {reference} evaluation in the ladder report yet\n({n_ref_runs} {reference} runs with scores; the figure fills in when they land)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color=S.SERIES[1],
                    bbox=dict(facecolor=S.SURFACE, edgecolor=S.SERIES[1], pad=6))
    top_y = G._header(fig, f"Size generalisation: the {reference} rung as the reference, on the {len(fams)} families it holds"
                      + (f" (design set `{design}`)" if design != "all" else ""),
                      f"Every family with a final at {reference} ({', '.join(fams) if fams else 'none yet'}); pairs at the grid seed, multi-axis "
                      f"({len(pair_sets(design_axes(finals(tables['frame'])))['multi-axis']) if len(fams) else 0}) and mono-axis "
                      f"({len(pair_sets(design_axes(finals(tables['frame'])))['mono-axis']) if len(fams) else 0}) sets (rule 15); a cell needs "
                      f"{MIN_PAIRS} pairs (rule 5); the gate is `{pool}`'s mask at the proxy and the same Wilson rule on the {reference} runs "
                      f"(rule 1); BPB is left out of the pooled lines. (a) and (c) read the same families to {TARGET_SIZE} for comparison, "
                      f"so the two references differ in nothing but the reference. The count under a point is the tasks behind it.")
    fig.tight_layout(rect=(0, 0, 1, top_y))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, reference: str, design: str, stem: str, tables: dict, fams: list, n_ref_runs: int) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel, gh = f"{stage}/{pool}", f"{GITHUB}/rq10_size_generalisation/{stage}/{pool}"
    pooled_ = tables["pooled"]
    key = f"above-reference-{reference}" + (f"-design{design}" if design != "all" else "")
    if not len(pooled_):
        status = (f"**Status: waiting for the {reference} evaluations.** The ladder report holds {n_ref_runs} {reference} runs with "
                  f"scores, so every table is written with its headers only and the figure says so; the driver reruns this step "
                  f"every pass and the block fills in by itself.")
        table = ""
    else:
        g = pooled_[pooled_["frac"] == 1.0]
        status = (f"**Population.** {len(fams)} families with a final at {reference} ({', '.join(fams)}); DA-size pooled over the gated "
                  f"benchmark tasks with ≥ {MIN_PAIRS} pairs.")
        table = md_table(["axes", "proxy", f"DA-size → {reference}", "tasks", f"DA-size → {TARGET_SIZE} (same families)"],
                         [[r["axes"], r["size"], f"{r['da']:.2f}", int(r["n_tasks"]),
                           (lambda q: f"{q['da'].iloc[0]:.2f}" if len(q) else "—")(
                               tables["pooled_1.7B"][(tables["pooled_1.7B"]["axes"] == r["axes"]) & (tables["pooled_1.7B"]["size"] == r["size"]) & (tables["pooled_1.7B"]["frac"] == 1.0)])]
                          for _, r in g.iterrows() if r["size"] != reference])
    body = "\n\n".join(filter(None, [
        f"## The {reference} rung as the reference" + (f" — preview on the `{design}` design set" if design != "all" else ""),
        f"**DA-size and DA-goal · reference {reference} · multi-axis and mono-axis pairs at the grid seed · gate `{pool}` at the proxy, "
        f"the Wilson rule on the {reference} runs at the reference · no filter.** "
        f"Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool {pool} --reference {reference}"
        + (f" --design {design}" if design != "all" else "") + "`.",
        status,
        f"![Size generalisation to {reference}]({rel}/{stem}.png)",
        table,
        f"Files: [`{stem}.png`]({gh}/{stem}.png), [`{stem}.csv`]({gh}/{stem}.csv), [`{stem}_per_task.csv`]({gh}/{stem}_per_task.csv)."]))
    replace_block(OUT_ROOT / "README.md", key, body,
                  f"above_reference.py --pool {pool} --reference {reference}" + (f" --design {design}" if design != "all" else ""))


def run(pool: str, reference: str, design: str, out_dir: Path) -> dict:
    df = ladder_frame(POOL, above_reference=True)
    df = df[on_shared_grid(df)]
    fin = finals(df)
    attrs = design_axes(fin)
    fams = sorted(set(fin.loc[fin["size"] == reference, "family"]) & set(attrs.index[DESIGNS[design](attrs)]))
    n_ref_runs = fin.loc[fin["size"] == reference, "model"].nunique()
    mask = gate_mask(df, pool, reference)
    stem = stem_for(reference, design)
    pt = per_task(df, reference, fams, mask)
    # the same families read to the current reference: the comparison of (a) and (c)
    fams17 = sorted(set(fin.loc[fin["size"] == TARGET_SIZE, "family"]) & set(fams)) if reference != TARGET_SIZE else fams
    pt17 = per_task(df, TARGET_SIZE, fams17, mask) if reference != TARGET_SIZE else pt
    tables = {"per_task": pt, "pooled": pooled(pt), "per_task_1.7B": pt17, "pooled_1.7B": pooled(pt17),
              "frame": df[df["family"].isin(fams)]}
    pt.to_csv(out_dir / f"{stem}_per_task.csv", index=False)
    tables["pooled"].assign(reference=reference).to_csv(out_dir / f"{stem}.csv", index=False)
    figure(tables, out_dir / f"{stem}.png", pool, reference, design, fams, n_ref_runs)
    if out_dir == OUT_ROOT / load_pools()[pool].get("stage", "pretraining") / pool:
        generate_readme(pool, reference, design, stem, tables, fams, n_ref_runs)
    print(f"--- {stem}: {len(fams)} families at {reference} ({n_ref_runs} runs) ---")
    if len(tables["pooled"]):
        print(tables["pooled"][tables["pooled"]["frac"] == 1.0][["axes", "size", "da", "n_tasks", "n_pairs"]].to_string(index=False))
    return tables


def check_against_rq02(tables: dict, pool: str) -> None:
    """Known answer: with the reference at 1.7B and every family, the DA-size per
    task and pair set is rq02's `decision_acc_size_<proxy>` in the
    `predictivity_schemes` table — the pool whose pairs (every scheme at the
    grid seed) the rq02 decision figures are computed over; the `predictivity`
    folder's table holds the A/B-only pool and differs by construction."""
    stage = load_pools()[pool].get("stage", "pretraining")
    raw = pd.read_csv(DECISION_ACCURACY / stage / "predictivity_schemes" / "da_per_task.csv")
    pt = tables["per_task"]
    pt = pt[(pt["frac"] == 1.0) & pt["da"].notna()]
    for axes in PAIR_AXES[:2]:
        da = raw[raw["axes"] == axes].set_index("task")
        cols = [c for c in da.columns if c.startswith("decision_acc_size_") and "_to_" not in c]
        ref = (da[cols].reset_index().melt(id_vars="task", var_name="col", value_name="rq02")
               .assign(size=lambda x: x["col"].str.replace("decision_acc_size_", "")))
        m = pt[pt["axes"] == axes].merge(ref, on=["task", "size"]).dropna(subset=["rq02"])
        diff = (m["da"] - m["rq02"]).abs()
        print(f"known-answer check vs rq02 predictivity_schemes/da_per_task.csv ({axes}): "
              f"{len(m)} cells, {int((diff < 1e-9).sum())} exact, max |diff| = {diff.max():.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="the gate's pool and the output folder")
    ap.add_argument("--reference", default=REFERENCE, choices=EVAL_SIZES, help="the rung read as the reference (default: %(default)s)")
    ap.add_argument("--design", default="all", choices=list(DESIGNS), help="restrict the families to a rung's design set")
    ap.add_argument("--check", action="store_true", help="with --reference 1.7B: compare against rq02's per-task table")
    ap.add_argument("--out-dir", default=None, help="write elsewhere (the known-answer check, so it leaves no table in the RQ folder)")
    args = ap.parse_args()
    out = Path(args.out_dir) if args.out_dir else OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out.mkdir(parents=True, exist_ok=True)
    t = run(args.pool, args.reference, args.design, out)
    if args.check:
        check_against_rq02(t, args.pool)
