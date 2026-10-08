"""English only, scaling regimes: the K = 1 cells under the scaling-predictability fits.

The scaling-predictability figure (rq01 `regimes.py`) places every task by the
median R² and Spearman ρ of its gated log-N fits (one per L) and the median R²
of its training-trajectory fits (one per (L, size)), over the deep data-A
seed-1904 cells of every L. This draws the same two panels for the three
monolingual English corpora at L = 1, one family each (deep, seed 1904,
90M–1.7B): DCLM-Edu (data A, the English every other cell trains on), DCLM
without the edu filter (DCLMP) and FineWeb (FWEB). The fits, the gate (rule 1,
the `predictivity` mask), the trajectory minimum and the regimes are rq01's
own (`fit_table`, `regime_table`); the one difference is forced by the design:
one L, so a task has one size fit per corpus instead of a median over L
(`min_fits=1`), while the trajectory R² stays a median over at least two sizes.
The L1 cells train English only, so the tasks are the English benchmarks and
the English validation BPB (rule 2). The grey points behind are the tasks of
rq01's figure (every L, every trained language), the multilingual reading the
K = 1 points are compared with. A second figure keeps only the three L1 cells
on the English benchmarks: the accuracy-scored English tasks (originals, RF and
LLM-RF twins; `utils.variant` scoring `acc`), without the bBPB twins and
`bpb_dclm` (`utils.lower_is_better`), and with no point from an L > 1 cell,
the grey backdrop included.

    english_only_scaling_regimes.png / .csv         the figure (README) and its table: one row per (corpus, task),
                                                    plus rq01's points as corpus `every K (rq01)`
    english_only_scaling_regimes_paper.png / .csv   the same, bare (rule 18), for the paper
    english_only_scaling_regimes_summary.csv        per corpus: tasks and the share in each regime; and the same
                                                    single-fit reading of the English tasks of the deep data-A
                                                    cell at every L (`L<k> DCLM-Edu`), so L1 is read against
                                                    the other L on one fit each, not against rq01's medians
    english_only_scaling_regimes_l1_accuracy.png / .csv        the L1 cells alone on the English accuracy tasks,
                                                               no grey points; one row per (corpus, task) with its format
    english_only_scaling_regimes_l1_accuracy_paper.png / .csv  the same, bare (rule 18), for the paper

    python analysis/rq13_english_only/english_regimes.py --pool predictivity
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq01_scaling_predictability.analyze import fit_table  # noqa: E402
from analysis.rq01_scaling_predictability.regimes import (  # noqa: E402
    CANONICAL as RQ01_POOL, DECLINES, MIN_POINTS, QUADRANTS, R2_SPLIT, REGIME_SHORT, _edged, _quadrants, regime_table)
from analysis.utils import GRID_SEED, finals, ladder_frame, lower_is_better, variant  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPORA = {"A": "DCLM-Edu", "DCLMP": "DCLM", "FWEB": "FineWeb"}     # the L1 data builds, in the order drawn
BACKGROUND = "every K (rq01)"
REGIMES = [v[0] for v in QUADRANTS.values()] + [DECLINES]
L1_ACCURACY = "english_only_scaling_regimes_l1_accuracy"
mpl.rcParams.update(S.RC)


def cells_of(data: str, L: int = 1):
    """The one family of a corpus at L: deep, seed 1904."""
    return lambda d: d[(d["seed"] == GRID_SEED) & (d["ladder"] == "deep") & (d["L"] == L) & (d["data"] == data)]


def table(pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The K = 1 regimes per corpus with rq01's points behind (the figure's
    table), and the single-fit regimes of the English tasks of the deep data-A
    cell at every L (for the summary)."""
    df = ladder_frame(pool)
    fin = finals(df)

    def one(data, L):
        fits, _ = fit_table(fin, pool, cells=cells_of(data, L))
        return regime_table(fits, df, pool, cells=cells_of(data, L), min_fits=1)
    parts = [one(data, 1).assign(corpus=name) for data, name in CORPORA.items()]
    rq01 = pd.read_csv(SCALING_PREDICTABILITY / load_pools()[RQ01_POOL].get("stage", "pretraining") / RQ01_POOL
                       / "scaling_regimes.csv")
    by_L = pd.concat([one("A", L).assign(corpus=f"L{L} {CORPORA['A']}") for L in sorted(fin["L"].unique()) if L > 1])
    return pd.concat(parts + [rq01.assign(corpus=BACKGROUND)], ignore_index=True), by_L[by_L["language"] == "en"]


def l1_accuracy(t: pd.DataFrame) -> pd.DataFrame:
    """The figure's table cut to the L1 cells on the English benchmarks: the
    accuracy-scored tasks (originals, RF, LLM-RF), no bBPB twin, no `bpb_dclm`,
    no rq01 point."""
    k1 = t[(t["corpus"] != BACKGROUND) & ~t["task"].map(lower_is_better)]
    k1 = k1.assign(format=k1["task"].map(lambda x: variant(x)[0]))
    assert (k1["language"] == "en").all() and (k1["task"].map(lambda x: variant(x)[1]) == "acc").all()
    return k1.reset_index(drop=True)


def summary(t: pd.DataFrame, by_L: pd.DataFrame) -> pd.DataFrame:
    t = pd.concat([t, by_L])
    s = t.groupby("corpus")["regime"].value_counts(normalize=True).unstack().reindex(columns=REGIMES).fillna(0)
    s.insert(0, "n_tasks", t.groupby("corpus").size())
    return s.reindex([*CORPORA.values(), *by_L["corpus"].unique(), BACKGROUND]).reset_index()


def figure(t: pd.DataFrame, path: Path, paper: bool) -> None:
    """Two rows (panel (a) and (b) of rq01's figure) by one column per corpus,
    rq01's points in grey behind when `t` holds them."""
    back = t[t["corpus"] == BACKGROUND]
    fig, axes = plt.subplots(2, len(CORPORA), figsize=(7.0, 4.9), sharex=True)
    for j, (name, colour) in enumerate(zip(CORPORA.values(), S.SERIES)):
        g = t[t["corpus"] == name]
        a, b = axes[0, j], axes[1, j]
        _quadrants(b, labels=False)
        for ax, y in ((a, "rho_size"), (b, "r2_trajectory")):
            ax.scatter(back["r2_size"], back[y], s=4, color=S.MUTED, alpha=.18, lw=0, zorder=1)
            ax.scatter(g["r2_size"], g[y], **_edged(dict(s=9, color=colour, alpha=.85, zorder=2), g))
            ax.set_xlim(-0.02, 1.02); ax.set_xticks([0, .5, 1]); ax.grid(color=S.GRID, lw=.5); S.clean(ax)
            ax.set_box_aspect(1)
        a.set_title(f"{name} ({len(g)} tasks)", fontsize=8, loc="left")
        a.set_ylim(-1.05, 1.05); a.axhline(0, color=S.GRID, lw=.6)
        b.set_ylim(-0.02, 1.02)
        b.set_xlabel("R² of the model-size scaling fit" if j == 1 else "")
    axes[0, 0].set_ylabel("Spearman ρ with model size")
    axes[1, 0].set_ylabel("Median R² across\ntraining-trajectory fits")
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    if paper:
        fig.tight_layout(h_pad=0.6, w_pad=0.4)
        S.save_paper(fig, path.with_suffix(""))
        return
    l1 = back.empty
    top = G._header(
        fig, "Scaling regimes of the English-only cells" + (" on the English benchmarks" if l1 else "")
        + ", one per English corpus",
        "point = one English " + ("accuracy task (original, RF or LLM-RF; no bBPB twin, no validation BPB)" if l1 else "task")
        + f" of the L1 deep seed-{GRID_SEED} cell of that corpus, 90M–1.7B, at the sizes where the task is "
        "above chance (rule 1): (a) R² and Spearman ρ of its final score ~ log N fit (one fit, one L), (b) the same R² against the "
        f"median R² of score ~ log tokens over each run's checkpoints (≥ {MIN_POINTS} points, ≥ 2 sizes); quadrants split at "
        f"R² = {R2_SPLIT}, black edge = ρ < 0 ('{DECLINES}'); "
        + ("no point from an L > 1 cell" if l1 else "grey = the tasks of rq01's figure (every L, medians over L)"))
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def generate_readme(t: pd.DataFrame, s: pd.DataFrame, pool: str) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    si = s.set_index("corpus")
    share = lambda c, r: f"{si.at[c, r]:.0%}"
    k1 = t[t["corpus"] != BACKGROUND]
    med = k1.groupby("corpus")[["r2_size", "r2_trajectory"]].median()
    acc = k1[~k1["family"].str.startswith(("bbpb", "bpb"))]
    both = "predictable across both"
    by_L = [c for c in si.index if c not in CORPORA.values() and c != BACKGROUND]
    lo, hi = si.loc[by_L, both].min(), si.loc[by_L, both].max()
    rows = [f"- **{c}** ({int(si.at[c, 'n_tasks'])} tasks, {int((acc['corpus'] == c).sum())} accuracy-scored): "
            + ", ".join(f"{REGIME_SHORT[r]} {share(c, r)}" for r in REGIMES)
            + f"; median R² across size {med.at[c, 'r2_size']:.2f}, along training {med.at[c, 'r2_trajectory']:.2f}; "
            f"the accuracy-scored tasks alone: both {(acc.loc[acc['corpus'] == c, 'regime'] == both).mean():.0%}."
            for c in CORPORA.values()]
    rows.append(f"- **The same single-fit reading at every other L** (the English tasks of the deep data-A cell, "
                + "–".join(dict.fromkeys(str(int(n)) for n in sorted(si.loc[by_L, "n_tasks"]))) + " tasks): predictable across both "
                f"{lo:.0%}–{hi:.0%} at L2–L50 ("
                + ", ".join(f"{c.split()[0]} {si.at[c, both]:.0%}" for c in by_L) + ").")
    rows.append(f"- **{BACKGROUND}** ({int(si.at[BACKGROUND, 'n_tasks'])} tasks, every trained language, medians over "
                "the L): " + ", ".join(f"{REGIME_SHORT[r]} {share(BACKGROUND, r)}" for r in REGIMES)
                + ". A median over six fits is smoother than one fit, so this is the figure's backdrop, not the "
                "like-for-like comparison (the bullet above is).")
    body = "\n\n".join([
        f"![Scaling regimes at L1]({stage}/{pool}/english_only_scaling_regimes.png)",
        f"Pool `{pool}`: the L1 deep seed-{GRID_SEED} cell of each English corpus (DCLM-Edu = data A, DCLM = DCLMP, "
        f"FineWeb = FWEB), 90M–1.7B, gate `{CANONICAL_POOL}`; the fits and regimes of the scaling-predictability figure, one "
        "size fit per task (one L) and the median trajectory R² over ≥ 2 sizes; English tasks (originals, RF, LLM-RF, bBPB "
        "twins) and `bpb_dclm`. The populations differ: the corpora share the gate, so they read the same tasks, and rq01's "
        "grey points are every trained language. Regenerate with "
        f"`python analysis/rq13_english_only/english_regimes.py --pool {pool}`.",
        "Key findings (share of tasks per regime; both = predictable across size and training):",
        "\n".join(rows)])
    replace_block(HERE / "README.md", "english-regimes", body, f"english_regimes.py --pool {pool}")


def generate_readme_l1(a: pd.DataFrame, pool: str) -> None:
    """The block of the L1-only figure, every number read from its table `a`."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    sets = {c: frozenset(g["task"]) for c, g in a.groupby("corpus")}
    same = len(set(sets.values())) == 1
    n = a.groupby("corpus").size()
    share = a.groupby("corpus")["regime"].value_counts(normalize=True).unstack().reindex(columns=REGIMES).fillna(0)
    med = a.groupby("corpus")[["r2_size", "r2_trajectory"]].median()
    fmt = a[a["corpus"] == next(iter(CORPORA.values()))]["format"].value_counts()
    both = "predictable across both"
    agree = a.pivot(index="task", columns="corpus", values="regime").dropna()
    rows = [f"- **{c}** ({n[c]} tasks): " + ", ".join(f"{REGIME_SHORT[r]} {share.at[c, r]:.0%}" for r in REGIMES)
            + f"; median R² across size {med.at[c, 'r2_size']:.2f}, along training {med.at[c, 'r2_trajectory']:.2f}."
            for c in CORPORA.values()]
    by_fmt = a.assign(b=a["regime"] == both).groupby(["format", "corpus"])["b"].mean()
    rows.append("- **Predictable across both, per format**: " + "; ".join(
        f"{G._TWIN_NAMES.get(f, f)} ({n_f} task{'s' * (n_f != 1)}) " + ", ".join(f"{c} {by_fmt[(f, c)]:.0%}" for c in CORPORA.values())
        for f, n_f in fmt.items()) + ".")
    rows.append(f"- **Same regime in all three corpora**: {(agree.nunique(axis=1) == 1).sum()} of {len(agree)} tasks; "
                f"predictable across both in all three: {(agree == both).all(axis=1).sum()}.")
    body = "\n\n".join([
        f"![Scaling regimes at L1 on the English benchmarks]({stage}/{pool}/{L1_ACCURACY}.png)",
        f"Pool `{pool}`: the L1 deep seed-{GRID_SEED} cell of each English corpus, 90M–1.7B, gate `{CANONICAL_POOL}`, the fits "
        "and regimes of the figure above; English accuracy tasks only ("
        + ", ".join(f"{v} {G._TWIN_NAMES.get(k, k)}" for k, v in fmt.items()) + "), no bBPB twin, no `bpb_dclm`, and no point from an L > 1 cell. "
        + ("The three corpora read the same tasks." if same else "The populations differ: the corpora read different tasks.")
        + f" Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool {pool}`.",
        "Key findings (share of tasks per regime; both = predictable across size and training):",
        "\n".join(rows),
        " · ".join(f"[{L1_ACCURACY}.{x}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/"
                   f"analysis/{HERE.name}/{stage}/{pool}/{L1_ACCURACY}.{x})" for x in ("png", "csv"))])
    replace_block(HERE / "README.md", "english-regimes-l1-accuracy", body, f"english_regimes.py --pool {pool}")


def main(pool: str) -> None:
    out_dir = HERE / load_pools()[pool].get("stage", "pretraining") / pool
    t, by_L = table(pool)
    s = summary(t, by_L)
    for name in ("english_only_scaling_regimes", "english_only_scaling_regimes_paper"):
        t.to_csv(out_dir / f"{name}.csv", index=False)
        figure(t, out_dir / f"{name}.png", paper=name.endswith("_paper"))
    s.to_csv(out_dir / "english_only_scaling_regimes_summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    generate_readme(t, s, pool)
    a = l1_accuracy(t)
    for name in (L1_ACCURACY, f"{L1_ACCURACY}_paper"):
        a.to_csv(out_dir / f"{name}.csv", index=False)
        figure(a, out_dir / f"{name}.png", paper=name.endswith("_paper"))
    generate_readme_l1(a, pool)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
