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
    english_only_scaling_regimes_l1_accuracy_tagged.png / .csv        the L1 accuracy figure's panel (b), one panel per
                                                                      corpus, every point named; the table adds the
                                                                      English accuracy tasks the gate filters out
                                                                      (regime `filtered`, no fit) and each task's name
    english_only_scaling_regimes_l1_accuracy_tagged_paper.png / .csv  the same, bare (rule 18), for the paper

    python analysis/rq13_english_only/english_regimes.py --pool predictivity
"""
from __future__ import annotations

import argparse
import re
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
from analysis.utils import GRID_SEED, assign_language, benchmark_family, finals, ladder_frame, lower_is_better, variant  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPORA = {"A": "DCLM-Edu", "DCLMP": "DCLM", "FWEB": "FineWeb"}     # the L1 data builds, in the order drawn
BACKGROUND = "every K (rq01)"
REGIMES = [v[0] for v in QUADRANTS.values()] + [DECLINES]
L1_ACCURACY = "english_only_scaling_regimes_l1_accuracy"
L1_TAGGED = f"{L1_ACCURACY}_tagged"
FILTERED = "filtered"      # an English accuracy task the gate leaves without a size fit or a trajectory fit
# a name's colour in the tagged figure: its quadrant's hue, dark enough for 6 pt text
REGIME_INK = {"predictable across both": "#1c5cab", "predictable during training only": "#3d7a2a",
              "predictable across size only": "#3f6670", "weak / unpredictable in both": "#a0522d", DECLINES: S.INK}
# the well-known English benchmarks the README and the paper name, by their task key (original and twins)
POPULAR = ["mmlu", "rf_mmlu", "global_mmlu_full_en", "rf_global_mmlu_full_en", "arc_easy", "arc_challenge", "hellaswag",
           "openbookqa", "xwinograd_en", "global_piqa_parallel_cloze_eng_latn", "lambada_openai_mt_en", "commonsense_qa",
           "rf_commonsense_qa", "belebele_eng_Latn", "rf_belebele_eng_Latn", "truthfulqa_mc2", "xstorycloze_en", "xnli_en"]
# task names the generic `tag` would read badly
TAGS = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "global_piqa_parallel_cloze_eng_latn": "Global PIQA",
        "lambada_openai_mt_en": "LAMBADA", "truthfulqa-multi_mc1_en": "TruthfulQA-Multi", "blend_us_sample": "BLEnD (US)"}
SUBTASKS = {"logical_deduction_three_objects": "logical deduction 3", "logical_deduction_five_objects": "logical deduction 5",
            "logical_deduction_seven_objects": "logical deduction 7", "reasoning_about_colored_objects": "colored objects",
            "tracking_shuffled_objects_three_objects": "shuffled objects 3",
            "tracking_shuffled_objects_five_objects": "shuffled objects 5",
            "tracking_shuffled_objects_seven_objects": "shuffled objects 7",
            "salient_translation_error_detection": "translation errors", "disambiguation_qa": "disambiguation QA", "united_kingdom": "UK", "united_states": "US",
            "south_africa": "South Africa"}
mpl.rcParams.update(S.RC)


def cells_of(data: str, L: int = 1):
    """The one family of a corpus at L: deep, seed 1904."""
    return lambda d: d[(d["seed"] == GRID_SEED) & (d["ladder"] == "deep") & (d["L"] == L) & (d["data"] == data)]


def tag(task: str) -> str:
    """A task as the tagged figure names it: the paper's benchmark name, the
    subtask, the twin (`rf_bbh_mcq_date_understanding` -> `BBH date understanding RF`)."""
    if task in TAGS:
        return TAGS[task]
    fmt = variant(task)[0]
    t = task.removeprefix(f"{fmt}_") if fmt != "original" else task
    fam = G.base(benchmark_family(t))
    sub = re.sub(r"(^|_)(eng_Latn|eng_latn|eng|en)$", "", t.removeprefix(fam).strip("_"))
    name = G.paper_name(fam).replace(" (MCQ)", "")
    if sub:
        name += " " + SUBTASKS.get(sub, sub.replace("_", " "))
    return name + ("" if fmt == "original" else f" {G._TWIN_NAMES[fmt]}")


def table(pool: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The K = 1 regimes per corpus with rq01's points behind (the figure's
    table), the single-fit regimes of the English tasks of the deep data-A
    cell at every L (for the summary), and the English accuracy tasks of each
    L1 cell that the gate filters out (no regime)."""
    df = ladder_frame(pool)
    fin = finals(df)
    filtered = []

    def one(data, L):
        fits, _ = fit_table(fin, pool, cells=cells_of(data, L))
        t = regime_table(fits, df, pool, cells=cells_of(data, L), min_fits=1)
        if L == 1:
            f = fits[(fits["kind"] == "benchmark") & ~fits["task"].isin(t["task"]) & ~fits["task"].map(lower_is_better)]
            f = f[(f["task"].map(assign_language) == "en") & (f["task"].map(lambda x: variant(x)[1]) == "acc")]
            filtered.append(f[["task", "family"]].drop_duplicates().assign(language="en", regime=FILTERED,
                                                                         corpus=CORPORA[data]))
        return t
    parts = [one(data, 1).assign(corpus=name) for data, name in CORPORA.items()]
    rq01 = pd.read_csv(SCALING_PREDICTABILITY / load_pools()[RQ01_POOL].get("stage", "pretraining") / RQ01_POOL
                       / "scaling_regimes.csv")
    by_L = pd.concat([one("A", L).assign(corpus=f"L{L} {CORPORA['A']}") for L in sorted(fin["L"].unique()) if L > 1])
    return (pd.concat(parts + [rq01.assign(corpus=BACKGROUND)], ignore_index=True), by_L[by_L["language"] == "en"],
            pd.concat(filtered, ignore_index=True))


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


def tagged(a: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    """The L1 accuracy table with each task's name, plus the English accuracy
    tasks the gate filters out in each L1 cell (regime FILTERED, no fit)."""
    out = pd.concat([a, filtered.assign(format=filtered["task"].map(lambda x: variant(x)[0]))], ignore_index=True)
    return out.assign(name=out["task"].map(tag), popular=out["task"].isin(POPULAR))


def _column(ax, g: pd.DataFrame, left: bool, fontsize: float) -> None:
    """Name the points of `g` in a column outside the axes (left or right), in
    the order of their y and spread just enough not to overlap, each name in
    its regime's colour with a leader to its point."""
    g = g.sort_values("r2_trajectory")
    lo, hi = ax.get_ylim()
    step = 1.2 * fontsize / 72 / (ax.get_position().height * ax.figure.get_figheight()) * (hi - lo)
    bottom, top = lo - .06 * (hi - lo), hi + .06 * (hi - lo)
    y = g["r2_trajectory"].to_numpy(float, copy=True)
    for i in range(len(y)):
        y[i] = max(y[i], y[i - 1] + step if i else bottom)
    for i in range(len(y) - 1, -1, -1):
        y[i] = min(y[i], y[i + 1] - step if i < len(y) - 1 else top)
    for r, ty in zip(g.itertuples(), y):
        ax.annotate(r.name, (r.r2_size, r.r2_trajectory), xytext=(-0.12 if left else 1.03, ty),
                    textcoords=("axes fraction", "data"), ha="right" if left else "left", va="center",
                    fontsize=fontsize, color=REGIME_INK[r.regime], annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=.4, alpha=.6, shrinkA=1, shrinkB=2,
                                    relpos=(1, .5) if left else (0, .5)))


def figure_tagged(a: pd.DataFrame, path: Path, paper: bool) -> None:
    """Panel (b) of the L1 accuracy figure, one panel per corpus stacked, every
    point named beside the panel: the points left of R² = 0.5 on the left, the
    rest on the right (the filtered tasks have no point)."""
    a = a[a["regime"] != FILTERED]
    fig, axes = plt.subplots(len(CORPORA), 1, figsize=(7.0, 9.0 if paper else 13.5), sharex=True)
    top = 0.97 if paper else G._header(
        fig, "Scaling regimes of the English-only cells on the English benchmarks, every task named",
        "panel (b) of the figure above, one panel per English corpus: point = one English accuracy task (original, RF or "
        f"LLM-RF) of the L1 deep seed-{GRID_SEED} cell, 90M–1.7B, above chance at ≥ 3 sizes (rule 1); R² of its final "
        f"score ~ log N fit against the median R² of score ~ log tokens over each run's checkpoints; quadrants split at "
        f"R² = {R2_SPLIT}, black edge and black name = ρ < 0 ('{DECLINES}'), a name's colour is its regime; the tasks the "
        "gate filters out have no point (the CSV lists them)") - 0.02
    fig.subplots_adjust(left=0.29, right=0.71, top=top, bottom=0.06, hspace=0.2)
    for ax, name, colour in zip(axes, CORPORA.values(), S.SERIES):
        g = a[a["corpus"] == name]
        _quadrants(ax, labels=False)
        ax.scatter(g["r2_size"], g["r2_trajectory"], **_edged(dict(s=12, color=colour, alpha=.9, zorder=3), g))
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        for left in (True, False):
            _column(ax, g[(g["r2_size"] < R2_SPLIT) == left], left, fontsize=6 if paper else 6.5)
        ax.grid(color=S.GRID, lw=.5); S.clean(ax)
        ax.set_title(f"{name} ({len(g)} tasks)", fontsize=8.5, loc="left")
    axes[-1].set_xlabel("R² of the model-size scaling fit")
    fig.supylabel("Median R² across training-trajectory fits", x=0.02, fontsize=8.5)
    if paper:
        S.save_paper(fig, path.with_suffix(""))
        return
    S.save(fig, path)


def generate_readme_tagged(a: pd.DataFrame, pool: str) -> None:
    """The block of the tagged figure: the regime of every well-known English
    benchmark per corpus, read from its table `a`."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    short = {**REGIME_SHORT, FILTERED: "filtered by the gate"}
    wide = a.pivot(index="task", columns="corpus", values="regime").reindex(columns=list(CORPORA.values()))
    names = a.drop_duplicates("task").set_index("task")["name"]

    def reading(task):
        r = wide.loc[task]
        if r.nunique() == 1:
            return f"{short[r.iloc[0]]} in all three"
        return ", ".join(f"{short[v]} ({c})" for c, v in r.items())
    pop = [k for k in POPULAR if k in wide.index]
    both = [k for k in pop if (wide.loc[k] == "predictable across both").all()]
    rest = [k for k in pop if k not in both]
    n = a[a["regime"] != FILTERED].groupby("corpus").size()
    nf = a[a["regime"] == FILTERED].groupby("corpus").size()
    body = "\n\n".join([
        f"![Scaling regimes at L1, every task named]({stage}/{pool}/{L1_TAGGED}.png)",
        f"Pool `{pool}`: the figure above with every point named, one panel per corpus (its panel (b)). The L1 deep "
        f"seed-{GRID_SEED} cell of each English corpus, 90M–1.7B, gate `{CANONICAL_POOL}`, English accuracy tasks "
        f"(drawn: {', '.join(f'{c} {n[c]}' for c in CORPORA.values())}). The table adds the English accuracy tasks the "
        f"gate filters out (regime `{FILTERED}`: above chance at fewer than three sizes, so no fit): "
        + ", ".join(f"{c} {nf.get(c, 0)}" for c in CORPORA.values()) + ". Regenerate with "
        f"`python analysis/rq13_english_only/english_regimes.py --pool {pool}`.",
        "Key findings (the well-known English benchmarks, per corpus):",
        "\n".join([f"- **Predictable across both in all three corpora**: " + ", ".join(names[k] for k in both) + "."]
                   + [f"- **{names[k]}**: {reading(k)}." for k in rest]),
        " · ".join(f"[{L1_TAGGED}.{x}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/"
                   f"analysis/{HERE.name}/{stage}/{pool}/{L1_TAGGED}.{x})" for x in ("png", "csv"))])
    replace_block(HERE / "README.md", "english-regimes-l1-tagged", body, f"english_regimes.py --pool {pool}")


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
    same_n = si.loc[list(CORPORA.values()), "n_tasks"].nunique() == 1
    rows = [f"- **{c}** ({int(si.at[c, 'n_tasks'])} tasks, {int((acc['corpus'] == c).sum())} accuracy-scored): "
            + ", ".join(f"{REGIME_SHORT[r]} {share(c, r)}" for r in REGIMES)
            + f". Median R² {med.at[c, 'r2_size']:.2f} across size and {med.at[c, 'r2_trajectory']:.2f} along training. "
            f"On the accuracy-scored tasks alone, both {(acc.loc[acc['corpus'] == c, 'regime'] == both).mean():.0%}."
            for c in CORPORA.values()]
    rows.append(f"- **The same single-fit reading at every other L** (the English tasks of the deep data-A cell, "
                + "–".join(dict.fromkeys(str(int(n)) for n in sorted(si.loc[by_L, "n_tasks"]))) + " tasks): predictable across both "
                f"{lo:.0%}–{hi:.0%} at L2–L50 ("
                + ", ".join(f"{c.split()[0]} {si.at[c, both]:.0%}" for c in by_L) + ").")
    rows.append(f"- **{BACKGROUND}** ({int(si.at[BACKGROUND, 'n_tasks'])} tasks, every trained language, medians over "
                "the L): " + ", ".join(f"{REGIME_SHORT[r]} {share(BACKGROUND, r)}" for r in REGIMES)
                + ". A median over six fits is smoother than a single fit, so this is the figure's backdrop. The "
                "bullet above is the like-for-like comparison.")
    body = "\n\n".join([
        f"![Scaling regimes at L1]({stage}/{pool}/english_only_scaling_regimes.png)",
        f"Pool `{pool}`: the L1 deep seed-{GRID_SEED} cell of each English corpus (DCLM-Edu = data A, DCLM = DCLMP, "
        f"FineWeb = FWEB), 90M–1.7B, gate `{CANONICAL_POOL}`. The fits and regimes are those of the scaling-predictability "
        "figure, with one size fit per task (a single L) and the median trajectory R² over ≥ 2 sizes. The tasks are the "
        "English tasks (originals, RF, LLM-RF, bBPB twins) and `bpb_dclm`. The corpora share the gate, "
        + ("and they read the same number of tasks." if same_n else "but their task counts differ (in brackets below).")
        + " rq01's grey points cover every trained language. Regenerate with "
        f"`python analysis/rq13_english_only/english_regimes.py --pool {pool}`.",
        "Key findings (share of tasks per regime, where both = predictable across size and along training):",
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
            + f". Median R² {med.at[c, 'r2_size']:.2f} across size and {med.at[c, 'r2_trajectory']:.2f} along training."
            for c in CORPORA.values()]
    by_fmt = a.assign(b=a["regime"] == both).groupby(["format", "corpus"])["b"].mean()
    rows.append("- **Predictable across both, per format**: " + "; ".join(
        f"{G._TWIN_NAMES.get(f, f)} ({n_f} task{'s' * (n_f != 1)}) " + ", ".join(f"{c} {by_fmt[(f, c)]:.0%}" for c in CORPORA.values())
        for f, n_f in fmt.items()) + ".")
    rows.append(f"- **Same regime in all three corpora**: {(agree.nunique(axis=1) == 1).sum()} of {len(agree)} tasks, "
                f"{(agree == both).all(axis=1).sum()} of them predictable across both.")
    body = "\n\n".join([
        f"![Scaling regimes at L1 on the English benchmarks]({stage}/{pool}/{L1_ACCURACY}.png)",
        f"Pool `{pool}`: the L1 deep seed-{GRID_SEED} cell of each English corpus, 90M–1.7B, gate `{CANONICAL_POOL}`, with "
        f"the fits and regimes of the figure above. Only English accuracy tasks are drawn ({next(iter(CORPORA.values()))}: "
        + ", ".join(f"{v} {G._TWIN_NAMES.get(k, k)}" for k, v in fmt.items()) + "): no bBPB twin, no `bpb_dclm` and no point from an L > 1 cell. "
        + ("The three corpora read the same tasks." if same else "The populations differ: the corpora read different tasks.")
        + f" Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool {pool}`.",
        "Key findings (share of tasks per regime, where both = predictable across size and along training):",
        "\n".join(rows),
        " · ".join(f"[{L1_ACCURACY}.{x}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/"
                   f"analysis/{HERE.name}/{stage}/{pool}/{L1_ACCURACY}.{x})" for x in ("png", "csv"))])
    replace_block(HERE / "README.md", "english-regimes-l1-accuracy", body, f"english_regimes.py --pool {pool}")


def main(pool: str) -> None:
    out_dir = HERE / load_pools()[pool].get("stage", "pretraining") / pool
    t, by_L, filtered = table(pool)
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
    at = tagged(a, filtered)
    for name in (L1_TAGGED, f"{L1_TAGGED}_paper"):
        at.to_csv(out_dir / f"{name}.csv", index=False)
        figure_tagged(at, out_dir / f"{name}.png", paper=name.endswith("_paper"))
    generate_readme_tagged(at, pool)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
