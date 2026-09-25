"""rq01 — tokens seen: is a language's evaluation a function of how much of
that language the proxy has trained on?

Both figures share one x axis: the training tokens of ONE language a checkpoint
has seen — the language's share of the mixture (L, scheme), from the build's own
plan (`utils.language_token_share`), times the cell's budget D(N), times the
checkpoint's share of the run. The share is a relabelling of the (L, size,
checkpoint) grid, not a new measurement, and the plan files live on capstor: a
cell whose share is unreachable carries NaN tokens, is left out of the figure
and is counted in the `!!!` line and the README.

A) da_goal_multi_axes_across_langs_bpb.png/.pdf/.csv, _cells.csv
   DA-goal of a language's BPB against the tokens of that language the proxies
   had seen. DA-goal = the share of pairs of design variants the proxy
   checkpoint orders like the reference's final (rq02's kernel; rule 15's
   multi-axis set: every pair at the grid seed of every scheme, the pair set of
   rq02's pooled panel). A BPB task is read only on the variants that train its
   language (rule 2, the loader), so the variants behind one cell differ in L,
   list and temperature and saw different amounts of the language: a cell's x is
   the checkpoint's share of the run times the MEAN over the pair set's proxy
   variants of their tokens of the language (the cells table keeps the min and
   max). A cell needs MIN_PAIRS pairs (rule 5). Per (proxy size, tenth of the
   run): the mean DA over languages, its standard error over languages and the
   geometric mean of their x — one line per proxy size, the reference's own line
   being its early checkpoints against its final.

B) pass_prob_vs_train_tokens_by_benchmark_<population>.png/.pdf/.csv, _points.csv
   Per benchmark and size: the share of its (language, L) cells that are above
   chance, against the tokens of the language the cell trained on, one line per
   size, the cells binned on log10 tokens (BINS_PER_DECADE per decade, the cell
   count on every point). The `.csv` is the cell table, `_points.csv` the binned
   values drawn. Two populations of runs behind a (task, size, L) cell:
     _deep_A_1904   the plan grid (deep, scheme A, seed 1904): one run per cell,
                    above chance = its one-sided 95 % Wilson lower bound clears
                    chance (rule 1's per-run test, `above_random.above_chance`)
     _1904          every seed-1904 run at the (size, L) that trains the language
                    (every scheme, both architectures): above chance when at least
                    MIN_SHARE of them are (rule 1's cell rule); the score and the
                    tokens are their means
   Only benchmarks with a chance level appear (BPB and the generative tasks
   cannot be gated). A `_ckpts` twin of each population reads the cells at
   every evaluated tenth of the run instead of the final alone: a cell is
   (task, size, L, tenth), above chance by the same rule on that checkpoint's
   scores, and its x the tokens of the language seen BY THAT CHECKPOINT (the
   tenth × the run's tokens) — ten times the cells and a token axis that runs
   through every training run, so the bins fill where the final-only version
   has one cell per (language, L, size).

    python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import MIN_SHARE, above_chance  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import _scores_at, compute_early_small_decision_accuracy  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import SAFE_DA  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, MIN_PAIRS, SHARED_FRACS, TARGET_SIZE, assign_language, at_fraction, benchmark_family, finals,
    ladder_frame, language_tokens, languages_only, size_order)

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
BINS_PER_DECADE = 3
DA_NAME = "da_goal_multi_axes_across_langs_bpb"
PASS_NAME = "pass_prob_vs_train_tokens_by_benchmark"
# population -> (the runs of a (task, size, L) cell, how the cell is called above chance)
POPULATIONS = {
    "deep_A_1904": ("the plan grid, deep / scheme A / seed 1904: one run per cell",
                    "the run's one-sided 95 % Wilson lower bound clears chance"),
    "1904": (f"every seed-{GRID_SEED} run at the (size, L) that trains the language, every scheme and architecture",
             f"at least {MIN_SHARE:.0%} of the cell's runs clear chance (Wilson lower bound); score and tokens are their means"),
}
mpl.rcParams.update(S.RC)


def _tokens(L, scheme, size, arch, lang) -> float:
    """Tokens of `lang` a full run of the cell trains on; NaN when the build's
    plan is unreachable. A trained language is always in the share, so a
    missing key is a bug and raises."""
    t = language_tokens(int(L), scheme, size, arch)
    return np.nan if t is None else t[lang]


# --- A: DA-goal of a language's BPB against the tokens of the language seen -------------

def da_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (BPB task, proxy size, tenth) with >= MIN_PAIRS pairs: DA-goal
    against the reference final over every design pair, and the tokens of the
    task's language the pair set's proxies had seen at that checkpoint (mean,
    min and max over the variants)."""
    bpb = df[df["kind"] == "bpb"].assign(language=lambda d: d["task"].map(assign_language))
    bpb = languages_only(bpb)
    attrs = bpb[["family", "L", "scheme", "arch"]].drop_duplicates().set_index("family")
    rows = []
    for task, dft in bpb.groupby("task", sort=False):
        lang = dft["language"].iloc[0]
        ref = set(_scores_at(dft, TARGET_SIZE, 1.0))
        for c in compute_early_small_decision_accuracy(dft, fracs=SHARED_FRACS):
            fams = sorted(set(_scores_at(dft, c["proxy_size"], c["frac"])) & ref)
            tok = np.array([c["frac"] * _tokens(attrs.at[f, "L"], attrs.at[f, "scheme"], c["proxy_size"], attrs.at[f, "arch"], lang)
                            for f in fams])
            rows.append({"task": task, "language": lang, "proxy_size": c["proxy_size"], "frac": c["frac"], "da": c["da"],
                         "n_pairs": c["n_pairs"], "n_variants": len(fams), "tokens": tok.mean(),
                         "tokens_min": tok.min(), "tokens_max": tok.max()})
    return pd.DataFrame(rows)


def da_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (proxy size, tenth): mean DA over the languages with a token count,
    its standard error over languages, the geometric mean of their tokens."""
    ok = cells.dropna(subset=["tokens"])
    return (ok.groupby(["proxy_size", "frac"])
            .agg(da=("da", "mean"), da_se=("da", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                 n_languages=("task", "nunique"), median_pairs=("n_pairs", "median"),
                 tokens=("tokens", lambda v: 10 ** np.log10(v).mean()))
            .reset_index())


def plot_da(summary: pd.DataFrame, cells: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for s in size_order(summary["proxy_size"].unique()):
        g = summary[summary["proxy_size"] == s].sort_values("frac")
        n = g["n_languages"]
        langs = f"{n.min()} languages" if n.min() == n.max() else f"{n.min()}–{n.max()} languages"
        ax.errorbar(g["tokens"], g["da"], yerr=g["da_se"].fillna(0), color=S.SIZE_COLOR.get(s, S.MUTED), marker="o",
                    ms=3.5, lw=1.4, capsize=2, label=f"{s}  ({langs})")
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("training tokens of the language seen by the proxy checkpoint (log)")
    ax.set_ylabel(f"mean DA-goal of the language's BPB vs the {TARGET_SIZE} final")
    ax.legend(frameon=False, loc="lower right", title="proxy size"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    top = G._header(fig, "Does a language's BPB rank the design variants like the reference once the proxy has seen enough of it?",
                    f"point = one proxy size at one tenth of its run: mean over languages of DA-goal (share of design-variant pairs "
                    f"the proxy's BPB of that language orders like the {TARGET_SIZE} final; every pair at seed {GRID_SEED} of every "
                    f"scheme, on the variants that train the language, ≥ {MIN_PAIRS} pairs), bar = standard error over languages; "
                    f"x = geometric mean over languages of the tokens of the language the pair set's proxies had seen (mean over "
                    f"variants); {cells.dropna(subset=['tokens'])['task'].nunique()} languages, dotted = {SAFE_DA}; the "
                    f"{TARGET_SIZE} line is its own early checkpoints against its final")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, DA_NAME)


# --- B: share of a benchmark's cells above chance against the tokens of the language -------------

def gate_cells(fin: pd.DataFrame, ckpts: bool = False) -> pd.DataFrame:
    """One row per (task, size, L) of the gateable benchmarks — per (task,
    size, L, tenth) with `ckpts`, `fin` then holding the scores at the tenths:
    the runs' mean score, the share of them above chance (rule 1's per-run
    test), the verdict at MIN_SHARE, and the tokens of the task's language they
    had trained on (the tenth's share of the run's tokens with `ckpts`; mean
    over the runs; NaN when any run's build plan is unreachable)."""
    fin = fin[fin["kind"] == "benchmark"].copy()
    fin["above"] = above_chance(fin["primary_score"].to_numpy(), fin["task"].to_numpy()).to_numpy()
    fin = fin.dropna(subset=["above"])            # no chance level or item count: cannot be gated
    fin["language"] = fin["task"].map(assign_language)
    fin = languages_only(fin)
    fin["tokens"] = [_tokens(L, s, size, a, lang)
                     for L, s, size, a, lang in zip(fin["L"], fin["scheme"], fin["size"], fin["arch"], fin["language"])]
    keys = ["task", "language", "size", "L"]
    if ckpts:
        fin["tokens"] *= fin["frac"]
        keys.append("frac")
    cells = (fin.groupby(keys)
             .agg(task_score=("primary_score", "mean"), share_above=("above", "mean"), n_runs=("model", "nunique"),
                  train_tokens=("tokens", "mean"), missing=("tokens", lambda v: v.isna().any()))
             .reset_index())
    cells.loc[cells["missing"], "train_tokens"] = np.nan
    cells["above_chance"] = cells["share_above"] >= MIN_SHARE
    cells["benchmark"] = cells["task"].map(benchmark_family)
    cells["language_scheme"] = "L" + cells["L"].astype(int).astype(str)
    return cells.rename(columns={"size": "model_size"})[
        ["benchmark", "task", "language", "model_size", "language_scheme"] + (["frac"] if ckpts else [])
        + ["train_tokens", "task_score", "above_chance", "share_above", "n_runs"]]


def bin_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, size, log-token bin): the share of the cells above chance,
    their count and the geometric mean of their tokens."""
    c = cells.dropna(subset=["train_tokens"]).copy()
    c["bin"] = np.floor(np.log10(c["train_tokens"]) * BINS_PER_DECADE).astype(int)
    out = (c.groupby(["benchmark", "model_size", "bin"])
           .agg(share_above=("above_chance", "mean"), n_cells=("task", "size"),
                tokens=("train_tokens", lambda v: 10 ** np.log10(v).mean()))
           .reset_index())
    out["bin_lo"], out["bin_hi"] = 10 ** (out["bin"] / BINS_PER_DECADE), 10 ** ((out["bin"] + 1) / BINS_PER_DECADE)
    return out


def plot_pass(points: pd.DataFrame, cells: pd.DataFrame, out_dir: Path, population: str, ckpts: bool = False) -> None:
    fams = G.panel_order(points["benchmark"].unique())
    sizes = size_order(points["model_size"].unique())
    ncols = 5
    nrows = -(-len(fams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows + 1.0), sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for i, (ax, fam) in enumerate(zip(flat, fams)):
        g0 = points[points["benchmark"] == fam]
        for s in sizes:
            g = g0[g0["model_size"] == s].sort_values("tokens")
            ax.plot(g["tokens"], g["share_above"], color=S.SIZE_COLOR.get(s, S.MUTED), marker="o", ms=3, lw=1.2)
            for x, y, n in zip(g["tokens"], g["share_above"], g["n_cells"]):
                ax.annotate(str(n), (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=4.5, color=S.MUTED)
        nc = cells[cells["benchmark"] == fam]
        ax.set_title(f"{G.display(fam)}  ({nc['language'].nunique()} languages, {len(nc)} cells)", loc="left", fontsize=8)
        ax.set_xscale("log"); ax.set_ylim(-0.04, 1.12); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        if i + ncols >= len(fams):                # the last panel of its column carries the x label
            ax.set_xlabel("training tokens of the task's language (log)"); ax.tick_params(labelbottom=True)
    for ax in flat[len(fams):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("cells above chance")
    flat[0].legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR.get(s, S.MUTED), lw=2, label=s) for s in sizes],
                   fontsize=6.5, frameon=False, loc="upper left", title="model size", title_fontsize=6.5)
    runs, verdict = POPULATIONS[population]
    stem = f"{PASS_NAME}_{population}" + ("_ckpts" if ckpts else "")
    unit, where = (("(task, size, L, tenth of the run)", "that checkpoint had seen (the tenth × the run's tokens)") if ckpts
                   else ("(task, size, L)", "a full run of the cell trains on"))
    top = G._header(fig, f"Share of a benchmark's {'(language, L, checkpoint)' if ckpts else '(language, L)'} cells above chance "
                    f"against the tokens of the language seen — {population}{' at the ten evaluated checkpoints' if ckpts else ''}",
                    f"cell = one {unit}: {runs}; above chance = {verdict}; x = the tokens of the task's language {where} "
                    f"(its share of the mixture × D(N)), cells binned {BINS_PER_DECADE} per decade, point = share "
                    f"of the bin's cells above chance, small number = cells in the bin, x = their geometric mean; one line per size; "
                    f"benchmarks with a chance level only, trained languages (rule 2), parent tasks (rule 6); "
                    f"{cells.dropna(subset=['train_tokens'])['task'].nunique()} tasks, {cells['language'].nunique()} languages")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, stem)


# --- README ----------------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, summary: pd.DataFrame, cells_a: pd.DataFrame,
                    cells_b: dict[str, pd.DataFrame]) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    sizes = size_order(summary["proxy_size"].unique())
    at = lambda s, f: summary[(summary["proxy_size"] == s) & (np.isclose(summary["frac"], f))].iloc[0]  # noqa: E731
    rows = [[s, f"{at(s, .2)['tokens'] / 1e9:.2f} B", fmt(at(s, .2)["da"]), f"{at(s, 1.0)['tokens'] / 1e9:.2f} B",
             fmt(at(s, 1.0)["da"]), int(at(s, 1.0)["n_languages"])]
            for s in sizes if s != TARGET_SIZE and not summary[(summary["proxy_size"] == s) & np.isclose(summary["frac"], 1.0)].empty]
    missing_a = int(cells_a["tokens"].isna().sum())
    missing_b = {p: int(c["train_tokens"].isna().sum()) for p, c in cells_b.items()}
    unreachable = ("" if not missing_a and not any(missing_b.values()) else
                   f"\n\n!!! Token counts unreachable (the build's plan is not readable from where this ran): {missing_a} cells of "
                   f"the DA figure, " + ", ".join(f"{n} of `{p}`" for p, n in missing_b.items()) + " — those cells are left out of "
                   "the figures and carry NaN in the tables.")
    body = "\n\n".join([
        "## Tokens seen: exposure to a language against its evaluation",
        f"`tokens_seen.py`, `{pool}` pool: both figures put a language's evaluation against the training tokens of that language "
        f"the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the "
        f"run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool {pool}`." + unreachable,
        f"**A. DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages "
        f"of the share of design-variant pairs the proxy's BPB orders like the {TARGET_SIZE} final (rq02's kernel, every pair at "
        f"seed {GRID_SEED} of every scheme on the variants that train the language, ≥ {MIN_PAIRS} pairs; rule 15's multi-axis "
        f"set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of "
        f"the language they had seen, and a point's x the geometric mean over languages "
        f"({cells_a.dropna(subset=['tokens'])['task'].nunique()} languages; `{DA_NAME}_cells.csv` has the per-language cells "
        f"with the min and max over variants). The {TARGET_SIZE} line is its own early checkpoints against its final.",
        md_table(["proxy size", "tokens of a language at 1C", "DA at 1C", "tokens at 5C", "DA at 5C", "languages"], rows),
        f"![DA-goal of BPB vs tokens seen]({rel}/{DA_NAME}.png)",
        f"**B. Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, "
        f"language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned {BINS_PER_DECADE} "
        f"per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two "
        f"populations: `deep_A_1904`, {POPULATIONS['deep_A_1904'][0]}; `1904`, {POPULATIONS['1904'][0]}. The `.csv` next to each "
        f"figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, "
        f"above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at "
        f"every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, "
        f"and a token axis that runs through every training run. Cells above chance: "
        + ", ".join(f"`{p}` {int(c['above_chance'].sum())} of {len(c)} ({c['task'].nunique()} tasks)" for p, c in cells_b.items()) + ".",
    ] + [f"![Share above chance vs tokens seen, {p}]({rel}/{PASS_NAME}_{p}.png)" for p in cells_b])
    replace_block(OUT_ROOT / "README.md", "tokens-seen", body, f"tokens_seen.py --pool {pool}")
    print(f"Wrote auto README block → {OUT_ROOT / 'README.md'}")


# --- driver ------------------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    df = df[df["seed"] == GRID_SEED].copy()       # the grid seed: a replicate is a draw of one design, not a second design
    df["bucket"] = df["size"].map(size_bucket)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}' at seed {GRID_SEED}: {df['model'].nunique()} cells")

    cells = da_cells(df)
    summary = da_summary(cells)
    cells.to_csv(out_dir / f"{DA_NAME}_cells.csv", index=False)
    summary.to_csv(out_dir / f"{DA_NAME}.csv", index=False)
    missing = cells[cells["tokens"].isna()]
    print(f"A: {len(cells)} (language, size, tenth) cells over {cells['task'].nunique()} languages with ≥ {MIN_PAIRS} pairs "
          f"(rule 5; the kernel leaves the others out), {cells['n_variants'].min()}–{cells['n_variants'].max()} variants per cell")
    if len(missing):
        print(f"!!! A: {len(missing)} cells over {missing['task'].nunique()} languages have no token count (build plan unreachable) "
              f"and are left out of the figure")
    if not summary.empty:
        plot_da(summary, cells, out_dir)

    fin = finals(df)
    # the same runs at every evaluated tenth: the `_ckpts` twins (`frac` = the tenth asked for)
    tenths = pd.concat([at_fraction(df, f).assign(frac=f) for f in SHARED_FRACS], ignore_index=True)
    cells_b = {}
    for population in POPULATIONS:
        for ckpts, frame in ((False, fin), (True, tenths)):
            sub = frame[(frame["arch"] == "deep") & (frame["scheme"] == "A")] if population == "deep_A_1904" else frame
            stem = f"{PASS_NAME}_{population}" + ("_ckpts" if ckpts else "")
            c = gate_cells(sub, ckpts)
            c.to_csv(out_dir / f"{stem}.csv", index=False)
            points = bin_cells(c)
            points.to_csv(out_dir / f"{stem}_points.csv", index=False)
            cells_b[stem.removeprefix(PASS_NAME + "_")] = c
            n_missing = int(c["train_tokens"].isna().sum())
            print(f"B [{population}{', tenths' if ckpts else ''}]: {len(c)} cells over {c['task'].nunique()} tasks, "
                  f"{c['benchmark'].nunique()} benchmarks, {c['language'].nunique()} languages; {int(c['above_chance'].sum())} above chance"
                  + (f"; !!! {n_missing} cells without a token count (build plan unreachable), left out of the figure" if n_missing else ""))
            if not points.empty:
                plot_pass(points, c, out_dir, population, ckpts)
    generate_readme(pool, out_dir, summary, cells, cells_b)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL, help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
