"""Above chance on languages the model never trained: does training a language
help a model clear chance on the untrained languages of the same family?

No decision accuracy here, only the gate's run-level verdict (rule 1,
`above_random.above_chance`: the one-sided 95 % Wilson lower bound of the run's
final accuracy clears the task's chance level), read from the gate's own table
`rq00_gate_and_curves/.../above_random_runs.csv` (pool `predictivity`, seed
1904, final checkpoints, accuracy-scored parent tasks).

Cells: the L1, L2 and L8 settings of the deep ladder, every data build the
launcher has there (`DATA_SCHEMES`: L1 A / DCLMP / FWEB, L2 A / ZH / ES, L8 A /
B). A cell's trained languages are `launch_trainings.cell_languages`; every
other language is untrained. Only the cells evaluated beyond their own
languages have untrained-language results: the scheme-A deep seed-1904 cells
(`auto_evals_cscs.ALL_LANGUAGES_RUNS`, every size but the L1 350M cell, whose
evaluation is incomplete) and the scheme-B L8 deep cell at 350M, 600M and 1B. The
DCLMP, FWEB, ZH and ES cells have none, so they are named and not drawn.

A language is in the trained languages' family when `configs/languages.json`
gives it the family of a trained NON-English language of the cell, at two
levels: the top level (the part before the comma, "Indo-European") and the
subfamily (the full string, "Indo-European, Slavic"; a family without a comma is
its own subfamily). An isolate (Basque) is a family of its own and never matches. L1 trains English alone, so every
untrained language is "other family" there: it is the baseline. Languages
without a family in languages.json are left out (and counted).

The measure is the share of a cell's untrained-language tasks above chance,
per size, split by family. Benchmarks differ between the two groups, so each
panel also draws the English-only L1 cell on exactly the same tasks: the gap to
that line is what training the family's language adds.

    above_chance_untrained_by_family.png / .csv   (+ _paper) a row per cell (L2 A, L8 A, L8 B),
                                                  a column per family level
    above_chance_untrained_lift_by_subfamily.png / .csv (+ _paper) per subfamily and cell, the share above
                                                  chance minus the L1 A cell's on the same (task, size), over the
                                                  sizes both have; filled = a trained language of the cell is in it;
                                                  drawn on >= MIN_VERDICTS verdicts
    above_chance_untrained_by_family_per_task.csv every (cell, size, task) verdict with its groups

    python analysis/rq06_language_transfer/family_transfer.py --pool predictivity
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
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.utils import ANALYSIS_SIZES, GRID_SEED, assign_language, benchmark_family  # noqa: E402
from pretrain.launch_trainings import DATA_SCHEMES, cell_languages  # noqa: E402
from pretrain.pretrain_progress import NAME_RE, SCHEME_OF_LABEL  # noqa: E402

HERE = Path(__file__).resolve().parent
SETTINGS = (1, 2, 8)
BASELINE = (1, "A")                 # English only, evaluated on every language
LEVELS = {"top": "Top-level family", "sub": "Subfamily"}
MIN_VERDICTS = 20                   # a (cell, subfamily) lift on fewer verdicts stays in the CSV, not in the figure
LANGS = json.loads((_SRC.parent / "configs" / "languages.json").read_text())["languages"]
mpl.rcParams.update(S.RC)


def family(lang: str, level: str) -> str | None:
    f = LANGS.get(lang, {}).get("family")
    if f is None:
        return None
    if f == "Isolate":                   # a family of its own: never the family of another language
        return f"Isolate ({lang})"
    return f.split(",")[0].strip() if level == "top" else f


def verdicts(pool: str) -> tuple[pd.DataFrame, list]:
    """Every untrained-language accuracy verdict of the deep seed-1904 L1/L2/L8 cells, with the
    family groups; and the cells that have none."""
    runs = pd.read_csv(GATE_AND_CURVES / load_pools()[pool].get("stage", "pretraining") / pool / "above_random_runs.csv")
    runs = runs[runs["above"].notna()]
    m = runs["model"].str.extract(NAME_RE)
    runs = runs.assign(size=m["size"], L=pd.to_numeric(m["L"]), data=m["scheme"].fillna("").map(SCHEME_OF_LABEL),
                       ladder=m["ladder"], seed=pd.to_numeric(m["seed"]), language=runs["task"].map(assign_language))
    runs = runs[(runs["ladder"] == "deep") & (runs["seed"] == GRID_SEED) & runs["L"].isin(SETTINGS)
                & runs["size"].isin(ANALYSIS_SIZES) & ~runs["language"].isin(["multi", "??"])]
    cells = [(L, d) for d, v in DATA_SCHEMES.items() for L in SETTINGS if L in v["langs"] and v["temp"] == 1]
    rows, empty = [], []
    for L, d in cells:
        trained = cell_languages(L, d)
        g = runs[(runs["L"] == L) & (runs["data"] == d) & ~runs["language"].isin(trained)]
        if g.empty:
            empty.append(f"L{L} {d}")
            continue
        g = g.assign(cell=f"L{L} {d}", cell_L=L, cell_data=d)
        for level in LEVELS:
            fams = {family(l, level) for l in trained - {"en"}} - {None}
            g[f"family_{level}"] = g["language"].map(lambda l: family(l, level))
            g[f"same_{level}"] = g[f"family_{level}"].isin(fams)
        rows.append(g)
    return pd.concat(rows, ignore_index=True), empty


def shares(v: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, level, group, size): the cell's share above chance, and the L1 English-only cell's on
    exactly the same tasks."""
    base = v[(v["cell_L"] == BASELINE[0]) & (v["cell_data"] == BASELINE[1])].set_index(["size", "task"])["above"]
    out = []
    for cell, g in v.groupby("cell"):
        if (g["cell_L"].iloc[0], g["cell_data"].iloc[0]) == BASELINE:
            continue
        g = g[g["family_top"].notna()]
        for level in LEVELS:
            for same, h in g.groupby(f"same_{level}"):
                for size, k in h.groupby("size"):
                    b = base.reindex(pd.MultiIndex.from_arrays([k["size"], k["task"]]))
                    out.append({"cell": cell, "level": level, "group": "same family" if same else "other family",
                                "size": size, "share_above": k["above"].mean(), "n_tasks": len(k),
                                "n_languages": k["language"].nunique(), "baseline_share_above": b.mean(),
                                "baseline_n_tasks": int(b.notna().sum()),
                                "languages": " ".join(sorted(k["language"].unique()))})
    t = pd.DataFrame(out)
    t["lift"] = t["share_above"] - t["baseline_share_above"]
    return t


def cell_label(cell: str, paper: bool) -> str:
    """`L8 A` as the README writes it, `K = 8, scheme A` in a paper figure (the paper's K)."""
    L, data = cell.split()
    return f"K = {L[1:]}, scheme {data}" if paper else cell


def figure(t: pd.DataFrame, path: Path, empty: list, paper: bool) -> None:
    cells = sorted(t["cell"].unique(), key=lambda c: (int(c.split()[0][1:]), c))
    sizes = [s for s in ANALYSIS_SIZES if s in set(t["size"])]
    fig, axes = plt.subplots(len(cells), 2, figsize=(6.6, 1.9 * len(cells) + 0.6), sharex=True, sharey=True, squeeze=False)
    colours = {"same family": S.SERIES[0], "other family": S.SERIES[1]}
    for i, cell in enumerate(cells):
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            for group, col in colours.items():
                g = t[(t["cell"] == cell) & (t["level"] == level) & (t["group"] == group)].set_index("size").reindex(sizes)
                if g["share_above"].notna().sum() == 0:
                    continue
                n = int(g["n_languages"].max())
                ax.plot(range(len(sizes)), g["share_above"], "-o", color=col, ms=3.5, lw=1.4,
                        label=f"{group.capitalize()} ({n} languages)")
                ax.plot(range(len(sizes)), g["baseline_share_above"], ":", color=col, lw=1.1)
            ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes)
            ax.set_ylim(0, 0.75); ax.grid(color=S.GRID, lw=.5); S.clean(ax)
            ax.plot([], [], ":", color=S.MUTED, lw=1.1, label="English only, same tasks")
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
            if i == 0:
                ax.set_title(LEVELS[level], fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{cell_label(cell, paper)}\nShare above chance", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Model size")
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    top = G._header(fig, "Above chance on untrained languages, by whether a trained language shares their family",
                    "share of the cell's untrained-language accuracy tasks whose final run clears chance (rule 1, the gate's "
                    "run-level verdict), seed 1904, deep; same family = the language shares the top-level family / the "
                    "subfamily of a trained non-English language (configs/languages.json); dotted = the English-only L1 A "
                    "cell on exactly the same tasks; no untrained-language results for " + ", ".join(empty))
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def lift_by_subfamily(v: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, subfamily): the share above chance and the L1 A cell's on the same (size, task) pairs,
    over every size both have, and its lift."""
    base = v[(v["cell_L"] == BASELINE[0]) & (v["cell_data"] == BASELINE[1])].set_index(["size", "task"])["above"]
    g = v[(v["cell"] != f"L{BASELINE[0]} {BASELINE[1]}") & v["family_sub"].notna()].copy()
    g["baseline"] = base.reindex(pd.MultiIndex.from_arrays([g["size"], g["task"]])).to_numpy()
    g = g.dropna(subset=["baseline"])
    t = (g.groupby(["cell", "family_sub", "same_sub"])
         .agg(n_verdicts=("above", "size"), n_languages=("language", "nunique"), share_above=("above", "mean"),
              baseline_share_above=("baseline", "mean"), sizes=("size", lambda x: " ".join(sorted(set(x), key=ANALYSIS_SIZES.index))),
              languages=("language", lambda x: " ".join(sorted(set(x)))))
         .reset_index())
    t["lift"] = t["share_above"] - t["baseline_share_above"]
    return t


def figure_lift(t: pd.DataFrame, path: Path, paper: bool) -> None:
    """Rows = subfamilies (ordered by their mean lift), x = lift over L1 A; colour = cell, filled = trained;
    only the (cell, subfamily) lifts on at least MIN_VERDICTS verdicts."""
    t = t[t["n_verdicts"] >= MIN_VERDICTS]
    order = t.groupby("family_sub")["lift"].mean().sort_values().index.tolist()
    cells = sorted(t["cell"].unique(), key=lambda c: (int(c.split()[0][1:]), c))
    colours = dict(zip(cells, S.SERIES))
    fig, ax = plt.subplots(figsize=(4.6, 0.24 * len(order) + 1.0))
    off = dict(zip(cells, np.linspace(-0.22, 0.22, len(cells))))
    for c in cells:
        g = t[t["cell"] == c].set_index("family_sub").reindex(order)
        y = np.arange(len(order)) + off[c]
        filled = g["same_sub"].fillna(False).astype(bool).to_numpy()
        ax.scatter(g["lift"][filled], y[filled], s=22, color=colours[c], zorder=3,
                   label=f"{cell_label(c, paper)}, trained subfamily")
        ax.scatter(g["lift"][~filled], y[~filled], s=16, facecolor="white", edgecolor=colours[c], lw=.8, zorder=2,
                   label=f"{cell_label(c, paper)}, other subfamily")
    ax.axvline(0, color=S.MUTED, lw=.7, ls=":")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Share above chance minus English only")
    ax.grid(axis="x", color=S.GRID, lw=.5); S.clean(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.1),
              ncol=2)
    if paper:
        fig.tight_layout()
        S.save_paper(fig, path.with_suffix(""))
        return
    top = G._header(fig, "What training a language adds on the untrained languages of its subfamily",
                    "per (cell, subfamily): the cell's share of untrained-language accuracy tasks above chance (rule 1) "
                    "minus the English-only L1 A cell's on the same (size, task) verdicts, pooled over the sizes both "
                    f"have; filled = the cell trains a non-English language of that subfamily; drawn where >= {MIN_VERDICTS} "
                    "verdicts")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path)


def generate_readme(pool: str, t: pd.DataFrame, v: pd.DataFrame, empty: list) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    nofam = sorted(set(v.loc[v["family_top"].isna(), "language"]))
    lines = []
    for (cell, level), g in t.groupby(["cell", "level"]):
        p = g.pivot(index="size", columns="group", values="lift").reindex([s for s in ANALYSIS_SIZES if s in set(g["size"])])
        s = g.pivot(index="size", columns="group", values="share_above").reindex(p.index)
        lines.append(f"- **{cell}, {LEVELS[level].lower()}**: share above chance, same family "
                     + ", ".join(f"{z} {s.at[z, 'same family']:.2f}" for z in p.index)
                     + "; other family " + ", ".join(f"{z} {s.at[z, 'other family']:.2f}" for z in p.index)
                     + ". Lift over L1 A on the same tasks: same family " + ", ".join(f"{p.at[z, 'same family']:+.2f}" for z in p.index)
                     + ", other family " + ", ".join(f"{p.at[z, 'other family']:+.2f}" for z in p.index) + ".")
    body = "\n\n".join([
        "## Above chance on untrained languages, by language family",
        f"Pool `{pool}`, the gate's run-level verdicts (rule 1) at the final checkpoint, deep seed-{GRID_SEED} cells at "
        "L1, L2 and L8, accuracy-scored parent tasks in languages the cell does not train; same family = the language "
        "shares the top-level family or the subfamily (`configs/languages.json` `family`) of a trained non-English "
        "language. Untrained-language results exist for the scheme-A cells (every size but L1 350M) and the L8 scheme-B "
        f"cell (350M–1B) only: none for {', '.join(empty)}. Left out: {len(nofam)} languages without a family in "
        f"languages.json. Regenerate with `python analysis/rq06_language_transfer/family_transfer.py --pool {pool}`.",
        f"![Above chance on untrained languages by family]({stage}/{pool}/above_chance_untrained_by_family.png)",
        f"![Lift over English only by subfamily]({stage}/{pool}/above_chance_untrained_lift_by_subfamily.png)",
        "Numbers (generated):", "\n".join(lines),
        "GitHub: " + " · ".join(
            f"[{n}](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/"
            f"rq06_language_transfer/{stage}/{pool}/{n})"
            for n in ("above_chance_untrained_by_family.png", "above_chance_untrained_by_family.csv",
                      "above_chance_untrained_lift_by_subfamily.png", "above_chance_untrained_lift_by_subfamily.csv",
                      "above_chance_untrained_by_family_per_task.csv"))])
    replace_block(HERE / "README.md", "family-transfer", body, f"family_transfer.py --pool {pool}")


def main(pool: str) -> None:
    out_dir = HERE / load_pools()[pool].get("stage", "pretraining") / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    v, empty = verdicts(pool)
    print(f"no untrained-language results: {', '.join(empty)}")
    v.drop(columns=["model"]).assign(family=v["task"].map(benchmark_family)).to_csv(
        out_dir / "above_chance_untrained_by_family_per_task.csv", index=False)
    t = shares(v)
    for name in ("above_chance_untrained_by_family", "above_chance_untrained_by_family_paper"):
        t.to_csv(out_dir / f"{name}.csv", index=False)
        figure(t, out_dir / f"{name}.png", empty, paper=name.endswith("_paper"))
    print(t.drop(columns="languages").round(3).to_string(index=False))
    lift = lift_by_subfamily(v)
    for name in ("above_chance_untrained_lift_by_subfamily", "above_chance_untrained_lift_by_subfamily_paper"):
        lift.to_csv(out_dir / f"{name}.csv", index=False)
        figure_lift(lift, out_dir / f"{name}.png", paper=name.endswith("_paper"))
    print(lift.drop(columns=["languages"]).round(3).to_string(index=False))
    generate_readme(pool, t, v, empty)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
