"""What leaving out the cells below a benchmark's token threshold does to its
aggregate score: the standard error of the aggregate, and the decision
accuracy of the ranking the aggregate makes.

A (task, language) cell is below threshold when the run's training tokens of
the language are below the pooled 50 % level of its benchmark family in
`pass_threshold_tokens_by_benchmark_1904_ckpts.csv` (tokens_seen.py, part C):
a family that `never` reaches the level loses every cell, one at 50 % from
its first bin (`<= lowest bin`) loses none. Per seed-1904 run at its final
checkpoint (pool `predictivity`, 90M–1.7B) and benchmark family, the
aggregate is the mean of the family's per-language parent tasks over the
languages the run trains (rule 2), under three policies:

  all       every trained-language cell, the excluded ones kept
  tokens    the cells at or above the family's threshold (the exclusion)
  gate      the cells above chance at the run's size and at 1.7B (rule 1), for comparison

  SE        the standard deviation of the per-language values over the square
            root of their count (two languages at least)
  DA-size   each proxy's aggregate against the 1.7B aggregate under the same
            policy, over the design pairs at the grid seed (`axes`: multi-axis
            and mono-axis, rule 15), a family cell needing MIN_PAIRS pairs (rule
            5), pooled over families as matching pairs over pairs.
            `reference` = `all cells` holds the 1.7B aggregate at every cell
            instead, so the reference ranking is the same for every policy.

!!! RULE 1: the `all` policy keeps the at-chance cells on purpose. It is the
counterfactual the exclusion is measured against. A pair that moves L compares
aggregates over two language sets, as an aggregate score does.

    token_exclusion_da_size_per_family_both_axes.csv  per (reference, policy, axes, size, benchmark):
                                                      DA-size, pairs, median SE and languages, share of cells kept
    token_exclusion_da_size_both_axes.csv             pooled over the families every policy scores (rule 13),
                                                      per (reference, policy, axes, size)
    token_exclusion_da_size_mono_axis.csv / .tex      the paper table: per size, the median SE and the
                                                      mono-axis DA-size of each policy, on those families

    python analysis/rq00_chance_vs_train_tokens/token_exclusion.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import CHANCE_VS_TRAIN_TOKENS  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import decisions, reliability  # noqa: E402
from analysis.utils import (  # noqa: E402
    MIN_PAIRS, PAIR_AXES, TARGET_SIZE, assign_language, benchmark_family, design_axes, finals, ladder_frame,
    language_tokens, pair_sets, passes_gate, size_order)

GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens"
THRESHOLDS = CHANCE_VS_TRAIN_TOKENS / "pretraining" / "predictivity_seeds" / "pass_threshold_tokens_by_benchmark_1904_ckpts.csv"
LEVEL = 0.5                                    # the threshold level: half of the family's cells above chance
POLICIES = ("all", "tokens", "gate")
REFERENCES = ("same policy", "all cells")
STEM = "token_exclusion_da_size"


def thresholds() -> pd.Series:
    """benchmark family -> the token count below which a cell is left out."""
    t = pd.read_csv(THRESHOLDS)
    t = t[(t["sizes"] == "all") & (t["level"] == LEVEL)].set_index("benchmark")
    return t["bin_lo"].where(t["status"] == "bin", np.where(t["status"] == "never", np.inf, 0.0))


def cells(pool: str) -> pd.DataFrame:
    """Every final (run, family, trained-language task) of a thresholded family,
    with the run's tokens of the language and the two keep flags."""
    thr = thresholds()
    fin = finals(ladder_frame(pool))
    fin = fin[fin["kind"] == "benchmark"].assign(benchmark=lambda d: d["task"].map(benchmark_family))
    fin = fin[fin["benchmark"].isin(thr.index)].assign(language=lambda d: d["task"].map(assign_language))
    tok = {k: language_tokens(*k) or {} for k in set(zip(fin["L"], fin["data"], fin["size"], fin["ladder"]))}
    fin["tokens"] = [tok[k].get(lang, np.nan) for k, lang in
                     zip(zip(fin["L"], fin["data"], fin["size"], fin["ladder"]), fin["language"])]
    fin["keep_all"] = True
    fin["keep_tokens"] = fin["tokens"] >= fin["benchmark"].map(thr)
    mask = load_mask(pool)
    fin["keep_gate"] = False
    for size, g in fin.groupby("size"):
        fin.loc[g.index, "keep_gate"] = passes_gate(mask, g["task"], size, TARGET_SIZE).to_numpy()
    return fin


def aggregates(fin: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, benchmark, run family, size): the aggregate, its SE, its languages."""
    out = []
    for pol in POLICIES:
        g = fin[fin[f"keep_{pol}"]]
        a = (g.groupby(["benchmark", "family", "size"])
             .agg(primary_score=("primary_score", "mean"), sd=("primary_score", "std"),
                  n_languages=("language", "nunique"), compute=("compute", "mean")).reset_index())
        a["se"] = (a["sd"] / np.sqrt(a["n_languages"])).where(a["n_languages"] >= 2)
        out.append(a.assign(policy=pol))
    return pd.concat(out, ignore_index=True)


def da_size(fin: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """DA-size of the aggregate ranking per (reference, policy, axes, size, benchmark)."""
    groups = {k: v for k, v in pair_sets(design_axes(fin)).items() if k in PAIR_AXES[:2]}
    proxies = [s for s in size_order(fin["size"].unique()) if s != TARGET_SIZE]
    as_tasks = agg.rename(columns={"benchmark": "task"}).assign(frac=1.0)
    out = []
    for ref in REFERENCES:
        for pol, a in as_tasks.groupby("policy"):
            truth = as_tasks[as_tasks["policy"] == ("all" if ref == "all cells" else pol)]
            c = reliability(decisions(a[a["size"] != TARGET_SIZE], groups, proxies, truth, fracs=(1.0,)))
            out.append(c.assign(reference=ref, policy=pol))
    d = pd.concat(out, ignore_index=True).rename(columns={"task": "benchmark", "group": "axes", "n_comparable": "n_pairs"})
    d.loc[d["n_pairs"] < MIN_PAIRS, "da"] = np.nan                                          # rule 5
    return d.drop(columns=["frac", "compute"])


def tables(fin: pd.DataFrame, agg: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The per-family table; the pooled one over the families every policy
    scores (rule 13); and the paper's: per size and policy, the median SE and
    the mono-axis DA-size against the same policy at the reference."""
    se = agg.groupby(["policy", "size", "benchmark"]).agg(se_median=("se", "median"),
                                                          n_languages_median=("n_languages", "median")).reset_index()
    kept = (fin.groupby(["size", "benchmark"])[["keep_tokens", "keep_gate"]].mean()
            .rename(columns=lambda c: c.replace("keep", "share_kept")).reset_index())
    per = d.merge(se, on=["policy", "size", "benchmark"], how="left").merge(kept, on=["size", "benchmark"], how="left")
    ok = per[per["reference"] == "same policy"].dropna(subset=["da"])
    fams = set.intersection(*[set(ok.loc[ok["policy"] == p, "benchmark"]) for p in POLICIES])
    sub = per[per["benchmark"].isin(fams) & per["da"].notna()]
    pooled = (sub.groupby(["reference", "policy", "axes", "size"])
              .agg(n_matching=("n_matching", "sum"), n_pairs=("n_pairs", "sum"), n_benchmarks=("benchmark", "nunique"))
              .reset_index())
    pooled["da"] = pooled["n_matching"] / pooled["n_pairs"]
    paper = (agg[agg["benchmark"].isin(fams)].groupby(["policy", "size"])
             .agg(se_median=("se", "median"), n_benchmarks=("benchmark", "nunique")).reset_index())
    paper["share_kept_tokens"] = paper["size"].map(fin[fin["benchmark"].isin(fams)].groupby("size")["keep_tokens"].mean())
    mono = pooled[(pooled["reference"] == "same policy") & (pooled["axes"] == "mono-axis")]
    paper = paper.merge(mono[["policy", "size", "da", "n_pairs"]], on=["policy", "size"], how="left")
    paper["axes"] = "mono-axis"
    return per, pooled, paper


def write_tex(paper: pd.DataFrame, path: Path) -> None:
    """Per size: median SE (all, tokens) and mono-axis DA-size (all, tokens, gate)."""
    def v(pol, size, col):
        r = paper[(paper["policy"] == pol) & (paper["size"] == size)][col].dropna()
        return f"{r.iloc[0]:.3f}" if len(r) else "--"
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{SE of the aggregate} & \\multicolumn{3}{c}{DA-size, single-axis pairs} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
             "Size & All cells & Above threshold & All cells & Above threshold & Above chance \\\\", "\\midrule"]
    for s in size_order(paper["size"].unique()):
        lines.append(" & ".join([s, v("all", s, "se_median"), v("tokens", s, "se_median")]
                                + [v(pol, s, "da") for pol in POLICIES]) + " \\\\")
    path.with_suffix(".tex").write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""]))


def generate_readme(pool: str, pooled: pd.DataFrame, paper: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    order = pooled.assign(o=pooled["size"].map(size_order(pooled["size"].unique()).index)).sort_values(["reference", "axes", "o"])
    for (ref, axes, size), g in order.groupby(["reference", "axes", "size"], sort=False):
        g = g.set_index("policy")
        rows.append([ref, axes, size] + [f"{g.at[pol, 'da']:.3f}" if pol in g.index else "—" for pol in POLICIES]
                    + [int(g["n_pairs"].min())])
    se_rows = [[s] + [f"{paper[(paper['policy'] == pol) & (paper['size'] == s)]['se_median'].iloc[0]:.4f}" for pol in POLICIES]
               + [f"{paper[paper['size'] == s]['share_kept_tokens'].iloc[0]:.2f}"] for s in size_order(paper["size"].unique())]
    rel = f"{stage}/{pool}"
    body = "\n\n".join([
        "## Aggregate scores with and without the cells below the token threshold",
        f"**DA-size of an aggregate ranking · pool `{pool}` (seed 1904), finals 90M–{TARGET_SIZE} · multi-axis and mono-axis "
        f"pairs · the {int(paper['n_benchmarks'].max())} benchmark families every policy scores · no filter.** Per run and "
        f"family, the aggregate is the mean of the family's per-language tasks over the languages the run trains: every "
        f"cell (`all`), the cells at or above the family's pooled {LEVEL:.0%} token threshold (`tokens`), or the cells "
        f"above chance at the size and at {TARGET_SIZE} (`gate`). A family that never reaches the threshold has no "
        f"`tokens` aggregate and is left out of every pooled row (rule 13). `reference` says whether the {TARGET_SIZE} "
        f"aggregate takes the same policy or every cell. The `all` policy keeps at-chance cells on purpose (rule 1): it "
        f"is the counterfactual. Regenerate with "
        f"`python analysis/rq00_chance_vs_train_tokens/token_exclusion.py --pool {pool}`.",
        md_table(["reference", "axes", "proxy", "DA all", "DA tokens", "DA gate", "pairs (min)"], rows),
        md_table(["size", "median SE all", "median SE tokens", "median SE gate", "cells kept by tokens"], se_rows),
        f"Files: [`{STEM}_both_axes.csv`]({GITHUB}/{rel}/{STEM}_both_axes.csv), "
        f"[`{STEM}_mono_axis.csv`]({GITHUB}/{rel}/{STEM}_mono_axis.csv) / [`.tex`]({GITHUB}/{rel}/{STEM}_mono_axis.tex) "
        f"(the paper table), [`{STEM}_per_family_both_axes.csv`]({GITHUB}/{rel}/{STEM}_per_family_both_axes.csv)."])
    replace_block(CHANCE_VS_TRAIN_TOKENS / "README.md", "token-exclusion", body, f"token_exclusion.py --pool {pool}")


def main(pool: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("!!! RULE 1: the `all` policy keeps at-chance cells on purpose; it is the counterfactual of the exclusion")
    fin = cells(pool)
    agg = aggregates(fin)
    per, pooled, paper = tables(fin, agg, da_size(fin, agg))
    per.to_csv(out_dir / f"{STEM}_per_family_both_axes.csv", index=False)
    pooled.to_csv(out_dir / f"{STEM}_both_axes.csv", index=False)
    paper.to_csv(out_dir / f"{STEM}_mono_axis.csv", index=False)
    write_tex(paper, out_dir / f"{STEM}_mono_axis")
    print(pooled.pivot_table(index=["reference", "axes", "size"], columns="policy", values="da").round(3).to_string())
    print(paper.pivot_table(index="size", columns="policy", values=["se_median", "da"]).round(4).to_string())
    if out_dir == CHANCE_VS_TRAIN_TOKENS / load_pools()[pool].get("stage", "pretraining") / pool:
        generate_readme(pool, pooled, paper)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    ap.add_argument("--out-dir", default=None, help="write elsewhere (a test run)")
    a = ap.parse_args()
    main(a.pool, Path(a.out_dir) if a.out_dir else CHANCE_VS_TRAIN_TOKENS / load_pools()[a.pool].get("stage", "pretraining") / a.pool)
