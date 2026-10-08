"""Which proxy disagreements with the reference are seed noise, and which are opposite effects?

DA-size counts a (pair, task) comparison as a miss when the proxy's score
difference between the two design variants has the other sign than the 1.7B
reference's. This script splits every comparison of a proxy against the
reference on two questions, per design axis:

    (a) |proxy difference| > K x the seed sd of the proxy's final score
        (the two variants are distinguishable at the proxy), K in (1, sqrt 2);
    (b) sign(proxy difference) != sign(1.7B difference) (a DA-size miss).

(a) without (b): the proxy separates the variants the reference's way.
(b) without (a): the proxy cannot tell the variants apart and the miss is a coin
flip. (a) with (b): the design choice has opposite measured effects at the two
sizes; `n_a_b_ref_clear` counts those whose 1.7B difference also clears the
same threshold (the reference has no replicate seed, so the proxy's sd stands
in for it).

The seed sd is rq05's `seed_sd` (per task, the median over the baseline deep
data-A cells with replicate seeds of the sample sd of the final score, read
from `predictivity_seeds`) on two scopes: `size`, the proxy's own x3 cells
(175M, 600M and 1B have them, at L1, L2, L50 for 175M/600M and L1, L2, L30 for
1B; 90M and 350M have none and take the pooled sd, `sd_source` says so), and
`pooled`, every x3 cell from 175M to 1B (rq05's and rq02_decisive_pairs'
convention). A per-language BPB task at 600M has its sd from the L50 cell
alone, except English (L1, L2, L50) and Russian (L2, L50).

Pairs and scores are rq02's (`compute_da.score_cube`, pool `predictivity`, seed
1904, final checkpoint), so the (b) share is 1 - DA-size pooled over
comparisons (rq02 reports per-language BPB at 600M as 0.74, the mean over
tasks, and 0.785 pooled). Mono-axis pairs move one design axis
(`utils.pair_sets`, `moved_axes`); the multi-axis rows pool every design pair,
and split it by whether the pair moves the depth. Rule 1 at the proxy and at
the reference (benchmarks), rule 2 by the loader (a language counts for a pair
only where both variants train it), rule 5 (a row over fewer than MIN_PAIRS
design pairs carries its counts but no shares). The 600M depth pairs are not
size-matched (`utils.size_matched`): the `size_matched` column is False on
every row that holds one.

    flip_attribution_da_size_by_axis_both_axes.csv   per (population, sd scope, K, proxy size, pair group)
    flip_attribution_da_size_by_axis_both_axes.tex   the 600M per-language BPB rows, booktabs, no caption
    flip_attribution_da_size_depth_by_seed_mono_axis.csv   the depth decision on per-language BPB with the deep
                                                     side at each of its replicate seeds (175M, 600M, 1B)
    python analysis/rq03_noise_and_snr/flip_attribution.py --pool predictivity
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
from analysis import grids as G  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import NOISE_AND_SNR  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import pair_agree, score_cube  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import AXIS_LABEL  # noqa: E402
from analysis.rq05_design_decisions.analyze import seed_sd  # noqa: E402
from analysis.utils import (  # noqa: E402
    DESIGN_AXES, GRID_SEED, MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, design_axes, finals, ladder_frame, moved_axes, pair_sets,
    passes_gate, size_matched, variant)

SEED_POOL = "predictivity_seeds"     # the replicate seeds the sd is read from
KS = {"1 sd": 1.0, "sqrt2 sd": float(np.sqrt(2))}
FOCUS = "600M"                       # the proxy the question is about
NAME = "flip_attribution_da_size_by_axis_both_axes"
SEED_NAME = "flip_attribution_da_size_depth_by_seed_mono_axis"


def _populations(tasks: list[str]) -> pd.Series:
    """task index -> population: the per-language BPB tasks (the 50 trained
    languages; `bpb_macro` and the loss are not a language, rule 7) and the
    benchmark tasks scored by accuracy (originals, rf and rfgm twins; the bBPB
    twins are left out, they are a BPB of their own)."""
    meta = G.add_meta(pd.DataFrame({"task": tasks}))
    meta = meta[meta["family"] != "loss"]
    acc = meta["task"].map(lambda t: variant(t)[1] == "acc")
    pop = pd.Series(np.where(meta["family"] == "bpb", "bpb", np.where(acc, "accuracy", "")), index=meta.index)
    return pop[pop != ""]


def _groups(attrs: pd.DataFrame) -> tuple[dict[tuple[str, str], list], dict]:
    """(axes, group label) -> design pairs: one group per mono-axis axis, every
    mono-axis pair, every multi-axis pair, and the multi-axis pairs split by
    whether they move the depth; and pair -> the axes it moves."""
    ps = pair_sets(attrs)
    moved = {p: moved_axes(attrs.loc[p[0]], attrs.loc[p[1]]) for p in ps["multi-axis"]}
    out = {}
    for ax in DESIGN_AXES:
        pairs = [p for p in ps["mono-axis"] if moved[p] == [ax]]
        if pairs:
            out[("mono-axis", ax)] = pairs
    out[("mono-axis", "all")] = ps["mono-axis"]
    out[("multi-axis", "all")] = ps["multi-axis"]
    out[("multi-axis", "moves arch")] = [p for p in ps["multi-axis"] if "arch" in moved[p]]
    out[("multi-axis", "arch held")] = [p for p in ps["multi-axis"] if "arch" not in moved[p]]
    return out, moved


def run(pool: str, fin: pd.DataFrame) -> pd.DataFrame:
    df, tasks, fams, col, S_, P = score_cube(pool)
    attrs = design_axes(df)
    groups, moved = _groups(attrs)
    fi = {f: i for i, f in enumerate(fams)}
    sd_pooled = seed_sd(fin).reindex(tasks).to_numpy(float)
    rep = fin[(fin["ladder"] == "deep") & (fin["data"] == "A")].groupby(["size", "L"])["seed"].nunique()
    rep_cells = {s: sorted(int(L) for (sz, L), n in rep.items() if sz == s and n >= 2) for s in SMALL_SIZES}
    pops = _populations(tasks)
    mask = load_mask(pool)
    rc = col[(TARGET_SIZE, 1.0)]
    rows = []
    for size in SMALL_SIZES:
        pc = col[(size, 1.0)]
        ok = passes_gate(mask, tasks, size, TARGET_SIZE).to_numpy()            # rule 1 at the proxy and the reference
        own = rep_cells[size]
        sd_size = (seed_sd(fin[fin["size"] == size]).reindex(tasks).to_numpy(float) if own else sd_pooled)
        scopes = {"size": (sd_size, f"{size} x3 cells (L{', L'.join(map(str, own))})" if own
                           else f"pooled 175M-1B x3 cells (no replicates at {size})"),
                  "pooled": (sd_pooled, "pooled 175M-1B x3 cells")}
        for (axes, group), pairs in groups.items():
            I = np.array([fi[a] for a, b in pairs]); J = np.array([fi[b] for a, b in pairs])
            A, V = pair_agree(S_, P, I, J, pc, rc)
            dp = np.abs(S_[:, I, pc] - S_[:, J, pc]); dr = np.abs(S_[:, I, rc] - S_[:, J, rc])
            axis_list = sorted({a for p in pairs for a in moved[p]})
            matched = all(size_matched(a, size) for a in axis_list)
            for pop, idx in pops.groupby(pops).groups.items():
                t = np.asarray(idx)
                for scope, (sd, source) in scopes.items():
                    has_sd = np.isfinite(sd[t]) & (sd[t] > 0)
                    base = V[t] & ok[t][:, None]
                    C = base & has_sd[:, None]
                    n = int(C.sum())
                    n_pairs = int((C.sum(0) > 0).sum())
                    for klab, k in KS.items():
                        thr = k * sd[t][:, None]
                        with np.errstate(invalid="ignore"):
                            a = C & (dp[t] > thr)
                            rclear = C & (dr[t] > thr)
                        b = C & ~A[t]
                        cnt = {"n_a_b": int((a & b).sum()), "n_a_not_b": int((a & ~b).sum()),
                               "n_not_a_b": int((~a & b & C).sum()), "n_not_a_not_b": int((~a & ~b & C).sum())}
                        enough = n_pairs >= MIN_PAIRS and n > 0                         # rule 5
                        sh = (lambda x, d=n: x / d if enough and d else np.nan)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            ratio = dp[t] / sd[t][:, None]; rratio = dr[t] / sd[t][:, None]
                        na = cnt["n_a_b"] + cnt["n_a_not_b"]
                        rows.append({
                            "population": pop, "axes": axes, "axis": group,
                            "moved_axes": ",".join(axis_list), "size": size, "reference": TARGET_SIZE,
                            "sd_scope": scope, "sd_source": source, "threshold": klab, "k": k,
                            "size_matched": matched, "n_pairs": n_pairs, "n_tasks": int((C.sum(1) > 0).sum()),
                            "n_comparisons": n, "n_comparisons_no_sd": int((base & ~has_sd[:, None]).sum()),
                            "share_a": sh(na), "share_b": sh(cnt["n_a_b"] + cnt["n_not_a_b"]),
                            "da": sh(cnt["n_a_not_b"] + cnt["n_not_a_not_b"]),
                            **cnt,
                            **{f"share_{c[2:]}": sh(v) for c, v in cnt.items()},
                            "share_b_given_a": cnt["n_a_b"] / na if enough and na else np.nan,
                            "share_b_given_not_a": cnt["n_not_a_b"] / (n - na) if enough and n - na else np.nan,
                            "n_a_b_ref_clear": int((a & b & rclear).sum()),
                            "share_ref_clear": sh(int(rclear.sum())),
                            "median_proxy_diff_over_sd": float(np.median(ratio[C])) if n else np.nan,
                            "median_ref_diff_over_sd": float(np.median(rratio[C])) if n else np.nan})
                        if not enough and n:
                            print(f"!!! RULE 5: {pop} {axes} {group} at {size}: {n_pairs} design pair(s) "
                                  f"< {MIN_PAIRS}, counts kept, shares blank")
    return pd.DataFrame(rows)


def depth_by_seed(fin: pd.DataFrame) -> pd.DataFrame:
    """The depth decision with the deep side re-drawn: per proxy size, (L, deep
    seed), the per-language BPB comparisons of deep data-A (that seed) against
    shallow data-A (seed 1904) and how many have the other sign than the 1.7B
    pair (seed 1904 both). Only the L with replicate deep cells (shallow has
    none); `L` = "all" pools them, and its share is blank below MIN_PAIRS L
    (rule 5: one L is one design pair)."""
    bpb = set(G.add_meta(pd.DataFrame({"task": fin["task"].unique()})).query("family == 'bpb'")["task"])
    f = fin[fin["task"].isin(bpb) & (fin["data"] == "A") & fin["ladder"].isin(["deep", "shallow"])]
    key = ["L", "task"]
    ref = f[f["size"] == TARGET_SIZE].pivot_table(index=key, columns="ladder", values="primary_score")
    ref = ref["deep"] - ref["shallow"]
    rows = []
    for size in SMALL_SIZES:
        g = f[f["size"] == size]
        deep = g[g["ladder"] == "deep"].pivot_table(index=key, columns="seed", values="primary_score")
        shallow = g[(g["ladder"] == "shallow") & (g["seed"] == GRID_SEED)].set_index(key)["primary_score"]
        Ls = sorted(L for L, n in deep.notna().sum(axis=1).groupby(level="L").max().items() if n >= 2)
        if not Ls:                     # 90M and 350M have no replicate seeds
            continue
        for seed in deep.columns:
            per = []
            for L in Ls:
                d = (deep[seed] - shallow).loc[L].dropna()
                r = ref.loc[L].reindex(d.index)
                per.append({"size": size, "L": str(L), "deep_seed": int(seed), "shallow_seed": GRID_SEED,
                            "n_pairs": 1, "n_comparisons": len(d), "n_b": int((np.sign(d) != np.sign(r)).sum()),
                            "median_proxy_diff": float(d.median()), "median_ref_diff": float(r.median())})
            tot = pd.DataFrame(per)
            n_pairs = int((tot["n_comparisons"] > 0).sum())
            rows += per + [{"size": size, "L": "all", "deep_seed": int(seed), "shallow_seed": GRID_SEED,
                            "n_pairs": n_pairs, "n_comparisons": int(tot["n_comparisons"].sum()),
                            "n_b": int(tot["n_b"].sum()),
                            "share_b": tot["n_b"].sum() / tot["n_comparisons"].sum() if n_pairs >= MIN_PAIRS else np.nan}]
    return pd.DataFrame(rows)


def _label(axes: str, group: str) -> str:
    if axes == "mono-axis":
        return "All mono-axis pairs" if group == "all" else AXIS_LABEL.get(group, group)[0].upper() + AXIS_LABEL.get(group, group)[1:]
    return {"all": "Multi-axis pairs", "moves arch": "Multi-axis, depth moved",
            "arch held": "Multi-axis, depth held"}[group]


ORDER = ["arch", "L", "scheme", "T", "activation", "optimizer", "all"]


def _focus(t: pd.DataFrame, pop: str = "bpb", size: str = FOCUS, scope: str = "size", k: str = "1 sd") -> pd.DataFrame:
    f = t[(t["population"] == pop) & (t["size"] == size) & (t["sd_scope"] == scope) & (t["threshold"] == k)
          & (t["n_comparisons"] > 0)].copy()      # activation and optimizer have no BPB pair against 1.7B yet
    f["_o"] = [(0 if a == "mono-axis" else 1, ORDER.index(g) if g in ORDER else 9 + i)
               for i, (a, g) in enumerate(zip(f["axes"], f["axis"]))]
    return f.sort_values("_o").drop(columns="_o")


def _pct(v) -> str:
    return "--" if not np.isfinite(v) else f"{100 * v:.0f}"


def latex(t: pd.DataFrame, path: Path) -> None:
    """Booktabs tabular of the 600M per-language BPB rows (1 x the 600M seed sd); no caption."""
    f = _focus(t)
    lines = [r"\begin{tabular}{lcrrrrrrrrr}", r"\toprule",
             r" & Size & & & \multicolumn{2}{c}{Share (\%)} & \multicolumn{4}{c}{Cross tab (\% of comparisons)} & \\",
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-10}",
             r"Design axis & matched & Pairs & Comparisons & (a) & (b) & (a), (b) & (a), not (b) & not (a), (b) "
             r"& not (a), not (b) & (a), (b), 1.7B clear \\", r"\midrule"]
    for i, r in enumerate(f.itertuples()):
        if i and r.axes == "multi-axis" and f.iloc[i - 1]["axes"] == "mono-axis" or (r.axes == "mono-axis" and r.axis == "all"):
            lines.append(r"\midrule")
        lines.append(" & ".join([_label(r.axes, r.axis), "yes" if r.size_matched else "no", str(r.n_pairs),
                                 str(r.n_comparisons), _pct(r.share_a), _pct(r.share_b), _pct(r.share_a_b),
                                 _pct(r.share_a_not_b), _pct(r.share_not_a_b), _pct(r.share_not_a_not_b),
                                 str(r.n_a_b_ref_clear)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def generate_readme(pool: str, t: pd.DataFrame, seeds: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    f = _focus(t)
    rows = [[_label(r.axes, r.axis), "yes" if r.size_matched else "**no**", r.n_pairs, r.n_comparisons,
             _pct(r.share_a), _pct(r.share_b), _pct(r.share_a_b), _pct(r.share_a_not_b), _pct(r.share_not_a_b),
             _pct(r.share_not_a_not_b), r.n_a_b_ref_clear] for r in f.itertuples()]
    src = f["sd_source"].iloc[0] if len(f) else ""
    # every proxy, the same groups: (b) / (a) / P(b | a)
    by_size = {s: _focus(t, size=s).set_index(["axes", "axis"]) for s in SMALL_SIZES}
    sizes_rows = [[_label(a, g), *["" if (a, g) not in by_size[s].index or not np.isfinite(by_size[s].loc[(a, g), "share_b"])
                                   else " / ".join(_pct(by_size[s].loc[(a, g), c]) for c in ("share_b", "share_a", "share_b_given_a"))
                                   for s in SMALL_SIZES]] for a, g in zip(f["axes"], f["axis"])]
    body = "\n\n".join([
        "## Attribution of the proxy-reference disagreements",
        f"Pool `{pool}` (pairs and scores, seed 1904, final checkpoint), seed sd from `{SEED_POOL}` (rq05's `seed_sd`). "
        f"(a) = |proxy difference| > 1 x the seed sd of the proxy's final score; (b) = the proxy difference has the "
        f"other sign than the {TARGET_SIZE} one (a DA-size miss, so (b) = 1 - DA-size pooled over comparisons). "
        f"Regenerate with `python analysis/rq03_noise_and_snr/flip_attribution.py --pool {pool}`; the sqrt(2) x sd "
        f"threshold, the pooled sd and the benchmark-accuracy population are in the CSV.",
        f"**{FOCUS}, per-language BPB** (sd: {src}; a comparison is one (design pair, trained language); "
        f"shares in % of the comparisons; last column = comparisons in (a) and (b) whose {TARGET_SIZE} difference also "
        f"clears the threshold):",
        md_table(["pairs", "size-matched", "design pairs", "comparisons", "(a)", "(b)", "(a)(b)", "(a) not (b)",
                  "not (a) (b)", "not (a) not (b)", f"(a)(b), {TARGET_SIZE} clear"], rows),
        "**Every proxy, per-language BPB** (cell = (b) / (a) / share of (a) that is also (b), in %; sd of the "
        "proxy's own x3 cells at 175M, 600M and 1B, the pooled sd at 90M and 350M):",
        md_table(["pairs", *SMALL_SIZES], sizes_rows),
        f"**The depth decision with the deep side's seed re-drawn** (per-language BPB, deep data A at each of its "
        f"three seeds against shallow data A at seed {GRID_SEED}; cell = comparisons whose sign differs from the "
        f"{TARGET_SIZE} pair / comparisons; shallow has no replicate, so only the deep side moves):",
        md_table(["proxy", "deep seed", "per L", "pooled (b)", "median deep − shallow BPB"],
                 [[s, int(sd), ", ".join(f"L{r.L} {r.n_b}/{r.n_comparisons}" for r in g[g["L"] != "all"].itertuples()),
                   _pct(g[g["L"] == "all"]["share_b"].iloc[0]),
                   " / ".join(f"{v:+.4f}" for v in g[g["L"] != "all"]["median_proxy_diff"])]
                  for (s, sd), g in seeds.groupby(["size", "deep_seed"], sort=False)]),
        f"Tables: `{stage}/{pool}/{NAME}.csv`, `{stage}/{pool}/{SEED_NAME}.csv`; LaTeX: `{stage}/{pool}/{NAME}.tex`."])
    replace_block(NOISE_AND_SNR / "README.md", "flip-attribution", body, f"flip_attribution.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    args = ap.parse_args()
    out_dir = NOISE_AND_SNR / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    fin = finals(ladder_frame(SEED_POOL))
    t = run(args.pool, fin)
    seeds = depth_by_seed(fin)
    seeds.to_csv(out_dir / f"{SEED_NAME}.csv", index=False)
    print(seeds[seeds["L"] == "all"].round(3).to_string(index=False))
    t.to_csv(out_dir / f"{NAME}.csv", index=False)
    latex(t, out_dir / f"{NAME}.tex")
    cols = ["axes", "axis", "size_matched", "n_pairs", "n_comparisons", "share_a", "share_b", "share_a_b",
            "share_a_not_b", "share_not_a_b", "share_not_a_not_b", "n_a_b_ref_clear"]
    print(_focus(t)[cols].round(3).to_string(index=False))
    print(f"Wrote {out_dir / NAME}.csv / .tex")
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, t, seeds)
