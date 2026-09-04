"""Small-to-large proxy predictivity over the (proxy size, language count) grid.

The plan's question (plan/small-to-large-predictivity-training-plan.md,
"Analysis"): at a given number of languages L, does a proxy size rank a design
choice the way the reference size — the largest model trained at that L —
does? Three reads, all from the ladder report (rq00–rq05 ask which
*benchmarks* are reliable; this RQ asks which *model sizes* are):

  1. intervention DA   — per (intervention, L, proxy size): the fraction of
                         population items on which the proxy and the reference
                         agree about which level of the intervention is better.
                         Two levels per intervention: model depth (deep vs
                         shallow, every L) and data scheme (A vs B, L ∈
                         {8, 15, 30}). Populations: per-language BPB on the
                         languages both levels train (`bpb_trained`), on all
                         100 validation languages (`bpb_all`), the benchmark
                         tasks (`benchmark`), and the single macro-BPB decision
                         (`bpb_macro`). `decision_acc_fast` on the two models
                         of one item is exactly that agreement.
  2. scaling-law error — per (L, arch, scheme, language): log BPB = a − α log N
                         fitted on the proxy rungs up to a ladder top, predicting
                         the reference's BPB; relative error. The plan's
                         "prediction ability" read.
  3. effect vs noise   — per (size, L, task): the intervention's |Δ| against
                         seed noise (std over the seed replicates, where the ×3
                         cells exist) and late-checkpoint noise (std over the
                         last N checkpoints — raw, and detrended because under
                         WSD the final window is still descending). A decision
                         whose effect sits inside the noise is a coin flip
                         whatever the DA says (the plan's caveat); this table is
                         also ladder_report.md's "effect of each transformation",
                         resolved per benchmark and per language with a proper
                         seed std.

    python analysis/rq06_proxy_predictivity/analyze.py --pool predictivity_seeds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import bucket_order, load_pools  # noqa: E402
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402
from pretrain.ladder_report import NON_EMB, _fit  # noqa: E402
from snr.metrics import decision_acc_fast  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import PROXY_PREDICTIVITY  # noqa: E402
from analysis.utils import LAST_N, assign_language, benchmark_family, build_snr_pool  # noqa: E402

OUT_ROOT = PROXY_PREDICTIVITY
CANONICAL = "predictivity_seeds"      # the pool whose numbers the README carries
GRID_SEED = 1904                      # the plan grid's seed
MIN_ITEMS = 3                         # fewest population items for a DA cell
INTERVENTIONS = {"arch": ("deep", "shallow"), "scheme": ("A", "B")}
# The other axis is held at its baseline level while an intervention is read.
BASELINE = {"arch": ("scheme", "A"), "scheme": ("arch", "deep")}


def _size_order(sizes) -> list[str]:
    order = bucket_order()
    return sorted(set(sizes), key=lambda s: order.index(s) if s in order else 99)


def _finals(df: pd.DataFrame) -> pd.DataFrame:
    """Each cell's last checkpoint per task."""
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def _trained(L: int, scheme: str) -> set[str]:
    return {"bpb_dclm"} | {f"bpb_{s}" for s in cell_fineweb_subsets(L, scheme)}


def _population(fin: pd.DataFrame, name: str, L: int, levels: tuple, axis: str) -> pd.DataFrame:
    """Rows of `fin` belonging to one population at language setting L."""
    sub = fin[fin["L"] == L]
    if name == "benchmark":
        return sub[sub["kind"] == "benchmark"]
    if name == "bpb_macro":
        return sub[sub["task"] == "bpb_macro"]
    bpb = sub[(sub["kind"] == "bpb") & (sub["task"] != "bpb_macro")]
    if name == "bpb_all":
        return bpb
    # bpb_trained: languages every level of the intervention trains on
    schemes = levels if axis == "scheme" else ("A",)
    keep = set.intersection(*(_trained(L, s) for s in schemes))
    return bpb[bpb["task"].isin(keep)]


# --- 1. intervention decision accuracy ---------------------------------------

def intervention_da(fin: pd.DataFrame) -> pd.DataFrame:
    fin = fin[fin["seed"] == GRID_SEED]
    rows = []
    for axis, levels in INTERVENTIONS.items():
        base_axis, base_level = BASELINE[axis]
        sub = fin[fin[base_axis] == base_level]
        for L in sorted(sub["L"].unique()):
            for pop in ("bpb_trained", "bpb_all", "benchmark", "bpb_macro"):
                piv = (_population(sub, pop, L, levels, axis)
                       .pivot_table(index=["size", "task"], columns=axis,
                                    values="primary_score"))
                if not set(levels) <= set(piv.columns):
                    continue
                piv = piv.dropna(subset=list(levels))
                sizes = _size_order(piv.index.get_level_values("size"))
                if len(sizes) < 2:
                    continue
                ref = sizes[-1]
                ref_scores = piv.xs(ref, level="size")
                for proxy in sizes[:-1]:
                    p = piv.xs(proxy, level="size")
                    items = p.index.intersection(ref_scores.index)
                    if len(items) < (1 if pop == "bpb_macro" else MIN_ITEMS):
                        continue
                    agree = [decision_acc_fast(p.loc[t, list(levels)].to_numpy(),
                                               ref_scores.loc[t, list(levels)].to_numpy())
                             for t in items]
                    d_ref = (ref_scores.loc[items, levels[0]] - ref_scores.loc[items, levels[1]])
                    d_proxy = (p.loc[items, levels[0]] - p.loc[items, levels[1]])
                    # the first level wins an item when its score is higher on a
                    # benchmark, lower on BPB
                    first_better = (d_ref > 0) if pop == "benchmark" else (d_ref < 0)
                    rows.append({
                        "intervention": axis, "population": pop, "L": L,
                        "proxy_size": proxy, "reference_size": ref, "n_items": len(items),
                        "decision_acc": float(np.mean(agree)),
                        "mean_abs_delta_proxy": float(d_proxy.abs().mean()),
                        "mean_abs_delta_ref": float(d_ref.abs().mean()),
                        # which level the reference prefers on the majority of items
                        "reference_prefers": levels[0] if first_better.mean() > 0.5 else levels[1],
                    })
    return pd.DataFrame(rows)


# --- 2. scaling-law error on per-language BPB ---------------------------------

def scaling_law_error(fin: pd.DataFrame) -> pd.DataFrame:
    sub = fin[(fin["seed"] == GRID_SEED) & (fin["kind"] == "bpb")]
    rows = []
    for (L, arch, scheme, task), g in sub.groupby(["L", "arch", "scheme", "task"]):
        g = g.assign(N=g["size"].map(NON_EMB)).dropna(subset=["N"]).sort_values("N")
        if len(g) < 4:                      # three proxy rungs + the reference
            continue
        ref = g.iloc[-1]
        proxies = g.iloc[:-1]
        for top in range(3, len(proxies) + 1):
            pts = proxies.iloc[:top]
            fit = _fit(list(zip(pts["N"], pts["primary_score"])))
            if not fit:
                continue
            slope, icpt = fit
            pred = float(np.exp(icpt + slope * np.log(ref["N"])))
            rows.append({
                "L": L, "arch": arch, "scheme": scheme, "task": task,
                "language": assign_language(task),
                "trained": task in _trained(L, scheme),
                "ladder_top": pts["size"].iloc[-1], "n_points": top,
                "reference_size": ref["size"], "alpha": -slope,
                "predicted": pred, "observed": float(ref["primary_score"]),
                "rel_error": (pred - float(ref["primary_score"])) / float(ref["primary_score"]),
            })
    return pd.DataFrame(rows)


# --- 3. intervention effect vs seed / checkpoint noise ------------------------

def _late_std(scores: np.ndarray, detrend: bool) -> float:
    s = np.asarray(scores[-LAST_N:], dtype=float)
    if len(s) < 2:
        return float("nan")
    if detrend:
        x = np.arange(len(s))
        s = s - np.polyval(np.polyfit(x, s, 1), x)
    return float(np.std(s))


def effect_vs_noise(df: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    key = ["size", "L", "task"]
    grid = fin[fin["seed"] == GRID_SEED]
    out = None
    for axis, levels in INTERVENTIONS.items():
        base_axis, base_level = BASELINE[axis]
        piv = (grid[grid[base_axis] == base_level]
               .pivot_table(index=key, columns=axis, values="primary_score"))
        if set(levels) <= set(piv.columns):
            eff = (piv[levels[0]] - piv[levels[1]]).abs().rename(f"effect_{axis}")
            out = eff.to_frame() if out is None else out.join(eff, how="outer")
    if out is None:
        return pd.DataFrame()
    # seed noise: the baseline cell's finals across seeds (sample std, n >= 2)
    base = fin[(fin["arch"] == "deep") & (fin["scheme"] == "A")]
    seed = base.groupby(key)["primary_score"].agg(["std", "count"])
    out["seed_noise"] = seed.loc[seed["count"] >= 2, "std"]
    out["n_seeds"] = seed["count"]
    # checkpoint noise: the grid seed's baseline cell, last N checkpoints
    curve = df[(df["seed"] == GRID_SEED) & (df["arch"] == "deep") & (df["scheme"] == "A")]
    curve = curve.sort_values("step").groupby(key)["primary_score"].apply(np.asarray)
    out["ckpt_noise"] = curve.map(lambda s: _late_std(s, detrend=False))
    out["ckpt_noise_detrended"] = curve.map(lambda s: _late_std(s, detrend=True))
    out = out.reset_index()
    out["family"] = out["task"].map(benchmark_family)
    out["population"] = np.where(out["family"] == "bpb", "bpb", "benchmark")
    for axis in INTERVENTIONS:
        col = f"effect_{axis}"
        if col in out:
            out[f"{col}_over_seed"] = out[col] / out["seed_noise"]
            out[f"{col}_over_ckpt"] = out[col] / out["ckpt_noise_detrended"]
    return out


# --- figures ----------------------------------------------------------------

def plot_da_grid(da: pd.DataFrame, path: Path) -> None:
    pops = [p for p in ("bpb_trained", "benchmark", "bpb_all") if p in set(da["population"])]
    if not pops:
        return
    axes_ = list(INTERVENTIONS)
    fig, axes = plt.subplots(len(axes_), len(pops),
                             figsize=(4.2 * len(pops), 3.0 * len(axes_)), squeeze=False)
    for i, axis in enumerate(axes_):
        for j, pop in enumerate(pops):
            ax = axes[i][j]
            sub = da[(da["intervention"] == axis) & (da["population"] == pop)]
            if sub.empty:
                ax.set_visible(False)
                continue
            Ls = sorted(sub["L"].unique())
            sizes = _size_order(sub["proxy_size"])
            mat = np.full((len(sizes), len(Ls)), np.nan)
            for _, r in sub.iterrows():
                mat[sizes.index(r["proxy_size"]), Ls.index(r["L"])] = r["decision_acc"]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for a in range(len(sizes)):
                for b in range(len(Ls)):
                    if np.isfinite(mat[a, b]):
                        n = int(sub[(sub["proxy_size"] == sizes[a]) & (sub["L"] == Ls[b])]["n_items"].iloc[0])
                        ax.text(b, a, f"{mat[a, b]:.2f}\n(n={n})", ha="center", va="center",
                                fontsize=6, color="white" if mat[a, b] < 0.6 else "black")
            ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L{L}" for L in Ls], fontsize=7)
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes, fontsize=7)
            ax.set_title(f"{axis}: {' vs '.join(INTERVENTIONS[axis])} — {pop}", fontsize=8)
            ax.set_xlabel("languages"); ax.set_ylabel("proxy size")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="decision accuracy (proxy vs reference)",
                 fraction=0.02)
    fig.suptitle("Does a proxy size rank the intervention like the reference size at that L?",
                 fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_error(sle: pd.DataFrame, path: Path) -> None:
    sub = sle[(sle["arch"] == "deep") & (sle["scheme"] == "A") & sle["trained"]]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for L, g in sub.groupby("L"):
        med = g.groupby("ladder_top")["rel_error"].apply(lambda v: np.median(np.abs(v)))
        tops = _size_order(med.index)
        ax.plot(tops, [med[t] for t in tops], marker="o", label=f"L{L}")
    ax.set_xlabel("largest proxy rung in the fit")
    ax.set_ylabel("median |relative error| of predicted BPB")
    ax.set_title("Per-language BPB power-law prediction of the reference rung", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_effect_vs_noise(evn: pd.DataFrame, path: Path) -> None:
    cols = [c for c in ("effect_arch_over_seed", "effect_scheme_over_seed",
                        "effect_arch_over_ckpt", "effect_scheme_over_ckpt") if c in evn]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.2), squeeze=False)
    for ax, col in zip(axes[0], cols):
        for pop, g in evn.groupby("population"):
            med = g.groupby("size")[col].median().dropna()
            sizes = _size_order(med.index)
            if sizes:
                ax.plot(sizes, [med[s] for s in sizes], marker="o", label=pop)
        ax.axhline(1, ls="--", lw=0.8, color="grey")
        ax.set_yscale("log"); ax.set_title(col, fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlabel("size")
    axes[0][0].set_ylabel("median |effect| / noise")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Intervention effect against seed and late-checkpoint noise (1 = same model)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# --- README ------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, da: pd.DataFrame, sle: pd.DataFrame,
                    evn: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    bullets, blocks = [], []
    if not da.empty:
        core = da[(da["intervention"] == "arch") & (da["population"] == "bpb_trained")]
        if not core.empty:
            grid = core.pivot_table(index="proxy_size", columns="L", values="decision_acc")
            grid = grid.reindex(_size_order(grid.index))
            # smallest proxy that reaches 0.75 at each L
            first = {L: next((s for s in grid.index if grid.loc[s, L] >= 0.75), "—")
                     for L in grid.columns}
            bullets.append(
                "- **Depth decision (deep vs shallow) on per-language BPB** — smallest proxy "
                "reaching DA ≥ 0.75 against the reference: "
                + ", ".join(f"L{L}: {s}" for L, s in first.items()) + ".")
            rows = [[s] + [fmt(grid.loc[s, L]) for L in grid.columns] for s in grid.index]
            blocks += ["**Intervention DA, depth on per-language BPB** (rows: proxy size; "
                       "columns: language setting; reference = largest size at that L):",
                       md_table(["proxy"] + [f"L{L}" for L in grid.columns], rows)]
        bench = da[(da["intervention"] == "arch") & (da["population"] == "benchmark")]
        if not bench.empty:
            m = bench.groupby("proxy_size")["decision_acc"].mean()
            bullets.append("- **Depth decision on benchmarks** — mean DA over L by proxy: "
                           + ", ".join(f"{s} {fmt(m[s])}" for s in _size_order(m.index)) + ".")
        blocks.append(f"![Intervention DA grid]({stage}/{pool}/intervention_da.png)")
    if not sle.empty:
        core = sle[(sle["arch"] == "deep") & (sle["scheme"] == "A") & sle["trained"]]
        if not core.empty:
            med = (core.groupby(["L", "ladder_top"])["rel_error"]
                   .apply(lambda v: float(np.median(np.abs(v)))).unstack("ladder_top"))
            med = med[_size_order(med.columns)]
            rows = [[f"L{L}"] + [fmt(med.loc[L, c], 3) for c in med.columns] for L in med.index]
            bullets.append(
                "- **Scaling-law error** — median |relative error| of the reference's "
                "per-language BPB predicted from the proxy ladder: "
                + ", ".join(f"L{L} {fmt(med.loc[L].dropna().iloc[-1], 3)}" for L in med.index)
                + " (largest proxy ladder at that L).")
            blocks += ["**Scaling-law error on trained-language BPB** (median |relative "
                       "error|; columns: largest proxy rung in the fit):",
                       md_table(["L"] + list(med.columns), rows),
                       f"![Scaling-law error]({stage}/{pool}/scaling_law_error.png)"]
    if not evn.empty:
        rows = []
        for pop, g in evn.groupby("population"):
            for col in ("effect_arch_over_seed", "effect_scheme_over_seed",
                        "effect_arch_over_ckpt", "effect_scheme_over_ckpt"):
                if col in g and g[col].notna().any():
                    rows.append([pop, col.replace("effect_", "").replace("_over_", " / "),
                                 fmt(g[col].median()), int(g[col].notna().sum())])
        seed_vs_ckpt = (evn["seed_noise"] / evn["ckpt_noise_detrended"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if not seed_vs_ckpt.empty:
            bullets.append(f"- **Seed noise vs detrended checkpoint noise** — median ratio "
                           f"{fmt(seed_vs_ckpt.median())} over {len(seed_vs_ckpt)} "
                           f"(size, L, task) cells with seed replicates.")
        arch = evn.get("effect_arch_over_seed")
        if arch is not None and arch.notna().any():
            arch = arch.dropna()
            bullets.append(f"- **Depth effect vs seed noise** — median |Δ|/seed-std "
                           f"{fmt(arch.median())}; {(arch > 2).mean():.0%} of {len(arch)} cells "
                           f"above 2× (a distinct model for SNR, not a re-roll).")
        blocks += ["**Intervention effect against noise** (median over cells):",
                   md_table(["population", "effect / noise", "median", "n"], rows),
                   f"![Effect vs noise]({stage}/{pool}/effect_vs_noise.png)"]
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq06_proxy_predictivity/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- driver -------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = build_snr_pool(pool)
    fin = _finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, "
          f"{fin['task'].nunique()} tasks, seeds {sorted(df['seed'].unique())}")

    da = intervention_da(fin)
    da.to_csv(out_dir / "intervention_da.csv", index=False)
    print(f"Wrote → {out_dir / 'intervention_da.csv'} ({len(da)} cells)")
    if not da.empty:
        plot_da_grid(da, out_dir / "intervention_da.png")

    sle = scaling_law_error(fin)
    sle.to_csv(out_dir / "scaling_law_error.csv", index=False)
    print(f"Wrote → {out_dir / 'scaling_law_error.csv'} ({len(sle)} fits)")
    if not sle.empty:
        plot_scaling_error(sle, out_dir / "scaling_law_error.png")

    evn = effect_vs_noise(df, fin)
    evn.to_csv(out_dir / "effect_vs_noise.csv", index=False)
    print(f"Wrote → {out_dir / 'effect_vs_noise.csv'} ({len(evn)} cells)")
    if not evn.empty:
        plot_effect_vs_noise(evn, out_dir / "effect_vs_noise.png")

    generate_readme(pool, out_dir, da, sle, evn)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL}; "
                        "all seeds are needed for the seed-noise column).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
