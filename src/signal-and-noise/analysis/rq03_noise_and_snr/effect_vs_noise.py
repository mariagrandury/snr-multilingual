"""Intervention effect against seed and checkpoint noise.

Per (size, L, task): each intervention's |Δ| at the grid seed is put against
the seed noise (sample std over the seed replicates of the baseline cell,
where ≥ 2 seeds exist) and the late-checkpoint noise (std over the last
`last_n` checkpoints of the baseline cell — raw, and detrended because under
WSD the final window is still descending). A ratio near 1 means the two
levels are the same model as far as a ranking is concerned (the "read this
against the seed row" rule of `ladder_report.md`): a decision on such a cell
is a coin flip whatever its decision accuracy (rq05) says. The seed-over-
checkpoint ratio is also what says how optimistic the checkpoint-noise SNR
of `run_apertus_snr_variants.py` is.

    effect_vs_noise.csv   per (size, L, task): |Δ| per intervention, seed noise, raw and detrended checkpoint noise, ratios
    effect_vs_noise.png

    python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_all
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
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import NOISE_AND_SNR  # noqa: E402
from analysis.rq05_design_decisions.analyze import INTERVENTIONS  # noqa: E402
from analysis.utils import GRID_SEED, LAST_N, benchmark_family, finals, ladder_frame, size_order  # noqa: E402

OUT_ROOT = NOISE_AND_SNR
CANONICAL = "predictivity_all"      # every seed: the seed-noise column needs the replicates
mpl.rcParams.update(S.RC)


def _late_std(scores: np.ndarray, detrend: bool) -> float:
    s = np.asarray(scores[-LAST_N:], dtype=float)
    if len(s) < 2:
        return float("nan")
    if detrend:
        x = np.arange(len(s))
        s = s - np.polyval(np.polyfit(x, s, 1), x)
    return float(np.std(s))


def effect_vs_noise(df: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """Per (size, L, task): each intervention's |Δ| at the grid seed, the seed
    noise of the baseline cell and its late-checkpoint noise."""
    key = ["size", "L", "task"]
    grid = fin[fin["seed"] == GRID_SEED]
    out = None
    for k, (_label, axis, levels, (hcol, hval)) in INTERVENTIONS.items():
        piv = grid[grid[hcol] == hval].pivot_table(index=key, columns=axis, values="primary_score")
        if set(levels) <= set(piv.columns):
            eff = (piv[levels[0]] - piv[levels[1]]).abs().rename(f"effect_{k}")
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
    for k in INTERVENTIONS:
        col = f"effect_{k}"
        if col in out:
            out[f"{col}_over_seed"] = out[col] / out["seed_noise"]
            out[f"{col}_over_ckpt"] = out[col] / out["ckpt_noise_detrended"]
    return out



def plot_effect_vs_noise(evn: pd.DataFrame, path: Path) -> None:
    cols = [c for c in evn.columns if c.startswith("effect_") and ("_over_seed" in c or "_over_ckpt" in c)
            and evn[c].notna().any()]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.2), squeeze=False)
    for ax, col in zip(axes[0], cols):
        for pop, g in evn.groupby("population"):
            med = g.groupby("size")[col].median().dropna()
            sizes = size_order(med.index)
            if sizes:
                ax.plot(sizes, [med[s] for s in sizes], marker="o", label=pop)
        ax.axhline(1, ls="--", lw=0.8, color=S.MUTED)
        ax.set_yscale("log"); ax.set_title(col.replace("effect_", "").replace("_over_", " / "), loc="left", fontsize=8)
        ax.grid(color=S.GRID, lw=.6); ax.set_xlabel("size"); S.clean(ax)
    axes[0][0].set_ylabel("median |effect| / noise")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Intervention effect against seed and late-checkpoint noise (1 = the same model)", y=1.0)
    fig.tight_layout(); S.save(fig, path, dpi=140)



def generate_readme(pool: str, out_dir: Path, evn: pd.DataFrame) -> None:
    if pool != CANONICAL or evn.empty:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    bullets, rows = [], []
    seed_vs_ckpt = (evn["seed_noise"] / evn["ckpt_noise_detrended"]).replace([np.inf, -np.inf], np.nan).dropna()
    if not seed_vs_ckpt.empty:
        bullets.append(f"- **Seed noise vs detrended checkpoint noise** — median ratio "
                       f"{fmt(seed_vs_ckpt.median())} over {len(seed_vs_ckpt)} (size, L, task) cells with seed replicates.")
    arch = evn.get("effect_arch_over_seed")
    if arch is not None and arch.notna().any():
        arch = arch.dropna()
        bullets.append(f"- **Depth effect vs seed noise** — median |Δ|/seed-std {fmt(arch.median())}; "
                       f"{(arch > 2).mean():.0%} of {len(arch)} cells above 2× (a distinct model for SNR, not a re-roll).")
    for pop, g in evn.groupby("population"):
        for k in INTERVENTIONS:
            for kind in ("seed", "ckpt"):
                col = f"effect_{k}_over_{kind}"
                if col in g and g[col].notna().any():
                    rows.append([pop, f"{k} / {kind}", fmt(g[col].median()), int(g[col].notna().sum())])
    body = "\n\n".join([
        "## Intervention effect against noise",
        f"Numbers from the `{pool}` pool. Regenerate with "
        f"`python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool {pool}`.",
        "\n".join(bullets),
        "**Effect over noise** (median over (size, L, task) cells):",
        md_table(["population", "effect / noise", "median", "n"], rows),
        f"![Effect vs noise]({rel}/effect_vs_noise.png)"])
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "effect-vs-noise", body, f"effect_vs_noise.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, seeds {sorted(df['seed'].unique())}")
    evn = effect_vs_noise(df, fin)
    evn.to_csv(out_dir / "effect_vs_noise.csv", index=False)
    print(f"Wrote → {out_dir / 'effect_vs_noise.csv'} ({len(evn)} cells)")
    if not evn.empty:
        plot_effect_vs_noise(evn, out_dir / "effect_vs_noise.png")
    generate_readme(pool, out_dir, evn)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL}; the seed "
                        "replicates are what the seed-noise column is made of).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
