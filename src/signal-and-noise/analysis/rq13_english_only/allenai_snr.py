"""Does SNR predict decision accuracy on the AllenAI benchmarks when the models
are monolingual? The check behind the surrogates' negative result: if our SNR
code is right, it must reproduce the SNR -> DA relation on DataDecide and show
it on our English-only models, and the relation must fade as the same English
benchmarks are read on multilingual models.

  benchmarks  the OLMES tasks of the SNR paper that we also evaluate
              (`english_only.ALLENAI_TASKS` plus BoolQ, PIQA and Winogrande;
              the letter `commonsense_qa` stays out, CSQA is read as its
              answer-string twin like DataDecide's RC format)
  ours        `english_only.py`'s tables: per (task, group, size) the SNR of
              the mirrored four-family groups (L1: DCLM-Edu, DCLM, FineWeb,
              shallow; L2, L15, L30: their multilingual counterparts) and the
              multi-axis DA-size of the same families against 1.7B (6 pairs),
              both gated (rule 1); proxies 90M-1B
  DataDecide  rq07's `allenai_snr_variants_per_task.csv`: the same SNR variant
              and DA kernel on DataDecide's 25 recipes, proxies 150M-750M
              against 1B
  statistic   Spearman rho of log10 SNR against DA-size over the (task, proxy)
              points of a group, with a 95 % bootstrap interval that
              resamples whole benchmarks

    english_only_allenai_snr_da.csv            one row per (group, task, proxy)
    english_only_allenai_snr_da_summary.csv    one row per group: rho, interval, points
    english_only_allenai_snr_da_paper.png      one panel per group
    english_only_allenai_snr_da.tex            per benchmark: median SNR and DA-size per group

    python analysis/rq13_english_only/allenai_snr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.rq13_english_only.english_only import ALLENAI_CSV, ALLENAI_TASKS, MIRROR, VARIANT  # noqa: E402
from analysis.utils import SMALL_SIZES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "pretraining" / "predictivity"
TASKS = {**{o: t for o, t in ALLENAI_TASKS.items() if o != "commonsense_qa"},
         "boolq": "boolq", "piqa": "piqa", "winogrande": "winogrande"}
DATADECIDE = "DataDecide"
GROUPS = [DATADECIDE, "L1", "L2", "L15", "L30"]
N_BOOT = 1000
mpl.rcParams.update(S.RC)


def points() -> pd.DataFrame:
    snr = pd.read_csv(OUT / "english_only_snr_per_task.csv")
    snr = snr[(snr["families"] == MIRROR) & snr["all_families"] & snr["task"].isin(TASKS)]
    da = pd.read_csv(OUT / "english_only_da_size_per_task_both_axes.csv")
    da = da[(da["axes"] == "multi-axis") & (da["population"] == "all benchmarks") & da["all_families"] & ~da["gated"]]
    ours = snr.merge(da[["task", "group", "size", "da", "n_pairs"]], on=["task", "group", "size"])
    ours = ours[ours["size"].isin(SMALL_SIZES)][["group", "task", "size", "snr", "da", "n_pairs"]]
    al = pd.read_csv(ALLENAI_CSV).set_index("task")
    sizes = [c.removeprefix("decision_acc_size_") for c in al.columns if c.startswith("decision_acc_size_")]
    theirs = pd.DataFrame([{"group": DATADECIDE, "task": o, "size": z, "snr": al.at[t, f"snr_{VARIANT}_{z}"],
                            "da": al.at[t, f"decision_acc_size_{z}"], "n_pairs": 300}
                           for o, t in TASKS.items() if t in al.index for z in sizes])
    p = pd.concat([theirs, ours], ignore_index=True).dropna(subset=["snr", "da"])
    p = p[p["snr"] > 0]
    names = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}        # one family, two tasks
    return p.assign(log10_snr=np.log10(p["snr"]), benchmark=p["task"].map(lambda t: names.get(t, G.paper_name(t))))


def summary(p: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for grp in GROUPS:
        g = p[p["group"] == grp]
        tasks = g["task"].unique()
        boot = []
        for _ in range(N_BOOT):              # resample whole benchmarks: a benchmark's proxies share it
            d = pd.concat([g[g["task"] == t] for t in rng.choice(tasks, len(tasks))])
            if d["da"].nunique() > 1 and d["log10_snr"].nunique() > 1:
                boot.append(spearmanr(d["log10_snr"], d["da"]).statistic)
        rows.append({"group": grp, "rho": spearmanr(g["log10_snr"], g["da"]).statistic,
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "n_points": len(g), "n_benchmarks": len(tasks), "median_snr": g["snr"].median(),
                     "median_da": g["da"].median()})
    return pd.DataFrame(rows)


def figure(p: pd.DataFrame, s: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.2, 1.9), sharey=True, sharex=True)
    benches = sorted(p["benchmark"].unique())
    colour = dict(zip(benches, plt.cm.tab10.colors))
    title = {DATADECIDE: "DataDecide (English)", "L1": "Ours, $K=1$ (English)", "L2": "Ours, $K=2$",
             "L15": "Ours, $K=15$", "L30": "Ours, $K=30$"}
    for ax, grp in zip(axes, GROUPS):
        g = p[p["group"] == grp]
        ax.scatter(g["log10_snr"], g["da"] + np.random.default_rng(0).uniform(-.015, .015, len(g)), s=9,
                   c=[colour[b] for b in g["benchmark"]], lw=0, alpha=.85)
        r = s.set_index("group").loc[grp]
        ax.set_title(f"{title[grp]}\n$\\rho$ = {r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", fontsize=6.5)
        ax.axhline(.5, color=S.MUTED, lw=.6, ls=":")
        ax.grid(color=S.GRID, lw=.5); S.clean(ax); ax.tick_params(labelsize=6)
        ax.set_xlabel("SNR (log$_{10}$)", fontsize=6.5)
    axes[0].set_ylabel("DA-size", fontsize=6.5)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, color=colour[b], label=b) for b in benches]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.12), ncol=len(benches), fontsize=5.5, frameon=False)
    fig.tight_layout(w_pad=.4)
    S.save_paper(fig, path.with_suffix(""))


def table(p: pd.DataFrame, s: pd.DataFrame, path: Path) -> None:
    med = p.groupby(["benchmark", "group"])[["snr", "da"]].median().unstack("group")
    head = ["Benchmark"] + [f"{g}" for g in GROUPS]
    lines = ["\\begin{tabular}{l" + "rr" * len(GROUPS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{g}}}" for g in GROUPS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(GROUPS))),
             head[0] + " & " + " & ".join(["SNR & DA"] * len(GROUPS)) + " \\\\", "\\midrule"]
    f = lambda v, d: "--" if pd.isna(v) else f"{v:.{d}f}"
    for b in med.index:
        lines.append(" & ".join([b] + [x for g in GROUPS for x in (f(med.at[b, ("snr", g)], 1), f(med.at[b, ("da", g)], 2))])
                     + " \\\\")
    r = s.set_index("group")
    lines += ["\\midrule", "Median & " + " & ".join(f"{r.at[g, 'median_snr']:.1f} & {r.at[g, 'median_da']:.2f}" for g in GROUPS)
              + " \\\\", "$\\rho$(log SNR, DA) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{r.at[g, 'rho']:.2f}}}" for g in GROUPS)
              + " \\\\", "\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    p = points()
    s = summary(p)
    p.to_csv(OUT / "english_only_allenai_snr_da.csv", index=False)
    s.to_csv(OUT / "english_only_allenai_snr_da_summary.csv", index=False)
    figure(p, s, OUT / "english_only_allenai_snr_da_paper.png")
    table(p, s, OUT / "english_only_allenai_snr_da.tex")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
