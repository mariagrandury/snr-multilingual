"""Family-level SNR ranking with subset-gain segments.

For each multilingual benchmark family, show the gap between the
full-language-set SNR and the best subset-SNR. y-tick labels are
colored by task format so the n_options story from
benchmark_creation is folded in.

Inputs (pooled `seeds_28_1797_1904` pool):
  - benchmark_creation/.../per_family_snr.csv  (family format + n_options)
  - smooth_subtasks/.../summary.csv            (full vs best SNR per family)
Output:
  - benchmark_creation/.../family_subset_gain.png
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import pandas as pd

POOL = "seeds_28_1797_1904"
FAMSRC = _REPO / "results" / "benchmark_creation" / POOL / "per_family_snr.csv"
SUBSRC = _REPO / "results" / "smooth_subtasks" / POOL / "summary.csv"
OUT = _REPO / "results" / "benchmark_creation" / POOL / "family_subset_gain.png"

FORMAT_COLOR = {
    "minimal_pair": "#1f77b4",
    "completion": "#2ca02c",
    "classification": "#ff7f0e",
    "mcq_question_only": "#d62728",
    "mrc_passage": "#9467bd",
}


def main() -> None:
    fam = pd.read_csv(FAMSRC).set_index("family")
    summary = pd.read_csv(SUBSRC)
    c1 = summary[(summary["case"] == "case1_per_benchmark")].dropna(subset=["snr_gain"])
    best_per_fam = c1.loc[c1.groupby("task")["snr_gain"].idxmax()].set_index("task")

    keep = fam.index.intersection(best_per_fam.index)
    fam = fam.loc[keep]
    best_per_fam = best_per_fam.loc[keep]
    merged = fam.join(best_per_fam[["size", "full_set_snr", "best_n", "best_snr",
                                    "snr_gain", "best_subset_short"]])
    merged = merged.sort_values("full_set_snr", ascending=True)

    def _short(subset: str, max_len: int = 38) -> str:
        return subset if len(subset) <= max_len else subset[: max_len - 1] + "…"

    snr_max = merged["best_snr"].max()
    x_annot = snr_max + 0.25
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(merged) + 2))
    y = list(range(len(merged)))
    for yi, (name, row) in zip(y, merged.iterrows()):
        ax.plot([row["full_set_snr"], row["best_snr"]], [yi, yi],
                color="#888", linewidth=2.5, zorder=1)
        ax.scatter(row["full_set_snr"], yi, color="white",
                   edgecolor="black", s=70, zorder=2)
        ax.scatter(row["best_snr"], yi, color="#d62728",
                   edgecolor="black", s=90, zorder=3)
        gain_tag = f"+{row['snr_gain']:.2f}"
        ax.text((row["full_set_snr"] + row["best_snr"]) / 2, yi + 0.18,
                gain_tag, ha="center", va="bottom", fontsize=8,
                color="#d62728", fontweight="bold")
        annot = f"{_short(row['best_subset_short'])}   ({row['size']}, n={int(row['best_n'])})"
        ax.text(x_annot, yi, annot, va="center", fontsize=8, color="#333",
                family="monospace")

    ax.set_yticks(y)
    labels = []
    for name in merged.index:
        fmt = merged.loc[name, "format"]
        nopt = int(merged.loc[name, "n_options"])
        labels.append(f"{name}  ({fmt}, {nopt}-opt)")
    ax.set_yticklabels(labels)
    for tick, name in zip(ax.get_yticklabels(), merged.index):
        tick.set_color(FORMAT_COLOR.get(merged.loc[name, "format"], "black"))

    ax.set_xlabel("SNR (compute_snr_small_scale; 9 (mix, seed) model_families per size)")
    ax.set_title(
        f"Family SNR: full language set (○) → best subset (●)   |   pool {POOL}",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, x_annot + 5)
    ax.axvline(x=snr_max + 0.1, color="black", linewidth=0.5, alpha=0.4)

    legend = [plt.Line2D([], [], marker="s", linestyle="", color=c,
                         markeredgecolor="black", label=fmt, markersize=9)
              for fmt, c in FORMAT_COLOR.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=8, title="task format",
              title_fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
