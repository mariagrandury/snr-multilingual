"""Two-panel composite: each SNR variant's correlation with decision
accuracy on Apertus (12 multilingual languages) AND its cross-corpus
transfer to AllenAI DataDecide. Same variant order on both panels.

Inputs (pooled `seeds_28_1797_1904` pool):
  - snr_definition/.../top_variants_overall.csv
  - allenai_comparison/.../pearson_r_per_variant.csv
Output:
  - results/snr_definition/seeds_28_1797_1904/variants_apertus_vs_allenai.png
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
APERTUS = _REPO / "results" / "snr_definition" / POOL / "top_variants_overall.csv"
ALLEN = _REPO / "results" / "allenai_comparison" / POOL / "pearson_r_per_variant.csv"
OUT = _REPO / "results" / "snr_definition" / POOL / "variants_apertus_vs_allenai.png"

# Mathematical-family groupings, mirroring the README's clusters.
FAMILY_COLOR = {
    "dispersion": "#1f77b4",      # range, dispersion, dist_std, aad, mpd, quartile_deviation, rms_deviation, mad
    "rel_spread": "#ff7f0e",      # iqr, rel_std, rel_mpd, rel_dispersion, rel_mpsd, rel_star_discrepancy
    "discrepancy": "#2ca02c",     # discrepancy, star_discrepancy, star_discrepancy_shifted, dispersion_shifted, gini
    "depth": "#d62728",           # tukey, projection
    "other": "#7f7f7f",           # mpsd
}
FAMILY = {
    "range": "dispersion", "dispersion": "dispersion", "dist_std": "dispersion",
    "aad": "dispersion", "mpd": "dispersion", "quartile_deviation": "dispersion",
    "rms_deviation": "dispersion", "mad": "dispersion",
    "iqr": "rel_spread", "rel_std": "rel_spread", "rel_mpd": "rel_spread",
    "rel_dispersion": "rel_spread", "rel_mpsd": "rel_spread",
    "rel_star_discrepancy": "rel_spread",
    "discrepancy": "discrepancy", "star_discrepancy": "discrepancy",
    "star_discrepancy_shifted": "discrepancy", "dispersion_shifted": "discrepancy",
    "gini": "discrepancy",
    "tukey": "depth", "projection": "depth",
    "mpsd": "other",
}


def main() -> None:
    apertus = pd.read_csv(APERTUS).set_index("variant")
    allen = pd.read_csv(ALLEN).set_index("variant")

    merged = apertus.join(allen[["r"]].rename(columns={"r": "r_allenai"}))
    merged = merged.sort_values("mean_r_overall", ascending=True)
    colors = [FAMILY_COLOR[FAMILY[v]] for v in merged.index]

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 7), sharey=True,
        gridspec_kw=dict(wspace=0.05),
    )

    ax = axes[0]
    ax.barh(merged.index, merged["mean_r_overall"], color=colors,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean Pearson r vs DA (12 langs, Apertus)")
    ax.set_title("Apertus: which SNR variant tracks decision accuracy?", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(merged.index, merged["r_allenai"], color=colors,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cross-corpus Pearson r (7 shared tasks)")
    ax.set_title("Transfer to AllenAI DataDecide", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    legend = [plt.Rectangle((0, 0), 1, 1, color=c, label=f) for f, c in FAMILY_COLOR.items()]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8, title="family",
                   title_fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"SNR variant performance: Apertus DA-correlation (left) vs AllenAI transfer (right) — pool {POOL}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
