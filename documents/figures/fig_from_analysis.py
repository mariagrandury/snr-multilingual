"""Rasterise the analysis figures the deck embeds.

analysis/report_figures/make_figures.py writes PDFs for the report. Slidev wants
PNGs, so this converts the three the deck shows. Run make_figures.py first.
"""
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parents[2]
PDFS = REPO / "src" / "signal-and-noise" / "analysis" / "report_figures" / "figures"
OUT = REPO / "documents" / "public" / "ladder"
NAMES = ("fig1_gate", "fig3_reliability_map", "fig4_subset_sweep")

if __name__ == "__main__":
    for name in NAMES:
        pymupdf.open(PDFS / f"{name}.pdf")[0].get_pixmap(dpi=200).save(OUT / f"{name}.png")
        print(f"wrote {OUT / name}.png")
