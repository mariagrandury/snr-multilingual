"""One palette for every analysis and deck figure.

Blue sequential ramp for magnitude, blue-to-red diverging for "better or worse
than", grey for "no data". Steps come from the validated ramp: the light end
clears 2:1 against the slide surface, adjacent steps differ enough to read.
Import this, never redefine a colour in a figure script.
``documents/figures/style.py`` re-exports it for the deck.
"""
import functools
import importlib.util
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, GRID, SURFACE = "#1b1b19", "#5c5c57", "#e3e2de", "#ffffff"
NODATA, ALERT = "#e8e6e1", "#b3261e"

# Ordinal ramp, light to dark. Use in this order, never cycled.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# Categorical slots, in fixed order, for series that are different KINDS of
# thing rather than steps of one magnitude. Never cycled past slot 3.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SIZE_COLOR = {"90M": "#cde2fb", "175M": "#86b6ef", "350M": "#3987e5",
              "600M": "#1c5cab", "1B": "#0d366b", "1.7B": "#061d3a"}
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]
# A cell's other two axes on a curve: its ladder (deep, shallow, swiglu, muon) takes
# the line width, its data build (the frame's `data`) the dash pattern, so
# colour stays free for the size.
LADDER_WIDTH = {"deep": 1.4, "shallow": 0.8, "swiglu": 2.2, "muon": 3.0}
DATA_DASH = {"A": "-", "B": "--", "AT3": ":", "BT3": (0, (5, 2)), "ZH": "-.",
               "ES": (0, (3, 1, 1, 1)), "DCLMP": (0, (6, 2, 2, 2)),
               "FWEB": (0, (5, 1, 1, 1, 1, 1))}         # dash-dot-dot: AT3 is the only dotted one

SEQ = LinearSegmentedColormap.from_list("snr_seq", ["#eaf2fd", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "snr_div", ["#8c1d18", "#d6a29e", "#f0efec", "#86b6ef", "#0d366b"])
for cm in (SEQ, DIV):
    cm.set_bad(NODATA)

# The paper's rcParams; a script applies them with ``mpl.rcParams.update(RC)``.
RC = {"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
      "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
      "svg.hashsalt": "snr"}   # fixed SVG element ids: an unchanged figure keeps its bytes (the date follows SOURCE_DATE_EPOCH)


def clean(ax, spines=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax


def title(fig, text, y=1.02, size=12):
    fig.suptitle(text, fontsize=size, color=INK, y=y)


PAPER = Path(__file__).resolve().parents[3] / "documents" / "paper"


@functools.cache
def paper_names() -> dict[str, str]:
    """Analysis figure (absolute path, no extension) -> the name the paper gives
    it: the PNG entries of make_rq_figures.FIGURES."""
    spec = importlib.util.spec_from_file_location("make_rq_figures", PAPER / "figures" / "make_rq_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {str(src.resolve()): stem for stem, (src, exts) in mod.FIGURES.items() if "png" in exts}


def save(fig, png, dpi=200):
    """Write the PNG `png`, and for a figure the paper embeds its SVG and PDF under
    the paper's name in documents/paper/figures_svg/ and figures_pdf/: the only
    SVG and PDF figures the analysis writes."""
    png = Path(png)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    stem = paper_names().get(str(png.resolve().with_suffix("")))
    for ext in ("svg", "pdf") if stem else ():
        (PAPER / f"figures_{ext}").mkdir(exist_ok=True)
        fig.savefig(PAPER / f"figures_{ext}" / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    mpl.pyplot.close(fig)
    print(f"wrote {png}" + (f" (paper {stem}.svg/.pdf)" if stem else ""))


ADVISORY = "advisory: "                    # rule 18's "try to": reported, not refused
PAPER_BANNED = ("—", "–", " - ", ";")      # rule 18: no dash used as punctuation, no ';' (a hyphen inside a word is fine)


def paper_problems(fig) -> list[str]:
    """Rule 18 (RULES.md): what a `_paper` figure must not carry. No figure
    title and no description text; axis labels capitalized; no dash or ';' in
    a label, panel title or legend entry; the same number of entries in every
    legend column, which is advisory (`ADVISORY`): the rule says to try."""
    out = []
    sup = fig._suptitle.get_text() if fig._suptitle is not None else ""
    if sup.strip():
        out.append(f"a figure title {sup!r}")
    sups = {id(t) for t in (getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None)) if t is not None}
    out += [f"a description {t.get_text()[:40]!r}" for t in fig.texts if id(t) not in sups and t.get_text().strip()]
    labels = [t.get_text() for t in fig.texts if id(t) in sups]
    labels += [lab for ax in fig.axes for lab in (ax.get_xlabel(), ax.get_ylabel())]
    for lab in filter(str.strip, labels):
        first = next((ch for ch in lab if ch.isalpha()), "")
        if first and not first.isupper():
            out.append(f"axis label not capitalized {lab!r}")
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    texts = labels + [ax.get_title(loc) for ax in fig.axes for loc in ("left", "center", "right")]
    texts += [t.get_text() for leg in legends for t in [*leg.get_texts(), leg.get_title()]]
    out += [f"{b!r} in {t!r}" for t in texts for b in PAPER_BANNED if b in t]
    for leg in legends:
        n, ncol = len(leg.get_texts()), leg._ncols
        if ncol > 1 and n % ncol:
            out.append(f"{ADVISORY}legend of {n} entries in {ncol} columns (unequal columns)")
    return out


def save_paper(fig, path, dpi=200):
    """Write a `_paper` figure: `path` without its extension, as PNG (and SVG and
    PDF for the paper, see `save`). Refuses (ValueError) a figure that breaks
    rule 18 instead of writing it, so a paper figure on disk is always one that
    passed."""
    path = Path(path)
    problems = paper_problems(fig)
    for p in (p for p in problems if p.startswith(ADVISORY)):
        print(f"!!! RULE 18: {path.name}: {p}")
    if bad := [p for p in problems if not p.startswith(ADVISORY)]:
        mpl.pyplot.close(fig)
        raise ValueError(f"RULE 18: {path.name}: " + " | ".join(bad))
    save(fig, path.parent / f"{path.name}.png", dpi)


def save_figure(fig, out_dir, name, dpi=200):
    """`save` by folder and stem: the PNG for the READMEs and the site."""
    save(fig, Path(out_dir) / f"{name}.png", dpi)
