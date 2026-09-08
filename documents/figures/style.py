"""One palette for every deck figure.

Blue sequential ramp for magnitude, blue-to-red diverging for "better or worse
than", grey for "no data". Steps come from the validated ramp: the light end
clears 2:1 against the slide surface, adjacent steps differ enough to read.
Import this, never redefine a colour in a figure script.
"""
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

SEQ = LinearSegmentedColormap.from_list("snr_seq", ["#eaf2fd", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "snr_div", ["#8c1d18", "#d6a29e", "#f0efec", "#86b6ef", "#0d366b"])
for cm in (SEQ, DIV):
    cm.set_bad(NODATA)


def clean(ax, spines=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax


def title(fig, text, y=1.02, size=12):
    fig.suptitle(text, fontsize=size, color=INK, y=y)


def save(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    mpl.pyplot.close(fig)
    print(f"wrote {path}")
