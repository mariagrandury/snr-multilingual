"""Shared loading for the deck figures: the ladder report, one place."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
from evals.scripts.utils.configs import fineweb_language, load_languages  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402

# Every language the ladder pretrains on, up to the 50-language setting.
TRAINED_LANGS = load_languages()["groups"]["trained"]


def wide():
    return pd.read_csv(ladder_dir() / "ladder_report.csv", low_memory=False).dropna(subset=["cell"])


def cells(w):
    """One row per cell: axes plus the run-level flags."""
    return w.groupby("cell")[["size", "L", "arch", "scheme", "seed", "run__complete",
                              "run__diverged", "run__off_trend", "run__final_loss"]].first()


def finals(w):
    """Each cell's last checkpoint."""
    return w.sort_values("iter").groupby("cell").tail(1).set_index("cell")


def healthy(w):
    c = cells(w)
    return c[(c["run__complete"] == 1) & (c["run__diverged"] != 1) & (c["run__off_trend"] != 1)]


def bpb_matrix(w, arch="deep", scheme="A", seed=1904):
    """(size, L) -> BPB at the final checkpoint, one column per FineWeb-2 subset.

    Columns stay subset codes (`rus_Cyrl`, `dclm`) rather than language tags:
    several subsets share a tag (three Arabic variants, two Chinese scripts) and
    collapsing them would hide which one a number belongs to.
    """
    h = healthy(w)
    h = h[(h["arch"] == arch) & (h["scheme"] == scheme) & (h["seed"] == seed)]
    f = finals(w)
    cols = [c for c in w.columns if c.startswith("bpb__")]
    rows = {}
    for cell, r in h.iterrows():
        if cell not in f.index:
            continue
        v = f.loc[cell, cols].astype(float)
        v.index = [c[len("bpb__"):] for c in cols]
        rows[(r["size"], int(r["L"]))] = v
    return pd.DataFrame(rows).T.rename_axis(index=["size", "L"])


def subset_label(subset: str) -> str:
    """`rus_Cyrl` -> `rus`, `dclm` -> `eng`. Script kept only where a language
    has two (`zho_Hans` / `zho_Hant`)."""
    if subset == "dclm":
        return "eng"
    iso3, _, script = subset.partition("_")
    return iso3


def subset_lang(subset: str) -> str:
    """Project language tag of a subset, for grouping (`rus_Cyrl` -> `ru`)."""
    return fineweb_language(subset)
