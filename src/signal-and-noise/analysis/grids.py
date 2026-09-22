"""Panel figures — the per-benchmark and per-language counterpart of every
aggregated figure.

An aggregate over all benchmarks hides which benchmark, or which language,
carries a result. Next to its aggregated figure every RQ therefore writes two
long figures from the same long table of cells, one subplot per page:

    <name>_by_benchmark.png   bits per byte first, then every benchmark alphabetically
    <name>_by_language.png    one subplot per language

`cells` is a long frame with one row per (task, row key, column key): the
columns `family` and `language`, the two grid keys and the value. A cell of a
subplot is the mean of the value over the subplot's tasks, annotated with the
number of tasks behind it, so a one-task 1.00 and a forty-task 0.96 do not
read alike. Every subplot keeps the full row and column order, and an empty
cell says why it is empty:

    white   no value: the benchmark does not exist in that language, or the
            cell was never trained or evaluated
    grey    filtered out: the value exists but the above-random gate removed it
            (rows of `cells` with `gated` true and no value)

Every figure writes the table it draws next to it: `<name>.csv` for
`<name>.png`, and for the figures that share one table `<name>.csv` for
`<name>_by_benchmark.png` and `<name>_by_language.png`. `note` is the line
under the title that says how a cell is computed.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

from analysis import style as S

FIRST = ("bpb", "loss")          # panels drawn before the alphabetical benchmarks
# The reformulated twins are ordinary benchmarks (`rf_belebele`), but reading
# them as `belebele-rf` keeps a twin next to its original instead of stranding
# every one of them under "r", and says which of the three sets a panel is.
_TWIN = re.compile(r"^(rfgm|rf)_(.+)$")


def display(key) -> str:
    """A benchmark family as a figure reads it: `rf_belebele` -> `belebele-rf`."""
    m = _TWIN.match(str(key))
    return f"{m.group(2)}-{m.group(1)}" if m else str(key)
NEVER = "#d6a29e"                # a level map's "never reached" (grey is kept for "filtered out")
GATED, NEVER_CODE = -2.0, -1.0   # a level map's codes below the levels' own indices
# Every run trains D(N) = 100 N tokens, five times the Chinchilla-optimal 20 N:
# a share of the run reads as a multiple of Chinchilla, 20 % = 1C ... 100 % = 5C.
CHINCHILLA_AT_FULL = 5


def chinchilla(frac: float) -> str:
    return f"{frac * CHINCHILLA_AT_FULL:g}C"


def mark_gated(df: pd.DataFrame, pool: str, size_col: str, value: str, reference: str | None = None) -> pd.DataFrame:
    """Blank `value` and set `gated` where the above-random gate filters the
    row out: the task is at chance at its own size, or at `reference` (the
    size it is ranked against). Tasks without a chance level (BPB) and sizes
    the gate has no column for are never gated."""
    from analysis.autodoc import CANONICAL_POOL
    from analysis.rq00_gate_and_curves.above_random import load_mask
    df = df.copy()
    mask = load_mask(pool)
    if mask is None:            # only the canonical pool has a gate report: same tasks and sizes, its seed
        print(f"  ({pool}: no above-random mask of its own, gating with {CANONICAL_POOL}'s)")
        mask = load_mask(CANONICAL_POOL)
    if mask is None:
        return df.assign(gated=False)
    at_chance = {(t, b) for b in mask.columns for t in mask.index[(mask[b] == 0).fillna(False)]}
    df["gated"] = [(t, b) in at_chance or (reference is not None and (t, reference) in at_chance)
                   for t, b in zip(df["task"], df[size_col])]
    df.loc[df["gated"], value] = np.nan
    return df


def add_meta(df: pd.DataFrame, task_col: str = "task") -> pd.DataFrame:
    """`family` and `language` from the task name; language aggregates and
    unassigned tasks are dropped (a panel is one benchmark or one language)."""
    from analysis.utils import assign_language, benchmark_family
    df = df.copy()
    df["family"] = df[task_col].map(benchmark_family)
    df["language"] = df[task_col].map(assign_language)
    return df[~df["language"].isin(["??", "multi"])]


def panel_order(keys, first=FIRST) -> list:
    keys = [k for k in pd.unique(pd.Series(list(keys))) if k not in ("??", "", None) and k == k]
    head = [k for k in first if k in keys]
    return head + sorted((k for k in keys if k not in head), key=display)


def _draw(ax, mat: pd.DataFrame, cnt: pd.DataFrame | None, *, vmin, vmax, cmap, fmt, fontsize, center=None,
          labels: dict | None = None, rotate: bool = False, gated: np.ndarray | None = None):
    vals = mat.to_numpy(dtype=float)
    if gated is not None:           # grey under the filtered-out cells, white under the ones with no value
        ax.imshow(np.where(gated & ~np.isfinite(vals), 1.0, 0.0), cmap=ListedColormap([S.SURFACE, S.NODATA]),
                  vmin=0, vmax=1, aspect="auto")
    cmap = cmap.copy(); cmap.set_bad(alpha=0)
    vals_drawn = np.ma.masked_invalid(vals)
    # a diverging scale is centred on `center` (chance, a ratio of 1), not on the middle of the range
    norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if center is not None else None
    im = ax.imshow(vals_drawn, cmap=cmap, aspect="auto", **({"norm": norm} if norm is not None else
                                                       {"vmin": vmin, "vmax": vmax}))

    def _dark(v) -> bool:                       # white ink on the saturated ends
        if center is None:
            return v > vmin + 0.7 * (vmax - vmin)
        return abs(v - center) > 0.6 * ((vmax - center) if v > center else (center - vmin))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            txt = labels.get(v, "") if labels is not None else fmt.format(v)
            if cnt is not None and np.isfinite(cnt.iloc[i, j]):
                txt += f"\n{int(cnt.iloc[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, rotation=90 if rotate else 0,
                    color="white" if _dark(v) else S.INK)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], fontsize=fontsize + .5, rotation=90 if rotate else 0)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels([display(r) for r in mat.index], fontsize=fontsize + .5)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    return im


def _csv_path(png: Path) -> Path:
    """`<name>.csv` for `<name>.png`, `<name>_by_benchmark.png` and `<name>_by_language.png`."""
    stem = png.stem
    for suffix in ("_by_benchmark", "_by_language"):
        stem = stem.removesuffix(suffix)
    return png.with_name(stem + ".csv")


def _header(fig, title: str, note: str) -> float:
    """Title and the note under it, laid out in inches so a 3-inch and a
    100-inch figure read alike; returns the top of the area left for the axes."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=10)
    if not note:
        return 1 - 0.3 / h
    width = max(60, int(fig.get_figwidth() * 17))          # characters of 7 pt text that fit the figure
    note = "\n".join(textwrap.fill(line, width) for line in note.split("\n"))
    fig.text(0.01, 1 - 0.27 / h, note, ha="left", va="top", fontsize=7, color=S.MUTED)
    return 1 - (0.36 + 0.13 * (note.count("\n") + 1)) / h


def panel_grid(cells: pd.DataFrame, path: Path, *, by: str, row: str, col: str, value: str, title: str,
               note: str = "", row_order: list | None = None, col_order: list | None = None, col_label=str,
               ncols: int = 4, vmin: float = 0.0, vmax: float = 1.0, cmap=None, fmt: str = "{:.2f}",
               cbar: str = "", xlabel: str = "", ylabel: str = "", counts: bool = True,
               first=FIRST, order: list | None = None, cell_w: float = 0.62, center: float | None = None,
               csv: bool = True) -> int:
    """One figure, one heat-map subplot per value of `by`. Returns the number
    of subplots (0 writes nothing). With languages on the columns pass a narrow
    `cell_w` (and `ncols=1`): the subplots stack as wide strips and the cell
    text is rotated. Rows of `cells` with `gated` true and no value are the
    grey cells. `csv=False` when the caller's table already carries the
    figure's name."""
    if "gated" not in cells:
        cells = cells.assign(gated=False)
    cells = cells[cells[value].notna() | cells["gated"]]
    keys = [k for k in order if k in set(cells[by])] if order is not None else panel_order(cells[by].unique(), first)
    if not keys:
        return 0
    rows = list(row_order or sorted(cells[row].unique(), key=str))
    cols = list(col_order or sorted(cells[col].unique(), key=str))
    ncols = min(ncols, len(keys)); nrows = (len(keys) + ncols - 1) // ncols
    w = cell_w * len(cols) + 1.5; h = (0.34 if cell_w >= 0.5 else 0.42) * len(rows) + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols + 0.8, h * nrows + 1.2), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(keys):]:
        ax.axis("off")
    im = None
    for ax, key in zip(flat, keys):
        g = cells[cells[by] == key]
        have = g.dropna(subset=[value])
        mat = have.pivot_table(index=row, columns=col, values=value, aggfunc="mean").reindex(index=rows, columns=cols)
        cnt = (have.pivot_table(index=row, columns=col, values=value, aggfunc="count").reindex(index=rows, columns=cols)
               if counts else None)
        gated = (g[g["gated"]].assign(_g=1.0).pivot_table(index=row, columns=col, values="_g", aggfunc="max")
                 .reindex(index=rows, columns=cols).notna().to_numpy())
        mat.columns = [col_label(c) for c in mat.columns]
        im = _draw(ax, mat, cnt, vmin=vmin, vmax=vmax, cmap=cmap or S.SEQ, fmt=fmt, fontsize=5.5 if cell_w >= 0.5 else 4.6,
                   rotate=cell_w < 0.5, center=center, gated=gated)
        ax.set_title(f"{display(key)}  ({have['task'].nunique()} tasks)" if "task" in g else display(key),
                     loc="left", fontsize=8)
    for ax in flat[:len(keys)]:
        ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
    legend = "white = no value" + (", grey = filtered out by the above-random gate" if cells["gated"].any() else "")
    if counts:
        legend = "small number = tasks behind the cell; " + legend
    top = _header(fig, title, (note + "\n" if note else "") + legend)
    fig.tight_layout(rect=(0, 0, 0.965, top))
    if im is not None:
        cax = fig.add_axes([0.972, 0.25, 0.008, 0.5])
        cb = fig.colorbar(im, cax=cax); cb.set_label(cbar, fontsize=8); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    if csv:
        keep = [c for c in dict.fromkeys(["task", "family", "language", by, row, col, value, "gated", "n_pairs"]) if c in cells]
        cells[keep].to_csv(_csv_path(path), index=False)
    S.save(fig, path, dpi=110 if len(keys) <= 60 else 80)
    print(f"Wrote {path.name} ({len(keys)} panels)")
    return len(keys)


def benchmark_and_language_panels(cells: pd.DataFrame, out_dir: Path, name: str, *, lang_ncols: int = 6, **kw) -> None:
    """`<name>_by_benchmark.png` and `<name>_by_language.png` from one table of cells, `<name>.csv`."""
    panel_grid(cells, out_dir / f"{name}_by_benchmark.png", by="family", **kw)
    panel_grid(cells, out_dir / f"{name}_by_language.png", by="language", ncols=lang_ncols, csv=False, **kw)


def level_heatmap(mats, path: Path, *, levels: list, title: str, note: str = "", cbar: str = "",
                  level_label=str, never: str = "—", xlabel: str = "language", ylabel: str = "benchmark",
                  rows: list | None = None, cols: list | None = None, separators: list = ()) -> None:
    """Language x benchmark maps whose cell is a *level* (the smallest size,
    Chinchilla multiple or compute at which something holds). `mats` is one
    frame or a dict label -> frame (stacked subplots, e.g. one per model
    size); a frame holds the level's index in `levels`, NEVER_CODE where no
    level reaches it, GATED where the gate filtered every level out (grey),
    NaN where there is no value (white). Rows follow the panel order, bits
    per byte first; every subplot keeps the same rows and columns. Writes
    `<name>.csv` next to the figure."""
    if isinstance(mats, pd.DataFrame):
        mats = {"": mats}
    mats = {k: m for k, m in mats.items() if m is not None and not m.empty}
    if not mats:
        return
    if rows is None:                # a square map passes one order for both so its diagonal is a thing with itself
        rows = panel_order(pd.unique(np.concatenate([m.index.to_numpy() for m in mats.values()])))
    if cols is None:
        cols = sorted(set().union(*[set(m.columns) for m in mats.values()]), key=str)
    colours = [S.SEQ(x) for x in np.linspace(0.15, 0.95, len(levels))]
    cmap = ListedColormap([S.NODATA, NEVER] + colours); cmap.set_bad(S.SURFACE)
    norm = BoundaryNorm(np.arange(-2.5, len(levels) + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(len(mats), 1, figsize=(0.30 * len(cols) + 3.4, (0.30 * len(rows) + 1.3) * len(mats) + 0.9),
                             squeeze=False)
    long = []
    for ax, (label, mat) in zip(axes[:, 0], mats.items()):
        mat = mat.reindex(index=rows, columns=cols)
        vals = mat.to_numpy(dtype=float)
        ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
        for k in separators:            # a square map's block boundaries, drawn on both axes
            ax.axhline(k - 0.5, color=S.INK, lw=0.6); ax.axvline(k - 0.5, color=S.INK, lw=0.6)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v) and v > GATED:
                    ax.text(j, i, never if v < 0 else level_label(levels[int(v)]), ha="center", va="center",
                            fontsize=4.4, rotation=90, color="white" if v >= 0.6 * len(levels) else S.INK)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6, rotation=90)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([display(r) for r in rows], fontsize=6.5)
        ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
        if label:
            ax.set_title(str(label), loc="left", fontsize=9)
        S.clean(ax, spines=()); ax.tick_params(length=0)
        t = mat.rename_axis(index="family", columns="language").stack().dropna().rename("level_index").reset_index()
        t["level"] = ["filtered out" if i == GATED else "never" if i < 0 else level_label(levels[int(i)]) for i in t["level_index"]]
        long.append(t.assign(panel=str(label)))
    any_gated = any((m == GATED).any().any() for m in mats.values())
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours + [NEVER] + ([S.NODATA] if any_gated else [])]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=S.SURFACE, edgecolor=S.GRID))
    axes[0, 0].legend(handles, [level_label(l) for l in levels] + [f"{never} never"]
                      + (["filtered out by the gate"] if any_gated else []) + ["no value"],
                      title=cbar, fontsize=6.5, title_fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    top = _header(fig, title, note)
    fig.tight_layout(rect=(0, 0, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(long)[["panel", "family", "language", "level_index", "level"]].to_csv(path.with_suffix(".csv"), index=False)
    S.save(fig, path, dpi=130)
    print(f"Wrote {path.name} ({len(mats)} panel(s), {len(rows)} benchmarks x {len(cols)} languages)")


def with_gated(level: pd.DataFrame, gated_pairs) -> pd.DataFrame:
    """A benchmark x language level matrix with GATED where the pair has no
    level at all and the gate is why (`gated_pairs`: the (family, language)
    pairs that had rows filtered out)."""
    level = level.copy()
    for fam, lang in gated_pairs:
        if fam not in level.index or lang not in level.columns or not np.isfinite(level.at[fam, lang]):
            level.loc[fam, lang] = GATED
    return level


def smallest_safe(levels_ok: pd.DataFrame) -> pd.Series:
    """Per row of a boolean/NaN frame whose columns are ordered levels: the
    index of the smallest level from which the condition holds at every
    larger level with information (NEVER_CODE if the largest informed level
    fails, NaN if no level has information). "Safe" means it does not come undone
    further up."""
    out = {}
    for key, r in levels_ok.iterrows():
        known = [(i, bool(v)) for i, v in enumerate(r.to_numpy()) if v == v and v is not None]
        if not known:
            out[key] = np.nan
            continue
        best = -1
        for i, ok in reversed(known):
            if not ok:
                break
            best = i
        out[key] = best
    return pd.Series(out, dtype=float)


# --- the condensed figure of an RQ -------------------------------------------
# `highlights.png`: a handful of small panels that say what the long grids say,
# built from these three axes. Each returns the table it drew; `save_highlights`
# writes them together as `highlights.csv`.

def matrix_ax(ax, mat: pd.DataFrame, title: str, *, cnt: pd.DataFrame | None = None, vmin=0.0, vmax=1.0, cmap=None,
              fmt="{:.2f}", center=None, xlabel: str = "", ylabel: str = "", gated: pd.DataFrame | None = None) -> pd.DataFrame:
    """A small heat map (e.g. benchmark x size, mean over languages). `gated`,
    a boolean frame aligned with `mat`, greys the cells the above-random gate
    filtered out (rule 12: white = no value, grey = gated)."""
    g = gated.reindex(index=mat.index, columns=mat.columns).fillna(False).to_numpy(dtype=bool) if gated is not None else None
    _draw(ax, mat, cnt, vmin=vmin, vmax=vmax, cmap=cmap or S.SEQ, fmt=fmt, fontsize=6.5, center=center, gated=g)
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel(xlabel, fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)
    t = mat.rename_axis(index="row", columns="col").stack().dropna().rename("value").reset_index()
    return t.assign(panel=title)


def stack_ax(ax, level: pd.DataFrame, title: str, *, levels: list, level_label=str, xlabel: str = "share of languages") -> pd.DataFrame:
    """A level map condensed: per benchmark, the share of its languages at
    each level (never in red, filtered out by the gate in grey)."""
    level = level.reindex(index=panel_order(level.index))
    codes = [float(i) for i in range(len(levels))] + [NEVER_CODE, GATED]
    names = [level_label(l) for l in levels] + ["never", "filtered out"]
    colours = [S.SEQ(x) for x in np.linspace(0.15, 0.95, len(levels))] + [NEVER, S.NODATA]
    share = pd.DataFrame({n: (level == c).sum(axis=1) for n, c in zip(names, codes)})
    total = share.sum(axis=1)
    share = share.div(total.where(total > 0), axis=0)
    left = np.zeros(len(share))
    for n, c in zip(names, colours):
        ax.barh(range(len(share)), share[n].fillna(0), left=left, color=c, label=n, height=0.8)
        left += share[n].fillna(0).to_numpy()
    ax.set_yticks(range(len(share))); ax.set_yticklabels([f"{f} ({int(n)})" for f, n in zip(share.index, total)], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_title(title, loc="left", fontsize=8.5); S.clean(ax); ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=6, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    t = share.rename_axis(index="row", columns="col").stack().dropna().rename("value").reset_index()
    return t.assign(panel=title)


def level_ax(ax, level: pd.DataFrame, title: str, *, levels: list, level_label=str, never: str = "—",
             xlabel: str = "", ylabel: str = "", fontsize: float = 6.5) -> pd.DataFrame:
    """A small level map (rows x columns, cell = index in `levels`, NEVER_CODE,
    GATED or NaN as in `level_heatmap`), drawn with the level's own colour
    scale and legend; returns its long table."""
    colours = [S.SEQ(x) for x in np.linspace(0.15, 0.95, len(levels))]
    cmap = ListedColormap([S.NODATA, NEVER] + colours); cmap.set_bad(S.SURFACE)
    norm = BoundaryNorm(np.arange(-2.5, len(levels) + 0.5, 1), cmap.N)
    vals = level.to_numpy(dtype=float)
    ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v) and v > GATED:
                ax.text(j, i, never if v < 0 else level_label(levels[int(v)]), ha="center", va="center",
                        fontsize=fontsize, color="white" if v >= 0.6 * len(levels) else S.INK)
    ax.set_xticks(range(vals.shape[1])); ax.set_xticklabels([str(c) for c in level.columns], fontsize=fontsize + .5)
    ax.set_yticks(range(vals.shape[0])); ax.set_yticklabels([str(r) for r in level.index], fontsize=fontsize + .5)
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel(xlabel, fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    any_gated = (vals == GATED).any()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours + [NEVER] + ([S.NODATA] if any_gated else [])]
    ax.legend(handles, [level_label(l) for l in levels] + [f"{never} never"] + (["filtered out"] if any_gated else []),
              fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    t = level.rename_axis(index="row", columns="col").stack().dropna().rename("value").reset_index()
    return t.assign(panel=title)


def level_lines_ax(ax, level: pd.DataFrame, title: str, *, levels: list, level_label=str, xlabel: str = "",
                   ylabel: str = "", colours: list | None = None, styles: list | None = None) -> pd.DataFrame:
    """The same level map as lines: one line per row, y = the columns in
    order, x = the level (never drawn one step past the last level, no value
    = a gap; rows jittered a little so coincident lines stay visible).
    Returns the long table."""
    cols = list(level.columns)
    colours = colours or [plt.cm.tab20(i % 20) for i in range(len(level))]
    jitter = np.linspace(-0.18, 0.18, len(level)) if len(level) > 1 else [0.0]
    for (name, r), c, dy, ls in zip(level.iterrows(), colours, jitter, styles or ["-"] * len(level)):
        x = r.to_numpy(dtype=float)
        x = np.where(x == GATED, np.nan, np.where(x == NEVER_CODE, len(levels), x))
        ax.plot(x, np.arange(len(cols)) + dy, marker="o", ms=3.5, lw=1.2, ls=ls, color=c, label=str(name))
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([str(c) for c in cols])
    ax.set_xticks(range(len(levels) + 1)); ax.set_xticklabels([level_label(l) for l in levels] + ["never"])
    ax.set_xlim(-0.5, len(levels) + 0.5); ax.set_ylim(-0.5, len(cols) - 0.5); ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlabel(xlabel, fontsize=7.5); ax.set_ylabel(ylabel, fontsize=7.5)
    ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1 if len(level) <= 12 else 2)
    t = level.rename_axis(index="row", columns="col").stack().dropna().rename("value").reset_index()
    return t.assign(panel=title)


def rank_ax(ax, values: pd.Series, title: str, *, k: int = 8, xlabel: str = "", fmt="{:.2f}", ref: float | None = None,
            ascending: bool = False) -> pd.DataFrame:
    """The k best and the k worst of a series (a language or a benchmark
    each); `ascending` when the smallest value is the best one."""
    v = values.dropna().sort_values(ascending=ascending)
    shown = v if len(v) <= 2 * k else pd.concat([v.head(k), v.tail(k)])
    colours = [S.RAMP[2]] * len(shown) if len(v) <= 2 * k else [S.RAMP[2]] * k + [S.SERIES[1]] * k
    ax.barh(range(len(shown)), shown.to_numpy(), color=colours, height=0.75)
    for i, x in enumerate(shown.to_numpy()):
        ax.text(x, i, " " + fmt.format(x), va="center", ha="left" if x >= 0 else "right", fontsize=6)
    if ref is not None:
        ax.axvline(ref, color=S.MUTED, lw=.8, ls=":")
    ax.set_yticks(range(len(shown))); ax.set_yticklabels([str(i) for i in shown.index], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel(xlabel, fontsize=7.5); S.clean(ax); ax.tick_params(axis="y", length=0)
    ax.set_title(title + (f"  (best {k}, worst {k} of {len(v)})" if len(v) > 2 * k else ""), loc="left", fontsize=8.5)
    return shown.rename("value").rename_axis("row").reset_index().assign(col="", panel=title)


def save_highlights(fig, out_dir: Path, title: str, note: str, tables: list, name: str = "highlights") -> None:
    """`<name>.png` and `<name>.csv`: the highlights page, or one of its panels on its own."""
    top = _header(fig, title, note)
    fig.tight_layout(rect=(0, 0, 1, top))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables)[["panel", "row", "col", "value"]].to_csv(out_dir / f"{name}.csv", index=False)
    S.save(fig, out_dir / f"{name}.png", dpi=150)
