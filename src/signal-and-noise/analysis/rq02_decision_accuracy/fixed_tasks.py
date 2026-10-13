"""The rq02 paper figures on one task set per line.

A line of decision accuracy against size or training averages, at each point,
the tasks that are above chance there and have enough pairs, and the gate keeps
more tasks at larger sizes. So a line can rise because its population changed
(rule 13). Each figure here is redrawn over `utils.fixed_population`'s tasks:
per line, the tasks with a value at EVERY point of it, solid, with the
moving-population line of the committed figure dashed behind it.

    rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper   the paper's rq2 figure
    scale_convergence_da_size_multi_axes_fixed_tasks_paper                 the rq02 appendix figure

Each CSV has one row per (reading, line, point): `reading` is `moving` (the
committed figure's population) or `fixed`, `n_tasks` the tasks behind the
point; `documents/paper/figures/make_rq_appendix.py` reads the impact off it.

    python analysis/rq02_decision_accuracy/fixed_tasks.py --pool predictivity
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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.crossfit_reliable import (  # noqa: E402
    AXIS_LABEL, VARIANT, Cube, figure_rq2, pooled_size, rq2_lines, verdict)
from analysis.utils import NON_EMB, SMALL_SIZES, TARGET_SIZE, moved_axes, pair_sets  # noqa: E402


def rq2(c: Cube) -> pd.DataFrame:
    mono = pair_sets(c.attrs)["mono-axis"]
    axis_of = {p: AXIS_LABEL[moved_axes(c.attrs.loc[p[0]], c.attrs.loc[p[1]])[0]] for p in mono}
    rel = verdict(c, mono)
    return pd.concat([pd.DataFrame(rq2_lines(c, mono, axis_of, rel, fixed=f)).assign(reading=r)
                      for f, r in ((False, "moving"), (True, "fixed"))], ignore_index=True)


def size_twin(c: Cube) -> pd.DataFrame:
    """Pooled DA-size per proxy size over the multi-axis pairs, every task and
    the tasks with a value at every proxy size, per population."""
    multi = pair_sets(c.attrs)["multi-axis"]
    rows = []
    for pop, tasks in (("all benchmarks", c.bench), ("bpb", c.bpb)):
        das = np.stack([c.cell(multi, (s, 1.0), (TARGET_SIZE, 1.0), c.gate_ref[s])[0] for s in SMALL_SIZES], 1)
        for reading, keep in (("moving", tasks), ("fixed", tasks & np.isfinite(das).all(1))):
            rows += [r | {"population": pop, "line": pop, "reading": reading} for r in pooled_size(c, multi, keep)]
    return pd.DataFrame(rows)


def figure_size(t: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    xs = [NON_EMB[s] for s in SMALL_SIZES]
    for ax, (pop, lab) in zip(axs, (("all benchmarks", "Benchmarks"), ("bpb", "BPB"))):
        for reading, ls, col, name in (("fixed", "-", S.INK, "Same tasks at every size"),
                                       ("moving", (0, (3, 2)), S.MUTED, "Tasks above chance at each size")):
            g = t[(t["population"] == pop) & (t["reading"] == reading)].set_index("x").reindex(SMALL_SIZES)
            ax.plot(xs, g["da"], color=col, ls=ls, marker="o" if reading == "fixed" else None, ms=3.5, lw=1.5, label=name)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(SMALL_SIZES); ax.minorticks_off()
        ax.set_xlabel("Non-embedding parameters (log)"); ax.set_ylim(0, 1.02); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axs[0].set_ylabel(f"Decision reliability vs {TARGET_SIZE} final")
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    t.to_csv(path.with_suffix(".csv"), index=False)
    S.save_paper(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    args = ap.parse_args()
    out = DECISION_ACCURACY / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    c = Cube(args.pool)
    t = rq2(c)
    print(t[t["panel"] == "DA-size"].round(3).to_string(index=False))
    figure_rq2(t, out / f"rq2_da_all_{VARIANT}_transformation_mono_axis_fixed_tasks_paper", solid="fixed", dashed="moving",
               dashed_label="Tasks above chance at each point")
    t = size_twin(c)
    print(t.round(3).to_string(index=False))
    figure_size(t, out / "scale_convergence_da_size_multi_axes_fixed_tasks_paper")
