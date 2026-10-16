"""Local redraws of a few paper figures, and rq2's `_font` twin.

Not part of the pipeline. Each figure's own script is re-run (one at a time)
with `Figure.savefig` replaced, so nothing but the figures named here is
written: no other figure, no table that exists already (a local pandas formats
floats differently from the cluster's) and no README block.

PLAIN   paper figures whose script changed after the last refresh: written as
        the script draws them (PNG in the analysis tree and here, SVG and PDF
        in ../figures_svg and ../figures_pdf), until the next refresh redraws them
FONT    rq2_font.png: rq2 with its axis labels at LABEL_PT, its tick labels
        at TICK_PT and its legends at LEGEND_PT once the figure is scaled to the width it is
        included at, the legends inside their panels (`rq2_twin`)

A figure `w` inches wide shown at `d` inches has its text scaled by d / w, so
text of PT * w / d points reads as PT on the page.

    python make_font_figures.py [script ...]     # from anywhere; the scripts run from src/signal-and-noise
"""
import os
import pickle
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SNR = REPO / "src" / "signal-and-noise"
LABEL_PT, TICK_PT, LEGEND_PT = 8.0, 7.0, 6.0
W = 455.24 / 72                                    # acl.sty \textwidth (in)
FONT = {"rq2": W}                                  # paper name -> included width
PLAIN = {"rq2", "app_rq02_bbpb_acc_gate", "app_rq02_agreement_correlation",
         "app_rq02_da_goal_multi_axes_bpb"}
A = "analysis/"
RUNS = [
    (A + "rq02_decision_accuracy/paper_rq2.py", "--pool", "predictivity", "--axes", "mono-axis"),
    (A + "rq01_scaling_predictability/tokens_seen.py", "--pool", "predictivity_seeds"),
    (A + "rq02_decision_accuracy/agreement.py", "--pool", "predictivity", "--paper"),
    ("bbpb_acc_gate",),
]


def bbpb_acc_gate():
    """app_rq02_bbpb_acc_gate from the committed tables: bench_bpb_da.py draws it
    only after reading the cluster-only per-item store."""
    import pandas as pd
    from analysis.rq02_decision_accuracy import bench_bpb_da as B
    out = B.DECISION_ACCURACY / "pretraining" / "predictivity"
    ref = pd.read_csv(out / "bench_bpb_da_size_by_acc_gate_multi_axes_paper.csv").groupby("size")["bpb_macro"].first()
    B.acc_gate_paper(pd.read_csv(out / "bench_bpb_da_size_by_acc_gate_multi_axes.csv"), ref.to_dict(), out)


def rq2_twin(fig):
    """rq2_font's panels: y from 0.3, DA-goal read as DA-target, the design-axis
    key of panel (a) top left in two columns (the long names in the first, so
    the second ends before the lines converging on the 1.7B reference), and the
    proxy-size key of panel (b) without its title."""
    for ax in fig.axes:
        ax.set_ylim(0.3, 1.02)
    fig.axes[2].set_ylabel("DA-target")
    ax = fig.axes[0]
    leg = ax.get_legend()
    names = [re.sub(r" \(.*\)", "", t.get_text()).replace("language count", "lang. count") for t in leg.get_texts()]
    entries = sorted(zip(leg.legend_handles, names), key=lambda e: e[1] != "all pairs")     # all pairs first
    rest = sorted(entries[1:], key=lambda e: -len(e[1]))
    blank = (Line2D([], [], alpha=0), "")       # tops the second column, so all pairs has the first row alone
    handles, labels = zip(*(entries[:1] + rest[:3] + [blank] + rest[3:]))
    x90 = min(t for t in ax.get_xticks() if t > 0)                     # the first size tick, 90M
    fs, hl = LEGEND_PT * fig.get_size_inches()[0] / FONT["rq2"], 0.8   # font() sets the same size
    at = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig, x=-hl * fs / 2, units="points")
    ax.legend(handles, labels, ncol=2, loc="upper left", frameon=False, borderaxespad=0, borderpad=0,
              bbox_to_anchor=(x90, 0.99), bbox_transform=at, fontsize=fs,     # the first markers on the 90M line
              columnspacing=1.5, handlelength=hl, handletextpad=0.3)
    leg = fig.axes[1].get_legend()
    fig.axes[1].legend(leg.legend_handles, [t.get_text() for t in leg.get_texts()], ncol=leg._ncols,
                       loc="lower right", frameon=False)


def font(fig, width):
    """Axis labels at LABEL_PT, tick labels at TICK_PT and legends at LEGEND_PT on the page;
    an x row of tick labels that then overlaps is rotated."""
    scale = fig.get_size_inches()[0] / width
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(LABEL_PT * scale); ax.yaxis.label.set_fontsize(LABEL_PT * scale)
        ax.tick_params(labelsize=TICK_PT * scale)
        if (leg := ax.get_legend()) is not None:
            for t in [*leg.get_texts(), leg.get_title()]:
                t.set_fontsize(LEGEND_PT * scale)
    fig.tight_layout()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        boxes = sorted((t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()), key=lambda b: b.x0)
        if any(a.x1 > b.x0 for a, b in zip(boxes, boxes[1:])):
            for t in ax.get_xticklabels():
                t.set_rotation(45); t.set_ha("right"); t.set_rotation_mode("anchor")
    fig.tight_layout()
    fig.align_xlabels(fig.axes)                     # one baseline for the panels' x labels


def patch():
    """savefig writes only PLAIN and FONT; to_csv and write_text leave an
    existing file alone and no README block is rewritten."""
    import pandas as pd
    from analysis import autodoc, style as S
    autodoc.rewrite = lambda path, edit: path
    names = {Path(p + ".png"): stem for p, stem in S.paper_names().items()}     # not with_suffix: "1.7B" has a dot
    names |= {S.PAPER / f"figures_{e}" / f"{stem}.{e}": stem for stem in PLAIN for e in ("svg", "pdf")}
    save = Figure.savefig

    def savefig(fig, fname, *args, **kw):
        path = Path(str(fname)).resolve() if isinstance(fname, (str, os.PathLike)) else None
        stem = names.get(path)
        if stem in PLAIN:
            save(fig, fname, *args, **kw)
            if path.suffix == ".png":
                shutil.copy(path, HERE / f"{stem}.png")
                print(f"PLAIN wrote {stem}.png")
        if stem in FONT and path.suffix == ".png":
            twin = pickle.loads(pickle.dumps(fig))       # a copy: the SVG and PDF saved next keep the plain figure
            rq2_twin(twin)
            font(twin, FONT[stem])
            save(twin, HERE / f"{stem}_font.png", dpi=200, bbox_inches="tight", facecolor=S.SURFACE)
            print(f"FONT wrote {stem}_font.png")
    Figure.savefig = savefig

    to_csv, write_text = pd.DataFrame.to_csv, Path.write_text
    pd.DataFrame.to_csv = lambda df, path=None, *a, **k: (None if isinstance(path, (str, os.PathLike)) and Path(path).exists()
                                                          else to_csv(df, path, *a, **k))
    Path.write_text = lambda path, *a, **k: None if path.exists() else write_text(path, *a, **k)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--run":    # one script, in this process, with savefig patched
        sys.path[:0] = [str(SNR), str(REPO / "src")]
        patch()
        sys.argv = sys.argv[2:]
        if sys.argv[0] == "bbpb_acc_gate":
            bbpb_acc_gate()
        else:
            runpy.run_path(sys.argv[0], run_name="__main__")
        return 0
    only = sys.argv[1:]
    env = {**os.environ, "PYTHONPATH": f"{SNR}:{REPO / 'src'}", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "HF_HUB_OFFLINE": "1", "SOURCE_DATE_EPOCH": "0", "MPLBACKEND": "Agg"}
    failed = [run[0] for run in RUNS                     # one at a time: never two of these at once
              if (not only or any(s in run[0] for s in only))
              and subprocess.run(["nice", "-n", "19", sys.executable, __file__, "--run", *run], cwd=SNR, env=env).returncode]
    print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
