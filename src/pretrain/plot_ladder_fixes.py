#!/usr/bin/env python3
"""
Plot the L2 / deep / seed-1904 ladder as trained against the two small-rung
fixes compared in plan/90M-rung-anomaly.md ("Choosing the 90M config"):

  * option 1 — AdEMAMix beta3 memory = 20% of the run (--ademamix-beta3-factor 0.2)
  * option 2 — a smaller batch at the same token budget, grid beta3
               (--gbs 84 at 90M, --gbs 168 at 175M: the nearest valid layout)

each drawn two ways: fixing only 90M (joined to the 175M as trained) and fixing
both 90M and 175M (joined to the 350M as trained), against a power law fitted
on the clean rungs, 350M-1.7B.

Every value is read from the training logs. "Final" is the doc's rule: the
median loss over the last 2% of logged iterations. The diagnostic runs are kept
out of ladder_report by design, so their logs are named explicitly here. The
1B L2 cell is aromanou's and its Slurm log is not under TRAIN_LOGS, so its loss
is read from the TensorBoard events in its run directory.

    conda activate snr                                      # matplotlib lives there
    python src/pretrain/plot_ladder_fixes.py                # -> plan/90M-ladder-fixes.png
    python src/pretrain/plot_ladder_fixes.py --out x.png
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import struct
import sys
from pathlib import Path

# On the login node OpenBLAS fails to start its 48 threads and the numpy
# import under matplotlib then hangs instead of erroring.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ladder_report import LOSS_RE, NON_EMB, _fit  # noqa: E402
from pretrain_progress import CKPT_ROOT, TRAIN_LOGS  # noqa: E402

GRID_LOGS = {s: f"pretrain-{s}-L2-deep-seed1904-[0-9]*.out"
             for s in ("90M", "175M", "350M", "600M", "1.7B")}
GRID_TB = {"1B": CKPT_ROOT / "lm-1B-L2-deep-seed1904" / "logging" / "tensorboard"}
OPTION1 = {"90M": "pretrain-diag-90M-L2-deep-seed1904-beta3f0.2-[0-9]*.out",
           "175M": "pretrain-diag-175M-L2-deep-seed1904-beta3f0.2-[0-9]*.out"}
OPTION2 = {"90M": "pretrain-diag-90M-L2-deep-seed1904-gbs84-tok9.29B-*.out",
           "175M": "pretrain-diag-175M-L2-deep-seed1904-gbs168-tok17.63B-*.out"}
CLEAN = ("350M", "600M", "1B", "1.7B")   # the rungs the trend is fitted on


def final_loss(curve: dict[int, float]) -> float:
    """Median loss over the last 2% of logged iterations."""
    its = sorted(curve)
    return statistics.median(curve[i] for i in its[-math.ceil(len(its) * 0.02):])


def log_curve(pattern: str) -> dict[int, float]:
    """iter -> loss over every job of one run; a resumed run's later job wins."""
    curve = {}
    for f in sorted(TRAIN_LOGS.glob(pattern), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        for it, _target, loss in LOSS_RE.findall(f.read_text(errors="ignore")):
            curve[int(it)] = float(loss)
    if not curve:
        raise SystemExit(f"no training log matches {TRAIN_LOGS}/{pattern}")
    return curve


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(buf: bytes):
    """(field number, wire type, raw value) for each field of a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i:i + n], i + n
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield key >> 3, wire, value


def tb_curve(run_dir: Path, tag: str = "lm loss") -> dict[int, float]:
    """step -> scalar `tag` from TensorBoard event files. tensorboard is not
    installed in either env, so the TFRecord framing (length, crc, payload, crc)
    and the Event -> Summary -> Value messages are decoded directly."""
    curve = {}
    for path in sorted(run_dir.glob("events.out.tfevents.*")):
        data, i = path.read_bytes(), 0
        while i + 12 <= len(data):
            (n,) = struct.unpack("<Q", data[i:i + 8])
            event, i = data[i + 12:i + 12 + n], i + 12 + n + 4
            fields = {f: v for f, _w, v in _fields(event)}   # 2 = step, 5 = summary
            if 2 not in fields or 5 not in fields:
                continue
            for f, _w, value in _fields(fields[5]):
                parts = {f2: v2 for f2, w2, v2 in _fields(value)} if f == 1 else {}
                if parts.get(1) == tag.encode() and 2 in parts:   # 2 = simple_value
                    curve[fields[2]] = struct.unpack("<f", parts[2])[0]
    if not curve:
        raise SystemExit(f"no '{tag}' scalars under {run_dir}")
    return curve


def plot(grid: dict, opt1: dict, opt2: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    slope, icpt = _fit([(NON_EMB[s], grid[s]) for s in CLEAN])
    pred = lambda n: math.exp(icpt + slope * math.log(n))  # noqa: E731
    x = NON_EMB
    sizes = list(NON_EMB)
    ymax = 3.05

    # Reference categorical slots 1-3 and neutral chrome (dataviz palette).
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, muted, gridline, axis, surface = ("#0b0b0b", "#52514e", "#898781",
                                                 "#e1e0d9", "#c3c2b7", "#fcfcfb")

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ns = [x["90M"] * 0.85, x["1.7B"] * 1.1]
    ax.plot(ns, [pred(n) for n in ns], ls=(0, (4, 3)), lw=1.5, color=muted, zorder=1)
    ax.text(x["1.7B"] * 1.12, pred(x["1.7B"] * 1.1),
            f"power law fit on\n350M–1.7B (α = {-slope:.3f})", color=ink2, fontsize=8, va="center")

    trained = [s for s in sizes if s != "90M"]
    ax.plot([x[s] for s in trained], [grid[s] for s in trained], "-o", lw=2, ms=8,
            color=blue, mec=surface, mew=2, zorder=3)
    ax.annotate("", xy=(x["90M"], ymax - 0.01), xytext=(x["90M"], ymax - 0.13),
                arrowprops=dict(arrowstyle="-|>", color=blue, lw=2), zorder=3)
    ax.text(x["90M"] * 1.06, ymax - 0.07, f"as trained:\n{grid['90M']:.2f}, diverged",
            color=ink2, fontsize=8, va="center")

    for vals, color in ((opt1, orange), (opt2, aqua)):
        # fix only 90M: the step up to the 175M as trained
        ax.plot([x["90M"], x["175M"]], [vals["90M"], grid["175M"]], "-", lw=2, color=color, zorder=2)
        # fix 90M and 175M: rejoin the ladder at 350M
        ax.plot([x["90M"], x["175M"], x["350M"]], [vals["90M"], vals["175M"], grid["350M"]],
                ls=(0, (1, 2)), lw=1.5, color=color, zorder=2)
        ax.plot([x["90M"]], [vals["90M"]], "o", ms=8, color=color, mec=surface, mew=2, zorder=4)
        ax.plot([x["175M"]], [vals["175M"]], "o", ms=8, mfc=surface, mec=color, mew=2, zorder=4)
        ax.text(x["90M"] * 0.93, vals["90M"], f"{vals['90M']:.3f}", color=ink2, fontsize=8,
                ha="right", va="center")

    ax.text(x["175M"] * 1.25, grid["175M"] + 0.02,
            f"175M as trained {grid['175M']:.3f}\n"
            f"{grid['175M'] - pred(x['175M']):+.2f} vs the trend", color=ink, fontsize=8, va="bottom")

    handles = [
        Line2D([], [], color=blue, marker="o", lw=2, ms=7, mec=surface, mew=2,
               label=f"as trained (90M diverges: {grid['90M']:.2f})"),
        Line2D([], [], color=orange, marker="o", lw=2, ms=7, mec=surface, mew=2,
               label=f"option 1, β₃ memory 20% of run: 90M {opt1['90M']:.3f}, 175M {opt1['175M']:.3f}"),
        Line2D([], [], color=aqua, marker="o", lw=2, ms=7, mec=surface, mew=2,
               label=f"option 2, batch 84 / 168: 90M {opt2['90M']:.3f}, 175M {opt2['175M']:.3f}"),
        Line2D([], [], color=ink2, lw=2, label="solid: only 90M fixed"),
        Line2D([], [], color=ink2, lw=1.5, ls=(0, (1, 2)), marker="o", ms=7, mfc=surface,
               mec=ink2, mew=2, label="dotted, hollow: 90M and 175M fixed"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, labelcolor=ink2)

    ax.set_xscale("log")
    ax.set_xlim(x["90M"] * 0.62, x["1.7B"] * 2.6)
    ax.set_ylim(2.0, ymax)
    ax.set_xticks([x[s] for s in sizes])
    ax.set_xticklabels(sizes)
    ax.minorticks_off()
    ax.set_xlabel("non-embedding parameters", color=ink2)
    ax.set_ylabel("final training loss (L2, deep, seed 1904)", color=ink2)
    ax.grid(axis="y", color=gridline, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9)
    ax.set_title("Fixing only 90M breaks the ladder at 175M; fixing both restores it",
                 color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=surface)

    print(f"power law on {', '.join(CLEAN)}: alpha = {-slope:.4f}")
    print("| rung | trend | as trained | option 1 | option 2 |")
    for s in ("90M", "175M"):
        t = pred(x[s])
        print(f"| {s} | {t:.3f} | {grid[s]:.3f} ({grid[s] - t:+.2f}) | "
              f"{opt1[s]:.3f} ({opt1[s] - t:+.2f}) | {opt2[s]:.3f} ({opt2[s] - t:+.2f}) |")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path,
                   default=SCRIPT_DIR.parent.parent / "plan" / "90M-ladder-fixes.png")
    args = p.parse_args()
    grid = {s: final_loss(log_curve(pat)) for s, pat in GRID_LOGS.items()}
    grid |= {s: final_loss(tb_curve(d)) for s, d in GRID_TB.items()}
    grid = {s: grid[s] for s in NON_EMB}   # ladder order
    opt1 = {s: final_loss(log_curve(pat)) for s, pat in OPTION1.items()}
    opt2 = {s: final_loss(log_curve(pat)) for s, pat in OPTION2.items()}
    plot(grid, opt1, opt2, args.out)


if __name__ == "__main__":
    main()
