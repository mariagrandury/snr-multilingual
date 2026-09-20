"""The paper's RQ2 figure: decision accuracy at every one of the ten evaluated
checkpoints, per proxy size, for per-language BPB and for benchmark tasks.

It lives here, not in the paper folder, because every figure the paper embeds
has to be produced by the analysis and only copied into
`documents/paper/figures` (`make_rq_figures.py`). It draws rq02's own table,
`da_early_small_per_task.csv` (compute_da.py): one row per (task, proxy size,
fraction of the proxy's run) with the DA of the proxy's ranking of the design
variants against the reference's final ranking, over at least MIN_PAIRS pairs
(rule 5, enforced in the kernel), on the nine evaluated checkpoints before the
final plus the final (rule 3). The reference's own rows are its ranking at an
earlier checkpoint against its final one; its self-comparison at 100 % is 1 by
construction and is left out.

A benchmark task counts at a proxy size only where it passes the above-random
gate there and at the reference (rule 1); BPB has no chance level and is never
gated. The mean per (size, fraction) is over the tasks that remain, and the
task count is written next to it (rule 13). The loader has already reduced the
tasks to parents in trained languages (rules 2 and 6).

    rq2.csv               size x fraction x group: mean DA, tasks
    rq2.png / .svg        the figure the paper includes: no title, the x axis in
                          Chinchilla multiples (every half-C point drawn, the
                          whole ones labelled), one boxed key inside the axes

    python analysis/rq02_decision_accuracy/paper_ten_checkpoints.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.utils import SMALL_SIZES, TARGET_SIZE, passes_gate, size_order  # noqa: E402

POOL, STAGE = "predictivity", "pretraining"
OUT = DECISION_ACCURACY / STAGE / POOL
COLORS = {"175M": "#86b6eb", "350M": "#438cdd", "600M": "#2464aa", "1B": "#123f78", "1.7B": "#061f3e"}
KEY = "#5c6066"             # neutral ink for the line-style samples: the style carries the meaning, not the colour
FULL = 5.0                  # every cell trains 5x Chinchilla, so 100% of training is 5C and the ten points are half-C steps
GROUPS = [("bpb", "", "per-language BPB"), ("benchmarks", 'stroke-dasharray="9,6"', "benchmark tasks")]


def summary() -> pd.DataFrame:
    """(size, frac, group) -> mean DA and task count over the gated tasks."""
    d = pd.read_csv(OUT / "da_early_small_per_task.csv").dropna(subset=["da"])
    d = d[~d["task"].isin(["bpb_macro", "train_loss"])]                       # whole-mixture aggregates, not tasks (rule 7)
    d["group"] = d["task"].str.startswith("bpb_").map({True: "bpb", False: "benchmarks"})
    mask = load_mask(POOL)
    ok = pd.Series(True, index=d.index)
    for size in d["proxy_size"].unique():                                       # rule 1: gated at the proxy and at the reference
        rows = (d["proxy_size"] == size) & (d["group"] == "benchmarks")
        ok[rows] = passes_gate(mask, d.loc[rows, "task"], size, TARGET_SIZE).to_numpy()
    d = d[ok & ~((d["proxy_size"] == TARGET_SIZE) & (d["frac"] >= 1.0))]       # the reference against itself is 1
    s = (d.groupby(["proxy_size", "frac", "group"]).agg(mean_da=("da", "mean"), tasks=("task", "nunique"))
         .reset_index().rename(columns={"proxy_size": "size"}))
    s["size"] = pd.Categorical(s["size"], size_order(s["size"].unique()))
    return s.sort_values(["group", "size", "frac"]).reset_index(drop=True)


def draw(name: str, s: pd.DataFrame, *, W=1000, H=620, margins=(100, 950, 45, 520)) -> None:
    left, right, top, bottom = margins
    X = lambda f: left + (float(f) - .1) / .9 * (right - left)          # noqa: E731
    Y = lambda v: bottom - (float(v) - .55) / .45 * (bottom - top)      # noqa: E731
    a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>', '<g font-family="DejaVu Sans, sans-serif" fill="#20252a">']

    def text(x, y, t, size=19, anchor="start", extra=""):
        a.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" {extra}>{t}</text>')

    def sample(x, y, w, stroke, dash, width=2.6):
        a.append(f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke="{stroke}" stroke-width="{width}" {dash}/>')

    # every measured fraction keeps its point; the labels are the whole Chinchilla multiples (rule 3)
    for f in (0.2, 0.4, 0.6, 0.8, 1.0):
        x = X(f); a.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#edf0f2"/>'); text(x, bottom + 30, f"{round(f * FULL):g}C", 17, "middle")
    for j in range(55, 101, 5):
        y = Y(j / 100); a.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e1e5e8"/>'); text(left - 13, y + 6, f"{j / 100:.2f}", 17, "end")
    a.append(f'<line x1="{left}" y1="{Y(.75)}" x2="{right}" y2="{Y(.75)}" stroke="#7e858c" stroke-dasharray="3,5" stroke-width="1.5"/>')
    for group, dash, _ in GROUPS:
        for size, c in COLORS.items():
            g = s[(s["group"] == group) & (s["size"] == size)].sort_values("frac")
            if g.empty:
                continue
            a.append(f'<polyline points="{" ".join(f"{X(f):.2f},{Y(v):.2f}" for f, v in zip(g["frac"], g["mean_da"]))}" fill="none" stroke="{c}" stroke-width="2.6" {dash}/>')
            for f, v in zip(g["frac"], g["mean_da"]):
                a.append(f'<circle cx="{X(f)}" cy="{Y(v)}" r="3.5" fill="{c}"/>')
    # the key sits in the empty bottom-right corner, below every curve: size colours left, line styles right
    bw, bh = 268, 104; bx, by = right - bw - 14, bottom - bh - 12
    a.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="white" fill-opacity="0.92" stroke="#c8ccd0" stroke-width="1" rx="4"/>')
    text(bx + 12, by + 18, "proxy size", 14, extra='fill="#5c6066"'); text(bx + 128, by + 18, "measurement", 14, extra='fill="#5c6066"')
    for i, (size, c) in enumerate(COLORS.items()):
        y = by + 34 + i * 14; sample(bx + 12, y, 20, c, "", 3.2); text(bx + 38, y + 5, size, 14)
    for i, (dash, label) in enumerate([(d, l) for _, d, l in GROUPS] + [('stroke-dasharray="3,5"', "DA = 0.75")]):
        y = by + 34 + i * 14; sample(bx + 128, y, 20, KEY, dash, 2.6); text(bx + 154, y + 5, label, 14)
    text((left + right) / 2, bottom + 62, "Proxy's training progress (Chinchilla multiples)", 21, "middle")
    text(29, (top + bottom) / 2, f"Mean decision accuracy vs. {TARGET_SIZE} final", 21, "middle", f'transform="rotate(-90 29 {(top + bottom) / 2})"')
    a.extend(["</g>", "</svg>"])
    (OUT / f"{name}.svg").write_text("\n".join(a) + "\n")
    subprocess.run(["convert", "-background", "white", "-density", "180", str(OUT / f"{name}.svg"), str(OUT / f"{name}.png")], check=True)
    print(f"wrote {name}.png/.svg")


def main() -> None:
    s = summary()
    s.to_csv(OUT / "rq2.csv", index=False)
    draw("rq2", s)
    final = s[s["frac"] >= 1.0].pivot(index="size", columns="group", values="mean_da").round(3)
    print("mean DA at the proxies' final checkpoint (tasks in rq2.csv):\n" + final.to_string())
    for size in SMALL_SIZES:
        g = s[(s["size"] == size) & (s["group"] == "benchmarks")]
        if g["tasks"].nunique() > 1:
            print(f"  rule 13: benchmark tasks at {size} range {g['tasks'].min()}-{g['tasks'].max()} across fractions (the gate, per size)")


if __name__ == "__main__":
    main()
