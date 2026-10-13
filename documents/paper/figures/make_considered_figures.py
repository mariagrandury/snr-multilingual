"""The rq00 paper figures over every CONSIDERED benchmark (analysed, screened and
discarded), with the 3B rung added. Not part of the pipeline: the screened
benchmarks' gate lives on the cluster scratch (utils.PROBE_GATE_DIR), so this
reads a copy of it.

    analysed (`auto`) tasks: 90M-1.7B from the committed gate (as the paper
        figures), 3B from the probe gate's 3B column (trained runs only)
    screened / discarded: 600M, 1B, 1.7B, 3B real (re-read on the runs that
        train the language, `considered_gate`), 90M-350M set to below chance

Inputs: the committed rq00 gate (analysis/rq00_gate_and_curves/pretraining/
predictivity/above_random_*.csv), the ladder report cache
(src/signal-and-noise/data/ladder-report, for `considered_tasks`) and a copy of
the probe gate in src/signal-and-noise/data/probe-gate (copied read-only from
/iopsstor/scratch/cscs/mariagrandury/snr-probe-gate, last written 2026-10-09).
Outputs, in documents/paper/figures/:

    app_rq00_chance_full_considered_3b.png         language x benchmark: smallest size above chance
    app_rq00_chance_share_considered_3b.png        per benchmark, the share of its languages per size
    app_rq00_chance_share_considered_counts_3b.png the same in languages, in the paper's reading order
    app_rq00_chance_reformulation_considered_3b.png  original vs RF / LLM-RF, McNemar per size

Bold benchmark names are cloze-format (every task scores the answer text: the
tasks.json example or the harness's doc_to_choice; RF and LLM-RF versions are
cloze by construction); a * marks a benchmark that was only screened.

    cd src/signal-and-noise && PYTHONPATH=$PWD:$PWD/../../src python ../../documents/paper/figures/make_considered_figures.py
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "src" / "signal-and-noise" / "data"
os.environ.setdefault("SNR_PROBE_GATE_DIR", str(DATA / "probe-gate"))
os.environ.setdefault("SNR_LADDER_DIR", str(DATA / "ladder-report"))
OUT = DATA / "considered"
OUT.mkdir(exist_ok=True)
FIGURES = REPO / "documents" / "paper" / "figures"
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis.grids as G  # noqa: E402
import analysis.style as S  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import considered_gate, GATE_DIR  # noqa: E402
from analysis.rq00_gate_and_curves import panels as P  # noqa: E402
from analysis.rq00_task_reformulation import reformulations_gate as R  # noqa: E402
from analysis.utils import FAMILY_MERGE, PROBE_GATE_DIR, benchmark_family, considered_tasks, languages_only, variant  # noqa: E402

mpl.rcParams.update(S.RC)
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"]
FILLED = ["90M", "175M", "350M"]
POOL = "predictivity"
group = considered_tasks()

# --- the cell table: one row per (task, size) on trained runs ---------------------
g = considered_gate(POOL)                      # auto 90M-1.7B; probe/discarded 600M-3B
pg = PROBE_GATE_DIR / "pretraining" / POOL
m3 = pd.read_csv(pg / "above_random_mask.csv")[["task", "family", "language", "random_baseline", "3B"]]
pop3 = pd.read_csv(pg / "above_random_share.csv", usecols=["task", "3B__population"])
a3 = m3.merge(pop3, on="task")
a3 = a3[(a3["3B__population"] == "trained") & a3["3B"].notna() & a3["task"].map(group).eq("auto")]
a3 = languages_only(a3.rename(columns={"3B": "above"}).drop(columns="3B__population").assign(size="3B"))
a3 = a3.assign(above=a3["above"].astype(int), group="auto", format=a3["task"].map(lambda t: variant(t)[0]))
scr = g[g["group"] != "auto"]
fill = scr.drop_duplicates("task")[["task", "family", "language", "random_baseline", "group", "format"]]
fill = pd.concat([fill.assign(size=s, above=0) for s in FILLED], ignore_index=True)
cells = pd.concat([g, a3, fill], ignore_index=True)
cells["family"] = cells["task"].map(benchmark_family).map(lambda f: FAMILY_MERGE.get(f, f))   # one row per paper benchmark
cells = cells.drop_duplicates(["task", "size"], keep="first")
cells.to_csv(OUT / "considered_cells.csv", index=False)
screened_fams = set(cells.loc[cells["group"] != "auto", "family"]) - set(cells.loc[cells["group"] == "auto", "family"])
name = lambda f: G.paper_name(f) + ("*" if f in screened_fams else "")

# --- chance_full and chance_share: (family, language) level, as gate_panels ----------
ok = cells.groupby(["family", "language", "size"])["above"].mean().unstack("size").reindex(columns=SIZES)
ok = ok.ge(P.MIN_SHARE).where(ok.notna())
level = G.smallest_safe(ok).rename("level").rename_axis(["family", "language"]).reset_index() \
         .pivot(index="family", columns="language", values="level")
iso2 = P.load_languages()["fineweb_iso2"]
langs = [l for l in ["en"] + [iso2[s.split("_")[0]] for s in P.cell_fineweb_subsets(50, "A")] if l in level.columns]
level = level[langs]
order = list((level >= 0).sum(axis=1).sort_values(ascending=False, kind="stable").index)
G.level_heatmap(level.T, OUT / "app_rq00_chance_full_considered_paper.png", levels=SIZES, title="",
                cbar="Smallest size above chance", xlabel="", ylabel="", rows=langs, cols=order,
                names=("language", "benchmark"), name=name, cell_text=False, cell_w=0.18)
height = 0.17 * len(order) + 0.9
fig, ax = plt.subplots(figsize=(5.4, height))
tab = G.stack_ax(ax, level, "", levels=SIZES, rows=order, name=name,
                 xlabel="Share of languages above the chance threshold")
ax.set_ylim(len(order) - 0.5, -0.5)
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
fig.legend(handles, labels, fontsize=6, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0))
fig.tight_layout(rect=(0, 0.3 / height, 1, 1))
tab.drop(columns="panel").rename(columns={"row": "benchmark", "col": "level", "value": "share"}) \
   .to_csv(OUT / "app_rq00_chance_share_considered_paper.csv", index=False)
S.save_paper(fig, OUT / "app_rq00_chance_share_considered_paper")

# --- chance_reformulation: McNemar over the task mask ---------------------------------
# analysed tasks as the paper reads them (the committed mask), 3B from the probe gate;
# screened / discarded from the trained-run cells above
canon = pd.read_csv(GATE_DIR / "pretraining" / POOL / "above_random_mask.csv")
canon = canon[canon["task"].map(group).eq("auto")][["task", "family", "language", *SIZES[:-1]]]
canon = canon.merge(pd.read_csv(pg / "above_random_mask.csv")[["task", "3B"]], on="task", how="left")
wide = cells[cells["group"] != "auto"].pivot_table(index="task", columns="size", values="above").reindex(columns=SIZES)
meta = cells.drop_duplicates("task").set_index("task")[["family", "language"]]
mask = pd.concat([canon, wide.join(meta).reset_index()], ignore_index=True)
mc = R.mcnemar(mask, SIZES)
# the bars are a share of TASKS: name a benchmark with its task count where it has more tasks than languages
n_tasks = mc.groupby("family")["languages"].max()
n_lang = mc.groupby("family")["n_languages"].max()
mc["n_languages"] = mc["family"].map(lambda f: str(n_lang[f]) if n_tasks[f] == n_lang[f]
                                     else f"{n_lang[f]}, {n_tasks[f]}")
path = OUT / "app_rq00_chance_reformulation_considered_paper.png"
fig, a = plt.subplots(figsize=(6.4, 3.4))
R._twins_ax(a, mc, SIZES)
h, labels = a.get_legend_handles_labels()
rf = {l.removeprefix("rf twin at "): x for x, l in zip(h, labels) if l.startswith("rf twin")}
gm = [x for x, l in zip(h, labels) if l == "rfgm twin"]
orig = plt.Rectangle((0, 0), 1, 1, facecolor=S.GRID, edgecolor=S.MUTED, lw=.4)
star = plt.Line2D([], [], marker="$*$", ls="none", color=S.INK, ms=7)
# five columns of two, filled column by column: Original + 90M | 175M 350M | 600M 1B | 1.7B 3B | LLM-RF + McNemar
items = [(orig, "Original")] + [(rf[s], f"RF at {s}") for s in SIZES] + [(gm[0], "LLM-RF"), (star, f"McNemar p < {R.P_SIG}")]
a.legend([x for x, _ in items], [l for _, l in items], fontsize=6, frameon=False, ncol=5,
         loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.4)
a.set_ylabel("Share of tasks above threshold")
n_lab = mc.groupby("family")["n_languages"].max()
a.set_xticklabels([f"{G.paper_name(t.get_text())} ({n_lab[t.get_text()]})" for t in a.get_xticklabels()],
                  rotation=30, ha="right", fontsize=7)
fig.tight_layout()
mc.to_csv(path.with_suffix(".csv"), index=False)
S.save_paper(fig, path.with_suffix(""))

# --- the share figure, restyled, and its counts version ------------------------------
# cloze-format families (bold); RF / LLM-RF versions are cloze by construction
CLOZE = {'acp_bench_cloze', 'arc', 'arc_mt', 'bbh_cloze', 'blimp_nl', 'boolq', 'copal_id', 'cultural_bench_hard', 'french_bench', 'global_piqa_nonparallel_cloze', 'global_piqa_parallel_cloze', 'headqa', 'hellaswag', 'ibero_arc', 'ibero_cola', 'ibero_copa', 'ibero_nli', 'ibero_openbookqa', 'ibero_paraphrase', 'ibero_piqa', 'ibero_siqa', 'ibero_truthfulqa', 'include_v2_en', 'include_v2_og', 'kobest', 'm_truthfulqa_mc1', 'm_truthfulqa_mc2', 'mathqa', 'multiblimp', 'multiblimp-extra', 'oall_acva', 'oall_alghafa', 'oall_mt', 'openbookqa', 'paws', 'piqa', 'toksuite', 'toksuite_math', 'toksuite_stem', 'toxigen', 'truthfulqa-multi_mc1', 'truthfulqa-multi_mc2', 'truthfulqa_mc1', 'truthfulqa_mc2', 'turblimp', 'winogrande', 'xcopa', 'xnli', 'xstorycloze', 'xwinograd', 'zhoblimp'}
cloze = {f: f in CLOZE or f.startswith(("rf_", "rfgm_")) for f in set(cells["family"])}
screened = set(cells.loc[cells["group"] != "auto", "family"]) - set(cells.loc[cells["group"] == "auto", "family"])
name = lambda f: G.paper_name(f) + ("*" if f in screened else "")

ok = cells.groupby(["family", "language", "size"])["above"].mean().unstack("size").reindex(columns=SIZES)
ok = ok.ge(P.MIN_SHARE).where(ok.notna())
level = G.smallest_safe(ok).rename("level").rename_axis(["family", "language"]).reset_index() \
         .pivot(index="family", columns="language", values="level")
iso2 = P.load_languages()["fineweb_iso2"]
langs = [l for l in ["en"] + [iso2[s.split("_")[0]] for s in P.cell_fineweb_subsets(50, "A")] if l in level.columns]
level = level[langs]
order = list((level >= 0).sum(axis=1).sort_values(ascending=False, kind="stable").index)
blue = dict(zip(SIZES[:-1], [S.SEQ(x) for x in np.linspace(0.15, 0.95, len(SIZES) - 1)]))   # the paper's 6-level scale


# the counts figure's order: share above chance at 90M, then languages above chance (any size), then the smallest
# size with a pass, then languages, then analysed before screened
lv = level.reindex(order)
n_lang = lv.notna().sum(axis=1)
first = lv.where(lv >= 0).min(axis=1).fillna(len(SIZES))
key = pd.DataFrame({"s90": -(lv == 0).sum(axis=1) / n_lang, "above": -(lv >= 0).sum(axis=1), "first": first,
                    "n": -n_lang, "screened": [f in screened for f in lv.index]})
counts_order = list(key.sort_values(list(key.columns), kind="stable").index)


def draw(absolute: bool, stem: str, order: list = order) -> None:
    height = 0.17 * len(order) + 0.9
    fig, ax = plt.subplots(figsize=(5.4, height))
    tab = G.stack_ax(ax, level, "", levels=SIZES, rows=order, name=name,
                     xlabel="Number of languages above the chance threshold" if absolute
                     else "Share of languages above the chance threshold")
    total = level.reindex(order).notna().sum(axis=1).to_numpy()
    for cont in ax.containers:
        colour = S.SIZE_COLOR["3B"] if cont.get_label() == "3B" else blue.get(cont.get_label())
        for i, r in enumerate(cont.patches):
            if colour is not None:
                r.set_facecolor(colour)
            if absolute:
                r.set_x(r.get_x() * total[i]); r.set_width(r.get_width() * total[i])
    if absolute:
        ax.set_xlim(0, total.max())
    for lab, fam in zip(ax.get_yticklabels(), order):
        lab.set_fontweight("bold" if cloze.get(fam) else "normal")
    ax.set_ylim(len(order) - 0.5, -0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, fontsize=6, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.3 / height, 1, 1))
    out = tab.drop(columns="panel").rename(columns={"row": "benchmark", "col": "level", "value": "share"})
    out["languages"] = out["benchmark"].map(dict(zip(order, total))) * out["share"]
    out["cloze"] = out["benchmark"].map(lambda f: bool(cloze.get(f)))
    out.to_csv(OUT / f"{stem}.csv", index=False)
    S.save_paper(fig, OUT / stem)


draw(False, "app_rq00_chance_share_considered_paper")
draw(True, "app_rq00_chance_share_considered_counts_paper", counts_order)

for name in ("chance_full", "chance_share", "chance_share_considered_counts", "chance_reformulation"):
    stem = f"app_rq00_{name}" + ("" if name.endswith("counts") else "_considered")
    shutil.copy(OUT / f"{stem}_paper.png", FIGURES / f"{stem}_3b.png")   # _3b: the 3B rung is in
