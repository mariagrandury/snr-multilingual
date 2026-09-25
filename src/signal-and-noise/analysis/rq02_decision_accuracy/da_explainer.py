"""Decision accuracy explained on a TOY ladder: the three DA kinds, the two
pair sets and the values a cell can take.

Nothing here is measured. Four labelled design variants (Deep-A-T1, Deep-B-T1,
Shallow-A-T1, Deep-A-T3: depth, language list, sampling temperature) get
hand-written scores at five sizes and along one run, chosen so that the
rankings move with size and with training and one pair ties — the situations
the real kernel (`snr.metrics.decision_acc_fast`, the sign-of-difference rule;
`analysis.utils.agreement_measures`) has to handle. Every DA in the figure is
computed by that kernel on the toy scores, so the figure and the pipeline
cannot disagree on a definition.

  (a) the toy ladder: every variant's FINAL score at every size — the ranking
      the reference (1.7B) settles on is not the ranking a small size shows;
  (b) one proxy run (350M) along training with the reference's final order
      beside it — the ranking also moves within a run;
  (c) the six pairs decided three ways: DA-size (proxy final vs reference
      final), DA-ckpt (a checkpoint vs the proxy's OWN final), DA-goal (a
      checkpoint vs the reference final); a pair counts when the sign of the
      score difference agrees, a pair tied on one side is a miss (the rule of
      the kernel); the DA over every pair is the multi-axis reading, the DA
      over the pairs that move ONE design axis the mono-axis one (rule 15);
      with n pairs a DA is k/n, so the values are a lattice, and the cells at
      or above the reliability cut (0.66, `above_66_*`) are ringed;
  (d) where each definition lives on the (size × checkpoint) grid, and the two
      identities: DA-goal at the final checkpoint IS DA-size, and at the
      reference size DA-ckpt IS DA-goal.

    da_explainer.png / .csv   the toy scores (kind = score) and the decided
                              pairs (kind = pair) behind panel (c), with the
                              multi-axis (`da`) and mono-axis (`da_mono`) DA

    python analysis/rq02_decision_accuracy/da_explainer.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
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
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import THRESHOLDS  # noqa: E402
from analysis.utils import CKPT_DA_EARLY_FRACS, MIN_PAIRS, TARGET_SIZE, agreement_measures  # noqa: E402

GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis"
mpl.rcParams.update(S.RC)

SIZES = ["175M", "350M", "600M", "1B", "1.7B"]
PROXY, FRAC = "350M", 0.5
FRACS = [*CKPT_DA_EARLY_FRACS, 1.0]
CUT = min(THRESHOLDS)                 # 0.66, the reliability cut of the `above_66_*` figures (reliable_tasks.py)
# The four toy variants: depth, language list, sampling temperature (T1 is the
# default temperature and is written out, as the cell names do).
VARIANTS = {"Deep-A-T1": dict(arch="deep", list="A", T=1),
            "Deep-B-T1": dict(arch="deep", list="B", T=1),
            "Shallow-A-T1": dict(arch="shallow", list="A", T=1),
            "Deep-A-T3": dict(arch="deep", list="A", T=3)}
AXIS_LABEL = {"arch": "depth", "list": "language list", "T": "temperature"}
FAM_COLOUR = dict(zip(VARIANTS, [S.RAMP[3], S.RAMP[1], S.SERIES[1], S.SERIES[2]]))
FAM_MARK = dict(zip(VARIANTS, ["o", "s", "^", "D"]))
# Final scores per size: the reference orders B > A > AT3 > S, the 350M proxy
# orders A = AT3 > B > S (one tie, two swaps).
FINALS = pd.DataFrame({"175M": [.30, .29, .31, .28], "350M": [.36, .35, .34, .36], "600M": [.42, .43, .39, .41],
                       "1B": [.48, .50, .44, .47], "1.7B": [.55, .58, .50, .53]}, index=list(VARIANTS))
# The 350M run along training, ten evaluated tenths; the last column is FINALS["350M"].
RUN = pd.DataFrame([[.26, .27, .28, .29, .30, .32, .33, .34, .35, .36],
                    [.26, .28, .30, .31, .32, .33, .34, .34, .35, .35],
                    [.27, .29, .30, .31, .31, .32, .33, .33, .34, .34],
                    [.25, .26, .27, .28, .29, .31, .33, .34, .35, .36]], index=list(VARIANTS), columns=FRACS)
assert (RUN[1.0] == FINALS[PROXY]).all()


def moved_axis(a: str, b: str) -> str:
    differ = [k for k in ("arch", "list", "T") if VARIANTS[a][k] != VARIANTS[b][k]]
    return AXIS_LABEL[differ[0]] if len(differ) == 1 else "two axes"


def decide(proxy: pd.Series, ref: pd.Series) -> pd.DataFrame:
    """Every pair with the sign on both sides and whether it agrees — the
    kernel's rule, pair by pair."""
    rows = []
    for a, b in combinations(VARIANTS, 2):
        sp, sr = np.sign(proxy[a] - proxy[b]), np.sign(ref[a] - ref[b])
        rows.append({"family_a": a, "family_b": b, "axis": moved_axis(a, b), "proxy_sign": int(sp), "ref_sign": int(sr),
                     "tied": "both" if sp == sr == 0 else ("proxy" if sp == 0 else ("reference" if sr == 0 else "no")),
                     "match": int(sp == sr)})
    return pd.DataFrame(rows)


def order_text(s: pd.Series) -> str:
    """`Deep-B-T1 > Deep-A-T1 = Deep-A-T3 > Shallow-A-T1` from a score vector."""
    g = s.sort_values(ascending=False)
    out, prev = [], None
    for k, v in g.items():
        out.append(("= " if prev is not None and v == prev else ("> " if prev is not None else "")) + k)
        prev = v
    return " ".join(out)


def figure(out_dir: Path) -> pd.DataFrame:
    kinds = {"DA-size": (FINALS[PROXY], FINALS[TARGET_SIZE], f"{PROXY} final vs\n{TARGET_SIZE} final"),
             "DA-ckpt": (RUN[FRAC], FINALS[PROXY], f"{PROXY} at {G.chinchilla(FRAC)} vs\nits own final"),
             "DA-goal": (RUN[FRAC], FINALS[TARGET_SIZE], f"{PROXY} at {G.chinchilla(FRAC)} vs\n{TARGET_SIZE} final")}
    tables = {k: decide(p, r) for k, (p, r, _) in kinds.items()}
    stats = {k: agreement_measures(p.values, r.values) for k, (p, r, _) in kinds.items()}
    for k in kinds:                                    # the pair table and the kernel are one rule
        assert abs(tables[k]["match"].mean() - stats[k]["da"]) < 1e-12, k

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), gridspec_kw={"width_ratios": [1, 1.25]})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) the toy ladder's finals, the order per size listed under the curves
    for fam in VARIANTS:
        ax_a.plot(range(len(SIZES)), FINALS.loc[fam], marker=FAM_MARK[fam], ms=5, color=FAM_COLOUR[fam], lw=1.4, label=fam)
    orders = "\n".join(f"{s_:>5}: {order_text(FINALS[s_])}" for s_ in SIZES)
    ax_a.text(.98, .03, "the ranking at each size's final\n" + orders, transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=6.3, family="monospace", bbox=dict(facecolor=S.SURFACE, edgecolor=S.GRID, pad=3))
    ax_a.set_xticks(range(len(SIZES))); ax_a.set_xticklabels(SIZES); ax_a.set_ylim(.25, .62)
    ax_a.set_xlabel("size"); ax_a.set_ylabel("toy final score")
    ax_a.set_title("(a) four design variants at five sizes: the ranking moves with size", loc="left", fontsize=8.5)
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper left"); ax_a.grid(color=S.GRID, lw=.6); S.clean(ax_a)

    # (b) the proxy run along training, the reference's final order beside it
    x = np.array(FRACS) * G.CHINCHILLA_AT_FULL
    for fam in VARIANTS:
        ax_b.plot(x, RUN.loc[fam], marker=FAM_MARK[fam], ms=3.5, color=FAM_COLOUR[fam], lw=1.4, label=fam)
    ax_b.axvline(FRAC * G.CHINCHILLA_AT_FULL, color=S.MUTED, lw=.9, ls="--")
    ax_b.text(.02, .97, f"{TARGET_SIZE} final (the reference): {order_text(FINALS[TARGET_SIZE])}\n"
              f"{PROXY} final:                   {order_text(FINALS[PROXY])}\n"
              f"{PROXY} at {G.chinchilla(FRAC)} (read here):  {order_text(RUN[FRAC])}",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=6.5, family="monospace", color=S.INK)
    ax_b.set_xticks([1, 2, 3, 4, 5]); ax_b.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax_b.set_xlabel("training tokens of the proxy run (× Chinchilla)"); ax_b.set_ylabel(f"toy score along the {PROXY} run")
    ax_b.set_ylim(.24, .41)
    ax_b.set_title(f"(b) the {PROXY} run along training: the ranking also moves within a run", loc="left", fontsize=8.5)
    ax_b.grid(color=S.GRID, lw=.6); S.clean(ax_b)

    # (c) the decision table, the two pair sets, the cut ringed
    pairs = tables["DA-size"][["family_a", "family_b", "axis"]]
    ax_c.set_xlim(-2.1, 3.5); ax_c.set_ylim(len(pairs) + 2.6, -1.1); ax_c.axis("off")
    ax_c.set_title("(c) the six pairs decided three ways: sign agreement is a match, a one-sided tie a miss", loc="left", fontsize=8.5)
    sign = {1: ">", -1: "<", 0: "="}
    for j, k in enumerate(kinds):
        ax_c.text(j + 1, -.5, f"{k}\n{kinds[k][2]}", ha="center", va="bottom", fontsize=6.3, fontweight="bold")
    ax_c.text(-2.05, -.5, "pair (axis moved)\ncell: proxy sign · reference sign", ha="left", va="bottom", fontsize=6.3, fontweight="bold")
    for i, r in pairs.iterrows():
        mono = r["axis"] != "two axes"
        ax_c.text(-2.05, i, f"{'● ' if mono else '○ '}{r['family_a']} vs {r['family_b']}\n   ({r['axis']})", ha="left", va="center",
                  fontsize=6.3, color=S.INK if mono else S.MUTED)
        for j, k in enumerate(kinds):
            t = tables[k].iloc[i]
            face = {"1": "#dff2e8", "0": "#fbe3de"}[str(t["match"])]
            if t["tied"] != "no":
                face = "#fff1cc"
            ax_c.add_patch(plt.Rectangle((j + .55, i - .42), .9, .84, color=face, lw=0))
            txt = f"{sign[t['proxy_sign']]} · {sign[t['ref_sign']]}\n{'✓ match' if t['match'] else ('✗ one-sided tie' if t['tied'] != 'no' else '✗ miss')}"
            ax_c.text(j + 1, i, txt, ha="center", va="center", fontsize=6.3)
    y = len(pairs)
    for j, k in enumerate(kinds):
        st, tb = stats[k], tables[k]
        mono = tb[tb["axis"] != "two axes"]
        for row, (num, den, val) in enumerate([(st["concordant"] + st["tied_both"], st["n_pairs"], st["da"]),
                                               (int(mono["match"].sum()), len(mono), mono["match"].mean())]):
            yy = y + .15 + .85 * row
            ax_c.text(j + 1, yy, f"{num}/{den} = {val:.2f}", ha="center", va="center", fontsize=7.2, fontweight="bold")
            if val >= CUT:
                ax_c.add_patch(mpl.patches.Ellipse((j + 1, yy), .78, .62, fill=False, ec=S.SERIES[1], lw=1.4, zorder=4))
    ax_c.text(-2.05, y + .15, "DA multi-axis (all 6 pairs)", ha="left", va="center", fontsize=6.6, fontweight="bold")
    ax_c.text(-2.05, y + 1.0, "DA mono-axis (● the 3 single-axis pairs)", ha="left", va="center", fontsize=6.6, fontweight="bold")
    ax_c.text(-2.05, y + 1.85, f"DA = matching pairs / pairs, so a cell of n pairs takes the values k/n (3 pairs: 0, ⅓, ⅔, 1); a cell needs ≥ {MIN_PAIRS} "
              f"pairs (rule 5).\nRinged: at or above {CUT:g}, the reliability cut of the `above_66_*` figures (the pooled lines sum the pairs over tasks).",
              ha="left", va="center", fontsize=6.2, color=S.INK)

    # (d) where each definition lives on the size × checkpoint grid
    nF = len(FRACS)
    for i, s_ in enumerate(SIZES):
        for j, f in enumerate(FRACS):
            ax_d.add_patch(plt.Rectangle((j - .45, i - .42), .9, .84, facecolor="#f2f4f7", edgecolor="none"))
    ref_i = SIZES.index(TARGET_SIZE)
    ax_d.plot(nF - 1, ref_i, marker="*", ms=16, color=S.INK, zorder=5)
    ax_d.text(nF - .35, ref_i, "the reference:\n1.7B final", ha="left", va="center", fontsize=6.8, fontweight="bold")
    for i, s_ in enumerate(SIZES):
        if s_ == TARGET_SIZE:
            continue
        ax_d.annotate("", xy=(nF - 1.05, ref_i - .45), xytext=(nF - 1, i + .35),
                      arrowprops=dict(arrowstyle="-|>", color=S.RAMP[3], lw=1.5, shrinkA=0, shrinkB=0), zorder=4)
        ax_d.plot(nF - 1, i, "o", ms=9, color=S.RAMP[3], zorder=5)
        ax_d.annotate("", xy=(nF - 1.35, i), xytext=(2, i), arrowprops=dict(arrowstyle="-|>", color=S.SERIES[2], lw=1.3), zorder=3)
        ax_d.plot(range(nF - 1), [i] * (nF - 1), "o", ms=4, color=S.SERIES[2], zorder=4)
        ax_d.annotate("", xy=(nF - 1.25, ref_i - .3), xytext=(FRACS.index(FRAC), i + .3),
                      arrowprops=dict(arrowstyle="-|>", color=S.SERIES[1], lw=1.0, ls=(0, (3, 2)), alpha=.8), zorder=2)
    ax_d.plot(range(nF - 1), [ref_i] * (nF - 1), "o", ms=4, color=S.SERIES[2], zorder=4)
    ax_d.annotate("", xy=(nF - 1.35, ref_i), xytext=(2, ref_i), arrowprops=dict(arrowstyle="-|>", color=S.SERIES[2], lw=1.3), zorder=3)
    ax_d.set_xticks(range(nF)); ax_d.set_xticklabels([G.chinchilla(f) for f in FRACS], fontsize=6.5)
    ax_d.set_yticks(range(len(SIZES))); ax_d.set_yticklabels(SIZES); ax_d.set_xlim(-.6, nF + 1.9); ax_d.set_ylim(-2.3, len(SIZES) - .3)
    ax_d.set_xlabel("checkpoint of the proxy run (× Chinchilla)"); ax_d.set_ylabel("proxy size")
    ax_d.set_title("(d) where each definition lives, and the two identities", loc="left", fontsize=8.5)
    ax_d.legend(handles=[plt.Line2D([], [], color=S.RAMP[3], marker="o", lw=1.5, label="DA-size: a proxy's FINAL vs the 1.7B final"),
                         plt.Line2D([], [], color=S.SERIES[2], marker="o", lw=1.3, label="DA-ckpt: a checkpoint vs its OWN run's final"),
                         plt.Line2D([], [], color=S.SERIES[1], lw=1.0, ls=(0, (3, 2)), label=f"DA-goal: a checkpoint vs the 1.7B final (drawn at {G.chinchilla(FRAC)})"),
                         plt.Line2D([], [], ls="none", label="identities: DA-goal at the final checkpoint = DA-size (same two vectors);"),
                         plt.Line2D([], [], ls="none", label="at 1.7B, DA-ckpt = DA-goal (its own final IS the reference)")],
                fontsize=6.3, frameon=False, loc="lower left", handlelength=1.6)
    ax_d.tick_params(length=0); S.clean(ax_d, spines=())

    top = G._header(fig, "Decision accuracy, explained on a toy ladder (no measured number in this figure)",
                    "Four labelled design variants (depth × language list × sampling temperature; T1 is the default temperature) with hand-written "
                    "scores, chosen so that the rankings move with size and along a run and one pair ties. A decision is one pair of variants; DA is the "
                    "share of pairs the proxy orders like the reference (sign of the score difference on both sides: tied on both agrees, tied on one "
                    "is a miss — `decision_acc_fast`). The DAs in (c) are computed by the pipeline's kernel on the toy scores. In the real tables a cell "
                    f"is one (task, proxy size[, checkpoint]) over the ladder's families at the grid seed — {MIN_PAIRS} pairs at least (rule 5), gated "
                    "where either side is at chance (rule 1); the mono-axis set is the pairs that move exactly one design axis (rule 15).")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.2)
    S.save(fig, out_dir / "da_explainer.png", dpi=150)

    scores = pd.concat([FINALS.rename_axis("family").reset_index().melt(id_vars="family", var_name="size", value_name="score").assign(frac=1.0),
                        RUN.drop(columns=1.0).rename_axis("family").reset_index().melt(id_vars="family", var_name="frac", value_name="score").assign(size=PROXY)],
                       ignore_index=True).assign(kind="score")          # the 350M final is FINALS' row, not RUN's last column, once
    pairs_long = pd.concat([t.assign(kind="pair", da_kind=k, da=stats[k]["da"], da_mono=t.loc[t["axis"] != "two axes", "match"].mean())
                            for k, t in tables.items()], ignore_index=True)
    table = pd.concat([scores, pairs_long], ignore_index=True)
    table.to_csv(out_dir / "da_explainer.csv", index=False)
    return table


def generate_readme(pool: str, t: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel, gh = f"{stage}/{pool}", f"{GITHUB}/rq02_decision_accuracy/{stage}/{pool}"
    p = t[t["kind"] == "pair"]
    da = {k: g["da"].iloc[0] for k, g in p.groupby("da_kind")}
    mono = {k: g["da_mono"].iloc[0] for k, g in p.groupby("da_kind")}
    above = [f"{k} {w} {v:.2f}" for k in da for w, v in (("multi-axis", da[k]), ("mono-axis", mono[k])) if v >= CUT]
    body = "\n\n".join([
        "### Decision accuracy on a toy ladder",
        "**Toy, not measured.** Four labelled variants (Deep-A-T1, Deep-B-T1, Shallow-A-T1, Deep-A-T3) with hand-written scores; every DA in "
        "the figure is the pipeline's kernel run on them. (a) the finals per size and (b) one run along training show the two ways a ranking "
        "moves; (c) decides the six pairs three ways over both pair sets of rule 15 and rings the values at or above the reliability cut; "
        "(d) is where each definition sits on the size × checkpoint grid, with the two identities. "
        f"Regenerate with `python analysis/rq02_decision_accuracy/da_explainer.py --pool {pool}`.",
        f"![Decision accuracy explained on a toy ladder]({rel}/da_explainer.png)",
        "Key findings (definitions, so nothing to measure):",
        "\n".join([
            f"- On the toy, DA-size {da['DA-size']:.2f}, DA-ckpt {da['DA-ckpt']:.2f} and DA-goal {da['DA-goal']:.2f} over the six "
            f"multi-axis pairs; over the three mono-axis pairs {mono['DA-size']:.2f} / {mono['DA-ckpt']:.2f} / {mono['DA-goal']:.2f}. "
            "The tied pair (Deep-A-T1 = Deep-A-T3 at the proxy's final) is a miss wherever the other side decides it, an agreement only "
            "if both sides tie — `decision_acc_fast`'s convention, order-invariant.",
            f"- A cell of n pairs takes the values k/n: at the minimum of {MIN_PAIRS} pairs that is 0, ⅓, ⅔, 1, so a per-cell DA is read on "
            "its lattice and the figures draw the pooled ratio over tasks (`scale_convergence.py`) or the mean over cells (`by_L.py`), never one cell. "
            f"The ringed cells ({', '.join(above)}) are the ones the `above_66_*` filters would keep (cut {CUT:g}).",
            "- 0.5 is a coin flip on every untied pair; since a one-sided tie is a miss, an uninformative proxy sits below it — ≈ 0.47 "
            "on the ladder (7 % of pairs tied, the seed null of `seed_uncertainty.py`).",
            "- DA-goal at the final checkpoint is DA-size, and at the reference size DA-ckpt is DA-goal: the early-and-small grid's "
            "last column and last row are the other two figures' numbers.",
            "- `by transformation` is the mono-axis set split by the axis a pair moves; each group needs its own three pairs. On the ladder "
            "a (task, size) cell holds 0–4 pairs for the temperature axis, 0–6 for the list, 0–10 for depth and 0–39 for the language "
            "count, so the temperature and depth groups often fall below the minimum and are NaN "
            "(`early_small_by_transformation_*`, `scale_convergence_transformation_panels*`)."]),
        "Follow-ups:",
        "\n".join([
            "- A measured twin: the same panels on one real task (`hellaswag_de`, say) with the ladder's families, so the toy "
            "orders become the observed ones.",
            "- The lattice of the ladder's actual pair counts per cell (`median_pairs` in the scale-convergence CSVs), to show how coarse "
            "a per-cell DA is on each axis."]),
        f"Files: [`da_explainer.png`]({gh}/da_explainer.png), [`da_explainer.csv`]({gh}/da_explainer.csv)."])
    replace_block(DECISION_ACCURACY / "README.md", "da-explainer", body, f"da_explainer.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="only names the output folder and the README; the scores are the toy's")
    args = ap.parse_args()
    out = DECISION_ACCURACY / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out.mkdir(parents=True, exist_ok=True)
    t = figure(out)
    print(t[t["kind"] == "pair"].groupby("da_kind")[["da", "da_mono"]].first().to_string())
    generate_readme(args.pool, t)
