"""Seven research questions, one figure each.

Statistics and general analysis:
  A1 which benchmarks scale predictably   A2 which benchmarks are stable
  A3 which languages gain from scale      A4 which benchmarks separate our choices

Decision accuracy and its surrogates:
  B1 how early we can call the final ranking
  B2 how small an evaluation suite can be
  B3 which surrogate carries the decision
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S, data

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
P = "pretraining/predictivity"
OUT = REPO / "documents" / "public" / "ladder"
NPARAMS = {"90M": 9.0e7, "175M": 1.75e8, "350M": 3.5e8, "600M": 6.0e8, "1B": 1.0e9}
SIZES = ["175M", "350M", "600M", "1B"]          # 90M is off the ladder, see Finding 2


def _scores():
    return pd.read_csv(ANALYSIS / f"rq00_acc_vs_flops/{P}/above_random_scores.csv")


def _variants():
    return pd.read_csv(ANALYSIS / f"rq02_snr_definition/{P}/snr_variants_per_task.csv")


def _fam_of(v):
    from analysis.utils import benchmark_family
    return v["task"].map(benchmark_family)


# --- A1 ---------------------------------------------------------------------
def a1_scaling(out):
    """Fit each task's score against log parameters. R² says how much of the
    movement the fit explains, Spearman says whether it moves the right way."""
    d = _scores()
    x_all = np.log10([NPARAMS[s] for s in SIZES])
    rows = []
    for _, r in d.iterrows():
        y = r[SIZES].astype(float).to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        x, yy = x_all[ok], y[ok]
        b, a = np.polyfit(x, yy, 1)
        pred = b * x + a
        ss_res = float(((yy - pred) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        rho = spearmanr(x, yy).statistic if len(set(yy)) > 1 else np.nan
        rows.append({"task": r["task"], "family": r["family"],
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rho": rho, "slope": b})
    f = pd.DataFrame(rows)
    g = f.groupby("family").agg(r2=("r2", "median"), rho=("rho", "median"),
                                n=("task", "size"))
    g = g[g["n"] >= 2].sort_values("r2")

    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(g) + 2.0))
    y = np.arange(len(g))
    ax.hlines(y, 0, g["r2"], color=S.GRID, lw=1.4, zorder=1)
    ax.scatter(g["r2"], y, s=64, color=S.RAMP[2], zorder=3, label="R² of the fit")
    ax.scatter(g["rho"], y, s=64, marker="D", color=S.SERIES[1], zorder=3,
               label="Spearman ρ against size")
    ax.axvline(0, color=S.MUTED, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{i}  ({int(n)})" for i, n in zip(g.index, g["n"])],
                                         fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("median across the family's tasks", fontsize=9.5, color=S.MUTED)
    ax.annotate("bits per byte and loss fall as models grow, so ρ = −1 is their ideal",
                (.5, -.11), xycoords="axes fraction", ha="center", fontsize=8.5,
                color=S.MUTED)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left",
              bbox_to_anchor=(0, 1.06), ncol=2)
    S.title(fig, "Which benchmarks scale predictably, 175M to 1B", y=1.02)
    S.save(fig, out / "rq_a1_scaling.png")
    return f


# --- A2 ---------------------------------------------------------------------
def a2_stability(out):
    """Relative noise at 1B: the spread over a run's late checkpoints divided by
    the mean. Small means the number is repeatable."""
    v = _variants()
    v = v.assign(family=_fam_of(v))
    v = v[["family", "noise_rel_std_1B", "signal_rel_std_1B"]].dropna(subset=["noise_rel_std_1B"])
    g = v.groupby("family").agg(noise=("noise_rel_std_1B", "median"),
                                n=("noise_rel_std_1B", "size")).sort_values("noise")
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(g) + 1.9))
    y = np.arange(len(g))
    ax.barh(y, g["noise"], .58, color=S.RAMP[1])
    for i, (val, n) in enumerate(zip(g["noise"], g["n"])):
        ax.annotate(f"{val:.3f}  ({int(n)})", (val, i), xytext=(6, 0), va="center",
                    textcoords="offset points", fontsize=8.5, color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=9)
    ax.set_xlabel("relative noise over the last checkpoints, lower is steadier",
                  fontsize=9.5, color=S.MUTED)
    ax.set_xlim(0, g["noise"].max() * 1.32)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=S.RAMP[1], label="median over the family's tasks at 1B")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower right")
    S.title(fig, "Which benchmarks give the same answer twice", y=1.02)
    S.save(fig, out / "rq_a2_stability.png")


# --- A4 ---------------------------------------------------------------------
def a4_signal(out):
    """Signal is the spread across the models we want to tell apart. It is only
    useful above the noise, so both are drawn."""
    v = _variants()
    v = v.assign(family=_fam_of(v))
    v = v[["family", "signal_rel_std_1B", "noise_rel_std_1B"]].dropna()
    g = v.groupby("family").agg(signal=("signal_rel_std_1B", "median"),
                                noise=("noise_rel_std_1B", "median"),
                                n=("signal_rel_std_1B", "size"))
    g["ratio"] = g["signal"] / g["noise"]
    g = g.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(8.8, 0.34 * len(g) + 2.0))
    y = np.arange(len(g))
    for i, (_, r) in enumerate(g.iterrows()):
        lo, hi = sorted((r["signal"], r["noise"]))
        ax.hlines(i, lo, hi, color=S.GRID, lw=1.6, zorder=1)
    ax.scatter(g["noise"], y, s=62, color=S.SERIES[1], zorder=3, label="noise")
    ax.scatter(g["signal"], y, s=62, color=S.RAMP[3], zorder=3, label="signal")
    for i, r in enumerate(g["ratio"]):
        ax.annotate(f"{r:.1f}×", (max(g['signal'].iloc[i], g['noise'].iloc[i]), i),
                    xytext=(8, 0), va="center", textcoords="offset points",
                    fontsize=8.5, color=S.INK if r >= 1 else S.MUTED)
    ax.set_yticks(y); ax.set_yticklabels([f"{i}  ({int(n)})" for i, n in zip(g.index, g["n"])],
                                         fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("relative spread at 1B, log scale", fontsize=9.5, color=S.MUTED)
    ax.annotate("the number on the right is signal ÷ noise. Above 1× the benchmark "
                "separates the models, below it does not",
                (.5, -.14), xycoords="axes fraction", ha="center", fontsize=8.5,
                color=S.MUTED)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper right",
              bbox_to_anchor=(1.0, 1.07), ncol=2)
    S.title(fig, "Does the benchmark separate the models we are comparing", y=1.02)
    S.save(fig, out / "rq_a4_signal.png")


# --- A3 ---------------------------------------------------------------------
def a3_languages(out):
    """Baseline against gain, per validation language, on the L50 chain. The
    three patterns the team named are the three regions of this plot."""
    m = data.bpb_matrix(data.wide())
    chain = [s for s in SIZES if (s, 50) in m.index]
    base, top = m.loc[(chain[0], 50)], m.loc[(chain[-1], 50)]
    d = pd.DataFrame({"base": base, "gain": base - top}).dropna()
    med_b, med_g = d["base"].median(), d["gain"].median()

    rho = spearmanr(d["base"], d["gain"]).statistic
    # Three kinds of language behaviour, so three categorical slots.
    HARD_GAIN, EASY_GAIN, FLAT = S.SERIES[0], S.SERIES[2], S.SERIES[1]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axvline(med_b, color=S.GRID, lw=1); ax.axhline(med_g, color=S.GRID, lw=1)
    def slot(r):
        if r.gain <= med_g:
            return FLAT
        return HARD_GAIN if r.base > med_b else EASY_GAIN
    d["colour"] = [slot(r) for r in d.itertuples()]
    for c, g in d.groupby("colour"):
        ax.scatter(g["base"], g["gain"], s=28, color=c, alpha=.85, lw=0)
    for lang in d["gain"].nlargest(4).index.tolist() + d["gain"].nsmallest(3).index.tolist():
        ax.annotate(data.subset_label(lang), (d.loc[lang, "base"], d.loc[lang, "gain"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8, color=S.INK)
    ax.set_xlabel(f"bits per byte at {chain[0]}, further right means the language starts harder",
                  fontsize=9.5, color=S.MUTED)
    ax.set_ylabel(f"bits per byte saved going {chain[0]} to {chain[-1]}",
                  fontsize=9.5, color=S.MUTED)
    ax.annotate(f"Spearman ρ = {rho:.2f} between where a language starts and what it gains",
                (.5, -.16), xycoords="axes fraction", ha="center", fontsize=9, color=S.MUTED)
    ax.grid(color=S.GRID, lw=.6); ax.set_axisbelow(True)
    S.clean(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=HARD_GAIN,
               label="starts hard, gains a lot"),
        Line2D([], [], marker="o", ls="none", color=EASY_GAIN,
               label="starts easy, gains a lot"),
        Line2D([], [], marker="o", ls="none", color=FLAT,
               label="gains little, wherever it starts")],
        frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left")
    S.title(fig, "Which languages gain most from a bigger model, at 50 languages", y=1.02)
    S.save(fig, out / "rq_a3_languages.png")
    return d


# --- B1 ---------------------------------------------------------------------
def b1_early(out):
    """Decision accuracy of an early checkpoint against the run's own final
    ranking, per benchmark family, at each size."""
    v = _variants()
    v = v.assign(family=_fam_of(v))
    fracs = [("f20", 20), ("f40", 40), ("f60", 60), ("f80", 80)]
    rows = []
    for size in SIZES:
        for f, pct in fracs:
            c = f"decision_acc_ckpt_{f}_{size}"
            if c in v:
                sub = v[["family", c]].dropna()
                for fam, g in sub.groupby("family"):
                    rows.append({"size": size, "pct": pct, "family": fam,
                                 "da": g[c].mean(), "n": len(g)})
    d = pd.DataFrame(rows)
    top = SIZES[-1]
    cur = d[d["size"] == top]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    # Bits per byte against the benchmark families. The families are one kind of
    # thing, so they share a muted colour rather than six invented hues.
    bench = [f for f in cur["family"].unique() if f not in ("bpb", "loss")]
    for fam in bench:
        g = cur[cur.family == fam].sort_values("pct")
        if len(g) >= 2:
            ax.plot(g["pct"], g["da"], marker="o", ms=4, lw=1.3, color=S.GRID, zorder=2)
    spread = cur[cur.family.isin(bench)].groupby("pct")["da"]
    ax.plot(spread.median().index, spread.median().values, marker="s", ms=7, lw=2.0,
            color=S.SERIES[1], zorder=4,
            label=f"benchmark families, median of {len(bench)}")
    g = cur[cur.family == "bpb"].sort_values("pct")
    ax.plot(g["pct"], g["da"], marker="o", ms=9, lw=2.8, color=S.RAMP[3], zorder=5,
            label=f"bits per byte ({int(g['n'].max())} tasks)")

    ax.axhline(.5, color=S.MUTED, ls="--", lw=1)
    ax.annotate("coin flip", (81, .5), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=S.MUTED, ha="right")
    ax.set_xticks([20, 40, 60, 80])
    ax.set_xticklabels(["20 %", "40 %", "60 %", "80 %"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("checkpoint, as a share of the full run", fontsize=9.5, color=S.MUTED)
    ax.set_ylabel("agrees with the run's own final ranking", fontsize=9.5, color=S.MUTED)
    ax.grid(axis="y", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(handles=[
        Line2D([], [], color=S.RAMP[3], lw=2.8, marker="o", ms=9,
               label=f"bits per byte ({int(g['n'].max())} tasks)"),
        Line2D([], [], color=S.SERIES[1], lw=2.0, marker="s", ms=7,
               label=f"benchmark families, median of {len(bench)}"),
        Line2D([], [], color=S.GRID, lw=1.3, marker="o", ms=4,
               label="one benchmark family")],
        frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower right", ncol=1)
    S.title(fig, f"How early the final ranking is already decided, at {top}", y=1.02)
    S.save(fig, out / "rq_b1_early.png")
    return d


# --- B2 ---------------------------------------------------------------------
def b2_suite_size(out):
    """How many benchmark tasks it takes to rank the models the way the full
    per-language bits-per-byte picture does. Tasks are added best first, by how
    well each one alone tracks that reference."""
    from analysis.utils import build_snr_pool
    pool = build_snr_pool("predictivity")
    fin = pool.loc[pool.groupby(["model", "task"])["step"].idxmax()]
    bench = fin[fin["kind"] != "bpb"].pivot_table(index="model", columns="task",
                                                  values="primary_score")
    bpb = fin[fin["kind"] == "bpb"].pivot_table(index="model", columns="task",
                                                values="primary_score")
    # Reference: rank the models by their mean per-language bits per byte, which
    # is lower-is-better, so flip the sign to make every ranking higher-is-better.
    ref = -bpb.mean(axis=1).dropna()
    bench = bench.loc[bench.index.intersection(ref.index)]
    ref = ref.loc[bench.index]
    keep = bench.columns[bench.notna().sum() >= max(6, int(.6 * len(bench)))]
    bench = bench[keep]

    solo = {t: spearmanr(bench[t], ref, nan_policy="omit").statistic for t in bench.columns}
    order = [t for t, _ in sorted(solo.items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -9))]
    z = (bench - bench.mean()) / bench.std(ddof=0)
    ks, rhos = [], []
    for k in range(1, len(order) + 1):
        agg = z[order[:k]].mean(axis=1)
        rhos.append(spearmanr(agg, ref, nan_policy="omit").statistic)
        ks.append(k)
    rho = np.array(rhos, dtype=float)
    full = rho[-1]
    peak_i = int(np.nanargmax(rho))
    peak_k, peak = ks[peak_i], rho[peak_i]
    # smallest suite within 5 % of the best one, which is the answer to the question
    small = int(ks[int(np.argmax(rho >= .95 * peak))])
    beats = int((rho > full).sum())

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(ks, rho, lw=2.4, color=S.RAMP[2], zorder=3)
    ax.axhline(full, color=S.SERIES[1], ls="--", lw=1.4, zorder=2)
    ax.annotate(f"the whole suite of {len(order)} tasks, ρ = {full:.2f}",
                (len(order), full), xytext=(-6, -14), ha="right",
                textcoords="offset points", fontsize=9, color=S.SERIES[1])
    ax.scatter([peak_k], [peak], s=90, color=S.RAMP[3], zorder=4)
    ax.annotate(f"best suite: {peak_k} tasks, ρ = {peak:.2f}", (peak_k, peak),
                xytext=(10, 6), textcoords="offset points", fontsize=9.5, color=S.INK)
    ax.annotate(f"every suite size but the last beats the whole suite\n"
                f"({beats} of {len(order)})", (len(order) * .52, .16),
                fontsize=9, color=S.MUTED, ha="center")
    ax.set_xlabel(f"benchmark tasks in the suite, added best first",
                  fontsize=9.5, color=S.MUTED)
    ax.set_ylabel("agreement with the bits per byte ranking (Spearman ρ)",
                  fontsize=9.5, color=S.MUTED)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, len(order) + 4)
    ax.grid(color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], color=S.RAMP[2], lw=2.4, label="suite built best task first"),
                       Line2D([], [], marker="o", ls="none", color=S.RAMP[3], label="the best suite"),
                       Line2D([], [], color=S.SERIES[1], ls="--", lw=1.4, label="the whole suite"),
                       ],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower left")
    S.title(fig, "A small suite ranks the models better than the whole one", y=1.02)
    S.save(fig, out / "rq_b2_suite_size.png")
    return pd.DataFrame({"k": ks, "rho": rho}), peak_k, peak, small, full, len(order), beats


# --- B3 ---------------------------------------------------------------------
def b3_surrogate(out):
    """Which measurement carries the decision, ranked by how often a small proxy
    reproduces the reference's answer."""
    from predictivity import min_predictive_size, kind
    d = min_predictive_size()
    d = d.assign(kind=d["task"].map(kind))
    g = d.groupby("kind").agg(works=("min_size", lambda s: s.notna().mean()),
                              n=("min_size", "size"))
    small = d[d["min_size"] == "175M"].groupby("kind").size()
    g["at175"] = [small.get(k, 0) / n for k, n in zip(g.index, g["n"])]
    g = g.sort_values("works")

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    y = np.arange(len(g))
    ax.barh(y, g["works"], .5, color=S.RAMP[2], label="some proxy size works")
    ax.barh(y, g["at175"], .5, color=S.RAMP[0], label="175M is already enough")
    for i, (w, n) in enumerate(zip(g["works"], g["n"])):
        ax.annotate(f"{w:.0%}  of {int(n):,}", (w, i), xytext=(7, 0), va="center",
                    textcoords="offset points", fontsize=9, color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=9.5)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("share of measurements a proxy reproduces the reference on",
                  fontsize=9.5, color=S.MUTED)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower right")
    S.title(fig, "Which surrogate carries the decision", y=1.04)
    S.save(fig, out / "rq_b3_surrogate.png")
    return g


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    f = a1_scaling(OUT)
    print("A1 top families by R²:")
    print(f.groupby("family")["r2"].median().sort_values(ascending=False).head(5).round(3).to_string())
    a2_stability(OUT)
    d = a3_languages(OUT)
    print(f"\nA3: {len(d)} languages; median gain {d['gain'].median():.3f} bits/byte; "
          f"{int((d['gain'] > 0).sum())} gain, {int((d['gain'] <= 0).sum())} do not")
    a4_signal(OUT)
    b1 = b1_early(OUT)
    print("\nB1 mean DA by checkpoint fraction at the top size:")
    print(b1[b1['size'] == SIZES[-1]].groupby("pct")["da"].mean().round(3).to_string())
    curve, peak_k, peak, small, full, n, beats = b2_suite_size(OUT)
    print(f"\nB2: best suite {peak_k} tasks at rho {peak:.3f}; {small} tasks reach 95 % of it; "
          f"whole suite of {n} gives {full:.3f}; {beats} suite sizes beat the whole suite")
    g = b3_surrogate(OUT)
    print("\nB3:"); print(g.round(3).to_string())
