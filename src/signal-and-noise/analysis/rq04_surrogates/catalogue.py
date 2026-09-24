"""A catalogue of surrogates of decision accuracy beyond SNR and FineTasks.

Every statistic here is read on the PROXY alone (rule 11): its ten tenths, its
noise window (rule 4), and the smaller rungs of the ladder, never the 1.7B
reference. Each is scored against DA-size (proxy final -> 1.7B final, rule 9)
over the same pair set it was computed on (rule 15: `multi-axis` and
`mono-axis`). The catalogue and its sources are `SURROGATES` below and
`literature.md` next to this file.

Populations: the tasks above chance at the proxy and at the reference (rule 1);
benchmarks and per-language BPB separately; `bpb_macro` / `train_loss` in
neither (rule 7). Per language, a Spearman rho over the language's (task,
proxy) cells, at least MIN_LANG_TASKS tasks (rule 8), descriptive only.

It also measures the ceiling any surrogate faces: DA-size recomputed with the
reference read at 90 % instead of 100 % of its run (the checkpoint noise of
the truth itself). A surrogate cannot correlate with DA better than DA
correlates with itself.

    surrogate_values.csv          one row per (task, proxy, axes): every statistic + DA-size
    surrogate_correlations.csv    rho / r / p / n and a task-bootstrap 90 % band per (axes, kind, proxy, surrogate)
    surrogate_by_language.csv     rho per (axes, language, surrogate), benchmarks
    surrogate_by_group.csv        rho per language tier and per benchmark family
    surrogate_combined.csv        every statistic together, leave-one-language-out
    surrogate_definitions.csv     name, family, expected sign, formula, source
    da_retest.csv                 the ceiling: DA-size vs itself at a 90 % reference
    surrogates_catalogue.png, surrogates_catalogue_by_language.png (+ .csv)

    python analysis/rq04_surrogates/catalogue.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, pearsonr, rankdata, spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from snr.stats import calc_monotonicity, calc_total_variation  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import NOISE_AND_SNR, SURROGATES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_items, task_n_options  # noqa: E402
from analysis.rq02_decision_accuracy.language_tier import tier_of_language  # noqa: E402
from analysis.utils import (  # noqa: E402
    EVAL_SIZES, MIN_LANG_TASKS, MIN_PAIRS, NOISE_GRID, NON_EMB, PAIR_AXES, SMALL_SIZES, TARGET_SIZE, assign_language,
    benchmark_family, design_axes, ladder_frame, languages_only, noise_checkpoints, on_shared_grid,
    pair_agreement, pair_sets, passes_gate)

OUT_ROOT = SURROGATES
AXES = PAIR_AXES[:2]                   # the design pair sets; the seed null has no DA-size to predict
KINDS = ("benchmark", "bpb")
N_BOOT = 200
MIN_TASKS = 8                          # a pooled rho needs this many tasks, as analyze.py
Z_CAP = 50.0                           # a gap over noise past this is "separated"; keeps the median finite
REF_RETEST = 18                        # the reference read at 18/NOISE_GRID = 90 % for the ceiling
mpl.rcParams.update(S.RC)

# name -> (family, expected sign, formula, source). Sign: +1 = higher should mean higher DA.
SURROGATES = {
    # separation against the checkpoint noise (signal detection / discriminative power)
    "gap_over_noise": ("separation", +1, "median over pairs of |x_a - x_b| / sqrt(s_a^2 + s_b^2), s = window std",
                       "Heineman 2025 (pairwise form); Sakai 2006"),
    "resolved_pairs_noise": ("separation", +1, "share of pairs with |x_a - x_b| > 1.96 sqrt(s_a^2 + s_b^2)",
                             "Sakai 2006 discriminative power; Madaan 2024"),
    "retest_agreement": ("separation", +1, "mean over pairs of Phi(z)^2 + Phi(-z)^2, z = |x_a - x_b| / sqrt(s_a^2 + s_b^2): "
                         "chance two noisy re-measurements order the pair alike", "Voorhees & Buckley 2002 swap rate"),
    "icc_window": ("separation", +1, "ICC(1) of the window scores, families as subjects, window checkpoints as raters",
                   "Shrout & Fleiss 1979; Madaan 2024"),
    "binomial_snr": ("separation", +1, "std_f(x) / sqrt(p(1-p)/n_items): spread across variants over item-sampling noise",
                     "Card 2020; Miller 2024"),
    "resolved_pairs_binomial": ("separation", +1, "share of pairs a two-proportion z-test over n_items separates at 5 %",
                                "Card 2020; Dror 2018"),
    "tie_rate_items": ("separation", -1, "share of pairs closer than one item (1/n_items)", "Card 2020"),
    "noise_to_binomial": ("separation", -1, "pooled window std / binomial se sqrt(p(1-p)/n_items)",
                          "Madaan 2024; Wang 2025 (all the noises)"),
    "mde_resolved": ("separation", +1, "share of pairs above the minimum detectable effect (1.96 + 0.84) sqrt(2) s at 80 % power",
                     "Card 2020"),
    "prob_outperform": ("separation", +1, "mean over pairs of |P(window score a > window score b) - 1/2|",
                        "Bouthillier 2021"),
    "dior": ("separation", +1, "5th percentile of Kendall tau between the final ranking and rankings drawn from a random "
             "window checkpoint per variant", "Perlitz 2023 (DIoR)"),
    "eta2_window": ("separation", +1, "between-variant share of the window variance (ANOVA eta^2)", "Fisher; Madaan 2024"),
    "g_coefficient": ("separation", +1, "generalizability coefficient ICC(1,k) = (MSB - MSW) / MSB over the window",
                      "Bodoff & Li 2007; Urbano 2013"),
    "cronbach_alpha": ("separation", +1, "Cronbach's alpha, window checkpoints as items, variants as respondents",
                       "Cronbach 1951"),
    # rank stability inside the proxy run (FineTasks, IR stability)
    "consecutive_kendall": ("rank stability", +1, "mean Kendall tau between the rankings at consecutive tenths",
                            "FineTasks (Kydlicek 2024) ordering consistency"),
    "consecutive_kendall_late": ("rank stability", +1, "the same over the second half of the run (50-100 %)",
                                 "FineTasks (Kydlicek 2024)"),
    "window_kendall": ("rank stability", +1, "mean Kendall tau of each window checkpoint's ranking with the final one",
                       "Heineman 2025; Madaan 2024"),
    "kendall_w_window": ("rank stability", +1, "Kendall's W of the rankings at the window checkpoints",
                         "Kendall & Babington Smith 1939"),
    "sign_consistency_window": ("rank stability", +1, "share of pairs ordered the same way at every window checkpoint",
                                "Buckley & Voorhees 2000"),
    "crossings": ("rank stability", -1, "mean number of sign changes of x_a(t) - x_b(t) over the ten tenths",
                  "rank reversals / learning-curve crossings"),
    "da_ckpt_mean": ("rank stability", +1, "mean DA of the 10-90 % rankings against the proxy's final",
                     "Heineman 2025; Magnusson 2025 (DataDecide)"),
    "da_ckpt_half": ("rank stability", +1, "DA of the 50 % ranking against the proxy's final", "Magnusson 2025"),
    "settling_time": ("rank stability", -1, "first tenth after which Kendall tau with the final ranking stays >= 0.8",
                      "rank settling (Magnusson 2025)"),
    "sign_persistence": ("rank stability", +1, "mean over pairs of the share of tenths ordered as at the final",
                         "Buckley & Voorhees 2000"),
    "split_half": ("rank stability", +1, "Spearman-Brown split-half: Spearman of the 80/90/100 % vs 85/95 % means",
                   "Spearman 1910; Brown 1910"),
    # training-curve shape (FineTasks, Heineman stats.py)
    "monotonicity": ("curve shape", +1, "mean over variants of Spearman(checkpoint, score)", "FineTasks (Kydlicek 2024)"),
    "monotonicity_steps": ("curve shape", +1, "mean over variants of (#up - #down)/(#steps)", "Heineman 2025 (snr/stats.py)"),
    "total_variation": ("curve shape", -1, "mean over variants of mean|diff| / range of the curve",
                        "Heineman 2025 (snr/stats.py)"),
    "gain_over_noise": ("curve shape", +1, "mean over variants of (x(100 %) - x(10 %)) / pooled window std",
                        "Madaan 2024 (monotonic improvement)"),
    "autocorr": ("curve shape", +1, "mean over variants and lags 1-2 of |autocorrelation| of the curve",
                 "E2LM (2025) signal quality"),
    "late_slope_to_noise": ("curve shape", -1, "mean over variants of the 50-100 % OLS slope / its residual std",
                            "Madaan 2024; Hofmann 2025 (saturation)"),
    "item_information": ("curve shape", +1, "q(1 - q), q = (mean x - chance)/(1 - chance): aggregate test information",
                         "Rodriguez 2021; Vania 2021; Hofmann 2025"),
    "emerged_share": ("curve shape", +1, "share of variants above chance + 2 binomial se", "Du 2024; Schaeffer 2023"),
    "nonrandom": ("curve shape", +1, "(mean x - chance) / (1 - chance)", "FineTasks (Kydlicek 2024) non-randomness"),
    "z_above_chance": ("curve shape", +1, "(mean x - chance) / sqrt(chance(1-chance)/n_items)", "Madaan 2024; rq00 gate"),
    # the small ladder below the proxy (no reference)
    "prev_rung_da": ("small ladder", +1, "DA of the previous rung's final ranking against the proxy's",
                     "Magnusson 2025; Perlitz 2024 (BAT)"),
    "ladder_da": ("small ladder", +1, "mean DA over every pair of rungs up to the proxy", "Magnusson 2025"),
    "rung_sign_consistency": ("small ladder", +1, "share of pairs ordered alike at every rung up to the proxy",
                              "Magnusson 2025; Liu 2024 (RegMix rank invariance)"),
    "rung_gap_corr": ("small ladder", +1, "Pearson r of the pair gaps at the previous rung and at the proxy",
                      "Liu 2024 (RegMix)"),
    "projected_flip_rate": ("small ladder", -1, "share of pairs whose gap, extrapolated linearly in log N from the rungs up "
                            "to the proxy, changes sign by the reference's N", "Magnusson 2025 (crossovers)"),
    "pseudo_ref_da": ("pseudo-reference (reads 1B)", +1, "DA of the proxy's final ranking against the 1B final ranking "
                      "(reads the largest rung below the reference, not the proxy alone)", "Liu 2024; Magnusson 2025"),
    "size_monotonicity": ("small ladder", +1, "mean over variants of the (#up - #down)/#steps across rungs up to the proxy",
                          "Bhagia 2024; Schaeffer 2024"),
    "scale_gain_over_noise": ("small ladder", +1, "mean over variants of (x(proxy) - x(previous rung)) / pooled window std",
                              "Bhagia 2024"),
    # agreement with the loss and with other benchmarks
    "bpb_rank_agreement": ("agreement", +1, "DA of the task's final ranking against the language's BPB ranking",
                           "Thrush 2024; Du 2024; Schaeffer 2024"),
    "bpb_corr_training": ("agreement", +1, "Spearman of the task score with -BPB over (variant, tenth) cells",
                          "Thrush 2024 perplexity correlations; Gadre 2024"),
    "language_consensus": ("agreement", +1, "DA against the mean z-score of the language's other gated benchmarks",
                           "Perlitz 2024 (BAT); Zhang & Hardt 2024"),
    "item_total_corr": ("agreement", +1, "Spearman of the final scores with the mean z-score of every other gated "
                        "benchmark (corrected item-total)", "Ruan 2024 (common factor); Cronbach 1951"),
    "benchmark_consensus": ("agreement", +1, "DA against the mean z-score of the same benchmark in other languages",
                            "Perlitz 2024 (BAT)"),
    # controls
    "n_pairs": ("control", +1, "pairs the proxy ranks", "-"),
    "n_items": ("control", +1, "scored examples", "Card 2020"),
    "chance": ("control", -1, "1 / number of options", "-"),
}
# rq03's SNR, for reference on the same populations
SNR_REF = {"snr_aad": "aad", "snr_rel_std": "rel_std", "snr_dist_std": "dist_std"}


def _pairs_ij(fams: list[str], pairs: list) -> tuple[np.ndarray, np.ndarray]:
    pos = {f: i for i, f in enumerate(fams)}
    ij = np.array([(pos[a], pos[b]) for a, b in pairs if a in pos and b in pos], int).reshape(-1, 2)
    return ij[:, 0], ij[:, 1]


def _da(fams, x, y, pairs) -> float:
    """`pair_agreement` over the families with a value on both sides."""
    ok = np.isfinite(x) & np.isfinite(y)
    f = [a for a, k in zip(fams, ok) if k]
    return pair_agreement(dict(zip(f, x[ok])), dict(zip(f, y[ok])), pairs)[0]


def _mean_tau(a: np.ndarray, cols) -> float:
    taus = []
    for c0, c1 in cols:
        ok = np.isfinite(a[:, c0]) & np.isfinite(a[:, c1])
        if ok.sum() >= 3 and np.ptp(a[ok, c0]) and np.ptp(a[ok, c1]):
            taus.append(kendalltau(a[ok, c0], a[ok, c1]).statistic)
    return float(np.mean(taus)) if taus else np.nan


def _window_anova(w: np.ndarray) -> dict:
    """Variance-decomposition statistics of the complete rows of a family x
    window matrix: variants are the subjects, window checkpoints the raters."""
    w = w[np.isfinite(w).all(axis=1)]
    n, m = w.shape
    if n < 3 or m < 2:
        return {}
    msb = m * w.mean(axis=1).var(ddof=1)
    msw = ((w - w.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (m - 1))
    ss_tot = ((w - w.mean()) ** 2).sum()
    r = np.apply_along_axis(rankdata, 0, w).sum(axis=1)
    tot = w.sum(axis=1).var(ddof=1)
    return {"icc_window": (msb - msw) / (msb + (m - 1) * msw) if msb + (m - 1) * msw > 0 else np.nan,
            "g_coefficient": (msb - msw) / msb if msb > 0 else np.nan,
            "eta2_window": msb * (n - 1) / ss_tot if ss_tot > 0 else np.nan,
            "cronbach_alpha": m / (m - 1) * (1 - w.var(axis=0, ddof=1).sum() / tot) if tot > 0 else np.nan,
            "kendall_w_window": 12 * ((r - r.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))}


def _dior(W: np.ndarray, x: np.ndarray, rng, draws: int = 100) -> float:
    """5th percentile of Kendall tau between the final ranking and a ranking that
    reads each variant at a random window checkpoint (Perlitz 2023's DIoR)."""
    ok = np.isfinite(x) & np.isfinite(W).any(axis=1)
    if ok.sum() < 3:
        return np.nan
    Wk, xk = W[ok], x[ok]
    taus = []
    for _ in range(draws):
        pick = np.array([rng.choice(r[np.isfinite(r)]) for r in Wk])
        if np.ptp(pick):
            taus.append(kendalltau(pick, xk).statistic)
    return float(np.percentile(taus, 5)) if taus else np.nan


def _curve_stats(task: str, fams: list, C: np.ndarray, W: np.ndarray, pairs: list, rng) -> dict:
    """The statistics of one (task, proxy) read on its ten tenths `C` (family x 10)
    and its window `W` (family x 5)."""
    x = C[:, -1]
    i, j = _pairs_ij(fams, pairs)
    ok = np.isfinite(x[i]) & np.isfinite(x[j])
    i, j = i[ok], j[ok]
    out = {"n_pairs": len(i)}
    if len(i) < MIN_PAIRS:                                        # rule 5
        return out
    sd = np.nanstd(W, axis=1, ddof=1)
    pooled = np.sqrt(np.nanmean(sd ** 2))
    d, s = np.abs(x[i] - x[j]), np.sqrt(sd[i] ** 2 + sd[j] ** 2)
    z = np.where(s > 0, d / np.where(s > 0, s, 1), Z_CAP)          # a noiseless pair is perfectly separated
    zz = np.minimum(z[np.isfinite(s)], Z_CAP)
    if len(zz):
        out |= {"gap_over_noise": float(np.median(zz)), "resolved_pairs_noise": float(np.mean(zz > 1.96)),
                "retest_agreement": float(np.mean(norm.cdf(zz) ** 2 + norm.cdf(-zz) ** 2))}
    out |= _window_anova(W)
    out["mde_resolved"] = float(np.mean(d > (1.96 + 0.84) * np.sqrt(2) * pooled)) if pooled > 0 else np.nan
    wi, wj = W[i][:, :, None], W[j][:, None, :]
    fin = np.isfinite(wi) & np.isfinite(wj)
    wins = np.where(fin, (wi > wj) + .5 * (wi == wj), np.nan)
    out["prob_outperform"] = float(np.nanmean(np.abs(np.nanmean(wins.reshape(len(i), -1), axis=1) - .5)))
    out["dior"] = _dior(W, x, rng)
    n_items, n_opt = task_n_items(task), task_n_options(task)
    xs = x[np.isfinite(x)]
    if np.isfinite(n_items) and n_items > 0 and np.isfinite(n_opt) and 0 < xs.mean() < 1:
        p, c = xs.mean(), 1 / n_opt
        out["n_items"], out["chance"] = n_items, c
        out["binomial_snr"] = xs.std(ddof=1) / np.sqrt(p * (1 - p) / n_items)
        pbar = (x[i] + x[j]) / 2
        out["resolved_pairs_binomial"] = float(np.mean(d / np.sqrt(pbar * (1 - pbar) * 2 / n_items) > 1.96))
        out["tie_rate_items"] = float(np.mean(d < 1 / n_items))
        out["nonrandom"] = q = (p - c) / (1 - c)
        out["item_information"] = q * (1 - q)
        out["noise_to_binomial"] = pooled / np.sqrt(p * (1 - p) / n_items)
        out["emerged_share"] = float(np.mean(xs > c + 2 * np.sqrt(c * (1 - c) / n_items)))
        out["z_above_chance"] = (p - c) / np.sqrt(c * (1 - c) / n_items)
    out["consecutive_kendall"] = _mean_tau(C, [(k, k + 1) for k in range(9)])
    out["consecutive_kendall_late"] = _mean_tau(C, [(k, k + 1) for k in range(4, 9)])
    out["window_kendall"] = _mean_tau(W, [(k, W.shape[1] - 1) for k in range(W.shape[1] - 1)])
    sw = np.sign(W[i] - W[j])
    full = np.isfinite(sw).all(axis=1)
    if full.sum() >= MIN_PAIRS:
        out["sign_consistency_window"] = float(np.mean((np.abs(sw[full].sum(axis=1)) == sw.shape[1])))
    dc = np.sign(C[i] - C[j])
    out["sign_persistence"] = float(np.nanmean(np.where(np.isfinite(dc[:, :-1]), dc[:, :-1] == dc[:, -1:], np.nan)))
    tau_final = [_mean_tau(C, [(k, 9)]) for k in range(9)]
    low = [k for k, t in enumerate(tau_final) if not t >= .8]
    out["settling_time"] = (max(low) + 2 if low else 1) / 10
    out["crossings"] = float(np.nanmean([np.sum(np.diff(r[np.isfinite(r) & (r != 0)]) != 0) for r in dc]))
    das = [_da(fams, C[:, k], x, pairs) for k in range(9)]
    out["da_ckpt_mean"], out["da_ckpt_half"] = np.nanmean(das), das[4]
    a, b = np.nanmean(W[:, [0, 2, 4]], axis=1), np.nanmean(W[:, [1, 3]], axis=1)
    okh = np.isfinite(a) & np.isfinite(b)
    if okh.sum() >= 3 and np.ptp(a[okh]) and np.ptp(b[okh]):
        r = spearmanr(a[okh], b[okh]).statistic
        out["split_half"] = 2 * r / (1 + r) if r > -1 else np.nan
    rows = [c[np.isfinite(c)] for c in C]
    mono = [spearmanr(np.arange(len(c)), c).statistic for c in rows if len(c) >= 3 and np.ptp(c)]
    out["monotonicity"] = float(np.mean(mono)) if mono else np.nan
    out["monotonicity_steps"] = float(np.mean([calc_monotonicity(c) for c in rows if len(c) >= 2]))
    out["total_variation"] = float(np.mean([calc_total_variation(c, norm=True) for c in rows if len(c) >= 2]))
    ac = [abs(pd.Series(c).autocorr(lag)) for c in rows if len(c) >= 6 and np.ptp(c) for lag in (1, 2)]
    out["autocorr"] = float(np.nanmean(ac)) if ac else np.nan
    sl = []
    for c in C[:, 4:]:
        k = np.flatnonzero(np.isfinite(c))
        if len(k) >= 4:
            b, a = np.polyfit(k, c[k], 1)
            res = np.std(c[k] - (a + b * k), ddof=2)
            if res > 0:
                sl.append(b / res)
    out["late_slope_to_noise"] = float(np.mean(sl)) if sl else np.nan
    if pooled > 0:
        out["gain_over_noise"] = float(np.nanmean(C[:, -1] - C[:, 0]) / pooled)
    out["_pooled_sd"] = pooled
    return out


def _zscore(finals: pd.DataFrame) -> pd.DataFrame:
    """family x task finals, each task standardized over the families it has."""
    return (finals - finals.mean()) / finals.std(ddof=1)


def _ladder_stats(fams: list, lad: np.ndarray, sizes: list, pairs: list) -> dict:
    """Cross-rung statistics from the final scores `lad` (family x rung, the
    rungs up to the proxy, the proxy last)."""
    i, j = _pairs_ij(fams, pairs)
    g = lad[i] - lad[j]                                             # pair x rung
    out = {}
    full = np.isfinite(g).all(axis=1) & (g != 0).all(axis=1)
    if full.sum() >= MIN_PAIRS:
        out["rung_sign_consistency"] = float(np.mean(np.abs(np.sign(g[full]).sum(axis=1)) == g.shape[1]))
    ok = np.isfinite(g[:, -2]) & np.isfinite(g[:, -1])
    if ok.sum() >= MIN_PAIRS and np.ptp(g[ok, -2]) and np.ptp(g[ok, -1]):
        out["rung_gap_corr"] = pearsonr(g[ok, -2], g[ok, -1]).statistic
    logn = np.log([NON_EMB[s] for s in sizes])
    flips = []
    for row in g:
        k = np.isfinite(row)
        if k.sum() >= 2 and k[-1] and row[-1] != 0:
            b, a = np.polyfit(logn[k], row[k], 1)
            flips.append(np.sign(a + b * np.log(NON_EMB[TARGET_SIZE])) != np.sign(row[-1]))
    if len(flips) >= MIN_PAIRS:
        out["projected_flip_rate"] = float(np.mean(flips))
    return out


def compute(df: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (task, proxy, axes): every statistic, DA-size and the retest DA."""
    # Oriented so higher is better for every kind: BPB and loss flip sign. DA is
    # sign-invariant; the curve and ladder statistics that read a direction are not.
    df = df.assign(score=np.where(df["kind"].isin(["bpb", "loss"]), -df["primary_score"], df["primary_score"]))
    grid = df[on_shared_grid(df)].assign(k=lambda d: (d["frac"] * 10).round().astype(int))
    win = noise_checkpoints(df).assign(p=lambda d: (d["frac"] * NOISE_GRID).round().astype(int))
    curves = {s: g.pivot_table(index=["task", "family"], columns="k", values="score").reindex(columns=range(1, 11))
              for s, g in grid.groupby("size")}
    windows = {s: g.pivot_table(index=["task", "family"], columns="p", values="score")
               .reindex(columns=range(NOISE_GRID - 4, NOISE_GRID + 1)) for s, g in win.groupby("size")}
    finals = {s: c[10].unstack("task") for s, c in curves.items()}          # family x task
    psets = pair_sets(design_axes(df))
    tasks = sorted(set(df["task"]) - {"bpb_macro", "train_loss"})
    lang = {t: assign_language(t) for t in tasks}
    bpb_of = pd.Series({lang[t]: t for t in tasks if t.startswith("bpb_")})
    bench = {t: benchmark_family(t) for t in tasks}
    ref, ref_w = finals[TARGET_SIZE], windows[TARGET_SIZE]
    pseudo = SMALL_SIZES[-1]                                        # the largest rung below the reference
    rng = np.random.default_rng(0)
    rows = []
    for si, s in enumerate(SMALL_SIZES):
        C_s, W_s, F_s = curves[s], windows[s], finals[s]
        Z_s = _zscore(F_s)
        gate_s = passes_gate(mask, tasks, s)
        for t in tasks:
            if t not in F_s.columns or F_s[t].notna().sum() < 3:
                continue
            fams = F_s[t].dropna().index.tolist()
            C = C_s.loc[t].reindex(fams).to_numpy()
            W = W_s.loc[t].reindex(fams).to_numpy() if t in W_s.index.get_level_values(0) else np.full((len(fams), 5), np.nan)
            x = C[:, -1]
            yr = ref[t].reindex(fams).to_numpy() if t in ref.columns else np.full(len(fams), np.nan)
            yr90 = (ref_w.loc[t][REF_RETEST].reindex(fams).to_numpy()
                    if t in ref_w.index.get_level_values(0) else np.full(len(fams), np.nan))
            for axes in AXES:
                pairs = psets[axes]
                r = {"task": t, "language": lang[t], "benchmark": bench[t],
                     "kind": "bpb" if t.startswith("bpb_") else "benchmark", "proxy_size": s, "axes": axes,
                     "gated_in": bool(gate_s[t] and passes_gate(mask, [t], TARGET_SIZE).iloc[0]),
                     "da_size": _da(fams, x, yr, pairs), "da_size_ref90": _da(fams, x, yr90, pairs)}
                r |= _curve_stats(t, fams, C, W, pairs, rng)
                pooled = r.pop("_pooled_sd", np.nan)
                rungs = SMALL_SIZES[:si + 1]
                if s != pseudo:
                    r["pseudo_ref_da"] = _da(fams, x, finals[pseudo].get(t, pd.Series(dtype=float)).reindex(fams).to_numpy(), pairs)
                if si:
                    prev = finals[rungs[-2]].get(t, pd.Series(dtype=float)).reindex(fams).to_numpy()
                    r["prev_rung_da"] = _da(fams, prev, x, pairs)
                    r["ladder_da"] = np.nanmean([_da(fams, finals[a].get(t, pd.Series(dtype=float)).reindex(fams).to_numpy(),
                                                     finals[b].get(t, pd.Series(dtype=float)).reindex(fams).to_numpy(), pairs)
                                                 for ai, a in enumerate(rungs) for b in rungs[ai + 1:]])
                    lad = np.column_stack([finals[a].get(t, pd.Series(dtype=float)).reindex(fams).to_numpy() for a in rungs])
                    r |= _ladder_stats(fams, lad, rungs, pairs)
                    r["size_monotonicity"] = float(np.mean([calc_monotonicity(c[np.isfinite(c)]) for c in lad
                                                            if np.isfinite(c).sum() >= 2]))
                    if pooled > 0:
                        r["scale_gain_over_noise"] = float(np.nanmean(x - prev) / pooled)
                if r["kind"] == "benchmark":
                    b = bpb_of.get(lang[t])
                    if b is not None and b in F_s.columns:
                        r["bpb_rank_agreement"] = _da(fams, x, F_s[b].reindex(fams).to_numpy(), pairs)
                        cb = C_s.loc[b].reindex(fams).to_numpy() if b in C_s.index.get_level_values(0) else None
                        if cb is not None:
                            ok = np.isfinite(C) & np.isfinite(cb)
                            if ok.sum() >= 5:
                                r["bpb_corr_training"] = spearmanr(C[ok], cb[ok]).statistic
                    peers = [u for u in F_s.columns if u != t and lang.get(u) == lang[t] and bench.get(u) != bench[t]
                             and not u.startswith("bpb_") and u in gate_s.index and gate_s[u]]
                    if peers:
                        r["language_consensus"] = _da(fams, x, Z_s[peers].mean(axis=1).reindex(fams).to_numpy(), pairs)
                    peers = [u for u in F_s.columns if u != t and not u.startswith("bpb_") and u in gate_s.index
                             and gate_s[u]]
                    zm = Z_s[peers].mean(axis=1).reindex(fams).to_numpy()
                    ok = np.isfinite(zm) & np.isfinite(x)
                    if ok.sum() >= 3 and np.ptp(x[ok]):
                        r["item_total_corr"] = spearmanr(x[ok], zm[ok]).statistic
                    peers = [u for u in F_s.columns if u != t and bench.get(u) == bench[t] and lang.get(u) != lang[t]
                             and u in gate_s.index and gate_s[u]]
                    if peers:
                        r["benchmark_consensus"] = _da(fams, x, Z_s[peers].mean(axis=1).reindex(fams).to_numpy(), pairs)
                rows.append(r)
        print(f"  {s}: {sum(r['proxy_size'] == s for r in rows)} (task, axes) rows")
    return pd.DataFrame(rows)


def _rho(x, y) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan, np.nan, int(ok.sum())
    r = spearmanr(x[ok], y[ok])
    return r.statistic, r.pvalue, int(ok.sum())


def _partial_rho(x, y, z) -> float:
    """Spearman rho of x and y with the ranks of z regressed out of both."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < MIN_TASKS or np.ptp(z[ok]) == 0:
        return np.nan
    rx, ry, rz = (rankdata(a[ok]) for a in (x, y, z))
    res = [a - np.polyval(np.polyfit(rz, a, 1), rz) for a in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


def correlations(v: pd.DataFrame, names: list[str], rng) -> pd.DataFrame:
    """Spearman rho (and Pearson r) of each statistic with DA-size, per (axes,
    kind, proxy) and pooled over the proxies; the pooled row carries a 90 %
    band from resampling TASKS (a task sits at several proxies)."""
    rows = []
    for (axes, kind), g in v[v["gated_in"]].groupby(["axes", "kind"]):
        for proxy, h in [*g.groupby("proxy_size"), ("all", g)]:
            if h["task"].nunique() < MIN_TASKS:
                continue
            y = h["da_size"].to_numpy(float)
            for m in names:
                x = h[m].to_numpy(float)
                rho, p, n = _rho(x, y)
                if not np.isfinite(rho):
                    continue
                ok = np.isfinite(x) & np.isfinite(y)
                row = {"axes": axes, "kind": kind, "proxy_size": proxy, "surrogate": m, "rho": rho, "p": p, "n": n,
                       "n_tasks": h.loc[ok, "task"].nunique(), "pearson_r": pearsonr(x[ok], y[ok]).statistic}
                if proxy == "all":
                    row["rho_partial_items"] = _partial_rho(x, y, h["n_items"].to_numpy(float))
                    tk = h.loc[ok, "task"].to_numpy()
                    ut = np.unique(tk)
                    idx = {u: np.flatnonzero(tk == u) for u in ut}
                    xs, ys = x[ok], y[ok]
                    boot = []
                    for _ in range(N_BOOT):
                        sel = np.concatenate([idx[u] for u in rng.choice(ut, len(ut))])
                        boot.append(_rho(xs[sel], ys[sel])[0])
                    row["rho_lo"], row["rho_hi"] = np.nanpercentile(boot, [5, 95])
                rows.append(row)
    return pd.DataFrame(rows)


def by_group(v: pd.DataFrame, names: list[str], col: str, min_tasks: int) -> pd.DataFrame:
    """Spearman rho per value of `col` over its (task, proxy) cells, benchmarks only."""
    rows = []
    g0 = v[v["gated_in"] & (v["kind"] == "benchmark")]
    for (axes, key), g in g0.groupby(["axes", col]):
        y = g["da_size"].to_numpy(float)
        for m in names:
            x = g[m].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if g.loc[ok, "task"].nunique() < min_tasks:                # rule 8
                continue
            rho, p, n = _rho(x, y)
            if np.isfinite(rho):
                rows.append({"axes": axes, col: key, "surrogate": m, "rho": rho, "p": p, "n": n,
                             "n_tasks": g.loc[ok, "task"].nunique()})
    return pd.DataFrame(rows)


def retest(v: pd.DataFrame) -> pd.DataFrame:
    """The ceiling: Spearman of DA-size with the same DA read against the
    reference at 90 % of its run, per (axes, kind, proxy)."""
    rows = []
    for (axes, kind, proxy), g in v[v["gated_in"]].groupby(["axes", "kind", "proxy_size"]):
        rho, p, n = _rho(g["da_size"].to_numpy(float), g["da_size_ref90"].to_numpy(float))
        rows.append({"axes": axes, "kind": kind, "proxy_size": proxy, "rho": rho, "n": n,
                     "mean_abs_diff": (g["da_size"] - g["da_size_ref90"]).abs().mean()})
    return pd.DataFrame(rows)


def combined(v: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Every statistic (plus the proxy size) in one gradient-boosted model,
    fitted leave-one-language-out: the Spearman rho of its out-of-language
    prediction with DA-size, per (axes, kind)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for (axes, kind), g in v[v["gated_in"] & v["da_size"].notna()].groupby(["axes", "kind"]):
        X = g[names].assign(proxy=g["proxy_size"].map(EVAL_SIZES.index)).to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nan)
        X = X[:, (np.isfinite(X).sum(axis=0) >= MIN_TASKS)]                # BPB has no item count, no consensus
        y, groups = g["da_size"].to_numpy(float), g["language"].to_numpy()
        pred = np.full(len(y), np.nan)
        for lg in np.unique(groups):
            te = groups == lg
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=.05, random_state=0)
            pred[te] = model.fit(X[~te], y[~te]).predict(X[te])
        rho, p, n = _rho(pred, y)
        rows.append({"axes": axes, "kind": kind, "rho": rho, "p": p, "n": n, "n_languages": len(np.unique(groups))})
    return pd.DataFrame(rows)


def plot(corr: pd.DataFrame, ceiling: pd.DataFrame, out_dir: Path, names: list[str]) -> None:
    c = corr[(corr["proxy_size"] == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 9), sharey=True)
    order = (c[(c["kind"] == "benchmark") & (c["axes"] == AXES[0])].set_index("surrogate")["rho"]
             .reindex(names).sort_values().index.tolist())
    y = np.arange(len(order))
    for ax, kind in zip(axes, KINDS):
        for k, (axes_name, col) in enumerate(zip(AXES, S.SERIES)):
            g = c[(c["kind"] == kind) & (c["axes"] == axes_name)].set_index("surrogate").reindex(order)
            off = (k - .5) * .3
            ax.errorbar(g["rho"], y + off, xerr=[g["rho"] - g["rho_lo"], g["rho_hi"] - g["rho"]], fmt="o", ms=3.5,
                        color=col, lw=.8, label=axes_name)
        ceil = ceiling[(ceiling["kind"] == kind) & (ceiling["axes"] == AXES[0])]["rho"].mean()
        if np.isfinite(ceil):
            ax.axvline(ceil, color=S.MUTED, ls="--", lw=.8)
            ax.text(ceil, len(order) - .5, " DA retest", fontsize=6.5, color=S.MUTED, va="bottom")
        ax.axvline(0, color=S.MUTED, lw=.8)
        ax.set_yticks(y); ax.set_yticklabels([f"{m} ({SURROGATES[m][0] if m in SURROGATES else 'SNR, rq03'})"
                                              for m in order], fontsize=6.8)
        ax.set_xlabel("Spearman ρ with DA-size (proxy → 1.7B)"); ax.set_title(kind, loc="left")
        ax.grid(axis="x", color=S.GRID, lw=.6); ax.set_axisbelow(True); S.clean(ax); ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="lower right")
    top = G._header(fig, "Which proxy-only statistic predicts decision accuracy?",
                    f"point = Spearman ρ over the (task, proxy) cells above chance at the proxy and at {TARGET_SIZE}, proxies "
                    f"{', '.join(SMALL_SIZES)} pooled; bar = 90 % band from resampling tasks; dashed = DA-size against "
                    f"itself with the reference read at 90 % (the ceiling); n per point in the CSV")
    fig.tight_layout(rect=(0, 0, 1, top))
    c.to_csv(out_dir / "surrogates_catalogue.csv", index=False)
    S.save_figure(fig, out_dir, "surrogates_catalogue")


def plot_languages(bl: pd.DataFrame, out_dir: Path, names: list[str]) -> None:
    g = bl[bl["axes"] == AXES[0]]
    mat = g.pivot_table(index="surrogate", columns="language", values="rho").reindex(names).dropna(how="all")
    cnt = g.pivot_table(index="surrogate", columns="language", values="n_tasks").reindex(mat.index)
    mat = mat[mat.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(8, .32 * mat.shape[1] + 3), .28 * mat.shape[0] + 2))
    cmap = S.DIV.copy()
    cmap.set_bad("white")                                          # rule 12: white = no value
    im = ax.imshow(mat.to_numpy(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, fontsize=7)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.02, pad=.01, label="Spearman ρ")
    top = G._header(fig, "Surrogates of decision accuracy, per language (descriptive)",
                    f"cell = Spearman ρ of the statistic with DA-size over the language's (benchmark, proxy) cells above "
                    f"chance, multi-axis pairs; white = fewer than {MIN_LANG_TASKS} tasks (rule 8); task counts in the CSV")
    fig.tight_layout(rect=(0, 0, 1, top))
    mat.join(cnt.add_prefix("n_tasks_")).to_csv(out_dir / "surrogates_catalogue_by_language.csv")
    S.save_figure(fig, out_dir, "surrogates_catalogue_by_language")


def generate_readme(pool: str, corr, bl, bg, ceil, comb) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    blocks = []
    for axes in AXES:
        c = corr[(corr["axes"] == axes) & (corr["proxy_size"] == "all")]
        rows = []
        for m, g in c.groupby("surrogate"):
            b, p = g[g["kind"] == "benchmark"], g[g["kind"] == "bpb"]
            fam = SURROGATES[m][0] if m in SURROGATES else "SNR (rq03)"
            rows.append([f"`{m}`", fam,
                         f"{fmt(b['rho'].iloc[0])} [{fmt(b['rho_lo'].iloc[0])}, {fmt(b['rho_hi'].iloc[0])}] ({int(b['n'].iloc[0])})" if len(b) else "",
                         fmt(b["rho_partial_items"].iloc[0]) if len(b) else "",
                         f"{fmt(p['rho'].iloc[0])} ({int(p['n'].iloc[0])})" if len(p) else "",
                         b["rho"].iloc[0] if len(b) else -9])
        rows.sort(key=lambda r: -r[-1])
        blocks.append(f"**{axes} pairs** — Spearman ρ with DA-size over the (task, proxy) cells, proxies pooled, "
                      "90 % task-bootstrap band, n cells; `partial` = the benchmark ρ with the item count's ranks regressed out "
                      "(does the statistic say more than the benchmark's size?):")
        blocks.append(md_table(["statistic", "family", "benchmarks ρ [90 %] (n)", "partial", "BPB ρ (n)"],
                               [r[:-1] for r in rows]))
    ce = ceil[ceil["axes"] == AXES[0]].pivot_table(index="kind", columns="proxy_size", values="rho")
    blocks.append("**The ceiling** — Spearman ρ of DA-size with itself when the reference is read at 90 % instead of "
                  "100 % of its run (multi-axis). No surrogate can track DA better than DA tracks itself:")
    blocks.append(md_table(["kind"] + [s for s in SMALL_SIZES if s in ce.columns],
                           [[k] + [fmt(ce.loc[k, s]) for s in SMALL_SIZES if s in ce.columns] for k in ce.index]))
    blocks.append("**All statistics together** (gradient-boosted trees, leave-one-language-out; ρ of the held-out "
                  "prediction with DA-size):")
    blocks.append(md_table(["pairs", "kind", "ρ", "n", "languages"],
                           [[r.axes, r.kind, fmt(r.rho), r.n, r.n_languages] for r in comb.itertuples()]))
    g = bl[bl["axes"] == AXES[0]]
    best = g.loc[g.groupby("language")["rho"].idxmax()].sort_values("rho", ascending=False)
    wins = best["surrogate"].value_counts()
    blocks.append(f"**Per language** (descriptive, rule 8: ≥ {MIN_LANG_TASKS} tasks; multi-axis). Best statistic per "
                  f"language, {len(best)} languages; most frequent winners: "
                  + ", ".join(f"`{m}` {n}" for m, n in wins.head(5).items()) + ". Mean ρ over languages, top 8:")
    mean = g.groupby("surrogate").agg(rho=("rho", "mean"), langs=("language", "nunique"),
                                      pos=("rho", lambda r: int((r > .3).sum()))).sort_values("rho", ascending=False)
    blocks.append(md_table(["statistic", "mean ρ", "languages", "languages with ρ > 0.3"],
                           [[f"`{m}`", fmt(r.rho), int(r.langs), int(r.pos)] for m, r in mean.head(8).iterrows()]))
    for col, label in (("tier", "language tier (smallest L that trains it)"), ("benchmark", "benchmark family")):
        h = bg[(bg["axes"] == AXES[0]) & bg[col].notna()]
        top = h.loc[h.groupby(col)["rho"].idxmax()].sort_values(col)
        blocks.append(f"**Per {label}** — the best statistic of each group (multi-axis, ≥ {MIN_LANG_TASKS} tasks):")
        blocks.append(md_table([col, "best statistic", "ρ", "n cells", "tasks"],
                               [[r[col], f"`{r.surrogate}`", fmt(r.rho), int(r.n), int(r.n_tasks)] for _, r in top.iterrows()]))
    blocks += [f"![Surrogate catalogue]({stage}/{pool}/surrogates_catalogue.png)",
               f"![Surrogate catalogue per language]({stage}/{pool}/surrogates_catalogue_by_language.png)"]
    body = "\n\n".join([
        "## A catalogue of surrogates beyond SNR",
        f"Numbers from the `{pool}` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool {pool}`. "
        f"Every statistic is read on the proxy alone (rule 11): its ten tenths, its noise window and the rungs below it. "
        f"The truth is DA-size, proxy final → {TARGET_SIZE} final, over the same pair set. Population: the tasks above "
        f"chance at the proxy and at {TARGET_SIZE} (rule 1), so n differs per statistic and proxy (rule 13). "
        "Definitions and sources: [`literature.md`](literature.md).",
        *blocks])
    replace_block(OUT_ROOT / "README.md", "catalogue", body, f"catalogue.py --pool {pool}")


def main(pool: str, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    df = ladder_frame(pool)
    print(f"{pool}: {df['model'].nunique()} models, {df['task'].nunique()} tasks")
    v = compute(df, load_mask(pool))
    snr = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv", index_col=0)
    for col, var in SNR_REF.items():
        v[col] = [snr.at[t, f"snr_{var}_{s}"] if t in snr.index and f"snr_{var}_{s}" in snr.columns else np.nan
                  for t, s in zip(v["task"], v["proxy_size"])]
    v = languages_only(v)                                                      # rule 7
    v["tier"] = v["language"].map(tier_of_language())
    names = [m for m in [*SURROGATES, *SNR_REF] if m in v.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    v.to_csv(out_dir / "surrogate_values.csv", index=False)
    corr = correlations(v, names, np.random.default_rng(0))
    corr.to_csv(out_dir / "surrogate_correlations.csv", index=False)
    bl = languages_only(by_group(v, names, "language", MIN_LANG_TASKS))
    bl.to_csv(out_dir / "surrogate_by_language.csv", index=False)
    bg = pd.concat([by_group(v, names, "tier", MIN_LANG_TASKS), by_group(v, names, "benchmark", MIN_LANG_TASKS)])
    bg.to_csv(out_dir / "surrogate_by_group.csv", index=False)
    ceil = retest(v)
    ceil.to_csv(out_dir / "da_retest.csv", index=False)
    comb = combined(v, [m for m in names if m != "n_pairs"])
    comb.to_csv(out_dir / "surrogate_combined.csv", index=False)
    pd.DataFrame([{"surrogate": m, "family": f, "expected_sign": sg, "formula": fo, "source": so}
                  for m, (f, sg, fo, so) in SURROGATES.items()]).to_csv(out_dir / "surrogate_definitions.csv", index=False)
    plot(corr, ceil, out_dir, names)
    plot_languages(bl, out_dir, names)
    generate_readme(pool, corr, bl, bg, ceil, comb)
    print(f"Wrote → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
