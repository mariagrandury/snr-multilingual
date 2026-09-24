"""A catalogue of surrogates of decision accuracy, and the truths they are scored against.

Surrogates, one row per (task, proxy size, pair set), all read on the PROXY
alone (rule 11: its final checkpoint, its ten tenths, its noise window (rule
4) and the rungs below it, never the reference):

  * ~50 statistics from the literature (`SURROGATES`, sources in `literature.md`);
  * the SNR grid: every AllenAI signal (the 22 aggregators of
    `snr/snr_variants.py`) against every noise in `NOISES`, from rq03's own
    per-model arrays and aggregators (`per_model_inputs`,
    `variant_signal_noise_snr`) so the (`rel_std`, `ckpt_rel`) cell IS rq03's
    `snr_rel_std`. The noises: the AllenAI checkpoint noise (relative, as
    AllenAI; and absolute), the two depth noises of `tukey` / `projection`,
    and the benchmark (k-fold) noise, relative and absolute.

    The k-fold benchmark noise needs per-item outputs, which the ladder report
    does not carry. For n binary items at accuracy p, the fold accuracies of a
    random k-fold partition have expected variance p(1-p)(k-1)/(n-1) exactly
    (hypergeometric; E over partitions of (1/k) sum_i (m_i - p)^2), so its
    root is used: the noise the k-fold estimator measures on average, with
    k = K_FOLDS. Its ranking across tasks does not depend on k.

Truths, one long row per (task, proxy, fraction, kind, metric, pair set, L),
gated (rule 1: DA-size / DA-goal at the proxy and the reference, DA-ckpt at
the proxy, `reliable_tasks.GATE_REF`) and at >= MIN_PAIRS pairs (rule 5):

  kind    size (proxy final vs reference final), goal (proxy at 10-90 % vs
          reference final), ckpt (proxy at 10-90 % vs its own final)
  metric  da: rq02's own table (`reliable_tasks.long_da`), both pair sets;
          tau_b, rho: `utils.agreement_measures` on rq02's checkpoint choice
          (`compute_da._scores_at`), every pair (multi-axis) only — Spearman is
          not defined on a pair subset, and on a subset tau is DA under another
          tie convention (`agreement_measures`' identity)
  L       all, or one language count: DA over the pairs of variants that share
          the L (`by_L._da_table` on this pool; an L with < MIN_PAIRS pairs
          is left out)
  retest  the same truth with the reference (the proxy's own final for ckpt)
          read at 90 % of its run: the ceiling any surrogate faces

    surrogate_values.csv       surrogates per (task, proxy_size, axes)
    surrogate_targets.csv      truths, long
    surrogate_definitions.csv  every surrogate: family, expected sign, formula, source

`search.py` scores the surrogates against the truths.

    python analysis/rq04_surrogates/catalogue.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, pearsonr, rankdata, spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from snr.snr_variants import AGGREGATION_FUNCTIONS  # noqa: E402
from snr.stats import calc_monotonicity, calc_total_variation  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY, SURROGATES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_items, task_n_options  # noqa: E402
from analysis.rq02_decision_accuracy.by_L import _da_table  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import _scores_at  # noqa: E402
from analysis.rq02_decision_accuracy.language_tier import tier_of_language  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import GATE_REF, long_da  # noqa: E402
from analysis.rq03_noise_and_snr.run_apertus_snr_variants import (  # noqa: E402
    DISCREPANCY_UNIT_INTERVAL, per_model_inputs, variant_key, variant_signal_noise_snr)
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, MIN_PAIRS, NOISE_GRID, NON_EMB, PAIR_AXES, SMALL_SIZES, TARGET_SIZE, agreement_measures,
    assign_language, benchmark_family, design_axes, ladder_frame, languages_only, noise_checkpoints,
    on_shared_grid, pair_agreement, pair_sets, passes_gate)

OUT_ROOT = SURROGATES
AXES = PAIR_AXES[:2]                   # the design pair sets; the seed null has no decision to predict
FRACS = [*CKPT_DA_EARLY_FRACS, 1.0]
RETEST_FRAC = 0.9                      # the reference re-read one tenth earlier for the ceiling
K_FOLDS = 5                            # the benchmark noise's folds (the value of the 2026-04 slides)
Z_CAP = 50.0                           # a gap over noise past this is "separated"; keeps the median finite
DEPTH = ("tukey", "projection")        # the two aggregators whose noise is their own
NOISES = ("ckpt_rel", "ckpt_abs", "tukey_depth", "projection_depth", "kfold_rel", "kfold_abs")
SIGNALS = [variant_key(fd) for fd in AGGREGATION_FUNCTIONS]
N_JOBS = 4
_DF: pd.DataFrame | None = None        # the pool, shared with the workers by fork

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


def compute(df: pd.DataFrame, mask: pd.DataFrame | None, tasks: list[str]) -> pd.DataFrame:
    """One row per (task, proxy, axes) of `tasks`: every catalogue statistic, read on
    the proxy's final checkpoint, its ten tenths, its window and the rungs below it."""
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
    every = sorted(set(df["task"]) - {"bpb_macro", "train_loss"})
    lang = {t: assign_language(t) for t in every}
    bpb_of = pd.Series({lang[t]: t for t in every if t.startswith("bpb_")})
    bench = {t: benchmark_family(t) for t in every}
    pseudo = SMALL_SIZES[-1]                                        # the largest rung below the reference
    rows = []
    for si, s in enumerate(SMALL_SIZES):
        C_s, W_s, F_s = curves[s], windows[s], finals[s]
        Z_s = _zscore(F_s)
        gate_s = passes_gate(mask, every, s)
        for t in tasks:
            if t not in F_s.columns or F_s[t].notna().sum() < 3:
                continue
            fams = F_s[t].dropna().index.tolist()
            C = C_s.loc[t].reindex(fams).to_numpy()
            W = W_s.loc[t].reindex(fams).to_numpy() if t in W_s.index.get_level_values(0) else np.full((len(fams), 5), np.nan)
            x = C[:, -1]
            for axes in AXES:
                pairs = psets[axes]
                r = {"task": t, "language": lang[t], "benchmark": bench[t],
                     "kind": "bpb" if t.startswith("bpb_") else "benchmark", "proxy_size": s, "axes": axes}
                rng = np.random.default_rng(zlib.crc32(f"{t}|{s}|{axes}".encode()))   # DIoR's draws: fixed per cell
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
    return pd.DataFrame(rows)


def kfold_noise(p: np.ndarray, n_items: float) -> dict:
    """The benchmark (k-fold) noise of models at accuracies `p` on `n_items`
    binary items, in closed form (module docstring): per model the root of
    the expected fold-accuracy variance p(1-p)(K-1)/(n-1); relative (over p,
    averaged over models, as the k-fold definition) and absolute."""
    if not (np.isfinite(n_items) and n_items > 1) or not ((p > 0) & (p < 1)).all():
        return {"kfold_rel": np.nan, "kfold_abs": np.nan}
    sd = np.sqrt(p * (1 - p) * (K_FOLDS - 1) / (n_items - 1))
    return {"kfold_rel": float(np.mean(sd / p)), "kfold_abs": float(np.mean(sd))}


def snr_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, proxy): every AllenAI signal, every noise in NOISES
    and every signal / noise ratio, from rq03's per-model arrays and
    aggregators. The discrepancy signals are NaN off the benchmarks, as in rq03."""
    df = df.assign(bucket=df["size"].map(size_bucket))
    rows = []
    for (t, s), g in df[df["size"].isin(SMALL_SIZES)].groupby(["task", "size"], sort=False):
        inputs = per_model_inputs(g, t, s)
        if inputs is None or t in ("bpb_macro", "train_loss"):
            continue
        bench = benchmark_family(t) not in ("bpb", "loss")
        sig, noi = {}, {}
        for fd in AGGREGATION_FUNCTIONS:
            k = variant_key(fd)
            a, b, _ = variant_signal_noise_snr(inputs, fd["func"])
            sig[k] = np.nan if k in DISCREPANCY_UNIT_INTERVAL and not bench else a
            if k == "rel_std":
                noi["ckpt_rel"] = b                                  # AllenAI's: mean step std / mean window score
            if k in DEPTH:
                noi[f"{k}_depth"] = b
        noi["ckpt_abs"] = float(np.mean(inputs[0]))
        noi |= kfold_noise(inputs[1], task_n_items(t) if bench else np.nan)
        row = {"task": t, "proxy_size": s}
        row |= {f"signal__{k}": v for k, v in sig.items()} | {f"noise__{k}": noi[k] for k in NOISES}
        row |= {f"snr__{a}__{b}": sig[a] / noi[b] if noi[b] and np.isfinite(noi[b]) else np.nan
                for a in SIGNALS for b in NOISES}
        rows.append(row)
    return pd.DataFrame(rows)


def _frac(col: pd.Series, kind: pd.Series) -> np.ndarray:
    """rq02's column name -> the proxy's fraction of its run (1.0 for DA-size)."""
    f = col.str.extract(r"_f(\d+)_")[0].astype(float) / 100
    return np.where(kind == "size", 1.0, f)


def _agreement_task(t: str) -> list[dict]:
    """tau_b, rho and DA of one task, every (proxy, fraction), as
    `agreement_measures` reads rq02's checkpoint choice; `retest` = the same
    with the reference (DA-ckpt: the proxy's own final) at RETEST_FRAC."""
    dft = _DF[_DF["task"] == t]
    ref, ref_rt = _scores_at(dft, TARGET_SIZE, 1.0), _scores_at(dft, TARGET_SIZE, RETEST_FRAC)
    rows = []
    for b in SMALL_SIZES:
        own, own_rt = _scores_at(dft, b, 1.0), _scores_at(dft, b, RETEST_FRAC)
        for f in FRACS:
            got = _scores_at(dft, b, f)
            refs = [("size" if f == 1.0 else "goal", ref, ref_rt)]
            if f < 1.0:
                refs.append(("ckpt", own, own_rt if f < RETEST_FRAC else {}))
            for kind, y, y_rt in refs:
                m, rt = (_measures(got, z) for z in (y, y_rt))
                for metric in ("da", "tau_b", "rho"):
                    rows.append({"task": t, "proxy_size": b, "frac": f, "kind": kind, "metric": metric,
                                 "value": m.get(metric, np.nan), "retest": rt.get(metric, np.nan),
                                 "n_pairs": m.get("n_pairs", 0)})
    return rows


def _measures(got: dict, ref: dict) -> dict:
    common = sorted(set(got) & set(ref))
    if len(common) < 2:
        return {}
    return agreement_measures([got[c][0] for c in common], [ref[c][0] for c in common])


def _by_L_job(job: tuple) -> pd.DataFrame:
    L, axes, pairs = job
    d = pd.DataFrame(_da_table(_DF[_DF["L"] == L], L, pairs))
    if d.empty:
        return d
    return pd.concat([
        d[d["frac"] == 1.0].assign(kind="size", value=d["da_ref"], n_pairs=d["n_pairs_ref"]),
        d[d["frac"] < 1.0].assign(kind="goal", value=d["da_ref"], n_pairs=d["n_pairs_ref"]),
        d[d["frac"] < 1.0].assign(kind="ckpt", value=d["da_own"], n_pairs=d["n_pairs_own"])])[
        ["task", "proxy_size", "frac", "kind", "value", "n_pairs"]].assign(metric="da", axes=axes, L=str(L))


def _gate(t: pd.DataFrame, pool: str, cols=("value",)) -> pd.DataFrame:
    """Rule 1 per kind (`GATE_REF`), rule 5 on `n_pairs`; rows left with no value are dropped."""
    parts = []
    for kind, ref in GATE_REF.items():
        g = t[t["kind"] == kind]
        for c in cols:
            g = G.mark_gated(g, pool, "proxy_size", c, ref)
        parts.append(g)
    t = pd.concat(parts).drop(columns="gated")
    t.loc[t["n_pairs"] < MIN_PAIRS, list(cols)] = np.nan              # rule 5
    return t.dropna(subset=["value"])


def targets(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The long truth table (module docstring)."""
    global _DF
    _DF = df.assign(bucket=df["size"].map(size_bucket))
    stage = load_pools()[pool].get("stage", "pretraining")
    da = []
    for a in AXES:
        d = long_da(DECISION_ACCURACY / stage / pool, pool, axes=a)
        da.append(pd.DataFrame({"task": d["task"], "proxy_size": d["size"], "frac": _frac(d["col"], d["kind"]),
                                "kind": d["kind"], "metric": "da", "axes": a, "L": "all",
                                "value": d["da"], "n_pairs": d["n_pairs"]}))
    da = pd.concat(da)
    da = da[da["proxy_size"].isin(SMALL_SIZES)]                      # DA-ckpt of the reference itself is not a proxy's
    tasks = sorted(set(df["task"]) - {"bpb_macro", "train_loss"})
    with Pool(N_JOBS) as p:
        ag = pd.DataFrame([r for rows in p.map(_agreement_task, tasks) for r in rows]).assign(axes=AXES[0], L="all")
    ag = _gate(ag, pool, ("value", "retest"))
    # DA is rq02's; the DA agreement_measures recomputes must be the same number, or the
    # two tables do not describe the same decisions.
    chk = da[da["axes"] == AXES[0]].merge(ag[ag["metric"] == "da"], on=["task", "proxy_size", "frac", "kind"])
    bad = (chk["value_x"] - chk["value_y"]).abs() > 1e-9
    print(f"  DA check against rq02: {len(chk) - bad.sum()} of {len(chk)} cells identical")
    assert not bad.any(), chk[bad].head()
    da = da.merge(ag.loc[ag["metric"] == "da", ["task", "proxy_size", "frac", "kind", "retest"]].assign(axes=AXES[0]),
                  on=["task", "proxy_size", "frac", "kind", "axes"], how="left")
    mono = pair_sets(design_axes(df))[AXES[1]]
    jobs = [(L, a, p) for L, dl in df.groupby("L") if dl["family"].nunique() * (dl["family"].nunique() - 1) // 2 >= MIN_PAIRS
            for a, p in ((AXES[0], None), (AXES[1], mono))]
    with Pool(N_JOBS) as p:
        byl = _gate(pd.concat(p.map(_by_L_job, jobs)), pool)
    byl = byl[byl["proxy_size"].isin(SMALL_SIZES)]
    out = pd.concat([da, ag[ag["metric"] != "da"], byl], ignore_index=True)
    out["language"] = out["task"].map(assign_language)
    return languages_only(out)                                       # rule 7


def _compute_chunk(tasks: list[str]) -> pd.DataFrame:
    return compute(_DF, load_mask(_POOL), tasks)


def main(pool: str, out_dir: Path) -> None:
    global _DF, _POOL
    df = ladder_frame(pool)
    print(f"{pool}: {df['model'].nunique()} models, {df['task'].nunique()} tasks")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = targets(df, pool)
    t.to_csv(out_dir / "surrogate_targets.csv", index=False)
    print(f"  truths: {len(t)} rows → surrogate_targets.csv")
    _DF, _POOL = df, pool
    tasks = sorted(set(df["task"]) - {"bpb_macro", "train_loss"})
    with Pool(N_JOBS) as p:
        v = pd.concat(p.map(_compute_chunk, [tasks[i::N_JOBS] for i in range(N_JOBS)]), ignore_index=True)
    v = v.merge(snr_grid(df), on=["task", "proxy_size"], how="left")
    v = languages_only(v)                                            # rule 7
    v["tier"] = v["language"].map(tier_of_language())
    v.to_csv(out_dir / "surrogate_values.csv", index=False)
    print(f"  surrogates: {len(v)} rows x {v.shape[1]} columns → surrogate_values.csv")
    defs = [{"surrogate": m, "family": f, "expected_sign": sg, "formula": fo, "source": so}
            for m, (f, sg, fo, so) in SURROGATES.items()]
    defs += [{"surrogate": f"signal__{a}", "family": "AllenAI signal", "expected_sign": 1,
              "formula": f"snr_variants.{a}_snr, signal part", "source": "Heineman 2025"} for a in SIGNALS]
    defs += [{"surrogate": f"noise__{b}", "family": "noise", "expected_sign": -1, "formula": b,
              "source": "Heineman 2025" if not b.startswith("kfold") else "benchmark noise (closed form, K_FOLDS)"}
             for b in NOISES]
    defs += [{"surrogate": f"snr__{a}__{b}", "family": "SNR grid", "expected_sign": 1,
              "formula": f"signal {a} / noise {b}", "source": "Heineman 2025"} for a in SIGNALS for b in NOISES]
    pd.DataFrame(defs).to_csv(out_dir / "surrogate_definitions.csv", index=False)


_POOL = CANONICAL_POOL

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
