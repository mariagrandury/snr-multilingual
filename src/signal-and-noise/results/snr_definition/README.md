# SNR variants × decision accuracy on Apertus

> Of the 22 candidate SNR definitions, which best correlates with **decision
> accuracy** — the actual probability that a benchmark ranks two models the
> way a larger-model evaluation would? And does the answer hold when we
> change the training seed?

## TL;DR

**Use `quartile_deviation` (or any dispersion-family variant) as the global
default.** Across 12 languages, mean Pearson r between log10(SNR) and
decision accuracy is **+0.343** on the train pool — leading a tight 6-way
dispersion tie. The global ranking of variants is **highly stable across
seed splits** (Spearman ρ = +0.83 on DA-size, +0.91 on DA-ckpt), but the
*exact* per-language argmax is not — only 1/14 languages keep the same
pick, and 2/14 keep the same family. Per-language tuning that promises a
correlation above the dispersion baseline should be treated as overfitting
until validated on a second seed pool.

**Best benchmark in each language** under the global-best variant: a
`multiblimp_<lang>` row wherever it exists (ar, en, es, eu, hi, ru, tr) by
3–5× the runner-up. Without multiblimp, the top slot is
`xstorycloze_<lang>` (sw, th, vi, zh) or `hellaswag_<lang>` (ja). Drop
`xnli_<lang>` rows where DA-size = 0 — they're high-SNR but mis-rank
relative to the 1B target.

![Top-5 benchmarks per language by SNR (quartile_deviation @ 1B)](seeds_28_1797/top_benchmarks_per_language.png)

## Headline — global best variant (train pool: `seeds_28_1797`)

**`quartile_deviation`** at mean Pearson r ≈ **+0.343** (DA-size), narrowly
leading a 6-way dispersion-cluster tie. Numbers below are the first 8 rows
of [`seeds_28_1797/top_variants_overall.csv`](seeds_28_1797/top_variants_overall.csv)
(full table: all 22 variants):

| variant | DA-size r | DA-ckpt r | overall |
|---|---:|---:|---:|
| **`quartile_deviation`** | **0.343** | 0.314 | **0.329** |
| `aad` | 0.334 | 0.311 | 0.323 |
| `mpd` | 0.324 | 0.307 | 0.316 |
| `rms_deviation` | 0.323 | 0.305 | 0.314 |
| `dist_std` | 0.318 | 0.302 | 0.310 |
| `dispersion`, `range` | 0.308 | 0.293 | 0.300 |
| `gini` | 0.301 | 0.222 | 0.262 |
| `iqr` | 0.299 | 0.271 | 0.285 |

The dispersion block (`quartile_deviation`, `aad`, `mpd`, `rms_deviation`,
`dist_std`, `dispersion`, `range`) is algebraically redundant at our pool
size — inter-variant Pearson r ≥ 0.999 — so any member is interchangeable
as the default. **Avoid `tukey` and `projection`**: half-space depth needs
more dimensions than 3 mixes × 2 seeds = 6 model units provide, and the
correlation with DA collapses.

> **Pooled-pool note.** Numbers above come from the `seeds_28_1797` train
> pool. For the full-pool `seeds_28_1797_1904` run (recommended for
> downstream work), the winner shifts toward the **relative-spread**
> family — `rel_mpd` at r=+0.405 (DA-size), `aad` at r=+0.380 (DA-ckpt),
> with `quartile_deviation` / `rms_deviation` / `mpd` / `dist_std` / `iqr`
> all in the +0.36–0.38 band. The dispersion family stays on top; the
> rel_spread members climb noticeably as more (mix, seed) variation
> enters the pool. See
> [`seeds_28_1797_1904/top_variants_overall.csv`](seeds_28_1797_1904/top_variants_overall.csv).

## Best variant per language (train pool, DA-size)

Full table: [`seeds_28_1797/best_variant_per_language.csv`](seeds_28_1797/best_variant_per_language.csv).

| lang | best variant | r | runner-up | r |
|---|---|---:|---|---:|
| tr | `mpsd` | **+0.875** | `rel_mpsd` | +0.867 |
| vi | `dist_std` | **+0.683** | `rms_deviation` | +0.677 |
| ru | `iqr` | **+0.627** | `rel_mpd` | +0.621 |
| en | `iqr` | **+0.602** | `rel_mpd` | +0.589 |
| th | `tukey` | +0.563 | `rel_star_discrepancy` | +0.513 |
| hi | `star_discrepancy_shifted` | +0.543 | `gini` | +0.520 |
| es | `dispersion_shifted` | +0.538 | `gini` | +0.489 |
| ar | `discrepancy` | +0.534 | `star_discrepancy` | +0.495 |
| sw | `quartile_deviation` | +0.512 | `aad` | +0.462 |
| eu | `discrepancy` | +0.315 | `gini` | +0.277 |
| zh | `dispersion_shifted` | +0.258 | `star_discrepancy` | +0.217 |
| ja | `star_discrepancy_shifted` | +0.034 | `dist_std` | +0.025 |

`de` and `fr` are dropped — ≤4 valid (task, size) cells on either pool.
Per-language r values are **2–3× stronger than on the single-seed test
pool** because the train pool has 6 (mix, seed) units per size vs 3.

![Best variant per language — DA-size vs DA-ckpt](seeds_28_1797/best_variant_per_language.png)

Variant-family rollup (DA-size, train pool):

| family | langs |
|---|---|
| **dispersion** | sw, tr, vi |
| **discrepancy** | ar, es, eu, hi, ja, zh |
| **rel_spread** | en, ru |
| **robust (`mad`)** | (none in train) |
| **depth** | th (via `tukey`) |

Test pool (single seed, `seeds_1904`) — for comparison:

| family | langs |
|---|---|
| **dispersion** | eu, hi |
| **discrepancy** | en, es, sw, vi |
| **rel_spread** | zh |
| **robust (`mad`)** | ar, ja, tr |
| **depth** | th (via `projection`) |

Three things to read out of this:

1. **`discrepancy` is the largest family in both splits.** 3–5 of the 12
   covered languages no matter which seed pool — the most defensibly
   "portable" family beyond the dispersion default.
2. **Thai always wants `depth`**, but the within-family variant flips
   (`tukey` on train, `projection` on test) — the depth family makes sense
   for one specific language; the within-family choice is seed-dependent.
3. **No typological clustering.** The dispersion winners span
   Indo-European (eu, hi, sw) and Sinitic (zh); the discrepancy winners
   span Romance (es), Slavic (ru), Semitic (ar) and Basque (eu).
   Best-variant choice isn't tracking language family.

## Top benchmarks per language

Using the global-best DA-size variant **`quartile_deviation`** at 1B
(full table:
[`seeds_28_1797/top_benchmarks_per_language.csv`](seeds_28_1797/top_benchmarks_per_language.csv)):

- **`multiblimp_<lang>` is rank-1 wherever it exists** (ar, en, es, eu, hi,
  ru, tr) — by 3–5× the runner-up. Minimal-pair grammatical contrasts are
  uniquely sharp signal.
- **Without multiblimp**, the top slot is `xstorycloze_<lang>` (sw, th, vi,
  zh) or `hellaswag_<lang>` (ja).
- **`xnli_<lang>` rows have DA-size = 0** (perfect rank disagreement with
  the 1B target) at small sizes despite high SNR. The benchmark's
  mix-ranking points the wrong way — a misleading reliability signal.

**Selection recipe:** use `quartile_deviation` SNR at 1B as the primary
screen, then drop benchmarks where DA-size = 0 to filter out
high-SNR-but-misranked cases (mostly `xnli_<lang>`).

## Framework generalization — does this hold on a held-out seed?

Train on `{28, 1797}`, evaluate on `{1904}`. Full report:
[`seeds_28_1797__vs__seeds_1904/summary.md`](seeds_28_1797__vs__seeds_1904/summary.md).

Source: [`seeds_28_1797__vs__seeds_1904/headline_metrics.csv`](seeds_28_1797__vs__seeds_1904/headline_metrics.csv)
(one row per metric × DA flavor).

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | **7% (1/14)** | **7% (1/14)** |
| Family-level agreement (lang-level) | 14% (2/14) | 14% (2/14) |
| Pearson r between splits (over all 264 variant cells) | +0.300 | +0.696 |
| **Spearman ρ on global variant ranking** | **+0.831** | **+0.908** |
| Retention of train-best on test (r_train_pick / r_test_best) | 53% | 77% |

Per-language details:
[`seeds_28_1797__vs__seeds_1904/per_language_agreement_da_size.csv`](seeds_28_1797__vs__seeds_1904/per_language_agreement_da_size.csv)
and
[`…_da_ckpt.csv`](seeds_28_1797__vs__seeds_1904/per_language_agreement_da_ckpt.csv).
The 264-cell scatter:
[`variant_r_train_vs_test.csv`](seeds_28_1797__vs__seeds_1904/variant_r_train_vs_test.csv).

**The global ranking of which SNR variants correlate with DA is highly
stable across seed splits.** The dispersion cluster is consistently on top
in both pools (Spearman ρ = +0.83 / +0.91).

**The exact best variant per language does not generalize** — 1/14
languages keep the pick, 2/14 keep the family. But picking any
dispersion-cluster member retains 77% of the optimal DA-ckpt correlation
on the held-out seed.

![Per-(language, variant) Pearson r — train vs test split](seeds_28_1797__vs__seeds_1904/variant_r_train_vs_test.png)

The DA-ckpt cluster lies tightly along the diagonal (slope ≈ 1); the
DA-size cluster has a much weaker correlation. **DA-ckpt is more portable**
than DA-size — within-size early-vs-late ckpt ranking is less
seed-sensitive than cross-size ranking.

**Recommendation:** treat `quartile_deviation` (or any dispersion member)
as the **portable global default**. Per-language tuning that promises a
stronger correlation than the dispersion-family baseline does not
generalize across the seed splits we tested.

## Q1–Q4 — the four research questions

Each question gets a CSV + PNG under every seed pool. Read the train pool
for the recommendations; the test and pooled pools serve as cross-checks.

### Q1 — Best SNR definition per language

Train pool:
[`seeds_28_1797/best_variant_per_language.csv`](seeds_28_1797/best_variant_per_language.csv)
+ [`.png`](seeds_28_1797/best_variant_per_language.png).
Same-flavor agreement *within* the train pool is moderate — most languages
flip variant family between DA-size and DA-ckpt. The four languages that
keep the same family across both DA flavors are ar (discrepancy), en
(rel_spread), ru (rel_spread), th (robust) — but none of these survive
the seed split.

### Q2 — Variant-family rollup

[`seeds_28_1797/best_variant_family_per_language.csv`](seeds_28_1797/best_variant_family_per_language.csv)
+ [`.png`](seeds_28_1797/best_variant_family_per_language.png). Same tables
as above, rolled up to 5 mathematical families.

![Best variant family per language](seeds_28_1797/best_variant_family_per_language.png)

### Q3 — Top variants across languages

[`seeds_28_1797/top_variants_overall.csv`](seeds_28_1797/top_variants_overall.csv)
+ [`.png`](seeds_28_1797/top_variants_overall.png) (lollipop).
`quartile_deviation`, `aad`, `mpd`, `rms_deviation`, `dist_std`,
`dispersion` / `range` cluster within ~0.03 of each other on either DA
axis — the inter-variant correlation matrix
[`variant_correlation_matrix.png`](seeds_28_1797/variant_correlation_matrix.png)
shows r ≥ 0.999 for the dispersion block. Any one is interchangeable.

![Top variants overall — DA-size and DA-ckpt](seeds_28_1797/top_variants_overall.png)

### Q4 — Top benchmarks per language under the global-best variant

[`seeds_28_1797/top_benchmarks_per_language.csv`](seeds_28_1797/top_benchmarks_per_language.csv)
+ [`.png`](seeds_28_1797/top_benchmarks_per_language.png).
`multiblimp_<lang>` dominates wherever it exists; `xstorycloze_<lang>` /
`hellaswag_<lang>` cover the rest. SNR magnitudes are higher than the
historical single-seed runs because the train pool has 2× the (mix, seed)
units.

![Top-5 benchmarks per language by SNR with DA annotations](seeds_28_1797/top_benchmarks_per_language.png)

## Key insights

- **The dispersion family is the portable global default.**
  `quartile_deviation` / `aad` / `mpd` / `rms_deviation` / `dispersion` /
  `range` tie within ~0.04 at mean r ≈ 0.33 across the 12 languages, and
  the ranking holds on the held-out seed (Spearman ρ on the global variant
  ranking = **+0.83 DA-size, +0.91 DA-ckpt**; pooled per-cell Pearson r =
  +0.30 / +0.70).
- **Per-language tuning does not generalize across seed splits.** Only
  1/14 (7%) of languages keep the same best variant; 2/14 (14%) keep the
  family. Picking the train-best variant retains 53% of the optimal r on
  DA-size and 77% on DA-ckpt — useful but noticeably degraded.
- **DA-ckpt is more transferable than DA-size.** Within-size early-vs-late
  ckpt ranking is less seed-sensitive than cross-size ranking.
- **The `(mix, seed)` model unit doubles the signal pool.** With 6 model
  units per size (vs 3 under the single-seed Apertus baseline),
  per-language Pearson r values jump 2–3× and the dispersion-cluster
  variants cleanly separate from the others (instead of all sitting near
  +0.27).
- **Algebraic redundancy is unchanged.** Dispersion block r ≥ 0.999;
  relative-spread block (`iqr`, `rel_dispersion`, `rel_mpd`, `rel_std`)
  r ≥ 0.998; `{discrepancy, star_discrepancy}` r = 0.959.
- **Depth variants are mostly useless.** `tukey` and `projection`
  correlate ~0 with DA on the train pool (r = 0.009 and -0.025) —
  half-space depth doesn't work at small n.
- **`rel_std_snr` is no longer degenerate** (since the `data_noise` fix —
  see [CLAUDE.md](../../CLAUDE.md)). It now sits in the relative-spread
  cluster.
- **Sparse-coverage languages are skipped** — `de` and `fr` have ≤4 valid
  (task, size) cells on either pool.

## Methodology

**Three training seeds per (size, mix)** for Apertus: `seed28`, `seed1797`,
`seed1904`. Each (mix, seed) is a separate training run, giving up to
**9 model units per size** (3 mixes × 3 seeds). Plus reference HF models
at 3B/7B/8B/70B feed into the same per-size signal pool when their size
matches an Apertus size (none do yet — Qwen3-0.6B and Apertus-v1.5-1B are
planned).

Three pools coexist under `results/snr_definition/`:

| Pool | Apertus runs | model_families per size | Typical ckpts |
|---|---|---|---|
| `seeds_28_1797` (train) | seed28 + seed1797 × 3 mixes × 4 sizes | 6 | 10–16 |
| `seeds_1904` (test) | seed1904 × 3 mixes × 4 sizes | 3 | 13 |
| `seeds_28_1797_1904` (pooled) | all 3 seeds × 3 mixes × 4 sizes | 9 | 10–16 |

`model_family` is the cross-size identity — the model name with only the
size token stripped (so `apertus-175M-fwEdu30-fw270-seed28` and
`apertus-1B-fwEdu30-fw270-seed28` collapse into one family for DA-size,
but the seed is preserved as part of the family ID — two different seeds
at the same (mix, size) are different families). External HF models
(Qwen3-0.6B, Apertus-v1.5-1B, …) participate in the SNR signal pool by
`model` and contribute to DA via `model_family` when their base release
exists at multiple sizes — so the framework is seed-agnostic at the API
level.

The headline numbers above come from the **`seeds_28_1797` (train) split**
— that's the corpus we use to pick the recommended variant per language.
`seeds_28_1797__vs__seeds_1904/` then checks whether those picks hold on
the held-out seed.

> The parquet still ships per-(lang, subject) facets like
> `global_mmlu_full_ar_anatomy` alongside per-language aggregates.
> `_is_parent_task` in `run_apertus_snr_variants.py` filters those out,
> matching the cluster's old `aggregate_parents` semantics. The CSV is
> 115 parent tasks for the train pool, 121 for the test pool.

## Reproduce

```bash
# Per-pool variants CSV
python multilingual/run_apertus_snr_variants.py --pool seeds_28_1797
python multilingual/run_apertus_snr_variants.py --pool seeds_1904
python multilingual/run_apertus_snr_variants.py --pool seeds_28_1797_1904

# Plots from each CSV
for pool in seeds_28_1797 seeds_1904 seeds_28_1797_1904; do
    python multilingual/analyze_snr_variants.py --pool $pool
    python multilingual/snr_definition_postprocess.py --pool $pool
done

# Train/test framework generalization
python multilingual/compare_seed_splits.py \
    --train-pool seeds_28_1797 \
    --test-pool  seeds_1904
```

## Directory layout

```
results/snr_definition/
├── seeds_28_1797/                            ← train pool (recommended picks)
│   ├── snr_variants_per_task.csv             ← 115 tasks × 279 columns
│   ├── snr_variants_definitions.csv          ← 22 variants metadata
│   ├── best_variant_per_language.{csv,png}   ← Q1
│   ├── best_variant_family_per_language.{csv,png}  ← Q2 rollup
│   ├── variant_clusters.csv                  ← Q2 raw
│   ├── top_variants_overall.{csv,png}        ← Q3
│   ├── top_benchmarks_per_language.{csv,png} ← Q4
│   ├── variant_correlation_matrix.png        ← inter-variant redundancy
│   ├── da_size_vs_da_ckpt.png                ← DA-flavor sanity
│   ├── da_size/                              ← variants × DA-size grids + heatmap
│   │   ├── snr_vs_decision_accuracy.png
│   │   ├── snr_vs_decision_accuracy_<lang>.png  (12 langs)
│   │   └── heatmap_pearson_r.png
│   └── da_ckpt/
│       ├── da_ckpt_mix/                      ← cross-size pooled view
│       ├── da_ckpt_175M/  da_ckpt_350M/
│       ├── da_ckpt_600M/  da_ckpt_1B/
│       └── (each subdir mirrors the da_size/ layout)
├── seeds_1904/                               ← held-out test pool (same layout)
├── seeds_28_1797_1904/                       ← pooled all seeds (recommended downstream)
└── seeds_28_1797__vs__seeds_1904/            ← framework generalization
    ├── summary.md                            ← headline agreement metrics
    ├── per_language_agreement_da_size.{csv,png}
    ├── per_language_agreement_da_ckpt.{csv,png}
    ├── variant_r_train_vs_test.{csv,png}
    └── top_variants_train_vs_test.csv
```

Generated by:

- [`multilingual/run_apertus_snr_variants.py`](../../multilingual/run_apertus_snr_variants.py) — writes `snr_variants_per_task.csv`
- [`multilingual/analyze_snr_variants.py`](../../multilingual/analyze_snr_variants.py) — reads the CSV, writes the SNR-vs-DA grids and heatmaps
- [`multilingual/snr_definition_postprocess.py`](../../multilingual/snr_definition_postprocess.py) — Q1–Q4 tables and plots
- [`multilingual/compare_seed_splits.py`](../../multilingual/compare_seed_splits.py) — train/test generalization across seed pools
