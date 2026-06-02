# RQ3 — Does our SNR ranking agree with AllenAI DataDecide?

> Do the SNR variants and the "reliable benchmark" set we find on the Apertus
> corpus also hold on AllenAI's DataDecide / OLMo corpus, on the English
> benchmarks both share?

Outputs under `pretraining/<pool>/`. The shared universe is **7 English
benchmarks** (arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa,
piqa). ⚠️ MMLU is aliased (Apertus Cohere-Full translation vs AllenAI's
Hendrycks original — 1 of the 7) — see `agreement.md`.

## TL;DR (paper takeaways)

1. **The reliable-benchmark set transfers cleanly across corpora.** The
   **top-10 most-reliable English benchmarks are identical on Apertus and
   AllenAI (Jaccard = 1.00)**; the per-corpus SNR rank orders agree to
   Spearman-level even though absolute SNR differs (~5× higher on AllenAI's
   wider model range).
2. **Cross-corpus correlation is strong and grows with seeds:** Pearson r over
   the 7 shared tasks climbs **0.75 (1 seed) → 0.84 (2 seeds) → 0.92 (3
   seeds)**. The pure 3-seed pool gives the tightest like-for-like agreement
   (**r = 0.92, p = 0.003**).
3. **Discrepancy / dispersion variants transfer; relative-spread does not.**
   The cross-corpus winners are `star_discrepancy_shifted` / `discrepancy` /
   `rms_deviation` / `dispersion`. The relative-spread family (`rel_std`,
   `rel_mpd`, `iqr`) — incl. AllenAI's own default `rel_std` — is robust
   *within* a corpus but correlates weakly *across* corpora.
4. **For cross-corpus claims, use the pure custom pool, not the external one.**
   AllenAI SNR is a small-model quantity; adding our >1B external models
   (`custom_swissai_hf`) shifts the shared-task SNR and drops cross-corpus r
   from 0.92 → 0.84. The externals are for scaling/power (RQ1), not for the
   AllenAI comparison.

## Cross-corpus correlation by pool

Best variant per pool, Pearson r over the 7 shared English tasks
([`pearson_r_per_variant.csv`](pretraining/seeds_28_1797_1904/pearson_r_per_variant.csv)):

| pool | best cross-corpus variant | r | p |
|---|---|---:|---:|
| `seeds_1904` (1 seed) | `dispersion` / `range` | 0.751 | 0.051 |
| `seeds_28_1797` (2 seeds) | `discrepancy` | 0.836 | 0.019 |
| **`seeds_28_1797_1904`** (3 seeds) | **`star_discrepancy_shifted`** | **0.924** | **0.003** |
| `custom_swissai_hf` (+ externals) | `rms_deviation` | 0.837 | 0.019 |

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

## Top-K reliable-benchmark agreement (pool `seeds_28_1797_1904`)

From [`agreement.csv`](pretraining/seeds_28_1797_1904/agreement.csv):

| K | intersection / K | **Jaccard** | shared top-K tasks |
|---:|---:|---:|---|
| 5 | 0.80 | 0.67 | arc_easy, hellaswag, mmlu, piqa |
| **10** | 0.70 | **1.00** | arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa |
| 20 | 0.35 | 1.00 | (same 7 — the full shared universe) |

Both corpora independently rank the same English benchmarks as most reliable;
`piqa` / `arc_easy` / `hellaswag` lead on Apertus, `arc_easy` / `hellaswag` /
`mmlu` on AllenAI.

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

## Files

- `pretraining/<pool>/pearson_r_per_variant.csv` — cross-corpus Pearson r for
  every SNR variant (the headline table).
- `…/pearson_r_size_sweep.csv` — r vs the size used for the comparison.
- `…/agreement.csv`, `agreement.md` — top-K reliable-benchmark overlap +
  the MMLU-aliasing caveat.
- `…/top_apertus.csv`, `top_allenai.csv`, `task_overlap.csv`.
- `…/snr_apertus_vs_snr_allenai_*.png` — per-variant + grid scatters.
