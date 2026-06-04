# RQ3 — Does our SNR ranking agree with AllenAI DataDecide?

> Do the SNR variants and the "reliable benchmark" set we find on the Apertus
> corpus also hold on AllenAI's DataDecide / OLMo corpus, on the English
> benchmarks both share?

Outputs under `pretraining/<pool>/`. The shared universe is **7 English
benchmarks** (arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa,
piqa). ⚠️ MMLU is aliased (Apertus Cohere-Full translation vs AllenAI's
Hendrycks original — 1 of the 7) — see `agreement.md`.

> ⚠️ **Only 7 English benchmarks overlap.** With a 7-task universe, *set*-overlap
> metrics (top-K Jaccard) are uninformative — any K ≥ 7 spans the whole universe
> and is trivially 1.0. The real evidence is the **correlation of SNR over those
> 7 tasks** (values: Pearson r; ranking: Spearman ρ), not the overlap.

## TL;DR (paper takeaways)

1. **SNR *values* transfer across corpora.** Pearson r of log₁₀(SNR₁B) over the
   7 shared English tasks is **0.84** (comprehensive pool), and rises with seeds
   **0.75 (1 seed) → 0.84 (2 seeds) → 0.92 (3 seeds)**. The pure 3-seed pool
   gives the tightest like-for-like fit (**r = 0.92, p = 0.003**), even though
   absolute SNR is ~5× higher on AllenAI's wider model range.
2. **On the pure 3-seed pool the rank order agrees too** (Spearman ρ = **0.93**,
   best variant `star_discrepancy_shifted`). On the comprehensive pool the value
   correlation holds (0.84) but the rank is looser (ρ = 0.64) — at n=7 Spearman
   is noisy and variant-dependent. Top-5 sets share 4/5.
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

| pool | best cross-corpus variant | Pearson r | Spearman ρ |
|---|---|---:|---:|
| `seeds_1904` (1 seed) | `dispersion` / `range` | 0.751 | 0.786 |
| `seeds_28_1797` (2 seeds) | `discrepancy` | 0.836 | 0.643 |
| **`seeds_28_1797_1904`** (3 seeds) | **`star_discrepancy_shifted`** | **0.924** | **0.929** |
| `custom_swissai_hf` (+ externals) | `rms_deviation` | 0.837 | 0.643 |

(Pearson on log₁₀ SNR; Spearman on rank — both over the 7 shared tasks, so
n=7: indicative, not tight. The pure 3-seed pool agrees on **both** value and
rank; Spearman is variant-dependent and noisy elsewhere.)

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

## Rank agreement over the 7 shared tasks

The honest agreement metric, since the universe is only 7 tasks
([`shared_task_agreement.csv`](pretraining/seeds_28_1797_1904/shared_task_agreement.csv)):

| pool / best variant | Pearson r (values) | Spearman ρ (rank) |
|---|---:|---:|
| **`seeds_28_1797_1904`** / `star_discrepancy_shifted` | **0.92** | **0.93** |
| `custom_swissai_hf` / `rms_deviation` | 0.84 | 0.64 |

On the pure 3-seed pool both the SNR *values* and their *rank order* agree
across corpora. Per-corpus order (comprehensive pool, `rms_deviation`): Apertus
`piqa > arc_easy > hellaswag > csqa > mmlu > arc_challenge > openbookqa`;
AllenAI `arc_easy > hellaswag > mmlu > piqa > arc_challenge > csqa > openbookqa`
— extremes agree (`openbookqa` last; `arc_easy`/`hellaswag` near top), the
middle reshuffles.

> **Do not report top-K Jaccard here.** With 7 shared tasks, any K ≥ 7 returns
> the whole universe on both sides → Jaccard ≡ 1.0 by construction (the script
> now drops K ≥ N; only K=5 = 0.67 is non-degenerate). The result is the
> *correlation*, not the set overlap.

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

## Files

- `pretraining/<pool>/pearson_r_per_variant.csv` — cross-corpus Pearson r for
  every SNR variant (the headline table).
- `…/pearson_r_size_sweep.csv` — r vs the size used for the comparison.
- `…/agreement.csv`, `agreement.md` — top-K reliable-benchmark overlap +
  the MMLU-aliasing caveat.
- `…/top_apertus.csv`, `top_allenai.csv`, `task_overlap.csv`.
- `…/snr_apertus_vs_snr_allenai_*.png` — per-variant + grid scatters.
