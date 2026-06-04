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
   7 shared English tasks rises with seeds **0.75 (1 seed) → 0.84 (2 seeds) →
   0.92 (3 seeds)**. The pure 3-seed pool gives the tightest like-for-like fit
   (**r = 0.92, p = 0.003**), even though absolute SNR is ~5× higher on
   AllenAI's wider model range.
2. **On the pure 3-seed pool the rank order agrees too** (Spearman ρ = **0.93**,
   best variant `star_discrepancy_shifted`) — agreement on **both** value and
   rank over all 7 shared tasks. (At n=7, Spearman is noisy and variant-dependent
   in the other pools.)
3. **Discrepancy / dispersion variants transfer; relative-spread does not.**
   The cross-corpus winners are `star_discrepancy_shifted` / `discrepancy` /
   `rms_deviation` / `dispersion`. The relative-spread family (`rel_std`,
   `rel_mpd`, `iqr`) — incl. AllenAI's own default `rel_std` — is robust
   *within* a corpus but correlates weakly *across* corpora.
4. **For cross-corpus claims, use the pure custom pool, not the external one.**
   On `custom_swissai_hf` the above-random gate drops the at-chance translated
   MCQA, shrinking the shared universe from **7 to 4 tasks**; over those 4,
   `mpsd` gives r = 0.996 / ρ = 1.0 — perfect but on too few tasks to compare
   like-for-like with the pure-pool 7-task fit. The externals are for
   scaling/power (RQ1), not for the AllenAI comparison.

## Cross-corpus correlation by pool

Best variant per pool, Pearson r over the 7 shared English tasks
([`pearson_r_per_variant.csv`](pretraining/seeds_28_1797_1904/pearson_r_per_variant.csv)):

| pool | best cross-corpus variant | Pearson r | Spearman ρ |
|---|---|---:|---:|
| `seeds_1904` (1 seed) | `dispersion` / `range` | 0.751 | 0.786 |
| `seeds_28_1797` (2 seeds) | `discrepancy` | 0.836 | 0.643 |
| **`seeds_28_1797_1904`** (3 seeds) | **`star_discrepancy_shifted`** | **0.924** | **0.929** |
| `custom_swissai_hf` (+ externals) | `mpsd` | 0.996 | 1.000 |

(Pearson on log₁₀ SNR; Spearman on rank. The pure pools share all **7** tasks;
`custom_swissai_hf` shares only **4** after the above-random gate, so its
r = 0.996 / ρ = 1.0 is over a much smaller universe — indicative, not
comparable. The pure 3-seed pool is the like-for-like result.)

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

## Rank agreement over the 7 shared tasks

The honest agreement metric, since the universe is only 7 tasks
([`shared_task_agreement.csv`](pretraining/seeds_28_1797_1904/shared_task_agreement.csv)):

| pool / best variant | Pearson r (values) | Spearman ρ (rank) |
|---|---:|---:|
| **`seeds_28_1797_1904`** / `star_discrepancy_shifted` | **0.92** | **0.93** |
| `custom_swissai_hf` / `mpsd` (n=4) | 1.00 | 1.00 |

On the pure 3-seed pool both the SNR *values* and their *rank order* agree
across corpora over all 7 shared tasks (`star_discrepancy_shifted`, ρ = 0.93).
The comprehensive `custom_swissai_hf` pool now keeps only 4 shared tasks after
the gate, so its perfect ρ = 1.0 reflects a 4-task universe, not a tighter fit.

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
