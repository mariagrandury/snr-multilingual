# RQ3 — Does our SNR ranking agree with AllenAI DataDecide?

## Research question

> Do the SNR variants and the "reliable benchmark" set we find on the Apertus
> corpus also hold on AllenAI's DataDecide / OLMo corpus, on the English
> benchmarks both share?

> ⚠️ **Only 7 English benchmarks overlap** (arc_challenge, arc_easy, csqa,
> hellaswag, mmlu, openbookqa, piqa). With a 7-task universe, *set*-overlap
> metrics (top-K Jaccard) are uninformative — any K ≥ 7 spans the whole universe
> and is trivially 1.0. The real evidence is the **correlation of SNR over those
> 7 tasks** (values: Pearson r; ranking: Spearman ρ), not the overlap.

<!-- BEGIN auto:highlight (analyze.py --pool custom_swissai_hf) -->
## Highlighted result

- **On the pure 3-seed pool (`seeds_28_1797_1904`) both SNR values and rank order agree across corpora** — best variant `star_discrepancy_shifted`, Pearson r of log₁₀(SNR) **0.92**, Spearman ρ of the rank order **0.93** over the 7 shared English tasks.
- **The value correlation rises with seeds** — Pearson r 0.75 → 0.84 → 0.92 (1 → 2 → 3 seeds): more seeds tighten the cross-corpus SNR fit.
- **Dispersion + discrepancy families transfer; relative-spread does not** — the cross-corpus winners are discrepancy/dispersion variants (`dispersion`, `discrepancy`, `star_discrepancy_shifted`), not the mean-normalised relative-spread family (incl. AllenAI's own `rel_std`).
- **Only 7 English tasks overlap, so the *correlation* is the result, not top-K Jaccard** (any K ≥ 7 spans the whole universe → Jaccard ≡ 1.0). On `custom_swissai_hf` the above-random gate shrinks the shared set to n_shared = **4**, so use the pure pool for the like-for-like fit.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the four model-set tiers:
`seeds_1904` (1 seed) · `seeds_28_1797` (2 seeds) · `seeds_28_1797_1904`
(3 seeds) · `custom_swissai_hf` (3 seeds + externals). For each pool we compute,
over the shared English tasks, the cross-corpus Pearson r of log₁₀(SNR@1B)
(values) and Spearman ρ (rank order), reporting the best-correlating variant.
The pure 3-seed pool `seeds_28_1797_1904` is the canonical, like-for-like
comparison: adding external models shifts the shared-task SNR and the
above-random gate drops the at-chance translated MCQA, so the comprehensive
`custom_swissai_hf` pool ends up with a smaller shared universe (use it for
scaling/power in RQ1, not for the AllenAI comparison).

> ⚠️ **Methodological caveat — MMLU aliasing.** Apertus's
> `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's
> `mmlu[_<subject>]` rows so the comparison can use the MMLU subjects, but
> **the two are not the same content**: Apertus runs the Cohere-Full
> translation/post-edit of MMLU (English split), AllenAI runs the original
> Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may
> differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual
> Apertus checkpoints, then drop the alias and compare like-for-like. See
> `pretraining/<pool>/agreement.md` for the full caveat.

<!-- BEGIN auto:results (analyze.py --pool custom_swissai_hf) -->
## Results

Cross-corpus agreement by pool (headline = the pure 3-seed pool `seeds_28_1797_1904`). Regenerate with `python results/allenai_comparison/analyze.py --pool custom_swissai_hf`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), each pool's best cross-corpus variant. The pure pools share all 7 tasks; `custom_swissai_hf` shares fewer after the above-random gate, so it is indicative, not comparable:

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_1904` (1 seed) | `dispersion` | 0.75 | 0.79 | 7 |
| `seeds_28_1797` (2 seeds) | `discrepancy` | 0.84 | 0.64 | 7 |
| `seeds_28_1797_1904` (3 seeds) | `star_discrepancy_shifted` | 0.92 | 0.93 | 7 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)
<!-- END auto:results -->

## TODO

- [ ] Add `mmlu_pro` / BBH to widen the 7-task shared universe.
- [ ] Bootstrap CIs on the cross-corpus Pearson r and Spearman ρ.
- [ ] Re-run the original `mmlu` lm-eval task on Apertus and drop the MMLU alias
      for a like-for-like comparison.

## Files

- `pretraining/<pool>/pearson_r_per_variant.csv` — cross-corpus Pearson r for
  every SNR variant (the headline table).
- `…/shared_task_agreement.csv` — best cross-corpus variant + Pearson r /
  Spearman ρ over the shared tasks (the per-pool result row).
- `…/pearson_r_size_sweep.csv` — r vs the size used for the comparison.
- `…/agreement.csv`, `agreement.md` — top-K reliable-benchmark overlap +
  the MMLU-aliasing caveat.
- `…/top_apertus.csv`, `top_allenai.csv`, `task_overlap.csv`.
- `…/snr_apertus_vs_snr_allenai_*.png` — per-variant + grid scatters.
