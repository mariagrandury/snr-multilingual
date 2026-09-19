# RQ3 — Noise and SNR: how noisy is a measurement, and what is its signal-to-noise ratio?

## Research question

> How much does a measurement move when nothing but the seed or the
> checkpoint changes, how does that noise compare with the effect of a design
> decision, and what is each benchmark's signal-to-noise ratio under the 22
> candidate definitions? The per-task SNR table computed here is what rq04
> ranks against decision accuracy and what rq07 and rq09 read.

## Experimental setup

Outputs live under `pretraining/<pool>/` for the ladder pools: `predictivity`
(the plan grid, seed 1904 — the headline), `predictivity_seeds` (every seed;
replicates enter the signal pool as separate models), and the seed holdout
`predictivity_seeds_train` (seeds 64/313 at the ×3 cells) → `_test` (seed 1904
on the same cells). The signal population at a size is every design variant
trained there (language setting × depth × scheme); the noise is the
late-checkpoint std over the last 5 checkpoints on the shared grid (`effect_vs_noise.py`
below carries the seed-replicate noise). DA has two flavours: **DA-size**
(small→1.7B ranking, plus every other bucket pair) and **DA-ckpt**
(20/40/60/80 % → final within a size). The 22 variants are grouped into
families (dispersion / relative-spread / discrepancy / robust / depth); rq04
correlates them with DA per language. The per-language BPB tasks
(`bpb_<subset>`) take part like any benchmark.
The seed holdout is English-heavy by construction: among the ×3 cells
(L ∈ {1, 2, 50, 100}) a non-English harness task is only evaluated at L50 and
L100, so its per-language variant ranking rests on two language settings per
seed, while English and the BPB tasks cover all four.

## Methodology

- **Signal and noise per (task, size bucket).** Signal pool = every model at
  the bucket (`per_model_inputs`); each model's `data_score` is its final
  checkpoint, `step_noise` the std over its last `last_n` = 5 checkpoints on
  the shared grid. The 22 aggregators in `snr/snr_variants.py` differ in how
  they turn the cross-model scores into a dispersion; all but `rel_std` share
  `noise = mean(step_noise) / mean(last-N means)`. Cells the rq00 gate marks
  at chance are NaN. Variants of the discrepancy family need scores in
  [0, 1] and are undefined on per-language BPB and loss;
  `snr_variant_coverage.csv` records how many cells each variant covers per
  bucket, and the driver prints the ones that fall short.
- **Effect vs noise.** For every (size, L, task) the intervention's |Δ| is put
  against the seed noise (sample std over the seed replicates, where ≥ 2
  seeds exist) and the late-checkpoint noise (std over the last `last_n`
  checkpoints of the baseline cell, raw and detrended — under WSD the final
  window is still descending, so the raw std carries trend). A ratio near 1
  means the two levels are the same model as far as a ranking is concerned
  (the "read this against the seed row" rule of `ladder_report.md`); a
  decision on such a cell is a coin flip whatever its DA (rq05).
- **Seed holdout.** `compare_seed_splits.py` builds the per-language variant
  ranking on the replicate seeds (64/313 of the ×3 cells) and on seed 1904 of
  the same cells and writes `<train>__vs__<test>/headline_metrics.csv`;
  agreement is counted over the languages that have a best variant on both
  splits. rq04 reports the numbers next to the ranking they test.

<!-- BEGIN auto:effect-vs-noise (effect_vs_noise.py --pool predictivity_all) -->
## Intervention effect against noise

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_all`.

- **Seed noise vs detrended checkpoint noise** — median ratio 1.88 over 7870 (size, L, task) cells with seed replicates.
- **Depth effect vs seed noise** — median |Δ|/seed-std 1.37; 33% of 5509 cells above 2× (a distinct model for SNR, not a re-roll).

**Effect over noise** (median over (size, L, task) cells):

| population | effect / noise | median | n |
|---|---|---|---|
| benchmark | arch / seed | 1.31 | 4897 |
| benchmark | arch / ckpt | 2.24 | 20032 |
| benchmark | scheme / seed | 1.09 | 1797 |
| benchmark | scheme / ckpt | 1.91 | 34225 |
| benchmark | temperature / seed | 1.13 | 4456 |
| benchmark | temperature / ckpt | 2.07 | 8977 |
| benchmark | zh / seed | 1.24 | 216 |
| benchmark | zh / ckpt | 1.99 | 535 |
| benchmark | es / seed | 1.47 | 143 |
| benchmark | es / ckpt | 2.76 | 376 |
| bpb | arch / seed | 1.85 | 600 |
| bpb | arch / ckpt | 4.34 | 1900 |
| bpb | scheme / seed | 0.48 | 100 |
| bpb | scheme / ckpt | 5.53 | 1300 |
| bpb | temperature / seed | 4.07 | 200 |
| bpb | temperature / ckpt | 12.08 | 300 |
| bpb | zh / seed | 2.75 | 200 |
| bpb | zh / ckpt | 13.91 | 300 |
| bpb | es / seed | 2.98 | 200 |
| bpb | es / ckpt | 14.54 | 300 |
| bpb_macro | arch / seed | 3.02 | 6 |
| bpb_macro | arch / ckpt | 4.40 | 19 |
| bpb_macro | scheme / seed | 3.80 | 1 |
| bpb_macro | scheme / ckpt | 3.89 | 13 |
| bpb_macro | temperature / seed | 2.48 | 2 |
| bpb_macro | temperature / ckpt | 7.27 | 3 |
| bpb_macro | zh / seed | 8.77 | 2 |
| bpb_macro | zh / ckpt | 23.07 | 3 |
| bpb_macro | es / seed | 7.32 | 2 |
| bpb_macro | es / ckpt | 20.01 | 3 |
| loss | arch / seed | 2.02 | 6 |
| loss | arch / ckpt | 2.07 | 22 |
| loss | scheme / seed | 11.14 | 1 |
| loss | scheme / ckpt | 2.66 | 13 |
| loss | temperature / seed | 4.99 | 2 |
| loss | temperature / ckpt | 3.11 | 5 |
| loss | zh / seed | 4.17 | 3 |
| loss | zh / ckpt | 11.03 | 4 |
| loss | es / seed | 2.67 | 2 |
| loss | es / ckpt | 4.39 | 3 |

![Effect vs noise](pretraining/predictivity_all/effect_vs_noise.png)
<!-- END auto:effect-vs-noise -->

## Preliminary findings (ladder snapshot, 2026-09-01)

Noise on the ≤ 600M ladder before any seed replicate existed
(`plan/status-09-01.md`, §5; checkpoint noise = std over the last 3 evaluated
checkpoints, median 0.004): the across-L range of final benchmark scores at
600M has median 0.022 ≈ 4.4× that noise (56/60 benchmarks above 2×; top:
`xwinograd_en` 17×, `hellaswag_ru` 13×), and on BPB the separation is one to
two orders of magnitude above noise — the language-count axis gives genuinely
distinct models for SNR. The transformation table on the 14 September report
(18 seed pairs) puts the seed effect at Δ final loss −0.061 … +0.046,
Δ macro BPB −0.050 … +0.034 and Δ mean benchmark −0.004 … +0.007; the depth
effect on benchmarks is of the same order (−0.009 … +0.014 over the tasks each
pair shares), so
whether deep vs shallow is a distinct model for SNR is exactly what the
effect-vs-noise table above decides per task.

## TODO

- [ ] **The seed holdout ranks most tasks on a single model pair (open, 2026-09-17).**
      *What it is.* rq04 claims one SNR definition tracks decision accuracy
      best; the holdout (`rq03_noise_and_snr/compare_seed_splits.py`, reported
      in rq04's highlight and "Seed generalization" table) asks whether that
      ranking of definitions survives a change of seed: it is computed on
      `predictivity_seeds_train` (seeds 64, 313) and again on
      `predictivity_seeds_test` (seed 1904 of the same cells) and the two are
      compared.
      *The problem.* Replicate seeds exist only at 175M and 600M, so both
      splits hold those two sizes and DA-size there is 175M → 600M (the 1.7B
      reference never enters). English tasks and BPB rest on 15 model pairs.
      A non-English benchmark is trained only in the L50 cell, so its train
      split holds one pair — the two seeds of the same cell — and its
      "decision accuracy" is 0 or 1 and measures seed noise, not a ranking.
      The median task has one pair.
      *Implications.* The per-language agreement numbers (29 % / 57 %) and the
      low DA-ckpt ρ (0.07) are dominated by those one-pair tasks, so "the
      ranking does not survive a seed swap" may say more about the holdout
      than about the definitions; the DA-size ρ (0.75) leans on English + BPB.
      Nothing in the main `predictivity` tables is affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate — changes what the holdout means. (C) report the
      holdout for DA-ckpt only, whose pairs are checkpoints × variants within
      a size and do not shrink to one. (D) drop the holdout numbers from the
      highlight until more sizes have replicate seeds.
      *Recommendation.* A now (small change, honest scope, say "English and
      BPB" in the highlight), and revisit once the 90M / 1.7B-L15 cells and
      any further replicate seeds exist. Not implemented yet.

## Files

- `pretraining/<pool>/snr_variants_per_task.csv` — per-task SNR (every
  variant × size bucket) + the DA columns joined from rq02. Single source of
  truth for rq04, rq07 and rq09 (`run_apertus_snr_variants.py`).
- `…/snr_variants_definitions.csv`, `snr_variant_coverage.csv` — the 22
  definitions and how many cells each covers per bucket.
- `…/effect_vs_noise.csv`, `effect_vs_noise.png` — per (size, L, task): |Δ|
  per intervention, seed noise, raw and detrended checkpoint noise, and their
  ratios (`effect_vs_noise.py`).
- `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/` — the
  seed-holdout report (`compare_seed_splits.py`: `headline_metrics.csv`, the
  per-language agreement and the variant r train vs test).

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Regenerate with `python analysis/rq03_noise_and_snr/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq03 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark](pretraining/predictivity/snr_by_benchmark.png)

![SNR per language](pretraining/predictivity/snr_by_language.png)

![Depth effect over seed noise per benchmark](pretraining/predictivity_all/effect_over_seed_by_benchmark.png)

![Depth effect over seed noise per language](pretraining/predictivity_all/effect_over_seed_by_language.png)
<!-- END auto:panels -->
