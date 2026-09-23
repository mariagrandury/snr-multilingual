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
checkpoint std over the one noise window of rule 4 (`utils.noise_checkpoints`:
the k/20 points in the last 20 % of the run, 80/85/90/95/100 %, the same rows for
BPB and benchmarks; `effect_vs_noise.py` below carries the seed-replicate
noise). DA has two flavours: **DA-size** (small→1.7B ranking, plus every
other bucket pair) and **DA-ckpt** (10 … 90 % → final within a size). The 22
variants are grouped into families (dispersion / relative-spread /
discrepancy / robust / depth); rq04 correlates them with DA per language. The
per-language BPB tasks (`bpb_<subset>`) take part like any benchmark, except
under the discrepancy family (below).
The seed holdout is English-heavy by construction: the ×3 cells are the deep
scheme-A runs at 175M and 600M for L ∈ {1, 2, 50}, and a non-English harness
task is only trained at L50 (Russian also at L2), so its per-language variant
ranking rests on one language setting per seed, while English and the BPB
tasks cover all three.

## Methodology

- **Signal and noise per (task, size bucket).** Signal pool = every model at
  the bucket (`per_model_inputs`); each model's `data_score` is its final
  checkpoint, `step_noise` the std (ddof 0) over the noise window, the
  k/20 points in the last `noise_window` = 20 % of its run (80/85/90/95/100 %,
  rule 4 — before 2026-09-20 it was the last five checkpoints of whatever grid
  the task had, 80–100 % for BPB and 60–100 % for benchmarks, which put BPB's
  noise on a narrower window). The 22 aggregators in `snr/snr_variants.py`
  differ in how they turn the cross-model scores into a dispersion; all but
  `rel_std` share `noise = mean(step_noise) / mean(window means)`. Cells the
  rq00 gate marks at chance are NaN. The discrepancy family (`discrepancy`,
  `star_discrepancy`, `star_discrepancy_shifted`, `rel_star_discrepancy`)
  reads the scores as points of [0, 1] and returns finite but meaningless
  values on BPB and the loss, so the driver sets it to NaN on every task that
  is not a benchmark; `snr_variant_coverage.csv` records how many cells each
  variant covers per bucket, and the driver prints the ones that fall short.
  (Upstream's `signal_xlabel` / `noise_xlabel` strings are swapped; the
  definitions table writes them under the right name and no figure uses them.)
- **Effect vs noise.** For every (size, L, task) the intervention's |Δ| is put
  against the seed noise (sample std, n−1, over the seed replicates, where
  ≥ 2 seeds exist) and the checkpoint noise (std over the same 80/85/90/95/100 %
  window of the baseline cell, raw with n−1 and detrended with n−2 — under
  WSD the final window is still descending, so the raw std carries trend).
  Every std divides by its residual degrees of freedom. A ratio near 1 means
  the two levels are the same model as far as a ranking is concerned (the
  "read this against the seed row" rule of `ladder_report.md`); a decision on
  such a cell is a coin flip whatever its DA (rq05). Cells at chance at their
  size keep their row with no number (rule 1) and are grey in the panels.
- **Seed holdout.** `compare_seed_splits.py` builds the per-language variant
  ranking (rq04's Pearson table: at least `min_lang_tasks` = 3 distinct tasks
  per language, rule 8; `multi` is never a language, rule 7) on the replicate
  seeds (64/313 of the ×3 cells) and on seed 1904 of the same cells and writes
  `<train>__vs__<test>/headline_metrics.csv`; agreement is counted over the
  languages that have a best variant on both splits. Its README block below
  says whether the two pools hold the same cells. rq04 reports the numbers
  next to the ranking they test.

<!-- BEGIN auto:effect-vs-noise (effect_vs_noise.py --pool predictivity_all) -->
## Intervention effect against noise

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_all`.

- **Noise definitions.** Seed noise = sample std (n−1) of the final score across the replicate seeds of the deep scheme-A cell; checkpoint noise = std of the grid seed's run over the noise window, the k/20 points in the last 20% of the run (80/85/90/95/100 %, the same for BPB and benchmarks), raw (n−1) and detrended by a line (n−2). Every std divides by its residual degrees of freedom. The seed-over-checkpoint ratio compares run-to-run scatter with the within-run scatter of one run: above 1 a re-roll of the seed moves the score more than the late checkpoints do.
- **Gate.** 3642 of 7533 (size, L, task) cells are at chance at their size (rule 1); they keep their row, carry no number and enter no median below.
- **Seed noise vs detrended checkpoint noise** — median ratio 1.79 over 794 (size, L, task) cells with seed replicates.
- **Depth effect vs seed noise** — median |Δ|/seed-std 1.46; 35% of 793 cells above 2× (a distinct model for SNR, not a re-roll).

**Effect over noise** (median over the ungated (size, L, task) cells; `n` = cells behind the median):

| population | effect / noise | median | n |
|---|---|---|---|
| benchmark | arch / seed | 1.27 | 636 |
| benchmark | arch / ckpt | 2.14 | 2883 |
| benchmark | scheme / seed | 1.08 | 167 |
| benchmark | scheme / ckpt | 2.00 | 1131 |
| benchmark | temperature / seed | 1.40 | 559 |
| benchmark | temperature / ckpt | 2.39 | 2433 |
| benchmark | zh / seed | 1.52 | 28 |
| benchmark | zh / ckpt | 2.91 | 38 |
| benchmark | es / seed | 1.77 | 28 |
| benchmark | es / ckpt | 3.78 | 38 |
| bpb | arch / seed | 2.13 | 139 |
| bpb | arch / ckpt | 1.92 | 530 |
| bpb | scheme / seed | 0.21 | 26 |
| bpb | scheme / ckpt | 2.07 | 185 |
| bpb | temperature / seed | 8.72 | 100 |
| bpb | temperature / ckpt | 14.94 | 400 |
| bpb | zh / seed | 1.37 | 3 |
| bpb | zh / ckpt | 4.75 | 4 |
| bpb | es / seed | 4.44 | 3 |
| bpb | es / ckpt | 7.63 | 4 |
| bpb_macro | arch / seed | 1.91 | 9 |
| bpb_macro | arch / ckpt | 2.64 | 30 |
| bpb_macro | scheme / seed | 3.80 | 1 |
| bpb_macro | scheme / ckpt | 3.02 | 15 |
| bpb_macro | temperature / seed | 2.48 | 2 |
| bpb_macro | temperature / ckpt | 3.80 | 12 |
| bpb_macro | zh / seed | 9.48 | 3 |
| bpb_macro | zh / ckpt | 16.01 | 4 |
| bpb_macro | es / seed | 8.22 | 3 |
| bpb_macro | es / ckpt | 12.81 | 4 |
| loss | arch / seed | 1.33 | 9 |
| loss | arch / ckpt | 1.66 | 30 |
| loss | scheme / seed | 11.14 | 1 |
| loss | scheme / ckpt | 2.70 | 15 |
| loss | temperature / seed | 7.20 | 3 |
| loss | temperature / ckpt | 1.61 | 15 |
| loss | zh / seed | 4.17 | 3 |
| loss | zh / ckpt | 8.54 | 4 |
| loss | es / seed | 2.42 | 3 |
| loss | es / ckpt | 3.91 | 4 |

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

Regenerate with `python analysis/rq03_noise_and_snr/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate" (at chance at that size, rule 1); each figure's table sits next to it under the same name; sizes are 175M–1.7B (rule 10). SNR noise is the std over the 80/85/90/95/100 % checkpoints (rule 4).

![rq03 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark](pretraining/predictivity/snr_by_benchmark.png)

![SNR per language](pretraining/predictivity/snr_by_language.png)

![Depth effect over seed noise per benchmark](pretraining/predictivity_all/effect_over_seed_by_benchmark.png)

![Depth effect over seed noise per language](pretraining/predictivity_all/effect_over_seed_by_language.png)
<!-- END auto:panels -->

<!-- BEGIN auto:seed-holdout (compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test) -->
## Seed holdout

Regenerate with `python analysis/rq03_noise_and_snr/compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test`; the tables are under `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/`.

`predictivity_seeds_train` (seeds 64, 313) and `predictivity_seeds_test` (seeds 1904) hold the same 6 cells: 175M L1 deep scheme A, 600M L1 deep scheme A, 175M L2 deep scheme A, 600M L2 deep scheme A, 175M L50 deep scheme A, 600M L50 deep scheme A.

A language's r is rq04's Pearson r over its tasks' (log10 SNR, DA) points and needs at least 3 distinct tasks with a value (rule 8); `multi` and `??` are never a language (rule 7). DA-size on the holdout is 175M → 600M (scaling pair) (the pools stop at 600M, so the 1.7B reference never enters; rule 9); DA-ckpt is the within-size early → final ranking. Agreement is counted over the languages with a best variant on both splits.

| DA | languages | same variant | same family | Spearman ρ of the variant ranking |
|---|---|---|---|---|
| DA-size | 1 | 0 | 0 | -0.71 |
| DA-ckpt | 1 | 0 | 1 | 0.83 |
<!-- END auto:seed-holdout -->
