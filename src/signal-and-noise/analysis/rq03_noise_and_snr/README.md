# Noise and SNR: how noisy is a measurement, and what is its signal-to-noise ratio?

## Research question

> How much does a measurement move when nothing but the seed or the
> checkpoint changes, how does that noise compare with the effect of a design
> decision, and what is each benchmark's signal-to-noise ratio under the 22
> candidate definitions? The per-task SNR table computed here is what the
> surrogates analysis ranks against decision accuracy and what the external
> frameworks and benchmark design analyses read.

## Experimental setup

- **Pools** (`pretraining/<pool>/`, [RULES.md](../RULES.md), Definitions). `predictivity` is the headline: seed 1904, every cell, every data build and every ladder (deep, shallow, swiglu), sizes 90M–1.7B.
- **Seed pools.** `predictivity_seeds` adds every replicate seed as a separate model of the signal pool. The holdout trains on the replicate seeds (`predictivity_seeds_train`: 64/313 at 175M/600M, 28/1797 at 1B) and tests on seed 1904 of the same 10 cells (`predictivity_seeds_test`).
- **Seed cells.** The ×3 cells are the deep data-A runs at 175M and 600M for L ∈ {1, 2, 50} and at 1B for L ∈ {1, 2, 30}, plus the 1B L30 data-B run. A non-English harness task is trained only at L30/L50 (Russian also at L2), so the seed analyses are English-heavy by construction.
- **Signal and noise.** The signal population at a size is every design variant trained there (language setting × ladder × data build), with one data-scheme axis: scheme A/B/C × temperature T (AT3 = A at T 3; ZH, DCLMP = B; ES, FWEB = C). The noise is the checkpoint noise over the noise window of rule 4 (`utils.noise_checkpoints`: 80/85/90/95/100 % of the run, the same for BPB and benchmarks): since 2026-10-08 the residual std around a line through the window (`utils.checkpoint_noise`), with the raw std kept beside it as `ckpt_noise_raw_<size>`; `effect_vs_noise.py` adds the seed-replicate noise.
- **Decision accuracy and variants.** DA-size ranks small → 1.7B (plus every other bucket pair) and DA-ckpt ranks 10 … 90 % → final within a size. The 22 SNR variants fall into five families (dispersion, relative spread, discrepancy, robust, depth), and the per-language BPB tasks (`bpb_<subset>`) take part like any benchmark except under the discrepancy family.
- **bBPB twins.** The bBPB twins (`bbpb_*`) are scored at every checkpoint of every cell, seed replicates included, so they have a checkpoint noise and an SNR at every size and also enter the seed-noise rows and the seed holdout.

## Highlighted result

![Noise and SNR in one figure](pretraining/predictivity/highlights.png)

Pool `predictivity`, SNR variant `rel_std` (relative std of the final scores across the size's design variants over the relative checkpoint noise), gate applied (tasks at chance at a size are grey), sizes 90M–1.7B. "Benchmarks" below are the harness tasks without BPB and without the bBPB twins; the bBPB twins (lower is better, never gated) have their own bullet, and the figure's two rank panels list them too.

**Key findings**

- **SNR peaks at 350M and is lowest at the 1.7B reference.** The median log10 SNR of the ungated benchmark tasks is 0.30 at 90M (316 tasks), 0.28 at 175M (345), 0.41 at 350M (383), 0.33 at 600M (429), 0.27 at 1B (472) and 0.24 at 1.7B (512), so at 1.7B the typical benchmark separates the design variants by only 1.7× its checkpoint noise.
- **Few benchmark tasks are below the noise at 1.7B.** 38 of the 512 ungated benchmark tasks (7 %) have SNR < 1 at 1.7B, against 8 of 383 (2 %) at 350M; the per-language BPB falls from a median of 1.34 at 350M to 0.81 at 1.7B, with none of the 50 languages below 0.
- **The bBPB twins follow the same curve, one notch higher.** Over the 816 twin tasks the median log10 SNR is 0.58 at 90M, 0.58 at 175M, 0.75 at 350M, 0.50 at 600M, 0.44 at 1B and 0.36 at 1.7B, with 34 of 816 (4 %) below the noise at 1.7B.
- **The task populations differ across sizes** (rule 13): the gate keeps 316 to 512 of the 845–846 benchmark tasks, so a column's median is over a different task set at each size.

**Follow-ups**

- The number of design variants in each size's signal pool beside the median row, to tell whether the 350M peak is a property of the benchmarks or of that rung's set of variants.
- Each bBPB twin against its original on the tasks the gate keeps, to tell whether the higher twin SNR is the continuous score or the ungated population.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.csv). The SNR above uses checkpoint noise; [the next figure](#intervention-effect-against-noise) checks it against the seed noise.

## Methodology

- **Signal and noise per (task, size bucket).**
  - Signal pool = every model at the bucket (`per_model_inputs`); each model's
    `data_score` is its final checkpoint and `step_noise` its checkpoint noise
    (`utils.checkpoint_noise`: the residual SD around a line, n−2, by default;
    the raw std, ddof 0, kept as `ckpt_noise_raw_<size>`) over the noise window, the k/20 points in the last `noise_window` = 20 % of its
    run (80/85/90/95/100 %, rule 4). Before 2026-09-20 the window was the last
    five checkpoints of whatever grid the task had (80–100 % for BPB, 60–100 %
    for benchmarks), which put BPB's noise on a narrower window.
  - The 22 aggregators in `snr/snr_variants.py` differ in how they turn the
    cross-model scores into a signal (`rel_std` reads the cross-model std of
    the finals, CLAUDE.md bug #4). They share
    `noise = mean(step_noise) / mean(window means)`.
  - Cells the above-chance gate marks at chance are NaN.
  - The discrepancy family (`discrepancy`, `star_discrepancy`,
    `star_discrepancy_shifted`, `rel_star_discrepancy`) reads the scores as
    points of [0, 1] and returns finite but meaningless values on BPB and the
    loss, so the driver sets it to NaN on every task that is not a benchmark.
    `snr_variant_coverage.csv` records how many cells each variant covers per
    bucket, and the driver prints the ones that fall short.
  - Upstream's `signal_xlabel` / `noise_xlabel` strings are swapped; the
    definitions table writes them under the right name and no figure uses them.
- **Effect vs noise.**
  - For every (size, L, task) the intervention's |Δ| is put against the seed
    noise (sample std, n−1, over the seed replicates, where ≥ 2 seeds exist)
    and the checkpoint noise over the same 80/85/90/95/100 % window of the
    baseline cell (`utils.checkpoint_noise`: detrended, the residual std with
    n−2, the rule-4 default; and raw, ddof 0). Under WSD the final window is
    still descending, so the raw std carries trend.
  - A ratio near 1 means the two levels are the same model as far as a ranking
    is concerned (the "read this against the seed row" rule of
    `ladder_report.md`). A decision on such a cell is a coin flip whatever its
    DA (design decisions analysis).
  - The interventions are depth (deep vs shallow at scheme A, T 1), data scheme
    A vs B and A vs C (deep, T 1) and temperature T 1 vs 3 (deep, scheme A).
  - Cells at chance at their size keep their row with no number (rule 1) and
    are grey in the panels.
- **Seed holdout.**
  - `compare_seed_splits.py` builds the per-language variant ranking (the
    surrogates analysis' Pearson table: at least `min_lang_tasks` = 3 distinct
    tasks per language, rule 8; `multi` is never a language, rule 7) on the
    replicate seeds (64/313/28/1797 of the ×3 cells) and on seed 1904 of the
    same cells, and writes `<train>__vs__<test>/headline_metrics.csv`.
  - Agreement is counted over the languages that have a best variant on both
    splits.
  - Its README block below says whether the two pools hold the same cells; the
    surrogates analysis reports the numbers next to the ranking they test.

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-08 12:06** (regenerated locally: acc_norm on the cloze-format originals)
and the outputs of the 2026-10-08 refresh (detrended checkpoint noise).

<!-- BEGIN auto:effect-vs-noise (effect_vs_noise.py --pool predictivity_seeds) -->
## Intervention effect against noise

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_seeds`.

- **Noise definitions.** Seed noise = sample std (n−1) of the final score across the replicate seeds of the deep data-A cell; checkpoint noise = std of the grid seed's run over the noise window, the k/20 points in the last 20% of the run (80/85/90/95/100 %, the same for BPB and benchmarks), `utils.checkpoint_noise`: detrended by a line (residual SD, n−2, the rule-4 default) and raw (ddof 0, the labelled alternative). The seed-over-checkpoint ratio compares run-to-run scatter with the within-run scatter of one run: above 1 a re-roll of the seed moves the score more than the late checkpoints do.
- **Gate.** 9027 of 33438 (size, L, task) cells are at chance at their size (rule 1); they keep their row, carry no number and enter no median below.
- **Seed noise vs detrended checkpoint noise** — median ratio 1.75 over 4394 (size, L, task) cells with seed replicates.
- **Depth effect vs seed noise** — median |Δ|/seed-std 1.31; 31% of 4389 cells above 2× (a distinct model for SNR, not a re-roll).

**Effect over noise** (median over the ungated (size, L, task) cells; `n` = cells behind the median):

| population | effect / noise | median | n |
|---|---|---|---|
| benchmark | arch / seed | 1.26 | 4232 |
| benchmark | arch / ckpt | 2.20 | 20796 |
| benchmark | scheme_B / seed | 1.21 | 1679 |
| benchmark | scheme_B / ckpt | 2.17 | 9644 |
| benchmark | scheme_C / seed | 1.63 | 775 |
| benchmark | scheme_C / ckpt | 3.56 | 1436 |
| benchmark | temperature / seed | 1.79 | 3366 |
| benchmark | temperature / ckpt | 3.04 | 16660 |
| bpb | arch / seed | 1.77 | 139 |
| bpb | arch / ckpt | 1.95 | 636 |
| bpb | scheme_B / seed | 0.27 | 32 |
| bpb | scheme_B / ckpt | 1.31 | 234 |
| bpb | scheme_C / seed | 160.69 | 6 |
| bpb | scheme_C / ckpt | 43.92 | 12 |
| bpb | temperature / seed | 12.33 | 130 |
| bpb | temperature / ckpt | 11.77 | 570 |
| bpb_macro | arch / seed | 1.84 | 9 |
| bpb_macro | arch / ckpt | 1.04 | 36 |
| bpb_macro | scheme_B / seed | 3.84 | 7 |
| bpb_macro | scheme_B / ckpt | 2.36 | 30 |
| bpb_macro | scheme_C / seed | 31.32 | 6 |
| bpb_macro | scheme_C / ckpt | 10.91 | 12 |
| bpb_macro | temperature / seed | 3.05 | 3 |
| bpb_macro | temperature / ckpt | 1.21 | 18 |
| loss | arch / seed | 1.16 | 9 |
| loss | arch / ckpt | 1.44 | 36 |
| loss | scheme_B / seed | 7.54 | 7 |
| loss | scheme_B / ckpt | 3.68 | 30 |
| loss | scheme_C / seed | 2.92 | 6 |
| loss | scheme_C / ckpt | 2.86 | 12 |
| loss | temperature / seed | 7.20 | 3 |
| loss | temperature / ckpt | 1.41 | 18 |

![Effect vs noise](pretraining/predictivity_seeds/effect_vs_noise.png)
<!-- END auto:effect-vs-noise -->

GitHub: [effect_vs_noise.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_vs_noise.png) · [effect_vs_noise.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_vs_noise.csv)

The highlighted SNR divides by checkpoint noise; this figure asks whether that is the right noise by setting both noises, and each design decision's effect, against the seed replicates. Population: pool `predictivity_seeds`, the nine deep data-A seed cells (175M and 600M at L 1/2/50, 1B at L 1/2/30, three seeds each), gate applied (9027 of 33438 (size, L, task) cells at chance; the 15882 bBPB twin cells are never gated and, now that the per-item store holds the seed replicates, enter the seed-noise rows too: 2883 of the 4394 seed-over-checkpoint cells).

**Key findings**

- **Seed noise exceeds checkpoint noise.** The seed std is a median 1.75× the detrended checkpoint std over 4394 ungated cells and larger in 75 % of them (benchmark originals 1.72 over 1354 cells, bBPB twins 1.82 over 2883, per-language BPB 1.33 over 139); per size 2.08 at 175M, 1.72 at 600M and 1.52 at 1B.
- **So the checkpoint-noise SNR is optimistic.** A design difference that clears the checkpoint noise need not clear a re-roll of the seed: on the same 4389 seed cells, depth's median effect is 2.13× the detrended checkpoint noise but 1.31× the seed noise (52 % against 31 % above 2×).
- **No design axis clears twice the seed noise in half of its cells.** Against seed noise the median effect is 1.31 for depth, 1.20 for scheme A vs B (1725 cells, 29 % above 2×), 1.65 for scheme A vs C (793, 43 %) and 1.86 for temperature T 1 vs 3 (3502, 47 %); the per-population medians are in the table above.

**Follow-ups**

- An SNR variant with the seed std as its noise, ranked against decision accuracy next to the checkpoint-noise variants, because the seed is the noise a design decision has to beat.
- The ratio table split by size, to see whether the seed-over-checkpoint gap closes towards 1B as the runs get longer.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Regenerate with `python analysis/rq03_noise_and_snr/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate" (at chance at that size, rule 1); each figure's table sits next to it under the same name; sizes are 175M–1.7B (rule 10). SNR noise is the residual std around a line through the 80/85/90/95/100 % checkpoints (rule 4).

![rq03 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark](pretraining/predictivity/snr_by_benchmark.png)

![SNR per language](pretraining/predictivity/snr_by_language.png)

![Depth effect over seed noise per benchmark](pretraining/predictivity_seeds/effect_over_seed_by_benchmark.png)

![Depth effect over seed noise per language](pretraining/predictivity_seeds/effect_over_seed_by_language.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.csv) ·
[snr_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/snr_by_benchmark.png) ·
[snr_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/snr_by_language.png) ·
[snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/snr.csv) ·
[effect_over_seed_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_over_seed_by_benchmark.png) ·
[effect_over_seed_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_over_seed_by_language.png) ·
[snr_variants_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/snr_variants_per_task.csv)

The highlight gave medians; these grids show which benchmarks and languages carry the signal. Population: SNR grids on pool `predictivity` (`rel_std`, sizes 90M–1.7B; the generated line's "175M–1.7B" is stale generator text), depth-over-seed grids on `predictivity_seeds` (4321 ungated cells at 175M, 600M and 1B), gate applied.

**Key findings**

- **HellaSwag has the highest benchmark SNR at 1.7B.** Over 29 benchmark families with an ungated task (50 languages, 512 tasks) the highest median log10 SNR is `hellaswag` 0.62, `global_mmlu_full` 0.53 and `xstorycloze` 0.49; the lowest are `belebele` 0.11, `rf_belebele` 0.15, `rf_include_base_44` 0.16 and `global_piqa_parallel_cloze` 0.17.
- **Reformulation does not reliably raise SNR at 1.7B here.** Two of the three `rf` twins with an ungated original are lower (`rf_global_mmlu_full` 0.39 against `global_mmlu_full` 0.53, `rf_include_base_44` 0.16 against 0.28) and `rf_belebele` is higher (0.15 against 0.11), and of the `rfgm` twins `rfgm_include_base_44` is lower (0.24 against 0.28) but `rfgm_belebele` is higher (0.20 against 0.11). The gate keeps a different task set for each twin, so this is a population-level reading.
- **Every language but Lithuanian is above the noise at 1.7B.** On the benchmarks the median log10 SNR runs from −0.03 (Lithuanian, 8 tasks) to 0.51 (Slovenian, 3 tasks), 1 of 50 languages sits below 0 and English is 0.29 over 34 tasks; a language rests on 1 to 52 tasks, so the extremes are thin.
- **Depth beats the seed noise in most families.** The median log10 ratio is above 0 in 40 of the 51 families with at least 10 cells (the bBPB twins included), from −0.42 (`bbpb_blend_sample`, 21 cells) to 0.39 (`bbpb_belebele`, 165 cells); the 21 original families run from −0.01 (`rfgm_include_base_44`) to 0.25 (`paws`, 14 cells, and per-language BPB, 139 cells), and the 49 languages with at least 10 cells from −0.07 (Kazakh, 38 cells) to 0.31 (Slovak, 30 cells; Bosnian, the one language under 10 cells, 0.16 over 5), with English at 0.07 over 1154 cells.

**Follow-ups**

- The per-language SNR grid restricted to languages with at least 10 ungated tasks, because the extremes of the ranking rest on 1–7 tasks.
- Original vs reformulated twins on the tasks both keep after the gate, to separate the format effect from the population effect.

The SNR ranking and the 22 definitions feed the [surrogates analysis](../rq04_surrogates/README.md); whether that ranking of definitions survives a change of seed is the holdout below.

<!-- BEGIN auto:seed-holdout (compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test) -->
## Seed holdout

Regenerate with `python analysis/rq03_noise_and_snr/compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test`; the tables are under `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/`.

`predictivity_seeds_train` (seeds 28, 64, 313, 1797) and `predictivity_seeds_test` (seeds 1904) hold the same 10 cells: 175M L1 deep data A, 1B L1 deep data A, 600M L1 deep data A, 175M L2 deep data A, 1B L2 deep data A, 600M L2 deep data A, 1B L30 deep data A, 1B L30 deep data B, 175M L50 deep data A, 600M L50 deep data A.

A language's r is rq04's Pearson r over its tasks' (log10 SNR, DA) points and needs at least 3 distinct tasks with a value (rule 8); `multi` and `??` are never a language (rule 7). DA-size on the holdout is 175M → 600M (scaling pair) (the pools stop at 600M, so the 1.7B reference never enters; rule 9); DA-ckpt is the within-size early → final ranking. Agreement is counted over the languages with a best variant on both splits.

| DA | languages | same variant | same family | Spearman ρ of the variant ranking |
|---|---|---|---|---|
| DA-size | 1 | 0 | 1 | 0.55 |
| DA-ckpt | 2 | 0 | 2 | 0.70 |
<!-- END auto:seed-holdout -->

[headline_metrics.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv) ·
[summary.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/summary.md)

Population: train = the replicate seeds 28/64/313/1797, test = seed 1904, the same 10 cells, pairs within each split, gate of `predictivity`. DA-size exists only for 175M → 600M (208 of the 1714 train-split tasks: 206 English tasks, 98 of them bBPB twins, and 2 aggregates), so the 1B seeds do not enter it and neither does the 1.7B reference.

**Key findings**

- **DA-ckpt: the family survives a seed swap.** Two languages (English, Russian) have a best variant on both splits; the train pick is the same family in 2 of 2 (discrepancy for English, relative spread for Russian) but the same variant in 0 of 2, the global variant ranking agrees at Spearman ρ 0.70, the variant cells at Pearson r 0.34 (n = 44) and the train pick keeps 97 % of the test-best r.
- **DA-size: now agrees, but on one size pair and one language.** Only English has a best variant on both splits (`dispersion_shifted` on train, `star_discrepancy_shifted` on test, both discrepancy, so 1 of 1 at the family level), the global ranking agrees at ρ 0.55 and the cells at r 0.72 (n = 22), and the train pick keeps 85 % of the test-best r; on the 10-06 snapshot the test pick was `rms_deviation` and ρ was −0.28, so the sign rests on a single language.

**Follow-ups**

- The DA-size holdout restricted to tasks with at least six pairs on both splits (option A of the TODO below), stated as English and BPB only.


## TODO

- [ ] **The seed holdout ranks most tasks on a single model pair (open, 2026-09-17).**
      *What it is.* The surrogates analysis claims one SNR definition tracks
      decision accuracy best; the holdout (`compare_seed_splits.py`, reported
      in the surrogates highlight and "Seed generalization" table) asks
      whether that ranking of definitions survives a change of seed, by
      computing it on `predictivity_seeds_train` (seeds 64, 313, 28, 1797)
      and on `predictivity_seeds_test` (seed 1904 of the same 10 cells).
      *The problem (re-read 2026-10-08).* Since 2026-10-05 the 1B ×3 cells
      (seeds 28/1797) are in both splits, but DA-size still exists only for
      175M → 600M, so the 1.7B reference never enters. A non-English
      benchmark is trained only at L30/L50 (Russian also at L2), so its
      split holds no multi-axis pair and no DA-size: 208 of the 1714
      train-split tasks have a DA-size value, 206 English tasks (98 of them
      bBPB twins) and 2 aggregates.
      *Implications.* The DA-size agreement (1 of 1 language at the family
      level, ρ 0.55; 0 of 1 and ρ −0.28 on the 10-06 snapshot) rests on
      English alone, and DA-ckpt on two languages
      (English, Russian), so whether "the ranking survives a seed swap" says
      more about the holdout than about the definitions: one language flipped
      it between two snapshots.
      Nothing in the main `predictivity` tables is affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate — changes what the holdout means. (C) report the
      holdout for DA-ckpt only, whose pairs are checkpoints × variants within
      a size and do not shrink to one. (D) drop the holdout numbers from the
      highlight until more sizes have replicate seeds.
      *Recommendation.* A now (small change, honest scope, say "English and
      BPB" in the highlight), and revisit once replicate seeds exist at 1.7B
      or in non-English-heavy cells. Not implemented yet.

## Extensions from other sweeps

- From the **36-model sweep** (2026-04…06, 4 sizes × 3 data mixtures × 3
  seeds, pools `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`,
  `custom_swissai_hf`) and the public models (`all/external`),
  `pretraining/<pool>/snr_variants_per_task.csv` are the same 22 definitions
  on those pools, with the sweep's checkpoint window and its 1B reference.
  `pretraining/seeds_28_1797__vs__seeds_1904/` is the sweep's seed holdout
  (seeds 28/1797 → 1904).
- They are history, not regenerated, and never pooled with the ladder's
  table: a different harness, task set (the 86-task old list), noise window
  and reference size.
- The readings built on them — the variant ranking per tier, the holdout's
  Spearman ρ, the per-language anchors — are in
  [the surrogates analysis' extensions](../rq04_surrogates/README.md#extensions-from-other-sweeps).
  The dated comparison note that once accompanied them
  (`ANALYSIS_new_vs_previous.md`, removed 2026-09-23) is superseded by that section.

## Files

- `pretraining/<pool>/snr_variants_per_task.csv` — per-task SNR (every
  variant × size bucket) + the DA columns joined from the decision accuracy
  analysis. Single source of truth for the surrogates, external frameworks
  and benchmark design analyses (`run_apertus_snr_variants.py`).
- `…/snr_variants_definitions.csv`, `snr_variant_coverage.csv` — the 22
  definitions and how many cells each covers per bucket.
- `…/effect_vs_noise.csv`, `effect_vs_noise.png` — per (size, L, task): |Δ|
  per intervention, seed noise, raw and detrended checkpoint noise, and their
  ratios (`effect_vs_noise.py`).
- `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/` — the
  seed-holdout report (`compare_seed_splits.py`: `headline_metrics.csv`, the
  per-language agreement and the variant r train vs test).
