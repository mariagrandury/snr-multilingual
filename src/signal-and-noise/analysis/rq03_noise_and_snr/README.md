# Noise and SNR: how noisy is a measurement, and what is its signal-to-noise ratio?

## Research question

> How much does a measurement move when nothing but the seed or the
> checkpoint changes, how does that noise compare with the effect of a design
> decision, and what is each benchmark's signal-to-noise ratio under the 22
> candidate definitions? The surrogates analysis ranks the per-task SNR table
> computed here against decision accuracy, and the external-frameworks and
> benchmark-design analyses read it too.

## Experimental setup

**Setup.**
- **Measured.** We measure the effect of a design decision on a task's final score against two kinds of noise, per (size, L, task) cell.
- **Decisions.** The decisions are depth (deep vs shallow), data scheme A vs B, data scheme A vs C, sampling temperature T = 1 vs T = 3, activation (xIELU vs SwiGLU) and optimizer (AdEMAMix vs Muon), each against the baseline, between the seed-1904 runs of every cell, 90M–1.7B.
- **Seed noise.** The seed noise is the sample standard deviation of the final score across the three seeds of the replicated baseline cells (deep, scheme A: 175M and 600M at L ∈ {1, 2, 50}, 1B at L ∈ {1, 2, 30}).
- **Checkpoint noise.** The checkpoint noise is the standard deviation of the scores around a straight line fitted to the last 20 % of a run (the checkpoints at 80, 85, 90, 95 and 100 %).
- **How.** We divide the absolute difference between the two levels of a decision by each noise and take the median over the cells. We do this separately for the benchmarks (with their rf and bBPB twins), the per-language BPB, its macro average and the training loss, and we leave out cells at chance.
- **Flips.** For the per-language BPB we count, per design axis, the (pair, language) comparisons whose 600M difference has the opposite sign of their 1.7B difference, and whether that 600M difference exceeds one seed standard deviation (table: app_rq03_flip_attribution).

- **Pools** (`pretraining/<pool>/`, [RULES.md](../RULES.md), Definitions). `predictivity` is the headline: seed 1904, every cell, every data build and every ladder (deep, shallow, swiglu), sizes 90M–1.7B.
- **Seed pools.** `predictivity_seeds` adds every replicate seed as a separate model of the signal pool. The holdout trains on the replicate seeds (`predictivity_seeds_train`: 64/313 at 175M/600M, 28/1797 at 1B) and tests on seed 1904 of the same 10 cells (`predictivity_seeds_test`).
- **Seed cells.** The ×3 cells are the deep data-A runs at 175M and 600M for L ∈ {1, 2, 50} and at 1B for L ∈ {1, 2, 30}, plus the 1B L30 data-B run. A non-English harness task is trained only at L30/L50 (Russian also at L2), so the seed analyses are English-heavy by construction.
- **Signal and noise.** The signal population at a size is every design variant trained there (language setting × ladder × data build), with one data-scheme axis: scheme A/B/C × temperature T (AT3 is A at T 3, ZH and DCLMP are B, ES and FWEB are C). The noise is the checkpoint noise over the noise window of rule 4 (`utils.noise_checkpoints`: 80/85/90/95/100 % of the run, the same for BPB and benchmarks). Since 2026-10-08 it is the residual std around a line through the window (`utils.checkpoint_noise`), with the raw std kept beside it as `ckpt_noise_raw_<size>`. `effect_vs_noise.py` adds the seed-replicate noise.
- **Decision accuracy and variants.** DA-size ranks small → 1.7B (plus every other bucket pair) and DA-ckpt ranks 10 … 90 % → final within a size. The 22 SNR variants fall into five families (dispersion, relative spread, discrepancy, robust, depth), and the per-language BPB tasks (`bpb_<subset>`) take part like any benchmark except under the discrepancy family.
- **bBPB twins.** The bBPB twins (`bbpb_*`) are scored at every checkpoint of every cell, seed replicates included, so they have a checkpoint noise and an SNR at every size and also enter the seed-noise rows and the seed holdout.

## Highlighted result

![Noise and SNR in one figure](pretraining/predictivity/highlights.png)

Pool `predictivity`, SNR variant `rel_std` (relative std of the final scores across the size's design variants over the relative checkpoint noise), gate applied (tasks at chance at a size are grey), sizes 90M–1.7B. "Benchmarks" below are the harness tasks without BPB and without the bBPB twins. The bBPB twins (lower is better, never gated) have their own bullet, and the figure's two rank panels list them too.

**Key findings**

- **SNR peaks at 350M and is lowest at the 1.7B reference.** The median log10 SNR of the ungated benchmark tasks is 0.29 at 90M (381 tasks), 0.27 at 175M (412), 0.41 at 350M (446), 0.31 at 600M (496), 0.27 at 1B (540) and 0.23 at 1.7B (588), so at 1.7B the typical benchmark separates the design variants by only 1.7× its checkpoint noise.
- **One benchmark task in eight is below the noise at 1.7B.** 75 of the 588 ungated benchmark tasks (13 %) have SNR < 1 at 1.7B, against 18 of 446 (4 %) at 350M. The per-language BPB falls from a median of 1.21 at 350M to 0.80 at 1.7B, with none of the 50 languages below 0.
- **The bBPB twins follow the same curve, one notch higher.** Over the 816 twin tasks the median log10 SNR is 0.58 at 90M, 0.58 at 175M, 0.75 at 350M, 0.50 at 600M, 0.44 at 1B and 0.36 at 1.7B, with 34 of 816 (4 %) below the noise at 1.7B.
- **The task populations differ across sizes** (rule 13). The gate keeps 381 to 588 of the 922–937 benchmark tasks, so a column's median is over a different task set at each size.

**Follow-ups**

- **Variants per size.** The number of design variants in each size's signal pool, shown beside the median row, would tell whether the 350M peak is a property of the benchmarks or of that rung's set of variants.
- **Twin against original.** Each bBPB twin against its original on the tasks the gate keeps would tell whether the higher twin SNR comes from the continuous score or from the ungated population.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/highlights.csv). The SNR above uses checkpoint noise. [The next figure](#intervention-effect-against-noise) checks it against the seed noise.

## Methodology

- **Signal and noise per (task, size bucket).**
  - Signal pool = every model at the bucket (`per_model_inputs`). Each model's
    `data_score` is its final checkpoint and `step_noise` its checkpoint noise
    (`utils.checkpoint_noise`: by default the residual SD around a line, n−2,
    with the raw std, ddof 0, kept as `ckpt_noise_raw_<size>`) over the noise window, the k/20 points in the last `noise_window` = 20 % of its
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
  - Upstream's `signal_xlabel` / `noise_xlabel` strings are swapped. The
    definitions table writes them under the right name, and no figure uses them.
- **Effect vs noise.**
  - For every (size, L, task) the intervention's |Δ| is put against the seed
    noise (sample std, n−1, over the seed replicates, where ≥ 2 seeds exist)
    and the checkpoint noise over the same 80/85/90/95/100 % window of the
    baseline cell (`utils.checkpoint_noise`, once detrended as the residual std
    with n−2, the rule-4 default, and once raw with ddof 0). Under WSD the final
    window is still descending, so the raw std carries the trend.
  - A ratio near 1 means the two levels are the same model as far as a ranking
    is concerned (the "read this against the seed row" rule of
    `ladder_report.md`). A decision on such a cell is random whatever its
    DA (design decisions analysis).
  - The interventions are depth (deep vs shallow at scheme A, T 1), data scheme
    A vs B and A vs C (deep, T 1), temperature T 1 vs 3 (deep, scheme A), and
    activation (xIELU vs SwiGLU) and optimizer (AdEMAMix vs Muon), both on the
    deep ladder at scheme A, T 1.
  - Cells at chance at their size keep their row with no number (rule 1) and
    are grey in the panels.
- **Seed holdout.**
  - `compare_seed_splits.py` builds the per-language variant ranking (the
    surrogates analysis' Pearson table: at least `min_lang_tasks` = 3 distinct
    tasks per language, rule 8, and `multi` is never a language, rule 7) on the
    replicate seeds (64/313/28/1797 of the ×3 cells) and on seed 1904 of the
    same cells, and writes `<train>__vs__<test>/headline_metrics.csv`.
  - Agreement is counted over the languages that have a best variant on both
    splits.
  - Its README block below says whether the two pools hold the same cells. The
    surrogates analysis reports the numbers next to the ranking they test.

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended
checkpoint noise), re-read on 2026-10-10.

<!-- BEGIN auto:effect-vs-noise (effect_vs_noise.py --pool predictivity_seeds) -->
## Intervention effect against noise

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_seeds`.

- **Noise definitions.** The seed noise is the sample std (n−1) of the final score across the replicate seeds of the deep data-A cell. The checkpoint noise is the std of the grid seed's run over the noise window, the k/20 points in the last 20% of the run (80/85/90/95/100 %, the same for BPB and benchmarks). `utils.checkpoint_noise` computes it detrended by a line (residual SD, n−2, the rule-4 default) and raw (ddof 0, the labelled alternative). The seed-over-checkpoint ratio compares the scatter between runs with the scatter within one run. Above 1, a re-roll of the seed moves the score more than the late checkpoints do.
- **Gate.** 9267 of 36420 (size, L, task) cells are at chance at their size (rule 1). They keep their row, carry no number and enter no median below.
- **Seed noise vs detrended checkpoint noise.** The median ratio is 1.75 over 4653 (size, L, task) cells with seed replicates.
- **Depth effect vs seed noise.** The median |Δ|/seed-std is 1.29, and 30% of 4648 cells are above 2×, where the two depths count as distinct models for SNR rather than a re-roll.

**Effect over noise.** Each value is the median over the ungated (size, L, task) cells, and `n` counts the cells behind the median:

| population | effect / noise | median | n |
|---|---|---|---|
| benchmark | arch / seed | 1.25 | 4491 |
| benchmark | arch / ckpt | 2.20 | 23461 |
| benchmark | scheme_B / seed | 1.24 | 1757 |
| benchmark | scheme_B / ckpt | 2.18 | 10629 |
| benchmark | scheme_C / seed | 1.64 | 811 |
| benchmark | scheme_C / ckpt | 3.65 | 1662 |
| benchmark | temperature / seed | 1.73 | 3586 |
| benchmark | temperature / ckpt | 3.03 | 18856 |
| benchmark | activation / seed | 1.15 | 1040 |
| benchmark | activation / ckpt | 2.18 | 13259 |
| benchmark | optimizer / seed | 1.15 | 1040 |
| benchmark | optimizer / ckpt | 2.31 | 13259 |
| bpb | arch / seed | 1.77 | 139 |
| bpb | arch / ckpt | 1.95 | 636 |
| bpb | scheme_B / seed | 0.27 | 32 |
| bpb | scheme_B / ckpt | 1.31 | 234 |
| bpb | scheme_C / seed | 160.69 | 6 |
| bpb | scheme_C / ckpt | 43.92 | 12 |
| bpb | temperature / seed | 12.33 | 130 |
| bpb | temperature / ckpt | 11.77 | 570 |
| bpb | activation / seed | 0.45 | 30 |
| bpb | activation / ckpt | 1.22 | 318 |
| bpb | optimizer / seed | 0.79 | 30 |
| bpb | optimizer / ckpt | 1.75 | 318 |
| bpb_macro | arch / seed | 1.84 | 9 |
| bpb_macro | arch / ckpt | 1.04 | 36 |
| bpb_macro | scheme_B / seed | 3.84 | 7 |
| bpb_macro | scheme_B / ckpt | 2.36 | 30 |
| bpb_macro | scheme_C / seed | 31.32 | 6 |
| bpb_macro | scheme_C / ckpt | 10.91 | 12 |
| bpb_macro | temperature / seed | 3.05 | 3 |
| bpb_macro | temperature / ckpt | 1.21 | 18 |
| bpb_macro | activation / seed | 8.06 | 1 |
| bpb_macro | activation / ckpt | 0.56 | 18 |
| bpb_macro | optimizer / seed | 2.61 | 1 |
| bpb_macro | optimizer / ckpt | 1.47 | 18 |
| loss | arch / seed | 1.16 | 9 |
| loss | arch / ckpt | 1.44 | 36 |
| loss | scheme_B / seed | 7.54 | 7 |
| loss | scheme_B / ckpt | 3.68 | 30 |
| loss | scheme_C / seed | 2.92 | 6 |
| loss | scheme_C / ckpt | 2.86 | 12 |
| loss | temperature / seed | 7.20 | 3 |
| loss | temperature / ckpt | 1.41 | 18 |
| loss | activation / seed | 26.78 | 1 |
| loss | activation / ckpt | 0.77 | 19 |
| loss | optimizer / seed | 2.82 | 1 |
| loss | optimizer / ckpt | 0.66 | 18 |

![Effect vs noise](pretraining/predictivity_seeds/effect_vs_noise.png)
<!-- END auto:effect-vs-noise -->

GitHub: [effect_vs_noise.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_vs_noise.png) · [effect_vs_noise.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds/effect_vs_noise.csv)

The highlighted SNR divides by the checkpoint noise. This figure asks whether that is the right noise, by setting both noises and each design decision's effect against the seed replicates.

Population: pool `predictivity_seeds`, the nine deep data-A seed cells (175M and 600M at L 1/2/50, 1B at L 1/2/30, three seeds each), gate applied (9245 of 34905 (size, L, task) cells at chance). The 15882 bBPB twin cells are never gated and, now that the per-item store holds the seed replicates, enter the seed-noise rows too: 2883 of the 4650 seed-over-checkpoint cells.

**Key findings**

- **Seed noise exceeds checkpoint noise.** The seed std is a median 1.75× the detrended checkpoint std over 4650 ungated cells (2.04 at 175M, 1.72 at 600M and 1.53 at 1B) and is larger in 75 % of them. By population the ratio is 1.73 for the benchmark originals (1610 cells), 1.82 for the bBPB twins (2883) and 1.33 for per-language BPB (139).
- **So the checkpoint-noise SNR is optimistic.** A design difference that clears the checkpoint noise need not clear a re-roll of the seed: on the same 4562 seed cells, depth's median effect is 2.12× the detrended checkpoint noise but 1.30× the seed noise (52 % against 30 % of the cells above 2×).
- **No design axis clears twice the seed noise in half of its cells.** Against seed noise the median effect is 1.30 for depth, 1.22 for scheme A vs B (1779 cells, 29 % above 2×), 1.66 for scheme A vs C (808, 43 %), 1.80 for temperature T 1 vs 3 (3705, 46 %) and 1.10 for activation (1066, 27 %). Depth, scheme A vs B and activation are at the level of a pure seed re-roll (median 1.15, 29 % above 2×). The optimizer has a single seed cell. The per-population medians are in the table above.

**Follow-ups**

- **A seed-noise SNR.** An SNR variant with the seed std as its noise, ranked against decision accuracy next to the checkpoint-noise variants, because the seed is the noise a design decision has to beat.
- **The ratio table by size.** Splitting the table by size would show whether the seed-over-checkpoint gap closes towards 1B as the runs get longer.

<!-- BEGIN auto:flip-attribution (flip_attribution.py --pool predictivity) -->
## Attribution of the proxy-reference disagreements

Pool `predictivity` (pairs and scores, seed 1904, final checkpoint), with the seed sd from `predictivity_seeds` (rq05's `seed_sd`). (a) means |proxy difference| > 1 x the seed sd of the proxy's final score. (b) means the proxy difference has the other sign than the 1.7B one. That is a DA-size miss, so (b) = 1 - DA-size pooled over comparisons. Regenerate with `python analysis/rq03_noise_and_snr/flip_attribution.py --pool predictivity`. The sqrt(2) x sd threshold, the pooled sd and the benchmark-accuracy population are in the CSV.

**600M, per-language BPB.** The sd comes from the 600M x3 cells (L1, L2, L50). A comparison is one (design pair, trained language), shares are in % of the comparisons, and the last column counts the comparisons in (a) and (b) whose 1.7B difference also clears the threshold:

| pairs | size-matched | design pairs | comparisons | (a) | (b) | (a)(b) | (a) not (b) | not (a) (b) | not (a) not (b) | (a)(b), 1.7B clear |
|---|---|---|---|---|---|---|---|---|---|---|
| Depth (deep vs shallow) | **no** | 10 | 209 | 42 | 95 | 41 | 1 | 54 | 4 | 64 |
| Language count | yes | 45 | 378 | 74 | 15 | 3 | 71 | 12 | 14 | 1 |
| Data scheme (A vs B vs C) | yes | 12 | 80 | 39 | 2 | 1 | 38 | 1 | 60 | 0 |
| Temperature (T=1 vs T=3) | yes | 4 | 145 | 92 | 1 | 0 | 92 | 1 | 7 | 0 |
| Activation (XIELU vs SwiGLU) | yes | 3 | 53 | 21 | 83 | 19 | 2 | 64 | 15 | 6 |
| Optimizer | yes | 3 | 53 | 100 | 70 | 70 | 30 | 0 | 0 | 0 |
| All mono-axis pairs | **no** | 77 | 918 | 65 | 37 | 16 | 49 | 21 | 14 | 71 |
| Multi-axis pairs | **no** | 496 | 4299 | 80 | 21 | 10 | 70 | 11 | 9 | 147 |
| Multi-axis, depth moved | **no** | 220 | 2040 | 78 | 29 | 13 | 64 | 15 | 7 | 119 |
| Multi-axis, depth held | yes | 276 | 2259 | 82 | 14 | 7 | 75 | 7 | 11 | 28 |

**Every proxy, per-language BPB.** A cell reads (b) / (a) / share of (a) that is also (b), in %. The sd comes from the proxy's own x3 cells at 175M, 600M and 1B, and is the pooled sd at 90M and 350M:

| pairs | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| Depth (deep vs shallow) | 4 / 91 / 1 | 4 / 93 / 1 | 27 / 31 / 14 | 95 / 42 / 98 | 6 / 34 / 2 |
| Language count | 12 / 84 / 7 | 12 / 95 / 10 | 15 / 81 / 9 | 15 / 74 / 4 | 17 / 81 / 10 |
| Data scheme (A vs B vs C) | 5 / 50 / 0 | 0 / 96 / 0 | 0 / 39 / 0 | 2 / 39 / 3 | 22 / 52 / 2 |
| Temperature (T=1 vs T=3) | 1 / 97 / 0 | 1 / 98 / 0 | 3 / 93 / 1 | 1 / 92 / 0 | 3 / 89 / 0 |
| Activation (XIELU vs SwiGLU) | 8 / 49 / 0 | 13 / 79 / 5 | 4 / 53 / 0 | 83 / 21 / 91 | 6 / 19 / 0 |
| Optimizer | 62 / 45 / 54 | 70 / 98 / 69 | 70 / 100 / 70 | 70 / 100 / 70 | 70 / 21 / 55 |
| All mono-axis pairs | 11 / 81 / 5 | 11 / 94 / 9 | 17 / 67 / 12 | 37 / 65 / 24 | 16 / 62 / 7 |
| Multi-axis pairs | 7 / 90 / 4 | 8 / 97 / 7 | 15 / 84 / 11 | 21 / 80 / 13 | 10 / 80 / 4 |
| Multi-axis, depth moved | 6 / 94 / 4 | 7 / 98 / 5 | 16 / 80 / 10 | 29 / 78 / 17 | 9 / 79 / 3 |
| Multi-axis, depth held | 9 / 87 / 4 | 10 / 97 / 9 | 15 / 87 / 12 | 14 / 82 / 9 | 12 / 82 / 5 |

**The depth decision with the deep side's seed re-drawn.** Per-language BPB, with deep data A at each of its three seeds against shallow data A at seed 1904. A cell reads the comparisons whose sign differs from the 1.7B pair over all comparisons. Shallow has no replicate, so only the deep side moves:

| proxy | deep seed | per L | pooled (b) | median deep − shallow BPB |
|---|---|---|---|---|
| 175M | 64 | L1 0/1, L2 1/2, L50 0/50 | 2 | -0.0064 / -0.0055 / -0.0086 |
| 175M | 313 | L1 0/1, L2 1/2, L50 1/50 | 4 | -0.0054 / -0.0037 / -0.0079 |
| 175M | 1904 | L1 0/1, L2 1/2, L50 3/50 | 8 | -0.0058 / -0.0045 / -0.0074 |
| 600M | 64 | L1 0/1, L2 1/2, L50 15/50 | 30 | -0.0020 / -0.0009 / -0.0011 |
| 600M | 313 | L1 0/1, L2 1/2, L50 43/50 | 83 | -0.0015 / +0.0010 / +0.0011 |
| 600M | 1904 | L1 0/1, L2 1/2, L50 50/50 | 96 | -0.0025 / +0.0031 / +0.0094 |
| 1B | 28 | L1 0/1, L2 1/2, L30 0/30 | 3 | -0.0024 / -0.0018 / -0.0028 |
| 1B | 1797 | L1 0/1, L2 1/2, L30 30/30 | 94 | -0.0027 / -0.0021 / +0.0022 |
| 1B | 1904 | L1 0/1, L2 1/2, L30 0/30 | 3 | -0.0037 / -0.0022 / -0.0059 |

Tables: `pretraining/predictivity/flip_attribution_da_size_by_axis_both_axes.csv`, `pretraining/predictivity/flip_attribution_da_size_depth_by_seed_mono_axis.csv`. LaTeX: `pretraining/predictivity/flip_attribution_da_size_by_axis_both_axes.tex`.
<!-- END auto:flip-attribution -->

GitHub: [flip_attribution_da_size_by_axis_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/flip_attribution_da_size_by_axis_both_axes.csv) · [flip_attribution_da_size_by_axis_both_axes.tex](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/flip_attribution_da_size_by_axis_both_axes.tex) · [flip_attribution_da_size_depth_by_seed_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity/flip_attribution_da_size_depth_by_seed_mono_axis.csv)

The effect-vs-noise medians above say how often a design effect clears the seed noise. They do not say whether the proxy's misses against the reference are those noisy comparisons. Per-language BPB ranks the design variants like 1.7B from every proxy (DA-size 0.91–0.95) except 600M (0.73, [rq02](../rq02_decision_accuracy/README.md#key-finding)).

This table splits each 600M comparison by two tests: (a) "the 600M difference exceeds the 600M seed sd" and (b) "its sign differs from 1.7B's". (b) without (a) is a miss the proxy could not have avoided, because the variants are indistinguishable at 600M. (a) with (b) is a design choice whose measured effect reverses between the sizes.

Population: pool `predictivity` (seed 1904, every ladder and data build, final checkpoint), with the pairs and scores of rq02's score cube and DA-size against 1.7B. A comparison is one (design pair, trained language) on the 50 per-language BPB tasks (rule 2: both variants train the language). The pairs are the mono-axis pairs per design axis (rule 15), plus the multi-axis pairs pooled and split by whether they move the depth.

Seed sd: rq05's `seed_sd` on `predictivity_seeds`, the per-task median over the deep data-A ×3 cells of the proxy's own size. At 600M these are L1, L2 and L50, so a non-English language's sd is the L50 cell's alone. 175M and 1B use their own cells, and 90M and 350M the pooled 175M–1B sd. The 600M depth pairs are not size-matched (size_matched = no, [RULES.md](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)). Activation (no per-language BPB at 1.7B swiglu yet) and optimizer (no muon cell at 600M or 1.7B) have no comparison. (b) is pooled over comparisons, so the multi-axis row's 22 % is 1 − 0.777, rq02's pooled 600M figure (the 0.73 is the mean over tasks).

**Key findings**

- **The 600M BPB misses sit on the depth axis.** Of the 758 multi-axis comparisons whose sign differs from 1.7B (22 % of 3404), 568 are on pairs that move the depth: 33 % of the 1720 depth-moving comparisons miss, against 11 % of the 1684 with depth held, within the 7–11 % every proxy shows with depth held. Mono-axis, depth misses in 95 % of the comparisons (199 of 209) at 600M, against 4 %, 4 %, 27 % and 6 % at 90M, 175M, 350M and 1B. Language count (16 %), data scheme (2.5 %) and temperature (1.4 %) sit where the other proxies do.
- **On depth, 600M mostly cannot separate the variants, but its misses are not random.** With 1 × the 600M seed sd, 41 % of the depth comparisons are (a) with (b), 54 % (b) without (a), 1 % (a) without (b) and 4 % neither. With √2 × sd (the sd of a difference) the split is 31 % / 65 %, and with the pooled 175M–1B sd it is 53 % / 43 %. Yet 93 % of the comparisons below the sd still flip: the 600M depth difference is small (median 0.87 sd) and consistently in shallow's favour, a systematic reversal rather than noise around zero.
- **The other axes are distinguishable at 600M and agree with 1.7B.** The 600M difference clears 1 sd on 92 % of the temperature, 74 % of the language-count and 39 % of the data-scheme comparisons, and of those only 0 %, 4 % and 3 % flip. The language-count misses (54 of 347) are mostly below the sd (43 of 54).
- **The depth reversal is a draw of the deep run's seed, not a property of 600M.** With shallow fixed at seed 1904 and the deep side at each of its three seeds (64, 313, 1904), the 600M L50 depth pair flips on 15, 43 and 50 of 50 languages, with a median deep − shallow BPB of −0.0011, +0.0011 and +0.0094 against 1.7B's −0.0056. At 1B, deep seed 1797 flips all 30 L30 languages, where seeds 28 and 1904 flip none. So the grid seed's 600M L50 deep run is the worst of its three (its median deep − shallow sits 0.008–0.011 BPB above the other two seeds'), and a 1B re-roll produces the same reversal. The size mismatch (+3.7 % parameters, about +15 % compute for shallow) can tilt the comparison, but deep seed 64 recovers the reference's sign on 35 of 50 languages with the same shallow run.
- **The reference barely separates depth either.** The 1.7B depth difference is a median 1.10 600M seed sds (5.10 for temperature, 1.74 for language count), and 64 of the 86 (a)-with-(b) depth comparisons also clear 1 sd at 1.7B. With no 1.7B replicate, we cannot tell whether those are opposite effects or two seed draws.

**Follow-ups**

- **Shallow replicates.** Two replicate seeds of shallow 600M at L50 (and 1B at L30) would give both sides of the depth pair a seed noise. Today only the deep side moves, so the reading above is one-sided.
- **A size-matched 600M shallow cell.** It would separate the +3.7 % parameters from the seed draw.
- **The split on benchmark accuracy.** It is in the CSV (population `accuracy`) but not in the README. There every axis misses on 43–55 % of comparisons at 600M, and clearing the sd barely changes it (P(b | a) 0.39–0.54 against 0.49–0.57 below the sd), so the attribution question is a BPB one.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Regenerate with `python analysis/rq03_noise_and_snr/panels.py --pool predictivity`. In every grid white means "no value" and grey "filtered out by the gate" (at chance at that size, rule 1). Each figure's table sits next to it under the same name. The SNR grids span 90M–1.7B (rule 10), and the depth-over-seed grids the sizes with seed replicates. The SNR noise is the residual std around a line through the 80/85/90/95/100 % checkpoints (rule 4).

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

The highlight gave medians. These grids show which benchmarks and languages carry the signal. Population: SNR grids on pool `predictivity` (`rel_std`, sizes 90M–1.7B) and depth-over-seed grids on `predictivity_seeds` (4476 ungated cells at 175M, 600M and 1B), gate applied.

**Key findings**

- **HellaSwag has the highest benchmark SNR at 1.7B.** Over 40 benchmark families with an ungated task (50 languages, 588 tasks), the highest median log10 SNR is `hellaswag` 0.62 (26 tasks), then `toksuite_math` 0.62 and `global_mmlu_full` 0.53 (one ungated task each) and `xstorycloze` 0.49. The lowest are `kobest` −0.09, `noreval_extra` −0.07, `toksuite` −0.02 and `winogrande` −0.01, the only four families below the noise.
- **Reformulation does not reliably raise SNR at 1.7B here.** Two of the three `rf` twins with an ungated original are lower (`rf_global_mmlu_full` 0.39 against `global_mmlu_full` 0.53, `rf_include_base_44` 0.16 against 0.28) and `rf_belebele` is higher (0.15 against 0.11), and of the `rfgm` twins `rfgm_include_base_44` is lower (0.24 against 0.28) but `rfgm_belebele` is higher (0.20 against 0.11). The gate keeps a different task set for each twin, so this is a reading at the level of populations.
- **Every language but Lithuanian and Norwegian is above the noise at 1.7B.** On the benchmarks the median log10 SNR runs from −0.03 (Lithuanian, 8 tasks) to 0.51 (Slovenian, 3 tasks). Norwegian is the other language below 0 (−0.005 over 38 tasks), and English is 0.29 over 40 tasks. A language rests on 1 to 54 tasks, so the extremes are thin.
- **Depth beats the seed noise in most families.** The median log10 ratio is above 0 in 45 of the 56 families with at least 10 cells (the bBPB twins included), from −0.42 (`bbpb_blend_sample`, 21 cells) to 0.39 (`bbpb_belebele`, 165 cells). The 26 original families run from −0.01 (`rfgm_include_base_44`) to 0.25 (`paws`, 14 cells, and per-language BPB, 139 cells). The 49 languages with at least 10 cells run from −0.07 (Kazakh, 38 cells) to 0.31 (Slovak, 31 cells), Bosnian, the one language under 10 cells, reads 0.16 over 5, and English 0.07 over 1178 cells.

**Follow-ups**

- **Languages with enough tasks.** The per-language SNR grid restricted to languages with at least 10 ungated tasks, because the extremes of the ranking rest on few tasks.
- **Twins on a shared task set.** Original vs reformulated twins on the tasks both keep after the gate would separate the format effect from the population effect.

The SNR ranking and the 22 definitions feed the [surrogates analysis](../rq04_surrogates/README.md). Whether that ranking of definitions survives a change of seed is the holdout below.

<!-- BEGIN auto:seed-holdout (compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test) -->
## Seed holdout

Regenerate with `python analysis/rq03_noise_and_snr/compare_seed_splits.py --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test`. The tables are under `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/`.

`predictivity_seeds_train` (seeds 28, 64, 313, 1797) and `predictivity_seeds_test` (seeds 1904) hold the same 10 cells: 175M L1 deep data A, 1B L1 deep data A, 600M L1 deep data A, 175M L2 deep data A, 1B L2 deep data A, 600M L2 deep data A, 1B L30 deep data A, 1B L30 deep data B, 175M L50 deep data A, 600M L50 deep data A.

A language's r is rq04's Pearson r over its tasks' (log10 SNR, DA) points and needs at least 3 distinct tasks with a value (rule 8). `multi` and `??` are never a language (rule 7). DA-size on the holdout is 175M → 600M (scaling pair). The pools hold no 1.7B run, so the reference never enters (rule 9). DA-ckpt is the within-size early → final ranking. Agreement is counted over the languages with a best variant on both splits.

| DA | languages | same variant | same family | Spearman ρ of the variant ranking |
|---|---|---|---|---|
| DA-size | 1 | 0 | 1 | 0.55 |
| DA-ckpt | 2 | 0 | 1 | 0.73 |
<!-- END auto:seed-holdout -->

[headline_metrics.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv) ·
[summary.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/summary.md)

Population: train = the replicate seeds 28/64/313/1797, test = seed 1904, the same 10 cells, pairs within each split, gate of `predictivity`. DA-size exists only for 175M → 600M (214 of the 1806 train-split tasks: 212 English tasks, 98 of them bBPB twins, and 2 aggregates), so the 1B seeds do not enter it and neither does the 1.7B reference.

**Key findings**

- **DA-ckpt: the family survives a seed swap in one of two languages.** Two languages (English, Russian) have a best variant on both splits. The train pick is the same family in 1 of 2 (relative spread for Russian, while English moves from discrepancy on train to relative spread on test) and the same variant in 0 of 2. The global variant ranking agrees at Spearman ρ 0.72, the variant cells at Pearson r 0.29 (n = 44), and the train pick keeps 98 % of the test-best r.
- **DA-size: now agrees, but on one size pair and one language.** Only English has a best variant on both splits, `dispersion_shifted` on both (1 of 1 at the variant and the family level). The global ranking agrees at ρ 0.55 and the cells at r 0.62 (n = 22), and the train pick keeps 100 % of the test-best r. On the 10-06 snapshot the test pick was `rms_deviation` and ρ was −0.28, so the sign rests on a single language.

**Follow-ups**

- **A holdout on tasks with enough pairs.** The DA-size holdout restricted to tasks with at least six pairs on both splits (option A of the TODO below), stated as English and BPB only.


## TODO

- [ ] **The seed holdout ranks most tasks on a single model pair (open, 2026-09-17).**
      *What it is.* The surrogates analysis claims one SNR definition tracks
      decision accuracy best. The holdout (`compare_seed_splits.py`, reported
      in the surrogates highlight and "Seed generalization" table) asks
      whether that ranking of definitions survives a change of seed, by
      computing it on `predictivity_seeds_train` (seeds 64, 313, 28, 1797)
      and on `predictivity_seeds_test` (seed 1904 of the same 10 cells).
      *The problem (re-read 2026-10-10).* Since 2026-10-05 the 1B ×3 cells
      (seeds 28/1797) are in both splits, but DA-size still exists only for
      175M → 600M, so the 1.7B reference never enters. A non-English
      benchmark is trained only at L30/L50 (Russian also at L2), so its
      split holds no multi-axis pair and no DA-size: 214 of the 1806
      train-split tasks have a DA-size value, 212 English tasks (98 of them
      bBPB twins) and 2 aggregates.
      *Implications.* The DA-size agreement (1 of 1 language at the family
      level and ρ 0.55, against 0 of 1 and ρ −0.28 on the 10-06 snapshot)
      rests on English alone, and DA-ckpt on two languages (English, Russian).
      Whether "the ranking survives a seed swap" therefore says more about the
      holdout than about the definitions: one language flipped it between two
      snapshots.
      Nothing in the main `predictivity` tables is affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate. This changes what the holdout means. (C) report the
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
- The readings built on them (the variant ranking per tier, the holdout's
  Spearman ρ, the per-language anchors) are in
  [the surrogates analysis' extensions](../rq04_surrogates/README.md#extensions-from-other-sweeps).
  The dated comparison note that once accompanied them
  (`ANALYSIS_new_vs_previous.md`, removed 2026-09-23) is superseded by that section.

## Files

- `pretraining/<pool>/snr_variants_per_task.csv`: per-task SNR (every
  variant × size bucket) + the DA columns joined from the decision accuracy
  analysis. Single source of truth for the surrogates, external frameworks
  and benchmark design analyses (`run_apertus_snr_variants.py`).
- `…/snr_variants_definitions.csv`, `snr_variant_coverage.csv`: the 22
  definitions and how many cells each covers per bucket.
- `…/effect_vs_noise.csv`, `effect_vs_noise.png`: per (size, L, task), |Δ|
  per intervention, seed noise, raw and detrended checkpoint noise, and their
  ratios (`effect_vs_noise.py`).
- `pretraining/predictivity/flip_attribution_da_size_by_axis_both_axes.csv`
  / `.tex`: per (population, sd scope, threshold, proxy size, pair group),
  the comparisons against 1.7B split by (a) |proxy Δ| > K × seed sd and (b)
  a sign flip, with the cross tab and `size_matched`. The `.tex` holds the 600M
  per-language BPB rows. `flip_attribution_da_size_depth_by_seed_mono_axis.csv`
  is the depth pair with the deep side at each replicate seed
  (`flip_attribution.py`).
- `pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/`: the
  seed-holdout report (`compare_seed_splits.py`: `headline_metrics.csv`, the
  per-language agreement and the variant r train vs test).
