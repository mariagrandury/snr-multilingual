# Decision accuracy on the decisions the reference makes

**Question.** Does a proxy rank the design decisions better once the pairs the 1.7B reference itself cannot separate from seed noise are left out?

[DA-size](../rq02_decision_accuracy/README.md) counts every pair of design variants, including pairs whose 1.7B scores differ by less than seed noise. On those pairs the reference's order is a coin flip, so a proxy that "misses" it has missed nothing, and they pull every DA toward 0.5. The [permutation null](../rq02_permutation_null/README.md) asks how far DA is from chance; this analysis asks how much of the gap to 1 is the reference's own noise.

## Setup

- **Snapshot.** Ladder report of **2026-10-06 04:26** (`d053e00c`), pool `predictivity` (seed 1904, every ladder and data build), proxies 90M–1B against the 1.7B final.
- **Decision accuracy.** DA-size, no reliability filter, multi-axis and mono-axis pairs (rule 15), pooled as matching over comparable pairs. Gate `predictivity` at the proxy and at 1.7B (rule 1); a cell needs 3 kept pairs (rule 5).
- **Decisive pairs.** A pair is kept when its 1.7B gap exceeds K × √2 × the task's seed sd, K ∈ {1, 2} (√2: the sd of a difference of two runs). The seed sd is the [design-decisions analysis](../rq05_design_decisions/README.md)'s: per task, the median over the replicated baseline cells (deep, data A; 175M, 600M and 1B in `predictivity_seeds`) of the sd of the final score. There are no replicates at 1.7B, so the threshold assumes the seed noise does not grow with size (the [noise analysis](../rq03_noise_and_snr/README.md) finds it near size-invariant).
- **Population.** Every line reads the same tasks: those with a seed sd and at least 3 decisive pairs at K = 2, so the lines differ by their pairs alone (rule 13). The bBPB twins have no replicates (their store holds seed 1904 only), so "Benchmarks" here is the accuracy tasks only, unlike the rq02 figures.

## Decisive pairs

![DA-size on decisive pairs, multi-axis](pretraining/predictivity/decisive_pairs_da_size_paper.png)

DA-size (no filter, multi-axis pairs, gate `predictivity`) on every pair and on the decisive pairs, and the share of pairs kept. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.csv)

Key findings:

- **Benchmark DA rises on decisive pairs.** On the same 206–303 tasks, multi-axis DA-size goes from 0.55–0.58 on every pair to 0.61–0.65 above one seed sd and 0.67–0.72 above two; part of the gap to 1 is the reference's own noise, not the proxy's.
- **Most benchmark pairs are not decisive.** One seed sd keeps 56–57 % of the comparable pairs, two keep 26–27 %; the reference separates few of the design pairs on a single benchmark task.
- **BPB is near perfect on decisive pairs.** It reads 0.88–0.99 above one seed sd and 0.95–0.995 above two, with 75 % and 56 % of the pairs kept; its dip at 600M shrinks from 0.79 to 0.88 and 0.95.
- **Mono-axis decisions gain less.** On 131–193 tasks, benchmark DA goes from 0.53–0.54 to 0.57–0.60 and 0.61–0.66, on 57–58 % and 27–28 % of the pairs.

Follow-ups:

- **A seed sd measured at 1.7B.** Replicates at the reference would remove the size-invariance assumption.
- **Decisive pairs inside the reliable-task verdict.** Reading `above_66_either` on decisive pairs would separate tasks that rank badly from tasks whose reference cannot rank.

<!-- BEGIN auto:decisive-pairs (decisive_pairs.py --pool predictivity) -->
## Results

Pool `predictivity`, seed sd from `predictivity_seeds`. A pair counts when both proxies and both reference runs have a score; a cell is gated at the proxy and at 1.7B (rule 1), and every line reads the tasks with 3 decisive pairs at the largest K (rule 5), so the lines differ by their pairs alone. Regenerate with `python analysis/rq02_decisive_pairs/decisive_pairs.py --pool predictivity`.

![DA-size on decisive pairs](pretraining/predictivity/decisive_pairs_da_size_both_axes.png)

Pooled DA-size (tasks; share of the comparable pairs kept):

| pairs | population | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| multi-axis | Benchmarks | every pair | 0.550 (206; 100%) | 0.564 (229; 100%) | 0.552 (249; 100%) | 0.562 (271; 100%) | 0.576 (303; 100%) |
| multi-axis | BPB | every pair | 0.945 (50; 100%) | 0.944 (50; 100%) | 0.885 (50; 100%) | 0.785 (50; 100%) | 0.910 (50; 100%) |
| multi-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.607 (206; 57%) | 0.626 (229; 57%) | 0.613 (249; 56%) | 0.637 (271; 56%) | 0.648 (303; 56%) |
| multi-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.986 (50; 75%) | 0.986 (50; 75%) | 0.951 (50; 75%) | 0.882 (50; 75%) | 0.975 (50; 75%) |
| multi-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.674 (206; 27%) | 0.691 (229; 27%) | 0.690 (249; 26%) | 0.718 (271; 27%) | 0.722 (303; 26%) |
| multi-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.990 (50; 56%) | 0.993 (50; 56%) | 0.973 (50; 56%) | 0.945 (50; 56%) | 0.995 (50; 56%) |
| mono-axis | Benchmarks | every pair | 0.528 (131; 100%) | 0.541 (147; 100%) | 0.527 (154; 100%) | 0.539 (172; 100%) | 0.536 (193; 100%) |
| mono-axis | BPB | every pair | 0.919 (32; 100%) | 0.923 (32; 100%) | 0.836 (32; 100%) | 0.674 (32; 100%) | 0.865 (32; 100%) |
| mono-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.570 (131; 57%) | 0.585 (147; 58%) | 0.576 (154; 57%) | 0.599 (172; 57%) | 0.580 (193; 57%) |
| mono-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.987 (32; 60%) | 0.987 (32; 60%) | 0.940 (32; 60%) | 0.850 (32; 60%) | 0.969 (32; 60%) |
| mono-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.610 (131; 27%) | 0.634 (147; 28%) | 0.625 (154; 27%) | 0.658 (172; 28%) | 0.629 (193; 27%) |
| mono-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.991 (32; 37%) | 0.991 (32; 37%) | 0.962 (32; 37%) | 0.936 (32; 37%) | 0.991 (32; 37%) |

Table: `decisive_pairs_da_size_both_axes.csv`.
<!-- END auto:decisive-pairs -->
