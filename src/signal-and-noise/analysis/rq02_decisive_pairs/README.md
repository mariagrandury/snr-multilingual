# Decision accuracy on the decisions the reference makes

**Question.** Does a proxy rank the design decisions better once the pairs the 1.7B reference itself cannot separate from seed noise are left out?

[DA-size](../rq02_decision_accuracy/README.md) counts every pair of design variants, including pairs whose 1.7B scores differ by less than seed noise. On those pairs the reference's order is a coin flip, so a proxy that "misses" it has missed nothing, and they pull every DA toward 0.5. The [permutation null](../rq02_permutation_null/README.md) asks how far DA is from chance; this analysis asks how much of the gap to 1 is the reference's own noise.

## Setup

- **Snapshot.** Ladder report of **2026-10-08 12:06** and the outputs of the 2026-10-08 refresh (detrended checkpoint noise), pool `predictivity` (seed 1904, every ladder and data build), proxies 90M–1B against the 1.7B final.
- **Decision accuracy.** DA-size, no reliability filter, multi-axis and mono-axis pairs (rule 15), pooled as matching over comparable pairs. Gate `predictivity` at the proxy and at 1.7B (rule 1); a cell needs 3 kept pairs (rule 5).
- **Decisive pairs.** A pair is kept when its 1.7B gap exceeds K × √2 × the task's seed sd, K ∈ {1, 2} (√2: the sd of a difference of two runs). The seed sd is the [design-decisions analysis](../rq05_design_decisions/README.md)'s: per task, the median over the replicated baseline cells (deep, data A; 175M, 600M and 1B in `predictivity_seeds`) of the sd of the final score. There are no replicates at 1.7B, so the threshold assumes the seed noise does not grow with size (the [noise analysis](../rq03_noise_and_snr/README.md) finds it near size-invariant).
- **Population.** Every line reads the same tasks: those with a seed sd and at least 3 decisive pairs at K = 2, so the lines differ by their pairs alone (rule 13). The bBPB twins now have replicates (their store holds the seed runs), so "Benchmarks" here includes them, as in the rq02 figures.

## Decisive pairs

![DA-size on decisive pairs, multi-axis](pretraining/predictivity/decisive_pairs_da_size_paper.png)

DA-size (no filter, multi-axis pairs, gate `predictivity`) on every pair and on the decisive pairs, and the share of pairs kept. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.csv)

Key findings:

- **Benchmark DA rises on decisive pairs.** On the same 849–962 tasks, multi-axis DA-size goes from 0.58–0.59 on every pair to 0.64–0.66 above one seed sd and 0.71–0.73 above two; part of the gap to 1 is the reference's own noise, not the proxy's.
- **Most benchmark pairs are not decisive.** One seed sd keeps 52 % of the comparable pairs, two keep 25 %; the reference separates few of the design pairs on a single benchmark task.
- **BPB is near perfect on decisive pairs.** It reads 0.88–0.99 above one seed sd and 0.95–0.995 above two, with 75 % and 56 % of the pairs kept; its dip at 600M shrinks from 0.79 to 0.88 and 0.95.
- **Mono-axis decisions gain less.** On 544–623 tasks, benchmark DA goes from 0.55–0.57 to 0.59–0.62 and 0.63–0.66, on 54 % and 25 % of the pairs.

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
| multi-axis | Benchmarks | every pair | 0.595 (849; 100%) | 0.584 (873; 100%) | 0.586 (892; 100%) | 0.584 (924; 100%) | 0.590 (962; 100%) |
| multi-axis | BPB | every pair | 0.945 (50; 100%) | 0.944 (50; 100%) | 0.885 (50; 100%) | 0.785 (50; 100%) | 0.910 (50; 100%) |
| multi-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.660 (849; 52%) | 0.643 (873; 52%) | 0.651 (892; 52%) | 0.650 (924; 52%) | 0.656 (962; 52%) |
| multi-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.986 (50; 75%) | 0.986 (50; 75%) | 0.951 (50; 75%) | 0.882 (50; 75%) | 0.975 (50; 75%) |
| multi-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.729 (849; 25%) | 0.709 (873; 25%) | 0.726 (892; 25%) | 0.724 (924; 25%) | 0.725 (962; 25%) |
| multi-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.990 (50; 56%) | 0.993 (50; 56%) | 0.973 (50; 56%) | 0.945 (50; 56%) | 0.995 (50; 56%) |
| mono-axis | Benchmarks | every pair | 0.573 (544; 100%) | 0.553 (561; 100%) | 0.558 (569; 100%) | 0.549 (599; 100%) | 0.551 (623; 100%) |
| mono-axis | BPB | every pair | 0.919 (32; 100%) | 0.923 (32; 100%) | 0.836 (32; 100%) | 0.674 (32; 100%) | 0.865 (32; 100%) |
| mono-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.619 (544; 54%) | 0.588 (561; 54%) | 0.608 (569; 54%) | 0.598 (599; 54%) | 0.594 (623; 54%) |
| mono-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.987 (32; 60%) | 0.987 (32; 60%) | 0.940 (32; 60%) | 0.850 (32; 60%) | 0.969 (32; 60%) |
| mono-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.661 (544; 25%) | 0.625 (561; 25%) | 0.651 (569; 25%) | 0.654 (599; 25%) | 0.642 (623; 25%) |
| mono-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.991 (32; 37%) | 0.991 (32; 37%) | 0.962 (32; 37%) | 0.936 (32; 37%) | 0.991 (32; 37%) |

Table: `decisive_pairs_da_size_both_axes.csv`.
<!-- END auto:decisive-pairs -->
