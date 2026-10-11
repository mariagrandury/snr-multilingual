# Decision accuracy on the decisions the reference makes

**Question.** Does a proxy rank the design decisions better once the pairs the 1.7B reference itself cannot separate from seed noise are left out?

[DA-size](../rq02_decision_accuracy/README.md) counts every pair of design variants, including pairs whose 1.7B scores differ by less than seed noise. On those pairs the reference's order is random. A proxy that "misses" it has missed nothing, yet these pairs pull every DA toward 0.5. The [permutation null](../rq02_permutation_null/README.md) asks how far DA lies from chance. This analysis asks how much of the gap to 1 comes from the reference's own noise.

## Setup

**Setup.**
- **Measured.** We measure DA-size on the pairs that the 1.7B reference separates by more than seed noise.
- **Models and pairs.** We use the seed-1904 runs of every language setting, ladder and data build. The proxies (90M–1B) are compared against the final 1.7B checkpoint, on pairs that differ on any number of design axes.
- **Decisive pairs.** A pair is decisive for a task when its 1.7B gap exceeds k √2 times the task's seed standard deviation, for k ∈ {1, 2}. The factor √2 turns the noise of one run into the noise of a difference of two runs. A task's seed standard deviation is the standard deviation of its final score across seeds, taken as the median over the replicated baseline cells (175M, 600M and 1B).
- **Assumption.** There are no replicates at 1.7B, so the threshold assumes that seed noise does not grow with size.
- **Tasks.** Every line reads the same tasks: the benchmark tasks and their bBPB twins with a seed standard deviation and at least three decisive pairs at k = 2, above chance at the proxy and at 1.7B, and the per-language BPB of the 50 trained languages. Each line divides the matching pairs by the comparable pairs, summed over tasks, so a task weighs by its pairs: on BPB English holds 17 % of the decisive pairs at k = 2, and the mean over the 50 languages is higher at 600M (0.99 against 0.95, `da_cell_mean` in the CSV, as in the definition table of the DA appendix).

- **Snapshot.** Ladder report of **2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended checkpoint noise, prose re-read on 2026-10-10), pool `predictivity` (seed 1904, every ladder and data build), proxies 90M–1B against the 1.7B final.
- **Decision accuracy.** DA-size, no reliability filter, multi-axis and mono-axis pairs (rule 15), pooled as matching over comparable pairs. Gate `predictivity` at the proxy and at 1.7B (rule 1). A cell needs 3 kept pairs (rule 5).
- **Decisive pairs.** A pair is kept when its 1.7B gap exceeds K × √2 × the task's seed sd, K ∈ {1, 2} (√2: the sd of a difference of two runs). The seed sd is the one of the [design-decisions analysis](../rq05_design_decisions/README.md): per task, the median over the replicated baseline cells (deep, data A, at 175M, 600M and 1B in `predictivity_seeds`) of the sd of the final score. There are no replicates at 1.7B, so the threshold assumes the seed noise does not grow with size (the [noise analysis](../rq03_noise_and_snr/README.md) finds it near size-invariant).
- **Population.** Every line reads the same tasks: those with a seed sd and at least 3 decisive pairs at K = 2, so the lines differ by their pairs alone (rule 13). The bBPB twins now have replicates (their store holds the seed runs), so "Benchmarks" here includes them, as in the rq02 figures.

## Decisive pairs

![DA-size on decisive pairs, multi-axis](pretraining/predictivity/decisive_pairs_da_size_paper.png)

DA-size (no filter, multi-axis pairs, gate `predictivity`) on every pair and on the decisive pairs, and the share of pairs kept. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.csv)

Key findings:

- **Benchmark DA rises on decisive pairs.** On the same 876–1020 tasks, multi-axis DA-size goes from 0.58–0.59 on every pair to 0.64–0.66 above one seed sd and 0.71–0.73 above two, so part of the gap to 1 is the reference's own noise and not the proxy's.
- **Most benchmark pairs are not decisive.** One seed sd keeps 52–53 % of the comparable pairs and two keep 25 %, so on a single benchmark task the reference separates few of the design pairs.
- **BPB is near perfect on decisive pairs.** It reads 0.89–0.98 above one seed sd and 0.95–0.995 above two, with 72 % and 52 % of the pairs kept. Its dip at 600M shrinks, from 0.78 on every pair to 0.89 and 0.95.
- **Mono-axis decisions gain less.** On 563–661 tasks, benchmark DA goes from 0.55–0.57 on every pair to 0.59–0.62 above one seed sd and 0.62–0.66 above two, on 54 % and 25–26 % of the pairs.

Follow-ups:

- **A seed sd measured at 1.7B.** Replicates at the reference would remove the size-invariance assumption.
- **Decisive pairs inside the reliable-task verdict.** Reading `above_66_either` on decisive pairs would separate tasks that rank badly from tasks whose reference cannot rank.

<!-- BEGIN auto:decisive-pairs (decisive_pairs.py --pool predictivity) -->
## Results

Pool `predictivity`, seed sd from `predictivity_seeds`. A pair counts when both proxies and both reference runs have a score. A cell is gated at the proxy and at 1.7B (rule 1). Every line reads the tasks with 3 decisive pairs at the largest K (rule 5), so the lines differ by their pairs alone. Regenerate with `python analysis/rq02_decisive_pairs/decisive_pairs.py --pool predictivity`.

![DA-size on decisive pairs](pretraining/predictivity/decisive_pairs_da_size_both_axes.png)

Pooled DA-size, followed by the number of tasks and the share of the comparable pairs kept:

| pairs | population | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| multi-axis | Benchmarks | every pair | 0.586 (924; 100%) | 0.574 (950; 100%) | 0.588 (971; 100%) | 0.582 (1006; 100%) | 0.586 (1055; 100%) |
| multi-axis | BPB | every pair | 0.926 (50; 100%) | 0.916 (50; 100%) | 0.848 (50; 100%) | 0.789 (50; 100%) | 0.899 (50; 100%) |
| multi-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.646 (924; 52%) | 0.629 (950; 52%) | 0.653 (971; 52%) | 0.646 (1006; 52%) | 0.650 (1055; 52%) |
| multi-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.981 (50; 70%) | 0.976 (50; 70%) | 0.925 (50; 70%) | 0.898 (50; 70%) | 0.974 (50; 70%) |
| multi-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.706 (924; 24%) | 0.686 (950; 25%) | 0.727 (971; 24%) | 0.717 (1006; 25%) | 0.718 (1055; 24%) |
| multi-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.990 (50; 48%) | 0.990 (50; 48%) | 0.958 (50; 48%) | 0.951 (50; 48%) | 0.996 (50; 48%) |
| mono-axis | Benchmarks | every pair | 0.567 (607; 100%) | 0.543 (626; 100%) | 0.559 (640; 100%) | 0.546 (669; 100%) | 0.548 (701; 100%) |
| mono-axis | BPB | every pair | 0.889 (32; 100%) | 0.885 (32; 100%) | 0.819 (32; 100%) | 0.638 (32; 100%) | 0.839 (32; 100%) |
| mono-axis | Benchmarks | reference gap > 1 x sqrt(2) seed sd | 0.611 (607; 53%) | 0.575 (626; 53%) | 0.610 (640; 53%) | 0.595 (669; 53%) | 0.593 (701; 53%) |
| mono-axis | BPB | reference gap > 1 x sqrt(2) seed sd | 0.982 (32; 56%) | 0.980 (32; 56%) | 0.939 (32; 56%) | 0.851 (32; 56%) | 0.964 (32; 56%) |
| mono-axis | Benchmarks | reference gap > 2 x sqrt(2) seed sd | 0.649 (607; 24%) | 0.608 (626; 24%) | 0.656 (640; 24%) | 0.653 (669; 24%) | 0.638 (701; 24%) |
| mono-axis | BPB | reference gap > 2 x sqrt(2) seed sd | 0.992 (32; 31%) | 0.992 (32; 31%) | 0.964 (32; 31%) | 0.936 (32; 31%) | 0.992 (32; 31%) |

Table: `decisive_pairs_da_size_both_axes.csv`.
<!-- END auto:decisive-pairs -->
