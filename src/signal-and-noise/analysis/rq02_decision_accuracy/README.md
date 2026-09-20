# RQ2 — Does a benchmark rank models at a small size the way the reference size does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**) and across training (**DA-ckpt**)?

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 175M → 1.7B 0.58 / 0.75; 350M → 1.7B 0.57 / 0.97; 600M → 1.7B 0.58 / 0.99; 1B → 1.7B 0.59 / 0.79.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 175M 0.86; 350M 0.90; 600M 1.00; 1B 0.93.
- **DA-size of `train_loss`** (one task, kept out of the means above): 175M 0.85; 350M 0.90; 600M 0.83; 1B 0.89.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 175M 90 % (0.87).
<!-- END auto:highlight -->

## Experimental setup

Models are the predictivity ladder's cells (`configs/models.json` pool
`predictivity`: 90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50, 100} × deep/shallow ×
scheme A/B, seed 1904; `predictivity_seeds` adds the seed replicates as
separate models). The cross-size identity is the cell's `family`
(`lm-L8-schemeB-deep-seed1904`: everything but the size), so a design variant
present at two sizes is one pair.

- **DA-size** — `decision_acc_size_<small>`: the families' ranking at
  `<small>`'s final checkpoint vs at the reference size's (1.7B) final checkpoint;
  `decision_acc_size_<a>_to_<b>` for every other bucket pair with ≥ 2 shared
  families (90M→175M … 1B→1.7B). Multilingual tasks are only evaluated on
  cells that train the language, so each task's pair set is the families
  that exist at both sizes *and* were evaluated on it.
- **DA-ckpt** — `decision_acc_ckpt_f<frac>_<size>`: within one size, the
  ranking at the checkpoint nearest 20/40/60/80 % of each run (1×C … 4×C on
  the WSD schedule) vs at the final checkpoint.

Per-language BPB (`bpb_<subset>`) and the training loss are tasks too, so DA
is computed for the plan's outcome metric alongside the benchmarks. DA is
never gated: it is the truth the SNR proxies of rq03 and rq04 are scored against.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_per_task.csv`
(one row per parent task, one column per DA definition) from
`snr.metrics.decision_acc_fast`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. rq03 joins the SNR variants onto this table; rq05 asks the
complementary question — which proxy *size* ranks an intervention like the
reference, with languages as the population.

With few families at a bucket (the 1.7B row has five language settings; a
task in one language may have three), DA is quantised to 1/#pairs: read the
family-level averages, and `n` alongside every value
(`da_n_pairs_per_task.csv`).

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool (`da_per_task.csv`, pairs from `da_n_pairs_per_task.csv`, gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size** (`n` tasks; median pairs per cell):

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 175M → 1.7B | 0.58 | 64 | 15 | 0.75 | 11 |
| 350M → 1.7B | 0.57 | 80 | 15 | 0.97 | 11 |
| 600M → 1.7B | 0.58 | 85 | 15 | 0.99 | 11 |
| 1B → 1.7B | 0.59 | 90 | 10 | 0.79 | 11 |

![DA-size by family](pretraining/predictivity/da_size_by_family.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.59 | 0.58 | 0.59 | 0.73 | 0.75 | 0.73 | 0.76 | 0.81 | 0.87 |
| 350M | 0.52 | 0.57 | 0.61 | 0.61 | 0.64 | 0.65 | 0.71 | 0.76 | 0.83 |
| 600M | 0.54 | 0.57 | 0.58 | 0.60 | 0.59 | 0.64 | 0.62 | 0.67 | 0.79 |
| 1B | 0.55 | 0.62 | 0.63 | 0.60 | 0.64 | 0.68 | 0.67 | 0.71 | 0.76 |
| 1.7B | 0.56 | 0.61 | 0.62 | 0.64 | 0.63 | 0.64 | 0.64 | 0.69 | 0.75 |
<!-- END auto:results -->

## Departure from upstream: tie handling in `decision_acc_fast`

Upstream's kernel (`snr/metrics.py`, kept verbatim as
`decision_acc_fast_upstream`) compares `a > b` on both vectors, so a pair the
proxy ties but the target decides counts as an agreement or a disagreement
depending on which model is listed first — the value depends on the listing
order (alphabetical by family here), not on the benchmark. Since 2026-09-16
`decision_acc_fast` compares the sign of the difference on both sides
instead: tied in both agrees, tied in one and decided in the other is a miss,
and the result is order-invariant. Without ties the two are identical.

Measured on the `predictivity` pool at the final checkpoints (2026-09-16):
3,301 of 119,456 family pairs are exact ties (2.76 %); 4.05 % of benchmark
pairs, where accuracy over a fixed item count ties easily, and none of the
per-language BPB or loss pairs; 1,225 of 2,845 (task, size) cells contain at
least one tie. The tables committed on 2026-09-16 were produced with the
upstream kernel and are regenerated by `run_all_predictivity.sh`.

## Preliminary findings (ladder snapshot, 2026-09-01)

Computed on the eval results of the ≤ 600M ladder before this pipeline existed
(`plan/status-09-01.md`, §4): ranking the language-count recipes (L1…L50,
deep, seed 1904) by final score at a small size vs at 600M, 60 of 219 parent
benchmarks had ≥ 4 recipes in common.

- Best: `hellaswag_de` (0.94), `xwinograd_jp` (0.78), `arc_it` (0.78),
  `hellaswag_ru` (0.77), `global_mmlu_full_en` (0.73); the HellaSwag family is
  the most decision-reliable overall (mean 0.72; 0.91 deciding from 350M).
- At or below the coin flip: the Global-MMLU family (0.48), several
  INCLUDE / Belebele / XNLI variants, `multiblimp_spa` (0.22) — a benchmark
  at chance has nothing to rank (their across-recipe range at 600M is ~0.01).
- DA improves with the deciding size for the emerged families (HellaSwag
  0.52 → 0.91 from 90M → 350M; ARC 0.52 → 0.64) but not for chance-level ones.
- English benchmarks average 0.63 vs 0.55 for non-English, mostly the
  chance-level knowledge tasks dragging the non-English pool.

These numbers use the 4–6 language-count recipes as the model population;
the pipeline above uses every design variant at a size and will supersede
them.

## Files

- `pretraining/<pool>/da_per_task.csv` — the DA table (single source of truth
  for rq03).
- `…/da_per_benchmark.csv`, `da_per_benchmark_size.csv`,
  `da_per_benchmark_ckpt.csv` — long and wide per-(language, benchmark) views.

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

Numbers from the `predictivity` pool: every design variant at a proxy size, read at 1C–5C of training (C = the Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), ranked against the same variants at the 1.7B final checkpoint (the 5C column is DA-size, the 1.7B row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **175M at 1C** (0.78).
- **all benchmarks** — no (proxy, checkpoint) reaches a mean agreement of 0.75.
- **Smallest safe size per (benchmark, language)** — never: 70, 1B: 15, 350M: 8, 175M: 6, 600M: 3 of 102 cells.

![rq02 in one figure](pretraining/predictivity/highlights.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 11 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.40 | 0.78 | 0.73 | 0.74 | 0.99 | 0.75 | 0.44 | 0.75 | 0.75 | 0.75 |
| 350M | 0.84 | 0.97 | 0.98 | 0.97 | 0.97 | 0.97 | 0.97 | 0.94 | 0.97 | 0.97 |
| 600M | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.99 | 0.98 | 0.99 | 0.99 |
| 1B | 0.81 | 0.95 | 0.89 | 0.83 | 0.87 | 0.86 | 0.83 | 0.73 | 0.76 | 0.79 |
| 1.7B | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 106 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.54 | 0.49 | 0.57 | 0.56 | 0.58 | 0.59 | 0.56 | 0.55 | 0.56 | 0.58 |
| 350M | 0.46 | 0.48 | 0.54 | 0.54 | 0.57 | 0.57 | 0.57 | 0.56 | 0.56 | 0.57 |
| 600M | 0.50 | 0.50 | 0.53 | 0.51 | 0.53 | 0.55 | 0.54 | 0.57 | 0.56 | 0.58 |
| 1B | 0.54 | 0.59 | 0.59 | 0.56 | 0.55 | 0.59 | 0.57 | 0.58 | 0.58 | 0.59 |
| 1.7B | 0.56 | 0.61 | 0.62 | 0.64 | 0.63 | 0.64 | 0.64 | 0.69 | 0.75 |  |

![Early and small](pretraining/predictivity/early_small.png)

![Early and small per benchmark](pretraining/predictivity/early_small_by_benchmark.png)

![Early and small per language](pretraining/predictivity/early_small_by_language.png)

**Smallest safe level per language and benchmark** (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level — for FLOPs, at every costlier (size, checkpoint) cell; red = never, grey = filtered out by the above-random gate, white = no value). Each figure's table sits next to it under the same name:

![Smallest safe size](pretraining/predictivity/safe_size.png)

![Smallest safe checkpoint](pretraining/predictivity/safe_checkpoint.png)

![Smallest safe FLOPs](pretraining/predictivity/safe_flops.png)

![DA-size per benchmark](pretraining/predictivity/da_size_by_benchmark.png)

![DA-size per language](pretraining/predictivity/da_size_by_language.png)
<!-- END auto:early-small -->

<!-- BEGIN auto:by-L (by_L.py --pool predictivity) -->
## Per language count

**Pairs per L** — the design variants the grid plans at seed 1904, and per proxy size the pairs usable against 1.7B (both members planned at that size and at 1.7B): planned / with data today on BPB / on the benchmarks / on the training loss. ZH and ES stop at 1B, so they never pair against the reference; L1, L2 and L100 have one pair, so their per-task DA is 0 or 1; L15 has no 1.7B cell yet.

| L | variants | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| 1 | L1-deep, L1-shallow | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 |
| 8 | L8-deep, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow | 6 / 0/0/3 | 6 / 0/0/3 | 6 / 0/0/3 | 6 / 0/0/0 |
| 15 | L15-AT3-deep, L15-AT3-shallow, L15-deep, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow | 15 / 0/0/0 | 15 / 0/0/0 | 15 / 0/0/0 | 15 / 0/0/0 |
| 30 | L30-AT3-deep, L30-AT3-shallow, L30-deep, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow | 15 / 0/3/3 | 15 / 0/3/3 | 15 / 0/3/3 | 15 / 0/3/3 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 0/3/3 | 6 / 0/3/3 | 6 / 0/3/3 | 6 / 0/3/3 |

The early-and-small reading one L at a time: pairs of design variants that share the L (seed 1904 of every scheme, `predictivity_all`), against the 1.7B final ranking, on the ten evaluated checkpoints of every run; a cell needs ≥ 3 pairs (rq02's rule), which today leaves out every L with one pair (the table above); the first panel pools every pair of schemes A and B (the `predictivity` pool's, `da_pooled_per_task.csv`). `da_by_L_per_task.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity`.

![Early and small per L](pretraining/predictivity/early_small_by_L.png)
<!-- END auto:by-L -->

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Every parent task as the proxy for every other one (381 x 381): the cell is the smallest proxy size (DA-size, 16 variants at 1.7B, 120 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 513 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_size_by_family.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_ckpt_by_family.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_size_by_language.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_ckpt_by_language.png)

Full task-level maps: [`cross_task_size.png`](pretraining/predictivity/cross_task_size.png), [`cross_task_ckpt.png`](pretraining/predictivity/cross_task_ckpt.png).
<!-- END auto:cross-task -->
