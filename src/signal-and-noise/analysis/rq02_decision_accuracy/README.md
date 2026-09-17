# RQ2 — Does a benchmark rank models at a small size the way the reference size does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**) and across training (**DA-ckpt**)?

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 175M → 1B 0.60 / 0.77; 350M → 1B 0.68 / 0.84; 600M → 1B 0.72 / 0.86.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 175M 80 % (0.86).
<!-- END auto:highlight -->

## Experimental setup

Models are the predictivity ladder's cells (`configs/models.json` pool
`predictivity`: 90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50, 100} × deep/shallow ×
scheme A/B, seed 1904; `predictivity_seeds` adds the seed replicates as
separate models). The cross-size identity is the cell's `family`
(`lm-L8-schemeB-deep-seed1904`: everything but the size), so a design variant
present at two sizes is one pair.

- **DA-size** — `decision_acc_size_<small>`: the families' ranking at
  `<small>`'s final checkpoint vs at the target size's (1B) final checkpoint;
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
| 175M → 1B | 0.60 | 61 | 15 | 0.77 | 101 |
| 350M → 1B | 0.68 | 79 | 21 | 0.84 | 101 |
| 600M → 1B | 0.72 | 105 | 28 | 0.86 | 101 |

![DA-size by family](pretraining/predictivity/da_size_by_family.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 20 % | 40 % | 60 % | 80 % |
|---|---|---|---|---|
| 175M | 0.64 | 0.75 | 0.76 | 0.86 |
| 350M | 0.66 | 0.70 | 0.76 | 0.81 |
| 600M | 0.70 | 0.74 | 0.75 | 0.79 |
| 1B | 0.71 | 0.72 | 0.75 | 0.80 |
| 1.7B | 0.73 | 0.75 | 0.77 | 0.80 |
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
