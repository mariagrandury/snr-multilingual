# RQ1 — Does a benchmark rank models at a small size the way the reference size does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**) and across training (**DA-ckpt**)?

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
never gated: it is the truth the SNR proxies of rq02 are scored against.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_per_task.csv`
(one row per parent task, one column per DA definition) from
`snr.metrics.decision_acc_fast`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. rq02 joins the SNR variants onto this table; rq06 asks the
complementary question — which proxy *size* ranks an intervention like the
reference, with languages as the population.

With few families at a bucket (the 1.7B row has five language settings; a
task in one language may have three), DA is quantised to 1/#pairs: read the
family-level averages, and `n` alongside every value.

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
  for rq02).
- `…/da_per_benchmark.csv`, `da_per_benchmark_size.csv`,
  `da_per_benchmark_ckpt.csv` — long and wide per-(language, benchmark) views.
