# Design decisions: which proxy sizes rank a design decision like the reference, and how does that depend on the number of languages?

## Research question

> At a given number of languages L, when does a small model rank a design
> choice (depth, data scheme, temperature, activation, optimizer) the way the 1.7B reference
> (TARGET_SIZE, rule 9) does, and how does the answer change with L?
> This is the predictivity question of
> [`plan/small-to-large-predictivity-training-plan.md`](../../../../plan/small-to-large-predictivity-training-plan.md).
> The gate, scaling, decision-accuracy and noise analyses ask which *benchmarks*
> carry a reliable signal. This one asks which *model sizes* do and, with
> `early_decision.py`, how early in the proxy's run.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

- **Depth (deep vs shallow) on per-language BPB.** Smallest proxy from which every larger proxy keeps DA ≥ 0.75 against the reference: L8: 1B, L15: 1B, L30: 1B, L50: 1B.
- **Data scheme (A vs B) on per-language BPB.** Smallest proxy from which every larger proxy keeps DA ≥ 0.75 against the reference: L8: 90M, L15: 90M, L30: —.
- **Temperature (T=1 vs T=3) on per-language BPB.** Smallest proxy from which every larger proxy keeps DA ≥ 0.75 against the reference: L15: 90M, L30: 90M, L50: 90M.
- **Activation (xIELU vs SwiGLU) on per-language BPB.** Smallest proxy from which every larger proxy keeps DA ≥ 0.75 against the reference: L8: 90M, L15: 1B, L30: 1B.
- **Optimizer (AdEMAMix vs Muon) on per-language BPB.** Smallest proxy from which every larger proxy keeps DA ≥ 0.75 against the reference: L8: —, L15: —, L30: —.
- **Depth decision on benchmarks.** Mean DA over L by proxy: 90M 0.55, 175M 0.52, 350M 0.53, 600M 0.48, 1B 0.49. Kept and flagged (size_matched = False): 600M shallow is not size-matched to 600M deep: +3.7 % non-embedding parameters (616.6M vs 594.5M), width 2048 vs 1536 and about +15 % compute.
- **Is there a decision to make?** Median |Δ| at the reference, in seed sds. Depth (deep vs shallow): benchmarks 1.2×, bits per byte 1.3×. Data scheme (A vs B, B at L8–L30): benchmarks 1.1×, bits per byte 1.6×. Data scheme (A vs B, ZH at L2): benchmarks 1.1×. Data scheme (A vs B, DCLMP at L1): benchmarks 1.1×. Data scheme (A vs C, ES at L2): benchmarks 1.3×. Data scheme (A vs C, FWEB at L1): benchmarks 2.1×. Temperature (T=1 vs T=3): benchmarks 1.4×, bits per byte 5.4×. Activation (xIELU vs SwiGLU): benchmarks 1.1×, bits per byte 1.1×. Optimizer (AdEMAMix vs Muon): benchmarks 1.1×, bits per byte 0.3×.
<!-- END auto:highlight -->

## Experimental setup

**Setup.**
- **Measured.** We measure decision accuracy on six single-axis design decisions, each against the baseline (deep, scheme A, T = 1): depth (deep vs shallow at one nominal size, size-matched except at 600M, where shallow has 3.7 % more non-embedding parameters), data scheme A vs B, data scheme A vs C, sampling temperature T = 1 vs T = 3, activation (xIELU vs SwiGLU) and optimizer (AdEMAMix vs Muon). The optimizer decision has no 1.7B evaluation yet, so it has no DA-size, DA-ckpt or effect at the reference on BPB or benchmarks.
- **Models.** We read each decision on the seed-1904 runs of every ladder and data build (the replicate seeds only set the seed noise of the effect sizes). We read each decision per language setting L and then average over L.
- **Tasks.** We read each decision on the per-language BPB of the languages that both levels train, and on the benchmark tasks: the accuracy tasks above chance at both the proxy and 1.7B, and their bBPB twins, which have no chance level and are about half to four fifths of the benchmark items.
- **How.** DA-size compares the final checkpoint of each proxy, 90M–1B, with the final checkpoint of the 1.7B run. DA-ckpt compares the earlier checkpoints of the 1.7B run with its final checkpoint.

The grid is the predictivity ladder: sizes 90M–1.7B (non-embedding), language
settings L ∈ {1, 2, 8, 15, 30, 50}, and six interventions on the design axes
([RULES.md](../RULES.md), Definitions). Each intervention is read against the
baseline, the deep, scheme-A, T = 1 cell (the data-A cell):

- **Depth.** Deep (width/depth ≈ 64) against shallow (≈ 128), at equal
  non-embedding size.
- **Data scheme.** A vs B and A vs C at T = 1. A letter names a different recipe
  at each L, so the decision is read per L. B is the diversity-first lists at
  L8–L30, ZH at L2 and DCLMP at L1. C is ES at L2 and FWEB at L1. A figure or
  table that averages over L draws one scheme decision per build
  (`analyze.by_recipe`), and the early-decision heat map keeps the planned list
  decision (A vs B at L8–L30).
- **Temperature.** T = 1 vs T = 3 at scheme A (AT3 at L15, L30, L50).
- **Activation and optimizer.** The two model-recipe decisions, at scheme A,
  T = 1, L8–L30: activation (the `deep` ladder, xIELU, vs `swiglu`) and
  optimizer (`deep`, AdEMAMix, vs `muon`).

Each intervention is checked to be one of rq02's mono-axis pairs
(`analyze.mono_axis` reads `utils.moved_axes`). The activation and optimizer
pairs are therefore exactly the ones that rq02's per-axis tables count.

The data interventions are read on the single data-scheme axis (scheme × T) of
the 2026-10-06 refresh. The activation and optimizer decisions were added on
2026-10-10, on the ladder report of 2026-10-10 04:31. The swiglu ladder is
complete at 90M–1.7B. The muon ladder is fully evaluated at 90M–600M only. Its 1B
runs and its 1.7B runs at L15 and L30 have finished training, so the training
loss reaches 1.7B there, but their evaluations had not landed, and the L8 1.7B
run has not finished. Every optimizer cell on BPB and benchmarks therefore
stays "not yet" until a refresh brings the 1.7B results (rule 9: a cell is
never filled from a smaller size).

The pool is `predictivity_seeds` (every seed, ladder and data build), gated with
`predictivity`'s mask. Every read uses each cell's final checkpoint
(D = 100·N tokens, WSD-annealed) unless it says otherwise. The reference at each
L is TARGET_SIZE, 1.7B (rule 9). 19 of the 22 (intervention, L) settings have
it at both levels, and the three optimizer settings wait for the muon 1.7B
evaluations (`analyze.pending_references`). The proxies are 90M, 175M, 350M,
600M and 1B.

The diverged batch-504 90M and 175M runs (see
[`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md)) are dropped
at load in favour of their batch-84 / batch-168 retrains (rule 10). The loader
also excludes the runs that have not reached their target.

Each decision is read on three populations:

- **`bpb_trained`.** The per-language BPB of the languages that both levels
  train, the plan's primary outcome.
- **`benchmark`.** The benchmark tasks the cell was evaluated on, gated at the
  proxy and at the cell's reference (rule 1).
- **`bpb_macro`.** The single macro-BPB decision, the "aggregate criterion"
  that the plan asks to compare against the per-language one.

## Key figure

![Design decisions read by proxy size and by the reference's checkpoints](pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.png)

Population: pool `predictivity_seeds`, read on its seed-1904 runs (every ladder and data build; the other seeds enter only the seed sd), with the benchmarks gated by `predictivity`'s mask and the bBPB twins (never gated) among the benchmark items. Left: DA-size of the proxies 90M–1B at their final checkpoint. Right: DA-ckpt of the reference's own nine earlier checkpoints, 0.5C–4.5C. Both panels read the mono-axis decisions of the six interventions, one scheme decision per data build (the optimizer has no line yet, since it has no 1.7B evaluation). Each line is read against one reference size and the L's that share it, then averaged over L. Solid lines: per-language BPB of the languages both levels train. Dashed lines: the gated benchmark tasks. Dotted line: 0.75.

**Key finding (ladder report 2026-10-10 04:31).** **Per-language BPB reads the temperature and list decisions at every proxy size, but the gated benchmarks almost never do.** On the BPB of the trained languages, temperature is read at 0.95–0.99 (mean over L15–L50) and the language lists at 0.77–1.00 (L8–L30). On the benchmarks every intervention sits at 0.45–0.63, and only the two L1 English-data swaps come near 0.75 (DCLMP 0.52–0.79, with 0.79 at 350M, and FWEB 0.57–0.75, with 0.75 at 600M).

Activation (added 2026-10-10) behaves like depth. On BPB it is read at 0.90–0.98 at 90M, 175M, 350M and 1B but at 0.34 at 600M, and on the benchmarks at 0.48–0.56. The optimizer line waits for the muon 1.7B evaluations.

The depth line on per-language BPB swings from 0.00 (600M, whose depth pairs are not size-matched) to 0.98 (175M), and the reference's own depth effect is only 1.3 seed sds. On the benchmarks, the reference's own checkpoints reach 0.74–0.85 at 90 % of its run. Before 80 % of the run, only the L1 FWEB swap reaches 0.75.

The 600M point of every depth line compares cells that are not size-matched: shallow 600M has 3.7 % more non-embedding parameters, width 2048 against 1536 and about 15 % more compute ([RULES.md, "Size matching"](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)). This is the likeliest explanation of its 0.00. The pairs stay in, and the tables flag them with `size_matched`.

GitHub: [da_all_lines_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.png) · [da_all_lines_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.csv). The same on the decided items and per benchmark: [Per benchmark and per language](#per-benchmark-and-per-language).

## Methodology

- **Intervention decision accuracy.** For each item of a population, the
  decision is which level of the intervention is better. DA(proxy, L) is the
  fraction of items on which the proxy agrees with the reference.
  With two levels, the pairwise-ranking definition of Heineman et al. (2025)
  reduces to sign agreement between the two models of one item, which is
  `snr.metrics.decision_acc_fast` per item. Items that the reference ties are
  dropped rather than counted as misses, because they leave no decision to
  agree with. This is the one place where this analysis departs from the kernel.
- **Items per cell.** `n_items` is reported with every cell, and a benchmark
  cell needs ≥ 3 items. At the final checkpoint a benchmark cell holds 14–1,345
  gated tasks (14–133 at L1, 1,119–1,345 at L50), and a `bpb_trained` cell
  5–50 languages (5, 6 and 26 for the language lists at L8, L15, L30).
- **Effect at the reference.** Per intervention and L, the median |Δ| at
  the reference in per-task seed standard deviations (the seed sd of the
  baseline deep data-A cells with replicates). It answers "is there a decision
  to make?". Two other reads that this folder used to carry now live with their
  themes: the scaling-law error in the scaling analysis (`scaling_law_error.py`),
  and the effect against seed and checkpoint noise per cell in the noise-and-SNR
  analysis (`effect_vs_noise.py`).

Hand-written numbers in this README come from the ladder report **2026-10-10
04:31** (acc_norm on the cloze-format originals) and from the outputs of the
2026-10-10 refresh (detrended checkpoint noise since 2026-10-08). Every number
was re-read against that refresh's tables on 2026-10-10. The bBPB twins exist at
every checkpoint of every size since the 2026-10-07 refresh (288 tasks at 90M,
L50, before it, against 1,108 now), and for the seed replicates too since
2026-10-08. Every proxy's benchmark cell therefore holds the same task set.

<!-- BEGIN auto:results (analyze.py --pool predictivity_seeds) -->
## Results

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq05_design_decisions/analyze.py --pool predictivity_seeds`.

**depth (deep vs shallow), per-language BPB** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L8 | L15 | L30 | L50 |
|---|---|---|---|---|
| 90M | 1.00 | 1.00 | 0.97 | 0.92 |
| 175M | 1.00 | 1.00 | 1.00 | 0.94 |
| 350M | 0.25 | 0.00 | 0.97 | 0.74 |
| 600M | 0.00 | 0.00 | 0.00 | 0.00 |
| 1B | 1.00 | 1.00 | 1.00 | 0.78 |

**data scheme (A vs B), per-language BPB** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 1.00 | 1.00 | 1.00 |
| 175M | 1.00 | 1.00 | 1.00 |
| 350M | 1.00 | 1.00 | 1.00 |
| 600M | 1.00 | 0.83 | 1.00 |
| 1B | 1.00 | 1.00 | 0.31 |

**temperature (T=1 vs T=3), per-language BPB** (rows: proxy size, columns: L, reference L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 90M | 0.87 | 1.00 | 1.00 |
| 175M | 1.00 | 1.00 | 0.98 |
| 350M | 0.87 | 1.00 | 0.98 |
| 600M | 1.00 | 0.97 | 0.98 |
| 1B | 0.93 | 1.00 | 0.98 |

**activation (xIELU vs SwiGLU), per-language BPB** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 1.00 | 1.00 | 0.87 |
| 175M | 1.00 | 0.87 | 0.83 |
| 350M | 1.00 | 1.00 | 0.93 |
| 600M | 1.00 | 0.00 | 0.03 |
| 1B | 1.00 | 1.00 | 0.90 |

**optimizer (AdEMAMix vs Muon), per-language BPB** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 0.00 | 0.07 | 0.63 |
| 175M | 0.00 | 0.07 | 0.50 |
| 350M | 0.00 | 0.07 | 0.50 |
| 600M | 0.00 | 0.07 | 0.50 |
| 1B | 0.00 | 0.07 | 0.50 |

**depth (deep vs shallow), benchmarks** (rows: proxy size, columns: L, reference L1 → 1.7B, L2 → 1.7B, L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L1 | L2 | L8 | L15 | L30 | L50 |
|---|---|---|---|---|---|---|
| 90M | 0.53 | 0.54 | 0.54 | 0.58 | 0.54 | 0.55 |
| 175M | 0.53 | 0.58 | 0.53 | 0.48 | 0.56 | 0.46 |
| 350M | 0.60 | 0.55 | 0.48 | 0.56 | 0.53 | 0.50 |
| 600M | 0.49 | 0.45 | 0.47 | 0.49 | 0.54 | 0.42 |
| 1B | 0.49 | 0.44 | 0.46 | 0.49 | 0.57 | 0.50 |

**data scheme (A vs B), benchmarks** (rows: proxy size, columns: L, reference L1 → 1.7B, L2 → 1.7B, L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L1 | L2 | L8 | L15 | L30 |
|---|---|---|---|---|---|
| 90M | 0.59 | 0.47 | 0.42 | 0.47 | 0.49 |
| 175M | 0.59 | 0.48 | 0.55 | 0.49 | 0.41 |
| 350M | 0.62 | 0.51 | 0.45 | 0.49 | 0.42 |
| 600M | 0.53 | 0.57 | 0.48 | 0.46 | 0.49 |
| 1B | 0.51 | 0.47 | 0.47 | 0.54 | 0.52 |

**data scheme (A vs C), benchmarks** (rows: proxy size, columns: L, reference L1 → 1.7B, L2 → 1.7B):

| proxy | L1 | L2 |
|---|---|---|
| 90M | 0.58 | 0.48 |
| 175M | 0.69 | 0.53 |
| 350M | 0.67 | 0.52 |
| 600M | 0.76 | 0.48 |
| 1B | 0.75 | 0.45 |

**temperature (T=1 vs T=3), benchmarks** (rows: proxy size, columns: L, reference L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 90M | 0.57 | 0.62 | 0.72 |
| 175M | 0.58 | 0.61 | 0.67 |
| 350M | 0.59 | 0.61 | 0.67 |
| 600M | 0.60 | 0.60 | 0.66 |
| 1B | 0.56 | 0.63 | 0.64 |

**activation (xIELU vs SwiGLU), benchmarks** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 0.51 | 0.55 | 0.53 |
| 175M | 0.50 | 0.50 | 0.53 |
| 350M | 0.55 | 0.53 | 0.60 |
| 600M | 0.49 | 0.49 | 0.50 |
| 1B | 0.51 | 0.51 | 0.57 |

**optimizer (AdEMAMix vs Muon), benchmarks** (rows: proxy size, columns: L, reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 0.45 | 0.62 | 0.49 |
| 175M | 0.49 | 0.43 | 0.46 |
| 350M | 0.60 | 0.59 | 0.47 |
| 600M | 0.53 | 0.54 | 0.43 |
| 1B | 0.51 | 0.53 | 0.52 |

![Intervention DA grid](pretraining/predictivity_seeds/intervention_da_all_mono_axis.png)

**Effect at the reference in seed standard deviations** (median over items and L, one scheme decision per recipe, since a letter names a different build at each L):

| intervention | benchmarks | bits per byte |
|---|---|---|
| depth (deep vs shallow) | 1.2 | 1.3 |
| data scheme (A vs B, B at L8–L30) | 1.1 | 1.6 |
| data scheme (A vs B, ZH at L2) | 1.1 |  |
| data scheme (A vs B, DCLMP at L1) | 1.1 |  |
| data scheme (A vs C, ES at L2) | 1.3 |  |
| data scheme (A vs C, FWEB at L1) | 2.1 |  |
| temperature (T=1 vs T=3) | 1.4 | 5.4 |
| activation (xIELU vs SwiGLU) | 1.1 | 1.1 |
| optimizer (AdEMAMix vs Muon) | 1.1 | 0.3 |

![Interventions](pretraining/predictivity_seeds/rq4_interventions.png)
<!-- END auto:results -->

GitHub: [intervention_da_all_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_all_mono_axis.png) · [intervention_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_all_mono_axis.csv) ·
GitHub: [rq4_interventions.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_interventions.png) · [rq4_interventions.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_interventions.csv) ·
[rq4_effect_vs_seed.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_effect_vs_seed.csv) ·
[rq4_da_size_by_intervention_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_da_size_by_intervention_mono_axis.csv)

**Key findings** (`intervention_da_all_mono_axis`, `rq4_effect_vs_seed`: pool
`predictivity_seeds`, proxies 90M–1B at their final checkpoint, reference
1.7B at every (intervention, L), DA-size per L)

- **Per-language BPB reads temperature and the language lists in almost every
  cell.** Every proxy reads the temperature decision at 0.87–1.00 at every L,
  and the language-list decision at 0.83–1.00 in every cell except 1B at L30 (0.31).
- **The depth decision on per-language BPB collapses at 350M and 600M.** It is
  0.00 for 600M at all four L (the 600M depth pairs are not size-matched) and
  0.00–0.97 for 350M, while 90M, 175M and 1B read it at 0.78–1.00. The depth
  bullet under the per-benchmark figures explains why.
- **On the gated benchmarks only three cells, all at L1, reach 0.75.** The per-L
  cells span 0.40–0.79. The three at ≥ 0.75 are 350M at L1 for depth and for
  A vs DCLMP (0.79 each, on 14 tasks) and 600M at L1 for A vs FWEB (0.75, on 133 tasks).
- **Only temperature on BPB is far outside seed noise at the reference.** On
  the benchmarks every intervention's effect is 1.1–2.1 seed sds. On
  per-language BPB it is 1.1 for activation, 1.3 for depth, 1.6 for the
  language lists and 5.4 for temperature.
- **Activation (xIELU vs SwiGLU) is read on BPB except at 600M.** 90M, 175M,
  350M and 1B read it at 0.83–1.00 at every L. 600M reads it at 1.00 (L8),
  0.00 (L15) and 0.03 (L30), the same 600M dip as for depth. On the benchmarks
  it sits at 0.45–0.58. Its effect at the reference is 1.1 seed sds on BPB and
  1.2 on the benchmarks, as small as depth's.
- **Optimizer (AdEMAMix vs Muon): not yet.** The muon 1B cells have no
  evaluation in this report and the 1.7B cells only their early checkpoints
  (benchmarks up to 16,200 of 81,000 iterations), so the BPB and benchmark cells are empty
  (L8, L15, L30). The one quantity with a 1.7B value is the training loss
  (`loss` population, one item per L, in `intervention_da_all_mono_axis.csv`).
  At L15 and L30 the 1.7B AdEMAMix run ends at the lower loss, while every
  proxy from 90M to 600M ends lower with Muon (DA 0.00). 1B agrees with the
  reference at L15 only. This is a one-item read at two L, so it is a lead to
  check after the refresh, not a finding.

**Follow-ups**

- Bootstrap the per-L benchmark DA over tasks. The L1 cells hold 14–133
  tasks, so the three ≥ 0.75 cells there need an interval before they are
  quoted.
- Re-read the optimizer rows after the refresh that lands the muon 1B and
  1.7B evaluations. Until the bBPB store is rebuilt, the muon cells carry no
  bBPB twins (306–639 benchmark tasks against deep's 607–1,329), so their
  benchmark cells will hold fewer tasks than those of the other decisions.
- Check whether the 600M swiglu cell is size-matched to 600M deep, as for
  depth (RULES.md, "Size matching"). The 600M activation dip mirrors depth's.

## How small, and how early

### Setup

Everything comes from the decision table above. With two levels, decision
accuracy is the share of population items on which the proxy prefers the level
that the reference prefers. The reference is TARGET_SIZE (1.7B, rule 9) at its
final checkpoint. An (intervention, L) without a 1.7B cell at both levels is
skipped: in this refresh, the three optimizer settings. The proxy is every
smaller size, read at its ten evaluated checkpoints, 0.5C to 5C of training
(C is the Chinchilla-optimal 20 tokens per parameter, and 5C is the full run).
The 1.7B row compares the reference's own checkpoints with its final ranking
(DA-ckpt).

One grid answers both halves of the question: how small and how early. It has
two populations, and each cell needs at least three items: the per-language BPB
of the languages both levels train, and the gated benchmark tasks both levels
were evaluated on.

### Methodology

The per-(intervention, L, population, proxy size, fraction) rows of the four
decisions of the heat map (depth, the A vs B language lists, activation and
optimizer) are averaged over L, so that every language setting counts once.
`cells` in the table is the number of settings behind a mean: for depth 4 on
BPB (L8–L50) and 6 on benchmarks (L1–L50), and 3 for the lists and for
activation (L8–L30). The optimizer has a loss row only (2 settings) until its
1.7B evaluations land. The reference, 1.7B for all of them, is carried in `refs`.

<!-- BEGIN auto:early-decision (early_decision.py --pool predictivity_seeds) -->
## How small, and how early (paper RQ2)

Numbers from the `predictivity_seeds` decision table above. Regenerate with `python analysis/rq05_design_decisions/early_decision.py --pool predictivity_seeds`.

- **Depth (deep vs shallow), per-language bits per byte.** Final-checkpoint agreement by proxy: 90M 0.97, 175M 0.98, 350M 0.49, 600M 0.00, 1B 0.95. The smallest proxy from which every larger proxy stays at ≥ 0.75 is **1B**, which reaches it at 2C of training (5C = the full run).
- **Depth (deep vs shallow), benchmark tasks.** Final-checkpoint agreement by proxy: 90M 0.55, 175M 0.52, 350M 0.53, 600M 0.48, 1B 0.49. No proxy reaches 0.75.
- **Data scheme (A vs B, B at L8–L30), per-language bits per byte.** Final-checkpoint agreement by proxy: 90M 1.00, 175M 1.00, 350M 1.00, 600M 0.94, 1B 0.77. The smallest proxy from which every larger proxy stays at ≥ 0.75 is **90M**, which reaches it at 0.5C of training (5C = the full run).
- **Data scheme (A vs B, B at L8–L30), benchmark tasks.** Final-checkpoint agreement by proxy: 90M 0.46, 175M 0.48, 350M 0.45, 600M 0.48, 1B 0.51. No proxy reaches 0.75.
- **Activation (xIELU vs SwiGLU), per-language bits per byte.** Final-checkpoint agreement by proxy: 90M 0.96, 175M 0.90, 350M 0.98, 600M 0.34, 1B 0.97. The smallest proxy from which every larger proxy stays at ≥ 0.75 is **1B**, which reaches it at 2.5C of training (5C = the full run).
- **Activation (xIELU vs SwiGLU), benchmark tasks.** Final-checkpoint agreement by proxy: 90M 0.53, 175M 0.51, 350M 0.56, 600M 0.49, 1B 0.53. No proxy reaches 0.75.
- **Optimizer (AdEMAMix vs Muon), per-language bits per byte.** Final-checkpoint agreement by proxy: 90M 0.23, 175M 0.19, 350M 0.19, 600M 0.19, 1B 0.19. No proxy reaches 0.75.
- **Optimizer (AdEMAMix vs Muon), benchmark tasks.** Final-checkpoint agreement by proxy: 90M 0.52, 175M 0.46, 350M 0.55, 600M 0.50, 1B 0.52. No proxy reaches 0.75.

**depth (deep vs shallow), per-language bits per byte** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.97 | 0.97 | 0.98 | 0.94 | 0.93 | 0.96 | 0.95 | 0.93 | 0.96 | 0.97 |
| 175M | 0.94 | 0.86 | 0.90 | 0.98 | 0.93 | 0.95 | 0.92 | 0.93 | 0.96 | 0.98 |
| 350M | 0.25 | 0.26 | 0.20 | 0.17 | 0.36 | 0.45 | 0.29 | 0.31 | 0.44 | 0.49 |
| 600M | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| 1B | 0.27 | 0.28 | 0.54 | 0.83 | 0.85 | 0.87 | 0.90 | 0.92 | 0.94 | 0.95 |
| 1.7B | 0.69 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |  |

**depth (deep vs shallow), benchmark tasks** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.54 | 0.52 | 0.52 | 0.49 | 0.51 | 0.52 | 0.54 | 0.52 | 0.55 | 0.55 |
| 175M | 0.53 | 0.51 | 0.49 | 0.51 | 0.50 | 0.53 | 0.54 | 0.49 | 0.52 | 0.52 |
| 350M | 0.50 | 0.47 | 0.46 | 0.48 | 0.49 | 0.52 | 0.51 | 0.51 | 0.51 | 0.53 |
| 600M | 0.46 | 0.46 | 0.45 | 0.47 | 0.49 | 0.48 | 0.47 | 0.47 | 0.47 | 0.48 |
| 1B | 0.47 | 0.49 | 0.47 | 0.48 | 0.48 | 0.50 | 0.52 | 0.54 | 0.50 | 0.49 |
| 1.7B | 0.51 | 0.48 | 0.53 | 0.56 | 0.60 | 0.61 | 0.63 | 0.65 | 0.76 |  |

**data scheme (A vs B, B at L8–L30), per-language bits per byte** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| 175M | 0.96 | 0.99 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 350M | 0.86 | 0.88 | 0.94 | 0.94 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| 600M | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.87 | 0.94 | 0.88 | 0.94 | 0.94 |
| 1B | 0.79 | 0.91 | 0.87 | 0.83 | 0.83 | 0.83 | 0.81 | 0.72 | 0.77 | 0.77 |
| 1.7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |  |

**data scheme (A vs B, B at L8–L30), benchmark tasks** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.49 | 0.51 | 0.51 | 0.46 | 0.50 | 0.50 | 0.47 | 0.45 | 0.47 | 0.46 |
| 175M | 0.49 | 0.52 | 0.47 | 0.50 | 0.50 | 0.41 | 0.48 | 0.45 | 0.46 | 0.48 |
| 350M | 0.48 | 0.45 | 0.45 | 0.47 | 0.44 | 0.46 | 0.47 | 0.45 | 0.47 | 0.45 |
| 600M | 0.49 | 0.48 | 0.52 | 0.49 | 0.48 | 0.50 | 0.50 | 0.48 | 0.49 | 0.48 |
| 1B | 0.51 | 0.48 | 0.51 | 0.51 | 0.48 | 0.48 | 0.48 | 0.52 | 0.52 | 0.51 |
| 1.7B | 0.48 | 0.51 | 0.53 | 0.61 | 0.59 | 0.62 | 0.61 | 0.62 | 0.76 |  |

**activation (xIELU vs SwiGLU), per-language bits per byte** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.02 | 0.40 | 0.57 | 0.72 | 0.86 | 0.86 | 0.90 | 0.93 | 0.94 | 0.96 |
| 175M | 0.02 | 0.22 | 0.37 | 0.53 | 0.68 | 0.64 | 0.81 | 0.91 | 0.92 | 0.90 |
| 350M | 0.00 | 0.01 | 0.40 | 0.58 | 0.96 | 0.94 | 0.97 | 0.99 | 0.98 | 0.98 |
| 600M | 0.00 | 0.00 | 0.25 | 0.25 | 0.25 | 0.29 | 0.33 | 0.29 | 0.33 | 0.34 |
| 1B | 0.00 | 0.04 | 0.60 | 0.71 | 0.79 | 0.86 | 0.88 | 0.94 | 0.96 | 0.97 |
| 1.7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |  |

**activation (xIELU vs SwiGLU), benchmark tasks** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.50 | 0.52 | 0.52 | 0.49 | 0.51 | 0.50 | 0.52 | 0.55 | 0.53 | 0.53 |
| 175M | 0.52 | 0.51 | 0.51 | 0.46 | 0.49 | 0.50 | 0.54 | 0.50 | 0.50 | 0.51 |
| 350M | 0.51 | 0.51 | 0.52 | 0.54 | 0.53 | 0.55 | 0.55 | 0.57 | 0.55 | 0.56 |
| 600M | 0.47 | 0.46 | 0.48 | 0.48 | 0.46 | 0.49 | 0.50 | 0.48 | 0.50 | 0.49 |
| 1B | 0.43 | 0.46 | 0.53 | 0.46 | 0.47 | 0.49 | 0.52 | 0.50 | 0.53 | 0.53 |
| 1.7B | 0.54 | 0.53 | 0.56 | 0.56 | 0.59 | 0.61 | 0.61 | 0.64 | 0.76 |  |

**optimizer (AdEMAMix vs Muon), per-language bits per byte** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.42 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.23 |
| 175M | 0.23 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.19 |
| 350M | 0.19 | 0.18 | 0.21 | 0.58 | 0.81 | 0.81 | 0.81 | 0.81 | 0.20 | 0.19 |
| 600M | 0.19 | 0.47 | 0.58 | 0.79 | 0.81 | 0.81 | 0.81 | 0.81 | 0.22 | 0.19 |
| 1B | 0.20 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.19 |
| 1.7B | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 |  |

**optimizer (AdEMAMix vs Muon), benchmark tasks** (rows: proxy size, columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run, values: mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.50 | 0.51 | 0.46 | 0.44 | 0.48 | 0.45 | 0.46 | 0.48 | 0.52 | 0.52 |
| 175M | 0.48 | 0.49 | 0.48 | 0.46 | 0.42 | 0.45 | 0.46 | 0.42 | 0.42 | 0.46 |
| 350M | 0.53 | 0.51 | 0.48 | 0.49 | 0.45 | 0.48 | 0.47 | 0.52 | 0.57 | 0.55 |
| 600M | 0.53 | 0.49 | 0.46 | 0.43 | 0.44 | 0.51 | 0.48 | 0.43 | 0.49 | 0.50 |
| 1B | 0.46 | 0.46 | 0.51 | 0.46 | 0.46 | 0.48 | 0.46 | 0.45 | 0.49 | 0.52 |
| 1.7B | 0.50 | 0.48 | 0.50 | 0.47 | 0.55 | 0.50 | 0.56 | 0.57 | 0.73 |  |

![Early and small](pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.png)
<!-- END auto:early-decision -->

GitHub: [rq2_da_goal_early_small_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.png) · [rq2_da_goal_early_small_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.csv) ·
[rq2_da_all_decisions_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_all_decisions_mono_axis.csv) ·
[early_decision_facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/early_decision_facts.json)

**Key findings** (`rq2_da_goal_early_small_mono_axis`: pool `predictivity_seeds`,
proxies 90M–1B at ten checkpoints, 0.5C–5C, against the 1.7B final ranking,
mean over L, and the 1.7B row is DA-ckpt)

- **On per-language BPB, the smallest proxy is enough, and so is an early
  checkpoint.** 90M reads the depth decision at 0.93–0.98 and the list decision
  at 0.96–1.00 from 0.5C on.
- **Larger proxies are not safer on BPB.** The 1B depth read climbs from 0.27
  at 0.5C to 0.95 at 5C, and the 1B list read falls from 0.91 at 1C to
  0.72–0.77 from 4C on. 600M reads depth at 0.00–0.19 throughout (its depth
  pairs are not size-matched).
- **On the gated benchmarks no proxy reaches 0.75 at any checkpoint.** The
  proxies read 0.41–0.57 over both decisions, and the reference's own
  checkpoints sit at 0.48–0.64 until they reach 0.76 at 4.5C.
- **Activation (added 2026-10-10) needs a later checkpoint than depth.** On BPB,
  90M first reaches 0.75 at 2.5C, where depth and the lists reach it at 0.5C.
  The final reads are 0.90–0.98, except at 600M (0.34). On the benchmarks the
  proxies stay at 0.43–0.56, like the other decisions, and on BPB the
  reference's own checkpoints read it at 0.99–1.00 from 0.5C on.
- **Optimizer: not yet.** It has no 1.7B evaluation, and the bottom row of the
  heat map says so.

**Follow-ups**

- The benchmark rows now carry the bBPB twins at every checkpoint (a depth
  proxy row holds 3,153–3,791 task-cells, against 793–1,735 at 90M–600M
  before). The range stays at 0.41–0.57, so the twins do not rescue the early
  benchmark decision. Split the rows by twin vs native task to check that
  neither half alone reaches 0.75.

## Caveats to carry into the paper

- The depth intervention's effect is small by design, because the aspect
  ratio is near the optimum. The effect-vs-noise table decides whether its DA
  can be interpreted at all. A scheme intervention can change the language set
  (A vs B at L8–L30, A vs ZH/ES at L2), so its `bpb_trained` population is the
  languages both builds train.
- Checkpoint noise windows differ by size unless the shared grid is used
  (the loader's default). Five late checkpoints span the final 25 % of a
  20-checkpoint run but 12.5 % of a 40-checkpoint one
  ([`plan/1b-models.md`](../../../../plan/1b-models.md)).
- The reference is TARGET_SIZE, 1.7B (rule 9), and an (intervention, L) without
  a 1.7B cell at both levels is skipped. In the 2026-10-10 report, 19 of the 22
  (intervention, L) settings have one. The three optimizer settings do not yet:
  `analyze.pending_references` lists them, the script prints a `!!! RULE 9`
  line per setting, and the tables and figures say "not yet". The
  `reference_size` column names the reference per cell.

<!-- BEGIN auto:panels (panels.py --pool predictivity_seeds) -->
## Per benchmark and per language

The decision table and the early read above, without pooling the benchmarks (`predictivity_seeds` pool). Language aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), and a language's subplot averages its BPB item with its benchmark items. Regenerate with `python analysis/rq05_design_decisions/panels.py --pool predictivity_seeds`. White cells have no value. Each figure's table sits next to it under the same name (`intervention_da_size_by_<unit>_mono_axis.csv`).

![rq05 in one figure](pretraining/predictivity_seeds/highlights.png)

![Decisions by proxy size and checkpoint](pretraining/predictivity_seeds/da_all_lines_mono_axis.png)

![The same on the items decided outside seed noise](pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.png)

![Which depth wins, in seed sds](pretraining/predictivity_seeds/depth_crossover.png)

![Decisions by compute](pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.png)

![Decisions per benchmark](pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.png)

![Early and small per benchmark](pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.png)

![Decisions per language](pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.png)

![Early and small per language](pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.png)
<!-- END auto:panels -->

**Key findings** (`da_all_lines_mono_axis`, `da_all_lines_decided_mono_axis`, `depth_crossover`:
DA-size and DA-ckpt on mono-axis decisions, final checkpoint unless stated.
Population, sizes and reference as in the setup above: `predictivity_seeds`,
reference TARGET_SIZE, 1.7B, per (intervention, L) (rule 9), items that the
reference ties dropped)

- **On the shared languages, most decisions are barely decided at the
  reference.** The effect table puts depth at 1.3 seed sds on BPB (per-L
  medians 1.15–1.35) and the language lists at 1.6 (0.59–2.87), against 5.4
  for temperature (4.36–7.53). On the benchmarks every intervention sits at
  1.1–2.1.
- **Noise at the reference explains only part of the benchmarks' failure.**
  `da_all_lines_decided_mono_axis` keeps only the items whose reference |Δ|
  clears 2 sds of the two-run difference. On the 94 gated benchmark cells
  (proxy × intervention × L) that keep ≥ 3 decided tasks, the final-checkpoint
  DA rises from 0.54 to 0.60 (means, with a median of 56.5 decided tasks per
  cell). This is still short of 0.75.
- **The decided figure has no depth BPB line.** On per-language BPB the
  decided cut keeps 10–41 languages per cell for temperature but at most 2 for
  depth and activation. On `bpb_macro` and the loss it leaves one item per L,
  which gives a 0/1 reading.
- **Depth is a vanishing advantage, not a crossover.** In `depth_crossover.csv`
  (the batch-84 / batch-168 retrains), deep beats shallow by 1.9–2.1
  difference sds at 90M and by 1.1–1.6 at 175M. At 350M the gap is |z| ≤ 0.51.
  Shallow is ahead by 0.28–1.46 sds at 600M (whose depth pairs are not
  size-matched), and deep by 0.48–1.03 at 1B and 0.81–0.95 at 1.7B. From 350M
  on, every gap is inside 2 sds. The depth DA on the trained languages' BPB
  (0.92–1.00 at 90M, 0.94–1.00 at 175M, 0.00–0.97 at 350M, 0.00 at 600M,
  0.78–1.00 at 1B over L8–L50) therefore reads a reference that has no real
  preference.
- **Untrained languages are not a population here.** A score on a language
  that only one level trains is measured by the
  [language-transfer analysis](../rq06_language_transfer/README.md) (rule 2),
  and the loader no longer delivers those rows.

**Follow-ups**

- Items are not independent. A language's BPB items move together, so 50
  languages agreeing at L50 is closer to one decision measured 50 times than
  to 50 decisions. Next to the item share, report the number of decisions
  (intervention × L) that agree as a secondary number, and bootstrap over L.
- The reference has been 1.7B on every line since 2026-09-26 (`refs` in the
  table). Keep naming it in the legend.
- The seed sd is the median over the replicated deep data-A cells, applied
  to every size and scheme, on 3 seeds where they exist. The median of the sds
  is biased low by about 17 %. Use the size's own sd where the ×3 cells exist
  at the reference's size, and widen the decided cut to cover its sampling
  error.
- The training loss is one item per L, so its line is a 0/0.5/1 step
  function. Drop it from the paper version. `da_all_lines_flops_mono_axis`
  (21 lines of 59 points) is not readable. Draw one line per size, or use
  per-size markers.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/highlights.csv) ·
GitHub: [da_all_lines_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis.png) · [da_all_lines_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis.csv) ·
GitHub: [da_all_lines_decided_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.png) · [da_all_lines_decided_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.csv) ·
GitHub: [depth_crossover.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/depth_crossover.png) · [depth_crossover.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/depth_crossover.csv) ·
GitHub: [da_all_lines_flops_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.png) · [da_all_lines_flops_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.csv) ·
GitHub: [intervention_da_size_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.png) · [intervention_da_size_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.csv) ·
GitHub: [intervention_da_goal_early_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.png) · [intervention_da_goal_early_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.csv) ·
GitHub: [intervention_da_size_by_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.png) · [intervention_da_size_by_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.csv) ·
GitHub: [intervention_da_goal_early_by_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.png) · [intervention_da_goal_early_by_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.csv)

<!-- BEGIN auto:transformations (transformations.py --pool predictivity_seeds) -->
## Transformations on one item set

Mean decision accuracy over each transformation's pairs, on the items every transformation decides somewhere (benchmarks gated by rq00's above-random mask at the proxy and the reference). `transformation_da_size_mono_axis.csv` has every pair. Kept and flagged (size_matched = False): 600M shallow is not size-matched to 600M deep: +3.7 % non-embedding parameters (616.6M vs 594.5M), width 2048 vs 1536 and about +15 % compute. Regenerate with `python analysis/rq05_design_decisions/transformations.py --pool predictivity_seeds`.

**benchmarks** (rows: transformation, columns: proxy size, values: mean over pairs on the shared items. Pairs = read of planned (with data): a planned pair without a 1.7B result at both cells is not read yet. An empty row means that no pair has items there yet):

| transformation | pairs (with data) | items | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| activation (xIELU vs SwiGLU) | 3 of 3 (3) | 143 | 0.48 | 0.56 | 0.54 | 0.44 | 0.53 |
| data scheme (A vs B) | 5 of 5 (5) | 142 | 0.48 | 0.54 | 0.50 | 0.52 | 0.52 |
| data scheme (A vs C) | 2 of 2 (2) | 142 | 0.53 | 0.61 | 0.60 | 0.62 | 0.60 |
| depth (deep vs shallow) | 6 of 6 (6) | 142 | 0.52 | 0.51 | 0.53 | 0.45 | 0.47 |
| language count (L vs next L) | 5 of 5 (5) | 143 | 0.48 | 0.56 | 0.53 | 0.49 | 0.51 |
| optimizer (AdEMAMix vs Muon) | 3 of 3 (3) | 141 | 0.57 | 0.47 | 0.54 | 0.45 | 0.52 |
| temperature (T=1 vs T=3) | 3 of 3 (3) | 143 | 0.50 | 0.49 | 0.49 | 0.43 | 0.44 |

**per-language BPB (trained languages)** (rows: transformation, columns: proxy size, values: mean over pairs on the shared items. Pairs = read of planned (with data): a planned pair without a 1.7B result at both cells is not read yet. An empty row means that no pair has items there yet):

| transformation | pairs (with data) | items | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| activation (xIELU vs SwiGLU) | 3 of 3 (3) | 26 | 0.97 | 0.92 | 0.99 | 0.35 | 0.97 |
| data scheme (A vs B) | 5 of 5 (3) | 26 | 1.00 | 1.00 | 1.00 | 0.94 | 0.77 |
| depth (deep vs shallow) | 6 of 6 (4) | 26 | 0.99 | 1.00 | 0.49 | 0.00 | 0.96 |
| language count (L vs next L) | 5 of 5 (3) | 26 | 0.80 | 0.79 | 0.74 | 0.81 | 0.64 |
| optimizer (AdEMAMix vs Muon) | 3 of 3 (3) | 26 | 0.24 | 0.19 | 0.19 | 0.19 | 0.19 |
| temperature (T=1 vs T=3) | 3 of 3 (3) | 26 | 0.96 | 0.99 | 0.94 | 0.97 | 0.96 |

Not read: data scheme (A vs C) (no pair has a 1.7B result at both cells, or fewer than 3 items, on these items).

![Transformations](pretraining/predictivity_seeds/transformation_da_size_mono_axis.png)
<!-- END auto:transformations -->

GitHub: [transformation_da_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/transformation_da_size_mono_axis.png) · [transformation_da_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/transformation_da_size_mono_axis.csv)

**Key findings** (`transformation_da_size_mono_axis`: pool `predictivity_seeds`,
DA-size of the proxies 90M–1B against 1.7B, mono-axis pairs, 133–134 gated
benchmark tasks or 26 trained languages shared by every transformation with
data, 2026-10-10 report)

- **On one task set, no proxy reads any transformation reliably.** The means
  span 0.43 (temperature, 600M and 1B) to 0.62 (A vs C, 600M). The
  language-count pairs (L vs next L) sit at 0.48–0.60, like the design
  interventions. Activation (added 2026-10-10) reads 0.45–0.60. The optimizer
  row is not there yet (no muon 1.7B evaluation).
- **On the BPB of 26 shared trained languages, most transformations are read
  well, except at 600M.** Temperature reads 0.94–0.99 and the A vs B lists
  0.77–1.00. Activation reads 0.92–0.99 except at 600M (0.35). Depth reads
  0.96–1.00 at 90M, 175M and 1B but 0.49 at 350M and 0.00 at 600M. The
  language-count pairs read 0.64–0.81. A vs C is not read, because English is
  its only BPB item.

**Follow-ups**

- Read the BPB table against the benchmark table per transformation: on the
  same item set, BPB separates the data decisions that the benchmarks do not.

## The key figure on one task set per line

Each line averages, at each proxy size, the (task, L) items that are above chance at that size. A line can therefore move because its items changed (rule 13). The fixed-task twin keeps, per (intervention, L, population), only the items with a value at every point (solid line), then averages over L as the key figure does. The committed line is drawn dashed behind it. Rows are the populations, and the columns are DA-size and DA-ckpt.

![Design decisions on one task set per line](pretraining/predictivity_seeds/da_all_lines_mono_axis_fixed_tasks_paper.png)

DA-size and DA-ckpt against the 1.7B final, mono-axis pairs, every seed (pool `predictivity_seeds`), gate `predictivity`, no reliability filter. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_fixed_tasks_paper.csv)

Key findings:

- **The BPB lines and the checkpoint panel do not move.** BPB is never gated, and on the checkpoint axis the gate is read at the reference only, so the populations there are already fixed.
- **The temperature benchmark line is flat on both readings.** On the 2,584 items it has at every size, it reads 0.62–0.63 from 90M to 1B, against 0.61–0.63 for the committed line on 2,640–3,058 items. With the bBPB twins now at every size and seed, the jump at 175M of the 2026-10-07 refresh is gone.
- **The depth and scheme-B lines barely move.** They shift by at most 0.04 and 0.01 (3,072 and 1,251 fixed items), so their shape is not a population effect.
- **The activation lines barely move.** The benchmark line shifts by at most 0.01 on 1,205 fixed items, and the BPB line has the same 53 items at every point (the optimizer has no line yet).
- **The two L1 single-build lines rest on very few items.** DCLMP and FWEB keep 14 items, and their fixed lines move by up to 0.19, so they are not readable. ZH and ES keep 124–125 items and move by at most 0.01.

Follow-ups:

- **The per-benchmark grids on fixed tasks.** The per-benchmark and per-language figures have the same moving populations.

## Extensions from other sweeps

None. The design-decision analysis exists on the ladder only (`predictivity_seeds`).
The 36-model sweep had one intervention (three FineWeb-edu mixtures) and no
depth, language-count, list or temperature axis, so no decision table of this
kind was made on it. Its numbers would not be pooled with the ladder's in any
case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/intervention_da_all_mono_axis.csv`: one row per (intervention,
  population, L, proxy size): `decision_acc`, `n_items`, mean |Δ| at proxy and
  reference, the level the reference prefers.
- `…/rq4_da_size_by_intervention_mono_axis.csv`, `rq4_effect_vs_seed.csv`, `rq4_interventions.png`,
  `facts.json`: the paper's design-decision figure and the numbers it quotes.
- `…/rq2_da_all_decisions_mono_axis.csv`, `rq2_da_goal_early_small_mono_axis.csv`, `rq2_da_goal_early_small_mono_axis.png`,
  `early_decision_facts.json`: the paper's how-small-and-how-early figure (`early_decision.py`).
- `pretraining/<pool>/transformation_da_size_mono_axis.csv`, `transformation_da_size_mono_axis.png`: the
  transformations (language count, temperature, depth, data scheme A vs B and A vs C, activation, and the
  optimizer once its 1.7B results land) on one gated item set, so that their predictability can be compared
  (`transformations.py`, the block "Transformations on one item set" above).
- `…/intervention_da_all_mono_axis.png`.
