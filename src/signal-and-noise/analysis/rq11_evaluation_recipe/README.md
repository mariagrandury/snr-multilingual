# RQ11 — Which benchmark, posed how and scored how, gives a reliable decision?

## Research question

The rest of the analysis asks whether a cheap measurement reproduces the
1.7B reference's design decision. This RQ turns that into advice for a reader
who has to choose what to evaluate: for each benchmark, which **variant** —
the items as published, their `rf_` cloze rewrite (RF) or their `rfgm_` LLM
rewrite (LLM-RF), scored by **accuracy** or by the gold answer's **bits per
byte** (bBPB) — reads the reference's decision from the smallest proxy?
Reliable means DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`, the "safe" cut of
rq02's early-small maps), and every figure, table and sentence here uses that
one τ.

## Experimental setup

Snapshot: the ladder report of 2026-10-04 03:15 (`9f7fde6f`). Pool
`predictivity` (seed 1904, schemes A and B, 90M–1B proxies against the 1.7B
reference), rq02's per-task table `da_goal_early_small_per_task_both_axes.csv`,
both pair sets (multi-axis in the figures below, mono-axis in the `_mono_axis`
twins). Each task is one variant of one benchmark in one language; its
`format` and `scoring` are columns the loader writes (`utils.variant`).

The bBPB twins come from the rq08 per-item store, which holds each run's final
checkpoint only, so for now only DA-size can be compared across all six
variants. When the store covers every checkpoint, the DA-goal cells below
100 % and the safe-compute column fill in on the next refresh without a code
change.

## Methodology

- **Per task**: DA-size at each proxy (rule 1 at the proxy and the reference,
  where a bBPB task, having no chance level, always passes; rule 5's three
  pairs), the **safe size** (the smallest proxy from which DA-size stays ≥ τ
  at every larger proxy, rq02's rule) and the safe compute share.
- **Per benchmark × variant**: the share of its languages reliable at each
  proxy, the mean DA-size and the mean and median safe rank (0 = safe from
  90M, 4 = from 1B, 5 = never).
- **The recommendation**: per benchmark, the variant with the smallest mean
  safe rank, then the higher mean DA-size. The mean is used rather than the
  median because the median is "never" whenever half the languages never get
  there, which ties almost every variant.
- **Together and apart**: every finding is given per variant and for all
  variants pooled. A head-to-head between variants is read on the **paired**
  cells, a (benchmark, language) that also has an original-accuracy value at
  that proxy, because the gate treats bBPB and accuracy differently: bBPB is
  defined where accuracy is at chance. The unpaired reading answers a
  different question, whether a variant reads anything where the original
  cannot.

<!-- BEGIN auto:results (recipe.py --pool predictivity) -->
## Results

DA-size against the 1.7B final, multi-axis pairs (rule 15), pool `predictivity`, gate `predictivity` at the proxy and the reference (a bBPB task has no chance level and passes), ≥ 3 pairs; reliable = DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every larger proxy. 1340 tasks over 30 benchmarks. The bBPB twins are read at final checkpoints only for now; a mean over fewer than 5 tasks is left blank. Regenerate with `python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity`.

**The recommendation** (per benchmark: the variant with the smallest mean safe rank over its languages — 0 = safe from 90M, 4 = from 1B, 5 = never — then the higher mean DA-size; τ = 0.75), best first:

| benchmark | evaluate it as | mean safe rank | median safe size | languages safe by 1B | languages | mean DA-size |
|---|---|---|---|---|---|---|
| XStoryCloze | original · bBPB | 2.57 | 1B | 0.71 | 7 | 0.79 |
| HellaSwag | original · accuracy | 3.26 | 1B | 0.68 | 20 | 0.70 |
| INCLUDE | LLM-RF · bBPB | 4.38 | never | 0.31 | 26 | 0.63 |
| LAMBADA | original · accuracy | 4.60 | never | 0.40 | 5 | 0.65 |
| Belebele | LLM-RF · bBPB | 4.71 | never | 0.19 | 34 | 0.59 |
| CulturalBench-easy | RF · bBPB | 4.74 | never | 0.21 | 8 | 0.55 |
| Global-MMLU | RF · accuracy | 4.83 | never | 0.17 | 25 | 0.54 |
| XCOPA | original · bBPB | 4.86 | never | 0.14 | 7 | 0.59 |
| Global PIQA (parallel) | original · bBPB | 4.89 | never | 0.07 | 33 | 0.58 |
| ARC (MT) | original · bBPB | 4.91 | never | 0.09 | 11 | 0.45 |
| INCLUDE v2 (OG) | original · bBPB | 4.92 | never | 0.06 | 30 | 0.55 |
| MultiBLiMP | original · bBPB | 4.92 | never | 0.04 | 26 | 0.47 |
| INCLUDE v2 (EN) | original · bBPB | 4.94 | never | 0.06 | 30 | 0.52 |
| MMLU | RF · bBPB | 5.00 | never | 0.00 | 1 | 0.61 |
| CommonsenseQA | RF · bBPB | 5.00 | never | 0.00 | 1 | 0.59 |
| PAWS-X | original · bBPB | 5.00 | never | 0.00 | 7 | 0.58 |
| XNLI | original · bBPB | 5.00 | never | 0.00 | 13 | 0.58 |
| ACP-Bench (cloze) | original · bBPB | 5.00 | never | 0.00 | 1 | 0.57 |
| TruthfulQA-Multi | original · bBPB | 5.00 | never | 0.00 | 2 | 0.56 |
| ARC | original · accuracy | 5.00 | never | 0.00 | 21 | 0.56 |
| BLEnD | original · bBPB | 5.00 | never | 0.00 | 4 | 0.55 |
| ACP-Bench (MCQ) | RF · bBPB | 5.00 | never | 0.00 | 1 | 0.55 |
| OpenBookQA | original · bBPB | 5.00 | never | 0.00 | 1 | 0.54 |
| MathQA | original · bBPB | 5.00 | never | 0.00 | 1 | 0.52 |
| TruthfulQA (mc2) | original · bBPB | 5.00 | never | 0.00 | 3 | 0.51 |
| BBH (MCQ) | RF · bBPB | 5.00 | never | 0.00 | 1 | 0.51 |
| XWinograd | original · accuracy | 5.00 | never | 0.00 | 6 | 0.51 |
| BBH (cloze) | original · bBPB | 5.00 | never | 0.00 | 1 | 0.50 |
| ToxiGen | original · bBPB | 5.00 | never | 0.00 | 1 | 0.50 |
| CulturalBench-hard | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  | 8 |  |

**Every variant together and apart** — mean DA-size per proxy over every task (τ = 0.75):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy | 0.52 | 0.51 | 0.51 | 0.54 | 0.53 |
| RF · accuracy | 0.51 | 0.50 | 0.46 | 0.47 | 0.49 |
| LLM-RF · accuracy | 0.49 | 0.47 | 0.50 | 0.50 | 0.48 |
| original · bBPB | 0.57 | 0.53 | 0.57 | 0.55 | 0.52 |
| RF · bBPB | 0.53 | 0.53 | 0.53 | 0.55 | 0.55 |
| LLM-RF · bBPB | 0.65 | 0.60 | 0.61 | 0.56 | 0.61 |
| all variants | 0.55 | 0.53 | 0.54 | 0.54 | 0.53 |

The same over the paired cells (a (benchmark, language) that also has an original-accuracy value at that proxy, so the gate treats every variant alike):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy | 0.52 | 0.51 | 0.51 | 0.54 | 0.53 |
| original · bBPB | 0.60 | 0.51 | 0.55 | 0.57 | 0.54 |
| all variants | 0.56 | 0.51 | 0.53 | 0.55 | 0.54 |

![Variants together and apart](pretraining/predictivity/recipe_da_size_variants_multi_axes.png)

![Benchmark x variant heat map](pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png)

![The cheapest reliable proxy](pretraining/predictivity/recipe_da_size_ladder_multi_axes.png)

![Reliability profiles per benchmark](pretraining/predictivity/recipe_da_size_profiles_multi_axes.png)
<!-- END auto:results -->

**Key findings** (DA-size, multi-axis pairs, τ = 0.75, bBPB at final checkpoints)

- **Few benchmarks reach τ by 1B, whatever the variant.** Only XStoryCloze
  (bBPB) and HellaSwag (accuracy and bBPB) have their median language safe from
  1B; for
  every other benchmark the median language never stays ≥ τ, and the
  recommendation is decided by how many languages do.
- **bBPB is recommended for 24 of the 29 benchmarks with a value.** Pooled over
  every task, bBPB reads the reference better than accuracy in the same format
  at every proxy, with one exception (the original format at 1B, 0.52 against
  0.53), and LLM-RF bBPB is the highest line (0.56–0.65 against 0.51–0.54 for
  the original accuracy). Pooled over all variants, DA-size is 0.53–0.55.
- **The rewrites help through bBPB, not through accuracy.** RF and LLM-RF
  accuracy sit at or below the original accuracy (0.46–0.51 and 0.47–0.50),
  consistent with rq00's finding that the twins pass the gate but rank no
  better. INCLUDE and Belebele are recommended as LLM-RF bBPB, Global-MMLU as
  RF accuracy.
- **On the same cells, bBPB still wins.** Where the original accuracy has a
  value, original bBPB reads the reference at 0.51–0.60 against 0.51–0.54, ahead
  at four of the five proxies and level at 175M: the advantage is not only
  that bBPB is defined where accuracy is at chance.
- **The paired head-to-head of the rewrites is thin.** For the three
  reformulated families the original is at chance in most languages, so their
  paired cells rest on a handful of tasks and are blanked in the tables and
  figures; their gain from bBPB is mostly the "reads where the original cannot"
  kind.

**Follow-ups**

- Rebuild the per-item store over every checkpoint (`build_per_item_store.sbatch
  --pool predictivity_schemes`, then `--bench-bpb`) so that DA-goal and the
  safe compute share, and with them "how small AND how early", are compared
  across all six variants.
- Add rq03's SNR and the k-fold noise per variant to the table, so a reader
  can also see which variant is cheapest to measure reliably.
- Cost: a recommendation should weigh the evaluation cost of a variant (the
  LLM rewrite, a cloze evaluation's extra forward passes), not only its DA.

GitHub: [recipe_da_size_variants_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.png) · [recipe_da_size_variants_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.csv) ·
GitHub: [recipe_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png) · [recipe_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.csv) ·
GitHub: [recipe_da_size_ladder_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.png) · [recipe_da_size_ladder_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.csv) ·
GitHub: [recipe_da_size_profiles_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.png) · [recipe_da_size_profiles_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.csv)

## Files

- `pretraining/<pool>/recipe_da_all_per_task{_multi_axes,_mono_axis}.csv` — per task:
  variant, DA-size per proxy, gate flags, safe size and safe compute share, the
  per-(benchmark, language) recommendation.
- `…/recipe_da_size_by_variant*.csv` — benchmark × variant;
  `recipe_da_size_recommendation*.csv` — one row per benchmark;
  `recipe_da_size_overview*.csv` — per variant and pooled, every task and paired.
- `…/recipe_da_size_{variants,heatmap,ladder,profiles}*.png/.csv` — the figures, each
  with its table.
