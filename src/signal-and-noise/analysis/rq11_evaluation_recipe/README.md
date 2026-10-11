# RQ11: which benchmark, posed how and scored how, gives a reliable decision?

## Research question

The rest of the analysis asks whether a cheap measurement reproduces the
design decision of the 1.7B reference. This analysis turns that question into
advice for a reader who has to choose what to evaluate. A **variant** of a
benchmark is its items as published, their `rf_` cloze rewrite (RF) or their
`rfgm_` LLM rewrite (LLM-RF), each scored by **accuracy** or by the gold
answer's **bits per byte** (bBPB). For each benchmark, which variant recovers
the reference's decision from the smallest proxy?

We call a task reliable when its DA-size is at least τ = 0.75
(`utils.RELIABLE_DA`, the "safe" cut of decision accuracy's early-small
maps). Every figure, table and sentence here uses this one τ. Every number
below is from the ladder report of **2026-10-10 04:31** and the outputs of
the full refresh of 2026-10-10 (detrended checkpoint noise).

## Experimental setup

**Setup.**
- **Measured.** We compare the ways of evaluating a benchmark by the DA-size of the proxies 90M–1B against the 1.7B final checkpoint.
- **Models and pairs.** We use the seed-1904 runs of every cell and every design pair, the multi-axis set, which also holds the single-axis pairs. This is the pair set of the surrogate comparison in the main text. On the single-axis pairs alone the best mean DA-size is 0.585 (LLM-RF bBPB at 90M), and LLM-RF bBPB → 1.7B bBPB falls to 0.65 at 600M, below 0.75.
- **Variants.** Each benchmark is read in up to six ways: its items as published, as RF or as LLM-RF, each scored by accuracy or by the bits per byte of the gold answer (bBPB).
- **Tasks.** An accuracy variant counts on the tasks above chance at the proxy and at the reference. A bBPB variant has no chance level and is read against its original's 1.7B accuracy, so it counts where that accuracy is above chance.

- **Pool** `predictivity`: seed 1904, every cell (every data build, read as
  scheme A/B/C × temperature T, every L, and the deep, shallow and swiglu
  ladders), with the 90M–1B proxies against the 1.7B reference. Gate
  `predictivity`. Both pair sets are computed: multi-axis in the figures
  below, mono-axis in the `_mono_axis` twins.
- **Input**: [decision accuracy](../rq02_decision_accuracy/README.md#3-early-and-small-da-goal-and-da-ckpt-on-the-ten-checkpoints)'s
  per-task table `da_goal_early_small_per_task_both_axes.csv` (regenerated
  by the same refresh), plus the bBPB rows read against accuracy, computed
  here by the same kernel on the same frame, pairs and grid. `recipe.py`
  stops when that table was computed on another population than the pool
  loads now (its largest pair count differs from the pool's), so the two
  kinds of rows are never mixed.
- **Tasks**: 1,754 over 43 benchmarks in the multi-axis table (938 accuracy
  tasks and 816 bBPB twins read under bBPB → 1.7B accuracy, 409 of which are
  also read under bBPB → 1.7B bBPB, see the next bullets).
- **Variants**: each task is one variant of one benchmark in one language,
  its `format` (original, RF, LLM-RF) and `scoring` (accuracy, bBPB) the
  columns the loader writes (`utils.variant`).
- **The bBPB twins** come from the per-item store `predictivity`, which now
  holds every checkpoint. A twin therefore has a DA-size at all five proxies,
  and its safe size is decided on the same five proxies as an accuracy
  variant's. The tables and figures show DA-size. The checkpoint grid enters
  through the safe compute share (`safe_compute`, section 1).
- **Twin coverage**: a twin's cell with fewer pairs than its original's is
  left blank (`same_pairs`, the run prints how many), and in this refresh
  many are. 13 of the pool's models are missing from the per-item store (the
  twelve muon proxies and `lm-1.7B-L8-swiglu-seed1904`). The missing 1.7B
  twin blanks the bBPB → 1.7B bBPB reading wherever that cell is in a pair,
  so 409 of the 816 twins keep a value under that reading, none of them on an
  English-only benchmark.

## Key figure

![Mean DA-size per way of evaluating a benchmark](pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png)

Population: DA-size against the 1.7B final, multi-axis pairs of `predictivity` (seed 1904), gate `predictivity` at the proxy and the reference (a bBPB task has no chance level and passes), ≥ 3 pairs, no filter, every task. Per proxy this is 193–304 original, 97–165 RF and 69–84 LLM-RF accuracy tasks, plus 273 original, 139 RF and 89 LLM-RF bBPB twins at every proxy, and 860–1,054 tasks pooled under "all variants". bBPB is read against the 1.7B accuracy, at the final checkpoint of every proxy 90M–1B. A point on fewer than 5 tasks is not drawn.

**Key finding.** **No way of evaluating a benchmark reaches a mean DA-size of τ = 0.75 against the reference's accuracy decision at any proxy up to 1B: the best mean is 0.63 (LLM-RF bBPB at 90M).** Only the easier target, bBPB → 1.7B bBPB, clears τ (LLM-RF at 0.76–0.88 on 35 tasks, [section 2](#2-every-variant-together-and-apart)). Against the same truth, bBPB beats the original accuracy at all five proxies (0.57–0.60 against 0.53–0.57 over every task, and 0.59–0.64 against 0.54–0.58 on the paired tasks). The rewrites help only when scored by bBPB: RF and LLM-RF accuracy reach 0.48–0.56 across the five proxies, and their bBPB 0.57–0.63. The eight benchmarks with the highest DA-size are drawn per variant along size and training (figure: app_rq11_recipe_by_benchmark).

GitHub: [recipe_da_size_variants_multi_axes_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png) · [recipe_da_size_variants_multi_axes_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.csv). Variants together and apart, and the recommendation per benchmark: [Results](#results).

## Methodology

- **Two targets for a bBPB variant**, both in every table and figure, each
  row naming its own:
  - *bBPB → 1.7B accuracy*: the twin's −bBPB at the proxy against its
    original's accuracy at the 1.7B final (the `rf_`/`rfgm_` prefix kept, so
    RF bBPB is read against RF accuracy), gated on that accuracy at the
    reference only. Every variant then predicts the same truth, the
    reference's accuracy decision. To rounding, this is the bBPB → acc
    reading of
    [the bBPB comparison](../rq02_decision_accuracy/README.md#11-benchmark-bpb-a-continuous-score-on-the-same-items).
  - *bBPB → 1.7B bBPB*: the twin against its own 1.7B bBPB. It has no chance
    level and is never gated. This target is easier: it is the reference's
    bBPB ranking, not its accuracy decision.
  - The accuracy variants are read as accuracy → 1.7B accuracy under both
    targets, gated at the proxy and the reference ([the gate](../rq00_gate_and_curves/README.md#1-which-cells-carry-any-information-the-gate-per-family-and-language)).
- **Per task**: DA-size at each proxy (rule 1 as above, and rule 5's three
  pairs), the **safe size** (the smallest proxy from which DA-size stays ≥ τ
  at every larger proxy) and the safe compute share (the cheapest (proxy,
  checkpoint) cell, the reference's own checkpoints included, from which every
  costlier cell clears τ, as a share of the 1.7B run's compute).
- **Per benchmark × variant**: the share of its tasks reliable at each proxy,
  the mean DA-size, and the mean and median safe rank (0 = safe from 90M,
  4 = from 1B, 5 = never) over the **ranked** tasks: a task gated or under the
  pair minimum at every proxy has no rank, and the ranked languages and tasks
  are counted beside it (rule 13).
- **The recommendation**: per benchmark and target, the variant with the
  smallest mean safe rank. A tie goes to the higher mean DA-size over the
  (task, proxy) cells that every tied variant has. We use the mean because
  the median is "never" whenever half the languages never get there.
- **Paired cells**: a head-to-head between variants uses only the tasks
  whose own original (every prefix stripped) has both an accuracy value and a
  bBPB → 1.7B accuracy value at that proxy, so both sides read the same items
  and bBPB is not credited where accuracy is at chance. The
  unpaired reading asks whether a variant says anything where the original
  cannot.

## Results

### 1. The recommendation: which variant reaches τ from the smallest proxy

<!-- BEGIN auto:results (recipe.py --pool predictivity) -->
DA-size against the 1.7B final, multi-axis pairs (rule 15), pool `predictivity`, rq02's early-small table written 2026-10-11 05:19, ≥ 3 pairs. Reliable means DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`), and the safe size is the smallest proxy from which DA-size stays ≥ τ at every larger proxy. Every row says what it predicts. An accuracy variant predicts its accuracy at 1.7B (gate `predictivity` at the proxy and the reference). A bBPB variant predicts either its original's accuracy at 1.7B (**bBPB → 1.7B accuracy**, gated on that accuracy at the reference only, the reading of `bench_bpb_da.py`) or its own bBPB at 1.7B (**bBPB → 1.7B bBPB**, no chance level, never gated). 1844 tasks over 43 benchmarks. The bBPB twins are read on the same checkpoint grid as the accuracy variants. A mean over fewer than 5 tasks is left blank (`n < 5`) in the tables below and in the figures, but it stays in the CSVs. Regenerate with `python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity`.

**How the pick is made**, per bBPB reading and scoring of the pick, over the benchmarks with a value. Reliable somewhere = mean safe rank below 5 (a task of the pick is safe from some proxy on). Never reliable = 5.00 (no task of the pick is ever safe):

| bBPB read | pick | benchmarks | reliable somewhere | never reliable | won on the DA-size tie-break | only variant that competes | no accuracy variant competes |
|---|---|---|---|---|---|---|---|
| bBPB → 1.7B accuracy | bBPB | 29 | 15 | 14 | 13 | 1 | 1 |
| bBPB → 1.7B accuracy | accuracy | 7 | 5 | 2 | 0 | 3 | 0 |
| bBPB → 1.7B bBPB | bBPB | 31 | 20 | 11 | 5 | 7 | 7 |
| bBPB → 1.7B bBPB | accuracy | 11 | 3 | 8 | 1 | 8 | 0 |

**The recommendation**, per benchmark and bBPB reading, is the variant with the smallest mean safe rank over its ranked tasks (0 = safe from 90M, 4 = from 1B, 5 = never). A task gated or under the pair minimum at every proxy has no rank, so the ranked languages and tasks are counted beside it (rule 13). A tie goes to the higher mean DA-size over the (task, proxy) cells every tied variant has, given in brackets (τ = 0.75). The rows are ordered by the reading with bBPB → 1.7B accuracy:

| benchmark | bBPB read | evaluate it as | decided by | mean safe rank | median safe size | share safe by 1B | languages ranked / all | tasks ranked | mean DA-size |
|---|---|---|---|---|---|---|---|---|---|
| MultiBLiMP (extra) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 90M | 0.67 | 3 / 3 | 3 | n < 5 |
| MultiBLiMP (extra) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 90M | 1.00 | 3 / 3 | 3 | n < 5 |
| HellaSwag | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.96 | 1B | 0.69 | 26 / 26 | 26 | 0.79 |
| HellaSwag | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.27 | 90M | 1.00 | 26 / 26 | 26 | 0.85 |
| XStoryCloze | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 3.50 | 1B | 0.50 | 8 / 8 | 8 | 0.70 |
| XStoryCloze | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.62 | 90M | 0.88 | 8 / 8 | 8 | 0.83 |
| MultiBLiMP | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 3.97 | never | 0.26 | 34 / 34 | 34 | 0.65 |
| MultiBLiMP | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.21 | never | 0.26 | 34 / 34 | 34 | 0.68 |
| TokSuite math | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 1B | 1.00 | 1 / 1 | 1 | n < 5 |
| TokSuite math | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| PIQA | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 1B | 1.00 | 1 / 1 | 1 | n < 5 |
| PIQA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| Global-MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.31 | never | 0.21 | 29 / 29 | 29 | 0.68 |
| Global-MMLU | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.55 | never | 0.28 | 29 / 29 | 29 | 0.67 |
| Global PIQA (non-parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 1B | 0.50 | 2 / 2 | 2 | n < 5 |
| Global PIQA (non-parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 90M | 1.00 | 2 / 2 | 2 | n < 5 |
| LAMBADA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 4.60 | never | 0.40 | 5 / 5 | 5 | 0.67 |
| LAMBADA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 4.60 | never | 0.40 | 5 / 5 | 5 | 0.67 |
| INCLUDE | bBPB → 1.7B accuracy | LLM-RF · bBPB → 1.7B accuracy | safe rank | 4.60 | never | 0.20 | 30 / 36 | 30 | 0.63 |
| INCLUDE | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 2.78 | 1B | 0.72 | 36 / 36 | 36 | 0.79 |
| Belebele | bBPB → 1.7B accuracy | LLM-RF · bBPB → 1.7B accuracy | safe rank | 4.64 | never | 0.14 | 49 / 49 | 59 | 0.59 |
| Belebele | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 3.85 | never | 0.47 | 49 / 49 | 59 | 0.74 |
| MMLU (Okapi) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.67 | never | 0.15 | 27 / 27 | 27 | 0.59 |
| MMLU (Okapi) | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.35 | never | 0.31 | 26 / 26 | 26 | 0.67 |
| ARC | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.73 | never | 0.09 | 21 / 27 | 22 | 0.59 |
| ARC | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.25 | never | 0.21 | 27 / 27 | 28 | 0.66 |
| MMMLU | bBPB → 1.7B accuracy | RF · accuracy → 1.7B accuracy | safe rank | 4.80 | never | 0.20 | 10 / 12 | 10 | 0.59 |
| MMMLU | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | safe rank | 4.80 | never | 0.20 | 10 / 12 | 10 | 0.59 |
| PAWS-X | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.80 | never | 0.20 | 5 / 8 | 5 | 0.56 |
| PAWS-X | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.62 | never | 0.38 | 8 / 8 | 8 | 0.70 |
| INCLUDE v2 (OG) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.82 | never | 0.07 | 35 / 42 | 55 | 0.58 |
| INCLUDE v2 (OG) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.65 | never | 0.16 | 42 / 42 | 77 | 0.64 |
| NorEval | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.86 | never | 0.14 | 1 / 1 | 7 | 0.55 |
| NorEval | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.86 | never | 0.14 | 1 / 1 | 7 | 0.62 |
| XCOPA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.88 | never | 0.12 | 8 / 8 | 8 | 0.48 |
| XCOPA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.62 | never | 0.38 | 8 / 8 | 8 | 0.73 |
| NorEval (extra) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 1 / 1 | 30 | 0.46 |
| NorEval (extra) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.73 | never | 0.17 | 1 / 1 | 30 | 0.57 |
| INCLUDE v2 (EN) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 67 | 0.48 |
| INCLUDE v2 (EN) | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 67 | 0.48 |
| MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MMLU | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| WinoGrande | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| WinoGrande | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| KoBEST | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| KoBEST | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 350M | 0.50 | 1 / 1 | 2 | n < 5 |
| ARC (MT) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 11 / 11 | 11 | 0.61 (shared cells: 0.65) |
| ARC (MT) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.64 | never | 0.18 | 11 / 11 | 11 | 0.62 |
| OpenBookQA | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| OpenBookQA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| TokSuite STEM | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| TokSuite STEM | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| XNLI | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 14 / 15 | 14 | 0.54 (shared cells: 0.56) |
| XNLI | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.67 | never | 0.13 | 15 / 15 | 15 | 0.67 |
| XWinograd | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.54 |
| XWinograd | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.54 |
| BoolQ | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| BoolQ | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| Global PIQA (parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 7 / 46 | 9 | 0.50 (shared cells: 0.51) |
| Global PIQA (parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 3.92 | never | 0.38 | 46 / 46 | 63 | 0.70 |
| ACP-Bench (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 5 | 0.49 (shared cells: 0.49) |
| ACP-Bench (MCQ) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.53 (shared cells: 0.54) |
| TokSuite | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 5 / 5 | 5 | 0.47 (shared cells: 0.47) |
| TokSuite | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 5 / 5 | 5 | 0.42 (shared cells: 0.42) |
| BBH (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 10 | 0.47 (shared cells: 0.48) |
| BBH (MCQ) | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 17 | 0.52 (shared cells: 0.53) |
| CulturalBench-easy | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 5 / 8 | 7 | 0.42 (shared cells: 0.40) |
| CulturalBench-easy | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.95 | never | 0.05 | 8 / 8 | 19 | 0.57 |
| ACP-Bench (cloze) | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ACP-Bench (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.54 |
| BBH (cloze) | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BBH (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 6 | 0.50 |
| BLEnD | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 4 | 0 |  |
| BLEnD | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 4 / 4 | 5 | 0.55 |
| CulturalBench-hard | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| CulturalBench-hard | bBPB → 1.7B bBPB | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| ToxiGen | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ToxiGen | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| TruthfulQA (mc2) | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 3 | 0 |  |
| TruthfulQA (mc2) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 3 / 3 | 3 | n < 5 |
| TruthfulQA-Multi | bBPB → 1.7B accuracy | none (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 2 | 0 |  |
| TruthfulQA-Multi | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 2 / 2 | 2 | n < 5 |

![The cheapest reliable proxy](pretraining/predictivity/recipe_da_size_ladder_multi_axes.png)
<!-- END auto:results -->

**Key findings** (`recipe_da_size_recommendation_multi_axes.csv`, `recipe_da_all_per_task_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, τ = 0.75, bBPB at all five proxies, and each bullet names its target)

- **Against the same truth (bBPB → 1.7B accuracy), bBPB is the pick for 20
  of the 36 benchmarks with a value and accuracy for 16.** Half of the bBPB
  picks (10 of 20) are never reliable and nine win only on the DA-size
  tie-break, while 14 of the 16 accuracy picks are the only variant that
  competes. Seven of the 43 benchmarks have no value. Ten bBPB picks and
  seven accuracy picks are reliable somewhere (mean safe rank below 5), and
  HellaSwag is among the tie-break wins.
- **The pick compares like with like.** A bBPB task is ranked on the same
  five proxies as an accuracy task, so a twin that dips below τ at 600M ranks
  later. Under bBPB → 1.7B accuracy, for example, HellaSwag's original bBPB
  ties its accuracy at 3.12 (reliable share 0.35 at 600M) and wins on the
  tie-break. Under bBPB → 1.7B bBPB, Belebele LLM-RF sits at 3.29.
- **Against the easier target (bBPB → 1.7B bBPB), bBPB is the pick for 15
  of the 38 benchmarks with a value.** Three of them have no accuracy variant
  at all. XCOPA and INCLUDE v2 (EN) switch from accuracy to bBPB when the
  target changes. ARC (MT), XStoryCloze, MathQA, OpenBookQA, MMLU,
  CommonsenseQA, BBH (MCQ), CulturalBench-easy and ACP-Bench (MCQ) switch
  from bBPB to accuracy, in each case because accuracy is the only variant
  that competes (their twins are blank or on too few tasks, see Setup).
  Belebele switches from RF to LLM-RF bBPB.
- **Few benchmarks are safe by 1B in most languages.** With bBPB → 1.7B
  accuracy, the share of ranked tasks safe by 1B is 0.69 for HellaSwag (26
  tasks), 0.62 for XStoryCloze (8, bBPB) and 0.80 for LAMBADA (5, accuracy).
  Every other benchmark with more than three ranked tasks is at 0.50 or
  below (boolq reaches 1.00 on one task and MultiBLiMP (extra) 0.67 on
  three). With bBPB → 1.7B bBPB the share is 1.00 for Global PIQA
  non-parallel and PAWS-X (2 tasks each), 0.95 for HellaSwag (20), 0.93 for
  INCLUDE (14 tasks, LLM-RF bBPB), 0.80 for LAMBADA (5, accuracy), 0.67 for
  Belebele (21, LLM-RF bBPB) and 0.50 for XCOPA (6), besides the same two
  small benchmarks.
- **Along compute, the safe cell is mostly the reference itself.**
  `safe_compute` is filled for every reading. A cheapest safe (proxy,
  checkpoint) cell exists for 0.47 of the accuracy tasks, 0.18 under
  bBPB → 1.7B accuracy and 0.89 under bBPB → 1.7B bBPB. But it is a proxy
  cell (at most 1B) for only 0.04, 0.09 and 0.18 of them. The median safe
  cell sits at 0.9, 0.4 and 0.7 of the reference run's compute, mostly on the
  reference's own checkpoints (under bBPB → 1.7B accuracy, about half of the
  few safe cells are proxy cells).

**Follow-ups**

- A per-variant table and figure of `safe_compute` and of DA-ckpt (the store
  now holds every checkpoint), split into proxy cells and reference
  checkpoints, so that the compute ladder is read per variant and not only
  per task.
- Add the SNR and the k-fold noise per variant to the table, so that a
  reader can also see which variant is cheapest to measure reliably.
- Weigh the evaluation cost of a variant (the LLM rewrite, a cloze
  evaluation's extra forward passes), not only its DA.

GitHub: [recipe_da_size_ladder_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.png) · [recipe_da_size_ladder_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.csv)

### 2. Every variant together and apart

The recommendation picks one variant per benchmark. This figure asks how
far apart the variants are when pooled, and whether the gap holds on the
cells both have.

<!-- BEGIN auto:variants (recipe.py --pool predictivity) -->
**Every variant together and apart.** The mean DA-size per proxy over every task with a value (τ = 0.75). In brackets are the tasks behind each mean, which differ across proxies and variants because the gate keeps different tasks at different sizes (rule 13):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.53 (195) | 0.55 (220) | 0.54 (239) | 0.57 (268) | 0.57 (305) |
| RF · accuracy → 1.7B accuracy | 0.48 (104) | 0.50 (119) | 0.49 (131) | 0.51 (153) | 0.56 (166) |
| LLM-RF · accuracy → 1.7B accuracy | 0.48 (69) | 0.49 (71) | 0.52 (75) | 0.50 (80) | 0.54 (84) |
| original · bBPB → 1.7B accuracy | 0.58 (320) | 0.56 (320) | 0.57 (320) | 0.58 (315) | 0.57 (320) |
| RF · bBPB → 1.7B accuracy | 0.59 (178) | 0.59 (178) | 0.59 (178) | 0.60 (178) | 0.60 (178) |
| LLM-RF · bBPB → 1.7B accuracy | 0.62 (89) | 0.59 (89) | 0.60 (89) | 0.57 (89) | 0.60 (89) |
| original · bBPB → 1.7B bBPB | 0.64 (598) | 0.60 (598) | 0.62 (598) | 0.61 (598) | 0.62 (598) |
| RF · bBPB → 1.7B bBPB | 0.65 (207) | 0.64 (207) | 0.66 (207) | 0.65 (207) | 0.66 (207) |
| LLM-RF · bBPB → 1.7B bBPB | 0.79 (95) | 0.77 (95) | 0.77 (95) | 0.72 (95) | 0.75 (95) |
| all variants, bBPB → 1.7B accuracy | 0.56 (955) | 0.56 (997) | 0.56 (1032) | 0.56 (1083) | 0.57 (1142) |
| all variants, bBPB → 1.7B bBPB | 0.62 (1268) | 0.59 (1310) | 0.61 (1345) | 0.60 (1401) | 0.61 (1455) |

The same over the paired tasks (those whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix stripped, has an accuracy and a bBPB → 1.7B accuracy value at that proxy, so the gate treats every variant alike):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.53 (187) | 0.55 (210) | 0.54 (229) | 0.57 (250) | 0.56 (292) |
| original · bBPB → 1.7B accuracy | 0.61 (187) | 0.57 (210) | 0.59 (229) | 0.59 (250) | 0.58 (292) |
| original · bBPB → 1.7B bBPB | 0.70 (182) | 0.61 (205) | 0.67 (224) | 0.64 (250) | 0.64 (287) |
| RF · bBPB → 1.7B bBPB |  |  |  | n = 1 | n = 1 |
| LLM-RF · bBPB → 1.7B bBPB |  |  |  | n = 1 | n = 1 |
| all variants, bBPB → 1.7B accuracy | 0.57 (374) | 0.56 (420) | 0.57 (458) | 0.58 (500) | 0.57 (584) |
| all variants, bBPB → 1.7B bBPB | 0.61 (369) | 0.58 (415) | 0.61 (453) | 0.60 (502) | 0.60 (581) |

![Variants together and apart](pretraining/predictivity/recipe_da_size_variants_multi_axes.png)
<!-- END auto:variants -->

**Key findings** (`recipe_da_size_overview_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, gate `predictivity`, bBPB at all five proxies, and each bullet names its target)

- **Against the same truth, bBPB predicts the reference a little better than accuracy.**
  Over every task, original bBPB → 1.7B accuracy is 0.60, 0.57, 0.58, 0.59
  and 0.59 from 90M to 1B (273 tasks), against 0.53, 0.55, 0.54, 0.57 and
  0.57 for the original accuracy (193–304 tasks). The lead is largest at 90M.
- **The small lead holds on the paired tasks.** There bBPB reaches 0.64,
  0.59, 0.60, 0.60 and 0.60 (149–245 tasks), against 0.54, 0.57, 0.56, 0.58
  and 0.58 for the accuracy of the same tasks.
  So the lead does not come only from bBPB being defined where accuracy is at
  chance.
- **The easier target gives higher values.** bBPB → 1.7B bBPB gives the
  original 0.64–0.69 (284 tasks), RF 0.69–0.72 (90) and LLM-RF 0.76–0.88 (35)
  over the five proxies, against 0.56–0.58 for all variants pooled under
  bBPB → 1.7B accuracy. Part of bBPB's lead under the easier reading comes
  from the target, not from the score.
- **The rewrites help only when scored by bBPB.** RF and LLM-RF accuracy sit
  at 0.50–0.56 and 0.48–0.54 over the five proxies, mostly below the original
  accuracy's 0.53–0.57. Scored by bBPB → 1.7B accuracy, they reach 0.59–0.60
  (139 tasks) and 0.57–0.63 (89). On the paired tasks the rewrites have no
  bBPB → 1.7B accuracy task and one bBPB → 1.7B bBPB task each, at 600M and
  1B only (n = 1).
- **The populations differ by target.** 273 original twins have a value
  under bBPB → 1.7B accuracy (gated on the original's accuracy at 1.7B) and
  284 under bBPB → 1.7B bBPB (never gated, but blank where the per-item store
  lacks a family, see Setup).

**Follow-ups**

- The paired panel per format, so that each rewrite's head-to-head with its
  own accuracy is read on the same tasks.

GitHub: [recipe_da_size_variants_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.png) · [recipe_da_size_variants_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.csv)

### 3. Per benchmark and variant

The pooled lines hide which benchmarks carry the gap. The heat map shows it
per benchmark and proxy.

<!-- BEGIN auto:heatmap (recipe.py --pool predictivity) -->
![Benchmark x variant heat map](pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png)
<!-- END auto:heatmap -->

**Key findings** (`recipe_da_size_by_variant_multi_axes.csv`: per benchmark, the mean DA-size at each proxy 90M–1B, for the benchmarks with ≥ 5 tasks in both variants at that proxy, and each bullet names its target)

- **bBPB → 1.7B accuracy beats the original accuracy on most benchmarks at
  every proxy.** It is ahead on 5 of 5 benchmarks at 90M, 5 of 7 at 175M and
  350M, 8 of 8 at 600M and 7 of 9 at 1B. The largest leads are on
  XStoryCloze at 175M (0.72 against 0.51), XCOPA at 175M and 350M (0.64
  against 0.41 and 0.44) and HellaSwag at 90M (0.83 against 0.62). It is
  behind on INCLUDE v2 (EN) and MultiBLiMP at 175M, on XStoryCloze and XNLI
  at 350M (0.71 against 0.73, 0.50 against 0.52) and on XCOPA and HellaSwag
  at 1B (0.59 against 0.65, 0.80 against 0.82). At 600M its lead on
  HellaSwag and MultiBLiMP is below 0.005 (both 0.72 and both 0.66 at two
  decimals).
- **In the rewrites, bBPB beats the same format's accuracy almost
  everywhere.** For RF it is ahead on 3 of 4 benchmarks at 90M and 175M, 4
  of 5 at 350M and 6 of 6 at 600M and 1B (Global-MMLU RF 0.69 against 0.44 at
  90M, and 0.70 against 0.68 at 1B). For LLM-RF it is ahead on 2 of 2 at
  every proxy (INCLUDE 0.65 against 0.57 at 1B).
- **bBPB → 1.7B bBPB is above bBPB → 1.7B accuracy on nearly every original
  benchmark.** It is above on 8 of 8 at 90M and 350M and 7 of 8 at 175M,
  600M and 1B. The exception is INCLUDE v2 (EN) (0.32 against 0.41 at 175M,
  0.45 against 0.49 at 600M, 0.48 against 0.49 at 1B). So a heat map of
  bBPB → bBPB cells overstates what bBPB says about the accuracy decision.

**Follow-ups**

- The same map on the paired cells only.

GitHub: [recipe_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png) · [recipe_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.csv)

### 4. Where each variant starts to read the reference

The heat map gives means. The profiles give the share of a benchmark's
tasks that are reliable at each proxy, which is what the safe rank
summarises.

<!-- BEGIN auto:profiles (recipe.py --pool predictivity) -->
![Reliability profiles per benchmark](pretraining/predictivity/recipe_da_size_profiles_multi_axes.png)
<!-- END auto:profiles -->

**Key findings** (`recipe_da_size_by_variant_multi_axes.csv`, `reliable_share_1B` over the benchmarks with a value at 1B, and each bullet names its target)

- **Few benchmarks have half of their tasks reliable at the 1B proxy.** With
  original accuracy it is 4 of 27 (HellaSwag and LAMBADA, plus boolq on one
  task and MultiBLiMP (extra) on three). With original bBPB → 1.7B accuracy
  it is 3 of 17 (HellaSwag, XStoryCloze, and Global PIQA non-parallel on 2
  tasks). With original bBPB → 1.7B bBPB it is 5 of 17 (those three, PAWS-X
  on 2 tasks and XCOPA).
- **No RF variant gets there under either target.** It is 0 of 10 for RF
  accuracy, 0 of 8 for RF bBPB → 1.7B accuracy and 0 of 3 for RF
  bBPB → 1.7B bBPB. Both LLM-RF benchmarks get there with bBPB → 1.7B bBPB
  (INCLUDE 0.93, Belebele 0.67).

**Follow-ups**

- Mark the proxy where each recommended variant first passes half of its
  tasks, to read the profiles at a glance.

GitHub: [recipe_da_size_profiles_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.png) · [recipe_da_size_profiles_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.csv)

### 5. The best benchmarks along size and training (paper figure)

The profiles use one decision accuracy, at final checkpoints. This figure
follows the eight best benchmarks through all three (DA-size, DA-ckpt and
DA-goal), so a variant that wins at the 90M final can also be checked early
in the run.

**DA-size, DA-ckpt, DA-goal · no reliability filter · multi-axis pairs · pairs
from `predictivity` at seed 1904 · gate `predictivity`.** The gate applies to
accuracy at the proxy and at 1.7B, and to DA-ckpt at the proxy only.
bBPB → 1.7B accuracy is gated on the accuracy of the same items at 1.7B, and
bBPB → 1.7B bBPB is never gated. Snapshot: the 2026-10-10 04:31 ladder report
and the 2026-10-10 refresh, the one rq02's tables on disk were computed on.
Rows: Overall (BPB of the 50 trained languages, accuracy over every format,
every twin against the 1.7B bBPB), then the eight benchmarks with the highest
`mean_da_size` (`recipe_da_size_by_variant_multi_axes.csv`: the mean over
tasks per proxy, then over the proxies 90M–1B with a value). Each benchmark
is shown in the better of its original and RF formats, among the formats with
≥ 5 ranked tasks (the ranking is in the CSV's `ranking` rows). Columns:
DA-size per proxy, then DA-ckpt (rq02's
`da_all_pooled_per_task_multi_axes.csv`, `da_own`) and DA-goal
(`recipe_da_goal_per_task_multi_axes.csv`) per tenth of the run, each task
averaged over the proxies 90M–1B first. The bBPB lines are the twins of the
ranking format. bBPB → 1.7B accuracy has no DA-ckpt, because its reference
would be the proxy's own accuracy, a different score. Bands: 95 % bootstrap
over the tasks of a line (1,000 resamples). A point on < 5 tasks is not
drawn. Task counts differ across lines and, for accuracy, across proxies
(because of the gate, see the CSV's `n_tasks`).

![DA-size, DA-ckpt and DA-goal per benchmark](pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.png)

**Key findings** (`recipe_da_all_by_benchmark_multi_axes_paper.csv`)

- **HellaSwag and LAMBADA lead the ranking.** The order is HellaSwag 0.72,
  LAMBADA 0.71, MultiBLiMP 0.65, XStoryCloze 0.64, ARC 0.62 (original), MMMLU
  0.59 (RF), NorEval 0.57 (original) and ARC (MT) 0.56 (original), with
  Global-MMLU (RF) next at 0.56. ARC (MT) and Global-MMLU differ only in the
  third decimal (0.560 against 0.558).
- **At the 90M proxy, bBPB → 1.7B accuracy beats the benchmark's accuracy on
  both benchmarks where both are drawn.** HellaSwag reaches 0.83 against 0.62
  and MultiBLiMP 0.65 against 0.63. At 1B, accuracy catches up only on
  HellaSwag (0.82 against 0.80).
- **DA-goal keeps that order along the run.** On the five benchmarks with a
  twin, bBPB → 1.7B accuracy is above every accuracy format at 50–100 % of
  the tenths of the run (all ten on HellaSwag and XStoryCloze, nine on
  ARC (MT), eight on MultiBLiMP and half on ARC). HellaSwag accuracy climbs
  from 0.51 at 0.5C to 0.72 at 5C and stays under its twin (0.74–0.80).
- **DA-ckpt is highest for bBPB → 1.7B bBPB on every benchmark and tenth but
  one.** It is −0.01 to +0.28 above the best accuracy format, and below it
  only on ARC (MT) at 80 % of the run (−0.01), as in rq02. A twin's ranking
  persists within its own run more than accuracy's does, but persistence is
  not a decision.
- **In the Overall row, BPB is highest and accuracy lowest.** DA-size on
  accuracy goes from 0.51 at 90M to 0.56 at 1B, while the bBPB twins stay at
  0.67–0.71 and BPB at 0.73–0.95. DA-goal gives 0.50–0.54, 0.67–0.70 and
  0.81–0.89 for the same three lines.

**Follow-ups**

- The same figure on mono-axis pairs (the rq2 pair set), once the
  bBPB → 1.7B accuracy cells are read there too.
- A per-language version of one row (e.g. HellaSwag), to see whether the bBPB
  lead at 90M comes from a few languages or from all of them.

GitHub: [recipe_da_all_by_benchmark_multi_axes_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.png) · [recipe_da_all_by_benchmark_multi_axes_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.csv)

## The key figure on one task set per line

Each line averages, at each proxy size, the tasks that pass the gate at that size, so a line can move just because its tasks changed (rule 13). This twin figure keeps, for each line, only the tasks with a value at every size (solid), with the committed line dashed behind it.

![Mean DA-size per variant, one task set per line](pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.png)

DA-size against 1.7B, multi-axis pairs, gate `predictivity`, no reliability filter. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.csv)

Key findings:

- **The pooled line barely moves.** On the 852 tasks it has at every size it reads 0.57, 0.56, 0.56, 0.57 and 0.59, against the committed 0.56, 0.56, 0.56, 0.57 and 0.58 on 860–1,054 tasks. The bBPB variants have every task at every proxy, so only the accuracy variants can shift it.
- **Original accuracy rises a little more on one task set.** Its 189 tasks go from 0.53 to 0.59 between 90M and 1B, against 0.53 to 0.57 on 193–304 tasks. The tasks that join at larger sizes rank worse.
- **The verdict does not change.** No line reaches τ = 0.75 on either task set, and the bBPB lines (the same tasks at every size) do not move.

Follow-ups:

- **The recommendation table on fixed tasks.** The per-benchmark pick reads the same moving populations.

## Files

- `pretraining/<pool>/recipe_da_all_per_task{_multi_axes,_mono_axis}.csv`:
  per task and reading (`reading`: acc_acc, bbpb_acc, bbpb_bbpb), the
  variant, DA-size per proxy, gate flags, safe size, safe rank and
  `safe_compute` (-1 = no cell from which every costlier one clears τ, empty
  for a task with no ungated cell, every reading on the full checkpoint grid
  since the 2026-10-07 refresh), and the per-(benchmark, language)
  recommendation per bBPB reading (`recommended_bbpb_acc`,
  `recommended_bbpb_bbpb`).
- `…/recipe_da_size_by_variant*.csv`: benchmark × variant per bBPB reading
  (`bbpb_reading`), with `languages_ranked` and `n_tasks`.
  `recipe_da_size_recommendation*.csv` has one row per benchmark and bBPB
  reading, with `decided_by`. `recipe_da_size_overview*.csv` gives each
  variant and the pool, over every task and over the paired tasks, with the
  task counts.
- `…/recipe_da_size_{variants,heatmap,ladder,profiles}*.png/.csv`: the
  figures, each with its table.
- `…/recipe_da_goal_per_task*.csv`: the early-small cells every table reads,
  per task, reading, proxy and tenth, with rule 5 applied and the gate
  verdict in `gated` (DA-size is its `frac` = 1.0 rows).
- `…/recipe_da_all_by_benchmark_multi_axes_paper.png/.csv`: the paper's
  per-benchmark figure (section 5). Per row, line and DA kind it gives the
  mean, the 95 % bootstrap band, `n_tasks` and `drawn`, and the benchmark
  ranking is in the `ranking` rows.
