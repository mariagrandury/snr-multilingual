# RQ11 — Which benchmark, posed how and scored how, gives a reliable decision?

## Research question

The rest of the analysis asks whether a cheap measurement reproduces the
1.7B reference's design decision. This analysis turns that into advice for a
reader who has to choose what to evaluate: for each benchmark, which
**variant** — the items as published, their `rf_` cloze rewrite (RF) or their
`rfgm_` LLM rewrite (LLM-RF), scored by **accuracy** or by the gold answer's
**bits per byte** (bBPB) — reads the reference's decision from the smallest
proxy?

Reliable means DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`, the "safe" cut of
decision accuracy's early-small maps), and every figure, table and sentence
here uses that one τ. Every number below is from the ladder report of
**2026-10-07 15:51** (the full refresh of 2026-10-07, commit 7966367c).

## Experimental setup

- **Pool** `predictivity`: seed 1904, every cell (every data build, read as
  scheme A/B/C × temperature T; every L; the deep, shallow and swiglu
  ladders), 90M–1B proxies against the 1.7B reference. Gate `predictivity`;
  both pair sets (multi-axis in the figures below, mono-axis in the
  `_mono_axis` twins).
- **Input**: [decision accuracy](../rq02_decision_accuracy/README.md#3-early-and-small-da-goal-and-da-ckpt-on-the-ten-checkpoints)'s
  per-task table `da_goal_early_small_per_task_both_axes.csv` (regenerated
  by the same refresh), plus the bBPB rows read against accuracy, computed
  here by the same kernel on the same frame, pairs and grid. `recipe.py`
  stops when that table was computed on another population than the pool
  loads now (its largest pair count differs from the pool's), so the two
  kinds of rows are never mixed.
- **Tasks**: 1,662 over 31 benchmarks in the multi-axis table (846 accuracy
  tasks, 816 bBPB twins, each twin read under both targets below).
- **Variants**: each task is one variant of one benchmark in one language,
  its `format` (original, RF, LLM-RF) and `scoring` (accuracy, bBPB) the
  columns the loader writes (`utils.variant`).
- **The bBPB twins** come from the per-item store `predictivity`, which now
  holds every checkpoint, so a twin has a DA-size at all five proxies and its
  safe size is decided on the same five as an accuracy variant's. The tables
  and figures are DA-size; the checkpoint grid enters through the safe
  compute share (`safe_compute`, section 1).
- **Twin coverage**: a twin's cell with fewer pairs than its original's is
  left blank (`same_pairs`, the run prints how many); in this refresh none is.

## Key figure

![Mean DA-size per way of evaluating a benchmark](pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png)

Population: DA-size against the 1.7B final, multi-axis pairs of `predictivity` (seed 1904), gate `predictivity` at the proxy and the reference (a bBPB task has no chance level and passes), ≥ 3 pairs, no filter, every task (155–236 original, 77–130 RF and 69–84 LLM-RF accuracy tasks per proxy; 255 original, 139 RF and 89 LLM-RF bBPB twins at every proxy; 784–933 tasks pooled under "all variants"); bBPB read against the 1.7B accuracy, at the final checkpoint of every proxy 90M–1B; a point on fewer than 5 tasks is not drawn.

**Key finding.** No way of evaluating a benchmark reads the reference's accuracy decision at τ = 0.75 on average at any proxy up to 1B: the best mean is 0.63 (LLM-RF bBPB at 90M; only the easier bBPB → 1.7B bBPB target, LLM-RF at 0.71–0.80, clears it, [section 2](#2-every-variant-together-and-apart)). Against the same truth, bBPB matches or edges the original accuracy at all five proxies (every task: 0.59–0.61 against 0.55–0.60; paired: 0.60–0.64), and the rewrites help only when scored by bBPB (RF and LLM-RF accuracy 0.48–0.56 across the five proxies, their bBPB 0.57–0.63).

GitHub: [recipe_da_size_variants_multi_axes_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png) · [recipe_da_size_variants_multi_axes_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.csv). Variants together and apart, and the recommendation per benchmark: [Results](#results).

## Methodology

- **Two targets for a bBPB variant**, both in every table and figure, each
  row naming its own:
  - *bBPB → 1.7B accuracy*: the twin's −bBPB at the proxy against its
    original's accuracy at the 1.7B final (the `rf_`/`rfgm_` prefix kept, so
    RF bBPB is read against RF accuracy), gated on that accuracy at the
    reference only. Every variant then predicts the same truth, the
    reference's accuracy decision; it is
    [the bBPB comparison](../rq02_decision_accuracy/README.md#11-benchmark-bpb-a-continuous-score-on-the-same-items)'s
    bBPB → acc reading, to rounding.
  - *bBPB → 1.7B bBPB*: the twin against its own 1.7B bBPB, no chance level,
    never gated. An easier target: the reference's bBPB ranking, not its
    decision.
  - The accuracy variants read accuracy → 1.7B accuracy, gated at the proxy
    and the reference ([the gate](../rq00_gate_and_curves/README.md#1-which-cells-carry-any-information-the-gate-per-family-and-language)), in both.
- **Per task**: DA-size at each proxy (rule 1 as above; rule 5's three
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
  smallest mean safe rank; a tie goes to the higher mean DA-size over the
  (task, proxy) cells every tied variant has. The mean is used because the
  median is "never" whenever half the languages never get there.
- **Paired cells**: a head-to-head between variants reads the tasks whose
  own original (every prefix stripped) has an accuracy value at that proxy,
  because bBPB is defined where accuracy is at chance. The unpaired reading
  asks whether a variant reads anything where the original cannot.

## Results

### 1. The recommendation: which variant reaches τ from the smallest proxy

<!-- BEGIN auto:results (recipe.py --pool predictivity) -->
DA-size against the 1.7B final, multi-axis pairs (rule 15), pool `predictivity`, rq02's early-small table written 2026-10-07 18:19, ≥ 3 pairs; reliable = DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every larger proxy. Every row says what it predicts: an accuracy variant its accuracy at 1.7B (gate `predictivity` at the proxy and the reference); a bBPB variant either its original's accuracy at 1.7B (**bBPB → 1.7B accuracy**, gated on that accuracy at the reference only, the reading of `bench_bpb_da.py`) or its own bBPB at 1.7B (**bBPB → 1.7B bBPB**, no chance level, never gated). 1662 tasks over 31 benchmarks. The bBPB twins are read at final checkpoints only for now. A mean over fewer than 5 tasks is left blank (`n < 5`) in the tables below and the figures; it stays in the CSVs. Regenerate with `python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity`.

**How the pick is made**, per bBPB reading and scoring of the pick (benchmarks with a value; reliable somewhere = mean safe rank below 5, a task of the pick safe from some proxy on; never reliable = 5.00, no task of the pick ever safe):

| bBPB read | pick | benchmarks | reliable somewhere | never reliable | won on the DA-size tie-break | only variant with a value | no accuracy variant has a value |
|---|---|---|---|---|---|---|---|
| bBPB → 1.7B accuracy | bBPB | 17 | 9 | 8 | 7 | 1 | 1 |
| bBPB → 1.7B accuracy | accuracy | 6 | 5 | 1 | 0 | 2 | 0 |
| bBPB → 1.7B bBPB | bBPB | 25 | 14 | 11 | 4 | 8 | 8 |
| bBPB → 1.7B bBPB | accuracy | 5 | 3 | 2 | 1 | 2 | 0 |

**The recommendation** (per benchmark and bBPB reading: the variant with the smallest mean safe rank over its ranked tasks — 0 = safe from 90M, 4 = from 1B, 5 = never; a task gated or under the pair minimum at every proxy has no rank, so the ranked languages and tasks are counted beside it (rule 13) — then the higher mean DA-size over the (task, proxy) cells every tied variant has, given in brackets; τ = 0.75), ordered by the reading with bBPB → 1.7B accuracy:

| benchmark | bBPB read | evaluate it as | decided by | mean safe rank | median safe size | share safe by 1B | languages ranked / all | tasks ranked | mean DA-size |
|---|---|---|---|---|---|---|---|---|---|
| HellaSwag | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.44 | 1B | 0.80 | 25 / 26 | 25 | 0.82 |
| HellaSwag | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.46 | 90M | 1.00 | 26 / 26 | 26 | 0.87 |
| Global PIQA (non-parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 350M | 0.50 | 2 / 2 | 2 | n < 5 |
| Global PIQA (non-parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 90M | 1.00 | 2 / 2 | 2 | n < 5 |
| XStoryCloze | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.88 | 1B | 0.62 | 8 / 8 | 8 | 0.74 |
| XStoryCloze | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.50 | 90M | 1.00 | 8 / 8 | 8 | 0.86 |
| MultiBLiMP | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 3.71 | never | 0.32 | 34 / 34 | 34 | 0.66 |
| MultiBLiMP | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.03 | never | 0.29 | 34 / 34 | 34 | 0.65 |
| LAMBADA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| LAMBADA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| Global-MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.03 | never | 0.28 | 29 / 29 | 29 | 0.68 |
| Global-MMLU | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.52 | never | 0.28 | 29 / 29 | 29 | 0.64 |
| ARC | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.44 | never | 0.25 | 15 / 27 | 16 | 0.65 |
| ARC | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.93 | never | 0.39 | 27 / 27 | 28 | 0.67 |
| INCLUDE | bBPB → 1.7B accuracy | LLM-RF · bBPB → 1.7B accuracy | safe rank | 4.57 | never | 0.23 | 30 / 36 | 30 | 0.64 |
| INCLUDE | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 2.25 | 600M | 0.81 | 36 / 36 | 36 | 0.80 |
| Belebele | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.58 | never | 0.14 | 49 / 49 | 57 | 0.58 |
| Belebele | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 3.95 | never | 0.46 | 49 / 49 | 59 | 0.74 |
| INCLUDE v2 (OG) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.71 | never | 0.11 | 35 / 42 | 55 | 0.59 |
| INCLUDE v2 (OG) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.66 | never | 0.14 | 42 / 42 | 77 | 0.64 |
| PAWS-X | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | n < 5 | never | 0.25 | 4 / 8 | 4 | n < 5 |
| PAWS-X | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.75 | never | 0.25 | 8 / 8 | 8 | 0.70 |
| XCOPA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.75 | never | 0.25 | 8 / 8 | 8 | 0.49 |
| XCOPA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.50 | 1B | 0.50 | 8 / 8 | 8 | 0.75 |
| ARC (MT) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.90 | never | 0.10 | 10 / 11 | 10 | 0.58 |
| ARC (MT) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.55 | never | 0.27 | 11 / 11 | 11 | 0.64 |
| INCLUDE v2 (EN) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 67 | 0.48 |
| INCLUDE v2 (EN) | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 67 | 0.48 |
| MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MMLU | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| XNLI | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 14 / 15 | 14 | 0.53 (shared cells: 0.55) |
| XNLI | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.67 | never | 0.13 | 15 / 15 | 15 | 0.70 |
| XWinograd | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.53 |
| XWinograd | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.53 |
| Global PIQA (parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 46 | 1 | n < 5 |
| Global PIQA (parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 3.79 | never | 0.40 | 46 / 46 | 63 | 0.73 |
| ACP-Bench (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 5 | 0.50 (shared cells: 0.50) |
| ACP-Bench (MCQ) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.53 (shared cells: 0.54) |
| BBH (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 10 | 0.46 (shared cells: 0.47) |
| BBH (MCQ) | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 17 | 0.52 (shared cells: 0.53) |
| CulturalBench-easy | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 5 / 8 | 7 | 0.43 (shared cells: 0.41) |
| CulturalBench-easy | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.89 | never | 0.11 | 8 / 8 | 19 | 0.56 |
| ACP-Bench (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ACP-Bench (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.53 |
| BBH (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BBH (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 6 | 0.50 |
| BLEnD | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 4 | 0 |  |
| BLEnD | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 4 / 4 | 5 | 0.54 |
| CulturalBench-hard | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| CulturalBench-hard | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| OpenBookQA | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| OpenBookQA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| ToxiGen | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ToxiGen | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| TruthfulQA (mc2) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 3 | 0 |  |
| TruthfulQA (mc2) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 3 / 3 | 3 | n < 5 |
| TruthfulQA-Multi | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 2 | 0 |  |
| TruthfulQA-Multi | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 2 / 2 | 2 | n < 5 |

![The cheapest reliable proxy](pretraining/predictivity/recipe_da_size_ladder_multi_axes.png)
<!-- END auto:results -->

**Key findings** (`recipe_da_size_recommendation_multi_axes.csv`, `recipe_da_all_per_task_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, τ = 0.75, bBPB at all five proxies; each bullet names the target it reads)

- **Against the same truth (bBPB → 1.7B accuracy), bBPB is the pick for 17
  of the 23 benchmarks with a value and accuracy for 6** (8 of 31 have none).
  Nine bBPB picks and five accuracy picks are reliable somewhere (mean safe
  rank below 5); seven bBPB picks win only on the DA-size tie-break (MathQA
  moved from accuracy to bBPB on it).
- **The pick is now like for like.** A bBPB task is ranked on the same five
  proxies as an accuracy task; the 90M–600M twins left every pick unchanged
  except MathQA's (a one-task tie-break), and moved mean safe ranks mostly
  later, where a twin dips below τ at 350M–600M (HellaSwag 1.80 to 2.44 under
  bBPB → 1.7B accuracy, Belebele LLM-RF 3.47 to 3.95 under bBPB → 1.7B bBPB).
- **Against the easier target (bBPB → 1.7B bBPB), bBPB takes 25 of the 30
  benchmarks with a value**, 8 of them with no accuracy variant at all.
  ARC (MT), PAWS-X and XCOPA switch from accuracy to bBPB with the target,
  MultiBLiMP and MMLU from bBPB to accuracy, Belebele from
  RF to LLM-RF bBPB and ACP-Bench (MCQ) from RF to original bBPB (the latter
  on the DA-size tie-break).
- **Few benchmarks are safe by 1B in most languages.** With bBPB → 1.7B
  accuracy, the share of ranked tasks safe by 1B is 0.80 for HellaSwag (25
  tasks), 0.62 for XStoryCloze (8, bBPB) and 1.00 for LAMBADA (5, accuracy);
  every other benchmark is at 0.50 or below. With bBPB → 1.7B bBPB it is 1.00
  for HellaSwag, XStoryCloze, Global PIQA non-parallel (2 tasks) and LAMBADA
  (5, accuracy), 0.81 for
  INCLUDE (36 tasks, LLM-RF bBPB) and 0.50 for XCOPA (8).
- **Along compute the safe cell is mostly the reference itself.**
  `safe_compute` is filled for every reading: a cheapest safe (proxy,
  checkpoint) cell exists for 0.53 of the accuracy tasks, 0.25 under
  bBPB → 1.7B accuracy and 0.88 under bBPB → 1.7B bBPB, but it is a proxy cell
  (at most 1B) for only 0.05, 0.08 and 0.14 of them; the median safe cell sits
  at 0.9, 0.8 and 0.8 of the reference run's compute, its own late checkpoints.

**Follow-ups**

- A per-variant table and figure of `safe_compute` and of DA-ckpt (the store
  now holds every checkpoint), split into proxy cells and reference
  checkpoints, so the compute ladder is read per variant and not only per task.
- Add the SNR and the k-fold noise per variant to the table, so a reader can
  also see which variant is cheapest to measure reliably.
- Weigh the evaluation cost of a variant (the LLM rewrite, a cloze
  evaluation's extra forward passes), not only its DA.

GitHub: [recipe_da_size_ladder_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.png) · [recipe_da_size_ladder_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_ladder_multi_axes.csv)

### 2. Every variant together and apart

The recommendation picks one variant per benchmark; this figure asks how
far apart the variants are when pooled, and whether the gap holds on the
cells both have.

<!-- BEGIN auto:variants (recipe.py --pool predictivity) -->
**Every variant together and apart** — mean DA-size per proxy over every task with a value (τ = 0.75; in brackets the tasks behind each mean, which differ across proxies and variants because the gate keeps different tasks at different sizes, rule 13):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.55 (155) | 0.57 (174) | 0.57 (194) | 0.60 (215) | 0.59 (236) |
| RF · accuracy → 1.7B accuracy | 0.48 (77) | 0.52 (90) | 0.48 (102) | 0.50 (121) | 0.56 (130) |
| LLM-RF · accuracy → 1.7B accuracy | 0.49 (69) | 0.49 (72) | 0.53 (76) | 0.50 (80) | 0.54 (84) |
| original · bBPB → 1.7B accuracy | 0.61 (255) | 0.59 (255) | 0.59 (255) | 0.61 (255) | 0.59 (255) |
| RF · bBPB → 1.7B accuracy | 0.59 (139) | 0.59 (139) | 0.59 (139) | 0.60 (139) | 0.60 (139) |
| LLM-RF · bBPB → 1.7B accuracy | 0.63 (89) | 0.60 (89) | 0.62 (89) | 0.57 (89) | 0.61 (89) |
| original · bBPB → 1.7B bBPB | 0.66 (552) | 0.60 (552) | 0.63 (552) | 0.62 (552) | 0.63 (552) |
| RF · bBPB → 1.7B bBPB | 0.65 (169) | 0.65 (169) | 0.64 (169) | 0.64 (169) | 0.66 (169) |
| LLM-RF · bBPB → 1.7B bBPB | 0.80 (95) | 0.77 (95) | 0.78 (95) | 0.71 (95) | 0.77 (95) |
| all variants, bBPB → 1.7B accuracy | 0.58 (784) | 0.57 (819) | 0.57 (855) | 0.58 (899) | 0.59 (933) |
| all variants, bBPB → 1.7B bBPB | 0.63 (1117) | 0.61 (1152) | 0.62 (1188) | 0.61 (1232) | 0.62 (1266) |

The same over the paired tasks (those whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix stripped, has an accuracy value at that proxy, so the gate treats every variant alike):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.55 (155) | 0.57 (174) | 0.57 (194) | 0.60 (215) | 0.59 (236) |
| RF · accuracy → 1.7B accuracy |  |  |  | n = 1 |  |
| LLM-RF · accuracy → 1.7B accuracy |  |  |  | n = 1 |  |
| original · bBPB → 1.7B accuracy | 0.64 (147) | 0.60 (165) | 0.60 (184) | 0.61 (204) | 0.60 (225) |
| RF · bBPB → 1.7B accuracy |  |  |  | n = 1 |  |
| LLM-RF · bBPB → 1.7B accuracy |  |  |  | n = 1 |  |
| original · bBPB → 1.7B bBPB | 0.71 (147) | 0.62 (165) | 0.68 (184) | 0.65 (204) | 0.65 (225) |
| RF · bBPB → 1.7B bBPB |  |  |  | n = 2 | n = 1 |
| LLM-RF · bBPB → 1.7B bBPB |  |  |  | n = 2 | n = 1 |
| all variants, bBPB → 1.7B accuracy | 0.60 (302) | 0.59 (339) | 0.58 (378) | 0.60 (423) | 0.60 (461) |
| all variants, bBPB → 1.7B bBPB | 0.63 (302) | 0.59 (339) | 0.62 (378) | 0.62 (425) | 0.62 (463) |

![Variants together and apart](pretraining/predictivity/recipe_da_size_variants_multi_axes.png)
<!-- END auto:variants -->

**Key findings** (`recipe_da_size_overview_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, gate `predictivity`; bBPB at all five proxies; each bullet names its target)

- **On the same truth, bBPB reads the reference about as well as accuracy.**
  Over every task, original bBPB → 1.7B accuracy is 0.61, 0.59, 0.59, 0.61
  and 0.59 from 90M to 1B (255 tasks) against the original accuracy's 0.55,
  0.57, 0.57, 0.60 and 0.59 (155–236 tasks); the lead is largest at 90M.
- **On the paired tasks the small lead holds**: 0.64, 0.60, 0.60, 0.61 and
  0.60 (147–225 tasks) against 0.55–0.60, so it is not only that bBPB is
  defined where accuracy is at chance.
- **The easier target reads higher.** bBPB → 1.7B bBPB gives the original
  0.60–0.66 (552 tasks), RF 0.64–0.66 (169) and LLM-RF 0.71–0.80 (95) over
  the five proxies, against 0.57–0.59 for all variants pooled under
  bBPB → 1.7B accuracy; part of bBPB's lead under that reading is the target,
  not the score.
- **The rewrites help through bBPB, not accuracy.** RF and LLM-RF accuracy
  sit at 0.48–0.56 and 0.49–0.54 over the five proxies, below the original
  accuracy's 0.55–0.60; their bBPB → 1.7B accuracy reads 0.59–0.60 (139
  tasks) and 0.57–0.63 (89); paired, they have one or two tasks each, at 600M
  and 1B only (n = 1–2).
- **Populations differ by target**: 255 original twins have a value under
  bBPB → 1.7B accuracy (gated on the original's accuracy at 1.7B) and 552
  under bBPB → 1.7B bBPB (never gated).

**Follow-ups**

- The paired panel per format, so the rewrites' head-to-head with their own
  accuracy is read on the same tasks.

GitHub: [recipe_da_size_variants_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.png) · [recipe_da_size_variants_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes.csv)

### 3. Per benchmark and variant

The pooled lines hide which benchmarks carry the gap; the heat map gives it
per benchmark and proxy.

<!-- BEGIN auto:heatmap (recipe.py --pool predictivity) -->
![Benchmark x variant heat map](pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png)
<!-- END auto:heatmap -->

**Key findings** (`recipe_da_size_by_variant_multi_axes.csv`: per benchmark, the mean DA-size at each proxy 90M–1B; benchmarks with ≥ 5 tasks in both variants at that proxy; each bullet names its target)

- **bBPB → 1.7B accuracy beats the original accuracy on 6 of 6 benchmarks at
  90M, 5 of 7 at 175M and 350M, 6 of 7 at 600M and 6 of 8 at 1B**, most on
  XStoryCloze at 175M (0.75 against 0.53), XCOPA at 90M–350M (0.62–0.65
  against 0.40–0.46) and HellaSwag at 90M (0.87 against 0.69). It is behind on
  INCLUDE v2 (EN) and MultiBLiMP at 175M, XNLI and XStoryCloze at 350M (0.51
  against 0.52, 0.73 against 0.76), HellaSwag at 600M (0.74 against 0.79) and
  XCOPA and HellaSwag at 1B (0.60 against 0.65, 0.81 against 0.83).
- **In the rewrites bBPB beats the same format's accuracy almost
  everywhere**: RF on 2 of 4 benchmarks at 90M, 3 of 4 at 175M, 4 of 5 at
  350M, 6 of 6 at 600M and 5 of 6 at 1B (Global-MMLU RF 0.68 against 0.44 at
  90M, but 0.68 against 0.69 at 1B), LLM-RF on 2 of 2 at every proxy
  (INCLUDE 0.66 against 0.57 at 1B).
- **bBPB → 1.7B bBPB is above bBPB → 1.7B accuracy on 10 of 10 original
  benchmarks at 90M, 350M and 1B, 8 of 10 at 175M and 7 of 10 at 600M**
  (behind at 600M: ARC 0.68 against 0.70, ARC (MT) 0.63 against 0.67, INCLUDE
  v2 (EN) tied at 0.50), so a heat map of bBPB → bBPB cells overstates what
  bBPB tells about the accuracy decision.

**Follow-ups**

- The same map on the paired cells only.

GitHub: [recipe_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png) · [recipe_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.csv)

### 4. Where each variant starts to read the reference

The heat map gives means; the profiles give the share of a benchmark's tasks
reliable at each proxy, the quantity the safe rank summarises.

<!-- BEGIN auto:profiles (recipe.py --pool predictivity) -->
![Reliability profiles per benchmark](pretraining/predictivity/recipe_da_size_profiles_multi_axes.png)
<!-- END auto:profiles -->

**Key findings** (`recipe_da_size_by_variant_multi_axes.csv`, `reliable_share_1B` over the benchmarks with a value at 1B; each bullet names its target)

- **Few benchmarks have half their tasks reliable at the 1B proxy.** Original
  accuracy: 2 of 15 (HellaSwag, LAMBADA); original bBPB → 1.7B accuracy: 3 of
  16 (HellaSwag, XStoryCloze, Global PIQA non-parallel on 2 tasks); original
  bBPB → 1.7B bBPB: 4 of 28 (those three and XCOPA).
- **No RF variant gets there under either target** (0 of 8), and of the two
  LLM-RF benchmarks only INCLUDE does, with bBPB → 1.7B bBPB.

**Follow-ups**

- Mark the proxy where each recommended variant first passes half of its
  tasks, to read the profiles at a glance.

GitHub: [recipe_da_size_profiles_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.png) · [recipe_da_size_profiles_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_profiles_multi_axes.csv)

### 5. The best benchmarks along size and training (paper figure)

The profiles read one decision accuracy at final checkpoints; this figure
follows the eight best benchmarks through all three, so a variant that wins at
the 90M final can be checked early in the run too.

**DA-size, DA-ckpt, DA-goal · no reliability filter · multi-axis pairs · pairs
from `predictivity` at seed 1904 · gate `predictivity`** (accuracy at the proxy
and 1.7B, DA-ckpt at the proxy only; bBPB → 1.7B accuracy on the accuracy of
the same items at 1.7B; bBPB → 1.7B bBPB never gated). Snapshot: the
2026-10-07 15:51 ladder report, the one rq02's tables on disk were computed on.
Rows: Overall (BPB of the 50 trained languages, accuracy over every format,
every twin against the 1.7B bBPB), then the eight benchmarks with the highest
`mean_da_size` (`recipe_da_size_by_variant_multi_axes.csv`: mean over tasks
per proxy, then over the proxies 90M–1B with a value), the better of the
original and RF format, among the formats with ≥ 5 ranked tasks (the ranking
is the CSV's `ranking` rows). Columns: DA-size per proxy; DA-ckpt (rq02's
`da_all_pooled_per_task_multi_axes.csv`, `da_own`) and DA-goal
(`recipe_da_goal_per_task_multi_axes.csv`) per tenth, each task averaged over
the proxies 90M–1B first. The bBPB lines are the twins of the ranking format;
bBPB → 1.7B accuracy has no DA-ckpt (its reference would be the proxy's own
accuracy, another score). Bands: 95 % bootstrap over the tasks of a line
(1,000 resamples); a point on < 5 tasks is not drawn. Task counts differ
across lines and, for accuracy, across proxies (the gate; the CSV's `n_tasks`).

![DA-size, DA-ckpt and DA-goal per benchmark](pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.png)

**Key findings** (`recipe_da_all_by_benchmark_multi_axes_paper.csv`)

- **Ranking**: HellaSwag 0.76, LAMBADA 0.72, XStoryCloze 0.66, MultiBLiMP
  0.65, ARC (MT) 0.58 (original), Global-MMLU 0.56 and INCLUDE 0.54 (RF),
  INCLUDE v2 (OG) 0.54 (original); next XWinograd 0.53. INCLUDE and INCLUDE
  v2 (OG) differ in the fourth decimal.
- **At the 90M proxy bBPB → 1.7B accuracy beats the benchmark's accuracy on
  4 of the 5 benchmarks where both have a value** (HellaSwag 0.87 against
  0.69, Global-MMLU 0.68 against 0.44 for RF, INCLUDE v2 (OG) 0.60 against
  0.47, MultiBLiMP 0.66 against 0.63; INCLUDE ties, 0.63 against 0.63 for RF).
  At 1B accuracy catches up on HellaSwag (0.83 against 0.81) and Global-MMLU
  (0.69 against 0.68).
- **DA-goal keeps that order along the run**: bBPB → 1.7B accuracy is above
  every accuracy format at 80–100 % of the tenths on the seven benchmarks with
  a twin. HellaSwag accuracy climbs from 0.54 at 0.5C to 0.75 at 5C, still
  under its twin (0.77–0.82).
- **DA-ckpt is highest for bBPB → 1.7B bBPB on every benchmark and tenth**
  (0.06–0.30 above the best accuracy format), as in rq02: a twin persists
  within its own run more than accuracy does, which is not a decision.
- **Overall**, DA-size on accuracy goes from 0.52 at 90M to 0.57 at 1B, the
  bBPB twins stay at 0.63–0.67 and BPB at 0.74–0.96; DA-goal reads 0.51–0.55,
  0.63–0.66 and 0.84–0.90.

**Follow-ups**

- The same figure on mono-axis pairs (the rq2 pair set), once the bBPB → 1.7B
  accuracy cells are read there too.
- A per-language twin of a row (e.g. HellaSwag), to see whether the bBPB lead
  at 90M is a few languages or all of them.

GitHub: [recipe_da_all_by_benchmark_multi_axes_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.png) · [recipe_da_all_by_benchmark_multi_axes_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_all_by_benchmark_multi_axes_paper.csv)

## The key figure on one task set per line

A line averages, at each proxy size, the tasks that pass the gate there, so it can move because its tasks changed (rule 13). The twin keeps, per line, the tasks with a value at every size (solid), the committed line dashed behind.

![Mean DA-size per variant, one task set per line](pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.png)

DA-size against 1.7B, multi-axis pairs, gate `predictivity`, no reliability filter. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_fixed_tasks_paper.csv)

Key findings:

- **The pooled line loses its zigzag.** On the 296 tasks it has at every size it reads 0.52, 0.55, 0.54, 0.56, 0.59; the committed 0.57 at 175M is the bBPB variants entering at 175M and 1B only.
- **Original accuracy rises a little more on one task set.** Its 152 tasks read 0.55 → 0.62 from 90M to 1B, against 0.55 → 0.59 on 153–237 tasks; the tasks that join at larger sizes rank worse.
- **The verdict does not change.** No line reaches τ = 0.75 on either task set, and the bBPB lines (the same tasks at 175M and 1B) do not move.

Follow-ups:

- **The recommendation table on fixed tasks.** The per-benchmark pick reads the same moving populations.

## Files

- `pretraining/<pool>/recipe_da_all_per_task{_multi_axes,_mono_axis}.csv` —
  per task and reading (`reading`: acc_acc, bbpb_acc, bbpb_bbpb): variant,
  DA-size per proxy, gate flags, safe size, safe rank and `safe_compute`
  (-1 = no cell from which every costlier one clears τ; empty for a task with
  no ungated cell, every reading on the full checkpoint grid since the
  2026-10-07 refresh), the per-(benchmark, language)
  recommendation per bBPB reading (`recommended_bbpb_acc`,
  `recommended_bbpb_bbpb`).
- `…/recipe_da_size_by_variant*.csv` — benchmark × variant per bBPB reading
  (`bbpb_reading`), with `languages_ranked` and `n_tasks`;
  `recipe_da_size_recommendation*.csv` — one row per benchmark and bBPB
  reading, with `decided_by`; `recipe_da_size_overview*.csv` — per variant
  and pooled, every task and paired, with the task counts.
- `…/recipe_da_size_{variants,heatmap,ladder,profiles}*.png/.csv` — the
  figures, each with its table.
- `…/recipe_da_goal_per_task*.csv` — the early-small cells every table reads:
  per task, reading, proxy and tenth, rule 5 applied, the gate verdict in
  `gated` (DA-size is its `frac` = 1.0 rows).
- `…/recipe_da_all_by_benchmark_multi_axes_paper.png/.csv` — the paper's
  per-benchmark figure (section 5): per row, line and DA kind the mean, the
  95 % bootstrap band, `n_tasks` and `drawn`; the benchmark ranking in the
  `ranking` rows.
