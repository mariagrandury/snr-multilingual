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
**2026-10-06 04:26** (the full refresh of 2026-10-07, outputs written 08:26).

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
- **The bBPB twins** come from the per-item store at final checkpoints only,
  so every table and figure here is DA-size. In these outputs a twin has a
  value at the **175M and 1B proxies only** (90M, 350M and 600M are empty for
  every bBPB row), so a bBPB safe size is 175M, 1B or never, decided on two
  proxies where an accuracy variant's is decided on five.
- **Twin coverage**: a twin exists only where the store holds the run, so a
  twin's cell with fewer pairs than its original's is left blank (the run
  prints how many) until the store covers the pool.

## Key figure

![Mean DA-size per way of evaluating a benchmark](pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png)

Population: DA-size against the 1.7B final, multi-axis pairs of `predictivity` (seed 1904), gate `predictivity` at the proxy and the reference (a bBPB task has no chance level and passes), ≥ 3 pairs, no filter, every task (153–237 original, 76–130 RF and 69–84 LLM-RF accuracy tasks per proxy; 255 original, 139 RF and 89 LLM-RF bBPB twins; 298–934 tasks pooled under "all variants"); bBPB read against the 1.7B accuracy, at final checkpoints of the 175M and 1B proxies only; a point on fewer than 5 tasks is not drawn.

**Key finding.** No way of evaluating a benchmark reads the reference's accuracy decision at τ = 0.75 on average by 1B: the best mean is 0.61 (LLM-RF bBPB at 1B; only the easier bBPB → 1.7B bBPB target, LLM-RF at 0.77, clears it, [section 2](#2-every-variant-together-and-apart)). Against the same truth, bBPB matches or edges the original accuracy at the two proxies where both have values (every task: 0.59 against 0.57 at 175M, 0.59 and 0.59 at 1B; paired: 0.60 and 0.60), and the rewrites help only when scored by bBPB (RF and LLM-RF accuracy 0.48–0.56 across the five proxies, their bBPB 0.59–0.61).

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
  at every larger proxy) and the safe compute share (empty while a task has
  final checkpoints only).
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
DA-size against the 1.7B final, multi-axis pairs (rule 15), pool `predictivity`, rq02's early-small table written 2026-10-07 05:11, ≥ 3 pairs; reliable = DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every larger proxy. Every row says what it predicts: an accuracy variant its accuracy at 1.7B (gate `predictivity` at the proxy and the reference); a bBPB variant either its original's accuracy at 1.7B (**bBPB → 1.7B accuracy**, gated on that accuracy at the reference only, the reading of `bench_bpb_da.py`) or its own bBPB at 1.7B (**bBPB → 1.7B bBPB**, no chance level, never gated). 1662 tasks over 31 benchmarks. The bBPB twins are read at final checkpoints only for now. A mean over fewer than 5 tasks is left blank (`n < 5`) in the tables below and the figures; it stays in the CSVs. Regenerate with `python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity`.

**How the pick is made**, per bBPB reading and scoring of the pick (benchmarks with a value; reliable somewhere = mean safe rank below 5, a task of the pick safe from some proxy on; never reliable = 5.00, no task of the pick ever safe):

| bBPB read | pick | benchmarks | reliable somewhere | never reliable | won on the DA-size tie-break | only variant with a value | no accuracy variant has a value |
|---|---|---|---|---|---|---|---|
| bBPB → 1.7B accuracy | bBPB | 16 | 9 | 7 | 6 | 1 | 1 |
| bBPB → 1.7B accuracy | accuracy | 7 | 5 | 2 | 1 | 2 | 0 |
| bBPB → 1.7B bBPB | bBPB | 25 | 15 | 10 | 3 | 8 | 8 |
| bBPB → 1.7B bBPB | accuracy | 5 | 2 | 3 | 2 | 2 | 0 |

**The recommendation** (per benchmark and bBPB reading: the variant with the smallest mean safe rank over its ranked tasks — 0 = safe from 90M, 4 = from 1B, 5 = never; a task gated or under the pair minimum at every proxy has no rank, so the ranked languages and tasks are counted beside it (rule 13) — then the higher mean DA-size over the (task, proxy) cells every tied variant has, given in brackets; τ = 0.75), ordered by the reading with bBPB → 1.7B accuracy:

| benchmark | bBPB read | evaluate it as | decided by | mean safe rank | median safe size | share safe by 1B | languages ranked / all | tasks ranked | mean DA-size |
|---|---|---|---|---|---|---|---|---|---|
| HellaSwag | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 1.80 | 175M | 0.80 | 25 / 26 | 25 | 0.84 |
| HellaSwag | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.12 | 175M | 1.00 | 26 / 26 | 26 | 0.89 |
| XStoryCloze | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.88 | 350M | 0.62 | 8 / 8 | 8 | 0.76 |
| XStoryCloze | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.38 | 175M | 1.00 | 8 / 8 | 8 | 0.88 |
| Global PIQA (non-parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 600M | 0.50 | 2 / 2 | 2 | n < 5 |
| Global PIQA (non-parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 175M | 1.00 | 2 / 2 | 2 | n < 5 |
| MultiBLiMP | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 3.79 | never | 0.32 | 34 / 34 | 34 | 0.65 |
| MultiBLiMP | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.03 | never | 0.29 | 34 / 34 | 34 | 0.65 |
| LAMBADA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| LAMBADA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| Global-MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.00 | never | 0.28 | 29 / 29 | 29 | 0.69 |
| Global-MMLU | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.10 | never | 0.28 | 29 / 29 | 29 | 0.66 |
| INCLUDE | bBPB → 1.7B accuracy | LLM-RF · bBPB → 1.7B accuracy | safe rank | 4.37 | never | 0.23 | 30 / 36 | 30 | 0.64 |
| INCLUDE | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 2.11 | 175M | 0.81 | 36 / 36 | 36 | 0.81 |
| Belebele | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.49 | never | 0.14 | 49 / 49 | 57 | 0.59 |
| Belebele | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 3.47 | never | 0.46 | 49 / 49 | 59 | 0.74 |
| ARC | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.56 | never | 0.25 | 15 / 27 | 16 | 0.65 |
| ARC | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.86 | never | 0.39 | 27 / 27 | 28 | 0.67 |
| INCLUDE v2 (OG) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.67 | never | 0.11 | 35 / 42 | 55 | 0.58 |
| INCLUDE v2 (OG) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.66 | never | 0.14 | 42 / 42 | 77 | 0.62 |
| PAWS-X | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | n < 5 | never | 0.25 | 4 / 8 | 4 | n < 5 |
| PAWS-X | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.00 | never | 0.25 | 8 / 8 | 8 | 0.69 |
| XCOPA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.75 | never | 0.25 | 8 / 8 | 8 | 0.49 |
| XCOPA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.75 | 1B | 0.50 | 8 / 8 | 8 | 0.74 |
| ARC (MT) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.90 | never | 0.10 | 10 / 11 | 10 | 0.58 |
| ARC (MT) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.73 | never | 0.27 | 11 / 11 | 11 | 0.63 |
| INCLUDE v2 (EN) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 67 | 0.48 |
| INCLUDE v2 (EN) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.95 | never | 0.01 | 42 / 42 | 77 | 0.47 |
| MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MMLU | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| XNLI | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 14 / 15 | 14 | 0.53 (shared cells: 0.54) |
| XNLI | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.87 | never | 0.13 | 15 / 15 | 15 | 0.68 |
| XWinograd | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.53 |
| XWinograd | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.53 |
| MathQA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| ACP-Bench (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 5 | 0.49 (shared cells: 0.51) |
| ACP-Bench (MCQ) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.54 (shared cells: 0.53) |
| BBH (MCQ) | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 10 | 0.45 (shared cells: 0.46) |
| BBH (MCQ) | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | DA-size tie-break | 5.00 | never | 0.00 | 1 / 1 | 17 | 0.53 (shared cells: 0.53) |
| Global PIQA (parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 46 | 1 | n < 5 |
| Global PIQA (parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 3.84 | never | 0.40 | 46 / 46 | 63 | 0.73 |
| CulturalBench-easy | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 5 / 8 | 7 | 0.43 (shared cells: 0.40) |
| CulturalBench-easy | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.89 | never | 0.11 | 8 / 8 | 19 | 0.56 |
| ACP-Bench (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ACP-Bench (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 7 | 0.52 |
| BBH (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BBH (cloze) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 1 / 1 | 6 | 0.48 |
| BLEnD | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 4 | 0 |  |
| BLEnD | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 5.00 | never | 0.00 | 4 / 4 | 5 | 0.46 |
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

**Key findings** (`recipe_da_size_recommendation_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, τ = 0.75, bBPB at final checkpoints of 175M and 1B only; each bullet names the target it reads)

- **Against the same truth (bBPB → 1.7B accuracy), bBPB is the pick for 16
  of the 23 benchmarks with a value and accuracy for 7** (8 of 31 have none).
  Nine bBPB picks and five accuracy picks are reliable somewhere (mean safe
  rank below 5); six bBPB picks win only on the DA-size tie-break.
- **The pick is not like for like yet.** A bBPB task is ranked on two proxies
  (175M, 1B) and an accuracy task on five, so a bBPB pick can be safe "from
  175M" while dipping below τ at 350M–600M unseen; the comparison waits for
  twins at every proxy.
- **Against the easier target (bBPB → 1.7B bBPB), bBPB takes 25 of the 30
  benchmarks with a value**, 8 of them with no accuracy variant at all.
  ARC (MT), INCLUDE v2 (EN), PAWS-X and XCOPA switch from accuracy to bBPB
  with the target, MultiBLiMP and MMLU from bBPB to accuracy, Belebele from
  RF to LLM-RF bBPB and ACP-Bench (MCQ) from RF to original bBPB (the latter
  on the DA-size tie-break).
- **Few benchmarks are safe by 1B in most languages.** With bBPB → 1.7B
  accuracy, the share of ranked tasks safe by 1B is 0.80 for HellaSwag (25
  tasks), 0.62 for XStoryCloze (8, bBPB) and 1.00 for LAMBADA (5, accuracy);
  every other benchmark is at 0.50 or below. With bBPB → 1.7B bBPB it is 1.00
  for HellaSwag, XStoryCloze, Global PIQA non-parallel (2 tasks) and LAMBADA
  (5, accuracy), 0.81 for
  INCLUDE (36 tasks, LLM-RF bBPB) and 0.50 for XCOPA (8).
- **Twin coverage.** A bBPB cell with fewer pairs than its original is blank
  until the store covers the pool; the run prints how many.

**Follow-ups**

- Comparing DA-goal and DA-ckpt across the variants needs new code here, a
  per-variant table and figure of the cheapest safe (proxy, checkpoint) and of
  DA-ckpt, beside the store rebuild over every checkpoint
  (`build_per_item_store.sbatch --pool predictivity`, then `--bench-bpb`).
- Re-read every bBPB pick once the twins have a value at 90M, 350M and 600M
  too, so bBPB and accuracy are ranked on the same five proxies.
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
| original · accuracy → 1.7B accuracy | 0.55 (153) | 0.57 (175) | 0.57 (192) | 0.60 (211) | 0.59 (237) |
| RF · accuracy → 1.7B accuracy | 0.48 (76) | 0.52 (91) | 0.48 (102) | 0.50 (121) | 0.56 (130) |
| LLM-RF · accuracy → 1.7B accuracy | 0.49 (69) | 0.49 (72) | 0.53 (77) | 0.50 (81) | 0.54 (84) |
| original · bBPB → 1.7B accuracy |  | 0.59 (255) |  |  | 0.59 (255) |
| RF · bBPB → 1.7B accuracy |  | 0.59 (139) |  |  | 0.60 (139) |
| LLM-RF · bBPB → 1.7B accuracy |  | 0.60 (89) |  |  | 0.61 (89) |
| original · bBPB → 1.7B bBPB |  | 0.60 (552) |  |  | 0.63 (552) |
| RF · bBPB → 1.7B bBPB |  | 0.65 (169) |  |  | 0.66 (169) |
| LLM-RF · bBPB → 1.7B bBPB |  | 0.77 (95) |  |  | 0.77 (95) |
| all variants, bBPB → 1.7B accuracy | 0.52 (298) | 0.57 (821) | 0.54 (371) | 0.55 (413) | 0.59 (934) |
| all variants, bBPB → 1.7B bBPB | 0.52 (298) | 0.61 (1154) | 0.54 (371) | 0.55 (413) | 0.62 (1267) |

The same over the paired tasks (those whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix stripped, has an accuracy value at that proxy, so the gate treats every variant alike):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.55 (153) | 0.57 (175) | 0.57 (192) | 0.60 (211) | 0.59 (237) |
| original · bBPB → 1.7B accuracy |  | 0.60 (165) |  |  | 0.60 (226) |
| original · bBPB → 1.7B bBPB |  | 0.62 (165) |  |  | 0.65 (226) |
| RF · bBPB → 1.7B bBPB |  |  |  |  | n = 1 |
| LLM-RF · bBPB → 1.7B bBPB |  |  |  |  | n = 1 |
| all variants, bBPB → 1.7B accuracy | 0.55 (153) | 0.59 (340) | 0.57 (192) | 0.60 (211) | 0.60 (463) |
| all variants, bBPB → 1.7B bBPB | 0.55 (153) | 0.59 (340) | 0.57 (192) | 0.60 (211) | 0.62 (465) |

![Variants together and apart](pretraining/predictivity/recipe_da_size_variants_multi_axes.png)
<!-- END auto:variants -->

**Key findings** (`recipe_da_size_overview_multi_axes.csv`: DA-size, multi-axis pairs, pool `predictivity`, gate `predictivity`; bBPB at 175M and 1B only; each bullet names its target)

- **On the same truth, bBPB reads the reference about as well as accuracy.**
  Over every task, original bBPB → 1.7B accuracy is 0.59 at 175M and 0.59 at
  1B (255 tasks) against the original accuracy's 0.57 (175 tasks) and 0.59
  (237); nothing can be said yet about 90M, 350M or 600M.
- **On the paired tasks the small lead holds**: 0.60 at 175M (165 tasks) and
  0.60 at 1B (226) against 0.57 and 0.59, so it is not only that bBPB is
  defined where accuracy is at chance.
- **The easier target reads higher.** bBPB → 1.7B bBPB gives the original
  0.60 and 0.63 (552 tasks), RF 0.65 and 0.66 (169), LLM-RF 0.77 and 0.77
  (95) at 175M and 1B, against 0.57 and 0.59 for all variants pooled under
  bBPB → 1.7B accuracy; part of bBPB's lead under that reading is the target,
  not the score.
- **The rewrites help through bBPB, not accuracy.** RF and LLM-RF accuracy
  sit at 0.48–0.56 and 0.49–0.54 over the five proxies, below the original
  accuracy's 0.55–0.60; their bBPB → 1.7B accuracy reads 0.59–0.60 (139
  tasks) and 0.60–0.61 (89); paired, they have only one task each, under
  bBPB → 1.7B bBPB at 1B (n = 1).
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

**Key findings** (`recipe_da_size_by_variant_multi_axes.csv`: per benchmark, the mean DA-size at the 175M and 1B proxies, the only two where bBPB has a value; benchmarks with ≥ 5 tasks in both variants at that proxy; each bullet names its target)

- **bBPB → 1.7B accuracy beats the original accuracy on 5 of 7 benchmarks at
  175M and 6 of 8 at 1B**, most on XStoryCloze (0.75 against 0.53 at 175M,
  0.77 against 0.69 at 1B) and XCOPA at 175M (0.65 against 0.43). It is
  behind on INCLUDE v2 (EN) and MultiBLiMP at 175M (0.43 against 0.47, 0.65
  against 0.66) and on XCOPA and HellaSwag at 1B (0.60 against 0.65, 0.81
  against 0.83).
- **In the rewrites bBPB beats the same format's accuracy almost
  everywhere**: RF on 3 of 4 benchmarks at 175M and 5 of 6 at 1B (Global-MMLU
  RF 0.69 against 0.57 at 175M, but 0.68 against 0.69 at 1B), LLM-RF on 2 of 2
  at both (INCLUDE 0.66 against 0.57 at 1B).
- **bBPB → 1.7B bBPB is above bBPB → 1.7B accuracy on 8 of 10 original
  benchmarks at 175M and 10 of 10 at 1B** (behind at 175M: ARC (MT) 0.60
  against 0.62, INCLUDE v2 (EN) 0.41 against 0.43), so a heat map of bBPB → bBPB cells
  overstates what bBPB tells about the accuracy decision.

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

## Files

- `pretraining/<pool>/recipe_da_all_per_task{_multi_axes,_mono_axis}.csv` —
  per task and reading (`reading`: acc_acc, bbpb_acc, bbpb_bbpb): variant,
  DA-size per proxy, gate flags, safe size, safe rank and `safe_compute`
  (-1 = no cell from which every costlier one clears τ; empty when only final
  checkpoints exist, bBPB for now), the per-(benchmark, language)
  recommendation per bBPB reading (`recommended_bbpb_acc`,
  `recommended_bbpb_bbpb`).
- `…/recipe_da_size_by_variant*.csv` — benchmark × variant per bBPB reading
  (`bbpb_reading`), with `languages_ranked` and `n_tasks`;
  `recipe_da_size_recommendation*.csv` — one row per benchmark and bBPB
  reading, with `decided_by`; `recipe_da_size_overview*.csv` — per variant
  and pooled, every task and paired, with the task counts.
- `…/recipe_da_size_{variants,heatmap,ladder,profiles}*.png/.csv` — the
  figures, each with its table.
