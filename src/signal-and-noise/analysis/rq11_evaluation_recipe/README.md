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

- **Pool** `predictivity` (seed 1904, every cell: each scheme, T, ladder and
  architecture; 90M–1B proxies against the 1.7B reference), the gate
  `predictivity`, both pair sets (multi-axis in the figures below, mono-axis
  in the `_mono_axis` twins). The snapshot is the one the results block names.
- **Input**: [decision accuracy](../rq02_decision_accuracy/README.md#3-early-and-small-da-goal-and-da-ckpt-on-the-ten-checkpoints)'s
  per-task table `da_goal_early_small_per_task_both_axes.csv`, plus the bBPB
  rows read against accuracy, computed here by the same kernel on the same
  frame, pairs and grid. `recipe.py` stops when that table was computed on
  another population than the pool loads now (its largest pair count differs
  from the pool's), so the two kinds of rows are never mixed. The outputs on
  disk today read a recomputation of that table on the current pool made
  outside the driver (`compute_da.py`'s kernel, benchmark tasks only), because
  the committed rq02 table predates the four-pool layout; the next
  regeneration replaces both.
- **Variants**: each task is one variant of one benchmark in one language,
  its `format` (original, RF, LLM-RF) and `scoring` (accuracy, bBPB) the
  columns the loader writes (`utils.variant`).
- **The bBPB twins** come from the per-item store, which holds each run's
  final checkpoint only, so every table and figure here is DA-size. A store
  that covers every checkpoint fills the twins' DA-goal and DA-ckpt cells in
  decision accuracy's tables; here it changes only the `safe_compute` column.
- **Twin coverage**: a twin exists only where the store holds the run, so a
  twin's cell with fewer pairs than its original's is left blank (the run
  prints how many) until the store covers the pool.

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
DA-size against the 1.7B final, multi-axis pairs (rule 15), pool `predictivity`, rq02's early-small table written 2026-10-07 01:53, ≥ 3 pairs; reliable = DA-size ≥ τ = 0.75 (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every larger proxy. Every row says what it predicts: an accuracy variant its accuracy at 1.7B (gate `predictivity` at the proxy and the reference); a bBPB variant either its original's accuracy at 1.7B (**bBPB → 1.7B accuracy**, gated on that accuracy at the reference only, the reading of `bench_bpb_da.py`) or its own bBPB at 1.7B (**bBPB → 1.7B bBPB**, no chance level, never gated). 1564 tasks over 31 benchmarks. The bBPB twins are read at final checkpoints only for now. A mean over fewer than 5 tasks is left blank (`n < 5`) in the tables below and the figures; it stays in the CSVs. Regenerate with `python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity`.

**How the pick is made**, per bBPB reading and scoring of the pick (benchmarks with a value; reliable somewhere = mean safe rank below 5, a task of the pick safe from some proxy on; never reliable = 5.00, no task of the pick ever safe):

| bBPB read | pick | benchmarks | reliable somewhere | never reliable | won on the DA-size tie-break | only variant with a value | no accuracy variant has a value |
|---|---|---|---|---|---|---|---|
| bBPB → 1.7B accuracy | bBPB | 11 | 9 | 2 | 2 | 0 | 0 |
| bBPB → 1.7B accuracy | accuracy | 11 | 5 | 6 | 0 | 7 | 0 |
| bBPB → 1.7B bBPB | bBPB | 17 | 14 | 3 | 0 | 4 | 4 |
| bBPB → 1.7B bBPB | accuracy | 9 | 3 | 6 | 0 | 7 | 0 |

**The recommendation** (per benchmark and bBPB reading: the variant with the smallest mean safe rank over its ranked tasks — 0 = safe from 90M, 4 = from 1B, 5 = never; a task gated or under the pair minimum at every proxy has no rank, so the ranked languages and tasks are counted beside it (rule 13) — then the higher mean DA-size over the (task, proxy) cells every tied variant has, given in brackets; τ = 0.75), ordered by the reading with bBPB → 1.7B accuracy:

| benchmark | bBPB read | evaluate it as | decided by | mean safe rank | median safe size | share safe by 1B | languages ranked / all | tasks ranked | mean DA-size |
|---|---|---|---|---|---|---|---|---|---|
| Global PIQA (non-parallel) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | n < 5 | 350M | 0.50 | 2 / 2 | 2 | n < 5 |
| Global PIQA (non-parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | n < 5 | 90M | 1.00 | 2 / 2 | 2 | n < 5 |
| HellaSwag | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.54 | 1B | 0.79 | 24 / 25 | 24 | 0.82 |
| HellaSwag | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.36 | 90M | 1.00 | 25 / 25 | 25 | 0.87 |
| XStoryCloze | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 2.57 | 1B | 0.71 | 7 / 7 | 7 | 0.75 |
| XStoryCloze | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 1.14 | 90M | 1.00 | 7 / 7 | 7 | 0.89 |
| MultiBLiMP | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 3.67 | never | 0.33 | 33 / 33 | 33 | 0.66 |
| MultiBLiMP | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.03 | never | 0.29 | 34 / 34 | 34 | 0.65 |
| LAMBADA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| LAMBADA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 3.80 | 1B | 1.00 | 5 / 5 | 5 | 0.72 |
| Global-MMLU | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.00 | never | 0.29 | 28 / 28 | 28 | 0.68 |
| Global-MMLU | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.50 | never | 0.29 | 28 / 28 | 28 | 0.64 |
| ARC | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.38 | never | 0.23 | 13 / 26 | 13 | 0.65 |
| ARC | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.85 | never | 0.42 | 26 / 26 | 26 | 0.68 |
| Belebele | bBPB → 1.7B accuracy | RF · bBPB → 1.7B accuracy | safe rank | 4.57 | never | 0.14 | 48 / 48 | 56 | 0.58 |
| Belebele | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 3.93 | never | 0.47 | 48 / 48 | 58 | 0.75 |
| INCLUDE | bBPB → 1.7B accuracy | LLM-RF · bBPB → 1.7B accuracy | safe rank | 4.58 | never | 0.23 | 31 / 36 | 31 | 0.63 |
| INCLUDE | bBPB → 1.7B bBPB | LLM-RF · bBPB → 1.7B bBPB | safe rank | 2.25 | 600M | 0.81 | 36 / 36 | 36 | 0.80 |
| INCLUDE v2 (OG) | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | safe rank | 4.78 | never | 0.09 | 35 / 42 | 55 | 0.59 |
| INCLUDE v2 (OG) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.66 | never | 0.14 | 42 / 42 | 77 | 0.64 |
| PAWS-X | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.80 | never | 0.20 | 5 / 8 | 5 | 0.58 |
| PAWS-X | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.57 | never | 0.29 | 7 / 7 | 7 | 0.72 |
| XCOPA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.88 | never | 0.12 | 8 / 8 | 8 | 0.49 |
| XCOPA | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 3.50 | 1B | 0.50 | 8 / 8 | 8 | 0.75 |
| ARC (MT) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.90 | never | 0.10 | 10 / 11 | 10 | 0.58 |
| ARC (MT) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.55 | never | 0.27 | 11 / 11 | 11 | 0.64 |
| INCLUDE v2 (EN) | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 68 | 0.48 |
| INCLUDE v2 (EN) | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | safe rank | 4.97 | never | 0.03 | 41 / 42 | 68 | 0.48 |
| MMLU | bBPB → 1.7B accuracy | RF · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MMLU | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B accuracy | RF · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| CommonsenseQA | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| XNLI | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | 5.00 | never | 0.00 | 13 / 14 | 13 | 0.53 (shared cells: 0.56) |
| XNLI | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | safe rank | 4.64 | never | 0.14 | 14 / 14 | 14 | 0.70 |
| XWinograd | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.52 |
| XWinograd | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 6 / 6 | 6 | 0.52 |
| MathQA | bBPB → 1.7B accuracy | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| MathQA | bBPB → 1.7B bBPB | original · accuracy → 1.7B accuracy | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |
| BBH (MCQ) | bBPB → 1.7B accuracy | RF · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 1 / 1 | 9 | 0.47 |
| BBH (MCQ) | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 1 / 1 | 9 | 0.47 |
| CulturalBench-easy | bBPB → 1.7B accuracy | original · bBPB → 1.7B accuracy | DA-size tie-break | n < 5 | never | 0.00 | 1 / 7 | 1 | n < 5 |
| CulturalBench-easy | bBPB → 1.7B bBPB | RF · bBPB → 1.7B bBPB | safe rank | 4.92 | never | 0.08 | 7 / 7 | 12 | 0.53 |
| ACP-Bench (MCQ) | bBPB → 1.7B accuracy | RF · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 1 / 1 | 5 | 0.43 |
| ACP-Bench (MCQ) | bBPB → 1.7B bBPB | RF · accuracy → 1.7B accuracy | only variant | 5.00 | never | 0.00 | 1 / 1 | 5 | 0.43 |
| ACP-Bench (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ACP-Bench (cloze) | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BBH (cloze) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BBH (cloze) | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| BLEnD | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 4 | 0 |  |
| BLEnD | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 3 / 3 | 4 | n < 5 |
| CulturalBench-hard | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| CulturalBench-hard | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 8 | 0 |  |
| Global PIQA (parallel) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 46 | 0 |  |
| Global PIQA (parallel) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | 3.77 | never | 0.40 | 45 / 45 | 62 | 0.74 |
| OpenBookQA | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| OpenBookQA | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ToxiGen | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| ToxiGen | bBPB → 1.7B bBPB | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 1 | 0 |  |
| TruthfulQA (mc2) | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 3 | 0 |  |
| TruthfulQA (mc2) | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 2 / 2 | 2 | n < 5 |
| TruthfulQA-Multi | bBPB → 1.7B accuracy | — (no variant has a value: every cell at chance or under the pair minimum) |  |  |  |  | 0 / 2 | 0 |  |
| TruthfulQA-Multi | bBPB → 1.7B bBPB | original · bBPB → 1.7B bBPB | only variant | n < 5 | never | 0.00 | 1 / 1 | 1 | n < 5 |

![The cheapest reliable proxy](pretraining/predictivity/recipe_da_size_ladder_multi_axes.png)
<!-- END auto:results -->

**Key findings** (DA-size, multi-axis pairs, τ = 0.75, bBPB at final checkpoints; each bullet names the target it reads)

- **Against the same truth (bBPB → 1.7B accuracy), bBPB and accuracy split
  the benchmarks.** 22 of the 31 benchmarks have a value; bBPB is the pick for
  11 and accuracy for 11. Nine of the bBPB picks are reliable somewhere (a
  mean safe rank below 5) and two win the DA-size tie-break (Cultural Bench
  easy, XNLI); seven of the accuracy picks are the only variant with a value,
  because the twin is gated where its original is at chance at 1.7B.
- **Against the easier target (bBPB → 1.7B bBPB), bBPB takes 17 of 26.** Four
  of those have no accuracy variant at all (BLEnD, Global PIQA parallel, both
  TruthfulQA tasks). ARC multilingual, PAWS and XCOPA switch from accuracy to
  bBPB with the target, and MultiBLiMP from bBPB to accuracy.
- **The median language is safe by 1B for few benchmarks.** With bBPB →
  1.7B accuracy: HellaSwag and XStoryCloze (bBPB) and LAMBADA (accuracy) from
  1B, Global PIQA non-parallel from 350M on 2 tasks. With bBPB → 1.7B bBPB:
  HellaSwag, XStoryCloze and Global PIQA non-parallel from 90M, INCLUDE
  (LLM-RF bBPB) from 600M, XCOPA and LAMBADA from 1B.
- **Twin coverage.** 1,960 of the 16,320 bBPB cells have fewer pairs than
  their original (the six L1 FineWeb cells have no twin yet) and are blank;
  they return once the store covers the pool.

**Follow-ups**

- Comparing DA-goal and DA-ckpt across the variants needs new code here, a
  per-variant table and figure of the cheapest safe (proxy, checkpoint) and of
  DA-ckpt, beside the store rebuild over every checkpoint
  (`build_per_item_store.sbatch --pool predictivity`, then `--bench-bpb`);
  the rebuild alone moves only `safe_compute`.
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
| original · accuracy → 1.7B accuracy | 0.55 (157) | 0.57 (180) | 0.57 (195) | 0.60 (214) | 0.59 (240) |
| RF · accuracy → 1.7B accuracy | 0.49 (77) | 0.51 (91) | 0.47 (106) | 0.49 (116) | 0.55 (128) |
| LLM-RF · accuracy → 1.7B accuracy | 0.49 (68) | 0.49 (70) | 0.53 (76) | 0.50 (81) | 0.54 (85) |
| original · bBPB → 1.7B accuracy | 0.61 (253) | 0.58 (253) | 0.59 (253) | 0.60 (253) | 0.59 (253) |
| RF · bBPB → 1.7B accuracy | 0.61 (118) | 0.61 (118) | 0.61 (118) | 0.61 (118) | 0.62 (118) |
| LLM-RF · bBPB → 1.7B accuracy | 0.63 (89) | 0.60 (89) | 0.61 (89) | 0.57 (89) | 0.61 (89) |
| original · bBPB → 1.7B bBPB | 0.67 (490) | 0.61 (490) | 0.65 (490) | 0.64 (490) | 0.64 (490) |
| RF · bBPB → 1.7B bBPB | 0.68 (134) | 0.67 (134) | 0.67 (134) | 0.67 (134) | 0.68 (134) |
| LLM-RF · bBPB → 1.7B bBPB | 0.80 (94) | 0.77 (94) | 0.78 (94) | 0.72 (94) | 0.77 (94) |
| all variants, bBPB → 1.7B accuracy | 0.58 (762) | 0.57 (801) | 0.57 (837) | 0.57 (871) | 0.59 (913) |
| all variants, bBPB → 1.7B bBPB | 0.64 (1020) | 0.61 (1059) | 0.62 (1095) | 0.62 (1129) | 0.63 (1171) |

The same over the paired tasks (those whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix stripped, has an accuracy value at that proxy, so the gate treats every variant alike):

| variant | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| original · accuracy → 1.7B accuracy | 0.55 (157) | 0.57 (180) | 0.57 (195) | 0.60 (214) | 0.59 (240) |
| RF · accuracy → 1.7B accuracy | n = 2 |  |  | n = 1 | n = 3 |
| LLM-RF · accuracy → 1.7B accuracy | n = 2 | n = 1 | n = 2 | n = 1 | n = 3 |
| original · bBPB → 1.7B accuracy | 0.64 (143) | 0.60 (163) | 0.60 (178) | 0.61 (195) | 0.60 (221) |
| RF · bBPB → 1.7B accuracy | n = 2 | n = 1 | n = 1 | n = 1 | n = 3 |
| LLM-RF · bBPB → 1.7B accuracy | n = 3 | n = 2 | n = 2 | n = 2 | n = 4 |
| original · bBPB → 1.7B bBPB | 0.71 (143) | 0.62 (163) | 0.68 (178) | 0.64 (195) | 0.65 (221) |
| RF · bBPB → 1.7B bBPB | n = 3 | n = 2 | n = 2 | n = 3 | n = 4 |
| LLM-RF · bBPB → 1.7B bBPB | n = 3 | n = 2 | n = 2 | n = 2 | n = 4 |
| all variants, bBPB → 1.7B accuracy | 0.59 (309) | 0.58 (347) | 0.58 (378) | 0.61 (414) | 0.59 (474) |
| all variants, bBPB → 1.7B bBPB | 0.63 (310) | 0.59 (348) | 0.62 (379) | 0.62 (416) | 0.62 (475) |

![Variants together and apart](pretraining/predictivity/recipe_da_size_variants_multi_axes.png)
<!-- END auto:variants -->

**Key findings** (the tables above; each bullet names its target)

- **On the same truth, bBPB reads the reference a little better than
  accuracy.** Over every task, original bBPB → 1.7B accuracy is 0.58–0.61
  against the original accuracy's 0.55–0.60: ahead at 90M–600M (0.61 against
  0.55 at 90M) and level at 1B (0.59 and 0.59).
- **On the paired tasks the lead holds at every proxy**: 0.60–0.64 against
  0.55–0.60, so it is not only that bBPB is defined where accuracy is at
  chance.
- **The easier target reads higher.** bBPB → 1.7B bBPB gives the original
  0.61–0.67, LLM-RF 0.72–0.80 and all variants pooled 0.61–0.64, against
  0.57–0.59 pooled with bBPB → 1.7B accuracy; part of bBPB's lead under that
  reading is the target, not the score.
- **The rewrites help through bBPB, not accuracy.** RF and LLM-RF accuracy
  sit at 0.47–0.55 and 0.49–0.54, at or below the original accuracy; their
  bBPB → 1.7B accuracy reads 0.61–0.62 and 0.57–0.63. Their paired cells rest
  on 1–4 tasks and are blank in the tables.
- **Populations differ by target**: 253 original twins have a value under
  bBPB → 1.7B accuracy (gated on the original's accuracy at 1.7B) and 490
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

**Key findings** (per benchmark, the mean DA-size over the five proxies of `recipe_da_size_by_variant_multi_axes.csv`; each bullet names its target)

- **bBPB → 1.7B accuracy beats the original accuracy on 10 of the 14
  benchmarks that have both**, most on XCOPA (0.63 against 0.49), Cultural
  Bench easy (0.46 against 0.30) and XStoryCloze (0.75 against 0.66). It is
  behind on Belebele (0.51 against 0.58), INCLUDE base (0.49 against 0.57), PAWS
  (0.54 against 0.58) and Global PIQA non-parallel (0.75 against 0.79).
- **In the rewrites it beats the same format's accuracy everywhere both
  exist**: 4 of 4 RF benchmarks (Global-MMLU RF 0.68 against 0.55) and 2 of 2
  LLM-RF ones.
- **bBPB → 1.7B bBPB is above bBPB → 1.7B accuracy on 14 of 14 original
  benchmarks**, so a heat map of bBPB → bBPB cells overstates what bBPB tells
  about the accuracy decision.

**Follow-ups**

- The same map on the paired cells only.

GitHub: [recipe_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.png) · [recipe_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_heatmap_multi_axes.csv)

### 4. Where each variant starts to read the reference

The heat map gives means; the profiles give the share of a benchmark's tasks
reliable at each proxy, the quantity the safe rank summarises.

<!-- BEGIN auto:profiles (recipe.py --pool predictivity) -->
![Reliability profiles per benchmark](pretraining/predictivity/recipe_da_size_profiles_multi_axes.png)
<!-- END auto:profiles -->

**Key findings** (each bullet names its target)

- **Few benchmarks have half their tasks reliable by 1B.** At the 1B proxy:
  2 of 16 benchmarks with original accuracy, 3 of 14 with original bBPB →
  1.7B accuracy, 4 of 19 with original bBPB → 1.7B bBPB, and no RF variant
  under either target.

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
