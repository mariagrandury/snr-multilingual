# English only — do the monolingual-English cells read the English benchmarks better?

## Question

> The L1 cells are monolingual English. If the trainings are correct they
> should do clearly better on the (English) benchmarks than the multilingual
> cells of the same size: higher scores, above chance earlier, and at least
> as predictable in size, as reliable in their decisions and as high in SNR.
> Does the ladder show that, and do the L1 numbers land where AllenAI's
> DataDecide puts the same English tasks?

A quick check of the ladder itself, read with the existing computations:
[the above-random gate](../rq00_gate_and_curves/README.md),
[scaling predictability](../rq01_scaling_predictability/README.md#methodology),
[decision accuracy](../rq02_decision_accuracy/README.md#methodology),
[noise and SNR](../rq03_noise_and_snr/README.md#methodology), the relation
[the surrogates](../rq04_surrogates/README.md) start from, and the AllenAI
table of [the external frameworks comparison](../rq07_external_frameworks/README.md#methodology).
The sections run from the plainest expectation (scores) to the most
definition-dependent one (DataDecide), and the [verdict](#validation-verdict)
collects them.

## Setup

- **The cells.** An L1 cell trains on English only; every other language
  setting gives English half of its tokens (the 50/50 mixture), so at the same
  size an L1 cell has seen twice the English. L1 has four families in pool
  `predictivity` (seed 1904): the deep and shallow cells of the baseline build
  (scheme A, T = 1: DCLM through the educational-quality classifier, the
  English every other cell trains on), and two deep cells that change only the
  English corpus — DCLM without the edu classifier (`dclmP`, scheme B at L1)
  and plain FineWeb up to 2022 (`fweb`, scheme C at L1).
- **No swiglu.** The swiglu ladders (deep, L8/L15/L30) have no L1
  counterpart beyond one 90M cell, so every comparison here reads the deep and
  shallow ladders only.
- **Scores, the gate, scaling.** The L1 baseline cell against the
  baseline cell (scheme A, T = 1) of the same depth at every other L (deep
  against deep, shallow against shallow), final checkpoints, 90M–1.7B. The
  seed yardstick is the deep L1 cell's sample std over its three seeds in
  `predictivity_seeds` (1904 with 64/313 at 175M and 600M, with 28/1797 at
  1B), the definition the noise-and-SNR analysis uses.
- **Decision accuracy and SNR** need several families, and the number of
  families moves both. So L1's four families are compared with the L's that
  also have four built the same way (the deep and shallow baseline cells plus
  every deep cell of another data build): L2 (A, ZH = B, ES = C) and L15/L30
  (A, AT3 = A at T 3, B); L8 and L50 have three and are left out.
- **One population per size.** A decision-accuracy or SNR cell counts only
  where every family of its group has the task at that size, and a size reads
  only the tasks every group has (rule 13). A gap in one L1 family therefore
  removes a task from every group's line; the block below says which gaps do.
- **Thin sizes.** A size whose task count falls below half the largest of its
  line is kept in the CSVs but neither drawn, quoted nor counted in the
  verdict. The generated blocks name the thin sizes and the coverage gap
  behind them.
- **What the decisions are about differs by L.** Each group has six pairs,
  one of them the depth pair. At L1 three move only the English corpus
  (scheme A/B/C), a decision English benchmarks should see directly, and two
  move it with the depth; at L2 three move only the second language (scheme
  A/B/C = Russian, Chinese, Spanish); at L15/L30 one moves the scheme (A vs B),
  one the temperature (T 1 vs 3) and three two axes at once. So the
  four-family DA and SNR are **not like-for-like** across L; two readings are:
  - the SNR of the deep and shallow baseline cells at every L, the one
    decision every L shares (two families: rule 5 is about decision accuracy,
    an SNR needs two runs);
  - the decision accuracy split by the axis a pair moves, whose depth pairs
    are that shared decision inside the pooled lines.
- **English benchmarks** are the tasks the registry tags English
  (`utils.assign_language`), every variant: originals, their RF and LLM-RF
  rewrites, and the bBPB twins (lower is better, oriented before any gap is
  read; no chance level, so never gated). The English validation BPB
  (`bpb_dclm`) is read beside them, never pooled with them.
- **The bBPB twins exist at every checkpoint of every size** since this
  snapshot, so they enter the decision-accuracy lines at every drawn proxy (98
  of 121–129 tasks) and their SNR rows read the full noise window; both
  sections give the accuracy-only and the twin-only reading beside the pooled
  one, because the twins outnumber the accuracy tasks three to one.
- **Rules.** The gate of rule 1 blanks every at-chance (task, size) cell
  (decision accuracy at the proxy and the reference); rule 5's three pairs per
  DA cell; the noise window of rule 4 for the SNR; sizes 90M–1.7B (rule 10);
  the per-size task populations move and are counted (rule 13).

<!-- BEGIN auto:english-setup (english_only.py --pool predictivity) -->
Pool `predictivity`, ladder-report snapshot **2026-10-07 15:51**; seed 1904 for every score, the replicate seeds of `predictivity_seeds` for the seed yardstick; gate `predictivity`. 206 English tasks: 107 accuracy-scored (71 original, 35 rf, 1 rfgm), 98 bBPB twins and the English validation BPB `bpb_dclm`.

- **Mirrored family groups** (decision accuracy and the first SNR reading): L1 — `lm-L1-dclmP-deep-seed1904`, `lm-L1-deep-seed1904`, `lm-L1-fweb-deep-seed1904`, `lm-L1-shallow-seed1904`; L2 — `lm-L2-ES-deep-seed1904`, `lm-L2-ZH-deep-seed1904`, `lm-L2-deep-seed1904`, `lm-L2-shallow-seed1904`; L15 — `lm-L15-AT3-deep-seed1904`, `lm-L15-deep-seed1904`, `lm-L15-schemeB-deep-seed1904`, `lm-L15-shallow-seed1904`; L30 — `lm-L30-AT3-deep-seed1904`, `lm-L30-deep-seed1904`, `lm-L30-schemeB-deep-seed1904`, `lm-L30-shallow-seed1904`. Left out, with another number of such families: L8 (3), L50 (3).
- **Deep against shallow** (the second SNR reading): the two baseline cells at L1, L2, L8, L15, L30, L50.
- **Coverage that moves the populations**: `lm-L1-deep-seed1904` at 350M has 18 of the 206 English tasks it has elsewhere. A decision-accuracy or SNR cell missing a family of its group is left out of the lines (it stays in the per-task CSVs), so a gap in one L1 family removes that (task, size) from every group's line.
- **The decision-accuracy population** (multi-axis; English benchmarks above chance at the proxy and the reference that every family of every group has): 90M 121 (23 acc, 98 bbpb), 175M 123 (25 acc, 98 bbpb), 350M 14 (6 acc, 8 bbpb), 600M 129 (31 acc, 98 bbpb), 1B 129 (31 acc, 98 bbpb); thin, so neither drawn nor quoted: 350M.
<!-- END auto:english-setup -->

## Highlighted result

![English scores](pretraining/predictivity/english_only_scores.png)

Pool `predictivity` (seed 1904, every data build, swiglu left out), ladder-report snapshot 2026-10-07 15:51, prose re-read 2026-10-07; final checkpoints, 90M–1.7B, gate `predictivity`. The L1 baseline cell against the same-depth baseline cell of L2, L8, L15, L30 and L50, on the 26–33 accuracy-scored English tasks above chance at each drawn size (350M thin for deep).

Key findings:

- **The English-only cells are better at English, by a small margin.** Deep L1 wins more than half of the English accuracy tasks against every other L at every drawn size (25 of 25 (size, L) cells; win share 0.60–0.79 per size, every L pooled), but the mean gap at 1.7B is 1.2 points (33 tasks, every L pooled); shallow L1 wins 29 of 30 cells, 0.6 points at 1.7B.
- **English BPB is the cleanest separation.** L1 has the lower `bpb_dclm` in all 30 (size, L) comparisons at both depths; the deep gap is 0.049 bits per byte at 90M and shrinks to 0.028 at 1.7B (0.782 against 0.810).
- **Part of the lead is beyond seed noise.** Deep L1 is ahead by more than 2 seed sd in 0.55 (175M), 0.45 (600M) and 0.45 (1B) of the (task, L) comparisons and behind by as much in 0.03–0.07.
- **Clearing chance earlier holds per task, not per share.** Against each of the five L, deep L1 clears chance from a smaller size on more tasks than from a larger one (92 against 30 (task, L) pairs), but its share of English tasks above chance is ahead in only 14 of 25 (size, L) cells (0.42 against a median 0.36 at 1.7B).
- **Scaling is about as regular.** The 1.7B English BPB forecast from 90M–1B misses by 1.0 % at L1 against 1.1–1.4 % elsewhere; the median R² of the English accuracy fits is 0.87 at L1 against 0.81–0.93 (31 fits each).
- **Decision accuracy is higher at L1, but the decisions differ.** With four families per L, DA-size (multi-axis, no filter) is 0.58–0.63 at 90M–1B for L1 against 0.45–0.53 for L2/L15/L30, on the depth pair alone 0.51 against 0.45–0.47. On the shared deep-against-shallow decision, accuracy tasks alone, L1's median log10 SNR is highest at every drawn size but 1B, where it ties (−0.01 against −0.22 to 0.01); on the bBPB twins alone it is mid-pack, so the pooled verdict is "partly" (ahead in 11, tied in 4, behind in 10 of 25 cells).
- **Next to DataDecide.** L1's DA-size is below / inside / above DataDecide's range in 3 / 5 / 3 of 11 format-matched (task, size) cells, median ratio 0.95; the SNRs are not comparable.

[english_only_scores.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.png) · [english_only_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.csv) · [english_only_verdict.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_verdict.csv)

## Scores

The first expectation, and the one a broken training would fail first: at
matched size and depth, the English-only cell scores higher on English.

<!-- BEGIN auto:english-scores (english_only.py --pool predictivity) -->
![English scores](pretraining/predictivity/english_only_scores.png)

![English scores per benchmark](pretraining/predictivity/english_only_scores_by_benchmark.png)

Key findings (accuracy-scored English tasks above chance at the size; the population moves with the size, the counts are drawn on the pooled line and listed below):

- **deep**: L1 ahead in 0.60–0.79 of the (task, L) comparisons per size; mean gap 90M -4.9, 175M 2.4, 600M 2.0, 1B 1.2, 1.7B 1.2 accuracy points (positive = L1 better; tasks 90M 29, 175M 26, 350M 6, 600M 31, 1B 32, 1.7B 33; 350M thin, left out).
- **shallow**: L1 ahead in 0.59–0.82 of the (task, L) comparisons per size; mean gap 90M -2.9, 175M 2.9, 350M 1.4, 600M 1.5, 1B 0.9, 1.7B 0.6 accuracy points (positive = L1 better; tasks 90M 29, 175M 26, 350M 28, 600M 31, 1B 32, 1.7B 33).
- **Against the seed noise** (deep, where replicates exist): L1 ahead by more than 2 seed sd in 175M 0.55, 600M 0.45, 1B 0.45 of the comparisons, behind by as much in 175M 0.03, 600M 0.03, 1B 0.07.
- **Where L1 is far behind** (deep, mean over every other L below −10 pp): `cultural_bench_easy_canada` at 90M (-69 pp), `cultural_bench_easy_australia` at 90M (-51 pp), `cultural_bench_easy_south_africa` at 90M (-37 pp), `cultural_bench_easy_nigeria` at 90M (-32 pp). A letter-format task can pass the gate on a constant answer that matches the majority gold label (rule 1 does not test for it), so a cell where every L but one sits far above chance and the other near zero reads a letter preference, not English skill; the medians are 90M 0.9, 175M 1.7, 600M 2.0, 1B 1.6, 1.7B 1.4 pp.
- **English BPB** (bpb_dclm, deep): L1 lower than the other L's in 1.00–1.00 of the comparisons per size, by 90M 0.049, 175M 0.045, 350M 0.042, 600M 0.041, 1B 0.033, 1.7B 0.028 bits per byte (L1 at 1.7B: 0.782 against 0.810).
- **bBPB** (deep, lower is better, oriented): L1 ahead in 0.54–0.80 of the comparisons per size; mean gap 90M 0.061, 175M 0.081, 600M 0.160, 1B -0.320, 1.7B 0.099 bits, median 90M 0.062, 175M 0.072, 600M 0.066, 1B 0.008, 1.7B 0.027 (positive = L1 better; 350M thin: 8 tasks). At 1B the L1 deep value sits more than 0.3 bits above both neighbouring sizes on 21 of 98 tasks (bbh_mcq-bbpb 16, acp_bench_cloze-bbpb 2, bbh_cloze-bbpb 2, bbh_mcq-rf-bbpb 1), against 8 of 490 (task, L) cells of the other L's deep cells; without those tasks the 1B mean gap is 0.024 instead of -0.320: a spike of that one cell, not of English (the store holds finals only and no replicate seed, so it cannot be told from seed noise).
- **By L at 1.7B** (deep, mean gap / win share): L2 0.9 pp / 0.70; L8 1.3 pp / 0.73; L15 0.8 pp / 0.61; L30 1.5 pp / 0.73; L50 1.7 pp / 0.70.

[english_only_scores.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.png) · [english_only_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.csv) · [english_only_scores_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_by_benchmark.png) · [english_only_scores_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_by_benchmark.csv) · [english_only_scores_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_per_task.csv) · [english_only_scores_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_summary.csv)
<!-- END auto:english-scores -->

Follow-ups:

- The same gaps per English training-token count instead of per size: an L1
  cell of size N has seen as much English as an L>1 cell of roughly 2N, which
  says whether L1's lead is the token count alone.
- The DCLM-without-edu and FineWeb L1 cells against the baseline L1 cell: the
  English-corpus effect next to the language-count effect, on the same tasks.
- The bBPB of the L1 deep cell at 1B (mean gap −0.320 bits against a median
  of 0.008) is not one bad checkpoint: its `bbh_mcq` twins rise from 1.6 bits
  at 10 % of the run to 3.2–3.5 over the last four checkpoints, against
  1.5–1.8 for L2 and L15 deep at the same steps (`bench_bpb.csv`); a replicate
  seed of that cell would say whether it is the seed.
- The 90M deep mean gap (−7.3 points against a median of 0.8) without the
  `cultural_bench_easy` letter tasks the outlier bullet names, to see whether
  L1 trails at 90M at all.
- Next: whether higher scores also mean clearing chance sooner,
  [Above random](#above-random).

## Above random

Higher scores should show up as English benchmarks clearing chance from a
smaller size; this is [the gate's](../rq00_gate_and_curves/README.md) own
run-level verdict per cell.

<!-- BEGIN auto:english-gate (english_only.py --pool predictivity) -->
![Above random](pretraining/predictivity/english_only_above_random.png)

Key findings:

- **Population**: the English tasks with a chance level every drawn cell scores, 90M 106, 175M 106, 350M 9, 600M 106, 1B 106, 1.7B 106; a size below half the largest population is left out of the lines and of the shares below (350M).
- **deep**: share of English tasks above chance, L1 90M 0.26, 175M 0.27, 600M 0.36, 1B 0.32, 1.7B 0.42; other L's median 90M 0.29, 175M 0.26, 600M 0.30, 1B 0.33, 1.7B 0.36.
- **shallow**: share of English tasks above chance, L1 90M 0.25, 175M 0.30, 600M 0.35, 1B 0.32, 1.7B 0.40; other L's median 90M 0.27, 175M 0.28, 600M 0.30, 1B 0.31, 1.7B 0.33.
- **deep, L1 vs L2** (47 tasks either clears): L1 clears from a smaller size on 0.38, the same on 0.51, a larger one on 0.11.
- **deep, L1 vs L8** (49 tasks either clears): L1 clears from a smaller size on 0.37, the same on 0.49, a larger one on 0.14.
- **deep, L1 vs L15** (47 tasks either clears): L1 clears from a smaller size on 0.45, the same on 0.49, a larger one on 0.06.
- **deep, L1 vs L30** (51 tasks either clears): L1 clears from a smaller size on 0.33, the same on 0.51, a larger one on 0.16.
- **deep, L1 vs L50** (50 tasks either clears): L1 clears from a smaller size on 0.36, the same on 0.50, a larger one on 0.14.

[english_only_above_random.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random.png) · [english_only_above_random.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random.csv) · [english_only_above_random_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random_per_task.csv)
<!-- END auto:english-gate -->

Follow-ups:

- The same crossing read per benchmark family, to see which English
  benchmarks the extra English tokens lift over chance first.
- Next: whether the extra English makes the ladder scale more regularly,
  [Scaling predictability](#scaling-predictability).

## Scaling predictability

More English tokens should not make English scale less regularly. Two
readings from [scaling predictability](../rq01_scaling_predictability/README.md#methodology):
the error of a power law fitted on the proxies for the 1.7B English BPB (a
forecast), and the R² of the log-linear fit of every English accuracy task
over the rungs every L shares, the 1.7B reference included (a property of the
ladder, rule 11).

<!-- BEGIN auto:english-scaling (english_only.py --pool predictivity) -->
![Scaling](pretraining/predictivity/english_only_scaling.png)

Key findings:

- **English BPB, fit on 90M–1B** (a forecast from the proxies): |error| of the 1.7B prediction L1 deep 1.1 %; L1 deep dclmP 1.1 %; L1 deep fweb 2.0 %; L1 shallow 0.8 %; L2 deep 1.3 %; L2 shallow 1.0 %; L8 deep 1.7 %; L8 shallow 1.0 %; L15 deep 1.9 %; L15 shallow 0.9 %; L30 deep 1.5 %; L30 shallow 1.1 %; L50 deep 1.9 %; L50 shallow 0.9 %.
- **English accuracy tasks** (deep, median R² of the log-N fit over the rungs the L1 deep cell has, 1.7B included — a reference-size quantity (rule 11) that describes the ladder, not a forecast; fits in brackets): L1 0.87 (31); L2 0.82 (31); L8 0.93 (31); L15 0.90 (31); L30 0.93 (31); L50 0.81 (31).

[english_only_scaling.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling.png) · [english_only_scaling.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling.csv)
<!-- END auto:english-scaling -->

Follow-ups:

- The power law on the English BPB of each L1 build's OWN corpus (the
  DCLM-without-edu and FineWeb cells are scored here on the edu-filtered DCLM
  validation set, off their training distribution).
- Next: whether the English benchmarks fall into the same scaling regimes at
  L1 as on the multilingual ladder, [Scaling regimes at L1](#scaling-regimes-at-l1).

### Scaling regimes at L1

The [scaling regimes](../rq01_scaling_predictability/README.md#1-scaling-regimes-the-paper-figure)
of the paper's benchmark scaling behaviour section, drawn for the monolingual
comparison point: the three English corpora at L1, one family each, under
the same gate and the same fits. All three have finished and been evaluated at
every rung from 90M to 1.7B; the DCLM-Edu cell's 350M evaluation covers 20 of
its 208 tasks, so that size drops out of its trajectory medians.

<!-- BEGIN auto:english-regimes (english_regimes.py --pool predictivity) -->
![Scaling regimes at L1](pretraining/predictivity/english_only_scaling_regimes.png)

Pool `predictivity`: the L1 deep seed-1904 cell of each English corpus (DCLM-Edu = data A, DCLM = DCLMP, FineWeb = FWEB), 90M–1.7B, gate `predictivity`; the fits and regimes of the scaling-predictability figure, one size fit per task (one L) and the median trajectory R² over ≥ 2 sizes; English tasks (originals, RF, LLM-RF, bBPB twins) and `bpb_dclm`. The populations differ: the corpora share the gate, so they read the same tasks, and rq01's grey points are every trained language. Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool predictivity`.

Key findings (share of tasks per regime; both = predictable across size and training):

- **DCLM-Edu** (130 tasks, 31 accuracy-scored): both 42%, training only 5%, size only 18%, weak 20%, declines 15%; median R² across size 0.71, along training 0.50; the accuracy-scored tasks alone: both 55%.
- **DCLM** (130 tasks, 31 accuracy-scored): both 41%, training only 3%, size only 18%, weak 23%, declines 15%; median R² across size 0.62, along training 0.48; the accuracy-scored tasks alone: both 45%.
- **FineWeb** (130 tasks, 31 accuracy-scored): both 72%, training only 4%, size only 5%, weak 11%, declines 8%; median R² across size 0.86, along training 0.68; the accuracy-scored tasks alone: both 55%.
- **The same single-fit reading at every other L** (the English tasks of the deep data-A cell, 130 tasks): predictable across both 42%–53% at L2–L50 (L2 42%, L8 46%, L15 53%, L30 50%, L50 49%).
- **every K (rq01)** (952 tasks, every trained language, medians over the L): both 79%, training only 1%, size only 10%, weak 7%, declines 3%. A median over six fits is smoother than one fit, so this is the figure's backdrop, not the like-for-like comparison (the bullet above is).
<!-- END auto:english-regimes -->

Follow-ups:

- The same panels per benchmark family: FineWeb's lead is in the bBPB twins
  (98 of the 130 tasks; on the accuracy-scored tasks alone it ties DCLM-Edu),
  so which twins it scales on is the next question.
- A replicate seed of the FineWeb L1 cell: one family per corpus cannot tell
  a corpus effect from a seed effect.

- Next: whether a small L1 proxy also ranks the L1 variants like 1.7B does,
  [Decision accuracy](#decision-accuracy).

## Decision accuracy

Does a small L1 proxy rank the L1 design variants as the 1.7B L1 cells do, at
least as well as a proxy at another L ranks that L's variants? The kernel,
the pair sets and the band are [decision accuracy's](../rq02_decision_accuracy/README.md#methodology);
the split by axis says how much of the answer is the decision mix.

<!-- BEGIN auto:english-da (english_only.py --pool predictivity) -->
![DA-size, multi-axis](pretraining/predictivity/english_only_da_size_by_L_multi_axes.png)

![DA-size, mono-axis](pretraining/predictivity/english_only_da_size_by_L_mono_axis.png)

Key findings (DA-size against 1.7B, no reliability filter, pooled over the English benchmarks above chance at the proxy and the reference (gate `predictivity`) that every family of every group has, ≥ 3 pairs per task, ± the 90 % leave-one-family-out half-width; the decisions differ by L):

- **multi-axis**: L1 0.58 ±0.09, 0.60 ±0.05, 0.60 ±0.20, 0.63 ±0.16; L2 0.53 ±0.06, 0.51 ±0.03, 0.53 ±0.11, 0.46 ±0.05; L15 0.51 ±0.12, 0.45 ±0.04, 0.46 ±0.04, 0.51 ±0.10; L30 0.48 ±0.03, 0.52 ±0.06, 0.45 ±0.06, 0.51 ±0.08 (90M, 175M, 600M, 1B).
- **mono-axis**: L1 0.60 ±0.18, 0.59 ±0.07, 0.61 ±0.23, 0.60 ±0.21; L2 0.52 ±0.12, 0.51 ±0.02, 0.51 ±0.10, 0.46 ±0.10; L15 0.47, 0.44, 0.44, 0.49; L30 0.50, 0.49, 0.41, 0.49 (90M, 175M, 600M, 1B).
- **By the axis a pair moves** (multi-axis, pooled over the proxies drawn; decisions in brackets): L1 depth (deep vs shallow) 0.51 (502), two axes at once 0.60 (1004), data build 0.63 (1506); L2 depth (deep vs shallow) 0.46 (502), two axes at once 0.52 (1004), data build 0.51 (1506); L15 temperature (T=1 vs T=3) 0.44 (502), depth (deep vs shallow) 0.45 (502), two axes at once 0.51 (1506), data build 0.49 (502); L30 temperature (T=1 vs T=3) 0.45 (502), depth (deep vs shallow) 0.47 (502), two axes at once 0.50 (1506), data build 0.50 (502). The depth pair (deep against shallow) is the one decision every group shares.
- **bBPB twins** (not in the lines: some family of a group has none; multi-axis, every cell with ≥ 3 pairs, pooled over the proxies, NOT one population): L1 0.61 (6 pairs per task); L2 0.50 (6 pairs per task); L15 0.50 (6 pairs per task); L30 0.49 (6 pairs per task).
- **English BPB** (multi-axis, mean over the proxies): L1 0.97; L2 0.87; L15 0.70; L30 0.80.

[english_only_da_size_by_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_multi_axes.png) · [english_only_da_size_by_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_multi_axes.csv) · [english_only_da_size_by_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_mono_axis.png) · [english_only_da_size_by_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_mono_axis.csv) · [english_only_da_size_by_L_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_both_axes.csv) · [english_only_da_size_by_pair_axis_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_pair_axis_both_axes.csv) · [english_only_da_size_per_task_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_per_task_both_axes.csv)
<!-- END auto:english-da -->

Key findings, read from `english_only_da_size_per_task_both_axes.csv` and `english_only_da_size_by_L_both_axes.csv` (multi-axis, the lines' shared cells, 6 pairs per task):

- **Four proxies are drawn, and the twins dominate them.** The lines read 121 tasks at 90M (23 accuracy, 98 bBPB), 123 at 175M (25, 98) and 129 at 600M and 1B (31, 98); 350M is thin (14 tasks) because the L1 deep cell there has 18 of the 206 English tasks.
- **The generated bBPB bullet's "not in the lines" disagrees with the setup block**, which counts the 98 twins in every drawn line; the per-task CSV marks them shared at every drawn size for every group.
- **L1's lead holds on each scoring alone.** Accuracy tasks: L1 0.59, 0.59, 0.56 and 0.63 (90M, 175M, 600M, 1B) against 0.43–0.48, 0.43–0.51, 0.39–0.53 and 0.45–0.51 for L2/L15/L30; bBPB twins: L1 0.58, 0.60, 0.61 and 0.63 against 0.49–0.54, 0.46–0.53, 0.44–0.53 and 0.44–0.52.
- **The band separates L1 at 175M only, and not from every L.** L1's 90 % band at 175M (0.548–0.655) just clears L2's and L15's upper ends (0.546, 0.490) but overlaps L30's (0.456–0.585); at 90M (0.49–0.67), 600M (0.40–0.80) and 1B (0.47–0.79) it overlaps all three.
- **Most of the gap is the decision, not the proxy.** On the depth pair, the one decision every group shares, L1 is at 0.51 against 0.45–0.47; the L1 pairs that move the English corpus reach 0.63 (data build) and 0.60 (two axes at once).

Follow-ups:

- The seed-null DA of the L1 replicates at 175M, 600M and 1B, as the floor
  these lines should be read against.
- The English tasks the 350M L1 deep cell lacks (it has 18 of 206), which
  keep 350M thin in every section.
- Next: whether the same families separate more cleanly than they wobble,
  [Noise and SNR](#noise-and-snr).

## Noise and SNR

The SNR of [noise and SNR](../rq03_noise_and_snr/README.md#methodology)
(`rel_std`) over the same mirrored groups, the like-for-like deep-against-shallow
reading at every L, and the relation [the surrogates](../rq04_surrogates/README.md)
are built on (SNR at the proxy against DA-size at the proxy).

<!-- BEGIN auto:english-snr (english_only.py --pool predictivity) -->
![SNR](pretraining/predictivity/english_only_snr.png)

Key findings:

- **The families mirroring L1's**, median log10 SNR over the paired tasks (90M: 127, 175M: 124, 350M: 14, 600M: 129, 1B: 130, 1.7B: 131 tasks; thin: 350M): L1 0.45, 0.41, 0.44, 0.33, 0.29; L2 0.18, 0.33, 0.29, 0.23, 0.13; L15 0.24, 0.23, 0.19, 0.16, 0.04; L30 0.19, 0.21, 0.21, 0.13, 0.13 (90M, 175M, 600M, 1B, 1.7B).
- **Deep against shallow**, median log10 SNR over the paired tasks (90M: 116, 175M: 118, 350M: 14, 600M: 117, 1B: 122, 1.7B: 120 tasks; thin: 350M): L1 0.08, 0.11, 0.03, 0.02, -0.12; L2 -0.00, 0.16, 0.04, -0.12, -0.13; L8 0.12, 0.17, -0.04, 0.01, -0.20; L15 0.11, 0.04, 0.07, 0.02, -0.12; L30 0.10, -0.03, 0.07, -0.05, -0.18; L50 0.06, -0.02, -0.06, 0.11, -0.10 (90M, 175M, 600M, 1B, 1.7B).
- **SNR → DA-size, Spearman ρ per proxy** (mirrored families, the DA lines' tasks): L1 0.08, 0.27, 0.37, 0.30; L2 0.24, 0.12, 0.07, 0.01; L15 -0.06, -0.05, 0.03, -0.01; L30 -0.06, -0.01, 0.16, 0.02 (90M, 175M, 600M, 1B).

[english_only_snr.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr.png) · [english_only_snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr.csv) · [english_only_snr_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr_per_task.csv)
<!-- END auto:english-snr -->

Key findings, read from `english_only_snr_per_task.csv` (ungated tasks every group of the reading has, median log10 SNR at 90M / 175M / 600M / 1B / 1.7B):

- **Mirrored families: on accuracy tasks alone L1 leads at every drawn size but 175M.** L1 0.36 / 0.34 / 0.36 / 0.28 / 0.25 against L2 0.22 / 0.35 / 0.23 / 0.15 / 0.09, L15 0.28 / 0.23 / 0.09 / 0.11 / 0.01 and L30 0.28 / 0.14 / 0.13 / 0.04 / −0.04 (29 / 26 / 31 / 32 / 33 tasks). On the 98 twins alone, now on the full noise window, L1 leads at every size (0.30–0.46 against 0.06–0.32), so the pooled lead (15 of 15 in the verdict) holds on each scoring.
- **On the shared decision, accuracy tasks alone, L1 is highest or tied.** Deep against shallow (18 / 20 / 19 / 24 / 22 tasks), L1's median log10 SNR is 0.21 / 0.17 / 0.16 / −0.01 / −0.15 against −0.02 to 0.16 / −0.10 to 0.06 / −0.11 to 0.04 / −0.22 to 0.01 / −0.31 to −0.17 for the other L's, tied with L8 at 1B (0.01 against −0.01). On the twins alone it is mid-pack (−0.11 to 0.10 against −0.17 to 0.19), so the pooled reading in the generated block and the verdict's 11 / 4 / 10 of 25 are the twins'.

Follow-ups:

- The signal and the noise parts separately (both are in
  `english_only_snr_per_task.csv`): whether L1's SNR comes from a wider
  spread across its families or from quieter late checkpoints.
- The deep-against-shallow SNR of the twins per benchmark family, to see
  which twins put L1 mid-pack where its accuracy tasks lead.
- Next: whether the L1 numbers land where an external English ladder puts the
  same tasks, [Next to AllenAI DataDecide](#next-to-allenai-datadecide).

## Next to AllenAI DataDecide

The L1 numbers next to DataDecide's on the English tasks both evaluate:

- **Source.** DataDecide's per-task SNR and DA from [the external frameworks
  comparison](../rq07_external_frameworks/README.md#methodology)
  (`allenai_snr_variants_per_task.csv`, built by `build_allenai_variants.py`
  with the same per-model arrays and aggregators).
- **Tasks, paired by format.** lm-eval's ARC, HellaSwag and OpenBookQA score
  the answer strings, like OLMES's RC; our `rf_` twins are the answer-string
  rewrite of a letter task (`rf_mmlu` ↔ `mmlu`, `rf_commonsense_qa` ↔ `csqa`);
  the letter originals meet OLMES's `:mc` (`commonsense_qa` ↔ `csqa:mc`;
  DataDecide has no `mmlu:mc` aggregate).
- **Sizes**, paired as the external frameworks comparison pairs them.
- **Definitions.** The table at the top of the block lists what differs
  between the two sides, so the comparison reads levels and directions, not
  decimals.

<!-- BEGIN auto:english-allenai (english_only.py --pool predictivity) -->
|  | ours (L1) | DataDecide |
|---|---|---|
| models behind the signal | 4 families (dclmP, deep, fweb, shallow) | 25 data recipes |
| DA pairs | 6 (multi-axis; a cell counts only with all 4 families), so DA moves in steps of 1/6 | 300 |
| decisions | depth and the English corpus | the data recipe |
| reference | 1.7B | 1B (so no DataDecide DA-size at 1B) |
| proxies | 175M / 350M / 600M / 1B (non-embedding) | 150M / 300M / 750M / 1B |
| token budget | 100 N (5× Chinchilla) | 100 tokens per parameter (5× Chinchilla), per the DataDecide release |
| checkpoint noise | std over 80/85/90/95/100 % of the run (rule 4) | std over the last five saved checkpoints (the table predates rule 4) |
| metric | lm-eval `acc` (originals), `acc_norm` (RF twins) | the release's OLMES primary metric (not re-read here: the `core` parquet is not cached on the cluster) |
| gate | rule 1 at the proxy (and the reference for DA) | none |

![AllenAI](pretraining/predictivity/english_only_allenai.png)

Key findings:

- **da-size**: L1 / DataDecide on the same task, median 0.95 (range 0.71–1.15); the other L groups' median over DataDecide 0.69.
- **snr**: L1 / DataDecide on the same task, median 0.29 (range 0.05–0.90); the other L groups' median over DataDecide 0.08.
- **No L1 DA-size** (at chance on our side, rule 1, or a family missing): `arc_challenge` 175M, 350M, `openbookqa` 175M, 350M, 600M, `rf_commonsense_qa` 350M, `rf_mmlu` 350M, `commonsense_qa` 175M, 350M, 600M. DataDecide is not gated, so its range is taken only over the tasks where L1 has a value at that size.

| ours | DataDecide | quantity | sizes (ours / theirs) | L1 | L1 pairs / runs | other L median | DataDecide | DataDecide range (L1's tasks) | L1 vs range | L1 / DataDecide |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | arc_easy | da-size | 175M / 150M | 1.00 | 6 | 0.67 | 0.93 | 0.70–0.93 | above | 1.07 |
| arc_easy | arc_easy | snr | 175M / 150M | 5.61 | 4 | 1.02 | 19.82 | 6.62–19.82 | (not graded) | 0.28 |
| arc_easy | arc_easy | da-size | 350M / 300M | 1.00 | 6 | 0.33 | 0.96 | 0.82–0.96 | above | 1.05 |
| arc_easy | arc_easy | snr | 350M / 300M | 10.11 | 4 | 2.19 | 25.18 | 11.32–25.18 | (not graded) | 0.40 |
| arc_easy | arc_easy | da-size | 600M / 750M | 0.83 | 6 | 0.17 | 0.96 | 0.76–0.97 | inside | 0.86 |
| arc_easy | arc_easy | snr | 600M / 750M | 5.56 | 4 | 1.52 | 26.19 | 8.64–26.19 | (not graded) | 0.21 |
| arc_easy | arc_easy | snr | 1B / 1B | 5.72 | 4 | 1.33 | 17.03 | 6.60–17.03 | (not graded) | 0.34 |
| arc_challenge | arc_challenge | da-size | 600M / 750M | 0.83 | 6 | 0.67 | 0.91 | 0.76–0.97 | inside | 0.92 |
| arc_challenge | arc_challenge | snr | 600M / 750M | 7.21 | 4 | 1.98 | 12.04 | 8.64–26.19 | (not graded) | 0.60 |
| arc_challenge | arc_challenge | snr | 1B / 1B | 4.84 | 4 | 0.41 | 11.47 | 6.60–17.03 | (not graded) | 0.42 |
| hellaswag | hellaswag | da-size | 175M / 150M | 0.83 | 6 | 1.00 | 0.72 | 0.70–0.93 | inside | 1.15 |
| hellaswag | hellaswag | snr | 175M / 150M | 0.96 | 4 | 1.40 | 6.82 | 6.62–19.82 | (not graded) | 0.14 |
| hellaswag | hellaswag | da-size | 350M / 300M | 0.83 | 6 | 0.83 | 0.82 | 0.82–0.96 | inside | 1.02 |
| hellaswag | hellaswag | snr | 350M / 300M | 1.91 | 4 | 1.81 | 11.32 | 11.32–25.18 | (not graded) | 0.17 |
| hellaswag | hellaswag | da-size | 600M / 750M | 0.67 | 6 | 0.67 | 0.94 | 0.76–0.97 | below | 0.71 |
| hellaswag | hellaswag | snr | 600M / 750M | 1.11 | 4 | 0.54 | 22.16 | 8.64–26.19 | (not graded) | 0.05 |
| hellaswag | hellaswag | snr | 1B / 1B | 1.19 | 4 | 0.47 | 11.95 | 6.60–17.03 | (not graded) | 0.10 |
| rf_commonsense_qa | csqa | da-size | 175M / 150M | 0.67 | 6 | 0.33 | 0.70 | 0.70–0.93 | below | 0.95 |
| rf_commonsense_qa | csqa | snr | 175M / 150M | 5.99 | 4 | 1.45 | 6.62 | 6.62–19.82 | (not graded) | 0.90 |
| rf_commonsense_qa | csqa | da-size | 600M / 750M | 0.67 | 6 | 0.50 | 0.76 | 0.76–0.97 | below | 0.88 |
| rf_commonsense_qa | csqa | snr | 600M / 750M | 2.60 | 4 | 0.95 | 8.64 | 8.64–26.19 | (not graded) | 0.30 |
| rf_commonsense_qa | csqa | snr | 1B / 1B | 1.88 | 4 | 1.77 | 6.60 | 6.60–17.03 | (not graded) | 0.28 |
| rf_mmlu | mmlu | da-size | 175M / 150M | 0.83 | 6 | 0.33 | 0.89 | 0.70–0.93 | inside | 0.93 |
| rf_mmlu | mmlu | snr | 175M / 150M | 3.33 | 4 | 1.46 | 9.74 | 6.62–19.82 | (not graded) | 0.34 |
| rf_mmlu | mmlu | da-size | 600M / 750M | 1.00 | 6 | 0.67 | 0.97 | 0.76–0.97 | above | 1.03 |
| rf_mmlu | mmlu | snr | 600M / 750M | 5.71 | 4 | 0.48 | 24.10 | 8.64–26.19 | (not graded) | 0.24 |
| rf_mmlu | mmlu | snr | 1B / 1B | 4.94 | 4 | 1.03 | 16.84 | 6.60–17.03 | (not graded) | 0.29 |

Reading the two sides:

- **DA-size shares the kernel, not the decisions.** Both sides count agreeing pairs; ours is conditional on the task passing the gate at the proxy and the reference, theirs is not, and ours ranks 4 families that differ in depth and English corpus, theirs 25 recipes. A cell above DataDecide's range is a 6-of-6 (DA moves in steps of 1/6 against 1/300).
- **The SNRs are not comparable** and are not graded: DataDecide's signal is the spread of 25 recipes over very different corpora, ours the spread of families of which two differ in depth only or in one filter, and the noise uses different checkpoint windows. The like-for-like SNR reading is deep against shallow at every L ([Noise and SNR](#noise-and-snr)); the four-family direction against the other L groups is a within-ladder reading over different decisions per L (Setup).

[english_only_allenai.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai.png) · [english_only_allenai.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai.csv) · [english_only_allenai_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai_per_task.csv)
<!-- END auto:english-allenai -->

Follow-ups:

- Rebuild the AllenAI table under rule 4's window, so the noise halves of the
  two SNRs share a definition.
- Evaluate `piqa`, `winogrande`, `boolq` and `social_iqa` on the L1 cells (the
  `allenai` task group): four more format-matched pairs.

## Validation verdict

One row per expectation, computed from the tables above. "yes": L1 is ahead
of, or within 0.01 of, every comparator at every size drawn; "partly": behind
somewhere and ahead somewhere.

- **not like-for-like**: the groups' decisions differ by L, so the row gives
  the numbers without a grade.
- **not comparable**: the definitions differ (the DataDecide SNR).
- **not computed**: the pool lacks the families the row needs.

<!-- BEGIN auto:english-verdict (english_only.py --pool predictivity) -->
| expectation | measure | L1 | comparator | confirmed? |
|---|---|---|---|---|
| higher English accuracy (deep) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 25, tied 0, behind 0 of 25 (size, L) cells | 1.2 pp at 1.7B | yes |
| higher English accuracy (shallow) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 29, tied 0, behind 1 of 30 (size, L) cells | 0.6 pp at 1.7B | partly |
| lower English BPB (deep) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 0.03 bits at 1.7B | yes |
| lower English BPB (shallow) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 0.03 bits at 1.7B | yes |
| clears chance earlier (deep) | comparators against which L1 stays above chance from a smaller size on more tasks than from a larger one; L1's share of English tasks above chance against each L, per size | 5 of 5 comparators (92 earlier, 30 later, (task, L) pooled) | share: ahead 14, tied 6, behind 5 of 25 (size, L) cells | partly |
| English BPB at least as predictable in size | median absolute relative error of the 1.7B prediction from 90M–1B (baseline build, deep and shallow), L1 against each L | 1.0 % | L2 1.1 %, L8 1.3 %, L15 1.4 %, L30 1.3 %, L50 1.4 % | yes |
| English accuracy at least as predictable in size | median R² of score vs log10 N (deep; the rungs the L1 deep cell has, 1.7B included: a reference-size quantity, rule 11), L1 against each L | 0.87 | L2 0.82, L8 0.93, L15 0.90, L30 0.93, L50 0.81 | partly |
| higher decision accuracy (mirrored families) | pooled DA-size per proxy, multi-axis, ahead 12, tied 0, behind 0 of 12 (size, L) cells; the decisions differ by L (at L1 three of six pairs move the English corpus) | mean 0.60 | L2 0.51, L15 0.48, L30 0.49; on the depth pairs alone: L1 0.51, L2 0.46, L15 0.45, L30 0.47 | not like-for-like |
| higher SNR (mirrored families) | median log10 SNR over the paired tasks per size, ahead 15, tied 0, behind 0 of 15 (size, L) cells; the families' differences differ by L | mean 0.38 | L2 0.23, L15 0.17, L30 0.17 | not like-for-like |
| higher SNR (deep against shallow, the shared decision) | median log10 SNR over the paired tasks per size, ahead 11, tied 4, behind 10 of 25 (size, L) cells | mean 0.02 | L2 -0.01, L8 0.01, L15 0.02, L30 -0.02, L50 -0.00 | partly |
| L1 DA-size in DataDecide's range | (task, size) cells below / inside / above DataDecide's min–max over the shared tasks where L1 has a value at the matched size; median L1 / DataDecide on the same task (ours gated, decisions differ: see the README) | 3 / 5 / 3 of 11 | ratio 0.95 | yes |
| L1 SNR next to DataDecide | not comparable: DataDecide's signal is the spread of 25 recipes, ours of L1's families; the noise uses different checkpoint windows (the like-for-like reading is the deep-against-shallow row above) | 16 cells | ratio 0.29 (shown, not graded) | not comparable |

[english_only_verdict.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_verdict.csv)
<!-- END auto:english-verdict -->

## Outputs

Under `pretraining/<pool>/`; the driver runs `predictivity`, which holds
every L1 family (deep, shallow, DCLM without edu, FineWeb) and writes the
blocks above.

| file | what |
|---|---|
| `english_only_scores.png/.csv` | mean gap (L1 − L) and win shares per size, deep and shallow, with the task counts |
| `english_only_scores_by_benchmark.png/.csv` | the deep gap and win share per English benchmark × size |
| `english_only_scores_per_task.csv`, `english_only_scores_summary.csv` | one row per (task, size, depth, comparator L); the summary per (scoring, depth, size, comparator), with `thin` |
| `english_only_above_random.png/.csv`, `english_only_above_random_per_task.csv` | share of English tasks above chance per cell; which cell clears chance first (with its task counts); the run-level verdicts |
| `english_only_scaling.png/.csv` | English BPB power-law error per ladder top; R² of the English accuracy fits per L, with the fit counts |
| `english_only_da_size_by_L_multi_axes.png/.csv`, `…_mono_axis.png/.csv` | pooled DA-size per family group: the lines, the jackknife band, the pair and task counts |
| `english_only_da_size_by_L_both_axes.csv` | the pooled lines in full (both pair sets): reliability, band, pair and task counts, `thin` |
| `english_only_da_size_by_pair_axis_both_axes.csv` | the drawn decisions split by the axis each pair moves |
| `english_only_da_size_per_task_both_axes.csv` | every (task, group, size) DA cell, both pair sets: gated (rule 1) and short (rule 5) cells NaN with their pair count; `all_families`, `shared` = what the lines read |
| `english_only_snr.png/.csv`, `english_only_snr_per_task.csv` | SNR per group (mirrored families and deep against shallow) with the task counts, per task at 1B, and the SNR → DA-size Spearman ρ |
| `english_only_allenai.png/.csv`, `english_only_allenai_per_task.csv` | ours next to DataDecide on the format-matched shared tasks, with L1's pair or run count |
| `english_only_verdict.csv` | the verdict table |
| `english_only_scaling_regimes.png/.csv`, `…_paper.png/.csv` | `english_regimes.py`: the scaling regimes of each English corpus at L1, one row per (corpus, task), with rq01's points as `every K (rq01)` |
| `english_only_scaling_regimes_summary.csv` | the share of tasks per regime for each corpus, and the same single-fit reading of the English tasks of the deep data-A cell at every other L |

## How to run

```bash
cd src/signal-and-noise
python analysis/rq13_english_only/english_only.py --pool predictivity
# --readme writes the blocks from another pool (by default only `predictivity` writes them)
python analysis/rq13_english_only/english_regimes.py --pool predictivity   # after rq01's regimes.py
```

It reads the above-random mask (the pool's own, else `predictivity`'s), the
replicate seeds of `predictivity_seeds` and the AllenAI table. Without the
mask or an L1 cell it writes nothing and says why; without the AllenAI table
it skips that comparison.

When L1 has fewer than three families in the pool, decision accuracy is not
computed (rule 5) and the mirrored-family SNR, the surrogate and the AllenAI
comparison are skipped with it, by this analysis's choice of matched groups.
The run says so in a `!!! RULE 5` line, writes the deep-against-shallow SNR
and the verdict rows as "not computed", and leaves the README to a pool that
has the families.
