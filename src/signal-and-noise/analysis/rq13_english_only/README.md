# English only: do the monolingual-English cells read the English benchmarks better?

## Question

> The English-only cells (L = 1) train on English alone, while every other
> cell gives English half of its tokens. Do they score better on the English
> benchmarks than multilingual cells of the same size? If the runs are
> correct, they should score higher, clear chance at a
> smaller size, and be at least as predictable across sizes, as reliable in
> their decisions (decision accuracy) and as high in SNR. Do their numbers
> also land where AllenAI's DataDecide puts the same English tasks?

This is a sanity check of the ladder itself. It reuses the existing
computations: [the above-random gate](../rq00_gate_and_curves/README.md),
[scaling predictability](../rq01_scaling_predictability/README.md#methodology),
[decision accuracy](../rq02_decision_accuracy/README.md#methodology),
[noise and SNR](../rq03_noise_and_snr/README.md#methodology), the SNR-to-DA
relation [the surrogates](../rq04_surrogates/README.md) start from, and the
AllenAI table of [the external frameworks comparison](../rq07_external_frameworks/README.md#methodology).
The sections go from the simplest expectation (scores) to the one that
depends most on definitions (DataDecide). The [verdict](#validation-verdict)
collects them.

## Setup

**Setup.**
- **Measured.** We measure, in accuracy points above chance, how much better a model trained on English alone scores on English benchmarks than one that gives English half of its tokens.
- **Models.** We compare the English-only cells (L = 1) with the baseline cells (scheme A, T = 1) of the same depth at every other language setting: deep against deep and shallow against shallow. All cells use seed 1904 and their final checkpoints, 90M–1.7B.
- **Tasks.** We use the English accuracy tasks that are above chance at each size, so the task set changes with the size.
- **How.** For each size, task and language setting, we subtract the multilingual cell's score from the English-only cell's, then average over the tasks and count the share of (task, L) comparisons where the English-only cell is ahead.
- **Scaling regimes.** We also draw the scaling regimes of the English tasks (whether a task scales predictably across sizes, along training, both or neither) for the English-only cell of each English corpus (figure: app_rq13_english_scaling_regimes), and the same regimes on the English accuracy tasks alone (figures: app_rq13_english_scaling_regimes_l1_accuracy, app_rq13_english_scaling_regimes_l1_accuracy_tagged).

- **The cells.** An L1 cell trains on English only. Every other language
  setting gives English half of its tokens (the 50/50 mixture), so at the same
  size an L1 cell has seen twice as much English. Pool `predictivity` (seed
  1904) has four L1 families. Two are the deep and shallow cells of the
  baseline build (scheme A, T = 1), whose English is DCLM filtered by the
  educational-quality classifier, the same English every other cell trains
  on. The other two are deep cells that change only the English corpus: DCLM
  without the edu classifier (`dclmP`, scheme B at L1) and plain FineWeb up to
  2022 (`fweb`, scheme C at L1).
- **No swiglu.** The swiglu ladders (deep, L8/L15/L30) have no L1
  counterpart beyond one 90M cell. Every comparison here therefore reads the
  deep and shallow ladders only.
- **Scores, the gate, scaling.** The L1 baseline cell is compared with the
  baseline cell (scheme A, T = 1) of the same depth at every other L (deep
  against deep, shallow against shallow), on final checkpoints, 90M–1.7B. The
  seed yardstick is the sample std of the deep L1 cell over its three seeds in
  `predictivity_seeds` (1904 with 64/313 at 175M and 600M, with 28/1797 at
  1B). This is the definition the noise-and-SNR analysis uses.
- **Decision accuracy and SNR** need several families, and the number of
  families changes both. So L1's four families are compared only with the L's
  that also have four families built the same way (the deep and shallow
  baseline cells plus every deep cell of another data build). These are L2
  (A, ZH = B, ES = C) and L15/L30 (A, AT3 = A at T 3, B). L8 and L50 have
  three and are left out.
- **One population per size.** A decision-accuracy or SNR cell counts only
  where every family of its group has the task at that size. A size reads
  only the tasks every group has (rule 13). A gap in one L1 family therefore
  removes a task from every group's line. The block below says which gaps do.
- **Thin sizes.** A size whose task count falls below half the largest count
  of its line stays in the CSVs, but it is not drawn, quoted or counted in
  the verdict. The generated blocks name the thin sizes and the coverage gap
  behind them.
- **The decisions differ by L.** Each group has six pairs, and one of them is
  the depth pair. At L1, three pairs move only the English corpus (scheme
  A/B/C), a decision English benchmarks should see directly, and two move it
  together with the depth. At L2, three pairs move only the second language
  (scheme A/B/C = Russian, Chinese, Spanish). At L15/L30, one pair moves the
  scheme (A vs B), one the temperature (T 1 vs 3), and three move two axes at
  once. So the four-family DA and SNR are **not like-for-like** across L. Two
  readings are:
  - the SNR of the deep and shallow baseline cells at every L, the one
    decision every L shares. Two families are enough here, because rule 5
    applies to decision accuracy and an SNR needs only two runs.
  - the decision accuracy split by the axis a pair moves. Its depth pairs are
    that shared decision inside the pooled lines.
- **English benchmarks** are the tasks the registry tags English
  (`utils.assign_language`), in every variant: the originals, their RF and
  LLM-RF rewrites, and the bBPB twins. A bBPB twin scores the gold answer's
  bits per byte, so lower is better. We flip its sign before reading any gap,
  and the gate never applies to it because it has no chance level. The
  English validation BPB (`bpb_dclm`) is read beside them, never pooled with
  them.
- **The bBPB twins exist at every checkpoint of every size** in the ladder
  report of 2026-10-10 04:31, the snapshot this README reads. They therefore enter the decision-accuracy lines at every drawn
  proxy (98 of 122–134 tasks), and their SNR rows read the full noise window.
  Because the twins outnumber the accuracy tasks about three to one, both
  sections give the accuracy-only and the twin-only reading beside the pooled
  one.
- **Rules.** The gate of rule 1 blanks every at-chance (task, size) cell (for
  decision accuracy, at the proxy and at the reference). Rule 5 asks for three
  pairs per DA cell, and rule 4 sets the noise window of the SNR. Sizes run
  from 90M to 1.7B (rule 10), and the per-size task populations move and are
  counted (rule 13).

<!-- BEGIN auto:english-setup (english_only.py --pool predictivity) -->
Pool `predictivity`, ladder-report snapshot **2026-10-11 04:43**, gate `predictivity`. Every score is from seed 1904, and the seed yardstick uses the replicate seeds of `predictivity_seeds`. 217 English tasks: 113 accuracy-scored (77 original, 35 rf, 1 rfgm), 103 bBPB twins and the English validation BPB `bpb_dclm`.

- **Mirrored family groups** (decision accuracy and the first SNR reading): L1 — `lm-L1-dclmP-deep-seed1904`, `lm-L1-deep-seed1904`, `lm-L1-fweb-deep-seed1904`, `lm-L1-shallow-seed1904`; L2 — `lm-L2-ES-deep-seed1904`, `lm-L2-ZH-deep-seed1904`, `lm-L2-deep-seed1904`, `lm-L2-shallow-seed1904`; L15 — `lm-L15-AT3-deep-seed1904`, `lm-L15-deep-seed1904`, `lm-L15-schemeB-deep-seed1904`, `lm-L15-shallow-seed1904`; L30 — `lm-L30-AT3-deep-seed1904`, `lm-L30-deep-seed1904`, `lm-L30-schemeB-deep-seed1904`, `lm-L30-shallow-seed1904`. Left out because they have a different number of such families: L8 (3), L50 (3).
- **Deep against shallow** (the second SNR reading): the two baseline cells at L1, L2, L8, L15, L30, L50.
- **The decision-accuracy population** (multi-axis, the English benchmarks above chance at the proxy and the reference that every family of every group has): 90M 127 (29 acc, 98 bbpb), 175M 129 (31 acc, 98 bbpb), 350M 132 (34 acc, 98 bbpb), 600M 137 (39 acc, 98 bbpb), 1B 138 (40 acc, 98 bbpb).
<!-- END auto:english-setup -->

## Highlighted result

![English scores](pretraining/predictivity/english_only_scores.png)

Pool `predictivity` (seed 1904, every data build, swiglu left out), ladder-report snapshot 2026-10-10 04:31, outputs of the 2026-10-10 refresh (checkpoint noise detrended). Final checkpoints, 90M–1.7B, gate `predictivity`. The L1 baseline cell against the same-depth baseline cell of L2, L8, L15, L30 and L50, on the 30–40 accuracy-scored English tasks above chance at each drawn size (350M is thin for deep).

Key findings:

- **The English-only cells are better at English, but only by about one accuracy point at 1.7B.** Deep L1 wins more than half of the English accuracy tasks against every other L at every drawn size (25 of 25 (size, L) cells), and its mean gap at 1.7B is 1.1 points over 40 tasks. Pooled over every L, it wins 0.65–0.76 of the comparisons per size. Shallow L1 wins 30 of 30 cells, with a gap of 0.8 points at 1.7B.
- **L1 has the lower English BPB in every comparison.** Its `bpb_dclm` is lower in all 30 (size, L) comparisons at both depths, and the deep gap shrinks from 0.049 bits per byte at 90M to 0.028 at 1.7B (0.782 against 0.810).
- **Part of the lead is larger than the seed noise.** Deep L1 is ahead by more than two seed standard deviations (sd) in 0.50 (175M), 0.43 (600M) and 0.47 (1B) of the (task, L) comparisons, and behind by as much in only 0.03–0.07. The seed sd is the std of the deep L1 cell over its three seeds.
- **L1 clears chance at a smaller size on more tasks, but its share of tasks above chance leads less often.** Against each of the five L, deep L1 clears chance from a smaller size on more tasks than from a larger one (99 against 30 (task, L) pairs). Its share of English tasks above chance is ahead in only 15 of 25 (size, L) cells, tied in 5 and behind in 5 (0.44 against a median of 0.39 at 1.7B).
- **English scales about as regularly at L1 as elsewhere.** The 1.7B English BPB forecast from 90M–1B misses by 1.0 % at L1, against 1.1–1.6 % at the other L. The median R² of the English accuracy fits is 0.88 at L1, against 0.83–0.93 elsewhere (39 fits each).
- **Decision accuracy is higher at L1, but the decisions differ.** With four families per L, DA-size (multi-axis, no filter) is 0.58–0.63 at 90M–1B for L1, against 0.45–0.53 for L2/L15/L30. On the depth pair alone, the one decision all groups share, it is 0.51 against 0.44–0.47.
- **On the shared deep-against-shallow decision, L1's SNR leads on accuracy tasks but not on the bBPB twins.** On accuracy tasks alone, L1's median log10 SNR is highest at 90M, 175M and 1.7B, 0.01 below L8 at 600M, and behind L8 and L30 at 1B (0.06 against 0.18 and 0.17). On the bBPB twins alone it is mid-pack, so the pooled verdict is "partly" (ahead in 15, tied in 3, behind in 7 of 25 cells).
- **L1's DA-size lands near DataDecide's.** It is below, inside and above DataDecide's range in 5, 6 and 2 of 13 format-matched (task, size) cells, with a median ratio of 0.90. The SNRs are not comparable.

[english_only_scores.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.png) · [english_only_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.csv) · [english_only_verdict.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_verdict.csv)

## Scores

This is the first expectation, and the one a broken training would fail
first: at matched size and depth, the English-only cell scores higher on
English.

<!-- BEGIN auto:english-scores (english_only.py --pool predictivity) -->
![English scores](pretraining/predictivity/english_only_scores.png)

![English scores per benchmark](pretraining/predictivity/english_only_scores_by_benchmark.png)

Key findings (on the accuracy-scored English tasks above chance at each size, so the population changes with the size and the task counts are listed per size):

- **Deep**: L1 is ahead in 0.65–0.76 of the (task, L) comparisons per size. Mean gap 90M -3.5, 175M 2.2, 350M 1.7, 600M 2.0, 1B 1.5, 1.7B 1.1 accuracy points (positive = L1 better), over 90M 35, 175M 32, 350M 34, 600M 39, 1B 41, 1.7B 40 tasks.
- **Shallow**: L1 is ahead in 0.62–0.81 of the (task, L) comparisons per size. Mean gap 90M -2.1, 175M 2.5, 350M 1.6, 600M 1.9, 1B 1.1, 1.7B 0.8 accuracy points (positive = L1 better), over 90M 35, 175M 32, 350M 34, 600M 39, 1B 41, 1.7B 40 tasks.
- **Against the seed noise** (deep, sizes with replicate seeds): L1 is ahead by more than 2 seed sd in 175M 0.50, 600M 0.43, 1B 0.46 of the comparisons and behind by as much in 175M 0.03, 600M 0.04, 1B 0.06.
- **Where L1 is far behind** (deep, mean gap over every other L below −10 pp): `cultural_bench_easy_canada` at 90M (-69 pp), `cultural_bench_easy_australia` at 90M (-51 pp), `cultural_bench_easy_south_africa` at 90M (-37 pp), `cultural_bench_easy_nigeria` at 90M (-32 pp), `rf_cultural_bench_easy_united_states` at 350M (-11 pp). A letter-format task can pass the gate with a constant answer that matches the majority gold label, which rule 1 does not test for. So when every L but one sits far above chance and the other sits near zero, the cell shows a letter preference, not English skill. The median gaps are 90M 1.5, 175M 1.7, 350M 1.7, 600M 2.0, 1B 1.6, 1.7B 1.6 pp.
- **English BPB** (bpb_dclm, deep): L1 is lower than every other L at every size, by a mean of 90M 0.049, 175M 0.045, 350M 0.042, 600M 0.041, 1B 0.033, 1.7B 0.028 bits per byte (at 1.7B, L1 0.782 against 0.810).
- **bBPB** (deep, sign flipped so that positive = L1 better): L1 is ahead in 0.54–0.80 of the comparisons per size. Mean gap 90M 0.063, 175M 0.080, 350M 0.069, 600M 0.153, 1B -0.304, 1.7B 0.094 bits, median 90M 0.065, 175M 0.071, 350M 0.048, 600M 0.062, 1B 0.009, 1.7B 0.027. At 1B the L1 deep value sits more than 0.3 bits above both neighbouring sizes on 21 of 103 tasks (bbh_mcq-bbpb 16, acp_bench_cloze-bbpb 2, bbh_cloze-bbpb 2, bbh_mcq-rf-bbpb 1), against 8 of 515 (task, L) cells for the other L's deep cells. Without those tasks the 1B mean gap is 0.023 instead of -0.304, so the spike belongs to that one cell, not to English. On those tasks the L1 deep 1B run of seed 1904 averages 3.17 bits, against 1.71 for seed 28, 2.14 for seed 1797 (the replicate seeds). It sits more than 0.3 bits above every replicate, so the spike belongs to that run, not to the size.
- **By L at 1.7B** (deep, mean gap / win share): L2 1.2 pp / 0.75, L8 1.0 pp / 0.70, L15 0.8 pp / 0.62, L30 1.2 pp / 0.70, L50 1.5 pp / 0.72.

[english_only_scores.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.png) · [english_only_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores.csv) · [english_only_scores_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_by_benchmark.png) · [english_only_scores_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_by_benchmark.csv) · [english_only_scores_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_per_task.csv) · [english_only_scores_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_summary.csv)
<!-- END auto:english-scores -->

Follow-ups:

- The same gaps per English training token instead of per size. An L1 cell
  of size N has seen as much English as an L > 1 cell of roughly 2N, so this
  tells whether L1's lead is just the token count.
- The DCLM-without-edu and FineWeb L1 cells against the baseline L1 cell.
  This puts the English-corpus effect next to the language-count effect, on
  the same tasks.
- The bBPB gap of the L1 deep cell at 1B (mean −0.320 bits against a median
  of 0.008) does not come from one bad checkpoint. Its `bbh_mcq` twins rise
  from 1.6 bits at 10 % of the run to 3.2–3.5 over the last four checkpoints,
  against 1.5–1.8 for L2 and L15 deep and 1.6–2.3 for the cell's replicate
  seeds 28 and 1797 at the same steps (`bench_bpb.csv`). The rise belongs to
  that run, so read the 1B bBPB gap on the replicate mean.
- The 90M deep mean gap (−3.5 points against a median of 1.5) without the
  `cultural_bench_easy` letter tasks the outlier bullet names, to see whether
  L1 trails at 90M at all.
- Next: whether higher scores also mean clearing chance sooner,
  [Above random](#above-random).

### The scores on one task set per line

The gap lines average, at each size, the English tasks above chance at that size (30–40 per size). A line can therefore move just because its tasks changed (rule 13). This figure keeps, per line, only the (task, L) comparisons that have a value at every size (solid), with the moving line dashed behind. Panel (c)'s shares use the same comparisons, so they get the same treatment.

![English scores on one task set per line](pretraining/predictivity/english_only_scores_fixed_tasks_paper.png)

Pool `predictivity`, seed 1904, finals, accuracy-scored English tasks above chance. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scores_fixed_tasks_paper.csv)

Key findings:

- **The negative L1 gap at 90M is a population effect.** On the 29 tasks every drawn size has, deep L1 leads the other language settings by 2.6 points at 90M, against −3.5 on the 35 tasks above chance there. Shallow L1 goes from −2.4 to +1.4 (27 tasks at every size).
- **On one task set, L1 is ahead at every size.** The deep gap reads 2.6, 2.4, 1.7, 1.5 and 1.2 points at 90M, 175M, 600M, 1B and 1.7B (no 350M point), and the shallow gap 0.8–2.7. Above 90M, the fixed and the moving readings differ by less than half a point.
- **The share of tasks L1 leads rises at 90M.** It goes from 0.65 to 0.78 for deep and from 0.60 to 0.70 for shallow.

Follow-ups:

- **The other rq13 figures on fixed tasks.** The gate and DA figures read the same moving English populations.

## Above random

Higher scores should also show up as English benchmarks clearing chance at a
smaller size. This section reads [the gate's](../rq00_gate_and_curves/README.md)
own run-level verdict per cell.

<!-- BEGIN auto:english-gate (english_only.py --pool predictivity) -->
![Above random](pretraining/predictivity/english_only_above_random.png)

Key findings:

- **Population**: the English tasks with a chance level that every drawn cell scores, 90M 112, 175M 112, 350M 112, 600M 112, 1B 112, 1.7B 112. A size with fewer than half the largest population is left out of the lines and of the shares below (none).
- **Deep**: share of English tasks above chance, L1 90M 0.30, 175M 0.32, 350M 0.35, 600M 0.41, 1B 0.38, 1.7B 0.46, against a median over the other L of 90M 0.32, 175M 0.29, 350M 0.31, 600M 0.36, 1B 0.38, 1.7B 0.40.
- **Shallow**: share of English tasks above chance, L1 90M 0.29, 175M 0.37, 350M 0.36, 600M 0.40, 1B 0.37, 1.7B 0.43, against a median over the other L of 90M 0.31, 175M 0.33, 350M 0.30, 600M 0.35, 1B 0.37, 1.7B 0.38.
- **Deep, L1 vs L2** (53 tasks that either cell clears): L1 clears chance at a smaller size on 0.36 of them, at the same size on 0.53 and at a larger size on 0.11.
- **Deep, L1 vs L8** (55 tasks that either cell clears): L1 clears chance at a smaller size on 0.36 of them, at the same size on 0.49 and at a larger size on 0.15.
- **Deep, L1 vs L15** (53 tasks that either cell clears): L1 clears chance at a smaller size on 0.42 of them, at the same size on 0.47 and at a larger size on 0.11.
- **Deep, L1 vs L30** (57 tasks that either cell clears): L1 clears chance at a smaller size on 0.35 of them, at the same size on 0.51 and at a larger size on 0.14.
- **Deep, L1 vs L50** (56 tasks that either cell clears): L1 clears chance at a smaller size on 0.36 of them, at the same size on 0.50 and at a larger size on 0.14.

[english_only_above_random.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random.png) · [english_only_above_random.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random.csv) · [english_only_above_random_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_above_random_per_task.csv)
<!-- END auto:english-gate -->

Follow-ups:

- The same crossing read per benchmark family, to see which English
  benchmarks the extra English tokens lift over chance first.
- Next: whether the extra English makes the ladder scale more regularly,
  [Scaling predictability](#scaling-predictability).

## Scaling predictability

More English tokens should not make English scale less regularly. We take
two readings from [scaling predictability](../rq01_scaling_predictability/README.md#methodology).
The first is a forecast: the error of a power law fitted on the proxies when
it predicts the 1.7B English BPB. The second describes the ladder (rule 11):
the R² of the log-linear fit of every English accuracy task over the rungs
every L shares, the 1.7B reference included.

<!-- BEGIN auto:english-scaling (english_only.py --pool predictivity) -->
![Scaling](pretraining/predictivity/english_only_scaling.png)

Key findings:

- **English BPB, fit on 90M–1B** (a forecast from the proxies): absolute error of the 1.7B prediction, L1 deep 1.1 %, L1 deep dclmP 1.1 %, L1 deep fweb 2.0 %, L1 shallow 0.8 %, L2 deep 1.3 %, L2 shallow 1.0 %, L8 deep 1.7 %, L8 muon 0.6 %, L8 shallow 1.0 %, L8 swiglu 1.4 %, L15 deep 1.9 %, L15 muon 0.7 %, L15 shallow 0.9 %, L15 swiglu 1.6 %, L30 deep 1.5 %, L30 muon 0.7 %, L30 shallow 1.1 %, L30 swiglu 1.4 %, L50 deep 1.9 %, L50 shallow 0.9 %.
- **English accuracy tasks** (deep): median R² of the log-N fit, with the number of fits in brackets, L1 0.84 (39), L2 0.81 (39), L8 0.88 (39), L15 0.87 (39), L30 0.90 (39), L50 0.84 (39). The fit uses the rungs the L1 deep cell has, 1.7B included, so it describes the ladder and is not a forecast (rule 11).

[english_only_scaling.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling.png) · [english_only_scaling.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling.csv)
<!-- END auto:english-scaling -->

Follow-ups:

- The power law on the English BPB of each L1 build's own corpus. Here the
  DCLM-without-edu and FineWeb cells are scored on the edu-filtered DCLM
  validation set, which is off their training distribution.
- Next: whether the English benchmarks fall into the same scaling regimes at
  L1 as on the multilingual ladder, [Scaling regimes at L1](#scaling-regimes-at-l1).

### Scaling regimes at L1

This figure draws the [scaling regimes](../rq01_scaling_predictability/README.md#1-scaling-regimes-the-paper-figure)
of the paper's section on benchmark scaling behaviour for the monolingual
comparison point. A task's scaling regime says whether its score follows a
fit across sizes, along training, both or neither. The three English corpora
at L1 get one family each, under the same gate and the same fits. All three
have finished and been evaluated at every rung from 90M to 1.7B. The
DCLM-Edu cell's 350M evaluation covers only 18 of the 212 English tasks it
has at the other sizes, so that size drops out of its trajectory medians.

<!-- BEGIN auto:english-regimes (english_regimes.py --pool predictivity) -->
![Scaling regimes at L1](pretraining/predictivity/english_only_scaling_regimes.png)

Pool `predictivity`: the L1 deep seed-1904 cell of each English corpus (DCLM-Edu = data A, DCLM = DCLMP, FineWeb = FWEB), 90M–1.7B, gate `predictivity`. The fits and regimes are those of the scaling-predictability figure, with one size fit per task (a single L) and the median trajectory R² over ≥ 2 sizes. The tasks are the English tasks (originals, RF, LLM-RF, bBPB twins) and `bpb_dclm`. The corpora share the gate, and they read the same number of tasks. rq01's grey points cover every trained language. Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool predictivity`.

Key findings (share of tasks per regime, where both = predictable across size and along training):

- **DCLM-Edu** (143 tasks, 39 accuracy-scored): both 45%, training only 5%, size only 13%, weak 22%, declines 16%. Median R² 0.70 across size and 0.52 along training. On the accuracy-scored tasks alone, both 54%.
- **DCLM** (143 tasks, 39 accuracy-scored): both 45%, training only 3%, size only 17%, weak 21%, declines 15%. Median R² 0.68 across size and 0.53 along training. On the accuracy-scored tasks alone, both 54%.
- **FineWeb** (143 tasks, 39 accuracy-scored): both 69%, training only 4%, size only 8%, weak 12%, declines 7%. Median R² 0.87 across size and 0.68 along training. On the accuracy-scored tasks alone, both 49%.
- **The same single-fit reading at every other L** (the English tasks of the deep data-A cell, 143 tasks): predictable across both 45%–56% at L2–L50 (L2 45%, L8 48%, L15 56%, L30 51%, L50 52%).
- **every K (rq01)** (1107 tasks, every trained language, medians over the L): both 78%, training only 1%, size only 9%, weak 8%, declines 4%. A median over six fits is smoother than a single fit, so this is the figure's backdrop. The bullet above is the like-for-like comparison.
<!-- END auto:english-regimes -->

Follow-ups:

- The same panels per benchmark family. FineWeb's lead is in the bBPB twins
  (98 of its 132 tasks). On the accuracy-scored tasks alone it ties DCLM at
  48 %, and DCLM-Edu's 54 % comes from six tasks only that cell has (next
  section). Which twins FineWeb scales on is the next question.
- A replicate seed of the FineWeb L1 cell. With one family per corpus, a
  corpus effect cannot be told from a seed effect.
- Next: the same panels on the English benchmarks alone, without the bBPB
  twins and the grey backdrop,
  [Scaling regimes at L1 on the English benchmarks](#scaling-regimes-at-l1-on-the-english-benchmarks).

### Scaling regimes at L1 on the English benchmarks

The figure above mixes two things into the L1 reading. The first is the bBPB
twins (98 of its 132–138 tasks, never gated). The second is rq01's grey
points, which come from every L. This version keeps only what the
monolingual English cells score on the English benchmarks: the
accuracy-scored English tasks (originals, RF and LLM-RF twins, each gated as
in rq01). It drops the bBPB twins, `bpb_dclm` and every point from an L > 1
cell. The RF twins stay because they are ordinary benchmarks in the pool
(RULES.md), scored by accuracy against a chance level. The bBPB twins do not
qualify. They score the gold answer's bits per byte rather than the
benchmark's own metric, and they have no chance level, so they pass the gate
at every size.

<!-- BEGIN auto:english-regimes-l1-accuracy (english_regimes.py --pool predictivity) -->
![Scaling regimes at L1 on the English benchmarks](pretraining/predictivity/english_only_scaling_regimes_l1_accuracy.png)

Pool `predictivity`: the L1 deep seed-1904 cell of each English corpus, 90M–1.7B, gate `predictivity`, with the fits and regimes of the figure above. Only English accuracy tasks are drawn (DCLM-Edu: 20 RF, 18 original, 1 LLM-RF): no bBPB twin, no `bpb_dclm` and no point from an L > 1 cell. The three corpora read the same tasks. Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool predictivity`.

Key findings (share of tasks per regime, where both = predictable across size and along training):

- **DCLM-Edu** (39 tasks): both 54%, training only 3%, size only 13%, weak 18%, declines 13%. Median R² 0.84 across size and 0.56 along training.
- **DCLM** (39 tasks): both 54%, training only 5%, size only 23%, weak 8%, declines 10%. Median R² 0.88 across size and 0.59 along training.
- **FineWeb** (39 tasks): both 49%, training only 5%, size only 15%, weak 26%, declines 5%. Median R² 0.89 across size and 0.53 along training.
- **Predictable across both, per format**: RF (20 tasks) DCLM-Edu 30%, DCLM 25%, FineWeb 30%; original (18 tasks) DCLM-Edu 78%, DCLM 83%, FineWeb 67%; LLM-RF (1 task) DCLM-Edu 100%, DCLM 100%, FineWeb 100%.
- **Same regime in all three corpora**: 19 of 39 tasks, 17 of them predictable across both.

[english_only_scaling_regimes_l1_accuracy.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling_regimes_l1_accuracy.png) · [english_only_scaling_regimes_l1_accuracy.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling_regimes_l1_accuracy.csv)
<!-- END auto:english-regimes-l1-accuracy -->

Key findings:

- **On the English benchmarks alone, the corpus effect of the figure above
  disappears.** FineWeb's 70 % against 42 % came from the bBPB twins. On the
  33 accuracy tasks the three corpora share, each corpus puts 16 (48 %) in
  the predictable regime. DCLM-Edu's 54 % comes from six tasks only its cell
  has (BoolQ, PIQA, WinoGrande and three ToKSuite tasks), five of them
  predictable across both.
- **What stays out of that regime is mostly the RF twins of BBH and
  ACP-Bench.** Of their 14 English tasks, only date understanding (all three
  corpora) and penguins in a table (FineWeb) are predictable across both.

Follow-ups:

- The BBH and ACP-Bench RF twins along training, per size. Do they sit near
  chance on the small rungs (a fit on few sizes), or stay flat above it?
- Next: which benchmarks are in each regime, and whether the well-known ones
  (MMLU, ARC, HellaSwag, OpenBookQA) are among the predictable ones,
  [Which English benchmarks scale at L1](#which-english-benchmarks-scale-at-l1).

### Which English benchmarks scale at L1

The figure above gives shares. This version names every point, with one
panel per corpus (the figure's bottom row, the one that sets the regime). Its
table adds the English accuracy tasks the gate filters out, so a benchmark
that never leaves chance shows up as `filtered` instead of missing.

<!-- BEGIN auto:english-regimes-l1-tagged (english_regimes.py --pool predictivity) -->
![Scaling regimes at L1, every task named](pretraining/predictivity/english_only_scaling_regimes_l1_accuracy_tagged.png)

Pool `predictivity`: the figure above with every point named, one panel per corpus (its panel (b)). The L1 deep seed-1904 cell of each English corpus, 90M–1.7B, gate `predictivity`, English accuracy tasks (drawn: DCLM-Edu 39, DCLM 39, FineWeb 39). The table adds the English accuracy tasks the gate filters out (regime `filtered`: above chance at fewer than three sizes, so no fit): DCLM-Edu 74, DCLM 74, FineWeb 74. Regenerate with `python analysis/rq13_english_only/english_regimes.py --pool predictivity`.

Key findings (the well-known English benchmarks, per corpus):

- **Predictable across both in all three corpora**: MMLU RF, Global-MMLU RF, ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA, XWinograd, LAMBADA, CommonsenseQA RF, Belebele RF, XStoryCloze, XNLI.
- **MMLU**: filtered by the gate in all three.
- **Global-MMLU**: filtered by the gate in all three.
- **Global PIQA**: weak (DCLM-Edu), size only (DCLM), weak (FineWeb).
- **CommonsenseQA**: filtered by the gate in all three.
- **Belebele**: filtered by the gate in all three.
- **TruthfulQA (mc2)**: filtered by the gate in all three.

[english_only_scaling_regimes_l1_accuracy_tagged.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling_regimes_l1_accuracy_tagged.png) · [english_only_scaling_regimes_l1_accuracy_tagged.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_scaling_regimes_l1_accuracy_tagged.csv)
<!-- END auto:english-regimes-l1-tagged -->

Key findings:

- **The well-known benchmarks are the predictable ones.** ARC-Easy,
  ARC-Challenge, HellaSwag, OpenBookQA, LAMBADA, XWinograd, XStoryCloze, XNLI
  and the RF versions of MMLU, Global-MMLU, Belebele and CommonsenseQA are
  predictable across both in all three corpora. They are 12 of the 14 tasks
  in that regime everywhere.
- **The popular benchmarks that drop out never leave chance.** MMLU,
  Global-MMLU, Belebele and CommonsenseQA in their letter format, and
  TruthfulQA, are filtered by the gate in all three corpora, so they never
  get a fit. The RF twin of each of the four letter-format benchmarks passes
  the gate.
- **Global PIQA (English) is the one well-known benchmark above chance that
  does not scale.** It is weak in DCLM-Edu and FineWeb and size only in DCLM.
  The other tasks outside the predictable regime are the BBH and ACP-Bench RF
  subtasks, the two CulturalBench-easy RF tasks, MathQA, MultiBLiMP and
  PAWS-X, and BoolQ (size only, scored by the DCLM-Edu cell alone). All but
  BoolQ and the US CulturalBench-easy RF task (weak in all three) change
  regime with the corpus.

Follow-ups:

- Global PIQA along training at each size. Does the English task sit just
  above chance (a fit on noise), or does it plateau early?

- Next: whether a small L1 proxy also ranks the L1 variants like 1.7B does,
  [Decision accuracy](#decision-accuracy).

## Decision accuracy

Does a small L1 proxy rank the L1 design variants as the 1.7B L1 cells do?
And does it do so at least as well as a proxy at another L ranks that L's
variants? The kernel, the pair sets and the band come from
[decision accuracy](../rq02_decision_accuracy/README.md#methodology). The
split by axis shows how much of the answer comes from the mix of decisions.

<!-- BEGIN auto:english-da (english_only.py --pool predictivity) -->
![DA-size, multi-axis](pretraining/predictivity/english_only_da_size_by_L_multi_axes.png)

![DA-size, mono-axis](pretraining/predictivity/english_only_da_size_by_L_mono_axis.png)

Key findings (DA-size against 1.7B, no reliability filter, ≥ 3 pairs per task, ± the half-width of the 90 % leave-one-family-out band). The lines pool the English benchmarks above chance at the proxy and the reference (gate `predictivity`) that every family of every group has. The decisions differ by L.

- **Multi-axis**: L1 0.58 ±0.07, 0.60 ±0.05, 0.62 ±0.04, 0.61 ±0.18, 0.63 ±0.15; L2 0.51 ±0.06, 0.51 ±0.04, 0.51 ±0.06, 0.53 ±0.11, 0.45 ±0.04; L15 0.51 ±0.10, 0.45 ±0.05, 0.49 ±0.09, 0.46 ±0.05, 0.50 ±0.08; L30 0.49 ±0.03, 0.51 ±0.06, 0.48 ±0.08, 0.45 ±0.05, 0.51 ±0.09 (90M, 175M, 350M, 600M, 1B).
- **Mono-axis**: L1 0.59 ±0.15, 0.59 ±0.07, 0.62 ±0.04, 0.62 ±0.21, 0.60 ±0.20; L2 0.51 ±0.13, 0.51 ±0.03, 0.50 ±0.08, 0.51 ±0.11, 0.45 ±0.08; L15 0.48, 0.44, 0.47, 0.43, 0.49; L30 0.51, 0.48, 0.51, 0.42, 0.50 (90M, 175M, 350M, 600M, 1B).
- **By the axis a pair moves** (multi-axis, pooled over the drawn proxies, number of decisions in brackets): L1 depth (deep vs shallow) 0.52 (663), two axes at once 0.61 (1326), data build 0.63 (1989); L2 depth (deep vs shallow) 0.48 (663), two axes at once 0.52 (1326), data build 0.50 (1989); L15 temperature (T=1 vs T=3) 0.44 (663), depth (deep vs shallow) 0.44 (663), two axes at once 0.51 (1989), data build 0.51 (663); L30 temperature (T=1 vs T=3) 0.48 (663), depth (deep vs shallow) 0.49 (663), two axes at once 0.49 (1989), data build 0.49 (663). The depth pair (deep against shallow) is the one decision every group shares.
- **bBPB twins alone** (multi-axis, every cell with ≥ 3 pairs at every proxy, including thin sizes and cells that miss a family, so not one population): L1 0.61 (6 pairs per task); L2 0.51 (6 pairs per task); L15 0.50 (6 pairs per task); L30 0.49 (6 pairs per task).
- **English BPB** (multi-axis, mean over the proxies): L1 0.97; L2 0.87; L15 0.70; L30 0.80.

[english_only_da_size_by_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_multi_axes.png) · [english_only_da_size_by_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_multi_axes.csv) · [english_only_da_size_by_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_mono_axis.png) · [english_only_da_size_by_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_mono_axis.csv) · [english_only_da_size_by_L_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_L_both_axes.csv) · [english_only_da_size_by_pair_axis_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_by_pair_axis_both_axes.csv) · [english_only_da_size_per_task_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_da_size_per_task_both_axes.csv)
<!-- END auto:english-da -->

Key findings, read from `english_only_da_size_per_task_both_axes.csv` and `english_only_da_size_by_L_both_axes.csv` (multi-axis, the lines' shared cells, 6 pairs per task):

- **Four proxies are drawn, and the twins dominate them.** The lines read 122 tasks at 90M (24 accuracy, 98 bBPB), 124 at 175M (26, 98) and 134 at 600M and 1B (36, 98). 350M is thin (14 tasks) because the L1 deep cell there has only 18 of the 212 English tasks.
- **L1's lead holds on each scoring alone.** On accuracy tasks, L1 reaches 0.60, 0.60, 0.60 and 0.64 (90M, 175M, 600M, 1B) against 0.44–0.51, 0.44–0.48, 0.39–0.54 and 0.49–0.50 for L2/L15/L30. On the bBPB twins, L1 reaches 0.58, 0.60, 0.61 and 0.63 against 0.49–0.54, 0.46–0.53, 0.44–0.53 and 0.44–0.52.
- **The band separates L1 from L15 at 175M, and nowhere else.** L1's 90 % leave-one-family-out band at 175M (0.548–0.657) clears L15's upper end (0.502). It touches L2's upper end (0.548) and overlaps L30's band (0.445–0.582). At 90M (0.49–0.67), 600M (0.42–0.79) and 1B (0.48–0.78) it overlaps all three.
- **Most of the gap comes from the decisions, not from the proxy.** On the depth pair, the one decision every group shares, L1 is at 0.51 against 0.44–0.47. The L1 pairs that move the English corpus reach 0.64 (data build) and 0.61 (two axes at once).

Follow-ups:

- The seed-null DA of the L1 replicates at 175M, 600M and 1B (the DA that
  pairs of seeds of one design reach), as the floor to read these lines
  against.
- The English tasks the 350M L1 deep cell lacks (it has 18 of 212). They
  keep 350M thin in every section.
- Next: whether the same families separate more cleanly than they wobble,
  [Noise and SNR](#noise-and-snr).

## Noise and SNR

This section has three readings. The first is the SNR of
[noise and SNR](../rq03_noise_and_snr/README.md#methodology) (`rel_std`) over
the same mirrored groups, the four-family groups of the Setup. The second is
the like-for-like deep-against-shallow reading at every L. The third is the
relation [the surrogates](../rq04_surrogates/README.md) are built on: SNR at
the proxy against DA-size at the proxy.

<!-- BEGIN auto:english-snr (english_only.py --pool predictivity) -->
![SNR](pretraining/predictivity/english_only_snr.png)

Key findings:

- **The families mirroring L1's**: median log10 SNR over the paired tasks (90M: 132, 175M: 129, 350M: 132, 600M: 137, 1B: 139, 1.7B: 138 tasks): L1 0.49, 0.48, 0.55, 0.47, 0.39, 0.32; L2 0.25, 0.39, 0.46, 0.34, 0.27, 0.20; L15 0.25, 0.28, 0.44, 0.21, 0.24, 0.11; L30 0.23, 0.29, 0.55, 0.24, 0.21, 0.22 (90M, 175M, 350M, 600M, 1B, 1.7B).
- **Deep against shallow**: median log10 SNR over the paired tasks (90M: 121, 175M: 127, 350M: 131, 600M: 122, 1B: 131, 1.7B: 128 tasks): L1 0.13, 0.17, 0.17, 0.10, 0.08, -0.05; L2 0.08, 0.21, 0.29, 0.08, -0.06, -0.09; L8 0.17, 0.18, 0.27, 0.10, 0.06, -0.08; L15 0.17, 0.15, 0.23, 0.10, 0.10, -0.02; L30 0.10, 0.08, 0.19, 0.10, 0.05, -0.11; L50 0.11, 0.05, 0.38, -0.01, 0.12, -0.03 (90M, 175M, 350M, 600M, 1B, 1.7B).
- **SNR → DA-size, Spearman ρ per proxy** (mirrored families, the DA lines' tasks): L1 0.15, 0.36, 0.36, 0.36, 0.37; L2 0.29, 0.13, 0.04, 0.13, 0.19; L15 -0.08, -0.03, -0.02, -0.04, 0.04; L30 -0.04, 0.03, -0.08, 0.14, 0.07 (90M, 175M, 350M, 600M, 1B).

[english_only_snr.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr.png) · [english_only_snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr.csv) · [english_only_snr_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_snr_per_task.csv)
<!-- END auto:english-snr -->

Key findings, read from `english_only_snr_per_task.csv` (ungated tasks every group of the reading has, median log10 SNR at 90M / 175M / 600M / 1B / 1.7B):

- **On the mirrored families, L1 leads at every drawn size on accuracy tasks alone.** L1 reads 0.41 / 0.37 / 0.42 / 0.34 / 0.31 against L2 0.24 / 0.34 / 0.28 / 0.22 / 0.12, L15 0.27 / 0.29 / 0.12 / 0.25 / 0.12 and L30 0.32 / 0.17 / 0.19 / 0.14 / 0.14 (30 / 27 / 33 / 34 / 34 tasks). On the 98 twins alone, now on the full noise window, L1 also leads at every size (0.34–0.54 against 0.09–0.40). The pooled lead (15 of 15 in the verdict) therefore holds on each scoring.
- **On the shared decision, L1 is highest at three of the five drawn sizes on accuracy tasks alone.** Deep against shallow (19 / 20 / 19 / 23 / 25 tasks), L1's median log10 SNR is 0.32 / 0.23 / 0.10 / 0.06 / −0.03, against −0.00 to 0.17 / −0.10 to 0.16 / −0.10 to 0.11 / −0.10 to 0.18 / −0.21 to −0.04 for the other L's. It is 0.01 below L8 at 600M and behind L8 and L30 at 1B (0.18, 0.17). On the twins alone it is mid-pack (−0.05 to 0.16 against −0.07 to 0.23). The pooled reading in the generated block and the verdict's 15 / 3 / 7 of 25 therefore follow the twins.

Follow-ups:

- The signal and the noise parts separately (both are in
  `english_only_snr_per_task.csv`). Does L1's SNR come from a wider spread
  across its families or from quieter late checkpoints?
- The deep-against-shallow SNR of the twins per benchmark family, to see
  which twins put L1 mid-pack where its accuracy tasks lead.
- Next: whether the L1 numbers land where an external English ladder puts the
  same tasks, [Next to AllenAI DataDecide](#next-to-allenai-datadecide).

## SNR against decision accuracy on the AllenAI benchmarks

The surrogates find no statistic that predicts decision accuracy well on the
ladder. Is that multilinguality or our code? `allenai_snr.py` reads the SNR →
DA-size relation with one implementation on DataDecide, on L1 and on the
multilingual mirrored groups, all on the nine OLMES benchmarks we evaluate
(ARC-Easy, ARC-Challenge, BoolQ, CommonsenseQA RF, HellaSwag, MMLU RF,
OpenBookQA, PIQA, WinoGrande).

**SNR (`rel_std`) and multi-axis DA-size · our side: the mirrored four-family
groups, seed 1904, proxies 90M–1B against 1.7B, gated (rule 1), 6 pairs ·
DataDecide: 25 recipes, 150M–750M against 1B (rq07's table).** Statistic:
Spearman ρ of log10 SNR against DA-size over the (benchmark, proxy) points,
95 % bootstrap over benchmarks. Snapshot: the committed rq13 and rq07 tables
(2026-10-11 refresh).

![SNR against DA-size](pretraining/predictivity/english_only_allenai_snr_da_paper.png)

Key findings (`english_only_allenai_snr_da_summary.csv`):

- **The code reproduces the English result.** DataDecide: ρ = 0.89 [0.57, 0.96] over 27 points.
- **L1 keeps a positive relation, the many-language groups lose it.** L1 0.46 [−0.08, 0.81], L2 0.43 [−0.06, 0.75], L15 0.04 [−0.39, 0.33], L30 −0.10 [−0.40, 0.16] (39 points each).
- **SNR and DA fall with K on the same benchmarks.** Median SNR 6.0 (DataDecide), 2.9, 2.2, 1.7, 1.8; median DA-size 0.76, 0.67, 0.67, 0.50, 0.50.

Follow-ups:

- The decisions differ by group (recipes, English corpora, language lists), so
  a like-for-like reading needs one decision at every K: deep against shallow
  has one pair per cell, too few for a DA-size distribution.

[english_only_allenai_snr_da_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai_snr_da_paper.png) · [english_only_allenai_snr_da.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai_snr_da.csv) · [english_only_allenai_snr_da_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai_snr_da_summary.csv)

## Next to AllenAI DataDecide

The L1 numbers next to DataDecide's on the English tasks both evaluate:

- **Source.** DataDecide's per-task SNR and DA from [the external frameworks
  comparison](../rq07_external_frameworks/README.md#methodology)
  (`allenai_snr_variants_per_task.csv`, built by `build_allenai_variants.py`
  with the same per-model arrays and aggregators).
- **Tasks, paired by format.** lm-eval's ARC, HellaSwag and OpenBookQA score
  the answer strings, like OLMES's cloze-style RC format. Our
  `rf_` twins are the answer-string rewrite of a letter task, so they meet RC
  too (`rf_mmlu` ↔ `mmlu`, `rf_commonsense_qa` ↔ `csqa`). The letter originals
  meet OLMES's `:mc` (`commonsense_qa` ↔ `csqa:mc`). DataDecide has no
  `mmlu:mc` aggregate.
- **Sizes**, paired as the external frameworks comparison pairs them.
- **Definitions.** The table at the top of the block lists what differs
  between the two sides. The comparison therefore reads levels and
  directions, not decimals.

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

- **DA-size**: the ratio L1 / DataDecide on the same task has a median of 0.90 (range 0.68–1.40). The same ratio for the other L groups has a median of 0.55.
- **SNR**: the ratio L1 / DataDecide on the same task has a median of 0.54 (range 0.17–1.08). The same ratio for the other L groups has a median of 0.18.
- **No L1 DA-size** (at chance on our side, rule 1, or a family missing): `arc_challenge` 175M, 350M. DataDecide is not gated, so its range is taken only over the tasks where L1 has a value at that size.

| ours | DataDecide | quantity | sizes (ours / theirs) | L1 | L1 pairs / runs | other L median | DataDecide | DataDecide range (L1's tasks) | L1 vs range | L1 / DataDecide |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | arc_easy | da-size | 175M / 150M | 0.83 | 6 | 0.50 | 0.93 | 0.71–0.93 | inside | 0.90 |
| arc_easy | arc_easy | snr | 175M / 150M | 8.36 | 4 | 1.79 | 14.97 | 4.24–14.97 | (not graded) | 0.56 |
| arc_easy | arc_easy | da-size | 350M / 300M | 0.83 | 6 | 0.33 | 0.95 | 0.71–0.95 | inside | 0.87 |
| arc_easy | arc_easy | snr | 350M / 300M | 10.53 | 4 | 2.87 | 18.62 | 4.11–18.62 | (not graded) | 0.57 |
| arc_easy | arc_easy | da-size | 600M / 750M | 0.83 | 6 | 0.67 | 0.97 | 0.76–0.97 | inside | 0.86 |
| arc_easy | arc_easy | snr | 600M / 750M | 8.18 | 4 | 1.67 | 16.41 | 4.11–16.41 | (not graded) | 0.50 |
| arc_easy | arc_easy | snr | 1B / 1B | 7.23 | 4 | 1.88 | 15.67 | 4.09–15.67 | (not graded) | 0.46 |
| arc_challenge | arc_challenge | da-size | 600M / 750M | 0.83 | 6 | 0.50 | 0.92 | 0.76–0.97 | inside | 0.91 |
| arc_challenge | arc_challenge | snr | 600M / 750M | 7.47 | 4 | 2.65 | 11.85 | 4.11–16.41 | (not graded) | 0.63 |
| arc_challenge | arc_challenge | snr | 1B / 1B | 8.53 | 4 | 1.31 | 12.61 | 4.09–15.67 | (not graded) | 0.68 |
| hellaswag | hellaswag | da-size | 175M / 150M | 0.67 | 6 | 0.83 | 0.74 | 0.71–0.93 | below | 0.90 |
| hellaswag | hellaswag | snr | 175M / 150M | 2.81 | 4 | 3.02 | 5.69 | 4.24–14.97 | (not graded) | 0.49 |
| hellaswag | hellaswag | da-size | 350M / 300M | 0.67 | 6 | 0.67 | 0.83 | 0.71–0.95 | below | 0.80 |
| hellaswag | hellaswag | snr | 350M / 300M | 2.95 | 4 | 4.18 | 8.58 | 4.11–18.62 | (not graded) | 0.34 |
| hellaswag | hellaswag | da-size | 600M / 750M | 0.67 | 6 | 0.50 | 0.94 | 0.76–0.97 | below | 0.71 |
| hellaswag | hellaswag | snr | 600M / 750M | 3.59 | 4 | 2.27 | 12.48 | 4.11–16.41 | (not graded) | 0.29 |
| hellaswag | hellaswag | snr | 1B / 1B | 3.97 | 4 | 1.82 | 15.34 | 4.09–15.67 | (not graded) | 0.26 |
| openbookqa | openbookqa | da-size | 175M / 150M | 0.83 | 6 | 0.33 | 0.71 | 0.71–0.93 | inside | 1.17 |
| openbookqa | openbookqa | snr | 175M / 150M | 2.25 | 4 | 2.18 | 4.24 | 4.24–14.97 | (not graded) | 0.53 |
| openbookqa | openbookqa | da-size | 350M / 300M | 1.00 | 6 | 0.67 | 0.71 | 0.71–0.95 | above | 1.40 |
| openbookqa | openbookqa | snr | 350M / 300M | 2.43 | 4 | 2.22 | 4.11 | 4.11–18.62 | (not graded) | 0.59 |
| openbookqa | openbookqa | da-size | 600M / 750M | 1.00 | 6 | 0.67 | 0.76 | 0.76–0.97 | above | 1.32 |
| openbookqa | openbookqa | snr | 600M / 750M | 2.62 | 4 | 1.82 | 4.11 | 4.11–16.41 | (not graded) | 0.64 |
| openbookqa | openbookqa | snr | 1B / 1B | 1.73 | 4 | 1.32 | 4.09 | 4.09–15.67 | (not graded) | 0.42 |
| rf_commonsense_qa | csqa | da-size | 175M / 150M | 0.67 | 6 | 0.33 | 0.77 | 0.71–0.93 | below | 0.87 |
| rf_commonsense_qa | csqa | snr | 175M / 150M | 6.46 | 4 | 1.84 | 5.96 | 4.24–14.97 | (not graded) | 1.08 |
| rf_commonsense_qa | csqa | da-size | 350M / 300M | 0.50 | 6 | 0.17 | 0.74 | 0.71–0.95 | below | 0.68 |
| rf_commonsense_qa | csqa | snr | 350M / 300M | 4.10 | 4 | 3.39 | 7.45 | 4.11–18.62 | (not graded) | 0.55 |
| rf_commonsense_qa | csqa | da-size | 600M / 750M | 0.67 | 6 | 0.50 | 0.79 | 0.76–0.97 | below | 0.84 |
| rf_commonsense_qa | csqa | snr | 600M / 750M | 2.81 | 4 | 1.01 | 7.22 | 4.11–16.41 | (not graded) | 0.39 |
| rf_commonsense_qa | csqa | snr | 1B / 1B | 1.61 | 4 | 2.48 | 9.29 | 4.09–15.67 | (not graded) | 0.17 |
| rf_mmlu | mmlu | da-size | 175M / 150M | 0.83 | 6 | 0.33 | 0.89 | 0.71–0.93 | inside | 0.94 |
| rf_mmlu | mmlu | snr | 175M / 150M | 3.72 | 4 | 1.17 | 7.74 | 4.24–14.97 | (not graded) | 0.48 |
| rf_mmlu | mmlu | da-size | 350M / 300M | 1.00 | 6 | 0.50 | 0.91 | 0.71–0.95 | above | 1.10 |
| rf_mmlu | mmlu | snr | 350M / 300M | 7.38 | 4 | 1.98 | 10.69 | 4.11–18.62 | (not graded) | 0.69 |
| rf_mmlu | mmlu | da-size | 600M / 750M | 1.00 | 6 | 0.67 | 0.95 | 0.76–0.97 | above | 1.05 |
| rf_mmlu | mmlu | snr | 600M / 750M | 11.84 | 4 | 0.49 | 14.80 | 4.11–16.41 | (not graded) | 0.80 |
| rf_mmlu | mmlu | snr | 1B / 1B | 11.83 | 4 | 1.90 | 13.30 | 4.09–15.67 | (not graded) | 0.89 |

Reading the two sides:

- **DA-size shares the kernel, not the decisions.** Both sides count agreeing pairs. Ours counts only tasks that pass the gate at the proxy and the reference, and theirs is not gated. Ours ranks 4 families that differ in depth and English corpus, theirs 25 recipes. A cell above DataDecide's range is a 6-of-6 (our DA moves in steps of 1/6, theirs in steps of 1/300).
- **The SNRs are not comparable**, so they are not graded. DataDecide's signal is the spread of 25 recipes over very different corpora. Ours is the spread of families of which two differ only in depth or in one filter, and the two noises use different checkpoint windows. The like-for-like SNR reading is deep against shallow at every L ([Noise and SNR](#noise-and-snr)). The four-family comparison with the other L groups stays within the ladder and covers different decisions per L (Setup).

[english_only_allenai.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai.png) · [english_only_allenai.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai.csv) · [english_only_allenai_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_allenai_per_task.csv)
<!-- END auto:english-allenai -->

Follow-ups:

- Rebuild the AllenAI table under rule 4's window, so that the noise halves of
  the two SNRs share a definition.
- Evaluate `piqa`, `winogrande`, `boolq` and `social_iqa` on every L1 family
  (the `allenai` task group). Today only the DCLM-Edu deep cell scores the
  first three (and not at 350M), while a cell needs all four families. This
  would add four format-matched pairs.

## Validation verdict

One row per expectation, computed from the tables above. The grades are:

- **yes**: L1 is ahead of every comparator, or within 0.01 of it, at every
  drawn size.
- **partly**: L1 is behind somewhere and ahead somewhere.
- **not like-for-like**: the groups' decisions differ by L, so the row gives
  the numbers without a grade.
- **not comparable**: the definitions differ (the DataDecide SNR).
- **not computed**: the pool lacks the families the row needs.

<!-- BEGIN auto:english-verdict (english_only.py --pool predictivity) -->
| expectation | measure | L1 | comparator | confirmed? |
|---|---|---|---|---|
| higher English accuracy (deep) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 1.1 pp at 1.7B | yes |
| higher English accuracy (shallow) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 0.8 pp at 1.7B | yes |
| lower English BPB (deep) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 0.03 bits at 1.7B | yes |
| lower English BPB (shallow) | share of the tasks L1 wins against each L at each size (thin sizes left out), against one half; mean gap at the reference, every L pooled (positive = L1 better) | ahead 30, tied 0, behind 0 of 30 (size, L) cells | 0.03 bits at 1.7B | yes |
| clears chance earlier (deep) | comparators against which L1 stays above chance from a smaller size on more tasks than from a larger one; L1's share of English tasks above chance against each L, per size | 5 of 5 comparators (101 earlier, 36 later, (task, L) pooled) | share: ahead 20, tied 5, behind 5 of 30 (size, L) cells | partly |
| English BPB at least as predictable in size | median absolute relative error of the 1.7B prediction from 90M–1B (baseline build, deep and shallow), L1 against each L | 1.0 % | L2 1.1 %, L8 1.2 %, L15 1.3 %, L30 1.2 %, L50 1.4 % | yes |
| English accuracy at least as predictable in size | median R² of score vs log10 N (deep; the rungs the L1 deep cell has, 1.7B included: a reference-size quantity, rule 11), L1 against each L | 0.84 | L2 0.81, L8 0.88, L15 0.87, L30 0.90, L50 0.84 | partly |
| higher decision accuracy (mirrored families) | pooled DA-size per proxy, multi-axis, ahead 15, tied 0, behind 0 of 15 (size, L) cells; the decisions differ by L (at L1 three of six pairs move the English corpus) | mean 0.61 | L2 0.50, L15 0.48, L30 0.49; on the depth pairs alone: L1 0.52, L2 0.48, L15 0.44, L30 0.49 | not like-for-like |
| higher SNR (mirrored families) | median log10 SNR over the paired tasks per size, ahead 17, tied 1, behind 0 of 18 (size, L) cells; the families' differences differ by L | mean 0.45 | L2 0.32, L15 0.25, L30 0.29 | not like-for-like |
| higher SNR (deep against shallow, the shared decision) | median log10 SNR over the paired tasks per size, ahead 14, tied 4, behind 12 of 30 (size, L) cells | mean 0.10 | L2 0.08, L8 0.12, L15 0.12, L30 0.07, L50 0.10 | partly |
| L1 DA-size in DataDecide's range | (task, size) cells below / inside / above DataDecide's min–max over the shared tasks where L1 has a value at the matched size; median L1 / DataDecide on the same task (ours gated, decisions differ: see the README) | 6 / 6 / 4 of 16 | ratio 0.90 | yes |
| L1 SNR next to DataDecide | not comparable: DataDecide's signal is the spread of 25 recipes, ours of L1's families; the noise uses different checkpoint windows (the like-for-like reading is the deep-against-shallow row above) | 22 cells | ratio 0.54 (shown, not graded) | not comparable |

[english_only_verdict.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq13_english_only/pretraining/predictivity/english_only_verdict.csv)
<!-- END auto:english-verdict -->

## Outputs

Under `pretraining/<pool>/`. The driver runs `predictivity`, which holds
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
| `english_only_scaling_regimes_l1_accuracy.png/.csv`, `…_paper.png/.csv` | `english_regimes.py`: the same regimes for the L1 cells alone on the English accuracy tasks (no bBPB twin, no `bpb_dclm`, no L > 1 point), one row per (corpus, task) with its `format` |

## How to run

```bash
cd src/signal-and-noise
python analysis/rq13_english_only/english_only.py --pool predictivity
# --readme writes the blocks from another pool (by default only `predictivity` writes them)
python analysis/rq13_english_only/english_regimes.py --pool predictivity   # after rq01's regimes.py
```

It reads the above-random mask (the pool's own, else `predictivity`'s), the
replicate seeds of `predictivity_seeds` and the AllenAI table. Without the
mask or an L1 cell, it writes nothing and says why. Without the AllenAI
table, it skips that comparison.

When L1 has fewer than three families in the pool, decision accuracy is not
computed (rule 5). Because this analysis compares matched groups, the
mirrored-family SNR, the surrogate and the AllenAI comparison are skipped
with it. The run says so in a `!!! RULE 5` line, writes the
deep-against-shallow SNR and the verdict rows as "not computed", and leaves
the README to a pool that has the families.
