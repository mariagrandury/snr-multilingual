# The above-random gate and the ladder's curves

## Research question

> Which benchmarks clear chance at which size of the ladder, and which separate
> the language settings most? Every later analysis reads only the cells this
> gate keeps.
>
> The folder also holds the ladder's descriptive curves (training loss and
> benchmark accuracy along the run), drawn from the analysis loader so they
> share every assumption of the later analyses.

<!-- BEGIN auto:highlight (run_apertus.py --pool predictivity) -->
## Highlighted result

- **The benchmarks that separate the language settings most: `cultural_bench_easy`, `bbpb_bbh_cloze`, `bbpb_bbh_mcq`** — top-3 families by Signal ((max−min)/mean of per-setting final scores) at 1.7B.
- **Above-random gate.** Of **1373 benchmarks, 821 clear chance at ≥1 size** and 778 at 1.7B (552 are random everywhere). The at-chance cells are removed before any SNR is computed; the breakdown by answer count below shows how much of the gate is an option-count effect.
<!-- END auto:highlight -->

## Setup

- **Snapshot.** Every hand-written number is from the ladder report
  2026-10-06 04:26, as regenerated on 2026-10-07 (commit `b316f53b`).
- **Gate pool.** `predictivity`: seed 1904, every cell of every ladder and
  data build, 161 runs with benchmark scores at 90M–1.7B (26–29 per size;
  deep 96, shallow 60, swiglu 5). By data scheme: A at T 1 77 runs, A at T 3
  24, B 48, C 12.
- **Curves pool.** `predictivity_seeds` (every seed) for the loss and
  benchmark curves of figure 5; sizes 90M–1.7B throughout (rule 10).
- **The gate covers every task the pool evaluates.** That includes the 532
  chance-level tasks no cell trains, read on every run; rule 2 drops them
  downstream, so the trained population is 841 of the 1,373 tasks with a
  chance level.
- **bBPB twins.** The `bbpb_` twins enter only the Signal ranking of figure
  2, and only as final-checkpoint values; nothing here reads them earlier in
  a run.

## Key findings (ladder report 2026-10-06 04:26)

![The gate in one figure](pretraining/predictivity/highlights.png)

*Pool `predictivity` (161 runs, seed 1904, 90M–1.7B). One-sided 95 % Wilson
gate; a (task, size) cell passes when at least half of the size's runs that
train the language clear chance (4 to 26 runs per cell).*

- **The gate removes most of the evaluation somewhere on the ladder.** On
  the 841 trained tasks, 312 (37 %) are above chance at 90M and 489 (58 %) at
  1.7B; 550 (65 %) are at chance at one size or more, 326 (39 %) at every size.
- **Over all 1,373 chance-level tasks the picture is the same.** 552 of
  1,372 (40 %) pass at 90M and 778 of 1,373 (57 %) at 1.7B; 552 (40 %) never
  pass.
- **It is an answer-format effect, not an option-count one.** On the trained
  languages at 1.7B the letter-answer families nearly vanish while their cloze
  twins pass: `belebele` 3 / 59 tasks against 57 / 59, `global_mmlu_full`
  1 / 29 against 29 / 29, `include_base_44` 3 / 36 against 29 / 36.
- **The format effect is not chance.** Over every language of each family,
  McNemar's discordant counts are 82 : 1, 34 : 0 and 29 : 1 (p < 10⁻⁷ each).
  Two- and four-option trained tasks pass alike at 1.7B (64 / 103, 62 %,
  against 402 / 691, 58 %).
- **The item count sets the bar.** The margin over chance a run needs for
  its Wilson bound to clear chance runs from +0.006 (the 14,042-item MMLU
  families) to +0.157 (`cultural_bench_easy`, 25 items), median task per
  family.
- **Much of what the ladder cannot read is a size floor or the majority
  rule (provisional).** Of the 86 tasks shared with the public models, 38 are
  at chance at every ladder size; public models read 27 of them at ≤ 1.7B and
  3 (`arc_eu`, `paws_ja`, `xnli_ar`) not up to 70B. The public models' mask
  still uses the retired +0.05 rule (figure 4).
- **Near misses.** Of the 29 of those 38 that a 1.7B cell trains, 23 have at
  least one 1.7B run whose Wilson bound clears chance. There the verdict is
  the half-of-the-runs rule, not only a floor.
- **Signal without the gate is not reliability.** The top task by Signal at
  1.7B, `rf_bbh_mcq_temporal_sequences` (3.39), scores 0.000–0.052 over the 26
  runs at 1.7B against a 0.25 chance: a large relative spread over a near-zero
  mean.

## Experimental setup

- **Accuracy-vs-FLOPs curves** (`pretraining/<pool>/{per_benchmark,per_language}/`):
  log-x FLOPs `6 × (N_non_emb + d·V) × D`, one curve per language setting
  (`plotted_mixes`: L1 … L50, deep, scheme A at T 1, seed 1904) and per size
  90M–1.7B, each rung at its own batch (rule 10). Only the top-3 families by
  Signal get curve grids; the 36-sweep pools overlay the external models to 70B.
- **Signal.** Per parent task (subjects collapse into the parent, languages
  stay distinct), (max − min) / mean of the six `plotted_mixes` cells' final
  scores at the 1.7B reference.
- **The above-random gate** ([`above_random.py`](above_random.py)) reads only
  the raw eval scores and the option counts (`n_options` in
  `configs/tasks.json`, derived from the evaluated samples where possible). It
  reads no other analysis's output, so every later analysis depends on the
  gate and never the reverse.
- **The rule.** A run is above chance on a task when the one-sided 95 % Wilson
  lower bound of its accuracy over the task's `n_items` clears `1/n_options`.
  A `(task, size)` cell is above random when at least half of the size's runs
  that train the language are (`MIN_SHARE`).
- **No fixed margin.** The old `mean score > 1/n_options + 0.05` rule is gone,
  so a 2-option task with 500 items and a 4-option task with 100 are held to
  the same evidence. The one-sided 95 % bound is the lower end of a two-sided
  90 % interval, which the auto table below calls "Wilson 90 %".
- **Propagation.** `run_apertus_snr_variants.py` NaN-s every at-chance cell,
  so the gate reaches every later analysis. Per-language BPB, the `bbpb_`
  twins and generative tasks have no chance level and are never gated.

<!-- BEGIN auto:above-random-example (above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary) -->
### The gate on one task: `include_v2_og_hungarian_hungary`

An educational reading of the rule above on one benchmark-language task (the gate is per task, never per family). (a) every run of `predictivity` that trains the language along training, the chance level and the score a run needs for its Wilson lower bound to clear chance at this task's item count, with one run's score − LCB band; (b) per size the final scores with their bounds, passers in colour, and the verdict; (c) the consequence — rq02's DA-size and DA-ckpt cells of the task, grey where the gate blanks them. The rule is the header's three lines. Regenerate with `python analysis/rq00_gate_and_curves/above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary`.

![The above-random gate on include_v2_og_hungarian_hungary](pretraining/predictivity/above_random_example.png)

Key findings:

- `include_v2_og_hungarian_hungary` has 4 options (0.25 chance) and 1,386 items, so a single run needs 0.270 (+0.020 over chance) for its LCB to clear chance; at n = 1,386 the bound sits 0.019 below the score.
- Verdict per size (runs passing / runs that train the language, ≥ 50% needed): 90M 1/9 → at chance, 175M 3/9 → at chance, 350M 4/9 → at chance, 600M 4/9 → at chance, 1B 5/9 → above, 1.7B 7/9 → above; the task enters every RQ from **1B** on and is grey below it.
- What that means in rq02 (panel c): a DA-size cell needs the proxy and the reference above chance, so `include_v2_og_hungarian_hungary` contributes only the 1B → 1.7B DA-size cell(s); a DA-ckpt cell needs the run's own size above chance, so it contributes only the 1B, 1.7B run(s). 8 of 11 cells are blanked; the task is never a decision at 90M, 175M, 350M, 600M.
- The highlighted run `lm-350M-L30-deep-seed1904` ends at 0.258 with LCB 0.239: not above chance — a score above the dotted line is not enough, the band's lower edge has to be.
- Population: 54 runs over 6 sizes (rule 2: only the runs that train the language count; a run scored on a language it never saw sits at chance and would drag the share down). The six runs of a size answer the same items, so their verdicts are correlated, not six independent trials.
- What the gate is not: chance is uniform guessing (1/4), so a run that always picks the majority gold label is not caught (on `include_v2_og_hungarian_hungary` that scores the majority label's share, above 0.25); and the gate is not a filter on DA — it blanks a cell by the gate alone, whatever the DA there (DA-size per proxy: 90M 0.50 gated; 175M 0.67 gated; 350M 0.39 gated; 600M 0.58 gated; 1B 0.53).

Follow-ups:

- `--task hellaswag_ta`: the clean step (0/6 at 175M–600M, 5/6 at 1B, 6/6 at 1.7B), with no run near the limit.
- `--task hellaswag_eu`: a task whose verdict is not monotone in size (mask 0/0/1/0/0/1), to show the share moving with the runs.
- The same figure on a 35-item task (`cultural_bench_easy_argentina`), where the needed margin is +0.12: the item count, not the model, decides.
- A `--pool predictivity_seeds` version with the replicate seeds, to see how much the per-run verdict moves with the seed.

Files: [`above_random_example.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_example.png), [`above_random_example.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_example.csv).
<!-- END auto:above-random-example -->

## Figures, in storyline order

### 1. Which cells carry any information: the gate per family and language

*Pool `predictivity` (161 runs, seed 1904, every ladder and data build),
sizes 90M–1.7B, one-sided 95 % Wilson gate (`MIN_SHARE` = half of the size's
runs that train the language); no task filter, and BPB and the loss have no
chance level.* The gate every later figure reads (on its 841 trained tasks),
and the first result of the study.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The gate and the curves without the aggregation (`predictivity` pool). Regenerate with `python analysis/rq00_gate_and_curves/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq00 in one figure](pretraining/predictivity/highlights.png)

![Smallest size above chance](pretraining/predictivity/first_size_above_random.png)

![Margin above chance per benchmark](pretraining/predictivity/gate_margin_by_benchmark.png)

![Margin above chance per language](pretraining/predictivity/gate_margin_by_language.png)

![What the gate asks, and what each rule keeps](pretraining/predictivity/above_random_thresholds.png)

Score along the run, one figure per language (50 languages, one subplot per benchmark, one line per size): `pretraining/predictivity/score_curves/<language>.png`, e.g.

![Score curves, German](pretraining/predictivity/score_curves/de.png)
<!-- END auto:panels -->

**Key findings**

- Most of the evaluation is at chance somewhere on the ladder: of the 1,373
  benchmark-language tasks with a chance level (`above_random_mask.csv`),
  552 / 589 / 629 / 676 / 731 / 778 are above chance at 90M / 175M / 350M /
  600M / 1B / 1.7B (1,372 tasks are scored below 600M). 851 (62 %) fail at
  one size or more and only 521 pass at all six.
- 821 clear the gate at ≥ 1 size and 552 nowhere; 43 are above chance at a
  smaller size but at chance at 1.7B (coded "never" in
  `first_size_above_random`, a different fact).
- On the 841 trained tasks alone the counts are 312 / 336 / 373 / 409 / 450 /
  489 at 90M … 1.7B; 515 pass at ≥ 1 size, 291 at all six. The other 532 are
  read on every run because no cell trains their language, and never reach a
  later table.
- Of the 862 (family, language) cells of `first_size_above_random.csv` (57
  families, the probe families and the `rfgm_` twins among them), 305 hold
  from 90M on, 33 from 175M, 30 from 350M, 36 from 600M, 42 from 1B, 50 only
  at 1.7B and 366 never.
- At 1.7B the letter-answer four-option families are nearly absent over all
  their languages: `global_mmlu_full` 1 / 37, `include_base_44` 3 / 43,
  `belebele` 5 / 105. Against them `multiblimp` passes 57 / 57, `xwinograd`
  6 / 6, `hellaswag` 26 / 31 and `xnli` 15 / 18.
- One cloze four-option family also stays at chance: `global_piqa_parallel_cloze`
  (103 items per task) passes 1 / 91 at 1.7B, 1 / 63 on trained languages.
- The reformulated twins pass where the originals fail: `rf_global_mmlu_full`
  35 / 37, `rf_belebele` 86 / 105, `rfgm_belebele` 59 / 59,
  `rf_include_base_44` 31 / 43, `rfgm_include_base_44` 32 / 43 (figure 3).
- The 80 % / 49 % two- / four-option split of figure 2's table is not an
  option-count effect: on the trained tasks at 1.7B the two kinds pass alike
  (64 / 103 against 402 / 691). The gap comes from 255 untrained two-option
  tasks, mostly BLiMP-style minimal pairs (`zhoblimp` 118, `blimp_nl` 84).
- What the Wilson bound asks (`above_random_thresholds.csv`, median task per
  family): +0.006 on the 14,042-item MMLU families, +0.007 on `hellaswag`
  (9,275), +0.043 on `xwinograd` (409), +0.046–0.077 on the 130–250-item
  `bbh` / `acp_bench` probes, +0.09 on the 100-item
  `global_piqa_nonparallel_cloze` and `cultural_bench_hard`, +0.157 on
  `cultural_bench_easy` (25 items).
- A model below 1B rarely delivers the last on a four-option task in a
  non-English language, so the gate measures the format's floor more than the
  languages' difficulty.
- Statistically above chance is not usefully above chance: the thinnest
  passing HellaSwag cell is `hellaswag_mr` at 90M, mean 0.257 against 0.25
  over 9,279 items (+0.007). `above_random_thresholds.png` shows what each
  rule keeps; an effect floor on top of the test is the paper's call.
- The gate does not catch a letter preference: `cultural_bench_easy` passes
  at 90M in 14 of its 19 languages and in none from 175M on. The 26 runs at
  90M that train its languages have median scores (over the languages) from
  0.07 to 0.65 against a 0.25 chance, the
  signature of an answer-letter bias meeting an unbalanced gold key rather
  than knowledge (not checked item by item).

**Follow-ups**

- Draw the run count per cell on the gate figures: a trained cell rests on 4
  to 26 runs, and no figure shows it. The 16 languages only the L50 cells
  train (`az`, `bs`, `ca`, `et`, `hr`, `kk`, `lt`, `lv`, `mr`, `ms`, `ne`,
  `sk`, `sl`, `sq`, `sr`, `ur`) have four runs per size, so two runs decide.
- The runs of a size share the test set, so "at least half of the runs" is a
  majority rule, not a pooled test. The six TruthfulQA tasks have a variable
  option count per item, so their chance is an approximation (mc1 0.2253, mc2
  0.449); a pooled test per cell is the rigorous variant.
- A letter-bias control: score each task's pick-one-letter rate per run and
  gate against it, which would test the `cultural_bench_easy` 90M passes above.
- The population can change along the size axis (a size with no cell that
  trained the language falls back to every cell; `above_random_share.csv`
  carries a `population` column, 5 046 cells `trained`, 3 189 `all`); a figure
  variant that greys the fallback cells would make the switch visible.
- Bug #16 (`../CLAUDE.md`): these panels are drawn from the mask at the time
  the driver reaches them; on 2026-09-23 the mask was rewritten after the
  panels and they had to be regenerated by hand, so the driver now draws them
  right after the gate.

GitHub: [first_size_above_random.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_above_random.png) · [first_size_above_random.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_above_random.csv) ·
GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/highlights.csv) ·
[gate_margin_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_margin_by_benchmark.png) ·
[gate_margin_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_margin_by_language.png) ·
[gate_margin.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_margin.csv) ·
[above_random_thresholds.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_thresholds.png) ·
[above_random_thresholds.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_thresholds.csv) ·
[above_random_mask.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_mask.csv) ·
[score_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/score_curves.csv)

What the gate leaves open — whether the at-chance families are unreadable
for any model of these sizes or only for this ladder — is figure 4; whether
the format is the cause is figure 3.

### 2. The gate by answer count, and the Signal ranking

*Pool `predictivity`; the Signal of a task is (max − min) / mean of the
per-setting final scores at 1.7B over the six L1 … L50 deep scheme-A (T 1)
seed-1904 cells, computed on every task **without the gate**. 1,279 of the
1,714 parent tasks have a value, the `bbpb_` twins (final checkpoint) among
them.*

<!-- BEGIN auto:results (run_apertus.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate: `python analysis/rq00_gate_and_curves/above_random.py --only predictivity` and `python analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity`.

**Top benchmarks by Signal across language settings** (full ranking in `pretraining/predictivity/acc_vs_flops_signal.csv`):

| task | family | lang | Signal |
|---|---|---|---|
| `rf_bbh_mcq_temporal_sequences` | rf_bbh_mcq | en | 3.391 |
| `cultural_bench_easy_australia` | cultural_bench_easy | en | 2.526 |
| `cultural_bench_easy_canada` | cultural_bench_easy | en | 2.308 |
| `cultural_bench_easy_united_states` | cultural_bench_easy | en | 2.143 |
| `cultural_bench_easy_south_africa` | cultural_bench_easy | en | 2.113 |

![top-Signal family accuracy vs FLOPs](pretraining/predictivity/per_benchmark/cultural_bench_easy.png)

**Above-random gate** — a (benchmark, size) cell is kept when at least 50% of the size's runs clear chance (`1/n_options`) with the Wilson 90% lower bound of their accuracy over the task's items; `run_apertus_snr_variants.py` NaN-s every at-chance `(benchmark, size)` SNR cell, so the gate propagates to all RQs:

| options | chance | above ≥1 size | above @1.7B |
|---|---|---|---|
| 2 | 0.50 | 289 / 358 | 285 / 358 |
| 3 | 0.33 | 19 / 24 | 17 / 24 |
| 4 | 0.25 | 494 / 932 | 459 / 932 |
| 5 | 0.20 | 14 / 43 | 13 / 43 |
| 6 | 0.17 | 1 / 4 | 1 / 4 |
| 7 | 0.14 | 1 / 7 | 1 / 7 |
| 8 | 0.12 | 2 / 2 | 1 / 2 |
| 10 | 0.10 | 0 / 2 | 0 / 2 |
| 12 | 0.08 | 1 / 1 | 1 / 1 |
<!-- END auto:results -->

**Key findings**

- Over all 1,373 tasks the table splits two- from four-option tasks: at 1.7B
  285 / 358 (80 %) against 459 / 932 (49 %), at 90M 241 / 358 against
  289 / 932. The split is the population, not the option count: on the trained
  tasks the two pass alike (figure 1).
- The Signal ranking is a selection on noise, not a reliability ranking. Its
  top families by mean Signal are `cultural_bench_easy` (1.35, 19 tasks of
  7–59 items), `bbpb_bbh_cloze` (0.78, 6) and `bbpb_bbh_mcq` (0.60, 17).
- The first is at chance at 1.7B on all 19 of its tasks; the other two are
  bits-per-byte twins of short BBH answers. The top task,
  `rf_bbh_mcq_temporal_sequences` (3.39), scores at most 0.052 against a 0.25
  chance.
- Read on the tasks above chance at 1.7B, the ranking tops out at 0.42
  (`include_v2_og_spanish_rep_blica_dominicana`), then 0.38 and 0.35 for two
  more `include_v2_og` regional tasks and ≤ 0.29 for the `rf_` cultural, BBH
  and ACP twins. The gated readings are
  [decision accuracy](../rq02_decision_accuracy/README.md) and
  [noise and SNR](../rq03_noise_and_snr/README.md).

**Follow-ups**

- Apply the gate to the Signal ranking (mask the at-chance cells before
  ranking), and rank on the gated 1.7B cells only.
- Rank the accuracy tasks and the `bbpb_` twins separately: a bits-per-byte
  spread and an accuracy spread are not on one scale.
- `per_benchmark/<family>.png` curves are redrawn only with `CURVES=1`
  (`refresh_analysis.sh --curves`). The top-3 grids on disk
  (`cultural_bench_easy.png`, `bbpb_bbh_cloze.png`, `bbpb_bbh_mcq.png`) date
  from 2026-10-06, before this refresh, so they need that redraw.

[acc_vs_flops_signal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/acc_vs_flops_signal.csv) ·
[above_random_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_scores.csv) ·
[above_random_share.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_share.csv) ·
[above_random_runs.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_runs.csv)

### 3. The reformulated twins move whole families across the gate

*Pool `predictivity` (the gate mask over its 161 runs, seed 1904, every
ladder and data build), the `rf_` (letters → answer strings) and `rfgm_`
(Gemini-rewritten items) twins against their originals, paired on the
language over every language of the family; McNemar's exact test on the
discordant languages. The figure and its auto block live in
[`../rq00_task_reformulation/README.md`](../rq00_task_reformulation/README.md#the-twins-and-the-gate-with-significance)
(`reformulations_gate.py`); the pipeline that produced the twins is that
folder's README.*

![The reformulations and the gate](../rq00_task_reformulation/reformulations_gate.png)

*(a) Per family with a twin, the share of its languages above the gate for
the original letter format and the `rf_` cloze twin at each size, paired on
the language; diamonds the `rfgm_` rewritten twin; a star where McNemar's
exact test on the discordant languages gives p < 0.05. (b)–(d) the gate's
pass share, the mean DA-size on the tasks above chance at the proxy and at
the reference (rule 1; the red line is the tie null, what a proxy with no
signal reads because a pair tied on one side counts as a miss) and the share
of reliable tasks, read on every task, on the originals alone and on the
twins alone.*

Every headline reading of the gate, scaling predictability and decision
accuracy can be read with and without the twins from
`reformulations_gate.csv` (DA-size on the multi-axis pairs, the `bbpb_` twins
excluded; the R² column is the scaling-predictability fit on
`predictivity_seeds`):

| population | gate at 90M → 1.7B | mean DA-size 90M → 1B (gated at both sizes) | tasks with median DA-size ≥ 0.66 | scaling-fit median R² (tasks) |
|---|---|---|---|---|
| every task | 0.40 → 0.57 | 0.52 → 0.57 | 18 % of 520 | 0.887 (342) |
| originals only | 0.37 → 0.49 | 0.55 → 0.59 | 23 % of 292 | 0.911 (199) |
| twins only | 0.52 → 0.80 | 0.49 → 0.55 | 11 % of 228 | 0.854 (143) |

**Key findings**

- The twins' effect on the gate is not chance. At 1.7B the cloze twin clears
  the gate in 82 Belebele languages where the original does not against 1 the
  other way (the rewritten `rfgm_` twin 56 against 0 over its 59 languages),
  34 against 0 for Global-MMLU and 29 against 1 for INCLUDE (30 against 1 for
  `rfgm_`), McNemar p < 10⁻⁷ each.
- The smaller families point the same way at 1.7B: BBH 10 against 0
  (p = 0.002), `cultural_bench_easy` 7 against 0 (p = 0.016), `acp_bench_mcq`
  5 against 0 (p = 0.06) (`reformulations_gate_mcnemar.csv`). At 90M
  `cultural_bench_easy` runs the other way (0 against 13), the letter-bias
  passes of figure 1.
- The twins pass the gate far more often and are read from 90M, but they do
  not rank the design variants better: gated at both sizes they read mean
  DA-size 0.49–0.55 across the proxies against 0.55–0.60 for the originals.
  Half as many clear the 0.66 cut (11 % of 228 against 23 % of 292).
- Their spread across variants is larger: median relative signal
  (`signal_rel_std`) 0.026 on 214 twins against 0.022 on 236 originals at 1B,
  on the tasks above chance at 1B (noise-and-SNR `snr_variants_per_task.csv`).
- The family mix moves the twins' mean: at 1B `rf_belebele` reads DA-size
  0.54 on 57 tasks and `rf_global_mmlu_full` 0.69 on 28
  (`agreement_da_size_per_cell_multi_axes.csv`). The tie null is ≈ 0.49 today
  (2.9 % one-sided ties per multi-axis cell), above the 0.47 the figure's red
  line still hard-codes.
- The reformulation decides whether a family exists for the rest of the
  analysis: on the trained languages at 1.7B `belebele` passes 3 / 59 tasks
  and `global_mmlu_full` 1 / 29, against 57 / 59 and 29 / 29 for their `rf_`
  twins ([scaling predictability, figure 2](../rq01_scaling_predictability/README.md#2-survivorship-what-the-gate-and-the-fit-minimum-removed)).
  In all three rows of the table the gate share and DA-size rise from 90M to
  1B, so no headline reading flips with or without the twins.

**Follow-ups**

- Formulation as a variable: per family with a twin, the first size above
  chance, DA-size and SNR per formulation (letters vs cloze vs rewritten) in
  one figure; the `rfgm_` evals now cover INCLUDE (43 languages) and Belebele
  (59) at every size.
- Redraw the red line of panel (c) at today's tie null (≈ 0.49 from the
  2.9 % one-sided tie share) instead of the hard-coded 0.47.
- The twins are the same items in another formulation, so a language can
  clear rule 8's three-task floor with the twins of one benchmark; a
  per-language panel that rests on twins alone should say so (RULES.md).
- The length tell (`rf_acp_bench_mcq`: the gold answer is the strictly
  shortest option in 35 % of items, up to 50 % in three subtasks) means those
  tasks' gate should be read against the pick-shortest rate, not 1/4 —
  [`../rq00_task_reformulation/README.md`](../rq00_task_reformulation/README.md#the-length-tell-and-where-it-actually-is).

GitHub: [reformulations_gate.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.png) · [reformulations_gate.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.csv) ·
[reformulations_gate_mcnemar.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate_mcnemar.csv)

### 4. Benchmark floors: is a gated task a size problem or a benchmark problem?

*Pool `predictivity` (161 runs, seed 1904, final checkpoint, trained
languages) against the external tier's mask (`all/external/`, public base and
post-trained models 270M–70B); the overlap is the 36-sweep's 86-task list (no
twins, no probe families). The external floor is a model floor, not a
parameters-at-5C floor.*

*Provisional: the external mask on disk dates from 2026-09-21 and still
applies the retired `score > 1/n_options + 0.05` rule (it agrees with that
rule on 1,169 of its 1,171 cells), not the Wilson gate the auto block names.*

<!-- BEGIN auto:above-random-external (above_random_external.py --pool predictivity) -->
## Benchmark floors: the ladder's gate against the public models'

Of the 86 tasks both tiers score, 38 are at chance at every ladder size. Where the external models (270M–70B, every release the external tier holds, base and post-trained, same gate) first read them: ≤ 600M 8, 1B–1.7B 19, 3B–4B 1, 7B–14B 5, ≥ 27B 2, never 3. A task readable at ≤ 1.7B by a public model is a size/recipe floor (the 5×-Chinchilla ladder does not reach it; public models of that size train on 10–36 T tokens); a task that needs ≥ 3B or is never read is a benchmark or language-resource floor. "Never reads" is the gate's verdict: at every ladder size fewer than 50% of the size's runs that train the task's language clear the one-sided 95 % Wilson bound over chance; single runs may clear it. Overlap is the 36-sweep's 86-task list only. Regenerate with `python analysis/rq00_gate_and_curves/above_random_external.py --pool predictivity`.

| family | ≤ 600M | 1B–1.7B | 3B–4B | 7B–14B | ≥ 27B | never |
|---|---|---|---|---|---|---|
| belebele | 3 | 9 | 0 | 0 | 0 | 0 |
| global_mmlu_full | 1 | 9 | 0 | 0 | 0 | 0 |
| arc | 0 | 0 | 0 | 2 | 0 | 1 |
| xnli | 0 | 0 | 0 | 2 | 0 | 1 |
| paws | 0 | 1 | 0 | 0 | 0 | 1 |
| truthfulqa_mc2 | 2 | 0 | 0 | 0 | 0 | 0 |
| xstorycloze | 0 | 0 | 0 | 1 | 1 | 0 |
| commonsense_qa | 1 | 0 | 0 | 0 | 0 | 0 |
| mmlu | 1 | 0 | 0 | 0 | 0 | 0 |
| openbookqa | 0 | 0 | 1 | 0 | 0 | 0 |
| xcopa | 0 | 0 | 0 | 0 | 1 | 0 |

![The gate on the public models](pretraining/predictivity/above_random_external.png)

The paper version, `above_random_external_paper.png` (alternatives `_paper_b`, one cell per task, and `_paper_c`, a family by bucket grid), recomputes the floor on the public base releases alone (no post-trained release, none of apertus3-a06, ap-from8b-TOP256): ≤ 600M 19, 1B–1.7B 9, 3B–4B 4, 7B–14B 0, ≥ 27B 4, never 2. `above_random_external_models.tex` lists those models per line and size bucket.

Population: the `predictivity` pool (seed 1904, 175M–1.7B, final checkpoint, trained languages) against every external release (`all/external`, base and post-trained, same gate; the paper version uses the public base releases only; panels (c) and (d) use the six public lines gemma-3, Qwen3, OLMo-2, Olmo-3, Apertus, apertus3-a06); no task filter beyond the 84-task overlap.

Key findings:
- 38 of the 86 shared tasks are gated at every ladder size; 27 of them are read by an external model ≤ 1.7B, 3 by none up to 70B.
- (c) the ladder reads 37% of the shared tasks at 90M and 56% at 1.7B; the public base models ≤ 1.7B read Qwen3-1.7B-Base 87%, Qwen3-0.6B-Base 81%, OLMo-2-0425-1B 64%, gemma-3-1b-pt 64%, apertus3-1b-21-nodes 62%, gemma-3-270m 58% (a model's own run clears the bound; the ladder's share is its mask, half of the runs).
- (d) 29 of the 38 ladder-gated tasks have a trained run at 1.7B (no 1.7B cell trains eu, sw, rule 2); 23 of them have a single run whose lower bound clears chance (near misses: the gate wants half of the runs), and the best public base model ≤ 1.7B beats the ladder's best run on 29.

Follow-ups:
- (c) with the post-trained releases as open markers: whether instruction tuning moves a task over the gate at the same size.
- (d) at every ladder size, not only the reference: the size at which each near miss first has a passing run.
- (c)/(d) on the `rf_` cloze twins once the external tier is evaluated on them: the twins pass the gate on the ladder, so the overlap would stop being the old 86-task list.

Figure: https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.png · tables: https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.csv, https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external_lines.csv
<!-- END auto:above-random-external -->

**Key findings**

- Most of what the ladder cannot read is a size or recipe floor, not a
  benchmark floor: of the 38 shared tasks at chance at every ladder size,
  public models read 8 at ≤ 600M and 19 at 1B–1.7B, 1 at 3B–4B
  (`openbookqa`), 5 at 7–14B, 2 at ≥ 27B and 3 never (`arc_eu`, `paws_ja`,
  `xnli_ar`).
- The ≤ 1.7B group is every Belebele and Global-MMLU task of the list, `mmlu`,
  `commonsense_qa`, `truthfulqa_{vi,zh}_mc2` and `paws_zh`. The 7–14B group is
  `arc_ar`, `arc_hi`, `xnli_eu`, `xnli_sw`, `xstorycloze_sw`; the ≥ 27B one
  `xstorycloze_eu`, `xcopa_eu`.
- Public models of the ladder's sizes read far more of the shared tasks:
  Qwen3-0.6B-Base 81 % and Qwen3-1.7B-Base 87 % (of the 84 they are scored
  on), against 37 % of 86 for the ladder's mask at 90M and 56 % at 1.7B
  (`above_random_external_lines.csv`).
- Basque and Swahili, which no 1.7B ladder cell trains, account for 6 of the
  10 tasks that need ≥ 7B or are never read: they are the language-resource
  floors.
- Near misses: 29 of the 38 tasks have a trained run at 1.7B, and on 23 of
  them at least one run's Wilson bound clears chance. The ladder's verdict
  there is the half-of-the-runs rule as much as a floor.

**Follow-ups**

- Regenerate the external mask with the Wilson gate (`above_random.py --only
  external`), so both tiers use one rule; the floors above are provisional
  until then.
- Extend the external gate to the `auto` list (the reference models' twin and
  probe evals), so the floors cover the twins and probe families, the first
  item of `plan/next_analyses.md` §2.
- A token-axis floor from the six public step series (the token count at
  which a fixed-size model leaves chance per task), which turns "how many
  parameters" into "how much data".
- A floor-conditioned reading of decision accuracy: DA against the proxy's distance
  above the task's floor in doublings; if the lines coincide, the external
  floors predict how far above its floor a proxy must sit for a task the
  ladder never reads (`plan/next_analyses.md` §7c).

GitHub: [above_random_external.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.png) · [above_random_external.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.csv)

### 5. The ladder's curves

*Pool `predictivity_seeds`, every trained cell, the shared checkpoint grid:
181 cells (seed 1904 161; seeds 64 and 313 three each at 175M and at 600M,
six per seed, 28 and 1797 four each at 1B; none at 1.7B). By data scheme A at
T 1 95, A at T 3 24, B 50, C 12; deep 116, shallow 60, swiglu 5.*

Descriptive: the counterpart of the progress report's figures on the
analysis' own cells, so a reader can check any assumption of the analyses
above against the raw runs.

<!-- BEGIN auto:curves (curves.py --pool predictivity_seeds) -->
## Curves on the analysis' cells

Every cell the `predictivity_seeds` pool holds (all seeds and data builds), after the loader has dropped diverged and unfinished runs and restricted checkpoints to the shared grid: the detailed counterpart of the progress report's figures. Loss per L, the whole run and its last 10 % (capped at 3.5 nats, where ladder and data build separate); benchmark accuracy as the mean over the tasks in the languages the cell trains on, one line per cell, chance from the option count, along the run in Chinchilla multiples; a family's cloze twin sits next to it, titled `(rf)`. Regenerate with `python analysis/rq00_gate_and_curves/curves.py --pool predictivity_seeds`.

![Loss curves](pretraining/predictivity_seeds/loss_curves.png)

![Benchmark curves](pretraining/predictivity_seeds/benchmark_curves.png)

The paper version, `benchmark_curves_paper.png` (`--paper`, redrawn from `benchmark_curves.csv`), drops the header for a legend of the line encoding; its size twin, `benchmark_size_curves_paper.png`, draws each design's final accuracy against non-embedding parameters, colour = L (the scaling-predictability appendix's size figure).
<!-- END auto:curves -->

*The benchmark panel draws 176 of the 181 cells: the five swiglu cells have
loss curves only.*

**Key findings**

- Letter-format knowledge MCQA rises well off chance in one cell only, and
  only in the last fifth of the run (between 4C and 5C): on `mmlu` the English-only
  `lm-1.7B-L1-deep-seed1904` goes 0.274 → 0.316 → 0.337 at 4C / 4.5C / 5C
  (chance 0.25). On `global_mmlu_full_en` it goes 0.270 → 0.303 → 0.314, on
  `commonsense_qa` 0.202 → 0.256 → 0.257 (chance 0.20).
- It is the only one of 175 cells to end above 0.28 on `mmlu`. Its shallow
  twin ends at 0.277, the English-only scheme-B (DCLMP) and scheme-C (FWEB)
  cells at 1.7B at 0.239 and 0.247, while the cloze `rf_mmlu` of the same
  cell moves smoothly (0.367 → 0.374 → 0.379).
- Two families move away from chance with training, downwards: TruthfulQA
  mc2 falls from 0.420 at 0.5C to 0.386 at 5C (mean over 175 cells; chance
  0.449) and ToxiGen from 0.523 to 0.449 (chance 0.5), in 87 % and 83 % of
  the cells.
- Their "never above chance" verdict is therefore a trend below chance, not
  noise around it.

**Follow-ups**

- Label the x axis of the loss figure in Chinchilla multiples (1C–5C) as the
  benchmark figure does (rule 3).
- Mark the last 20 % of the run on the benchmark panels, where the `mmlu`
  jump happens. It is one run of one corpus and no 1.7B cell has a seed
  replicate, so whether it is the corpus or the draw is open.

[loss_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/loss_curves.png) ·
[loss_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/loss_curves.csv) ·
[benchmark_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_curves.png) ·
[benchmark_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_curves.csv)

## Extensions from other sweeps

Everything below comes from the 36-model sweep (2026-04…06: 4 sizes × 3 data
mixtures × 3 seeds, `apertus-*`, 12 languages, pools `custom`, `seeds_*`,
`custom_swissai_hf`, `external`; sizes 175M–1B, reference 1B) or from the
public models of the external tier. Its harness, task set, reference size and
gate rule (a fixed +0.05 margin) differ from the ladder's, so its numbers are
replications of the findings above, never rows of the same table
(`../CLAUDE.md`, "Two sweeps").

## Custom vs. external: the at-chance problem is a capability artifact

*36-sweep numbers (pools `custom` and `external`, 2026-06, custom sizes
175M–1B); the ladder's counterpart is figure 4
above (`above_random_external`).*

This is the foundational result the rest of the paper rests on. The
above-random gate is identical for every model set, but what it removes depends
entirely on *who is being evaluated*.

**On the custom pretrains the gate is brutally selective.** Of 118 benchmarks,
only **44 clear chance at ≥1 size and 74 are random everywhere** — the 175M–1B
models we train are simply too weak to register signal on most translated
knowledge MCQA.

The loss is concentrated in the 4-option families: only **9 / 63**
clear chance, so `belebele`, `global_mmlu_full`, `arc`, and `truthfulqa` are
NaN-ed out before any SNR is computed. The surviving pool is therefore *almost
entirely 2-option*, which is exactly what later biases the design-feature
analyses toward "fewer options ⇒ higher SNR".

**Re-running the identical gate on the external tier dissolves the penalty.** The
`external` model set (every non-custom model — `reference_hf` + a06 + distillation
+ posttraining, sizes 270M…70B; gate report in `all/external/`, curves skipped)
clears chance on **122 / 124** benchmarks, including **68 / 69** four-option
families.

A 4-option translated benchmark is not intrinsically low-signal — it
only looks that way under models too small to beat chance.

| model set | models | benchmarks | above ≥1 size | 2-opt | 3-opt | 4-opt | 5-opt |
|---|---|---|---|---|---|---|---|
| `custom` (SNR-gate domain) | 175M–1B custom pretrains | 118 | **44 (37%)** | 28 / 42 | 7 / 11 | **9 / 63** | 0 / 2 |
| `external` (`all/external`) | 270M–70B reference / a06 / distill / post | 124 | **122 (98%)** | 42 / 42 | 10 / 11 | **68 / 69** | 2 / 2 |

The same benchmark tells the story directly: on the `custom_swissai_hf`
acc-vs-FLOPs curves, Belebele sits flat at chance across the custom 175M–1B
sweep, then the overlaid external final-checkpoint markers climb steeply toward
0.8+ out to 70B.

![Belebele: custom at chance, externals climb to 0.8+](pretraining/custom_swissai_hf/per_benchmark/belebele.png)

**Implication for the paper.** The custom-pool gate measures *our models' scale*,
not benchmark reliability; the external tier is the control that separates the
two.

This single distinction drives the external-tier findings downstream: option
count stops predicting SNR, and 4-option HellaSwag becomes the
*highest*-SNR family once it clears the gate.

[belebele.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/per_benchmark/belebele.png)

## Methodology — scaling beyond 1B (36-sweep pools)

The `custom_swissai_hf` pool extends the compute axis past the custom 1B ceiling
by folding in the apertus3 a06, distilled, and reference-HF trajectories. Three
mechanisms make that work:

- **Bucketed size axis.** The four custom sizes (175M / 350M / 600M / 1B) stay
  singletons; nearby external/large sizes pool into buckets so each holds ≥2
  models. A bucket with ≥2 models contributes a computable cross-model signal and
  decision accuracy; a singleton bucket (e.g. an isolated 70B) still plots its
  final-checkpoint marker on the curves but is dropped from the noise/DA pool.
- **Relative-fraction ckpt-DA.** External trajectories never hit the custom
  absolute megatron iters, so checkpoint-DA selects each model's early checkpoint
  at a *fraction* of its own max step rather than a fixed iter — letting the a06,
  distillation, SmolLM3, Olmo-3 and Apertus-8B series participate and extending
  ckpt-DA into the large buckets. (The computation lives in
  [compute_da.py](../rq02_decision_accuracy/compute_da.py); the gate and curves
  here just consume its trajectories.)
- **Cross-bucket scaling-DA is family-coverage-limited.** Scaling-DA auto-detects
  bucket pairs where ≥2 model *families* span both sizes (via within-family
  ladders such as gemma-3 / OLMo). Above 1B this is sparse — few families ship
  multiple sizes — so the few detectable pairs saturate at DA 1.0 on a handful of
  shared tasks: directionally useful, not yet statistically strong. acc-vs-FLOPs
  curves and the signal pool carry the >1B scaling story for now.

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **The benchmarks that separate data mixtures most: `agieval_sat`, `belebele`, `arabic_leaderboard_alghafa_mcq_exams_test`** — top-3 families by mixture-Signal ((max−min)/mean of per-mix final scores) at 1B.
- **Mixture-Signal ≠ reliability.** These top-Signal families are exactly the ones the above-random gate **removes** — they sit at chance, so they never enter the SNR analysis. Of **118 benchmarks, 44 clear chance at ≥1 size** (74 are random everywhere) — almost entirely an answer-count effect.

### Results

Headline numbers from the `custom_swissai_hf` pool (Signal) and the `custom` above-random report. Regenerate: `python analysis/rq00_gate_and_curves/run_apertus.py --pool custom_swissai_hf` and `python analysis/rq00_gate_and_curves/above_random.py`.

**Top benchmarks by mixture-Signal** (full ranking in `pretraining/custom_swissai_hf/acc_vs_flops_signal.csv`):

| task | family | lang | Signal |
|---|---|---|---|
| `agieval_sat_en` | agieval_sat | en | 0.268 |
| `belebele_hin_Deva` | belebele | hi | 0.245 |
| `belebele_zho_Hans` | belebele | zh | 0.236 |
| `global_piqa_completions_arb_arab` | global_piqa_completions | ar | 0.229 |
| `belebele_eng_Latn` | belebele | en | 0.222 |

![top-Signal family accuracy vs FLOPs](pretraining/custom_swissai_hf/per_benchmark/agieval_sat.png)

**Above-random gate** — a benchmark must beat chance (`1/n_options`) by +0.05; `run_apertus_snr_variants.py` NaN-s every random `(benchmark, size)` SNR cell, so the gate propagates to all RQs. Almost entirely an answer-count effect:

| options | chance | above ≥1 size | above @1B |
|---|---|---|---|
| 2 | 0.50 | 28 / 42 | 28 / 42 |
| 3 | 0.33 | 7 / 11 | 7 / 11 |
| 4 | 0.25 | 9 / 63 | 7 / 63 |
| 5 | 0.20 | 0 / 2 | 0 / 2 |

[agieval_sat.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/per_benchmark/agieval_sat.png) ·
[acc_vs_flops_signal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/acc_vs_flops_signal.csv)

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task Signal (full
  ranking; all parent tasks, ungated).
- `pretraining/predictivity/above_random_{scores,share,mask,runs,thresholds}.csv`
  — the ladder's gate (`above_random.py`): mean score, share of runs above
  chance, the 1/0/blank mask every later analysis reads, the per-run Wilson bounds, and
  the margin each rule implies. `first_size_above_random.csv`,
  `gate_margin.csv`, `highlights.csv`, `score_curves.csv`,
  `above_random_thresholds.csv` sit next to the panels of the same name.
- `pretraining/predictivity/above_random_external.{png,csv}` — the ladder's
  floors against the public models' (`above_random_external.py`; the former
  `benchmark_floor.*` name is retired).
- `pretraining/seeds_28_1797_1904/`, `pretraining/custom_swissai_hf/`,
  `all/external/` — the 36-sweep's and the public models'
  `above_random_scores.csv` / `above_random_mask.csv` (pure-custom,
  all-models, and non-custom reports, pool-named); `all/external/` is also
  the mask figure 4 and decision accuracy's public-ladder extension read.
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid (36-sweep pools); `…/per_language/<lang>.png`
  — per language, subplots = top-3 families. Redrawn with `CURVES=1` only.
- `pretraining/predictivity_seeds/loss_curves.png`, `benchmark_curves.png` — the
  ladder's curves on the analysis' cells (`curves.py`).
- `pretraining/predictivity/above_random_example.{png,csv}` — the gate on
  one task, an educational reading of the rule (`above_random_example.py
  --task <task>`; its block sits under the setup above).
- Scripts: `above_random.py` (gate), `run_apertus.py` (accuracy vs FLOPs,
  Signal), `curves.py`, `panels.py`, `above_random_external.py`,
  `above_random_example.py`.
