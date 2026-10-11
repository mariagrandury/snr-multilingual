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
- **Above-random gate.** Of **1696 benchmarks, 980 clear chance at ≥1 size** and 941 at 1.7B (716 are random everywhere). The at-chance cells are removed before any SNR is computed; the breakdown by answer count below shows how much of the gate is an option-count effect.
<!-- END auto:highlight -->

## Setup

**Setup.**
- **Measured.** We measure whether a benchmark task is above chance at a model size.
- **Models.** We read the final checkpoints of the seed-1904 runs of every language setting, ladder and data build, 90M–1.7B.
- **Tasks.** The tasks are every benchmark task with a chance level, in the languages the sweep trains. Per-language BPB and the bBPB twins have no chance level and are never gated.
- **Rule.** A run clears chance on a task when the one-sided 95 % Wilson lower bound of its accuracy over the task's items is above the chance level. A (task, size) cell is above chance when at least half of the runs of that size that train the task's language clear it.
- **Beyond the analysed set.** We also apply the same rule to the benchmarks we screened, which were evaluated only on the final checkpoints of the larger models (figure: app_rq00_chance_considered, table: app_rq00_pass_counts_considered). On the tasks that clear chance at very small token counts of their language, we check which option the runs pick (table: app_rq00_early_passes).

- **Snapshot.** Every hand-written number is from the ladder-report snapshot
  2026-10-10 04:31 and the outputs of the 2026-10-10 refresh (detrended
  checkpoint noise), unless a section names another date.
- **Gate pool.** `predictivity`: seed 1904, every cell of every ladder and
  data build, 192 runs with benchmark scores at 90M–1.7B (31–33 per size,
  and by ladder deep 96, shallow 60, swiglu 19, muon 17). By data scheme: A at
  T 1 108 runs, A at T 3 24, B 48, C 12.
- **Curves pool.** `predictivity_seeds` (every seed) for the loss and
  benchmark curves of figure 5, sizes 90M–1.7B throughout (rule 10).
- **The gate covers every task the pool evaluates.** That includes the 763
  chance-level tasks that no cell trains (734 of them scored on every run, 29
  never scored). Rule 2 drops them downstream, so the trained population is
  933 of the 1,696 tasks with a chance level. It includes the benchmarks whose
  ladder evals are still pending (rf_m_mmlu, rf_mmmlu, toksuite), which
  section 1a counts as screened.
- **bBPB twins.** The per-item store holds the `bbpb_` twins at all 12
  checkpoints of 173 of the 192 seed-1904 ladder cells (not the 17 muon cells
  and two swiglu cells) and of the 20 replicate-seed cells. Here they enter
  only the Signal ranking of figure 2, which reads the final scores of every task.

## Key findings (ladder report 2026-10-10 04:31)

![The gate in one figure](pretraining/predictivity/highlights.png)

*Pool `predictivity` (192 runs, seed 1904, 90M–1.7B). One-sided 95 % Wilson
gate. A (task, size) cell passes when at least half of the size's runs that
train the language clear chance (1 to 32 runs per cell, 4 to 32 outside
rf_m_mmlu, rf_mmmlu and multiblimp-extra at 90M–350M).*

- **The gate removes most of the evaluation somewhere on the ladder.** On
  the 933 trained tasks, 377 (40 %) are above chance at 90M and 599 (64 %) at
  1.7B. At one size or more 577 (62 %) are at chance, and 313 (34 %) are at
  chance at every size.
- **The picture is the same over all 1,696 tasks with a chance level.** Of the
  1,511 scored at 90M, 619 (41 %) pass, and 946 of 1,667 (57 %) pass at 1.7B.
  710 (42 %) never pass, 29 of them because they are never scored.
- **The answer format, more than the option count, decides the gate.** On the
  trained languages at 1.7B the letter-answer families nearly vanish while
  their cloze twins pass: `belebele` 3 / 59 tasks against 57 / 59,
  `global_mmlu_full` 1 / 29 against 29 / 29, `include_base_44` 3 / 36 against
  29 / 36.
- **The format effect is not chance.** Over every language of each family,
  McNemar's discordant counts are 82 : 1, 34 : 0 and 29 : 1 (p < 10⁻⁷ each).
  Two- and four-option trained tasks pass at similar rates at 1.7B (83 / 122,
  68 %, against 488 / 759, 64 %).
- **The item count sets the bar.** The margin over chance that a run needs
  for its Wilson bound to clear chance runs from +0.006 (the 14,042-item MMLU
  families) to +0.157 (`cultural_bench_easy`, 25 items), median task per
  family.
- **Much of what the ladder cannot read is a size floor or an effect of the
  half-of-the-runs rule (provisional).** Of the 74 tasks shared with the
  public models, 27 are at chance at every ladder size. Public models read 24
  of them at ≤ 1.7B, and no model up to 70B reads 2 (`paws_ja`, `xnli_ar`).
  The public models' mask still uses the retired +0.05 rule (figure 4).
- **Near misses.** All 27 have a 1.7B cell that trains them, and 24 have at
  least one 1.7B run whose Wilson bound clears chance. For those, the verdict
  comes from the half-of-the-runs rule as well as from a floor.
- **Signal without the gate is not reliability.** The top task by Signal at
  1.7B, `rf_bbh_mcq_temporal_sequences` (3.39), scores 0.000–0.068 over the 29
  runs at 1.7B against a 0.25 chance. Its large relative spread sits on a
  near-zero mean.

## Experimental setup

- **Accuracy-vs-FLOPs curves** (`pretraining/<pool>/{per_benchmark,per_language}/`):
  FLOPs `6 × (N_non_emb + d·V) × D` on a log x axis, one curve per language setting
  (`plotted_mixes`: L1 … L50, deep, scheme A at T 1, seed 1904) and per size
  90M–1.7B, each rung at its own batch (rule 10). Only the top-3 families by
  Signal get curve grids. The 36-sweep pools overlay the external models up to 70B.
- **Signal.** Per parent task (subjects collapse into the parent, languages
  stay distinct), (max − min) / mean of the six `plotted_mixes` cells' final
  scores at the 1.7B reference.
- **The above-random gate** ([`above_random.py`](above_random.py)) reads only
  the raw eval scores and the option counts (`n_options` in
  `configs/tasks.json`, derived from the evaluated samples where possible). It
  reads no output of another analysis, so every later analysis depends on the
  gate and never the reverse.
- **The rule.** A run is above chance on a task when the one-sided 95 % Wilson
  lower bound of its accuracy over the task's `n_items` clears `1/n_options`.
  A `(task, size)` cell is above random when at least half of the size's runs
  that train the language clear it (`MIN_SHARE`).
- **No fixed margin.** The old `mean score > 1/n_options + 0.05` rule is gone,
  so a 2-option task with 500 items and a 4-option task with 100 items are held to
  the same standard of evidence. The one-sided 95 % bound is the lower end of a two-sided
  90 % interval, which the auto table below calls "Wilson 90 %".
- **Propagation.** `run_apertus_snr_variants.py` sets every at-chance cell to NaN,
  so the gate reaches every later analysis. Per-language BPB, the `bbpb_`
  twins and generative tasks have no chance level and are never gated.

<!-- BEGIN auto:above-random-example (above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary) -->
### The gate on one task: `include_v2_og_hungarian_hungary`

The rule above, read on one benchmark-language task (the gate is per task, never per family). (a) Every run of `predictivity` that trains the language, along training, with the chance level and the score a run needs for its Wilson lower bound to clear chance at this task's item count. One run's band between its score and its Wilson lower bound (LCB) is drawn. (b) Per size, the final scores with their bounds, the passing runs in colour, and the verdict. (c) The consequence: rq02's DA-size and DA-ckpt cells of the task, grey where the gate blanks them. The figure's header states the rule in three lines. Regenerate with `python analysis/rq00_gate_and_curves/above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary`.

![The above-random gate on include_v2_og_hungarian_hungary](pretraining/predictivity/above_random_example.png)

Key findings:

- `include_v2_og_hungarian_hungary` has 4 options (0.25 chance) and 1,386 items, so a single run needs 0.270 (+0.020 over chance) for its LCB to clear chance. At n = 1,386 the bound sits 0.019 below the score.
- Verdict per size (runs passing / runs that train the language, ≥ 50% needed): 90M 1/11 → at chance, 175M 5/11 → at chance, 350M 6/11 → above, 600M 5/11 → at chance, 1B 7/11 → above, 1.7B 9/11 → above. The task enters every RQ from **350M** on and is grey below it.
- What that means in rq02 (panel c): a DA-size cell needs the proxy and the reference above chance, so `include_v2_og_hungarian_hungary` contributes only the 350M → 1.7B, 1B → 1.7B DA-size cell(s). A DA-ckpt cell needs the run's own size above chance, so it contributes only the 350M, 1B, 1.7B run(s). 6 of 11 cells are blanked, and the task is never a decision at 90M, 175M, 600M.
- The highlighted run `lm-350M-L30-deep-seed1904` ends at 0.258 with LCB 0.239: not above chance. A score above the dotted line is not enough: the band's lower edge has to be above it.
- Population: 66 runs over 6 sizes. Only the runs that train the language count (rule 2), because a run scored on a language it never saw sits at chance and would drag the share down. The runs of a size answer the same items, so their verdicts are correlated, not independent trials.
- What the gate is not: chance is uniform guessing (1/4), so a run that always picks the majority gold label is not caught (on `include_v2_og_hungarian_hungary` that run scores the majority label's share, above 0.25). Nor is the gate a filter on DA: it blanks a cell by the gate alone, whatever the DA there (DA-size per proxy: 90M 0.56 gated, 175M 0.69 gated, 350M 0.40, 600M 0.64 gated, 1B 0.51).

Follow-ups:

- `--task arc_ca`: a clean step (0/4 runs at 90M–600M, 4/4 at 1B and 1.7B).
- `--task include_v2_og_spanish_jamaica`: a task whose verdict is not monotone in size (mask 1/0/0/1/1/1), to show the share moving with the runs.
- The same figure on a 35-item task (`cultural_bench_easy_argentina`), where the needed margin is +0.12: the item count, not the model, decides.
- A `--pool predictivity_seeds` version with the replicate seeds, to see how much the per-run verdict moves with the seed.

Files: [`above_random_example.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_example.png), [`above_random_example.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_example.csv).
<!-- END auto:above-random-example -->

## Figures, in storyline order

### 1. Which cells carry any information: the gate per family and language

*Pool `predictivity` (192 runs, seed 1904, every ladder and data build),
sizes 90M–1.7B, one-sided 95 % Wilson gate (`MIN_SHARE` = half of the size's
runs that train the language). No task filter, and BPB and the loss have no
chance level.* This is the gate every later figure reads (on its 933 trained
tasks), and the first result of the study.

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

- **Most of the evaluation is at chance somewhere on the ladder.** Of the
  1,696 benchmark-language tasks with a chance level (`above_random_mask.csv`),
  619 / 663 / 704 / 801 / 873 / 946 are above chance at 90M / 175M / 350M /
  600M / 1B / 1.7B (1,511 tasks are scored below 600M, 1,667 from 600M on).
  1,049 (62 %) fail at one size or more, and only 585 pass at all six.
- **986 clear the gate at ≥ 1 size and 710 nowhere (29 of them never
  scored).** 40 are above chance at a smaller size but at chance at 1.7B.
  `first_size_above_random` codes them "never", which is a different fact.
- **On the 933 trained tasks alone the counts are 377 / 411 / 447 / 498 / 551 /
  599 at 90M … 1.7B.** 620 pass at ≥ 1 size and 356 at all six. The other 763
  tasks (29 of them never scored) are in languages no cell trains. The gate
  reads them on every run, and they never reach a later table.
- **Of the 689 (family, language) cells of `first_size_above_random.csv` (52
  families, the `rfgm_` twins among them), 283 hold from 90M on.** 32 hold
  from 175M, 29 from 350M, 50 from 600M, 40 from 1B, 42 only at 1.7B and 213
  never.
- **At 1.7B the letter-answer four-option families are nearly absent over all
  their languages.** `global_mmlu_full` passes 1 / 37, `include_base_44` 3 / 43
  and `belebele` 5 / 105. By contrast `multiblimp` passes 57 / 57, `xwinograd`
  6 / 6, `hellaswag` 31 / 31 and `xnli` 15 / 18.
- **One cloze four-option family also stays mostly at chance.**
  `global_piqa_parallel_cloze` (103 items per task) passes 10 / 91 at 1.7B,
  and 10 / 63 on trained languages.
- **The reformulated twins pass where the originals fail.** At 1.7B
  `rf_global_mmlu_full` passes 35 / 37, `rf_belebele` 86 / 105,
  `rfgm_belebele` 59 / 59, `rf_include_base_44` 31 / 43 and
  `rfgm_include_base_44` 32 / 43 (figure 3).
- **The 77 % / 51 % two- / four-option split of figure 2's table is mostly not
  an option-count effect.** On the trained tasks at 1.7B the two kinds pass at
  similar rates (83 / 122 against 488 / 759). The gap comes from 304 untrained
  two-option tasks, mostly BLiMP-style minimal pairs (`zhoblimp` 118,
  `blimp_nl` 84).
- **The fewer the items, the larger the margin the Wilson bound asks.** Per
  family (`above_random_thresholds.csv`, median task): +0.006 on the
  14,042-item MMLU families, +0.007 on `hellaswag` (9,275), +0.043 on
  `xwinograd` (409), +0.046–0.077 on the 130–250-item `bbh` / `acp_bench`
  probes, +0.09 on the 100-item `global_piqa_nonparallel_cloze` and
  `cultural_bench_hard`, +0.125 on `toksuite` (40 items) and +0.157 on
  `cultural_bench_easy` (25 items).
- **The gate measures the format's floor more than the languages'
  difficulty.** A model below 1B rarely clears chance on a four-option
  letter-answer task in a non-English language.
- **Statistically above chance is not usefully above chance.** The thinnest
  passing HellaSwag cell is `hellaswag_hy` at 175M, mean 0.269 against 0.25
  over 8,632 items (+0.019), read on untrained runs since no cell trains
  Armenian. On trained cells the thinnest is `hellaswag_ne` at 90M, 0.274
  over 9,318 items (+0.024). `above_random_thresholds.png` shows what each
  rule keeps. Whether to add an effect floor on top of the test is left to the
  paper.
- **The gate does not catch a letter preference.** `cultural_bench_easy`
  passes at 90M in 10 of its 19 languages and in none from 175M on. The 32
  runs at 90M that train its languages have median scores (over the
  languages) from 0.07 to 0.65 against a 0.25 chance. This points to an
  answer-letter bias meeting an unbalanced gold key, not to knowledge (not
  checked item by item here, see section 1b).

**Follow-ups**

- Draw the run count per cell on the gate figures: a trained cell rests on 1
  to 32 runs, and no figure shows it. The 16 languages that only the L50 cells
  train (`az`, `bs`, `ca`, `et`, `hr`, `kk`, `lt`, `lv`, `mr`, `ms`, `ne`,
  `sk`, `sl`, `sq`, `sr`, `ur`) have four runs per size, so two runs decide.
- The runs of a size share the test set, so "at least half of the runs" is a
  majority rule, not a pooled test. The six TruthfulQA tasks have a variable
  option count per item, so their chance is an approximation (mc1 0.2253, mc2
  0.449). A pooled test per cell would be the rigorous variant.
- A letter-bias control: score each task's pick-one-letter rate per run and
  gate against it, which would test the `cultural_bench_easy` 90M passes above.
- The population can change along the size axis: a size with no cell that
  trained the language falls back to every cell. `above_random_share.csv`
  carries a `population` column (5,598 cells `trained`, 3,936 `all`). A figure
  variant that greys the fallback cells would make the switch visible.
- Bug #16 (`../CLAUDE.md`): these panels are drawn from the mask at the time
  the driver reaches them. On 2026-09-23 the mask was rewritten after the
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

**The map condensed, for the paper** (`first_size_share_paper.png`, one bar
per benchmark, and its full-width twin `first_size_share_paper_horizontal.png`,
the same bars standing along x, `panels.py --paper`). *Pool `predictivity`,
sizes 90M–1.7B, the same gate. Only the trained languages of the L50 list, with
a benchmark's language count in brackets. 52 benchmarks, the `rf_`/`rfgm_` twins
and the still-pending rf_m_mmlu, rf_mmmlu and toksuite among them. The two
CSVs are identical.*

![Share of languages above chance from each size](pretraining/predictivity/first_size_share_paper.png)

![Share of languages above chance from each size, horizontal](pretraining/predictivity/first_size_share_paper_horizontal.png)

**Key findings** (`first_size_share_paper_horizontal.csv`)

- **12 of the 52 benchmarks never clear the gate in any trained language,
  and 15 clear it from 90M in all of them.** The 12 are `mmlu`,
  `commonsense_qa`, `bbh_mcq` / `bbh_cloze`, `acp_bench_mcq` /
  `acp_bench_cloze`, `cultural_bench_easy` / `_hard`, `blend_sample`,
  `toxigen` and both TruthfulQA families. The 15 are `hellaswag`,
  `multiblimp`, `mathqa`, `openbookqa`, `rfgm_belebele`, `rf_mmlu`,
  `rf_commonsense_qa`, `boolq`, `piqa`, `multiblimp-extra`, `noreval`,
  `noreval_extra`, `toksuite`, `toksuite_math` and `toksuite_stem`.
- **The twins flip the verdict.** `belebele` never clears in 0.94 of its
  languages, and `rf_belebele` clears in all of them (0.86 from 90M).
  `global_mmlu_full` never clears in 0.97, and `rf_global_mmlu_full` clears in
  all (figure 3).

**Follow-ups**

- Order the bars by benchmark kind (original, `rf_`, `rfgm_`) beside the
  share order, so each twin sits next to its original.

GitHub: [first_size_share_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_paper.png) ·
[first_size_share_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_paper.csv) ·
[first_size_share_paper_horizontal.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_paper_horizontal.png) ·
[first_size_share_paper_horizontal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_paper_horizontal.csv)

The gate leaves two questions open. Figure 4 asks whether the at-chance
families are unreadable for any model of these sizes or only for this ladder.
Figure 3 asks whether the format is the cause.

### 1a. Every benchmark we considered, the screened ones included

Figure 1 and the counts above cover the analysed population, the `auto` group.
We also screened benchmarks that we did not adopt: the `auto_probe` benchmarks
outside `auto`, and the `discarded` group. From the probe jobs of 2026-09-22
on, they were evaluated only on the final checkpoint of the 600M, 1B and 1.7B
cells (`auto_evals_cscs.py --group auto_probe --size 600M,1B,1.7B
--final-only`, gated by `../rq00_task_reformulation/probe.sh`). From the
2026-10-09 probe jobs on, they are also evaluated on six 3B cells (deep L8,
L15, L30 and L50, scheme-B L8 and L15, iteration 145200). This variant
therefore reads them at `utils.PROBE_SIZES` (600M, 1B, 1.7B and 3B) wherever
the probe gate has a result, on the runs that train each task's language, and
marks them apart from the analysed ones.

A size without a probe result has no verdict. It never counts as a fail.
Until the 3B jobs land and `probe.sh` re-runs the probe gate (with
`--above-reference`, the 3B column), every output covers 600M to 1.7B. The
analysed group stays at 90M to 1.7B (rule 10), so the all-considered row is
read at 600M, 1B and 1.7B alone. `discarded` keeps its definition: at chance
at 1B and at 1.7B (`utils.DISCARD_SIZES`, the rule `configs/tasks.json` was
built with). `gate_screened_verdicts_all_considered.csv` gives
`discard_rule`, `fails_600M` and `passes_3B` for each screened task. From it
one can list the tasks at chance at 1B and 1.7B that clear chance at 3B, and
the tasks a 600M pass would keep.

**The nine benchmarks promoted into `auto` on 2026-10-09** (`plan/probe-promotion-2026-10-09.md`:
rf_m_mmlu, rf_mmmlu, kobest, multiblimp-extra, noreval, toksuite, piqa,
winogrande, boolq) enter the analysis once their ladder evals land. Rule 3
reads a task along the run at the ten tenths, while the probe gave final
checkpoints only. `utils.pending_benchmarks()` holds an `auto` benchmark out of
the analysed group until at least half the seed-1904 cells that score it have
ten checkpoints or more in the ladder report. On the 2026-10-10 report six of
the nine are analysed, and rf_m_mmlu, rf_mmmlu and toksuite are still
pending. This variant counts a pending benchmark with the screened ones (it
was screened, and its verdicts are the probe's), and leaves out a benchmark
that has no result at all. The paper's counts (`make_appendix_tables.py`) do
the same. The split updates by itself as the ladder evals arrive. It is built
in the shared layer (`utils.considered_tasks`, `above_random.considered_gate`)
and no pool reads it: the DA and SNR populations stay on `auto`, and the
canonical figures and tables of figure 1 do not change. Snapshot: the
canonical gate of the 2026-10-10 08:19 refresh and the probe gate written by
`probe.sh` on 2026-10-09 (no 3B column yet).

<!-- BEGIN auto:considered (panels.py --pool predictivity) -->
## Every considered benchmark, the screened ones included

**Pool `predictivity` (seed 1904), the same gate, a cell read on the runs that train the task's language.** Analysed is the `auto` group at every size (the canonical tables above), minus the benchmarks whose ladder evals have not landed (`utils.pending_benchmarks`: none), which count as screened. Screened is the `auto_probe` benchmarks outside the analysed group, and discarded is the `discarded` group. Both were evaluated on the final checkpoint of the 600M, 1B and 1.7B cells only (`rq00_task_reformulation/probe.sh`, gate tables in `utils.PROBE_GATE_DIR`), so their columns below 600M are empty and the all-considered row is read at 600M, 1B and 1.7B. Tasks without a chance level have no verdict. No pool reads this variant: the DA and SNR populations stay on `auto`. Regenerate with `python analysis/rq00_gate_and_curves/panels.py --paper`.

| group | format | benchmarks | tasks | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|---|---|---|
| Analysed | original | 40 | 630 | 204 / 630 | 219 / 630 | 242 / 630 | 265 / 630 | 303 / 630 | 328 / 630 |
| Analysed | RF | 10 | 208 | 104 / 208 | 119 / 208 | 131 / 208 | 153 / 208 | 166 / 208 | 178 / 208 |
| Analysed | LLM-RF | 2 | 95 | 69 / 95 | 71 / 95 | 75 / 95 | 80 / 95 | 85 / 95 | 89 / 95 |
| Analysed | all | 52 | 933 | 377 / 933 | 409 / 933 | 448 / 933 | 498 / 933 | 554 / 933 | 595 / 933 |
| Screened, not analysed | original | 24 | 288 |  |  |  | 245 / 288 | 276 / 288 | 282 / 288 |
| Screened, not analysed | RF | 4 | 4 |  |  |  | 2 / 4 | 4 / 4 | 4 / 4 |
| Screened, not analysed | all | 28 | 292 |  |  |  | 247 / 292 | 280 / 292 | 286 / 292 |
| Discarded | original | 28 | 228 |  |  |  | 2 / 228 | 0 / 228 | 0 / 228 |
| Discarded | RF | 1 | 1 |  |  |  | 0 / 1 | 0 / 1 | 0 / 1 |
| Discarded | all | 29 | 229 |  |  |  | 2 / 229 | 0 / 229 | 0 / 229 |
| All considered | all | 93 | 1454 |  |  |  | 747 / 1454 | 834 / 1454 | 881 / 1454 |

![Every considered benchmark against the gate](pretraining/predictivity/first_size_share_all_considered_paper.png)

Files: [`gate_pass_counts_all_considered.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_pass_counts_all_considered.csv) / [`.tex`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_pass_counts_all_considered.tex), [`first_size_share_all_considered_paper.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_all_considered_paper.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/first_size_share_all_considered_paper.csv), [`gate_screened_verdicts_all_considered.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/gate_screened_verdicts_all_considered.csv).
<!-- END auto:considered -->

**Key findings** (`gate_pass_counts_all_considered.csv`, `first_size_share_all_considered_paper.csv`)

- **We considered 93 benchmarks with a chance level, 1,454 tasks on the 50
  trained languages.** 47 are analysed (887 tasks), 33 are screened and not
  analysed (338 tasks, among them the three pending benchmarks rf_m_mmlu,
  rf_mmmlu and toksuite), and 29 hold the 229 `discarded` tasks (most share a
  benchmark with screened tasks). With the tasks that have no chance level
  (LAMBADA, MELA, XQuAD and the f1 tasks), the paper counts 97 benchmarks and
  1,539 tasks (`make_appendix_tables.py`, which counts on the configs: 51
  screened benchmarks, 626 tasks of which 229 discarded).
- **Almost every screened task that was kept passes.** 287 of 338 (85 %) pass
  at 600M, 322 (95 %) at 1B and 332 (98 %) at 1.7B. 316 pass at both 1B and
  1.7B and none at neither (the 129 that failed both went to `discarded` on
  2026-10-09). Six fail at 1.7B after passing at 1B (three KMMLU subjects,
  TruthfulQA mc1 ca and pt, one ZhoBLiMP paradigm), and two pass at 600M and
  fail later (truthfulqa_ru_mc1, rf_m_mmlu_ne). The screened benchmarks were
  not adopted because of cost, redundancy with an analysed benchmark, or a
  pending decision (`plan/probe-promotion-2026-10-09.md`), not because the
  gate rejected them.
- **All 229 discarded tasks fail at 1B and at 1.7B, and 227 at 600M too.**
  The two that pass at 600M alone are `evalita-mp_te_prompt-3` and
  `truthfulqa_mr_mc2`. A discard rule read at 600M, 1B and 1.7B (at chance at
  all three) would keep them screened and discard 227
  (`gate_screened_verdicts_all_considered.csv`, `discard_rule` and
  `fails_600M`). `configs/tasks.json` keeps the 1B/1.7B rule (229).
- **Over everything considered, the gate keeps 747 of 1,454 tasks (51 %) at
  600M, 831 (57 %) at 1B and 885 (61 %) at 1.7B,** against 458, 509 and 553
  of the 887 analysed (52 %, 57 %, 62 %).
- **The format effect holds on the screened set.** At 1.7B the RF twins pass
  43 of 43 screened tasks, against 289 of 295 originals (98 %). The
  originals that failed are now in `discarded` (228 of its 229 tasks), so
  the screened gap is small by construction. Among the analysed tasks, RF
  passes 139 of 169 (82 %), LLM-RF 89 of 95 (94 %) and the originals 325 of
  623 (52 %).
- **16 benchmarks of the screened panel never pass at 1.7B, eight of them
  multiple-choice knowledge exams** (ArabicMMLU, CareQA, C-Eval, MMMLU, the
  OALL Arabic MMLU and EXAMS, TurkishMMLU, HAE-RAE Bench). The others are ACVA
  (OALL), Bangla, the FrenchBench extra RF twin, TruthfulQA (mc1),
  TruthfulQA-Multi (mc2), and kobest, noreval and noreval_extra, whose only
  tasks in the panel are their discarded ones. 20 screened benchmarks pass in
  every task at 1.7B, among them the RF twins of MMLU (Okapi), ArabicMMLU and
  CareQA, whose letter originals pass in 1 of 27, 0 of 1 and 0 of 1 tasks.

**Follow-ups**

- Re-run `probe.sh` and `panels.py --paper` once the six 3B probe jobs land
  (the 3B column), and again as the promoted benchmarks' ladder evals arrive
  (`pending_benchmarks` then empties and they move to the analysed group).
- The `lm-1.7B-L8-swiglu-seed1904` cell finished after the first probe ran
  and has no probe result for the screened tasks (D6 of the promotion plan).

### 1b. The early passes: a constant answer, the longest option, or a related language?

Figure 1 found CulturalBench-easy above chance at 90M and at chance from 175M on, and the token curves (`../rq00_chance_vs_train_tokens`) show a few tasks above chance at very small token counts of their language and at chance at intermediate ones. The paper names two candidate causes: a preference for the longest option early in training, and the tokens of a related language. The letter bias suspected above is a third. The table tests the three on the stored per-option log-likelihoods and on the all-language evaluation of the deep scheme-A ladder.

<!-- BEGIN auto:early-passes (early_passes.py --pool predictivity) -->
## The early passes: a constant answer, the longest option, or a related language?

**Gate verdicts per run · deep scheme-A seed-1904 ladder, every evaluated tenth, trained and untrained languages (a rule 2 opt-out, since this is a control of the gate, not a population).** The tasks are chosen on the seed-1904 cell table of `../rq00_chance_vs_train_tokens` (every build and ladder). A task is chosen when it is above chance in at least half of its cells below 1e+08 tokens of the language and in fewer than half of those between 1e+08 and 1e+10, with at least 10 cells on each side. 11 tasks qualify. For each task the table gives the share of runs above chance with few (`low`) and intermediate (`mid`) tokens of the language. It also gives that share for the untrained runs (no tokens of the language and no trained language of its subfamily) at the same (size, checkpoint) points as the low ones. The group `untrained, related` has 0 runs. Over the passing low-token runs, it then gives the median share of items answered with the run's most picked option, the share of those runs whose most picked option alone would clear chance, and the median share of picks and of gold answers on the single longest option. Regenerate with `python analysis/rq00_gate_and_curves/early_passes.py --pool predictivity`.

| task | lang. | items | pass low (runs) | pass mid (runs) | pass untrained (runs) | passing low runs | one-option share | constant passes | picks longest | gold longest |
|---|---|---|---|---|---|---|---|---|---|---|
| `cultural_bench_easy_argentina` | es | 35 | 0.79 (14) | 0.31 (226) | 0.88 (8) | 11 | 0.91 | 0.91 | 0.13 | 0.23 |
| `cultural_bench_easy_chile` | es | 22 | 0.64 (14) | 0.23 (226) | 0.62 (8) | 9 | 0.91 | 0.89 | 0.26 | 0.32 |
| `cultural_bench_easy_india` | hi | 46 | 0.38 (87) | 0.09 (33) | 0.31 (176) | 33 | 0.78 | 0.97 | 0.18 | 0.22 |
| `cultural_bench_easy_japan` | ja | 53 | 0.79 (14) | 0.28 (226) | 0.75 (8) | 11 | 0.94 | 0.91 | 0.20 | 0.28 |
| `cultural_bench_easy_mexico` | es | 49 | 0.64 (14) | 0.26 (226) | 0.75 (8) | 9 | 0.94 | 1.00 | 0.26 | 0.24 |
| `cultural_bench_easy_morocco` | ar | 17 | 0.45 (71) | 0.22 (109) | 0.41 (75) | 32 | 0.88 | 1.00 | 0.24 | 0.18 |
| `cultural_bench_easy_peru` | es | 19 | 0.57 (14) | 0.20 (226) | 0.62 (8) | 8 | 0.97 | 1.00 | 0.06 | 0.31 |
| `cultural_bench_easy_spain` | es | 40 | 0.64 (14) | 0.27 (226) | 0.62 (8) | 9 | 0.95 | 1.00 | 0.27 | 0.35 |
| `include_base_44_georgian` | ka | 500 | 0.30 (60) | nan (0) | 0.33 (300) | 18 | 0.97 | 0.94 | 0.22 | 0.23 |
| `paws_ja` | ja | 2000 | 0.64 (14) | 0.11 (226) | 0.12 (8) | 9 | 0.89 | 1.00 | 0.89 | 0.56 |
| `rf_m_mmlu_ml` | ml | 11028 | 0.33 (60) | nan (0) | 0.53 (300) | 20 | 0.28 | 0.95 | 0.38 | 0.28 |

Over the 169 passing low-token runs: 65% answer one option on at least 80% of the items, and in 96% that option alone clears chance.

Files: [`early_passes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/early_passes.csv) / [`.tex`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/early_passes.tex), [`early_passes_runs.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/early_passes_runs.csv).
<!-- END auto:early-passes -->

Key findings (`early_passes.csv`, `early_passes_runs.csv`, snapshot of the 2026-10-10 refresh):
- **Ten tasks pass early and fail later: eight CulturalBench-easy countries, INCLUDE Georgian and PAWS-X Japanese.** CulturalBench-easy asks in English about a country and is placed at the tokens of the country's language, so its token axis says little about it.
- **The early passes are a constant answer.** Over the 149 passing low-token runs, 74 % give one option on at least 80 % of the items (median share 0.78–0.97 per task), and in 97 % that option alone would clear chance. The answer keys of these 17–53-item tasks are unbalanced, and in the median run the model's accuracy is within 0.02 of the constant answer's (0.13 at the 90th percentile).
- **They fail later because the preferred option moves.** In the failing runs the most picked option sits on a minority gold position (median constant accuracy 0.11–0.32 per CulturalBench-easy task), and the share of items it takes drops to 0.55–0.72.
- **A longest-option preference explains only PAWS-X Japanese.** There the passing runs pick the longer continuation (the "no" answer) on 89 % of the items, and it is gold on 56 %. Elsewhere the picks on the longest option (0.06–0.27) sit at or below the gold share on it (0.18–0.35).
- **Related-language tokens are not needed.** None of the ten languages has a trained relative in the untrained deep scheme-A runs. Those runs, which see no token of the language, pass as often at the same size and checkpoint (0.31–0.88 against 0.30–0.79 for the low-token trained runs). The exception is PAWS-X Japanese (0.12 against 0.64).

Follow-ups:
- Gate every task against its constant-answer accuracy (the share of its majority gold label) instead of 1/n, which would remove these passes at the source.
- Run the same per-option test on every task at 90M, to count the passes that the constant answer explains beyond these ten.

### 1c. A majority-class chance level: how much of the gate rests on uniform guessing?

Section 1b found passes that a constant answer explains, and its first follow-up proposed holding every task to its constant-answer accuracy. The canonical gate keeps uniform guessing as its chance level (rule 1), so this is a sensitivity analysis. We rerun the same gate with max(uniform, majority) as the chance level. The majority baseline is the share of a task's evaluated items whose gold option index is the most frequent one, i.e. the accuracy of always answering "A" (or always "yes"). It is read from the harness samples of the ladder's final checkpoints for 1,664 of the 1,669 accuracy tasks of `configs/tasks.json` (`auto`, `auto_probe` and `discarded`, under their `majority` key). The rerun lives in [`majority_chance/`](majority_chance/), never in the tables the RQs read. A constant preference can reach the majority baseline only where each answer position carries a fixed label on every item (a letter, Yes / No, True / False: 559 tasks). On cloze tasks, whose answers change per item, the scored log-likelihoods never see the position. A third gate therefore applies the majority level to the labelled tasks alone.

![Majority-class chance: which tasks flip, and what the gate keeps](majority_chance/majority_chance.png)

*Pool `predictivity` (seed 1904, 90M–1.7B), the same runs and Wilson rule as the canonical gate. The rerun reproduces the canonical mask on every task whose chance level does not move (0 differing cells). Population: the 933 trained tasks with a chance level, cells read on the runs that train the language. Regenerate with `python analysis/rq00_gate_and_curves/majority_chance.py` (`--derive --write-tasks-json` rereads the samples).*

Key findings (`majority_pass_counts.csv`, `majority_flips.csv`, `majority_by_benchmark.csv`, `majority_downstream.csv`, ladder report 2026-10-10 04:31):
- **Taken at face value, the majority level removes almost half of the gate's passes.** Trained tasks above chance fall from 377 to 146 of 933 at 90M and from 599 to 426 at 1.7B, and the tasks never above chance rise from 313 to 505. Of the 2,883 passing trained cells, 1,302 (45 %, from 447 tasks) flip from pass to fail (231 at 90M, 173 at 1.7B).
- **Almost all of it is answer order, which a log-likelihood model cannot use.** 92 % of the flipped cells (1,199 of 1,302) are cloze tasks with per-item answers. MultiBLiMP lists the grammatical sentence first and TruthfulQA its true answers first (majority 1.0, so every MultiBLiMP pass flips). The harness also keeps the source datasets' gold-position skew in INCLUDE v2 (majority up to 0.84), in the RF and LLM-RF twins of Belebele, Global-MMLU and INCLUDE (0.27–0.28 against 0.25, enough over 900–14,000 items), and in ARC, XStoryCloze and Global PIQA. Among the newly analysed benchmarks, NorEval (extra) flips 121 cells, rf_m_mmlu 98 and rf_mmmlu 37.
- **Where a constant answer is possible, the gate barely moves.** On the labelled tasks alone the counts are 352 at 90M and 579 at 1.7B (against 377 and 599), 103 cells of 43 tasks. They come from CulturalBench-easy (10 countries, letter keys with majorities of 0.45–0.86 over 7–59 items), PAWS-X (5 languages, majority 0.55, 14 cells), the letter-format Belebele, Global-MMLU and INCLUDE (13 tasks that the uniform gate barely passed), BBH (MCQ), BLEnD, NorEval (extra) (10 tasks, 55 cells, majority 0.67–0.69), NorEval and KoBEST (one task each). XNLI's labels are balanced (0.333). BoolQ is on the ladder (one task, above chance at all six sizes) but has no majority baseline yet, so the rerun gates it at the uniform level only.
- **The downstream headline numbers hold.** rq02's DA-size on accuracy moves by at most 0.006 under either majority gate (0.51–0.57 from 90M to 1B). rq01's share of accuracy and BPB tasks predictable along size and training rises from 74 % (311 of 420) to 87 % (219 of 251) under the full majority gate, because the 169 tasks it drops are less often predictable (92 of them, 54 %). On the labelled tasks alone it is 76 % of 408. The median size-fit R² of the accuracy tasks is 0.87 (390 tasks) under the uniform gate and 0.92 (221) under the full majority gate.

Follow-ups:
- Count the constant-answer passes the labelled gate removes per checkpoint, not only at the final one, to see whether the early passes of section 1b are all of them.
- Read the majority level per run instead of per task (the option the run picks most often, as in 1b), which bounds what a run's constant preference can reach rather than what the key allows.

[`majority_chance.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/majority_chance/majority_chance.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/majority_chance/majority_chance.csv), [`majority_baselines.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/majority_chance/majority_baselines.csv), [`majority_downstream.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/majority_chance/majority_downstream.csv).


### 2. The gate by answer count, and the Signal ranking

*Pool `predictivity`. The Signal of a task is (max − min) / mean of the
per-setting final scores at 1.7B over the six L1 … L50 deep scheme-A (T 1)
seed-1904 cells, computed on every task **without the gate**. 1,360 of the
1,806 parent tasks have a value, the `bbpb_` twins among them.*

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
| 2 | 0.50 | 317 / 426 | 312 / 426 |
| 3 | 0.33 | 23 / 31 | 21 / 31 |
| 4 | 0.25 | 597 / 1121 | 568 / 1121 |
| 5 | 0.20 | 36 / 75 | 34 / 75 |
| 6 | 0.17 | 1 / 4 | 1 / 4 |
| 7 | 0.14 | 3 / 34 | 3 / 34 |
| 8 | 0.12 | 2 / 2 | 1 / 2 |
| 10 | 0.10 | 0 / 2 | 0 / 2 |
| 12 | 0.08 | 1 / 1 | 1 / 1 |
<!-- END auto:results -->

**Key findings**

- **Over all 1,696 tasks the table splits two- from four-option tasks.** Of
  the tasks scored at the size, 312 / 403 (77 %) two-option tasks pass at 1.7B
  against 573 / 1,115 (51 %) four-option ones, and 262 / 386 against
  337 / 1,040 at 90M. The split reflects the population, not the option count:
  on the trained tasks the two kinds pass at similar rates (figure 1).
- **The Signal ranking is a selection on noise, not a reliability ranking.**
  Its top families by mean Signal are `cultural_bench_easy` (1.35, 19 tasks of
  7–59 items), `bbpb_bbh_cloze` (0.78, 6) and `bbpb_bbh_mcq` (0.60, 17).
- **The first is at chance at 1.7B on all 19 of its tasks.** The other two are
  bits-per-byte twins of short BBH answers, and the top task,
  `rf_bbh_mcq_temporal_sequences` (3.39), scores at most 0.068 against a 0.25
  chance.
- **Read on the tasks above chance at 1.7B, the ranking tops out at 0.65
  (`include_v2_og_spanish_nicaragua`).** Next come two more `include_v2_og`
  regional tasks (0.42 and 0.38), `toksuite_farsi_canonical` (0.35),
  `include_v2_og_arabic_morocco` (0.35), `nortruthfulqa_mc_nno_p1` (0.33),
  `global_piqa_parallel_cloze_eng_latn` (0.32) and `toksuite_math` (0.30).
  Everything else is at ≤ 0.29, among it the other Global PIQA tasks and the
  `rf_` cultural, BBH and ACP twins. For the gated readings, see
  [decision accuracy](../rq02_decision_accuracy/README.md) and
  [noise and SNR](../rq03_noise_and_snr/README.md).

**Follow-ups**

- Apply the gate to the Signal ranking (mask the at-chance cells before
  ranking), and rank on the gated 1.7B cells only.
- Rank the accuracy tasks and the `bbpb_` twins separately: a bits-per-byte
  spread and an accuracy spread are not on one scale.
- `per_benchmark/<family>.png` curves are redrawn only with `CURVES=1`
  (`refresh_analysis.sh --curves`). The top-3 grids (`cultural_bench_easy.png`,
  `bbpb_bbh_cloze.png`, `bbpb_bbh_mcq.png`) were redrawn on 2026-10-10, but
  that redraw is partial and some other grids still date from 2026-10-06.

[acc_vs_flops_signal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/acc_vs_flops_signal.csv) ·
[above_random_scores.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_scores.csv) ·
[above_random_share.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_share.csv) ·
[above_random_runs.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_runs.csv)

### 3. The reformulated twins move whole families across the gate

*Pool `predictivity` (the gate mask over its 192 runs, seed 1904, every
ladder and data build), the `rf_` (letters → answer strings) and `rfgm_`
(Gemini-rewritten items) twins against their originals, paired on the
language over every language of the family, with McNemar's exact test on the
discordant languages. The figure and its auto block live in
[`../rq00_task_reformulation/README.md`](../rq00_task_reformulation/README.md#the-twins-and-the-gate-with-significance)
(`reformulations_gate.py`). That README also describes the pipeline that
produced the twins.*

![The reformulations and the gate](../rq00_task_reformulation/reformulations_gate.png)

*(a) Per family with a twin, the share of its languages above the gate for
the original letter format and for the `rf_` cloze twin at each size, paired
on the language. Diamonds mark the `rfgm_` rewritten twin, and a star marks a
size where McNemar's exact test on the discordant languages gives p < 0.05.
(b)–(d) The gate's pass share, the mean DA-size on the tasks above chance at
the proxy and at the reference (rule 1), and the share of reliable tasks, each
read on every task, on the originals alone and on the twins alone. The red
line is the tie null: what a proxy with no signal reads, because a pair tied
on one side counts as a miss.*

Every headline reading of the gate, scaling predictability and decision
accuracy can be read with and without the twins from
`reformulations_gate.csv`. DA-size is read on the multi-axis pairs with the
`bbpb_` twins excluded, and the R² column is the scaling-predictability fit on
`predictivity_seeds`:

| population | gate at 90M → 1.7B | mean DA-size 90M → 1B (gated at both sizes) | tasks with median DA-size ≥ 0.66 | scaling-fit median R² (tasks) |
|---|---|---|---|---|
| every task | 0.41 → 0.57 | 0.51 → 0.56 | 17 % of 619 | 0.868 (390) |
| originals only | 0.37 → 0.49 | 0.53 → 0.57 | 22 % of 354 | 0.882 (223) |
| twins only | 0.54 → 0.82 | 0.49 → 0.56 | 11 % of 265 | 0.851 (167) |

**Key findings**

- **The twins' effect on the gate is not chance.** At 1.7B the cloze twin clears
  the gate in 82 Belebele languages where the original does not, against 1 the
  other way (the rewritten `rfgm_` twin 56 against 0 over its 59 languages).
  The counts are 34 against 0 for Global-MMLU and 29 against 1 for INCLUDE (30
  against 1 for `rfgm_`), McNemar p < 10⁻⁷ each.
- **The smaller families point the same way at 1.7B.** BBH reads 10 against 0
  (p = 0.002), `cultural_bench_easy` 7 against 0 (p = 0.016), `acp_bench_mcq`
  5 against 0 (p = 0.06) (`reformulations_gate_mcnemar.csv`). At 90M
  `cultural_bench_easy` runs the other way (1 against 10): these are the
  letter-bias passes of figure 1.
- **The twins pass the gate far more often and are read from 90M, but they do
  not rank the design variants better.** Gated at both sizes they read mean
  DA-size 0.49–0.56 across the proxies, against 0.53–0.57 for the originals,
  and they clear the 0.66 cut half as often (11 % of 265 against 22 % of
  354).
- **Their spread across variants is only marginally larger.** The median
  relative signal (`signal_rel_std`) at 1B is 0.025 on 244 twins against 0.024
  on 298 originals, on the tasks above chance at 1B (noise-and-SNR
  `snr_variants_per_task.csv`, `bbpb_` twins excluded).
- **The family mix moves the twins' mean.** At 1B `rf_belebele` reads DA-size
  0.54 on 57 tasks and `rf_global_mmlu_full` 0.68 on 28
  (`agreement_da_size_per_cell_multi_axes.csv`). The tie null is ≈ 0.49 today
  (2.4 % one-sided ties per multi-axis cell), above the 0.47 that the figure's
  red line still hard-codes.
- **The reformulation decides whether a family exists for the rest of the
  analysis.** On the trained languages at 1.7B `belebele` passes 3 / 59 tasks
  and `global_mmlu_full` 1 / 29, against 57 / 59 and 29 / 29 for their `rf_`
  twins ([scaling predictability, figure 2](../rq01_scaling_predictability/README.md#2-survivorship-what-the-gate-and-the-fit-minimum-removed)).
  In all three rows of the table the gate share and DA-size rise from 90M to
  1B, so no headline reading flips with or without the twins.

**Follow-ups**

- Formulation as a variable: per family with a twin, the first size above
  chance, DA-size and SNR per formulation (letters vs cloze vs rewritten) in
  one figure. The `rfgm_` evals now cover INCLUDE (43 languages) and Belebele
  (59) at every size.
- Redraw the red line of panel (c) at today's tie null (≈ 0.49 from the
  2.4 % one-sided tie share) instead of the hard-coded 0.47.
- The twins are the same items in another formulation, so a language can
  clear rule 8's three-task floor with the twins of one benchmark. A
  per-language panel that rests on twins alone should say so (RULES.md).
- The length tell (`rf_acp_bench_mcq`: the gold answer is the strictly
  shortest option in 35 % of items, up to 50 % in three subtasks) means those
  tasks' gate should be read against the pick-shortest rate, not 1/4 (see
  [`../rq00_task_reformulation/README.md`](../rq00_task_reformulation/README.md#the-length-tell-and-where-it-actually-is)).

GitHub: [reformulations_gate.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.png) · [reformulations_gate.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.csv) ·
[reformulations_gate_mcnemar.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate_mcnemar.csv)

### 4. Benchmark floors: is a gated task a size problem or a benchmark problem?

*Pool `predictivity` (192 runs, seed 1904, final checkpoint, trained
languages) against the external tier's mask (`all/external/`, public base and
post-trained models 270M–70B). The overlap is the 100 tasks of the 36-sweep's
task list that both masks hold, minus the 26 in a language that no ladder cell
trains at some size: 74 tasks, with no twins and no probe families. The
external floor is the size of the smallest public model that reads a task,
whatever its training budget, not a size at the ladder's 5C budget.*

*Provisional: the external mask on disk dates from 2026-09-21. It still
applies the retired `score > 1/n_options + 0.05` rule (it agrees with that
rule on 1,169 of its 1,171 cells), not the Wilson gate the auto block names.*

<!-- BEGIN auto:above-random-external (above_random_external.py --pool predictivity) -->
## Benchmark floors: the ladder's gate against the public models'

Of the 74 tasks both tiers score, 27 are at chance at every ladder size. The external models (270M–70B, every release the external tier holds, base and post-trained, same gate) first read them at: ≤ 600M 8, 1B–1.7B 16, 3B–4B 0, 7B–14B 1, ≥ 27B 0, never 2. A task that a public model reads at ≤ 1.7B has a size or recipe floor: the 5×-Chinchilla ladder does not reach it, while public models of that size train on 10–36 T tokens. A task that needs ≥ 3B, or that no model reads, has a benchmark or language-resource floor. "Never reads" is the gate's verdict: at every ladder size fewer than 50% of the size's runs that train the task's language clear the one-sided 95 % Wilson bound over chance. Single runs may clear it. The overlap is the 100 tasks of the 36-sweep's task list that both masks hold, minus the 26 tasks in a language that no ladder cell trains at some size (there the gate would read untrained runs, which measures transfer, rule 2). Regenerate with `python analysis/rq00_gate_and_curves/above_random_external.py --pool predictivity`.

| family | ≤ 600M | 1B–1.7B | 3B–4B | 7B–14B | ≥ 27B | never |
|---|---|---|---|---|---|---|
| belebele | 3 | 7 | 0 | 0 | 0 | 0 |
| global_mmlu_full | 1 | 8 | 0 | 0 | 0 | 0 |
| paws | 0 | 1 | 0 | 0 | 0 | 1 |
| truthfulqa_mc2 | 2 | 0 | 0 | 0 | 0 | 0 |
| arc | 0 | 0 | 0 | 1 | 0 | 0 |
| commonsense_qa | 1 | 0 | 0 | 0 | 0 | 0 |
| mmlu | 1 | 0 | 0 | 0 | 0 | 0 |
| xnli | 0 | 0 | 0 | 0 | 0 | 1 |

![The gate on the public models](pretraining/predictivity/above_random_external.png)

The paper version, `above_random_external_paper.png` (alternatives `_paper_b`, one cell per task, and `_paper_c`, a family by bucket grid), recomputes the floor on the public base releases alone (no post-trained release, none of apertus3-a06, ap-from8b-TOP256): ≤ 600M 16, 1B–1.7B 6, 3B–4B 1, 7B–14B 0, ≥ 27B 2, never 2. `above_random_external_models.tex` lists those models per line and size bucket.

Population: the `predictivity` pool (seed 1904, 90M–1.7B, final checkpoint, trained languages) against every external release (`all/external`, base and post-trained, same gate). The paper version uses the public base releases only, and panels (c) and (d) use the six public lines gemma-3, Qwen3, OLMo-2, Olmo-3, Apertus, apertus3-a06. There is no task filter beyond the 74-task overlap.

Key findings:
- 27 of the 74 shared tasks are gated at every ladder size. An external model ≤ 1.7B reads 24 of them, and no model up to 70B reads 2.
- (c) the ladder reads 42% of the shared tasks at 90M and 64% at 1.7B. The public base models ≤ 1.7B read Qwen3-1.7B-Base 93%, Qwen3-0.6B-Base 85%, OLMo-2-0425-1B 67%, gemma-3-1b-pt 65%, apertus3-1b-21-nodes 64%, gemma-3-270m 60% (for a public model, its own run clears the bound, while the ladder's share is its mask, half of the runs).
- (d) 27 of the 27 ladder-gated tasks have a trained run at 1.7B. 25 of them have a single run whose lower bound clears chance. These are near misses, since the gate wants half of the runs. The best public base model ≤ 1.7B beats the ladder's best run on 25.

Follow-ups:
- (c) with the post-trained releases as open markers, to see whether instruction tuning moves a task over the gate at the same size.
- (d) at every ladder size, not only the reference, to find the size at which each near miss first has a passing run.
- (c) and (d) on the `rf_` cloze twins once the external tier is evaluated on them. The twins pass the gate on the ladder, so the overlap would no longer be the 36-sweep's task list alone.

Figure: https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.png · tables: https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.csv, https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external_lines.csv
<!-- END auto:above-random-external -->

**Key findings**

- **Most of what the ladder cannot read is a size or recipe floor, not a
  benchmark floor.** Of the 27 shared tasks at chance at every ladder size,
  public models read 8 at ≤ 600M and 16 at 1B–1.7B, none at 3B–4B, 1 at
  7–14B (`arc_hi`), none at ≥ 27B and 2 never (`paws_ja`, `xnli_ar`).
- **The ≤ 1.7B group is every Belebele and Global-MMLU task of the list,
  `mmlu`, `commonsense_qa`, `truthfulqa_{vi,zh}_mc2` and `paws_zh`.** The
  7–14B group is `arc_hi` alone, and no shared task first passes at ≥ 27B.
- **Public models of the ladder's sizes read far more of the shared tasks.**
  Qwen3-0.6B-Base reads 85 % and Qwen3-1.7B-Base 93 % (of the 72 they are
  scored on), against 42 % of 74 for the ladder's mask at 90M and 64 % at
  1.7B (`above_random_external_lines.csv`).
- **No language-resource floor is left in the overlap.** Basque and Swahili,
  which no 1.7B ladder cell trains, left the overlap with the other tasks in a
  language that no ladder cell trains at some size. The 3 tasks that need ≥ 7B
  or are never read are in Hindi, Japanese and Arabic.
- **Near misses.** All 27 tasks have a trained run at 1.7B, and on 24 of them
  at least one run's Wilson bound clears chance. There the ladder's verdict
  reflects the half-of-the-runs rule as much as a floor.

**Follow-ups**

- Regenerate the external mask with the Wilson gate (`above_random.py --only
  external`), so that both tiers use one rule. The floors above are
  provisional until then.
- Extend the external gate to the `auto` list (the reference models' twin and
  probe evals), so the floors cover the twins and probe families, the first
  item of `plan/next_analyses.md` §2.
- A token-axis floor from the six public step series (the token count at
  which a fixed-size model leaves chance per task), which turns "how many
  parameters" into "how much data".
- A floor-conditioned reading of decision accuracy: DA against the proxy's distance
  above the task's floor in doublings. If the lines coincide, the external
  floors predict how far above its floor a proxy must sit for a task the
  ladder never reads (`plan/next_analyses.md` §7c).

GitHub: [above_random_external.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.png) · [above_random_external.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_external.csv)

### 5. The ladder's curves

*Pool `predictivity_seeds`, every trained cell, the shared checkpoint grid:
212 cells. 192 are at seed 1904, six each at seeds 64 and 313 (three at 175M
and three at 600M), four each at seeds 28 and 1797 (at 1B), and none of the
replicates is at 1.7B. By data scheme: A at T 1 126, A at T 3 24, B 50, C 12.
By ladder: deep 116, shallow 60, swiglu 19, muon 17.*

These figures are descriptive. They redraw the progress report's figures on
the analysis' own cells, so a reader can check any assumption of the analyses
above against the raw runs.

<!-- BEGIN auto:curves (curves.py --pool predictivity_seeds) -->
## Curves on the analysis' cells

Every cell the `predictivity_seeds` pool holds (all seeds and data builds), after the loader has dropped diverged and unfinished runs and kept the checkpoints on the shared grid. These are the detailed counterpart of the progress report's figures. The loss is drawn per L, over the whole run and over its last 10 % (capped at 3.5 nats, where ladders and data builds separate). Benchmark accuracy is the mean over the tasks in the languages the cell trains on, one line per cell, with chance from the option count, along the run in Chinchilla multiples. A family's cloze twin sits next to it, titled `(rf)`. Regenerate with `python analysis/rq00_gate_and_curves/curves.py --pool predictivity_seeds`.

![Loss curves](pretraining/predictivity_seeds/loss_curves.png)

![Benchmark curves](pretraining/predictivity_seeds/benchmark_curves.png)

The paper version, `benchmark_curves_paper.png` (`--paper`, redrawn from `benchmark_curves.csv`), replaces the header with a legend of the line encoding. Its size twin, `benchmark_size_curves_paper.png`, draws each design's final accuracy against non-embedding parameters, coloured by L (the size figure of the scaling-predictability appendix).

![Benchmark curves, paper version](pretraining/predictivity_seeds/benchmark_curves_paper.png)

![Final accuracy against size, paper version](pretraining/predictivity_seeds/benchmark_size_curves_paper.png)
<!-- END auto:curves -->

*The benchmark panel draws 206 of the 212 cells. Five of the 17 muon cells
(1.7B L15 and L30, 1B L8, L15 and L30) and one of the 19 swiglu cells (90M L1)
have loss curves only.*

**Key findings**

- **Letter-format knowledge MCQA rises well off chance in one cell only, and
  only in the last fifth of the run (between 4C and 5C).** On `mmlu` the
  English-only `lm-1.7B-L1-deep-seed1904` goes 0.274 → 0.316 → 0.337 at 4C /
  4.5C / 5C (chance 0.25). On `global_mmlu_full_en` it goes 0.270 → 0.303 →
  0.314, and on `commonsense_qa` 0.202 → 0.256 → 0.257 (chance 0.20).
- **It is the only one of 205 cells to end above 0.28 on `mmlu`.** Its shallow
  twin ends at 0.277, and the English-only scheme-B (DCLMP) and scheme-C (FWEB)
  cells at 1.7B at 0.239 and 0.247. The cloze `rf_mmlu` of the same cell moves
  smoothly (0.367 → 0.374 → 0.379).
- **Two families move below chance with training, so their "never above
  chance" verdict reflects a downward trend, not noise around chance.** The
  English TruthfulQA mc2 task falls from 0.421 at 0.5C to 0.387 at 5C (mean
  over 205 cells, chance 0.449), and ToxiGen from 0.525 to 0.450 (chance 0.5),
  in 88 % and 82 % of the cells.

**Follow-ups**

- Label the x axis of the loss figure in Chinchilla multiples (1C–5C) as the
  benchmark figure does (rule 3).
- Mark the last 20 % of the run on the benchmark panels, where the `mmlu`
  jump happens. It is one run of one corpus and no 1.7B cell has a seed
  replicate, so it is open whether the jump comes from the corpus or from the
  seed.

[loss_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/loss_curves.png) ·
[loss_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/loss_curves.csv) ·
[benchmark_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_curves.png) ·
[benchmark_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_curves.csv)

**The two paper grids** (`benchmark_curves_paper.png`, `benchmark_size_curves_paper.png`,
five panels per row, `curves.py --paper`). *Pool `predictivity_seeds`, the
same cells as above, no gate, no task filter. The size twin reads each
design's final checkpoint, one line per (L, ladder, data build, seed) across
its rungs: 46 designs, 29 of them at 1.7B, 53 families. Each L averages over
its own trained languages, so the populations differ across L.*

**Key findings** (`benchmark_size_curves_paper.csv`)

- **The cloze and completion families climb with size, and the letter-format
  ones do not.** Mean over the designs, 90M → 1.7B: `hellaswag` 0.304 →
  0.533, `xwinograd` 0.573 → 0.747, `rf_commonsense_qa` 0.304 → 0.462, while
  `mmlu`, `global_mmlu_full`, `include_base_44`, `bbh_mcq`, `acp_bench_mcq`
  and `commonsense_qa` move by 0.013 or less. `cultural_bench_easy`
  (0.393 → 0.259), `toxigen` and both TruthfulQA families fall.
- **At 1.7B the English-only cell scores highest on most families.** On the
  deep data-A seed-1904 designs, L1 is the top line on 28 of the 39 families
  with a value at every L and above L50 on 32 (`hellaswag` 0.675 against
  0.450, `arc` 0.576 against 0.322). L1 averages its English tasks only, so
  this is partly the task population, not a ranking of the mixtures.

**Follow-ups**

- A twin of the size figure on the English tasks alone, so that the L lines
  read one task population.

GitHub: [benchmark_curves_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_curves_paper.png) ·
[benchmark_size_curves_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_size_curves_paper.png) ·
[benchmark_size_curves_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity_seeds/benchmark_size_curves_paper.csv)

## Extensions from other sweeps

Everything below comes from the 36-model sweep (2026-04…06: 4 sizes × 3 data
mixtures × 3 seeds, `apertus-*`, 12 languages, pools `custom`, `seeds_*`,
`custom_swissai_hf`, `external`, sizes 175M–1B, reference 1B) or from the
public models of the external tier. Its harness, task set, reference size and
gate rule (a fixed +0.05 margin) differ from the ladder's, so its numbers are
replications of the findings above, never rows of the same table
(`../CLAUDE.md`, "Two sweeps").

## Custom vs. external: the at-chance problem is a capability artifact

*36-sweep numbers (pools `custom` and `external`, 2026-06, custom sizes
175M–1B). The ladder's counterpart is figure 4
above (`above_random_external`).*

The rest of the 36-sweep analysis built on this result. The above-random gate
is identical for every model set, but what it removes depends on which models
are evaluated.

**On the custom pretrains the gate removes most benchmarks.** Of 118 benchmarks,
only **44 clear chance at ≥1 size and 74 are random everywhere**. The 175M–1B
models we trained are too weak to show signal on most translated knowledge
MCQA.

The loss is concentrated in the 4-option families: only **9 / 63**
clear chance, so `belebele`, `global_mmlu_full`, `arc` and `truthfulqa` are
set to NaN before any SNR is computed. The surviving pool is therefore *almost
entirely 2-option*, which later biases the design-feature analyses toward
"fewer options ⇒ higher SNR".

**The same gate on the external tier removes almost nothing.** The `external`
model set (every non-custom model: `reference_hf`, a06, distillation and
posttraining, sizes 270M…70B, with the gate report in `all/external/` and the
curves skipped) clears chance on **122 / 124** benchmarks, including **68 / 69**
four-option families.

A 4-option translated benchmark is not low-signal in itself. It only looks
that way under models too small to beat chance.

| model set | models | benchmarks | above ≥1 size | 2-opt | 3-opt | 4-opt | 5-opt |
|---|---|---|---|---|---|---|---|
| `custom` (SNR-gate domain) | 175M–1B custom pretrains | 118 | **44 (37%)** | 28 / 42 | 7 / 11 | **9 / 63** | 0 / 2 |
| `external` (`all/external`) | 270M–70B reference / a06 / distill / post | 124 | **122 (98%)** | 42 / 42 | 10 / 11 | **68 / 69** | 2 / 2 |

Belebele shows this directly. On the `custom_swissai_hf` acc-vs-FLOPs curves
it sits flat at chance across the custom 175M–1B sweep, while the overlaid
external final-checkpoint markers climb toward 0.8+ up to 70B.

![Belebele: custom at chance, externals climb to 0.8+](pretraining/custom_swissai_hf/per_benchmark/belebele.png)

**Implication for the paper.** The custom-pool gate measures *our models' scale*,
not benchmark reliability. The external tier is the control that separates the
two.

This distinction explains the external-tier findings downstream: option
count stops predicting SNR, and 4-option HellaSwag becomes the
*highest*-SNR family once it clears the gate.

[belebele.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/per_benchmark/belebele.png)

## Methodology: scaling beyond 1B (36-sweep pools)

The `custom_swissai_hf` pool extends the compute axis past the custom 1B ceiling
by folding in the apertus3 a06, distilled, and reference-HF trajectories. Three
mechanisms make that work:

- **Bucketed size axis.** The four custom sizes (175M / 350M / 600M / 1B) stay
  singletons, and nearby external or large sizes pool into buckets so each holds ≥2
  models. A bucket with ≥2 models contributes a computable cross-model signal and
  decision accuracy. A singleton bucket (e.g. an isolated 70B) still plots its
  final-checkpoint marker on the curves but is dropped from the noise/DA pool.
- **Relative-fraction ckpt-DA.** External trajectories never hit the custom
  absolute megatron iters, so checkpoint-DA selects each model's early checkpoint
  at a *fraction* of its own max step rather than at a fixed iter. This lets the
  a06, distillation, SmolLM3, Olmo-3 and Apertus-8B series take part and extends
  ckpt-DA into the large buckets. (The computation lives in
  [compute_da.py](../rq02_decision_accuracy/compute_da.py). The gate and curves
  here only read its trajectories.)
- **Cross-bucket scaling-DA is family-coverage-limited.** Scaling-DA auto-detects
  bucket pairs where ≥2 model *families* span both sizes (via within-family
  ladders such as gemma-3 / OLMo). Above 1B this is sparse (few families ship
  multiple sizes), so the few detectable pairs saturate at DA 1.0 on a handful of
  shared tasks. They show the direction but are not yet statistically strong.
  For now the acc-vs-FLOPs curves and the signal pool carry the >1B scaling
  evidence.

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history. The predictivity ladder regenerates the blocks above.

### Highlighted result

- **The benchmarks that separate data mixtures most: `agieval_sat`, `belebele`, `arabic_leaderboard_alghafa_mcq_exams_test`.** These are the top-3 families by mixture-Signal ((max−min)/mean of per-mix final scores) at 1B.
- **Mixture-Signal ≠ reliability.** These top-Signal families are the ones the above-random gate **removes**: they sit at chance, so they never enter the SNR analysis. Of **118 benchmarks, 44 clear chance at ≥1 size** (74 are random everywhere), almost entirely an answer-count effect.

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

**Above-random gate.** A benchmark must beat chance (`1/n_options`) by +0.05. `run_apertus_snr_variants.py` sets every random `(benchmark, size)` SNR cell to NaN, so the gate propagates to all RQs. The split is almost entirely an answer-count effect:

| options | chance | above ≥1 size | above @1B |
|---|---|---|---|
| 2 | 0.50 | 28 / 42 | 28 / 42 |
| 3 | 0.33 | 7 / 11 | 7 / 11 |
| 4 | 0.25 | 9 / 63 | 7 / 63 |
| 5 | 0.20 | 0 / 2 | 0 / 2 |

[agieval_sat.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/per_benchmark/agieval_sat.png) ·
[acc_vs_flops_signal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/custom_swissai_hf/acc_vs_flops_signal.csv)

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv`: per-task Signal (full
  ranking, all parent tasks, ungated).
- `pretraining/predictivity/above_random_{scores,share,mask,runs,thresholds}.csv`:
  the ladder's gate (`above_random.py`): mean score, share of runs above
  chance, the 1/0/blank mask every later analysis reads, the per-run Wilson bounds, and
  the margin each rule implies. `first_size_above_random.csv`,
  `gate_margin.csv`, `highlights.csv`, `score_curves.csv`,
  `above_random_thresholds.csv` sit next to the panels of the same name.
- `pretraining/predictivity/above_random_external.{png,csv}`: the ladder's
  floors against the public models' (`above_random_external.py`, the former
  `benchmark_floor.*` name is retired).
- `pretraining/seeds_28_1797_1904/`, `pretraining/custom_swissai_hf/`,
  `all/external/`: the 36-sweep's and the public models'
  `above_random_scores.csv` / `above_random_mask.csv` (pure-custom,
  all-models, and non-custom reports, pool-named). `all/external/` is also
  the mask figure 4 and decision accuracy's public-ladder extension read.
- `…/per_benchmark/<family>.png`: top-3 families, subplots per language,
  external scaling markers overlaid (36-sweep pools), and `…/per_language/<lang>.png`
  per language, with the top-3 families as subplots. Redrawn with `CURVES=1` only.
- `pretraining/predictivity_seeds/loss_curves.png`, `benchmark_curves.png`: the
  ladder's curves on the analysis' cells (`curves.py`).
- `pretraining/predictivity/above_random_example.{png,csv}`: the gate on
  one task, an educational reading of the rule (`above_random_example.py
  --task <task>`, its block sits under the setup above).
- `pretraining/predictivity/early_passes.{csv,tex}`, `early_passes_runs.csv`:
  the tasks that pass at very few tokens and fail later, tested for a constant
  answer, the longest option and a related language (`early_passes.py`, section 1b).
- `pretraining/predictivity/gate_pass_counts_all_considered.{csv,tex}`,
  `first_size_share_all_considered_paper.{png,csv}`: the gate over every
  benchmark we considered, the screened and discarded ones at 600M, 1B and
  1.7B (and 3B once the 3B probe results are in the probe gate) beside the
  analysed ones (`panels.py`, section 1a, read from `utils.PROBE_GATE_DIR`
  for the screened tasks), and `gate_screened_verdicts_all_considered.csv`
  (per screened task: `discard_rule`, `fails_600M`, `passes_3B`).
- `majority_chance/`: the majority-class chance sensitivity (`majority_chance.py`, section 1c):
  `majority_baselines.csv` (per task: items, majority baseline, its index, whether the
  answer positions carry fixed labels), the rerun gate under `gate/pretraining/predictivity/`,
  `majority_pass_counts.csv`, `majority_flips.csv`, `majority_by_benchmark.csv`,
  `majority_downstream.csv`, and `majority_chance{,_paper}.{png,csv}`.
- Scripts: `above_random.py` (gate), `run_apertus.py` (accuracy vs FLOPs,
  Signal), `curves.py`, `panels.py`, `above_random_external.py`,
  `above_random_example.py`, `early_passes.py`, `majority_chance.py`.
