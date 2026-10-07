# Size generalisation — does a ranking that holds at 1.7B still hold one rung above it, at 3B?

## Research question

Every other analysis stops at the reference: a proxy is judged by whether it
makes the decisions the 1.7B model makes (rule 10, `analysis/RULES.md`). The
3B rung asks the one question that frame cannot: whether the reference itself
is a proxy for the next rung.

Seven cells are planned at 3B for it (deep, seed 1904: L ∈ {8, 15, 30, 50} in
scheme A and L ∈ {8, 15, 30} in scheme B; `plan/3b_models.md`), and a cell
enters each analysis once its 3B final is evaluated in the ladder report. This
folder is therefore the only reader of `build_snr_pool(above_reference=True)`
and the only one the rule-10 checker exempts (`check_rules.EXEMPT`).

Snapshot: the ladder report of **2026-10-07 15:51**, regenerated the same day
(branch `feat/rq_figures`, commit 32474921), pool `predictivity` (seed 1904).
Six 3B cells are in the report and five are scored at both 1.7B and 3B (A-L8,
A-L15, A-L50, B-L8, B-L15): A-L30's last benchmark evaluation is at iteration
137,940 of 145,200, so its final has not landed, and B-L30 is not trained yet.

The rung is asked two things:

- **does it lift more benchmarks above chance** — the gate,
  [below](#what-the-3b-rung-can-measure-that-the-17b-reference-cannot).
- **do the decision-accuracy and SNR readings stay the same when 3B replaces
  1.7B as the reference** — [the 3B rung as the reference](#the-3b-rung-as-the-reference)
  and [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b).

## Setup

- **Population.** Pool `predictivity` (seed 1904); the five families with a
  final at 3B (deep: A-L8, A-L15, A-L50, B-L8, B-L15), proxies 90M–1.7B. Pairs
  at the grid seed: ten multi-axis and six mono-axis (rule 15), fewer on a task
  whose language not every family trains (rule 2), and a cell needs three
  (rule 5).
- **Lattice.** With 10 and 6 pairs a per-task DA sits on a k/10, k/6 lattice.
  The lines therefore draw the pooled ratio over tasks (matching pairs / all
  pairs), not a mean of per-task values.
- **Gate.** `predictivity`'s mask at the proxy; at the reference rung the same
  one-sided 95 % Wilson rule computed on the reference's own runs
  (`above_random.scores_and_mask`), since the committed mask stops at 1.7B.
- **Two channels, pooled separately.** Benchmark accuracy and per-language
  BPB are never averaged together: DA-size sits near 0.5 on the first and at
  0.75–0.96 on the second, so one mean would report neither. BPB carries no
  chance level and is never gated, and the A-L50 3B cell carries no BPB score,
  so the BPB channel rests on the four L8/L15 families and 6–7 tasks against
  the benchmark channel's 115–180 at 3B.
- **Comparison.** Panels (a) and (c) read the same five families to 1.7B, so
  the two readings differ in the reference and in the bBPB twins (next
  bullet). The reference-consistency section reads both references on
  identical decisions without the twins; the preview (`--reference 1.7B
  --design 3B`) reads the whole 3B design set, A-L30 and B-L30 included, to
  1.7B.
- **The bBPB twins at the 1.7B reference.** The per-item store now holds every
  checkpoint of every seed-1904 cell up to 1.7B and no 3B cell, so the twins
  enter every 1.7B-reference benchmark row and no 3B-reference row. The
  "same families" column and the preview table below therefore pool them; the
  benchmark-only numbers are quoted in the Key findings.
- **Known-answer check.** `--reference 1.7B --check` reproduces the
  decision-accuracy tables' `decision_acc_size_<proxy>` per task for both
  pair sets (`predictivity/da_all_per_task_both_axes.csv`). It was exact on
  all 4,490 cells on 2026-10-02 (max |diff| 1.11e-16) and has not been re-run
  on today's pools.

## Key figure

![The benchmarks the 3B rung lifts above chance, and decision accuracy to 3B beside 1.7B](pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png)

Population: pool `predictivity` (seed 1904); the five deep families scored at both 1.7B and 3B (A-L8, A-L15, A-L50, B-L8, B-L15), each (family, task) scored at both rungs. Left: per benchmark with at least five tasks whose share moves, the share of its tasks the above-random gate admits at each rung (rule 1, recomputed on those families). Right: DA-size from the finals of 90M–1B to the 3B final (solid) and to the 1.7B final (dashed) on the same mono-axis decisions, the task above chance at the proxy, 1.7B and 3B, no filter, ≥ 3 pairs; benchmark accuracy and per-language BPB pooled separately, 90 % leave-one-family-out jackknife bands.

**Key finding.** The 3B rung measures more than the reference but decides no
differently: on the same five families it lifts the above-chance count from
510 to 674 of 841 tasks (+32 %), while mono-axis DA-size on identical
decisions stays at 0.47–0.52 in benchmark accuracy to either reference and at
0.75–0.92 in per-language BPB.

GitHub: [gate_share_and_da_size_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png) · [gate_share_and_da_size_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.csv). The full figures: [the gate per benchmark](#the-share-of-each-benchmark-above-chance-at-17b-and-at-3b) and [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b). The previous key figure, DA-size to the 3B final alone, is still written as `above_reference_3B_paper.png` ([the 3B rung as the reference](#the-3b-rung-as-the-reference)).

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Population.** 5 families with a final at 3B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

| axes | proxy | DA-size → 3B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 3B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.52 | 116 | 0.53 | 0.92 | 6 |
| mono-axis | 175M | 0.48 | 115 | 0.54 | 0.92 | 6 |
| mono-axis | 350M | 0.51 | 131 | 0.52 | 0.75 | 6 |
| mono-axis | 600M | 0.51 | 149 | 0.53 | 0.88 | 6 |
| mono-axis | 1B | 0.52 | 157 | 0.53 | 0.92 | 6 |
| mono-axis | 1.7B | 0.47 | 166 | — | 0.96 | 6 |
| multi-axis | 90M | 0.51 | 125 | 0.53 | 0.95 | 7 |
| multi-axis | 175M | 0.49 | 126 | 0.55 | 0.95 | 7 |
| multi-axis | 350M | 0.49 | 142 | 0.54 | 0.82 | 7 |
| multi-axis | 600M | 0.51 | 160 | 0.55 | 0.90 | 7 |
| multi-axis | 1B | 0.53 | 170 | 0.53 | 0.92 | 7 |
| multi-axis | 1.7B | 0.49 | 180 | — | 0.92 | 7 |

Files: [`above_reference_3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png), [`above_reference_3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.csv), [`above_reference_3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_per_task.csv).
<!-- END auto:above-reference-3B -->

Key findings:

- **In benchmark accuracy no proxy predicts the 3B decision, the 1.7B
  reference included.** On the five families' gated tasks, DA-size → 3B is
  0.47–0.52 mono-axis over 115–166 tasks and 0.49–0.53 multi-axis over
  125–180; the 1.7B rung is the lowest of its column (0.47 mono-axis, 0.49
  multi-axis), so more model is not a better proxy of the next rung here.
- **The "same families" column is not benchmark-only:** its 0.52–0.54
  mono-axis and 0.53–0.55 multi-axis pool the bBPB twins, which exist at 1.7B
  and not at 3B (hence panel (a)'s dashed lines on a median of 423 and 456
  tasks against 140 and 151 to 3B). The benchmark-only comparison on
  identical decisions is [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b):
  0.48–0.53 to 1.7B against 0.47–0.52 to 3B.
- **In per-language BPB the reference is a good proxy for the next rung: 0.96
  mono-axis, 0.92 multi-axis.** Every proxy from 90M up reaches at least 0.75
  (lowest at 350M: 0.75 mono-axis, 0.82 multi-axis), over the four L8/L15
  families.
- **`bpb_macro` agrees on every pair, `train_loss` on most.** From 1.7B to 3B,
  `bpb_macro` matches 4/4 mono-axis and 6/6 multi-axis pairs; `train_loss`,
  which all five families carry, matches 4/6 and 8/10.
- **Panel (d) says the same per benchmark family** (multi-axis, 1.7B → 3B, 26
  families): `bpb` leads at 0.90 (7 tasks), then `loss` at 0.80,
  `lambada_openai_mt` at 0.71 (5) and `arc_mt` at 0.65 (4). 14 of the 26
  families sit at or below 0.50.
- **The BPB claim rests on 6–7 tasks**: four languages every L8 and L15 family
  trains (cmn, deu, jpn, rus), `bpb_dclm` and `bpb_macro`, plus fas multi-axis.
  This is the figure's binding limitation, and a BPB score for the L30 and L50
  cells at 3B is what would widen it.
- **DA-goal along the run (panel b) never leaves chance in benchmark
  accuracy.** Every checkpoint before the final of every proxy reads the 3B
  final at 0.42–0.55 multi-axis over 125–180 tasks (0.39–0.55 mono-axis);
  per-language BPB (in `above_reference_3B.csv`, not drawn in panel b) stays
  at 0.77–0.97 multi-axis (0.71–1.00 mono-axis).

Follow-ups:

- Panel (c) pairs per-task DA-size 1.7B → 3B against 1B → 1.7B at Pearson
  r = 0.20 over 175 tasks (multi-axis): the tasks whose ranking converged by
  1.7B are barely the ones that keep it at 3B. On a k/10 lattice this is a
  weak instrument, so quote the pooled lines, not r.
- Score the 3B cells in the per-item store so the "same families" column
  compares like with like; until then read the reference-consistency table.
- Add the reference's earlier checkpoints as proxies (DA-ckpt at 3B) once its
  k/10 grid is evaluated; `per_task()` already takes any `frac`.
- The rung's families are all deep: no depth pair, so the by-transformation
  reading is limited to the language count and the scheme.

### The preview: the 3B design set read to 1.7B

The control that separates "3B is unpredictable" from "this population is
unpredictable at any reference". It reads all seven 3B-design families at
1.7B, where A-L30 and B-L30 already have finals, so its population is wider
than the 3B table's five.

<!-- BEGIN auto:above-reference-1.7B-design3B (above_reference.py --pool predictivity --reference 1.7B --design 3B) -->
## The 1.7B rung as the reference — preview on the `3B` design set

**DA-size and DA-goal · reference 1.7B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 1.7B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B`.

**Population.** 7 families with a final at 1.7B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L30-deep-seed1904, lm-L30-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 1.7B](pretraining/predictivity/above_reference_1.7B_design3B.png)

| axes | proxy | DA-size → 1.7B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 1.7B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.53 | 681 | 0.53 | 0.88 | 21 |
| mono-axis | 175M | 0.52 | 700 | 0.52 | 0.88 | 21 |
| mono-axis | 350M | 0.51 | 726 | 0.51 | 0.85 | 21 |
| mono-axis | 600M | 0.52 | 752 | 0.52 | 0.88 | 21 |
| mono-axis | 1B | 0.52 | 772 | 0.52 | 0.76 | 21 |
| multi-axis | 90M | 0.53 | 842 | 0.53 | 0.91 | 31 |
| multi-axis | 175M | 0.52 | 870 | 0.52 | 0.91 | 31 |
| multi-axis | 350M | 0.52 | 900 | 0.52 | 0.87 | 31 |
| multi-axis | 600M | 0.53 | 932 | 0.53 | 0.91 | 31 |
| multi-axis | 1B | 0.53 | 955 | 0.53 | 0.77 | 31 |

Files: [`above_reference_1.7B_design3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.png), [`above_reference_1.7B_design3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.csv), [`above_reference_1.7B_design3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B_per_task.csv).
<!-- END auto:above-reference-1.7B-design3B -->

Key findings:

- **On the seven design families, DA-size to 1.7B is a coin flip from every
  proxy size.** Benchmark tasks only (the bBPB twins left out): 0.49 / 0.51 /
  0.48 / 0.49 / 0.50 mono-axis at 90M / 175M / 350M / 600M / 1B over 179–270
  gated tasks, and 0.49 / 0.50 / 0.49 / 0.50 / 0.52 multi-axis over 223–336;
  the 3B question therefore starts from a population whose 1.7B ranking the
  ladder does not resolve either.
- **Every row of the table pools the bBPB twins.** The 502 mono-axis (619
  multi-axis) twin tasks alone give 0.53–0.54 (0.53–0.55) at every proxy size,
  which lifts the rows to 0.51–0.53 over 681–955 tasks.
- **The pair sets are larger than at 3B.** Seven families give 21 multi-axis
  and 12 mono-axis pairs against 10 and 6 for the five at 3B, so the two
  tables differ in population as well as in reference.
- **BPB on these families reaches 0.76–0.91 at the 1.7B reference**
  (0.76–0.88 mono-axis over 21 tasks, 0.77–0.91 multi-axis over 31), against
  0.75–0.96 to 3B over 6–7: the channel that works does not degrade with the
  rung.

Follow-ups: see above; and consider whether the rung's design set should be
widened (a shallow cell) before it is read as a generalisation test.

### The prior question: what the rung can measure at all

DA asks whether a ranking survives, which it can only ask of a benchmark
above chance at both rungs. The gate is a property of (task, size), so a
benchmark the reference cannot resolve may become measurable one rung up —
and the reference's own mask can never say so.

<!-- BEGIN auto:gate-crossover (gate_crossover.py --pool predictivity) -->
## What the 3B rung can measure that the 1.7B reference cannot

**The above-random gate at two rungs · the 5 families with a 3B final · both columns recomputed on those families (rule 1) so the rung is the only difference.** Regenerate with `python analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity`.

**Population.** 841 tasks carry a chance level at both rungs; BPB and the generative tasks have none, so they never enter the gate. The gate admits **510** of them at 1.7B and **674** at 3B (**+164**, +32%).

![Gate crossover](pretraining/predictivity/gate_crossover.png)

|  | at chance at 3B | above chance at 3B |
|---|---|---|
| **above chance at 1.7B** | 6 | 504 |
| **at chance at 1.7B** | 161 | 170 |

**170 tasks cross into the gate at 3B** and **6** drop out of it. Where they come from:

| benchmark | gained | lost | net |
|---|---|---|---|
| belebele | 40 | 0 | +40 |
| include_base_44 | 30 | 0 | +30 |
| global_mmlu_full | 28 | 0 | +28 |
| cultural_bench_easy | 16 | 0 | +16 |
| global_piqa_parallel_cloze | 11 | 1 | +10 |
| include_v2_og | 9 | 2 | +7 |
| arc | 5 | 0 | +5 |
| blend_sample | 5 | 0 | +5 |
| rf_include_base_44 | 5 | 0 | +5 |
| include_v2_en | 4 | 1 | +3 |
| rfgm_include_base_44 | 4 | 0 | +4 |
| cultural_bench_hard | 3 | 0 | +3 |

Files: [`gate_crossover.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.png), [`gate_crossover.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.csv), [`gate_crossover_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_per_task.csv) (every task's run shares and verdicts, `status` = `newly above` for the tasks that cross in).

### The share of each benchmark above chance at 1.7B and at 3B

Per benchmark, the share of its tasks the gate admits at each rung, on the same (family, task) cells; 40 benchmarks, 17 gain, 0 lose, 3 have no task above chance at 1.7B and at least one at 3B.

![Gate share by benchmark](pretraining/predictivity/gate_crossover_by_benchmark.png)

| benchmark | tasks | above at 1.7B | above at 3B | share change |
|---|---|---|---|---|
| blend_sample | 5 | 0 (0%) | 5 (100%) | +100% |
| commonsense_qa | 1 | 0 (0%) | 1 (100%) | +100% |
| mmlu | 1 | 0 (0%) | 1 (100%) | +100% |
| global_mmlu_full | 29 | 1 (3%) | 29 (100%) | +97% |
| cultural_bench_easy | 19 | 1 (5%) | 17 (89%) | +84% |
| include_base_44 | 36 | 6 (17%) | 36 (100%) | +83% |
| belebele | 59 | 17 (29%) | 57 (97%) | +68% |
| paws | 8 | 5 (62%) | 7 (88%) | +25% |
| arc | 28 | 17 (61%) | 22 (79%) | +18% |
| global_piqa_parallel_cloze | 63 | 1 (2%) | 11 (17%) | +16% |
| rf_cultural_bench_easy | 19 | 7 (37%) | 10 (53%) | +16% |
| cultural_bench_hard | 19 | 1 (5%) | 4 (21%) | +16% |
| rf_include_base_44 | 36 | 27 (75%) | 32 (89%) | +14% |
| rfgm_include_base_44 | 36 | 30 (83%) | 34 (94%) | +11% |
| include_v2_og | 77 | 57 (74%) | 64 (83%) | +9% |
| include_v2_en | 77 | 66 (86%) | 69 (90%) | +4% |
| hellaswag | 26 | 25 (96%) | 26 (100%) | +4% |

Files: [`gate_crossover_by_benchmark.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_by_benchmark.png), [`gate_crossover_by_benchmark.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_by_benchmark.csv).
<!-- END auto:gate-crossover -->

Key findings:

- **Yes, the 3B rung lifts more benchmarks above chance: it measures 32 % more
  of the suite than the 1.7B reference.** On the five families scored at both
  rungs, of the 841 tasks with a chance level the gate admits 510 at 1.7B and
  674 at 3B (+164 net): 170 cross in, 6 drop out, 504 stay above and 161 stay
  at chance at both rungs.
- **The crossings are concentrated in the multilingual knowledge and
  comprehension benchmarks.** Net gains: `belebele` +40, `include_base_44`
  +30, `global_mmlu_full` +28, `cultural_bench_easy` +16,
  `global_piqa_parallel_cloze` +10, together 124 of the 164.
- **Per language the gain is largest in the high-resource head**: en +16,
  es +13, zh +7, pt +6, ar +6, 48 of the 164 net. 48 languages gain at least
  one task net, and none outside the top five gains more than 5.
- **This is the strongest argument for the 3B rung, and it is independent of
  decision accuracy**: 170 tasks the reference reports as at chance become
  measurable. It is also a caution for every other analysis, whose gate stops
  at 1.7B by rule 10.
- **The 6 losses are small and thin**: `bbh_mcq_disambiguation_qa` (English,
  5 runs per rung) and five tasks on 2–3 runs per rung (`include_v2_en` and
  `include_v2_og` Arabic-Kuwait, `include_v2_og` Greek-Cyprus,
  `global_piqa_parallel_cloze` Polish, `rf_belebele` arb_Latn). On two runs
  the Wilson rule can move that many by chance.

Follow-ups:

- The gate rests on 1–5 runs per (task, rung), and 58 of the 170 newly
  admitted tasks on a single run (a language only the L50 family trains).
  More 3B cells would tighten the gate directly; this, not the
  decision-accuracy table, is where A-L30's final and B-L30 would pay.
- Ask in the gate analysis (`../rq00_gate_and_curves/README.md`) whether the
  newly admitted tasks' DA is any better than the already-admitted ones', i.e.
  whether measurability buys reliability.

Key findings (the per-benchmark shares, `gate_crossover_by_benchmark.png`):

- **The rung lifts the multiple-choice knowledge benchmarks the most.**
  `global_mmlu_full` goes from 3 % to 100 % of its 29 tasks, `include_base_44`
  from 17 % to 100 % of 36, `cultural_bench_easy` from 5 % to 89 % of 19 and
  `belebele` from 29 % to 97 % of 59; of the 40 benchmarks, 17 gain share and
  none loses it (`rf_belebele` and `bbh_mcq` trade one task each way).
- **Their reformulated twins were already measurable at 1.7B.**
  `rf_global_mmlu_full` and `rfgm_belebele` sit at 100 % and `rf_belebele` at
  98 % at both rungs, so what 3B adds is the original format, not the
  knowledge.
- **Three benchmarks have no task above chance at 1.7B and at least one at
  3B:** `blend_sample` (0 → 5 of 5), `commonsense_qa` and `mmlu` (0 → 1 of 1).
- **Six benchmarks stay at chance on every task at 3B:** `acp_bench_cloze`
  (7 tasks), `acp_bench_mcq` (7), `bbh_cloze` (6), `truthfulqa_mc2` (3),
  `truthfulqa-multi_mc1` (2) and `toxigen` (1). `bbh_mcq` admits 1 of 17 and
  `global_piqa_parallel_cloze` 11 of 63 at 3B, so neither is measurable on
  this ladder yet.

Follow-ups:

- Draw the per-benchmark shares with each task's run count
  (`gate_crossover_per_task.csv`, `runs_<size>`), so a share resting on one
  run per rung reads as such.

### The framework at the new reference: does DA and SNR read the same?

The gate says what the rung can measure; the question left open is whether
the DA/SNR framework built on the 1.7B reference reads the same at 3B. This
compares DA-size to the two references on the same decisions, the ranking of
benchmarks by DA and by SNR under each, and the SNR at the two rungs.

<!-- BEGIN auto:reference-consistency (reference_consistency.py --pool predictivity) -->
## Is the framework consistent when the reference moves from 1.7B to 3B?

**DA-size to two references and the SNR at two rungs · 5 families scored at both (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904) · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy and at 1.7B, the Wilson rule on the 3B runs · no filter.** Regenerate with `python analysis/rq10_size_generalisation/reference_consistency.py --pool predictivity`.

**Population.** 10 multi-axis and 6 mono-axis pairs over the 5 families (fewer on a task whose language not every family trains, rule 2). A DA cell is a (task, pair set, proxy) whose decisions both references score, the task above chance at the proxy, at 1.7B and at 3B; 2045 cells over 266 tasks have fewer than 3 pairs and are NaN (rule 5, pair counts in the per-task table). Bands are the 90 % leave-one-family-out jackknife.

![DA-size to two references, multi-axis](pretraining/predictivity/reference_consistency_da_size_multi_axes.png)

![DA-size to two references, mono-axis](pretraining/predictivity/reference_consistency_da_size_mono_axis.png)

Benchmark accuracy, pooled DA-size [90 % band]:

| axes | proxy | tasks | → 1.7B | → 3B |
|---|---|---|---|---|
| mono-axis | 90M | 102 | 0.52 [0.44, 0.60] | 0.51 [0.46, 0.56] |
| mono-axis | 175M | 114 | 0.52 [0.44, 0.60] | 0.47 [0.38, 0.56] |
| mono-axis | 350M | 129 | 0.48 [0.43, 0.53] | 0.50 [0.36, 0.64] |
| mono-axis | 600M | 147 | 0.48 [0.35, 0.61] | 0.51 [0.48, 0.55] |
| mono-axis | 1B | 154 | 0.52 [0.43, 0.61] | 0.52 [0.46, 0.59] |
| multi-axis | 90M | 111 | 0.51 [0.43, 0.59] | 0.50 [0.44, 0.56] |
| multi-axis | 175M | 125 | 0.52 [0.46, 0.57] | 0.48 [0.43, 0.53] |
| multi-axis | 350M | 140 | 0.49 [0.45, 0.52] | 0.49 [0.39, 0.59] |
| multi-axis | 600M | 158 | 0.50 [0.35, 0.66] | 0.51 [0.48, 0.53] |
| multi-axis | 1B | 167 | 0.53 [0.43, 0.62] | 0.52 [0.47, 0.58] |

Spearman between the two references' DA, over the benchmark tasks and over the benchmarks (mean DA of their tasks):

| axes | proxy | ρ tasks | tasks | ρ benchmarks | benchmarks |
|---|---|---|---|---|---|
| mono-axis | 90M | 0.11 | 102 | 0.06 | 21 |
| mono-axis | 175M | 0.09 | 114 | 0.03 | 22 |
| mono-axis | 350M | 0.08 | 129 | -0.05 | 22 |
| mono-axis | 600M | 0.01 | 147 | 0.27 | 23 |
| mono-axis | 1B | 0.23 | 154 | 0.35 | 23 |
| multi-axis | 90M | 0.18 | 111 | 0.11 | 21 |
| multi-axis | 175M | 0.09 | 125 | -0.07 | 22 |
| multi-axis | 350M | 0.10 | 140 | 0.02 | 22 |
| multi-axis | 600M | -0.00 | 158 | 0.38 | 23 |
| multi-axis | 1B | 0.16 | 167 | 0.32 | 23 |

![SNR at two rungs](pretraining/predictivity/reference_consistency_snr.png)

SNR (`rel_std`) at 1.7B against 3B, tasks above chance at both:

| channel | ρ tasks | tasks | ρ benchmarks | benchmarks | median log10 SNR 1.7B | median log10 SNR 3B | share higher at 3B |
|---|---|---|---|---|---|---|---|
| benchmarks | 0.35 | 300 | 0.45 | 24 | -0.02 | -0.16 | 0.33 |
| bpb | 0.94 | 12 | — | 1 | -0.58 | -0.82 | 0.00 |

Files: [`reference_consistency_da_size_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.csv), [`reference_consistency_da_size_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.csv), [`reference_consistency_da_size_per_task_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_per_task_both_axes.csv), [`reference_consistency_snr.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.csv), [`reference_consistency_snr_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr_per_task.csv).
<!-- END auto:reference-consistency -->

Key findings:

- **Pooled DA-size does not depend on the reference.** On identical
  decisions (five families, benchmark tasks above chance at the proxy, 1.7B
  and 3B), mono-axis DA-size is 0.48–0.52 to 1.7B and 0.47–0.52 to 3B over
  102–154 tasks, multi-axis 0.49–0.53 and 0.48–0.52 over 111–167; every 90 %
  band contains 0.50.
- **In per-language BPB both references are well predicted.** DA-size is
  0.79–0.88 to 1.7B and 0.75–0.92 to 3B mono-axis (6 tasks), 0.82–0.90 and
  0.82–0.95 multi-axis (7 tasks, four families).
- **Which tasks a proxy ranks well does not survive the move.** Spearman ρ
  between the two references' per-task DA is −0.00 to 0.23 over 102–167 tasks,
  and only 17–25 % of the benchmark cells give the same DA to both (mean
  |difference| 0.26–0.31).
- **The ranking of benchmarks by DA survives weakly, and only from the larger
  proxies.** Over 21–23 benchmarks ρ is −0.07 to 0.11 from 90M–350M and
  0.27–0.38 from 600M and 1B, both pair sets.
- **The SNR ranking survives better than the DA ranking.** Over 300 benchmark
  tasks above chance at both rungs ρ is 0.35 (0.45 over 24 benchmarks), and
  over 12 BPB tasks 0.94.
- **SNR is lower at 3B.** Median log10 SNR drops from −0.02 to −0.16 on the
  benchmark tasks, with 33 % of them higher at 3B, and from −0.58 to −0.82 on
  BPB, where none of the 12 is higher; the largest benchmark drops are
  `arc_mt` (median −0.55 over 6 tasks), `arc` (−0.39, 12) and `hellaswag`
  (−0.30, 13), and of the benchmarks with more than one task only
  `multiblimp` (+0.12, 17) and `rf_bbh_mcq` (+0.08, 10) rise.

Follow-ups:

- With five families at 3B the per-task DA sits on a k/10, k/6 lattice and
  2,045 cells over 266 tasks fall under `MIN_PAIRS`; read the pooled lines
  and the benchmark-level ρ before the per-task scatter.
- Split the SNR drop into its signal (spread of the five families) and noise
  (checkpoint window) halves, to say whether the designs converge at 3B or
  the late checkpoints get noisier.

## Extensions from other sweeps

None: the 36-model sweep has no rung above its 1B reference, and the public
models' size steps (the decision-accuracy README, "Extensions") are between-lab decisions, not this
ladder's.

## Files

- `above_reference.py` — the DA half; `--reference` picks the rung,
  `--design 3B` restricts to the rung's design set, `--check` compares with
  the decision-accuracy tables, `--out-dir` writes elsewhere (the check).
- `gate_crossover.py` — the gate half; `--reference` picks the rung read
  above the 1.7B reference.
- `reference_consistency.py` — the framework at the two references: DA-size
  to 3B against 1.7B on the same decisions, the DA and SNR rankings of the
  benchmarks under each, the SNR at both rungs; and the key figure
  (`--paper` redraws it from the CSVs on disk).
- `pretraining/predictivity/above_reference_<ref>[_design<d>].{png,csv}`,
  `..._per_task.csv`, `gate_crossover{,_by_benchmark}.{png,csv}`,
  `gate_crossover_per_task.csv`,
  `reference_consistency_da_size_{multi_axes,mono_axis}.{png,csv}`,
  `reference_consistency_da_size_per_task_both_axes.csv`,
  `reference_consistency_snr.{png,csv}`, `reference_consistency_snr_per_task.csv`,
  `gate_share_and_da_size_mono_axis_paper.{png,svg,csv}`.
