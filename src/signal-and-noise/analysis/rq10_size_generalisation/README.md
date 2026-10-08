# Size generalisation — does a ranking that holds at 1.7B still hold one rung above it, at 3B?

## Research question

Every other analysis stops at the reference: a proxy is judged by whether it
makes the decisions the 1.7B model makes (rule 10, `analysis/RULES.md`). The
3B rung asks the one question that frame cannot: whether the reference itself
is a proxy for the next rung.

Seven cells were trained at 3B for it (deep, seed 1904: L ∈ {8, 15, 30, 50} in
scheme A and L ∈ {8, 15, 30} in scheme B; `plan/3b_models.md`); a cell enters
each analysis once its 3B evaluations are in the ladder report. This folder is therefore the only reader of
`build_snr_pool(above_reference=True)` and the only one the rule-10 checker
exempts (`check_rules.EXEMPT`).

Snapshot: the full refresh of **2026-10-07** (commit b316f53b) on the ladder
report of **2026-10-06 04:26**, pool `predictivity` (seed 1904). The 3B
population is the same four families as before the pool change; the bBPB
twins are finals-only in these outputs, so nothing here is claimed about them
at early checkpoints.

The question has two halves, and they answer differently:

- **can the decision be ranked one rung up** — the decision-accuracy half,
  below. In benchmark accuracy, no (DA-size 0.43–0.52); in per-language BPB,
  yes (0.75–0.96, and 0.96 mono-axis from the 1.7B reference).
- **can the benchmark be measured at all one rung up** — the gate half,
  [below](#what-the-3b-rung-can-measure-that-the-17b-reference-cannot). The 3B
  rung lifts 98 of 573 tasks above chance that the reference leaves at chance.

## Setup

- **Population.** Pool `predictivity` (seed 1904); the four families with a
  final at 3B (deep, L8 and L15, schemes A and B), proxies 90M–1.7B. Pairs at
  the grid seed: six multi-axis and four mono-axis (rule 15), above rule 5's
  minimum of three.
- **Lattice.** With 6 and 4 pairs a per-task DA sits on a k/6, k/4 lattice.
  The lines therefore draw the pooled ratio over tasks (matching pairs / all
  pairs), not a mean of per-task values.
- **Gate.** `predictivity`'s mask at the proxy; at the reference rung the same
  one-sided 95 % Wilson rule computed on the reference's own runs
  (`above_random.scores_and_mask`), since the committed mask stops at 1.7B.
- **Two channels, pooled separately.** Benchmark accuracy and per-language
  BPB are never averaged together: DA-size sits near 0.5 on the first and at
  0.75–0.96 on the second, so one mean would report neither. BPB carries no
  chance level and is never gated; only the languages every paired family
  scores clear `MIN_PAIRS`, so the BPB channel rests on 6–7 tasks against the
  benchmark channel's 59–90 at 3B.
- **Comparison.** Panels (a) and (c) read the same four families to 1.7B, so
  the two readings differ in nothing but the reference. The standalone
  preview (`--reference 1.7B --design 3B`) is that comparison line on its own.
- **The bBPB twins at the 1.7B reference.** In this refresh the twins exist
  at finals only and only where the store holds them: none at 3B, but 187
  multi-axis (175 mono-axis) twin tasks at 175M and 1B against the 1.7B final.
  They enter the 1.7B-reference benchmark rows at those two sizes (the
  "same families" column and the preview below); the clean benchmark-only
  numbers are quoted in the Key findings.
- **Known-answer check.** `--reference 1.7B --check` reproduces the
  decision-accuracy tables' `decision_acc_size_<proxy>` per task for both
  pair sets (`predictivity/da_all_per_task_both_axes.csv`). It was exact on
  all 4,490 cells on 2026-10-02 (max |diff| 1.11e-16) and has not been re-run
  on today's pools.

## Key figure

![The benchmarks the 3B rung lifts above chance, and decision accuracy to 3B beside 1.7B](pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png)

Population: pool `predictivity` (seed 1904); the deep families scored at both 1.7B and 3B, each (family, task) scored at both rungs. Left: per benchmark with at least five tasks whose share moves, the share of its tasks the above-random gate admits at each rung (rule 1, recomputed on those families). Right: DA-size from the finals of 90M–1B to the 3B final (solid) and to the 1.7B final (dashed) on the same mono-axis decisions, the task above chance at the proxy, 1.7B and 3B, no filter, ≥ 3 pairs; benchmark accuracy and per-language BPB pooled separately, 90 % leave-one-family-out jackknife bands.

**Key finding.** Pending the regeneration.

GitHub: [gate_share_and_da_size_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png) · [gate_share_and_da_size_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.csv). The full figures: [the gate per benchmark](#the-share-of-each-benchmark-above-chance-at-17b-and-at-3b) and [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b). The previous key figure, DA-size to the 3B final alone, is still written as `above_reference_3B_paper.png` ([the 3B rung as the reference](#the-3b-rung-as-the-reference)).

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Population.** 5 families with a final at 3B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

| axes | proxy | DA-size → 3B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 3B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.53 | 117 | 0.52 | 0.92 | 6 |
| mono-axis | 175M | 0.48 | 115 | 0.54 | 0.92 | 6 |
| mono-axis | 350M | 0.51 | 131 | 0.48 | 0.75 | 6 |
| mono-axis | 600M | 0.51 | 147 | 0.49 | 0.88 | 6 |
| mono-axis | 1B | 0.52 | 157 | 0.53 | 0.92 | 6 |
| mono-axis | 1.7B | 0.47 | 166 | — | 0.96 | 6 |
| multi-axis | 90M | 0.51 | 125 | 0.52 | 0.95 | 7 |
| multi-axis | 175M | 0.49 | 126 | 0.55 | 0.95 | 7 |
| multi-axis | 350M | 0.49 | 142 | 0.49 | 0.82 | 7 |
| multi-axis | 600M | 0.51 | 158 | 0.51 | 0.90 | 7 |
| multi-axis | 1B | 0.53 | 170 | 0.53 | 0.92 | 7 |
| multi-axis | 1.7B | 0.49 | 180 | — | 0.92 | 7 |

Files: [`above_reference_3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png), [`above_reference_3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.csv), [`above_reference_3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_per_task.csv).
<!-- END auto:above-reference-3B -->

Key findings:

- **In benchmark accuracy nothing predicts 3B, but nothing predicts 1.7B
  either.** DA-size → 3B is 0.43–0.52 from every proxy over 59–90 gated tasks,
  the 1.7B reference included (0.43 mono-axis over 83 tasks, 0.44 multi-axis
  over 90), while the same four families read to 1.7B give 0.44–0.48
  mono-axis and 0.44–0.49 multi-axis on benchmark tasks alone: the 3B number
  is this population's chance-level ranking arriving one rung later, not a
  property of 3B.
- **The "same families" column at 175M and 1B is not benchmark-only.** Its
  0.51 / 0.53 (mono-axis) and 0.51 / 0.51 (multi-axis) pool the bBPB twins,
  which exist at those sizes and not at 3B; without them the 1.7B-reference
  benchmark DA-size is 0.48 / 0.48 mono-axis and 0.49 / 0.48 multi-axis.
- **In per-language BPB the reference is a good proxy for the next rung: 0.96
  mono-axis, 0.92 multi-axis.** Every proxy from 90M up reaches at least 0.75 (lowest
  at 350M: 0.75 mono-axis, 0.82 multi-axis).
- **`bpb_macro` agrees on every pair, `train_loss` on all but one.**
  1.7B → 3B, `bpb_macro` matches 4/4 mono-axis and 6/6 multi-axis pairs;
  `train_loss` matches 3/4 and 5/6.
- **Panel (d) says the same per benchmark family** (multi-axis, 1.7B → 3B,
  26 families): `bpb` leads at 0.90 (7 tasks), then `paws` and `loss` at 0.83
  and `hellaswag` at 0.67. 18 of the 26 families sit at or below 0.50.
- **The BPB claim rests on 6–7 tasks**: five languages both an L8 and an L15
  family train (cmn, deu, fas, jpn, rus) plus `bpb_macro` and `bpb_dclm`
  multi-axis, one fewer mono-axis. This is the figure's binding limitation,
  and what A-L30 + B-L30 at 3B would widen.
- **DA-goal along the run (panel b) never rises above chance in benchmark
  accuracy.** Multi-axis, every checkpoint of every proxy reads the 3B final at 0.37–0.55;
  per-language BPB (in `above_reference_3B.csv`, not drawn in panel b) stays
  at 0.77–0.97 multi-axis across checkpoints (0.71–1.00 mono-axis).

Follow-ups:

- Panel (c) pairs per-task DA-size 1.7B → 3B against 1B → 1.7B at Pearson
  r = 0.36 over 89 tasks (multi-axis): the tasks whose ranking converged by
  1.7B are only weakly the ones that keep it at 3B. On a k/4, k/6 lattice
  this is a weak instrument, so quote the pooled lines, not r.
- Add the reference's earlier checkpoints as proxies (DA-ckpt at 3B) once its
  k/10 grid is evaluated; `per_task()` already takes any `frac`.
- The rung has four families, all deep: no depth pair, so the
  by-transformation reading is limited to the language count and the list.

### The preview: the four 3B-design families read to 1.7B

The comparison line of panel (a) on its own — the control that separates "3B
is unpredictable" from "this population is unpredictable at any reference".

<!-- BEGIN auto:above-reference-1.7B-design3B (above_reference.py --pool predictivity --reference 1.7B --design 3B) -->
## The 1.7B rung as the reference — preview on the `3B` design set

**DA-size and DA-goal · reference 1.7B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 1.7B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B`.

**Population.** 7 families with a final at 1.7B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L30-deep-seed1904, lm-L30-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 1.7B](pretraining/predictivity/above_reference_1.7B_design3B.png)

| axes | proxy | DA-size → 1.7B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 1.7B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.49 | 177 | 0.49 | 0.88 | 21 |
| mono-axis | 175M | 0.52 | 702 | 0.52 | 0.88 | 21 |
| mono-axis | 350M | 0.50 | 303 | 0.50 | 0.85 | 21 |
| mono-axis | 600M | 0.49 | 250 | 0.49 | 0.88 | 21 |
| mono-axis | 1B | 0.52 | 773 | 0.52 | 0.76 | 21 |
| multi-axis | 90M | 0.49 | 220 | 0.49 | 0.91 | 31 |
| multi-axis | 175M | 0.52 | 872 | 0.52 | 0.91 | 31 |
| multi-axis | 350M | 0.51 | 474 | 0.51 | 0.87 | 31 |
| multi-axis | 600M | 0.50 | 310 | 0.50 | 0.91 | 31 |
| multi-axis | 1B | 0.53 | 956 | 0.53 | 0.77 | 31 |

Files: [`above_reference_1.7B_design3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.png), [`above_reference_1.7B_design3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.csv), [`above_reference_1.7B_design3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B_per_task.csv).
<!-- END auto:above-reference-1.7B-design3B -->

Key findings:

- **On these four families alone, DA-size to 1.7B is a coin flip from every
  proxy size.** Benchmark tasks only (the bBPB twins left out): mono-axis
  0.48 / 0.48 / 0.46 / 0.44 / 0.48 at 90M / 175M / 350M / 600M / 1B over 53–76
  gated tasks, multi-axis 0.47 / 0.49 / 0.47 / 0.44 / 0.48 over 56–82; this
  is the control for the 3B table above, since the 3B question starts from
  a population whose 1.7B ranking the ladder does not resolve either.
- **The table's 175M and 1B rows (234–269 tasks) pool the bBPB twins.** The
  187 multi-axis (175 mono-axis) twin tasks alone give 0.52 / 0.53 multi-axis
  and 0.52 / 0.56 mono-axis at 175M / 1B, which lifts those rows to 0.51–0.53;
  the twins are finals-only in this refresh, so read nothing into them at
  other sizes or checkpoints.
- **The six multi-axis pairs are two scheme pairs, two L pairs and two that
  change both.** The four mono-axis pairs are the scheme and L pairs alone.
- **BPB on the same families reaches 0.79–0.90 at the 1.7B reference**
  (0.79–0.88 mono-axis, 0.82–0.90 multi-axis), against 0.75–0.96 at 3B: the
  channel that works does not degrade with the rung.
- **The gated task count is 53–82** because the four families train 8 or 15
  languages (rule 2): the 3B answer is about those languages' tasks.

Follow-ups: see above; and consider whether the rung's design set should be
widened (a shallow cell, or L30) before it is read as a generalisation test.

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

- **The 3B rung measures 27 % more of the benchmark suite than the 1.7B
  reference does.** On the four 3B-design families, of the 573 tasks with a
  chance level the gate admits 337 at 1.7B and 429 at 3B (+92 net): 98 cross
  in, 6 drop out, 331 stay above and 138 stay at chance at both rungs.
- **The crossings are concentrated in the multilingual knowledge and
  comprehension benchmarks.** Net gains: `global_mmlu_full` +17, `belebele`
  +14, `cultural_bench_easy` +14, `include_base_44` +13,
  `global_piqa_parallel_cloze` +8.
- **Per language the gain is largest in the high-resource head**: en +16,
  es +12, ar +7, zh +7, 42 of the 92 net. 23 languages gain at least one
  task, and none outside the top five gains more than 5.
- **This is the strongest argument for the 3B rung, and it is independent of
  decision accuracy**: 98 tasks the reference reports as at chance become
  measurable. It is also a caution for every other analysis, whose gate stops
  at 1.7B by rule 10.
- **The 6 losses are small and scattered** (3 `belebele`, 1 each `bbh_mcq`
  and `include_base_44` in the table above, one more outside it). With 4 runs
  per rung the Wilson rule can plausibly move that many by chance.

Follow-ups:

- The two columns rest on 4 runs at 3B against 4 at 1.7B. More 3B cells would
  tighten the gate estimate directly; this, not the decision-accuracy table,
  is where A-L30 + B-L30 would pay.
- Ask in the gate analysis (`../rq00_gate_and_curves/README.md`) whether the
  newly admitted tasks' DA is any better than the already-admitted ones', i.e.
  whether measurability buys reliability.

Key findings (the per-benchmark shares, `gate_crossover_by_benchmark.png`):

- **Benchmarks the rung lifts the most:** pending the regeneration
  (`gate_crossover_by_benchmark.csv`, `delta`).
- **Benchmarks with no task above chance at 1.7B and at least one at 3B:**
  pending the regeneration.
- **Benchmarks still at chance at 3B on every task:** pending the
  regeneration.

Follow-ups:

- Read the newly passing tasks' run counts (`gate_crossover_per_task.csv`,
  `runs_<size>`): a task only the L50 family trains rests on one run per rung.

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
| mono-axis | 90M | 101 | 0.52 [0.44, 0.60] | 0.51 [0.46, 0.56] |
| mono-axis | 175M | 114 | 0.52 [0.44, 0.60] | 0.47 [0.38, 0.56] |
| mono-axis | 350M | 129 | 0.48 [0.43, 0.53] | 0.50 [0.36, 0.64] |
| mono-axis | 600M | 145 | 0.48 [0.36, 0.61] | 0.51 [0.47, 0.55] |
| mono-axis | 1B | 154 | 0.52 [0.43, 0.61] | 0.52 [0.46, 0.59] |
| multi-axis | 90M | 109 | 0.51 [0.44, 0.58] | 0.50 [0.44, 0.56] |
| multi-axis | 175M | 125 | 0.52 [0.46, 0.57] | 0.48 [0.43, 0.53] |
| multi-axis | 350M | 140 | 0.49 [0.45, 0.52] | 0.49 [0.39, 0.59] |
| multi-axis | 600M | 156 | 0.51 [0.35, 0.66] | 0.51 [0.48, 0.54] |
| multi-axis | 1B | 167 | 0.53 [0.43, 0.62] | 0.52 [0.47, 0.58] |

Spearman between the two references' DA, over the benchmark tasks and over the benchmarks (mean DA of their tasks):

| axes | proxy | ρ tasks | tasks | ρ benchmarks | benchmarks |
|---|---|---|---|---|---|
| mono-axis | 90M | 0.11 | 101 | 0.14 | 20 |
| mono-axis | 175M | 0.09 | 114 | 0.03 | 22 |
| mono-axis | 350M | 0.08 | 129 | -0.05 | 22 |
| mono-axis | 600M | 0.02 | 145 | 0.43 | 23 |
| mono-axis | 1B | 0.23 | 154 | 0.35 | 23 |
| multi-axis | 90M | 0.19 | 109 | 0.16 | 21 |
| multi-axis | 175M | 0.09 | 125 | -0.07 | 22 |
| multi-axis | 350M | 0.10 | 140 | 0.02 | 22 |
| multi-axis | 600M | 0.01 | 156 | 0.56 | 23 |
| multi-axis | 1B | 0.16 | 167 | 0.32 | 23 |

![SNR at two rungs](pretraining/predictivity/reference_consistency_snr.png)

SNR (`rel_std`) at 1.7B against 3B, tasks above chance at both:

| channel | ρ tasks | tasks | ρ benchmarks | benchmarks | median log10 SNR 1.7B | median log10 SNR 3B | share higher at 3B |
|---|---|---|---|---|---|---|---|
| benchmarks | 0.32 | 300 | 0.39 | 24 | -0.03 | -0.16 | 0.35 |
| bpb | 0.94 | 12 | — | 1 | -0.51 | -0.76 | 0.00 |

Files: [`reference_consistency_da_size_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.csv), [`reference_consistency_da_size_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.csv), [`reference_consistency_da_size_per_task_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_per_task_both_axes.csv), [`reference_consistency_snr.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.csv), [`reference_consistency_snr_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr_per_task.csv).
<!-- END auto:reference-consistency -->

Key findings:

- **DA-size to 3B against DA-size to 1.7B on the same decisions:** pending
  the regeneration (benchmark accuracy and BPB, both pair sets).
- **Does the ranking of benchmarks by DA survive the reference:** pending the
  regeneration (Spearman over tasks and over benchmarks, per proxy).
- **SNR at 3B against 1.7B:** pending the regeneration (Spearman over tasks
  and over benchmarks, the median log10 SNR at each rung).

Follow-ups:

- With few families at 3B the per-task DA sits on a coarse lattice and many
  cells fall under `MIN_PAIRS`; read the pooled lines and the benchmark-level
  ρ before the per-task scatter.

## The DA panel on one task set per line

The DA lines pool, at each proxy size, the tasks above chance at the proxy and at both references, so they can move because their tasks changed (rule 13). The twin redraws the DA panel only (the gate panel is a share, not a mean over tasks), one panel per reference: per channel, the tasks with a value at every proxy size (solid), the committed line dashed behind. A cell is kept for both references at once, so one task set serves both.

![DA to 1.7B and to 3B on one task set per line](pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.png)

DA-size (no filter, mono-axis pairs, gate `predictivity` at the proxy, the references' own gate at 1.7B and 3B), pooled matching over comparable pairs. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.csv)

Key findings:

- **The fixed task set changes nothing that matters.** The benchmark lines keep 100 of 101–154 tasks and move by 0.03 at most (to 1.7B, at 600M); against 3B they read 0.47–0.52 on both task sets.
- **BPB does not move.** The same 6 languages at every size.

Follow-ups:

- **The jackknife band on the fixed set.** The per-task table has no family pairs, so the twin draws no band.

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
