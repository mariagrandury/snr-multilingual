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

Snapshot: the ladder report of **2026-10-08 12:06** (regenerated locally:
acc_norm on the cloze-format originals), outputs of the full refresh of
2026-10-08 (detrended checkpoint noise), pool `predictivity` (seed 1904).
Six 3B cells are in the report and all six are scored at both 1.7B and 3B
(A-L8, A-L15, A-L30, A-L50, B-L8, B-L15): A-L30's benchmark final has landed
but its last BPB evaluation is at iteration 123,420 of 145,200, and B-L30 is
not in the report yet.

The rung is asked two things:

- **does it lift more benchmarks above chance** — the gate,
  [below](#what-the-3b-rung-can-measure-that-the-17b-reference-cannot).
- **do the decision-accuracy and SNR readings stay the same when 3B replaces
  1.7B as the reference** — [the 3B rung as the reference](#the-3b-rung-as-the-reference)
  and [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b).

## Setup

- **Population.** Pool `predictivity` (seed 1904); the six families with a
  final at 3B (deep: A-L8, A-L15, A-L30, A-L50, B-L8, B-L15), proxies 90M–1.7B. Pairs
  at the grid seed: fifteen multi-axis and nine mono-axis (rule 15), fewer on a task
  whose language not every family trains (rule 2), and a cell needs three
  (rule 5).
- **Lattice.** With 15 and 9 pairs a per-task DA sits on a k/15, k/9 lattice.
  The lines therefore draw the pooled ratio over tasks (matching pairs / all
  pairs), not a mean of per-task values.
- **Gate.** `predictivity`'s mask at the proxy; at the reference rung the same
  one-sided 95 % Wilson rule computed on the reference's own runs
  (`above_random.scores_and_mask`), since the committed mask stops at 1.7B.
- **Two channels, pooled separately.** Benchmark accuracy and per-language
  BPB are never averaged together: DA-size sits near 0.5 on the first and at
  0.81–0.96 on the second, so one mean would report neither. BPB carries no
  chance level and is never gated, and the A-L30 3B cell carries no BPB final,
  so the BPB channel rests on the five other families and 10–12 tasks against
  the benchmark channel's 457–618 at 3B.
- **Comparison.** Panels (a) and (c) read the same six families to 1.7B, so
  the two readings differ in the reference and in the bBPB twins' coverage
  (next bullet). The reference-consistency section reads both references on
  identical decisions, twins included; the preview (`--reference 1.7B
  --design 3B`) reads the whole 3B design set, B-L30 included, to
  1.7B.
- **The bBPB twins at both references.** The per-item store now holds every
  checkpoint of every seed-1904 cell up to 1.7B and of five of the six 3B
  cells (not A-L30, `bench_bpb.csv`), so the twins enter every benchmark row
  at both references, on five families at 3B (293 mono-axis and 315
  multi-axis twin tasks). Every table below therefore pools them; the
  benchmark-only numbers are quoted in the Key findings.
- **Known-answer check.** `--reference 1.7B --check` reproduces the
  decision-accuracy tables' `decision_acc_size_<proxy>` per task for both
  pair sets (`predictivity/da_all_per_task_both_axes.csv`). It was exact on
  all 4,490 cells on 2026-10-02 (max |diff| 1.11e-16) and has not been re-run
  on today's pools.

## Key figure

![The benchmarks the 3B rung lifts above chance, and decision accuracy to 3B beside 1.7B](pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png)

Population: pool `predictivity` (seed 1904); the six deep families scored at both 1.7B and 3B (A-L8, A-L15, A-L30, A-L50, B-L8, B-L15), each (family, task) scored at both rungs. Left: per benchmark with at least five tasks whose share moves, the share of its tasks the above-random gate admits at each rung (rule 1, recomputed on those families). Right: DA-size from the finals of 90M–1B to the 3B final (solid) and to the 1.7B final (dashed) on the same mono-axis decisions, the task above chance at the proxy, 1.7B and 3B, no filter, ≥ 3 pairs; benchmark accuracy and per-language BPB pooled separately, 90 % leave-one-family-out jackknife bands.

**Key finding.** **The 3B rung measures more than the reference but decides
no differently.** On the same six families it lifts the above-chance count
from 522 to 687 of 841 tasks (+32 %), while mono-axis DA-size on identical
decisions stays at 0.49–0.54 in benchmark accuracy (bBPB twins pooled) to
either reference and at 0.81–0.96 in per-language BPB.

GitHub: [gate_share_and_da_size_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.png) · [gate_share_and_da_size_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_paper.csv). The full figures: [the gate per benchmark](#the-share-of-each-benchmark-above-chance-at-17b-and-at-3b) and [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b). The previous key figure, DA-size to the 3B final alone, is still written as `above_reference_3B_paper.png` ([the 3B rung as the reference](#the-3b-rung-as-the-reference)).

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Population.** 6 families with a final at 3B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L30-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

| axes | proxy | DA-size → 3B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 3B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.53 | 457 | 0.53 | 0.96 | 10 |
| mono-axis | 175M | 0.50 | 466 | 0.53 | 0.96 | 10 |
| mono-axis | 350M | 0.49 | 487 | 0.52 | 0.81 | 10 |
| mono-axis | 600M | 0.53 | 516 | 0.53 | 0.94 | 10 |
| mono-axis | 1B | 0.54 | 537 | 0.52 | 0.90 | 10 |
| mono-axis | 1.7B | 0.50 | 551 | — | 0.96 | 10 |
| multi-axis | 90M | 0.54 | 504 | 0.54 | 0.94 | 12 |
| multi-axis | 175M | 0.51 | 517 | 0.54 | 0.94 | 12 |
| multi-axis | 350M | 0.50 | 540 | 0.54 | 0.88 | 12 |
| multi-axis | 600M | 0.54 | 573 | 0.54 | 0.91 | 12 |
| multi-axis | 1B | 0.55 | 599 | 0.53 | 0.90 | 12 |
| multi-axis | 1.7B | 0.51 | 618 | — | 0.95 | 12 |

Files: [`above_reference_3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png), [`above_reference_3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.csv), [`above_reference_3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_per_task.csv).
<!-- END auto:above-reference-3B -->

Key findings:

- **In benchmark accuracy no proxy predicts the 3B decision, the 1.7B
  reference included.** On the six families' gated benchmark tasks (the bBPB
  twins left out), DA-size → 3B is 0.47–0.52 mono-axis over 164–258 tasks and
  0.48–0.53 multi-axis over 189–303; the 1.7B rung (0.50 mono-axis, 0.52
  multi-axis) sits below 1B (0.52, 0.53), so more model is not a better proxy
  of the next rung here.
- **The "same families" column is not benchmark-only, and neither is the 3B
  column:** its 0.52–0.53 mono-axis and 0.53–0.54 multi-axis pool the bBPB
  twins, which exist at 1.7B on all six families and at 3B on five (panel
  (a)'s dashed lines rest on a median of 622 and 728 tasks against 501 and
  556 to 3B). The benchmark-only comparison on identical decisions, from
  [the framework at the new reference](#is-the-framework-consistent-when-the-reference-moves-from-17b-to-3b)'s
  per-task table, is 0.47–0.52 to 1.7B against 0.47–0.52 to 3B.
- **In per-language BPB the reference is a good proxy for the next rung: 0.96
  mono-axis, 0.95 multi-axis.** Every proxy from 90M up reaches at least 0.81
  (lowest at 350M: 0.81 mono-axis, 0.88 multi-axis), over the five families
  with a BPB final at 3B.
- **`bpb_macro` agrees on every pair, `train_loss` on fewer than half the
  mono-axis pairs.** From 1.7B to 3B, `bpb_macro` matches 6/6 mono-axis and
  10/10 multi-axis pairs; `train_loss`, which all six families carry, matches
  4/9 and 10/15.
- **Panel (d) says the same per benchmark family** (multi-axis, 1.7B → 3B, 64
  families, 37 of them bBPB twins): `bpb` leads at 0.96 (12 tasks), then
  `bbpb_xstorycloze` and `bbpb_commonsense_qa` at 0.80 (4, 1), `openbookqa`
  at 0.73 (1) and `arc_mt` at 0.70 (6). 28 of the 64 families sit at or
  below 0.50.
- **The BPB claim rests on 10–12 tasks**: four languages every family with a
  3B BPB final trains (cmn, deu, jpn, rus), `bpb_dclm` and `bpb_macro`, plus
  fas, fra, ita and spa, and ell and tha multi-axis. This is the figure's
  binding limitation, and a BPB final for the A-L30 cell at 3B is what would
  widen it.
- **DA-goal along the run (panel b) never leaves chance in benchmark
  accuracy.** Every checkpoint before the final of every proxy reads the 3B
  final at 0.48–0.56 multi-axis over 504–618 tasks (0.47–0.55 mono-axis),
  bBPB twins pooled; per-language BPB (in `above_reference_3B.csv`, not drawn
  in panel b) stays at 0.78–0.98 multi-axis (0.69–0.98 mono-axis).

Follow-ups:

- Panel (c) pairs per-task DA-size 1.7B → 3B against 1B → 1.7B at Pearson
  r = 0.16 over 610 tasks (multi-axis): the tasks whose ranking converged by
  1.7B are barely the ones that keep it at 3B. On a k/15 lattice this is a
  weak instrument, so quote the pooled lines, not r.
- Score A-L30's 3B cell in the per-item store so the "same families" column
  compares like with like; until then read the reference-consistency table.
- Add the reference's earlier checkpoints as proxies (DA-ckpt at 3B) once its
  k/10 grid is evaluated; `per_task()` already takes any `frac`.
- The rung's families are all deep: no depth pair, so the by-transformation
  reading is limited to the language count and the scheme.

### The preview: the 3B design set read to 1.7B

The control that separates "3B is unpredictable" from "this population is
unpredictable at any reference". It reads all seven 3B-design families at
1.7B, where B-L30 already has a final, so its population is wider
than the 3B table's six.

<!-- BEGIN auto:above-reference-1.7B-design3B (above_reference.py --pool predictivity --reference 1.7B --design 3B) -->
## The 1.7B rung as the reference — preview on the `3B` design set

**DA-size and DA-goal · reference 1.7B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 1.7B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B`.

**Population.** 7 families with a final at 1.7B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L30-deep-seed1904, lm-L30-schemeB-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 1.7B](pretraining/predictivity/above_reference_1.7B_design3B.png)

| axes | proxy | DA-size → 1.7B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 1.7B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.53 | 680 | 0.53 | 0.88 | 21 |
| mono-axis | 175M | 0.52 | 704 | 0.52 | 0.88 | 21 |
| mono-axis | 350M | 0.51 | 728 | 0.51 | 0.85 | 21 |
| mono-axis | 600M | 0.52 | 761 | 0.52 | 0.88 | 21 |
| mono-axis | 1B | 0.52 | 785 | 0.52 | 0.76 | 21 |
| multi-axis | 90M | 0.53 | 843 | 0.53 | 0.91 | 31 |
| multi-axis | 175M | 0.52 | 875 | 0.52 | 0.91 | 31 |
| multi-axis | 350M | 0.52 | 904 | 0.52 | 0.87 | 31 |
| multi-axis | 600M | 0.53 | 944 | 0.53 | 0.91 | 31 |
| multi-axis | 1B | 0.53 | 972 | 0.53 | 0.77 | 31 |

Files: [`above_reference_1.7B_design3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.png), [`above_reference_1.7B_design3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.csv), [`above_reference_1.7B_design3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B_per_task.csv).
<!-- END auto:above-reference-1.7B-design3B -->

Key findings:

- **On the seven design families, DA-size to 1.7B is a coin flip from every
  proxy size.** Benchmark tasks only (the bBPB twins left out): 0.50 / 0.50 /
  0.48 / 0.49 / 0.50 mono-axis at 90M / 175M / 350M / 600M / 1B over 178–283
  gated tasks, and 0.49 / 0.50 / 0.49 / 0.50 / 0.52 multi-axis over 224–353;
  the 3B question therefore starts from a population whose 1.7B ranking the
  ladder does not resolve either.
- **Every row of the table pools the bBPB twins.** The 502 mono-axis (619
  multi-axis) twin tasks alone give 0.53–0.54 (0.53–0.55) at every proxy size,
  which lifts the rows to 0.51–0.53 over 680–972 tasks.
- **The pair sets are larger than at 3B.** Seven families give 21 multi-axis
  and 12 mono-axis pairs against 15 and 9 for the six at 3B, so the two
  tables differ in population as well as in reference.
- **BPB on these families reaches 0.76–0.91 at the 1.7B reference**
  (0.76–0.88 mono-axis over 21 tasks, 0.77–0.91 multi-axis over 31), against
  0.81–0.96 to 3B over 10–12: the channel that works does not degrade with the
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

**The above-random gate at two rungs · the 6 families with a 3B final · both columns recomputed on those families (rule 1) so the rung is the only difference.** Regenerate with `python analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity`.

**Population.** 841 tasks carry a chance level at both rungs; BPB and the generative tasks have none, so they never enter the gate. The gate admits **522** of them at 1.7B and **687** at 3B (**+165**, +32%).

![Gate crossover](pretraining/predictivity/gate_crossover.png)

|  | at chance at 3B | above chance at 3B |
|---|---|---|
| **above chance at 1.7B** | 4 | 518 |
| **at chance at 1.7B** | 150 | 169 |

**169 tasks cross into the gate at 3B** and **4** drop out of it. Where they come from:

| benchmark | gained | lost | net |
|---|---|---|---|
| belebele | 45 | 0 | +45 |
| global_mmlu_full | 29 | 0 | +29 |
| include_base_44 | 28 | 0 | +28 |
| cultural_bench_easy | 15 | 0 | +15 |
| global_piqa_parallel_cloze | 14 | 1 | +13 |
| include_v2_og | 8 | 1 | +7 |
| blend_sample | 5 | 0 | +5 |
| cultural_bench_hard | 4 | 0 | +4 |
| rf_cultural_bench_easy | 4 | 0 | +4 |
| rf_include_base_44 | 4 | 0 | +4 |
| include_v2_en | 3 | 1 | +2 |
| rfgm_include_base_44 | 3 | 0 | +3 |

Files: [`gate_crossover.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.png), [`gate_crossover.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.csv), [`gate_crossover_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_per_task.csv) (every task's run shares and verdicts, `status` = `newly above` for the tasks that cross in).

### The share of each benchmark above chance at 1.7B and at 3B

Per benchmark, the share of its tasks the gate admits at each rung, on the same (family, task) cells; 40 benchmarks, 17 gain, 0 lose, 5 have no task above chance at 1.7B and at least one at 3B.

![Gate share by benchmark](pretraining/predictivity/gate_crossover_by_benchmark.png)

| benchmark | tasks | above at 1.7B | above at 3B | share change |
|---|---|---|---|---|
| global_mmlu_full | 29 | 0 (0%) | 29 (100%) | +100% |
| blend_sample | 5 | 0 (0%) | 5 (100%) | +100% |
| commonsense_qa | 1 | 0 (0%) | 1 (100%) | +100% |
| mmlu | 1 | 0 (0%) | 1 (100%) | +100% |
| cultural_bench_easy | 19 | 2 (11%) | 17 (89%) | +79% |
| include_base_44 | 36 | 5 (14%) | 33 (92%) | +78% |
| belebele | 59 | 10 (17%) | 55 (93%) | +76% |
| rf_cultural_bench_easy | 19 | 7 (37%) | 11 (58%) | +21% |
| cultural_bench_hard | 19 | 0 (0%) | 4 (21%) | +21% |
| global_piqa_parallel_cloze | 63 | 14 (22%) | 27 (43%) | +21% |
| paws | 8 | 5 (62%) | 6 (75%) | +12% |
| rf_include_base_44 | 36 | 29 (81%) | 33 (92%) | +11% |
| include_v2_og | 77 | 57 (74%) | 64 (83%) | +9% |
| rfgm_include_base_44 | 36 | 31 (86%) | 34 (94%) | +8% |
| arc | 28 | 22 (79%) | 24 (86%) | +7% |
| include_v2_en | 77 | 66 (86%) | 68 (88%) | +3% |
| rf_belebele | 59 | 57 (97%) | 58 (98%) | +2% |

Files: [`gate_crossover_by_benchmark.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_by_benchmark.png), [`gate_crossover_by_benchmark.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover_by_benchmark.csv).
<!-- END auto:gate-crossover -->

Key findings:

- **Yes, the 3B rung lifts more benchmarks above chance: it measures 32 % more
  of the suite than the 1.7B reference.** On the six families scored at both
  rungs, of the 841 tasks with a chance level the gate admits 522 at 1.7B and
  687 at 3B (+165 net): 169 cross in, 4 drop out, 518 stay above and 150 stay
  at chance at both rungs.
- **The crossings are concentrated in the multilingual knowledge and
  comprehension benchmarks.** Net gains: `belebele` +45, `global_mmlu_full`
  +29, `include_base_44` +28, `cultural_bench_easy` +15,
  `global_piqa_parallel_cloze` +13, together 130 of the 165.
- **Per language the gain is largest in the high-resource head**: en +16,
  es +13, zh +9, hi +7, ar +7, 52 of the 165 net. 45 languages gain at least
  one task net, and none outside the top five gains more than 5.
- **This is the strongest argument for the 3B rung, and it is independent of
  decision accuracy**: 169 tasks the reference reports as at chance become
  measurable. It is also a caution for every other analysis, whose gate stops
  at 1.7B by rule 10.
- **The 4 losses are small and thin**: `bbh_mcq_disambiguation_qa` (English,
  6 runs per rung) and three tasks on 2–4 runs per rung (`include_v2_en`
  Arabic-Kuwait, `include_v2_og` Greek-Cyprus,
  `global_piqa_parallel_cloze` Swedish). On two runs
  the Wilson rule can move that many by chance.

Follow-ups:

- The gate rests on 1–6 runs per (task, rung), and 34 of the 169 newly
  admitted tasks on a single run (a language only the L50 family trains).
  More 3B cells would tighten the gate directly; this, not the
  decision-accuracy table, is where B-L30 would pay.
- Ask in the gate analysis (`../rq00_gate_and_curves/README.md`) whether the
  newly admitted tasks' DA is any better than the already-admitted ones', i.e.
  whether measurability buys reliability.

Key findings (the per-benchmark shares, `gate_crossover_by_benchmark.png`):

- **The rung lifts the multiple-choice knowledge benchmarks the most.**
  `global_mmlu_full` goes from 0 % to 100 % of its 29 tasks, `include_base_44`
  from 14 % to 92 % of 36, `cultural_bench_easy` from 11 % to 89 % of 19 and
  `belebele` from 17 % to 93 % of 59; of the 40 benchmarks, 17 gain share and
  none loses it (`bbh_mcq` trades one task each way).
- **Their reformulated twins were already measurable at 1.7B.**
  `rf_global_mmlu_full` and `rfgm_belebele` sit at 100 % at both rungs and
  `rf_belebele` at 97–98 %, so what 3B adds is the original format, not the
  knowledge.
- **Five benchmarks have no task above chance at 1.7B and at least one at
  3B:** `global_mmlu_full` (0 → 29 of 29), `blend_sample` (0 → 5 of 5),
  `cultural_bench_hard` (0 → 4 of 19), `commonsense_qa` and `mmlu` (0 → 1 of 1).
- **Six benchmarks stay at chance on every task at 3B:** `acp_bench_cloze`
  (7 tasks), `acp_bench_mcq` (7), `bbh_cloze` (6), `truthfulqa_mc2` (3),
  `truthfulqa-multi_mc1` (2) and `toxigen` (1). `bbh_mcq` admits 1 of 17 and
  `global_piqa_parallel_cloze` 27 of 63 at 3B, so neither is measurable on
  most of its tasks on this ladder yet.

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

**DA-size to two references and the SNR at two rungs · 6 families scored at both (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L30-deep-seed1904, lm-L50-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904) · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy and at 1.7B, the Wilson rule on the 3B runs · no filter.** Regenerate with `python analysis/rq10_size_generalisation/reference_consistency.py --pool predictivity`.

**Population.** 15 multi-axis and 9 mono-axis pairs over the 6 families (fewer on a task whose language not every family trains, rule 2). A DA cell is a (task, pair set, proxy) whose decisions both references score, the task above chance at the proxy, at 1.7B and at 3B; 3730 cells over 499 tasks have fewer than 3 pairs and are NaN (rule 5, pair counts in the per-task table). Bands are the 90 % leave-one-family-out jackknife.

![DA-size to two references, multi-axis](pretraining/predictivity/reference_consistency_da_size_multi_axes.png)

![DA-size to two references, mono-axis](pretraining/predictivity/reference_consistency_da_size_mono_axis.png)

Benchmark accuracy, pooled DA-size [90 % band]:

| axes | proxy | tasks | → 1.7B | → 3B |
|---|---|---|---|---|
| mono-axis | 90M | 446 | 0.52 [0.45, 0.60] | 0.54 [0.52, 0.55] |
| mono-axis | 175M | 465 | 0.53 [0.50, 0.57] | 0.50 [0.48, 0.52] |
| mono-axis | 350M | 486 | 0.51 [0.46, 0.56] | 0.49 [0.45, 0.53] |
| mono-axis | 600M | 515 | 0.52 [0.46, 0.58] | 0.53 [0.49, 0.58] |
| mono-axis | 1B | 536 | 0.52 [0.49, 0.54] | 0.54 [0.51, 0.57] |
| multi-axis | 90M | 492 | 0.53 [0.44, 0.61] | 0.55 [0.51, 0.59] |
| multi-axis | 175M | 516 | 0.54 [0.50, 0.58] | 0.51 [0.48, 0.54] |
| multi-axis | 350M | 539 | 0.53 [0.49, 0.56] | 0.50 [0.47, 0.53] |
| multi-axis | 600M | 572 | 0.54 [0.47, 0.61] | 0.54 [0.51, 0.56] |
| multi-axis | 1B | 597 | 0.53 [0.48, 0.57] | 0.55 [0.52, 0.58] |

Spearman between the two references' DA, over the benchmark tasks and over the benchmarks (mean DA of their tasks):

| axes | proxy | ρ tasks | tasks | ρ benchmarks | benchmarks |
|---|---|---|---|---|---|
| mono-axis | 90M | -0.01 | 446 | -0.08 | 60 |
| mono-axis | 175M | 0.03 | 465 | 0.20 | 60 |
| mono-axis | 350M | 0.02 | 486 | -0.06 | 60 |
| mono-axis | 600M | 0.03 | 515 | 0.34 | 62 |
| mono-axis | 1B | 0.11 | 536 | 0.21 | 62 |
| multi-axis | 90M | 0.04 | 492 | -0.06 | 60 |
| multi-axis | 175M | 0.06 | 516 | 0.14 | 60 |
| multi-axis | 350M | 0.07 | 539 | -0.05 | 60 |
| multi-axis | 600M | 0.04 | 572 | 0.41 | 62 |
| multi-axis | 1B | 0.11 | 597 | 0.20 | 62 |

![SNR at two rungs](pretraining/predictivity/reference_consistency_snr.png)

SNR (`rel_std`) at 1.7B against 3B, tasks above chance at both:

| channel | ρ tasks | tasks | ρ benchmarks | benchmarks | median log10 SNR 1.7B | median log10 SNR 3B | share higher at 3B |
|---|---|---|---|---|---|---|---|
| benchmarks | 0.05 | 943 | 0.18 | 63 | 0.11 | 0.01 | 0.40 |
| bpb | 0.95 | 25 | — | 1 | 0.38 | 0.21 | 0.08 |

Files: [`reference_consistency_da_size_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_multi_axes.csv), [`reference_consistency_da_size_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_mono_axis.csv), [`reference_consistency_da_size_per_task_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_da_size_per_task_both_axes.csv), [`reference_consistency_snr.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.png) / [`.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr.csv), [`reference_consistency_snr_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/reference_consistency_snr_per_task.csv).
<!-- END auto:reference-consistency -->

Key findings:

- **Pooled DA-size does not depend on the reference.** On identical
  decisions (six families, benchmark tasks and their bBPB twins above chance
  at the proxy, 1.7B and 3B), mono-axis DA-size is 0.51–0.53 to 1.7B and
  0.49–0.54 to 3B over 446–536 tasks, multi-axis 0.53–0.54 and 0.50–0.55 over
  492–597; every 90 % band to 1.7B contains 0.50, and five of the ten bands
  to 3B sit just above it (lower edges 0.51–0.52).
- **In per-language BPB both references are well predicted.** DA-size is
  0.85–0.92 to 1.7B and 0.81–0.96 to 3B mono-axis (10 tasks), 0.88–0.91 and
  0.88–0.94 multi-axis (12 tasks, five families).
- **Which tasks a proxy ranks well does not survive the move.** Spearman ρ
  between the two references' per-task DA is −0.01 to 0.11 over 446–597 tasks,
  and only 20–25 % of the benchmark cells give the same DA to both (mean
  |difference| 0.26–0.29).
- **The ranking of benchmarks by DA survives weakly, best from 600M.** Over
  60–62 benchmarks ρ is −0.08 to 0.20 from 90M–350M, 0.34–0.41 from 600M
  and 0.20–0.21 from 1B, both pair sets.
- **The SNR ranking survives no better than the DA ranking, except in BPB.**
  Over 943 benchmark tasks above chance at both rungs ρ is 0.05 (0.18 over 63
  benchmarks), and over 25 BPB tasks 0.95.
- **SNR is lower at 3B.** Median log10 SNR drops from 0.11 to 0.01 on the
  benchmark tasks, with 40 % of them higher at 3B, and from 0.38 to 0.21 on
  BPB, where 2 of the 25 are higher; among the original benchmarks the
  largest drops are `arc_mt` (median −0.36 over 10 tasks),
  `global_piqa_parallel_cloze` (−0.36, 10) and `hellaswag` (−0.29, 20), and
  of those with more than one task only `xcopa` (+0.13, 7), `rf_bbh_mcq`
  (+0.05, 10) and `xnli` (+0.03, 12) rise.

Follow-ups:

- With six families at 3B the per-task DA sits on a k/15, k/9 lattice and
  3,730 cells over 499 tasks fall under `MIN_PAIRS`; read the pooled lines
  and the benchmark-level ρ before the per-task scatter.
- Split the SNR drop into its signal (spread of the six families) and noise
  (checkpoint window) halves, to say whether the designs converge at 3B or
  the late checkpoints get noisier.

## The DA panel on one task set per line

The DA lines pool, at each proxy size, the tasks above chance at the proxy and at both references, so they can move because their tasks changed (rule 13). The twin redraws the DA panel only (the gate panel is a share, not a mean over tasks), one panel per reference: per channel, the tasks with a value at every proxy size (solid), the committed line dashed behind. A cell is kept for both references at once, so one task set serves both.

![DA to 1.7B and to 3B on one task set per line](pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.png)

DA-size (no filter, mono-axis pairs, gate `predictivity` at the proxy, the references' own gate at 1.7B and 3B), pooled matching over comparable pairs. GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_share_and_da_size_mono_axis_fixed_tasks_paper.csv)

Key findings:

- **The fixed task set changes nothing that matters.** The benchmark lines keep 444 of 446–536 tasks and move by 0.02 at most (to 1.7B, at 600M); against 3B they read 0.49–0.54 on both task sets.
- **BPB does not move.** The same 10 BPB tasks at every size.

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
  `gate_share_and_da_size_mono_axis_paper.{png,csv}`.
