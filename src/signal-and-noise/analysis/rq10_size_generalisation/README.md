# Size generalisation — does a ranking that holds at 1.7B still hold one rung above it, at 3B?

## Research question

Every other analysis stops at the reference: a proxy is judged by whether it
makes the decisions the 1.7B model makes (rule 10, `analysis/RULES.md`). The
3B rung asks the one question that frame cannot: whether the reference itself
is a proxy for the next rung.

Four cells were trained at 3B for it (deep, L ∈ {8, 15}, schemes A and B, seed
1904; `plan/3b_models.md`). This folder is therefore the only reader of
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

![Decision accuracy against the 3B rung, benchmark accuracy and BPB](pretraining/predictivity/above_reference_3B_paper.png)

Population: pool `predictivity` (seed 1904); DA-size from the finals of 90M–1.7B to the 3B final, 6 multi-axis and 4 mono-axis pairs over the 4 families with a 3B final; gate `predictivity` at the proxy and the Wilson rule on the 3B runs at the reference, no filter; benchmark accuracy (59–90 gated tasks) and per-language BPB (6–7 tasks) pooled separately.

**Key finding.** Benchmark accuracy reads the 3B ranking at chance from every proxy (DA-size 0.43–0.52, the 1.7B reference included), while per-language BPB reads it at 0.75–0.96. From the 1.7B reference itself BPB reaches 0.92 (multi-axis) and 0.96 (mono-axis).

GitHub: [above_reference_3B_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_paper.png) · [above_reference_3B_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_paper.csv). The full figure, with the same families read to 1.7B: [The 3B rung as the reference](#the-3b-rung-as-the-reference).

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Population.** 4 families with a final at 3B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

| axes | proxy | DA-size → 3B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 3B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.51 | 63 | 0.48 | 0.92 | 6 |
| mono-axis | 175M | 0.49 | 59 | 0.51 | 0.92 | 6 |
| mono-axis | 350M | 0.47 | 68 | 0.46 | 0.75 | 6 |
| mono-axis | 600M | 0.52 | 73 | 0.44 | 0.88 | 6 |
| mono-axis | 1B | 0.49 | 77 | 0.53 | 0.92 | 6 |
| mono-axis | 1.7B | 0.43 | 83 | — | 0.96 | 6 |
| multi-axis | 90M | 0.51 | 66 | 0.47 | 0.95 | 7 |
| multi-axis | 175M | 0.49 | 62 | 0.51 | 0.95 | 7 |
| multi-axis | 350M | 0.46 | 71 | 0.47 | 0.82 | 7 |
| multi-axis | 600M | 0.51 | 77 | 0.44 | 0.90 | 7 |
| multi-axis | 1B | 0.49 | 83 | 0.51 | 0.92 | 7 |
| multi-axis | 1.7B | 0.44 | 90 | — | 0.92 | 7 |

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

**Population.** 4 families with a final at 1.7B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 1.7B](pretraining/predictivity/above_reference_1.7B_design3B.png)

| axes | proxy | DA-size → 1.7B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 1.7B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.48 | 53 | 0.48 | 0.88 | 6 |
| mono-axis | 175M | 0.51 | 234 | 0.51 | 0.88 | 6 |
| mono-axis | 350M | 0.46 | 68 | 0.46 | 0.79 | 6 |
| mono-axis | 600M | 0.44 | 73 | 0.44 | 0.83 | 6 |
| mono-axis | 1B | 0.53 | 251 | 0.53 | 0.88 | 6 |
| multi-axis | 90M | 0.47 | 56 | 0.47 | 0.87 | 7 |
| multi-axis | 175M | 0.51 | 249 | 0.51 | 0.87 | 7 |
| multi-axis | 350M | 0.47 | 71 | 0.47 | 0.85 | 7 |
| multi-axis | 600M | 0.44 | 77 | 0.44 | 0.82 | 7 |
| multi-axis | 1B | 0.51 | 269 | 0.51 | 0.90 | 7 |

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

**The above-random gate at two rungs · the 4 families with a 3B final · both columns recomputed on those families (rule 1) so the rung is the only difference.** Regenerate with `python analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity`.

**Population.** 573 tasks carry a chance level at both rungs; BPB and the generative tasks have none, so they never enter the gate. The gate admits **337** of them at 1.7B and **429** at 3B (**+92**, +27%).

![Gate crossover](pretraining/predictivity/gate_crossover.png)

|  | at chance at 3B | above chance at 3B |
|---|---|---|
| **above chance at 1.7B** | 6 | 331 |
| **at chance at 1.7B** | 138 | 98 |

**98 tasks cross into the gate at 3B** and **6** drop out of it. Where they come from:

| benchmark | gained | lost | net |
|---|---|---|---|
| belebele | 17 | 3 | +14 |
| global_mmlu_full | 17 | 0 | +17 |
| cultural_bench_easy | 14 | 0 | +14 |
| include_base_44 | 14 | 1 | +13 |
| global_piqa_parallel_cloze | 8 | 0 | +8 |
| blend_sample | 5 | 0 | +5 |
| include_v2_og | 5 | 0 | +5 |
| rf_cultural_bench_easy | 5 | 0 | +5 |
| cultural_bench_hard | 4 | 0 | +4 |
| rf_include_base_44 | 2 | 0 | +2 |
| rfgm_include_base_44 | 2 | 0 | +2 |
| bbh_mcq | 1 | 1 | +0 |

Files: [`gate_crossover.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.png), [`gate_crossover.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/gate_crossover.csv).
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
- `pretraining/predictivity/above_reference_<ref>[_design<d>].{png,csv}`,
  `..._per_task.csv`, `gate_crossover.{png,csv}`.
