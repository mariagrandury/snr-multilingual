# RQ10 — Does a ranking that holds at 1.7B still hold one rung above it, at 3B?

## Research question

Every other question stops at the reference: a proxy is judged by whether it
makes the decisions the 1.7B model makes (rule 10, `analysis/RULES.md`). The
3B rung exists to ask the one question that cannot be asked inside that
frame — whether the reference itself is a proxy for the next rung. Four cells
were trained at 3B for it (deep, L ∈ {8, 15}, schemes A and B;
`plan/3b_models.md`), so this RQ is the only reader of
`build_snr_pool(above_reference=True)` and the only folder the rule-10 checker
exempts (`check_rules.EXEMPT`).

Snapshot: the ladder report of **2026-10-02 06:24** (5,631 rows). All four 3B
cells are trained to 145,200 iterations and evaluated, so every table below
carries real numbers.

The RQ has two halves, and they answer differently:

- **can the decision be ranked one rung up** — the DA question, below. In
  benchmark accuracy, no; in BPB, yes, at 0.96.
- **can the benchmark be measured at all one rung up** — the gate question,
  [below](#what-the-3b-rung-can-measure-that-the-17b-reference-cannot). The 3B
  rung lifts 98 tasks above chance that the reference leaves at chance.

## Setup

- **Population.** Every family with a final at the reference rung; pairs at
  the grid seed, multi-axis and mono-axis (rule 15). With the four 3B
  families that is six multi-axis and four mono-axis pairs — above rule 5's
  minimum, but a per-task DA sits on a coarse lattice (k/6, k/4), so the
  pooled ratio over tasks is what the lines draw.
- **Gate.** `predictivity`'s mask at the proxy; at the reference rung the same
  one-sided 95 % Wilson rule computed on the reference's own runs
  (`above_random.scores_and_mask`), since the committed mask stops at 1.7B.
- **Two channels, pooled separately.** Benchmark accuracy and per-language
  BPB are never averaged together: DA-size sits at ~0.50 on the first and
  ~0.9 on the second, so one mean over both would report neither (rq02 keeps
  `bpb_macro` and `train_loss` out of its benchmark means for the same
  reason). BPB carries no chance level, so it is never gated; only the
  languages every paired family scores clear `MIN_PAIRS`, which is why the
  BPB column rests on 6–7 tasks against the benchmark column's 60–93.
- **Comparison.** Panels (a) and (c) read the same families to 1.7B, so the
  two references differ in nothing but the reference. The standalone preview
  (`--reference 1.7B --design 3B`) is that comparison line on its own.
- **Known-answer check.** `--reference 1.7B --check` reproduces rq02's
  `decision_acc_size_<proxy>` per task for both pair sets
  (`predictivity_schemes/da_all_per_task_both_axes.csv`, the pool the rq02 decision figures
  pair over) — exact on all 4,490 cells on 2026-10-02, max |diff| 1.11e-16.
  The script and rq02 share the pair sets and the kernel; the `predictivity`
  folder's table is the A/B-only pool and differs by construction.

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Population.** 4 families with a final at 3B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs, and separately over the per-language BPB tasks (never gated — no chance level; only the languages every paired family scores clear 3 pairs, which is why the BPB column rests on far fewer tasks).

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

| axes | proxy | DA-size → 3B | tasks | DA-size → 1.7B (same families) | BPB DA-size → 3B | BPB tasks |
|---|---|---|---|---|---|---|
| mono-axis | 90M | 0.50 | 60 | 0.48 | 0.92 | 6 |
| mono-axis | 175M | 0.48 | 58 | 0.47 | 0.92 | 6 |
| mono-axis | 350M | 0.47 | 68 | 0.46 | 0.75 | 6 |
| mono-axis | 600M | 0.52 | 76 | 0.44 | 0.88 | 6 |
| mono-axis | 1B | 0.49 | 78 | 0.48 | 0.92 | 6 |
| mono-axis | 1.7B | 0.43 | 85 | — | 0.96 | 6 |
| multi-axis | 90M | 0.51 | 63 | 0.47 | 0.95 | 7 |
| multi-axis | 175M | 0.49 | 61 | 0.49 | 0.95 | 7 |
| multi-axis | 350M | 0.46 | 71 | 0.47 | 0.82 | 7 |
| multi-axis | 600M | 0.51 | 80 | 0.44 | 0.90 | 7 |
| multi-axis | 1B | 0.49 | 83 | 0.48 | 0.92 | 7 |
| multi-axis | 1.7B | 0.44 | 93 | — | 0.92 | 7 |

Files: [`above_reference_3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png), [`above_reference_3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.csv), [`above_reference_3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_per_task.csv).
<!-- END auto:above-reference-3B -->

Key findings:

- **In benchmark accuracy nothing predicts 3B — but nothing predicts 1.7B
  either.** DA-size → 3B is 0.43–0.52 from every proxy, the 1.7B reference
  included (0.43 mono-axis, 0.44 multi-axis). Read alone that says the rung is
  unpredictable. It is not: the same four families read to 1.7B give 0.44–0.49,
  and the full 25-family population at 1.7B gives 0.51–0.52 (mono-axis) and
  0.54–0.57 (multi-axis). The benchmark channel is at chance at every
  reference and at every population size, so the 3B number is the study's
  standing result arriving one rung later, not a property of 3B.
- **In BPB the reference IS a good proxy for the next rung: 0.96 mono-axis,
  0.92 multi-axis.** Every proxy from 90M up clears 0.75, and the ordering is
  not merely preserved but close in magnitude — on macro-BPB the four
  mono-axis pairs move +0.0679 → +0.0593, +0.0400 → +0.0414, +0.0057 →
  +0.0009, −0.0222 → −0.0169 from 1.7B to 3B, 4/4 agreeing in sign. Train
  loss agrees on 3 of 4; its one flip (`L`, scheme B) is +0.0102 against
  −0.0183, both inside the noise of a near-tie.
- Panel (d) says the same per family: `bpb` leads at 0.90, `loss` 0.83,
  and everything from `hellaswag` (0.67) down is at or under chance.
- **The BPB claim rests on 6–7 tasks** — the languages an L8 and an L15 family
  both score, plus `bpb_macro` and `bpb_dclm`. This is the figure's binding
  limitation, and the one thing more 3B cells would buy: A-L30 + B-L30 would
  widen it to the L30 intersection.
- DA-goal along the run (panel b) never leaves 0.40–0.60 for any proxy size,
  consistent with the benchmark reading.

Follow-ups:

- Panel (c) pairs per-task DA-size 1.7B → 3B against 1B → 1.7B at Pearson
  r = 0.36 over 89 tasks: the tasks whose ranking converged by 1.7B are only
  weakly the ones that keep it at 3B. With 4 and 6 pairs the per-task DA sits
  on a k/4, k/6 lattice, so this is a weak instrument — quote the pooled
  lines, not r.
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
| mono-axis | 175M | 0.47 | 58 | 0.47 | 0.88 | 6 |
| mono-axis | 350M | 0.46 | 68 | 0.46 | 0.79 | 6 |
| mono-axis | 600M | 0.44 | 76 | 0.44 | 0.83 | 6 |
| mono-axis | 1B | 0.48 | 77 | 0.48 | 0.88 | 6 |
| multi-axis | 90M | 0.47 | 56 | 0.47 | 0.87 | 7 |
| multi-axis | 175M | 0.49 | 61 | 0.49 | 0.87 | 7 |
| multi-axis | 350M | 0.47 | 71 | 0.47 | 0.85 | 7 |
| multi-axis | 600M | 0.44 | 80 | 0.44 | 0.82 | 7 |
| multi-axis | 1B | 0.48 | 82 | 0.48 | 0.90 | 7 |

Files: [`above_reference_1.7B_design3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.png), [`above_reference_1.7B_design3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.csv), [`above_reference_1.7B_design3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B_per_task.csv).
<!-- END auto:above-reference-1.7B-design3B -->

Key findings:

- On these four families alone, DA-size to 1.7B sits at a coin flip from every
  proxy size (mono-axis 0.48 / 0.47 / 0.46 / 0.44 / 0.48 at 90M / 175M / 350M /
  600M / 1B over 53–77 gated tasks; multi-axis 0.47 / 0.49 / 0.47 / 0.44 /
  0.48), against 0.51 → 0.57 on the full population (rq02, figure 1). Four
  families give six pairs, three of them between the two language lists at
  one L — decisions rq02 already reads at chance — so the 3B question starts
  from a population whose 1.7B ranking the ladder does not resolve either.
  **This is the control for the 3B table above.**
- BPB on the same families reaches 0.79–0.90 at the 1.7B reference, against
  0.75–0.96 at 3B: the channel that works does not degrade with the rung.
- The gated task count is 53–82 because the four families train 8 or 15
  languages (rule 2): the 3B answer is about those languages' tasks.

Follow-ups: see above; and consider whether the rung's design set should be
widened (a shallow cell, or L30) before it is read as a generalisation test.

### The prior question: what the rung can measure at all

DA asks whether a ranking survives. It can only ask that of a benchmark that
is above chance at both rungs. The gate is a property of (task, size), so a
benchmark the reference cannot resolve may become measurable one rung up —
and the reference's own mask can never say so.

<!-- BEGIN auto:gate-crossover (gate_crossover.py --pool predictivity) -->
## What the 3B rung can measure that the 1.7B reference cannot

**The above-random gate at two rungs · the 4 families with a 3B final · both columns recomputed on those families (rule 1) so the rung is the only difference.** Regenerate with `python analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity`.

**Population.** 541 tasks carry a chance level at both rungs; BPB and the generative tasks have none, so they never enter the gate. The gate admits **305** of them at 1.7B and **397** at 3B (**+92**, +30%).

![Gate crossover](pretraining/predictivity/gate_crossover.png)

|  | at chance at 3B | above chance at 3B |
|---|---|---|
| **above chance at 1.7B** | 6 | 299 |
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

- **The 3B rung measures 30 % more of the benchmark suite than the reference
  does**: of the 541 tasks with a chance level, the gate admits 305 at 1.7B
  and 397 at 3B. 98 cross in, 6 drop out.
- The crossings are concentrated in the hardest multilingual knowledge
  benchmarks — `global_mmlu_full` +17, `belebele` +14 net, `cultural_bench_easy`
  +14, `include_base_44` +13 net — i.e. exactly the families the paper wants
  to say something about and that sit at chance for most of the ladder.
- Per language the gain is broadest in the high-resource head (en +16, es +12,
  ar +7, zh +7), which is where these benchmarks have the most items: the
  3B rung does not rescue the tail, it rescues the tasks that were close.
- **This is the strongest argument for the 3B rung**, and it is independent of
  DA: 98 tasks that the reference reports as noise become measurable. It is
  also a caution for every other RQ, whose gate stops at 1.7B by rule 10.
- The 6 losses (3 `belebele`, 1 each `bbh_mcq`, `include_base_44`,
  `include_v2_en`) are within what the 4-run Wilson rule will move by chance;
  they are not a size effect.

Follow-ups:

- The two columns rest on 4 runs at 3B against 4 at 1.7B. More 3B cells would
  tighten the gate estimate directly — this, not the DA table, is where
  A-L30 + B-L30 would pay.
- Worth asking in rq00 whether the newly-admitted tasks' DA is any better than
  the already-admitted ones', i.e. whether measurability buys reliability.

## Extensions from other sweeps

None: the 36-model sweep has no rung above its 1B reference, and the public
models' size steps (rq02, "Extensions") are between-lab decisions, not this
ladder's.

## Files

- `above_reference.py` — the DA half; `--reference` picks the rung,
  `--design 3B` restricts to the rung's design set, `--check` compares with
  rq02, `--out-dir` writes elsewhere (the check).
- `gate_crossover.py` — the gate half; `--reference` picks the rung read
  above the 1.7B reference.
- `pretraining/predictivity/above_reference_<ref>[_design<d>].{png,csv}`,
  `..._per_task.csv`, `gate_crossover.{png,csv}`.
