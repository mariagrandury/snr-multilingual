# Status — 2026-09-02

A docs/analysis day: no cells trained, no evals submitted. Eight open
discussion points from [status-09-01.md](status-09-01.md) were settled or
quantified, and the answers are written into the plan docs rather than left in
chat. Implementation lands tomorrow.

## What was decided or measured today

- **Per-language benchmark coverage is now generated, not hand-written** —
  `scripts/wire_harness_tasks.py --report` rewrites the table in
  [benchmark_selection.md](benchmark_selection.md) from `configs/tasks.json`,
  so it cannot drift. 95 languages, 463 tasks at L100.
- **Every trained language has ≥ 1 benchmark family**, but 16 have exactly one,
  and for 5 of those (`cy`, `ga`, `la`, `kmr`, `ug`) it is MultiBLiMP — a
  grammaticality probe, not a task. Those languages have no task benchmark.
- **We do not have every high-quality benchmark.** Three gap classes documented:
  in-harness but deliberately unwired (okapi mmlu/truthfulqa, `mmlu_prox`,
  `indicxnli`, ~15 single-language native suites), published but unported
  (SIB-200, Taxi1500, MILU, Uhura, LORAXBENCH, TUMLU), and languages with
  nothing anywhere.
- **LAMBADA correction**: upstream marks `lambada_multilingual` (what we wire)
  as legacy in favour of `lambada_multilingual_stablelm`. Switch or drop it.
- **INCLUDE v2 is the best single addition available** — 113 language–country
  pairs / 89 languages, big African and South Asian expansion. 19 of our
  trained languages gain a family; **Odia goes 1 → 2**, and `as`/`pa`/`sd`/`so`
  go 2 → 3. Blocker: the shipped tasks are generative CoT (`generate_until`,
  4096 gen tokens, answer-tag extraction) — unusable for 90M–1.7B base models.
  The data is plain 4-option MC, so an `output_type: multiple_choice` variant
  is needed first.
- **Seed change is free today**: all 16 replicate cells (8× seed28, 8×
  seed1797) are unstarted. Renaming to 1904/313/64 costs nothing now; the same
  change after they start would forfeit **~10,900 node-hours** (43,700 GPU-h,
  29 % of the sweep) — 98 % of it at the 1B rung.
- **FLOPs convention pinned** to 6 × (n_non_emb + d_model × vocab) × D, one
  definition in `configs.flops_params`, with a `flops_basis` tag so external
  models on a nominal total are visible as such. Our own numbers are unchanged.
- **Milestone evals cost ~1 extra eval per run** and turn the IsoFLOP reads
  (1e19 / 3.2e19 / 1e20, 3–4 sizes each) into measured points. **No retraining
  needed** — the checkpoints already exist on the save grid.
- **Sampling temperature is the biggest methodological finding.** At L100 with
  T = 1 the head/tail share ratio is 16,581 : 1; at 90M, **66 of 99 languages
  get under 10 M tokens** and the smallest gets 80 K. Including the 8 languages
  swapped into L100 *because they have benchmarks*. Recommendation: **T = 2
  sweep-wide** — lifts the floor 40× (0.8 M → 33 M tokens at the 1B rung) and
  repeats nothing (max 0.3 epochs). Never vary T with L: that confounds the
  intervention.
- **Dropping 1.7B frees 40 % of the sweep** (15,086 of 37,860 node-hours) and
  makes 1B the reference at every L — which makes the temperature problem
  worse, not better.
- **5 × C vs ATLAS is testable without new full runs**: 12 WSD cooldown
  branches (350M/600M × L ∈ {1, 30, 100} × f ∈ {0.25, 0.5}) ≈ 6 % of a level.
  Do the cheap falsification first — the tail may be data-starved, not
  under-trained.
- **Grid redefinition (≥ 3 models per cell) dropped** at the user's call; the
  seed change may cover it.

Detail lives in
[small-to-large-predictivity-training-plan.md](small-to-large-predictivity-training-plan.md)
("The compute axis", "Sampling temperature", "Is 5 × C the right budget") and
[benchmark_selection.md](benchmark_selection.md) (coverage table, the three gap
classes, INCLUDE v2).

## Still open from 09-01

- **Eval walltime at L ≥ 30 is unresolved and blocking.** L100 = 463 tasks →
  1.7B ≈ 1,356 min against the 719-min queue cap; over the cap already at
  1.7B/L30 (726) and 1B/L50 (728). `BATCH_TASKS=1` writes nothing on a
  walltime kill, so an over-cap job is resubmitted and killed forever.
- The L100 data mixture still needs rebuilding (8-language swap), and the 8
  swap-ins must be verified present in the swiss-ai filtered dir first.
- 90M β₃ confirming run; BPB backlog; the 12 remaining ≤600M cells.

## Tomorrow, in order

1. Decide **T** (2 vs 1 + token floor). It gates every data rebuild below, and
   the L100 rebuild we already owe should absorb it.
2. Verify the 8 L100 swap-in languages exist in the swiss-ai filtered dir, then
   rebuild the L100 mixture + validation entries.
3. Fix the eval walltime: split jobs, or cap the during-training task set.
4. Change the seeds to 1904 / 313 / 64 (free today, expensive later).
5. Generate the INCLUDE v2 MC-format tasks and wire them.
6. Switch or drop LAMBADA-MT.
7. `derive_task_options.py` on the cluster for the newly wired tasks
   (no `n_options` yet).
8. Measure real per-task eval minutes on the first big job and recalibrate.
