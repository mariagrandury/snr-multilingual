# Adding a 3B rung: which cells, and why

*2026-09-19. Decision doc for the extrapolation check above the 1.7B reference.
Numbers: `src/pretrain/ladder_report.csv` (final checkpoints, deep, seed 1904),
`src/signal-and-noise/analysis/rq02_decision_accuracy/.../da_size_per_benchmark_multi_axes.csv`
and `rq00_gate_and_curves/.../above_random_share.csv`, regenerated 2026-09-19 21:04.*

## The question

The paper's claim is decision accuracy (DA) across size on **benchmarks**: does the
ranking of language mixes at a small size predict the ranking at the reference?
With 90M out of the analysis the reference is 1.7B. A 3B rung tests whether the
1.7B ranking is itself final. It only pays for pairs whose ranking is **still
open at 1.7B**; a pair already ordered the same way from 350M up is confirmed,
not tested, and a pair inside seed noise is a coin flip at any size.

## Constraints

| | value | source |
| --- | --- | --- |
| Tokens per run | 300B (D = 100N), 150B from the multilingual half | ladder rule |
| Data on disk | every FineWeb-2 build is ≤ 92B → a 3B repeats data unless rebuilt | builds |
| 165B builds at T = 1 | feasible, no language exhausted: A-L8 240B, A-L15 287B, B-L8 199B, B-L15 209B available | build logs |
| Cost per run | **~1,630 node-h** (measured 2026-09-22: 1,831–2,059 ms/iter wall, median 1,924, over all 8 jobs of the four live cells × 145,200 iters = 77.6 h × 21 nodes; the earlier ~1,900 extrapolated the 1.7B runs by N²) | job logs |
| Wall time per run | ~78 h on 21 nodes = 8 segments (a 12 h job trains for `JOB_TRAIN_SEC` = 10.83 h once startup and the exit-save grace are taken off), ~1 week with queueing | `ITER_MS` |
| Evals + BPB per cell | 12 benchmark checkpoints (the grid the analysis reads — `due_iters`, 2026-09-21) and 60 BPB; BPB ~1 h/ckpt at 3B (1.7B: 36 min) → ~60 node-h, which is nearly all of it | `score_bpb` |
| Deadline | paper 2026-09-25: **nothing lands in time**; this is for the revision | plan |
| 1.7B evals | still landing (B-L15 has no final-checkpoint benchmarks yet) | report |

## What the ladder already says

**Benchmarks are mostly at chance at 1.7B.** Share of cells above random and
DA toward the 1.7B reference (median over a family's tasks):

| family | gate@1.7B | DA 350M→1.7B | DA 1B→1.7B | read |
| --- | --: | --: | --: | --- |
| global_piqa (cloze) | 0.01 | 0.40 | 0.47 | at chance: DA is noise |
| include_base_44 | 0.11 | 0.50 | 0.53 | at chance |
| global_mmlu_full | 0.13 | 0.50 | 0.54 | at chance |
| belebele | 0.13 | 0.47 | 0.57 | at chance |
| arc | 0.46 | 0.61 | 0.76 | half gated |
| xnli / xcopa / paws | 0.66–0.83 | 0.61–0.64 | 0.62–0.65 | gated, DA flat |
| rf_global_mmlu / rf_belebele | 0.79–0.81 | 0.60 | 0.60–0.80 | reformulated twins are gated |
| multiblimp / xstorycloze | 0.69–1.00 | 0.67–0.70 | 0.73–0.76 | gated |
| hellaswag | 0.85 | 0.82 | 0.87 | resolved |

Overall: median DA(1B → 1.7B) = **0.60**; 382 of 653 tasks below 0.7, most of them
in the at-chance families. A 3B rung addresses this in two ways: 300B tokens
(1.8× the 1.7B's) may lift belebele / mmlu / include above chance so DA becomes
measurable on them at all, and it gives the gated families a target beyond 1.7B.

**Which pairs are open at 1.7B.** Family-mean score over the tasks both cells
were evaluated on; sign of (first − second) at 175M, 350M, 600M, 1B, 1.7B
(`·` = one cell not evaluated at that size); seed noise on a family mean is
±0.007.

| pair | ordering across sizes (gated families) | gap at top | status |
| --- | --- | --: | --- |
| **A8 vs B8** | A ahead through 600M, **B ahead at 1.7B** on arc, xnli, xcopa, xwinograd, lambada, belebele, include, multiblimp | 0.016–0.075 | **reverses at the reference: open** |
| A15 vs B15 | A ahead at 1B on hellaswag, xcopa, xstorycloze, xwinograd, lambada; 1.7B pending | 0.014–0.082 (1B) | open until 1.7B evals land |
| A8 vs A15 | L15 ahead at every size on hellaswag, multiblimp, xcopa, xnli, xstorycloze | 0.013–0.030 | predictable from 175M |
| B8 vs B15 | consistent signs, gaps ≤ 0.009 on gated families | ≤ 0.024 | inside noise |
| A8 vs A30 | L30 ahead from 350M on every gated family | 0.016–0.067 | predictable |
| A15 vs A30 | L30 ahead at every size on hellaswag, multiblimp, xnli, xstorycloze | 0.003–0.044 | predictable |

On BPB the same pairs read differently (A vs B is within noise per language,
L8 vs L30 is the resolvable pair); the paper's DA is on benchmarks, so the
benchmark table decides.

## Options

| option | cells (deep, T = 1, seed 1904) | pairs | node-h | what it tests |
| --- | --- | --: | --: | --- |
| **1. 2×2 (proposed)** | A8, B8, A15, B15 | 6 | ~6,500 | scheme ranking at both L (the open pairs), L8/L15 at both schemes, interaction |
| 2. L-axis + one scheme pair | A8, A15, A30, B15 | 6 | ~6,500 | the resolvable L pairs + one open scheme pair |
| 3. three cells | A8, B8, A15 | 3 | ~4,900 | the open L8 scheme pair, L8/L15, one diagonal |
| 4. none now | — | 0 | 0 | finish the 1.7B evals; decide from the complete 1.7B DA |

| | pros | cons |
| --- | --- | --- |
| **1. 2×2** | both scheme pairs are the ones whose ranking is open at 1.7B (A8/B8 reverses there); every cell has a full 175M–1.7B series behind it; the scheme × L interaction is estimable; four 165B builds all feasible | ~68% of the compute the sweep has kept so far; the L8/L15 pairs add little (predictable or inside noise); the reversal rests on 1.7B evals that are still landing; four concurrent 21-node runs will serialize (2–4 weeks) |
| 2. L-axis + B15 | keeps the L30 pairs, which carry the largest gated-family gaps | those pairs are already ordered the same way from 350M up — a 3B confirms; drops A8/B8, the one pair that moves with size |
| 3. three cells | keeps the open A8/B8 pair and the L8/L15 contrast at −25% cost | loses A15/B15, the second open pair; 3 pairs is a coarse DA |
| 4. none now | free; the 1.7B row's DA is not yet complete, so the "open pairs" list may still change | no extrapolation check in the revision |

## Recommendation

**Option 1, gated on two cheap steps first:**

1. Finish the 1.7B benchmark evals (B-L15 final checkpoint, and every 1.7B cell
   with a mid-run reference) and re-read the pair table. If A8/B8 still reverses
   and A15/B15 is open, the 2×2 is the right set; if both are settled, fall back
   to option 4 and spend the compute on seeds.
2. Build the four 165B mixtures (`data/launch_builds.sh`, the `REBUILD_165`
   tier) — ~1 day each, parallel — so the runs are not blocked on data.
   Submitted 2026-09-20 (singleton chains, `infra01`):

   | Job | Mixture | Builds into | Stages to |
   |---|---|---|---|
   | 3448609 | `build-a-L8-165b` | `predictivity-data/rebuild-165B/fineweb_L8` | `data-165B/fineweb_L8` |
   | 3448610 | `build-a-L15-165b` | `predictivity-data/rebuild-165B/fineweb_L15` | `data-165B/fineweb_L15` |
   | 3448611 | `build-b-L8-165b` | `predictivity-data/rebuild-165B/schemeB/fineweb_L8` | `data-165B/schemeB/fineweb_L8` |
   | 3448612 | `build-b-L15-165b` | `predictivity-data/rebuild-165B/schemeB/fineweb_L15` | `data-165B/schemeB/fineweb_L15` |

Then launch A8 and B8 first (the open pair), A15 and B15 second.

## What is wired (2026-09-19)

| piece | change |
| --- | --- |
| Architecture | `hyperparams_deep.json` 3B: 36 layers × 2816, 44 heads / 11 KV groups, FFN 11264, 2.997B non-emb, LR 5.99e-4, 145,200 iters, warmup 5,800, decay 29,000, 60 checkpoints every 2,420, MBS 1 on 21 nodes |
| Grid | `LADDER` gains 3B; `SIZE_LANG_SETTINGS["3B"] = [8, 15]`; deep only via `arches_for()` (the shallow file has no 3B); seed 1904 only |
| Data | four builds now sized 165B → a second rebuild tier (`rebuild-165B`, staged to `data-165B`); the launcher reads it only for cells the 92B copies cannot feed (`CSCS_REBUILD_DATA_DIRS`) — nothing else moves |
| Evals | `models.json` gains the four cells (sync); the watcher, BPB chain and report pick them up by name; eval walltime uses an extrapolated 0.85 min/task |
| Verified | every non-3B cell's launch command is byte-identical to before; the 3B cells report `skip [data undersized]` until the 165B copies are staged |

Launch, once the data is staged: `python3.11 src/pretrain/launch_trainings.py cscs --size 3B --dry-run`, then the same per scheme without `--dry-run`. The dry-run is queued as job 3449193 (`dryrun-3B`, `src/pretrain/dryrun_after_build.sbatch`): it follows the `build-a-L8-165b` chain segment by segment and writes its output beside the build logs on capstor.

---

# Which 3B cells next (2026-09-30)

The 2×2 above is **trained**: A8, A15, B8, B15 all reached 145,200 iterations.
This section is the follow-on decision, and it is driven by a counting fact
that the options table above never looked at.

## The 2×2 cannot report a single mono-axis decision

Rule 5 needs `MIN_PAIRS = 3` pairs differing on exactly one design axis. Four
cells in a 2×2 give six pairs, but only four are mono-axis, and they split two
and two:

| cells at 3B | `L` | `list` | `T` | axes clearing MIN_PAIRS |
| --- | --: | --: | --: | --- |
| **A8, A15, B8, B15 (today)** | 2 | 2 | 0 | **NONE** |
| + A-L30, B-L30 | 6 | **3** | 0 | **L, list** |
| + A-L50 | 9 | 3 | 0 | L, list |
| + AT3 at L15, L30, L50 | 12 | 3 | **3** | L, list, T |

So rq10 — *does a ranking that holds at 1.7B still hold at 3B* — cannot be
answered for any single design axis with what is on disk. That is a property
of the 2×2's SHAPE, not of the data, and it does not improve with more
evaluation. It is the thing the next cells have to buy.

**A-L30 + B-L30 is the minimum that buys it**, and the only two-cell move that
does: one more cell at any third `L` clears `L` alone and leaves `list`
stranded at 2. A-L50 then takes `L` to 9 and gives the top rung the full
language range. The temperature axis is poor value at 3B — each AT3 cell
contributes exactly one `T` pair, so it needs three of them.

## Every setting can feed a 3B

A 3B draws 150B from the multilingual half, and a T=1 allocation is
**proportional** to what each language has, so every language draws the same
fraction of its own capacity and none is exhausted while the target is under
the total. At a 165B target that fraction is 47%:

| build | total available | languages over capacity at 165B |
| --- | --: | --: |
| A-L30 | 353.6B | 0 of 29 |
| B-L30 | 341.4B | 0 of 29 |
| A-L50 | 373.7B | 0 of 49 |

## First, finish the evaluations — the ranking inputs are stale

`ladder_report.csv` is from **2026-09-28** and predates the 3B finals. On that
snapshot two of the four trained 3B cells carry **zero** benchmark results,
and scheme B is under-evaluated one rung down as well:

| cell | benchmark tasks at final ckpt | eval dirs on disk |
| --- | --: | --: |
| 3B-L8-deep | 3543 | 82 |
| 3B-L15-deep | **0** | 75 |
| 3B-L8-schemeB | **0** | 79 |
| 3B-L15-schemeB | 1116 | 81 |
| 1.7B-L8-schemeB | 951 (A: 3898) | 61 |

This is not missing work — every cell has 75–82 checkpoint directories and the
evals are landing — it is a stale report. But it matters for the choice,
because the premise this document used to rank the scheme axis first (A8/B8
*reverses* at 1.7B) currently reads −0.024 on gated families over only **37 of
202** task columns, the ones where the under-evaluated B cell happens to
overlap. A reversal measured on a biased task subset is not a basis for 3,260
node-hours. The recommendation above already gated the 2×2 on "finish the 1.7B
benchmark evals and re-read the pair table"; that gate was never satisfied,
and it still is not.

**So: republish the report and re-read before launching any training.** The
pair-count table is not affected — it is structural — but which cells are worth
3,260 node-hours does depend on numbers that are two days old and moving.

## What has been changed now (data only)

The builds are long (~1 day each) and nothing depends on the eval outcome, so
they are started while the report catches up:

- `SIZE_LANG_SETTINGS["3B"]` → `[8, 15, 30, 50]`. This is what makes
  `fineweb_target_tokens` size the L30/L50 builds at 165B instead of 92B —
  the target is derived from the grid's largest rung, so the grid has to say
  the 3B trains there before the build can be made for it.
- `DATA_SCHEMES["AT3"]["max_size"]` gains `50: "1.7B"`. AT3 was capped at the
  reference for L15 and L30 but not L50, so opening the 3B rung at L50 would
  otherwise have created an AT3 3B cell nobody planned.
- `REBUILD_165` gains `A:30 B:30 A:50`.

Net: the 3B rung goes from 4 planned cells to **7** — the four trained ones
plus A-L30, B-L30, A-L50. No AT3 cell, no training launched.

Until the builds stage, the three cells are **refused**, not merely skipped:

```
skip [data undersized]: lm-3B-L30-deep-seed1904 — draws 149.9B (1.63 epochs)
  from a 92.0B build the grid now sizes at 165.0B
skip [data undersized]: lm-3B-L50-deep-seed1904 — draws 149.9B (2.88 epochs)
  from a 52.0B build the grid now sizes at 165.0B
```

That is `undersized_build` doing its job: registering the cells cannot start a
3B on repeated data by accident. **Once the builds stage that guard lifts**, and
a bare `launch_trainings.py cscs --size 3B` would submit them — so launch by
`--scheme`/`--langs` while the decision is open.

## Cost

~1,630 node-h of training per cell (measured, 2026-09-22) plus ~60 of eval;
~78 h on 21 nodes ≈ 8 walltime segments ≈ a week queued. A-L30 + B-L30 is
~3,400 node-h all in; adding A-L50 makes it ~5,100.

