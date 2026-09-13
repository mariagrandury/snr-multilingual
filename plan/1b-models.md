# The 1B row: two checkpoint schedules, and whether that matters

*Written 2026-09-03. Covers the 1B cells trained by `aromanou` from 2026-09-02
alongside the two trained under `mariagrandury`.*

## TL;DR

Fourteen 1B jobs (thirteen cells with checkpoints so far) were launched from
a clone predating three grid decisions.
They train to **45,740 iterations saving every 2,287** (20 checkpoints); the
current code says **45,720 saving every 1,143** (40 checkpoints). The two
grids are incommensurate — `2287 != 2 x 1143` — and the targets differ by 20
iterations.

**This is almost entirely harmless.** Token counts at matched checkpoints
differ by a constant **0.044 %**, and the 20 checkpoints land exactly on the
shared `k/20` grid that every other size uses. Cross-size comparability, the
scaling fit, and the seed-variance estimate are all unaffected.

The one real loss is the *extra* density the 1B reference row was meant to
carry (40 points, every 2.5 % of training). That is not recoverable after
the fact — the intermediate checkpoints were never written — and it splits
the noise estimate within the 1B row until the row is read on a common
subset.

**Decision: do not retrain. Let the runs finish on their own schedule, and
do not pull mid-flight.** The off-grid seed/language combinations are being
adopted into the grid rather than discarded.

**Open question — the one decision left:** the noise estimator is already
fixed at *5 late checkpoints*, and that window scales with checkpoint count,
so it spans the final **25 %** of training on a 20-checkpoint run against
**12.5 %** on a 40-checkpoint one. The 1B row therefore has to be made
self-consistent. The cheap fix is to read the whole 1B row on the shared
`k/20` subset; the expensive one is retraining. See "The one real problem".

## Decision

| | |
|---|---|
| Retrain her 1B cells? | **No** |
| Let them finish on 45,740 / 2,287? | **Yes** — do not pull mid-flight |
| Adopt the off-grid seeds/languages into the grid? | **Yes** (`seeds_for()`) |
| Keep `mariagrandury`'s two cells at 40 checkpoints? | **Yes** — denser is strictly more information |
| Read the 1B row's noise on the shared `k/20` subset? | **Yes** — the only way to make the row self-consistent without retraining |

Retraining the nine scheme-A cells would cost roughly the ~2,185 node-hours
already spent, to buy a 0.044 % token alignment and 2x density. That is not
a good trade unless the paper's central claim turns on step-to-step noise
measured at 2.5 % resolution *at 1B specifically*.

## What is actually on disk

Fourteen 1B jobs ran or are running under `aromanou` (thirteen have written
checkpoints; two scheme-B cells were still queued), each on 21 nodes
(~253 nodes committed, ~2,185 node-hours consumed as of 2026-09-03 12:50).
All markers point at valid checkpoints — `.metadata` plus 168 `.distcp`
shards each, no async-save shell directories.

| Cell | Iter | % of 45,740 | Saves | State |
|---|---:|---:|---:|---|
| `lm-1B-L2-deep-seed28` | 34,305 | 75 % | 15 | running |
| `lm-1B-L2-deep-seed1797` | 34,305 | 75 % | 15 | running |
| `lm-1B-L2-deep-seed1904` | 34,305 | 75 % | 15 | running |
| `lm-1B-L30-deep-seed28` | 32,018 | 70 % | 14 | running |
| `lm-1B-L30-deep-seed1797` | 32,018 | 70 % | 14 | running |
| `lm-1B-L30-deep-seed1904` | 32,018 | 70 % | 14 | running |
| `lm-1B-L1-deep-seed28` | 29,731 | 65 % | 13 | running |
| `lm-1B-L1-deep-seed1797` | 27,444 | 60 % | 12 | running |
| `lm-1B-L1-deep-seed1904` | 27,444 | 60 % | 12 | running |
| `lm-1B-L8-schemeB-deep-seed1904` | 27,444 | 60 % | 12 | running |
| `lm-1B-L15-schemeB-deep-seed1904` | 18,296 | 40 % | 8 | running |
| `lm-1B-L30-schemeB-deep-seed28` | 11,435 | 25 % | 5 | running |
| `lm-1B-L15-deep-seed1904` | 22,870 | 50 % | 10 | **NODE_FAIL, stalled** |

`lm-1B-L30-schemeB-deep-seed1797` and `-seed1904` are queued with no
checkpoints yet. Under `mariagrandury`: `lm-1B-L8-deep-seed1904` (42,194)
and `lm-1B-L50-deep-seed1904` (41,614), both on the *current* 45,720 / 1,143
schedule and both needing a resume. `lm-1B-L100-deep-seed1904` has never
started.

## The three deviations

All are consistent with a clone predating the relevant commits; the runs
started 2026-09-02 19:57, after all three had been pushed to the branch.

| # | Current code | These runs | Introduced by |
|---|---|---|---|
| 1 | 45,720 iters, save every **1,143** (40 ckpts) | 45,740, every **2,287** (20 ckpts) | `db52eb9` *Denser reference checkpoints: 40 at 1B* (2026-08-23) |
| 2 | `SEED_TRIPLE = [64, 313, 1904]` | seeds **28, 1797, 1904** | the 36-sweep seed set |
| 3 | x3 row at 175M/600M, `L in {1,2,50,100}` | x3 at **1B**, `L in {1,2,30}` | `70c2aa7` *grid: move the x3 row to L50* (2026-09-02) |

Deviations 2 and 3 are label questions, not physics: seed identity is
arbitrary, and what the x3 row measures is spread across three seeds. They
are resolved by adopting the combinations into `seeds_for()` rather than by
re-running anything. Deviation 1 is the substantive one.

## Why 0.044 % is the whole numerical story

`tokens_per_iter = 2,064,384` for every size in the ladder (GBS 504 x seq
4096), so the arithmetic is exact.

| | 45,740 schedule | 45,720 schedule | Delta |
|---|---:|---:|---:|
| Total tokens | 94.42 B | 94.38 B | 41.3 M |
| Relative | — | — | **0.044 %** |
| Checkpoint *k* of 20 | `k x 2287` | `k x 2286` | `k` iters |

The offset is a constant 0.044 % at *every* matched checkpoint, not a drift
that accumulates. It sits orders of magnitude below seed-to-seed variance,
and the analysis plots against tokens or FLOPs rather than raw iteration
index, so the two grids overlay cleanly.

## Why cross-size comparability is intact

`n_checkpoints()` returns 20 for 90M-600M and 40 at 1B specifically so that
every size lands on a shared `k/20` grid — 40 and 60 are multiples of 20, so
the denser rungs *add* points between the shared ones rather than moving
them:

> 40 and 60, being multiples of 20, keep every size on the shared k/20 grid
> (1B: every 2nd checkpoint, 1.7B: every 3rd).

A 20-checkpoint 1B run therefore has exactly the grid every other size has.
It is missing the interleaved extras, not sitting on a different grid. Every
one of its checkpoints has a counterpart in a 40-checkpoint run (her
`k x 2287` against your `2k x 1143 = k x 2286`), within that same 0.044 %.

What does **not** transfer is step-to-step noise. Variance between adjacent
checkpoints depends on their spacing, and hers are 2x further apart, so a
naive adjacent-checkpoint noise estimate is systematically larger. Scores
compare freely; noise does not, without correcting for spacing.

Seed variance is unaffected: all three seeds at each `L` share an identical
45,740 / 2,287 schedule, so the x3 row measures exactly what it was built to
measure.

## The one real problem: the noise window

The estimator is **not** an open choice — `plan/small-to-large-predictivity-training-plan.md`
settles it:

> Note the deviation from "the last 30": with 20 checkpoints per run (40/60
> at the reference rungs) the whole grid is smaller than that, and the dense
> tail is 5. Checkpoint noise is therefore estimated over 5 late
> checkpoints, not 30.

Five is an *absolute count*, so the window it covers scales with the
checkpoint count:

| Checkpoints | Last 5 span | As a fraction of training |
|---:|---|---:|
| 20 | 80 % -> 100 % | final **25 %** |
| 40 | 90 % -> 100 % | final **12.5 %** |
| 60 | 93 % -> 100 % | final **8.3 %** |

This is inherent to the current design, not something these runs introduced:
the noise window already differs across the ladder by construction. But it
does mean the 1B row is now internally split — `L1`/`L2`/`L30` would have
their noise measured over the final 25 % of training at 2,287-iteration
spacing, while `L8`/`L50` use the final 12.5 % at 1,143. Noise measured over
a wider, coarser window is systematically larger, so those two groups are
not directly comparable on the noise axis.

Note that this makes the affected cells consistent with **every other size**
(90M-600M all have 20 checkpoints and therefore the same 25 % window) and
inconsistent only with the two 1B cells that got the intended denser
sampling.

**The fix costs nothing.** The plan already anticipates the mechanism:

> The odd late checkpoints are converted to HF and kept, so the
> checkpoint-noise window can be densified later by lowering `--every`
> without retraining.

That works downward as well as upward. Reading the whole 1B row on the
shared `k/20` subset — every 2nd checkpoint of the 40-checkpoint cells —
gives all 1B cells 20 points, the same 25 % window, and consistency both
within the row and with the rest of the ladder. What is forfeited is the
extra 1B resolution `db52eb9` intended; retraining is the only way to
recover that, because the intermediate checkpoints were never written.

One thing that survives intact either way: the **1xC operating point**
(`train_iters / 5`) lands exactly on-grid at checkpoint *n*/5 on both
schedules — iteration 9,148 (checkpoint 4 of 20) and 9,144 (checkpoint 8 of
40). The plan's "checkpointing at defined token counts" requirement holds
for these runs.

## A note on the analysis code

The SNR analysis for our own ladder is not written yet.
`src/signal-and-noise/snr/mask_analysis.py` and everything under
`allenai_analysis/` operate on AllenAI's DataDecide parquet
(`pull_predictions_from_hf("allenai/signal-and-noise")`), and both hard-code
a `[-30:]` / `int(-0.3*len(...))` window rather than the 5 this ladder uses.
The only own-model path, `snr/download/apertus.py`, still matches
`apertus-<size>-fwEdu<N>-fw<M>-seed<S>` — the finished 36-sweep naming — and
will not pick up any `lm-*` cell. Whoever writes the ladder-side analysis
needs to implement the 5-checkpoint window explicitly; neither existing
implementation does.

## Why not to pull mid-flight

`SAVE_INTERVAL` is a plain `--save-interval` CLI argument re-exported on
every submission, not read from the checkpoint, so a resumed job starts
saving every 1,143 immediately — and not retroactively. Because
`2287 != 2 x 1143`, the new points land one iteration off the old grid per
step, leaving the cell on *neither* schedule.

The resume itself would not crash: `megatron_args.sh` pins
`--use-checkpoint-opt_param-scheduler`, so the saved LR schedule wins and
the loop exits 20 iterations early (see `src/pretrain/CLAUDE.md` #5). But
`ADEMAMIX_WARMUP` would also shift 45,740 -> 45,720, and that file is
explicit that the warmup should stay at the target across resumes so every
submission runs the same optimizer schedule.

A clean coarse grid beats a mixed one. Let the runs finish.

## Follow-ups

1. Adopt the off-grid combinations in `seeds_for()`
   (`src/pretrain/launch_trainings.py`) — a 1B-specific row, *not* a widened
   `SEED_TRIPLE`, which would demand five seeds per existing x3 cell. Twelve
   cells at 175M/600M are already trained with seeds 64/313.
2. Read the 1B row's checkpoint noise on the shared `k/20` subset, so the
   row is self-consistent and matches the rest of the ladder.
3. Re-point `snr/download/apertus.py` at the `lm-*` naming, and implement
   the 5-late-checkpoint noise window — no existing code path does.
4. Resume the three stalled scheme-A cells:
   ```bash
   cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
   for L in 8 15 50; do
     python3.11 launch_trainings.py cscs --dry-run --arch deep --size 1B --seed 1904 --langs $L
   done
   ```
   Drop `--dry-run` to submit. Do not omit `--langs`: that also picks up
   `L100` as `[fresh]` and launches a fourth 21-node job. Note `L15` is on
   the 2,287 schedule, so resuming it mixes that one cell.
5. `175M-L30` also has off-grid `seed28`/`seed1797` cells on disk — same
   situation, not covered by follow-up 1. Decide separately.
