# The 90M rung does not sit on the scaling curve

**Status: open.** The 90M models train cleanly and reach a loss far worse than
the ladder predicts, and score at chance on every benchmark. Every rung from
175M up is healthy. This matters because 90M is the bottom anchor of the
predictivity fit: a degenerate point there distorts every extrapolation built
on it.

Recorded 2026-08-21 from the first complete L1/L2 ladder.

## The observation

Final `lm loss` (all cells trained to their own D = 100·N budget, LR fully
decayed, exact token counts):

| arch    | L | 90M       | 175M  | 350M  | 600M  |
| ------- | - | --------- | ----- | ----- | ----- |
| deep    | 1 | **5.628** | 3.161 | 2.731 | 2.591 |
| deep    | 2 | **5.762** | 2.904 | 2.480 | 2.297 |
| shallow | 2 | **4.846** | 3.071 | 2.474 | 2.289 |

From 175M upward the steps are textbook diminishing returns: −0.42 then −0.18
per doubling (deep L2). The 90M → 175M step is **−2.86 for 1.9× parameters** —
six times larger, and it reproduces in all three ladders.

Evaluation agrees. Over 10 checkpoints per cell on the `auto` benchmark set:

* **175M-L2-shallow** learns: `arc_easy` 0.332 → 0.449 (chance 0.25),
  `xnli_en` 0.334 → 0.409, `xwinograd_en/ru` +0.035/+0.048.
* **Both 90M cells** sit at or below chance on all 16 tasks, and most deltas
  are *negative* over training (`arc_easy` 0.306 → 0.295, `xwinograd_en`
  0.516 → 0.490).

Loss 5.76 is perplexity ≈ 317, so the 90M has learned the token distribution
(from 11.78 = ln 131072 at init) but nothing that transfers to a task.

## What is already ruled out

* **Not a training bug.** Initial loss matches ln(vocab); loss decreases
  monotonically; **zero NaN and zero skipped iterations**; small final grad
  norms; LR reaches 0; token counts exact.
* **Not caused by any recent change.** Every number reproduces across two
  independent generations — old architecture/LR/capstor data vs new
  architecture/LR/iopsstor data — to within 0.06 nats (90M-L2: 5.746 → 5.762;
  175M-L2-shallow: 3.012 → 3.071). Flash attention, the storage move, the 6ND
  learning rate and the rebuilt shallow ladder are all exonerated.
* **Not the architecture family.** Both 90M variants fail: deep (h=768) at
  5.762 and shallow (h=1024) at 4.846.
* **Not solely the learning rate** — see below.

## The learning-rate sweep

Diagnostic runs on `90M-L2-deep`, identical to the production cell except LR
(jobs 3138085 / 3138086, run names `diag-90M-L2-deep-lr*`):

| LR                     | final loss |
| ---------------------- | ---------- |
| 3e-4                   | 5.833      |
| **6e-4**               | **5.032**  |
| 1.4276e-3 (production) | 5.762      |

The 6ND-law LR is **~2.4× too hot at 90M**, worth 0.73 nats — a real and
fixable problem, and evidence the law overshoots increasingly as N shrinks.
But the optimum found so far still leaves 90M at 5.03 against a trend-implied
~3.4–3.6, so **roughly 1.7 nats remain unexplained**. LR alone does not rescue
the rung.

## Hypotheses still open

1. **The LR law is systematically too hot, at every size.** If 175M also
   improves at a lower LR, the whole ladder is mistuned and the 90M is simply
   where it hurts most. This is the highest-stakes possibility: it would mean
   re-tuning before the expensive rungs run.
2. **Vocabulary bottleneck.** At h=768 the 131 072-token vocabulary dominates
   the model: ~201M embedding parameters against a 93M transformer body, over
   2:1. Widening from 768 to 1024 buys ~0.9 nats, which is consistent — but
   deep-175M at the same h=1024 reaches 2.90, so width is not the whole story.
3. **Capacity threshold.** The rung may sit below the point where this vocab
   and data mixture support task-transferable structure, in which case 90M is
   not recoverable and should be dropped from the fit rather than repaired.
4. **Budget-limited, not capacity-limited.** D = 100·N is only 9.3B tokens at
   90M. If the model is merely undertrained, more tokens would keep improving
   it — distinguishable from (3) by a single longer run.

## Jobs proposed to close this out

All are 90M/175M scale: 3 nodes (~1.6 h) or 6 nodes (~1.9 h) per run, so the
whole study is well under 20 node-hours. Run them as `diag-*` cells so they
never collide with grid checkpoints or W&B run ids.

| # | Run | Purpose | Decides |
| - | --- | ------- | ------- |
| 1 | `90M-L2-deep` at LR 8e-4 and 1e-3 (2 jobs) | bracket the optimum between the measured 6e-4 and 1.4276e-3 | the best achievable 90M loss, and how far the 6ND law overshoots |
| 2 | `175M-L2-deep` at LR 6e-4 (1 job) | is the 175M also mistuned? | **hypothesis 1** — if it improves materially, re-tune the LR law for the whole ladder before the expensive rungs |
| 3 | `90M-L2-deep` at 2× tokens (9000 iters), best LR from (1) | undertrained vs capacity-limited | **hypotheses 3 vs 4** |
| 4 | `90M` at h=1280 (~5 layers, same non-emb budget), best LR | isolate width at fixed parameters | **hypothesis 2** — a third width point after 768 → 1024 |

Run (1) and (2) first: they are the cheapest and (2) is the one that could
change the plan for every remaining rung. (3) and (4) are only worth running
if the gap survives the corrected LR.

**Decision to make once these land:** if the 90M cannot be brought near the
curve, drop it from the ladder rather than let it anchor the predictivity fit,
and record the exclusion — the fit is the deliverable, and a degenerate bottom
rung biases the slope everywhere.
