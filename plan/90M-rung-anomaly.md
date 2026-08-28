# The 90M rung does not sit on the scaling curve

**Status: cause identified 2026-08-28 — the 90M runs DIVERGE.** They are not
converging to a poor loss; they reach ~4.5 about a fifth of the way in and
then get steadily worse for the remaining 80% of training. The final
checkpoint — the one that is converted, evaluated and shipped — is materially
worse than one already on disk at iter ~800.

This retracts the claim below that the loss "decreases monotonically". It does
not, and that mistake is why the rung looked like a capacity problem for a
week. Everything in "The observation" and "The learning-rate sweep" is still
accurate as *measurements*; their interpretation changes.

Recorded 2026-08-21 from the first complete L1/L2 ladder; cause added
2026-08-28.

## The divergence (2026-08-28)

`ladder_report.py --check loss` compares each run's final loss against its own
best. **9 of the 10 completed 90M runs diverge; no other size does.**

| cell | best loss | at iter | final | delta |
| ---- | --------: | ------: | ----: | ----: |
| 90M-L1-deep     | 4.339 | 840 (18%) | 5.628 | +1.29 |
| 90M-L1-shallow  | 4.697 | 654 (15%) | 5.887 | +1.19 |
| 90M-L2-deep     | 4.397 | 897 (19%) | 5.762 | +1.36 |
| 90M-L15-deep    | 4.595 | 698 (15%) | 6.282 | +1.69 |
| 90M-L30-deep    | 4.475 | 766 (17%) | 6.389 | +1.91 |
| 90M-L50-deep    | 4.538 | 790 (17%) | 6.349 | +1.81 |

Every one peaks between iter 600 and 900. 90M-L1-deep spikes to **13.27** at
iter 901 — worse than initialisation (11.90). For contrast, 175M-L2 takes one
spike (6.94 at iter 1709) and fully recovers, and 350M/600M reach their best
loss at >90% of the run with no spike at all.

On a single-epoch budget there is no overfitting available to explain this.

**Held-out data confirms it independently.** `score_bpb.py` on
90M-L2-deep, iter 450 vs iter 4500:

| | English (dclm) | Russian | macro over 100 langs |
| - | ---: | ---: | ---: |
| iter 450  | 1.675 | 1.171 | 3.72 |
| iter 4500 | 1.954 | 1.445 | 12.52 |

BPB gets **worse** with training, and per-language perplexity on unseen
languages blows past 1e32 — the output distribution has degenerated, not
merely failed to improve.

## What it is not: the learning rate

The obvious reading is that 90M gets the ladder's hottest LR (1.4276e-3 at
90M, falling to 8.976e-4 at 600M, because peak LR is derived per run from its
own compute budget). **The diagnostic runs rule this out.** Re-reading them
for divergence rather than final loss:

| LR | best loss | at iter | final | verdict |
| -- | --------: | ------: | ----: | ------- |
| 3e-4 (4.8x below production) | 4.504 | 1121 (24%) | 5.833 | **diverges** (spike to 8.71) |
| 6e-4 | 4.569 | 479 (10%) | 5.032 | **diverges** |
| 1.4276e-3 (production) | 4.397 | 897 (19%) | 5.762 | **diverges** |

Every LR reaches ~4.5 and then degrades. LR changes how *badly* it degrades,
not *whether* it does. A step-size problem would be cured by a 4.8x smaller
step; this is not.

## Leading hypothesis: the run is shorter than the optimizer's memory

`megatron_args.sh` sets `--ademamix-beta3 0.9999`, whose slow-EMA timescale is
1/(1-0.9999) = **10 000 steps**, with alpha ramping to 8 over the full run
(`ADEMAMIX_WARMUP` = the cell's target iters). Against each rung's length:

| size | iters | iters / 10k timescale | outcome |
| ---- | ----: | --------------------: | ------- |
| 90M   |  4 500 | **0.45x** | diverges at every LR tried |
| 175M  |  8 540 | **0.85x** | one spike, recovers |
| 350M  | 16 660 | 1.7x | clean |
| 600M  | 28 800 | 2.9x | clean |
| 1B    | 45 740 | 4.6x | not yet run |
| 1.7B  | 81 000 | 8.1x | not yet run |

The severity ordering is exact, and it is the only 90M-specific quantity found
so far that is not also true of the healthy rungs. The mechanism would be that
alpha grows to 8 on a slow-momentum buffer that has never equilibrated — the
90M finishes training having seen less than half of its optimizer's own
memory — so the update is increasingly dominated by a stale direction.

**This is a hypothesis, not a demonstrated cause.** The evidence is the exact
severity ordering plus the elimination of LR. One run confirms or kills it:
90M-L2-deep with `ADEMAMIX_BETA3` set so the timescale is a fixed fraction of
the run (see "Fixing it" below). If it still diverges, the cause is elsewhere
and hypotheses 2-4 come back into play.

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

## Fixing it while keeping a ladder

The constraint that rules out most fixes: a ladder is only a ladder if every
rung runs the *same procedure*. Anything applied to 90M alone turns the bottom
point into a different experiment, which is exactly what a scaling fit cannot
absorb.

### A. Tie beta3 to the run length (recommended)

`beta3 = 1 - 1/(f x train_iters)`, one `f` for the whole ladder — e.g. f=0.2
puts the slow-EMA timescale at 20% of every run: 90M 0.99889, 1.7B 0.999938.

* **Pro** — this makes the ladder *more* controlled, not less. A fixed beta3
  across runs from 4.5k to 81k steps means the optimizer has a qualitatively
  different memory at each rung; that is an uncontrolled variable the ladder
  is currently carrying silently. Tying it to the run length removes it.
* **Pro** — fixes 175M's marginal spike (0.85x) at the same time.
* **Pro** — cheap where it matters: 90M and 175M are 19 and 48 node-hours for
  a full L1..L50 deep row.
* **Con** — strictly, every rung's optimizer changes, so 350M/600M would need
  re-running for exact comparability (~525 node-hours).
* **Mitigation** — 350M and 600M are already at 1.7x and 2.9x, where the
  change is small by construction. Run ONE 350M control with the new beta3; if
  it lands within noise of the existing run, keep 350M+ as they are and
  document the control. That caps the real cost at ~70 node-hours plus one
  control.

### B. Shorten the optimizer memory globally (fixed smaller beta3)

Pick a single smaller beta3 (e.g. 0.999, timescale 1000 steps) for every rung.

* **Pro** — one constant, no formula; the ladder stays trivially uniform.
* **Con** — throws away the long-horizon momentum that is the point of
  AdEMAMix at the large rungs, where nothing is broken. Fixing the small end
  by degrading the large end is the wrong trade for a study whose expensive
  rungs are 1B and 1.7B.

### C. Drop 90M from the ladder

* **Pro** — free, and defensible if the rung is genuinely below the capacity
  threshold.
* **Con** — the evidence now says it is *not* a capacity limit: 90M reaches
  4.4 before degrading, against a trend-implied ~3.5. That is a training
  failure, not a floor. Dropping it would discard a recoverable anchor, and
  175M (0.85x) is likely mildly affected too, so the next rung up is not a
  clean substitute.

### D. Extend the 90M run past D = 100xN

* **Con** — breaks the definition of the ladder (every rung at 5x Chinchilla).
  It would trade a training bug for a confound in the headline axis. Rejected.

### Recommendation

**A, staged.** (1) One 90M-L2-deep run with beta3 tied to the run length — it
confirms or kills the hypothesis for ~2 node-hours. (2) If confirmed, adopt
the formula, re-run the 90M and 175M rows, and run one 350M control to justify
keeping 350M+ as they are. (3) Hold the queued 90M jobs until (1) reports;
they will diverge exactly like the ten already on disk.

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
