# The 90M rung does not sit on the scaling curve

<!-- ------------------------------------------------------------------ -->
<!-- APPENDIX DRAFT — self-contained, paper-ready. Everything below the  -->
<!-- "Working record" divider is the internal investigation log.         -->
<!-- ------------------------------------------------------------------ -->

## Appendix: the smallest rung of the ladder

Our predictivity study trains a ladder of models at six parameter counts,
each on its own compute-optimal-multiple budget, and asks how benchmark
rankings evolve with compute. A ladder of this kind is only as good as its
span: the bottom rung sets the left-hand anchor of every scaling fit, and a
displaced anchor tilts the fitted slope across the whole range. We therefore
examined the smallest rung, at 90M non-embedding parameters, before using it.

The 90M rung does not lie on the curve traced by the larger rungs, and the
reason is not that the models are too small. Nine of the ten completed 90M
runs *diverge*: each reaches its best training loss between 15% and 19% of
the way through training and then degrades monotonically for the remaining
four fifths, ending between 1.2 and 1.9 nats worse than its own best. No
larger rung shows this. The 175M runs take at most a single loss spike and
recover; the 350M and 600M runs reach their best loss after more than 90% of
training, as expected under a warmup-stable-decay schedule.

Two observations rule out the obvious explanations. First, this is not
overfitting: every run is single-epoch, so no example is seen twice. Second,
it is not a capacity floor. A capacity-limited model converges to a poor
loss and stays there; these models reach roughly 4.4 — against a value near
3.5 implied by extrapolating the larger rungs — and then get worse. Held-out
evidence confirms the degradation is real rather than an artefact of the
training objective: bits-per-byte measured on a disjoint validation set rises
with training on every language we checked (English 1.68 to 1.95, Russian
1.17 to 1.45 between 10% and 100% of training), and per-language perplexity
on the rarest languages diverges outright. The output distribution
deteriorates; it does not merely stop improving.

Nor is it the learning rate. Our peak learning rates come from a compute-based
scaling law evaluated at each run's own budget, which is hottest at the small
end. We trained the 90M rung at three learning rates spanning a factor of
nearly five, down to a value well below the law's prescription. All three
diverge, with the same signature and at the same point in training.

The explanation consistent with all of the evidence is an interaction between
the optimizer's memory and the length of the run. We train with AdEMAMix,
which augments the usual momentum with a second, much slower exponential
moving average of past gradients, governed by a decay coefficient beta3. That
coefficient sets a timescale, of order 1/(1 - beta3) steps, over which the
slow average accumulates. We held beta3 fixed across the ladder — the natural
choice for keeping rungs comparable — at a value whose timescale is 10,000
optimizer steps. But the rungs differ enormously in length, because each
trains on a budget proportional to its own parameter count:

| rung | training steps | steps / optimizer timescale |
| ---- | -------------: | --------------------------: |
| 90M   |  4,500 | 0.45 |
| 175M  |  8,540 | 0.85 |
| 350M  | 16,660 | 1.7 |
| 600M  | 28,800 | 2.9 |
| 1B    | 45,740 | 4.6 |
| 1.7B  | 81,000 | 8.1 |

The 90M run is less than half the optimizer's own averaging window. It never
reaches the regime the optimizer was configured for; the slow average remains
dominated by gradients from early training, and is applied with a large
multiplier throughout. The ordering of this ratio matches the ordering of the
observed severity exactly: 90M diverges badly, 175M shows a single recovered
spike, and everything at 1.7 and above is clean. A fixed beta3 across a ladder
whose rungs differ 18-fold in length is thus not the neutral choice it appears
to be — it silently gives each rung a qualitatively different optimizer.

We tested this directly. In a control run at the 90M rung we set beta3 so
that the optimizer's timescale is a fixed fraction of the run (one fifth)
rather than a fixed number of steps, changing nothing else. The divergence
disappears completely:

| 90M, 2 languages | best loss | final loss | drift |
| ---------------- | --------: | ---------: | ----: |
| fixed timescale (10,000 steps) | 4.397 | 5.762 | +1.365 |
| timescale = 0.2 x run (900 steps) | 2.767 | **2.778** | **+0.011** |

The drift falls to that of a healthy rung, and the final loss improves by
2.98 nats — **1.62 nats better than the uncorrected run's own best**, so the
rung had been discarding not merely its last four fifths but most of the
value of the entire run.

We also checked the competing explanation, that our compute-based learning
rate law is simply too hot at the small end, by retraining the 175M rung at
roughly half its prescribed rate. That made it *worse* (2.904 to 3.015) and it
had not been diverging in the first place, so the law is not the cause and we
left it unchanged for the remaining rungs.

One consequence of the correction deserves care. The corrected 90M reaches a
lower loss (2.778) than the uncorrected 175M (2.904) despite training on half
as many tokens. A smaller model should not outperform a larger one on the same
data, so this is evidence that the 175M rung — at 0.85 of the optimizer
timescale, the second-shortest run — is also depressed, though not visibly
diverging. A control at 175M with the same correction is running. Until it
reports, the ladder's absolute losses below roughly 350M should be read as
upper bounds on those architectures' loss rather than as measurements of it.

**Treatment in the reported analysis.** *[Completed at revisit.]* Where the
90M rung is excluded, it is excluded as a rung whose optimizer configuration
is known to be mismatched to its run length — a documented and reproducible
training defect — and not as an unexplained outlier. We report the ladder both
with and without it so the effect on the fitted slope is visible. Correcting
it for the whole ladder would require retraining every rung, since the
correction changes the optimizer at all of them; that was outside the compute
budget of this study, and we note it as a recommendation for future ladders:
**express every optimizer timescale as a fraction of the run, not as a fixed
number of steps.**

Auditing the rest of the configuration against that rule, beta3 turned out to
be the extreme case rather than the only one. Because each rung trains a
budget proportional to its own parameter count, run lengths span 18x, and
three further constants are written in steps: the second-moment decay (a
1,000-step timescale, 22% of the shortest run and 1% of the longest), the
first-moment decay (10 steps, negligible at every rung), and decoupled weight
decay, which applies once per step and therefore shrinks the largest rung 18x
more in total than the smallest. None is implicated in the divergence, and we
did not vary them; we record them because a ladder that compares across run
lengths silently varies any quantity defined in steps, and the reader should
know which ones those were.

<!-- ------------------------------------------------------------------ -->
<!-- Working record — internal. Not for the paper.                      -->
<!-- ------------------------------------------------------------------ -->

## Working record

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

### Recommendation (superseded — see the decision below)

**A, staged.** (1) One 90M-L2-deep run with beta3 tied to the run length — it
confirms or kills the hypothesis for ~2 node-hours. (2) If confirmed, adopt
the formula, re-run the 90M and 175M rows, and run one 350M control to justify
keeping 350M+ as they are. (3) Hold the queued 90M jobs until (1) reports;
they will diverge exactly like the ten already on disk.

## Decision (2026-09-01): defer, do not retrain

Step (1) is being done. Steps (2) and (3) are **not**.

The schedule is tight and the priority is finishing the planned grid for
90M..600M. Adopting option A would change the optimizer at every rung, which
means retraining everything already on disk — 24 cells — to keep the ladder
internally comparable. That is not affordable now, and it is not what the
paper needs: the ladder can be reported with the 90M rung's behaviour
*explained* rather than *fixed*.

Concretely:

* **The grid keeps its current config.** `lm-90M-L100-deep-seed1904` will be
  trained with beta3 = 0.9999 like the other ten 90M cells, and will diverge
  like them. That is deliberate — a rung trained differently from its own row
  would be worse than a rung that is uniformly wrong.
* **90M stays in the grid.** Whether it enters the scaling fit is an
  analysis-time decision made from the trained curves, not a training-time
  one. Excluding it early would throw away the evidence that justifies the
  exclusion.
* **Two diagnostics settle the cause** (below), so the appendix can say the
  divergence is understood and attributable rather than unexplained. That
  distinction is the entire return on the ~11 node-hours.
* **Revisit 2026-09-08.** If the 90M..600M grid finishes early and the 175M
  diagnostic shows the second-shortest rung is also affected, retraining the
  175M row becomes worth discussing. Otherwise the appendix stands. The
  revisit must also decide the pending 350M diagnostic (see "TODO: the fourth
  diagnostic" below) — that one is gated on the 175M result, not on the
  calendar, so it may come due sooner.

## The config invariant

Options A and B both violate a rule this repo now enforces mechanically:
**a training run must reproduce the config the already-pretrained cells
used.** The comparability argument at the top of this section is why, and 24
trained cells are what is at stake.

So the experimental knobs added for the diagnostics —
`launch_trainings.py --lr` and `--ademamix-beta3-factor` — are opt-in and
never defaults. `megatron_args.sh` keeps `--ademamix-beta3 ${ADEMAMIX_BETA3:-0.9999}`,
so a run that does not set the variable is unchanged; the launcher emits the
variable only when the flag is passed. Any run that passes either flag is
**renamed `diag-*`**, which is not optional: `diag-` matches neither
`pretrain_progress.NAME_RE` nor `ladder_report.LOG_RE`, and
`sync_models_json` builds its keys from `exp_name()`, so a non-standard run
is structurally unable to be mistaken for a ladder rung, occupy a grid cell's
checkpoint directory, or reuse its W&B run id. The flags also require a
`--size/--langs/--seed` filter, so one of them cannot fan a non-standard
config across the whole grid by accident.

Verified by diffing `launch_trainings.py cscs --dry-run` output across the
change: byte-identical for a normal launch.

The `diag-` prefix hides these runs from the grid tooling but **not** from
durable storage — `mirror_eval_logs.sbatch` globs `Meg-Runs/msnr/*/logging`,
deliberately not `lm-*`, so the diagnostics reach capstor with everything
else. They are the evidence this appendix rests on.

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

### Launched 2026-09-01

Per the decision above, only the two cheapest run, and the beta3 control
replaces job (1)'s LR bracket as the first question — the LR sweep already
showed all three rates diverge, so beta3 is the live hypothesis:

| run | command | tests |
| --- | ------- | ----- |
| `diag-90M-L2-deep-seed1904-beta3f0.2` | `launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --ademamix-beta3-factor 0.2` | beta3 = 0.998889 (timescale 900 steps vs the 4,500-iter run). Does the divergence disappear? |
| `diag-175M-L2-deep-seed1904-lr0.0006` | `launch_trainings.py cscs --size 175M --langs 2 --seed 1904 --lr 6e-4` | job (2): is the 175M — a rung we are keeping, at 0.85x — also mistuned? |

Compare against `lm-90M-L2-deep-seed1904` / `lm-175M-L2-deep-seed1904` in
W&B (project `msnr`, run name = the cell name) or the raw
`logs/slurm/training/pretrain-diag-*.out`. **`ladder_report.py --check loss`
will not show them** — its `LOG_RE` requires the `lm-` grid shape, which is
the same exclusion that keeps them out of the ladder.

Findings go in the appendix at the top of this file, replacing its two
placeholder paragraphs.

### Results (2026-09-01, both runs complete)

Both trained their full schedule (4500/4500 and 8540/8540) and wrote their
final checkpoints. Slurm reports CANCELLED because the tail was killed after
training finished — the runs themselves are complete.

| run | best | final | drift |
| --- | ---: | ----: | ----: |
| `lm-90M-L2-deep-seed1904` (grid, beta3 0.9999) | 4.397 | 5.762 | +1.365 |
| `diag-90M-L2-deep-seed1904-beta3f0.2` (0.998889) | 2.767 | **2.778** | **+0.011** |
| `lm-175M-L2-deep-seed1904` (grid, LR 1.217e-3) | 2.870 | 2.904 | +0.034 |
| `diag-175M-L2-deep-seed1904-lr0.0006` | 2.976 | 3.015 | +0.039 |

**beta3 confirmed, and the effect is far larger than this doc predicted.**
The divergence does not shrink, it disappears: +1.365 -> +0.011, the drift of
a healthy rung. Final loss 5.762 -> 2.778, which is 1.62 nats better than the
diverged run's *best*. The hypothesis in "Leading hypothesis" above is
correct.

**The LR explanation is dead.** Halving the 175M's LR made it worse and it
was never diverging, so the 6ND law is not too hot at the small end. This
closes the open question flagged at `compute-budget.md:460` — do not re-tune
the law for the remaining rungs. It also retires proposed jobs (1), (3) and
(4) in the table above: they all presuppose an LR problem.

**New question, sharper than the original.** The corrected 90M (2.778) beats
the grid 175M (2.904) on half the tokens. A 90M should not outperform a 175M
on identical data, so the 175M is very likely depressed too — consistent with
its 0.85x ratio and with the "fixes 175M's marginal spike" claim in option A.
`diag-175M-L2-deep-seed1904-beta3f0.2` (job 3256931, beta3 0.99941452) is
running to settle it. If it lands materially below 2.904, the annex framing
"exclude 90M, the ladder starts at 175M" does not survive either, and the
question becomes how far up the ladder the effect reaches.

Caveats: these are TRAINING losses, and the ladder's stated metric is held-out
BPB. The diag checkpoints are not converted or BPB-scored — diag cells have no
models.json entry, so the eval path skips them; that needs doing by hand
before any of this is quoted as a held-out result. One cell (L2), one seed.

### Why a third diagnostic (`diag-175M-L2-deep-seed1904-beta3f0.2`)

Job 3256931, beta3 0.99941452 (= 0.2 x 8540 steps), LR left at the grid value.
It exists because of one number: the CORRECTED 90M reaches 2.778 while the
grid 175M reaches 2.904, on half the tokens. A 90M outperforming a 175M on
identical data is not a result, it is a symptom — the most economical
explanation is that the 175M is impaired by the same mechanism, just less
visibly (0.85x of the timescale, +0.034 drift, no divergence, but a depressed
absolute loss).

This matters more than the 90M question it grew out of. If the 175M is also
depressed, "exclude the 90M and start the ladder at 175M" does not rescue the
analysis, and the question becomes how far up the ladder the effect reaches —
350M sits at 1.7x, which is above 1 but not by much. The run costs ~6
node-hours and is the cheapest way to bound that.

**Interim reading (2026-09-01, job still running, 5,429 / 8,540 = 64%.)**
Against the grid cell at matched iterations. Per-step losses are too noisy to
read directly, so these are 500-iteration block means over the 5,475 iterations
both runs have in common:

| iters | grid `lm-175M-L2` | `diag-...-beta3f0.2` | gap |
| ----- | ----------------: | -------------------: | --: |
| 0-499 | 5.649 | 5.515 | +0.134 |
| 500-999 | 3.865 | 3.445 | +0.420 |
| 1000-1499 | 3.315 | 3.093 | +0.222 |
| 1500-1999 | *7.340* | 2.947 | *+4.393* |
| 2000-2499 | 3.240 | 2.863 | +0.377 |
| 2500-2999 | 3.078 | 2.809 | +0.269 |
| 3000-3499 | 3.036 | 2.773 | +0.263 |
| 3500-3999 | 3.002 | 2.744 | +0.258 |
| 4000-4499 | 2.981 | 2.723 | +0.259 |
| 4500-4999 | 2.960 | 2.705 | +0.255 |
| 5000-5499 | 2.957 | 2.689 | +0.267 |

Two things. The grid run's 1500-1999 block mean of 7.34 is the loss spike this
document already noted at the 175M rung; the corrected run has nothing like it.
And from 2500 onwards the gap is **flat at ~0.26 nats**, not widening — the
corrected run is not pulling away, it is holding a constant offset.

The offset is already decisive on its own: at 64% of training and before any
WSD decay, the corrected run's block mean (2.689) is **below the grid run's
fully annealed final loss of 2.904**. Still not a final answer — the decay has
not started for either — but "only the bottom rung is affected" should not be
assumed.

### TODO: the fourth diagnostic (350M), conditional on the third

If the 175M run above finishes materially ahead of its grid cell, the next
question is whether 350M is affected too — and 350M matters more than either,
because it is a rung we are keeping and have already trained AND evaluated in
full. Its run is only 1.7x the fixed 10,000-step timescale (16,660 iters), the
same ratio quoted just above — the optimizer's memory still spans 60% of
training there. If 350M is also depressed, the finding stops being
"the bottom rung was miscalibrated" and becomes "the small end was", which
changes what the paper can claim rather than just which rungs it reports.

No code change is needed — `--ademamix-beta3-factor` (commit `c0ae655`) derives
beta3 from each cell's own `iters`, so it is size-generic:

```bash
python3.11 src/pretrain/launch_trainings.py cscs --size 350M --langs 2 --seed 1904 \
    --ademamix-beta3-factor 0.2 --dry-run
```

giving `diag-350M-L2-deep-seed1904-beta3f0.2` with `ADEMAMIX_BETA3=0.99969988`
(a 3,332-step timescale on a 16,660-iter run). Holding f = 0.2 across all three
sizes is the point of a factor rather than an absolute beta3: 90M, 175M and
350M then differ only in size, each against an already-trained grid baseline at
the same L.

Cost: 14 nodes x ~2h50 = **~39 node-hours**, from the grid cell's own `sacct`
elapsed. That is ~3.5x the two diagnostics already spent, which is why it is
gated on the 175M result rather than launched alongside it — a negligible 175M
gap would confine the effect to the bottom rung and make this unnecessary.

## Audit: which hyperparameters are fixed in STEPS rather than fractions

Prompted by the beta3 result. Because D = 100 x N at a fixed batch and
sequence length, iterations are proportional to N, so the ladder spans **4,500
to 81,000 steps — an 18x range**. Any hyperparameter expressed as a number of
steps therefore means something different at every rung, silently. Every
valued argument in `megatron_args.sh` was classified.

**Correctly scaled (fraction of the run):**

| argument | value | across the ladder |
| -------- | ----- | ----------------- |
| `--lr-warmup-iters` | per cell | 3.5-4.4% of the run |
| `--lr-wsd-decay-iters` | per cell | 19.8-20.1% |
| `--ademamix-beta3-warmup`, `--ademamix-alpha-warmup` | `ADEMAMIX_WARMUP` = iters | 100% |
| `--save-interval` | iters / 20 | 5% |
| `--lr` | 6ND law at each run's own budget | per size |

**Dimensionless, correctly fixed:** `--ademamix-alpha 8`, `--clip-grad 0.1`,
both dropouts, `--min-lr 0`, `--split`, the architecture and parallelism args,
`--seq-length`, `--global-batch-size`, rope parameters,
`--make-vocab-size-divisible-by`. `--init-method-std` is width-scaled, which is
right — initialisation is a function of width, not of run length.
`--log-interval` and `--manual-gc-interval` affect only logging and
performance.

**Fixed in steps — the beta3 class.** Four, in descending severity:

| argument | timescale | 90M | 1.7B | spread |
| -------- | --------: | --: | ---: | -----: |
| `--ademamix-beta3 0.9999` | 10,000 steps | **2.22x the run** | 0.12x | 18x |
| `--adam-beta2 0.999` | 1,000 steps | 0.22x | 0.01x | 18x |
| `--weight-decay 0.1` | per step, so total shrinkage ∝ iters | 450 | 8,100 | 18x |
| `--adam-beta1 0.9` | 10 steps | 0.002x | 0.0001x | 18x |

Reading this:

* **beta3 is confirmed broken and is the extreme case** — the only one whose
  timescale exceeds the run itself, and at 90M it exceeds it twice over.
* **beta2 is the same defect one decade smaller.** At 90M the second-moment
  estimate averages over 22% of the entire run; at 1.7B over 1%. Nothing
  observed pins a failure on it — the corrected 90M converged cleanly with
  beta2 untouched — but it is an uncontrolled variable across the size axis,
  and it should be stated rather than discovered later.
* **Weight decay is the one most likely to be overlooked.** Decoupled weight
  decay applies once per step, so at a fixed coefficient the 1.7B receives 18x
  the total shrinkage of the 90M. This is a real difference in effective
  regularisation along the very axis the scaling fit is measured over.
* **beta1 is negligible** — 10 steps is a rounding error at every rung.

**Conclusion.** The design was systematically fraction-scaled everywhere the
authors thought about it; the four exceptions are all *optimizer* constants
that convention writes as fixed numbers. For a single-model run that
convention is harmless. For a ladder whose whole purpose is to compare across
run lengths, it silently varies the optimizer along the axis being measured.
Recommendation for any future ladder: **express every optimizer timescale as a
fraction of the run, and check the total weight-decay budget explicitly.**

No change is proposed to the trained cells — see the decision above.

## The token budget, and why intermediate checkpoints are not (N, D) points

The budget is D(N) = 5 x 20 x N = 100 x N (see the training plan's "Token
budget"): a fixed multiple of Chinchilla per rung, the standard ladder setup
(Bhagia et al. span 1-10xC, DataDecide uses 5xC), chosen so the ~2B rung lands
near 200B tokens, inside ATLAS's 144-283B pretrain-vs-finetune crossover band
for multilingual models, and so that within a size only the language mix
varies. That reasoning stands and 5xC should be kept — changing it means
retraining everything.

**The compute axis is already two-dimensional.** Each run saves 20 evenly
spaced checkpoints, so checkpoint k sits at k/20 of 5C = k/4 x C, index-aligned
at every rung (8 of 40 at 1.7B):

| checkpoint | 4 | 8 | 12 | 16 | 20 |
| ---------- | - | - | -- | -- | -- |
| budget | 1xC | 2xC | 3xC | 4xC | 5xC |

Evaluation covers every 2nd checkpoint, so all five multiples are evaluated at
all six sizes: 30 (N, D) points before counting BPB on all 20.

**But those points are not what they look like.** WSD decay starts at exactly
80% of every run — checkpoint 16. So checkpoints 4, 8 and 12 (1xC, 2xC, 3xC)
are all mid-run at constant peak LR and have **never been annealed**. A model
stopped mid-schedule has a systematically higher loss than one trained to that
budget and decayed to zero. Only checkpoint 20 is a real 5xC model.

### Options

| | pros | cons |
| - | ---- | ---- |
| **A. Use the checkpoints as-is, state the caveat** | Free; the data exists. For SNR's *ranking* questions the bias is plausibly harmless if it is monotone and similar across sizes — and rankings, not absolute losses, are the deliverable. | Absolute loss/BPB at 1-3xC is biased high by an unquantified amount; a scaling-law fit over D is not defensible from them. |
| **B. Anneal from a checkpoint (cooldown)** | The modern standard. Cheap: ~10% of the tokens to that point, so a 1xC 175M point costs ~0.35B tokens (~0.1 node-hours). Turns 30 pseudo-points into 30 real ones. | Needs load-from-A / save-to-B, which the wrapper cannot express (`--save`/`--load` are one dir), plus `--override-opt_param-scheduler`, which CLAUDE.md #5 forbids for resumes. The cooldown fraction must match the grid's 20% or finals and cooldowns are not comparable. |
| **C. A fresh short run per budget** | Cleanest: each (N, D) point is a properly trained, properly annealed model, and it needs no checkpoint surgery — only the WSD schedule scaled to the shorter run. | 5 budgets x 56 cells = 280 runs against a 6,290 node-hour budget. Affordable only for spot checks. |
| **D. Use final checkpoints only** | Rigorous and free; all six points annealed. | Collapses to one axis and discards the within-run trajectory that "how do rankings evolve with training" needs. |

**Recommendation: A for the ranking analyses, and quantify the bias once
before relying on it.** SNR itself — signal, noise, decision accuracy — compares
models at matched compute, where a shared mid-schedule bias largely cancels.
The scaling-law-error analysis is the one that genuinely needs annealed
points. One measurement settles how much this matters: train a fresh 175M to
1,708 iters (= 1xC) with the WSD schedule scaled to that length, and compare
against checkpoint 4 of `lm-175M-L2-deep-seed1904`, which is already on disk.
~4 node-hours. If the gap is small, option A covers everything and B/C are
unnecessary; if it is large, no fit over D should use the raw checkpoints.

**Not implemented.** The measurement above is option C in miniature and would
need one more diagnostic flag — a `--budget-iters N` that scales warmup and
decay to N rather than truncating the full schedule. Note it is NOT
`--training-steps`, which caps a resume and deliberately keeps the original
curve (CLAUDE.md #5); using that would produce exactly the un-annealed object
we are trying to avoid. Deferred, not rejected.

**Decision to make once these land:** if the 90M cannot be brought near the
curve, drop it from the ladder rather than let it anchor the predictivity fit,
and record the exclusion — the fit is the deliverable, and a degenerate bottom
rung biases the slope everywhere.
