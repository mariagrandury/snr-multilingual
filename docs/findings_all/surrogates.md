---
title: "RQ4 · Cheap predictors"
---

# RQ4 · Which cheap statistic predicts decision accuracy?

**The question.** Decision accuracy needs the large model you want to avoid
training. Can a number computed on the small model alone — an SNR, its
signal or noise part, early-checkpoint agreement, scaling-fit quality, the
margin above chance — tell you in advance whether its decision will hold?

**How we measure it.** For each proxy size (90M–1B) we take the tasks above
chance at the proxy and at the 1.7B reference and correlate each statistic
with the tasks' DA-size against 1.7B (Spearman ρ). The paper pools each
benchmark task's accuracy with its bBPB twin (the same items scored as bits
per byte of the correct answer); below we also split the two and add the
per-language bits per byte.

## Explore

<div class="viz" data-viz="surrogates"></div>

**Where does the signal come from?** The same grid with a scoring toggle.
Switch between the pooled tasks, accuracy alone, the bBPB twins alone and
the per-language BPB, or show all four side by side, one panel per
statistic.

<div class="viz" data-viz="surrogatesScoring"></div>

**One dot per task.** The statistic on the x axis, the task's DA-size on the
y axis; the bold line is the median DA-size along x. A useful statistic
makes the line climb; search for a benchmark or a language to find its tasks.

<div class="viz" data-viz="surrogatesScatter"></div>

The best SNR definition, language by language:

<div class="viz" data-viz="bestVariant"></div>

**The 22 SNR definitions at a glance.** Each row is one definition and its
family; the two dots are its mean correlation with the two kinds of
decision accuracy.

<div class="viz" data-viz="surrogatesVariants"></div>

## What we find

- **No single surrogate leads at every proxy size.** On the pooled
  benchmark tasks (accuracy + bBPB, 1274–1461 tasks) early-checkpoint
  agreement reads ρ 0.54–0.62 and the relative-std SNR 0.53–0.60, trading
  the lead: early agreement at 90M, 175M, 600M and 1B, SNR at 350M. The
  scaling-fit R² follows at 0.45–0.49 from 350M. The signal is there at
  90M already and does not grow with the proxy.
- **The predictive part of SNR is its noise.** Signal alone (ρ −0.16 to 0.00) and the margin above chance (−0.03 to 0.09) carry nothing. The
  inverted noise alone reads 0.44–0.56.
- **The ρ ≈ 0.6 level comes from the bBPB twins.** On the accuracy tasks
  alone (359–544) the relative-std SNR reads 0.31–0.36 and leads from
  90M to 600M (tied with the inverted noise at 600M). At 1B the inverted
  noise leads (0.38). Early agreement reads 0.20–0.37. On the 816 twins
  early agreement reads 0.57–0.70.
- **Per-language BPB turns negative at 600M.** Every statistic but the
  inverted noise is positive from 90M to 350M and drops below zero at 600M
  (SNR −0.37, scaling-fit R² −0.60), the proxy whose depth pairs disagree
  with 1.7B ([RQ3](noise-snr.md)). No proxy statistic can see a reversal
  that only happens at the reference.
- **Among the 22 SNR definitions the relative-spread family wins as a
  block** (`rel_std`, `rel_mpd`, `rel_dispersion`: mean r 0.52–0.53 with DA-size, 0.52–0.54 with DA-ckpt), and it is the best family for DA-size in
  26 of 50 languages; the depth definitions are null.

<!-- highlight: rq04_surrogates -->

!!! success "Takeaway"
    Use SNR and early-checkpoint agreement to *shortlist* benchmarks, not to
    certify a decision: they rank tasks only moderately like their
    decision accuracy (ρ ≈ 0.6). Prefer a relative-spread definition and
    estimate its noise well, since the noise is what predicts; read accuracy
    and bBPB apart, and distrust any proxy statistic on a size where the
    decision itself may reverse.

[Full analysis →](../signal-noise/surrogates.md)
