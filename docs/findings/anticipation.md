---
title: "Anticipation: Can We Predict Ranking Agreement?"
---

# Anticipation: Can We Predict Ranking Agreement?

Computing DA requires access to both proxy and reference evaluations,
limiting its utility for guiding development decisions before larger models
are trained. Prior work has proposed several surrogate metrics, most notably
signal-to-noise ratio (SNR), which compares the separation between model
variants to the variability of their evaluation scores.

Specifically, we test whether statistics computed from proxy evaluations
alone can serve as surrogates for DA. For each benchmark–language task and
proxy size, we compute each surrogate without using the reference model. We
then measure its Spearman correlation ($\rho$) with DA across
benchmark–language tasks at each proxy size. Figure 21 reports these
correlations across proxy sizes for tasks passing the above-chance
threshold. We compare scaling-fit $R^2$, SNR variants and their signal/noise
components, margin above chance, and early-checkpoint ranking agreement.

<div class="viz" data-viz="paperRq3"></div>

***Figure 21:** Spearman rank correlation ($\rho$) between each surrogate and DA for the
1.7B model across benchmark–language tasks. Cells report the $\rho$ and
sample size (n). This figure corresponds to Figure 4 from the paper.*
{ #figure-21 }

We find that no single surrogate consistently predicts DA across
multilingual settings. Early-checkpoint ranking agreement
($\rho = 0.54$–$0.62$) and the relative-standard-deviation SNR
($\rho = 0.53$–$0.60$) show the strongest overall associations, followed by
scaling-fit $R^2$ ($\rho = 0.45$–$0.49$). In contrast, signal alone and
margin above chance show little association with DA. For SNR-based
surrogates, however, performance depends strongly on how noise is defined,
with some noise estimates yielding substantially stronger associations than
others.
Stronger associations emerge for individual benchmarks, up to $\rho=0.93$
for HellaSwag, demonstrating that predictability is benchmark-dependent.

<div class="viz" data-viz="surrogatesVariants"></div>

***Figure 22: The relative-spread SNR definitions track decision accuracy best, as a
block.** For each of the 22 SNR definitions, the mean over languages of the Pearson
r between log10 SNR and decision accuracy across the tasks of one language,
for DA-size (blue) and DA-ckpt (orange). This figure corresponds to Figure 19 from the paper.*
{ #figure-22 }

<div class="viz" data-viz="surrogatesBench"></div>

***Figure 23: Stronger associations emerge for individual benchmarks.** Spearman ρ with DA-size within one benchmark, for the
relative-standard-deviation SNR and the strongest SNR definition.*
{ #figure-23 }

Moderate
correlations are already present at small scales and do not consistently
improve as models grow. Early-checkpoint agreement reaches $\rho=0.62$ at
175M, while relative-standard-deviation noise reaches $\rho=0.56$ at 350M.
Scaling-fit $R^2$ becomes available from 350M and remains relatively stable
($\rho = 0.45$–$0.49$) up to 1B. Overall, surrogate signal emerges at small
scales but does not consistently strengthen with model size. For example, at 600M,
per-language BPB shows low evaluation noise (checkpoint noise of 0.4% against
1.3% of the mean score), strong scaling fits (median $R^2$ of 1.00), and
clear score differences between variants (median relative-standard-deviation
SNR of 8.7 against 2.0 for benchmark tasks), yet DA against the 1.7B
reference remains at $0.70$–$0.74$ throughout training.

<div class="viz" data-viz="surrogatesScoring"></div>

***Figure 24: Surrogates split by scoring.** Spearman ρ between each surrogate and DA-size against 1.7B at each proxy
size, on the benchmark tasks above chance scored by accuracy, their bBPB
versions and per-language BPB; n is the number of tasks. This figure corresponds to Figure 20 from the paper.*
{ #figure-24 }

!!! success "Summary"
    No metrics computed from proxy evaluations predicts DA consistently across
    benchmarks or proxy sizes. The two metrics with the strongest association
    are early-checkpoint ranking agreement, the DA between a run's 10%
    checkpoint and final checkpoint ($\rho = 0.54$–$0.62$), and the
    relative-standard-deviation SNR, the spread of the variants' final scores
    divided by checkpoint noise ($\rho = 0.53$–$0.60$).
