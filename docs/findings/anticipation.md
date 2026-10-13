---
title: "Anticipation: Can We Predict Ranking Agreement?"
---

# Anticipation: Can We Predict Ranking Agreement?

Computing DA requires access to both proxy and reference evaluations, limiting
its usefulness for guiding development decisions before larger models are
trained. We therefore ask whether statistics computed from proxy evaluations
alone can indicate how well their rankings will agree with those at reference
scale. Prior work has proposed several such surrogate metrics, most notably
signal-to-noise ratio (SNR), which compares the separation between model
variants to the variability of their evaluation scores. These methods have
primarily been studied in English settings and at scales where downstream
benchmark signal is already measurable. In this analysis section, we
investigate whether existing surrogate metrics are associated with DA in the
multilingual setting and whether they provide useful signal at the smallest
model scale.

We test whether statistics computed from proxy evaluations alone can serve as
surrogates for DA. For each benchmark–language task and proxy size, we compute
each surrogate without using the reference model. We then measure its Spearman
correlation ($\rho$) with DA across benchmark–language tasks at each proxy
size. The figure below reports these correlations across proxy sizes for tasks
passing the above-chance threshold. We compare scaling-fit $R^2$, SNR variants
and their signal/noise components, margin above chance, and early-checkpoint
ranking agreement. These capture, respectively, how well scores follow a
log-linear scaling trend, the separation and variability of model scores,
performance relative to chance, and ranking agreement between early and final
checkpoints at the same size.

<div class="viz" data-viz="paperRq3"></div>

*Surrogate predictivity across proxy model sizes. Spearman rank correlation
($\rho$) between each surrogate statistic and DA across benchmark–language
tasks at each proxy size. Each entry reports the correlation and the number of
benchmark tasks (n) used to compute it. Higher values indicate stronger
associations with ranking agreement at the 1.7B reference scale. Scaling-fit
$R^2$ is reported only from 350M onward, as fitting requires at least three
model sizes.*

We find that no single surrogate consistently predicts DA across multilingual
settings. Early-checkpoint ranking agreement ($\rho = 0.54$–$0.62$) and the
relative-standard-deviation SNR ($\rho = 0.53$–$0.60$) show the strongest
overall associations, followed by scaling-fit $R^2$ ($\rho = 0.45$–$0.49$). In
contrast, signal alone and margin above chance show little association with
DA. For SNR-based surrogates, however, performance depends strongly on how
noise is defined, with some noise estimates yielding substantially stronger
associations than others. These findings suggest that benchmarks whose
rankings stabilize early in training and whose scores exhibit less evaluation
noise are also more likely to preserve the ranking of model variants across
scales. Stronger associations emerge for individual benchmarks, reaching
$\rho=0.93$ for HellaSwag, compared with a maximum of $\rho=0.75$ across all
benchmarks combined, demonstrating that surrogate effectiveness is
benchmark-dependent.

We next examine how these associations change with model size. Moderate
correlations are already present at small scales and do not consistently
improve as models grow. Early-checkpoint agreement reaches $\rho=0.62$ at
175M, while relative-standard-deviation noise reaches $\rho=0.56$ at 350M.
Scaling-fit $R^2$ becomes available from 350M and remains relatively stable
($\rho = 0.45$–$0.49$) up to 1B. Overall, surrogate signal emerges at small
scales but does not consistently strengthen with model size. These findings
suggest that reliable evaluation at smaller scales does not necessarily imply
consistent model rankings at larger scales. For example, at 600M, per-language
BPB shows low evaluation noise, strong scaling fits, and clear score
differences between variants (median relative-standard-deviation SNR of 8.7
against 2.0 for the benchmark tasks, checkpoint noise of 0.4% against 1.3% of
the mean score, and a median scaling-fit $R^2$ of 1.00), yet DA against the
1.7B reference remains at $0.70$–$0.74$ throughout training. This highlights
the distinction between reliably measuring performance differences at the
proxy scale and preserving model rankings across scales.

!!! success "Finding"
    Benchmarks whose ranking settles early in training and whose scores vary
    little across checkpoints are more likely to preserve the ranking of model
    variants across scales.
