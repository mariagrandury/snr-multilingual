# Methodology

Here is the review of 3.1 and 3.2. The issues here are mostly internal consistency, notation, and two technical inaccuracies. I give the reasoning before each change.

## 3.1 Decision Accuracy

**Issue 1. The notation only covers one of the four DA variants.** The equation defines $s_a$ as a small proxy model and $m_a$ as a larger target model. That fits DA by size, but not DA by checkpoint (early vs late checkpoint of the same model), cross-context DA (short vs long context), or cross-stage DA (earlier vs later stage). Right now the three other variants reuse an equation whose symbols do not describe them. The fix is to define DA once over a generic proxy setting and target setting, then present the four variants as instances. This follows your style guide rule of defining symbols once and reusing them cleanly.

```latex
Decision accuracy (DA) measures whether the ranking of models in a cheap proxy
setting matches their ranking in an expensive target setting. Following
\citet{heineman_signal_2025}, we define it as
\begin{equation}
\text{DA} = \frac{1}{|\mathcal{P}|} \sum_{(a,b) \in \mathcal{P}}
\mathbb{1}\big[\text{sign}(B(p_a) - B(p_b)) = \text{sign}(B(t_a) - B(t_b))\big]
\end{equation}
where $\mathcal{P}$ is a set of model pairs, $p_a$ is the model trained with
configuration $a$ evaluated in the proxy setting, $t_a$ is the same
configuration evaluated in the target setting, and $B(\cdot)$ returns the
benchmark score. A higher DA means rankings observed in the cheap setting
transfer to the expensive one, and therefore that the benchmark can guide
development decisions. Each DA variant below instantiates the proxy and target
settings differently.
```

Then the four mini-header paragraphs need only one-line adjustments, for example "DA by size sets the proxy to a smaller model and the target to a larger model of the same family."

**Issue 2. Checkpoint percentages contradict Section 4.1.** Here you write 10%, 33%, 66%, or 100%. Section 4.1 says 10%, 33%, and 50%. One of the two is wrong, and a reader comparing methodology to analysis will catch it. I cannot tell from the text which schedule you actually ran. Pick the one matching the evaluations in the repo and use it in both places. I can verify against the results directory when we get to Section 4 if you want.

**Issue 3. The size example does not match your model suite.** "for example 1B and 8B parameters" describes only the open-source extension. Since the custom suite is 175M to 1B, a better example covers both regimes: "for example 350M and 1B for the custom suite, or 1B and 8B for open-source families." This is minor but it preempts a reviewer question about whether DA by size was even computable on the custom models.

## 3.2 Signal and Noise

**Issue 4. The star discrepancy definition is technically wrong.** You write "the largest difference between any point and the uniform distribution." Star discrepancy compares the empirical distribution of the point set against the uniform distribution, not individual points. Heineman et al. describe it as measuring how well a set of points covers a space. Corrected sentence:

```latex
Star discrepancy is the largest difference between the empirical distribution
of the scores and the uniform distribution.
```

**Issue 5. Dispersion is missing the normalization.** Your equation gives the raw maximum pairwise distance, but the framework divides every signal and noise measure by the mean score so that accuracy and unbounded metrics like BPB are comparable. Your own preliminary report adopts relative dispersion, the normalized version, and validates it against relative spread (R = 0.811 vs 0.791). Two changes follow. First, fix the equation. Second, since this is a progress report and not a proposal, state which definition you adopted and why, with a forward reference to the validation. Scores are scalars, so the norm should also be an absolute value:

```latex
Dispersion is the largest pairwise distance. We normalize it by the mean score
so that bounded and unbounded metrics are comparable,
\begin{equation}
\text{Rel. Dispersion}(C_\text{final}) =
\frac{\max_{i \neq j} |c_i - c_j|}{\bar{c}}.
\end{equation}
We adopt relative dispersion as our signal metric, since it yields the
strongest correlation with decision accuracy in our validation
(Section~\ref{subsec:exp-snr-da}).
```

**Issue 6. The noise paragraph has a typo, a symbol collision, an undefined symbol, and an inconsistent citation key.** In order: "the the standard deviation" should be "the standard deviation". The checkpoint noise equation uses $k$ for the number of final checkpoints while the benchmark noise three lines later uses $k$ for the number of folds. The symbol $U(t_j)$ is never defined and clashes with $B(\cdot)$ from 3.1. And `heineman2025signalnoiseframeworkreducing` is a second key for the paper you cite everywhere else as `heineman_signal_2025`. Rewritten paragraph:

```latex
\textbf{Noise candidates.} We consider four sources of noise that the framework
can in principle account for. Seed noise is the standard deviation of the final
checkpoint score across training runs with different random seeds. Data order
noise is the equivalent across runs that vary the order of training documents.
Total variation is the average change in score across training checkpoints
minus an improvement term. Checkpoint noise is proposed by
\citet{heineman_signal_2025} as a proxy when the first three are too expensive
to estimate at large scale. It is the relative standard deviation of the
benchmark score $B$ over the final $n$ training checkpoints
$t_{T-n+1}, \dots, t_T$ of a model,
\begin{equation}
\text{Checkpoint noise} =
\frac{\sigma\big(\{B(t_j)\}_{j=T-n+1}^{T}\big)}{\bar{B}}.
\end{equation}
```

Note I also added the normalization by the mean here, for the same comparability reason as the signal. This matches the relative standard deviation definition in your slides.

**Issue 7. The benchmark noise claim overstates its applicability.** The closing sentence says the framework becomes "applicable to any model with public scores." Computing the folds requires the per-question outputs of the evaluation run, not the aggregate score, as your own preceding sentence says. A reviewer will flag the gap. Corrected closing:

```latex
The fold scores are obtained by partitioning the stored per-question outputs
of a single evaluation run, so no inference is repeated and no intermediate
checkpoints are needed. This makes the framework applicable to any model whose
per-question evaluation outputs are available, which holds for any model the
practitioner can run once.
```

**Issue 8 (small).** "we consider five candidate signal definitions to capture a different notion of how spread out model scores are" has a number agreement problem. Use "five candidate signal definitions, each capturing a different notion of how spread out model scores are."

One structural comment to close. The section opener promises "multiple possible definitions for signal and noise" and then 3.2 lists candidates without resolving them. For a report, resolve them: you adopted relative dispersion as signal, and you compare checkpoint noise against your proposed benchmark noise. Saying this upfront in the 3.2 opening paragraph turns the section from a menu into a method, and it sets up the k-fold estimator as the contribution it is.

Want me to move on to 3.3 Models and 3.4 Benchmarks, or jump to Section 4 with the repo check for the checkpoint percentages?
