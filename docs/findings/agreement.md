---
title: "Agreement: How Early Can We Trust Proxy Evaluations?"
---

# Agreement: How Early Can We Trust Proxy Evaluations?

We study ranking
agreement by comparing the relative ordering of model variants at a cheaper
proxy point (model size and training checkpoint) with a reference point (the
final checkpoint at the same or largest model size). We examine three
settings: *Within-model training agreement*, which compares intermediate
checkpoints with the final checkpoint at the same model size
([§2.1](#21-within-model-training-agreement)); *Scale agreement*, which
compares fully trained smaller models with the fully trained largest model
([§2.2](#22-scale-agreement)); and *Target agreement*, which evaluates
whether intermediate checkpoints across model sizes recover the rankings of
the fully trained largest model ([§2.3](#23-target-agreement)). These
settings are summarised in Table 3 below.

<div class="da-settings" markdown>

| Setting | DA | Proxy | Reference | Research Question |
|---|---|---|---|---|
| Scale agreement | DA-size | ($N$, $f=1$) | ($N_{\mathrm{ref}}$, $f=1$) | How small can the proxy model be? |
| Within-model training agreement | DA-ckpt | ($N$, $f<1$) | ($N$, $f=1$) | How early does the ranking stabilize? |
| Target agreement | DA-target | ($N$, $f<1$) | ($N_{\mathrm{ref}}$, $f=1$) | How small and early can the proxy be? |

</div>

***Table 3:** Ranking-agreement settings. $N$ is the non-embedding parameter
count, $N_{\mathrm{ref}}$ the reference size (1.7B) and $f$ the training
progress. This table corresponds to Table 8 from the paper.*
{ #table-3 }

We quantify ranking agreement using decision accuracy (DA), defined as the
fraction of pairwise model preferences preserved between the proxy and
reference points. We compute DA only for ranking comparisons involving at
least three variant pairs, since comparisons between only two pairs yield
binary agreement scores. We compare model variants across the intervention
dimensions of the [model ladder](../ladder.md), using model sizes ranging
from 90M to 1.7B parameters. We focus on accuracy and bBPB tasks whose median
DA across proxy points reaches at least 0.66 in either model-size or
training-checkpoint DA.

## 2.1 Within-Model Training Agreement

<div class="viz" data-viz="paperRq2" data-panel="ckpt"></div>

***Figure 13: Within-model training agreement.** DA of each intermediate
checkpoint relative to the final checkpoint of the same model size, averaged
over tasks, over the benchmark–language tasks (accuracy and bBPB) whose median DA in
the scale or within-model training agreement setting is at least 0.66. This figure corresponds to
Figure 3b from the paper.*
{ #figure-13 }

Figure 13 shows the mean DA across benchmark–language tasks and model-variant
comparisons at each training checkpoint, using the final checkpoint at the
same model size as the reference. Ranking agreement generally increases with
training progress across all model sizes, rising from approximately 62–68%
at early checkpoints to 83–93% in the later stages of training. Although 350M models generally exhibit the
highest agreement, we observe no consistent relationship between model size
and ranking agreement.

## 2.2 Scale Agreement

<div class="viz" data-viz="paperRq2" data-panel="size"></div>

***Figure 14: Scale agreement.** DA of the final checkpoint of each
model size against the final 1.7B ranking, pooled over decisions, one line
per design axis and a black line over all pairs, over the benchmark–language tasks (accuracy and bBPB) whose median DA in
the scale or within-model training agreement setting is at least 0.66. This figure corresponds to
Figure 3a from the paper.*
{ #figure-14 }

Figure 14 shows the mean DA between fully trained models at different scales
and the fully trained 1.7B reference, across model-variant comparisons
grouped by intervention type. Overall, agreement remains modest (59–61%
across all pairs) and shows no consistent improvement with increasing model
size. However, agreement varies substantially across intervention types,
with sampling temperature consistently exhibiting higher agreement (72–75%)
than other design choices. These results suggest that the reliability of
cross-scale decisions depends more on the type of model intervention than on
the size of the proxy model alone.

## 2.3 Target Agreement

<div class="viz" data-viz="paperRq2" data-panel="target"></div>

***Figure 15: Target agreement.** DA of each intermediate checkpoint
relative to the final 1.7B reference model, measuring how early partially
trained and smaller models recover the decision at the reference size, over the benchmark–language tasks (accuracy and bBPB) whose median DA in
the scale or within-model training agreement setting is at least 0.66. This figure corresponds to
Figure 3c from the paper.*
{ #figure-15 }

Figure 15 shows the mean DA between intermediate checkpoints across model
sizes and the target (the fully trained 1.7B reference model), measuring the
combined effects of model scale and training progress. For smaller models,
agreement remains modest (approximately 58–64%) and shows little systematic
improvement with additional training. In contrast, agreement at 1.7B
increases steadily throughout training, consistent with the within-model
training agreement observed in Figure 13. To test whether
reliable cross-scale ranking agreement emerges at larger proxy sizes, we
extend the model ladder to 3B for a subset of configurations. Even at 1.7B,
benchmark accuracy shows low agreement with the 3B reference, with DA of
0.51–0.52.

<div class="viz" data-viz="sgReference"></div>

***Figure 16: A 3B reference ranks the proxies no better than the 1.7B one.** On identical decisions, DA-size against 3B is 0.49–0.54 for accuracy and
0.78–0.91 for per-language BPB, against the 1.7B and the 3B final
checkpoints.*
{ #figure-16 }

<div class="viz" data-viz="sgGate"></div>

***Figure 17: At 3B more tasks pass the above-chance threshold.** From 612 to 778 of 933 tasks (+27%), mostly the letter-format Belebele,
Global-MMLU and INCLUDE tasks: the share of each benchmark's tasks above
chance at 1.7B and at 3B, on the six configurations trained to 3B.*
{ #figure-17 }

## 2.4 Sensitivity Analysis

We further examine the sensitivity of ranking agreement to near-tied model
scores (i.e., model variants with very close rankings), evaluation scoring
protocols, and per-language training-token budgets. Accounting for near-tied
scores as expected increases DA without substantially changing the main
trends, while BPB-based evaluations generally preserve rankings more reliably
than benchmark accuracy, particularly at smaller scales and earlier training
stages.

At $\epsilon = 0.5$ pp, 755 reversed comparisons are reclassified as correct,
increasing to 1,099 at 1 pp and 1,308 at 2 pp.

| $\epsilon$ (pp) | Pairs within tolerance at reference | Ranking reversals treated as ties | Already correct |
|---|---|---|---|
| 0.1 | 357 | 184 | 173 |
| 0.25 | 982 | 436 | 546 |
| 0.5 | 1,949 | 755 | 1,194 |
| 1.0 | 3,268 | 1,099 | 2,169 |
| 2.0 | 4,301 | 1,308 | 2,993 |

***Table 4: Sensitivity of decision accuracy to the reference-gap tolerance
$\epsilon$.** "Within tolerance" counts pairs whose absolute performance
difference at the reference scale is at most $\epsilon$. Only reversed pairs are
newly reclassified as correct; pairs whose rankings already agree remain
unchanged. This table corresponds to Table 9 from the paper.*
{ #table-4 }

<div class="viz" data-viz="daChannels"></div>

***Figure 18: Where accuracy is at chance at the proxy, bBPB still ranks the design
variants like the 1.7B bBPB, though far below per-language BPB.** DA-size against the 1.7B final checkpoint on multi-axis pairs, for the
tasks above chance at the proxy, at chance at the proxy and at chance at
every size. Blue: accuracy against the 1.7B accuracy. Orange: the proxy's
bBPB against the 1.7B accuracy. Green: bBPB against the 1.7B bBPB. Black
tick: the macro average of per-language BPB. Dashed: 0.5. This figure corresponds to Figure 16 from the paper.*
{ #figure-18 }

<div class="viz" data-viz="daDecisive"></div>

***Figure 19: Decision accuracy on the pairs the reference separates.** On the design-variant pairs that the 1.7B reference separates by more than
two seed standard deviations (blue), the share ordered correctly is 0.69 to
0.73, while per-language BPB orders 0.74–0.94 of all pairs (grey) like the
reference.*
{ #figure-19 }

<div class="viz" data-viz="ltCross"></div>

***Figure 20: On HellaSwag, the languages that the K = 8 list trains together predict
each other's 1.7B ranking, while English does not.** DA-size of the ranking on the HellaSwag task of one language at the proxy
(rows) against the 1.7B ranking on another language (columns), averaged
over 90M–1B; the diagonal is the task's own DA-size. This figure corresponds to Figure 15 from the paper.*
{ #figure-20 }

!!! success "Summary"
    On accuracy-scored benchmarks, the share of variant pairs that a proxy
    point orders the same way as the final 1.7B checkpoint is close to
    pairwise chance: 0.51-0.56 at every proxy size. On per-language BPB, the
    share is higher, ranging from 0.74 to 0.94 over the same proxies.
    Seed-to-seed noise in the reference models accounts for part of the
    disagreement. On the design-variant pairs that the 1.7B reference model
    separates by more than two seed standard deviations, the share ordered
    correctly is 0.69 to 0.73.
