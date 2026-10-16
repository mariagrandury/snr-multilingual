---
title: "Agreement: How Early Can We Trust Proxy Evaluations?"
---

# Agreement: How Early Can We Trust Proxy Evaluations?

Building on the results from the [previous section](measurability.md), we
investigate whether evaluations at smaller model scales or earlier training
stages can reliably inform decisions made at full scale. We study ranking
agreement by comparing the relative ordering of model variants at a cheaper
proxy point (model size and training checkpoint) with a reference point (the
final checkpoint at the same or largest model size). We examine three
settings: *Within-model training agreement*, which compares intermediate
checkpoints with the final checkpoint at the same model size ([Within-Model
Training Agreement](#within-model-training-agreement)); *Scale agreement*,
which compares fully trained smaller models with the fully trained largest
model ([Scale Agreement](#scale-agreement)); and *Target agreement*, which
evaluates whether intermediate checkpoints across model sizes recover the
rankings of the fully trained largest model ([Target
Agreement](#target-agreement)). These settings are summarised in the table
below. We additionally examine the sensitivity of ranking agreement to
near-tied model scores, evaluation scoring protocols, and per-language
training-token budgets ([Sensitivity Analysis](#sensitivity-analysis)).

| Setting | DA | Proxy | Reference | Question |
|---|---|---|---|---|
| Scale agreement | DA-size | ($N$, $f=1$) | ($N_{\mathrm{ref}}$, $f=1$) | How small can the proxy model be? |
| Within-model training agreement | DA-ckpt | ($N$, $f<1$) | ($N$, $f=1$) | How early does the ranking stabilize? |
| Target agreement | DA-goal | ($N$, $f<1$) | ($N_{\mathrm{ref}}$, $f=1$) | How small and early can the proxy be? |

We quantify ranking agreement using decision accuracy (DA), defined as the
fraction of pairwise model preferences preserved between the proxy and
reference points. We compute DA only for ranking comparisons involving at
least three variant pairs, since comparisons between only two pairs yield
binary agreement scores. We compare model variants across the intervention
dimensions of the [model ladder](../ladder.md), using model sizes ranging
from 90M to 1.7B parameters. We focus on accuracy and bBPB tasks whose median
DA across proxy points reaches at least 0.66 in either model-size or
training-checkpoint DA. This heuristic threshold selects tasks with higher
observed ranking agreement.

<div class="viz" data-viz="paperRq2"></div>

*Decision accuracy of proxy models for the 1.7B reference, over the
benchmark–language tasks (accuracy and bBPB) whose median DA in the scale or
within-model training agreement setting is at least 0.66. **(a) Scale
agreement**: DA of the final checkpoint of each model size against the final
1.7B ranking, pooled over decisions, one line per design axis and a black
line over all pairs. **(b) Within-model training agreement**: DA of each
intermediate checkpoint relative to the final checkpoint of the same model
size, averaged over tasks. **(c) Target agreement**: DA of each intermediate
checkpoint relative to the final 1.7B reference model, measuring how early
partially trained and smaller models recover the decision at the reference
size.*

## Within-Model Training Agreement

Panel (b) shows the mean DA across benchmark–language tasks and model-variant
comparisons at each training checkpoint, using the final checkpoint at the
same model size as the reference. Ranking agreement generally increases with
training progress across all model sizes, rising from approximately 62–68%
at early checkpoints to 83–93% in the later stages of training.
Nevertheless, disagreement persists even near the end of training,
indicating that intermediate evaluations do not always reproduce final
model-selection decisions. Although 350M models generally exhibit the
highest agreement, we observe no consistent relationship between model size
and ranking agreement.

## Scale Agreement

Panel (a) shows the mean DA between fully trained models at different scales
and the fully trained 1.7B reference, across model-variant comparisons
grouped by intervention type. Overall, agreement remains modest (59–61%
across all pairs) and shows no consistent improvement with increasing model
size. However, agreement varies substantially across intervention types,
with sampling temperature consistently exhibiting higher agreement (72–75%)
than other design choices. These results suggest that the reliability of
cross-scale decisions depends more on the type of model intervention than on
the size of the proxy model alone.

## Target Agreement

Panel (c) shows the mean DA between intermediate checkpoints across model
sizes and the target (the fully trained 1.7B reference model), measuring the
combined effects of model scale and training progress. For smaller models,
agreement remains modest (approximately 58–64%) and shows little systematic
improvement with additional training. In contrast, agreement at 1.7B
increases steadily throughout training, consistent with the within-model
training agreement observed in panel (b). This contrast indicates that
improvements in ranking agreement during training do not necessarily
transfer across model scales, limiting the ability of smaller, partially
trained models to predict decisions at the reference size. To test whether
reliable cross-scale ranking agreement emerges at larger proxy sizes, we
extend the model ladder to 3B for a subset of configurations. Even at 1.7B,
benchmark accuracy shows low agreement with the 3B reference, with DA of
0.51–0.52. This indicates that increasing proxy model size to 1.7B is still
not sufficient to recover reliable rankings at the reference size from
multilingual benchmark accuracy.

## Sensitivity Analysis

We further examine the sensitivity of ranking agreement to near-tied model
scores (i.e., model variants with very close rankings), evaluation scoring
protocols, and per-language training-token budgets. Accounting for near-tied
scores as expected increases DA without substantially changing the main
trends, while BPB-based evaluations generally preserve rankings more reliably
than benchmark accuracy, particularly at smaller scales and earlier training
stages.

!!! success "Summary"
    On accuracy-scored benchmarks, the share of variant pairs that a proxy
    point orders the same way as the final 1.7B checkpoint is close to
    pairwise chance: 0.51-0.56 at every proxy size. On per-language BPB, the
    share is higher, ranging from 0.74 to 0.94 over the same proxies.
    Seed-to-seed noise in the reference models accounts for part of the
    disagreement. On the design-variant pairs that the 1.7B reference model
    separates by more than two seed standard deviations, the share ordered
    correctly is 0.69 to 0.73.
