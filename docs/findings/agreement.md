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
checkpoints with the final checkpoint at the same model size
([§2.1](#21-within-model-training-agreement)); *Scale agreement*, which
compares fully trained smaller models with the fully trained largest model
([§2.2](#22-scale-agreement)); and *Target agreement*, which evaluates
whether intermediate checkpoints across model sizes recover the rankings of
the fully trained largest model ([§2.3](#23-target-agreement)). These
settings are summarised in Table 3 below. We additionally examine the
sensitivity of ranking agreement to near-tied model scores, evaluation
scoring protocols, and per-language training-token budgets
([§2.4](#24-sensitivity-analysis)).

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
training-checkpoint DA. This heuristic threshold selects tasks with higher
observed ranking agreement.

## 2.1 Within-Model Training Agreement

<div class="viz" data-viz="paperRq2" data-panel="ckpt" id="figure-13" data-caption="Figure 13: Within-model training agreement." data-paper="Figure 3b"></div>

Figure 13 shows the mean DA across benchmark–language tasks and model-variant
comparisons at each training checkpoint, using the final checkpoint at the
same model size as the reference. Ranking agreement generally increases with
training progress across all model sizes, rising from approximately 62–68%
at early checkpoints to 83–93% in the later stages of training.
Nevertheless, disagreement persists even near the end of training,
indicating that intermediate evaluations do not always reproduce final
model-selection decisions. Although 350M models generally exhibit the
highest agreement, we observe no consistent relationship between model size
and ranking agreement.

## 2.2 Scale Agreement

<div class="viz" data-viz="paperRq2" data-panel="size" id="figure-14" data-caption="Figure 14: Scale agreement." data-paper="Figure 3a"></div>

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

<div class="viz" data-viz="paperRq2" data-panel="target" id="figure-15" data-caption="Figure 15: Target agreement," data-text="how early partially trained and smaller models recover the decision at the reference size." data-paper="Figure 3c"></div>

Figure 15 shows the mean DA between intermediate checkpoints across model
sizes and the target (the fully trained 1.7B reference model), measuring the
combined effects of model scale and training progress. For smaller models,
agreement remains modest (approximately 58–64%) and shows little systematic
improvement with additional training. In contrast, agreement at 1.7B
increases steadily throughout training, consistent with the within-model
training agreement observed in Figure 13. This contrast indicates that
improvements in ranking agreement during training do not necessarily
transfer across model scales, limiting the ability of smaller, partially
trained models to predict decisions at the reference size. To test whether
reliable cross-scale ranking agreement emerges at larger proxy sizes, we
extend the model ladder to 3B for a subset of configurations. Even at 1.7B,
benchmark accuracy shows low agreement with the 3B reference, with DA of
0.51–0.52. This indicates that increasing proxy model size to 1.7B is still
not sufficient to recover reliable rankings at the reference size from
multilingual benchmark accuracy.

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

<div class="viz" data-viz="daChannels" id="figure-16" data-caption="Figure 16: Where accuracy is at chance at the proxy, bBPB still ranks the design variants like the 1.7B bBPB, though far below per-language BPB." data-text="Blue: accuracy against the 1.7B accuracy. Orange: the proxy&#x27;s bBPB against the 1.7B accuracy. Green: bBPB against the 1.7B bBPB." data-paper="Figure 16"></div>

<div class="viz" data-viz="ltCross" id="figure-17" data-caption="Figure 17: On HellaSwag, the languages that the K = 8 list trains together predict each other&#x27;s 1.7B ranking, while English does not." data-paper="Figure 15"></div>

!!! success "Summary"
    On accuracy-scored benchmarks, the share of variant pairs that a proxy
    point orders the same way as the final 1.7B checkpoint is close to
    pairwise chance: 0.51-0.56 at every proxy size. On per-language BPB, the
    share is higher, ranging from 0.74 to 0.94 over the same proxies.
    Seed-to-seed noise in the reference models accounts for part of the
    disagreement. On the design-variant pairs that the 1.7B reference model
    separates by more than two seed standard deviations, the share ordered
    correctly is 0.69 to 0.73.
