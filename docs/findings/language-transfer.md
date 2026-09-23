---
title: "RQ6 · Language transfer"
---

# RQ6 · Does scaling transfer to other languages?

**The question.** If we measured a language only on one or two small models,
or the model never trained on it, can we still predict its bits per byte at
the reference size?

**How we measure it.** We fit a power law with the exponent pooled over the
other languages and anchor it on the rungs observed for the target language.
We compare it with the language's own fit and with simply reading the largest
proxy.

## Explore

<div class="viz" data-viz="transferSummary"></div>

Predicted vs observed, language by language:

<div class="viz" data-viz="transfer"></div>

## What we find

<!-- highlight: rq06_language_transfer -->

!!! success "Takeaway"
    For a language you cannot evaluate at scale, one small measurement plus
    the pooled exponent is a good forecast — better than reading the
    largest proxy as is.

[Full analysis →](../signal-noise/language_transfer.md)
