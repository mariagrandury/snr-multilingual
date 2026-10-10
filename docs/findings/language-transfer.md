---
title: "RQ6 · Language transfer"
---

# RQ6 · Does what we measure carry over to other languages?

**The question.** Three versions of one question. If a language was measured
only on one or two small models, or the model never trained on it, can we
still predict its bits per byte at the reference size? Does a design decision
a proxy reads on its trained languages carry over to the languages it did not
train? And does training a language help its untrained relatives?

**How we measure it.**

- *Forecasting BPB.* We fit a power law per language, with the exponent
  pooled over the other languages and the intercept anchored on the rungs
  observed for the target language. We compare it with the language's own
  fit and with simply reading the largest proxy. 100 evaluation languages at
  six language settings L: 106 trained and 494 never-trained (L, language)
  cases.
- *Transferring a decision.* The language-list decision (scheme A vs B at L8,
  L15 and L30) read on the BPB of four groups of languages: trained by both
  lists, by one list, untrained but written in a trained script, and
  untrained in an untrained script. DA-size compares each proxy's final
  checkpoint with the 1.7B final.
- *Language against language.* Whether the ranking of the design variants
  on one language's task, at the proxies, predicts the 1.7B ranking on
  another language's task (cross-task DA-size, mean over 90M–1B). A model's
  score on a language it does not train counts here.
- *Untrained relatives.* How often the tasks in untrained languages clear
  chance, split by whether the language shares the family of a trained
  non-English language, at two family levels (top-level family such as
  Indo-European, subfamily such as Slavic), against the English-only cell on
  exactly the same tasks. These views read the benchmarks every cell is
  evaluated on (the `auto` set: 941 tasks of 27 benchmarks), not the
  candidates still being screened.

## Explore

### Forecasting a language's BPB

<div class="viz" data-viz="transferSummary"></div>

Predicted vs observed, language by language:

<div class="viz" data-viz="transfer"></div>

**The same forecast, on a map of the languages.** Every chip is one
language, grouped by family and subfamily; the darker its swatch, the closer
the forecast lands to the observed 1.7B BPB. Outlined chips are trained at
the chosen L.

<div class="viz" data-viz="ltTree"></div>

### Does a design decision transfer to untrained languages?

**Rows are groups of languages, columns proxy sizes.** A filled dot means the
proxy picks the same language list as the 1.7B reference on at least the
threshold share of that group's languages; the strip on the right is the
reference's own run, one tick per checkpoint.

<div class="viz" data-viz="ltGroups"></div>

### Which language predicts which?

**Pick a language and see whose 1.7B ranking it predicts** (or, flipped, who
predicts it). Blue swatches agree with the target's ranking more often than
chance, orange less; click any chip to pick it.

<div class="viz" data-viz="ltCross"></div>

### Do untrained relatives benefit?

**Each dot is one subfamily of untrained languages under one cell**, placed
by how much more often its tasks clear chance than under English-only
training; filled dots are subfamilies the cell trains a language of. Switch
to "share above chance by size" for the two family levels side by side with
their English-only baselines.

<div class="viz" data-viz="ltFamily"></div>

## What we find

<!-- highlight: rq06_language_transfer -->

- **One small rung plus the pooled exponent beats the largest proxy read as
  is.** From the 175M rung alone the 1.7B BPB is predicted within 4.4 %
  (trained) and 5.0 % (never trained), median; the 1B proxy's own BPB taken
  as the reference's misses by 7.6 % and 6.6 %. The error grows with L: from
  3.6 % at L1 to 7.6 % at L50 for never-trained languages.
- **The list decision does not transfer to untrained languages at a proxy's
  scale.** Languages whose script a list trains are read at 0.63 (90M) and
  0.72–0.73 (175M–600M), 0.81 only at 1B; unseen scripts at 0.42–0.73, never
  0.75. The trained languages are read at 0.77–1.00 (both lists) and 1.00
  (one list, where the model that saw the language wins).
- **Within one benchmark, the languages a mixture trains together predict
  each other.** On HellaSwag, German, Spanish, French and Italian, the four
  European languages of the L8 list, predict each other's 1.7B ranking at
  0.82 on average, as well as each predicts itself. English ranks itself at
  0.78 but the other languages at 0.43, and they rank it at 0.50; Basque,
  which no list trains, is predicted by no other language (0.46 on average).
- **Across benchmarks the transfer is weak.** On every task above chance the
  own-task diagonal averages 0.51 and every off-diagonal kind 0.48; only 4 %
  of the off-diagonal cells reach 0.75.
- **Training a language helps its own subfamily a little, and no one else.**
  Five of the six (cell, trained subfamily) lifts over English only are
  positive, +0.03 to +0.06 (Romance under L8 A the largest: 0.55 of the
  Catalan, Galician, Portuguese and Romanian tasks above chance against
  0.49); Germanic under L8 A is −0.01. The 54 lifts on subfamilies a cell
  does not train centre on zero (median 0.00, 44 % positive). Russian lifts
  Slavic at every L (+0.03 to +0.04). The raw same-family share sits above
  the other-family one at both family levels, but English only shows most of
  that gap too, so the gap is mostly which languages and benchmarks fall in
  each group.

!!! success "Takeaway"
    For a language you cannot evaluate at scale, one small measurement plus
    the pooled exponent is a good forecast, better than reading the largest
    proxy as is. A design decision, though, has to be read on the languages
    you care about: it does not carry over to untrained ones, and English is
    a poor stand-in for the rest.

[Full analysis →](../signal-noise/language_transfer.md)
