---
name: draft
description: Write or rewrite sections of NLP research papers — abstract, introduction, related work, method, experimental setup, results, limitations, captions. Use this skill whenever the user is drafting, rewriting, tightening, or asking for feedback on academic paper prose: rough notes that need to become a section, a paragraph that "doesn't sound like a paper", an abstract that needs restructuring, or a request to match a specific paper voice. Trigger it even when the user does not mention a style guide or name a section, as long as the text is research writing for a paper submission.
---

# NLP Paper Style

House style for academic paper prose.

The style is consistent enough to be treated as a set of rules. Follow them.

## Before writing

Settle three things first. Infer them from context where possible; ask only if a wrong guess would waste the user's time.

1. **Which section?** Abstract, Introduction, Related Work, Method, Experimental Setup, Results, Limitations, and captions each follow different patterns. Section identity drives everything below.
2. **What is fixed?** Numbers, citations, technical claims, and anything the user wraps in `[KEEP: ...]` stay verbatim.
3. **Rewrite or draft?** A rewrite changes prose only. A draft from notes may restructure, but never adds technical content.

**Never invent.** Do not add numbers, dataset names, citations, or results that are not in the source. If a slot needs a number the user has not given, write a visible placeholder (`[X%]`, `[CITE]`) and say so at the end. A fabricated citation is worse than a gap.

## Core rules (all sections)

**Voice.** Active first-person plural. *We propose*, *we evaluate*, *we observe*, *we find*, *we show*. Avoid *it is shown that*, *it can be seen that*.

**Hedge speculation, not results.** Measured outcomes are stated flatly: *Our method improves F1 by 3.2%.* Never *may improve*. Interpretations get hedges: *We hypothesize that*, *One possible explanation is*, *This suggests that*. Mixing these up is the single most common failure — a hedged result reads as unconfident, an unhedged interpretation reads as overclaiming.

**Lead with the claim.** The first sentence of a paragraph carries the point. The rest supports it. The last sentence says what it means: *This indicates that...* / *This suggests that...* / *This highlights...*

**One claim per paragraph.** If a paragraph makes two points, split it.

**Short sentences.** Two clear sentences beat one with three subordinate clauses. Cut *in order to* → *to*, *with respect to* → *for*, *it is important to note that* → delete.

**Enumerate in prose, not bullets.** *First, ... Second, ... Finally, ...* inside running text. Bullets are for definitions, metric lists, and formal frameworks only — never inside narrative paragraphs.

**Bolded mini-headers.** Inside a subsection, label parallel items with a bold noun phrase ending in a period, then continue in the same paragraph:

> **Soft selection.** Given a batch S, we select negative pairs as samples with different labels.
>
> **Hard selection.** Given a batch S with script and domain labels, negative pairs are chosen according to Algorithm 1.

Use this for strategies, training stages, datasets, models, baselines, metrics, and ablation variants. It is the most recognizable feature of this style.

**Transitions.** Use these, and use them deliberately:

| Purpose | Word |
|---|---|
| Add concrete detail | *Specifically,* |
| Flag a surprise | *Notably,* / *Interestingly,* |
| Contrast | *However,* / *In contrast,* / *By contrast,* |
| Consequence | *Consequently,* / *As a result,* |
| Zoom into one case | *In particular,* |
| Add support | *Moreover,* / *Furthermore,* |

**Acronyms.** Define on first use: *supervised contrastive learning (SCL)*. Use the short form thereafter.

**Section cross-references.** `§3.1`, `Sec. 3.1`, or `Section 3.1` — pick one and stay consistent with the surrounding document.

## Section recipes

### Abstract (~150–250 words)

Five moves, in order, roughly one to two sentences each:

1. **Setup** — what the topic is and why it matters.
2. **Gap** — what is missing, signaled with *However,* or *While X, Y*.
3. **Proposal** — *To address this, we propose X* / *In this work, we introduce X*.
4. **Approach** — how it works, in plain terms, no notation.
5. **Results** — concrete numbers and what they show.

Skeleton:

> Tokenization is the first step of most NLP pipelines. Standard algorithms favor high-resource languages, leaving low-resource ones with longer, lower-quality tokenizations. To remedy this, we introduce X, which does Y. Our method improves Z by N% with negligible cost on global metrics.

### Introduction

The same five moves, expanded to paragraphs:

1. **Broad context** (1 paragraph) — the field, why it matters, what exists. Cite generously.
2. **Specific problem** (1 paragraph) — narrow the focus. *However,* or *While ..., ...* introduces the gap. Often two reasons: *First, ... Second, ...*
3. **Proposed approach** (1–2 paragraphs) — *To mitigate these limitations, we propose ...* Explain conceptually; no equations yet.
4. **Evaluation summary** (1 paragraph) — datasets, headline numbers, what they show.
5. **Contributions** (optional, last paragraph) — *Our key contributions are as follows: ...* Three or four short sentences.

Never open with *In recent years, X has gained significant attention.* Open with a substantive claim about the field.

### Related Work

Organize by theme with `###` subsections or bolded mini-headers (*Tool-Augmented LLMs*, *Multilingual Tokenization*). Each paragraph groups several citations, then ends by positioning the current work against them: *Different from the above, we focus on ...* Cite densely here.

### Method / Methodology

- Open with a roadmap: *In the following, we describe X (§3.1), Y (§3.2), and Z (§3.3).*
- Use mini-header paragraphs for parallel components.
- Number equations, keep them simple, and follow each with a one-sentence English gloss.
- Define each symbol once, then reuse it cleanly.
- Algorithms go in a numbered Algorithm block, referenced from the prose.
- Cite sparingly — only for methods being borrowed.

### Experimental Setup

Pure mini-header territory: **Datasets.** **Models.** **Baselines.** **Metrics.** **Hyperparameters.** Each is a short factual paragraph. State what was done and why when the choice is non-obvious; push the rest to the appendix.

### Results and Analysis

- **Table first.** *Table 2 presents the overall results for X, Y, and Z.* Then walk the reader through the rows.
- **Headline finding first.** *We observe that X improves Y by N%.* Then explain why.
- **Comparative verbs.** *outperforms*, *lags behind*, *matches*, *is competitive with*, *falls short of*.
- **Stratify.** Break results by language, domain, script, resource level, or model size, each under its own mini-header (**Resource Analysis.** **Domain Analysis.** **Script Analysis.**).
- **Always interpret.** A number without an interpretation is an unfinished paragraph.

### Limitations

Candid and direct, three to five sentences. Name what was not tested, what is narrow, what is uncertain. No over-apologizing, no defensiveness. Close with future work where natural: *We leave X to future work.*

### Captions

Self-contained — the reader should understand the table or figure from the caption alone. Define every column and every symbol. Note that best results are **bold** and second-best underlined. Explain asterisks or daggers used for significance. For multi-panel figures, walk through (a), (b), (c) in order.

## Word choice

| Use | Not |
|---|---|
| use | utilize |
| show | demonstrate (occasionally fine, not repeatedly) |
| help, improve | facilitate, enable |
| we propose | we put forth |
| to | in order to |
| for | with respect to |
| because | due to the fact that |

Keep field terminology exact and consistent: *out-of-distribution*, *in-domain*, *fine-tuning*, *zero-shot*, *low-resource*, *baseline*, *ablation*, *held-out*.

## What to avoid

- Hedging on measured results (*may*, *might*, *could potentially*).
- Long sentences with stacked subordinate clauses.
- Filler openings (*It is important to note that*, *It is worth mentioning that*).
- Bullets inside narrative paragraphs.
- Vague verbs (*leverage*, *facilitate*, *enable*) where a concrete one exists.
- Marketing adjectives (*novel*, *cutting-edge*, *state-of-the-art*) more than once in a paper.
- Em-dash-heavy asides and *not X, but Y* constructions.
- Restating the same finding in three consecutive sentences.

## Self-check before returning

Run through this list. It catches most of what goes wrong:

1. Does every paragraph open with its claim and close with its implication?
2. Are measured results unhedged and interpretations hedged?
3. Any sentence over ~35 words that should be two?
4. Any bullets that belong in prose?
5. Are parallel items using bolded mini-headers?
6. Every acronym defined at first use?
7. Every number and citation traceable to the source the user gave?
8. Any *utilize*, *leverage*, *facilitate*, *in order to*, *it is important to note*?

## Worked example

**Input (user's draft):**

> In order to evaluate our approach, we utilized three different benchmark datasets. It is important to note that the results demonstrate that our method may potentially outperform the baselines, particularly in the case of low-resource languages, which is likely due to the fact that the contrastive objective facilitates better generalization across domains, and we also observed improvements on high-resource languages, although these were smaller.

**Output:**

> We evaluate our approach on three benchmark datasets. Our method outperforms the cross-entropy baselines, with the largest gains on low-resource languages. Improvements on high-resource languages are more modest. We hypothesize that the contrastive objective helps the model learn domain-invariant representations, which matters most where training data is concentrated in a single domain.

What changed: *in order to* → *to*; *utilized* → *evaluate*; filler opening removed; *may potentially outperform* → flat claim (it is a measured result); the causal explanation became an explicit hypothesis; one 60-word sentence became four short ones; the paragraph now closes with an interpretation.

## Handling the common requests

**"Rewrite this paragraph."** Return the rewrite, then two or three lines on what changed and why. Do not return a diff unless asked.

**"Turn these notes into a section."** Restructure freely, add transitions, add mini-headers. Add no technical content. Flag any gap where the section recipe expects something the notes do not have.

**"What's wrong with this?"** Give feedback only — do not rewrite. Point to specific sentences and name the rule each one breaks.

**"Make it tighter."** Cut 20% of the words without losing content. Target filler, redundant restatement, and subordinate clauses first.

**"Give me two versions."** One tight, one slightly more detailed. Label them and let the user choose.

When the user's own preferences conflict with this guide, their preferences win. Say so briefly rather than silently overriding either.
