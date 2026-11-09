# Website: what is left to make it support the paper

The site (`https://scaling-reliability.netlify.app`, footnote 2 of the
submission) must help reviewers, and the LLM they may use, read the paper. It
must never add a number, a name or a claim that the paper does not have.
Figure and table numbers below are the paper's; "site Figure 8" is the site's
own number.

## Done

- 2026-10-18: the repo-docs pages are no longer built (`exclude_docs`); the
  Findings pages use the paper's subsection titles, numbered 1.x–3.x for its
  Sections 4–6; every figure and table has a site number and, where it has
  one, names its paper counterpart; every figure opens on the paper's view;
  "DA-target" replaces "DA-goal"; paper Figures 2 and 3 are one site figure
  per panel; the ladder page says how the benchmark counts relate; run names
  read `L<K>`; the Benchmarks table has a Paper column.
- 2026-11-02:
  - The Recommender's look-up (site Table 5) is called a stricter, site-only
    rule, with a link to site Table 2 (the paper's Table 7) as the paper's
    guide. The Home card "A guide to evaluation configuration" points to site
    Table 2. The Recommender's SNR column is named for the
    relative-standard-deviation SNR it uses, and BPB's above-chance cell reads
    "n/a".
  - Home says the charts are interactive versions of the paper's figures.
  - `mkdocs.yml` pins d3 7.9.0, Plot 0.6.17 and MathJax 3.2.2; every chart box
    carries a `<noscript>` note (`mkdocs_hooks.on_post_page`).
  - The Findings text is cut to the sentences that carry a result: whole
    sentences removed, no wording changed.
  - The appendix views are back on the Findings pages (site Figures 4–21,
    site Table 4):

| Site | View | Paper |
|---|---|---|
| Figure 4 | `gateMarginCalc` | App. D.1 margins (no figure) |
| Figure 5 | `firstSize` | Figure 6 (buttons: Figure 5-style bars, original versions only, every considered benchmark; 3B from the six configurations trained to 3B) |
| Figure 7 | `llmRewrite` | §4.2 / D.2 (no figure) |
| Figure 8 | `reformulation` | Figure 7 |
| Figure 11 | `scalingCurves` | Figures 9 and 10 |
| Figure 12 | `englishRegimes` | Figure 11 |
| Table 4 | static | Table 9 |
| Figure 16 | `daChannels` | Figure 16 |
| Figure 17 | `ltCross` | Figure 15 |
| Figure 19 | `surrogatesVariants` | Figure 19 |
| Figure 20 | `surrogatesBench` (new) | §6 "up to ρ = 0.93 for HellaSwag" |
| Figure 21 | `surrogatesScoring` | Figure 20 (its "accuracy + bBPB" grid repeats site Figure 18) |

  Removed again on 2026-11-03: `sgReference`, `sgGate` (App. E.6) and
  `daDecisive` (§5 Summary). Left out: `gateMajority`, `formatExample` (a
  subset of `llmRewrite`), `bestVariant`, and the views of questions the paper
  does not ask.
- 2026-11-03: the Findings text is the paper's main text again, word for
  word; a collapsed box gives the formal definition of the above-chance
  threshold (App. D.1); site Figures 9 and 10 open on every task with bBPB.

## Proposal: site Figure 11 (accuracy curves, `scalingCurves`)

Not done yet, for discussion:

1. Move it next to the reformulation figures (§1.2): its paper caption, "the
   reformulated versions rise with model size while the letter-format
   original benchmarks stay at chance up to 1.7B", is a §4.2 result, while
   §1.3 is about fits.
2. Show less by default: one line per K (the mean over ladders, builds and
   seeds) instead of every run, so the legend is the five K values only;
   "every run" becomes a button. Drop the "one task, with its fits" mode, or
   keep it collapsed under the chart.
3. Keep one of site Figure 3 (Model ladder score curves) and Figure 11: both
   draw benchmark accuracy across sizes and training and both point to the
   paper's Figures 9 and 10.

## Open

1. **INCLUDE v2 links, after the review.** The benchmark card (Model ladder,
   site Figure 3) links INCLUDE v1 (`CohereForAI/include-base-44`) as the data
   source and no paper. Set the data source to
   `https://huggingface.co/datasets/epfl-nlp/include-2.0` and the paper to
   `https://aclanthology.org/2026.emnlp-main.1344/`: `fallback_source` in
   `scripts/build_site_data.py` (`evaluation()`), the `paper` field of
   `include_v2_og` / `include_v2_en` under `benchmarks` in
   `configs/tasks.json`, and `evaluation.json`. Not done during the review:
   the dataset org names the affiliation, which the anonymity guard of
   `mkdocs_hooks.py` blocks (the strict build fails).
2. **Paper Figures 13, 14 and 18 have no data in the repo** (near-tie
   tolerance curves, tie categories, DA-target of BPB against the tokens of
   the language). Table 9 is typed in from the PDF.
3. **Where the paper's text and the data behind its figures differ.** The
   site's figures show the data; the site's text quotes the paper. A reader
   who hovers can see both:
   - §4.2 / D.2 at 1.7B: "from 5 to 86 of 105 languages for Belebele", "from 3
     to 31 of 43 for INCLUDE", LLM-RF INCLUDE "32 of 43", McNemar
     "p ≤ 5.8e-8". Figure 7's table (`reformulations_gate_paper.csv`, site
     Figure 8) reads 4 → 85, 2 → 31, 31 of 43 and at most 2.98e-8; Global-MMLU
     (1 → 35 of 37) matches.
   - §4 "938 analysed benchmark–language tasks": the Figure 6 table (site
     Figure 5) and the 38% / 34% shares are over 933.
   - §5.2 "59–61% across all pairs": Figure 3a (site Figure 14) reads
     0.60–0.64.
   - §5 Summary "0.51–0.56" on accuracy: the data give 0.50–0.56. App. E.3
     "0.58, 0.57, 0.58, 0.58, 0.58" on every pair: the decisive-pairs view (now off the site) read 0.57–0.59, because its table keeps only the tasks with a seed SD.
   - App. F.4 accuracy rel-std SNR "0.33–0.40": the data give 0.326–0.395.
     Figure 20's caption "on accuracy alone the relative-standard-deviation SNR
     leads or ties" fails at 350M and 1B; Figure 19's caption says "ordered by
     their DA-size correlation", the figure orders by the mean of DA-size and
     DA-ckpt (site Figures 19 and 21 keep the figure's data and drop those
     claims from their captions).
   - Figure 5 counts INCLUDE v2 (OG) in 36 languages, the site in 42; the
     site has no 3B or screened-benchmark data for its Figure 5-style bars.

## 4. Pending website changes (asked on 2026-11-09)

1. Center Table 1 on the Model ladder page. It is aligned to the left.
2. One caption per figure. Each chart has its own note inside the box and a
   second caption below it in the markdown. Merge them into a single caption
   inside the figure that starts with "Figure N" and names the paper figure
   when there is one.
3. Move Figure 4 (the required margin above chance) into the "Formal
   definition of the above-chance threshold" box on the Measurability page.
4. Landing page: add the opening question of the slides ("People choose
   expensive pretraining options from evaluations of small or partially
   trained models. When does that practice make the same decision as the full
   model in multilingual pretraining?") and `motivation.gif`
   (`documents/public/motivation.gif`).
5. Agreement page: update the colour palette of the bar plot.
