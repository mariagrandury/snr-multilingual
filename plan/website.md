# Website: what is left to make it support the paper

The site (`https://scaling-reliability.netlify.app`, footnote 2 of the
submission) must help reviewers, and the LLM they may use, read the paper. It
must never add a number, a name or a claim that the paper does not have. The
review of 2026-10-18 compared the site with the submitted PDF. The fixes
already made: the repo-docs pages are no longer built (`exclude_docs`), the
Findings pages use the paper's subsection titles, numbered 1.x, 2.x and 3.x
for the paper's Sections 4, 5 and 6, every figure and table has a site number
and, where it has one, names its paper counterpart, every figure opens on the paper's view (other filters are buttons), "DA-target"
replaces "DA-goal", paper Figures 2 and 3 are split into one site figure per
panel, each under its subsection, the
ladder page says how the benchmark counts relate, run names read `L<K>`, and
the Benchmarks table has a Paper column (🔬 analysed, 🔍 screened, - not
evaluated). This file lists what is still open. Figure and table numbers
below are the paper's; "site Table 2" is the site's own number.

## 1. The two guides

The paper's footnote 2 sends reviewers to "the per-benchmark guide". The site
has two tables that answer "from how many tokens of a language is this
benchmark usable?", and they use different rules, so they give different
numbers for the same benchmark and size.

| | Table 7 (site Table 2, Measurability page, `gateThresholdTable`) | Look-up table (Recommender, `lookup`) |
|---|---|---|
| In the paper | yes: Table 7, Appendix D.3 | no |
| Unit | benchmark × answer format × model size, languages pooled | benchmark × language × model size |
| Population | non-English (language, K, checkpoint) evaluations, seed 1904 | the same evaluations, per language |
| Rule | a logistic fit of the share above chance against log tokens of the language; the threshold is the token count where the fit reaches 50 % (and 90 %) | a point is above chance when at least half of the benchmark's tasks in that language are; the threshold is the fewest tokens from which **every** later point stays above chance |
| Output | two token counts per cell, `≤ X` / `> X` at the ends of the trained range | one token count per cell, `?` outside the trained range |
| Behaviour | smooth, reads a share | strict: one later dip below chance removes the threshold |

So the look-up is stricter than Table 7 and per language. Both are correct
for what they compute, but only Table 7 is the paper's guide.

The Recommender's second table (`recommend`) adds a third ingredient that is
not in Table 7: per task and size it marks "use" when the task is above
chance (at least half of the runs) **and** its DA-size or DA-ckpt is at least
0.66, the paper's heuristic of §5. Its `log₁₀ SNR` column does not say which
of the paper's 22 SNR definitions it uses (it should be the
relative-standard-deviation SNR of Figure 4), and it shows "100 % runs above
chance" for BPB, which the paper exempts from the filter.

Decision (2026-11-02): option 1 is done, a sentence under the Recommender's
look-up (site Table 4) calls it a stricter, site-only rule and links site
Table 2 (the paper's Table 7) as the paper's guide. Options 2 and 3 stay open:

1. Label the look-up as a stricter, site-only rule and link Table 7 as the
   paper's guide (one sentence on the Recommender page).
2. Point the Home card "A guide to evaluation configuration" at Table 7
   (`findings/measurability/#table-2`) instead of the Recommender.
3. Rename the `log₁₀ SNR` column to the SNR it uses, and show BPB's
   above-chance cell as "n/a (no chance level)".

## 2. Not yet done

0. **INCLUDE v2 links, after the review.** The benchmark card (Model ladder,
   site Figure 3) links INCLUDE v1 (`CohereForAI/include-base-44`) as the data
   source and no paper. Set the data source to
   `https://huggingface.co/datasets/epfl-nlp/include-2.0` and the paper to
   `https://aclanthology.org/2026.emnlp-main.1344/`: `fallback_source` in
   `scripts/build_site_data.py` (`evaluation()`), the `paper` field of
   `include_v2_og` / `include_v2_en` under `benchmarks` in
   `configs/tasks.json`, and `evaluation.json`. Not done during the review:
   the dataset org names the affiliation, which the anonymity guard of
   `mkdocs_hooks.py` blocks (the strict build fails).
1. **Map the paper to the site on Home.** One line saying the charts are
   interactive versions of the paper's figures, drawn from the same tables,
   and a short map: contribution 1 → Model ladder, contribution 2 (the guide)
   → Table 7, contribution 3 → Findings.
2. **Text-only readers.** The charts are drawn in JavaScript from JSON, so a
   page fetched by an LLM tool shows the captions only. Keep every key number
   in the prose, and add a fallback image of the paper's figure (`<noscript>`
   or an image under each chart).
3. **Pin the chart libraries.** `mkdocs.yml` loads `d3@7`, `plot@0.6` and
   `mathjax@3` from jsdelivr, which resolve to the newest release in that
   range. A release during the review could break the charts. Pin exact
   versions (d3 7.9.0, Plot 0.6.17, MathJax 3.2.2 are the current ones) or
   copy the files into `docs/interactive/vendor/`.

## 3. Figures to re-add to the Findings pages

The PR12 deploy (before PR13) had one page per RQ; its views still exist in
`docs/interactive/views/*.js` and the pages in `docs/findings_all/`. These
views illustrate a claim of the paper or redraw an appendix figure, so they
help a reviewer check the paper. Rules for each one:

- the paper's names only (no `rf_include_base_44`, `blend_sample`, "RQ10",
  "gate", "L", "DA-goal");
- the paper's caption and figure number, and the default view exactly as the
  paper's figure, other settings as buttons;
- data regenerated from the tables behind the paper, then checked against
  the PDF: PR12's surrogate grid read 0.57 at 90M and 0.40 for the 1B
  dist_std SNR where Figure 20 reads 0.58 and 0.39, and its reformulation
  bars counted Belebele 105, CulturalBench-easy 19 and BBH (MCQ) 17 tasks
  where Figure 7 counts 94, 8 and 1.

### Measurability (§4)

| View | Paper | Why |
|---|---|---|
| `formatExample`, `llmRewrite` | §4.2 | One INCLUDE item as letter, RF and LLM-RF. The paper describes the reformulation but shows no example; reviewers will want to see one, and the Limitations mention a manual review of the rewrites. |
| `firstSize` | Figures 6 and 5 | The smallest size above chance per benchmark and language, with the rewritten versions as a button. Backs "34 % of tasks never pass" and "letter formats stay at chance". |
| `reformulation` (with the McNemar stars) | Figure 7 | Backs "from 5 to 86 of 105 languages for Belebele" and the other §4.2 counts. |
| `gateMarginCalc` | §D.1, per-benchmark margin | Shows why small benchmarks need a larger margin (9.00 / 2.44 / 0.74 / 0.60 pp). |
| `scalingCurves` | Figures 9 and 10 | The accuracy curves behind Figure 2, per benchmark. |
| `englishRegimes` | Figure 11 | Backs "the same pattern holds for monolingual English models". |

### Agreement (§5)

| View | Paper | Why |
|---|---|---|
| `daDecisive` (was on the site, hidden 2026-11-02) | §5 Summary | The pairs the 1.7B reference separates by more than two seed SDs: 0.69–0.73 (benchmarks, pooled over pairs) and 0.74–0.94 (BPB, mean over languages, as App. E.3). Its default already matches the paper (2 seed SD, a button for 1 SD); review it with the other appendix views. |
| `daChannels` | Figure 16 | Accuracy against bBPB against per-language BPB; backs §5.4 "BPB-based evaluations generally preserve rankings more reliably". |
| near-ties sensitivity (no view yet) | Figure 13, Table 9 | Backs "accounting for near-tied scores increases DA without changing the main trends". |
| `ltCross` | Figure 15 | HellaSwag cross-language DA; English predicts the others at 0.45, as in §E.5. |
| `sgReference`, `sgGate` | Figure 21, §E.6 | The 3B reference (DA 0.51–0.52) and 612 → 778 tasks above chance (+27 %), as in §5.3 and §E.6. |

### Anticipation (§6)

| View | Paper | Why |
|---|---|---|
| `surrogatesScoring` | Figure 20 | Figure 4 split by scoring (accuracy, bBPB, BPB). |
| `surrogatesVariants` | Figure 19 | The 22 SNR definitions; backs "performance depends strongly on how noise is defined". |
| per-benchmark ρ | §6 | Backs "up to ρ = 0.93 for HellaSwag". |

### Leave out

These answer questions the paper does not ask and carry numbers it does not
report, so they would add confusion: DataDecide (`external`), `subsets`,
`benchmark-design`, `noise-snr`, `above-chance-items`, the design-effect
views (`effect`, `interventions`, `ddLadder`, `early`, `ddClock`), BPB
forecasting (`transferSummary`, `transfer`, `ltTree`), `recipe`, the
public-model floors (`floors`, `gateFloorsLines`), `gateEarly`,
`gateExclusion` and `gateMajority`.
