# `rfgm` pilot — automated + manual review of the Gemini rewrites

Reviewed: the 23 `pilot_*.jsonl` files in this directory (214 rows, written
2026-09-20 by `rewrite_items_gemini.py pilot`), against the `SYSTEM` prompt,
the per-family `NOTES` and the `validate()` gate in
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/scripts/rewrite_items_gemini.py`.
Mechanical checks were scripted; every item in Spanish, Chinese, Greek,
Turkish, Basque and Arabic, and most in Persian and Hindi, was then read.

No pilot file was modified.

## Verdict

**Yes — spend the money on belebele + `include_base_44`.** The prompt works.
Across 144 items in those two families there is zero script drift, zero
option-reordering, zero option letters/numbers leaking into the continuations,
the negation rule is honoured in 22/22 negated items, terminal punctuation is
consistent within every item, and the answer-shape leak is **not** raised over
the originals (see the table). The defect rate that would actually corrupt a
score is roughly **4–6 %** — one degenerate stem, a handful of embellished gold
options, a few mangled blank-medial items — which is comparable to the noise
already in these source datasets (belebele #874's option set is awkward in
every language; `include_base_44_arabic#467` ships three identical options
upstream).

Two qualifications:

1. **Do not extend the same prompt to `global_mmlu_full` without the fix in
   systematic issue #1.** 11 of 70 Global-MMLU items had their stimulus
   (a quoted source text, a lettered list, a multi-sentence scenario)
   *compressed away*, and at least 5 are now unanswerable. That family is not
   in this $94 run — keep it that way until the prompt tells the model to
   reproduce the premise verbatim. The same defect hit 2 `include_base_44`
   items (`hindi#437`, `turkish#73`), so it is worth fixing before the run
   regardless: it is one sentence of prompt.
2. **The pilot is thin.** 144 items over 16 tasks, ~9 per task, with only two
   languages per family read end-to-end by a fluent speaker. The findings below
   are what 214 items can show; a 116k-item run will surface tail cases this
   sample cannot.

### Pilot files with fewer than 10 items

Two files come from the earlier truncated test run and carry **2 rows each**:

- `pilot_belebele_spa_Latn.jsonl` — 2 rows
- `pilot_include_base_44_basque.jsonl` — 2 rows (1 of them a reject)

Every other file has exactly 10. Spanish belebele and Basque
`include_base_44` are therefore effectively unreviewed; `pilot --force` on
those two tasks costs cents and should be run before the production submit.
Note this also skews the per-file leak numbers (Basque shows "100 %" on n=2).

## Mechanical statistics

| family | tasks | items | rejected by `validate()` | script drift (whole item) | per-choice script flip | median stem tok | median choice tok | choice tok IQR | gold-longest **rewritten** | gold-longest **original** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| belebele | 8 | 72 | 1 (1.4 %) | 0 | 0 | 23 | 11.9 | 8.3–17.7 | **19.4 %** | 18.1 % |
| global_mmlu_full | 7 | 70 | 1 (1.4 %) | 0 | 0 | 34 | 9.7 | 3.5–17.3 | **17.1 %** | 15.7 % |
| include_base_44 | 8 | 72 | 2 (2.8 %) | 0 | 8 | 35 | 11.7 | 5.8–17.6 | **19.4 %** | 20.8 % |
| **all** | 23 | 214 | 4 (1.9 %) | 0 | 8 | — | — | — | **18.7 %** | 18.2 % |

Tokens are `len(utf8)/3.5`, the driver's own convention.

**Answer-shape leak.** Chance is 25 %. The rewrite sits at 18.7 % overall
against 18.2 % for the originals — +0.5 pp, i.e. one item in 214, and *below*
chance in both directions. Per family the rewrite moves +1.3 pp (belebele),
+1.4 pp (global_mmlu), **−1.4 pp** (include_base_44). A stronger directional
test agrees: measuring how much each option *grew* relative to its original,
the gold grew more than all three distractors in only 10 % / 9 % / 7 % of items
by family (chance 25 %), and mean character growth is essentially identical for
gold and distractors (belebele +3.0 vs +2.9; global_mmlu +2.0 vs +1.8;
include_base_44 +0.85 vs +0.82). **There is no systematic length tell.** The
length leaks that exist are one-offs, listed below.

Other mechanical results (all 214 items):

- **Order preserved**: 0/214 items where a permutation of `orig_choices`
  matches `choices` better than the identity (best-permutation vs identity
  similarity, threshold 0.35).
- **`gold` in range, `text == context + "\n" + stem`**: 214/214.
- **Option letters/numbers in continuations**: 0.
- **A choice appearing verbatim in the stem**: 0.
- **Stem ends with terminal punctuation**: 4 (all four are the colon rejects).
- **Stem contains a question mark**: 0.
- **Continuations starting with whitespace or punctuation**: 0.
- **Mixed terminal punctuation inside one item**: 0 (Hindi uses `।`
  consistently, Chinese `。`, everything else `.` — 184 items punctuate all
  four, 30 punctuate none, never a mix).
- **Duplicate continuations**: 1 (`include_base_44_arabic#467`, and the
  *source* item has the duplicate).
- **Belebele stem quoting ≥40 chars of the passage**: 0.
- **Negation preserved**: 22/22 negated source questions carry an explicit
  language-native negation in the stem. This rule is working well — see
  `belebele_spa_Latn#316` ("no es correcto afirmar sobre Blake que"),
  `belebele_hin_Deva#600` ("यह घटना नहीं घटी थी कि"),
  `include_base_44_greek#467` ("δεν ανήκει η"),
  `global_mmlu_full_es#2860` ("es falso afirmar que"). No English template
  appeared anywhere.

## Items a human must review

Ranked by severity. Family tags: **[P]** = in the production run
(belebele / include_base_44), **[G]** = global_mmlu_full, not in this run but
blocking a later one.

### A. Item is unanswerable or its stimulus was destroyed

1. **[P] `include_base_44_hindi#437`** — the question's four lettered
   sub-statements were dropped, but the options still refer to them. Stem:
   *"सामाजिक विज्ञान के मूल्यांकन के दौरान इसका उद्देश्य समझने में मदद करने वाले प्रश्नों में शामिल हैं"*,
   options *"केवल (a) और (d)।"* — nothing in the item says what (a)–(d) are.
   The 376-char original listed them. Item is now pure guessing.
2. **[G] `global_mmlu_full_el#5813`** — stem says
   *"Σύμφωνα με το απόσπασμα από την πλατφόρμα του Προοδευτικού Κόμματος του 1912…"*
   but the 1 452-char platform excerpt is gone. The four options are quotes
   *from* that excerpt.
3. **[G] `global_mmlu_full_tr#5813`** — same item, same defect:
   *"1912 İlerici Parti Platformu metnine göre…"* with no metin.
4. **[G] `global_mmlu_full_fa#5813`** — same item; `validate()` rejected it for
   the trailing colon, but the retry will regenerate the same passage-less stem.
5. **[G] `global_mmlu_full_el#5845`** — the Reagan 1975 quotation (1 512 chars)
   is dropped *and* unreferenced: the stem is now a bare world-knowledge
   question, *"Ένας παράγοντας που δεν συνέβαλε στην ανάδειξη του συντηρητισμού
   στις ΗΠΑ…"*. Different item from the original.
6. **[G] `global_mmlu_full_tr#5845`** / **`fa#5845`** — same, and worse: the
   stem *cites* the missing text (*"Ronald Reagan'ın 1975'teki değerlendirmeleri
   ışığında"* / *"با توجه به دیدگاه‌های رونالد ریگان در دهه ۱۹۷۰"*).
7. **[G] `global_mmlu_full_zh#6078`** — two primary sources (Bentinck 1829,
   Simpson 1826) replaced by a 58-char summary
   *"结合威廉·本廷克关于禁止殉夫习俗的言论与乔治·辛普森对印第安人的看法…"*.
8. **[P] `include_base_44_turkish#73`** — the CO2/temperature premise paragraph
   (508 → 124 chars) is compressed; worth a read to confirm the item still
   determines its answer.

### B. Degenerate or ungrammatical stem

9. **[P] `belebele_pes_Arab#316`** — **the worst item in the pilot.** Stem is
   *"با توجه به متن،"* ("According to the passage,") — a framing phrase with no
   predicate. The four continuations are complete independent sentences
   (*"این حادثه هیچ تلفات جانی نداشت."*), so the item collapses into
   "which sentence sounds most plausible", not a reading-comprehension test.
   `validate()` passes it. Note the rewriter also had to *add* a subject
   (*"این حادثه"*) to option 0 to make it stand alone.
10. **[P] `include_base_44_chinese#17`** — the mid-sentence blank was moved to
    the end and the result is word salad:
    *"…应当按规定并在距来车方向50米至100米处设置警告标志前开启"*. The original read
    *"…应当按规定开启____并在距来车方向50米至100米处设置警告标志。"*; `按规定` is now
    stranded without its verb.
11. **[P] `include_base_44_persian#404`** — verb-final clash. Stem
    *"با توجه به حکایت «خوش‌اخلاقی»، پزشک عامل اصلی خشمگین نشدن را می‌دانست"* +
    *"صبر و آرامش."* puts the finite verb before its complement; grammatical
    Persian needs *"… را صبر و آرامش می‌دانست"*.
12. **[P] `belebele_zho_Hans#874`** — the stem adds a verb that only some
    options fit: *"…的好处是能看到"* + *"避免雨季的情况。"* = "the benefit is being
    able to *see* the avoiding of the rainy season". Two of three distractors
    become unreadable, which is a leak toward the gold.
13. **[P] `belebele_pes_Arab#600`** — distractor 3 became self-referential
    nonsense: *"زمین‌لرزه سبب"* + *"شدت 6.5 ریشتری آن شد."* ("the earthquake
    caused its own 6.5 magnitude"). One distractor eliminated for free.
14. **[P] `belebele_pes_Arab#339`** (rejected: trailing colon) — the stem is
    also truncated mid-word in the file: *"مت وزیر تجارت به این شخص برسد:"*.
    Check what the retry produces.
15. **[P] `include_base_44_basque#419`** (rejected: trailing colon) — the
    negation *was* handled correctly (*"…baliozkotuta ez dagoen egiaztagiria
    honako hau da:"*); only the colon is wrong. A trailing-colon strip would
    recover this item without a second API round.

### C. Meaning changed, option enriched, or a free cue added

16. **[P] `belebele_tur_Latn#874`** — gold embellished and made the longest of
    the four: *"Şelalelerin dramatik görünümü"* → *"şelalelerin **daha
    etkileyici ve** dramatik görünümüdür."*
17. **[P] `belebele_arb_Arab#874`** — same item, same direction:
    *"مناظر درامية للشلالات"* → *"رؤية مناظر **أكثر إثارة** للشلالات."*
18. **[P] `belebele_ell_Grek#874`** — *"Εντυπωσιακή όψη"* → *"**πιο**
    εντυπωσιακή όψη"*.
19. **[P] `belebele_eus_Latn#874`** — *"Ur-jauzien ikuspegi ikusgarria"* →
    *"ur-jauziak **ikusgarriagoak izango direla**"* (comparative + future).
    Items 16–19 are one source item rewritten in four languages, and the model
    added an intensifier to the gold in **all four**. Either the source item is
    unsalvageable or the model is quietly answering it; worth one human call on
    whether belebele #874 should be dropped.
20. **[P] `include_base_44_persian#318`** — the question's role labels were
    changed. Original asks for *"به‌ترتیب پیامد، علت و نتیجۀ"*; the stem says
    *"به‌ترتیب **معلول، علت و پیامد**"*. Different mapping, same options.
21. **[P] `belebele_pes_Arab#711`** — under a negation stem, distractor 0 gained
    a named entity: *"دولت مرکزی وجود نداشت"* → *"**اسپانیا** بدون دولت مرکزی
    ماند."* Adding a subject under negation can flip truth value.
22. **[P] `belebele_zho_Hans#339`** — distractor 2 gained factual content:
    *"受影响的宠物"* → *"受影响宠物**的身体状况**"*.
23. **[P] `belebele_tur_Latn#527`** — gold gained a passage detail not in the
    option: *"Alonso'nun pit duruşunun hemen ardından"* → *"Alonso'nun **erken
    bir** pit stopunun hemen ardından"*.
24. **[G] `global_mmlu_full_es#2034`** — two degenerate options were
    de-degenerated and a restriction was added. *"Ambos"* → *"tanto la fuente de
    voltaje como la de corriente."* (now the longest), and *"Fuente de voltaje"*
    → *"la fuente de voltaje **exclusivamente**."* The "exclusivamente" is added
    factual content; the item is easier than the original.
25. **[G] `global_mmlu_full_ar#1742`** — a meta-option became a substantive one:
    *"لا يوجد."* ("none of these") → *"صفرًا."* ("zero"). Different option.
26. **[P] `include_base_44_greek#404`** — the four options are Fontaine stages
    written with **Greek** capital iota (`Ι`, `ΙΙα`, `ΙΙβ`, `ΙΙΙ`) and came back
    in **Latin** (`I.`, `IIα.`, `IIβ.`, `III.`). `validate()`'s majority-script
    check is blind to this because the stem is Greek. Mixed-script options
    tokenize differently from anything the model saw in pretraining.
27. **[P] `belebele_ell_Grek#880`** — the stem narrows the question from "how is
    volunteering beneficial for a budget traveller" to *"μπορεί να ωφελήσει
    **τον προϋπολογισμό**"*, which makes the room-and-board gold the only
    budget-relevant option. Compare the Chinese rewrite of the same item, which
    kept the original framing.
28. **[P] `include_base_44_arabic#467`** (rejected: duplicate choices) — the
    *source* item has `["السفر", "السفرا", "السفر", "السفر"]`. Nothing the
    rewriter can do; it will be rejected twice and cost two batch rounds.

## Systematic issues (patterns, not one-offs)

Ordered by how much they would cost the production run.

1. **Premise / stimulus compressed away — 13 items (6.1 %): 11 global_mmlu,
   2 include_base_44.** Any question that carries a quoted source, a
   lettered/numbered list or a multi-sentence scenario gets summarised into the
   stem, and sometimes dropped entirely while the stem still cites it. Detected
   as `len(stem)/len(orig_question) < 0.40` on questions over 200 chars.
   *Prompt fix:* "If the question contains a quoted passage, a lettered or
   numbered list, or a multi-sentence scenario, reproduce it verbatim at the
   start of the stem — never summarise, paraphrase or omit it."

2. **Under-specified stems — 13 items (6.1 %), 1 pathological.** Stems of ≤4
   words, or ending at a comma after a framing phrase
   (`belebele_pes_Arab#316`: *"با توجه به متن،"*). Most are harmless
   (`belebele_tur_Latn#527`: *"Kaza"* + *"… gerçekleşmiştir."* is fine Turkish),
   but the pathological case turns the item into sentence-plausibility.
   *Prompt fix:* "The stem must state the full proposition minus its final
   constituent; it may not end at a comma, nor consist only of a framing phrase
   such as the input-language equivalent of 'According to the passage,'."

3. **One option embellished with an intensifier, comparative or added noun —
   ~12 items (5.6 %), and it lands on the gold more often than on a distractor
   in the cases that matter** (items 16–23 above). The aggregate length
   statistics do *not* show this (gold grows no faster than distractors overall),
   so it is a qualitative leak, not a length leak. It concentrates on items
   whose original option set is a bare noun phrase.
   *Prompt fix:* "Never add an intensifier, comparative, quantifier or
   qualifying phrase to an option that did not have one, even to make the
   sentence read better."

4. **Mid-sentence blanks relocated badly — 11 items have a blank in the source
   question, and ~4 of the rewrites are awkward or ungrammatical**
   (`include_base_44_chinese#17` is broken; `persian#404`, `persian#43`,
   `hindi#144` are strained). Verb-final and blank-medial languages are where
   "stop the stem where the answer goes" fights the grammar.
   *Prompt fix:* "If the blank is not at the end of the sentence, restructure
   the whole sentence so the missing constituent is final — do not simply move
   the surrounding words around the blank."

5. **Quotation marks / brackets added around all four options — 10 items
   (4.7 %).** `«…»` in Greek, Persian and Spanish, `“…”` in Chinese, `"…"` in
   Turkish, on options that had none. Applied uniformly so it is not a leak, but
   it changes the surface form against the original and costs output tokens.
   *Prompt fix:* "Do not add quotation marks, guillemets or brackets that the
   original option did not have."

6. **Stems ending in a gendered or definite article — 9 items (4.2 %).** e.g.
   `global_mmlu_full_es#7868` *"…es **el**"*, `include_base_44_greek#467`
   *"…δεν ανήκει **η**"*, `include_base_44_spanish#303` *"…transformar **la**"*.
   All nine happen to have four agreeing options, so nothing broke — but the
   model demonstrably *rewrites an option's head noun to force agreement*
   (`greek#467`: *"Υπορενιναιμικός υποαλδοστερονισμός"* → *"**κατάσταση**
   υπορενιναιμικού υποαλδοστερονισμού"*). Wherever one option's gender differs,
   this either leaks the answer or silently alters an option.
   *Prompt fix:* "Do not end the stem with an article, determiner, classifier or
   any word whose form depends on the choice that follows."

7. **Digit-system conversion — 5 items (2.3 %), all Persian.** ASCII digits in
   the source options come back as Eastern Arabic-Indic numerals
   (`['2020','2050','2025','2040']` → `['۲۰۲۰.','۲۰۵۰.','۲۰۲۵.','۲۰۴۰.']`).
   Meaning-preserving but a different surface string, and tokenized differently
   from the original and `rf_` twins, which weakens the three-way comparison.
   *Prompt fix:* "Copy numerals exactly as written in the input; never convert
   between digit systems."

8. **Trailing-colon stems — 4 items (1.9 %), the only recurring `validate()`
   reject.** All four are otherwise correct rewrites. The `submit --retry`
   round re-sends the identical prompt, so the model has no reason to answer
   differently; three of the four would be recovered by simply stripping a
   trailing `:` in `row()` before validating.
   *Prompt fix (or driver fix):* strip a trailing colon and re-validate before
   sending the item to the retry file.

9. **Mixed-script option sets survive `validate()` — 1 observed
   (`include_base_44_greek#404`), and the check cannot see it by construction.**
   `script_of()` takes the majority script of stem + all choices together, so a
   short option flipping scripts is invisible.
   *Driver fix:* also compare `script_of(choices[i])` against
   `script_of(orig_choices[i])` per choice, where both are non-empty.

10. **Degenerate source items are paid for twice — 1 observed
    (`include_base_44_arabic#467`, three identical source options).**
    *Driver fix:* skip items whose `orig_choices` already contain duplicates in
    `items()`, before building the request.

## Suggested minimal change set before the production submit

- The four one-line prompt additions for issues **1, 2, 3, 6** (premise
  verbatim, no bare-frame stems, no added intensifiers, no stem-final
  determiner). These cover every defect in the production families that a
  prompt can reach, and add ~40 tokens to a 550-token scaffold (≈ +7 % on the
  input bill, i.e. under $4 on this run).
- The three driver-side fixes for issues **8, 9, 10** (strip trailing colon,
  per-choice script check, pre-filter duplicate source options). All three are
  a few lines and save API rounds rather than costing them.
- Re-run `pilot --force --tasks belebele_spa_Latn,include_base_44_basque` so
  those two tasks have 10 items each.
