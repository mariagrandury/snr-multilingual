# `rfgm` pilot — automated + manual review of the Gemini rewrites

Reviewed: the 23 `pilot_*.jsonl` files in this directory, written by
`rewrite_items_gemini.py pilot` (2026-09-20, Spanish belebele and Basque
`include_base_44` regenerated after the first pass of this review).
**230 items, 23 tasks, every task at 10 items** — no truncated files remain.
Checked against the `SYSTEM` prompt, the per-family `NOTES` and the
`validate()` gate in
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/scripts/rewrite_items_gemini.py`.
Mechanical checks were scripted; every item in Spanish, Basque, Chinese,
Greek, Turkish and Arabic, and most in Persian and Hindi, was then read.

No pilot JSONL was modified.

## Verdict

**Yes — spend the money on belebele + `include_base_44`.** With both
production families now fully covered (160 items), the prompt holds up: zero
script drift at item level, zero option reordering, zero option letters or
numbers in the continuations, negation correct in **24/24** negation-framed
source questions (native wording, never an English template), terminal
punctuation internally consistent in every item, `text == context+"\n"+stem`
and `gold` in range 230/230. The answer-shape leak is **exactly flat** against
the originals corpus-wide (19.6 % vs 19.6 %). The rate of defects that would
actually corrupt a score is ~5–7 %, comparable to the noise already in the
sources.

The full-corpus refresh changed two things about the risk picture, both worth
acting on before the submit:

1. **The trailing-colon reject is language-correlated, not cosmetic.** Basque
   lost **3 of 10** items to it; Persian 2 of 20; every other language 0.
   Basque, Turkish and other verb-final languages reach for a cataphoric
   `… honako hau da:` / `… erabiltzen da:` construction, which is natural
   Basque and correct except for the colon. At production scale this is a
   per-language item-loss gradient — Basque tasks could come back ~30 % shorter
   than Spanish ones, which biases the per-language comparison the whole RQ
   rests on. The `submit --retry` round re-sends the identical prompt, so it
   is not obviously self-healing. **Strip a trailing `:` in `row()` before
   validating** (3 of 5 colon rejects are otherwise-perfect rewrites) and add
   the explicit no-colon line to the prompt.
2. **Position-dependent options ("all of the above", "answers b and c are
   correct") are broken by the format itself, not by the rewriter** — see the
   dedicated section below. 6 items (2.6 %), 3 of them in the production
   families. Pre-filter them.

Two standing qualifications:

- **Do not extend this prompt to `global_mmlu_full` without the fix in
  systematic issue #1.** 11 of 70 Global-MMLU items had their stimulus
  (a quoted source, a lettered list, a multi-sentence scenario) compressed
  away, and at least 5 are now unanswerable. That family is not in this $94
  run. The same defect hit 2 `include_base_44` items (`hindi#437`,
  `turkish#73`), so fix it regardless: it is one sentence of prompt.
- **The pilot is a sample of a distribution, not a preview of the output.**
  `belebele_spa_Latn#316` was rewritten twice across the two pilot passes and
  gave two different (both acceptable) stems — *"no es correcto afirmar sobre
  Blake que"* the first time, *"no es correcto afirmar que Blake"* the second.
  Nothing here is reproducible item-by-item; 230 items over 23 tasks show the
  failure *modes*, not the exact production set.

Every pilot file now holds 10 items. The earlier truncated pair
(`pilot_belebele_spa_Latn.jsonl`, `pilot_include_base_44_basque.jsonl`, 2 rows
each) has been regenerated and is reviewed in full below.

## Mechanical statistics

| family | tasks | items | rejected by `validate()` | script drift (whole item) | per-choice script flip | median stem tok | median choice tok | choice tok IQR | gold-longest **rewritten** | gold-longest **original** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| belebele | 8 | 80 | 1 (1.2 %) | 0 | 0 | 22 | 11.6 | 8.0–17.4 | **20.0 %** | 20.0 % |
| global_mmlu_full | 7 | 70 | 1 (1.4 %) | 0 | 0 | 34 | 9.7 | 3.5–17.3 | **17.1 %** | 15.7 % |
| include_base_44 | 8 | 80 | 4 (5.0 %) | 0 | 8 | 34 | 12.0 | 6.0–18.8 | **21.2 %** | 22.5 % |
| **all** | 23 | 230 | 6 (2.6 %) | 0 | 8 | — | — | — | **19.6 %** | 19.6 % |

Tokens are `len(utf8)/3.5`, the driver's own convention. The 8 per-choice
script flips are two items: `include_base_44_greek#404` (Greek iota → Latin I,
a real defect) and `include_base_44_hindi#437` (Latin `(a) और (b)` gained
Devanagari `केवल`, benign). `include_base_44`'s 5.0 % reject rate is entirely
Basque (3) plus the two Arabic/duplicate and colon cases.

**Answer-shape leak.** Chance is 25 %. Corpus-wide the rewrite sits at
**19.6 % against the originals' 19.6 % — a difference of zero items in 230**,
and both are well below chance. Per family the rewrite moves **±0.0 pp**
(belebele, 20.0 → 20.0), **+1.4 pp** (global_mmlu), **−1.3 pp**
(include_base_44). A stronger directional test agrees: measuring how much each
option *grew* relative to its original, the gold grew more than all three
distractors in only 10 % / 9 % / 6 % of items by family (chance 25 %), and mean
character growth is near-identical for gold and distractors (belebele +2.91 vs
+2.90; global_mmlu +2.04 vs +1.77; include_base_44 +0.76 vs +0.31).
**There is no systematic length tell.** The length leaks that exist are
one-offs, listed below.

Per-file, the highest leak rate is `include_base_44_basque` at 50 % rewritten —
but the *source* options leak at exactly 50 % too. That task's IVAP-exam items
genuinely have a long correct answer; the rewrite adds nothing to it. The
production run should not be judged on that file's absolute number.

Other mechanical results (all 230 items):

- **Order preserved**: 0/230 items where a permutation of `orig_choices`
  matches `choices` better than the identity (best-permutation vs identity
  similarity, threshold 0.35).
- **`gold` in range, `text == context + "\n" + stem`**: 230/230.
- **Option letters/numbers in continuations**: 0.
- **A choice appearing verbatim in the stem**: 0.
- **Stem ends with terminal punctuation**: 6 — the 5 colon cases
  `validate()` rejects, plus `belebele_pes_Arab#316`, which ends in an Arabic
  comma `،` and therefore passes the gate.
- **Stem contains a question mark**: 0.
- **Continuations starting with whitespace or punctuation**: 0.
- **Mixed terminal punctuation inside one item**: 0 (200 items punctuate all
  four, 30 punctuate none — the 30 are every Hindi item, which uses `।`;
  Chinese uses `。`, everything else `.`; never a mix inside an item).
- **Duplicate continuations**: 1 (`include_base_44_arabic#467`, and the
  *source* item has the duplicate).
- **Belebele stem quoting ≥40 chars of the passage**: 0.
- **Negation preserved**: 24/24. All 21 items a multilingual negation regex
  flagged, plus the 3 Basque items it missed because the source is ALL-CAPS,
  carry an explicit language-native negation in the stem; all 6 items the
  regex called "lost" were manually verified as false positives. Examples:
  `belebele_spa_Latn#316` (*"no es correcto afirmar que Blake"*),
  `include_base_44_basque#431` (*"Ez da Euskal Autonomia Erkidegoko Talde
  Psikosozial Judizialaren funtzioa"*), `belebele_hin_Deva#600`
  (*"यह घटना नहीं घटी थी कि"*), `include_base_44_greek#467`
  (*"δεν ανήκει η"*), `global_mmlu_full_es#2860` (*"es falso afirmar que"*).
  This rule is the prompt's strongest result.

## Position-dependent and option-referring options

Checked across the whole corpus, as asked. An option that names other options
by letter, or that says "all/none of the above", is **broken by the cloze
format itself**: the harness scores `stem + " " + choice` one choice at a
time, so such an option never co-occurs with its antecedent. This is true of
the `rf_` twins as well as `rfgm_`, and it is true whether or not the
rewriter touches the option.

**6 items (2.6 % of 230); 3 of them in the production families.**

| item | option | source | rewrite | verdict |
|---|---|---|---|---|
| **[P] `include_base_44_basque#366`** | 3 (**gold**) | *"Zuzenak dira b) eta c) erantzunak."* | *"aurreko bi aukeretan aipatutakoak daude."* | **broken** — letters dropped as instructed, replaced by "those mentioned in the previous two options", which has no antecedent in a continuation prompt |
| **[P] `include_base_44_hindi#437`** | all 4 | *"(a) और (d)"* etc. | *"केवल (a) और (d)।"* | **broken** — letters kept verbatim *and* the (a)–(d) list was deleted from the stem; nothing to refer to |
| **[P] `include_base_44_arabic#172`** | 3 (**gold**) | *"جميع ماذكر"* | *"جميع ما ذكر."* | **broken** — "all of the mentioned" with nothing mentioned |
| [G] `global_mmlu_full_hi#9797` | 3 | *"ऊपर के सभी"* | *"उपर्युक्त सभी हैं।"* | **broken** — same |
| [G] `global_mmlu_full_ar#9633` | 3 (**gold**) | *"كل ما سبق أسباب للارتفاع."* | *"جميع العوامل السابقة مجتمعة."* | **handled** — still positional, but the stem now enumerates the causes, so it reads |
| [G] `global_mmlu_full_es#2034` | 2 (**gold**), 3 | *"Ambos"* / *"Ninguno de esos."* | *"tanto la fuente de voltaje como la de corriente."* / *"ninguna de las fuentes mencionadas."* | **handled well** — expanded into explicit content, which is the only correct move; note it also adds content (see item 26 below) |

So **4 of 6 are unusable**, and the two that work only work because the model
expanded the option into explicit content — which the prompt otherwise forbids
("must not add factual content"). The rewriter cannot win here: obeying the
no-added-content rule guarantees a dangling reference.

**Verdict: pre-filter them out of the production run.** A cheap deterministic
filter in `items()` — drop any item where an option matches a letter-reference
or an "all/none of the above" pattern in that language — costs nothing, removes
a class the format cannot represent, and is more honest than paying Gemini to
produce an item no model can answer. Two conditions:

- **Apply the same filter to the `rf_` twins and to the originals' item
  counts**, or the three sets cover different items and the original/rf/rfgm
  comparison stops being like-for-like. `derive_task_options.py` must run
  after, as the README already requires for the extra rfgm drops.
- Log the dropped ids. At 2.6 % this is ~3 000 items of the 116k run; on a
  per-task basis it will be lumpy (Basque and Hindi `include_base_44` have
  proportionally more of them), and a task that silently loses 10 % of its
  items has a different noise floor from one that loses none.

## Items a human must review

Ranked by severity. **[P]** = in the production run (belebele /
include_base_44); **[G]** = global_mmlu_full, not in this run but blocking a
later one.

### A. Item is unanswerable or its stimulus was destroyed

1. **[P] `include_base_44_hindi#437`** — the question's four lettered
   sub-statements were dropped, but the options still refer to them. Stem:
   *"सामाजिक विज्ञान के मूल्यांकन के दौरान इसका उद्देश्य समझने में मदद करने वाले प्रश्नों में शामिल हैं"*,
   options *"केवल (a) और (d)।"* — nothing says what (a)–(d) are. Pure guessing.
2. **[P] `include_base_44_basque#366`** — gold is
   *"aurreko bi aukeretan aipatutakoak daude"* ("those named in the previous
   two options are [entitled]") after the stem
   *"Konstituzio Auzitegiaren aurrean babes-errekurtsoa jartzeko legitimatuta"*.
   In a continuation prompt there are no previous options. Compounded by a
   meaning shift in distractor 2: *"Interes legitimoak **bultzatutako** edozein
   pertsona **natural** edo juridiko"* → *"interes legitimoa **duen** edozein
   pertsona **fisiko** zein juridiko"* — a legal term of art changed.
3. **[G] `global_mmlu_full_el#5813`** — stem says
   *"Σύμφωνα με το απόσπασμα από την πλατφόρμα του Προοδευτικού Κόμματος του 1912…"*
   but the 1 452-char excerpt is gone. The four options are quotes *from* it.
4. **[G] `global_mmlu_full_tr#5813`** — same item, same defect:
   *"1912 İlerici Parti Platformu metnine göre…"* with no metin.
5. **[G] `global_mmlu_full_fa#5813`** — same item; rejected for the trailing
   colon, but the retry will regenerate the same passage-less stem.
6. **[G] `global_mmlu_full_el#5845`** — the Reagan 1975 quotation (1 512 chars)
   is dropped *and* unreferenced; the stem is now a bare world-knowledge
   question, *"Ένας παράγοντας που δεν συνέβαλε στην ανάδειξη του συντηρητισμού
   στις ΗΠΑ…"*. Different item from the original.
7. **[G] `global_mmlu_full_tr#5845`** / **`fa#5845`** — same, and worse: the
   stem *cites* the missing text (*"Ronald Reagan'ın 1975'teki değerlendirmeleri
   ışığında"* / *"با توجه به دیدگاه‌های رونالد ریگان در دهه ۱۹۷۰"*).
8. **[G] `global_mmlu_full_zh#6078`** — two primary sources (Bentinck 1829,
   Simpson 1826) replaced by a 58-char summary
   *"结合威廉·本廷克关于禁止殉夫习俗的言论与乔治·辛普森对印第安人的看法…"*.
9. **[P] `include_base_44_arabic#172`** — gold *"جميع ما ذكر."* ("all of the
   mentioned") after a stem that mentions nothing.
10. **[P] `include_base_44_turkish#73`** — the CO2/temperature premise
    (508 → 124 chars) is compressed; read it to confirm the item still
    determines its answer.

### B. Degenerate or ungrammatical stem

11. **[P] `belebele_pes_Arab#316`** — **still the worst item in the pilot.**
    Stem is *"با توجه به متن،"* ("According to the passage,") — a framing phrase
    with no predicate, ending in an Arabic comma that `validate()` does not
    catch. The four continuations are complete independent sentences
    (*"این حادثه هیچ تلفات جانی نداشت."*), so the item collapses into
    "which sentence sounds most plausible". The rewriter also had to *add* a
    subject (*"این حادثه"*) to option 0 to make it stand alone.
12. **[P] `include_base_44_basque#56`** — the same failure in the production
    set, and the corpus's only comma-terminated stem:
    *"Administrazio-prozedurako behin-behineko neurriei dagokienez,"*
    ("As regards provisional measures in administrative procedure,"). The
    source "question" is just the topic label *"BEHIN-BEHINEKO NEURRIAK:"*, and
    the four options are independent legal statements, so the rewrite had
    nothing to build a predicate from. Compounded: distractor 2 gained
    *"**beti**"* ("always") — an added universal quantifier that makes a legal
    statement more obviously false than its source.
13. **[P] `include_base_44_chinese#17`** — the mid-sentence blank was moved to
    the end and the result is word salad:
    *"…应当按规定并在距来车方向50米至100米处设置警告标志前开启"*. The original read
    *"…应当按规定开启____并在距来车方向50米至100米处设置警告标志。"*; `按规定` is now
    stranded without its verb.
14. **[P] `include_base_44_persian#404`** — verb-final clash. Stem
    *"با توجه به حکایت «خوش‌اخلاقی»، پزشک عامل اصلی خشمگین نشدن را می‌دانست"* +
    *"صبر و آرامش."* puts the finite verb before its complement; grammatical
    Persian needs *"… را صبر و آرامش می‌دانست"*.
15. **[P] `belebele_zho_Hans#874`** — the stem adds a verb that only some
    options fit: *"…的好处是能看到"* + *"避免雨季的情况。"* = "the benefit is being
    able to *see* the avoiding of the rainy season". Two distractors become
    unreadable, which is a leak toward the gold.
16. **[P] `belebele_pes_Arab#600`** — distractor 3 became self-referential
    nonsense: *"زمین‌لرزه سبب"* + *"شدت 6.5 ریشتری آن شد."* ("the earthquake
    caused its own 6.5 magnitude"). One distractor eliminated for free.
17. **[P] `belebele_pes_Arab#339`** (rejected: colon) — the stem is also
    truncated mid-word in the file: *"مت وزیر تجارت به این شخص برسد:"*.
    Check what the retry produces.

### C. Basque colon rejects — recoverable, but the biggest item-loss risk

18. **[P] `include_base_44_basque#419`**, **`#130`**, **`#26`** — three of
    Basque's ten items, all rejected only for a trailing colon, all otherwise
    correct rewrites that preserve meaning and (in #419 and #130) the negation:
    *"…C1 mailarekin parekatuta ez dagoen titulua edo eskakizuna honako hau
    da:"*, *"Hizkuntzaren ikuspegitik okerra den esaldia hau da:"*,
    *"…teknika zehazki honetarako erabiltzen da:"*. A trailing-colon strip
    recovers all three without a second API round. Secondary note on **#130**:
    the source question is only *"AUKERATU OKERRA."* ("choose the wrong one"),
    and the rewrite invented the criterion *"Hizkuntzaren ikuspegitik"* ("from a
    linguistic point of view") — a reasonable inference, but content the source
    does not contain.

### D. Meaning changed, option enriched, or a free cue added

19. **[P] `belebele_spa_Latn#874`** — gold embellished with a comparative and
    two distractors padded to fit the stem's added verb *"se tienen"*:
    *"Impresionantes vistas de las cataratas"* → *"vistas **más**
    impresionantes de las cataratas."*; *"Menos turistas"* → *"menos turistas
    **alrededor**."*; *"Evitar la temporada de lluvias"* → *"**posibilidades
    de** evitar la temporada de lluvias."*
20. **[P] `belebele_tur_Latn#874`** — *"Şelalelerin dramatik görünümü"* →
    *"şelalelerin **daha etkileyici ve** dramatik görünümüdür."* (now the
    longest of the four).
21. **[P] `belebele_arb_Arab#874`** — *"مناظر درامية للشلالات"* →
    *"رؤية مناظر **أكثر إثارة** للشلالات."*
22. **[P] `belebele_ell_Grek#874`** — *"Εντυπωσιακή όψη"* → *"**πιο**
    εντυπωσιακή όψη"*. **[P] `belebele_eus_Latn#874`** —
    *"Ur-jauzien ikuspegi ikusgarria"* → *"ur-jauziak **ikusgarriagoak izango
    direla**"*. With items 15 and 19–21 this is **one belebele source item
    damaged in six of its eight languages**, always in the same direction: an
    intensifier or comparative added to the gold, or the stem reshaped so only
    the gold reads. Either belebele #874 should be dropped, or this is the
    clearest evidence in the pilot that the model works out the answer and
    writes around it. Worth one human call.
23. **[P] `include_base_44_persian#318`** — the question's role labels were
    changed. Original asks for *"به‌ترتیب پیامد، علت و نتیجۀ"*; the stem says
    *"به‌ترتیب **معلول، علت و پیامد**"*. Different mapping, same options.
24. **[P] `belebele_spa_Latn#339`** — **the stem-final-article risk, confirmed
    in a production family.** Stem ends *"los investigadores determinaron
    **la**"*, so distractor 2 was rewritten to agree: *"Mascotas afectadas"* →
    *"**presencia de** mascotas afectadas."* An added noun with no source. (The
    verb also drifted, *observaron* → *determinaron*.)
25. **[P] `belebele_pes_Arab#711`** — under a negation stem, distractor 0
    gained a named entity: *"دولت مرکزی وجود نداشت"* → *"**اسپانیا** بدون دولت
    مرکزی ماند."* Adding a subject under negation can flip truth value.
26. **[P] `belebele_spa_Latn#638`** — gold rewritten from a source
    mistranslation: *"Visitar el pase de Angkor"* → *"**contar con** el pase de
    Angkor."* The correction is right, but it turns the only nonsensical option
    into the only sensible one.
27. **[P] `belebele_zho_Hans#339`** — distractor 2 gained factual content:
    *"受影响的宠物"* → *"受影响宠物**的身体状况**"*.
    **[P] `belebele_tur_Latn#527`** — gold gained a passage detail:
    *"Alonso'nun pit duruşunun hemen ardından"* → *"Alonso'nun **erken bir**
    pit stopunun hemen ardından"*.
28. **[P] `include_base_44_greek#404`** — the four options are Fontaine stages
    written with **Greek** capital iota (`Ι`, `ΙΙα`, `ΙΙβ`, `ΙΙΙ`) and came back
    in **Latin** (`I.`, `IIα.`, `IIβ.`, `III.`). `validate()`'s majority-script
    check is blind to this because the stem is Greek. Mixed-script options
    tokenize unlike anything in pretraining.
    **[G] `global_mmlu_full_es#2034`** — *"Fuente de voltaje"* → *"la fuente de
    voltaje **exclusivamente**."*: added factual content, item now easier.
    **[G] `global_mmlu_full_ar#1742`** — *"لا يوجد."* ("none of these") →
    *"صفرًا."* ("zero"): a meta-option became a substantive one.
    **[P] `belebele_ell_Grek#880`** — stem narrowed to *"μπορεί να ωφελήσει
    **τον προϋπολογισμό**"*, making the room-and-board gold the only
    budget-relevant option; the Chinese rewrite of the same item kept the
    original framing.
    **[P] `include_base_44_arabic#467`** (rejected: duplicate choices) — the
    *source* has `["السفر", "السفرا", "السفر", "السفر"]`; nothing the rewriter
    can do, and it will be rejected twice, costing two batch rounds.

## Systematic issues (patterns, not one-offs)

Ordered by how much they would cost the production run. Rates are over all
230 items unless stated.

1. **Premise / stimulus compressed away — 13 items (5.7 %): 11 global_mmlu,
   2 include_base_44.** Any question carrying a quoted source, a
   lettered/numbered list or a multi-sentence scenario gets summarised into the
   stem, and sometimes dropped while the stem still cites it. Detected as
   `len(stem)/len(orig_question) < 0.40` on questions over 200 chars. Character
   unchanged by the refresh — it remains a global_mmlu-dominated problem.
   *Prompt fix:* "If the question contains a quoted passage, a lettered or
   numbered list, or a multi-sentence scenario, reproduce it verbatim at the
   start of the stem — never summarise, paraphrase or omit it."

2. **Trailing-colon stems — 5 items (2.2 %), and the distribution is the
   finding: Basque 3/10, Persian 2/20, all six other languages 0/180.**
   Previously read as a flat cosmetic 1.9 %; with Basque covered it is clearly
   a verb-final-language pattern (cataphoric `honako hau da:`,
   `erabiltzen da:`) and therefore a *per-language item-loss gradient*, which
   is the one kind of bias this RQ cannot absorb. All five are otherwise
   correct rewrites.
   *Driver fix (do this one):* strip a trailing `:` in `row()` and re-validate
   before writing the item to the retry file. *Prompt fix:* "The stem must not
   end with a colon, even where the language would normally introduce the
   completion with one."

3. **Option embellished with an intensifier, comparative, quantifier or added
   noun — ~17 items (7.4 %).** Up from ~12: the Spanish and Basque refresh
   added `spa#874` (comparative on gold, padding on two distractors),
   `spa#339` (*"presencia de"*), `spa#638` (*"contar con"*), `basque#56`
   (*"beti"*), `basque#366` (*"fisiko"/"duen"*). The aggregate length
   statistics do **not** show this — gold grows no faster than distractors —
   so it is a qualitative leak, not a length leak, and it concentrates on items
   whose source options are bare noun phrases.
   *Prompt fix:* "Never add an intensifier, comparative, quantifier or
   qualifying phrase to an option that did not have one, even to make the
   sentence read better."

4. **Under-specified stems — 15 items (6.5 %), 2 pathological.** Stems of ≤4
   words, or a framing phrase with no predicate. Most are harmless
   (`belebele_spa_Latn#527`: *"El accidente ocurrió"*; `belebele_tur_Latn#527`:
   *"Kaza"* + *"… gerçekleşmiştir."* is fine Turkish). The two pathological
   cases — `belebele_pes_Arab#316` and now `include_base_44_basque#56`, one in
   each production family — turn the item into sentence-plausibility. Both
   arise where the *source* question is a bare topic label.
   *Prompt fix:* "The stem must state the full proposition minus its final
   constituent; it may not end at a comma, nor consist only of a framing phrase
   such as the input-language equivalent of 'According to the passage,' or a
   bare topic label."

5. **Quotation marks or brackets added around all four options — 11 items
   (4.8 %).** `«…»` in Greek, Persian, Spanish and now Basque (`#130`), `“…”`
   in Chinese, `"…"` in Turkish, on options that had none. Applied uniformly so
   it is not a leak, but it changes the surface form against the original and
   the `rf_` twin, and costs output tokens.
   *Prompt fix:* "Do not add quotation marks, guillemets or brackets that the
   original option did not have."

6. **Stems ending in a gendered or definite article — 10 items (4.3 %), and
   the risk is now confirmed rather than latent.** e.g.
   `global_mmlu_full_es#7868` *"…es **el**"*, `include_base_44_spanish#303`
   *"…transformar **la**"*, `include_base_44_greek#467` *"…δεν ανήκει **η**"*.
   In **two** of them the model rewrote an option's head noun purely to force
   agreement — `greek#467` (*"Υπορενιναιμικός υποαλδοστερονισμός"* →
   *"**κατάσταση** υπορενιναιμικού υποαλδοστερονισμού"*) and the newly
   reviewed `belebele_spa_Latn#339` (*"Mascotas afectadas"* → *"**presencia
   de** mascotas afectadas"*). Wherever an option's gender differs, this either
   leaks the answer or silently alters the option.
   *Prompt fix:* "Do not end the stem with an article, determiner, classifier
   or any word whose form depends on the choice that follows."

7. **Mid-sentence blanks relocated badly — 11 source items carry a blank, ~4
   rewrites are awkward or ungrammatical** (`include_base_44_chinese#17` is
   broken; `persian#404`, `persian#43`, `hindi#144` are strained). Verb-final
   and blank-medial languages are where "stop the stem where the answer goes"
   fights the grammar. Unchanged by the refresh.
   *Prompt fix:* "If the blank is not at the end of the sentence, restructure
   the whole sentence so the missing constituent is final — do not simply move
   the surrounding words around the blank."

8. **Position-dependent / option-referring options — 6 items (2.6 %), 3 in the
   production families.** New pattern; full analysis in its own section above.
   4 of 6 are unusable in a continuation format regardless of what the
   rewriter does.
   *Fix:* pre-filter in `items()`, and apply the same filter to `rf_` and to
   the originals' item counts.

9. **Digit-system conversion — 5 items (2.2 %), all Persian.** ASCII digits in
   the source come back as Eastern Arabic-Indic numerals
   (`['2020','2050','2025','2040']` → `['۲۰۲۰.','۲۰۵۰.','۲۰۲۵.','۲۰۴۰.']`).
   Meaning-preserving but a different surface string from the original and the
   `rf_` twin, which weakens the three-way comparison.
   *Prompt fix:* "Copy numerals exactly as written in the input; never convert
   between digit systems."

10. **Mixed-script option sets survive `validate()` — 1 observed
    (`include_base_44_greek#404`), and the check cannot see it by
    construction.** `script_of()` takes the majority script of stem + all
    choices together, so a short option flipping scripts is invisible.
    *Driver fix:* also compare `script_of(choices[i])` against
    `script_of(orig_choices[i])` per choice, where both are non-empty.

11. **Degenerate source items are paid for twice — 1 observed
    (`include_base_44_arabic#467`, three identical source options).**
    *Driver fix:* skip items whose `orig_choices` contain duplicates in
    `items()`, before building the request.

## Suggested minimal change set before the production submit

- **Driver, do these first** (they cost nothing and buy the most): strip a
  trailing colon before `validate()` (issue 2 — the Basque item-loss risk);
  pre-filter position-dependent and option-referring options (issue 8),
  applying the same filter to `rf_`; pre-filter duplicate source options
  (issue 11); per-choice script check (issue 10).
- **Prompt**, five one-line additions covering issues 1, 2, 3, 4 and 6
  (premise verbatim; no stem-final colon; no added intensifiers; no bare-frame
  or comma-ending stems; no stem-final determiner). ~50 tokens on a 550-token
  scaffold, ≈ +9 % on the input bill — under $5 on this run.
- Issues 5, 7 and 9 (added quotes, relocated blanks, digit conversion) are
  cosmetic-to-moderate; fold their prompt lines in if the prompt is being
  edited anyway, but none of them justifies a re-pilot on its own.
