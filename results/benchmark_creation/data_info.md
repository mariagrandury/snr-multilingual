# Benchmark metadata for `benchmark_creation/` analysis

Fill one row per benchmark family. `family` keys must match
`multilingual.analyze_snr_variants.benchmark_family` — i.e., the
prefix of the multilingual task name with the language token stripped.
The 12 families currently in scope (from
`collect_multilingual_families`) are listed below.

The paragraphs that follow each summarise the underlying paper for that
family, cross-referenced against the
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
task READMEs which fix the actual datasets the eval harness loads. The
table at the bottom encodes the same information in the schema
consumed by `INSTRUCTIONS.md` Step 2.

A non-trivial number of `_eu` (Basque) tasks are **independently
created** datasets sourced from different papers than the rest of
their family. These are flagged per-family below and rolled up in the
final notes.

---

## Per-benchmark notes

### arc

**ARC** ([Clark et al., 2018](https://arxiv.org/abs/1803.05457)) is the
AI2 Reasoning Challenge: 7,787 grade-school science multiple-choice
questions authored for human standardized tests, partitioned into a
2,590-question Challenge set (only items missed by both a retrieval and
a word-co-occurrence baseline) and a 5,197-question Easy set. The
multilingual variants used here (`arc_<lang>`) come from
[Okapi (Lai et al., 2023)](https://arxiv.org/abs/2307.16039), which
machine-translated ARC from English into 26 languages — including
Basque — using ChatGPT (`okapi/arc_multilingual` in the harness).
Apertus exposes the English `arc_challenge` / `arc_easy` plus 9 Okapi
translations (ar, de, es, eu, fr, hi, ru, vi, zh); the `arc_eu` row
is therefore part of the same Okapi release as the rest, not an
external addition.

### belebele

**Belebele** ([Bandarkar et al., 2024](https://aclanthology.org/2024.acl-long.44/))
is a parallel multiple-choice machine reading comprehension benchmark
covering 122 language variants. Each item gives a short passage drawn
from FLORES-200 plus a question with four answer options. The dataset
was constructed end-to-end by human experts fluent in English and the
target language — explicitly *without* machine translation — making it
the largest fully-human multilingual MRC resource at the time of
release. Apertus uses 12 BCP-47 / FLORES-coded subsets (arb_Arab,
eng_Latn, eus_Latn, hin_Deva, jpn_Jpan, rus_Cyrl, spa_Latn, swh_Latn,
tha_Thai, tur_Latn, vie_Latn, zho_Hans).

### global_mmlu (Lite-style split)

**Global-MMLU** ([Singh et al., 2024](https://arxiv.org/abs/2412.03304))
is a multilingual extension of MMLU
([Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300)) released by
Cohere Labs across 42 languages. The release ships two splits: a
**Lite** version restricted to 14 languages and containing only items
that are fully human-translated or human post-edited (200 "culturally
sensitive" + 200 "culturally agnostic" questions per language); and a
**Full** version (see next entry). The Apertus parquet exposes
`global_mmlu_<lang>` for six languages (ar, de, es, fr, hi, zh) — all
within the Lite-eligible 14, and the per-row counts (~400/language)
match the Lite design. Each language is further broken down into five
MMLU super-categories (business, humanities, medical, other,
social_sciences, stem) which are filtered out of the per-language
aggregate by `collect_multilingual_families`.

### global_mmlu_full

**Global-MMLU-Full**, from the same Cohere release as above, contains
all ~14k MMLU questions translated into 42 languages using a
multi-stage pipeline that combines machine translation with
crowd-sourced and expert post-editing, but does *not* enforce full
human translation for every item. Apertus uses 10 language subsets
(ar, bn, de, es, fr, hi, id, it, sw, zh) and exposes per-subject
keys `global_mmlu_full_<lang>_<subject>` for the 57 MMLU subjects.

### global_piqa_completions

**Global-PIQA** ([Arnett et al., 2025](https://arxiv.org/abs/2510.24081))
is the MRL 2025 shared-task benchmark for physical commonsense
reasoning across 116 language varieties, hand-built by ~320 native-
speaker researchers from 65 countries. Unlike translation-based
multilingual benchmarks, Global-PIQA is a *participatory* dataset:
items are written in the target language from scratch and over half
incorporate culturally-specific references (local foods, customs,
traditions). The harness ships two task variants: `*_completions`
(log-probability completion, intended for pretrained models) and
`*_prompted` (string-matched generation, intended for instruction-
tuned models). Apertus uses the completions variant on 11 language /
script subsets (arb_arab, cmn_hans, eng_latn, hin_deva, jpn_jpan,
rus_cyrl, spa_latn / spa_latn_spai, swh_latn, tha_thai, tur_latn,
vie_latn).

### hellaswag

**HellaSwag** ([Zellers et al., 2019](https://aclanthology.org/P19-1472/))
is a sentence-completion commonsense benchmark built by Adversarial
Filtering: candidate endings are sampled from a generative model and
iteratively filtered so that surface-form lexical cues do not give the
right answer away. Items come from ActivityNet captions and WikiHow
articles. The multilingual variants here are again from
[Okapi (Lai et al., 2023)](https://arxiv.org/abs/2307.16039), which
ChatGPT-translated HellaSwag into 30 languages including Basque
(`okapi/hellaswag_multilingual` in the harness). Apertus uses the
English HellaSwag plus 8 Okapi translations (ar, de, es, eu, fr, hi,
ru, vi, zh).

### multiblimp

**MultiBLiMP 1.0**
([Jumelet et al., 2025](https://arxiv.org/abs/2504.02768)) extends BLiMP
([Warstadt et al., 2020](https://aclanthology.org/2020.tacl-1.25/)) to a
massively multilingual scale: 125k+ minimal pairs over 101 languages
covering subject-verb agreement phenomena. The pairs are produced by
a fully automated pipeline that recombines morphological and syntactic
information from Universal Dependencies and UniMorph — there is no
translation, and no native-speaker authoring of items. Performance is
measured by whether the model assigns higher probability to the
grammatical sentence than to its minimally-modified ungrammatical
counterpart. Apertus uses 7 ISO-639-3-coded subsets (arb, eng, eus,
hin, rus, spa, tur). Note that Basque is part of the original
MultiBLiMP release — `multiblimp_eus` is **not** an independent
addition.

### paws

**PAWS-X** ([Yang et al., 2019](https://aclanthology.org/D19-1382/)) is
the cross-lingual extension of PAWS
([Zhang et al., 2019](https://aclanthology.org/N19-1131/)), a binary
paraphrase-identification benchmark whose hardness comes from
adversarial sentence pairs that share most of their bag-of-words but
differ in word order or syntactic structure. Yang et al. produced
**human translations** of 23,659 evaluation pairs into six
typologically diverse languages (de, es, fr, ja, ko, zh) — the
training pairs are machine-translated, but the evaluation set is
fully human-translated. Apertus uses paws_en + 3 PAWS-X test subsets
(es, ja, zh) — and one **independently-created** Basque subset:
`paws_eu` is [HiTZ/PAWS-eu](https://huggingface.co/datasets/HiTZ/PAWS-eu),
a 2,000-pair professional Basque translation of the original English
PAWS, commissioned by HiTZ (UPV/EHU) within the ILENIA project and
released with [IberoBench (Baucells et al., 2025)](https://aclanthology.org/2025.coling-main.699/).

### xcopa

**XCOPA** ([Ponti et al., 2020](https://aclanthology.org/2020.emnlp-main.185/))
is the cross-lingual Choice-of-Plausible-Alternatives benchmark for
causal commonsense reasoning. The validation and test sets of English
COPA ([Roemmele et al., 2011](https://people.ict.usc.edu/~gordon/copa.html))
were carefully translated and re-annotated by trained native speakers
into 11 typologically diverse languages: Estonian, Haitian Creole,
Indonesian, Italian, Cusco-Collao Quechua, Kiswahili, Tamil, Thai,
Turkish, Vietnamese, Mandarin Chinese — **Basque is not in this
list**. Apertus uses 5 of those (sw, th, tr, vi, zh) plus an
**independently-created** Basque subset: `xcopa_eu` is
[HiTZ/XCOPA-eu](https://huggingface.co/datasets/HiTZ/XCOPA-eu), a
professional Basque translation of the *original English COPA*
(Roemmele et al., 2011) commissioned by HiTZ (UPV/EHU) within ILENIA
and released with
[IberoBench (Baucells et al., 2025)](https://aclanthology.org/2025.coling-main.699/).

### xnli

**XNLI** ([Conneau et al., 2018](https://aclanthology.org/D18-1269/)) is
the cross-lingual extension of MultiNLI
([Williams et al., 2018](https://aclanthology.org/N18-1101/)). The
authors took 7,500 MultiNLI dev/test pairs and had **professional
translators** render premise + hypothesis into 14 additional languages
(fr, es, de, el, bg, ru, tr, ar, vi, th, zh, hi, sw, ur), preserving
the English entailment label. Apertus uses 10 of the original XNLI
languages (en, ar, es, hi, ru, sw, th, tr, vi, zh) plus
**independently-created** `xnli_eu`:
[XNLIeu (Heredia et al., 2024, NAACL)](https://aclanthology.org/2024.naacl-long.234/),
a Basque XNLI built by machine-translating the English XNLI test set
into Basque and then **manually post-editing** every example. The
harness ships three XNLIeu variants (`xnli_eu` post-edited from MT,
`xnli_eu_mt` raw MT, `xnli_eu_native` natively authored); Apertus
uses the post-edited default.

### xstorycloze

**XStoryCloze** ([Lin et al., 2022](https://aclanthology.org/2022.emnlp-main.616/))
is the multilingual companion to the XGLM paper. Lin et al. took the
Spring-2016 validation split of the English Story Cloze Test
([Mostafazadeh et al., 2016](https://aclanthology.org/N16-1098/)) and
commissioned **professional human translations** into 10 additional
languages: ru, zh-Hans, es-LatAm, ar, hi, id, te, sw, eu, my — Basque
is part of Meta's original release. Each item is a four-sentence
story plus two candidate endings (one plausible, one not). Apertus
uses 8 of the 11 released subsets (ar, en, es, eu, hi, ru, sw, zh);
`xstorycloze_eu` is therefore not an independent dataset.

### xwinograd

**XWinograd** ([Tikhonov & Ryabinin, 2021](https://aclanthology.org/2021.findings-acl.310/)),
later expanded by [Muennighoff et al., 2022](https://arxiv.org/abs/2211.01786),
is a multilingual Winograd Schema Challenge corpus. Rather than
translating English Winograd schemas, the authors **aggregated
existing native or pre-existing-translation Winograd schema resources
in five non-English languages and standardised the annotation
pipeline**: English (combining the original Winograd Schema Challenge
items, the SuperGLUE WSC items, and the Definite Pronoun Resolution
dataset), French, Japanese, Portuguese, Russian, and Chinese
(extended in the harness with 488 additional schemas from
clue/cluewsc2020). Items are pronoun-coreference disambiguation
problems judged by log-probability comparison between the two
candidate completions. Apertus uses 4 of those subsets (en, jp, ru,
zh).

---

## Schema-encoded table

Definitions of the columns:
- `data_source`: where the underlying QA / NLI / etc. items came from.
- `curation_process`: high-level method by which the per-language
  evaluation items were produced (machine translation, professional
  human translation, originally-multilingual / participatory,
  template-generated, …).
- `task_format`: `mc` (multiple-choice with N candidate completions
  scored by log-probability) or `gen` (free-form generation). All
  twelve families here are scored as `mc`/completion-style.
- `domain`: high-level subject area being probed.
- `n_languages`: number of per-language aggregate tasks present in the
  Apertus parquet for this family.

| family | data_source | curation_process | task_format | domain | n_languages |
|---|---|---|---|---|---|
| arc | English ARC science questions (Clark et al. 2018) translated to 9 langs by Okapi (Lai et al. 2023) — Basque included | machine translation by ChatGPT (uniform across all 9 non-English subsets, including eu) | mc | grade-school science / general knowledge | 11 |
| belebele | FLORES-200 passages with 4-option MRC questions (Bandarkar et al. 2024, Meta) | human authoring + human translation by bilingual experts (no MT) | mc | reading comprehension | 12 |
| global_mmlu | English MMLU (Hendrycks et al. 2021) Lite-eligible items in Global-MMLU (Singh et al. 2024, Cohere Labs) | professional human translation + human post-editing (Lite-style: items rated fully human-translated/post-edited) | mc | general knowledge / STEM (57 subjects, 5 super-categories) | 6 |
| global_mmlu_full | English MMLU translated into 42 langs in Global-MMLU (Singh et al. 2024, Cohere Labs) | machine translation + crowd / expert post-editing (mixed quality across items) | mc | general knowledge / STEM (57 subjects) | 10 |
| global_piqa_completions | Originally-multilingual physical-commonsense items hand-written by native-speaker researchers (Arnett et al. 2025, MRL shared task) | participatory native-speaker authoring (no translation) | mc | physical commonsense reasoning | 11 |
| hellaswag | English HellaSwag (Zellers et al. 2019) translated to 8 langs by Okapi (Lai et al. 2023) — Basque included | machine translation by ChatGPT (uniform across all 8 non-English subsets, including eu) | mc | commonsense sentence completion | 9 |
| multiblimp | Universal Dependencies + UniMorph subject-verb agreement minimal pairs across 101 langs (Jumelet et al. 2025) — Basque included | template-based automatic generation from UD/UniMorph (uniform across all 7 subsets) | mc | grammaticality / syntactic minimal pairs | 7 |
| paws | English PAWS adversarial paraphrase pairs (Zhang et al. 2019) — non-English from PAWS-X (Yang et al. 2019) for es/ja/zh + HiTZ/PAWS-eu (IberoBench, Baucells et al. 2025) for eu | professional human translation across all subsets (PAWS-X for es/ja/zh; HiTZ ILENIA-funded translation for eu — separate effort) | mc | paraphrase identification / NLI | 5 |
| xcopa | English COPA (Roemmele et al. 2011) — non-English from XCOPA (Ponti et al. 2020) for sw/th/tr/vi/zh + HiTZ/XCOPA-eu (IberoBench, Baucells et al. 2025) for eu | professional human translation + native-speaker re-annotation (XCOPA design) for sw/th/tr/vi/zh; professional human translation of original English COPA (no XCOPA re-annotation step) for eu — separate effort | mc | causal commonsense reasoning | 6 |
| xnli | MultiNLI (Williams et al. 2018) — non-English from XNLI (Conneau et al. 2018) for ar/es/hi/ru/sw/th/tr/vi/zh + XNLIeu (Heredia et al. 2024) for eu | professional human translation for the 9 XNLI subsets; machine translation + manual post-editing for eu — separate paper | mc | natural language inference | 11 |
| xstorycloze | English Story Cloze Test (Mostafazadeh et al. 2016) Spring-2016 validation, professionally translated to 10 langs by Lin et al. 2022 (Meta) — Basque included | professional human translation (uniform across all 8 subsets, including eu) | mc | narrative commonsense (story completion) | 8 |
| xwinograd | Aggregated native or pre-existing Winograd-schema resources in 6 langs, standardised by Tikhonov & Ryabinin 2021 (extended in harness with extra Chinese schemas from clue/cluewsc2020) | originally-multilingual aggregation of native-language schemas (no translation step) | mc | pronoun coreference / commonsense | 4 |

Notes:
- **Heterogeneous-curation families.** In `paws`, `xcopa`, and `xnli`,
  the Basque subset is curated differently from the rest of the
  family and is sourced from a separate paper. Concretely: `paws_eu`
  and `xcopa_eu` come from HiTZ via IberoBench (Baucells et al.
  2025; ILENIA project) — both professional human translations of the
  *original* English source (PAWS, COPA), which is a milder process
  than the adversarial-pair authoring of PAWS-X or the
  re-annotation step of XCOPA. `xnli_eu` is XNLIeu (Heredia et al.
  2024, NAACL), MT + manual post-editing — strictly different from
  the professional-translation method used in the rest of XNLI.
  When aggregating SNR per family, this means a single
  `curation_process` cell hides within-family variation; consider
  carrying a per-task curation column as a follow-up.
- **`arc_eu`, `hellaswag_eu`, `multiblimp_eus`, `xstorycloze_eu`**
  are all part of the *same* primary release as their family
  (Okapi-30 / Okapi-30 / MultiBLiMP-101 / XStoryCloze-11
  respectively). Treat their `curation_process` as identical to the
  rest of the family.
- `n_languages` counts per-language *aggregate* tasks, not the
  per-(language, subject) facets that exist for `global_mmlu` and
  `global_mmlu_full`. These per-subject keys are filtered out by
  `multilingual.smooth_subtasks.collect_multilingual_families`.
- `arc` n_languages = 11 because the family includes the English
  `arc_challenge` and `arc_easy` splits (collapsed via
  `_BENCHMARK_FAMILY_OVERRIDES`) plus 9 non-English translations,
  per `multilingual.analyze_snr_variants.benchmark_family`.
- All 12 families here are scored as multiple-choice
  log-probability tasks (the `_completions` variant of Global-PIQA,
  the cloze-style XStoryCloze / XCOPA / XWinograd, the binary-choice
  PAWS, the entailment-class XNLI, the next-sentence HellaSwag, the
  per-option ARC / MMLU / Belebele, and the minimal-pair MultiBLiMP).

When this table is filled in (it now is), run the analysis described
in [INSTRUCTIONS.md](INSTRUCTIONS.md) Step 2 onward.
