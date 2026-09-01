# Benchmark selection for the predictivity sweep

Status 2026-08-21. Companion to the team decisions in
[team-discussion-2026-08-21.md](team-discussion-2026-08-21.md) (point 4):
**evaluate every cell on all the benchmarks the lm-eval harness offers for the
languages it trains on**, and **choose the L100 languages so that every trained
language has at least one benchmark** (fill-ins need two). Per-language BPB on
the fixed validation set remains the primary metric; benchmarks are the
secondary, SNR-style signal.

How this is wired (all reproducible from scripts):

- `scripts/wire_harness_tasks.py` scans the installed harness (`lm_eval/tasks`,
  the `task:` field of every yaml), maps each per-language task of the families
  below onto the FineWeb-2 subset it evaluates, and adds it to
  `configs/tasks.json` under the project's language tag (`configs/languages.json`
  `fineweb_iso2`). It also writes the `benchmarks` section (paper, venue,
  languages, format per family) and keeps every pretraining-stage family in the
  `auto` group the watchers evaluate during training.
- `src/pretrain/data/generate_language_sets.py` builds both language-set JSONs
  from the FineWeb-2 distribution and tasks.json (benchmark availability).
- The auto set per cell is `auto` families ∩ trained languages
  (`tasks_for_benchmarks` × `cell_languages`): L1 15 · L2 25 · L8 90 · L15 153 ·
  L30 238 · L50 334 · **L100 463** tasks. Eval walltime at L ≥ 50 for the 1B/1.7B
  rungs must be measured on the first reference checkpoint and the jobs split if
  needed (see the discussion guide, 4B) — the user will revisit the per-language
  lists after this investigation.

## The benchmark families

Provenance and construction matter for a predictivity study: a benchmark that is
machine-translated, auto-generated, or near chance for sub-1B models contributes
noise rather than signal. Venues checked 2026-08-21.

| Family (tasks.json `benchmark`) | Paper · venue | Languages | Format / size | Construction | Notes for us |
|---|---|---|---|---|---|
| `belebele` | [Bandarkar et al., ACL 2024](https://aclanthology.org/2024.acl-long.44/) | 122 | 4-way MC reading comprehension, 900 items/lang, parallel | Human-translated (FLORES passages), QA'd | The backbone: widest *human* coverage; identical items across languages → clean cross-lingual comparison. 87 variants wired. |
| `global_piqa_completions` | [Chang et al., arXiv 2510.24081](https://arxiv.org/abs/2510.24081) — MRL shared task @ EMNLP 2025 | 116 varieties | 2-way physical-commonsense completion | Hand-written per language by 335 native-speaker researchers; parallel + culturally specific splits | Completion format suits small base models; the only benchmark for bs, nn, fo. Not (yet) a main-conference paper; small per-variety sets → noisy per language. |
| `global_mmlu_full` | [Singh et al., ACL 2025](https://aclanthology.org/2025.acl-long.919/) | 42 | MC knowledge QA (14k MMLU items) | Professional + community translation, culturally-sensitive tagging | Knowledge-heavy: expect near-chance at 90M–600M; mainly a 1B/1.7B signal. 37 wired. |
| `include_base_44` | [Romanou et al., ICLR 2025](https://openreview.net/forum?id=k3gCieTXeY) | 44 | MC regional exam questions (~500/lang in base_44) | Natively sourced (not translated) | High quality, EPFL-authored; knowledge-heavy like MMLU. All 44 wired (tags fixed 2026-08-21). |
| `multiblimp` | [Jumelet et al., TACL 2026](https://aclanthology.org/2026.tacl-1.10/) | 101 | Grammatical minimal pairs (subject–verb agreement) | Auto-generated from Universal Dependencies + UniMorph | A grammatical-competence *probe*, not a task: gives signal very early in training, but shallow; count it as a weak family. |
| `hellaswag`, `arc` (Okapi m_hellaswag / m_arc) | [Lai et al., EMNLP 2023 demo](https://aclanthology.org/2023.emnlp-demo.28/) | 31 | 4-way completion / MC science QA | **Machine-translated (ChatGPT)** from the English originals | Widely used and good early signal (hellaswag), but translation artifacts; treat per-language absolute scores with care. |
| `xnli` | [Conneau et al., EMNLP 2018](https://aclanthology.org/D18-1269/) | 15 | 3-way NLI | Professional translation | Classic; NLI is hard for small base models (near chance early). |
| `xstorycloze` | [Lin et al., EMNLP 2022 (XGLM)](https://aclanthology.org/2022.emnlp-main.616/) | 11 | 2-way story ending | Professional translation | Good small-model signal; the only second family for Burmese. |
| `xcopa` | [Ponti et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.185/) | 11 | 2-way causal commonsense, 500 items | Professional translation | Small; covers ht, et, ta, th, id, it, sw, tr, vi, zh. |
| `xwinograd` | [Tikhonov & Ryabinin, Findings ACL 2021](https://aclanthology.org/2021.findings-acl.310/) | 6 | Winograd coreference | Human | Small; en, fr, jp, pt, ru, zh. |
| `paws` (PAWS-X) | [Yang et al., EMNLP 2019](https://aclanthology.org/D19-1382/) | 7 | Paraphrase identification | Professional translation | Near chance for small base models. |
| `lambada_openai_mt` | [EleutherAI task](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/lambada_multilingual) (LAMBADA: Paperno et al., ACL 2016) | 5 | Last-word prediction | **Machine-translated, no paper** | Weakest provenance in the set; kept because completion-style tasks track pretraining well. |
| `afrixnli`, `afrimmlu` (IrokoBench) | [Adelani et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.139/) | 16 African + en/fr | 3-way NLI / 5-subject MC QA | **Human-translated by native speakers** (Masakhane, Lacuna Fund) | See below. Wired as the `_prompt_1` variant (the harness ships 5 prompt templates). |
| `afrimgsm` (IrokoBench) | same | 16 + en/fr | Grade-school math generation (MGSM) | Human-translated | Generation/maths → midtraining stage, like `mgsm_direct`. |
| `mgsm_direct` | [Shi et al., ICLR 2023](https://openreview.net/forum?id=fR3wGCk-IXp) | 10 | Grade-school math generation | Human-translated | Midtraining stage. |
| `truthfulqa-multi_mc1` | [Calvo Figueras et al., arXiv 2502.09387](https://arxiv.org/abs/2502.09387) | en, es, eu, ca, gl | MC truthfulness | Professional translation | Small-model signal is weak; kept for the Iberian languages. |

## IrokoBench (AfriXNLI / AfriMMLU / AfriMGSM) — the new afri* families

- **Venue and quality.** Published at NAACL 2025 (main conference) after the
  June-2024 arXiv release; produced by the Masakhane community with Lacuna Fund
  support, with professional/native-speaker human translation of three existing
  benchmarks (XNLI, a 5-subject MMLU subset, MGSM) into 16 typologically diverse
  African languages, plus English and French as references. It is the standard
  African-language evaluation suite in 2025–26 model reports (used for Aya,
  Gemma, Llama-family evaluations). Quality is the human-translated kind we
  prefer; the limitation is size — a few hundred items per language per task — so
  per-language scores are noisier than Belebele's 900.
- **Languages.** amh, ewe, fra, hau, ibo, kin, lin, lug, orm, sna, sot, swa, twi,
  wol, xho, yor, zul (+ eng). Of these, our lists now train amh, swh, kin, xho,
  zul, ibo, sot (hau is absent from the swiss-ai data dir; yor, sna, lin, lug,
  gaz (orm) sit below rank 99).
- **How we use it.** `afrixnli_<lang>_prompt_1` and `afrimmlu_direct_<lang>_prompt_1`
  in the pretraining-stage auto set (one prompt template; the 5-template groups
  would multiply cost ×5 for a prompt-robustness measurement we are not making);
  `afrimgsm_<lang>` in the midtraining stage with MGSM.
- **Why it matters for L100.** The swap-in languages kin, xho, zul, ibo, sot get
  their second/third human-made family from IrokoBench; without it they would
  rest on Belebele alone.

## Coverage of the trained languages by region

Families in the during-training auto set per language (after the 2026-08-21
L100 swap). Arabic dialect subsets (ary, arz, ars, apc) share the `ar` tag, so
they inherit the MSA tasks and the dialect Belebele/Global-PIQA variants run for
every Arabic-training cell — a folding to keep in mind when reading per-dialect
numbers.

### South Asian

| Language | Benchmark families (auto set) |
|---|---|
| hin_Deva Hindi | arc, belebele, global_mmlu, global_piqa, hellaswag, include, multiblimp, xnli, xstorycloze (9) |
| ben_Beng Bengali | arc, belebele, global_mmlu, global_piqa, hellaswag, include, multiblimp (7) |
| urd_Arab Urdu | belebele, global_piqa, include, multiblimp, xnli (5) |
| mar_Deva Marathi | arc, belebele, global_piqa, hellaswag, multiblimp (5) |
| npi_Deva Nepali | arc, belebele, global_mmlu, global_piqa, hellaswag, include (6) |
| tam_Taml Tamil | arc, belebele, global_piqa, hellaswag, include, multiblimp, xcopa (7) |
| tel_Telu Telugu | arc, belebele, global_mmlu, global_piqa, hellaswag, include, xstorycloze (7) |
| kan_Knda Kannada | arc, belebele, global_piqa, hellaswag (4) |
| mal_Mlym Malayalam | arc, belebele, global_piqa, hellaswag, include (5) |
| guj_Gujr Gujarati | arc, belebele, global_piqa, hellaswag, multiblimp (5) |
| pan_Guru Panjabi | belebele, global_piqa (2) |
| sin_Sinh Sinhala | belebele, global_mmlu, global_piqa (3) |
| asm_Beng Assamese | belebele, global_piqa (2) |
| ory_Orya Odia | belebele (1) |

### African

| Language | Benchmark families (auto set) |
|---|---|
| swh_Latn Swahili | belebele, global_mmlu, global_piqa, xcopa, xnli, xstorycloze (6) |
| amh_Ethi Amharic | afrimmlu, afrixnli, belebele, global_mmlu, global_piqa, multiblimp (6) |
| som_Latn Somali | belebele, global_mmlu (2) |
| plt_Latn Plateau Malagasy | belebele, global_mmlu (2) |
| kin_Latn Kinyarwanda | afrimmlu, afrixnli, belebele, global_piqa (4) |
| xho_Latn Xhosa | afrimmlu, afrixnli, belebele (3) |
| zul_Latn Zulu | afrimmlu, afrixnli, belebele, global_piqa (4) |
| ibo_Latn Igbo | afrimmlu, afrixnli, belebele, global_piqa (4) |
| sot_Latn Southern Sotho | afrimmlu, afrixnli, belebele (3) |
| ary_Arab / arz_Arab (fold into ar) | arc, belebele, global_mmlu, global_piqa, hellaswag, include, multiblimp, xnli, xstorycloze (9) |

### Central Asian / Turkic / Caucasus

| Language | Benchmark families (auto set) |
|---|---|
| kaz_Cyrl Kazakh | belebele, global_piqa, include, multiblimp (4) |
| kir_Cyrl Kyrgyz | belebele, global_mmlu, global_piqa, multiblimp (4) |
| uzn_Cyrl / uzn_Latn Uzbek | belebele, global_piqa, include (3) |
| azj_Latn Azerbaijani | belebele, global_piqa, include (3) |
| tgk_Cyrl Tajik | belebele (1) |
| khk_Cyrl Mongolian | belebele (1) |
| uig_Arab Uyghur | global_piqa, multiblimp (2) |
| kat_Geor Georgian | belebele, global_piqa, include, multiblimp (4) |
| hye_Armn Armenian | arc, belebele, global_piqa, hellaswag, include, multiblimp (6) |
| pbt_Arab Pashto | belebele (1) |
| ckb_Arab Central Kurdish | belebele, global_piqa (2) |
| kmr_Latn Northern Kurdish | multiblimp (1) |
| snd_Arab Sindhi | belebele, global_piqa (2) |

### South-East Asian

| Language | Benchmark families (auto set) |
|---|---|
| ind_Latn Indonesian | arc, belebele, global_mmlu, global_piqa, hellaswag, include, xcopa, xstorycloze (8) |
| vie_Latn Vietnamese | arc, belebele, global_mmlu, global_piqa, hellaswag, include, xcopa, xnli (8) |
| tha_Thai Thai | belebele, global_piqa, xcopa, xnli (4) |
| zsm_Latn Malay | belebele, global_mmlu, global_piqa, include (4) |
| fil_Latn Filipino | belebele, global_piqa, include (3) |
| mya_Mymr Burmese | belebele, xstorycloze (2) |
| khm_Khmr Khmer | belebele (1) |
| lao_Laoo Lao | belebele (1) |
| jav_Latn Javanese | belebele, global_piqa (2) |
| bod_Tibt Tibetan | belebele (1) |

### Other one-family European languages

lvs (belebele), afr (belebele), nno (global_piqa), cym (multiblimp), gle
(multiblimp), mlt (belebele), lat (multiblimp), bos (global_piqa); fao and hat
have two (global_piqa + multiblimp; belebele + xcopa).

## Candidates for underserved languages (not in our harness fork yet)

Verified to exist; each needs porting into the swiss-ai harness fork (or pulling
the upstream task) before `wire_harness_tasks.py` can pick it up.

| Benchmark | Paper · venue | Languages | Why | Caveat |
|---|---|---|---|---|
| **MILU** | [Verma et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.507/) · [AI4Bharat](https://github.com/AI4Bharat/MILU) | 11 Indic: bn, gu, hi, kn, ml, mr, **or, pa**, ta, te + en | India-centric MC knowledge (8 domains, 41 subjects, regional exams); natively sourced. Ported to upstream lm-eval-harness ([PR #2482](https://github.com/EleutherAI/lm-evaluation-harness/pull/2482)). Adds a second family to Odia and Panjabi, a third to Kannada. | Knowledge-heavy → small-model signal mostly at 1B+. |
| IndicMMLU-Pro | [arXiv 2501.15747](https://arxiv.org/abs/2501.15747) (GEM workshop 2026) | hi, bn, gu, mr, kn, pa, ta, te, ur | MMLU-Pro in 9 Indic languages | Machine-translated with IndicTrans2 (back-translation QA) — MT caveat as for Okapi. |
| TUMLU | Isbarov et al., 2025 (Turkic MMLU) | az, crh, kaa, kk, tt, tr, ug, uz | Native middle/high-school exam questions for Turkic languages | Would give Uyghur and Kazakh a native knowledge benchmark; verify harness port. |
| SeaExam / SEA-HELM | AI Singapore / DAMO (2024–25) | id, vi, th, ms, fil, … | Native SEA exam questions | Strengthens id/vi/th/ms/fil only; khm, lao, bod stay Belebele-only. |

No native second family exists anywhere in the harness for khm, lao, bod, tgk,
pbt, ory, lvs, afr, mlt — Belebele is their floor until one is ported.

Recommendation: port **MILU** first (NAACL 2025, natively sourced, already in
upstream harness form; closes the Odia/Panjabi single-family gap), then TUMLU
for Uyghur/Kazakh/Uzbek; skip MT-based suites unless a language has nothing else.

## Caveats to carry into the analysis

- **Chance level differs by family** (2-way: global_piqa, xstorycloze, xcopa;
  3-way: xnli, afrixnli; 4-way: belebele, hellaswag, arc, afrimmlu, global_mmlu,
  include). Normalise before aggregating, and expect the knowledge families
  (global_mmlu, include, afrimmlu) to sit at chance for the small rungs — the
  SNR machinery will flag them as low-signal early in training, which is itself
  a result.
- **Translation provenance**: Okapi (hellaswag/arc) and LAMBADA-MT are machine
  translations; everything else wired is human-made or professionally
  translated; MultiBLiMP is auto-generated from treebanks.
- **Folded tags**: Arabic dialects → `ar`; romanised subsets (hin_Latn,
  urd_Latn, …) are excluded from the language lists precisely because they would
  inherit the native-script benchmarks.
- **Prompt variants**: IrokoBench ships 5 prompt templates; we run template 1.
