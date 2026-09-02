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

## Coverage of the trained languages

What every trained language is actually evaluated on during training (the auto
group intersected with the cell's languages), and the setting it enters at.
Generated — run `python scripts/wire_harness_tasks.py --report` after any change
to configs/tasks.json or the language sets. Arabic dialect subsets (ary, arz,
ars, apc) share the `ar` tag, so they inherit the MSA tasks and their dialect
Belebele / Global-PIQA variants run for every Arabic-training cell.

<!-- BEGIN generated: scripts/wire_harness_tasks.py --report -->

Tasks per cell (auto group x trained languages): L1 15 · L2 25 · L8 90 · L15 153 · L30 238 · L50 334 · L100 463.

| Enters at | Language | Families | Tasks | Benchmark families |
|---|---|---|---:|---|
| L1 | `en` English | 14 | 15 | afrimmlu, afrixnli, arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, lambada_openai_mt, multiblimp, paws, truthfulqa-multi_mc1, xnli, xstorycloze, xwinograd |
| L2 | `ru` Russian | 10 | 10 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp, xnli, xstorycloze, xwinograd |
| L8 | `fr` French | 13 | 14 | afrimmlu, afrixnli, arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, xnli, xwinograd |
| L8 | `es` Spanish | 12 | 14 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, truthfulqa-multi_mc1, xnli, xstorycloze |
| L8 | `de` German | 10 | 10 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, xnli |
| L8 | `zh` Mandarin Chinese | 10 | 12 | arc, belebele, global_mmlu_full, global_piqa_completions, include_base_44, paws, xcopa, xnli, xstorycloze, xwinograd |
| L8 | `it` Italian | 9 | 9 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, lambada_openai_mt, multiblimp, xcopa |
| L8 | `ja` Japanese | 6 | 6 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, paws, xwinograd |
| L15 | `ar` Levantine Arabic | 9 | 21 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp, xnli, xstorycloze |
| L15 | `id` Indonesian | 8 | 8 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, xcopa, xstorycloze |
| L15 | `pt` Portuguese | 8 | 9 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp, xwinograd |
| L15 | `vi` Vietnamese | 8 | 8 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, xcopa, xnli |
| L15 | `nl` Dutch | 7 | 7 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp |
| L15 | `fa` Persian | 5 | 5 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp |
| L15 | `pl` Polish | 5 | 5 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp |
| L30 | `hi` Hindi | 9 | 10 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp, xnli, xstorycloze |
| L30 | `bn` Bengali | 7 | 9 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp |
| L30 | `tr` Turkish | 7 | 7 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp, xcopa, xnli |
| L30 | `uk` Ukrainian | 7 | 7 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, multiblimp |
| L30 | `el` Modern Greek (1453-) | 6 | 6 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp, xnli |
| L30 | `hu` Hungarian | 6 | 6 | arc, belebele, global_piqa_completions, hellaswag, include_base_44, multiblimp |
| L30 | `ro` Romanian | 6 | 6 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, multiblimp |
| L30 | `sv` Swedish | 6 | 6 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, multiblimp |
| L30 | `bg` Bulgarian | 5 | 5 | belebele, global_piqa_completions, include_base_44, multiblimp, xnli |
| L30 | `ko` Korean | 5 | 5 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, paws |
| L30 | `cs` Czech | 4 | 4 | belebele, global_mmlu_full, global_piqa_completions, multiblimp |
| L30 | `da` Danish | 4 | 4 | arc, belebele, hellaswag, multiblimp |
| L30 | `fi` Finnish | 4 | 4 | belebele, global_piqa_completions, include_base_44, multiblimp |
| L30 | `th` Thai | 4 | 4 | belebele, global_piqa_completions, xcopa, xnli |
| L30 | `no` Norwegian Bokmål | 2 | 2 | belebele, global_piqa_completions |
| L50 | `ca` Catalan | 8 | 8 | arc, belebele, global_piqa_completions, hellaswag, multiblimp, paws, xnli, xstorycloze |
| L50 | `ta` Tamil | 7 | 7 | arc, belebele, global_piqa_completions, hellaswag, include_base_44, multiblimp, xcopa |
| L50 | `ne` Nepali (individual language) | 6 | 7 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44 |
| L50 | `sr` Serbian | 6 | 7 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44 |
| L50 | `et` Standard Estonian | 5 | 5 | belebele, global_piqa_completions, include_base_44, multiblimp, xcopa |
| L50 | `he` Hebrew | 5 | 5 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp |
| L50 | `hr` Croatian | 5 | 5 | arc, belebele, global_piqa_completions, hellaswag, include_base_44 |
| L50 | `lt` Lithuanian | 5 | 5 | belebele, global_mmlu_full, global_piqa_completions, include_base_44, multiblimp |
| L50 | `ml` Malayalam | 5 | 5 | arc, belebele, global_piqa_completions, hellaswag, include_base_44 |
| L50 | `mr` Marathi | 5 | 5 | arc, belebele, global_piqa_completions, hellaswag, multiblimp |
| L50 | `sk` Slovak | 5 | 6 | arc, belebele, global_piqa_completions, hellaswag, multiblimp |
| L50 | `ur` Urdu | 5 | 7 | belebele, global_piqa_completions, include_base_44, multiblimp, xnli |
| L50 | `ka` Georgian | 4 | 4 | belebele, global_piqa_completions, include_base_44, multiblimp |
| L50 | `kk` Kazakh | 4 | 4 | belebele, global_piqa_completions, include_base_44, multiblimp |
| L50 | `ms` Standard Malay | 4 | 4 | belebele, global_mmlu_full, global_piqa_completions, include_base_44 |
| L50 | `az` North Azerbaijani | 3 | 3 | belebele, global_piqa_completions, include_base_44 |
| L50 | `sl` Slovenian | 3 | 4 | belebele, global_piqa_completions, multiblimp |
| L50 | `sq` Tosk Albanian | 3 | 3 | belebele, global_piqa_completions, include_base_44 |
| L50 | `bs` Bosnian | 1 | 1 | global_piqa_completions |
| L50 | `lv` Standard Latvian | 1 | 1 | belebele |
| L100 | `eu` Basque | 10 | 10 | arc, belebele, hellaswag, include_base_44, multiblimp, paws, truthfulqa-multi_mc1, xcopa, xnli, xstorycloze |
| L100 | `te` Telugu | 7 | 7 | arc, belebele, global_mmlu_full, global_piqa_completions, hellaswag, include_base_44, xstorycloze |
| L100 | `am` Amharic | 6 | 6 | afrimmlu, afrixnli, belebele, global_mmlu_full, global_piqa_completions, multiblimp |
| L100 | `gl` Galician | 6 | 6 | belebele, global_piqa_completions, multiblimp, paws, xnli, xstorycloze |
| L100 | `hy` Armenian | 6 | 6 | arc, belebele, global_piqa_completions, hellaswag, include_base_44, multiblimp |
| L100 | `sw` Swahili (individual language) | 6 | 6 | belebele, global_mmlu_full, global_piqa_completions, xcopa, xnli, xstorycloze |
| L100 | `gu` Gujarati | 5 | 5 | arc, belebele, global_piqa_completions, hellaswag, multiblimp |
| L100 | `ig` Igbo | 4 | 4 | afrimmlu, afrixnli, belebele, global_piqa_completions |
| L100 | `kn` Kannada | 4 | 4 | arc, belebele, global_piqa_completions, hellaswag |
| L100 | `ky` Kirghiz | 4 | 4 | belebele, global_mmlu_full, global_piqa_completions, multiblimp |
| L100 | `mk` Macedonian | 4 | 4 | belebele, global_piqa_completions, include_base_44, multiblimp |
| L100 | `rw` Kinyarwanda | 4 | 4 | afrimmlu, afrixnli, belebele, global_piqa_completions |
| L100 | `zu` Zulu | 4 | 4 | afrimmlu, afrixnli, belebele, global_piqa_completions |
| L100 | `be` Belarusian | 3 | 3 | global_piqa_completions, include_base_44, multiblimp |
| L100 | `is` Icelandic | 3 | 3 | belebele, global_piqa_completions, multiblimp |
| L100 | `si` Sinhala | 3 | 4 | belebele, global_mmlu_full, global_piqa_completions |
| L100 | `st` Southern Sotho | 3 | 3 | afrimmlu, afrixnli, belebele |
| L100 | `tl` Filipino | 3 | 3 | belebele, global_piqa_completions, include_base_44 |
| L100 | `uz` Northern Uzbek | 3 | 3 | belebele, global_piqa_completions, include_base_44 |
| L100 | `xh` Xhosa | 3 | 3 | afrimmlu, afrixnli, belebele |
| L100 | `as` Assamese | 2 | 2 | belebele, global_piqa_completions |
| L100 | `ckb` Central Kurdish | 2 | 2 | belebele, global_piqa_completions |
| L100 | `fo` Faroese | 2 | 2 | global_piqa_completions, multiblimp |
| L100 | `ht` Haitian | 2 | 2 | belebele, xcopa |
| L100 | `jv` Javanese | 2 | 2 | belebele, global_piqa_completions |
| L100 | `mg` Plateau Malagasy | 2 | 2 | belebele, global_mmlu_full |
| L100 | `my` Burmese | 2 | 2 | belebele, xstorycloze |
| L100 | `pa` Panjabi | 2 | 2 | belebele, global_piqa_completions |
| L100 | `sd` Sindhi | 2 | 3 | belebele, global_piqa_completions |
| L100 | `so` Somali | 2 | 2 | belebele, global_mmlu_full |
| L100 | `ug` Uighur | 2 | 2 | global_piqa_completions, multiblimp |
| L100 | `af` Afrikaans | 1 | 1 | belebele |
| L100 | `bo` Tibetan | 1 | 1 | belebele |
| L100 | `cy` Welsh | 1 | 1 | multiblimp |
| L100 | `ga` Irish | 1 | 1 | multiblimp |
| L100 | `km` Khmer | 1 | 1 | belebele |
| L100 | `kmr` Northern Kurdish | 1 | 1 | multiblimp |
| L100 | `la` Latin | 1 | 1 | multiblimp |
| L100 | `lo` Lao | 1 | 1 | belebele |
| L100 | `mn` Halh Mongolian | 1 | 1 | belebele |
| L100 | `mt` Maltese | 1 | 1 | belebele |
| L100 | `nn` Norwegian Nynorsk | 1 | 1 | global_piqa_completions |
| L100 | `or` Odia | 1 | 1 | belebele |
| L100 | `ps` Southern Pashto | 1 | 1 | belebele |
| L100 | `tg` Tajik | 1 | 1 | belebele |

Languages by number of families: 1→16 · 2→12 · 3→10 · 4→13 · 5→13 · 6→11 · 7→6 · 8→4 · 9→3 · 10→4 · 12→1 · 13→1 · 14→1.

<!-- END generated -->

## What we are NOT evaluating on, and why (reviewed 2026-09-01)

"Every benchmark available" currently means *every parallel, multi-language
family the harness ships*. Three classes sit outside that, in increasing order
of work.

### 1. In the upstream harness, deliberately unwired

| Family | Coverage | Why it is out |
|---|---|---|
| `okapi/mmlu_multilingual` (m_mmlu) | 34, MT | Duplicates `global_mmlu_full` (human) wherever both exist; machine-translated knowledge QA is the weakest signal at our sizes. Adds a knowledge family to ~10 languages that have none — revisit only if the knowledge axis turns out to matter. |
| `okapi/truthfulqa_multilingual` | 31, MT | Truthfulness is a post-training property; near-chance and non-monotone during pretraining. |
| [`mmlu_prox`](https://aclanthology.org/2025.emnlp-main.79/) (EMNLP 2025) | 31 incl. af, wo, yo, zu, sw, ne, mr, sr, te, ur | MMLU-Pro difficulty (10-way, reasoning-first): at chance for every rung below ~1B, so it would contribute noise, not signal. Would give Afrikaans a second family. |
| `indicxnli` | gu | One task; adds NLI to Gujarati. Cheap — wire it if NLI is kept in the auto set. |
| `lambada_multilingual_stablelm` | 5 | **Upstream says prefer this over the legacy `lambada_multilingual` we wire.** Switch, or drop LAMBADA-MT entirely (weakest provenance in our set). |
| Language-specific native suites: `noreval` (no), `icelandic_winogrande` (is), `greekmmlu` (el), `basque_bench`/`eus_*` (eu), `catalan_bench` (ca), `galician_bench` (gl), `french_bench`, `evalita_LLM` (it), `arabicmmlu`, `kmmlu`/`kobest`/`click` (ko), `cmmlu`/`ceval` (zh), `bangla_*` (bn), `copal_id` (id), `blimp_nl` (nl), `mlqa` (7) | 1 language each | **Not comparable across languages** — different items, formats and difficulty, so they cannot enter a macro-average or a cross-language SNR comparison. They are the right instrument for a per-language sanity check on final checkpoints, not for the during-training set. `noreval` would fix Norwegian's thin coverage (2 families) if a per-language read is wanted. |

### 2. Published, not in the harness — would need porting

| Benchmark | Paper · venue | Languages | Why it matters here |
|---|---|---|---|
| **INCLUDE v2** | EPFL, in-house (`~/Projects/epfl/include-private`; data at [`epfl-nlp/include-89`](https://huggingface.co/datasets/epfl-nlp/include-89) / [`include-results/include-128`](https://huggingface.co/datasets/include-results/include-128)) | **113 language–country pairs, 89 languages** — v1's 44 plus a large African set (amharic, dagbani, dangme, ekpeye, embu, esan, ewe, fante, fula, ga, hausa, ibibio, idoma, igala, igbo, jju, kinyarwanda, luo, makhuwa, nyanja, obolo, oromo, sena, somali, swahili, tangale, tigrinya, twi, tyap, yoruba, darija) and more South Asian (assamese, dogri, maithili, oriya, punjabi, sindhi, sinhala) | **The best single addition available to us.** 19 of our trained languages gain a family, and it is the one that fixes the worst gaps: **`or` Odia 1 → 2** (its only benchmark today is Belebele), and `as`, `pa`, `sd`, `so` all 2 → 3. Natively sourced exam questions, same construction as v1 — the highest-provenance family in our set. See the caveat below. |
| **SIB-200** | [Adelani et al., EACL 2024](https://aclanthology.org/2024.eacl-long.14/) | **205** | The single biggest coverage win: 7-way topic classification would give a second family to *every* one-family language (af, bo, km, lo, mn, mt, or, ps, tg, cy, ga, la, kmr, lv, …). Caveat: built on FLORES-200, the same source text as Belebele → domain-correlated, not an independent second opinion; short inputs make it measurable at 90M. |
| **Taxi1500** | [Ma et al., NAACL 2025 (short)](https://aclanthology.org/2025.naacl-short.36/) | 1502 | 6-way classification covering literally every language we train. Bible domain — a strong register mismatch with FineWeb-2 web text; treat as a floor-level probe. |
| **MILU** | [Verma et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.507/) · [AI4Bharat](https://github.com/AI4Bharat/MILU) | 11 Indic incl. **or, pa** | Natively sourced MC knowledge; closes the Odia/Panjabi single-family gap. Already in upstream harness form ([PR #2482](https://github.com/EleutherAI/lm-evaluation-harness/pull/2482)). Knowledge-heavy → 1B+ signal. |
| **Uhura** | [Bayes et al., arXiv 2412.00948](https://arxiv.org/abs/2412.00948) | 6 African | Human-translated ARC-Easy + TruthfulQA; ARC-Easy is a format small models can do, unlike most African-language benchmarks. |
| LORAXBENCH | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.881/) | 20 Indonesian | Covers `jav` natively (currently belebele + global_piqa). |
| TUMLU | Isbarov et al., 2025 | az, kk, tt, tr, ug, uz | Native Turkic exams; would give Uyghur a task benchmark (it has only multiblimp). |
| IndicMMLU-Pro | [arXiv 2501.15747](https://arxiv.org/abs/2501.15747) | 9 Indic | Machine-translated (IndicTrans2) — MT caveat as for Okapi. |
| SeaExam / SEA-HELM | AI Singapore / DAMO | id, vi, th, ms, fil | Strengthens languages that are already well covered; khm, lao, bod stay Belebele-only. |

**INCLUDE v2 caveat — the shipped tasks are the wrong format for us.**
`include-private/include-tasks/include_cot_reasoning/` generates
`output_type: generate_until` tasks: a CoT system prompt, `max_gen_toks` 4096,
and an `<answer>A/B/C/D</answer>` extraction filter, scored by exact match.
That measures instruction-following, which 90M–1.7B *base* models do not have —
every cell would score at or near zero, and 113 pairs × 4096 generated tokens
per item is also far outside the eval walltime cap. The underlying data is
plain 4-option MC (`question`, `choices[0..3]`, `answer`, `country`), so the
fix is a loglikelihood variant: the same `_base_og.yaml` with
`output_type: multiple_choice` and `doc_to_choice`, exactly the shape
`include_base_44` already has in the harness. That is a small change to
`generate_include_tasks.py`, and it is a prerequisite — do not wire the CoT
tasks into the `auto` group.

### 3. Still nothing anywhere

No second *parallel* family exists for khm, lao, bod, tgk, pbt, ory, lvs, afr,
mlt short of SIB-200/Taxi1500. Belebele (or, for cy/ga/la/kmr/ug, MultiBLiMP
alone) is their floor.

Recommendation, in order: (1) generate the **INCLUDE v2** MC variant and wire
it — natively sourced, ours to change, 19 trained languages gain a family and
Odia stops being a single-benchmark language; (2) switch LAMBADA to the
stablelm variant or drop it; (3) port **SIB-200** — one task family, 205
languages, fixes the remaining one-family languages at once, and its FLORES
provenance is a known quantity; (4) **Uhura** for the African set. MILU becomes
optional once INCLUDE v2 covers or/pa. Skip MT-based and MMLU-Pro-class suites:
at 90M–1.7B they measure nothing.

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
