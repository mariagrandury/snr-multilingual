# Probe candidates from the website list (2026-10-08)

The question: which benchmarks on the website's list (`configs/multilingual_benchmarks.csv`, 304 rows) have task configs in the pinned harness and have never been in `auto` or `auto_probe`? The candidates are grouped by trained language, starting with the languages that have the fewest benchmarks.

## Method

- **Already done** means `groups.auto` ∪ `groups.auto_probe` in `configs/tasks.json`. It is matched the way `configs.tasks_for_benchmarks` matches: the exact benchmark name or a `<name>_…` prefix, in the pretraining stage, with `jp` read as `ja`.
- **Available** means the pinned harness has a config for the task. The harness is `/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/lm-evaluation-harness` at `51d6f4b6`, and its `lm_eval/tasks` directory listing matches the wheel `lm_eval-0.4.13.dev0`. Task names were taken from `task:`/`group:`/`tag:` in the YAMLs, and formats from `output_type` and `doc_to_choice`.
- **Coverage** is the number of families each trained language has in `auto`. The `rf_`/`rfgm_` twins are left out, and `include_v2_en`/`_og` and `cultural_bench_easy`/`_hard` count as separate families. The number in brackets adds the families that are only in `auto_probe`. This count is not the paper's "Fam." column in `app_02_table_languages.tex`. That column counts what the harness offers, not what we evaluate.
- **Format labels:**
  - **letters**: the answer is a letter (A–D) scored by loglikelihood. At 90M–1.7B these sit at chance, so they are only useful through an `rf_` twin (rq00).
  - **cloze**: loglikelihood over the answer strings or label words.
  - **minimal pair**: a 2-way grammatical vs ungrammatical choice.
  - **gen**: `generate_until`. It is costly and almost always at floor at our sizes.
- **Item counts** come from the CSV `n_items` unless marked *(cache)*. Those were read from `dataset_info.json` in `hf_home/datasets`, which also means the dataset is already built. A dataset that is not cached has to be added to `src/evals/configs/eval_datasets.txt` and built first, because batch mode aborts on any uncached dataset.
- **Gate reminder:** about 71 items or more for a 2-way task, and about 40 or more for a 4-way task.

CSV totals: 53 rows are already covered (fully or partly), 149 have harness configs but are not covered, 89 are lighteval-only, and 13 are in neither framework at our pin. Some rows the CSV marks as lighteval-only do exist in our harness under another name: `m_mmlu_*` (okapi_mmlu), `eus_exams_*`, `kbl_*`, `norquad`, `ask_gec`, `libra`, `cabreu` and `xlsum_*`. They are counted as available below.

### Registered but never evaluated

These tasks are already in `configs/tasks.json` but sit outside `auto` and `auto_probe`. They need no registration work, only a group entry:

- `truthfulqa` (okapi m_truthfulqa ar/es/eu/hi/ru, mc1+mc2)
- `truthfulqa_mc1` (vi, zh)
- `mgsm_direct` / `mgsm_native_cot` (gen)
- `mlqa` (gen)
- `aexams`
- `arabic_leaderboard_complete`
- `cmmlu`
- `agieval`
- `afrimmlu`
- `afrixnli`

### Partly covered families with uncovered trained languages

- **multiblimp:** the harness ships `multiblimp_lav`, `multiblimp_sqi` and `multiblimp_hbs`, and all three are cached. None of them is registered.
- **okapi m_truthfulqa:** only `truthfulqa_vi_mc2` and `truthfulqa_zh_mc2` are in `auto`, through the `truthfulqa_mc2` prefix. The data for all 31 languages is cached.
- **truthfulqa-multi:** `_ca` and `_gl` are not registered.
- **xquad:** only `es` is in the probe. The other 12 languages are generative.

## Trained languages, lowest coverage first

### bs (Bosnian): 1 family (global_piqa)
- **MultiBLiMP Serbo-Croatian**: `multiblimp_hbs`; minimal pair; 3,286 items *(cache)*. The UD-based hbs set is shared by bs, hr and sr. Before registering it, decide which language tag it gets (or tag it three times). Belebele has no Bosnian, so this is the only cheap addition.

### lv (Latvian): 1 family (belebele)
- **MultiBLiMP Latvian**: `multiblimp_lav`; minimal pair; 3,032 items *(cache)*. This is the best single addition on the list.
- No other harness candidate. `lextreme` (lighteval, gen) and `wmt` (lv pairs are not in our harness, which only has wmt14 en-fr and wmt16 de/ro) do not apply.

### sl (Slovenian): 3 families
- No harness candidate. `multiblimp_slv` is already in `auto`, and `flores200` in our harness covers only the Iberian pairs. The only list entry with Slovenian is `lextreme` (lighteval, gen).

### th (Thai): 4 families
- `mgsm_direct_th` (registered), `xquad_th` and `mmlu_prox_th`/`_lite_th` are all gen and costly, and should be skipped.
- The cheap Thai options on the list are all lighteval-only: `thai_exams`, `m3exam` (th), `meta_mmlu` (th), `xnli2.0` (th) and `community_hellaswag_tha`. Adding any of them means porting it.

### no (Norwegian): 4 families [+1: noreval]
The current `noreval` registration has only `ncb`, `norbelebele` and `norcommonsenseqa`, five prompts each. The NorEval sets below are not registered:
- **NorOpenBookQA**: `noropenbookqa_nob` / `_nno` (p0–p4); cloze over `choices.text`; 376 nb + 90 nn. Not cached.
- **NRK-Quiz-QA**: `nrk_quiz_qa_nob` / `_nno`; cloze; 3,600 nb + 1,330 nn; 2–5 options. Native knowledge. Not cached.
- **NorTruthfulQA MC**: `nortruthfulqa_mc_nob` / `_nno`; cloze over `mc1_targets`; 488 nb + 57 nn (nn is under the gate floor).
- **NoReC document / sentence**: `norec_document`, `norec_sentence`; cloze over "negativ"/"positiv"; 2,955 / 583 items. Sentiment, which is easy to clear but weakly tied to knowledge.
- **Okapi MMLU nb**: `m_mmlu_nb`; letters. Only useful through an `rf_` twin.
- Gen, skip: `noridiom`, `norquad`, `nortruthfulqa_gen`, `norsumm`, `tatoeba_*`, `ask_gec`, `norrewrite_instruct`, `norsummarize_instruct`.

### az (Azerbaijani): 5 families
- No harness candidate. `tumlu_mini` is lighteval-only.

### sq (Albanian): 5 families
- **MultiBLiMP Albanian**: `multiblimp_sqi`; minimal pair; 243 items *(cache)*. This clears the 2-way gate with a margin.

### cs (Czech): 6 families
- Only `mmlu_prox_cs` (gen, 10 options, 5-shot CoT), so skip it. `lextreme` is lighteval-only.

### ro (Romanian): 6 families
- **Okapi TruthfulQA**: `truthfulqa_ro_mc1` / `_mc2`; cloze; 779 items *(cache)*. TruthfulQA is often at or below chance for small models. `truthfulqa_mc2` was promoted on 2026-09-23, so check its gate rows before adding the whole okapi set.
- **Okapi MMLU**: `m_mmlu_ro`; letters, so it needs an `rf_` twin.
- Gen: `xquad_ro`, `wmt16-en-ro` / `-ro-en`.

### ms (Malay), ka (Georgian), kk (Kazakh): 6 families each
- No harness candidate for any of them. `tumlu_mini` (kk) is lighteval-only.

### da (Danish): 7 families
- `truthfulqa_da_mc1` / `_mc2`: cloze, 781 items *(cache)*.
- `m_mmlu_da`: letters, needs an `rf_` twin.

### sk (Slovak): 7 families
- `truthfulqa_sk_mc1` / `_mc2`: cloze, 778 items *(cache)*.
- `m_mmlu_sk`: letters.

### hr (Croatian): 7 families
- `multiblimp_hbs`: see bs.
- `truthfulqa_hr_mc1` / `_mc2`: cloze, 769 items *(cache)*.
- `m_mmlu_hr`: letters.

### mr (Marathi): 7 families
- `truthfulqa_mr_mc1` / `_mc2`: cloze, 764 items *(cache)*.
- `m_mmlu_mr`: letters.
- **IndicParam Marathi**: `indicparam_marathi`; letters (a–d). Graduate-level, so it would be at chance even through an `rf_` twin. Skip it.
- `mmlu_prox_mr`: gen.

### ml (Malayalam): 7 families
- `truthfulqa_ml_mc1` / `_mc2`: cloze, 694 items *(cache)*.
- `m_mmlu_ml`: letters.

### bg, he, lt, et, ur, fa: 7 families each
- No new harness candidate from the list.
- Lighteval-only options: `exams` (bg, lt), `xnli2.0` (bg, ur), `indicxcopa` (ur), `xcodah`/`xcsqa` (ur).
- `mmlu_prox_ur` is gen.
- fa has `toksuite` in the probe already.

### ko (Korean): 7 families [+2: haerae, kmmlu]
- **KoBEST**: `kobest_boolq`, `_copa`, `_hellaswag`, `_sentineg`, `_wic`; all cloze or label words (예/아니오, 부정/긍정); roughly 1,404 / 1,000 / 500 / 397 / 1,260 test items. Not cached. This is the strongest Korean addition.
- **CLIcK**: `click_cul_*`, `click_lang_*`; letters (A–E).
- **CSAT-QA**: `csatqa_*`; MC over option strings; 187 items over 5 options, which is borderline for the gate.
- Skip: `kormedmcqa` (gen, letter), `kbl` (gen), `hrm8k` (gen), `mmmlu_ko_kr` (letters), `mmlu_prox_ko` (gen).

### fi (Finnish): 8 families
- No harness candidate. `tydiqa` and `mkqa` are lighteval-only and gen.

### sr (Serbian): 8 families
- `multiblimp_hbs`: see bs.
- `truthfulqa_sr_mc1` / `_mc2`: cloze, 785 items *(cache)*.
- `m_mmlu_sr`: letters.
- `mmlu_prox_sr`: gen.
- The `serbian_evals` suite (arc, hellaswag, piqa, winogrande, boolq, openbook, mmlu, oz) is lighteval-only. Porting it would be the big Serbian win.

### ne (Nepali): 8 families
- `truthfulqa_ne_mc1` / `_mc2`: cloze, 774 items *(cache)*.
- `m_mmlu_ne`: letters.
- `indicparam_nepali`: letters, graduate level, skip.
- `mmlu_prox_ne`: gen.

### nl (Dutch): 8 families [+1: blimp_nl]
- **LAMBADA (StableLM re-translation)**: `lambada_openai_mt_stablelm_nl`; last-word loglikelihood (cheap despite the CSV's "generative"); about 5,153 items. Not cached.
- `truthfulqa_nl_mc1` / `_mc2`: cloze, 785 items *(cache)*.
- `m_mmlu_nl`: letters.

### ca (Catalan): 8 families [+5 ibero_*]
The IberoBench leftovers:
- **COPA-ca**: `copa_ca`; cloze; 500 items.
- **XNLI-va**: `xnli_va`; cloze, NLI written as "…, correcte? Sí/No/…"; 5,010 items (Valencian).
- **TE-ca**: `teca`; same cloze-NLI style; 3-way.
- **Parafraseja**: `parafraseja`; cloze paraphrase; 2-way.
- **TruthfulQA-multi ca**: `truthfulqa-multi_mc1_ca` / `_mc2_ca`; cloze; 817 items. The family is already in `auto` for en/es/eu, so this is the cheapest of the five: it only needs a tasks.json row.
- `truthfulqa_ca_mc1` / `_mc2` (okapi): cloze, 777 items *(cache)*.
- `m_mmlu_ca`: letters.
- Skip:
  - `cabbq` (BBQ-style; `bbq` was dropped at 23 min per checkpoint)
  - `wnli_ca` (71 items, under the gate)
  - gen tasks: `catalanqa`, `coqcat`, `xquad_ca`, `mgsm_direct_ca`, `cocoteros_va`, `truthfulqa_va`, `phrases_*`, `cabreu`

### pl (Polish): 9 families
- Only `polemo2_in` / `_out`, which is gen (letter generation; 722 + 494 items). Skip it.

### uk (Ukrainian), sv (Swedish), hu (Hungarian), ta (Tamil): 9 families each
- `truthfulqa_{uk,sv,hu,ta}_mc1` / `_mc2`: cloze; 770 / 774 / 771 / 743 items *(cache)*.
- `m_mmlu_{uk,sv,hu,ta}`: letters.
- `mmlu_prox_{uk,hu}`: gen.

### bn (Bengali): 9 families [+1: bangla]
- **BoolQ-BN**: `boolqa_bn`; cloze yes/no; 432 items. The only Bangla set the `bangla` probe left out.
- `truthfulqa_bn_mc1` / `_mc2`: cloze, 781 items *(cache)*.
- `m_mmlu_bn`, `mmmlu_bn_bd`: letters.
- `mgsm_direct_bn` (registered) and `mmlu_prox_bn`: gen.

### tr (Turkish): 9 families [+3: toksuite, turblimp, turkishmmlu]
- Nothing new in the harness beyond `xquad_tr` (gen).
- `arc_tr`, `community_hellaswag_tur`, `tumlu_mini` and `tquad_v2` are lighteval-only.

### id (Indonesian): 10 families
- **COPAL-ID**: `copal_id_standard`, `copal_id_colloquial`; cloze; 559 + 559 items. Natively written and culture-specific.
- `truthfulqa_id_mc1` / `_mc2`: cloze, 778 items *(cache)*.
- `m_mmlu_id`, `mmmlu_id_id`: letters.

### el (Greek): 10 families
- Only `xquad_el` (gen). `greekmmlu` (`gmmlu`) is not in our pin.

### ja (Japanese): 10 families [+1: mela]
- **JGLUE**:
  - `ja_leaderboard_jcommonsenseqa`: cloze over choices; about 1,119 items.
  - `ja_leaderboard_jnli`: label words 含意/矛盾/中立; about 2,434 items.
  - `ja_leaderboard_marc_ja`: ポジティブ/ネガティブ; about 5,654 items.
  
  All three are validation splits, use instruction-style prompts and default to 3-shot, so set the shots to match our 0-shot recipe. Not cached.
- `mmmlu_ja_jp`: letters.
- Gen: `ja_leaderboard_jaqket_v2`, `ja_leaderboard_jsquad`, `jfinqa`, `mgsm_direct_ja`, `mmlu_prox_ja`.

### pt (Portuguese): 11 families
- **LAMBADA StableLM pt**: `lambada_openai_mt_stablelm_pt`; last-word loglikelihood; about 5,153 items.
- **ASSIN**: `assin_entailment`, `assin_paraphrase`; cloze-NLI ("…, certo? Também/Não…"); 2-way. Mixes pt-BR and pt-PT.
- **ASSIN 2 RTE**: `assin2_rte`; cloze-NLI. (`assin2_sts` is gen and should be skipped.)
- `truthfulqa_pt_mc1` / `_mc2`: cloze, 788 items *(cache)*.
- `m_mmlu_pt`, `mmmlu_pt_br`: letters.

### it (Italian): 12 families [+3: evalita_llm, mela, toksuite]
- The `evalita_llm` probe covers only `_at`, `_te` and `_wic`.
- **Evalita leftovers**:
  - `evalita-mp_faq`: letters A–D; 401 items.
  - `evalita-mp_sa`: label words.
  - `evalita-mp_hs`: Vero/Falso.
  
  Each has 6 prompts, and the `_ls`, `_ner`, `_re` and `_sum` tasks are gen.
- `truthfulqa_it_mc1` / `_mc2`: cloze, 783 items *(cache)*.
- `lambada_openai_mt_stablelm_it`: a re-translation that duplicates the `lambada_openai_mt_it` already in `auto`.
- `m_mmlu_it`, `mmmlu_it_it`: letters.

### vi (Vietnamese), hi (Hindi): 13 families each
- vi:
  - `truthfulqa_vi_mc1` (registered, mc2 already in `auto`)
  - `m_mmlu_vi`: letters
  - `xquad_vi` and `mlqa` vi: gen
- hi:
  - **BHS Hindi**: `bhs_hindi`; minimal pair (agreement); 6 paradigms of about 1,000 items each.
  - `truthfulqa_hi_*`: registered, not in `auto`.
  - `m_mmlu_hi`, `mmmlu_hi_in`: letters.

### de (German), fr (French): 13 families each
- de:
  - `truthfulqa_de_mc1` / `_mc2`: cloze, 788 items *(cache)*
  - `lambada_openai_mt_stablelm_de` (duplicate)
  - `m_mmlu_de`, `mmmlu_de_de`: letters
  - `mgsm_direct_de`, `xquad_de`, `pisa`: gen
- fr (the `french_bench` probe has only arc, grammar, reading_comp, vocab and xnli):
  - `french_bench_hellaswag`: cloze; 9,338 items. Strong.
  - `french_bench_boolqa`: Oui/Non; 178 items.
  - `french_bench_topic_based_nli`: 3 labels; 600 items.
  - `french_bench_fquadv2_bool`: about 400 items.
  - `histoires_morales`: cloze, 2-way; 12,000 items.
  - `truthfulqa_fr_*`: cloze, 787 items *(cache)*.
  - `glianorex_fr`, `m_mmlu_fr`, `mmmlu_fr_fr`: letters.
  - `french_bench_trivia`, `_multifquad`, `_orangesum_*`, `_fquadv2_genq`, `_wikitext_fr` (perplexity): gen.

### ru (Russian), ar (Arabic): 14 families each
- ru:
  - `truthfulqa_ru_*`: registered, not in `auto`
  - `m_mmlu_ru`: letters
  - `mgsm_direct_ru`, `xquad_ru`: gen
  - The MERA tasks (`parus`, `rcb`, `ruopenbookqa`, `ruworldtree`, `mathlogicqa`) are lighteval-only.
- ar: a lot is cached, through OALL:
  - **AlGhafa-translated**: `arabic_mt_{arc_challenge,arc_easy,boolq,copa,hellaswag,mmlu,openbook_qa,piqa,race,sciq,toxigen}`; cloze over `choices`, apart from `arabic_mt_mmlu`. All cached.
  - **AlGhafa-native**: `arabic_leaderboard_alghafa`. Cached.
  - **ACVA**: `arabic_leaderboard_acva_*`; 2-way True/False; 8,710 items. Cached, but the statements are GPT-generated.
  - **ArabCulture completion**: `arab_culture_completion_*`; cloze; 3,482 items. (`arab_culture` is the letter variant.)
  - **AraDiCE MSA**: `AraDiCE_{boolq,piqa,openbookqa,winogrande,truthfulqa_mc1}_msa`; cloze.
  - `arabicmmlu`, `arabic_leaderboard_arabic_mmlu`, `mmmlu_ar_xy`, `m_mmlu_ar`, `aexams`: letters.
  - `copa_ext_ar` / `arabic_mt_copa`: only 90 items, which is gate-marginal.

### zh (Chinese): 16 families [+4]
- **ACLUE**: `aclue_*`; letters; 4,967 items (classical Chinese).
- **TMMLU+**: `tmmluplus_*`; letters; 22,690 items (Traditional Chinese, Taiwan).
- **TMLU**: `tmlu_*`; letters (Taiwan).
- **AGIEval zh**: `agieval_cn` (registered); letters.
- **CMMLU**: `cmmlu` (registered); letters. The script-format dataset cannot be built with datasets v3.
- `m_mmlu_zh`, `mmmlu_zh_cn`: letters.
- `truthfulqa_zh_mc1`: registered.
- `mgsm_direct_zh`, `xquad_zh`, `mlqa`: gen.
- None of the Chinese candidates are cloze. They are all letter-only, so `rf_` twins would be needed.

### es (Spanish): 18 families [+4]
- **COPA-es**: `copa_es`; cloze; 500 items.
- **HEAD-QA es**: `headqa_es`; cloze over answer text; about 2,700 items.
- **EusExams es**: `eus_exams_es_*`; letters (Basque public-service exams in Spanish).
- **CareQA es**: `careqa_es`; letters; 5,621 items.
- `truthfulqa_es_*` (okapi): registered.
- `m_mmlu_es`, `mmmlu_es_la`: letters.
- Skip: `esbbq` (BBQ, costly), `wnli_es` (71 items), `mgsm_direct_es`, `cocoteros_es`, `noticia` and `spanish_bench` gen tasks.

### en: 25 families
- Out of scope for multilingual coverage. The list's English entries are mostly the `_en` halves of multilingual sets (`afrimmlu`, `bertaqa_en`, `careqa_en`, `headqa_en`, `glianorex_en`, `injongointent`, `nollysenti`, `uhura_arc_easy`).

## Languages outside the trained set (brief)

These are inert unless a pass is run with `--all-languages`.

- **eu**: `eus_exams_eu_*`, `eus_proficiency`, `eus_reading`, `eus_trivia`, `bertaqa_eu`, `bhs_basque`, BasqueGLUE (script dataset, cannot be built), `wnli_eu` (71 items), `m_mmlu_eu`, `truthfulqa_eu_*`.
- **gl**: `galcola`, `parafrases_gl`, `truthfulqa-multi_*_gl`, `mgsm_direct_gl`, `summarization_gl` (gen).
- **is**: `icelandic_winogrande` (1,095 items), `m_mmlu_is`, `arc_challenge_mt_is`.
- **hy / gu / kn / te**: `truthfulqa_{hy,gu,kn,te}_*`, `m_mmlu_*`. Indic: `indicparam_*` (gu, or, …).
- **sw**: `bhs_swahili`.
- **African languages**: `afrimmlu`, `afrixnli`, `afrisenti`, `masakhanews`, `sib`, `naijarc`, `uhura_arc_easy`, `nollysenti` (mostly cloze or label-word tasks).
- **Arabic dialects**: `darijammlu`, `darijahellaswag`, `egymmlu`, `egyhellaswag`, `AraDiCE_*_egy` / `_lev`.
- **ky**: `ulut`, `uleval`.
- **ug**: `lambada_uyghur`.

## Top-10 for the lowest-coverage languages

1. **`multiblimp_lav`** (lv, 1 → 2 families). Minimal pairs, 3,032 items, cached; only a tasks.json row is needed.
2. **`multiblimp_hbs`** (bs 1 → 2, and also hr and sr). 3,286 items, cached; decide the language tag first.
3. **`multiblimp_sqi`** (sq, 5 → 6). 243 items, cached.
4. **NorEval MC extras** (no, 4 → up to 8): `noropenbookqa_nob`, `nrk_quiz_qa_nob`, `nortruthfulqa_mc_nob` and `norec_document`/`_sentence`, all cloze. They can be added to the existing `noreval` probe benchmark, but the datasets need building.
5. **Okapi m_truthfulqa mc1/mc2** for da, sk, hr, ro, mr, ml, sr, ne, uk, sv, hu, ta, nl, bn, id, ca, pt, it, fr, de. One family adds +1 to twenty trained languages, it is cloze, and all 31 languages are cached. Check the `truthfulqa_mc2` gate first: TruthfulQA can stay at or below chance at 1.7B.
6. **KoBEST** (ko, 7 → 8): five cloze tasks, native Korean.
7. **Okapi `m_mmlu_*` with an `rf_` twin** (ro, da, sk, hr, mr, ml, sr, ne, nb, uk, sv, hu, ta, …). This adds a knowledge family to most low-coverage languages that have none, but the letter original sits at chance. It is only worth it with the `rf_` twin (`make_rf_tasks.py`), so it is a bigger job than the others.
8. **Catalan IberoBench leftovers** (ca, 8 → about 12): `copa_ca`, `xnli_va`, `teca`, `parafraseja`, `truthfulqa-multi_mc1_ca`, all cloze. TruthfulQA-multi is already an `auto` family.
9. **LAMBADA StableLM nl / pt** (nl 8 → 9, pt 11 → 12): cheap loglikelihood, about 5k items each.
10. **COPAL-ID** (id, 10 → 11): 2 × 559 cloze items, native Indonesian.

Close behind: JGLUE (`jcommonsenseqa`/`jnli`/`marc_ja`, ja), `boolqa_bn` (bn), and the cached cloze AlGhafa-translated set (ar).

**No harness candidate at all:** sl, th, az, cs, ms, ka, kk, fi, bg, he, lt, et, el, pl, fa. Their cheap options on the list are lighteval-only: `serbian_evals`, `thai_exams`, `tumlu_mini`, `exams`, `xnli2.0`, `indicxcopa`, `lextreme`. Raising their coverage means porting from lighteval or adding datasets that are not on the list.

## Final list and answers (added 2026-10-08, registered in `configs/tasks.json`)

### Where the tasks are

- **Groups.** Everything is in the one existing group, `auto_probe` (user decision, 2026-10-08): the 159 originals, their `rf_` twins, and `m_mmlu`/`mmmlu` with their twins. One group means one job per checkpoint that runs every missing task, old candidates included. The watcher reads only `auto` unless it is given `--group`, and no watcher is running, so nothing is evaluated until the pass below is launched. `probe.sh` (`SNR_TRAINED_GROUPS=auto,auto_probe`) sees all of them.
- **Discarded (2026-10-08).** The old candidates (in `auto_probe` before today, never promoted) that fail the gate at both 1B and 1.7B now sit in `groups.discarded`, a list of 100 task names that `tasks_for_benchmarks` never selects. The verdict is rule 1 (`above_random.scores_and_mask`) on the predictivity pool with `SNR_TRAINED_GROUPS=auto,auto_probe`, from the ladder report of 2026-10-07 15:45, on the runs that trained the task's language. Reason for every one: not above chance at 1B nor at 1.7B. The paper's Table `tab:discarded-benchmarks` (`make_appendix_tables.py`) is generated from the group.
  - C-Eval 51 of 52 (zh; `ceval-valid_art_studies` passes at 1.7B and stays), ZhoBLiMP 10 of 118, NorEval 9 of 16, TurkishMMLU 9 of 9, BLiMP-NL 8 of 84, HAE-RAE 5 of 5, Bangla 3 of 4, TurBLiMP 3 of 16, EVALITA-LLM 2 of 18.
  - `haerae` and `turkishmmlu` lost every task, so they also left the `auto_probe` benchmark list.
  - Not discarded, because the gate gives no verdict: `xquad_es` (generative, no chance level), `arc_eu_challenge` (eu is trained by no cell, so its 0/0 comes from the untrained population), and 77 tasks that ran but have no score in the ladder report (`kmmlu_*` 45, `mela_*` 10, EVALITA `_te_`/`_wic_` 12, `french_bench_grammar`/`_vocab`/`_reading_comp`/`_xnli`, `catcola`/`escola`, `siqa_ca`, `piqa_bn`, `ncb`, `truthfulqa_gl_mc1`). Their tasks.json `metric: acc_norm` is not what the harness emits (`acc`, or `mcc` for MELA), so `results_io.flatten` drops them: the same bug as `truthfulqa_gl_mc1` below, not fixed here.
- **`auto` is unchanged.** It selects the same 846 tasks before and after the edit.
- **New benchmark names avoid the prefix rule.** None of them is equal to an `auto` benchmark or starts with `<auto benchmark>_`, because `tasks_for_benchmarks` would otherwise pull them into `auto`. That is why the new names are `multiblimp-extra` (not `multiblimp_*`), `m_truthfulqa_mc2` (not `truthfulqa_mc2`) and `truthfulqa-multi_mc1_ca` → `ibero_truthfulqa`. `noreval_extra`, `evalita_llm_extra` and `french_bench_extra` still fall under the `auto_probe` families by prefix.
- **Relabelled entries.** 13 existing Okapi TruthfulQA rows that were registered but never in a group (`truthfulqa` = ar/es/eu/hi/ru; `truthfulqa_mc1` = vi/zh) now carry `m_truthfulqa_mc1/_mc2`.
- **Metadata.** `n_options` and `n_items` were measured through a real `TaskManager`, as `add_harness_tasks.py` does. `metric: acc_norm` is set only where the harness emits it. TruthfulQA, KoBEST (except hellaswag), NoReC, EVALITA, COPA, NLI, CareQA, ArabicMMLU and m_mmlu emit `acc` only, and an `acc_norm` override would make `results_io.flatten` drop the task.
- **Pre-existing bug, not fixed here.** `truthfulqa_gl_mc1` has `metric: acc_norm`, but the task emits only `acc`. It is inert because gl is not trained.

### Task list (trained languages only)

| benchmark | tasks | languages | format |
|---|---|---|---|
| multiblimp-extra | multiblimp_lav, _sqi, _hbs | lv, sq, hr (hbs is shared by bs/hr/sr and tagged hr) | minimal pair |
| m_truthfulqa_mc1 / _mc2 | truthfulqa_{l}_mc1 ×26, _mc2 ×24 | ar bn ca da de es fr hi hr hu id it ml mr ne nl pt ro ru sk sr sv ta uk (+vi, zh mc1; their mc2 is already in auto) | cloze |
| truthfulqa_mc1 | truthfulqa_mc1 | en | cloze |
| truthfulqa-multi_mc2 | truthfulqa-multi_mc2_{en,es,ca} | en es ca | cloze |
| ibero_truthfulqa | + truthfulqa-multi_mc1_ca | ca | cloze |
| noreval_extra | noropenbookqa_{nob,nno}_p0-4, nrk_quiz_qa_{nob,nno}_p0-4, nortruthfulqa_mc_{nob,nno}_p0-4, norec_{document,sentence}_p0-4 (40) | no | cloze; **p2/p3 of noropenbookqa and nrk_quiz_qa are lettered**, their p0/p1/p4 siblings are the cloze form |
| kobest | kobest_boolq, copa, hellaswag, sentineg, wic | ko | cloze/label words |
| evalita_llm_extra | evalita-mp_{faq,hs,sa}_prompt-1..6 (18) | it | cloze; **prompt-3/4 are lettered**, prompts 1/2/5/6 are the cloze form |
| ibero_copa | copa_es, copa_ca | es ca | cloze |
| ibero_nli | teca, xnli_va, assin2_rte | ca ca pt | label words |
| ibero_paraphrase | parafraseja | ca | label words |
| copal_id | copal_id_standard, _colloquial | id | cloze |
| headqa | headqa_es | es | cloze |
| careqa | careqa_es | es | **letters → rf_careqa_es** |
| french_bench_extra | french_bench_boolqa, _hellaswag, _topic_based_nli, _fquadv2_bool | fr | cloze; **fquadv2_bool lettered → rf_** |
| oall_alghafa | arabic_leaderboard_alghafa_* (9 native tasks) | ar | options listed, answer strings scored |
| oall_mt | arabic_mt_{arc_challenge,arc_easy,boolq,copa(90 items),hellaswag,mmlu,openbook_qa,piqa,race,sciq,toxigen} | ar | answer strings scored |
| oall_acva | arabic_leaderboard_acva (58-subject group) | ar | True/False words |
| oall_exams | arabic_exams | ar | **letters → rf_arabic_exams** |
| oall_arabic_mmlu | arabic_leaderboard_arabic_mmlu (57-subject group) | ar | **letters → rf_ group of 57 leaf twins** |
| arabicmmlu | arabicmmlu (40-subject group, OALL v2) | ar | **letters → rf_arabicmmlu over the `All` config (same 14,455 items)** |
| m_mmlu | m_mmlu_{l} ×27 | ar bn ca da de es fr hi hr hu id it ml mr nb(no) ne nl pt ro ru sk sr sv ta uk vi zh | **letters → rf_m_mmlu_*** |
| mmmlu | mmmlu_{ar_xy,bn_bd,de_de,es_la,fr_fr,hi_in,id_id,it_it,ja_jp,ko_kr,pt_br,zh_cn} | 12 | **letters → rf_mmmlu_* (whole split, as rf_mmlu)** |

**Left out:**
- `aclue`: a script dataset that cannot be built. A parquet export exists at `refs/convert/parquet`, so it would need a parquet wrapper YAML, like `include_v2`.
- `tmlu`: `miulab/tmlu` returns 404 on the Hub.
- `assin_entailment` / `assin_paraphrase`: the harness YAML has no `dataset_name`, and `nilc-nlp/assin` has three configs, so it is ambiguous offline.
- `wnli_es` / `wnli_ca`: 71 items.
- `paws_es_spanish_bench` and `xnli_es_spanish_bench`: duplicates of the `paws`/`xnli` es tasks already in `auto`.
- The OALL `_light` variants: 10% subsets of the full tasks.
- `alghafa/copa_ar`, `piqa_ar`: not part of OALL.
- All `flores_*` tasks: see below.

**rf twins.**
- They live under `src/evals/tasks/rf/{careqa,french_bench_extra,oall_exams,oall_arabic_mmlu,arabicmmlu,m_mmlu,mmmlu}/` and are registered as `rf_<family>` with `metric: acc_norm`.
- The 7 templates are now in `make_rf_tasks.TEMPLATES` (with `RF_ONLY`, `GROUP_FAMILIES`, `TAG` and `FILTER_FIELDS`), and the twins are generated by `make_rf_tasks.py --family careqa french_bench_extra oall_exams oall_arabic_mmlu arabicmmlu m_mmlu mmmlu`. The regeneration is byte-identical to the first build (108 of 109 files). The one intended difference is the docstring of `arabicmmlu/utils.py`. The templates:
  - careqa: `{{question}}\nAnswer:` / `[op1..op4]` / `cop-1`
  - fquadv2_bool: `context + question + "D'après le contexte, répondre à la question est"` / `['possible','impossible']`
  - oall_exams and oall_arabic_mmlu: `[A,B,C,D]` / `gold`
  - arabicmmlu: `utils.process_docs` builds `rf_text`/`rf_choices`/`rf_gold` from `Option 1-5`
  - m_mmlu: `{{instruction}}` / `[option_a..d]`
  - mmmlu: `openai/MMMLU <XX_YY>` / `[A..D]`
- All 44 twins load offline and their item counts match the originals. Rows with blank options are dropped: m_mmlu_sk −1, mmmlu de −2, ja −1, zh −2.
- The 57 stray `*_light` twins from the first build were deleted with approval. The generator skips `_light` leaves.

**Datasets.** 185 configs of 27 repos were built into `hf_home/datasets` on the login node. They are listed in `src/evals/configs/eval_datasets.txt` under `# probe 2026-10-08`:
- ltg/{noropenbookqa,nrk_quiz_qa,nortruthfulqa_mc,norec_document,norec_sentence}
- skt/kobest_v1
- evalitahf/{faq,hatespeech_detection,sentiment_analysis}
- BSC-LT/COPA-es, projecte-aina/{COPA-ca,teca,Parafraseja}, gplsi/xnli_va, nilc-nlp/{assin,assin2}
- haryoaw/COPAL, EleutherAI/headqa, HPAI-BSC/CareQA
- manu/{french_boolq,fquad2_test,french_bench_hellaswag,topic_based_nli_test}
- OALL/Arabic_MMLU, MBZUAI/ArabicMMLU
- alexandrainst/m_mmlu, openai/MMMLU

### Answers

**FLORES.**
- **Not MCQA.** Every flores task in the pinned harness is `generate_until` with BLEU/TER/chrF. The trained pairs are in `flores_es`, `flores_ca` and `flores_pt` (the IberoBench hubs), 1,012 devtest sentences per pair: 8 directed pairs at L8, 18 at L15/L30 and 30 at L50.
- **Cannot be built.** It reads `facebook/flores`, a loading-script dataset that `datasets` 4.x refuses, so it would need a port to `openlanguagedata/flores_plus`.
- **Cost.** The closest measured generative task is `xquad_es`: 1,190 items in 107 s on the 1.7B, one GPU worker. Its responses average about 3,100 characters, because the base models run on towards `max_gen_toks` 2048 instead of stopping at `\n`. On that basis FLORES costs about 90–110 s per pair at 1.7B: about 50 GPU-min per L50 checkpoint and about 13 GPU-min at L8. MMLU, for comparison, takes 183 s.
- **Not added.**

**m_mmlu vs MMMLU.**
- **m_mmlu** is Okapi (alexandrainst): MMLU machine-translated with ChatGPT into 34 languages, 27 of them trained. It is one task per language, about 11–13k items, scored by letter.
- **MMMLU** is OpenAI's professional human translation into 14 languages, 12 of them trained. It is a 57-subject group with 14,042 items, also scored by letter.
- **Cost per language** is the same as MMLU: about 3–3.5 GPU-min at 1.7B, measured from `mmlu` (183 s) and `global_mmlu_full_*` (170–250 s). An L50 1.7B checkpoint costs:
  - m_mmlu alone: about 1.5 GPU-h
  - MMMLU alone: about 0.7 GPU-h
  - both plus twins: about 3.6 GPU-h, or 215 worker-min
- **A final-only pass** over the 101 cells at 600M/1B/1.7B, with both benchmarks and their twins, is roughly 150–200 GPU-h (40–50 node-h). That is affordable once on preemptable. Promoting both into `auto` would multiply it by 12.
- **Overlap.** All 12 trained MMMLU languages are already covered by `global_mmlu_full` in `auto`. m_mmlu adds MMLU coverage for only 9 trained languages: ca, da, hr, hu, ml, mr, no, sk, ta.
- **Recommendation was:** m_mmlu (with its `rf_` twin) for those 9 languages, and skip MMMLU.
- **Decision (2026-10-08):** run both, in full: m_mmlu in all 27 trained languages and MMMLU in all 12, each with its `rf_` twin. Both are in `auto_probe`, and are screened at the final checkpoint only. The overlap with Global-MMLU becomes a comparison of three translations of the same items: Okapi machine translation, OpenAI professional translation and Global-MMLU.
- **Walltime.** The MMLU-sized tasks (11–14.5k items: `m_mmlu_*`, `mmmlu_*`, `arabicmmlu`, `arabic_leaderboard_arabic_mmlu`, `arabic_mt_mmlu` and the `rf_` twins) now weigh six tasks in `auto_evals_cscs.submit_eval`, like `rf_global_mmlu_full`. `mmlu` and `global_mmlu_full_*` take 183–199 s per worker-task at 1.7B (per-task timings, seed 1904) against the 41 s the fit assumes, so without the weight 64 of the 100 jobs were undersized (a 600M all-languages job: 50 min requested, about 80 needed). With it one pass covers everything. `--group auto_probe --size 600M,1B,1.7B --final-only --dry-run` (2026-10-08, after the discard): 100 eval jobs and 1 conversion (lm-1.7B-L8-swiglu, iters 76950–81000), 16,531 task runs: 12,474 new, 3,615 old candidates on the 22 cells that never had the first probe (swiglu, the replicate seeds), and 442 `auto` top-ups on the unfinished swiglu cells. The largest job is lm-1.7B-L30-swiglu-seed1904 (754 tasks, 03:45:00). The requests total 133 node-hours, an upper bound (the probe job of 2026-09-25 ran 403 small tasks in 12 min).

**Translated ARC/HellaSwag/MMLU versions** (selected by `auto` / `auto_probe`):
- **ARC:**
  - Okapi m_arc `arc_<l>`, 31 languages (auto)
  - LumiOpen `arc_mt` (`arc_challenge_mt_*`), 11 languages (auto)
  - IberoBench `arc_ca_*` (auto_probe)
  - `french_bench_arc_challenge` (auto_probe)
  - new: OALL `arabic_mt_arc_*`
- **HellaSwag:**
  - Okapi m_hellaswag `hellaswag_<l>`, 30 languages (auto). This is the only multilingual translated HellaSwag in the pin.
  - new: `french_bench_hellaswag`, `arabic_mt_hellaswag`, and the native `kobest_hellaswag`
- **MMLU:**
  - `mmlu` en (auto)
  - Global-MMLU `global_mmlu_full`, 37 languages (auto)
  - new: OALL `arabic_mt_mmlu`, `arabic_leaderboard_arabic_mmlu`, ArabicMMLU
  - new (probe): m_mmlu, MMMLU
- **TruthfulQA:**
  - `truthfulqa_mc2` en/vi/zh (auto)
  - `truthfulqa-multi_mc1` en/es (auto)
  - new: the rest of the Okapi set, multi mc2, and NorTruthfulQA
- **PIQA/COPA/Winograd:** `global_piqa`, `xcopa` and `xwinograd` are in auto. New: `copa_es/ca`, `kobest_copa`, `copal_id`, `arabic_mt_piqa/copa`.

**Is MMLU evaluated for every model?**
- `mmlu` and `rf_mmlu` are in `auto` (en), and every cell trains en.
- `ladder_report.csv` (2026-10-07) has 12/12 grid checkpoints for 190 of 201 cells.
- **Gaps that are still in progress:** the swiglu cells (1.7B-L8/L15/L30, 1B-L8, 600M-L8/L15, 350M-L8, 175M-L30) and 3B-L30-deep (11/12).
- **Off-ladder:** 90M-L1-b84-swiglu.
- **One real gap:** lm-350M-L1-deep-seed1904 has `mmlu` at 2/12 while `global_mmlu` is at 12/12.
