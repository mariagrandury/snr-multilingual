# Open questions (night of 10-06 → 10-07)

Defaults in **bold** are what is implemented now; change any of them.

## Website: anonymity
1. Plain words that identify us: CSCS / Clariden / Alps, the pool `custom_swissai_hf`, run_apertus*.py file names, "her scratch". **Redacted in the review build by the hook (CSCS → our cluster, etc.); file names kept.** Hide the Docs pages during review instead?
2. Links to the public Apertus models (huggingface.co/swiss-ai/...): **replaced by the notice.** Keep them?
3. Repeated "(Link momentarily removed for double blind review)" after every repo file in READMEs: **kept as is.** Prefer a short marker with a footnote?
4. 15 figures in the custom_swissai_hf and all/external scopes are **withheld from the site**; regenerate them without the internal lines (apertus3-a06, distillation)?
5. The Slidev deck (/slides/) still says EPFL, Swiss-AI and aromanou: **not anonymized.** Drop it from the review build?

## Website: content
6. configs/languages.json: hello, thank you, speakers, family, script for the 50 languages were **drafted from general knowledge** — please check.
7. Data-mixture scheme dropdown: **A / B / ZH / ES** (AT3 = A at T = 3), plus extra dropdowns for the second language and English data. OK?
8. Mixture token counts: **shares × the 1.7B budget.** Want a size dropdown instead?
9. Benchmark cards: the examples live in configs/tasks.json (358 KB → 1.3 MB). **Kept there**; move them to configs/task_examples.json?
10. 11 benchmarks have no citation entry in tasks.json (blend_sample, commonsense_qa, cultural_bench_easy/hard, global_piqa_*, mathqa, mmlu, openbookqa, toxigen, truthfulqa_mc2): fill them from multilingual_benchmarks.csv?
11. INCLUDE / INCLUDE v2: **bylines removed** (Romanou is a co-author). Is it OK to show INCLUDE v2 items, since the dataset looks unreleased?
12. Public-model floors: **internal lines left out of the hover list**, but the colours still come from a CSV that includes them; regenerate without them? Keep Apertus 8B and 70B?
13. Recommender: reformulated (rf/rfgm) rows **kept**; budgets below the smallest one we measured show **"?"**; thresholds **pool every seed-1904 run** of a size (vs the deep/A grid only).

## Analysis
14. README rule 6 (a GitHub link after every figure) vs anonymity: **the READMEs keep the links; the website hook replaces them.** OK?
15. check_rules scan of docs/, mkdocs.yml and documents/paper/ for identities: **not added yet** (it would fail until the paper's `% generator:` comments are cleaned). Add it now?
16. Refresh speedup: is a ~1.2 GB report cache per argument set on iopsstor OK?
