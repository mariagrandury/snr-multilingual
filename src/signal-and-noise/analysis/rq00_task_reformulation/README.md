# Task reformulation for the auto evals

Status 2026-09-18. Companion to [plan/benchmark_selection.md](../../../../plan/benchmark_selection.md)
(what we evaluate on) and [rq00_gate_and_curves](../rq00_gate_and_curves/) (the chance gate). The
reference implementation is [lighteval_reference.py](lighteval_reference.py)
(HuggingFace's FineWeb-edu ablation tasks) — it is a *reference*: everything
here runs inside the lm-eval harness we already use.

## Why

Three quarters of the auto suite is at chance at these sizes: 113 of 461
gated tasks clear chance at any size, and only 21 of 252 four-option tasks;
`global_mmlu_full`, `global_piqa_*` and `truthfulqa-multi_mc1` never do.
Base models at 90M–1.7B do not pick up the "answer with a letter" convention,
so a lettered multiple-choice prompt measures the convention, not the
knowledge. The lighteval file's fix for MMLU: no letters, no few-shot, the
full answer string scored as the continuation after `Answer:`, `acc_norm`
(length-normalised) as the metric.

## What the harness already does per family

467 auto tasks, all zero-shot (`NUM_FEWSHOT` is never set), 1.19M questions.
Counts read from the offline cache (`$HF_HOME/datasets/*/dataset_info.json`).

| family | tasks | prompt format in the harness | questions | tok/q | Mtok |
|---|---:|---|---:|---:|---:|
| belebele | 105 | **letters A–D** (passage + question + 4 options) | 94,500 | 300 | 28.4 |
| global_mmlu_full | 37 | **letters A–D** (57 subject subtasks over one 14,042-row split) | 519,554 | 190 | 98.7 |
| include_base_44 | 43 | **letters A–D** | 22,100 | 115 | 2.5 |
| arc | 33 | cloze (answer strings) | 39,200 | 120 | 4.7 |
| hellaswag | 31 | cloze (4 endings) | 280,000 | 400 | 112 |
| global_piqa | 95 | cloze (2/4 solutions) | 9,773 | 185 | 1.8 |
| xnli | 18 | cloze, templated `…, right? Yes/Also/No, …` | 49,859 | 70 | 3.5 |
| paws | 10 | cloze, templated `…, right? Yes/No, …` | 20,000 | 75 | 1.5 |
| xstorycloze | 13 | cloze (2 endings) | 19,643 | 140 | 2.75 |
| xcopa | 11 | cloze (2 clauses) | 5,500 | 40 | 0.22 |
| truthfulqa-multi_mc1 | 3 | cloze (mc1 strings) | 2,451 | 100 | 0.25 |
| xwinograd | 6 | two contexts, shared continuation | 4,449 | 40 | 0.18 |
| multiblimp | 57 | minimal pair, no context | 94,450 | 140 | 13.2 |
| lambada_openai_mt | 5 | last-word log-likelihood | 25,765 | 100 | 2.6 |

Only the first three families are letter-format; that is the whole scope of
the programmatic reformulation. arc and hellaswag are already cloze and still
mostly at chance, so a prompt change cannot help them — only a rewrite of the
items themselves (below) can. Two asides the survey turned up: `xnli` for the
15 `facebook/xnli` languages is scored on the 2,490-row *validation* split
(the common YAML sets no `test_split`) while `xnli_eu`/`xnli_gl` use their
5,010-row test split, and 10 of the 57 `multiblimp` tasks have under 200
items (`guj` 7, `ben` 21, `gle` 28).

## Tier 1 — programmatic reformulation (implemented)

[`src/evals/scripts/make_rf_tasks.py`](../../../evals/scripts/make_rf_tasks.py) writes one self-contained harness YAML
per original task under `src/evals/tasks/rf/<family>/rf_<task>.yaml` — same
`dataset_path`, `dataset_name` and split, read from the pinned harness
checkout — with:

| family | `doc_to_text` | `doc_to_choice` |
|---|---|---|
| belebele | `{passage}\nQuestion: {question}\nAnswer:` | the four `mc_answer*` strings |
| global_mmlu_full | `The following are questions about {subject}.\nQuestion: {question}\nAnswer:` | `option_a..d` |
| include_base_44 | `Question: {question}\nAnswer:` | `option_a..d` |

`output_type: multiple_choice`, `num_fewshot: 0`, metrics `acc` and
`acc_norm`. The global_mmlu and include twins are one task per language over
the whole split rather than a group of subject/domain subtasks — the report
aggregates to the family anyway, and a 57-way group is what made the
originals slow.

Plumbing, all keyed on the new task names so originals and twins coexist:

- `configs/tasks.json`: one entry per twin (`language`, `benchmark:
  rf_<family>`, `n_options: 4`, `metric: acc_norm`), group `auto_rf`,
  `benchmarks` metadata. The `rf_` **prefix** is deliberate:
  `tasks_for_benchmarks` matches `<benchmark>_…`, so a `_rf` suffix would be
  selected by the original family. Registered entries are what keep the
  analysis from folding `rf_belebele_eng_Latn` into `belebele` (`results_io.
  aggregate_parents`, `analysis/utils._is_language_aggregate`) and what gives
  `ladder_report` its `chance` column.
- `evaluate.sbatch`: `HARNESS_INCLUDE_PATH` → `eval_worker.py --include_path`
  (already accepted, never set before). The pinned wheel is untouched.
- `auto_evals_cscs.py --reformulated`: the `auto_rf` group instead of `auto`,
  jobs named `eval-<cell>-iter<N>-rf` (the original and the rf set of one
  checkpoint are different work — neither watcher may read the other's job
  as its own); the include path is always exported. Per-task idempotency, walltime sizing,
  hold-backs and the W&B push need nothing — the metric override makes the
  series `rf_<task>/acc_norm`.
- Per cell: L1 2 · L2 5 · L8 24 · L15 49 · L30 84 · L50 124 rf tasks (185 in
  every language).

### First evals to launch

The question is whether the reformulation moves scores off the chance line
at all, and whether it opens a gap between sizes. Answer it on final
checkpoints before touching the grid:

1. **Two checkpoints, one job each**: the strongest model and a small one at
   the same setting — `lm-1.7B-L8-deep-seed1904` and `lm-175M-L8-deep-seed1904`,
   final iter only (`--every 1000` leaves just the final save due). Both are
   ALL_LANGUAGES runs, so every rf task runs (185) and the comparison covers
   trained and untrained languages. Read: per family and language, rf
   `acc_norm` vs the original `acc` on the same checkpoint (both on disk),
   against chance 0.25 + 0.05. If the 1.7B stays at chance, stop: the
   families need Tier 2, not a prompt change.
2. **The L8 size ladder finals** (90M … 1.7B, 6 jobs): does rf order the
   sizes, and from which size does it clear the gate?
3. **The deep-A-seed1904 ladder** at the normal `--every 2` cadence, under
   `--watch`, throttled with `--max-submit`.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
# 1. dry-run, then two jobs
python3.11 auto_evals_cscs.py --reformulated --name lm-1.7B-L8-deep-seed1904 --every 1000 --dry-run
python3.11 auto_evals_cscs.py --reformulated --name lm-1.7B-L8-deep-seed1904 --every 1000
python3.11 auto_evals_cscs.py --reformulated --name lm-175M-L8-deep-seed1904 --every 1000
# 2. the L8 finals of every size
for s in 90M 175M 350M 600M 1B 1.7B; do
  python3.11 auto_evals_cscs.py --reformulated --name lm-$s-L8-deep-seed1904 --every 1000
done
# 3. the whole seed-1904 deep ladder, every 2nd checkpoint
python3.11 auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904 --max-submit 20 --watch 1800
```

The walltime estimate is fitted on the original task mix; the rf
global_mmlu tasks score four answer strings per question over 14k questions,
so the first jobs may hit their limit. That costs only the tasks in flight:
the next pass resubmits what is missing, sized to it.

### Which metric to compare

The originals ask for a letter, so every option is one token of the same
length (" A" … " D") and byte-length normalisation cannot change the ranking:
their `acc` and `acc_norm` are identical, and comparing the rf `acc_norm`
with the original `acc` is acc_norm against acc_norm. The rf tasks are the
only place the two differ — the answers have different lengths — and
`acc_norm` is the standard choice for cloze scoring (the lighteval reference
scores `loglikelihood_acc_norm`). It is the metric the pipeline carries for
the rf tasks: tasks.json's per-task `metric`, honoured by `results_io.flatten`
(W&B) and `ladder_report._primary` (the report), so both show the same number.
`compare.py` reads that `primary_score` from the ladder report through the
rq00 gate (`above_random.scores_and_mask` on the `predictivity` pool, deep
scheme-A cells): so the table below is `original acc → rf acc_norm`, and it
moves only after `ladder_report.py --plot --publish --push-hf` has picked up
new rf results and `scripts/refresh_analysis.sh` has re-run. Until then `compare.py` refuses to run against a report without the `rf_*` columns (it would empty the table); point `SNR_LADDER_DIR` at a report that has them. The conclusion
does not depend on the metric — in the pilot at 1.7B in the trained
languages belebele gains +0.14 on acc_norm and +0.16 on acc, Global-MMLU
+0.08 and +0.06 (the per-task `acc` stays in the eval logs and W&B).

### Pilot results (2026-09-18, final checkpoints, all 185 rf tasks)

| family | 175M rf `acc_norm` | 175M orig `acc` | 1.7B rf `acc_norm` | 1.7B orig `acc` | tasks > 0.30 at 1.7B (rf / orig) |
|---|---:|---:|---:|---:|---:|
| belebele (105) | 0.277 | 0.232 | 0.303 | 0.238 | 46 / 0 |
| global_mmlu_full (37) | 0.255 | 0.245 | 0.274 | 0.247 | 6 / 0 |
| include_base_44 (43) | 0.259 | 0.259 | 0.279 | 0.249 | 11 / 0 |

The reformulation moves the letter families off the chance line at 1.7B and
opens the size gap the originals never showed: belebele 0.30 vs 0.24, Global-
MMLU in the trained languages en 0.37 · de/es/fr 0.32 · it 0.31 · ja 0.30 · zh
0.32 (untrained languages stay at 0.25), INCLUDE fr 0.46 · es/it 0.38 · jp 0.35
· de/bg 0.30. At 175M only belebele moves (+0.045); the knowledge families
stay at chance, as they should for a 175M model. Cost: the rf Global-MMLU
tasks take 2.7–3.3 min per worker-task (one 14k-row task per language) against
0.1–0.3 for belebele and INCLUDE, so the watcher weights them ×6 in the
walltime. Two data faults surfaced and are fixed: 37 Global-MMLU rows and one
INCLUDE row with a `None`/blank option (dropped by `process_docs`), and the
cross-user `datasets` lock files in the shared cache (see
`src/evals/CLAUDE.md`, "lock files"). Steps 2 and 3 were launched the same
day (`eval-<cell>-iter<N>-rf` jobs).

## Tier 2 — rewriting the items with Gemini (not built; costed)

For families that are cloze already and still at chance (arc, hellaswag,
global_piqa) and for the three letter families after Tier 1, the remaining
lever is the items themselves: shorter contexts, plainer wording, answers
that read as natural continuations. That is a rewritten dataset per task
(a JSONL per task, `dataset_path: json` in a second set of YAMLs, same
`--include_path` route), produced once with an LLM and kept alongside the
originals — never pushed publicly with gold labels.

Cost model: per item, input = the item (question + all options) + ~150
tokens of instruction and JSON scaffolding; output = the rewritten item,
about the item's own length. Token counts are bytes/3.5 estimates from the
cache (±30 %, CJK/Indic scripts heavier). Prices per 1M tokens, September
2026 ([1](https://developer.puter.com/tutorials/gemini-api-pricing/),
[2](https://www.cloudzero.com/blog/gemini-pricing/),
[3](https://benchlm.ai/google/api-pricing)): Gemini 3.1 Flash-Lite
$0.25 in / $1.50 out, Gemini 3 Flash $0.50 / $3.00, Gemini 3 Pro
$2 / $12; the Batch API halves all three.

| scope | questions | in Mtok | out Mtok | Flash-Lite | Flash | Pro |
|---|---:|---:|---:|---:|---:|---:|
| all four-option MC families (arc, belebele, global_mmlu_full, hellaswag, include_base_44) | 955k | 389 | 246 | $466 (batch $233) | $933 ($466) | $3,730 ($1,865) |
| every auto benchmark (14 families) | 1,192k | 450 | 272 | $521 ($260) | $1,041 ($520) | $4,164 ($2,082) |

Caveats: Gemini 3 bills thinking tokens as output — set the thinking budget
to minimal or the output side grows 1.5–2×; global_mmlu and hellaswag are 77 %
of the tokens, and rewriting only the question and options while keeping
belebele passages and hellaswag contexts verbatim cuts the output by about
40 %; multiblimp, lambada and xwinograd are structure-bound (minimal pairs,
last word, coreference) and cannot be rewritten without changing what they
measure, so the "every benchmark" row is an upper bound; a 1k-item pilot per
family (≈ $1) should precede any full run.

<!-- BEGIN auto:rf-compare (analysis/rq00_task_reformulation/compare.py) -->
Gate cells (median task margin over chance 0.25, trained languages, deep scheme-A seed-1904 ladder, from the ladder report). Cell: original acc → rf acc_norm, **Δ** = rf − original; n = rf tasks.

| family | 175M | 350M | 600M | 1B | 1.7B |
|---|---:|---:|---:|---:|---:|
| belebele | -0.007 → +0.030, **+0.037**, n=59 | -0.013 → +0.057, **+0.070**, n=59 | -0.007 → +0.092, **+0.099**, n=59 | — | +0.011 → +0.150, **+0.139**, n=59 |
| global_mmlu_full | -0.006 → +0.006, **+0.012**, n=29 | -0.000 → +0.019, **+0.019**, n=29 | -0.004 → +0.028, **+0.032**, n=29 | — | -0.010 → +0.066, **+0.076**, n=29 |
| include_base_44 | +0.004 → +0.005, **+0.001**, n=36 | — | — | — | +0.005 → +0.124, **+0.119**, n=36 |

![family x size](rf_gate.png)

![per language](rf_gate_by_language.png)
<!-- END auto:rf-compare -->
