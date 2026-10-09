# PIQA / WinoGrande / BoolQ into `auto`, and the probe promotion (2026-10-09)

Status: **landed 2026-10-09.** The tasks.json edit below is in the live `configs/tasks.json` (its separate
`majority` keys untouched), the BoolQ line in `src/evals/configs/eval_datasets.txt`, and the pretrain README
pointer went in with `ddc04add`. The sections below are the proposal as it was approved; the cost and the
3B scope are the estimates made before landing.

## Decisions taken (2026-10-09, final selection) and what is still open

Everything ships as **one in-place tasks.json edit**, so every due checkpoint gets one
top-up job carrying the new tasks together with the all-languages backlog left by
the 03:40 cancellation of every eval job.

| | decision |
|---|---|
| D0 | Part A ships together with the promotions, one edit, one job per checkpoint. |
| D1 | **Keep** the letter originals in `auto` (mmlu, commonsense_qa, belebele, global_mmlu_full, include_base_44, bbh_mcq, acp_bench_mcq, cultural_bench_easy): rq00 compares each with its twin, and at 3B 687 of 841 trained tasks pass (522 at 1.7B), mostly those originals. Discarding is for probe candidates only. |
| D2 | BoolQ chance **0.5**, like every binary task. No majority level in the gate (the per-task `majority` key, added separately for a sensitivity appendix, is preserved by the edit). |
| D3 | `m_truthfulqa_mc1`: **not adopted**. It stays in `auto_probe`; its 5 tasks failing at both sizes are discarded. |
| D4/D5 | Promote **rf_m_mmlu** (27 languages), **rf_mmmlu** (12), **multiblimp-extra**, **noreval** (+`noreval_extra` by prefix), **kobest** (its 2 failing tasks discarded), **toksuite** (+`_math`, `_stem`). Every other borderline benchmark stays out. |
| Discards | The **129** probe tasks failing the gate at 1B and at 1.7B (trained runs) go to `groups.discarded` (100 → 229). |
| D6 | Verdict **accepted without re-probing** `lm-1.7B-L8-swiglu-seed1904` (that needs a probe job); its promoted tasks are evaluated anyway once in `auto`. |
| 3B scope | Every considered task NOT in `auto` (the `auto_probe` remainder + `discarded`) is evaluated on the FINAL checkpoint of each trained 3B cell (below). |

**The edit** (`snr-probe-gate/promotion-2026-10-09/tasks.json.final.diff`, against the live
file with its 1,664 `majority` keys, none touched; +147/−3 lines):
`auto` 40 → 49 (+ boolq, kobest, multiblimp-extra, noreval, piqa, rf_m_mmlu, rf_mmmlu,
toksuite, winogrande); piqa `metric: acc_norm`; `n_items` + `source` on the three Part A
tasks; +129 `discarded`. `scripts/build_final.py` applies it **in place** to the live file
(never a whole-file copy, so keys added meanwhile survive); `--land` reads, edits and writes
in one process through `write_tasks_json`, which refuses if the file changed in between.

**BoolQ is cached.** `aps/super_glue@3de24cf8` built on the login node with
`download_eval_datasets.py` into `hf_home/datasets/aps___super_glue/` (all 11 configs);
`boolq` loads with `HF_DATASETS_OFFLINE=1` (3270 items, 2033 yes); its 11 lock files were
created and opened with `setfacl -m m::rwx`, as `share_dataset_locks` does.

### Cost of the first watcher pass after landing

Counted with the watcher's own functions on the proposed tasks.json (`scripts/cost_pass.py`):
due checkpoints (`saved_valid_iters` + `due_iters`), each cell's eval languages
(`all_languages_run` included), what is left on disk (`remaining_tasks`), priced with
`eval_minutes` and `submit_eval`'s weights (6× for the MMLU-sized tasks). No eval job is in
the queue (all cancelled 03:40), so nothing is deduplicated against running work.

| category | task-evals | compute node-h (MIN_PER_TASK, 4 workers) |
|---|---:|---:|
| new `auto` tasks (Part A + promotions, every eval language of the cell) | 133,589 | 991 |
| all-languages backlog (old `auto` tasks in untrained languages of the all-languages runs: ZH/ES L2 and schemeB L8, 15 cells, 163 checkpoints) | 116,737 | 341 |
| routine (old `auto`, trained languages, still missing; 1.7B) | 5,527 | 18 |
| **total** | **255,853** | **≈ 1,350** |

- **Jobs:** 2,548 due checkpoints over 214 cells = 2,548 jobs, none over the 12 h cap. 3 checkpoints (1.7B) need a convert first.
- **Requested:** 2,104 node-h. Expect roughly 1,450–1,700 node-h burned once cold start is added.
- **Calendar:** at yesterday's peak rate (69 eval jobs, ~50 node-h in one hour) that is about 1.5–2 days, and 3–4 days if the queue is contended. `rf_m_mmlu` + `rf_mmmlu` are most of the new-task compute.

### 3B scope: the probe remainder on the 3B finals

`scripts/probe3b.py` builds the list itself: `discarded` tasks are skipped by
`tasks_for_benchmarks`, so `auto_evals_cscs.py --group auto_probe` cannot reach them. It hands
the list to the watcher's `submit_eval` (same sbatch, per-task idempotent, job suffix `-probe3b`).
It is a dry-run by default; `--submit` submits.

| 3B cell (final 145200, converted) | considered tasks (trained languages) | requested wall |
|---|---:|---:|
| lm-3B-L8-deep-seed1904 | 257 | 1:30 |
| lm-3B-L15-deep-seed1904 | 389 | 2:15 |
| lm-3B-L30-deep-seed1904 | 514 | 3:00 |
| lm-3B-L50-deep-seed1904 | 550 | 3:20 |
| lm-3B-L8-schemeB-deep-seed1904 | 188 | 1:05 |
| lm-3B-L15-schemeB-deep-seed1904 | 259 | 1:30 |
| **6 jobs** | **2,157** | **12.7 node-h** |

`lm-3B-L30-schemeB-deep-seed1904` was removed from the grid before it trained (`ddc04add`). The verdict then comes from `probe.sh`
(`PROBE_GATE` → `SNR_GATE_DIR`, so the canonical gate tables are not overwritten), which now gates with
`above_random.py --above-reference`: the probe gate carries a 3B column, read by rq00's "all considered"
variant at `utils.PROBE_SIZES` (600M, 1B, 1.7B, 3B; `discarded` stays defined at 1B and 1.7B).

### Still open

- The borderline set stays out: kmmlu, the BLiMPs, ibero_*, French, Arabic, EVALITA, headqa/careqa, copal_id.
- `m_truthfulqa_mc1` is not adopted.
- Label follow-ups (`labels_followup.diff`: `PAPER_NAMES`, regimes `SHORT`, appendix `BENCH_META`, `N_OPTIONS`) are not landed, because those live files carry other uncommitted edits. `BENCH_META` also lacks rows for the six promoted probe benchmarks.

### To land (your go)

1. `bash /iopsstor/scratch/cscs/mariagrandury/snr-probe-gate/promotion-2026-10-09/land.sh`. It does two things:
   - `build_final.py --land`, the in-place edit of the live tasks.json;
   - `git apply support.diff`: the BoolQ manifest line and the pretrain README pointer.
2. Restart the watcher: `bash scripts/nightly.sh evals`, or wait for 00:00. **This is the launch.**
3. Run `python3.11 …/scripts/probe3b.py`, then `python3.11 …/scripts/probe3b.py --submit` for the 6 3B jobs.
4. When those land, run `bash src/signal-and-noise/analysis/rq00_task_reformulation/probe.sh`.

Sources: the probe gate written by `probe.sh` on 2026-10-08 18:04
(`/iopsstor/scratch/cscs/mariagrandury/snr-probe-gate/pretraining/predictivity/`,
seed-1904 `predictivity` pool, probe counted as trained), the canonical gate
(`src/signal-and-noise/analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_*.csv`,
18:08), `configs/tasks.json` at `2a3493ae`, and the plan notes listed at the end.

---

## Part A: PIQA, WinoGrande and BoolQ

### Status of the three tasks

| task (lm_eval name) | harness YAML (pinned checkout `51d6f4b6`) | dataset | in tasks.json | groups today | probed or discarded? |
|---|---|---|---|---|---|
| `piqa` | `lm_eval/tasks/piqa/piqa.yaml`, emits acc and acc_norm | `baber/piqa`, validation, 1838 items, labels 910/928 | yes: `benchmark: piqa`, `n_options: 2`, no `n_items`, no `metric` | `pretraining_full`, `pretraining_classic`, `allenai` | never: no msnr result, no row in either gate |
| `winogrande` | `lm_eval/tasks/winogrande/default.yaml` (`winogrande_xl`, partial scoring), emits acc only | `allenai/winogrande`, validation, 1267 items, answers 628/639 | yes: `benchmark: winogrande`, `n_options: 2` | `pretraining_full`, `pretraining_classic`, `allenai` | never |
| `boolq` | `lm_eval/tasks/super_glue/boolq/default.yaml`, emits acc only, choices `["no", "yes"]` | `aps/super_glue` config `boolq`, validation, 3270 items, **2033 yes / 1237 no** | yes: `benchmark: boolq`, `n_options: 2` | `allenai` | never |

None of the three is in `auto`, `auto_probe` or `discarded`. The benchmark
prefixes don't collide: no other task's benchmark is `piqa_…`, `winogrande_…`
or `boolq_…`. The Catalan, Basque and Bangla PIQA translations are
`ibero_piqa` and `bangla`, so adding `piqa` selects exactly one task. The
analysis side already handles them. `_is_parent_task`, `assign_language`
(`en`) and `benchmark_family` return the right thing for all three
(checked), and `above_random.N_OPTIONS` already lists piqa and winogrande.

The only earlier mentions say "evaluate them on the ladder":
`rq07_external_frameworks/README.md:133-156` (4 of the 10 DataDecide tasks
missing from the shared set) and `rq13_english_only/README.md:513` (a
follow-up). No note records a reason for leaving them out. The probe
instructions of 09-21 and 09-30 were about non-English candidates.

### Datasets (offline cache, `HF_HOME=/iopsstor/scratch/cscs/mariagrandury/hf_home`)

- **PIQA: cached.** `datasets/baber___piqa/default/0.0.0/142f6d73…`. Loads with `HF_DATASETS_OFFLINE=1`.
- **WinoGrande: cached.** `datasets/allenai___winogrande/winogrande_xl/0.0.0/01e74176…`. Loads offline.
- **BoolQ: was NOT cached (built 2026-10-09, see "Decisions taken").** `aps/super_glue` is missing from `hf_home/datasets`, from `HF_HUB_CACHE` and from
  `src/evals/configs/eval_datasets.txt`. On the Hub it is a parquet repo with no loading script (tip
  `3de24cf8022e94f4ee4b9d55a6f539891524d646`), so `load_dataset` builds it with datasets v3+.
  Since bug 13 a missing dataset fails only that task, not the job. Even so, the watcher would hold `boolq`
  on every checkpoint for three attempts before `fix_missing_dataset` repairs the cache. **Stage it first,
  on the login node:** add the pinned line to `eval_datasets.txt` (in the diff below) and run
  `python3.11 src/evals/scripts/download_eval_datasets.py <manifest with that one line>`. This builds all
  super_glue configs, including ReCoRD, which is the large one.
- The three tasks were not loaded through a real `TaskManager`: `lm_eval` is not in the `snr` env. They are
  stock harness YAMLs, and the datasets were loaded directly. A one-task smoke eval on one checkpoint would
  close this gap. It is a job, so it is yours to launch.

### Metric, chance level, overrides

| task | metric to pin | why | chance in the gate today | proposal |
|---|---|---|---|---|
| piqa | **`metric: acc_norm`** | It is a cloze task, and the 2026-10-08 audit rule (`a5ce9ff8`, `src/evals/CLAUDE.md` "Set `metric` only where `acc` is wrong") pins acc_norm for cloze families that emit it, as for ARC, HellaSwag, OBQA and Global-PIQA. | 0.5 (`n_options` 2) | 0.5. The labels are balanced (49.5 / 50.5 %). |
| winogrande | none (acc) | It emits no acc_norm. Partial scoring compares one continuation under two contexts, so length normalisation does not apply. The audit's "families that emit no acc_norm keep acc" rule covers it. | 0.5 | 0.5. Balanced (49.6 / 50.4 %). |
| boolq | none (acc) | It emits acc only, and the two answers are single tokens. | 0.5 | **Decision D2.** The majority class ("yes") is 0.6217. |

These pins are not part of the "77 overrides". Those were the 77 `auto_probe`
tasks whose `acc_norm` pin was wrong (`7b2f356d`: 61 removed, MELA set to mcc,
EVALITA wic set to f1). None of the three needs a non-accuracy metric.
`n_items` is set directly (1838 / 1267 / 3270), so the Wilson bound has a
denominator before `derive_task_options.py` runs. Without `n_items` the gate
drops a task silently.

**BoolQ and the gate (D2).** Rule 1 (`RULES.md:67`, `above_random.py:108-111`)
gates on uniform guessing and says outright that it does not test a constant
answer. For BoolQ that is a hole: a model that always says "yes" scores 0.62,
clears 0.5 with a Wilson lower bound of about 0.61 on 3270 items, and passes
at every size without reading the passage. The two options:

- (a) keep 0.5 and say in rule 1 that BoolQ passes on its prior;
- (b) add `"boolq": 0.6217` to `above_random.CHANCE`, a one-line exception to
  "uniform", documented as the majority-class baseline. That is the number
  every BoolQ paper reports.

I recommend (b). Small models typically sit at 0.55–0.63 on BoolQ, so with (b)
it may well fail the gate at 1B, and that is the honest reading. The worktree
diff does not include (b), because it is your call.

### Expected behaviour (36-model sweep, same harness tasks)

The 36-sweep parquet (`pretraining_custom`, final checkpoint, 9 runs per size)
had piqa and winogrande, but not boolq:

| task | 175M | 350M | 600M | 1B |
|---|---:|---:|---:|---:|
| piqa (acc) | 0.672 | 0.696 | 0.712 | 0.727 |
| winogrande (acc) | 0.522 | 0.541 | 0.563 | 0.590 |

PIQA should clear 0.5 from the smallest rung. WinoGrande is marginal below
350M (n = 1267: the lower bound of 0.522 is about 0.50) and clear from 600M.

### Cost (Part A)

The watcher's dry-run could not be pointed at another tasks.json, so I ran it
from the worktree. A pass takes over an hour (about a minute per cell, 219
cells), so I stopped it and counted the same quantities directly with the
watcher's own functions (`predictivity_cells`, `saved_valid_iters`,
`due_iters`, `eval_languages`) on the worktree code.

- **Due checkpoints on disk now: 2,547** over 219 cells (90M 384, 175M 456, 350M 372, 600M 456, 1B 444,
  1.7B 363, 3B 72). The grid at completion is 2,628 (12 per cell). Every cell trains English, so every due
  checkpoint gets the 3 tasks: **7,641 task-evals**.
- **Task compute:** about **93 GPU-hours, roughly 23 node-hours**, at `MIN_PER_TASK` per worker-task
  (0.39–0.85 min by size). BoolQ is counted as 2 tasks for its 96-word passages (6.5k requests against
  PIQA's 3.7k).
  The measured times put it lower. The "done <task> in Ns" lines in the msnr worker logs agree with
  `total_evaluation_time_seconds`. None of the three English tasks has ever run, so the times come from
  the closest tasks that have:
  - PIQA from piqa_ca and piqa_eu (the same 1838×2 items, translated): 0.15–0.20 min.
  - WinoGrande from xwinograd_en, scaled to 1267 items: 0.10–0.13 min.
  - BoolQ from arabic_mt_boolq (3260 items, 0.40–0.47 min at 600M–1.7B) and kobest_boolq: 0.38–0.58 min.

  Each of those ranges runs from 90M to 3B, and each value is per worker-task. Together that is
  0.63–0.91 worker-min per checkpoint. Over the 2,547 checkpoints it comes to **≈ 30 GPU-h, ≈ 8 node-h of
  task compute**. That is about a third of the `MIN_PER_TASK` figure, because those constants average
  over a mix that includes the 14k-item MMLU tasks.
- **Cold start dominates.** Done alone, this is one top-up job per checkpoint: about **2,547 sbatch jobs**
  in one watcher pass, since the nightly runs with no `--max-submit`. Recent October jobs that ran 1–6
  tasks took a median of **3.5 min** elapsed (p25 3.3, p75 6.6; n = 18 at 600M–1.7B). That makes it about
  **150 node-hours elapsed, up to ~250**. Each job requests 15 min (`eval_walltime`), about 640 node-hours
  of requested wall in the queue.
- **Cheaper:** ship Part A in the same edit as any Part B promotion. Every checkpoint then gets one job that
  carries both, and the 3 tasks add about 1 min per job.

### Proposed diff (worktree, `git -C …-wt-auto-add diff`)

```diff
--- a/configs/tasks.json
+++ b/configs/tasks.json
@@ -3601,7 +3601,10 @@   (tasks.piqa)
-      "n_options": 2
+      "n_options": 2,
+      "metric": "acc_norm",
+      "n_items": 1838,
+      "source": "baber/piqa (validation)"
@@ -3955,7 +3958,9 @@   (tasks.winogrande)
-      "n_options": 2
+      "n_options": 2,
+      "n_items": 1267,
+      "source": "allenai/winogrande (winogrande_xl, validation)"
@@ -11375,7 +11380,9 @@   (tasks.boolq)
-      "n_options": 2
+      "n_options": 2,
+      "n_items": 3270,
+      "source": "aps/super_glue (boolq, validation)"
@@ groups.auto (40 -> 43, kept sorted)
       "blend_sample",
+      "boolq",
       "commonsense_qa",
       ...
       "paws",
+      "piqa",
       "rf_acp_bench_mcq",
       ...
       "truthfulqa_mc2",
+      "winogrande",
       "xcopa",
```

Written with `utils.configs.read_tasks_json` / `write_tasks_json`, so the
rest of the file is byte-identical. The worktree also carries the supporting
edits:

| file | change |
|---|---|
| `src/evals/configs/eval_datasets.txt` | `+aps/super_glue@3de24cf8…` (pinned) |
| `src/signal-and-noise/analysis/rq00_gate_and_curves/above_random.py` | `N_OPTIONS += "boolq": 2` (cosmetic; tasks.json is read first). D2(b) would add `CHANCE["boolq"] = 0.6217` and a rule-1 sentence. |
| `src/signal-and-noise/analysis/grids.py` | `PAPER_NAMES` += PIQA, WinoGrande, BoolQ |
| `src/signal-and-noise/analysis/rq01_scaling_predictability/regimes.py` | `SHORT["piqa"] = "piqa-en"`. Global PIQA already uses the label `piqa`, so the two families would collide in the regime figures. |
| `documents/paper/figures/make_appendix_tables.py` | `BENCH_META` += three rows. The appendix benchmark table is generated from `groups.auto`, so the paper table grows. |
| `src/pretrain/README.md` | The hand-written 14-name `auto` list was already stale (40 today). It is replaced by a pointer to the group. |

Not edited, worth doing on adoption:
- `rq13_english_only/english_regimes.py` `POPULAR` / `TAGS`: the live file has your uncommitted edits, so I left it alone.
- `src/evals/CLAUDE.md` "Task list": a dated line for the promotion.
- `RULES.md`: the promotion paragraph. Promotion is a population change, and rule 2's trained set gains three English tasks in every cell.

Full patch: `snr-probe-gate/promotion-2026-10-09/worktree_full.diff`.

### Order of operations if you approve

1. ~~Login node: stage BoolQ~~ done 2026-10-09.
2. Copy the worktree's `configs/tasks.json` (+ the support files) into the live tree. For a 3-task top-up
   over 2,547 checkpoints, consider a one-shot `auto_evals_cscs.py --max-submit N` pass first so the queue
   is not flooded. The `--watch` watcher only sees the new group after its restart: the nightly at 00:00,
   or a manual restart. That restart is the launch.
3. After the top-ups land: `derive_task_options.py --only-missing`, refresh, and check the gate rows for
   the three tasks.

---

## Part B: the probe, reopened

### Where it stands

- **2026-09-23 (`b30800c5`):** 20 of 21 candidates promoted into `auto`, including the letter originals
  `mmlu` and `commonsense_qa` together with their `rf_` twins. bbq stayed out (23 min per checkpoint) and
  left `auto_probe` on 10-01.
- **2026-09-23 (`0f9781ad`):** `auto_probe` grows to 44 benchmarks: IberoBench, TokSuite and 13
  language leaderboards.
- **2026-10-08 (`2937c65e`):** the 159-task batch plus `m_mmlu` (27) and `mmmlu` (12) with twins.
  "100 old candidates not above chance at 1B nor 1.7B (rule 1 gate, trained runs) move to
  `groups.discarded`"; `auto` unchanged.
- **2026-10-08 (`7b2f356d`, `a5ce9ff8`, `14c343e2`, `28f136b5`):** metric overrides fixed; probe chance
  levels fixed (mc2 0.449, m_truthfulqa_mc1 0.227, arabicmmlu 0.2935); probe gate moved out of the repo
  (`SNR_GATE_DIR`).
- **Your 09-30 instruction (`plan/todos/status-09-30.md:482-489`):** "Check which pass the above chance
  threshold at least at 1B and 1.7B. Add those to auto." Not done yet for the 10-08 batch.

**How complete the probe is.** Against the seed-1904 cells the gate reads,
the probe is complete except for **`lm-1.7B-L8-swiglu-seed1904`**, which
finished after the probe ran. The 1B replicate seeds 28/1797 have no probe
evals, but the gate does not read them. The gaps that remain are task-level
failures, not missing jobs:

- `evalita_llm_extra`: the hs/sa prompts never produced a score, 102 of 378 expected runs at 1B.
- `ibero_nli`: the f1 tasks, 8 of 26.
- `mela` (mcc) and `xquad` (exact_match) have no chance level and are never gated.

No note marks the probe evals as "on hold". If they are paused, the only
verdict that could move is that of the swiglu cell.

Per-task table (930 probe tasks; mask, score, chance, run counts at 1B and
1.7B): `snr-probe-gate/promotion-2026-10-09/probe_tasks_1b_17b.csv`. Per benchmark:
`probe_benchmarks_1b_17b.csv`.

### Candidates not yet in `auto` (546 tasks, 50 benchmarks)

"Both" means the task passes the gate at 1B and at 1.7B (rule 1, trained
runs). Scores are means over the family's tasks at 1B → 1.7B.

#### Promote (pass at both sizes, and add language or format coverage)

| benchmark | languages | passes 1B / 1.7B / both | metric | score 1B→1.7B (chance) | twin / original in auto? | reason |
|---|---|---|---|---|---|---|
| `rf_m_mmlu` | 27 Okapi languages | 24 / 27 / 24 of 27 | acc_norm | 0.276→0.290 (0.25) | `rf_global_mmlu_full` is in auto; the `m_mmlu` original fails (D4) | MMLU-style knowledge in 9 languages Global-MMLU lacks (ca, da, hr, hu, ml, mr, nb, sk, ta); no, mr, ro, sk and ta have ≤ 5 passing families today |
| `multiblimp-extra` | hr, lv, sq | 3 / 3 / 3 | acc_norm | 0.909→0.923 (0.5) | MultiBLiMP family in auto | lv has 1 passing family in auto; cheap. It is near ceiling, so expect little DA. |
| `noreval` (+ `noreval_extra`, pulled in by the prefix) | no | 5+28 / 7+30 / 5+28 | acc / acc_norm | 0.42 (0.32) | — | Norwegian has 3 passing families in auto |
| `kobest` | ko | 2 / 3 / 2 of 5 | acc / acc_norm | 0.516→0.539 (0.45) | — | ko has 5 families; the two failing tasks (`kobest_boolq`, `kobest_wic`) are discarded |
| `toksuite` (+ `_math`, `_stem`, pulled in by the prefix) | en, fa, it, tr, zh | 7 / 7 / 7 | acc_norm | 0.705→0.786 (0.25) | — | A new format (tokenizer-perturbed items), a wide and growing margin, adds fa |

#### Borderline (pass, but redundant, tiny margin, saturated, or too costly for what they add)

| benchmark | languages | passes 1B / 1.7B / both | reason |
|---|---|---|---|
| `m_truthfulqa_mc1` | 27 | 17 / 19 / 15 of 27 | It passes on +0.03 over E[1/n] = 0.227 (0.260→0.259, no growth). The English original `truthfulqa_mc1` fails both, which points to translation artifacts rather than truthfulness. It is cheap (817 items). See D3. |
| `rf_mmmlu` | 12 | 11 / 12 / 11 | All 12 languages are already covered by `rf_global_mmlu_full`; it would be a third MMLU translation at the 6× task weight (≈ 300 node-h) |
| `kmmlu` | ko | 26 / 23 / 23 of 45 | Letter format at 0.254 against 0.25: it passes on n, not margin |
| `blimp_nl`, `zhoblimp`, `turblimp` | nl, zh, tr | 75 / 107 / 13 pass both | Saturated (0.82–0.85, flat from 1B to 1.7B), 197 subtasks, ≈ 860 node-h; the languages are well covered (≥ 7 families) |
| `ibero_*` (arc, cola, copa, nli, openbookqa, paraphrase, piqa, siqa, truthfulqa) | ca, es (eu, gl untrained) | 17 pass both | ca has 7 and es 14 passing families already; ≈ 46 node-h for ca only |
| `french_bench` (+ `_extra` by prefix) | fr | 4+1 / 4+1 | fr has 12 families; 3 of `_extra` fail |
| `oall_mt`, `oall_alghafa`, `rf_arabicmmlu`, `rf_oall_arabic_mmlu`, `rf_oall_exams` | ar | 7+2+1+1+1 pass both | ar already has 59 tasks in 9 passing families; ≈ 112 node-h |
| `evalita_llm` | it | 6 / 6 / 6 of 16 | The prefix pulls in `evalita_llm_extra` (0/18 at 1B); it has 11 families |
| `headqa`, `rf_careqa` | es | 1 / 1 each | es is the best-covered language after en |
| `m_truthfulqa_mc2` | ml, ta | 2 / 2 / 2 of 24 | 22 of 24 fail; the 2 passers sit at 0.43 against 0.449 in the other languages, so they are more likely noise than signal |
| `copal_id`, `evalita_llm_extra` | id, it | pass at 1.7B only (2/2, 2/18) | They don't meet "both sizes" |

#### Discard (fail at both 1B and 1.7B on trained runs → `groups.discarded`, 129 tasks)

| benchmark | tasks failing both | note |
|---|---|---|
| `m_mmlu` (letter original) | 26 of 27 | `m_mmlu_ne` passes at 1.7B only. Its `rf_` twin is the promote candidate. |
| `mmmlu` (letter original) | 12 of 12 | It carries `metric: acc_norm` although it is lettered, an inconsistency with the audit rule. Moot once discarded. |
| `arabicmmlu`, `oall_arabic_mmlu`, `oall_exams`, `careqa` (letter originals) | 1 each | Their twins pass |
| `m_truthfulqa_mc2` | 22 | |
| `truthfulqa-multi_mc2` | 3 | |
| `truthfulqa_mc1` (en) | 1 | |
| `kmmlu` | 19 | |
| `noreval_extra` | 10 | |
| `oall_alghafa` | 5 | |
| `oall_mt` | 4 | |
| `evalita_llm` | 4 | |
| `evalita_llm_extra` | 4 | |
| `french_bench` | 1 | |
| `french_bench_extra` | 3 | |
| `rf_french_bench_extra` | 1 | |
| `oall_acva` | 1 | |
| `kobest` | 2 | |
| `bangla` | 1 | |
| `ceval` | 1 | |
| `m_truthfulqa_mc1` | 5 | es, zh, fr, sk, sv (`truthfulqa_eu_mc1` has no trained run, so it is not discarded) |

Not discarded:
- `truthfulqa_eu_mc1`, `truthfulqa_gl_mc1` and `arc_eu_challenge` have no trained run (eu and gl are not
  trained). Their verdict comes from the untrained population, which the 10-08 discard rule does not use.
- `mela`, `xquad` and the f1 tasks have no chance level, so the gate gives no verdict.

#### Cost of promoting the "promote" set

Computed on the 2,547 due checkpoints with the watcher's task weights (the
6× weight for the MMLU-sized `rf_m_mmlu_*`) and `MIN_PER_TASK`; the failing
tasks are discarded. Each cell gets the tasks in its trained languages, and
every language for the seed-1904 all-languages runs.

| set | tasks | task-evals | GPU-h | node-h (compute) |
|---|---:|---:|---:|---:|
| Part A (piqa, winogrande, boolq) | 3 | 7,641 | 93 (measured analogs: ≈ 30) | 23 (measured: ≈ 8), + ~150 cold start if shipped alone |
| `rf_m_mmlu` alone | 27 | 40,272 | 2,217 | 554 |
| `rf_m_mmlu`, only the 9 non-Global-MMLU languages | 9 | 10,335 | 573 | 143 |
| multiblimp-extra / noreval(+extra) / kobest / toksuite(×3) | 3 / 33 / 3 / 7 | 2.9k / 46.4k / 4.7k / 15.0k | 27 / 427 / 44 / 137 | 7 / 107 / 11 / 34 |
| **Promote set, full `rf_m_mmlu`** | 77 | 109,226 | **2,850** | **≈ 713** |
| **Promote set, 9-language `rf_m_mmlu`** | 59 | 79,289 | **1,207** | **≈ 302** |

For scale: `plan/compute-budget.md` puts the whole current `auto` sweep at
about 930 node-h burned. The full promote set nearly doubles that, and
`rf_m_mmlu` is 78 % of it. The node-hours here are compute only. Shipping the
set together with Part A puts everything in one job per checkpoint, and the
cold start is paid once.

### Letter originals already in `auto` that fail the gate (the MMLU question)

Your position: the MMLU originals are letter format, do not pass at 1B or
1.7B, and discarding them is justified. The data agrees.
- **mmlu:** 0.249 → 0.258 against 0.25, mask 0 at every size. The audit's F3 finding explains why: bf16
  logit ties, 72.5 % "A" on Global-MMLU es.
- **commonsense_qa:** 0.202 → 0.205 against 0.20, likewise 0 everywhere.

Three consequences to decide with it (D1):

1. **The rule does not stop at mmlu.** Applied to `auto` as it is today (canonical gate, trained
   population), **331 of 841 trained tasks fail at both 1B and 1.7B**:
   - belebele 56/59, global_mmlu_full 28/29, include_base_44 32/36 (the rq00 letter originals);
   - cultural_bench_easy 19/19, cultural_bench_hard 19/19, bbh_mcq 16/17, acp_bench_mcq 7/7,
     acp_bench_cloze 7/7, bbh_cloze 6/6, blend_sample 5/5, truthfulqa_mc2 3/3,
     truthfulqa-multi_mc1 2/2, toxigen, commonsense_qa, mmlu;
   - also global_piqa_parallel_cloze 53/63. That mask may predate the acc_norm switch in `a5ce9ff8`, so
     re-check it after the next refresh.

   A "discard on fail" rule for mmlu alone would be inconsistent. Either it applies to every letter
   original (≈ 125 tasks) and every failing task, or mmlu stays with them.
2. **Discarding is not only an eval saving.** `analysis.utils.auto_tasks()` leaves `discarded` tasks out of
   rule 2's trained set, so a discarded original drops out of every pool, not just future evals.
   - Results on disk stay, and the rq00 comparison (`compare.py`, which uses `untrained=True`) keeps
     reading them.
   - Cells evaluated after the discard (muon, the 3B L30/L50 rungs, FWEB, 1.7B swiglu) would have the
     `rf_` twin and no original, so the paired original-vs-twin readings shrink to the cells evaluated
     before.
3. **What it saves.** The originals that fail cost one ordinary task each per checkpoint (mmlu is 14k items
   at ~3 min per worker-task at 1.7B). The gate already masks them out of every analysis (rule 1), so the
   analysis gains nothing from discarding them. Only compute is saved.

My suggestion:
- Discard the failing tasks that have **no twin and no role in rq00**: cultural_bench_hard, blend_sample,
  acp_bench_cloze, bbh_cloze, toxigen, truthfulqa_mc2, truthfulqa-multi_mc1 (en).
- Keep the letter originals that pair with an `rf_` twin (mmlu, commonsense_qa, belebele,
  global_mmlu_full, include_base_44, bbh_mcq, acp_bench_mcq, cultural_bench_easy) while rq00 reads the
  pair at every checkpoint.
- If you prefer to discard mmlu anyway, discard all the paired letter originals together and say in rq00
  that the pair is read on cells evaluated before 2026-10-09.

### Decisions for you

- **D0. Part A launch.** Approve copying the worktree `tasks.json` (+ support files) into the live tree,
  and choose: alone (≈ 2.5k small jobs, ~150 node-h, mostly cold start), or together with D4/D5 (one job
  per checkpoint).
- **D1. MMLU originals in `auto`.** Discard mmlu alone, all paired letter originals, or none (see above).
- **D2. BoolQ chance.** Uniform 0.5, or the majority class 0.6217 (`CHANCE["boolq"]`, with a rule-1 note).
- **D3. `m_truthfulqa_mc1`.** Promote its 15–19 passers (cheap, 15 languages, but a +0.03 margin and an
  English original at chance), or leave it in the probe.
- **D4. `rf_m_mmlu` scope.** All 27 languages (554 node-h) or the 9 Global-MMLU lacks (143 node-h). The
  9-language cut has no mechanism yet: `discarded` means "fails the gate", so the other 18 would need
  another way out, such as a `redundant` group that `tasks_for_benchmarks` also skips. **`rf_mmmlu`:**
  borderline/drop (no new language).
- **D5. The rest of the promote set** (multiblimp-extra, noreval(+extra), kobest, toksuite(+math/stem)),
  ≈ 160 node-h together. Then the 129 discards (`tasks.json.partB.diff`).
- **D6. Probe completeness.** Run `auto_evals_cscs.py --group auto_probe --final-only` once more for
  `lm-1.7B-L8-swiglu-seed1904` before the verdict, or accept the verdict without that cell.

Also noted: the uncommitted `plan/todos/status-09-30.md:506-516` says the
`rf_`/`rfgm_` twins "stay on acc". tasks.json pins every one of them to
acc_norm, as does `src/evals/CLAUDE.md:984`.

### Files

- Proposed Part A tree: `/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual-wt-auto-add` (branch
  `wt/auto-add`, uncommitted; `git diff` there).
- Artifacts (outside the repo, durable): `/iopsstor/scratch/cscs/mariagrandury/snr-probe-gate/promotion-2026-10-09/`
  - `tasks.proposed.json` and `tasks.json.partA.diff` (Part A);
  - `tasks.partB.proposed.json` and `tasks.json.partB.diff` (Part A + the promote set + the 129 discards);
  - `worktree_full.diff`;
  - `probe_tasks_1b_17b.csv` and `probe_benchmarks_1b_17b.csv`;
  - `probe_completeness.csv`;
  - `auto_tasks_fail_both_trained.csv`;
  - `cells_due.json`;
  - `cost_estimates.csv`;
  - `models.json.live-backup`;
  - `scripts/` (the scripts that computed these numbers).
- Notes read: `plan/signal-audit-2026-10-08.md` (F3, F4, fix 3), `plan/todos/probe-candidates-2026-10-08.md`,
  `plan/todos/status-09-30.md`, `src/signal-and-noise/analysis/rq00_task_reformulation/README.md`
  (probe-survivors block), `src/signal-and-noise/analysis/RULES.md` (rule 1), `src/evals/CLAUDE.md`.
