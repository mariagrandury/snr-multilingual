# Task reformulation for the auto evals

Status 2026-10-07. Every hand-written number outside the dated sections, and
every generated block, is from the ladder-report snapshot **2026-10-07 15:51**
(outputs regenerated in `7966367c`, prose re-read 2026-10-07), read from this folder's CSVs and the
`predictivity` gate mask (`../rq00_gate_and_curves/pretraining/predictivity/above_random_mask.csv`).

The dated sections keep their own dates: the harness survey and the pilot of
2026-09-18, the probe of 2026-09-23. The benchmark-BPB twins (`bbpb_*`) are
left out of every count here.

Companion to [plan/benchmark_selection.md](../../../../plan/benchmark_selection.md)
(what we evaluate on) and [the gate-and-curves README](../rq00_gate_and_curves/) (the chance gate).
The reference implementation is [lighteval_reference.py](lighteval_reference.py)
(HuggingFace's FineWeb-edu ablation tasks); everything here runs inside the lm-eval harness we already use.

## Research question

Do letter-format multiple-choice benchmarks leave chance at 90M–1.7B once the
letters are dropped (`rf_`) or the items are rewritten as cloze statements
(`rfgm_`), and do the reformulated twins then rank designs better?

On 2026-09-18, before the twins and the probe promotions, 113 of 461 gated
tasks cleared chance at any size, and only 21 of 252 four-option tasks. In
today's mask, 547 of the 1,041 non-twin benchmark tasks clear chance at some
size and 229 of the 615 four-option ones.

`global_mmlu_full` clears in 6 of 37 languages at 175M, 2 at 350M, 1 at 1.7B
and none at 90M, 600M or 1B. `global_piqa_*` clears in at most 3 of its 95
tasks (at 1.7B), and `truthfulqa-multi_mc1` in 1 of 3 at every size but 600M.

Base models at 90M–1.7B do not pick up the "answer with a letter" convention,
so a lettered multiple-choice prompt measures the convention, not the
knowledge. The lighteval file's fix for MMLU: no letters, no few-shot, the
full answer string scored as the continuation after `Answer:`, with `acc_norm`
(length-normalised) as the metric.

## Setup

Gate counts: the `predictivity` gate mask (seed 1904, every cell, data build
and ladder, 90M–1.7B), read on the cells that trained the task's language
(on every cell where none did). The twins exist only on the deep scheme-A seed-1904 cells, so a twin's verdict is
taken over those runs and the original's over every cell of the pool.

The mask carries 105 belebele, 37 Global-MMLU and 43 INCLUDE tasks, plus 45
tasks of the five promoted families with an `rf_` twin (mmlu 1,
commonsense_qa 1, bbh_mcq 17, acp_bench_mcq 7, cultural_bench_easy 19). The
margins and significance tests use the deep scheme-A seed-1904 finals (six
cells per size, L1–L50; five at 350M, no L1), trained languages only: 59
belebele, 29 Global-MMLU and 36 INCLUDE tasks.

## Highlighted result

![The reformulations and the gate](reformulations_gate_paper.png)

GitHub: [reformulations_gate_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate_paper.png) · [reformulations_gate_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate_paper.csv)

Population: the `predictivity` gate mask, per family and size, the original
(grey) against each twin, paired on the task; * marks McNemar p < 0.05. The
counts below are tasks above the gate (`reformulations_gate_mcnemar.csv`).

Key findings (snapshot 2026-10-07 15:51):

- **The letter-format originals sit on chance.** Their median margin over chance (trained languages, deep scheme-A finals) stays within ±0.013 at every size; at 1.7B it is +0.011 on belebele, −0.010 on Global-MMLU and +0.005 on INCLUDE. At 1.7B the gate keeps 5 of 105 belebele, 1 of 37 Global-MMLU and 3 of 43 INCLUDE originals.
- **Dropping the letters moves whole families across the gate.** At 1.7B the `rf_` twins pass in 86 of 105 belebele, 35 of 37 Global-MMLU and 31 of 43 INCLUDE tasks (McNemar twin-only : original-only 82 : 1, 34 : 0 and 29 : 1; p ≤ 5.8e-08). The McNemar test is significant at every size for belebele and Global-MMLU, and from 175M for INCLUDE.
- **The margin gain is larger at 1.7B than at 90M.** rf − original goes from +0.030 / +0.017 / +0.004 at 90M to **+0.098 / +0.058 / +0.047** at 1.7B (belebele / Global-MMLU / INCLUDE), significant at 1.7B in 56 of 59, 26 of 29 and 18 of 36 tasks. belebele and INCLUDE grow at every size, while Global-MMLU dips to +0.009 at 175M and passes its 90M value again only from 600M (+0.019).
- **The Gemini rewrite adds more where it exists.** `rfgm_` gains **+0.168** on belebele (59 of 59 tasks significant) and **+0.075** on INCLUDE (22 of 36) at 1.7B. `rfgm_belebele` passes the gate in 59 of 59 tasks at every size, and `rfgm_include_base_44` in 32 of 43 at 1.7B; Global-MMLU has no rewrite yet.
- **The gain is not the metric.** On plain `acc` on both sides, the median rf gain per (task, model) at 1.7B is +0.124 belebele, +0.047 Global-MMLU and +0.062 INCLUDE, against +0.104 / +0.064 / +0.078 on acc_norm (`rf_significance.csv`, 288 pairs).
- **Over the gate's whole population, the twins pass in 267 of 332 tasks at 1.7B (0.80)**, against 511 of 1,041 non-twin benchmark tasks (0.49). Their own 230 originals pass in 9.
- **Passing the gate does not make the twins rank designs better.** Mean DA-size (proxy → 1.7B, multi-axis pairs, gated at proxy and reference, `predictivity`) is 0.486 at 90M and 0.550 at 1B on the twins (146 and 214 tasks), against 0.550 and 0.593 on the non-twin tasks (155 and 236). The twins' pairs come from the deep scheme-A cells alone (the L axis), so the two means are not on the same pairs.

Follow-ups:

- Draw DA-size for the twins against their own originals on the same deep scheme-A pairs, so that the last bullet compares like with like.

## What the harness already does per family

The survey of 2026-09-18, before the twins and the probe promotions joined `auto`:
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
- `auto_evals_cscs.py --reformulated` (removed on 2026-10-03, with its groups
  retired on 2026-09-23: the twins are now in `auto`): the `auto_rf` group instead of `auto`,
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
   final iter plus the noise window (`--every 1000` coarsens the tenths away, but the 85/90/95/100 % points are added unconditionally, so five saves are due, not one). Both are
   ALL_LANGUAGES runs, so every rf task runs (185) and the comparison covers
   trained and untrained languages. Read: per family and language, rf
   `acc_norm` vs the original `acc` on the same checkpoint (both on disk),
   against chance 0.25 + 0.05. If the 1.7B stays at chance, stop: the
   families need Tier 2, not a prompt change.
2. **The L8 size ladder finals** (90M … 1.7B, 6 jobs): does rf order the
   sizes, and from which size does it clear the gate?
3. **The deep-A-seed1904 ladder** at the normal `--every 2` cadence, under
   `--watch`, throttled with `--max-submit`.

These are the commands as they were run. The flag no longer exists: an
ordinary `auto` pass now evaluates the twins.

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
chance gate (`above_random.scores_and_mask` on the `predictivity` pool, deep
scheme-A cells). So the `auto:rf-compare` table further down (after
[the length tell](#the-length-tell-and-where-it-actually-is)) is `original acc → rf acc_norm`.

It moves only after `ladder_report.py --plot --publish --push-hf` has picked up
new rf results and `scripts/refresh_analysis.sh` has re-run. Until then `compare.py` refuses to run against a report without the `rf_*` columns (it would empty the table); point `SNR_LADDER_DIR` at a report that has them.

The conclusion does not depend on the metric. On the 2026-10-07 15:51 snapshot,
the median rf gain per (task, model) at 1.7B is +0.104 on acc_norm and +0.124
on acc for belebele, +0.064 and +0.047 for Global-MMLU, and +0.078 and +0.062
for INCLUDE (`rf_significance.csv`, which carries the twin run's own `acc`, the
metric the significance test reads).

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

## Tier 2 — rewriting the items with Gemini (`rfgm_*`: Belebele and INCLUDE rewritten and evaluated, Global-MMLU not yet)

Tier 1 only drops the letters. Here the item itself changes: Gemini turns
the question into one declarative sentence that stops where the answer goes
(a stem) and the four options into short, parallel continuations, in the
item's own language — the cloze formulation the FineWeb-2 / FineTasks work
found readable at small scale. Decided 2026-09-19: the three letter families
in all their languages (185 tasks, ~636k items), `gemini-3.8-flash` through
the Batch API, evaluated on the deep data-A seed-1904 ladder at every
evaluated checkpoint (the `rf_*` coverage), so original / rf / rfgm are
compared on identical models. The families that are cloze already (arc,
hellaswag, global_piqa) are not rewritten.

The set is a third family prefix — `rfgm_` — on the exact `rf_` plumbing:
[`rewrite_items_gemini.py`](../../../evals/scripts/rewrite_items_gemini.py)
produces one JSONL per task under
`/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/rf-data/rfgm/` (never
pushed publicly: gold labels), `make_rf_tasks.py --set rfgm` writes one
`dataset_path: json` YAML per task under `src/evals/tasks/rfgm/` and the
tasks.json entries (`benchmark: rfgm_<family>`, `metric: acc_norm`, group
`auto_rfgm`), `auto_evals_cscs.py --reformulated rfgm` evaluates them with
`-rfgm` job names, and `compare.py` draws them next to `rf` (SETS).

Row schema of `<task>.jsonl` (`text`, `choices`, `gold` are what the YAML
reads by column name; the rest is for audits):

```json
{"id": 17, "text": "<passage>\n<stem>", "choices": ["…", "…", "…", "…"], "gold": 2,
 "stem": "…", "context": "<passage or ''>", "subject": "anatomy|''",
 "orig_question": "…", "orig_choices": ["…"], "lang": "deu_Latn"}
```

`text` is the full prompt: the belebele passage verbatim + newline + stem;
Global-MMLU / INCLUDE the stem only. Gold keeps the original position. The
stem carries no trailing space — the harness appends `" " + choice` itself.

### Step by step

**0. Once — Application Default Credentials.** We authenticate to Vertex AI
with ADC, not an API key: the Generative Language API refuses keys under an
organisation policy that blocks them, which is what `API_KEY_INVALID` on a
well-formed key means. ADC uses your own Google identity and the SDK picks
it up with no secret in the repo. Run Google's
[setup script](https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.sh)
— it installs the gcloud CLI, asks for your project ID, runs the login,
sets the quota project and enables `aiplatform.googleapis.com` — or do the
same by hand:

```bash
curl -sSL https://sdk.cloud.google.com | bash && exec -l $SHELL   # gcloud, into $HOME
gcloud auth application-default login --no-browser                # prints a command to run on your laptop
gcloud auth application-default set-quota-project <project-id>
```

**Run the login on the login node, not on your laptop.** ADC is a file,
`~/.config/gcloud/application_default_credentials.json`, and the driver
reads the one on the machine it runs on. `--no-browser` prints a
`gcloud … --remote-bootstrap="…"` command to paste into a laptop that has
gcloud and a browser; that returns a URL you paste back. If you already
authenticated on the laptop, copying the file over is equivalent and
quicker: `scp ~/.config/gcloud/application_default_credentials.json
clariden:.config/gcloud/`. Credentials refresh themselves and gcloud is not
needed again afterwards.

**Enabling the API is a different credential.** `gcloud auth
application-default login` writes ADC for client libraries; it does not log
the gcloud CLI itself in, so `gcloud services enable` answers "You do not
currently have an active account selected". Either run `gcloud auth login`
first, or just switch the API on in the console:
`console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=<project-id>`.
The project also needs billing — batch prediction is not a free-tier
feature.

**Cloud Storage.** A Vertex batch job reads its requests from Cloud Storage
and writes its answers back there (an uploaded file, which the Gemini
Developer API takes, is not a valid Vertex source). The driver keeps them
under `gs://<bucket>/rfgm/`, where `<bucket>` is `$RFGM_GCS_BUCKET`, else
`--bucket`, else `<project>-msnr-rfgm`, and creates it on first use with
the same ADC credential — which needs `storage.buckets.create` on the
project (`roles/storage.admin`, or `roles/storage.bucketCreator` plus
`roles/storage.objectAdmin`). Without it the driver stops with the 403 and
the role to ask for.

```bash
# 1. can this account create the bucket?
gcloud storage buckets create gs://$GOOGLE_CLOUD_PROJECT-msnr-rfgm \
    --project=$GOOGLE_CLOUD_PROJECT --location=US --uniform-bucket-level-access
# 2. if that is a 403, a project admin runs ONE of:
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member=user:<you@example.org> --role=roles/storage.admin     # then repeat step 1
gcloud storage buckets create gs://<name> --project=$GOOGLE_CLOUD_PROJECT --location=US
gcloud storage buckets add-iam-policy-binding gs://<name> \
    --member=user:<you@example.org> --role=roles/storage.objectAdmin   # then export RFGM_GCS_BUCKET=<name>
# 3. verify
gcloud storage ls gs://$GOOGLE_CLOUD_PROJECT-msnr-rfgm
```

The bucket may live in another project the account does own; Vertex reads
it as long as the caller has object access. Only `submit`/`fetch` need it —
`pilot` and `build` do not. About 250 MB of requests and a similar volume of
answers, at Standard-class prices a few cents a month; delete the bucket
when the rewritten sets are on capstor.

**The environment every command below starts from.** The location must be
`global`: the whole Gemini 3.x family is served only from the global
endpoint, and a regional job answers 404 *The PublisherModel does not
exist* (checked 2026-09-20 against `us-central1`, `europe-west4` and
`us-east5` — only 2.5 is regional). Batch prediction accepts `global`, and
the bucket's `US` multi-region default serves it.
`google-genai` is in `requirements-ml.txt` and already in the snr env; the
storage calls use `google-auth` and `requests`, which come with it.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<project-id> GOOGLE_CLOUD_LOCATION=global
export HF_HOME=/iopsstor/scratch/cscs/mariagrandury/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

`GOOGLE_GENAI_USE_VERTEXAI` decides the shape of the request files, so the
driver refuses to run if the SDK resolves a different backend than the one
it was configured for — export it before `build`, not between `build` and
`submit`. Everything runs on the login node (outbound internet, the snr
env, the offline datasets cache the items come from); the driver is
network-bound and light, and every mode is idempotent, so re-run after any
interruption. Vertex's batch quota is per project and region: `submit`
stops at the first `RESOURCE_EXHAUSTED` and is re-run later.

**1. Pilot (~$0.40, 15 min).** Synchronous calls with the batch's prompt
and schema on 10 items of each review task — the `REVIEW_TASKS` in the
driver: Spanish, Hindi, Turkish, Farsi, Greek, Arabic, Basque and Chinese
in every family that has them (8 belebele, 7 Global-MMLU, 8 INCLUDE; 230
items) — printed original → rewritten and written to `pilot_<task>.jsonl`
in this directory (accepted rows in the row schema, rejected ones with
their reason):

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py pilot
```

Read all of them: language kept, stem answerable, gold not leaked, options
parallel. The prompt (`SYSTEM`, `NOTES` in the driver) is tuned here and
nowhere else; the validator should pass ≥ 95 %. `--tasks a,b` / `--family`
narrow the pilot while iterating on the prompt.

**2. Build the request files.** One line per item in
`rf-data/rfgm/_requests/<task>.jsonl` (REST JSON: the user content, the
system instruction, `responseSchema`, `thinkingLevel: LOW`), rows with a
missing option dropped as in the rf twins; prints the token estimate and
the cost per family. `--dry-run` prints the estimate only.

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py build --dry-run
python3.11 src/evals/scripts/rewrite_items_gemini.py build
```

**3a. No bucket? Run it online instead.** `online` uses the same prompt and
the same validation but calls `generate_content` directly, so it needs
nothing but ADC — no Cloud Storage, no `build`/`submit`/`fetch`. It costs
**double** the batch price and takes wall-clock time instead of a queue,
so it is the fallback while `storage.buckets.create` is missing, not the
default.

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py online --family include_base_44 --L 50
python3.11 src/evals/scripts/rewrite_items_gemini.py online --family belebele --L 50
```

**`--L 50` is not an optimisation, it is the right task set.** The ladder
only scores a task on models that trained its language (rule 2,
`utils.trained_only`), so a twin outside the L-cell's languages is paid for
and never read — the chance gate's own task counts are exactly this set. L50
keeps 124 of the 185 tasks:

| family | all | L50 | items (L50) |
|---|---:|---:|---:|
| belebele | 105 | 59 | 53,100 |
| global_mmlu_full | 37 | 29 | 407,192 |
| include_base_44 | 43 | 36 | 18,337 |

On belebele that is $38 and ~3 hours saved online; on Global-MMLU, 112k
items. `--scheme` picks the scheme `--L` refers to (default A).

Run one family at a time. Two concurrent runs at the default concurrency
put 24 requests in flight, which is where the shared quota collapses — the
aggregate is slower than running them in sequence.

Every answer is appended and flushed as it lands and a re-run skips the ids
already written, so a killed login-node process loses at most the requests
in flight — re-run the same command to continue. `--limit N` caps new items
per task, `--concurrency` defaults to 12: Gemini 3.x online runs on Vertex's
dynamic shared quota, where 12 threads sustain ~2.9 items/s while 24 trips
429s and collapses throughput to 0.3 (measured 2026-09-20). It prints rate,
ETA and running cost every 200 items.

**3. Submit.** One batch job per task (185 files of 1–20 MB): each is
uploaded to `gs://<bucket>/rfgm/requests/<task>.jsonl` and becomes the
job's source, and the answers land in `…/<task>/dest`. Both URIs and the
job name go into `rf-data/rfgm/_jobs.json`; a task that has a job is
skipped, so re-run until `status` shows 185.

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py submit --max-jobs 40
python3.11 src/evals/scripts/rewrite_items_gemini.py status
```

**4. Fetch and validate.** Reads every SUCCEEDED job's destination prefix
in Cloud Storage, matches each answer back to its item — Vertex drops
everything outside `request` and echoes the request instead of a key, so
the match is on the prompt text, and any line that matches nothing is
counted and skipped rather than written — validates each item — JSON with a non-empty stem (a
trailing colon, dash or ellipsis is trimmed rather than rejected: five of
the six pilot rejects were a colon, and they fell on Basque 3/10 and
Persian 2/20 against 0 of 180 elsewhere, a per-language item loss this
comparison cannot absorb), exactly four non-empty pairwise-distinct
choices, no choice appearing as a word in the stem (answer leak), the
majority Unicode script unchanged, plus a per-choice script check for the
options the source wrote in the item's own script (the per-item majority was
blind to a Greek item whose four numerals came back in Latin; options that
were already off-script, such as a bare `"a, b"`, are skipped). Kana,
katakana, hangul and han count as one script: Japanese mixes kanji and kana
within a sentence, and comparing the per-choice majority without folding
them rejected 30.9 % of Japanese INCLUDE items against 0-2 % elsewhere — an
uneven item loss across languages is the one bias this comparison cannot
absorb. Script is the cheap "still in its language" check and cannot tell
Spanish from English; step 5 covers the Latin-script languages. Writes
`<task>.jsonl` (accepted) and `_rejects/<task>.jsonl` (with the reason and
what the model produced, so a reject can be re-judged without paying for the
call again; `online --retry-rejects` re-runs them). A task file is written only from a
SUCCEEDED job, so a partial run never leaves a half file; FAILED / EXPIRED
jobs are cleared and `submit` resubmits them (from the object already in
the bucket); rejected items get one more
round through `submit --retry` (then `fetch` again), a second rejection
drops the item. Batch jobs target 24 h; most finish sooner.

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py fetch
python3.11 src/evals/scripts/rewrite_items_gemini.py submit --retry && python3.11 src/evals/scripts/rewrite_items_gemini.py fetch
```

**5. Spot-check (45 min, human).** Per task the row and reject counts,
reject rate, median stem / choice length, the tokens actually spent and
their cost, and with `--show` five random items of each of the 23 review
tasks (the pilot's eight languages):

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py report --show
```

Look for translation drift, stems that give the answer away, options
collapsed to one word where the original was a clause. A task above 5 %
rejects: read its rejects before evaluating it.

**6. Register the tasks.** 185 YAMLs + tasks.json (idempotent; tasks whose
JSONL is missing are skipped and counted). `n_items` is filled from the
first results by `derive_task_options.py`.

```bash
python3.11 src/evals/scripts/make_rf_tasks.py --set rfgm
```

Steps 7 and 8 are the record of the 2026-09 launch; `--reformulated` and
`auto_rfgm` no longer exist, and the `rfgm_*` twins are in `auto`.

**7. First eval — one job (needs approval).** `test_new_tasks.py` checks
names against a harness clone and never passes `--include_path`, so the
smoke test is the watcher on one final checkpoint, all rfgm tasks, ~25
min on one node; the `per_task/rfgm_*` results must carry `acc,none` and
`acc_norm,none`, and `samples_rfgm_*.jsonl` the rewritten docs:

```bash
cd src/pretrain
python3.11 auto_evals_cscs.py --reformulated rfgm --name lm-175M-L8-deep-seed1904 --every 1000 --dry-run
python3.11 auto_evals_cscs.py --reformulated rfgm --name lm-175M-L8-deep-seed1904 --every 1000 --max-submit 1
```

**8. The ladder (needs approval).** The rf watcher's shape: every 2nd
checkpoint + final of the deep data-A seed-1904 cells, jobs
`eval-<cell>-iter<N>-rfgm`, 23–45 min each on one node, ~625 jobs, ~385
node-hours (rf took about two days at `--max-submit 20`). Held-back tasks:
a `--retry-held` pass. `pretrain_progress.py` does not count reformulated
sets; progress is `squeue --me | grep rfgm` and the watcher log.

```bash
python3.11 auto_evals_cscs.py --reformulated rfgm --arch deep --scheme A --seed 1904 --dry-run
nohup python3.11 -u auto_evals_cscs.py --reformulated rfgm --arch deep --scheme A --seed 1904 \
      --max-submit 20 --watch 1800 >> /iopsstor/scratch/cscs/mariagrandury/auto_evals_rfgm_watch.log 2>&1 &
```

Results land exactly as the originals' do: `per_task/<task>/…/results_*.json`
and the merged `results_*.json`, one `samples_<task>_*.jsonl` per task,
`job.json`, the Slurm log `src/evals/logs/eval-<cell>-iter<N>-rfgm_<jobid>`,
and `push_all_results.py` pushes `rfgm_<task>/acc_norm` to W&B (registered
tasks are never dropped).

**9. Report and analysis.** The report picks the `rfgm_*` columns up
(`metric_for` → acc_norm); `compare.py` then draws five panels — original,
rf, rfgm, rf − original, rfgm − original — with the significance counts
per set, the table below gets both deltas, and the gate outputs
(`first_size_above_random`, `gate_margin_by_benchmark`) list the `rf_*` and
`rfgm_*` families as rows next to the originals: that is the direct read of
the gate impact. Every other analysis (decision accuracy, noise and SNR) sees the new tasks
too, as it does the rf ones.

```bash
python3.11 src/pretrain/ladder_report.py --plot --publish --push-hf
bash scripts/refresh_analysis.sh
```

### Cost

| | items | in Mtok | out Mtok (incl. thinking) | batch $ | online $ |
|---|---:|---:|---:|---:|---:|
| belebele (105) | 94.5k | 75 | 5 | 38 | 76 |
| global_mmlu_full (37) | 519.5k | 352 | 47 | 220 | 440 |
| include_base_44 (43) | 22.1k | 15 | 2 | 9 | 18 |
| total | 636k | 441 | 54 | **267** | **533** |

Measured, not estimated: 16 real requests per family on 2026-09-20 gave
791/678/658 input and 55/90/88 output tokens per item (output includes
thinking at LOW). The earlier bytes/3.5 projection of 607 Mtok in was ~38 %
high, so the run is cheaper than the $525–570 first documented. Online
prices are double batch, and are what the run costs if it goes through
`generate_content` because no Cloud Storage bucket is available.

The instruction and the response schema are 550 tokens by `count_tokens`
and go out with every single request, against an item that is only 57
tokens for Global-MMLU and ~240 for belebele — so the scaffold is about
80 % of the input bill, and trimming instruction text is the only lever
that moves it. `scaffold()` estimates it as bytes/3.5, which runs ~30 %
high on English prose (719 vs 550), so `build --dry-run` moves with the
prompt but reads high. No cache discount is
available against it: implicit caching wants a 2 048-token shared prefix on
Vertex and this is a quarter of that, and batch and cache discounts would
not compound anyway.

`gemini-3.8-flash` batch: $0.375 / $1.875 per Mtok in / out through
2026-12-31, double from 2027 — the same on Vertex as on the Gemini API
([pricing](https://ai.google.dev/gemini-api/docs/pricing)), billed to the
Cloud project instead of the key;
thinking tokens bill as output and no Gemini 3 model turns thinking off,
only down to `LOW` / `MINIMAL`. Input = item + ~150 tokens of instruction
and schema; output = stem + four choices + JSON; belebele passages are
input only. Bytes/3.5 estimates, ±30 %. `build --dry-run` prints the same
table from the cache; `report` prints what was actually spent. Cheaper
alternatives at the same scope: `gemini-3.1-flash-lite` batch
($0.125 / $0.75 → ~$125–170), riskier on low-resource scripts;
`gemini-3.1-pro-preview` batch ($1 / $6 → ~$1,000–1,400).

### The prompt

**The rewriter is never told which option is correct.** The user turn carries
the language, the subject, the passage, the question and the four options in
their original order, and nothing else; `gold` stays in the driver and is
copied to the output row untouched. Telling it would invite the failure that
defeats the whole exercise: a model that knows the answer writes the true
statement more fully or more specifically than the false ones, and a small
model then scores above chance by following the style rather than the
content, which is the letter-format artefact traded for a subtler one. The
instruction to "preserve which choice is correct" is a prohibition on
flipping truth values, not a disclosure.

It can still often infer the answer, so `report` measures what is left: how
often the gold continuation is the single longest of the four, in the
rewrite and in the originals side by side. Chance is 25 %, the originals
already carry some of this artefact, and what matters is whether the rewrite
raised it.

System instruction (`SYSTEM` in the driver; `{family_note}` per family):

```
You rewrite multiple-choice test items so that a small language model can be
scored on them as text continuations. Rewrite the item in the SAME LANGUAGE
and script as the input — never translate, never switch to English.

Turn the question into ONE declarative sentence that stops exactly where the
answer would go (a "stem"). Turn each of the four options into a short
continuation that completes the stem into a true or false statement. Rules:
- The stem must not contain or hint at the correct answer, and must not say
  "which of the following".
- Keep the meaning of the original question and of every option; keep the
  options in the same order; keep the correct option correct.
- Make the four continuations parallel: same grammatical form, similar
  length (aim for 1-8 words), no option letters or numbers.
- The stem ends with no trailing space or punctuation; each continuation
  starts as it would follow a space after the stem (lowercase unless it is a
  name), and ends with a period if the stem+continuation is a full sentence.
- Do not add facts. If the question asks for the option that is NOT true,
  write the stem as "Of the following, the one that is not … is".
{family_note}
Return only JSON: {"stem": "...", "choices": ["...", "...", "...", "..."]}
```

Family notes — belebele: "A passage is given; the stem must be answerable
from that passage alone. Do not rewrite or quote the passage." Global-MMLU:
"The subject is given; the stem may name it." INCLUDE: "The item may test
regional knowledge (driving rules, local history); keep the local terms."
User content per item: `Language:`, `Subject:` (Global-MMLU, INCLUDE),
`Passage:` (belebele), `Question:`, `Options:` numbered 1–4. The response
schema pins `{"stem": string, "choices": [4 strings]}`.

## The probe (2026-09-23)

The same question, asked of new candidates: `configs/tasks.json` →
`groups.auto_probe` (BBH and ACP-Bench as cloze arms, mmlu, commonsense_qa,
cultural_bench, INCLUDE v2, …, and the `rf_` twins of the ones that ask for a
letter), evaluated at the last checkpoint of the 600M–1.7B cells. `probe.sh`
runs the chain — `derive_task_options` → `ladder_report --plot` →
`above_random` → `compare.py --tag probe` → `probe_survivors.py` →
`check_rules` — and leaves `rf_gate_probe.*` (the pairs, this README's
`rf-compare-probe` block) and `probe_survivors.csv` (the gate per language,
the `probe-survivors` block). `compare.py` subtracts each task's own chance
level, so the 2- to 10-way probe pairs read on the same scale as the 4-way
families above (whose numbers this does not change).

**Outcome (2026-09-23).** Twenty of the twenty-one candidates were promoted
into `groups.auto`, so every checkpoint is topped up with them and they enter
every analysis: a population change, stated in `RULES.md`.

`bbq` was not promoted: it cleared a 1/12 chance at 0.44 without telling us
much, and it cost 23 min per checkpoint against 2.9 for mmlu and 0.3 for a
belebele task.

### The length tell, and where it actually is

Replacing letters with answer strings hands the model a lever the lettered
form does not have: the options now differ in length. If the gold answer is
systematically the shortest (or longest) of its set, a model scores above
chance by preferring short (raw log-likelihood) or long (`acc_norm`, which
divides by length) strings without reading the stem — and the
original→`rf_` difference is then partly that artefact rather than the format
change. `acc_norm` normalises a candidate by its own length; it does not
remove a dataset-level correlation between *being the gold* and *being short*.

Measured off the cached items, not the model: how often the gold is the
**strictly** shortest (or longest) option, ties excluded, over every task of a
family (8 sampled where a family has more).

| family | chance | gold shortest | gold longest | items |
|---|---:|---:|---:|---:|
| rf_acp_bench_mcq | 25.0 % | **35.4 %** | 15.1 % | 900 |
| rf_bbh_mcq | 23.5 % | 13.1 % | 10.9 % | 1,896 |
| rf_belebele | 25.0 % | 23.7 % | 20.8 % | 7,200 |
| rf_commonsense_qa | 20.0 % | 11.1 % | 19.0 % | 1,221 |
| rf_cultural_bench_easy | 25.0 % | 24.0 % | 27.1 % | 221 |
| rf_global_mmlu_full | 25.0 % | 17.8 % | 23.3 % | 112,332 |
| rf_include_base_44 | 25.0 % | 15.8 % | 22.2 % | 4,328 |
| rf_mmlu | 25.0 % | 17.8 % | 23.7 % | 14,042 |
| rfgm_include_base_44 | 25.0 % | 17.6 % | 21.9 % | 4,243 |

So this is not a property of the reformulation: every family the study
already relies on sits at or below chance on both tells, the Gemini rewrite
included (17.6 %, which is what the driver's own length report is there to
keep). It is a property of ACP-Bench's published distractor sets — its items
are plans, and a wrong plan tends to be stated at greater length — and it
only becomes reachable once the strings are scored. Three of its seven
subtasks carry it:

| task | chance | gold shortest |
|---|---:|---:|
| rf_acp_bench_mcq_areach | 25 % | 50.0 % |
| rf_acp_bench_mcq_val | 25 % | 47.7 % |
| rf_acp_bench_mcq_prog | 25 % | 46.9 % |
| rf_acp_bench_mcq_land | 25 % | 30.0 % |
| rf_acp_bench_mcq_app | 25 % | 28.5 % |
| rf_acp_bench_mcq_reach | 25 % | 26.9 % |
| rf_acp_bench_mcq_just | 25 % | 19.2 % |

For those three, the reference a score has to beat is the pick-shortest rate
(0.50 / 0.48 / 0.47), not the 0.25 the gate uses, and the gate's verdict on
them should be read with that substitution — which is why the probe reports
`rf_acp_bench_mcq` as a family and not as one number. `rf_bbh_mcq_snarks`,
the two-way task where a tell would be cheapest, measures 54.8 % against its
50 % chance on 177 items, inside its own sampling error.

<!-- BEGIN auto:rf-compare (analysis/rq00_task_reformulation/compare.py) -->
Gate cells (median task margin over the task's chance level, trained languages, deep data-A seed-1904 ladder, from the ladder report; each set on the models that have the original and that twin scored — the original shown is the rf pairing). Cell: original acc, then per set `twin acc_norm (**Δ** = twin − original, n = tasks, sig)`; sig = tasks whose gain is significant for at least half of the size's models (two-proportion z-test of the original's acc against the twin run's own acc, p < 0.05; the acc_norm−acc offset is family-shaped, rf median +0.005, and exceeds half the plotted rf gain in 36 % of the pairs).

| family | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---:|---:|---:|---:|---:|---:|
| belebele | +0.004 · rf +0.033 (**+0.030**, n=59, sig=41) · rfgm +0.071 (**+0.067**, n=59, sig=53) | -0.003 · rf +0.039 (**+0.042**, n=59, sig=45) · rfgm +0.081 (**+0.084**, n=59, sig=58) | -0.013 · rf +0.046 (**+0.059**, n=59, sig=55) · rfgm +0.089 (**+0.102**, n=59, sig=57) | -0.007 · rf +0.058 (**+0.065**, n=59, sig=54) · rfgm +0.113 (**+0.120**, n=59, sig=59) | +0.000 · rf +0.083 (**+0.083**, n=59, sig=57) · rfgm +0.139 (**+0.139**, n=59, sig=59) | +0.011 · rf +0.109 (**+0.098**, n=59, sig=56) · rfgm +0.179 (**+0.168**, n=59, sig=59) |
| global_mmlu_full | -0.012 · rf +0.006 (**+0.017**, n=29, sig=2) · rfgm — | -0.000 · rf +0.009 (**+0.009**, n=29, sig=1) · rfgm — | -0.000 · rf +0.009 (**+0.010**, n=29, sig=3) · rfgm — | -0.004 · rf +0.015 (**+0.019**, n=29, sig=13) · rfgm — | -0.008 · rf +0.029 (**+0.037**, n=29, sig=24) · rfgm — | -0.010 · rf +0.048 (**+0.058**, n=29, sig=26) · rfgm — |
| include_base_44 | +0.003 · rf +0.007 (**+0.004**, n=36, sig=2) · rfgm +0.009 (**+0.006**, n=36, sig=1) | +0.001 · rf +0.013 (**+0.012**, n=36, sig=1) · rfgm +0.015 (**+0.015**, n=36, sig=3) | +0.001 · rf +0.023 (**+0.022**, n=36, sig=7) · rfgm +0.023 (**+0.022**, n=36, sig=7) | +0.001 · rf +0.027 (**+0.026**, n=36, sig=6) · rfgm +0.035 (**+0.035**, n=36, sig=12) | +0.002 · rf +0.039 (**+0.037**, n=36, sig=12) · rfgm +0.055 (**+0.053**, n=36, sig=15) | +0.005 · rf +0.052 (**+0.047**, n=36, sig=17) · rfgm +0.080 (**+0.075**, n=36, sig=21) |

![family x size](rf_gate.png)

![per language](rf_gate_by_language.png)
<!-- END auto:rf-compare -->

GitHub: [rf_gate.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate.png) · [rf_gate.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate.csv) ·
GitHub: [rf_gate_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_by_language.png) · [rf_gate_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_by_language.csv) ·
[rf_significance.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_significance.csv)

The opening figure showed the twins crossing the gate; this one measures how far above chance they land.
Population: `predictivity`, deep scheme-A seed-1904 cells (L1–L50; five at 350M, no L1), final checkpoints, 90M–1.7B, trained languages only; 59 belebele, 29 Global-MMLU and 36 INCLUDE tasks.

Cells are medians over tasks of the margin over chance, `acc` for the originals and acc_norm for the twins.

Key findings:

- **The originals stay within ±0.013 of chance at every size** in all three families.
- **The rf gain and its significance rise overall, but not at every size.** From 90M to 1.7B, belebele goes from +0.030 to +0.098 and INCLUDE from +0.004 to +0.047, rising at every size, while Global-MMLU goes from +0.017 to +0.058 with a dip to +0.009 at 175M. The significant-task counts by size (90M → 1.7B) are 41/45/55/54/57/56 of 59 on belebele, 6/14/17/15/24/26 of 29 on Global-MMLU and 7/3/9/10/14/18 of 36 on INCLUDE.
- **rfgm beats rf at every size on belebele** (+0.067 at 90M → +0.168 at 1.7B; 53 of 59 tasks significant at 90M, 59 of 59 at 600M, 1B and 1.7B). On INCLUDE, rfgm stays within 0.003 of rf up to 350M (+0.022 each there) and pulls ahead from 600M (+0.075 against +0.047 at 1.7B).
- **Per language at 1.7B** (`rf_gate_by_language.csv`), every belebele and Global-MMLU language gains. rf − original runs from +0.042 (hi) to +0.158 (ru) over 49 belebele languages (median +0.106; rfgm median +0.170) and from +0.011 (bn) to +0.107 (en) over 29 Global-MMLU languages (median +0.059).
- **INCLUDE loses in two of 36 languages under each twin**: ne and ur under rf, ml and ur under rfgm; its median language gains +0.067 (rf) and +0.082 (rfgm).
- **Part of every twin cell is the acc_norm choice.** Its mean offset (twin acc_norm − twin acc) over the 1,726 rf (task, model) pairs, 90M–1.7B, is −0.016 on belebele, +0.015 on Global-MMLU and +0.018 on INCLUDE, with a median of +0.004. The block's "rf median +0.005 … 36 % of the pairs" is a constant in `compare.py`; today |offset| exceeds half the pair's |acc_norm gain| in 45 % of the pairs.

Follow-ups:

- Draw each row once: `rf_gate.csv` holds 138 rows for 78 distinct cells, and `rf_gate_by_language.csv` 5,622 rows for 3,072, which makes the per-language figure unreadable. Splitting it by family would make the languages legible.
- Compute the acc_norm-offset sentence in the block and caption from `rf_significance.csv` on every run, instead of the constants.
- Add a same-metric panel (twin `acc` − original `acc`) beside the acc_norm one, since the z-test already runs on `acc`.
- Add a Global-MMLU rfgm set: it is the family with the most items and the only letter family with no rewrite.

<!-- BEGIN auto:rf-compare-probe (analysis/rq00_task_reformulation/compare.py --tag probe) -->
Gate cells (median task margin over the task's chance level, trained languages, deep data-A seed-1904 ladder, from the ladder report; each set on the models that have the original and that twin scored — the original shown is the rf pairing). Cell: original acc, then per set `twin acc_norm (**Δ** = twin − original, n = tasks, sig)`; sig = tasks whose gain is significant for at least half of the size's models (two-proportion z-test of the original's acc against the twin run's own acc, p < 0.05; 3 of 1419 (task, model) pairs untested, no harness results file).

| family | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---:|---:|---:|---:|---:|---:|
| mmlu | -0.007 · rf +0.023 (**+0.030**, n=1, sig=1) · rfgm — | +0.001 · rf +0.030 (**+0.029**, n=1, sig=1) · rfgm — | -0.005 · rf +0.036 (**+0.041**, n=1, sig=1) · rfgm — | +0.001 · rf +0.057 (**+0.057**, n=1, sig=1) · rfgm — | -0.004 · rf +0.082 (**+0.086**, n=1, sig=1) · rfgm — | +0.017 · rf +0.111 (**+0.093**, n=1, sig=1) · rfgm — |
| commonsense_qa | +0.004 · rf +0.107 (**+0.103**, n=1, sig=1) · rfgm — | +0.004 · rf +0.135 (**+0.131**, n=1, sig=1) · rfgm — | -0.000 · rf +0.148 (**+0.148**, n=1, sig=1) · rfgm — | +0.002 · rf +0.188 (**+0.186**, n=1, sig=1) · rfgm — | +0.001 · rf +0.214 (**+0.213**, n=1, sig=1) · rfgm — | +0.009 · rf +0.265 (**+0.256**, n=1, sig=1) · rfgm — |
| cultural_bench_easy | +0.175 · rf +0.010 (**-0.165**, n=19, sig=0) · rfgm — | -0.093 · rf +0.025 (**+0.118**, n=19, sig=7) · rfgm — | -0.026 · rf +0.021 (**+0.047**, n=19, sig=2) · rfgm — | -0.015 · rf +0.025 (**+0.039**, n=19, sig=7) · rfgm — | +0.021 · rf +0.065 (**+0.044**, n=19, sig=5) · rfgm — | +0.043 · rf +0.092 (**+0.049**, n=19, sig=7) · rfgm — |
| bbh_mcq | -0.000 · rf +0.024 (**+0.024**, n=17, sig=6) · rfgm — | -0.004 · rf +0.030 (**+0.034**, n=17, sig=6) · rfgm — | +0.005 · rf +0.036 (**+0.031**, n=17, sig=5) · rfgm — | +0.006 · rf +0.060 (**+0.054**, n=17, sig=7) · rfgm — | +0.000 · rf +0.075 (**+0.075**, n=17, sig=8) · rfgm — | -0.003 · rf +0.084 (**+0.087**, n=17, sig=10) · rfgm — |
| acp_bench_mcq | +0.003 · rf +0.047 (**+0.045**, n=7, sig=4) · rfgm — | -0.001 · rf +0.064 (**+0.065**, n=7, sig=5) · rfgm — | +0.010 · rf +0.062 (**+0.052**, n=7, sig=4) · rfgm — | +0.000 · rf +0.087 (**+0.087**, n=7, sig=5) · rfgm — | +0.006 · rf +0.105 (**+0.099**, n=7, sig=5) · rfgm — | +0.010 · rf +0.132 (**+0.122**, n=7, sig=5) · rfgm — |

![family x size](rf_gate_probe.png)

![per language](rf_gate_by_language_probe.png)
<!-- END auto:rf-compare-probe -->

GitHub: [rf_gate_probe.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_probe.png) · [rf_gate_probe.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_probe.csv) ·
GitHub: [rf_gate_by_language_probe.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_by_language_probe.png) · [rf_gate_by_language_probe.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_gate_by_language_probe.csv) ·
[rf_significance_probe.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/rf_significance_probe.csv)

The same reading on the promoted probe families, which tests whether the letter effect is specific to the three multilingual families.
Population: the same deep scheme-A seed-1904 finals, on the five promoted families with an `rf_` twin: mmlu (1 task), commonsense_qa (1), bbh_mcq (17 subtasks) and acp_bench_mcq (7), all English, plus cultural_bench_easy (19 tasks over 8 languages).

Key findings:

- **rf − original is larger at 1.7B than at 90M on all four English families**: commonsense_qa +0.103 → +0.256, acp_bench_mcq +0.045 → +0.122, mmlu +0.030 → +0.093 and bbh_mcq +0.024 → +0.087. Only commonsense_qa rises at every size: mmlu dips at 175M (+0.029), and bbh_mcq (+0.031) and acp_bench_mcq (+0.052) dip at 350M.
- **Significance:** mmlu and commonsense_qa are significant at every size, bbh_mcq's significant subtasks go from 6 to 12 of 17 (6/7/6/8/10/12, 90M → 1.7B), and acp_bench_mcq's stay at 4–5 of 7.
- **acp_bench_mcq's gain partly rides [the length tell](#the-length-tell-and-where-it-actually-is).** Under acc_norm, which the cell plots, only areach beats its pick-shortest rate at 1.7B (0.688 against 0.50; prog 0.350 and val 0.181 stay below theirs, medians over the six 1.7B cells). Plain `acc`, which the significance count reads, reaches 0.527 on prog, above its 0.469.
- **cultural_bench_easy's original is erratic**: +0.175 at 90M, −0.093 at 175M and +0.043 at 1.7B. Its rf − original (−0.165 at 90M, +0.049 with 7 of 19 tasks significant at 1.7B) reflects the original's instability below 600M, not the twin.

Follow-ups:

- Drop the two empty rfgm panels from `rf_gate_probe.png`, since no probe family has an rfgm set.
- Test acp_bench_mcq's three length-tell subtasks against their pick-shortest rate instead of the original's `acc`, so that the significance count stops crediting the tell.
- Check cultural_bench_easy's 90M original for a constant-answer bias before reading its small-size cells.

<!-- BEGIN auto:probe-survivors (analysis/rq00_task_reformulation/probe_survivors.py) -->
Probe survivors: which of the 59 candidate benchmarks in `auto_probe` clear the above-random gate (rule 1, read from the committed `predictivity` mask, cells that trained the language), over 50 languages. A cell counts languages (original | rf twin where one exists) — the population differs per cell (rule 13) — and the second table names the surviving benchmarks per language, the twin as `-rf`. Same items as `rf_gate_probe` (`compare.py --tag probe`), which measures how far above chance; this table is only the gate.

| benchmark | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---:|---:|---:|---:|---:|---:|
| acp_bench_cloze | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 |
| acp_bench_mcq | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| arabicmmlu | orig — | orig — | orig — | orig 0/1 · rf 0/1 | orig 0/1 · rf 0/1 | orig 0/1 · rf 1/1 |
| arc_mt | orig 0/11 | orig 0/11 | orig 0/11 | orig 2/11 | orig 10/11 | orig 11/11 |
| bangla | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 |
| bbh_cloze | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 |
| bbh_mcq | orig 1/1 · rf 1/1 | orig 1/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 1/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| blend_sample | orig 1/4 | orig 0/4 | orig 0/4 | orig 0/4 | orig 0/4 | orig 0/4 |
| blimp_nl | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| careqa | orig — | orig — | orig — | orig 0/1 · rf 0/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| ceval | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 |
| commonsense_qa | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| copal_id | orig — | orig — | orig — | orig 0/1 | orig 0/1 | orig 1/1 |
| cultural_bench_easy | orig 6/8 · rf 1/8 | orig 0/8 · rf 1/8 | orig 0/8 · rf 2/8 | orig 0/8 · rf 3/8 | orig 0/8 · rf 3/8 | orig 0/8 · rf 5/8 |
| cultural_bench_hard | orig 0/8 | orig 0/8 | orig 0/8 | orig 0/8 | orig 0/8 | orig 0/8 |
| evalita_llm | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| evalita_llm_extra | orig — | orig — | orig — | orig 0/1 | orig 0/1 | orig 1/1 |
| french_bench | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| french_bench_extra | orig — | orig — | orig — | orig 1/1 · rf 0/1 | orig 1/1 · rf 0/1 | orig 1/1 · rf 0/1 |
| headqa | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| ibero_arc | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 |
| ibero_cola | orig 1/2 | orig 0/2 | orig 0/2 | orig 0/2 | orig 1/2 | orig 1/2 |
| ibero_copa | orig — | orig — | orig — | orig 1/2 | orig 1/2 | orig 1/2 |
| ibero_nli | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| ibero_openbookqa | orig 2/3 | orig 3/3 | orig 3/3 | orig 3/3 | orig 3/3 | orig 3/3 |
| ibero_paraphrase | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| ibero_piqa | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 | orig 2/2 |
| ibero_siqa | orig 0/1 | orig 0/1 | orig 0/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| ibero_truthfulqa | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/2 | orig 0/2 | orig 0/2 |
| include_v2_en | orig 32/42 | orig 34/42 | orig 36/42 | orig 38/42 | orig 41/42 | orig 42/42 |
| include_v2_og | orig 13/42 | orig 16/42 | orig 23/42 | orig 29/42 | orig 31/42 | orig 35/42 |
| kmmlu | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 1/1 | orig 1/1 |
| kobest | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| m_mmlu | orig — | orig — | orig — | orig 0/27 · rf 23/27 | orig 0/27 · rf 24/27 | orig 0/27 · rf 27/27 |
| m_truthfulqa_mc1 | orig — | orig — | orig — | orig 6/26 | orig 6/26 | orig 3/26 |
| m_truthfulqa_mc2 | orig — | orig — | orig — | orig 24/24 | orig 24/24 | orig 24/24 |
| mathqa | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| mela | orig — | orig — | orig — | orig — | orig — | orig — |
| mmlu | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| mmmlu | orig — | orig — | orig — | orig 0/12 · rf 9/12 | orig 0/12 · rf 10/12 | orig 0/12 · rf 12/12 |
| multiblimp-extra | orig — | orig — | orig — | orig 3/3 | orig 3/3 | orig 3/3 |
| noreval | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| noreval_extra | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| oall_acva | orig — | orig — | orig — | orig 0/1 | orig 0/1 | orig 0/1 |
| oall_alghafa | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| oall_arabic_mmlu | orig — | orig — | orig — | orig 0/1 · rf 0/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| oall_exams | orig — | orig — | orig — | orig 0/1 · rf 0/1 | orig 0/1 · rf 1/1 | orig 0/1 · rf 1/1 |
| oall_mt | orig — | orig — | orig — | orig 1/1 | orig 1/1 | orig 1/1 |
| openbookqa | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| toksuite | orig 5/5 | orig 5/5 | orig 5/5 | orig 5/5 | orig 5/5 | orig 5/5 |
| toksuite_math | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| toksuite_stem | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| toxigen | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 | orig 0/1 |
| truthfulqa-multi_mc2 | orig — | orig — | orig — | orig 3/3 | orig 3/3 | orig 3/3 |
| truthfulqa_mc1 | orig — | orig — | orig — | orig 0/1 | orig 0/1 | orig 0/1 |
| truthfulqa_mc2 | orig 0/3 | orig 0/3 | orig 0/3 | orig 0/3 | orig 0/3 | orig 0/3 |
| turblimp | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |
| xquad | orig — | orig — | orig — | orig — | orig — | orig — |
| zhoblimp | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 | orig 1/1 |

| language | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| ar | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, oall_alghafa, oall_mt | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, oall_alghafa, oall_arabic_mmlu-rf, oall_exams-rf, oall_mt | arabicmmlu-rf, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, oall_alghafa, oall_arabic_mmlu-rf, oall_exams-rf, oall_mt |
| az | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| bg | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| bn | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| ca | ibero_arc, ibero_openbookqa, ibero_piqa | ibero_arc, ibero_openbookqa, ibero_piqa | ibero_arc, ibero_openbookqa, ibero_piqa | ibero_arc, ibero_nli, ibero_openbookqa, ibero_paraphrase, ibero_piqa, ibero_siqa, m_mmlu-rf, m_truthfulqa_mc2, truthfulqa-multi_mc2 | ibero_arc, ibero_nli, ibero_openbookqa, ibero_paraphrase, ibero_piqa, ibero_siqa, m_mmlu-rf, m_truthfulqa_mc2, truthfulqa-multi_mc2 | ibero_arc, ibero_nli, ibero_openbookqa, ibero_paraphrase, ibero_piqa, ibero_siqa, m_mmlu-rf, m_truthfulqa_mc2, truthfulqa-multi_mc2 |
| cs | include_v2_og | include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| da | — | — | — | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| de | — | — | include_v2_en | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| el | include_v2_en | include_v2_en | include_v2_en | include_v2_en | arc_mt, include_v2_en, include_v2_og | arc_mt, include_v2_en, include_v2_og |
| en | acp_bench_mcq-rf, bbh_mcq, bbh_mcq-rf, blend_sample, commonsense_qa-rf, cultural_bench_easy, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem | acp_bench_mcq-rf, bbh_mcq, bbh_mcq-rf, commonsense_qa-rf, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem | acp_bench_mcq-rf, bbh_mcq-rf, commonsense_qa-rf, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem | acp_bench_mcq-rf, bbh_mcq-rf, commonsense_qa-rf, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem, truthfulqa-multi_mc2 | acp_bench_mcq-rf, bbh_mcq, bbh_mcq-rf, commonsense_qa-rf, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem, truthfulqa-multi_mc2 | acp_bench_mcq-rf, bbh_mcq-rf, commonsense_qa-rf, cultural_bench_easy-rf, mathqa, mmlu-rf, openbookqa, toksuite, toksuite_math, toksuite_stem, truthfulqa-multi_mc2 |
| es | cultural_bench_easy, ibero_cola, include_v2_en, include_v2_og | ibero_openbookqa, include_v2_en, include_v2_og | ibero_openbookqa, include_v2_en, include_v2_og | arc_mt, cultural_bench_easy-rf, headqa, ibero_copa, ibero_openbookqa, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, truthfulqa-multi_mc2 | arc_mt, careqa-rf, cultural_bench_easy-rf, headqa, ibero_cola, ibero_copa, ibero_openbookqa, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, truthfulqa-multi_mc2 | arc_mt, careqa-rf, cultural_bench_easy-rf, headqa, ibero_cola, ibero_copa, ibero_openbookqa, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, truthfulqa-multi_mc2 |
| et | — | — | — | — | — | include_v2_en |
| eu | ibero_arc, ibero_piqa | ibero_arc, ibero_piqa | ibero_arc, ibero_piqa | ibero_arc, ibero_piqa | ibero_arc, ibero_piqa | ibero_arc, ibero_piqa |
| fa | toksuite | toksuite | toksuite | toksuite | include_v2_en, toksuite | include_v2_en, toksuite |
| fi | — | include_v2_en | include_v2_en | include_v2_en | include_v2_en | arc_mt, include_v2_en |
| fr | french_bench, include_v2_en | french_bench, include_v2_en | french_bench, include_v2_en | french_bench, french_bench_extra, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | french_bench, french_bench_extra, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | french_bench, french_bench_extra, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| gl | ibero_openbookqa | ibero_openbookqa | ibero_openbookqa | ibero_openbookqa | ibero_openbookqa | ibero_openbookqa |
| he | include_v2_en | include_v2_en | include_v2_en | include_v2_en | include_v2_en | include_v2_en, include_v2_og |
| hi | cultural_bench_easy, include_v2_en | include_v2_en | include_v2_en | include_v2_en, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 | cultural_bench_easy-rf, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| hr | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, multiblimp-extra | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, multiblimp-extra | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, multiblimp-extra |
| hu | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| id | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | copal_id, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| is | — | — | — | — | — | — |
| it | evalita_llm, include_v2_en, toksuite | evalita_llm, include_v2_en, toksuite | evalita_llm, include_v2_en, include_v2_og, toksuite | arc_mt, evalita_llm, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, toksuite | arc_mt, evalita_llm, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, toksuite | arc_mt, evalita_llm, evalita_llm_extra, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf, toksuite |
| ja | cultural_bench_easy, include_v2_en | include_v2_en | include_v2_en | include_v2_en, include_v2_og, mmmlu-rf | include_v2_en, include_v2_og, mmmlu-rf | cultural_bench_easy-rf, include_v2_en, include_v2_og, mmmlu-rf |
| ka | include_v2_en | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| kk | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| ko | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, kobest, mmmlu-rf | include_v2_en, include_v2_og, kmmlu, kobest, mmmlu-rf | include_v2_en, include_v2_og, kmmlu, kobest, mmmlu-rf |
| lt | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| lv | — | — | — | multiblimp-extra | multiblimp-extra | multiblimp-extra |
| ml | — | include_v2_en | include_v2_en | include_v2_en, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 |
| mr | — | — | — | m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 |
| ms | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og |
| ne | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| nl | blimp_nl, include_v2_en | blimp_nl, include_v2_en | blimp_nl, include_v2_en, include_v2_og | blimp_nl, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | blimp_nl, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | blimp_nl, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| no | noreval | noreval | noreval | m_mmlu-rf, noreval, noreval_extra | arc_mt, m_mmlu-rf, noreval, noreval_extra | arc_mt, m_mmlu-rf, noreval, noreval_extra |
| pl | include_v2_en | include_v2_en | include_v2_en | include_v2_en, include_v2_og | arc_mt, include_v2_en, include_v2_og | arc_mt, include_v2_en, include_v2_og |
| pt | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2, mmmlu-rf |
| ro | — | — | — | m_mmlu-rf, m_truthfulqa_mc2 | m_mmlu-rf, m_truthfulqa_mc2 | m_mmlu-rf, m_truthfulqa_mc2 |
| ru | cultural_bench_easy, include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| sk | — | — | — | m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 |
| sq | include_v2_en | include_v2_en | include_v2_en | include_v2_en, include_v2_og, multiblimp-extra | include_v2_en, include_v2_og, multiblimp-extra | include_v2_en, include_v2_og, multiblimp-extra |
| sr | include_v2_en | include_v2_en | include_v2_en | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 |
| sv | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | arc_mt, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| ta | include_v2_en | include_v2_en | include_v2_en | include_v2_en, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc1, m_truthfulqa_mc2 | include_v2_en, m_mmlu-rf, m_truthfulqa_mc2 |
| tr | include_v2_en, toksuite, turblimp | include_v2_en, include_v2_og, toksuite, turblimp | include_v2_en, include_v2_og, toksuite, turblimp | include_v2_en, include_v2_og, toksuite, turblimp | include_v2_en, include_v2_og, toksuite, turblimp | include_v2_en, include_v2_og, toksuite, turblimp |
| uk | include_v2_en | include_v2_en | include_v2_en, include_v2_og | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 | include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc2 |
| ur | — | — | — | include_v2_en | include_v2_en | include_v2_en, include_v2_og |
| vi | include_v2_en, include_v2_og | include_v2_en, include_v2_og | cultural_bench_easy-rf, include_v2_en, include_v2_og | cultural_bench_easy-rf, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc1 | cultural_bench_easy-rf, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc1 | cultural_bench_easy-rf, include_v2_en, include_v2_og, m_mmlu-rf, m_truthfulqa_mc1 |
| zh | cultural_bench_easy, include_v2_en, include_v2_og, toksuite, zhoblimp | include_v2_en, include_v2_og, toksuite, zhoblimp | include_v2_en, include_v2_og, toksuite, zhoblimp | include_v2_en, include_v2_og, m_mmlu-rf, mmmlu-rf, toksuite, zhoblimp | include_v2_en, include_v2_og, m_mmlu-rf, mmmlu-rf, toksuite, zhoblimp | include_v2_en, include_v2_og, m_mmlu-rf, mmmlu-rf, toksuite, zhoblimp |
<!-- END auto:probe-survivors -->

[probe_survivors.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/probe_survivors.csv)

The margins above leave open which candidates pass the gate at all, per language; this table answers that.
Population: the `predictivity` gate mask (seed 1904, every cell and data build), cells that trained the language, the 33 candidates over 47 languages.

Each cell counts languages, and the language count changes from row to row.

Key findings:

- **INCLUDE v2 has the broadest coverage.** include_v2_en clears the gate in 42 of 42 languages at 1.7B (32 of 42 at 90M), and include_v2_og in 35 of 42 (13 of 42 at 90M).
- **arc_mt switches on with size.** It clears in 0 of 11 languages through 350M, then 2 at 600M, 10 at 1B and 11 at 1.7B.
- **Three lettered originals never pass**: mmlu, commonsense_qa and acp_bench_mcq clear in 0 of 1 language at every size, while their `rf_` twins clear in 1 of 1. The bbh_mcq original passes only at 90M, 175M and 1B, its twin at every size.
- **cultural_bench_easy's original passes in 6 of 8 languages at 90M and in none from 175M up**, while its twin rises from 1 of 8 to 5 of 8 at 1.7B.
- **Ten candidates never pass in any language at any size:** acp_bench_cloze, bangla, bbh_cloze, ceval, cultural_bench_hard (0 of 8), haerae, openbookqa, toxigen, truthfulqa_mc2 (0 of 3) and turkishmmlu. blend_sample passes in 1 of 4 at 90M only, and xquad is generative and has no gate.

Follow-ups:

- Add a first-size-above-gate grid per candidate and language, as `first_size_above_random` draws for `auto`. It would condense the 47-row table above into one figure.

<!-- BEGIN auto:reformulations-gate (reformulations_gate.py --pool predictivity) -->
## The twins and the gate, with significance

Per family at 1.7B: the share of languages above the gate for the original and the twin, paired on the language; `twin only / original only` are the discordant languages McNemar's exact test is run on. Below, the gate's pass share per size on every task, the originals alone and the twins alone (`reformulations_gate.csv` carries the same three populations for mean DA-size, the reliable share and rq01's median R²). Regenerate with `python analysis/rq00_task_reformulation/reformulations_gate.py --pool predictivity`.

| family | twin | languages | original | twin | twin only / original only | p (McNemar) |
|---|---|---|---|---|---|---|
| acp_bench_mcq | rf | 7 | 0.00 | 0.71 | 5 / 0 | 0.0625 |
| arabicmmlu | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |
| bbh_mcq | rf | 17 | 0.00 | 0.59 | 10 / 0 | 0.00195 |
| belebele | rf | 105 | 0.05 | 0.82 | 82 / 1 | 1.74e-23 |
| belebele | rfgm | 59 | 0.05 | 1.00 | 56 / 0 | 2.78e-17 |
| careqa | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |
| commonsense_qa | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |
| cultural_bench_easy | rf | 19 | 0.00 | 0.37 | 7 / 0 | 0.0156 |
| french_bench_extra | rf | 1 | 0.00 | 0.00 | 0 / 0 | — |
| global_mmlu_full | rf | 37 | 0.03 | 0.95 | 34 / 0 | 1.16e-10 |
| include_base_44 | rf | 43 | 0.07 | 0.72 | 29 / 1 | 5.77e-08 |
| include_base_44 | rfgm | 43 | 0.07 | 0.74 | 30 / 1 | 2.98e-08 |
| m_mmlu | rf | 27 | 0.00 | 1.00 | 27 / 0 | 1.49e-08 |
| mmlu | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |
| mmmlu | rf | 12 | 0.00 | 1.00 | 12 / 0 | 0.000488 |
| oall_arabic_mmlu | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |
| oall_exams | rf | 1 | 0.00 | 1.00 | 1 / 0 | 1 |

| population | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| every task | 0.39 | 0.42 | 0.45 | 0.49 | 0.53 | 0.57 |
| originals only | 0.36 | 0.38 | 0.40 | 0.43 | 0.47 | 0.50 |
| twins only | 0.52 | 0.58 | 0.62 | 0.70 | 0.76 | 0.82 |

![The reformulations and the gate](reformulations_gate.png)
<!-- END auto:reformulations-gate -->

GitHub: [reformulations_gate.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.png) · [reformulations_gate.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate.csv) ·
[reformulations_gate_mcnemar.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_task_reformulation/reformulations_gate_mcnemar.csv)

This figure is the full version of the opening one: every size, plus the pooled gate share and the decision-accuracy reading.
Population: the `predictivity` gate mask; the McNemar rows pair each twin with its original on the task, and at 600M–1.7B the pooled rows split the 1,373 gated benchmark tasks into 1,041 non-twin tasks and 332 twins (1,372 and 1,040 at 90M–350M).

Key findings:

- **At 1.7B the large twins pass in significantly more tasks than their originals.** Twin-only : original-only is 82 : 1 for belebele rf (p 1.7e-23), 56 : 0 for belebele rfgm (2.8e-17), 34 : 0 for Global-MMLU rf (1.2e-10), 29 : 1 and 30 : 1 for INCLUDE rf and rfgm (5.8e-08, 3.0e-08), 10 : 0 for bbh_mcq (0.002) and 7 : 0 for cultural_bench_easy (0.016).
- **acp_bench_mcq does not reach p < 0.05** (5 : 0, p 0.06), and mmlu and commonsense_qa have one task each (p 1).
- **Across sizes, belebele and Global-MMLU are significant at every size, INCLUDE and bbh_mcq from 175M.** At 90M cultural_bench_easy is significant the other way: 12 of its 19 originals pass where the twin does not, against 1 the other way round.
- **The "originals only" row is every non-twin benchmark task** (511 of 1,041 pass at 1.7B, 0.49), not the twins' own originals, which pass in 9 of 230.
- **The twins do not rank designs better.** Mean DA-size (proxy → 1.7B, multi-axis pairs, gated at proxy and reference) is 0.486 / 0.505 / 0.499 / 0.501 / 0.550 on the twins from 90M to 1B, against 0.550 / 0.574 / 0.570 / 0.597 / 0.593 on the non-twin tasks. The twins' pairs are the deep scheme-A cells' alone (the L axis), so this is not a same-pairs comparison.

Follow-ups:

- Add the twins' own originals as a fourth pooled population, so that the "originals only" row stops reading as the twins' baseline.

The fuller reading (what the twins do to the gate, to DA-size, to the reliable
share and to the scaling fits) is [figure 3 of the gate-and-curves README](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate).
The former `twins_gate.*` name of these outputs is retired.

## Extensions from other sweeps

None. The reformulation exists on the ladder only (`predictivity`, the deep
scheme-A seed-1904 cells for the twin comparison): the 36-model sweep never
evaluated the `rf_` or `rfgm_` twins, and its numbers would not be pooled
with the ladder's in any case (a different harness, task set and reference
size).
