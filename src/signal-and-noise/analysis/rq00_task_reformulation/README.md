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

## Tier 2 — rewriting the items with Gemini (`rfgm_*`, driver built, data not yet produced)

Tier 1 only drops the letters. Here the item itself changes: Gemini turns
the question into one declarative sentence that stops where the answer goes
(a stem) and the four options into short, parallel continuations, in the
item's own language — the cloze formulation the FineWeb-2 / FineTasks work
found readable at small scale. Decided 2026-09-19: the three letter families
in all their languages (185 tasks, ~636k items), `gemini-3.8-flash` through
the Batch API, evaluated on the deep scheme-A seed-1904 ladder at every
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
counted and skipped rather than written — validates each item — JSON with a non-empty stem that ends
without whitespace or a colon, exactly four non-empty pairwise-distinct
choices, no choice appearing as a word in the stem (answer leak), the
majority Unicode script unchanged (the cheap "still in its language" check;
it cannot tell Spanish from English, step 5 covers the Latin-script
languages) — and writes `<task>.jsonl` (accepted) and
`_rejects/<task>.jsonl` (with reasons). A task file is written only from a
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
checkpoint + final of the deep scheme-A seed-1904 cells, jobs
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
per set, the table below gets both deltas, and the rq00 gate outputs
(`first_size_above_random`, `gate_margin_by_benchmark`) list the `rf_*` and
`rfgm_*` families as rows next to the originals: that is the direct read of
the gate impact. Every other RQ (decision accuracy, SNR) sees the new tasks
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

<!-- BEGIN auto:rf-compare (analysis/rq00_task_reformulation/compare.py) -->
Gate cells (median task margin over chance 0.25, trained languages, deep scheme-A seed-1904 ladder, from the ladder report; each set on the models that have the original and that twin scored — the original shown is the rf pairing). Cell: original acc, then per set `twin acc_norm (**Δ** = twin − original, n = tasks, sig)`; sig = tasks whose gain is significant for at least half of the size's models (two-proportion z-test of the original's acc against the twin run's own acc, p < 0.05; the acc_norm−acc offset is family-shaped, rf median +0.005, and exceeds half the plotted rf gain in 36 % of the pairs).

| family | 175M | 350M | 600M | 1B | 1.7B |
|---|---:|---:|---:|---:|---:|
| belebele | -0.007 · rf +0.030 (**+0.037**, n=59, sig=38) · rfgm — | -0.013 · rf +0.046 (**+0.059**, n=59, sig=55) · rfgm — | -0.007 · rf +0.058 (**+0.065**, n=59, sig=54) · rfgm — | +0.000 · rf +0.083 (**+0.083**, n=59, sig=57) · rfgm — | +0.011 · rf +0.109 (**+0.098**, n=59, sig=56) · rfgm — |
| global_mmlu_full | -0.006 · rf +0.006 (**+0.012**, n=29, sig=10) · rfgm — | -0.000 · rf +0.009 (**+0.010**, n=29, sig=17) · rfgm — | -0.004 · rf +0.015 (**+0.019**, n=29, sig=15) · rfgm — | -0.008 · rf +0.029 (**+0.037**, n=29, sig=24) · rfgm — | -0.010 · rf +0.048 (**+0.058**, n=29, sig=26) · rfgm — |
| include_base_44 | +0.004 · rf +0.006 (**+0.002**, n=36, sig=8) · rfgm — | +0.001 · rf +0.023 (**+0.022**, n=36, sig=9) · rfgm — | +0.001 · rf +0.027 (**+0.026**, n=36, sig=10) · rfgm — | +0.002 · rf +0.039 (**+0.037**, n=36, sig=14) · rfgm — | +0.005 · rf +0.052 (**+0.047**, n=36, sig=18) · rfgm — |

![family x size](rf_gate.png)

![per language](rf_gate_by_language.png)
<!-- END auto:rf-compare -->
