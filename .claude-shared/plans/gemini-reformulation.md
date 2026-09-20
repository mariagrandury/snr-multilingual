# Plan: Tier 2 — Gemini-rewritten twins (`rfgm_*`) of the three letter families

Status 2026-09-20: implemented (driver, generator, watcher flag, compare.py, docs),
on **Vertex AI via Application Default Credentials** — the organisation blocks
Generative Language API keys, so the Gemini Developer API is not available to us;
the Gemini run, the smoke test and the ladder evals are the user's steps below.
The living copy of the guide is the Tier 2 section of
`src/signal-and-noise/analysis/rq00_task_reformulation/README.md`.

## Context

Tier 1 (`rf_*`, committed) drops the A–D letters and scores the answer strings.
It lifts belebele from chance to +0.10 at 1.7B, Global-MMLU to +0.05, INCLUDE
to +0.05 — real but small, and the 175M–350M rungs stay near chance. The next
lever is the items themselves: turn each question into a declarative stem
with four short, parallel continuations, in the item's own language, with
Gemini. The user chose (2026-09-19): the 3 letter families in all their
languages (185 tasks, ~636k items), `gemini-3.8-flash` through the Batch API
(~$325 at list price, budget $450 with thinking at "low"), evaluated on the
deep scheme-A seed-1904 ladder at every evaluated checkpoint (the same
coverage as `rf_*`: ~625 jobs, ~385 node-hours), statement-stem style.

Everything downstream keys on the task name, so the new set is a third,
separate family prefix — `rfgm_` — that rides the exact `rf_` plumbing:
`--include_path` YAMLs, tasks.json entries with `metric: acc_norm`, a job
suffix, the ladder report, `compare.py`, and the rq00 gate (which picks up any
registered task automatically).

## Design decisions (fixed)

- **Names.** Tasks `rfgm_<task>`, benchmarks `rfgm_<family>`, group
  `auto_rfgm`, job suffix `-rfgm`, YAML dir `src/evals/tasks/rfgm/<family>/`.
  One prefix token: `tasks_for_benchmarks` matches `<benchmark>_…`, and
  `compare.py` maps a twin to its original with `removeprefix`, so `rf_gm_`
  would collide with the `rf_` strip.
- **Data.** One JSONL per task at
  `/capstor/store/cscs/swissai/infra01/msnr-harness/rf-data/rfgm/<task>.jsonl`
  (the dir is `infra01` group-rws with default ACLs; aromanou's jobs read it).
  Never pushed publicly — gold labels. Row schema, uniform across families:
  ```json
  {"id": 17, "text": "<passage>\n<stem>", "choices": ["…","…","…","…"], "gold": 2,
   "stem": "…", "context": "<passage or ''>", "subject": "anatomy|''",
   "orig_question": "…", "orig_choices": ["…"], "lang": "deu_Latn"}
  ```
  `text` is the full prompt (belebele passage verbatim + newline + stem;
  Global-MMLU/INCLUDE: stem only). Gold keeps the original position.
- **YAML** (one template for all three families, because the JSONL is uniform):
  ```yaml
  task: rfgm_belebele_deu_Latn
  dataset_path: json
  dataset_kwargs:
    data_files:
      test: /capstor/store/cscs/swissai/infra01/msnr-harness/rf-data/rfgm/belebele_deu_Latn.jsonl
  test_split: test
  output_type: multiple_choice
  num_fewshot: 0
  doc_to_text: text        # column name → raw value (harness task.py:1382-1386)
  doc_to_choice: choices   # column name → the list as-is (task.py:1464-1465)
  doc_to_target: gold      # column int → index into choices (task.py:1418-1422)
  metric_list: [acc, acc_norm]  (mean, higher_is_better)
  metadata: {version: 0.0}
  ```
  `load_dataset("json", data_files=<abs path>)` needs no network
  (harness `download`, task.py:993-997 splats `dataset_kwargs`; precedent
  `lm_eval/tasks/prost/corypaik_prost.yaml`). The stem must not end with a
  space: the harness appends `" " + choice` itself.
- **Gemini call.** Batch API on **Vertex AI**, `gemini-3.8-flash`,
  `responseMimeType: application/json` + `responseSchema`,
  `thinkingLevel: LOW` (no Gemini 3 model allows off; thinking bills as
  output), default temperature, one batch job per task (185 jobs, each file
  1–20 MB), 24 h turnaround target, 50 % price — the same rates as the
  Developer API, billed to the Cloud project.
- **Auth and storage (2026-09-20).** ADC (`GOOGLE_GENAI_USE_VERTEXAI=true`,
  `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`), the user running
  `gcloud auth application-default login --no-launch-browser` once. A Vertex
  batch job's source must be a `gs://` object — an uploaded file is a
  Developer-API-only source — so requests and answers live under
  `gs://<project>-msnr-rfgm/rfgm/`, created on first use and signed with the
  same ADC credential through `google-auth` + `requests` (no new dependency,
  no gcloud at run time). The location must be `global`: Gemini 3.x is served
  only from the global endpoint and a regional job 404s on the publisher
  model (verified 2026-09-20; batch prediction does accept `global`). Vertex
  ignores every field outside `request` and echoes the request instead of a
  key, so the request lines carry no `key` and `fetch` matches answers to
  items on the prompt text.
- **Where the driver runs.** The login node (`clariden-ln003`): it has
  outbound internet, the `snr` env, and the offline datasets cache the items
  come from. The driver is network-bound and light (no SIGKILL risk), and
  every mode is idempotent/resumable through a state file, so it can be
  re-run after any interruption.

## Files

| file | change |
|---|---|
| `src/evals/scripts/rewrite_items_gemini.py` (new, one mode-dispatched script) | `pilot` / `build` / `submit` / `status` / `fetch` / `report`; state in `rf-data/rfgm/_jobs.json`; reuses `make_rf_tasks.source_config` to find each task's dataset/config/split in the cache; `REVIEW_TASKS` = Spanish, Hindi, Turkish, Farsi, Greek, Arabic, Basque, Chinese in every family (23 tasks) for the pilot and `report --show` |
| `src/evals/scripts/make_rf_tasks.py` | `--set {rf,rfgm}` (default rf); for `rfgm` one `RFGM_YAML` template over the JSONLs that exist, entries `benchmark: rfgm_<fam>`, group `auto_rfgm`, benchmarks metadata `format: "Gemini statement rewrite of <fam> …"` |
| `configs/tasks.json` | generated: 185 `rfgm_*` entries, `groups.auto_rfgm`, 3 `benchmarks.rfgm_*` |
| `src/evals/tasks/rfgm/<family>/rfgm_<task>.yaml` | generated (185) |
| `src/pretrain/auto_evals_cscs.py` | `--reformulated` → `nargs="?", const="rf", choices=["rf","rfgm"]`; group `auto_<set>`, suffix `-<set>`; walltime weight `startswith(("rf_global_mmlu_full", "rfgm_global_mmlu_full"))` (same 14k rows × 4 strings) |
| `src/pretrain/compute_cost.py` | `kind_of` regex `-iter\d+(-rf|-rfgm)?$` |
| `src/signal-and-noise/analysis/rq00_task_reformulation/compare.py` | three sets: `SETS = {"rf": "reformulated (answer strings)", "rfgm": "rewritten (Gemini statements)"}`; `pool()` keeps a (model, base) row for a set when original + that set are both scored (pairwise, so rf stays complete while rfgm lands); `significance()` per set (plain acc read from `per_task/<set>_*`); panels `original | rf | rfgm | rf−orig | rfgm−orig` (5 columns, both figures); `rf_significance.csv` gains a `set` column; README table cell `orig → rf (Δ, sig) · rfgm (Δ, sig)` |
| `src/signal-and-noise/analysis/rq00_task_reformulation/README.md` | Tier 2 section becomes the guide below (prompt, commands, cost, validation); pilot spot-check table |
| `src/evals/README.md`, `src/evals/CLAUDE.md`, `src/pretrain/README.md`, `src/pretrain/CLAUDE.md` | the `rfgm` set next to `rf` in the existing sentences; `pip install google-genai` and the API-key file in the evals README setup |
| `requirements-ml.txt` | `google-genai` |

Not changed: `evaluate.sbatch` (`HARNESS_INCLUDE_PATH` already points at
`src/evals/tasks`, scanned recursively), `ladder_report.py` (`metric_for`
already picks `acc_norm` for any task whose entry says so), `above_random.py`
(any registered task with `n_options` is gated), `pretrain_progress.py`
(rf-unaware today; stays so).

## The guide (goes into the README's Tier 2 section)

### 0. Prerequisites (once)

1. A Google Cloud project with billing and `aiplatform.googleapis.com`
   enabled (batch prediction is not a free-tier feature).
2. ADC **on the login node** — it is a file the driver reads locally,
   `~/.config/gcloud/application_default_credentials.json`, so a login done
   on the laptop does not count. `curl -sSL https://sdk.cloud.google.com | bash`,
   then `gcloud auth application-default login --no-browser` (prints a
   `--remote-bootstrap` command for a laptop with gcloud and a browser; a
   laptop file can also just be scp'd over), then
   `gcloud auth application-default set-quota-project <project>`.
   Credentials refresh themselves; gcloud is not needed again. Note that
   `gcloud services enable` uses the CLI's own account, not ADC — run
   `gcloud auth login` first or enable aiplatform.googleapis.com in the
   console.
3. Export `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`,
   `GOOGLE_CLOUD_LOCATION=global` before `build` — the flag decides the
   request-file format and the driver refuses a mismatched backend.
   `google-genai` is in `requirements-ml.txt`, already in the snr env.
4. Vertex's batch quota is per project and region; the driver stops at the
   first `RESOURCE_EXHAUSTED` and is re-run later. The ADC account also
   needs `storage.buckets.create` (`roles/storage.admin`), or an admin
   creates the bucket and grants object access (`RFGM_GCS_BUCKET`).
5. `mkdir -p /capstor/store/cscs/swissai/infra01/msnr-harness/rf-data/rfgm`
   (inherits the infra01 group ACL). The Cloud Storage bucket
   (`$RFGM_GCS_BUCKET`, else `<project>-msnr-rfgm`) is created by the driver.

### 1. Pilot (~$0.40, 15 minutes)

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global
export HF_HOME=/iopsstor/scratch/cscs/mariagrandury/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python3.11 src/evals/scripts/rewrite_items_gemini.py pilot        # 10 items x the 23 REVIEW_TASKS
```
Synchronous `generate_content` with the same prompt/schema as the batch on
the review languages — Spanish, Hindi, Turkish, Farsi, Greek, Arabic,
Basque, Chinese (belebele spa_Latn / hin_Deva / tur_Latn / pes_Arab /
ell_Grek / arb_Arab / eus_Latn / zho_Hans, Global-MMLU es / hi / tr / fa /
el / ar / zh, INCLUDE spanish / hindi / turkish / persian / greek / arabic /
basque / chinese); prints original → rewritten side by side and writes
`analysis/rq00_task_reformulation/pilot_<task>.jsonl`. Read all 230 items:
language kept, stem answerable, gold not leaked, options parallel. Adjust the
prompt if not, re-run. This is the only step where the prompt is tuned.

### 2. Build the batch request files

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py build          # all 185 tasks
python3.11 src/evals/scripts/rewrite_items_gemini.py build --family belebele
```
For each task: load the split from the offline cache (`source_config`),
drop rows with a missing option (the same rule as `rf_` `process_docs`),
write `rf-data/rfgm/_requests/<task>.jsonl`, one line per item:
```json
{"key": "<task>:<row id>", "request": {"contents": [{"parts": [{"text": "<user content>"}]}],
 "config": {"system_instruction": {"parts": [{"text": "<SYSTEM>"}]},
            "response_mime_type": "application/json", "response_schema": {...},
            "thinking_config": {"thinking_level": "low"}}}}
```
Prints the token estimate per family and the cost at batch prices; `--dry-run`
prints only the estimate.

### 3. Submit

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py submit            # every built task not yet submitted
python3.11 src/evals/scripts/rewrite_items_gemini.py submit --max-jobs 40
```
Per task: upload `_requests/<task>.jsonl` to
`gs://<bucket>/rfgm/requests/<task>.jsonl`, then
`client.batches.create(model="gemini-3.8-flash", src=<that gs:// uri>, config={"display_name": "rfgm-"+task})`;
the answers land in `…/rfgm/requests/<task>/dest`. Job name, source and
destination go into `_jobs.json`. Idempotent: a task with a job is skipped.
Re-run until `status` shows 185 submitted.

### 4. Poll and fetch

```bash
python3.11 src/evals/scripts/rewrite_items_gemini.py status            # table: task, state, items, rejects
python3.11 src/evals/scripts/rewrite_items_gemini.py fetch             # every SUCCEEDED job not yet fetched
```
`fetch`: list the job's destination prefix in Cloud Storage and read every
`.jsonl` object there, match each `{"request","response"}` line back to its
item on the echoed prompt text (lines that match nothing are counted and
skipped), take `response.candidates[0].content.parts[0].text` as JSON,
**validate** (below), and write `rf-data/rfgm/<task>.jsonl` in the row
schema, plus `rf-data/rfgm/_rejects/<task>.jsonl` with the reason. A task
is written only when its job is SUCCEEDED, so a partial run never leaves a
half file. Jobs that end FAILED/EXPIRED are cleared from `_jobs.json` so
`submit` resubmits them; items rejected by validation are collected into one
retry request file per task and go through `submit --retry` once (a second
rejection drops the item — the count is in the report).

Validation per item: JSON parses; `stem` non-empty, no trailing whitespace
or colon; exactly 4 `choices`, each a non-empty string, pairwise distinct
after casefold/strip; no choice appears verbatim in the stem (answer leak);
majority Unicode script of stem + choices equals the original's (a cheap
"still in the right language" check; it cannot tell Spanish from English, so
the human spot-check in step 5 carries the Latin-script languages). `gold` is
copied from the original because the model is told to keep the option order;
validation only checks that choice `gold` is non-empty.

### 5. Spot-check (human, 30 minutes)

`report` prints per-task item counts, reject rates, length stats (median
stem/choice tokens), the tokens actually spent, and with `--show` 5 random
items of each of the 23 review tasks (the pilot's eight languages). Look
for: translated-to-English drift, stems that give the answer, options
collapsed to one word when the original was a clause. Reject rate > 5 % on a
task → read its rejects before evaluating.

### 6. Register the tasks

```bash
python3.11 src/evals/scripts/make_rf_tasks.py --set rfgm          # 185 YAMLs + tasks.json (idempotent)
```
Writes `src/evals/tasks/rfgm/<family>/rfgm_<task>.yaml` for every task whose
JSONL exists and the tasks.json entries (`benchmark: rfgm_<family>`,
`n_options: 4`, `metric: acc_norm`, `stages: ["pretraining"]`), group
`auto_rfgm`, benchmarks metadata. `n_items` is filled later by
`derive_task_options.py` from the first results.

### 7. Smoke-test in the harness: one watcher job (needs job approval)

`test_new_tasks.py` checks names against a harness clone and never passes
`--include_path`, so it cannot see repo-local tasks. The smoke test is the
watcher on one final checkpoint, every rfgm task, ~25 min on one node:

```bash
cd src/pretrain
python3.11 auto_evals_cscs.py --reformulated rfgm --name lm-175M-L8-deep-seed1904 --every 1000 --dry-run
python3.11 auto_evals_cscs.py --reformulated rfgm --name lm-175M-L8-deep-seed1904 --every 1000 --max-submit 1
```
Confirms the JSON loader works offline in the container, the
`text`/`choices`/`gold` columns resolve, both metrics land in
`per_task/rfgm_*/…/results_*.json` and `samples_rfgm_*.jsonl` carries the
rewritten docs. Note `--reformulated rfgm` raises `KeyError: 'auto_rfgm'`
until step 6 has run once: the group exists only when JSONLs do.

### 8. Launch the evals on the deep scheme-A seed-1904 ladder

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
python3.11 auto_evals_cscs.py --reformulated rfgm --arch deep --scheme A --seed 1904 --dry-run   # plan + job count
nohup python3.11 -u auto_evals_cscs.py --reformulated rfgm --arch deep --scheme A --seed 1904 \
      --max-submit 20 --watch 1800 >> /iopsstor/scratch/cscs/mariagrandury/auto_evals_rfgm_watch.log 2>&1 &
```
Same shape as the rf watcher (every 2nd checkpoint + final; jobs
`eval-<cell>-iter<N>-rfgm`, 23–45 min each on one node, ~625 jobs, ~385
node-h). Held-back tasks: `--retry-held` pass, as for rf. `pretrain_progress.py`
does not count rf/rfgm; progress = `squeue --me | grep rfgm` and
`grep -c COMPLETED` on `sacct --name`-style queries, or the watcher log.

### 9. Report and analysis

```bash
python3.11 src/pretrain/ladder_report.py --plot --publish --push-hf    # rfgm_* columns (acc_norm) join the report
bash scripts/refresh_analysis.sh                                        # rq00 gate + compare.py + figures + deck
```
`compare.py` then draws `rf_gate.png` / `rf_gate_by_language.png` with five
panels (original, rf, rfgm, rf−orig, rfgm−orig), the README table gets both
deltas with significance, and the rq00 gate outputs
(`first_size_above_random`, `gate_margin_by_benchmark`) list `rf_*` and
`rfgm_*` families as rows next to the originals — that is the direct read of
the gate impact. Every other RQ (DA, SNR) sees the new tasks too, same as rf.

### 10. Cost and time

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

Measured 2026-09-20 with the final prompt. The instruction plus schema is
~735 tokens on every request, the larger half of the input bill; `scaffold()`
measures it instead of assuming (a flat 150 understated the run by ~$130),
and implicit caching does not apply below Vertex's 2 048-token prefix
minimum. The rewriter is never sent the gold index.

`gemini-3.8-flash` batch: $0.375 / $1.875 per Mtok (through 2026-12-31; doubles
in 2027). Input = item + ~150 tokens of instruction and schema; output =
stem + 4 choices (+ JSON). Belebele passages are input only. Estimates are
bytes/3.5 with ±30 %. Wall time: batch jobs target 24 h; eval ~385 node-h at
the current throughput (rf took ~2 days with `--max-submit 20`).

## The prompt

System instruction (one string, reused for all items; `{family_note}` is
filled per family):

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
Family notes: belebele — "A passage is given; the stem must be answerable
from that passage alone. Do not rewrite or quote the passage." Global-MMLU —
"The subject is given; the stem may name it." INCLUDE — "The item may test
regional knowledge (driving rules, local history); keep the local terms."

User content per item:
```
Language: deu_Latn
Subject: anatomy            (Global-MMLU only)
Passage: <flores_passage>   (belebele only)
Question: <question>
Options:
1. <option a>
2. <option b>
3. <option c>
4. <option d>
```
`response_schema`: `{"type":"OBJECT","properties":{"stem":{"type":"STRING"},"choices":{"type":"ARRAY","items":{"type":"STRING"},"minItems":4,"maxItems":4}},"required":["stem","choices"]}`.

## Verification

1. `pilot` on the 23 review tasks: 230 items read by hand; the driver's
   validation passes ≥ 95 %.
2. `build --dry-run` prints ~636k items and a cost inside the table above.
3. `fetch` on the first SUCCEEDED job: no orphan lines reported, `wc -l` of
   the JSONL equals items − rejects; `python -c "import datasets; datasets.load_dataset('json', data_files={'test': …})"` with `HF_DATASETS_OFFLINE=1` loads it.
4. `make_rf_tasks.py --set rfgm --dry-run` lists 185 tasks; after the real
   run `python3.11 -c "from evals.scripts.utils.configs import tasks_for_benchmarks; …"` returns 185 for `auto_rfgm`; `auto_evals_cscs.py --reformulated rfgm --dry-run` names jobs `…-rfgm` and lists `rfgm_*` tasks; `--reformulated` alone still means rf.
5. Smoke test (step 7) produces `per_task/rfgm_belebele_deu_Latn/…/results_*.json` with `acc,none` and `acc_norm,none`, and `samples_rfgm_*.jsonl` at the run level.
6. After the first ~20 rfgm jobs: `ladder_report.py --plot --out-dir <scratch>` has `bench__rfgm_*` columns; `SNR_LADDER_DIR=<scratch> python3.11 analysis/rq00_task_reformulation/compare.py` draws five panels with the rfgm column partially filled and the rf column unchanged from today.
7. `review-snr` before the commit; the commit carries scripts, YAMLs, tasks.json, docs and the regenerated figures together (figure convention).
