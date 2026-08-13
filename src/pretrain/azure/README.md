# Pretraining and evaluating on Azure

This guide reproduces one cell of the SNR pretraining sweep on Azure —
**Apertus 175M, mixture 30% FineWeb-Edu / 70% FineWeb2-HQ, seed 28** — and
evaluates it on **hellaswag**, assuming you have never used Azure before.
Everything runs as [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/)
*command jobs*: you submit a YAML from your laptop, Azure boots a GPU node,
runs the job in a container, saves outputs to cloud storage and shuts the
node down. You never SSH anywhere and you only pay while a job runs.

The same scripts launch every other size × mixture × seed cell (step 9).

**The three stages** (do them in order — each one validates the next):

| Stage | What it proves | Compute | Time | Cost |
|---|---|---|---|---|
| Smoke test | container + Megatron fork + checkpointing work | 1× A100 | < 30 min | < $5 |
| Pilot (5.16B tokens) | data pipeline + full loop + conversion + eval work | 4× A100 | ~5–8 h | ~$100–150 |
| Full run (103.2B tokens) | the real cell | 4× A100 | ~4.5–6.5 days | ~$1,600–2,300 |

Per-size ballpark for full runs on 4× A100 (same data, same global batch):
175M ≈ 4.5–6.5 days; 350M ≈ 7–10 days; 600M ≈ 10–14 days; 1B ≈ 2.5–4 weeks.
The two big sizes are only sensible on Azure if you can get more/faster GPUs
(e.g. H100 quota) — the cluster used 24–84 GH200s per run.

**Prerequisites**

- An Azure account with a **pay-as-you-go subscription** (a free trial has no
  GPU quota). Create one at [azure.microsoft.com](https://azure.microsoft.com);
  you'll need a credit card. Budget ≥ $150 for smoke+pilot, ≥ $2,500 with the
  full run.
- A [wandb.ai](https://wandb.ai) account and API key (Settings → API keys) —
  W&B is the primary way you'll monitor training.
- No Hugging Face token needed: the tokenizer, FineWeb-Edu and FineWeb2-HQ
  are all public.
- The files referenced below all live in `src/pretrain/azure/`; run every
  command from that directory.

---

## 1. Install the Azure CLI and log in

```bash
# macOS
brew install azure-cli
# Ubuntu/Debian
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login                      # opens a browser
az account list --output table
```

Copy the `SubscriptionId` you want to bill (the UUID column) — you'll paste
it into `env.sh` next.

## 2. Configure names and create the workspace

Edit `env.sh`: set `AZ_SUBSCRIPTION` to your subscription id and
`AZ_LOCATION` to the region where you'll request quota in step 3 (pick one
and stick to it — quota, storage and compute are all per-region. For Europe
try `francecentral` or `swedencentral`; `eastus` / `southcentralus` often
have better GPU availability). Then:

```bash
source env.sh
bash setup_azure.sh
```

What this creates (all inside one **resource group**, a folder you can later
delete in one command):

- **Workspace** (`$AZ_WS`) — the Azure ML project hub. It auto-creates a
  **storage account** whose blob container is exposed to jobs as the
  `workspaceblobstore` *datastore*: that's where the tokenized data,
  checkpoints, converted models and eval results will live.
- **Environments** — pointers to the Docker images jobs run in:
  `apertus-nemo` (NGC NeMo 25.11, the x86 build of the same image the CSCS
  cluster uses) and `apertus-eval` (vLLM, for lm-eval).
- **Compute clusters** — `gpu-train` (one `Standard_NC96ads_A100_v4` node:
  4× A100 80GB, ~$14.7/h) and `gpu-single` (one `Standard_NC24ads_A100_v4`:
  1× A100, ~$3.7/h). Both have `min_instances: 0`: nodes exist only while a
  job runs, so an idle setup costs ~$0.

!!! warning "The compute creation fails until you have quota"
    A brand-new subscription has **0 GPU quota** — do step 3 first if
    `setup_azure.sh` fails on the compute step, then re-run it (it's
    idempotent).

## 3. Request GPU quota (the step that involves waiting)

Azure meters GPU access in *vCPUs of a VM family*. You need the
**`Standard NCADS_A100_v4 Family vCPUs`** family (sometimes rendered
"NCADSA100v4") in your `$AZ_LOCATION`:

1. Go to the [Azure portal](https://portal.azure.com) → search **Quotas** →
   **Compute** → filter by your region.
2. Search for `NCADS_A100_v4`, tick it, click **New quota request**.
3. Request **120 vCPUs** (= the 96-core 4-GPU machine + the 24-core 1-GPU
   machine). If you only get 96, skip `gpu-single` and change
   `compute: azureml:gpu-single` to `azureml:gpu-train` in
   `jobs/convert.yml` and `jobs/eval.yml` (or pass
   `--set compute=azureml:gpu-train` at submit time) — everything still
   works, small jobs just run on the bigger node.

Requests ≤ ~100 cores are usually auto-approved within minutes to hours;
larger ones open a support ticket (days). If denied, try another region
(then update `AZ_LOCATION` and re-run setup). Verify with:

```bash
az ml compute list-sizes --location $AZ_LOCATION --output table | grep NC96ads
```

Cost-saving option: `tier: low_priority` in `compute-*.yml` runs the same
hardware at Spot prices (often 3–5× cheaper) but the node can be evicted at
any time — fine for training here because resubmitting a job resumes from
the last checkpoint (step 7), annoying for the one-shot conversion/eval jobs.

## 4. Smoke test (do not skip)

This runs 20 training iterations on **mock data** on 1 GPU — it exercises
the exact code path of the real run (swiss-ai Megatron fork, xIELU/QK-norm
kernels, AdEMAMix optimizer, `torch_dist` checkpoint save) for pocket change:

```bash
export WANDB_API_KEY=<your key>       # optional but recommended
az ml job create --file jobs/train-smoke.yml $AZ_ML_ARGS \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY \
  --web                               # opens the job page in Azure ML Studio
```

Stream the logs from your terminal (or watch them on the job page under
**Outputs + logs → user_logs/std_log.txt**):

```bash
az ml job stream --name <job name printed above> $AZ_ML_ARGS
```

The first run takes ~10 min before training starts (image pull + Megatron
clone + tokenizer download + xIELU kernel compile). Healthy output looks
like:

```
 iteration       1/      20 | ... | lm loss: 1.18E+01 | ...
 iteration       2/      20 | ... | lm loss: 1.17E+01 | ...
```

Loss starting near 11.8 (= ln 131072, random init over the vocab) and
decreasing is the success signal, plus `successfully saved checkpoint at
iteration 20` at the end. If `WANDB_API_KEY` was set you'll also see the run
appear in W&B under `mariagrandury-epflnlp/data-mix-small`.

## 5. Prepare the pilot data

The cluster consumed mixtures pre-tokenized at CSCS; on Azure
`prepare_data.py` recreates them from the public sources: it streams
**FineWeb-Edu** (English) and **FineWeb2-HQ** per language, tokenizes with
`alehc/swissai-tokenizer` via Megatron's `preprocess_data.py`, and writes
`.bin`/`.idx` shards plus a `data_path.txt` manifest of mixture weights.

!!! note "Approximation"
    The exact language composition of the cluster's FineWeb2 corpus is not
    recoverable from this repo. The documented approximation is the
    project's `main` language group minus English
    (`configs/languages.json`: es ru hi zh ja ar vi tr th sw eu), weighted
    by each language's corpus size on the Hub; languages missing from
    FineWeb2-HQ fall back to unfiltered FineWeb-2. The realized per-source
    token counts are recorded in `data_path.txt`.

```bash
az ml job create --file jobs/prep.yml $AZ_ML_ARGS
az ml job stream --name <job name> $AZ_ML_ARGS
```

~1–2 h on the 96-core node (~$20). Verify the output in **Azure ML Studio →
Data → Datastores → workspaceblobstore → Browse** → `tokenized/mix_30_70/pilot`:
you should see `fineweb_edu_text_document.bin/.idx`, one `fw2_<lang>_...`
pair per language, and `data_path.txt`.

## 6. Pilot training run

```bash
az ml job create --file jobs/train-pilot.yml $AZ_ML_ARGS \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY
```

2,500 iterations × 504 × 4096 tokens ≈ 5.16B tokens, ~5–8 h on 4× A100
(gradient accumulation 18 to keep the cluster's global batch of 504).
Monitor in **W&B** (`data-mix-small` project, run
`apertus-175M-fwEdu30-fw270-seed28-azure-<id>`): `lm loss` should fall
steeply below ~7 in the first few hundred iterations and grind toward ~3–4;
`throughput/tokens_per_sec` should be stable after warmup.

Two behaviours worth knowing before the full run:

- **Resume = resubmit.** The job's checkpoint output is pinned to a fixed
  storage path and Megatron `--load`s from it: if the job dies (or a Spot
  node is evicted), `az ml job create -f jobs/train-pilot.yml ...` again and
  it continues from the last checkpoint. Iterations already completed are
  never retrained.
- The tokenized data is **downloaded to the node's local NVMe** at job start
  (`mode: download`, a few minutes) — don't change it to `ro_mount`;
  memory-mapped `.bin` reads over a blob mount are pathologically slow.

## 7. Full data + full training run

```bash
# ~415GB of tokenized data; several hours on the 96-core node
az ml job create --file jobs/prep.yml $AZ_ML_ARGS \
  --set display_name=prep-data-full \
        environment_variables.TOTAL_TOKENS_B=103.2 \
        environment_variables.EDU_CONFIG=sample-350BT \
        outputs.tokenized.path=azureml://datastores/workspaceblobstore/paths/tokenized/mix_30_70/full

# the real thing: 50,000 iters, checkpoint every 2,000 (~every 4 h)
az ml job create --file jobs/train-full.yml $AZ_ML_ARGS \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY
```

!!! danger "Cost"
    This job runs ~4.5–6.5 days at ~$14.7/h ≈ **$1,600–2,300**. Watch the
    first hour in W&B before walking away; you can kill it any time with
    `az ml job cancel --name <job name> $AZ_ML_ARGS` and lose at most the
    iterations since the last checkpoint (≤ 2,000 ≈ 4 GPU-hours). Resume by
    resubmitting.

## 8. Convert and evaluate on hellaswag

**Convert** the final Megatron checkpoint to a Hugging Face snapshot
(`ApertusForCausalLM`) — a few minutes on 1 GPU:

```bash
az ml job create --file jobs/convert.yml $AZ_ML_ARGS
```

For the pilot's checkpoint instead, override the paths and step:

```bash
az ml job create --file jobs/convert.yml $AZ_ML_ARGS \
  --set inputs.checkpoints.path=azureml://datastores/workspaceblobstore/paths/runs/apertus-175M-fwEdu30-fw270-seed28-pilot/checkpoints \
        environment_variables.CKPT_STEP=2500 \
        outputs.hf_model.path=azureml://datastores/workspaceblobstore/paths/models/apertus-175M-fwEdu30-fw270-seed28-pilot/iter_0002500
```

**Evaluate.** `eval.sh` reproduces the cluster's lm-eval invocation exactly
(vLLM backend, swiss-ai lm-evaluation-harness fork for the task definitions,
`add_bos_token=True`, no chat template, `--batch_size auto:20
--max_batch_size 32 --log_samples --write_out --trust_remote_code
--confirm_run_unsafe_code --gen_kwargs max_gen_toks=2048`, TP=PP=DP=1) and
writes results in the cluster's directory layout:

```bash
az ml job create --file jobs/eval.yml $AZ_ML_ARGS
az ml job stream --name <job name> $AZ_ML_ARGS
```

The job log ends with the parsed scores, e.g.:

```
hellaswag {'acc,none': 0.3311, 'acc_stderr,none': 0.0047, 'acc_norm,none': 0.3902, ...}
```

Sanity expectations: random = 0.25; a 175M model after the full 103B tokens
lands roughly at ~0.30–0.35 `acc` / ~0.35–0.42 `acc_norm`; the pilot
checkpoint will be barely above random — that's normal at 5B tokens. Other
tasks: `--set environment_variables.TASKS=hellaswag,hellaswag_es` (any
comma-separated lm-eval names; the multilingual `hellaswag_*` variants are
in the fork).

**W&B push happens inside the eval job** when you pass your key
(`--set environment_variables.WANDB_API_KEY=$WANDB_API_KEY`, which the
launchers do automatically when `WANDB_API_KEY` is exported): the job runs the
repo's standard `push_all_results.py` against its own results, appending one
step to the model's curve in `mariagrandury-epflnlp/snr-experiments` — same
project as the cluster evals. Without a key, push later from your laptop:

```bash
az ml job download --name <eval job name> --output-name results --download-path /tmp/azure-evals $AZ_ML_ARGS
cd ../../..   # repo root
LOGS_ROOT=/tmp/azure-evals/named-outputs/results \
  python src/evals/scripts/push_all_results.py \
  --entity mariagrandury-epflnlp --project snr-experiments \
  --name apertus-175M-fwEdu30-fw270-seed28-iter50000
```

(The NAME must be a `configs/models.json` key + `-iter<N>` — that's how the
pusher resolves the tokens/FLOPs axes. All 36 sweep cells plus the two
bilingual cells already have entries.)

## 9. Auto-evals every 5 checkpoints

To see whether a training is progressing well with more information than the
loss curve, every 5th saved checkpoint (with `SAVE_INTERVAL=2000`: iters
10000, 20000, 30000, 40000, 50000) gets evaluated automatically on the
**`auto`** benchmark group from `configs/tasks.json` — `hellaswag`,
`hellaswag_ru`, `global_mmlu_full_en`, `global_mmlu_full_ru`,
`global_piqa_completions_eng_latn`, `global_piqa_completions_rus_cyrl` —
and pushed to the same W&B project (`snr-experiments`) as every other eval.
Edit the group in `configs/tasks.json` to change the list.

Start the watcher in a terminal alongside your training run:

```bash
source env.sh && export WANDB_API_KEY=<key>
python auto_evals.py --watch 600        # one pass every 10 min; Ctrl-C to stop
```

Each pass lists the checkpoints in blob storage and, for every due iteration,
submits the one step that's missing: first a `convert.yml` job (Megatron →
HF), then — on a later pass, once the snapshot exists — an `eval.yml` job with
`TASKS=auto`. Everything already evaluated or currently running is skipped, so
the watcher is idempotent: stop it, restart it, run it twice — nothing
duplicates, and a killed pass just means the next one picks up the remainder.
`--dry-run` previews the submissions; `--every N` changes the cadence;
`--name <cell>` watches a single run. In W&B, the auto-eval points appear on
the model's per-benchmark curves (x-axis: flops/tokens/iter) as training
progresses.

## 10. The minimal plan: bilingual EN+RU, 90M and 1.7B

The first real experiment on Azure: two 2-language models — FineWeb-Edu
(English) + FineWeb2-HQ Russian, 50/50 by bytes — at **90M** (L15×d768,
92.9M non-embedding params) and **1.7B** (L30×d2304, 1.67B), both trained
like the sweep (50,000 iters ≈ 103.2B tokens, GBS 504, seq 4096; hyperparams
in `../hyperparams_deep.json`, cells in `configs/models.json` as
`apertus-{90M,1.7B}-fwEdu50-fw2ru50-seed28`), auto-evaluated every 5
checkpoints.

```bash
source env.sh && export WANDB_API_KEY=<key>

# 1. Bilingual mixture (~415GB tokenized; reusable by both sizes)
az ml job create --file jobs/prep.yml $AZ_ML_ARGS \
  --set display_name=prep-data-enru \
        environment_variables.TOTAL_TOKENS_B=103.2 \
        environment_variables.EDU_RATIO=0.5 \
        environment_variables.EDU_CONFIG=sample-350BT \
        environment_variables.EXTRA_ARGS="--languages ru" \
        outputs.tokenized.path=azureml://datastores/workspaceblobstore/paths/tokenized/mix_enru_50_50/full

# 2. Launch the trainings (queue both; with max_instances 1 they run in sequence)
python launch_azure_trainings.py --size 90M  --seed 28
python launch_azure_trainings.py --size 1.7B --seed 28

# 3. Auto-evals while they train
python auto_evals.py --watch 600
```

Expectations on the 4×A100 `gpu-train` cluster: **90M ≈ 2–3 days
(~$700–1,000)**; **1.7B ≈ 3–4 weeks (~$8–12k)** — for the 1.7B seriously
consider requesting H100-class quota (e.g. `Standard_ND96isr_H100_v5`, brings
it to ~4–5 days) or raising `max_instances` is no help (single-node training).
MBS is preset per size (21 for 90M, 2 for 1.7B) so the global batch of 504
divides evenly on 1 or 4 GPUs. The final checkpoints then take the same
convert → full-eval path as any other cell (`--ckpts final|full_eval`).

## 11. Launching the remaining cells

Every other size × mixture × seed cell uses the same `train-full.yml` /
`eval.yml` templates; the launchers fill in the per-cell architecture
(from `../hyperparams_deep.json`), mixture and paths — same filter flags as
the cluster's `launch_trainings.py`:

```bash
source env.sh && export WANDB_API_KEY=<key>

# preview, then launch: all three mixtures of 350M seed 28
python launch_azure_trainings.py --size 350M --seed 28 --dry-run
python launch_azure_trainings.py --size 350M --seed 28

# evals: the canonical 10-checkpoint set, full pretraining task list
python launch_azure_evals.py --size 350M --seed 28 --ckpts full_eval --tasks pretraining_full --dry-run
```

Each cell's mixture must exist first (`prep.yml` with
`environment_variables.EDU_RATIO=0.6` → `tokenized/mix_60_40/full`, etc.),
and each evaluated checkpoint needs a `convert.yml` run first. With one
`max_instances: 1` cluster, submitted jobs queue and run one at a time —
raise `max_instances` (and your quota) to run cells in parallel.

## 12. Where everything is stored (and getting it out)

All artifacts live in the workspace's blob storage under `workspaceblobstore`:

| Path | Contents |
|---|---|
| `tokenized/mix_<edu>_<fw2>/<scale>` | `.bin`/`.idx` shards + `data_path.txt` |
| `runs/<exp_name>/checkpoints` | Megatron `torch_dist` checkpoints (`iter_*/`) |
| `runs/<exp_name>/logs` | TensorBoard + W&B offline files |
| `models/<exp_name>/iter_<N>` | converted HF snapshots |
| `eval_logs/<entity>/<project>/<NAME>/harness/eval_*/` | lm-eval results + samples |

Browse it in Studio (**Data → Datastores → workspaceblobstore → Browse**) or
[Azure Storage Explorer](https://azure.microsoft.com/en-us/products/storage/storage-explorer).

**Register converted models** in the workspace's model registry so they're
versioned and survive any cleanup of the run directories:

```bash
az ml model create --name apertus-175M-fwEdu30-fw270-seed28 --version 50000 \
  --type custom_model \
  --path azureml://datastores/workspaceblobstore/paths/models/apertus-175M-fwEdu30-fw270-seed28/iter_0050000 \
  $AZ_ML_ARGS
```

**Download anything** with `az ml job download` (per-job outputs, as in
step 8) or `azcopy` for bulk paths.

**Optional: push to the Hugging Face Hub** with the sweep's naming
convention (repo per cell, branch per checkpoint — same as
`../conversion/push-snr.py` uses on the cluster):

```bash
huggingface-cli upload <your-org>/apertus-175M-fwEdu30-fw270-seed28 \
  ./iter_0050000 . --revision stage1-step-50000 --private
```

## 13. Teardown

Compute scales to zero by itself; a parked setup costs only blob storage
(~$12/month for ~600GB). To stop even that:

```bash
az ml compute delete --name gpu-train $AZ_ML_ARGS --yes    # keeps all data
az group delete --name $AZ_RG --yes                        # deletes EVERYTHING
```

!!! danger
    `az group delete` destroys the storage account too — checkpoints,
    models, eval results. Download (or push to HF / register elsewhere)
    anything you care about first.

## 14. Common mistakes

- **Compute create fails / job queues forever** → you have no quota in that
  region (step 3), or you edited `AZ_LOCATION` after creating the workspace
  (everything is per-region — start over in one region).
- **NCCL "unhandled system error" at startup** → shared memory too small.
  The train jobs set `resources.shm_size: 64g`; keep it if you copy the YAML.
- **Training is 10–100× slower than expected** → the data input was changed
  from `mode: download` to a mount. Don't stream `.bin` memmaps from blob.
- **Restarted pilot suddenly trains 50,000 iters** (or vice versa) → you
  reused another stage's checkpoint dir: `TRAINING_STEPS` and the output
  paths must change together. Smoke/pilot/full each pin their own
  `runs/...` paths for exactly this reason.
- **Conversion fails with a `transformers` import/version error** → run it
  via `jobs/convert.yml` only; it pins `transformers==4.57.6` inside its own
  job container (the training and eval images keep their own versions).
- **Eval crashes at vLLM init with a KV-heads error** → someone raised
  `TP`. Keep `TP=1`: the 350M/1B models have 5/7 KV heads, indivisible by
  higher TP.
- **`push_all_results.py` silently pushes nothing** → the eval `NAME`
  doesn't match a `configs/models.json` key + `-iter<N>`, or `LOGS_ROOT`
  doesn't point at the directory that *contains* `<entity>/<project>/...`.
- **No W&B run appeared** → `WANDB_API_KEY` wasn't passed to the job
  (`--set environment_variables.WANDB_API_KEY=$WANDB_API_KEY`); the job
  still trains, logging only to TensorBoard (`runs/<exp>/logs/tensorboard`).
