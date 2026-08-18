# Pretraining and evaluating on Azure

This guide sets up Azure from zero and runs predictivity-sweep training cells
on it, assuming you have never used Azure before. Everything runs as
[Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/)
_command jobs_: you submit a YAML from your laptop, Azure boots a GPU node,
runs the job in a container, saves outputs to cloud storage and shuts the
node down. You never SSH anywhere and you only pay while a job runs.

Azure runs the **exact same training logic as the CSCS cluster**: both
platforms build the Megatron command from the shared
[`megatron_args.sh`](../megatron_args.sh) and are driven by the same launcher
([`launch_trainings.py`](../launch_trainings.py) — `cscs` or `azure` as the
first argument). Only the wrapper differs
([`launch_pretraining_azure.sh`](../launch_pretraining_azure.sh): torchrun on
one node, vs sbatch/srun on the cluster).

**The three stages** (do them in order — each one validates the next):

| Stage | What it proves | Cost |
| ----- | -------------- | ---- |
| Smoke test (mock data) | container + Megatron fork + checkpointing work | < 30 min, < $2 |
| First real cell (90M, L2) | data pipeline + full loop + conversion + eval work | hours, ~tens of $ |
| The sweep | the real thing | see the compute-budget sheet |

Per-size cost anchors, measured at 103.2B tokens on `gpu-nc80-lp`
(2× H100 94GB, fixed low-priority meter ~$3.63/h): 175M ≈ 3 days / ~$240;
350M ≈ 5 days / ~$400; 600M ≈ 7 days / ~$620. Predictivity budgets are
D(N) = 100 × N tokens — scale those anchors by (100 N / 103.2B), e.g. the
175M predictivity cell (17.6B tokens) is ~$45. The 1B and 1.7B rungs belong
on the UK South `gpu-nd96-spot` pool (8× H100 + InfiniBand) — see the
compute-budget sheet next to the plan for the full budget.

**Prerequisites**

- An Azure account with a **pay-as-you-go subscription** (a free trial has no
  GPU quota). Create one at [azure.microsoft.com](https://azure.microsoft.com);
  you'll need a credit card.
- A [wandb.ai](https://wandb.ai) account and API key (Settings → API keys) —
  W&B is the primary way you'll monitor training.
- No Hugging Face token needed: the tokenizer is public, and all training
  data ships pre-tokenized from CSCS (§5) — nothing is downloaded from the
  HF Hub.
- CSCS access (login node) — the data mixtures are built there (§5).
- The files referenced below all live in `src/pretrain/`; run every
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
it into `azure/env.sh` next.

## 2. Configure names and create the workspace

Edit `azure/env.sh`: set `AZ_SUBSCRIPTION` to your subscription id. The
regions are already decided for this project (quota, storage and compute are
all per-region — see the compute-budget sheet): the **primary workspace is
Spain Central** (`snr-es-rg`/`snr-es-ws`, NC80adis H100 low-priority — every
size ≤600M plus evals) with a second workspace in **UK South**
(`snr-uk-rg`/`snr-uk-ws`, ND96isr 8×H100 Spot — the 1B/1.7B pool); §9 sets
up the UK one. If you're adapting this guide to another subscription, pick a
region where _your_ subscription can deploy the SKUs and stick to it. Then:

```bash
source azure/env.sh
bash azure/setup.sh
```

**Credentials & W&B config.** `source azure/env.sh` loads the Azure names.
The W&B **entity** is the constant `mariagrandury-epflnlp` (hardcoded in
`megatron_args.sh`) and the training **project** comes from
`configs/hf_wandb.json` (currently `msnr`), injected into every job by the
launcher. The two **secrets** — `WANDB_API_KEY` and your Hugging Face
token — are _not_ in any file and are never re-entered in this guide: they
live in your laptop's shell env / HF login. Each step that needs them just
verifies they're present first, e.g.:

```bash
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"
huggingface-cli whoami >/dev/null 2>&1 && echo "✓ HF logged in" || echo "✗ run: huggingface-cli login"
```

What this creates (all inside one **resource group**, a folder you can later
delete in one command):

- **Workspace** (`$AZ_WS`) — the Azure ML project hub. It auto-creates a
  **storage account** whose blob container is exposed to jobs as the
  `workspaceblobstore` _datastore_: that's where the tokenized data,
  checkpoints, converted models and eval results will live.
- **Environments** — pointers to the Docker images jobs run in:
  `apertus-nemo` (NGC NeMo 25.11, the x86 build of the same image the CSCS
  cluster uses) and `apertus-eval` (vLLM, for lm-eval).
- **Compute clusters** — every `compute-*.yml` whose SKU the region offers
  (the rest are skipped with a warning). In Spain Central that's
  `gpu-nc80-lp` (`Standard_NC80adis_H100_v5`: 2× H100 94GB at the fixed
  low-priority meter, ~$3.63/h); the UK South workspace instead gets
  `gpu-nd96-spot` (§9). Clusters have `min_instances: 0`: nodes exist only
  while a job runs, so an idle setup costs ~$0.

!!! warning "The compute creation fails until you have quota"
A brand-new subscription has **0 GPU quota** — do step 3 first if
`azure/setup.sh` fails on the compute step, then re-run it (it's
idempotent).

## 3. Request GPU quota (the step that involves waiting)

Azure meters GPU access in _vCPUs of a VM family_. This project needs
(both requests filed 2026-08-13; amounts and rationale in the
compute-budget sheet):

1. **`Standard NCadsH100v5 Family vCPUs`** in **Spain Central** — 160 cores
   = 2 NC80adis nodes. The `gpu-nc80-lp` cluster bills the _low-priority_
   meter, whose quota is a **separate counter** in Azure ML Studio →
   Quota — check it once the workspace exists (it often has a non-zero
   default).
2. **`Standard NDSH100v5 Family vCPUs`** in **UK South** — 96–192 cores
   (1–2 ND96isr nodes to start; the predictivity plan scales to 16) plus
   its Spot counter.

File absent families via Help + Support → _Service and subscription limits
(quotas)_; H100-class requests open a support ticket (days, not minutes),
so file early. Verify what the region offers with:

```bash
az ml compute list-sizes --location $AZ_LOCATION --output table | grep NC80adis
```

Both clusters already use the discounted tiers (`tier: low_priority` in
`compute-nc80-lowpri.yml` / `compute-nd96-spot.yml` — 77–81% below
dedicated). The trade-off: a node can be evicted at any time. That's fine
for training because resubmitting a job resumes from the last checkpoint;
if an eviction ever bites a one-shot conversion/eval job, just resubmit it.
Dedicated is the fallback only if evictions thrash — remove the `tier:`
line and re-create the compute.

## 4. Smoke test (do not skip)

This runs 20 training iterations on **mock data** on one node — it exercises
the exact code path of the real run (swiss-ai Megatron fork, xIELU/QK-norm
kernels, AdEMAMix optimizer, `torch_dist` checkpoint save) for pocket change:

```bash
source azure/env.sh   # Azure names; W&B key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (optional for the smoke test)"
az ml job create --file azure/jobs/smoke.yml $AZ_ML_ARGS \
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
appear in W&B under `mariagrandury-epflnlp/msnr`.

## 5. Generate the data mixtures (CSCS) and ship them to Azure

**All training data is built from the corpora curated at CSCS** — DCLM-edu
(English) and FineWeb-2-HQ (multilingual), the filtered swiss-ai variants on
`/capstor` — and tokenized *there*; nothing is downloaded from the HF Hub.
Two scripts under `data/` own the pipeline:

- [`data/create_data_mixture.py`](data/create_data_mixture.py) — the worker:
  streams the parquet sources, tokenizes with `swiss-ai/Apertus-70B-2509`,
  writes Megatron `.bin`/`.idx`, builds the fixed validation set, and
  excludes its rows from training via a manifest. Resumable after
  preemption.
- [`data/build_data_mixtures.py`](data/build_data_mixtures.py) — the driver:
  turns the language schemes (`data/language_sets_scheme{A,B}.json`) into
  per-build `create_data_mixture.py` calls with the right token targets.

### 5a. Build on CSCS — L2 first, then the rest

The English + L2 (Russian) pair unblocks the first real cells (§8), so build and
ship it first; the remaining multilingual builds run while those models
already train:

```bash
# CSCS login node, from src/pretrain/data/ ($OUT on /capstor or /iopsstor)
# 1. Fixed validation set + exclusion manifest (once — every build needs it)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage validation

# 2. The shared English dataset + the L2 FineWeb-2 build (rus_Cyrl)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage english
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage fineweb --settings 2

# 3. All remaining language settings (start §8 first, then run this)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage fineweb \
    --settings 8,15,30,50,100
```

Targets (printed by `--dry_run` first): 184.0B English; 92.0B FineWeb-2 where
the 1.7B trains (L2/L8/L30/L100) and
52B elsewhere (L15/L50); ~2.5TB of int32 `.bin`/`.idx` total. Blending happens at _training_ time via Megatron blend weights, so
each dataset is built exactly once.

### 5b. Ship to Azure with azcopy

`azcopy` is a single static binary — no root or install on the login node.
Upload once to the Spain workspace; the UK copy is server-side inside Azure
(nothing flows through CSCS twice):

**On your laptop** (has `az` + `azure/env.sh`)

1. Find each workspace's storage account and container behind the workspaceblobstore datastore:

```bash
az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_ES \
 --query '{account:account_name, container:container_name}'
az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_UK \
 --query '{account:account_name, container:container_name}'
```

The Spain (`AZ_ES_*`) storage behind `workspaceblobstore` — not secrets; re-run
`az ml datastore show` if you ever recreate the workspace and these change:

```
account   = snreswsstorage217e4ec3bb
container = azureml-blobstore-2ed07b42-5369-425f-b7f2-df29b3684e32
```

2. Mint a 7-day container SAS. It prints
   a token that IS a secret: paste it into `ES_SAS` on the CSCS side below, never
   into a committed file. The date expression is macOS/Linux portable:

```bash
az storage container generate-sas \
  --account-name snreswsstorage217e4ec3bb \
  --name azureml-blobstore-2ed07b42-5369-425f-b7f2-df29b3684e32 \
  --permissions racwl \
  --expiry "$(date -u -v+7d '+%Y-%m-%dT%H:%MZ' 2>/dev/null || date -u -d '+7 days' '+%Y-%m-%dT%H:%MZ')" \
  --auth-mode login --as-user -o tsv

# (For UK later: az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_UK
#  --query '{account:account_name, container:container_name}', then the same
#  generate-sas against those.)
```

**On the CSCS login node** — install azcopy, set the target + token once, then
copy every build into its own folder (`$OUT` = the §5a build dir):

```bash
# azcopy is a single static binary — no root/install:
wget -qO- https://aka.ms/downloadazcopy-v10-linux | tar xz
export PATH="$PWD/azcopy_linux_amd64"*:$PATH

# Spain target + the SAS token you just printed on your laptop:
export ES="https://snreswsstorage217e4ec3bb.blob.core.windows.net/azureml-blobstore-2ed07b42-5369-425f-b7f2-df29b3684e32"
export ES_SAS='PASTE_THE_TOKEN_HERE'

# English + L2 first (unblocks §8), then the rest — one build per folder:
azcopy copy "$OUT/english_dclm*" "$ES/predictivity/data/english_dclm/?$ES_SAS"
azcopy copy "$OUT/fineweb_L2*"   "$ES/predictivity/data/fineweb_L2/?$ES_SAS"
azcopy copy "$OUT/fineweb_L8*"   "$ES/predictivity/data/fineweb_L8/?$ES_SAS"
azcopy copy "$OUT/fineweb_L15*"  "$ES/predictivity/data/fineweb_L15/?$ES_SAS"
azcopy copy "$OUT/fineweb_L30*"  "$ES/predictivity/data/fineweb_L30/?$ES_SAS"
azcopy copy "$OUT/fineweb_L50*"  "$ES/predictivity/data/fineweb_L50/?$ES_SAS"
azcopy copy "$OUT/fineweb_L100*" "$ES/predictivity/data/fineweb_L100/?$ES_SAS"
azcopy copy "$OUT/validation*"   "$ES/predictivity/data/validation/?$ES_SAS"

# --- LATER (only after the UK workspace exists — §9) ----------------------
# The UK account/container don't exist until you run azure/setup.sh for the UK
# workspace, so `datastore show $AZ_ML_ARGS_UK` fails until then. Once it's up,
# get its account/container, mint a UK SAS the same way, and duplicate Spain ->
# UK entirely server-side (fast, no egress from CSCS):
export UK="https://<uk-account>.blob.core.windows.net/<uk-container>"
export UK_SAS='PASTE_THE_UK_TOKEN_HERE'
azcopy copy "$ES/predictivity/data/?$ES_SAS" "$UK/predictivity/?$UK_SAS" --recursive
```

(If `--auth-mode login --as-user` is rejected on your account, generate the
SAS from the portal instead: storage account → Containers → … → Generate SAS,
permissions Read+Add+Create+Write+List.)

Final blob layout, one folder per build — what the predictivity jobs (§8–§9)
mount directly:

```
predictivity/data/english_dclm/english_dclm.{bin,idx}
predictivity/data/fineweb_L2/fineweb_L2.{bin,idx}
...                fineweb_L100/...
predictivity/data/validation/validation*.{bin,idx} + validation.manifest.json
```

Verify in **Azure ML Studio → Data → Datastores → workspaceblobstore →
Browse**: `predictivity/data/` holds the per-build folders.

## 6. Convert and evaluate a checkpoint

**Convert** a Megatron checkpoint to a Hugging Face snapshot
(`ApertusForCausalLM`) — a few minutes on one node. The defaults in the YAML
are placeholders; point them at a real cell with `--set`:

```bash
az ml job create --file azure/jobs/convert.yml $AZ_ML_ARGS \
  --set inputs.checkpoints.path=azureml://datastores/workspaceblobstore/paths/predictivity/runs/<cell>/checkpoints \
        environment_variables.CKPT_STEP=<iter> \
        outputs.hf_model.path=azureml://datastores/workspaceblobstore/paths/models/<cell>/iter_<0-padded-iter>
```

**Evaluate.** `azure/eval.sh` reproduces the cluster's lm-eval invocation
exactly (vLLM backend, swiss-ai lm-evaluation-harness fork for the task
definitions, `add_bos_token=True`, no chat template, `--batch_size auto:20
--max_batch_size 32 --log_samples --write_out --trust_remote_code
--confirm_run_unsafe_code --gen_kwargs max_gen_toks=2048`, TP=PP=DP=1) and
writes results in the cluster's directory layout:

```bash
az ml job create --file azure/jobs/eval.yml $AZ_ML_ARGS \
  --set inputs.hf_model.path=azureml://datastores/workspaceblobstore/paths/models/<cell>/iter_<0-padded-iter> \
        environment_variables.NAME=<cell>-iter<iter>
az ml job stream --name <job name> $AZ_ML_ARGS
```

(Or use the launchers: `azure/launch_evals.py` fills the paths per cell and
checkpoint subset; `auto_evals.py` in §7 automates the whole chain.)

The job log ends with the parsed scores, e.g.:

```
hellaswag {'acc,none': 0.3311, 'acc_stderr,none': 0.0047, 'acc_norm,none': 0.3902, ...}
```

Sanity expectations: random = 0.25; small models early in training sit barely
above random — that's normal. Other tasks:
`--set environment_variables.TASKS=hellaswag,hellaswag_es` (any
comma-separated lm-eval names; the multilingual `hellaswag_*` variants are
in the fork).

**W&B push happens inside the eval job** when you pass your key
(`--set environment_variables.WANDB_API_KEY=$WANDB_API_KEY`, which the
launchers do automatically when `WANDB_API_KEY` is exported): the job runs the
repo's standard `push_all_results.py` against its own results, appending one
step to the model’s curve in `mariagrandury-epflnlp/msnr` — the same
project the training runs log to, so loss and benchmark curves live side by
side. Without a key, push later from your laptop:

```bash
az ml job download --name <eval job name> --output-name results --download-path /tmp/azure-evals $AZ_ML_ARGS
cd ../..   # repo root
LOGS_ROOT=/tmp/azure-evals/named-outputs/results \
  python src/evals/scripts/push_all_results.py \
  --entity mariagrandury-epflnlp --project msnr \
  --name <cell>-iter<iter>
```

(The NAME must be a `configs/models.json` key + `-iter<N>` — that's how the
pusher resolves the tokens/FLOPs axes.)

## 7. Auto-evals every 5 checkpoints

To see whether a training is progressing well with more information than the
loss curve, saved checkpoints get evaluated automatically on the **`auto`**
benchmark group from `configs/tasks.json` — `hellaswag`, `hellaswag_ru`,
`global_mmlu_full_en`, `global_mmlu_full_ru`,
`global_piqa_completions_eng_latn`, `global_piqa_completions_rus_cyrl` —
and pushed to the same W&B project (`msnr`) as the training runs. Due are
**every 2nd saved checkpoint plus the run's final one** whatever its iter
(predictivity targets end off the save grid, e.g. 4500 or 81000);
`--every N` changes the cadence. Edit the `auto` benchmark group in
`configs/tasks.json` to change what runs.

Start the watcher in a terminal alongside your training run:

```bash
source azure/env.sh   # Azure names; W&B key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"
python azure/auto_evals.py --watch 600        # one pass every 10 min; Ctrl-C to stop
```

Each pass lists the checkpoints in blob storage and, for every due iteration,
submits the one step that's missing: first a `azure/jobs/convert.yml` job
(Megatron → HF), then — on a later pass, once the snapshot exists — a
`azure/jobs/eval.yml` job with `TASKS=auto`. Everything already evaluated or
currently running is skipped, so the watcher is idempotent: stop it, restart
it, run it twice — nothing duplicates, and a killed pass just means the next
one picks up the remainder. `--dry-run` previews the submissions;
`--every N` changes the cadence; `--name <cell>` watches a single run. In
W&B, the auto-eval points appear on the model's per-benchmark curves
(x-axis: flops/tokens/iter) as training progresses.

## 8. Launch the predictivity trainings

Once the English + L2 data is on the blob store (§5), start with the cheapest real
cells and scale out. Same launcher, same filters as on the cluster; jobs
queue on the clusters and run as nodes free up:

```bash
source azure/env.sh   # Azure names; W&B key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"

python launch_trainings.py azure --dry-run              # the whole 51-job grid
python launch_trainings.py azure --langs 2 --size 90M   # first real cell
python launch_trainings.py azure --langs 1              # monolingual anchors
python launch_trainings.py azure                        # everything
python launch_trainings.py azure --arch shallow         # the depth-intervention variant
```

`--arch` picks the reviewed architecture family — `deep` (default baseline,
`hyperparams/hyperparams_deep.json`) or `shallow`
(`hyperparams/hyperparams_shallow.json`, same non-embedding sizes at
width/depth 128); the D(N) = 100 × N schedule comes from each config's
`predictivity` block. Runs are named `apertus-<size>-L<L>[-shallow]-seed<seed>`
and log to `mariagrandury-epflnlp/msnr`. Micro-batch sizes tuned for the
cluster are auto-shrunk per node (`launch_pretraining_azure.sh`) so the
global batch of 504 always divides; the 1.7B resolves to MBS 1 on the 8-GPU
nodes — override with `--set environment_variables.MBS=3` if it fits.

Two behaviours worth knowing:

- **Resume = resubmit.** Each cell's checkpoint output is pinned to a fixed
  storage path and Megatron `--load`s from it: if the job dies (or a Spot
  node is evicted), submit the same cell again and it continues from the
  last checkpoint. Iterations already completed are never retrained.
- The tokenized data is **downloaded to the node's local NVMe** at job start
  (`mode: download`, a few minutes) — don't change it to `ro_mount`;
  memory-mapped `.bin` reads over a blob mount are pathologically slow.

## 9. The UK South workspace (1B and 1.7B)

The launcher places 1B/1.7B cells on **UK South** (`gpu-nd96-spot`, 8×H100
Spot) automatically — it reads `AZ_ML_ARGS_UK` from `azure/env.sh`. To bring
that workspace up: re-export `AZ_LOCATION/AZ_RG/AZ_WS` to the `AZ_UK_*`
values, run `azure/setup.sh` once (computes whose SKU the region lacks are
skipped), then server-side-copy the data (§5b's LATER block).

## 10. Where everything is stored (and getting it out)

All artifacts live in the workspace's blob storage under `workspaceblobstore`:

| Path                                                  | Contents                                      |
| ----------------------------------------------------- | --------------------------------------------- |
| `predictivity/data/<build>/`                          | `.bin`/`.idx` per dataset build               |
| `predictivity/runs/<cell>/checkpoints`                | Megatron `torch_dist` checkpoints (`iter_*/`) |
| `predictivity/runs/<cell>/logs`                       | TensorBoard + W&B offline files               |
| `models/<cell>/iter_<N>`                              | converted HF snapshots                        |
| `eval_logs/<entity>/<project>/<NAME>/harness/eval_*/` | lm-eval results + samples                     |

Browse it in Studio (**Data → Datastores → workspaceblobstore → Browse**) or
[Azure Storage Explorer](https://azure.microsoft.com/en-us/products/storage/storage-explorer).

**Register converted models** in the workspace's model registry so they're
versioned and survive any cleanup of the run directories:

```bash
az ml model create --name <cell> --version <iter> \
  --type custom_model \
  --path azureml://datastores/workspaceblobstore/paths/models/<cell>/iter_<0-padded-iter> \
  $AZ_ML_ARGS
```

**Download anything** with `az ml job download` (per-job outputs, as in
step 6) or `azcopy` for bulk paths.

**Optional: push to the Hugging Face Hub** with the sweep's naming
convention (repo per cell, branch per checkpoint — same as
`conversion/push-snr.py` uses on the cluster):

```bash
# Uses your saved HF login (token on this laptop) — verify, don't re-enter:
huggingface-cli whoami >/dev/null 2>&1 && echo "✓ HF logged in" || echo "✗ run: huggingface-cli login"
huggingface-cli upload <your-org>/<cell> \
  ./iter_<0-padded-iter> . --revision stage1-step-<iter> --private
```

## 11. Teardown

Compute scales to zero by itself; a parked setup costs only blob storage
(~$12/month for ~600GB). To stop even that:

```bash
az ml compute delete --name gpu-nc80-lp $AZ_ML_ARGS --yes  # keeps all data
az group delete --name $AZ_RG --yes                        # deletes EVERYTHING
```

!!! danger
`az group delete` destroys the storage account too — checkpoints,
models, eval results. Download (or push to HF / register elsewhere)
anything you care about first.

## 12. Common mistakes

- **Compute create fails / job queues forever** → you have no quota in that
  region (step 3), or you edited `AZ_LOCATION` after creating the workspace
  (everything is per-region — start over in one region).
- **NCCL "unhandled system error" at startup** → shared memory too small.
  The train jobs set `resources.shm_size: 64g`; keep it if you copy the YAML.
- **Training is 10–100× slower than expected** → the data input was changed
  from `mode: download` to a mount. Don't stream `.bin` memmaps from blob.
- **A resubmitted cell suddenly trains a different number of iters** → you
  reused another cell's checkpoint dir: `TRAINING_STEPS` and the output
  paths must change together. The launcher pins each cell's own
  `predictivity/runs/<cell>/` paths for exactly this reason.
- **Conversion fails with a `transformers` import/version error** → run it
  via `azure/jobs/convert.yml` only; it pins `transformers==4.57.6` inside its
  own job container (the training and eval images keep their own versions).
- **Eval crashes at vLLM init with a KV-heads error** → someone raised
  `TP`. Keep `TP=1`: the 350M/1B models have 5/7 KV heads, indivisible by
  higher TP.
- **`push_all_results.py` silently pushes nothing** → the eval `NAME`
  doesn't match a `configs/models.json` key + `-iter<N>`, or `LOGS_ROOT`
  doesn't point at the directory that _contains_ `<entity>/<project>/...`.
- **No W&B run appeared** → `WANDB_API_KEY` wasn't passed to the job
  (`--set environment_variables.WANDB_API_KEY=$WANDB_API_KEY`); the job
  still trains, logging only to TensorBoard
  (`predictivity/runs/<cell>/logs/tensorboard`).
