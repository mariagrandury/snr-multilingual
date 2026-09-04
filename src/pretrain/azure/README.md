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
on the Canada Central `gpu-nd96-spot` pool (8× H100 + InfiniBand, §9) — see
the compute-budget sheet next to the plan for the full budget.

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
size ≤600M plus evals), and the ND (8×H100) pool lives in **Canada Central**
(`snr-ca-rg`/`snr-ca-ws`) — see §9. If you're adapting this guide to another
subscription, pick a region where _your_ subscription can deploy the SKUs and
stick to it. Then:

```bash
source azure/env.sh
bash azure/setup.sh
```

!!! note "Why Canada Central for ND, not UK South (2026-08-26)"
    UK South was the original ND region, and its SKU is still allow-list
    clear — but after 13 days and 23 quota requests the subscription holds
    **0 dedicated H100 cores anywhere**, so the choice is now driven by what
    can run *today*:

    - Canada Central has the cheapest meters of any allow-list-clear ND
      region: dedicated **$122.40**/node-h against UK South's $122.90, and
      Spot $21.80 vs $23.53 if the tier ever becomes reachable.
    - Italy North and Norway East look clear in the SKU API but have **no
      retail meters at all** for `ND96isr_H100_v5`, so quota there could
      never be billed. Those are the 768/768/1536-core tickets.

    Check any of this with `./quota_status.sh board`.

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
  low-priority meter, ~$3.63/h); the Canada Central workspace instead gets
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
2. **`standardNDv5H100Family`** in **Canada Central** — 96 cores per
   ND96isr node (1 node = 8×H100); 6 nodes = 576 cores covers the 1B rung,
   9 nodes = 864 covers 1B + 1.7B. File it as **`Type: Dedicated`** — that
   is the only tier H100 can use here, see "The four gates" below.

   If a region answers `NotAvailableForSubscription`, raise that as a
   **separate, explicit ask**: it is an allow-list decision, not a capacity
   shortage, so it is not gated on the global GPU crunch and a different
   team can act on it. Do not cancel those tickets — the ticket *is* the
   channel for lifting the restriction; reframe them from "more cores" to
   "please enable this VM family for the subscription".

File absent families via Help + Support → _Service and subscription limits
(quotas)_; H100-class requests open a support ticket (days, not minutes),
so file early.

### The four gates, and which signals lie

A GPU request passes four independent gates. They fail in this order, with
different remedies, and conflating them wastes days:

| # | Gate | Failure | Fixed by a quota ticket? |
| - | ---- | ------- | ------------------------ |
| 1 | **SKU allow-list** | `NotAvailableForSubscription` on `az vm list-skus` | **Yes** — but ask for *enablement*, not cores |
| 2 | **Tier policy** | `UnsupportedVMSizeForLowPriority` at cluster create | **No, never** — H100 is dedicated-only here |
| 3 | **Quota** | `ClusterMinNodesExceedCoreQuota` (dedicated) / job queues forever (low-priority) | **Yes** |
| 4 | **Capacity** | `OutOfCapacity` at scale-up | **No** — retry, or change region/SKU |

Gate 2 is family-level: **H100 cannot use low-priority at all** (verified on
`ND96isr_H100_v5` and `NC80adis_H100_v5`), while **A100 can**. Gate 3 is
asymmetric: a *dedicated* cluster is quota-checked at **create** and fails
loudly; a *low-priority* cluster is not checked at all and creates as
`provisioning_state: Succeeded` no matter the quota, so "the cluster exists"
proves nothing.

**Three signals that lied to us** (2026-08-26) — do not plan around any of
them:

- `az vm list-skus` → `LowPriorityCapable`: **True** for the refused
  `ND96isr_H100_v5`, **False** for the accepted `NC96ads_A100_v4`. Wrong in
  both directions.
- A "Low Priority" **retail meter exists** for `ND96isr_H100_v5`
  ($23.60/node-h) even though the tier is refused.
- The AML **quota counters** read `lowPriority: -1` (no per-family cap) and
  `TotalLowPriorityCores: 300`, which looks like 3 free ND nodes. Those
  cores are real but spendable only on A100.

**The only reliable test is the $0 probe.** `min_instances: 0` allocates
nothing, so creating a cluster costs nothing; the verdict is in
`properties.errors`, which the CLI does not surface:

```bash
az ml compute create --file azure/compute-nd96-spot.yml $AZ_ML_ARGS
az rest --method get --url \
  "https://management.azure.com/subscriptions/$AZ_SUBSCRIPTION/resourceGroups/$AZ_RG/providers/Microsoft.MachineLearningServices/workspaces/$AZ_WS/computes/gpu-nd96-spot?api-version=2024-10-01" \
  --query "properties.properties.errors"
```

Run that against any new SKU/region/tier **before** planning around it. A
non-empty result at gate 2 means no amount of waiting or refiling will help.

**Check status with `quota_status.sh` — never the portal.** Quota here is
split across three systems that each hold a third of the answer, and the
obvious CLI commands return empty rather than erroring (`Microsoft.Compute`
and `Microsoft.Quota` are both unregistered on this subscription, so
`az vm list-usage` yields `[]` and every `az quota` call fails silently).
Granted cores land in the **Azure ML** counters, because every ticket here
is filed as subType `BatchAml`.

```bash
./quota_status.sh board       # terminal report + shareable quota_status.png
./quota_status.sh limits      # granted cores per family, per region
./quota_status.sh tickets     # what was filed + the reply emails
```

`board` is the one to run: it joins granted cores × filed tickets × where
the SKU is actually deployable, and prints which tickets can **never**
approve (`NotAvailableForSubscription` is an allow-list denial, not a
capacity shortage — refiling it forever will not help) and which regions
are still worth filing in. It refreshes from Azure each run (~2 min);
`--offline` re-renders from the last fetch.

Before filing anywhere new, confirm two things hold for that region: the
SKU is not allow-list blocked, **and** Azure ML exists there (several
H100 regions, e.g. `southeastus` and `centraluseuap`, have no Azure ML at
all, so quota there would be unusable). `board` checks both.

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
Upload once to the Spain workspace; the Canada Central copy is server-side inside Azure
(nothing flows through CSCS twice).

**Copy from the capstor master, not the iopsstor stage.** capstor is the
durable copy every build writes to; iopsstor holds only what
`data/stage_to_iopsstor.sh` has staged so far, and is swept every ~30 days.

**What ships, and how much.** Scheme A is the full ladder; scheme B differs
from A only at L ∈ {8, 15, 30} (`SCHEME_B_LANGS` in `launch_trainings.py` —
every other setting reuses the scheme-A build, so there is nothing else to
upload). **~3.3 TB in total**, apparent size:

| scheme | builds | size |
| ------ | ------ | ---: |
| A | `english_dclm` 686G, `fineweb_L2` 272G, `L8` 343G, `L15` 194G, `L30` 343G, `L50` 194G, `L100` 343G | 2.4 TB |
| A | `validation.*` (one pair per language + manifest) | 2 GB |
| B | `fineweb_L8` 343G, `L15` 194G, `L30` 343G | 0.9 TB |

Scheme B's `english_dclm.*` and `validation.manifest.json` are **symlinks**
into the scheme-A directory, and azcopy skips symlinks unless you pass
`--follow-symlinks`. Leave them skipped: `azure/jobs/pretrain.yml` pins
`inputs.english` at the scheme-A `predictivity/data/english_dclm` folder for
every cell, and only `inputs.fineweb` is repointed at `data/schemeB/` — so
uploading a second 686 GB copy of English would buy nothing.

**On your laptop** (has `az` + `azure/env.sh`)

1. Find each workspace's storage account and container behind the workspaceblobstore datastore:

```bash
az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_ES \
 --query '{account:account_name, container:container_name}'
az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_CA \
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

# (For Canada Central later: az ml datastore show --name workspaceblobstore $AZ_ML_ARGS_CA
#  --query '{account:account_name, container:container_name}', then the same
#  generate-sas against those.)
```

**On the CSCS login node** — install azcopy once, set the target + token, then
copy every build into its own folder.

3. Install azcopy and point it at Spain. `$OUT` is the capstor master (§5a):

```bash
cd /iopsstor/scratch/cscs/mariagrandury
wget -qO- https://aka.ms/downloadazcopy-v10-linux | tar xz
export PATH="$PWD/azcopy_linux_amd64"*:$PATH

export OUT=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data
export ES="https://snreswsstorage217e4ec3bb.blob.core.windows.net/azureml-blobstore-2ed07b42-5369-425f-b7f2-df29b3684e32"
export ES_SAS='PASTE_THE_TOKEN_HERE'   # from step 2 — a secret, never commit it
```

4. **Scheme A.** One build per destination folder. English + L2 first: those
   two unblock the first real cells (§8), and the rest can upload while those
   models already train.

```bash
azcopy copy "$OUT/english_dclm.*" "$ES/predictivity/data/english_dclm/?$ES_SAS"
azcopy copy "$OUT/fineweb_L2.*"   "$ES/predictivity/data/fineweb_L2/?$ES_SAS"
azcopy copy "$OUT/validation.*"   "$ES/predictivity/data/validation/?$ES_SAS"

for L in 8 15 30 50 100; do
  azcopy copy "$OUT/fineweb_L$L.*" "$ES/predictivity/data/fineweb_L$L/?$ES_SAS"
done
```

5. **Scheme B** — only the three settings where it differs from A, into a
   `schemeB/` subfolder that mirrors the CSCS layout (this is exactly what
   `launch_trainings.py azure --scheme B` points `inputs.fineweb` at):

```bash
for L in 8 15 30; do
  azcopy copy "$OUT/schemeB/fineweb_L$L.*" "$ES/predictivity/data/schemeB/fineweb_L$L/?$ES_SAS"
done
```

6. **Run it detached.** At ~3.3 TB this is hours of wall-clock and the login
   node will drop the session first. There is **no `tmux` or `screen` on
   Clariden** (not installed, not a module) — use `nohup`, which is:

```bash
nohup bash -c '<the loops above>' > azcopy.log 2>&1 &
tail -f azcopy.log      # progress; safe to Ctrl-C, the copy keeps running
```

azcopy keeps a resumable job plan, so a dropped connection is not a lost
upload: `azcopy jobs list`, then `azcopy jobs resume <job-id> --source-sas
"$ES_SAS"`. Re-running a `copy` is also safe — it re-uploads, it does not
corrupt. If the 7-day SAS expires mid-transfer, mint a new one (step 2) and
resume.

7. **Verify** in Azure ML Studio → Data → Datastores → workspaceblobstore →
   Browse. `predictivity/data/` should hold one folder per build:

```
predictivity/data/english_dclm/english_dclm.{bin,idx}
predictivity/data/fineweb_L2/fineweb_L2.{bin,idx}
...                fineweb_L100/...
predictivity/data/validation/validation*.{bin,idx} + validation.manifest.json
predictivity/data/schemeB/fineweb_L{8,15,30}/fineweb_L*.{bin,idx}
```

**LATER — the Canada Central copy** (only after that workspace exists,
§9). Its account/container don't exist until `azure/setup.sh` has run for
it, so `datastore show $AZ_ML_ARGS_CA` fails before that. Once it's up, get
its account/container, mint a SAS the same way, and duplicate Spain → Canada
entirely server-side — fast, and no second egress from CSCS:

```bash
export CA="https://<ca-account>.blob.core.windows.net/<ca-container>"
export CA_SAS='PASTE_THE_CA_TOKEN_HERE'
azcopy copy "$ES/predictivity/data/?$ES_SAS" "$CA/predictivity/?$CA_SAS" --recursive
```

(If `--auth-mode login --as-user` is rejected on your account, generate the
SAS from the portal instead: storage account → Containers → … → Generate SAS,
permissions Read+Add+Create+Write+List.)

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

**Evaluate.** `azure/eval.sh` runs the cluster's own inner runner
(`src/evals/scripts/_run_per_task.sh`: one `eval_worker.py` per GPU, the
model loaded once, each task's results written as it finishes — so a
preempted Spot job keeps what it did and the watcher's next submission runs
only the rest) with the cluster's lm-eval arguments (vLLM backend, swiss-ai
lm-evaluation-harness fork for the task definitions, `add_bos_token=True`,
no chat template, `--batch_size auto:20 --max_batch_size 32 --log_samples
--write_out --trust_remote_code --confirm_run_unsafe_code --gen_kwargs
max_gen_toks=2048`, TP=PP=DP=1) and writes results in the cluster's
directory layout:

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
group from `configs/tasks.json` — a list of **benchmark names**; each cell
gets every listed benchmark's tasks in the languages it trains on (an L2
cell gets `hellaswag` + `hellaswag_ru` + …, an L1 cell only the English
variants) — and pushed to the same W&B project (`msnr`) as the training
runs. Due are
**every 2nd saved checkpoint plus the run's final one** whatever its iter
(predictivity targets end off the save grid, e.g. 4500 or 81000);
`--every N` changes the cadence. Edit the `auto` benchmark group in
`configs/tasks.json` to change what runs.

Start the watcher in a terminal alongside your training run:

```bash
source azure/env.sh   # Azure names; W&B key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"
python auto_evals_azure.py --watch 600                 # Spain workspace (<=600M)
python auto_evals_azure.py --workspace ca --watch 600  # Canada ND workspace (1B/1.7B)
```

The two workspaces have separate blob stores and compute, so run one
watcher per workspace; `--workspace ca` switches the az CLI to the
`AZ_CA_*` names and overrides the job YAMLs' Spain-only compute with
`gpu-nd96-spot` (do the same with `--set compute=azureml:gpu-nd96-spot`
when submitting `jobs/push.yml` in the Canada Central workspace). Note
`gpu-nd96-spot` cannot currently allocate — H100 has no reachable tier
here — so point evals at an A100 cluster if they must run before H100
quota lands.

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
`predictivity` block. Runs are named `lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>`
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

## 9. The ND workspace — Canada Central (1B and 1.7B)

The launcher places 1B/1.7B cells on the ND pool (`gpu-nd96-spot`, 8×H100)
automatically. Bring the workspace up with the `use_ca` helper from
`azure/env.sh` (it re-exports `AZ_LOCATION/AZ_RG/AZ_WS` **and**
`AZ_ML_ARGS` — setting only the first three leaves `az ml` pointed at Spain):

```bash
source azure/env.sh && use_ca
bash azure/setup.sh          # idempotent; skips SKUs the region lacks
```

Then server-side-copy the data (§5b's LATER block).

### Low-priority is NOT available for H100 (settled 2026-08-26)

The 300 regional low-priority vCPUs (`TotalLowPriorityCores`) are genuinely
granted, but **cannot be spent on any H100 SKU**. AML rejects them outright:

```
UnsupportedVMSizeForLowPriority: The VM size STANDARD_ND96ISR_H100_v5 is not
allowed for LowPriority. Please convert to Dedicated or use a different VM size.
```

Confirmed for both `Standard_ND96isr_H100_v5` and `Standard_NC80adis_H100_v5`.
**Dedicated quota is therefore the only path to H100** — the pending dedicated
tickets are the right ask after all.

The policy is **family-level, not SKU-level**: A100 *is* allowed on
low-priority (both `NC24ads_A100_v4` and `NC96ads_A100_v4` create clean), it
just returned `OutOfCapacity` in Canada Central when asked to allocate. That
failure is transient and needs no quota, so it is worth re-probing. Do not
use `az vm list-skus`'s `LowPriorityCapable` to predict any of this — it is
wrong in both directions (`True` for the rejected ND96isr_H100_v5, `False`
for the accepted NC96ads_A100_v4).

| What you see | Which gate failed | Fixable by a quota ticket? |
| --- | --- | --- |
| `UnsupportedVMSizeForLowPriority` at create | tier/SKU policy | **No** — never |
| `OutOfCapacity` on scale-up | physical capacity | No — retry, or change region/SKU |
| Job queues, counters stay 0 | quota | **Yes** |

!!! danger "This failure is invisible until a job hits it"
    The cluster **creates as `provisioning_state: Succeeded`** and looks
    healthy in `az ml compute list`. The rejection lives only in
    `properties.errors`, and reaches the job as the uninformative
    _"cluster has encountered unknown issue"_. Two other signals lie too:
    `az vm list-skus` reports `LowPriorityCapable=True` for ND96isr_H100_v5,
    and a "Low Priority" retail meter exists for it. Both are wrong.

    The only reliable probe — and it costs **$0**, since `min_instances: 0`
    allocates nothing:

    ```bash
    az ml compute create --file azure/compute-nd96-spot.yml $AZ_ML_ARGS
    az rest --method get --url \
      "https://management.azure.com/subscriptions/$AZ_SUBSCRIPTION/resourceGroups/$AZ_RG/providers/Microsoft.MachineLearningServices/workspaces/$AZ_WS/computes/gpu-nd96-spot?api-version=2024-10-01" \
      --query "properties.properties.errors"
    ```

    Run that after creating **any** new cluster, before submitting work to it.

### The A100 fallback (`--compute`)

Since no H100 is obtainable, `launch_trainings.py azure` takes `--compute` to
retarget any cell at another cluster. Two A100 clusters are defined:

| File | Cluster | Tier | State |
| --- | --- | --- | --- |
| `compute-nc96-a100-lp.yml` | `gpu-nc96-a100-lp` | low-priority | created; `OutOfCapacity` on scale-up |
| `compute-nc96-a100-ded.yml` | `gpu-nc96-a100-ded` | dedicated | **cannot be created** — 0 family quota |

```bash
source azure/env.sh && use_ca
python launch_trainings.py azure --size 1B --langs 1 --seed 1904 \
  --compute gpu-nc96-a100-lp --dry-run      # inspect, then drop --dry-run
```

Jobs are single-node on Azure regardless (`torchrun --standalone`), so
switching cluster only changes the per-node GPU count; the wrapper
re-resolves MBS against it (1B: MBS 6 on 4×A100 → grad accum 21).

### Smoke-test a cluster

`smoke.yml` runs on **mock data**, so it needs no dataset — the fastest way to
prove a new cluster end to end:

```bash
az ml job create --file azure/jobs/smoke.yml $AZ_ML_ARGS \
  --set compute=azureml:gpu-nc96-a100-lp --set display_name=smoke-a100-nc96 --web
```

!!! warning "Azure has no training data yet"
    `predictivity/data` is **empty in both workspaces** — the tokenized
    mixtures still live only on CSCS. Real cells cannot run until §5's
    ~2.5 TB upload completes; only mock-data smoke tests can.

```bash
source azure/env.sh && use_ca
az ml compute show --name gpu-nd96-spot $AZ_ML_ARGS \
  --query "{state:provisioning_state, tier:tier, size:size, max:max_instances}" -o yaml

# 20 iterations on mock data, one node (smoke.yml defaults to the Spain NC
# cluster, so point it at the ND one)
az ml job create --file azure/jobs/smoke.yml $AZ_ML_ARGS \
  --set compute=azureml:gpu-nd96-spot \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY --web
```

Quota and capacity fail at **different stages** — which one you hit matters:

| Symptom | Meaning |
| --- | --- |
| _"cluster has encountered unknown issue"_ | read `properties.errors` (above) — the cluster is misconfigured, e.g. an H100 SKU on `low_priority` |
| Job stuck in Queued, `list-nodes` empty, `TotalDedicatedCores` still `0` | no dedicated quota yet — this is the current state |
| Job stuck in Queued with quota granted | **no capacity** — a quota ticket will not fix this |
| `TotalDedicatedCores` moves and the job runs | working |

Watch all three with:

```bash
az ml compute list-nodes --name gpu-nd96-spot $AZ_ML_ARGS -o table
./quota_status.sh aml | grep -A3 -i canadacentral
```

Cost: ~$23.60/node-h, so the <30 min smoke test is **~$12** (not the "<$2" of
§4, which is a NC80adis node). `min_instances: 0` means an idle cluster is $0.

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
