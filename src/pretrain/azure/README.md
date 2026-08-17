# Pretraining and evaluating on Azure

This guide reproduces one cell of the SNR pretraining sweep on Azure —
**Apertus 175M, mixture 30% FineWeb-Edu / 70% FineWeb2-HQ, seed 28** — and
evaluates it on **hellaswag**, assuming you have never used Azure before.
Everything runs as [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/)
_command jobs_: you submit a YAML from your laptop, Azure boots a GPU node,
runs the job in a container, saves outputs to cloud storage and shuts the
node down. You never SSH anywhere and you only pay while a job runs.

The same scripts launch every other size × mixture × seed cell (step 9).

**The three stages** (do them in order — each one validates the next):

Every stage runs on `gpu-nc80-lp` (one `Standard_NC80adis_H100_v5` node:
2× H100 94GB at the fixed low-priority meter, ~$3.63/h — the plan's economy
pool, see the compute-budget sheet):

| Stage                    | What it proves                                     | Time          | Cost      |
| ------------------------ | -------------------------------------------------- | ------------- | --------- |
| Smoke test               | container + Megatron fork + checkpointing work     | < 30 min      | < $2      |
| Pilot (5.16B tokens)     | data pipeline + full loop + conversion + eval work | ~3–5 h        | ~$15      |
| Full run (103.2B tokens) | the real cell                                      | ~2.5–3.5 days | ~$220–300 |

Per-size ballpark for full 103.2B-token runs on `gpu-nc80-lp` (same data,
same global batch): 175M ≈ 3 days / ~$240; 350M ≈ 5 days / ~$400;
600M ≈ 7 days / ~$620; 1B ≈ 10–11 days / ~$920. The 1B (and anything
bigger) belongs on the UK South `gpu-nd96-spot` pool instead — 8× H100 +
InfiniBand cuts the 1B to ~2.5–3 days (§11).

**Prerequisites**

- An Azure account with a **pay-as-you-go subscription** (a free trial has no
  GPU quota). Create one at [azure.microsoft.com](https://azure.microsoft.com);
  you'll need a credit card. Budget ≥ $25 for smoke+pilot, ≥ $350 with the
  full run.
- A [wandb.ai](https://wandb.ai) account and API key (Settings → API keys) —
  W&B is the primary way you'll monitor training.
- No Hugging Face token needed: the tokenizer is public, and all training
  data ships pre-tokenized from CSCS (§5) — nothing is downloaded from the
  HF Hub.
- CSCS access (login node) — the data mixtures are built there (§5).
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

Edit `env.sh`: set `AZ_SUBSCRIPTION` to your subscription id. The regions are
already decided for this project (quota, storage and compute are all
per-region — see the compute-budget sheet): the **primary workspace is Spain
Central** (`snr-es-rg`/`snr-es-ws`, NC80adis H100 low-priority — every size
≤600M plus evals) with a second workspace in **UK South**
(`snr-uk-rg`/`snr-uk-ws`, ND96isr 8×H100 Spot — the 1B/1.7B pool); §11 sets
up the UK one. If you're adapting this guide to another subscription, pick a
region where _your_ subscription can deploy the SKUs and stick to it. Then:

```bash
source env.sh
bash setup_azure.sh
```

**Credentials & W&B config.** `source env.sh` loads the Azure names _and_ the
W&B **entity/project** from `configs/hf_wandb.json` (the single source of
truth — nothing else hardcodes them). The two **secrets** — `WANDB_API_KEY`
and your Hugging Face token — are _not_ in any file and are never re-entered
in this guide: they live in your laptop's shell env / HF login. Each step that
needs them just verifies they're present first, e.g.:

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
  low-priority meter, ~$3.63/h) — every job in this guide runs on it; the
  UK South workspace instead gets `gpu-nd96-spot` (§11). Clusters have
  `min_instances: 0`: nodes exist only while a job runs, so an idle setup
  costs ~$0.

!!! warning "The compute creation fails until you have quota"
A brand-new subscription has **0 GPU quota** — do step 3 first if
`setup_azure.sh` fails on the compute step, then re-run it (it's
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
for training because resubmitting a job resumes from the last checkpoint
(step 7); if an eviction ever bites a one-shot conversion/eval job, just
resubmit it. Dedicated is the fallback only if evictions thrash — remove
the `tier:` line and re-create the compute.

## 4. Smoke test (do not skip)

This runs 20 training iterations on **mock data** on one node — it exercises
the exact code path of the real run (swiss-ai Megatron fork, xIELU/QK-norm
kernels, AdEMAMix optimizer, `torch_dist` checkpoint save) for pocket change:

```bash
source env.sh   # W&B entity/project from configs/hf_wandb.json; key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (optional for the smoke test)"
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
appear in W&B under `mariagrandury-epflnlp/msnr`.

## 5. Generate the data mixtures (CSCS) and ship them to Azure

**All training data is built from the corpora curated at CSCS** — DCLM-edu
(English) and FineWeb-2-HQ (multilingual), the filtered swiss-ai variants on
`/capstor` — and tokenized *there*; nothing is downloaded from the HF Hub.
Two scripts under `src/pretrain/data/` own the pipeline:

- [`../data/create_data_mixture.py`](../data/create_data_mixture.py) — the worker:
  streams the parquet sources, tokenizes with `swiss-ai/Apertus-70B-2509`,
  writes Megatron `.bin`/`.idx`, builds the fixed validation set, and
  excludes its rows from training via a manifest. Resumable after
  preemption.
- [`../data/build_data_mixtures.py`](../data/build_data_mixtures.py) — the driver:
  turns the language schemes (`../data/language_sets_scheme{A,B}.json`) into
  per-build `create_data_mixture.py` calls with the right token targets.

### 5a. Build on CSCS — EN+RU first, then the rest

The English + Russian pair unblocks the minimal plan (§10), so build and
ship it first; the remaining multilingual builds run while those two models
already train:

```bash
# CSCS login node, from src/pretrain/data/ ($OUT on /capstor or /iopsstor)
# 1. Fixed validation set + exclusion manifest (once — every build needs it)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage validation

# 2. EN+RU: the shared English dataset + the L2 FineWeb-2 build (rus_Cyrl)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage english
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage fineweb --settings 2

# 3. All remaining language settings (start §10 first, then run this)
python build_data_mixtures.py --scheme A --output_dir /iopsstor/scratch/cscs/mariagrandury/data/ --stage fineweb \
    --settings 8,15,30,50,100
```

Targets (printed by `--dry_run` first): 184.0B English; 52B (L2/L15/L50) or
92.0B (L8/L30/L100) FineWeb-2 per setting; ~2.5TB of int32 `.bin`/`.idx`
total. Blending happens at _training_ time via Megatron blend weights, so
each dataset is built exactly once.

### 5b. Ship to Azure with azcopy

`azcopy` is a single static binary — no root or install on the login node.
Upload once to the Spain workspace; the UK copy is server-side inside Azure
(nothing flows through CSCS twice):

**On your laptop** (has `az` + `env.sh`)

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

# EN+RU first (unblocks §10), then the rest — one build per folder:
azcopy copy "$OUT/english_dclm*" "$ES/predictivity/data/english_dclm/?$ES_SAS"
azcopy copy "$OUT/fineweb_L2*"   "$ES/predictivity/data/fineweb_L2/?$ES_SAS"
azcopy copy "$OUT/fineweb_L8*"   "$ES/predictivity/data/fineweb_L8/?$ES_SAS"
azcopy copy "$OUT/fineweb_L15*"  "$ES/predictivity/data/fineweb_L15/?$ES_SAS"
azcopy copy "$OUT/fineweb_L30*"  "$ES/predictivity/data/fineweb_L30/?$ES_SAS"
azcopy copy "$OUT/fineweb_L50*"  "$ES/predictivity/data/fineweb_L50/?$ES_SAS"
azcopy copy "$OUT/fineweb_L100*" "$ES/predictivity/data/fineweb_L100/?$ES_SAS"
azcopy copy "$OUT/validation*"   "$ES/predictivity/data/validation/?$ES_SAS"

# --- LATER (only after the UK workspace exists — §11) ---------------------
# The UK account/container don't exist until you run setup_azure.sh for the UK
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

Final blob layout, one folder per build — what the predictivity jobs (§11)
mount directly:

```
predictivity/data/english_dclm/english_dclm.{bin,idx}
predictivity/data/fineweb_L2/fineweb_L2.{bin,idx}
...                fineweb_L100/...
predictivity/data/validation/validation*.{bin,idx} + validation.manifest.json
```

### 5c. The 36-cell sweep mixtures (pilot/full path, §6–§7 and §12)

The `mix_<edu>_<fw2>` mixtures are **not rebuilt** either — copy the
cluster's frozen tokenized mixtures as-is (their component `.bin`/`.idx`
are symlinks, so tell azcopy to follow them) and write the `data_path.txt`
manifest `train.sh` reads, weights proportional to the actual `.bin` sizes:

```bash
# Reuses $ES / $ES_SAS exported in §5b (same CSCS login session).
cd <CSCS_MIX_DIR>   # e.g. the frozen mix_100B_30_70
total=$(du -cbL *.bin | tail -1 | cut -f1)
for b in *.bin; do
  printf '%.6f %s\n' "$(python3 -c "print($(stat -Lc%s "$b")/$total)")" "${b%.bin}"
done > data_path.txt
azcopy copy "$PWD/*" \
  "$ES/tokenized/mix_30_70/full/?$ES_SAS" \
  --recursive --follow-symlinks
```

Verify in **Azure ML Studio → Data → Datastores → workspaceblobstore →
Browse**: `tokenized/mix_30_70/full` holds the `.bin`/`.idx` pairs plus
`data_path.txt`, and `predictivity/data/` the per-build folders.

## 6. Pilot training run

```bash
az ml job create --file jobs/train-pilot.yml $AZ_ML_ARGS \
  --set environment_variables.WANDB_API_KEY=$WANDB_API_KEY
```

2,500 iterations × 504 × 4096 tokens ≈ 5.16B tokens, ~3–5 h on
`gpu-nc80-lp` (2× H100; gradient accumulation keeps the cluster's global
batch of 504).
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

## 7. Full training run

The full mixture is already on the blob store from §5c — the pilot and the
full run read the same `tokenized/mix_30_70/full`, they just train a
different number of iterations:

```bash
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
(`ApertusForCausalLM`) — a few minutes on one node:

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
source env.sh   # W&B entity/project from configs/hf_wandb.json; key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"
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

The first real experiment on Azure: two 2-language models — DCLM-edu
(English) + FineWeb-2-HQ Russian, 50/50 — at **90M** (L15×d768, 92.9M
non-embedding params) and **1.7B** (L30×d2304, 1.67B), both trained like
the sweep (50,000 iters ≈ 103.2B tokens, GBS 504, seq 4096; hyperparams in
`../hyperparams/hyperparams_deep.json`, cells in `configs/models.json` as
`apertus-{90M,1.7B}-fwEdu50-fw2ru50-seed28`), auto-evaluated every 5
checkpoints.

The data is the CSCS-built EN+RU pair from §5a/§5b — the same
`english_dclm` + `fineweb_L2` builds the predictivity sweep uses (the 52B
L2 build covers this run's 51.6B Russian half). Compose the bilingual
mixture dir **server-side** from the already-uploaded builds (no re-upload;
repeat against the UK account for the 1.7B) and give it the 2-line blend
manifest `train.sh` reads:

```bash
source env.sh   # W&B entity/project from configs/hf_wandb.json; key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"

# 1. Compose tokenized/mix_enru_50_50/full from the predictivity builds
#    (same $ES / $ES_SAS as §5b; mint a fresh SAS if the old one expired):
ES="https://snreswsstorage217e4ec3bb.blob.core.windows.net/azureml-blobstore-2ed07b42-5369-425f-b7f2-df29b3684e32"
export ES_SAS='PASTE_THE_TOKEN_HERE'
azcopy copy "$ES/predictivity/data/english_dclm/?$ES_SAS" \
            "$ES/tokenized/mix_enru_50_50/full/?$ES_SAS" --recursive
azcopy copy "$ES/predictivity/data/fineweb_L2/?$ES_SAS" \
            "$ES/tokenized/mix_enru_50_50/full/?$ES_SAS" --recursive
printf '0.50 english_dclm/english_dclm\n0.50 fineweb_L2/fineweb_L2\n' > data_path.txt
azcopy copy data_path.txt "$ES/tokenized/mix_enru_50_50/full/?$ES_SAS"

# 2. Launch the trainings: 90M on Spain, 1.7B on the UK Spot pool
python launch_azure_trainings.py --size 90M --seed 28
AZ_RG=$AZ_UK_RG AZ_WS=$AZ_UK_WS python launch_azure_trainings.py \
    --size 1.7B --seed 28 --compute gpu-nd96-spot

# 3. Auto-evals while they train
python auto_evals.py --watch 600
```

Expectations: the **90M ≈ 1.5–2 days (~$150)** on `gpu-nc80-lp`; the
**1.7B belongs on the UK `gpu-nd96-spot` pool — ≈ 4–5 days (~$2,400)**
(on the 2-GPU Spain node it would take ~2.5 weeks; raising `max_instances`
doesn't help, training is single-node). MBS is preset per size (21 for 90M,
2 for 1.7B) so the global batch of 504 divides evenly on 2 or 8 GPUs. The
final checkpoints then take the same convert → full-eval path as any other
cell (`--ckpts final|full_eval`).

## 11. The predictivity sweep (Spain Central + UK South, discounted meters)

The small-to-large predictivity sweep (51 runs per intervention level; see
`.claude-shared/plans/small-to-large-predictivity-training-plan.md` and the
compute-budget sheet next to it) runs on **two workspaces**: Spain Central
(`gpu-nc80-lp`, 2×H100 at the fixed low-priority meter — every size ≤600M)
and UK South (`gpu-nd96-spot`, 8×H100 Spot — the 1B and 1.7B rungs). Set the
`AZ_ES_*` / `AZ_UK_*` names in `env.sh`, then run `setup_azure.sh` once per
workspace (export `AZ_LOCATION/AZ_RG/AZ_WS` to each region's values first;
computes whose SKU a region doesn't offer are skipped with a warning).

**Data**: the per-build folders under `predictivity/data/` in both
workspaces' blob stores, built at CSCS and shipped with azcopy — the whole
pipeline is §5 (build order: EN+RU first, then the remaining settings).

**Launch** (same filters as the cluster launcher; jobs queue on the clusters
and run as nodes free up — resubmitting any cell resumes it):

```bash
source env.sh   # W&B entity/project from configs/hf_wandb.json; key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"
python launch_azure_predictivity.py --dry-run          # the whole 51-job grid
python launch_azure_predictivity.py --langs 1          # monolingual anchors first
python launch_azure_predictivity.py                    # everything
python launch_azure_predictivity.py --arch shallow     # the depth-intervention variant
```

`--arch` picks the reviewed architecture family — `deep` (default baseline,
`../hyperparams/hyperparams_deep.json`) or `shallow` (`../hyperparams/hyperparams_shallow.json`, same
non-embedding sizes at width/depth 128); the D(N) = 100 × N schedule comes
from each config's `predictivity` block. Runs land in the W&B entity/project
from `configs/hf_wandb.json` (currently `msnr`) — injected into every job by
`launch_azure_predictivity.py`, no per-run setup — named
`apertus-<size>-L<L>-seed<seed>` (shallow runs as `...-L<L>-shallow-...`).
Micro-batch sizes tuned for the cluster are auto-shrunk per node (`train.sh`)
so the global batch of 504 always divides; the 1.7B resolves to MBS 1 on the
8-GPU nodes — override with `--set environment_variables.MBS=3` if it fits.

## 12. Launching the remaining cells

Every other size × mixture × seed cell uses the same `train-full.yml` /
`eval.yml` templates; the launchers fill in the per-cell architecture
(from `../hyperparams/hyperparams_deep.json`), mixture and paths — same filter flags as
the cluster's `launch_trainings.py`:

```bash
source env.sh   # W&B entity/project from configs/hf_wandb.json; key stays in your saved shell env
[ -n "$WANDB_API_KEY" ] && echo "✓ WANDB_API_KEY present" || echo "✗ set WANDB_API_KEY in your shell profile (~/.zshrc)"

# preview, then launch: all three mixtures of 350M seed 28
python launch_azure_trainings.py --size 350M --seed 28 --dry-run
python launch_azure_trainings.py --size 350M --seed 28

# evals: the canonical 10-checkpoint set, full pretraining task list
python launch_azure_evals.py --size 350M --seed 28 --ckpts full_eval --tasks pretraining_full --dry-run
```

Each cell's mixture must exist first — the cluster's frozen tokenized
mixture copied to `tokenized/mix_60_40/full` etc. per §5c — and each
evaluated checkpoint needs a `convert.yml` run first. With one
`max_instances: 1` cluster, submitted jobs queue and run one at a time —
raise `max_instances` (and your quota) to run cells in parallel.

## 13. Where everything is stored (and getting it out)

All artifacts live in the workspace's blob storage under `workspaceblobstore`:

| Path                                                  | Contents                                      |
| ----------------------------------------------------- | --------------------------------------------- |
| `tokenized/mix_<edu>_<fw2>/<scale>`                   | `.bin`/`.idx` shards + `data_path.txt`        |
| `runs/<exp_name>/checkpoints`                         | Megatron `torch_dist` checkpoints (`iter_*/`) |
| `runs/<exp_name>/logs`                                | TensorBoard + W&B offline files               |
| `models/<exp_name>/iter_<N>`                          | converted HF snapshots                        |
| `eval_logs/<entity>/<project>/<NAME>/harness/eval_*/` | lm-eval results + samples                     |

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
# Uses your saved HF login (token on this laptop) — verify, don't re-enter:
huggingface-cli whoami >/dev/null 2>&1 && echo "✓ HF logged in" || echo "✗ run: huggingface-cli login"
huggingface-cli upload <your-org>/apertus-175M-fwEdu30-fw270-seed28 \
  ./iter_0050000 . --revision stage1-step-50000 --private
```

## 14. Teardown

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

## 15. Common mistakes

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
  doesn't point at the directory that _contains_ `<entity>/<project>/...`.
- **No W&B run appeared** → `WANDB_API_KEY` wasn't passed to the job
  (`--set environment_variables.WANDB_API_KEY=$WANDB_API_KEY`); the job
  still trains, logging only to TensorBoard (`runs/<exp>/logs/tensorboard`).
