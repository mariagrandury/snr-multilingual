# SNR evaluation pipeline

> Evaluate language models at many training checkpoints to feed the SNR
> framework. Built on top of
> [`lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness)
> with W&B integration; runs on the CSCS Alps SLURM cluster.

Jobs install the harness from a **pinned shared checkout**, not from GitHub —
one clone per job broke at fleet scale (HTTP 401 at ~120 queued evals; a pass
now submits ~700) and silently tracked whatever HEAD was current, which makes
harness drift indistinguishable from checkpoint noise. Order of preference:
prebuilt wheel → in-tree build from the checkout → GitHub (only with an
explicit `LM_EVAL_HARNESS_BRANCH`, or if the shared tree is missing). Each
job logs and records in `job.json` which one it used.

```bash
HARNESS_SRC=/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/lm-evaluation-harness
git -C $HARNESS_SRC pull                       # refresh deliberately, then rebuild:
rsync -a --exclude .git --exclude build --exclude '*.egg-info' \
      $HARNESS_SRC/ /iopsstor/scratch/cscs/$USER/tmp-harness-build/src/
pip wheel --no-deps --no-build-isolation \
      -w $(dirname $HARNESS_SRC)/wheels /iopsstor/scratch/cscs/$USER/tmp-harness-build/src
```

Build out-of-tree: `pip install <dir>` writes `build/` **inside** the source,
so concurrent jobs delete each other's files mid-copy. Nothing checks that the
newest wheel matches the checkout — skip the rebuild and jobs keep installing
the old code, which is what `harness_src_commit` in `job.json` is for.

## What this produces

A single source of truth for every (model, checkpoint, task, metric)
score in the project, in three forms:

| Output | Where | Used by |
|---|---|---|
| Per-task JSON results + samples | `eval_logs/<entity>/<project>/<model>/harness/eval_<ts>_<jobid>/` | `build_hf_dataset.py` → HF dataset |
| HF dataset (`multilingual-snr/multilingual-snr-eval-results`) | three parquet splits: `pretraining_custom`, `pretraining_a06`, `reference_hf` | the SNR framework in [`../signal-and-noise/`](../signal-and-noise/) |
| W&B per-model curves | [`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments) | live dashboards, one line per benchmark per model |

The HF dataset is the canonical input to every analysis in
[`../signal-and-noise/`](../signal-and-noise/): the SNR-variant CSV, the
benchmark_creation per-family ranking, the AllenAI cross-corpus
comparison, and the subset-search outputs all start from it.

## Models in scope

**36 Apertus pretrains** (the canonical 4 sizes × 3 mixes × 3 seeds from
[`../pretrain/`](../pretrain/)), plus reference HF models (Qwen3, Gemma-3,
SmolLM3, Olmo-3, Apertus-8B/70B) and the a06 main runs (`apertus3-{1b,3b}-*-nodes`).

The full list lives in [`configs/models.json`](configs/models.json) (the
shared source of truth, read via
[`scripts/utils/configs.py`](scripts/utils/configs.py)). Pools group them
for downstream SNR analysis — see
[`../signal-and-noise/README.md`](../signal-and-noise/README.md).

## Tasks in scope

86 tasks per checkpoint — the deduplicated union of
[`configs/signal_to_ratio/tasks_pretraining.txt`](configs/signal_to_ratio/tasks_pretraining.txt)
and `tasks_pretraining_b.txt`, exposed as the launcher mode
`snr-pretraining-full`. Coverage includes per-language benchmarks
(`multiblimp_<lang>`, `xstorycloze_<lang>`, `xwinograd_<lang>`,
`hellaswag_<lang>`, `xnli_<lang>`, `xcopa_<lang>`, `paws_<lang>`,
`belebele_<lang>`, `global_mmlu_full_<lang>_<subject>`, …) for 12+
languages and standalone English benchmarks (`arc_challenge`, `arc_easy`,
`hellaswag`, `piqa`, `openbookqa`, `mmlu`, `commonsense_qa`, …).

Task lists live under [`configs/signal_to_ratio/`](configs/signal_to_ratio/):

| File | Purpose |
|---|---|
| `tasks_pretraining.txt` | 48 tasks — pretraining-stage subset A |
| `tasks_pretraining_b.txt` | 38 tasks — pretraining-stage subset B |
| `tasks_pretraining_full.txt` | 86 tasks — dedup union, used by `snr-pretraining-full` |
| `tasks_posttraining.txt` | post-training tasks (instruct/SFT) |
| `*_main_table.txt` | matching `task/metric` pairs for the W&B summary table |

### Reformulated tasks (`tasks/rf/`)

The three auto families that ask for a letter — `belebele`,
`global_mmlu_full`, `include_base_44` — sit at chance for base models at
these sizes. [`scripts/make_rf_tasks.py`](scripts/make_rf_tasks.py)
generates a cloze twin of each (`rf_<task>`: same dataset, config and
split, no lettered option list, the four answer strings scored as
continuations, zero-shot, `acc` + `acc_norm`) under
[`tasks/rf/`](tasks/rf/) and registers them in `configs/tasks.json`
(`benchmark: rf_<family>`, `metric: acc_norm`, group `auto_rf`). The YAMLs
reach the harness through `eval_worker.py --include_path`, which
`evaluate.sbatch` passes when `HARNESS_INCLUDE_PATH` is set, so the pinned
wheel is untouched. Re-run the generator after adding a language to any of
the three families; it is idempotent. The lettered probe families (below)
get twins from the same script, through an absolute `include:` of the
original YAML. The second twin, `rfgm_<task>`
(`--set rfgm`, group `auto_rfgm`, [`tasks/rfgm/`](tasks/rfgm/)), is the
same item rewritten by Gemini into a statement stem with four short
continuations: [`scripts/rewrite_items_gemini.py`](scripts/rewrite_items_gemini.py)
runs the Batch API from the login node and leaves one JSONL per task under
`/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/rf-data/rfgm/`, which the
YAMLs read through `dataset_path: json` (offline; never published, it
carries the gold labels). Design, the step-by-step guide, the prompt and
the cost: [`analysis/rq00_task_reformulation/`](../signal-and-noise/analysis/rq00_task_reformulation/README.md).
Its `compare.py` reads every task set off
the ladder report (the deep scheme-A seed-1904 cells, through the rq00
gate; `run_all_predictivity.sh` runs it) and rewrites the
figures and the generated table in that README (`rf_gate.png`, per
language `rf_gate_by_language.png`, a CSV each): the rq00 gate cell —
median task margin over chance, trained languages — original, rf, rfgm,
and each set's difference.

### Probe benchmarks (`groups.auto_probe`, `tasks/cloze/`, `tasks/include_v2/`)

A screening pass over candidate benchmarks at each cell's last checkpoint:
`auto_evals_cscs.py --group auto_probe --size 600M,1B,1.7B --final-only`.
A candidate that passes is promoted into `auto`: twenty of the
twenty-one were on 2026-09-23, all but `bbq`. Two generators feed it, both idempotent and both
registering in `configs/tasks.json`:

- [`scripts/make_cloze_tasks.py`](scripts/make_cloze_tasks.py) turns the
  closed-form subtasks of BBH and ACP-Bench into logprob tasks under
  [`tasks/cloze/`](tasks/cloze/), one benchmark per arm so the arms stay
  separable by name: `bbh_mcq` / `rf_bbh_mcq` (the reformulation pair),
  `bbh_cloze` (two-way, no twin), and the same three for `acp_bench`.
  Option and item counts are measured from the data and written as
  `n_options` / `n_items`.
- [`scripts/make_include_v2_tasks.py`](scripts/make_include_v2_tasks.py)
  writes INCLUDE v2 (`include-results/include-128`, the L50 pairs, OG and EN
  variants) as cloze tasks under [`tasks/include_v2/`](tasks/include_v2/),
  in the same group, read straight from the cached parquet.

Run the generators one at a time: each rewrites all of `tasks.json`, and
`utils.configs.write_tasks_json` refuses to write over a file that changed
underneath it. Load every new task through a `TaskManager` before launching;
`src/evals/CLAUDE.md` lists what only fails inside the job. Once the
results are in, `src/signal-and-noise/analysis/rq00_task_reformulation/probe.sh`
produces the verdict: the gate per language and original-vs-`rf_` on the
pairs. A candidate that earns its place is then added to `groups.auto`.

## How to run

The two-step flow: **generate a runner** from a models file (lists which
checkpoints to evaluate), then **launch** evaluations using the standard
launcher script. See
[`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md)
for the canonical SNR-experiments instructions.

The one-liner that drives the full sweep (idempotent — re-runnable):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals && git pull && \
  bash scripts/launch_evaluations.sh snr-pretraining-full \
      --script runners/snr_pretraining_all.sh --time 12:00:00
```

Re-running is safe at two layers:

| Layer | Where | When |
|---|---|---|
| Per-checkpoint | [`runners/hf_base_runner.sh`](runners/hf_base_runner.sh) (calls `scripts/_eval_status.py`) | Before each `sbatch` — skips submission entirely if every task already has results for that ckpt |
| Per-task | [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) (calls `scripts/_eval_status.py`) | Inside a running job — filters `$TASKS` down to remaining; logs skipped to `skipped_tasks.log`; exits cleanly with no work. One [`scripts/eval_worker.py`](scripts/eval_worker.py) per GPU then runs the rest, each task's results landing in `per_task/<task>/` as it finishes |

Both use the same disk-scan in
[`scripts/_eval_status.py`](scripts/_eval_status.py): a task is "done" iff
a non-empty `eval_*/per_task/<task>/` exists (killed runs) **or** any
`eval_*/results_*.json` lists it under `.results` (clean runs).

Live progress dashboard:

```bash
python3.11 scripts/snr_progress.py                                    # per-ckpt summary
python3.11 scripts/snr_progress.py --status not_submitted             # gaps
python3.11 scripts/snr_progress.py --details --filter <NAME-substr>   # per-task
```

What the jobs cost, and how much a killed one kept — the numbers behind
`auto_evals_cscs.MIN_PER_TASK`, split by pipeline generation (a run is
`worker` if its eval dir has a `job.json`, `batched` otherwise):

```bash
python3.11 scripts/eval_timing.py             # median min/task + kill survival
python3.11 scripts/eval_timing.py --detail    # one row per job
```

To prove a pipeline change did not move any score, re-evaluate one checkpoint
into a throwaway `WANDB_PROJECT` (a different project is a different
`eval_logs` subtree, so the per-task gate sees no prior results and runs
everything) and diff the two:

```bash
python3.11 scripts/diff_scores.py <NAME> msnr <the throwaway project>
```

System Python on login nodes is 3.6; use `python3.11` for the dashboard.

When `normal` is backed up, [`scripts/debug_drain.sh`](scripts/debug_drain.sh)
feeds already-pending convert/eval/bpb jobs through the idle `debug` partition
(capped at debug-qos' 1 running + 1 queued, each capped to debug's 1:30 wall).
It never submits anything new, so it cannot duplicate work; conversions go
first, since a cell cannot be evaluated before its HF snapshot exists.

Truncating a job to 1:30 is safe because all three kinds **resume**:
conversions and BPB write a per-checkpoint marker and skip what already
carries it, and evals write every task's results as it finishes (CLAUDE.md
bug 13) — the auto-eval watcher then resubmits a killed eval with only the
tasks still missing. So every pending job is moved regardless of the walltime
it asked for, conversions first, evals next, BPB last.

```bash
bash scripts/debug_drain.sh --dry-run   # what it would move
bash scripts/debug_drain.sh             # loop until nothing is pending
bash scripts/debug_drain.sh --ensure    # start one in the background if none runs
```

It **exits** once nothing movable is left — it drains a batch, it does not
stand guard. That is why the submitters call `--ensure`:
[`launch_bpb.sh`](scripts/launch_bpb.sh) after queueing BPB jobs, and
`auto_evals_cscs.py` at the end of every watch pass. `--ensure` starts a
detached drainer only when none is running (log in `logs/`), so calling it on
every submission cannot stack up drainers racing for the same two debug slots.

The same idea for a reservation:
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/reservation_drain.sh`
moves any of my pending `normal` jobs (pretrain, eval, convert, BPB) into the
reservation named at its top, holding at most `--max-nodes` of it (default
63) and always leaving `--min-free` nodes for others (default 50).
Everything it moves must be able to finish within `--hours` (integer, default
12) of the start: a job whose walltime no longer fits in the time left stays
on `normal`, and the loop exits when the window is over. Jobs go in
by most node-hours first, then job name; `--priority eval` (or `shallow`,
`L30`, any substring) puts the matching jobs ahead of that order. A job that
does not fit the remaining room is passed over and smaller jobs behind it
backfill; when a big job finishes, the freed room goes to the next big job
first, since it is earlier in the order. Jobs it has placed in the
reservation are left alone by the debug drainer.

```bash
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/reservation_drain.sh --dry-run
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/reservation_drain.sh --priority eval
```

And the same idea for the `preemptable` partition:
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/preempt_drain.sh`
moves pending convert/eval/pretrain/BPB jobs there, holding at most
`--max-nodes`
(default 100). `normal` is capped by its QOS at 480 nodes for the whole partition, so a
full cluster parks these jobs on `QOSGrpNodeLimit` for hours, while
`preemptable` has every node and no group cap — its price is preemption
(4 min grace, then cancelled: Clariden runs `JobRequeue=0`), which costs a
conversion its in-flight checkpoint and an eval its in-flight tasks, both of
which the next watcher pass resubmits and redoes. A BPB chain already has its
next link queued, except when a link dies before scoring its first
checkpoint: the chain's no-progress guard then ends it, and `launch_bpb.sh`
restarts it (it skips scored cells and cells with a job in flight).
Data builds are **not** moved (2026-09-20): the first six put there were
cancelled by the system 16 s after starting, before reaching the line that
queues their successor, and six chains died. A build asks for 32 CPUs and
`normal`, being `OverSubscribe=EXCLUSIVE`, has always given it a whole node
anyway; `preemptable` is `FORCE:1` and really did pack all six onto one node
beside a stranger's job. `scontrol` cannot add `--exclusive` to a queued job,
so builds belong on `preemptable` only by being *submitted* there —
`BUILD_PARTITION=preemptable ./launch_builds.sh`, which asks for the node and
queues two segments (../pretrain/README.md).

Pretrain jobs are moved only at the top rungs (3B, deep 1.7B and 1B) **and
only when they can come back from a preemption**, which is what
`launch_trainings.py --partition preemptable` arranges: it sets
`MEGATRON_EXIT_ON_SIGTERM=1`, so the patched handler catches the preemption
signal itself and Megatron checkpoints inside the 4 min grace, and
`PRETRAIN_CHAIN=1`, so the wrapper has already queued a singleton successor to
resume from that save. The drainer asks the controller per job
(`squeue -O Comment,Requeue`: `selfchain`, or `--requeue` for jobs still in
flight from before 2026-09-21) and skips the rest — for those a preemption
really would cost a save interval on 21 nodes.

Nothing is truncated (preemptable allows 24 h), so this one is simpler than
the debug drainer. It cannot *raise* a walltime either — a limit can only be
lowered after submission — so a job moved here keeps the 12 h `normal` wall;
submit to `preemptable` up front (`--partition preemptable`,
`--partition preemptable`) to get the full 24. Order: builds, conversions,
the final checkpoint of a model, the 3B / 1.7B-deep / 1B-deep pretrainings,
the 20/40/60/80 % points, the other evals, then anything with `bpb` in its
name — the fraction read from the size's own schedule, so it means the same at
every rung. Unlike the other two it does not exit when the queue empties; it
keeps moving what later watcher passes submit, until you kill it.

```bash
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/preempt_drain.sh --dry-run
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/preempt_drain.sh --max-nodes 100
```

## Bits-per-byte: the second way to evaluate a model

Benchmarks are not the study's outcome metric. The predictivity plan's outcome
is **per-language bits-per-byte on the fixed held-out validation set**, and it
is measured by a different path from everything above — no lm-eval, no vLLM,
no benchmark tasks:

```bash
bash scripts/launch_bpb.sh --dry-run    # every cell with unscored checkpoints
bash scripts/launch_bpb.sh              # submit them
sbatch --job-name=bpb-<cell> scripts/score_bpb.sbatch <cell> [iters...]   # one cell
```

[`scripts/launch_bpb.sh`](scripts/launch_bpb.sh) is the normal entry point and
is safe to re-run: it skips cells that are fully scored and cells that already
have a job in flight. That second check is not optional — `score_bpb.sbatch`
queues its own successor (below), so a cell mid-chain always has a PENDING job
that a naive loop would duplicate.

One job per cell scores every converted checkpoint it finds and writes
`<LOGS_ROOT>/<entity>/msnr/<cell>-iter<N>/bpb/bpb.json` — per language, the
NLL, the byte count, `bpb` and `ppl`. `ppl` is `Infinity` when a diverged
checkpoint averages more than ~709.8 nats/token, past what a double can
exponentiate; its `bpb` stays finite.

How it differs from the harness path, and why:

* **The data is the validation build, not a benchmark.**
  `build_data_mixtures.py --scheme A --stage validation` wrote one `.bin` per
  language plus `validation.manifest.json`, and every training build skipped
  exactly those rows, so train and validation are disjoint by construction.
  There is one validation build for the whole sweep — the other data schemes
  symlink this manifest into their own data dirs — so BPB is comparable
  across schemes, not just across sizes.
* **Nothing is generated.** It is a single teacher-forced forward pass per
  block, summing `-log p` over targets. Blocks overlap by one token so every
  token except the corpus's first is scored with a predecessor.
* **The denominator is measured, not assumed.** BPB divides by UTF-8 bytes, and
  the bytes are obtained by decoding the scored tokens — verified to reproduce
  the manifest byte count exactly on Latin, Cyrillic and CJK. Scaling the
  manifest total by a token fraction would be wrong for a prefix, because
  bytes-per-token varies per document.
* **Documents are concatenated without EOD**, so the numerator and denominator
  describe the same text. An inserted EOD would add likelihood cost that no
  byte in the denominator pays for.
* **It is comparable across tokenizers**, which accuracy is not — the
  denominator is bytes. That is why the plan chose it for the tokenizer
  intervention.

**A cell does not fit one debug slot**, so the job carries itself across
several. Scoring costs a flat amount per checkpoint — ~444 s at 90M, 826 s at
350M, 1,506 s at 1B, 2,141 s at 1.7B (medians over 1,023 jobs, <6% spread on
debug) — against 20 checkpoints per cell (40 at 1B, 60 at 1.7B), so a 1:30
debug slot gets through 11 of them at 90M and 2 at 1.7B. `score_bpb.sbatch`
therefore queues a `--dependency=singleton` successor **before** it starts
scoring (at the wall Slurm kills the batch script, so anything submitted
afterwards would never run), and refuses to start a checkpoint it cannot
finish in the time left, using the previous checkpoint's measured cost. The
successor re-derives what is still due and exits without chaining when
nothing is, so the chain ends itself; `MAX_CHAIN` (default 32 — a 60-save
1.7B cell drained to 1:30 debug slots needs ~30 links) bounds it against a
failure loop. A link that finds none of its predecessor's due checkpoints
scored exits non-zero without chaining, so a checkpoint that always fails
stops the chain after one wasted link instead of at that cap, and every link
exits with the scoring step's code, so a failed link shows as FAILED.

`--max-tokens` (default 1M/language) takes a deterministic leading-document
prefix, so every model is scored on byte-identical text; `--max-tokens 0` uses
the full ~5M. The offline flags in the sbatch are not optional: without
`HF_HUB_OFFLINE=1`, `from_pretrained` on a **local** path still calls the Hub
and blocks for ~25 min per checkpoint on a compute node.

Read the results with
[`../pretrain/ladder_report.py`](../pretrain/ladder_report.py), which also
cross-checks them against the loss curves and the benchmark scores.

## Outputs in detail

### SLURM logs (per job)

`<repo>/logs/<job_name>_<job_id>.{out,err}`. Job name pattern:

- single-node: `eval-<model_name>`
- split-K: `eval-<model_name>-split<i>` + `eval-<model_name>-aggregate`

Where `<model_name>` is `<base>-iter<N>` (Megatron) or `<base>-<branch>`
(HF).

### Harness + W&B logs (per checkpoint)

`$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/<model_name>/harness/eval_<timestamp>_<jobid>/`,
with SNR defaults:

- `LOGS_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs`
- `WANDB_ENTITY=mariagrandury-epflnlp`
- `WANDB_PROJECT=snr-experiments`

Each `eval_*/` holds the merged `results_<timestamp>.json` (every task the
job finished) plus one `samples_<task>_<timestamp>.jsonl` per task, and
beside them:

- `per_task/<task>/<model>/results_*.json` — each task's own results file,
  written *the moment the task completes* (`scripts/eval_worker.py`), so a
  walltime kill loses only the tasks in flight; kept after the merge for its
  per-task timing and config.
- `job.json` — Slurm ids, node, walltime, backend, TP/PP/workers, repo
  commit, container, the harness spec it installed (`harness` +
  `harness_src_commit`), start/end and the done/failed/skipped counts. Written
  before the eval too, so a killed job still leaves the header (and
  `status: started` is how a reader tells a kill from a clean failure).
- `worker_<i>.log` per worker, `failed_tasks.log` (`<task>\t<reason>`),
  `skipped_tasks.log` (already evaluated elsewhere), and `inflight/<task>/`
  only for a task that never published (its worker died, or it raised after
  writing partial output).

### W&B per-model curves

[`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments)
— one W&B run per *model* (not per ckpt), with each ckpt logged as one
history step. Pushed by
[`scripts/push_all_results.py`](scripts/push_all_results.py) from inside
the eval job and also runnable on the login node for bulk rescue. FLOPs
is the default x-axis (`define_metric(*, step_metric="flops")`); iter and
tokens are also defined and can be swapped in the W&B UI.

### HF dataset build

`build_hf_dataset.py` walks `eval_logs/.../snr-experiments/` and emits the
three parquet splits (`pretraining_custom`, `pretraining_a06`,
`reference_hf`) consumed by the SNR analysis pipeline. The build resolves
model metadata (size, params, tokens, family, split) from
`configs/models.json` via the shared `configs.py` loader and computes
`tokens` (Megatron iter × 504 × 4096 or HF branch value) and
`compute ≈ 6 × params × tokens` at build time.

## Repository structure (used paths only)

```
evals/
├── configs/
│   ├── models.json                      # shared model registry
│   ├── tasks.json                       # task → stage mapping (where applicable)
│   └── signal_to_ratio/                 # SNR experiment configs
│       ├── README.md                    # canonical SNR-experiments how-to
│       ├── models_pretraining_custom*.txt
│       ├── models_midtraining_hf.txt
│       ├── models_posttraining_hf.txt
│       ├── models_test_{hf,megatron}.txt
│       ├── tasks_pretraining{,_b,_full}.txt
│       └── tasks_posttraining.txt
├── scripts/
│   ├── launch_evaluations.sh            # primary entry point
│   ├── evaluate.sbatch                  # SLURM job script
│   ├── aggregate_splits.sbatch          # split-aggregation job
│   ├── generate_snr_runner.sh           # runner generator
│   ├── list_checkpoints.sh              # ckpt enumerator
│   ├── score_bpb.py                     # per-language BPB + perplexity
│   ├── score_bpb.sbatch                 # BPB job, one per cell (self-chaining)
│   ├── launch_bpb.sh                    # submit BPB for every cell still due
│   ├── mirror_eval_logs.sbatch          # rsync eval_logs -> capstor; touch ckpts vs purge
│   ├── snr_progress.py                  # progress dashboard
│   ├── _eval_status.py                  # idempotency disk scan
│   ├── _run_per_task.sh                 # inner runner: one eval_worker.py per GPU, merge at the end
│   ├── eval_worker.py                   # model loaded once, tasks one at a time, results per task
│   ├── eval_timing.py                   # min/task and kill-survival, batched vs worker pipeline
│   ├── diff_scores.py                   # same ckpt, two projects: did a change move any score?
│   ├── push_all_results.py              # W&B per-model push
│   ├── build_hf_dataset.py              # HF dataset builder
│   └── utils/configs.py                 # shared config loader
├── runners/
│   ├── hf_base_runner.sh                # submission loop, idempotency gate
│   ├── snr_pretraining_all.sh           # 36 models × 13 canonical iters = 468 cells
│   ├── snr_pretraining_local_hf.sh      # vLLM on converted Megatron ckpts
│   ├── snr_pretraining_hf_top.sh        # SmolLM3-3B / Olmo-3-7B / Apertus-8B (HF)
│   └── snr_pretraining_hf_70b.sh        # Apertus-70B (HF)
├── containers/
│   ├── env.toml                         # standard HF eval container
│   └── env_vllm.toml                    # vLLM container (recommended)
└── logs/                                # SLURM stdout/stderr
```

## Backends

| Backend | When | Notes |
|---|---|---|
| `vllm` | **Recommended.** Faster + dramatically more memory-efficient than the Megatron eval path. Used by `runners/snr_pretraining_local_hf.sh` (vLLM on converted Megatron ckpts) and the HF runners. Runs `EVAL_WORKERS` workers per job (default: one per GPU the TP × PP layout leaves free), each with its own model copy. | Generation tasks (gsm8k, squadv2) may differ slightly between backends — only compare results across models using the same backend. |
| `hf` (accelerate) | Default in `evaluate.sbatch`. | Slower; used when vLLM doesn't fit a particular task. |
| `megatron_lm` | Direct evaluation of Megatron ckpts (no HF conversion). | Has known memory pressure issues; prefer the local-HF (vLLM) runner instead. |

## Secrets

`evaluate.sbatch` reads env first, files in `scripts/` as fallback:

- `WANDB_API_KEY` — for the W&B push. Needs membership in
  `mariagrandury-epflnlp` entity.
- `HF_TOKEN` — for HF Hub fetches (reference HF models).
- `CSCS_SERVING_API` — for LLM-as-judge evals (e.g. AlpacaEval). Key at
  https://serving.swissai.cscs.ch.

`HF_HOME` and `HF_HUB_CACHE` are forwarded into the container by
`evaluate.sbatch` via `INNER_EXPORTS`; the populated cache lives at
`/capstor/store/cscs/swissai/infra01/users/$USER/hf_models` (~258 GB,
mounted by both `containers/env.toml` and `containers/env_vllm.toml`).

## See also

- [`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md)
  — canonical SNR-experiments how-to: generate runners, launch, smoke
  tests, rescue procedures.
- [`CLAUDE.md`](CLAUDE.md) — back-of-house notes: bug history (vLLM
  tokenizer revisions, Megatron container choice, idempotency edge cases,
  HF Hub rate limits, …), cluster gotchas, the W&B layout.
