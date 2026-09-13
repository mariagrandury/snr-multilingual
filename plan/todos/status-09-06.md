Useful commands:

bash evals/scripts/launch_bpb.sh --filter '600M-L50'
python3.11 pretrain/auto_evals_cscs.py --convert-only
pkill -f pretrain/auto_evals_cscs.py  # kill eval watcher(s)

to cancel:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1}' | xargs -r scancel
to preview:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1, $2}'

---

Evals:
- ✅ Remove afrimmmlu and afrixnli (for now) -> review list of languages and available benchmarks, bloks L100 analysis
- ✅ Job 3311744: eval-175M-L50-deep-seed1904-iter3416 COMPLETED with batch=1
- ✅ sbatch scripts/mirror_eval_logs.sbatch -> now it mirrors evals AND touches megatron ckpt files (Job ID 3312266)
- ✅ OPTIMIZE EVALS (details below)
- ✅ Review optimization and fix bugs
- ✅ Eval one ckpt per size to refit eval time estimations:
    - python3.11 pretrain/auto_evals_cscs.py --name lm-175M-L50-deep-seed1904 --max-submit 1
    - python3.11 pretrain/auto_evals_cscs.py --name lm-350M-L50-deep-seed1904 --max-submit 1
    - python3.11 pretrain/auto_evals_cscs.py --name lm-600M-L50-deep-seed1904 --max-submit 1
    - python3.11 pretrain/auto_evals_cscs.py --name lm-1B-L50-deep-seed1904 --max-submit 1 --every 4
    - python3.11 pretrain/auto_evals_cscs.py --arch deep --scheme A --max-submit 1 (90M)
- ✅ Verify prefix caching (diff_scores.py)
- Launch auto_eval_cscs.py --retry-held (and scancel the 90M models)

BPB:
- ✅ bash evals/scripts/launch_bpb.sh --filter '1B'
- bash evals/scripts/launch_bpb.sh --filter 'deep' (1.7B + 90M L8)
- bash evals/scripts/launch_bpb.sh --filter '600M' (40 ckpts, all shallow)
- loss/BPB scaling fits

1B & 1.7B:
- ✅ python3.11 pretrain/auto_evals_cscs.py --convert-only
- fit 1B and 1.7B training estimates
- resume training of 1.7B models -> maybe on Wednesday better
- launch 1Bs again?
- touch angelika's ckpts

SNR:
- Update the SNR module to new naming
- Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 6 language settings

L100:
- [Discuss] Update the list of available high-quality benchmarks for low resource languages.
- Launch the creation of the L100 data mixture with the new list of languages (1 day). Blocks pretraining of all L100 models. Blocked by update of available benchmarks for low resource languages.
- [Discuss] Update the model grid plan so each size-languages cell has at least 3 models (we need 3 to calculate DA).
- The 4 small sizes of L100 models (classic: deep, A).


## Optimize Evals I (VERIFIED)

Merge changes in evaluate.sbatch (if needed after other eval optimization fixes):

from

LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/swiss-ai/lm-evaluation-harness.git"
if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
fi

to

# Install the harness from a SHARED CHECKOUT, not from GitHub.
#
# Every job used to run `pip install git+https://github.com/swiss-ai/...`,
# i.e. one fresh clone per job. That is fine for a handful of jobs and fails
# as a fleet: with ~120 evals queued on 2026-09-02 the partial clone's
# promisor fetch started coming back `HTTP 401`, pip aborted, and the job died
# in 35 s on `lm_eval: command not found` — 90 failures in a row, none of
# which reached a single dataset. The clone is also the slowest part of a
# short eval, and it silently tracked whatever HEAD was at submission time,
# so two checkpoints of the same cell could be scored by different harness
# versions. A pinned local checkout fixes all three.
#
# Refresh it deliberately (login node, then note the commit in the eval log):
#   git -C $HARNESS_SRC pull
HARNESS_SRC=${HARNESS_SRC:-/capstor/store/cscs/swissai/infra01/msnr-harness/lm-evaluation-harness}
# A PREBUILT WHEEL is the first choice, because `pip install <dir>` builds
# in-tree: setuptools writes build/ INSIDE the source directory, so every
# concurrent job shares one build tree and they delete each other's files
# mid-copy — `error: [Errno 2] No such file or directory`, then
# `lm_eval: command not found`. Two of six jobs died that way on 2026-09-03;
# the ones that ran alone were fine, which is exactly the signature of a race.
# Installing an artifact writes nothing shared and skips the ~15 min build.
#
# Rebuild it after refreshing the checkout (login node, minutes):
#   git -C $HARNESS_SRC pull
#   rsync -a --exclude .git --exclude build --exclude '*.egg-info' \
#         $HARNESS_SRC/ /iopsstor/scratch/cscs/$USER/tmp-harness-build/src/
#   pip wheel --no-deps --no-build-isolation \
#         -w $(dirname $HARNESS_SRC)/wheels /iopsstor/.../tmp-harness-build/src
HARNESS_WHEEL=$(ls -t "$(dirname "$HARNESS_SRC")"/wheels/lm_eval-*.whl 2>/dev/null | head -1)
if [[ -z "$LM_EVAL_HARNESS_BRANCH" && -n "$HARNESS_WHEEL" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="$HARNESS_WHEEL"
    echo "Harness: $(basename "$HARNESS_WHEEL") (prebuilt wheel)"
elif [[ -z "$LM_EVAL_HARNESS_BRANCH" && -d "$HARNESS_SRC" ]]; then
    # No wheel yet: build from the checkout. Correct, but serialize the jobs —
    # see the race above.
    LM_EVAL_HARNESS_PIP_SPEC="$HARNESS_SRC"
    echo "Harness: $HARNESS_SRC @ $(git -C "$HARNESS_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') (in-tree build — no wheel found)"
else
    # Explicit branch, or the shared checkout is missing: fall back to GitHub.
    LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/swiss-ai/lm-evaluation-harness.git"
    if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
        LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
    fi
    echo "Harness: cloning $LM_EVAL_HARNESS_PIP_SPEC (no shared checkout at $HARNESS_SRC)"
fi

## Optimize Evals II

### Review

1. Scope
11 commits, 0e06c2a..bb3fdad — the worker-pool eval optimization (all dated 2026-09-04, on feat/optimize-evals, already pushed to origin). 20 files, +1,098/−407.

Left alone, as pre-existing in-flight work: configs/models.json and src/pretrain/eval_progress.png (rewritten by your watcher on the other login node), and the staged plan/todos/status-09-06.md.

2. Verdict
The design is sound and the code does what the docs claim — I verified the claim/publish/merge/resume protocol end-to-end against fixtures. One measurement bug made the tool that proves the change report numbers ~10× wrong; fixed. One live regression parks healthy checkpoints; it needs your decision.

3. Bugs found
a. eval_timing.py:64 — the A/B compared two different units. Fixed.

For batched runs the fallback counted results keys, but lm_eval expands a group into one row per subtask. On job 3199185 that is 1,067 rows for 100 queued tasks. Result: batched read as 0.08–0.10 min/task, so the tool would have reported the new pipeline as ~8× slower, and its stated purpose — re-fitting MIN_PER_TASK — biased every walltime toward undersizing. Subtracting group_subtasks fixes it, and the corrected tool now lands on the independently fitted constants:

size	measured now	MIN_PER_TASK
90M	0.77	0.67
175M	0.76	0.62
350M	0.84	0.75
600M	0.90	0.85
1B	0.91	2.0 (guess)
1.7B	0.96	2.8 (guess)
That is also your first real measurement at 1B/1.7B — both ~2–3× below the conservative guesses, so those rungs are over-requesting walltime today.

b. eval_timing.py:178 — asserted a conclusion its own table contradicts. Fixed. It printed "every kill kept 0"; the table right above says 5 kept 0 and 1 kept 100. Job 3199185 is real: the batched call finished and wrote at 14:31, then the wall hit during the W&B push. Your step-1 expectation ("the kill table should show every killed job kept zero") will not hold, and now the text explains why instead of denying it.

c. eval_worker.py:172 / _run_per_task.sh:143 — a cleanly-failed task left an empty inflight/<task>/. Fixed. The fixture reported "2 task(s) whose worker died" when one had raised and been logged normally; the worker's docstring claimed inflight residue was only killed tasks. No correctness leak (_eval_status.py globs eval_*/per_task/* and eval_*/results_*.json, never inflight/), but it mislabels the one signal you'd read after a crash. Now only genuinely-dead tasks remain — verified.

d. plan/compute-budget.md:407,425 — stale figures. Fixed. The worked examples were computed pre-afri-removal (1,965/615 vs the current 1,890/585), and "2.5–3.7×" understates the top and badly overstates the bottom: the real per-job range across the grid is 1.0×–3.8× — no gain at all at L1/L2, where the 15-min overhead plus 15-min rounding leave nothing to divide. The table still said "610 submittable" while the prose two paragraphs down said nothing is refused any more.

e. runners/snr_pretraining_local_hf.sh:36, src/evals/README.md:227 — comments the diff made false. Fixed. The runner still described BATCH_TASKS as a live toggle _run_per_task.sh implements; the README listed a job.json "unfinished" count that evaluate.sbatch deliberately does not write (there's a comment saying so).

4. Verification
Fixture, 2 workers × 5 tasks (scratchpad, fake worker — no real data): 3 published, 1 clean failure, 1 simulated dead worker → merge produced 3 tasks, samples moved up, both failure kinds logged exactly once, inflight/ left only the dead one.
Resume: re-run against a populated LOGS_ROOT skipped the 3 finished and ran only the 2 missing; top-up run exited 0 with "already have results".
auto_evals_cscs.py --dry-run: 698 eval submissions, 0 convert, walltimes exactly matching my independent recomputation (148 tasks/600M → 01:15, 233 → 01:45, 329 → 02:15). Reproduced the doc's 56 cells / 730 due / 120 over-cap / 610 submittable, confirming the cost model. configs/models.json was mutated by sync_models_json.sync() as the skill warns — restored to its exact pre-review bytes (ddd753d9…).
bash -n on all 5 changed shell scripts; py_compile on the 4 changed Python files; job_json exercised under login-node Python 3.6 with =-bearing, empty and bracketed values.
pretrain_progress.sync_docs() → "already in sync" for both generated docs; nothing touched.
Could not run: the vLLM path itself (needs a GPU node), auto_evals_azure.py (no az CLI), and enable_prefix_caching.
5. Comparability
One risk, and it's the one you already named. enable_prefix_caching=True (evaluate.sbatch:272) is the only change that can move a score. Everything else — worker pool, per-task writes, job.json, walltime sizing, drainer policy — is plumbing. Your instinct is right and the in-code comment says the right thing: re-run one already-evaluated checkpoint and diff before trusting any cross-time comparison, and drop the flag if scores move. I could not verify V1-default behaviour against the swiss-ai fork from the login node.

Not-fine list: nothing triggered. No task removed from a group here, no config change, no checkpoint or result rewritten.

6. Proposals
P1 — 361 batch-era failed_tasks.log files are being read as per-task evidence, and it has already parked healthy checkpoints. This is the one I'd act on today.

Under BATCH_TASKS=1, _run_per_task.sh wrote every queued task into failed_tasks.log when the single call died. The new failed_in() reads each line as an independent per-task failure. Across your tree: 361 eval dirs carry that old format, 0 carry the new one.

The dry-run consequence:


lm-175M-L50-deep-seed1904: 20 saved | convert - | eval - | HELD BACK on [854, 1708]
    iter 854:  110 task(s) held back — other: AttributeError: ... 'weighted_f1_score'
    iter 1708: 110 task(s) held back — other: AttributeError: ... 'weighted_f1_score'
Those two checkpoints have eval - — nothing pending, permanently parked — on evidence from four Sep-3 runs killed by the afri* defect you already fixed, twice over (removed from the group, and the worker pool isolates task failures now). Since held-back tasks are excluded from submission, the streak can never reset on its own. Two more are parked the same way (lm-600M-L1-deep-seed64 iter 20160, 13 tasks, "unknown: no eval job log found"; lm-175M-L1-deep-seed64 iter 854, 4 tasks, the global-piqa dataset).

Minimal fix — a batch-era log cannot distinguish "this task failed" from "the batch died", so it shouldn't be read as per-task evidence at all:


def failed_in(run: Path) -> dict[str, str]:
    try:
        lines = (run / "failed_tasks.log").read_text().splitlines()
    except OSError:
        return {}
    if not any("\t" in ln for ln in lines):
        return {}   # batched-era format: every queued task, not per-task evidence
    ...
Those runs still count as barren via wrote_results(), so a genuinely broken checkpoint is still caught — just at the checkpoint level, which is all that evidence supports.

P2 — debug_drain.sh now truncates evals, and three truncations during cold start park a checkpoint. The drainer moving over-cap evals is right given resumability, but a 1:30 slot on a large L100 cell can die before the first task lands (cold start alone is ~15 min). task_attempts counts a barren run as a strike against every remaining task, so three drains park it as "unknown". Precise fix: job.json distinguishes these — a wrapper that ran to completion writes status: finished|failed, so status == "started" means the job was killed and knows nothing about any task. Skip those the way attempted_nothing skips no-op runs.

P3 — re-fit MIN_PER_TASK now that (a) is fixed. The measured 1B/1.7B figures (0.91 / 0.96 vs the 2.0 / 2.8 guesses) would cut those rungs' requests ~2–3×, which is exactly the queue-priority win the constants exist for. Recomputing the budget: 940–1,420 node-hours for all 730 evals, i.e. 15–23% of the 6,290 training node-hours, against the doc's current 2,200–2,550 / 26–29%. I did not change those headline numbers — reconciling them to the 09-02 derivation is a decision, not a typo fix.

P4 — job 3312757 is queued as eval- with an empty $NAME. Submitted 11:58 today, 00:30:00, partition normal. It looks like your smoke test with the NAME= assignment lost — it'll write into .../msnr//harness/. Your call whether to cancel; I haven't touched it.

Minor, not fixed: commit 3ba6def's subject says "the GPUs the node actually has" but GPUS_PER_NODE=4 is still hardcoded (evaluate.sbatch:165) — correct on Clariden, just not what the sentence claims. And the flatten loop in _run_per_task.sh:117 is now unreachable (nothing writes a CLI-style top-level subdir any more).

7. Proposed commit message

evals: measure the worker pool in the same unit it is fitted in

eval_timing.py's batched fallback read a results file's `.results` keys as
the task count. lm_eval lists a group's subtasks individually there, so job
3199185's 100 queued tasks came back as 1,067 rows -- 0.08 min/task against
the 0.75 MIN_PER_TASK was fitted at. The two pipelines were being compared
in different units, and the tool would have reported its own subject as ~8x
slower; worse, its stated job is re-fitting MIN_PER_TASK, where the bias
undersizes every eval walltime. Subtracting `.group_subtasks` leaves the
top-level names the job was given.

Corrected, it reproduces the constants it is meant to re-fit -- 90M 0.77 vs
0.67, 175M 0.76 vs 0.62, 350M 0.84 vs 0.75, 600M 0.90 vs 0.85 -- and gives
1B and 1.7B their first measurement, 0.91 and 0.96 against guesses of 2.0
and 2.8.

The kill-survival note asserted "every kill kept 0" while the table above it
showed one batched job keeping 100: 3199185 finished its burst at 14:31 and
hit the wall during the W&B push. It now says what the data says.

eval_worker.py left an empty inflight/<task>/ behind for a task that raised
and was logged, so _run_per_task.sh counted it among the tasks "whose worker
died" -- the one signal that distinguishes a silent CUDA/OOM death from a
clean failure. Verified on a two-worker fixture: before, 2 reported for 1
real death; after, 1.

Docs: the deprecated local_hf runner still described BATCH_TASKS as a live
toggle _run_per_task.sh implements; the evals README listed a job.json
"unfinished" count evaluate.sbatch deliberately does not write.
compute-budget's worked examples predate the afri removal (1,965 -> 1,890,
615 -> 585) and its "2.5-3.7x" is really 1.0-3.8x, since L1/L2 gain nothing
against the 15-min overhead and rounding; the eval table still said "610
submittable" after the over-cap refusal was removed.

Verified: fixture run of _run_per_task.sh (2 workers, 5 tasks, one clean
failure, one simulated dead worker) merges 3 tasks and logs each failure
once; re-run skips the 3 finished and runs only the 2 missing; a top-up run
exits 0 with nothing to do. auto_evals_cscs.py --dry-run unchanged at 698
submissions with walltimes matching an independent recomputation of
eval_minutes; models.json restored after sync(). sync_docs() reports both
generated docs already in sync.

### Verifying the optimization

I added eval_timing.py because "better" needs numbers off disk, not assertions. It classifies each eval dir by pipeline generation, worker if it has the new job.json and batched otherwise, and reads wall-clock from sacct so both generations are measured identically.

1. Baseline, before running anything new. On the login node, this reports what the old pipeline cost and what its killed jobs kept:

cd src/evals && python3.11 scripts/eval_timing.py --detail

Today every row says batched, and the kill table should show every killed job kept zero tasks. That is the number the change has to beat.

(Output)

== minutes per task (wall-clock / tasks finished, sacct elapsed) ==
pipeline  size   jobs   tasks   median     p10     p90
batched   1.7B     24   18816     0.09    0.09    0.15
batched   175M    100   60968     0.09    0.07    0.12
batched   1B       20   26500     0.10    0.09    0.11
batched   350M     59   55253     0.09    0.08    0.13
batched   600M     79   57650     0.10    0.08    0.14
batched   90M      82   58796     0.09    0.07    0.11
Only COMPLETED jobs; a killed one's elapsed is its walltime, not its cost.

== what a killed job kept ==
pipeline   jobs  kept 0  kept >0  tasks saved
batched       6       5        1         1067
A batched job wrote everything at the end, so every kill kept 0 (evals CLAUDE.md bug 13).


2. Smoke test on a real checkpoint. Isolated by name suffix and W&B project, so it never mixes with sweep data:

cd src/evals

LM_EVAL_BACKEND=vllm TOKENIZER=swiss-ai/Apertus-70B-2509 BOS=true APPLY_CHAT_TEMPLATE=false TP=1 PP=1 EVAL_WORKERS=4 HARNESS_LIMIT=4 WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=msnr-workertest TASKS=hellaswag,hellaswag_ru,xnli_ru,xwinograd_ru sbatch --job-name=eval-lm-90M-L2-deep-seed1904-iter450-workertest --partition=debug --time=00:30:00 --export=ALL scripts/evaluate.sbatch /capstor/store/cscs/swissai/infra01/msnr-hf-models/lm-90M-L2-deep-seed1904/iter_0000450 lm-90M-L2-deep-seed1904-iter450-workertest

>> Submitted batch job 3312820

3. Confirm all four GPUs are busy. While it runs, this should show four python processes, one per GPU, where the old pipeline showed one:

srun --overlap --jobid=$(squeue --me -h -n eval-lm-90M-L2-deep-seed1904-iter450-workertest -o %i) nvidia-smi


+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GH200 120GB             Off |   00000009:01:00.0 Off |                    0 |
| N/A   23C    P0             92W /  900W |       4MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GH200 120GB             Off |   00000019:01:00.0 Off |                    0 |
| N/A   27C    P0             97W /  900W |       3MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GH200 120GB             Off |   00000029:01:00.0 Off |                    0 |
| N/A   25C    P0             42W /  900W |       5MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GH200 120GB             Off |   00000039:01:00.0 Off |                    0 |
| N/A   24C    P0             94W /  900W |       5MiB /  97871MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+


4. The recoverability proof, which is the main claim. Submit a task-heavy cell with a deliberately undersized walltime so Slurm kills it:

cd src/evals && NAME=lm-600M-L30-deep-seed1904-iter14400 LM_EVAL_BACKEND=vllm TOKENIZER=swiss-ai/Apertus-70B-2509 BOS=true APPLY_CHAT_TEMPLATE=false TP=1 PP=1 EVAL_WORKERS=4 WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=msnr TASKS="$(python3.11 -c "import sys;sys.path.insert(0,'../..');from src.evals.scripts.utils.configs import tasks_for_benchmarks;sys.path.insert(0,'../pretrain');from launch_trainings import cell_languages;import json;print(','.join(tasks_for_benchmarks(json.load(open('../../configs/tasks.json'))['groups']['auto'], cell_languages(30,'A'))))")" sbatch --job-name=eval-$NAME --time=00:20:00 --export=ALL scripts/evaluate.sbatch /capstor/store/cscs/swissai/infra01/msnr-hf-models/lm-600M-L30-deep-seed1904/iter_0014400 $NAME

(What I submitted)

LM_EVAL_BACKEND=vllm TOKENIZER=swiss-ai/Apertus-70B-2509 BOS=true APPLY_CHAT_TEMPLATE=false TP=1 PP=1 EVAL_WORKERS=4 WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=msnr TASKS="$(python3.11 -c "import sys;sys.path.insert(0,'../..');from src.evals.scripts.utils.configs import tasks_for_benchmarks;sys.path.insert(0,'../pretrain');from launch_trainings import cell_languages;import json;print(','.join(tasks_for_benchmarks(json.load(open('../../configs/tasks.json'))['groups']['auto'], cell_languages(30,'A'))))")" sbatch --job-name=eval-lm-600M-L30-deep-seed1904-iter14400 --time=00:20:00 --partition=debug --export=ALL scripts/evaluate.sbatch /capstor/store/cscs/swissai/infra01/msnr-hf-models/lm-600M-L30-deep-seed1904/iter_0014400 lm-600M-L30-deep-seed1904-iter14400

>> FileNotFoundError: [Errno 2] No such file or directory: '/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/configs/hf_wandb.json'
>> Submitted batch job 3312838

After it times out, count what survived. On the old pipeline this was always zero:

ls /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr/lm-600M-L30-deep-seed1904-iter14400/harness/eval_*/per_task | wc -l

>> 3

Then resubmit the same command. The job log's Skipping N task(s) with existing results line is the resume working, and job.json records tasks_skipped alongside tasks_done.

>> FileNotFoundError: [Errno 2] No such file or directory: '/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/configs/hf_wandb.json'
>>Submitted batch job 3312890
>> 5

5. The A/B number. Once a few worker jobs have completed, the same command from step 1 prints both generations side by side, with a speedup ratio per size:

cd src/evals && python3.11 scripts/eval_timing.py

Expect roughly a fourfold drop in minutes per task at fixed task count, since four workers replace one process. That is also how you re-fit MIN_PER_TASK in auto_evals_cscs.py, whose current figures were measured on the old single-process pipeline and are now per worker. Until that re-fit lands, the walltime requests are an upper bound.

(Output)

y
== minutes per task (wall-clock / tasks finished, sacct elapsed) ==
pipeline  size   jobs   tasks   median     p10     p90
batched   1.7B     24    1864     0.96    0.84    1.15
batched   175M    100    7050     0.76    0.63    0.92
batched   1B       20    2790     0.91    0.84    1.02
batched   350M     59    5600     0.84    0.79    1.01
batched   600M     79    5871     0.90    0.82    1.10
batched   90M      82    6006     0.77    0.67    0.89
Only COMPLETED jobs; a killed one's elapsed is its walltime, not its cost.

== what a killed job kept ==
pipeline   jobs  kept 0  kept >0  tasks saved
batched       6       5        1          100

### What still needs the cluster

- The per-task constants were fitted on the old pipeline. Treat the budget totals as an upper bound until the timing tool re-fits them from real jobs.

- Prefix caching is unverified on your container. I ported it from include-private. The V1 engine normally enables it anyway, so it should be a no-op, but I did not confirm that against the swiss-ai fork build. Since the sweep compares scores across time, re-run one already-evaluated checkpoint and diff before relying on it. The comment in the job script says so and says to drop the flag if scores move.


Re-eval a checkpoint that already has results
A different WANDB_PROJECT is a different eval_logs subtree, so the per-task gate finds nothing and runs everything — while NAME stays real so the W&B push can still resolve it (that's what broke 3312820).


cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals
NAME=lm-175M-L50-deep-seed1904-iter3416
TASKS="$(python3.11 -c "import sys;sys.path.insert(0,'../..');from src.evals.scripts.utils.configs import tasks_for_benchmarks;sys.path.insert(0,'../pretrain');from launch_trainings import cell_languages;import json;print(','.join(tasks_for_benchmarks(json.load(open('../../configs/tasks.json'))['groups']['auto'], cell_languages(50,'A'))))")"
echo "$TASKS" | tr ',' '\n' | grep -c .     # must print 329 before you submit

LM_EVAL_BACKEND=vllm TOKENIZER=swiss-ai/Apertus-70B-2509 BOS=true APPLY_CHAT_TEMPLATE=false \
TP=1 PP=1 EVAL_WORKERS=4 \
WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=msnr-prefixcache TASKS="$TASKS" \
sbatch --job-name=eval-$NAME --account=infra01 --time=01:00:00 --export=ALL \
  scripts/evaluate.sbatch \
  /capstor/store/cscs/swissai/infra01/msnr-hf-models/lm-175M-L50-deep-seed1904/iter_0003416 $NAME
Then diff — I added scripts/diff_scores.py for it (README updated), smoke-tested against msnr vs itself: 2716/2716 identical, max |Δ| 0:

python3.11 scripts/diff_scores.py lm-175M-L50-deep-seed1904-iter3416 msnr msnr-prefixcache

Any nonzero delta means enable_prefix_caching=True moved scores and the flag should come out.

