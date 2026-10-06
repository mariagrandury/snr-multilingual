# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel
scontrol update JobId=3593053 Partition=normal

## Pretraining

python3.11 pretrain/launch_trainings.py cscs --scheme FWEB --partition preemptable
python3.11 pretrain/launch_trainings.py cscs --size 3B --partition preemptable
python3.11 pretrain/launch_trainings.py cscs --activation swiglu --partition preemptable



## Convert and eval new ckpts

cd Projects/snr-multilingual/src/ && conda activate snr

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1800
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --retry-held

✅ probe new benchmarks:
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --group auto_probe --size 90M,175M,350M,600M,1B,1.7B --seed 1904 --final-only

✅ compare probe benchmarks and decide which to keep:
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise/analysis/rq00_task_reformulation/probe.sh

## Update analysis with new evals

✅ on the cluster, in the snr env: rebuild and publish the report:
python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git

✅ mirror eval logs to capstor:
sbatch evals/scripts/mirror_eval_logs.sbatch

✅ fetch to update local cache:
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual \
 && git fetch origin data/ladder-report \
 && git archive origin/data/ladder-report | tar -x -C src/signal-and-noise/data/ladder-report

✅ fetch new data to cache and rebuild every derived artefact (inc. documents):
✅ a) except the curves (FORCE=1 to force fetch, if the refresh was interrupted haflway just run without FORCE), <2h:
FORCE=1 bash scripts/refresh_analysis.sh
✅ b) with the curves (needs slurm):
sbatch --account=infra01 --partition=normal --nodes=1 --time=09:00:00 \
 --job-name=snr-analysis \
 --output=/iopsstor/scratch/cscs/mariagrandury/snr-analysis-%j.log \
 --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && FORCE=1 PY=python HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 bash scripts/refresh_analysis.sh --curves'

✅ fetch new data to cache and rebuild every derived artefact (only analysis/ figures):
FORCE=1 bash run_all_predictivity.sh

# Scheduled update

loginctl enable-linger

systemctl --user enable --now ladder-nightly.timer

> > Created symlink /users/mariagrandury/.config/systemd/user/timers.target.wants/ladder-nightly.timer → /users/mariagrandury/.config/systemd/user/ladder-nightly.timer.

systemctl --user list-timers ladder-nightly

> > NEXT LEFT LAST PASSED UNIT ACTIVATES  
> > Wed 2026-09-23 06:00:00 CEST 3h 41min - - ladder-nightly.timer ladder-nightly.service
> > 1 timers listed.
> > Pass --all to see loaded but inactive timers, too.

To see how it went:

cat /iopsstor/scratch/cscs/mariagrandury/logs/nightly-ladder/last-run.txt
systemctl --user status ladder-nightly.service # exit status of the last run

# Build L1 & L2

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
BUILD_PARTITION=preemptable ./launch_builds.sh # adds build-en-dclmp + build-en-fweb

monitor build:
bash /iopsstor/scratch/cscs/mariagrandury/buildwatch.sh

- FineWeb -> 25h -> Wed 21h

# 3B ETA

T=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training
for j in $(squeue --me -h -o "%i|%j" | grep "pretrain-3B" | cut -d'|' -f1); do
  f=$(ls -t $T/*-${j}.out 2>/dev/null | head -1); [ -z "$f" ] && continue
l=$(grep -h "iteration " "$f" 2>/dev/null | tail -1)
printf "%-40s %s | eta %s\n" "$(basename ${f%-$j.out})" \
 "$(sed -E 's/.*iteration +([0-9]+)\/ *([0-9]+).*/\1\/\2/'<<<"$l")" \
 "$(sed -E 's/.*eta: ([^|]+)\|.*/\1/'<<<"$l" | tr -s ' ')"
done

Friday midday

# Evals

Update tasks list and reeval:

- Add language-specific tasks
  - Check the results from La Leaderboard, see which benchmarks give signal at small scales for ES, CA, GL, EU
- 🛑 (Switch or drop LAMBADA-MT -> what was this about?
- 🛑 Reeval after worker implementation -> what was this about? are the current values above or below before implementing the eval workers
- mmlu, openbookqa, commonsense_qa, blend, cultural_bench, truthfulqa_mc2? triviaqa, squadv2, agieval, bbh, toxigen, multi-if, mbpp_instruct, mathqa, ifeval, humaneval_instruct, hendrycks_math, drop, bbq, acp_bench

# Task reformulation

## Programatically

- ✅ probe eval reformulations

## Gemini

- [BLOCKED] admin access to create bucket

✅ launch bucket reformulation:

(snr) mariagrandury@clariden-ln004:/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual> gcloud storage buckets create gs://silin-482809-msnr-rfgm \

>     --project=silin-482809 --location=US --uniform-bucket-level-access

Creating gs://silin-482809-msnr-rfgm/...
ERROR: (gcloud.storage.buckets.create) HTTPError 403: maria.grandury@epfl.ch does not have storage.buckets.create access to the Google Cloud project. Permission 'storage.buckets.create' denied on resource '//storage.googleapis.com/projects/\_/buckets/silin-482809-msnr-rfgm' (or it may not exist). This command is authenticated as maria.grandury@epfl.ch which is the active account specified by the [core/account] property.

✅ launch online reformulation:

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual

nohup env GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=silin-482809 GOOGLE_CLOUD_LOCATION=global \
 /users/mariagrandury/miniconda3/envs/snr/bin/python3.11 \
 src/evals/scripts/rewrite_items_gemini.py online \
 --family belebele --L 50 --retry-rejects \

> > /iopsstor/scratch/cscs/mariagrandury/rfgm_online.log 2>&1 &

python3.11 src/evals/scripts/make_rf_tasks.py --set rfgm --family belebele # registers 59 tasks + YAMLs

# Backlog

- what do we do with the seeds? -> error bars for DA
- notes from 09-16
- legacy June sweep
- external models
- varify EN versions in L1
- verify RU/ZH/ES in L2


---

# Review

run_tokens counted 90M/175M runs at batch 504 (the reviewer fixed this in your commit f4c81050). The reviewer described it as a reporting error, but it also chooses each cell's FineWeb-2 file: the 52B stage copy, or the 92B rebuild for cells the 52B copy is too small for. With the inflated 55.7B / 52.9B estimates, the 90M/175M cells at A-L15, A-L50 and B-L15 were probably sent to the 92B file. Their 350M–1B rungs read the 52B one.

This is the one finding that touches the trained grid. Those 90M/175M cells would have seen a different sample order and the 92B copy's newer crawls, unlike the 350M–1B rungs of the same settings. The 1.7B rung reads the 92B copy too, by design.
The fix retrains nothing: finished cells are skipped. It is correct for any future launch. pretrain_progress will now plan the 52B file for those cells, so its report will flag any that actually read the 92B copy.
Needed: keep the fix, and check on the cluster which file those cells read. A suggested read-only command; I haven't run it, and the log path is from CLAUDE.md:

grep -lE "data-92B" /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/pretrain-{90M,175M}-*-{L15,L50}-* 2>/dev/null

If they did read it, the honest fix is to state it in rule 10 and the paper. The alternative is a retrain of those cells, which is your decision.


ALSO


The batch confound at 90M/175M: LR not rescaled when the batch changed, together with the data-file question in #2. Both belong in rule 10 and the paper.


The batch confound at 90M/175M: LR not rescaled when the batch changed, together with the data-file question in #2. Both belong in rule 10 and the paper.
Recompute rf_significance.csv on the cluster.
Move the 15.6 MB ladder_report_bpb.png to LFS.


## Answer

Both answered. Nothing changed, nothing launched.

# 1. The 92B data routing — not an issue

Two independent lines of evidence, both negative.

**Empirically:** zero of the ~1,400 `-b84`/`-b168` training logs mention `data-92B` or `data-165B`. Every one that names a data root names `/iopsstor/.../data/` — the 52B stage. The only cells that ever read the 92B copy are the six 1.7B ones (A-L15, B-L15, A-L50, deep and shallow), exactly by design.

**Mechanically, the inflated numbers never crossed the threshold.** The reviewer's 55.7B / 52.9B are right as `run_tokens`, but `fineweb_source` passes that to `undersized_build`, which halves it internally (`draw = run_tokens * (100 − EN_SHARE) // 100`) because only half the blend is FineWeb-2:

| cell | run_tokens (buggy) | FineWeb-2 draw | build has | routed |
|---|---:|---:|---:|---|
| 90M-L15/L50-A, L15-B | 55.7B | **27.9B** | 52.0B | 52B stage |
| 175M-L15/L50-A, L15-B | 52.9B | **26.4B** | 52.0B | 52B stage |

`draw ≤ have` → `undersized_build` returns `None` → no rebuild lookup. I re-ran `fineweb_source` with both the fixed and the inflated token count for all six cells: the routing is identical. The bug was a reporting error only, exactly as the reviewer first called it. Keep the fix — it's correct for any future launch — but no trained cell is affected.

(My first attempt at this got the wrong answer because I pre-halved the token count before calling `undersized_build`, which halves it again. The table above is from the corrected run.)

# 2. The LR was not rescaled — confirmed, and deliberate

`scale_for_gbs` rescales `train_iters`, `lr_warmup_iters` and `lr_wsd_decay_iters` only. The peak LR is untouched, and the logs confirm it: `90M-L8-b84` ran at **0.0014276**, identical to its diverged batch-504 predecessor; `175M-L8-b168` at **0.00121655**, likewise.

**It's a documented choice, not an oversight.** `plan/90M-175M-batch-retrain.md:72-73` says "LR 0.0014276 *unchanged*", and `plan/90M-rung-anomaly.md:693` says "the LR is not rescaled for the smaller batch". More importantly, §"What it is not: the learning rate" rules the LR out as the cause: at batch 504 the 90M rung diverged at **3e-4, 6e-4 and 1.4276e-3** alike — "LR changes how badly it degrades, not whether it does". Holding the LR is what makes the retrain a single-variable change against the runs it replaces.

## Does it show up?

`ladder_report.py --check scaling --tol 0.15` → **nothing flagged**. Residuals aggregated over all 26 families:

| rung | mean residual | families below the fit |
|---|---:|---|
| 90M | −0.045 | 24/26 *(out-of-sample — the fit starts at 175M)* |
| **175M** | −0.023 | **26/26** |
| **350M** | +0.038 | **0/26** |
| 600M | −0.013 | 21/26 |
| 1B | +0.003 | 11/26 |
| 1.7B | −0.005 | 16/26 |

26/26 then 0/26 is a clean sign flip, and it lands exactly at the batch boundary (175M = 168, 350M = 504). But it is also precisely what a straight line fitted to a convex log-log curve does — ends below, middle above — and 1.7B being below on 16/26 fits that reading, not the batch one. The magnitudes are small against the 0.15 tolerance.

## So: is it an issue?

**Not for the headline results.** DA is a *ranking within a rung*. Every family at 90M shares the same batch and the same LR, so a rung-wide offset cancels exactly in the comparison. rq02 — DA-size, DA-ckpt, the reliable-task lists — is safe by construction, and that is where the paper's claims live.

**Marginally for rq01's scaling fits**, where 90M/175M are points on the curve. The fit already starts at 175M, the residuals are inside tolerance, and the check passes.

**The honest limitation** is that batch and rung are perfectly confounded: every 90M cell is at 84, every 350M+ cell at 504, and the only 90M/175M runs at 504 diverged. You cannot estimate the batch effect at those rungs from data that exists.

## How to close it, if you want to

One diagnostic settles it — **350M at batch 168, LR unchanged**, the smallest rung that currently uses 504 and so the cheapest place you can have both. If its loss lands on the batch-504 350M, the batch/LR change is proven harmless at a rung where both exist, and the 90M/175M rungs inherit that. The launcher already supports it and forces a `diag-` name, so it cannot occupy a grid cell:

```bash
python3.11 launch_trainings.py cscs --size 350M --langs 8 --gbs 168 --dry-run
# diag-350M-L8-deep-seed1904-gbs168-tok34.39B, 49,980 iters, 14 nodes, ~45 node-h
```

Otherwise the minimal action is documentation: state in the paper that the two smallest rungs train at batch 84/168 with the LR held, and that the rank-based results are unaffected because the offset is rung-wide. I'd do that regardless of whether you run the diagnostic.

## Dive deep

can i say that the 90m and 175m models are from the same ladder except the batch size? short answer

Yes — with one caveat worth a half-sentence.

**Accurate:** batch is the only *primitive* that changed. Everything else that differs is derived from it to hold the token budget D = 100·N and the same schedule *fractions*: iterations ×6 / ×3, warmup and WSD decay scaled with them, save interval and `ADEMAMIX_WARMUP` likewise. Peak LR, architecture, init, data, tokenizer and seed are identical to the rest of the ladder — I checked the logs: 90M ran at 0.0014276 and 175M at 0.00121655, the same values their batch-504 predecessors used and the same 6ND law every other rung uses.

**The caveat:** don't let "only the batch size" imply "and therefore nothing else changed". At a fixed LR a 6× smaller batch *is* a different optimization regime — 6× more updates at the same step size — and that is precisely why these rungs converge where batch-504 diverged. The LR was deliberately not rescaled (`plan/90M-rung-anomaly.md:693`).

So I'd write it as: *"the 90M and 175M rungs train at a reduced global batch (84 and 168 against 504), with the token budget and every schedule fraction held; the peak learning rate is unchanged."* That is defensible as written. "Identical except the batch size" is also true but invites the reader to assume the difference is cosmetic.

One thing it's worth knowing supports you: the batch change moved those rungs' AdEMAMix slow-EMA memory from 10,000/4,500 ≈ 2.2 runs (longer than the run — the bug) to ≈ 0.37 and 0.39, which is the same range as 350M (0.60) and 600M (0.35). On that axis the retrained rungs are *more* like the rest of the ladder than before, not less.


# New nightly

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
# 1. every artifact against the 10-05 report (includes by_L ×3, by_language, language_tier)
sbatch --parsable --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
  --job-name=ladder-refresh \
  --output=/iopsstor/scratch/cscs/mariagrandury/logs/nightly/2026-10-05.refresh-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && FORCE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 bash scripts/refresh_analysis.sh --no-fetch --no-deck'
# 2. the rq00 curve grids, afterany (not afterok) on job 1
sbatch --account=infra01 --partition=normal --nodes=1 --time=04:00:00 \
  --job-name=ladder-curves --dependency=afterany:<job1> \
  --output=/iopsstor/scratch/cscs/mariagrandury/logs/nightly/2026-10-05.curves-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise && HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 python3 analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity'

# To fix

- rq06 language_panel.py:107 keeps one task per language where a family has several


# 6 Oct: Agents Stopped

Both jobs are queued and on preemptable as you asked:

| Job | Partition | Estimated start |
|---|---|---|
| Paper job (3593053) | preemptable | 12:30 (was 12:03 on normal) |
| Per-item store (3593047) | preemptable | 12:34 |

Slurm's estimate got 30 min later after the move. To go back: `scontrol update JobId=3593053 Partition=normal`.

**Already committed, not pushed:**
- `78b00a2f`, the merge of your bBPB/rq11 commits plus the fix for the nightly crash.

**Reviewed and ready to commit:**
- **rq12 above-chance items** (worktree `wt/above-chance`) and **rq13 English-only** (worktree `wt/english-only`): built, reviewed twice with verification, and every confirmed finding fixed.
  - Commit them only after the running jobs finish (~13:30), because they change `run_all_predictivity.sh` and shared modules those jobs read.
- **Pools + scheme axis** (worktree `wt/pools-scheme-axis`): reviewed and fixed, but built on the code before the merge. It needs a port (12 known items) before it can be committed; that's the refactor you postponed.

**Not reviewed yet, agents still running:**
- rq11 with both truths, plus the 15 mechanical bBPB fixes (worktree `wt-fixes`).
- The prose audit for the gate and scaling-predictability READMEs. The prose itself is already edited in the main tree.
- `plan/optimize-refresh.md`.

These three agents use your credits. Tell me if you want any of them stopped.

**Next 12 h:**

| When | What | Who |
|---|---|---|
| ~11:30 | Gate and scaling-predictability README prose audited and final | agent |
| ~12:30–13:45 | Paper job runs decision accuracy (rest), noise/SNR, surrogates and the remaining analyses on the main pool, then the paper figures, tables and facts. Log: `/iopsstor/scratch/cscs/mariagrandury/logs/paperjob/paper-3593053.log` | Slurm |
| 13:18 | Nightly refresh times out; the curves job then runs, untouched | Slurm |
| ~13:45 | Prose for the remaining READMEs from the paper job's outputs, then sync the figures and numbers into your Overleaf source once it's in the repo | me / agents |
| ~14:00 | Review gate, then commit today's outputs, paper figures, rq12 and rq13 | me |
| ~12:30–16:30 | Store build | Slurm |
| after the store | Rerun rq12 at checkpoint level and the full rq11 | me |
| afternoon | rq11 fixes reviewed and committed; optimisation plan delivered | me |
| tonight 00:00 / 04:00 | Nightly runs again | Slurm |

**Risk for tonight:** without the compute_da speedup, tonight's 04:00 refresh will time out again. The quick fix is to raise the refresh's `--time` to 12 h in `scripts/nightly.sh`, or to land the top item from the optimisation plan before 04:00. I'll prepare whichever you prefer.

If my credits stop:
- the Slurm jobs finish on their own;
- the paper figures land in `documents/paper/figures/`;
- `facts.py` prints the numbers that moved at the end of the paper job's log.

---

The README audit agent has stopped. It made no further README edits. Its progress note is at `plan/todos/progress-2026-10-06-readme-audit.md`.

**Scaling predictability:** every earlier audit problem was already fixed this morning, and it re-checked those fixes against today's CSVs.

- **Re-checked and correct:** the first three headline bullets, all of figures 1, 2 and 4, and the bBPB wording ("better than accuracy, below BPB and the loss").
- **Not yet checked:** the 122 / 111 / 5 split of the 127 three-rung series, the 9 fits without 1.7B, every figure-3 number, the α values, and the setup's cell counts.
- **Small issues, not fixed:**
  - "1179 gated fits" counts 22 LAMBADA fits, which are never gated.
  - Some older figure-3 text still names analyses by number.
  - One paper follow-up bullet runs to three sentences.
- **Old data still flagged:** figure 5 keeps its "not refreshed today" note, because its decision-accuracy table is still the 02:00 one.

**Gate:** the agent read it but did not check its numbers against today's CSVs, so that audit is still to do. It also found several paragraphs over the two-sentence limit. The two inputs from 02:00 (the above-random example and figure 3's decision-accuracy columns) are already labelled in the prose as older tables.

Still waiting on the other two agents' progress files.

---

The rq11 / bBPB-fixes agent has also stopped. Its progress note is in the `wt-fixes` worktree: `PROGRESS-2026-10-06.md`.

- **rq11 code is mostly done but unverified:**
  - Both bBPB readings, bBPB → 1.7B accuracy and bBPB → 1.7B bBPB, are coded, and every table, figure and recommendation is labelled with its target.
  - Twin-to-original pairing and the language and task counts behind each ranking are coded.
  - Recommendation rows resting on fewer than 5 tasks now print "n < 5", with a table splitting how each pick was made.
  - None of this has been run or checked.
  - The rq11 CSVs and README on disk are older than the last code edit, so none of their numbers can be trusted yet.
- **Still open in rq11:**
  - Check the bBPB → 1.7B accuracy values against `bench_bpb_da.py`.
  - Regenerate the outputs.
  - Rewrite the README key findings for both targets.
  - Run `check_rules` and the tests.
- **The 15 mechanical bBPB fixes:** not started.

Waiting on the last agent's draft of `plan/optimize-refresh.md`.

---

All three agents have stopped and saved their progress:

| Work | Progress file |
|---|---|
| README audit (gate, scaling predictability) | `plan/todos/progress-2026-10-06-readme-audit.md` |
| rq11 two truths and the bBPB fixes | `snr-multilingual-wt-fixes/PROGRESS-2026-10-06.md` |
| Refresh optimisation | `plan/optimize-refresh.md`, marked as an interrupted draft with a "Not done yet" list |

The rq11 outputs in that worktree predate the last code edit, so rerun `recipe.py` before trusting any rq11 number.

What the optimisation draft measured:
- **The refresh uses 1 of the 288 CPUs it's given.** compute_da is 71% of the run and redoes the same checkpoint selection for every DA definition, pair set and pool, through ~130 pandas calls per task.
- **The data loader costs ~40 s per call, with no cache.** Across ~80 steps that's an estimated 50–60 min of reloading.
- **Proposed fix:** compute one score table per task, shared across all pools; cache the loaded pools; parallelise across the node. Target: under 45 min, from 216 min now. Those savings are estimates; the profiling and benchmarks weren't run.
- **Eight steps already failed on 10-04:** `by_L.py` (3 runs), `agreement.py`, `scaling_vs_ranking.py`, `public_ladders.py`, `finetasks_criteria.py`, `per_item_ladder.py`. Check the paper job's log for those in case it uses any of their figures.

Nothing else is running on my side. The paper job (3593053) and the store build (3593047) are still queued on preemptable and will run without me.
