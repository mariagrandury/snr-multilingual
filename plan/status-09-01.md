# Status

Where each stage stands against all 56 planned cells
stage	     done	         planned    %
Pretraining	 24 cells	     56	        43%
Eval	     229 ckpt-evals	 560	    41%
BPB	         150 ckpts	     1,120	    13%

That headline is misleading, though, because the compute is not evenly distributed. Split it at 600M:

cells                               done	  node-hours left
≤600M ladder (90M/175M/350M/600M)	24 / 36	  ~600
1B + 1.7B	                         0 / 20	  ~6,600

92% of the sweep's remaining compute is the two rungs that haven't started. Everything trained so far is ≤600M.

# Time predictions

Measured burn rate over the last 8 days: ~215 node-hours/day, very bursty (758 on Aug 27, 3 on Aug 29).

≤600M ladder — ~600 node-hours left: 12 cells to train (107 nh, mostly cheap 175M), ~120 new eval jobs (~300 nh), 570 BPB checkpoints (~103 nh). That's ~3 days of compute, realistically ~1 week wall-clock with queue waits. Call it ~Sep 8.

1B + 1.7B — ~6,600 node-hours: at the CSCS rate that's ~31 days (early October). The plan puts these on Azure (16× ND96isr Spot, UK) where the fleet math is ~4 days per level — but I can't verify Azure state from the login node (az isn't installed there), and nothing has started. plan/compute-budget.md schedules training to finish Aug 31 with a hard stop Sep 4; that date has passed with the entire big-rung block unstarted, so the schedule needs re-planning regardless of which cluster runs it.

# Blockers

The real blocker isn't compute
Two things gate the analysis no matter how many node-hours land:

All 24 trained cells are seed1904. All 16 replicate cells (8× seed28, 8× seed1797) are unstarted. SNR = signal / noise, and the noise half needs either seed replicates or the late-checkpoint variance. You can compute checkpoint noise today, but not seed noise — on any cell. That is a harder blocker than the 1B rung, and the replicates are cheap (they're at 175M and 1B; the 175M ones are ~3 nh each).

The 90M rung still diverges (item 15) — 9 of 10 runs peak at 14–19% of training then degrade, and BPB rises with training. Until the β₃ fix is confirmed, 90M is not usable as a ladder point, which turns your 4-size ladder into a 3-size ladder.

# Recommended order

1. Let the 40 running 600M evals land (today, no action). Closes benchmarks for 4 sizes × 6 language settings.
2. BPB on the remaining trained cells — ~330 checkpoints, ~66 node-hours, 1–2 days. Cheapest item on the board and it's the plan's stated outcome metric. Unlocks the loss/BPB scaling fits immediately.
3. The 90M β₃ confirming run (~2 node-hours). Tiny, and it decides whether the ladder has 3 or 4 rungs — which changes what you can claim. Do this before writing any scaling section.
4. The 8× 175M seed replicates (~26 node-hours). This is what makes SNR computable rather than just signal. Highest analysis-value per node-hour in the whole sweep.
5. The 12 remaining ≤600M cells (~107 nh) → convert → eval → BPB (~400 nh).
6. Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 7 language settings.
7. 1B/1.7B in parallel on Azure, positioned as the extrapolation check the paper validates against — not as a prerequisite for the analysis.

Steps 2–4 total under 100 node-hours and unblock essentially the whole analysis. I'd do them before anything else in the queue.

# Commands

## Step 5

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && timeout 900 python3.11 src/pretrain/pretrain_progress.py 2>&1 | awk '$2=="fresh" && $1 !~ /-1B-|-1\.7B-/ {print}'

lm-175M-L1-deep-seed28	    fresh	8540
lm-175M-L1-deep-seed1797	fresh	8540
lm-175M-L2-deep-seed28	    fresh	8540
lm-175M-L2-deep-seed1797	fresh	8540
lm-175M-L30-deep-seed28	    fresh	8540
lm-175M-L30-deep-seed1797	fresh	8540

lm-90M-L100-deep-seed1904	fresh	4500
lm-175M-L100-deep-seed28	fresh	8540
lm-175M-L100-deep-seed1797	fresh	8540
lm-175M-L100-deep-seed1904	fresh	8540
lm-350M-L100-deep-seed1904	fresh	16660
lm-600M-L100-deep-seed1904	fresh	28800

## Step 2 — BPB on the trained cells that still lack it

Derives the list at run time rather than hardcoding it, and skips cells already in flight (the chain means a cell can have a successor queued):

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals

STAGING=/capstor/store/cscs/swissai/infra01/msnr-hf-models
OUT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr

for c in $(ls "$STAGING" | grep '^lm-'); do
  squeue --me -h -n "bpb-$c" -o %i | grep -q . && { echo "skip  $c (in flight)"; continue; }
  due=0
  for it in $(find "$STAGING/$c" -name .hf_complete -printf '%h\n' 2>/dev/null \
              | xargs -r -n1 basename | sed 's/^iter_0*//'); do
    [ -s "$OUT/$c-iter$it/bpb/bpb.json" ] || due=$((due+1))
  done
  [ "$due" -eq 0 ] && { echo "skip  $c (complete)"; continue; }
  echo "SUBMIT $c ($due due)"
  # sbatch --job-name="bpb-$c" scripts/score_bpb.sbatch "$c"
done
Run it as-is first — it prints the plan with sbatch commented out. Uncomment that line to submit. Then keep the drainer alive, since it exits once nothing is movable:


nohup bash scripts/debug_drain.sh --interval 45 \
      > logs/debug_drain_$(date +%Y%m%d_%H%M%S).log 2>&1 &


Implemented a new script to launch with one command:

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals
bash scripts/launch_bpb.sh --dry-run    # then drop --dry-run


## Step 3 — the 90M β₃ confirming run: no command yet

There is no flag for this. launch_trainings.py:305-308 sets ADEMAMIX_WARMUP to the cell's own iters and there is no --beta3 argument — the fix from plan/90M-rung-anomaly.md (option A, tie β₃ to run length) is staged in the doc but not implemented in the code. So this step needs a one-line change to the env block first, then:

python3.11 src/pretrain/launch_trainings.py cscs \
    --size 90M --langs 2 --seed 1904 --training-steps 4500 --dry-run

...run under a diag- name so NAME_RE/sync_models_json don't absorb it into the grid. I'd rather implement that properly than hand you a command with an invented flag — say the word and I'll write it.

## Step 4 — the 8× 175M seed replicates

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
python3.11 src/pretrain/launch_trainings.py cscs --size 175M --seed 28   --dry-run
python3.11 src/pretrain/launch_trainings.py cscs --size 175M --seed 1797 --dry-run
Expect 4 cells each — L1, L2, L30, L100. Drop --dry-run to submit.




## Step 5 — the remaining 4 cells (all L100, seed1904)

Steps 4 + 5 together are the 12. After step 4, what's left is:

lm-90M-L100-deep-seed1904   lm-350M-L100-deep-seed1904
lm-175M-L100-deep-seed1904  lm-600M-L100-deep-seed1904

for s in 90M 175M 350M 600M; do
  python3.11 src/pretrain/launch_trainings.py cscs \
      --size $s --langs 100 --seed 1904 --dry-run
done
One size per call deliberately: --langs 100 --seed 1904 alone would also match the 1B and 1.7B L100 cells you don't want started yet.

Convert → eval happens on its own
launch_trainings.py starts the auto-eval watcher unless you pass --no-auto-evals, so conversion and evaluation follow the new checkpoints automatically. To drive it explicitly for cells already trained:

python3.11 src/pretrain/auto_evals_cscs.py --dry-run     # count the submit: lines
python3.11 src/pretrain/auto_evals_cscs.py --watch 900
Then re-run the step-2 loop once conversion has landed, to pick up BPB for the new cells.

Three things to know before you run these:

auto_evals_cscs.py --dry-run calls sync_models_json.sync(), which writes configs/models.json — and that file currently has 112 uncommitted lines from the pending review. Back it up and check git status configs/ afterwards, or commit the pending change first.

The uncommitted auto_evals_cscs.py over-cap guard will silently skip 600M-L100 (its 290-task eval needs ~711–1233 min against the 719-minute queue cap). So that cell will train and convert but not evaluate until eval splitting exists — item 5. Same for the 1B/1.7B L100 cells later.

600M-L100 is also the single most expensive item here: 21 nodes × ~4.4 h ≈ 53 node-hours, half of step 5's training cost on its own.

---

# Real updated plan

Today:
1. ✅ Let the 40 running 600M evals land (today, no action). Closes benchmarks for 4 sizes × 6 language settings.
1. ✅ (Deps on #1) Launched job to mirror eval logs to capstor.
1. ✅ Launched 3 jobs to finish pretraining the 175M scheme B models.
1. ✅ Launched 2 jobs to finish BPB calculation of L2 and L50 models.
3. ✅ The 90M β₃ confirming run (~2 node-hours). Tiny, and it decides whether the ladder has 3 or 4 rungs — which changes what you can claim. Do this before writing any scaling section.
5. ✅ BPB on the remaining trained cells — ~330 checkpoints, ~66 node-hours, 1–2 days. Cheapest item on the board and it's the plan's stated outcome metric. Unlocks the loss/BPB scaling fits immediately.
7. ✅ The 8× seed replicates: sizes 175M,600M x langs 1,50 x seeds 64,313

ToDo:
2. [Discuss] Update the list of available high-quality benchmarks for low resource languages.
2. 🛑 Launch the creation of the L100 data mixture with the new list of languages (1 day). Blocks pretraining of all L100 models. Blocked by update of available benchmarks for low resource languages.
4. Update the SNR module to new naming. Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 6 language settings.
6. [Discuss] Update the model grid plan so each size-languages cell has at least 3 models (we need 3 to calculate DA).
9. 🛑 (Deps on #2) The 4 small sizes of L100 models (classic: deep, A).
5. ✅ BPB on the remaining trained cells — ~330 checkpoints, ~66 node-hours, 1–2 days. Cheapest item on the board and it's the plan's stated outcome metric. Unlocks the loss/BPB scaling fits immediately.
10. 🛑 Further writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × all 7 language settings.
11. 🛑 (Deps on Azure capacity) 1B/1.7B in parallel on Azure.
11. 🛑 (Deps on #6) Selected 1B models in parallel on CSCS. Fit pretraining and eval estimates.

Ideas:
- schedule that periodically checks the squeue, reviews the logs of previous runs, and submits new jobs following a priority list, sends a message on failure or when decision is required
- evals idempotent and write eval results -> review how I implemented it for INCLUDE v2
- wire recover_results_from_samples.py. Verified: the only reference is its own docstring. Under BATCH_TASKS=1 a walltime kill writes nothing, and this is the tool that rescues those samples. Your own todo.md raised it; still open.
- evals: split over-cap eval jobs across Slurm jobs (NUM_SPLITS/SPLIT_INDEX in evaluate.sbatch + aggregate_splits.sbatch) and teach auto_evals_cscs.py to submit/dedupe the parts. Until then the watcher SKIPs 600M/1B/1.7B at L100 and 1.7B at L50 (need 711–1233 min vs the 719 cap) — see the SKIP lines in its output.
- HF model push is manual on CSCS: push-snr.py is invoked only from azure/jobs/push.yml. Nothing on the cluster side pushes converted checkpoints to the Hub — so it happens only when you remember. Your todo.md asked for somewhere to wire it that doesn't interfere.
- build_hf_dataset.py doesn't understand lm-* cells: Hence the project_legacy change now sitting uncommitted. Predictivity results can't be published to the HF dataset until the name regex, iter grid and project are taught the new sweep.
- ladder_report.check_bpb() does an unguarded json.loads while bpb_results() guards the same files — one bad file kills the whole report.
- score_bpb materialises 17 GB of fp32 logits per batch, capping batch size and making off-GPU runs impossible.
- azure/jobs/pretrain.yml's inputs.fineweb default still points at english_dclm — a hand-submitted L=1 job still pulls 686 GB.
- mirror_eval_logs.sbatch's #SBATCH --output dir is created by the script itself, after Slurm has already opened the file.
- compute-budget.md:52 still says L ∈ {1, 30, 100} (now L50), and the Azure tables still need the 56-run re-derivation — both under the existing staleness banner.
- periodically re-estimate with real values the eval and training job times for each language-size pair (and add them to the relevant docs, readme and plans) -> only update the estimates for which we have real significant results
- if there are no <2h jobs, the debug drainer could divide a >2h eval job into 2 debug jobs -> your answer to this does not correlate with you saying that the results are written per task are persistent i f a job times out, double check what really happens. Also, is the debug limit 1h30 or 2h?
- check whether it would be possible to use just one eval job to evaluate all the remaining eval ckpts of a model (i.e. instead of having 10 short eval jobs for 10 ckpts, queue just 1 eval job to eval the 10 ckpts). besides the feasability, check whether it would make sense taking into account the eval time

# Results report

## 1. Freshness: local artifacts are current; the capstor/HF copies of the ladder report are stale

Artifact	Location	State
Eval scores (eval_logs/.../msnr)	iopsstor	Up to date and live — 363 checkpoint dirs, 87 new results since Aug 30; the auto_evals_cscs.py --watch 1800 watcher is running, and right now there are 1 eval job (90M-L50 iter3600) and BPB jobs for 600M-L2/L50 in flight
Ladder report (csv/md/pngs)	iopsstor, src/pretrain/	Regenerated today 14:10 — current, though the 600M BPB runs finishing today will make it slightly behind again
Pretrain progress plots	iopsstor, src/pretrain/	Regenerated today 16:07 — current
Eval logs on capstor	/capstor/.../msnr-eval-logs	Fine — last full mirror Aug 30, and a fresh mirror-eval-logs job (3252060) is mirroring the 380 GB tree right now
Ladder report on capstor	/capstor/.../msnr-ladder-report	Stale: Aug 29 (3.0 MB CSV vs today's 4.4 MB)
Ladder report on HF	multilingual-snr/msnr-ladder-report (private)	Staler: Aug 28 14:12
Fix is one login-node command (no cluster job): python3.11 ladder_report.py --plot --publish --push-hf — ideally after today's 600M BPB jobs land so you publish once. Note also that multilingual-snr/multilingual-snr-eval-results (the parquet the signal-and-noise pipeline reads) was last built June 4 and only covers the legacy 36-sweep — the predictivity evals exist only in eval_logs/W&B/ladder-CSV form so far.

## 2. BPB evolution across languages and sizes

BPB coverage is still narrow: only L2 and L50 cells (plus a sliver of 90M-L15) at 90M–350M; the 600M runs are executing right now, L1/L8/L15/L30 have none.

Scaling behaves (except 90M): non-English macro BPB falls monotonically with size — L50: 7.03 → 1.61 → 1.38 across 90M/175M/350M — and falls over training within each healthy run.
90M is inverted: BPB rises during training (L2 macro 4.1 → 12.5 first→last checkpoint), the BPB-side signature of the known diverged 90M rung (all 90M cells flagged "diverged" in the report; your two diag- runs probing lr/beta3 are running now).
More languages is free English, cheaper everything else: English (dclm) BPB is identical between L2 and L50 at every size (0.947 vs 0.946 at 350M), while L50 beats L2 on 81–89 of 99 non-English languages, by ~0.4–0.5 bits/byte on average — enormous against a checkpoint noise of ~0.002.
Per-language spread at 350M-L50: best are non-Latin-script languages (tam/tha/ben/kat/mal ≈ 0.55–0.57), worst are low-resource Latin-script ones (som/mlt/kmr/uzn/cym ≈ 2.7).

## 3. Benchmark evolution across languages and sizes

Mean final score and first→last-checkpoint gain, per family (deep/seed1904 ladder):

Real signal already: multiblimp (0.65 → 0.92 going 90M → 600M, well above 0.5 chance), and clear size-monotonic growth on hellaswag (0.25 → 0.30), xnli (0.33 → 0.42), xstorycloze (0.48 → 0.57), xwinograd (0.51 → 0.64), xcopa.
Still at chance even at 600M: belebele, global_mmlu, include, and arc-multilingual all sit ≈ 0.24–0.26 (chance = 0.25) — these knowledge-heavy MC benchmarks haven't emerged at this compute scale.
90M shows ~zero within-run benchmark growth (deltas ≈ 0 or negative), consistent with the divergence.

## 4. Decision accuracy per benchmark

Setup: for each benchmark (language variants separate), rank the 6 language-count recipes (L1…L50, deep/seed1904) by final score at a small size vs at 600M; DA = pairwise agreement (decision_acc_fast). Non-English tasks are only evaluated on cells trained on that language, so they use each task's common recipe subset (≥4 recipes); 60 of 219 parent-level benchmarks qualify. Full table: decision_accuracy.csv in my scratchpad (columns per small size + mean).

Best: hellaswag_de (0.94), xwinograd_jp (0.78), arc_it (0.78), hellaswag_ru (0.77), global_mmlu_full_en (0.73). The hellaswag family is the most decision-reliable overall (mean 0.72, and 0.91 when deciding from 350M).
Worst (≤ coin flip): global_mmlu family (0.48), several include/belebele/xnli variants, multiblimp_spa (0.22). Unsurprising — a benchmark at chance has nothing to rank (their 600M across-recipe ranges are ~0.01).
DA generally improves with the deciding size for the emerged families (hellaswag 0.52 → 0.91 from 90M → 350M; arc 0.52 → 0.64), but not for chance-level families.
English benchmarks average 0.63 vs 0.55 for non-English, but the gap is mostly the chance-level knowledge tasks dragging the non-English pool.
Caveat: with 4–6 recipes DA is quantized to 6–15 pairs, so individual values carry ±0.1–0.17 granularity; family-level averages are the trustworthy read.

## 5. Do arch and language distribution count as "different" models?

No seed pairs exist yet (the ladder report's seed row is empty), so I used checkpoint noise — the std of each benchmark score over the last 3 evaluated checkpoints, median 0.004 — as the noise yardstick, which is the more conservative of the SNR paper's two noise definitions available here.

Language distribution: yes, clearly. At 600M the across-L range of final scores has median 0.022 ≈ 4.4× checkpoint noise, and 56/60 benchmarks separate by > 2× noise (top: xwinograd_en 17×, hellaswag_ru 13×). On BPB the separation is one to two orders of magnitude above noise. The L axis gives genuinely distinct models for SNR.
Arch (deep vs shallow): yes, but from thin evidence. The only clean matched pair with benchmarks on both sides is 175M-L2: median |Δ| 0.013 ≈ 5.7× noise, above 2× noise on 14/17 tasks; on BPB, median |Δ| 0.074 vs noise 0.002 (~35×), with deep slightly better on average (−0.084 bits/byte). The 90M pairs are contaminated by the divergence (the L2 BPB "arch effect" of +8.2 there is the anomaly, not architecture).
Honest caveat: without a seed reroll you can't strictly prove these deltas exceed seed variance — checkpoint noise underestimates seed noise (the SNR paper finds seed noise a few times larger). The L effect at 4–17× noise would survive that comfortably; the 175M-L2 arch effect at ~2–6× on benchmarks is likely real but is the one conclusion a single seed pair could still overturn. The seed-28/1797 cells now marked fresh in the launcher are exactly what closes this.