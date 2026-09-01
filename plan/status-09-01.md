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

1. ✅ Let the 40 running 600M evals land (today, no action). Closes benchmarks for 4 sizes × 6 language settings.
1. ✅ (Deps on #1) Launched job to mirror eval logs to capstor.
1. ✅ Launched 3 jobs to finish pretraining the 175M scheme B models.
1. ✅ Launched 2 jobs to finish BPB calculation of L2 and L50 models.
2. [Discuss] Update the list of available high-quality benchmarks for low resource languages.
2. 🛑 Launch the creation of the L100 data mixture with the new list of languages (1 day). Blocks pretraining of all L100 models. Blocked by update of available benchmarks for low resource languages.
3. The 90M β₃ confirming run (~2 node-hours). Tiny, and it decides whether the ladder has 3 or 4 rungs — which changes what you can claim. Do this before writing any scaling section.
4. Update the SNR module to new naming. Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 6 language settings.
5. [Delegate] BPB on the remaining trained cells — ~330 checkpoints, ~66 node-hours, 1–2 days. Cheapest item on the board and it's the plan's stated outcome metric. Unlocks the loss/BPB scaling fits immediately.
6. [Discuss] Update the model grid plan so each size-languages cell has at least 3 models (we need 3 to calculate DA).
7. 🛑 [Delegate] (Deps on #5) The 8× 175M seed replicates (~26 node-hours). This is what makes SNR computable rather than just signal. Highest analysis-value per node-hour in the whole sweep.
8. 🛑 [Delegate] (Deps on #6) The X remaining ≤600M cells (~107 nh) → convert → eval → BPB (~400 nh). X is currently 12 but might change when updating the model grid plan.
9. 🛑 (Deps on #2) The 4 small sizes of L100 models (classic: deep, A).
10. 🛑 Further writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × all 7 language settings.
11. 🛑 (Deps on Azure capacity) 1B/1.7B in parallel on Azure.
11. 🛑 (Deps on #6) Selected 1B models in parallel on CSCS.
