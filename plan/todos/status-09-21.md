# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel


## Pretraining

pretrain-1B-L2-shallow-seed1904 # aromanou

python3.11 pretrain/launch_trainings.py cscs --size 1B --arch shallow --seed 1904 --partition preemptable
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme ZH --seed 1904 --partition preemptable

python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --partition preemptable --time 23:59:00 

python3.11 pretrain/launch_trainings.py cscs --size 3B  --partition preemptable --time 23:59:00
python3.11 pretrain/launch_trainings.py cscs --size 3B --scheme B  --partition preemptable --time 23:59:00



python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M --arch shallow --scheme B --seed 1904 --partition preemptable
python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M --scheme AT3 --seed 1904 --partition preemptable
python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M --scheme AT3 --seed 1904 --arch shallow --partition preemptable
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /90M/ {print $1}' | xargs -r scancel


python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M,1B,1.7B --scheme BT3 --partition preemptable --dry-run

## Convert and eval new ckpts

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --watch 1200
python3.11 pretrain/auto_evals_cscs.py --retry-held
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1200
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --reformulated rf


## Build data

sbatch --job-name=build-bt3-L30 --dependency=singleton --time=23:59:00 --exclusive \
   --partition=preemptable \
   --export=ALL,BUILD_SCHEME=BT3,BUILD_STAGE=fineweb,BUILD_SETTING=30,BUILD_OUT=$OUT/BT3 \
   /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data/submit_build_one.sh
  
Submitted batch job 3471038



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
sbatch --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
  --job-name=snr-analysis \
  --output=/iopsstor/scratch/cscs/mariagrandury/snr-analysis-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && FORCE=1 PY=python HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 bash scripts/refresh_analysis.sh --curves'

✅ fetch new data to cache and rebuild every derived artefact (only analysis/ figures):
FORCE=1 bash run_all_predictivity.sh


# Figures

- final-final-final-review, commit and push, merge into main


# Evals

BPB:
- calculate BPB with final L100 validation
- calculate absolute BPB (see details below)

Update tasks list and reeval:
- Add language-specific tasks
- Add INCLUDE v2
- Switch or drop LAMBADA-MT
- Reeval after worker implementation


# Task reformulation

## Programatically

- ✅ eval reformulations
- ✅ fix comparison metrics
- ✅ calculate statistical significance

## Gemini

- ✅ write plan and implement
- ✅ need API key
- [BLOCKED] admin access to create bucket


(snr) mariagrandury@clariden-ln004:/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual> gcloud storage buckets create gs://silin-482809-msnr-rfgm \
>     --project=silin-482809 --location=US --uniform-bucket-level-access

Creating gs://silin-482809-msnr-rfgm/...
ERROR: (gcloud.storage.buckets.create) HTTPError 403: maria.grandury@epfl.ch does not have storage.buckets.create access to the Google Cloud project. Permission 'storage.buckets.create' denied on resource '//storage.googleapis.com/projects/_/buckets/silin-482809-msnr-rfgm' (or it may not exist). This command is authenticated as maria.grandury@epfl.ch which is the active account specified by the [core/account] property.



cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
nohup python3.11 src/evals/scripts/rewrite_items_gemini.py online --family belebele --L 50 --retry-rejects \
  >> /iopsstor/scratch/cscs/mariagrandury/rfgm_online.log 2>&1 &



# Update CLAUDE.md and check legacy grid/evals usefulness

The multilingual snr is a huge project with many experiments, and even has 2 periods (pre and post July).

1. Review the CLAUDE.md files and add any information relevant so all sessions are up-to-date
2. Review the old custom grid of models and think whether we could somehow include the evaluations as a rq or sub-rq to complement our results
3. Review also the external reference models evaluated (olmo, apertus, etc) to see if they could compliment our results and conclusions or they could be included as interesting sub-rqs


# Update model grid
- due_iters returns exactly 12 at every size — 247 future checkpoint-evals saved, 128 at 1B / 71 at 3B / 48 at 1.7B
- 24 never-launched seed replicates dropped (3,782 node-h); nothing on disk touched
- BT3 registered — and your job 3471038 is now building it
- ZH undersized exception lives in the registry; the launcher is idempotent again
- plan/l1_third_family.md

# Backlog

- plan/l1_third_family.md
- FineTasks
- what do we do with the seeds?
- notes from 09-16





Reliable benchmark-language cells (DA ≥ 0.8)
I had to make a call you should check. Reducing each task's DA to one number per axis, at your 0.8 cut:

reduction	DA-size ≥0.8	DA-ckpt ≥0.8	both	languages
mean over all cells	6	5	2	2
median	6	13	3	3
max over all cells	25	99	25	19
late (fixed cells)	13	47	12	11
mean is unusable (2 tasks). max is worse than unusable — a task has ~4 DA-size cells but ~45 DA-ckpt cells, so max-over-45 clears 0.8 for 99/113 tasks and the "both" test silently collapses to DA-size alone (0 tasks pass size but fail ckpt). I made the default late: DA-size at the largest proxy (1B→1.7B) and DA-ckpt at 90% of that same run — one fixed cell per axis, nothing chosen by its value. All three are still written to the CSV.

The 12 passing cells are HellaSwag in 11 languages, plus lambada_openai_mt_en:

da, en, es, fr, it, nl, pt, ro, ru, uk, vi
hellaswag	0.80–1.00 DA-size, 0.80–0.97 DA-ckpt
Per language, benchmarks passing both: en 2; da/es/fr/it/nl/pt/ro/ru/uk/vi 1 each; and 0 for all 21 others — ar, bg, bn, cs, de, el, fa, fi, he, hi, hu, id, ja, ka, no, pl, sv, ta, th, tr, zh.

Key findings:

One benchmark family carries the entire result. HellaSwag is the only benchmark that passes in more than one language. This independently reproduces your README's 2026-09-01 note that HellaSwag is the most decision-reliable family — computed from a different pipeline on different pairs.
The binding constraint is size, not checkpoint. 47 tasks clear DA-ckpt but only 13 clear DA-size. Reading a benchmark early is cheap; reading it small is what breaks.
Reliability tracks script/family, not resource rank. Every passing language is Latin or Cyrillic. German fails despite 7 benchmarks and high resource; zh, ja, el, ar, he, ta, ka — all non-Latin — pass nothing. That's a stronger pattern than the resource ordering and worth a sentence in the paper.


# Definition pool of DA

Good catch, but the two things are independent — and no, it doesn't follow.

The `_multi_axis` / `_one_axis` suffix is about the **axes mode**, which I'll implement the same way whatever we decide. What your preference actually constrains is the **output directory**, since outputs go to `analysis/<rq>/<stage>/<pool>/`. And the output directory does not have to equal the pool the pairs come from — `by_L.py` and `scale_convergence.py` already separate those: both read `predictivity_all` internally (`L_POOL` / `POOL`) while `--pool predictivity` picks only the gate and the output folder. Only `reliable_tasks.py` couples the two, because it reads `da_per_task.csv` out of the `--pool` directory.

So there's a third option that gives you exactly the side-by-side you want.

**A. Change the `predictivity` pool to include all six schemes**
- *Pros:* one pool everywhere; outputs stay in `predictivity/`; no new plumbing; `reliable_tasks` picks up ES/ZH/AT3/BT3 for free.
- *Cons:* SNR signal is `(max − min) / mean` across the pool's variants, so adding temperature and second-language interventions **inflates dispersion without widening the decision the pool measures** — precisely what that pool's own description warns against. It moves rq03's variant ranking, rq04's surrogates and rq09, i.e. the paper's whole SNR story, not just RQ2. The rq00 gate is also computed `--only predictivity`, so membership changes which (task, size) cells pass.

**B. Run rq2 with `--pool predictivity_schemes`**
- *Pros:* headline pool untouched; the pool already exists for exactly this.
- *Cons:* every rq2 output moves to a new directory — the cross-directory comparison you don't want — and the gate would have to be computed for that pool too.

**C. Keep `--pool predictivity` for the gate and the output directory; let `reliable_tasks` read the scheme-inclusive DA table (recommended)**
- *Pros:* all rq2 figures stay in `predictivity/`, so `rq2_above_66_both_multi_axis.png` and `..._one_axis.png` sit next to everything already there. rq03/rq04/rq09 untouched. It mirrors the decoupling `by_L`/`scale_convergence` already use, so it's a precedent, not a new concept. It also fixes the #9 mismatch — reliability would finally be judged on the same population the figures filter.
- *Cons:* one more distinction to hold in your head ("the DA table I read" vs "the pool I'm reported under"), `compute_da` has to run for `predictivity_schemes` as an extra step, and `predictivity/da_reliable_tasks.csv` needs a column or note saying which pool its DA came from, or a future reader will assume `predictivity`.

I'd go with **C**: it's the only one that gets you ES/ZH/AT3/BT3 in the pairs *and* the two suffixed plots in one directory *and* leaves the SNR numbers alone. The only real cost is documenting the provenance, which the `axes` column gives me a natural place to do.
