# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel


## Pretraining

python3.11 pretrain/launch_trainings.py cscs --size 3B  --partition preemptable --time 23:59:00
python3.11 pretrain/launch_trainings.py cscs --size 3B --scheme B  --partition preemptable --time 23:59:00

python3.11 pretrain/launch_trainings.py cscs --size 1.7B --langs 2 --scheme ES --seed 1904 --partition preemptable --time 23:59:00

python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M,1B,1.7B --scheme BT3 --partition preemptable --dry-run

## Convert and eval new ckpts

cd Projects/snr-multilingual/src/ && conda activate snr

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1200
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --retry-held

✅ probe new benchmarks:
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --group auto_probe --size 600M,1B,1.7B --seed 1904 --final-only

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
sbatch --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
  --job-name=snr-analysis \
  --output=/iopsstor/scratch/cscs/mariagrandury/snr-analysis-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && FORCE=1 PY=python HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 bash scripts/refresh_analysis.sh --curves'

✅ fetch new data to cache and rebuild every derived artefact (only analysis/ figures):
FORCE=1 bash run_all_predictivity.sh

# Scheduled update

loginctl enable-linger

systemctl --user enable --now ladder-nightly.timer
>> Created symlink /users/mariagrandury/.config/systemd/user/timers.target.wants/ladder-nightly.timer → /users/mariagrandury/.config/systemd/user/ladder-nightly.timer.

systemctl --user list-timers ladder-nightly
>> NEXT                             LEFT LAST PASSED UNIT                 ACTIVATES             
>> Wed 2026-09-23 06:00:00 CEST 3h 41min -         - ladder-nightly.timer ladder-nightly.service
>> 1 timers listed.
>> Pass --all to see loaded but inactive timers, too.

To see how it went:

cat /iopsstor/scratch/cscs/mariagrandury/logs/nightly-ladder/last-run.txt
systemctl --user status ladder-nightly.service     # exit status of the last run


# Build L1 & L2

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
BUILD_PARTITION=preemptable ./launch_builds.sh     # adds build-en-dclmp + build-en-fweb

monitor build:
bash /iopsstor/scratch/cscs/mariagrandury/buildwatch.sh 

- DCLMp -> 17h -> Wed 12h
- FineWeb -> 25h -> Wed 21h
- 1.7B EN pretraining done -> Thu 23h
- 1.7B ZH pretraining done -> Wed 00h
- 1.7B ES pretraining done -> Thu 23h

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

# L2 ES?

- checking data availability to pretrain 1.7B

# Evals

Update tasks list and reeval:
- Add language-specific tasks
    - Check the results from La Leaderboard, see which benchmarks give signal at small scales for ES, CA, GL, EU
- Add INCLUDE v2
- 🛑 (Switch or drop LAMBADA-MT -> what was this about?
- 🛑 Reeval after worker implementation -> what was this about? are the current values above or below before implementing the eval workers
- mmlu, openbookqa, commonsense_qa, blend, cultural_bench, truthfulqa_mc2? triviaqa, squadv2, agieval, bbh, toxigen, multi-if, mbpp_instruct, mathqa, ifeval, humaneval_instruct, hendrycks_math, drop, bbq, acp_bench

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain

# probe: originals + their rf_ twins, 84 jobs, ~48 node-h
SBATCH_PARTITION=preemptable /users/mariagrandury/miniconda3/envs/snr/bin/python3.11 \
  auto_evals_cscs.py --group auto_probe --size 600M,1B,1.7B --final-only --seed 1904

# INCLUDE v2 (OG + EN, L50), 77 jobs, ~39 node-h
SBATCH_PARTITION=preemptable /users/mariagrandury/miniconda3/envs/snr/bin/python3.11 \
  auto_evals_cscs.py --group auto_include_v2 --size 600M,1B,1.7B --final-only --seed 1904


# Task reformulation

## Programatically

- ✅ probe eval reformulations

## Gemini

- [BLOCKED] admin access to create bucket

✅ launch bucket reformulation:

(snr) mariagrandury@clariden-ln004:/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual> gcloud storage buckets create gs://silin-482809-msnr-rfgm \
>     --project=silin-482809 --location=US --uniform-bucket-level-access

Creating gs://silin-482809-msnr-rfgm/...
ERROR: (gcloud.storage.buckets.create) HTTPError 403: maria.grandury@epfl.ch does not have storage.buckets.create access to the Google Cloud project. Permission 'storage.buckets.create' denied on resource '//storage.googleapis.com/projects/_/buckets/silin-482809-msnr-rfgm' (or it may not exist). This command is authenticated as maria.grandury@epfl.ch which is the active account specified by the [core/account] property.

✅ launch online reformulation:

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual

nohup env GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=silin-482809 GOOGLE_CLOUD_LOCATION=global \
   /users/mariagrandury/miniconda3/envs/snr/bin/python3.11 \
   src/evals/scripts/rewrite_items_gemini.py online \
   --family belebele --L 50 --retry-rejects \
   >> /iopsstor/scratch/cscs/mariagrandury/rfgm_online.log 2>&1 &



python3.11 src/evals/scripts/make_rf_tasks.py --set rfgm --family belebele   # registers 59 tasks + YAMLs
# validate them through a real TaskManager, then:
python3.11 src/pretrain/auto_evals_cscs.py --dry-run                          # price the top-up FIRST
# only then add rfgm_belebele to groups.auto




# Update CLAUDE.md and check legacy grid/evals usefulness

The multilingual snr is a huge project with many experiments, and even has 2 periods (pre and post July).

1. Review the CLAUDE.md files and add any information relevant so all sessions are up-to-date
2. Review the old custom grid of models and think whether we could somehow include the evaluations as a rq or sub-rq to complement our results
3. Review also the external reference models evaluated (olmo, apertus, etc) to see if they could compliment our results and conclusions or they could be included as interesting sub-rqs
4. check whether we could compare our results with the ones from FineTasks (https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks) or get any insight from their analysis and replicate/extend it with our data


# Backlog

- what do we do with the seeds? -> error bars for DA
- notes from 09-16


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



Files to delete (nothing removed; all folded or superseded)

analysis/ANALYSIS_new_vs_previous.md, analysis/PARALLEL_SESSIONS.md, analysis/paper_figures.md
analysis/rq03_noise_and_snr/INSTRUCTIONS.md, rq07_external_frameworks/INSTRUCTIONS.md, rq08_subset_selection/INSTRUCTIONS.md, rq09_benchmark_design/INSTRUCTIONS.md
analysis/rq02_decision_accuracy/pretraining/predictivity/README.md (now a pointer)
rq00_gate_and_curves/pretraining/predictivity/benchmark_floor.png and .csv
rq00_task_reformulation/twins_gate.png, twins_gate.csv, twins_gate_mcnemar.csv, __pycache__/twins_gate.cpython-311.pyc
rq09_benchmark_design/pretraining/predictivity/finetasks_criteria.{csv,png}, finetasks_surrogates_rank.csv, finetasks_surrogates_scatter.{csv,png}




# 90m and 175M

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
sbatch refetch_fineweb.sh
Submitted batch job 3496755

i removed BT3 from the ladder since we're not going to train them finally and launched all pretrainings

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
for f in "" "--arch shallow" "--scheme AT3" "--scheme AT3 --arch shallow" \
         "--scheme B" "--scheme B --arch shallow" \
         "--scheme ZH" "--scheme ES" "--scheme DCLMP" "--scheme FWEB"; do
  for s in 90M 175M; do
    python3.11 launch_trainings.py cscs --size $s $f --partition preemptable
  done
done


# New eval probe

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src && \
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py \
    --group auto_probe --size 600M,1B,1.7B --final-only

