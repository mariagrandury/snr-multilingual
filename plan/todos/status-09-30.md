# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel

## Pretraining

python3.11 pretrain/launch_trainings.py cscs --scheme FWEB --partition preemptable

## Convert and eval new ckpts

cd Projects/snr-multilingual/src/ && conda activate snr

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1800
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
