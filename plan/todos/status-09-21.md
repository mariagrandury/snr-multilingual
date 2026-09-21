# Resume trainings and evals

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel


## Pretraining

✅ 1B original grid:
python3.11 pretrain/launch_trainings.py cscs --size 1B --arch shallow --seed 1904 --partition preemptable --time 23:59:00

✅ AT3 models:
python3.11 pretrain/launch_trainings.py cscs --size 1B --scheme AT3 --seed 1904 --partition preemptable --time 23:59:00 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --partition preemptable --time 23:59:00 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --arch shallow --partition preemptable --time 23:59:00 

✅ 3B models:
python3.11 pretrain/launch_trainings.py cscs --size 3B  --partition preemptable --time 23:59:00
python3.11 pretrain/launch_trainings.py cscs --size 3B --scheme B  --partition preemptable --time 23:59:00


## Convert and eval new ckpts

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --watch 1200
python3.11 pretrain/auto_evals_cscs.py --retry-held
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --watch 1200
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --reformulated rf


## Update analysis with new evals

✅ on the cluster, in the snr env: rebuild and publish the report:
python3.11 src/pretrain/ladder_report.py --plot --publish --push-hf --push-git

✅ mirror eval logs to capstor:
sbatch evals/scripts/mirror_eval_logs.sbatch 

✅ fetch to update local cache:
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual \
  && git fetch origin data/ladder-report \
  && git archive origin/data/ladder-report | tar -x -C src/signal-and-noise/data/ladder-report

✅ fetch new data to cache and rebuild every derived artefact (inc. documents):
✅ a) except the curves (FORCE=1 to force fetch, if the refresh was interrupted haflway just run without FORCE):
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



# Update CLAUDE.md and check legacy grid/evals usefulness

The multilingual snr is a huge project with many experiments, and even has 2 periods (pre and post July).

1. Review the CLAUDE.md files and add any information relevant so all sessions are up-to-date
2. Review the old custom grid of models and think whether we could somehow include the evaluations as a rq or sub-rq to complement our results
3. Review also the external reference models evaluated (olmo, apertus, etc) to see if they could compliment our results and conclusions or they could be included as interesting sub-rqs


# Backlog

- FineTasks
- what do we do with the seeds?
- notes from 09-16
