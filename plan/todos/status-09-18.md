# Commands

Finish training and eval grid.

✅ resume training of 1.7B models:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow
 
When we have L100:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1B --scheme AT3 --arch shallow

cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 84
scontrol update jobid=3398524 reservation=SD-69241-apertus-1-5-0 

bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/scripts/preempt_drain.sh --dry-run

squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /seed28/ {print $1}' | xargs -r scancel

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --watch 1200
python3.11 pretrain/auto_evals_cscs.py --retry-held
SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py --reformulated rf

✅ save new results:
python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
sbatch evals/scripts/mirror_eval_logs.sbatch 

cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 45  --interval 1800

# Regenerate all plots with new threshold

-> Needs slurm because >1h

python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git

sbatch --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
  --job-name=snr-analysis \
  --output=/iopsstor/scratch/cscs/mariagrandury/snr-analysis-%j.log \
  --wrap='source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise && FORCE=1 PY=python HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 SNR_LADDER_DIR=SNR_LADDER_DIR=/capstor/store/cscs/swissai/infra01/msnr-ladder-report bash run_all_predictivity.sh'


# ToDos

- task reformulation: programatically
- task reformulation: Gemini
- prep 3B architecture config

# Finish grid pretraining and eval

python3.11 pretrain/auto_evals_cscs.py --retry-held


# 90M

- decision: out of grid
- remove from auto evals and pretraining plan
- do not waste compute:
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /90M/ {print $1}' | xargs -r scancel

# Language reformulation

D=/iopsstor/scratch/cscs/mariagrandury/hf_home/datasets
find $D -maxdepth 1 -name '*.lock' ! -user mariagrandury -size 0 -delete
for cfg in $D/*/*/*/*/; do cfg=${cfg%/}; l="$D/$(echo "$cfg" | tr / _).lock"; [ -e "$l" ] || touch "$l"; done
find $D -maxdepth 1 -name '*.lock' -user mariagrandury -exec setfacl -m m::rwx {} +
# then restart the rf watcher on the new code
kill 225219; cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
nohup python3.11 -u auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904 --max-submit 20 --watch 1800 >> /iopsstor/scratch/cscs/mariagrandury/auto_evals_rf_watch.log 2>&1 &


python3.11 pretrain/auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904 --retry-held --max-submit 20

python3.11 pretrain/auto_evals_cscs.py --reformulated --arch deep --scheme A --seed 1904
