
# Clariden

## commit and push

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && git add . && git commit -q -m wip && git push 2>&1 | tail -8 && git status -sb | head -1 && git log --oneline -1"' 2>&1 | tail -20

## BPB

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src && bash evals/scripts/launch_bpb.sh"' 2>&1 | tail -30

## Auto evals

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src && SBATCH_PARTITION=preemptable python3.11 pretrain/auto_evals_cscs.py"' 2>&1 | tail -30

## Check Slurm queue and logs

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'ps -u mariagrandury -o pid,etime,args | grep -v grep | grep -E "auto_evals|launch_bpb|drain|ladder_report" ; echo "--- squeue"; squeue -u mariagrandury -o "%.10i %.12P %.40j %.2t %.10M" | head -25; echo "--- queued: $(squeue -u mariagrandury -h | wc -l)"; echo "--- newest logs"; ls -t /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/logs 2>/dev/null | head -3; ls -t /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/logs 2>/dev/null | head -3' 2>&1 | tail -45

## Ladder report

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src && mkdir -p /iopsstor/scratch/cscs/mariagrandury/logs && LOG=/iopsstor/scratch/cscs/mariagrandury/logs/ladder*report*\$(date +%Y%m%d*%H%M%S).log && nohup python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git > \$LOG 2>&1 < /dev/null & sleep 5; ps -u mariagrandury -o pid,etime,args | grep -v grep | grep ladder_report; ls -t /iopsstor/scratch/cscs/mariagrandury/logs/ladder_report*\*.log | head -1"' 2>&1 | tail -5

## Trainings

ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/cscs-key -o IdentitiesOnly=yes -J ela mariagrandury@clariden-ln002.cscs.ch 'bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr && cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src && \
echo \"=== [1] 3B A\" && python3.11 pretrain/launch_trainings.py cscs --size 3B --partition preemptable --time 23:59:00 2>&1 | tail -12; \
echo \"=== [2] 3B B\" && python3.11 pretrain/launch_trainings.py cscs --size 3B --scheme B --partition preemptable --time 23:59:00 2>&1 | tail -12; \
echo \"=== [3] DCLMP\" && python3.11 pretrain/launch_trainings.py cscs --scheme DCLMP --partition preemptable --time 23:59:00 2>&1 | tail -12; \
