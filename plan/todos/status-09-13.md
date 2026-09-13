✅ bash evals/scripts/launch_bpb.sh
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200
✅ cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 66  --interval 1800

✅ resume pretrainings:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --dry-run
python3.11 pretrain/launch_trainings.py cscs --scheme ZH --size 175M,350M
python3.11 pretrain/launch_trainings.py cscs --scheme ES --size 175M,350M
python3.11 pretrain/launch_trainings.py cscs --scheme AT3 --size 175M

✅ finish 92B rebuild:
scontrol update jobid=3378824 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3378823 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3378822 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3389994 reservation=SD-69241-apertus-1-5-0 

✅ resume pretrainings:
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
python3.11 pretrain/launch_trainings.py cscs --scheme AT3 --size 1B,1.7B
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B        # L30 only; running L8 is skipped
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow    # L1 only; the rest are running
scontrol update jobid=3389998 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3390000 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3390001 reservation=SD-69241-apertus-1-5-0 

✅ finish 90m tests:
python3.11 pretrain/launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --ademamix-beta3-factor 0.2 --gbs 252
scontrol update jobid=3390002 reservation=SD-69241-apertus-1-5-0 

✅ start training of 1.7B L15 and L50 with new data mixture:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --dry-run
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B --arch shallow
7 models * 21 nodes = 147
cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 150  --interval 1800

eval --all-languages for all deep models:
✅ implement change
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200
✅ squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /90M/ {print $1}' | xargs -r scancel

✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git

- ✅ resume 90M config comparison runs (4x)
- ✅ resume 1.7B models
- ✅ resume building AT3 data mixture
- ✅ resume building L15 and L50 92B data mixtures
- ✅ resume AT3 models
- calculate absolute BPB (see details below)
- Reeval after worker implementation
- Add language-specific tasks
- Change eval QA format
- Add INCLUDE v2
- Switch or drop LAMBADA-MT
- Get the changes from pretrain-eval-azure in local branch (pull --rebase)
- Implement flops params?
- Write methodology
- Consider models needed language interference study

aromanou:
- ✅ access to aromanou's logs
- launch pretraining of 1B shallow models
- launch pretraining of <=1B low-resource experiment models

# 90M config

cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
RUNS=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr
PLANS=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/conversion-plans
export HF_TOKENIZER=swiss-ai/Apertus-70B-2509
export STAGING_BASE=/capstor/store/cscs/swissai/infra01/msnr-hf-models
export TMP_TORCH_BASE=/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/_tmp_torch

## convert: one job per run, all 20 checkpoints
for spec in "diag-90M-L2-deep-seed1904-beta3f0.2 225" \
            "diag-90M-L2-deep-seed1904-gbs84-tok9.29B 1350"; do
  set -- $spec
  echo "MODEL $1 $RUNS/$1/checkpoints $(seq -s ' ' $2 $2 $(($2 * 20)))" > $PLANS/plan-$1.txt
  sbatch --partition=normal --time=00:45:00 --job-name=convert-snr-$1 \
         --export=ALL,PLAN_FILE=$PLANS/plan-$1.txt src/pretrain/conversion/convert-snr.sh
done

JOBS: 3391201 3391202

## BPB: starts after its conversion succeeds, scores every converted checkpoint
for exp in diag-90M-L2-deep-seed1904-beta3f0.2 diag-90M-L2-deep-seed1904-gbs84-tok9.29B; do
  dep=$(squeue --me -h -n convert-snr-$exp -o %i)
  sbatch ${dep:+--dependency=afterok:$dep} --job-name=bpb-$exp \
         src/evals/scripts/score_bpb.sbatch $exp
done

## 175M runs
python3.11 src/pretrain/launch_trainings.py cscs --size 175M --langs 2 --seed 1904 --gbs 168
3391171
python3.11 src/pretrain/launch_trainings.py cscs --size 175M --langs 2 --seed 1904 --gbs 72
3391172

rerunning 175M model: <1000 node hours
