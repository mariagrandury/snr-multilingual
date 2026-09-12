

✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
✅ sbatch scripts/mirror_eval_logs.sbatch 
✅ bash evals/scripts/launch_bpb.sh
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200
✅ cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 52 --interval 1800

✅ resume 1.7B models:
python3.11 launch_trainings.py cscs --size 1.7B                  # L2 deep
python3.11 launch_trainings.py cscs --size 1.7B --scheme B       # L8, L30 scheme B deep
python3.11 launch_trainings.py cscs --size 1.7B --arch shallow   # L1, L2, L8, L30 shallow
scontrol update jobid=3377944 reservation=SD-69241-apertus-1-5-0 # L2 deep

✅ resume 1B models:
python3.11 launch_trainings.py cscs --size 1B --langs 1 --seed 1904
python3.11 launch_trainings.py cscs --size 1B --langs 15 --seed 1904

✅ rebuild data for 1.7B size:
M=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data
R=$M/rebuild-92B
DST=/iopsstor/scratch/cscs/mariagrandury/data-92B
ONE=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data/submit_build_one.sh
.# the rebuild root needs the shared english + validation links (launch_builds.sh does this)
for d in $R $R/schemeB; do mkdir -p $d && for f in english_dclm.bin english_dclm.idx validation.manifest.json; do ln -sfn $M/$f $d/$f; done; done
sbatch --job-name=build-a-L15-92b --dependency=singleton --export=ALL,BUILD_SCHEME=A,BUILD_STAGE=fineweb,BUILD_SETTING=15,BUILD_OUT=$R,BUILD_DST=$DST $ONE
sbatch --job-name=build-a-L50-92b --dependency=singleton --export=ALL,BUILD_SCHEME=A,BUILD_STAGE=fineweb,BUILD_SETTING=50,BUILD_OUT=$R,BUILD_DST=$DST $ONE
sbatch --job-name=build-b-L15-92b --dependency=singleton --export=ALL,BUILD_SCHEME=B,BUILD_STAGE=fineweb,BUILD_SETTING=15,BUILD_OUT=$R/schemeB,BUILD_DST=$DST $ONE
scontrol update jobid=3378809 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3378808 reservation=SD-69241-apertus-1-5-0 
scontrol update jobid=3378807 reservation=SD-69241-apertus-1-5-0 

✅ resume trainings that took longer bc of mirror eval logs:
python3.11 launch_trainings.py cscs --scheme ZH --size 175M,350M
python3.11 launch_trainings.py cscs --scheme ES --size 175M,350M
python3.11 launch_trainings.py cscs --scheme AT3 --size 175M
python3.11 launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --ademamix-beta3-factor 0.2 --gbs 252


- ✅ launch 90M config comparison runs (4x)
- ✅ mirror_eval_logs time -> 12h
- ✅ Refit evals
- access to aromanou's logs
- launch pretraining of 1B shallow models
- calculate absolute BPB (see details below)
- Reeval after worker implementation
- Add language-specific tasks
- Change eval QA format
- Generate the INCLUDE v2 MC-format tasks and wire them
- Switch or drop LAMBADA-MT
- Get the changes from pretrain-eval-azure in local branch (pull --rebase)
- Implement flops params?
- Write methodology
- Consider models needed language interference study


## 90M config comparison

# B — batch size only, β₃ memory at 20% of each run (recommended)
python3.11 src/pretrain/launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --gbs 252 --ademamix-beta3-factor 0.2
python3.11 src/pretrain/launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --gbs 84  --ademamix-beta3-factor 0.2

# A — grid config except the batch size (β₃ 0.9999)
python3.11 src/pretrain/launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --gbs 252
python3.11 src/pretrain/launch_trainings.py cscs --size 90M --langs 2 --seed 1904 --gbs 84

## Access to aromanou's logs

Command for aromanou
Her training jobs write their Slurm logs to her own scratch, which we can't enter. She runs this once:

L=/iopsstor/scratch/cscs/aromanou/data-mix-small/Megatron-LM/logs
# let mariagrandury pass through down to the log dir, without listing anything
setfacl -m u:mariagrandury:--x /iopsstor/scratch/cscs/aromanou \
    /iopsstor/scratch/cscs/aromanou/data-mix-small \
    /iopsstor/scratch/cscs/aromanou/data-mix-small/Megatron-LM $L $L/slurm
# read the training logs, existing and future (mask pinned so csstaff gains nothing)
setfacl -m u:mariagrandury:r-x -d -m u:mariagrandury:r-x $L/slurm/training
find $L/slurm/training -type f -exec setfacl -m u:mariagrandury:r-- -m m::r-- {} +

Eval logs need nothing from her. Her eval results already sit in our tree and are readable (0 of her 380 entries are unreadable), so the mirror copies them. Her eval Slurm logs are in her home clone, which is private, but no tool reads eval Slurm logs.
Purge protection is a separate step (optional). The mirror's touch pass can't refresh her checkpoint files; only she can: 

find /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr -path '*/checkpoints/*' -user aromanou -exec touch -c {} +

She shouldn't git pull. Her 1B runs are on the old 20-checkpoint schedule, and the current launcher refuses to resume them ("resume it from the checkout that started it"). That old clone is also why her logs still go to her scratch.
After her grant, compute_cost.py picks her logs up automatically. ladder_report.py and the mirror still only read our log directory; each needs a small change, which I haven't made.

## To consider (BPB): Context resets at every scoring window

Production scores back-to-back 4,096-token windows, so the first tokens of each window predict with almost no context. On the same checkpoint, 1,024-token windows read higher than one continuous window by 0.019 bits/byte on English and 0.122 on Tibetan; scaled to production's window count per 1M tokens, that is roughly +0.007 and +0.03.

Impact: every model is scored on identical windows, so rankings and decision accuracy are unaffected. Absolute BPB is pessimistic, and more so for scripts with few bytes per token. Suggestion: if absolute values matter — for comparison with published numbers — score with a sliding window and a 512–1,024-token stride.

[report]
- set the flag to --rope-scaling-factor 8
- fix the “trained language” flag
- validation languages -> good catch! since we will change the final L100 list, we will have to change the validation 100 languages, how will this affect the validation rebuild and calculation of bpb scores? give me a short list of tasks to follow and what to remember when i do this (i will give this task to another session)
- The byte-count self-check never runs in production: explain further this caveat, why is it relevant and the proposed fix
- create a 2nd version of the artifact without the BPB implementation review section


# 1B and 1.7B training status

## 1B

- Done:
    - deep L1 (seeds 28 and 1797), L2 ×3 seeds, L8, L30 ×3 seeds, L50 seed 1904
    - deep scheme B
- Waiting for aromanou: L1-deep-seed1904 is at 41,166 of 45,720 and L15-deep-seed1904 at 36,592. Both stopped on node failures. Only her clone can resume them, because they're on the old save schedule that the current launcher refuses.
- Queued: L50-AT3 deep.
- Not started:
    - deep L50 (seeds 28 and 1797), ZH and ES
    - all 19 shallow 1B cells
- Blocked: L100-AT3, because its data isn't staged.

## 1.7B

- Done: deep L1 and L30.
- Running: deep L8 is at iteration 66,659 of 81,000, about 4.5 h from the end, which this job reaches.
- Queued: L50-AT3 deep.
- Not started: shallow L50-AT3, and shallow scheme B at L8 and L30.
- Blocked:
    - L15 and L50, both arches, and L15 scheme B, both arches: they wait for the 92B data rebuilds;
    - L100-AT3: its data isn't staged.
- Resumes with no job queued:

cell	            at     / target	first job requests	jobs to finish
L2 deep	            65,491 / 81,000	8:15	1
L8 scheme B deep    56,700 / 81,000	11:15	1
L30 scheme B deep	65,176 / 81,000	8:15	1
L1 shallow	        64,512 / 80,640	8:15	1
L2 shallow	        32,016 / 80,640	11:59	2
L8 shallow	        12,096 / 80,640	11:59	3
L30 shallow	        31,667 / 80,640	11:59	2

Each is a 21-node job, so all seven need 147 nodes at once.

The L8-scheme-B job hit TIMEOUT on 09-11 because the previous reservation ended under it. Job 3357160 (1.7B L8 shallow) is less clear: it saved iteration 12,096 at 23:58 and its tasks were terminated a moment later. The log shows no error, no walltime hit and no reservation end, so I don't know why it stopped. Resuming from 12,096 is safe.


cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
# preview all three first: each should list only [resume] lines
python3.11 launch_trainings.py cscs --size 1.7B --dry-run
python3.11 launch_trainings.py cscs --size 1.7B --scheme B --dry-run
python3.11 launch_trainings.py cscs --size 1.7B --arch shallow --dry-run

python3.11 launch_trainings.py cscs --size 1.7B                  # L2 deep
python3.11 launch_trainings.py cscs --size 1.7B --scheme B       # L8, L30 scheme B deep
python3.11 launch_trainings.py cscs --size 1.7B --arch shallow   # L1, L2, L8, L30 shallow


Commands for aromanou:

cd /users/aromanou/projects/mSNR/snr-multilingual/src/pretrain
python3.11 launch_trainings.py cscs --size 1B --langs 1 --seed 1904 --dry-run
python3.11 launch_trainings.py cscs --size 1B --langs 1 --seed 1904
python3.11 launch_trainings.py cscs --size 1B --langs 15 --seed 1904 --dry-run
python3.11 launch_trainings.py cscs --size 1B --langs 15 --seed 1904

## Reservation check

RES=SD-69241-apertus-1-5-0
# nodes held now, and nodes queued, per user
squeue -R $RES -h -t R  -o "%u %D" | awk '{n[$1]+=$2} END {for (u in n) print n[u], u}' | sort -rn
# running jobs, soonest end first: user, nodes, time left, expected end, name
squeue -R $RES -h -t R -o "%.14u %.4D %.11L %.20e %j" -S e
# nodes freeing up, per hour from now
squeue -R $RES -h -t R -o "%D %e" | while read n e; do echo "$n $(( ($(date -d "$e" +%s) - $(date +%s)) / 3600 ))"; done | awk '{f[$2]+=$1} END {for (h in f) print "+" h "h", f[h]}' | sort -t+ -k2 -n
# when Slurm expects the queued jobs to start
squeue -R $RES -h -t PD -o "%.14u %.4D %.20S %j" -S S
