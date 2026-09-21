# Commands

Finish training and eval grid.

✅ resume training of 1.7B models:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1B --scheme AT3 --arch shallow

cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 84
scontrol update jobid=3398524 reservation=SD-69241-apertus-1-5-0 

✅ eval new ckpts:
bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --watch 1200

✅ save new results:
python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
sbatch evals/scripts/mirror_eval_logs.sbatch 

cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 45  --interval 1800

# Training

Define L100:
- Add language-specific tasks
- Decide L100 list, build data mix, launch pretrainings

# Evals

BPB:
- calculate BPB with final L100 validation
- calculate absolute BPB (see details below)

Update tasks list and reeval:
- Add language-specific tasks
- Add INCLUDE v2
- Switch or drop LAMBADA-MT
- Reeval after worker implementation
- Change eval QA format
    - https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py
    - https://arxiv.org/pdf/2412.04403

# Analysis

Define RQs:
- Define final RQs
- Design main figure for each RQ

Backlog:
- Implement flops params?
- Get the changes from pretrain-eval-azure in local branch (pull --rebase)

# Paper

- Write methodology
- Draft analysis section with RQs, experimental setup and expected finding
- Write introduction
- Write abstract
- [DEADLINE] Friday 18/09: Abstract deadline
- [DEADLINE] Monday 21/09: Share paper with colleagues
- [DEADLINE] Friday 25/09: Paper deadline

# aromanou

- launch pretraining of 1B shallow models
- decide language distribution for low-resource language experiments
- launch pretraining of <=1B low-resource experiment models
- research RQ2 about 
