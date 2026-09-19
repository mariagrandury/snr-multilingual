# Training

✅ resume training of 1.7B L15 and L50 with new data mixture:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --dry-run
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B 
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme B --arch shallow

✅ resume training of L2 mixtures:
python3.11 pretrain/launch_trainings.py cscs --scheme ZH --size 1B
python3.11 pretrain/launch_trainings.py cscs --scheme ES --size 1B

✅ resume training of AT3 mixture:
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3
python3.11 pretrain/launch_trainings.py cscs --size 1.7B --scheme AT3 --arch shallow
python3.11 pretrain/launch_trainings.py cscs --size 1B --scheme AT3 --arch shallow


cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 84
scontrol update jobid=3398524 reservation=SD-69241-apertus-1-5-0 

# Evals

✅ bash evals/scripts/launch_bpb.sh
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200

✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
✅ sbatch evals/scripts/mirror_eval_logs.sbatch 

cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 45  --interval 1800

# ToDos

- decide L100 list, build data mix, launch pretrainings
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
- launch pretraining of 1B shallow models
- launch pretraining of <=1B low-resource experiment models
