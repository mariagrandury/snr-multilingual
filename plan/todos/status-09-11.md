Save night job progress:
✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
✅ sbatch evals/scripts/mirror_eval_logs.sbatch 

Download missing GL tasks:
✅ python3.11 evals/scripts/download_eval_datasets.py

Resubmit with --all-languages:
✅ python3.11 pretrain/auto_evals_cscs.py --arch deep --scheme A --seed 1904 --all-languages
✅ python3.11 pretrain/auto_evals_cscs.py --arch deep --scheme B --seed 1904 --all-languages
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /90M/ {print $1}' | xargs -r scancel

Cancel requeued jobs for ZH and ES since they are built:
✅ scancel 3353771 3353773

Launch 175M-600M pretraining of ZH and ES:
✅ python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M --scheme ZH
✅ python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M --scheme ES

Resume 1.7B deep models and start training of shallow models:
✅ python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch deep --dry-run
✅ python3.11 pretrain/launch_trainings.py cscs --size 1.7B --arch shallow --dry-run

eval new ckpts as they are available:
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200
✅ bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --retry-held

evening:
8x 1.7B (21 nodes):
✅ cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --priority build --max-nodes 50 --interval 1800 --hours 72

night:
✅ bash evals/scripts/launch_bpb.sh
✅ python3.11 pretrain/launch_trainings.py cscs --scheme AT3 --dry-run
✅ python3.11 pretrain/auto_evals_cscs.py --watch 1200
✅ cd Projects/snr-multilingual/ && bash scripts/reservation_drain.sh --max-nodes 150 --interval 1800

morning:
python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
sbatch scripts/mirror_eval_logs.sbatch 

- Refit evals
- Reeval after worker implementation
- Add language-specific tasks
- Change eval QA format
- Generate the INCLUDE v2 MC-format tasks and wire them
- Switch or drop LAMBADA-MT
- Get the changes from pretrain-eval-azure in local branch (pull --rebase)
- Implement flops params?
- Write methodology
- Consider models needed language interference study
