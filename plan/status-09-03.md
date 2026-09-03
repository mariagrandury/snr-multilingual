# TODOs

bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --convert-only
pkill -f auto_evals_cscs.py  # kill eval watcher(s)

## Pretraining

Finish my 1B pretraining jobs:
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1B --seed 1904 --langs 8 --dry-run 
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1B --seed 1904 --langs 50 --dry-run 

Add the 28 and 1797 seeds to 1B grid (aromanou).

Launch 1.7B pretrainings: L1, L8, L8B, L30, L30B
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1.7B --seed 1904 --langs 30 --scheme B --no-auto-evals 

## Evals

Test full eval of an L50 models to verify all new "auto" benchmarks work properly -> BATCH = 0 and time 12h. Job ID 3285073.
python3.11 pretrain/auto_evals_cscs.py --name lm-175M-L50-deep-seed1904 --max-submit 1

