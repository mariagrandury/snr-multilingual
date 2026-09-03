# TODOs

bash evals/scripts/launch_bpb.sh
python3.11 pretrain/auto_evals_cscs.py --convert-only
pkill -f pretrain/auto_evals_cscs.py  # kill eval watcher(s)


to cancel:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1}' | xargs -r scancel
to preview:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1, $2}'



## Pretraining

Finish my 1B pretraining jobs:
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1B --seed 1904 --langs 8 --dry-run 
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1B --seed 1904 --langs 50 --dry-run 

Add the 28 and 1797 seeds to 1B grid (aromanou).

Launch 1.7B pretrainings: L1, L8, L8B, L30, L30B
python3.11 pretrain/launch_trainings.py cscs --arch deep --size 1.7B --seed 1904 --langs 30 --scheme B --no-auto-evals 

## Evals

Test full eval of an L50 models to verify all new "auto" benchmarks work properly -> BATCH = 0 and time 12h. Job ID 3285073.

lines 320 auto_evals_cscs.py
    env = {**os.environ,
           "LM_EVAL_BACKEND": "vllm",
           "TOKENIZER": TOKENIZER_MODEL,
           "BOS": "true",
           "APPLY_CHAT_TEMPLATE": "false",
           "BATCH_TASKS": "0",
           "TP": "1", "PP": "1",
           "WANDB_ENTITY": WANDB_ENTITY,
           "WANDB_PROJECT": PROJECT_NAME,
           "LOGS_ROOT": str(logs_root),
           "TASKS": tasks}
    cmd = ["sbatch", f"--job-name={job_name('eval', name)}",
           f"--time=11:59:00",
           "--export=ALL", "scripts/evaluate.sbatch", str(hf_dir), name]

python3.11 pretrain/auto_evals_cscs.py --name lm-175M-L50-deep-seed1904 --max-submit 1

