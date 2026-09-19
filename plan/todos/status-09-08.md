Useful commands:

bash evals/scripts/launch_bpb.sh --filter '600M-L50'
python3.11 pretrain/auto_evals_cscs.py --convert-only
pkill -f pretrain/auto_evals_cscs.py  # kill eval watcher(s)


to preview:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1, $2}'
to cancel:
squeue -u $USER -h -o '%i|%j' | awk -F'|' '$2 ~ /^eval-(90M|1B)-/{print $1}' | xargs -r scancel

preview first
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /lm-1B/ {print $1, $2}'
then cancel
squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /L100/ {print $1}' | xargs -r scancel


---

# ToDo

- ✅ python3.11 pretrain/ladder_report.py --plot --publish --push-hf --push-git
- ✅ Fix HF_HUB_OFFLINE=1 for tokenizer

- ✅ python3.11 pretrain/auto_evals_cscs.py --convert-only
- ✅ bash evals/scripts/launch_bpb.sh
- ✅ python3.11 pretrain/auto_evals_cscs.py
- ✅ python launch_trainings.py cscs --size 175M,350M,600M,1.7B
- ✅ squeue --me -h -o '%i|%j' | awk -F'|' '$2 ~ /L100/ {print $1}' | xargs -r scancel


After first full round of evals:
- Rerun pre-optimization evals. The measurement this session produced says the worker pool itself moved scores: old-batched vs new-worker differs on 3.5% of accuracy metrics, max 0.0200, against a same-settings floor of 0.5%, max 0.0020. That is a real, systematic, time-correlated offset between checkpoints evaluated before and after 2026-09-04 — and the SNR noise estimate over 5 late checkpoints is exactly where such an offset would masquerade as noise. The pipeline change is already committed and already running; the open decision is whether to re-evaluate the pre-Sep-4 checkpoints. At ~19 node-hours for the current backlog it is cheap, and it is cheaper to decide now than after the fit.
- Re-fit SAFETY/OVERHEAD_MIN. Median actual/requested walltime is 0.05 across 103 completed jobs. Over-requesting 20× suppresses backfill, which is the mechanism that would give you the 14 concurrent nodes you had at peak. Not urgent, but it's now the biggest lever on queue throughput.
- Don't refit MIN_PER_TASK from the worker rows yet. The report now shows 66 worker jobs at 175M, but nearly all are resumes that got the cheap leftovers — the same bias that produced the bogus 13.1×. Only job 3314667 (329 tasks, 0 skipped) is clean. Wait for ~5 more full-list runs.


Pretrain:
- Get the changes from pretrain-eval-azure in local (pull --rebase)


## flops_params is documented as implemented; none of it exists
The sweep caught all five milestone_iters mentions (verified complete). It missed the sibling feature. At the reviewed ref, every one of these greps returns 0:


def flops_params                              0
sync_models_json → n_non_emb/d_model/vocab    0
push_all_results → flops_basis                0
build_hf_dataset → flops_basis column         0
models.json entries with n_non_emb            0
Yet src/evals/CLAUDE.md:728,737,748 and plan/small-to-large-predictivity-training-plan.md:281 state all of it as fact — the latter literally "Implemented as configs.flops_params()". Line 748, "Params: configs.flops_params (models.json), never parsed from the name", is actively misleading for anyone debugging the W&B push. Same treatment as milestone_iters would close it.

