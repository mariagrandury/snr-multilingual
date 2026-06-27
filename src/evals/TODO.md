# Evals — open TODOs

Snapshot of outstanding work from the 2026-05/06 eval-scaling + DA-grid
sessions. Paths are absolute (`/iopsstor/...`) since commands run from
arbitrary cluster CWDs.

## DA-grid (decision-accuracy checkpoint selection)

Chosen **`da_ckpts = {10, 20, 30, 40, 50, 100}%`**, computed as **iter /
last_iter** (value-based, NOT index — a dense tail must not drag the 50% mark
late; e.g. if iter 100k is 100%, iter 50k is the 50% ckpt regardless of how
many late ckpts exist). For continuation stages, **0% = end of the previous
stage** (SmolLM3 midtraining: 0% = stage1 end @ iter 3,440,000 → da at
9/19/31/41/50/100%). `full_eval = da_ckpts ∪ dense_tail`. Applied to all 44
da-stages in `configs/models.json`. `10_ckpts` (W&B curve density) left
unchanged.

- [ ] **(P1) Custom-apertus new-grid checkpoints may not be loadable — VERIFY before re-launch.**
  The 60 new-grid `pretraining_full` jobs for the 36 custom models were
  launched then **cancelled (2026-06-01)**. `launch_pretraining_megatron`
  reported `skip_no_ckpt=0` (iter dirs exist), but a present `iter_NNNNNNN/`
  does NOT mean the `.distcp` shards survived the checkpoint **deletion
  sweeps** — the new da iters not in the old `full_eval` (the ~30%/18k-type
  iters) are most at risk. Verify each new iter dir has `.metadata` + ≥1
  `.distcp` (same validity test as
  `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/pretrain_progress.py`),
  then re-launch only surviving iters (or reconvert from HF). a06 (3) +
  distill (6) core jobs were left running; only the custom 60 were cancelled.
- [ ] **(P2) Reference-model trajectories — DA them or keep as `main` anchors?**
  Config now defines a da-grid for them but they were NOT launched (~31 jobs):
  `Olmo-3-1025-7B` step branches, `SmolLM3-3B-checkpoints` stage1,
  `Apertus-8B/70B-2509` `stepN-tokensXXX` branches. They're external comparison
  points, not the Apertus size-scaling ladder — decide whether to evaluate
  their full pretraining trajectories.
- [ ] **(P2) `Apertus-8B-2509` da-grid wrinkle.** Its `all` contains
  `stepN-tokensXXX` branches, so the 100% anchor became `step2627139-tokens15T`
  and **`main` dropped** from `full_eval`. The in-progress `Apertus-8B-2509-main`
  eval (79/120) is now off-grid. Decide: keep `main` as the anchor vs switch to
  the step-branch trajectory.
- [ ] **(P3) After inspecting SNR values, consider extending da to the upper
  half** (60/70/80/90%) — currently 50→100% is unanchored except the endpoints.

## Samples recovery (2026-06-01 scratch-cleanup incident)

CSCS `/iopsstor` scratch auto-cleaned per-task `samples_*.jsonl` files for
ckpts not accessed in ~30 days. Aggregate scores survived on the hub
(`multilingual-snr/multilingual-snr-eval-results`); per-instance predictions
did not. 126 NAMEs ended up with zero samples on disk; 41 of those
correspond to hub-published (model, revision) rows and need re-eval to
restore the per-sample data.

- [ ] **(P2) Re-eval 41 (model, ckpt) jobs to restore lost per-instance
  samples.** Full per-NAME list in
  [`samples-recovery-20260601.csv`](samples-recovery-20260601.csv) —
  columns: NAME, model, ckpt, n_tasks, per_task_s_median, est_minutes.
  Totals: 32,981 task-evals, ~1,550 GPU-hours upper bound (~500-800h
  realistic with `BATCH_TASKS=1` amortising model load). 9 of the 41 jobs
  (`apertus3-3b-64-nodes` iter15k-155k) dominate at ~700 GPU-hours — defer
  or batch separately if budget-constrained. The
  `Olmo-3-1025-7B@stage3-step11921` estimate (110h) is an outlier from a
  high median processing_time (mgsm/triviaqa skew); spot-check before
  committing the wall-time.
- [ ] **(P3) Prevent recurrence: keep canonical eval samples on
  `/capstor`, not `/iopsstor`.** Stage 1 already done (608 GB rsync'd to
  `/capstor/store/cscs/swissai/infra01/users/mariagrandury/snr-multilingual-eval-logs`).
  Stage 2 = symlink `/iopsstor/.../eval_logs` to the capstor path once
  in-flight jobs finish, so future evals write to capstor by default.

## Eval-launch infrastructure

- [ ] **(P3) `snr_progress.csv` is a single shared file, overwritten per-pool.**
  The `launch_pretraining_*.sh` `--no-refresh` flag reuses whatever pool's
  snapshot was written last — so `--no-refresh` across *different* pools reads
  stale rows and submits the WRONG pool's jobs (caused a duplicate distill
  submission on 2026-06-01). Fix: pool-named CSVs (`snr_progress-<pool>.csv`),
  or drop `--no-refresh`. Until then, never `--no-refresh` when switching pools.
- [ ] **(P2) Task-group-specific job names / eval-log dirs (the "#8" fix).**
  The eval NAME / Slurm job name is per-(model, ckpt) only:
  `eval-<model>-iter<N>`. The same ckpt evaluated on different task groups
  (e.g. a06/distill on `midtraining` vs `pretraining_full`) collides — the
  second launch sees the first as `active` and skips, and both write into the
  SAME `eval_logs/<NAME>/` dir. Fix: fold the task group (or a hash of the task
  set) into the job name + eval-log dir. Until then, run gap-fills only after
  colliding jobs finish.
- [ ] **(P3) a06/distill midtraining experiment is partial.** Some midtraining
  jobs were cancelled as off-grid during the da-grid change; reconcile with a
  midtraining re-launch once the job-name collision (above) is fixed.
- [ ] **(P3) Review the launcher refactor.**
  `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh`
  now dispatches on `checkpoint_kind`: `hf_branch` → "hub mode" (eval HF-Hub
  repo at `REVISION`, model's own tokenizer, chat-template from task group);
  else → "local mode" (converted iter dir). `scripts/snr_progress.py` selects
  ckpts by the pool's declared `stage`. Default `PP` for unknown sizes → 4
  (`TP=1 PP=4`). Works in dry-run + live; wants a proper review pass.

## Config generation (build_configs.py)

- [x] **`scripts/build_configs.py` synced to live + `eval_groups` added** (2026-06-01).
  It was a stale one-time bootstrap (94 models, old da-grid); now regenerates
  the full live `configs/models.json` (130 models, 8 pools, the `{10,20,30,40,50,100}%`
  da-grid as a computed rule) and stamps stage-level `eval_groups` (per-pool,
  default stage→group: pretraining→pretraining_full, mid→midtraining,
  post→posttraining). `Qwen3.5-*`/`gemma-4-*` excluded (no eval_groups). The
  posttraining-source instruct models were moved to `swiss-ai-reference`
  (Apertus-Instruct) / `huggingface-reference` (gemma-it, Ministral, Mistral)
  so `posttraining_hf_reference` covers them → 21 models now have no
  eval_groups (genuinely not evaluated).
- [x] **`tasks.json` mgsm expansion adopted (2026-06-01).** `build_configs.py`
  builds `tasks.json` FROM the `tasks_*.txt` source files; `tasks_posttraining.txt`
  already carries the mgsm expansion (`mgsm_{direct,en_cot,native_cot}_*`, 22
  mgsm tasks, posttraining group = 127). The stale 115 was only the old
  `tasks.json`. Regenerated and kept the expanded version, so the `.txt` source,
  `tasks.json`, and the launcher eval list are all consistent at 127.
  **`build_configs.py` is now fully safe to re-run end-to-end** (models.json +
  tasks.json both reproduce the on-disk state).

## New architectures (not yet evaluable)

- [ ] **(P2) `qwen3_5` + `gemma4` architecture versioning.** Qwen3.5-* (model
  type `qwen3_5`, multimodal) and gemma-4-* (`gemma4`, multimodal) are in
  `configs/models.json` but NOT launched — the eval container's
  transformers/vLLM likely doesn't recognize these brand-new archs (same
  hard-load-failure class as the Olmo-retrofit, CLAUDE bug #8). Sort out a
  container/transformers version that supports the archs, then smoke-test one
  of each with `--limit` before fan-out. (lm-eval *can* evaluate the text
  modality of a multimodal model, so modality isn't the blocker — arch
  registration is.)

## Done (for context)

- SmolLM3-3B-checkpoints: added stage2/stage3 midtraining subset lists +
  posttraining `it-*` stage; posttraining trajectory evaluated.
- OLMo-2 / Olmo-3 / gemma-3 / gemma-4 families added to `models.json`;
  OLMo instruct/think source set to `huggingface-reference` for pool
  consistency.
- Qwen3-4B / Qwen3-4B-Base repo typos fixed; Qwen3.5-4B-Base added.
- `midtraining_hf_reference` pool added.
