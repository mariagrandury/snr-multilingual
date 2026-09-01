# CLAUDE.md

## Working rules

- **Never `rm -rf`, `mv`, or otherwise delete/relocate files or
  directories without first asking the user for permission.** Even if
  files look orphaned, redundant, or "obviously stale," ask first —
  the user may be intentionally keeping them, or moving them may
  break tracked paths. This applies to all destructive shell
  operations (`rm`, `mv` to a different parent, `git rm`, etc.).

- **Reuse existing code aggressively; keep new code simple and
  boilerplate-free.** Before writing a new helper, grep the repo for
  one that already does the job (e.g. `get_slice`,
  `signal_to_noise_ratio`, `decision_acc_fast`, `_is_language_aggregate`,
  the loader helpers in `snr/download/`, the shared CLI patterns in
  `analysis/`). Don't reimplement, don't wrap-for-wrap's-sake, and
  don't add defensive scaffolding ("just in case" config flags,
  pre-validation of arguments that won't be wrong, try/except around
  pure-Python logic). When extending a script, the new diff should
  read like a small addition, not a rewrite. Prefer one direct call
  over a chain of pass-through helpers.

- **Review before every commit.** When asked to commit — or to review
  pending work — run the `review-snr` skill first (`/review-snr`,
  `.claude-shared/skills/review-snr/SKILL.md`), report, then print the
  proposed commit message and wait for approval. Never `git commit` or
  `git push` unprompted. This is enforced, not just asked:
  `.claude-shared/hooks/review-snr-gate.sh` runs as a PreToolUse hook on
  Bash (user settings, `~/.claude/settings.json`) and refuses any
  `git commit` aimed at this repo unless the pending tree matches the
  fingerprint the skill wrote with `--mark` at the end of its last run.
  If it refuses, run the review — do not look for a way around it.
  Everything lives in `.claude-shared/` because `.claude/` is gitignored;
  a fresh clone or machine needs the discovery symlinks once:
  `mkdir -p .claude/skills ~/.claude/skills && ln -sfn ../../.claude-shared/skills/review-snr .claude/skills/review-snr && ln -sfn "$PWD/.claude-shared/skills/review-snr" ~/.claude/skills/review-snr`
  (the user-level link is what makes `/review-snr` work from the parent
  directory sessions are usually started in), plus the hook block in
  `~/.claude/settings.json`.

## Project Overview

This project extends the Signal-and-Noise (SNR) framework (Heineman et al., 2025) from English-only to multilingual settings.

**Research question:** Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

Two sweeps, in this order:

1. **The finished 36-model sweep** (4 sizes × 3 data mixtures × 3 seeds,
   `apertus-*`, W&B project `snr-experiments`) — done; its tooling evolved
   in place into the predictivity scripts.
2. **The predictivity sweep** (current work): a 6-rung ladder
   90M–1.7B × 7 language settings × deep/shallow × data schemes A/B, run
   across CSCS and Azure. Cells are named `lm-*` and log to W&B project
   **`msnr`**. Design: [`plan/small-to-large-predictivity-training-plan.md`](plan/small-to-large-predictivity-training-plan.md).

Read [`src/pretrain/CLAUDE.md`](src/pretrain/CLAUDE.md) and
[`src/evals/CLAUDE.md`](src/evals/CLAUDE.md) before touching either side —
they carry the failure modes, and they are more current than this file.

## Architecture

```
configs/          # tasks.json, models.json, languages.json, hf_wandb.json
documents/        # Slidev presentation (scholarly theme) + project documents
plan/             # the sweep design + compute budget (the planning docs)
scripts/          # build_configs.py, lint_models_json.py, grant_collaborator.sh
src/
  evals/          # evaluation harness wrapper (lm_eval integration)
  pretrain/       # the predictivity sweep: launchers, data build, auto-evals
  signal-and-noise/  # Allen AI reference implementation + its analysis
```

Eval results are NOT in the repo: they live on the cluster at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/`
(and checkpoints under `.../Meg-Runs/msnr/`).

## Key Concepts

- **Signal**: Relative dispersion of benchmark scores across training data mixtures (how well a benchmark separates models): `(max - min) / mean`
- **Noise**: Two variants:
  - Checkpoint noise: Score variability across late training checkpoints (requires many checkpoints)
  - Benchmark noise: k-fold split variability from a single evaluation run (more practical)
- **SNR**: Signal / Noise — higher means more reliable benchmark
- **Decision Accuracy**: Does the benchmark correctly rank model pairs? Pairwise ranking agreement between small and large models
- **Scaling-Law Error**: Can small model results predict large model performance?

## External Dependencies

- **lm_eval** (lm-evaluation-harness): the swiss-ai fork, installed per eval job
- **signal-and-noise** (Allen AI): reference implementation at `src/signal-and-noise/`
- **wandb**: entity `mariagrandury-epflnlp` (constant in `megatron_args.sh`);
  project comes from `configs/hf_wandb.json` — **`msnr`** for the predictivity
  sweep, `snr-experiments` only for the legacy 36-model infra in `src/evals/`

## Configuration

- `configs/tasks.json`: task lists by stage, plus the `auto` benchmark group
  the watchers evaluate
- `configs/models.json`: one entry per cell — generated by
  `src/pretrain/sync_models_json.py`, not hand-edited
- `configs/hf_wandb.json`: HF org + W&B project
- `configs/languages.json`: the FineWeb→ISO code table the task/language
  matching keys off
- Architectures live in `src/pretrain/hyperparams/hyperparams_{deep,shallow}.json`

## Development

The real entry points are documented in the two sub-READMEs
([`src/pretrain/README.md`](src/pretrain/README.md),
[`src/evals/README.md`](src/evals/README.md)). The short version, all from the
cluster login node with the `snr` conda env:

```bash
pip install -r requirements-ml.txt

# Pretraining: idempotent — re-run to drive the sweep forward
python3.11 src/pretrain/launch_trainings.py cscs --dry-run
python3.11 src/pretrain/pretrain_progress.py --plot     # status + heatmaps
python3.11 src/pretrain/data/data_progress.py           # data-mixture coverage

# Evals: the watcher converts + evaluates due checkpoints and pushes to W&B
python3.11 src/pretrain/auto_evals_cscs.py --dry-run

# Slides
cd documents && npx slidev --open
```

System Python on the login nodes is 3.6 — use `python3.11`.

## Conventions

- Source modules (`src/`) contain core logic with no CLI parsing
- Scripts (`scripts/`) are thin wrappers with argparse and minimal logic
- W&B logging is optional (--no-wandb flag) for local development
- Results are saved locally AND to W&B when enabled
- Checkpoint resolution supports: last N, total T (evenly spaced), or named list
- Presentation uses Slidev with `slidev-theme-scholarly`, config in `documents/`

Predictivity-sweep specifics (the 36-sweep's sizes and 30/70-style mixtures
are retired — do not carry them into new work):

- Sizes: 90M, 175M, 350M, 600M, 1B, 1.7B non-embedding
- Data: fixed 50/50 English (DCLM) + FineWeb-2, with L ∈ {1, 2, 8, 15, 30, 50,
  100} languages; L=1 is 100% English. The mixture varies the language *count*,
  not the English ratio.
- Cell name = Slurm job name = checkpoint dir = W&B run name:
  `lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>`
- Each size trains its own budget D(N) = 100 × N tokens (5× Chinchilla)
