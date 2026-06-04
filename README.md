# Signal-Aware Framework for Multilingual LM Evaluation

> Which (subsets of) benchmarks provide reliable signal at each stage of
> multilingual model training?

This project extends the Signal-and-Noise framework (Heineman et al.,
2025) from the original English DataDecide / OLMo ladder to a
controlled multilingual sweep: 4 sizes × 3 data mixtures × 3 seeds = 36
small Apertus pretrains, evaluated on 80+ multilingual tasks, then run
through the full SNR / decision-accuracy stack.

## Answer to the research question

**Two-level recommendation:**

1. **Use a dispersion-family variant as the global default.** `dist_std`
   leads on DA-size (mean Pearson r between log10(SNR) and decision
   accuracy ≈ **+0.32**, far ahead of the field), while the
   mean-pairwise-distance / relative-spread cluster (`rel_mpd` / `mpd` /
   `mpsd`) leads on DA-ckpt (≈ **+0.51**) — all dispersion-family, so
   recommend the *family*, not an exact variant. Never `tukey` /
   `projection` (anti-correlated, r ≤ 0).
2. **Use `multiblimp_<lang>` as the per-language reliability anchor**,
   with `xwinograd` / `xcopa` recurring. Under the chosen definition
   (`dist_std` @ 1B) `multiblimp` is the highest-SNR above-random
   benchmark in **7 of 11 languages**, with checkpoint-DA ≈ 0.85 — it
   ranks model variants the way a larger-model evaluation would.

**The framework generalizes at the global-ranking level.** Spearman rank
correlation between the train and test pools' global variant orderings
is **+0.80 (DA-size)** and **+0.93 (DA-ckpt)** — *which* variants are
good is stable across seed pools. But the *exact* per-language argmax
changes (0–1/14 languages keep the same pick), so per-language tuning
that beats the dispersion baseline should be treated as overfitting
until validated on at least one more seed.

**Cross-corpus check.** On the seven English benchmarks shared with the
AllenAI DataDecide ladder, the pure 3-seed Apertus SNR correlates with
AllenAI SNR at **Pearson r = 0.92** (Spearman ρ = 0.93) for the
discrepancy / dispersion family — the dispersion + discrepancy variants
transfer, the relative-spread family does not. With only 7 shared tasks,
top-K set overlap is uninformative (Jaccard ≡ 1.0 for K ≥ 7); the
correlation is the result. (Folding in the >1B external models is *not*
the right comparison: the above-random gate drops the at-chance MCQA,
shrinking the shared set to 4.)

**What predicts SNR.** The **answer-count penalty is enforced upstream by
the above-random gate** — every 4-option translated knowledge MCQA
(`belebele`, `global_mmlu_full`, `truthfulqa`) sits at chance and is
dropped before SNR is computed, leaving 9 mostly-2-option survivors.
Among those survivors no single design feature is individually
significant (family-level Kruskal-Wallis on n_options H = 1.8, p = 0.18;
format H = 0, p = 1.0 — too little variation left), and **curation method
(MT / human / template) does not predict SNR** (H = 0.5, p = 0.78). The
mechanism still holds qualitatively (fewer options → sharper per-item
log-likelihood), but the within-survivor test is underpowered at n = 9.

**What subsets elevate SNR.** Best subsets substantially beat full sets
on multilingual benchmarks: Belebele 350M `+1.16` SNR with a 3-language
subset; Global-MMLU 175M `+1.52` with `medical_genetics` alone;
per-language Global-MMLU-tr 1B `+1.56`. Subject picks are highly stable
across seed pools; language picks are partially stable; per-(language,
subject) picks are pool-sensitive.

**Benchmarks to de-prioritise.** `xnli_<lang>` rows often have high SNR
but DA-size = 0 — perfect rank disagreement with the 1B target. High
SNR there is misleading. `mgsm_direct` is currently broken in the
parquet (NaN scores).

The full per-language tables and seed-split summaries live in
[`src/signal-and-noise/results/`](src/signal-and-noise/results/) — one
subdir per research question.

## The pipeline in three sections

### 1. [Pretraining](src/pretrain/) — build the model pool

- **36 small multilingual Apertus models**: 4 sizes (175M, 350M, 600M,
  1B) × 3 data mixtures of FineWeb-Edu + FineWeb2-HQ (30/70, 60/40,
  90/10) × 3 seeds (28, 1797, 1904). All trained to iter 50 000
  (~100 B tokens at GBS 504 × seq 4096).
- Per-size cluster cost: 175M ~11 h, 350M ~7.8 h, 600M ~7.2 h, 1B ~9.9 h.
- The canonical entry point is the idempotent
  [`launch_resumes.sh`](src/pretrain/launch_resumes.sh) — drives the
  full sweep to 100% canonical coverage.

### 2. [Evaluation](src/evals/) — measure them

- Cluster-side SLURM pipeline built on
  [`lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness)
  with W&B integration.
- 86 tasks per checkpoint (per-language multilingual + standalone
  English benchmarks); 10 evenly-spaced + 4 dense-tail canonical iters
  per model.
- Results saved locally to `eval_logs/`, pushed to the
  [`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments)
  W&B project, and packaged into the public HF dataset
  [`multilingual-snr/multilingual-snr-eval-results`](https://huggingface.co/datasets/multilingual-snr/multilingual-snr-eval-results).
- The SNR experiments are organised into three stages
  ([pretraining](src/evals/configs/signal_to_ratio/), midtraining,
  posttraining) with separate runners + idempotent re-launch.

### 3. [Signal & Noise](src/signal-and-noise/) — analyse them

Four self-contained reports, one per research question. Each is read
results-first; the per-pool subdirs hold the CSVs and PNGs.

| Report | Question |
|---|---|
| [`results/snr_definition/`](src/signal-and-noise/results/snr_definition/) | Which SNR variant best correlates with decision accuracy across languages? Does the choice generalize across seeds? |
| [`results/benchmark_creation/`](src/signal-and-noise/results/benchmark_creation/) | What benchmark design features (curation, format, option count, item length) predict SNR? |
| [`results/allenai_comparison/`](src/signal-and-noise/results/allenai_comparison/) | Do our SNR rankings agree with AllenAI DataDecide on shared English tasks? |
| [`results/smooth_subtasks/`](src/signal-and-noise/results/smooth_subtasks/) | Per benchmark, can a language or MMLU-subject subset elevate SNR and DA? |

Each analysis is partitioned by Apertus seed pool: **`seeds_1904`**
(single-seed test), **`seeds_28_1797`** (held-out train), and
**`seeds_28_1797_1904`** (pooled all seeds, recommended for downstream
work). The `seeds_28_1797__vs__seeds_1904/` subdir under `snr_definition/`
holds the train/test framework-generalization summary.

## Project structure

- [`configs/`](configs/) — `tasks.json` and `models.json` define what to evaluate
- [`documents/`](documents/) — reports and slides
- [`src/`](src/) — core logic (pretrain, evaluate, analyse)
- [`scripts/`](scripts/) — thin runner wrappers (local + SLURM)
- [`results/`](results/) — local output (gitignored)
- [`preliminary-analysis/`](preliminary-analysis/) — early-version code +
  report, kept for reference
