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

1. **Use `quartile_deviation` (or any dispersion-family variant) as the
   global default.** Mean Pearson r between log10(SNR) and decision
   accuracy is ~+0.34 across 12 languages, and the dispersion-cluster
   ranking is stable across seed splits (DA-ckpt r between train and
   test pools = +0.70).
2. **Use `multiblimp_<lang>`, `xstorycloze_<lang>` and
   `hellaswag_<lang>` as the per-language reliability anchors.** They
   dominate the SNR ranking in every language where they exist and have
   decision-accuracy ≈ 1.0 — they rank model variants in the way large-
   model evaluations do.

**The framework generalizes at the global-ranking level.** Spearman rank
correlation between the train and test pools' global variant orderings
is **+0.83 (DA-size)** and **+0.91 (DA-ckpt)** — *which* variants are
good is stable across seed pools. But the *exact* per-language argmax
changes (only 1/14 languages keep the same pick), so per-language
tuning that beats the dispersion baseline should be treated as
overfitting until validated on at least one more seed.

**Cross-corpus check.** On the seven English benchmarks shared with the
AllenAI DataDecide ladder, the pooled Apertus SNR (9 model_families per
size) correlates with AllenAI SNR at **Pearson r = 0.935** for the
discrepancy family. Top-10 reliable benchmarks agree by Jaccard 1.0:
`arc_challenge`, `arc_easy`, `csqa`, `hellaswag`, `mmlu`, `openbookqa`,
`piqa`.

**What predicts SNR.** Task design — primarily the **number of answer
options** — explains most of what we can explain. Per-family Spearman
ρ = +0.77 against the random baseline (p = 0.006); family-level
Kruskal-Wallis on n_options is H = 5.5, p = 0.019. Curation method (MT /
human / template) does not predict SNR (H = 1.44, p = 0.49). Option
length follows the same qualitative mechanism (longer options →
sharper per-item log-likelihood) but doesn't reach significance at the
per-family level on n = 11.

**What subsets elevate SNR.** Best subsets substantially beat full sets
on multilingual benchmarks: Belebele 350M `+0.89` SNR with a 4-language
subset; Global-MMLU 175M `+0.96` with `international_law` alone. Subject
picks are highly stable across seed pools; language picks are partially
stable; per-(language, subject) picks are pool-sensitive.

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
