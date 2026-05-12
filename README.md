# Signal-Aware Framework for Multilingual LM Evaluation

The objective of this project is to answer this research question:

> Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

## Overview

- **Pretraining**: Pretrain custom small multilingual models. 4 sizes (175M, 350M, 600M, 1B) × 3 data mixtures (FineWeb-Edu 30/60/90) × 3 seeds (28, 1797, 1904) = 36 models. Run only on the cluster.
- **Evaluation**: Evaluate HuggingFace model checkpoints on lm-evaluation-harness tasks. Results saved locally and pushed to W&B and to a public HF dataset ([`multilingual-snr/multilingual-snr-eval-results`](https://huggingface.co/datasets/multilingual-snr/multilingual-snr-eval-results)).
- **SNR**: Calculate signal, noise, SNR, decision accuracy, and (where applicable) scaling-law error for all the benchmarks. 22 SNR variants × 4 sizes × the 115–121 multilingual parent tasks.
- **Analysis**: Per-language and across-language correlation of SNR-vs-DA, plus subset-search to find which subtasks elevate SNR for each benchmark family. Train/test seed split to test that the framework's recommendations generalize.
- **Slides**: [`documents/`](documents/) — Slidev presentation summarizing methodology and findings.

## Answer to the research question

**Two-level recommendation:**

1. **Use `quartile_deviation` (or any dispersion-family variant) as the
   global default.** Mean Pearson r between log10(SNR) and decision
   accuracy is ~+0.30 across 12 languages, and the dispersion-cluster
   ranking is stable across seed splits (DA-ckpt r between train and
   test splits = +0.69).
2. **Use `multiblimp_<lang>` and `xstorycloze_<lang>` /
   `hellaswag_<lang>` as the per-language reliability anchors.** They
   dominate the SNR ranking in every language where they exist and
   have decision-accuracy ≈ 1.0 — they actually rank model variants
   in the way large-model evaluations do.

**The framework generalizes at the global-ranking level.** The
Spearman rank correlation between the train and test pools' global
variant orderings is **+0.84 (DA-size)** and **+0.90 (DA-ckpt)** — the
"which variants are good" question is stable across seed pools. But
the *exact* per-language argmax changes (only 1/14 languages keep the
same pick), so per-language tuning that beats the dispersion baseline
should be treated as overfitting until validated on at least one more
seed.

**Benchmarks to de-prioritise:** `xnli_<lang>` rows often have high
SNR but DA-size = 0 (perfect rank disagreement with the 1B target),
so they're misleading reliability signals. `mgsm_direct` is currently
broken in the parquet (NaN scores).

See [`src/signal-and-noise/results/snr_definition/README.md`](src/signal-and-noise/results/snr_definition/README.md)
for the full per-language tables and the train/test split summary.

## Project structure

- [`configs/`](configs/) — `tasks.json` and `models.json` define what to evaluate
- [`documents/`](documents/) — reports and slides presenting the motivation and progress of the project
- [`src/`](src/) — core logic (config loading, model pretraining, evaluation, analysis)
- [`scripts/`](scripts/) — thin runner wrappers (local + SLURM)
- [`results/`](results/) — local output (gitignored)
- [`preliminary-analysis/`](preliminary-analysis/) — code and report from a preliminary analysis of the framework

## Key results dirs

Each analysis is partitioned by Apertus seed pool: **`seeds_1904`**
(single-seed test), **`seeds_28_1797`** (held-out train), and
**`seeds_28_1797_1904`** (pooled all seeds, recommended for downstream
work).

- [`src/signal-and-noise/results/snr_definition/seeds_<pool>/`](src/signal-and-noise/results/snr_definition/)
  — per-task SNR/DA tables + Q1–Q4 plots (best variant per language,
  top variants overall, top benchmarks per language)
- [`src/signal-and-noise/results/snr_definition/seeds_28_1797__vs__seeds_1904/`](src/signal-and-noise/results/snr_definition/seeds_28_1797__vs__seeds_1904/)
  — framework-generalization summary across train/test seed pools
  (agreement metrics, retention, Spearman ρ on variant ranking,
  scatter)
- [`src/signal-and-noise/results/smooth_subtasks/seeds_<pool>/`](src/signal-and-noise/results/smooth_subtasks/)
  — subset-search outputs (per-benchmark + per-(lang, subject) GMF +
  per_sample under [per_sample/](src/signal-and-noise/results/smooth_subtasks/per_sample/))
- [`src/signal-and-noise/results/allenai_comparison/seeds_<pool>/`](src/signal-and-noise/results/allenai_comparison/)
  — cross-corpus transfer of SNR rankings to AllenAI DataDecide
- [`src/signal-and-noise/results/benchmark_creation/seeds_<pool>/`](src/signal-and-noise/results/benchmark_creation/)
  — per-family SNR vs benchmark-design metadata (curation, format,
  option count, item length)
- [`src/signal-and-noise/results/acc_vs_flops/seeds_<seed>/`](src/signal-and-noise/results/acc_vs_flops/)
  — training curves (acc vs FLOPs) for each individual seed
