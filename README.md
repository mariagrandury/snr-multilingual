# Signal-Aware Framework for Multilingual LM Evaluation

The objective of this project is to answer this research question:

> Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

## Overview

- Pretraining: Pretrain custom small multilingual models. Run only on the cluster.
- Evaluation: Evaluate HuggingFace model checkpoints on lm-evaluation-harness tasks. Results saved locally and pushed to W&B.
- SNR: Calculate signal, noise, SNR, decision accuracy, and scaling-law error for all the benchmarks. Results saved locally and pushed to W&B.
- Analysis: Perform statistical analyses and generate visuals to understand the results and be able to make recommendations on which benchmarks (or subsets of benchmarks) to evaluate at each training stage.
- Slides: Document the methodology and results in clear slides to present our work to the research community.

## Project structure

- `configs/`: tasks.json and models.json define what to evaluate
- `documents/`: reports and slides presenting the motivation and progress of the project
- `src/`: core logic (config loading, model pretraining, evaluation, analysis)
- `scripts/`: thin runner wrappers (local + SLURM)
- `results/`: local output (gitignored)
- `preliminary-analysis/`: code and report from a preliminary analysis of the framework
