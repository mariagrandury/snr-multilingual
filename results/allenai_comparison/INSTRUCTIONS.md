# INSTRUCTIONS — `allenai_comparison`

Owner: one Claude session. Don't write outside this directory.

## Research question

Do **our** "best" SNR definitions correlate with the **AllenAI**
results? Concretely: taking each best-per-language definition from
Question 1, do the SAME benchmarks come out as "reliable" when we
recompute SNR on the AllenAI DataDecide / OLMo data?

This is a transfer-of-conclusions check: a benchmark deemed reliable
on Apertus pretraining experiments should also be reliable on the
DataDecide ladder.

## Inputs

- **Apertus side** — `results/snr_definition/snr_variants_per_task.csv`
  (must have been produced by Question 1's session — depends on it).
- **AllenAI side** — pulled with the upstream code:
  ```python
  from snr.download.hf import pull_predictions_from_hf
  path = pull_predictions_from_hf("allenai/signal-and-noise", split_name="core")
  df_allen = pd.read_parquet(path)
  ```
  This requires an internet connection. The dataset is public.
- `reference_hf-*.parquet` — load with
  `load_reference_hf_eval_results()`. Use as a **third corpus**: for
  each shared task, score = primary_score at the model's max step.
  Then we have three per-task vectors (Apertus-1B, AllenAI-750M,
  reference_hf-{Olmo3-7B, Apertus-8B, SmolLM3-3B}). The
  rank-agreement test runs pairwise across all three, giving more
  data points than the binary Apertus↔AllenAI comparison.

## Existing code (REUSE)

- [snr/snr_simple.py](../../snr/snr_simple.py) — upstream SNR pipeline
  (`compute_decision_accuracy`, `compute_snr_small_scale`,
  `calculate_results`). Use this on the AllenAI DataFrame to get a
  reference per-task SNR table at small sizes (150M/300M/750M) and
  decision accuracy targeted at 1B / 7B / 13B.
- [snr/snr_variants.py](../../snr/snr_variants.py) — same 22 variants
  list. Re-run `run_apertus_snr_variants.py`'s logic but feeding the
  AllenAI df. Easiest path: extract the per-task variant computation
  into a helper that takes `df, sizes` and returns the same wide CSV
  shape — then call it once for Apertus, once for AllenAI.

## Plan

### Step 1 — produce an AllenAI variants CSV

Reuse `multilingual/run_apertus_snr_variants.py` as a template. Write
`results/allenai_comparison/build_allenai_variants.py`:

- Load AllenAI parquet via `pull_predictions_from_hf`.
- Pick small sizes from
  `snr.constants.signal.SNR_MODELS` matching the upstream "core" split
  (150M / 300M / 750M with target = 1B is the conventional setup).
- Run the same variant loop as in `run_apertus_snr_variants.py`, with
  AllenAI's mix names (DataDecide names many).
- Write `allenai_snr_variants_per_task.csv` to this directory.

### Step 2 — task-name reconciliation

Apertus task names and AllenAI task names overlap on the English
core (`mmlu`, `arc_challenge`, `arc_easy`, `hellaswag`, `piqa`,
`openbookqa`, `commonsense_qa`, `truthfulqa_mc1`) and diverge on the
multilingual variants (AllenAI doesn't have `arc_de` etc.).

- Build `task_overlap.csv`: columns `task, in_apertus, in_allenai`.
- Restrict the comparison to the intersection.

### Step 3 — Pearson r between corpora

For each shared task and each variant V:
- `snr_apertus = log10(snr_<V>_<largest_apertus_size>)` (default 1B).
- `snr_allenai = log10(snr_<V>_<largest_allenai_small_size>)` (default
  750M, since AllenAI's "small" tops out at 750M before the 1B
  target).

Compute Pearson r across the shared-tasks vector. Write
`pearson_r_per_variant.csv` with columns `variant, n_shared_tasks, r`.

Render a scatter `snr_apertus_vs_snr_allenai_<best_variant>.png`
(one panel) and a per-variant grid
`snr_apertus_vs_snr_allenai_grid.png` (rows = variants, sorted by r).

### Step 3b — fold in reference_hf as a third corpus

For each shared task, take `primary_score` at each reference_hf
model's max step → 3 extra columns
(`refhf_olmo3_7b`, `refhf_apertus_8b`, `refhf_smollm3_3b`).

- Pairwise Pearson r between Apertus-SNR / AllenAI-SNR / each refhf
  rank → write `pairwise_corpus_pearson_r.csv` (rows = (corpus_a,
  corpus_b, variant), cell = r).
- Render a 3×3 (or 5×5 if you also include SmolLM3-Base which is 1
  ckpt) correlation heatmap
  `corpus_correlation_heatmap.png`.

Skip if reference_hf adds no new info beyond Apertus — note the
decision in the README.

### Step 4 — "reliable" benchmark agreement

Define a benchmark as **reliable** if its SNR (under the best
variant) is in the top-K (K=10) within a corpus.

- Build a 2-column ranked list:
  - `top_apertus.csv`: top 10 tasks by `snr_<V>_1B` from
    `results/snr_definition/snr_variants_per_task.csv`.
  - `top_allenai.csv`: top 10 by `snr_<V>_750M` from
    `allenai_snr_variants_per_task.csv`.
- `agreement.md`: tabulate intersection / Jaccard for K = 5, 10, 20.

### Step 5 — README

Write `results/allenai_comparison/README.md`:

- Research question (verbatim).
- Setup: 3-4 sentences naming both data sources, the variant choice,
  the "shared task" definition, and the K-cutoff agreement metric.
- Main results: embed `snr_apertus_vs_snr_allenai_<V>.png` and a
  short table summarising `pearson_r_per_variant.csv` (top + bottom
  3 variants).
- Headline finding: list the benchmarks that appear in **both**
  top-10s — those are the cross-corpus-reliable ones.
- Directory contents.

## Notes / gotchas

- AllenAI uses `model_path` and a different mix vocabulary than
  Apertus. `compute_snr_small_scale` works because it groups by
  `mix` regardless of names — but make sure the input df has 5+
  ckpts per mix (DataDecide does).
- The DataDecide parquet is large (~GB). `pull_predictions_from_hf`
  caches under `analysis/data/` (or `SNR_DATA_DIR`).
- `snr.snr_variants` aggregator inputs are `(step_noise, data_scores,
  data_noise, data_scores_last_n)` — same shape we already feed for
  Apertus; reuse `per_mix_inputs` from
  `multilingual/run_apertus_snr_variants.py`.
- Don't re-run `compute_scaling_law_error` — that requires
  `olmo-ladder` and won't work with our environment.

## Definition of done

- `allenai_snr_variants_per_task.csv` produced.
- `task_overlap.csv` and `pearson_r_per_variant.csv` produced.
- Headline scatter PNG produced.
- `top_apertus.csv` / `top_allenai.csv` + `agreement.md`.
- `README.md` embeds the headline plot and the agreement table, and
  lists every file in the directory.
