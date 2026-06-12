# Pretraining SNR analysis — new results vs previous

Comparison of the extended pipeline against the previous committed results
(git HEAD, `results/snr_definition/seeds_*`). Supersedes the earlier draft at
`results/pretraining/ANALYSIS_new_vs_previous.md`.

## Output layout (new)

`results/<analysis>/<stage>/<pool>/`, e.g.
`results/snr_definition/pretraining/custom_swissai_hf/`,
`results/acc_vs_flops/pretraining/seeds_28_1797_1904/`,
`results/benchmark_creation/pretraining/<pool>/`,
`results/allenai_comparison/pretraining/<pool>/`,
`results/smooth_subtasks/pretraining/<pool>/`.

**Model-set tiers** (pools): pure custom `seeds_1904` (1 seed), `seeds_28_1797`
(2 seeds), `seeds_28_1797_1904` (3 seeds), and `custom_swissai_hf` (3 seeds +
all external pretraining-checkpoint models: swiss-ai + HF references + a06 +
distillation). The previous `seeds_*` pools all folded externals, so the
closest old↔new pair is **`seeds_28_1797_1904` (old) ↔ `custom_swissai_hf`
(new)**.

## What changed (inputs + method)

- **More data**: folded in apertus3 a06 (1B, 3B) + distillation (600M, 1B) +
  expanded reference_hf; re-fetched the latest splits.
- **Instruct excluded**: reference_hf now ships instruct variants
  (gemma-3-*-it, OLMo-2-Instruct/SFT/DPO, Qwen3 non-Base, Apertus-8B-Instruct,
  aya). The external fold is now restricted to models declared at the
  **pretraining** stage, so 29 base/pretrained external models join the pool
  and no instruct checkpoint leaks in.
- **Size buckets (>1B)**: signal/noise + DA size axis pools nearby large sizes
  (`0.6B≡600M`, `7-9B`, `12-14B`, `27-32B`, …); the pool now spans 175M→32B.
- **Relative-fraction ckpt-DA**: early ckpts chosen per-model at ~12/36/56% of
  each model's own max step, so external multi-ckpt trajectories (a06,
  distillation, SmolLM3-checkpoints, Olmo-3, Apertus-8B) enter ckpt-DA — new
  ckpt-DA at 3B and 7-9B.
- **Subject-subtask aggregation everywhere except the per-subtask analysis**:
  acc-vs-flops now keeps only parent tasks (mmlu_anatomy→mmlu, agieval/
  arabic_leaderboard subjects collapsed; languages stay distinct), matching the
  variant pipeline. Task count drops 950→118.
- **acc-vs-flops top-N**: all per-task **Signal** ((max−min)/mean of per-mix
  final scores at the target size) is written to `acc_vs_flops_signal.csv`;
  only the top-3 benchmark families by Signal are plotted (307→3 per-benchmark
  grids).

## RQ1 — best SNR variant vs decision accuracy

Mean Pearson r across languages of log10(SNR) vs DA, top variant per pool:

| pool | top variant | DA-size r | DA-ckpt r | overall r |
|---|---|---:|---:|---:|
| **previous** `seeds_28_1797_1904` (ref-only externals, old defs) | `rel_mpd` | 0.405 | 0.362 | 0.383 |
| **new** `seeds_28_1797_1904` (pure custom, no externals) | `rel_mpd` | 0.392 | 0.379 | 0.386 |
| **new** `custom_swissai_hf` (+ a06/distill/swissai/HF, instruct-excluded) | `rel_mpd` | 0.400 | **0.519** | **0.460** |

- **Headline answer unchanged**: `rel_mpd` (and the dispersion / relative-spread
  family: `rel_std`, `iqr`, `quartile_deviation`, `aad`, `mpd`) is still the
  most DA-correlated SNR variant in every tier.
- **Externals mainly lift DA-ckpt**: at 3 seeds, adding the external
  pretraining models raises DA-ckpt r **0.379 → 0.519** and overall **0.386 →
  0.460**, while DA-size barely moves (0.392 → 0.400). The gain is the
  relative-fraction ckpt-DA now drawing on the external multi-checkpoint
  trajectories.
- vs the **previous** comprehensive result, DA-ckpt is up (0.362 → 0.519) and
  overall up (0.383 → 0.460); DA-size is essentially flat (0.405 → 0.400, the
  small dip from dropping instruct models that previously padded the pool).
- **Dose-response in seeds** (pure tiers): top DA-size r climbs 0.31 (1 seed) →
  0.33 (2 seeds) → 0.39 (3 seeds); the best variant shifts from the absolute
  dispersion cluster (`mpd`, tied, at 1 seed) to `rel_mpd` by 2–3 seeds.

### Seed generalization (holdout): does a 2-seed definition hold on the 3rd?

`seeds_28_1797 → seeds_1904` (held-out seed):

| | DA-size | DA-ckpt |
|---|---:|---:|
| **Spearman ρ on global variant ranking** | **+0.797** | **+0.925** |
| Pearson r between splits (all cells) | +0.569 | +0.725 |
| Family-level best-variant agreement | 14% | 36% |
| Retention of train-best r | 62% | 78% |

**The variant ranking transfers strongly to the held-out seed** (ρ ≈ 0.80
DA-size, 0.93 DA-ckpt). Exact per-language argmax flips (the dispersion family
is near-degenerate at n_mixes=3), so the seed-robust recommendation is at the
*family* level: "use a dispersion / relative-spread variant", and DA-ckpt is
the more portable axis.

## RQ4 — subtask subsets

`smooth_subtasks` reran for `seeds_28_1797_1904` and `custom_swissai_hf` over
the bucketed pool; per-benchmark and global-MMLU subject-subset CSVs refreshed
under `results/smooth_subtasks/pretraining/<pool>/`. Per-item (Option-D)
analysis reran on the committed intermediates (320 cells / 80 benchmarks);
per-sample raw logs remain cluster-only.

## Scaling (>1B)

- **acc-vs-flops** overlays every external pretraining model's final-ckpt
  (FLOPs, score) on the custom per-mix curves, extending the compute axis to
  70B; only the top-3 highest-Signal families are drawn (full Signal table in
  `acc_vs_flops_signal.csv`). Current top-3 by Signal: `agieval_sat`,
  `belebele`, `arabic_leaderboard_alghafa_mcq_exams_test`.
- **Scaling-DA** auto-detects cross-bucket pairs with ≥2 shared families
  (e.g. `4B→12-14B`, `7-9B→27-32B`); computable but sparse (≤2 families/pair),
  so it's directional, not yet statistically strong. Cross-size DA above 1B
  stays family-coverage-limited.

## RQ2 / RQ3

Both reran per tier against the new CSVs (paths now
`results/{benchmark_creation,allenai_comparison}/pretraining/<pool>/`):
- RQ2 (`benchmark_creation`): per-family SNR + length/format-feature
  correlations.
- RQ3 (`allenai_comparison`): Apertus 120 tasks vs AllenAI 241 — 7 shared
  English tasks; agreement tables refreshed.

## Bottom line

Adding the new models (and excluding instruct contamination) **keeps the
qualitative answer** — `rel_mpd` / the dispersion family — while **sharpening
the evidence**: DA-ckpt correlation up ~40–55%, the signal pool now spans
175M→32B, ckpt-DA now exists at 3B/7-9B, acc-vs-flops is parent-aggregated and
top-N (fast), and the variant ranking is shown to hold on a fully held-out seed
(Spearman ρ up to 0.93). Cross-size DA above 1B is the one dimension still
limited by external-family coverage.
