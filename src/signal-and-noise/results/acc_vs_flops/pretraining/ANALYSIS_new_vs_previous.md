# Pretraining SNR analysis — new results vs previous

Comparison of the extended pipeline (this run) against the previous committed
results (git HEAD, `results/snr_definition/seeds_*`). Generated 2026-06-02.

## What changed (inputs + method)

**More data folded in.** Previously the only external models in the SNR
signal/noise + decision-accuracy (DA) pool were the `reference_hf` rows. Now
the pool also folds in the **apertus3 a06** (1B, 3B) and **distillation**
(`ap-from8b`, 0.6B≡600M + 1B) checkpoint series, and the `reference_hf` split
itself expanded to 20 models (175M→70B). The distilled 0.6B run is the same
600M parameter scale as the custom 600M models (`0.6B→600M` alias), so it now
joins the **600M→1B** size-DA as a 4th family.

**Size buckets (>1B).** The signal/noise + DA size axis is now a *bucket*:
custom sizes stay singletons; nearby large sizes pool so each has ≥2 models.
Models per bucket in the comprehensive tier (`3seeds_swissai_hf`):

| 175M | 270M | 350M | 600M | 1B | 1.7B | 3B | 4B | 7-9B | 12-14B | 27-32B | 70B |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1 | 9 | 11 | 13 | 1 | 4 | 2 | 5 | 2 | 2 | 1 |

Buckets with ≥2 models (computable cross-model signal): 175M, 350M, 600M, 1B,
**3B, 4B, 7-9B, 12-14B, 27-32B** — the SNR signal pool now spans the full
ladder, not just ≤1B. Singletons (270M, 1.7B, 70B) contribute a final-score
point but are dropped from noise.

**Relative-fraction ckpt-DA.** ckpt-DA previously matched *absolute* custom
megatron iters (6000/18000/28000), which no external trajectory hits — so only
custom models ever entered ckpt-DA. It now selects each model's early
checkpoint at a *fraction* of its own max step (~12/36/56%), so the a06,
distillation, SmolLM3-checkpoints, Olmo-3 and Apertus-8B trajectories
participate. ckpt-DA now exists at the **3B (~88 tasks) and 7-9B (~87 tasks)**
buckets — entirely new.

**Model-set tiers + stage layout.** Outputs are now under
`results/pretraining/{snr_definition,acc_vs_flops,smooth_subtasks}/<tier>/`,
with four tiers: pure-custom `1seed`/`2seeds`/`3seeds` (no externals) and
`3seeds_swissai_hf` (all externals). The previous `seeds_*` pools all had
externals folded, so the closest old↔new pair is
**`seeds_28_1797_1904` (old) ↔ `3seeds_swissai_hf` (new)**.

## RQ1 — best SNR variant vs decision accuracy

Mean Pearson r across languages of log10(SNR) vs DA, top variant per pool:

| pool | top variant | DA-size r | DA-ckpt r | overall r |
|---|---|---:|---:|---:|
| **previous** `seeds_28_1797_1904` | `rel_mpd` | 0.405 | 0.362 | 0.383 |
| **new** `3seeds_swissai_hf` | `rel_mpd` | **0.418** | **0.555** | **0.487** |

- **The headline answer is unchanged and now stronger.** `rel_mpd`
  (relative mean pairwise distance) remains the single best variant; the
  dispersion/relative-spread family (`rel_mpd`, `rel_std`, `aad`,
  `quartile_deviation`, `mpd`, `iqr`) still occupies the top of the ranking.
- **DA-size r** rose modestly (0.405→0.418): the distilled family added a
  4th data point to 600M→1B (600M size-DA now covers 108/121 tasks, mean DA
  0.63).
- **DA-ckpt r jumped sharply (0.362→0.555, +53%)**, and overall r +27%
  (0.383→0.487). This is the largest effect of the new data: folding the
  external multi-checkpoint trajectories into a relative-fraction ckpt-DA gives
  the SNR↔DA-ckpt correlation far more (and more diverse) points to fit.

Pure-custom tiers (new, no externals) show the expected dose-response in
statistical power as seeds accumulate: top-variant DA-size r climbs
**0.31 (1seed) → 0.33 (2seeds) → 0.39 (3seeds)**, and the top variant shifts
from the absolute-dispersion cluster (`mpd`/`dispersion`/`range`, tied at 1
seed) to `rel_mpd` by 3 seeds — i.e. relative variants only reveal their edge
once there are enough runs to estimate dispersion stably.

### Seed generalization (holdout) — does a definition found on 2 seeds hold on the 3rd?

`compare_seed_splits.py`, train→test = held-out seed 1904:

| split | exact-variant | family-level | Pearson r (cells) | **Spearman ρ (global ranking)** | retention |
|---|---:|---:|---:|---:|---:|
| `2seeds`→`1seed`  (DA-size) | 0% | 14% | +0.569 | **+0.797** | 62% |
| `2seeds`→`1seed`  (DA-ckpt) | 7% | 36% | +0.725 | **+0.925** | 78% |
| `3seeds`→`1seed`  (DA-size) | 7% | 43% | +0.714 | **+0.746** | 72% |
| `3seeds`→`1seed`  (DA-ckpt) | 50% | 50% | +0.830 | **+0.940** | 87% |

**The variant *ranking* generalizes strongly to the held-out seed** (Spearman
ρ ≈ 0.75–0.94), and much better under DA-ckpt than DA-size. Exact per-language
best-variant identity transfers poorly (the dispersion-family members are
near-degenerate at n_mixes=3, so the argmax flips between near-ties), but
family-level agreement and retention are solid — i.e. "pick a
dispersion/relative-spread variant" is a seed-robust recommendation; "pick
exactly `rel_mpd` vs `mpd` in Swahili" is not.

## RQ4 — subtask / language / subject / per-item subsets

`smooth_subtasks.py` now runs over the bucketed pool (custom + externals).
The per-benchmark and global-MMLU subject-subset CSVs are regenerated for
`3seeds` and `3seeds_swissai_hf` under
`results/pretraining/smooth_subtasks/per_subtask/<tier>/`; the high-SNR-subset
story is preserved and now reported at the larger buckets too. Per-item
(Option-D) analysis (`analyze_per_sample_d.py`) reran on the committed
intermediates (320 cells / 80 benchmarks) — unchanged, since per-sample raw
logs are cluster-only and were not regenerated for the new models.

## Scaling (>1B)

- **acc-vs-flops** (`run_apertus.py --pool 3seeds_swissai_hf`) overlays every
  external model's final-ckpt (FLOPs, score) on the custom per-mix curves, so
  the compute axis now extends past the custom 1B ceiling to 70B.
- **Scaling-DA** auto-detected cross-bucket pairs with ≥2 shared families:
  `4B→12-14B` and `7-9B→27-32B` (both via the gemma-3 / OLMo within-family
  ladders). These are computable but **sparse**: only 2 families each, so DA
  saturates at 1.0 on the ~6 shared tasks — directionally useful, not yet
  statistically strong. Cross-bucket DA past 1B is limited by how few model
  families ship multiple sizes; acc-vs-flops + the signal pool carry the
  scaling story for now.

## RQ2 / RQ3 (downstream, refreshed)

Both re-ran against the new tier CSVs (paths now stage-aware):
- **RQ2** (`benchmark_creation`): per-family SNR + length/format-feature
  correlations regenerated for all tiers.
- **RQ3** (`allenai_comparison`): Apertus 120 tasks vs AllenAI 241; **7 shared
  English tasks** — agreement table regenerated. Shared-task coverage is
  unchanged (the new models add scale/power, not new English-benchmark
  overlap).

## Bottom line

Adding the apertus3, distilled, and expanded reference models **did not change
the qualitative answer** — `rel_mpd` and the dispersion/relative-spread family
remain the most DA-correlated SNR variants — but it **materially tightened the
evidence**: DA-ckpt correlation up >50%, the signal pool now spans 175M→32B
instead of ≤1B, ckpt-DA now exists at 3B/7-9B, and the variant ranking is shown
to hold on a fully held-out seed (Spearman ρ up to 0.94). The one place the new
data does *not* yet deliver is cross-size DA above 1B, which remains
family-coverage-limited.
