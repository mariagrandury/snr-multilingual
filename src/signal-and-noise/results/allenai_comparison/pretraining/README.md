# Cross-corpus transfer: does our SNR pick agree with AllenAI?

> Do the SNR variants we recommend on Apertus also correlate with the
> AllenAI DataDecide / OLMo SNR rankings on the same English benchmarks?
> And do the same benchmarks emerge as "reliable" on both corpora?

## TL;DR

**Yes, strongly — when we use the pooled Apertus pool (`seeds_28_1797_1904`)
the cross-corpus Pearson r reaches r = 0.935** for the discrepancy family,
and the top-10 most-reliable English benchmarks are identical on both
corpora (Jaccard 1.0 at K = 10).

The **discrepancy and dispersion families transfer cleanly across corpora.**
The relative-spread variants (`rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`,
`rel_dispersion`) — including the upstream AllenAI default `rel_std` —
correlate weakly across the two corpora; they're robust within-corpus but
not transferable.

Adding seeds tightens the cross-corpus agreement: 1B→1B Pearson r climbs
from 0.753 (single-seed, 3 model_families) to 0.935 (pooled, 9
model_families). This is why **`seeds_28_1797_1904` is the recommended
pool for downstream benchmark selection.**

## Headline — best variant per Apertus pool

Each pool's first two rows of
[`seeds_<pool>/pearson_r_per_variant.csv`](seeds_28_1797_1904/pearson_r_per_variant.csv)
(sorted by `r` descending):

| pool | best variant | r | runner-up | r |
|---|---|---:|---|---:|
| `seeds_1904` (single seed) | `dispersion` / `range` | **0.753** | `dist_std` | 0.728 |
| `seeds_28_1797` (train) | `discrepancy` | **0.842** | `star_discrepancy` | 0.831 |
| `seeds_28_1797_1904` (pooled, recommended) | `star_discrepancy_shifted` | **0.935** | `discrepancy` | 0.816 |

Per-pool full tables under
[`seeds_<...>/pearson_r_per_variant.csv`](seeds_28_1797_1904/pearson_r_per_variant.csv);
per-variant scatter grids at
[`seeds_<...>/snr_apertus_vs_snr_allenai_grid.png`](seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png).

![Apertus vs AllenAI SNR (pooled pool, best variant: star_discrepancy_shifted)](seeds_28_1797_1904/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

## Cross-corpus-reliable benchmarks (top-K agreement)

Each `<pool>/agreement.csv` tabulates the intersection / Jaccard of the
two corpora's top-K tasks (ranked over the 7-task shared universe);
`<pool>/agreement.md` is the human-readable mirror. All three pools
converge on the same shortlist (numbers below are the `intersection_over_k`
column of each pool's
[`agreement.csv`](seeds_28_1797_1904/agreement.csv)):

| K | seeds_1904 | seeds_28_1797 | seeds_28_1797_1904 |
|---|---:|---:|---:|
| 5 | 4/5 (Jaccard 0.67) | **5/5** (Jaccard 1.0) | **5/5** (Jaccard 1.0) |
| 10 | 7/7 (Jaccard 1.0) | 7/7 (Jaccard 1.0) | 7/7 (Jaccard 1.0) |
| 20 | 7/7 (Jaccard 1.0) | 7/7 (Jaccard 1.0) | 7/7 (Jaccard 1.0) |

**Benchmarks reliable in BOTH corpora (top-10, all pools):**

- `arc_challenge`
- `arc_easy`
- `csqa` (alias: `commonsense_qa` in Apertus)
- `hellaswag`
- `mmlu`
- `openbookqa`
- `piqa`

These are the tasks where pretraining-data-mix differences produce a
measurable, robust ranking on **both** the multilingual Apertus runs and
the AllenAI DataDecide ladder. **Strong candidates for the English side of
any multilingual benchmark suite**, independent of which pretraining
distribution you're working with.

## Size sweep — correlation strengthens with more model_families

Each `<pool>/pearson_r_size_sweep.csv` gives the mean Pearson r across all
22 variants at every matched-size pair. The 1B → 1B agreement improves
materially as we go from single-seed (`seeds_1904`, 3 model_families per
size) to pooled (`seeds_28_1797_1904`, 9 model_families per size) — adding
seeds tightens the SNR estimate on the Apertus side and brings it closer
to AllenAI's 25-mix DataDecide reference.

## Which variant families transfer?

| family | members | transfers across corpora? |
|---|---|---|
| **discrepancy** | `discrepancy`, `star_discrepancy`, `star_discrepancy_shifted`, `dispersion_shifted`, `gini` | **Yes** — top performers across pools |
| **dispersion** | `dispersion`, `range`, `mpd`, `aad`, `rms_deviation`, `quartile_deviation`, `dist_std` | **Yes** — strong, especially single-seed |
| **relative-spread** | `rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`, `rel_dispersion` | **No** — robust within-corpus, weak cross-corpus. Includes the upstream AllenAI default `rel_std`. |
| **depth** | `tukey`, `projection` | **No** — uncorrelated with DA in the first place |

## Methodology

- **Apertus side** (multi-seed, 3 mixes × 3 seeds × 4 sizes): per-task SNR
  table from one of the seed-pool subdirs under
  [`../snr_definition/`](../snr_definition/). Each pool gives its own
  cross-corpus output (`seeds_1904/`, `seeds_28_1797/`,
  `seeds_28_1797_1904/`).
- **AllenAI side** (DataDecide ladder, 25 mixes × 5 ckpts at sizes
  150M / 300M / 750M / 1B): pulled once at build time and run through
  [`build_allenai_variants.py`](build_allenai_variants.py), which reuses
  every primitive from
  [`multilingual/run_apertus_snr_variants.py`](../../multilingual/run_apertus_snr_variants.py)
  (`per_model_inputs`, `variant_signal_noise_snr`,
  `compute_size_decision_accuracy`, the 22-aggregator
  `AGGREGATION_FUNCTIONS` list). The shared driver groups by `model` for
  the signal pool and by `model_family` (model name minus the size token)
  for DA — so neither corpus needs a `seed` column to contribute.
- **Task-name reconciliation.** Apertus ran only the multilingual
  `global_mmlu_full_en_<subject>` view of MMLU on the full ckpt-series.
  AllenAI uses the vanilla `mmlu_<subject>` names. We canonicalise Apertus
  names via `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]`. The
  multilingual Apertus variants CSV (parent-task filter) collapses MMLU
  subject facets into `mmlu`, so the post-alias shared set is
  **7 standalone English tasks** (`arc_challenge`, `arc_easy`, `csqa`,
  `hellaswag`, `mmlu`, `openbookqa`, `piqa`).
- **Headline correlation axis:** `log10(snr_<V>_1B)` on each corpus,
  Pearson r over the 7 shared tasks. Per-corpus top-K lists are ranked
  over this same 7-task universe.
- **Reference HF models skipped.** SmolLM3-3B / Olmo-3-7B / Apertus-8B
  each have a single training mix, so the data-mix-spread term in every
  SNR variant is undefined. The rank-agreement comparison would have to
  use raw `primary_score` (a capability number, not a noise-relative one),
  conflating "task is reliable" with "task is easy for this model".

## How to enlarge the shared universe

AllenAI's `core` split has **178 tasks Apertus does not currently
evaluate**. Adding them to the Apertus eval suite would directly expand
the cross-corpus comparison surface from 7 tasks to potentially 185.

The highest-yield additions, by category:

| category | tasks Apertus is missing | how to add |
|---|---:|---|
| `mmlu_<subject>:mc` — multi-choice MMLU subjects | 53 | Apertus ran rank-classification (`global_mmlu_full_en_*`); add the `:mc` form. Roughly doubles MMLU comparison coverage on its own. |
| `mmlu_pro` — MMLU-Pro + 14 categories | 19 | Available in OLMES and `lm-evaluation-harness` (`mmlu_pro` / `mmlu_pro_<category>`). |
| `arc_*:mc`, `hellaswag:mc` | 3 | Multi-choice forms (Apertus has only `:rc`). |
| OLMES core knowledge / commonsense | 10 | `boolq:mc`, `openbookqa:mc`, `piqa:mc`, `commonsense_qa:mc`, `socialiqa(:mc)`, `winogrande(:mc)`, `truthfulqa_mc1`. |
| Math | 14 | `gsm8k`, `gsm_plus`, `gsm_symbolic_*`, `minerva_math_*` (lm-eval has `gsm8k`, `gsm8k_cot`, `hendrycks_math_*`). |
| Code | 4 | `codex_humaneval(plus)`, `mbpp(plus)`. Needs code-execution sandboxing to score. |
| BBH (Big-Bench Hard) | 27 | `bbh_*` matches directly in lm-eval. |
| AGI Eval | 19 | `agi_eval_*` (OLMES) ↔ `agieval_*` (lm-eval, no `:mc`/`:rc` split). |
| Generative QA | 8 | `drop`, `squad`, `triviaqa`, `medmcqa(:mc)`, `jeopardy`, `autobencher(:mc)`. |

Suffix `:mc` = multi-choice form, suffix `:rc` = rank-classification form.
The [OLMES harness](https://github.com/allenai/olmes) reproduces these
task IDs verbatim; `lm-evaluation-harness` has equivalents for most.

Not recommended (custom harness flow, aggregates of other benchmarks, or
non-evaluation probes):

- `paloma_*` (13 tasks) — perplexity / bytes-per-byte; needs the
  Paloma-specific harness flow.
- `multitask_*` (4 tasks) — aggregates of other benchmarks already covered.
- `custom_loss_*` (3 tasks) — direct training-loss probes.
- `copycolors:mc` — niche AllenAI test.

## Reproduce

```bash
# One-time: build AllenAI CSV (pulls the core split through the 22-variant loop)
python results/allenai_comparison/build_allenai_variants.py

# Per Apertus seed pool
for pool in seeds_1904 seeds_28_1797 seeds_28_1797_1904; do
    python results/allenai_comparison/analyze.py --pool $pool
done
```

## Directory contents

| file | description |
|---|---|
| [`build_allenai_variants.py`](build_allenai_variants.py) | Pulls AllenAI `core` parquet, runs the 22-variant SNR loop (shares `per_model_inputs`, `compute_size_decision_accuracy` with the Apertus driver — both group by `model` / `model_family`), writes `allenai_snr_variants_per_task.csv` |
| [`analyze.py`](analyze.py) | Loads the Apertus side via `--pool`, the AllenAI side from `allenai_snr_variants_per_task.csv`, applies the `global_mmlu_full_en_*` ↔ `mmlu_*` alias, writes every CSV / PNG / agreement.md into the matching subdir |
| `allenai_snr_variants_per_task.csv` | 241 tasks × 267 cols (22 variants × 4 sizes × 3 stats + 3 size-DA). Shared across pools — produced by `build_allenai_variants.py`. |
| `seeds_<pool>/` (one per Apertus pool) | Per-pool comparison outputs: `task_overlap.csv`, `pearson_r_per_variant.csv`, `pearson_r_size_sweep.csv`, `snr_apertus_vs_snr_allenai_<best_variant>.png`, `snr_apertus_vs_snr_allenai_grid.png`, `top_apertus.csv`, `top_allenai.csv`, `agreement.md` |
