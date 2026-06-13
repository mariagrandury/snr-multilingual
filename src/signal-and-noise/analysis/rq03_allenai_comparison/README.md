# RQ3 — Does our SNR ranking agree with AllenAI DataDecide?

## Research question

> Do the SNR variants and the "reliable benchmark" set we find on the Apertus
> corpus also hold on AllenAI's DataDecide / OLMo corpus, on the English
> benchmarks both share?

> ⚠️ **Only 7 English benchmarks overlap** (arc_challenge, arc_easy, csqa,
> hellaswag, mmlu, openbookqa, piqa). With a 7-task universe, *set*-overlap
> metrics (top-K Jaccard) are uninformative — any K ≥ 7 spans the whole universe
> and is trivially 1.0. The real evidence is the **correlation of SNR over those
> 7 tasks** (values: Pearson r; ranking: Spearman ρ), not the overlap.

<!-- BEGIN auto:highlight (analyze.py --pool custom_swissai_hf) -->
## Highlighted result

- **On the pure 3-seed pool (`seeds_28_1797_1904`) SNR values and rank order agree across corpora** — best variant `dispersion_shifted`, Pearson r of log₁₀(SNR) **0.98**, Spearman ρ **1.00**, but over only **4** shared English tasks after the above-random gate — near-saturated, so indicative rather than robust.
- **Seed-count trend is not robust** — Pearson r 0.90 → 1.00 → 0.98 (1 → 2 → 3 seeds) is over only ~4 shared tasks; with so few points the values saturate near 1.0 and don't form a reliable monotone trend.
- **Dispersion + discrepancy families transfer; relative-spread does not** — the cross-corpus winners are discrepancy/dispersion variants (`projection`, `dispersion_shifted`, `dispersion_shifted`), not the mean-normalised relative-spread family (incl. AllenAI's own `rel_std`).
- **Only 7 English tasks overlap the two corpora, and the above-random gate leaves just 4 of them** — so the evidence is the SNR *correlation* over that handful, not top-K Jaccard (trivially 1.0 on so small a universe). `custom_swissai_hf` keeps n_shared = **4**.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the four custom-pretraining tiers:
`seeds_1904` (1 seed) · `seeds_28_1797` (2 seeds) · `seeds_28_1797_1904`
(3 seeds) · `custom_swissai_hf` (3 seeds + externals) — plus the `external`
model-set tier under `all/external/` (see the dedicated section below). For each
pool we compute,
over the shared English tasks, the cross-corpus Pearson r of log₁₀(SNR@1B)
(values) and Spearman ρ (rank order), reporting the best-correlating variant.
The pure 3-seed pool `seeds_28_1797_1904` is the canonical, like-for-like
comparison: adding external models shifts the shared-task SNR and the
above-random gate drops the at-chance translated MCQA, so the comprehensive
`custom_swissai_hf` pool ends up with a smaller shared universe (use it for
scaling/power in RQ1, not for the AllenAI comparison).

> ⚠️ **Methodological caveat — MMLU aliasing.** Apertus's
> `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's
> `mmlu[_<subject>]` rows so the comparison can use the MMLU subjects, but
> **the two are not the same content**: Apertus runs the Cohere-Full
> translation/post-edit of MMLU (English split), AllenAI runs the original
> Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may
> differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual
> Apertus checkpoints, then drop the alias and compare like-for-like. See
> `pretraining/<pool>/agreement.md` for the full caveat.

## Methodology

- **Apertus side** (multi-seed, 3 mixes × 3 seeds × 4 sizes): the per-task SNR
  table from one of the seed-pool subdirs under
  [`../rq02_snr_definition/`](../rq02_snr_definition/). Each pool produces its own
  cross-corpus output (`pretraining/seeds_1904/`, `…/seeds_28_1797/`,
  `…/seeds_28_1797_1904/`, `…/custom_swissai_hf/`).
- **AllenAI side** (DataDecide ladder, 25 mixes × 5 ckpts at sizes
  150M / 300M / 750M / 1B): pulled once at build time and run through
  [build_allenai_variants.py](build_allenai_variants.py), which reuses every
  primitive from
  [run_apertus_snr_variants.py](../rq02_snr_definition/run_apertus_snr_variants.py)
  (`per_model_inputs`, `variant_signal_noise_snr`) plus
  `compute_size_decision_accuracy` from
  [compute_da.py](../rq01_decision_accuracy/compute_da.py) and the 22-aggregator
  `AGGREGATION_FUNCTIONS` list. The shared driver groups by `model` for the
  signal pool and by `model_family` (model name minus the size token) for DA — so
  neither corpus needs a `seed` column to contribute.
- **Task-name reconciliation.** Apertus ran only the multilingual
  `global_mmlu_full_en_<subject>` view of MMLU on the full ckpt-series; AllenAI
  uses the vanilla `mmlu_<subject>` names. We canonicalise the Apertus names via
  `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]`. The parent-task filter on the
  Apertus variants CSV collapses MMLU subject facets into `mmlu`, so the
  post-alias shared set is the **7 standalone English tasks** (`arc_challenge`,
  `arc_easy`, `csqa`, `hellaswag`, `mmlu`, `openbookqa`, `piqa`). This aliasing
  is the headline caveat above — same task IDs, different MMLU content.
- **Correlation axis.** `log10(snr_<V>_1B)` on each corpus, Pearson r (values)
  and Spearman ρ (rank) over the 7 shared tasks; per-corpus top-K lists are
  ranked over this same universe.
- **Reference HF models skipped.** SmolLM3-3B / Olmo-3-7B / Apertus-8B each have a
  single training mix, so the data-mix-spread term in every SNR variant is
  undefined; including them would force the comparison onto raw `primary_score` (a
  capability number), conflating "task is reliable" with "task is easy".

**Which variant families transfer across corpora** (qualitative, stable across
pools):

| family | members | transfers? |
|---|---|---|
| **discrepancy** | `discrepancy`, `star_discrepancy`, `star_discrepancy_shifted`, `dispersion_shifted`, `gini` | **Yes** — top cross-corpus performers |
| **dispersion** | `dispersion`, `range`, `mpd`, `aad`, `rms_deviation`, `quartile_deviation`, `dist_std` | **Yes** — strong, especially single-seed |
| **relative-spread** | `rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`, `rel_dispersion` | **No** — robust within-corpus, weak cross-corpus (includes AllenAI's default `rel_std`) |
| **depth** | `tukey`, `projection` | **No** — uncorrelated with DA in the first place |

**Enlarging the shared universe.** The 7-task overlap is the binding constraint;
AllenAI's `core` split has ~178 tasks Apertus does not yet evaluate. Highest-yield
additions, by category — adding them to the Apertus suite directly widens the
comparison surface:

| category | missing | how to add |
|---|---:|---|
| `mmlu_<subject>:mc` (multi-choice MMLU) | 53 | Apertus ran rank-classification (`global_mmlu_full_en_*`); add the `:mc` form. Roughly doubles MMLU coverage. |
| `mmlu_pro` (+ 14 categories) | 19 | In OLMES and lm-eval (`mmlu_pro` / `mmlu_pro_<category>`). |
| BBH (Big-Bench Hard) | 27 | `bbh_*` matches directly in lm-eval. |
| AGI Eval | 19 | `agi_eval_*` (OLMES) ↔ `agieval_*` (lm-eval). |
| Math | 14 | `gsm8k`, `gsm_plus`, `minerva_math_*` (lm-eval: `gsm8k`, `hendrycks_math_*`). |
| OLMES core knowledge / commonsense | 10 | `:mc` forms of boolq, openbookqa, piqa, commonsense_qa, socialiqa, winogrande, truthfulqa_mc1. |
| Generative QA | 8 | `drop`, `squad`, `triviaqa`, `medmcqa`, `jeopardy`. |
| `arc_*:mc`, `hellaswag:mc`, Code | 7 | Multi-choice ARC/HellaSwag; `codex_humaneval`, `mbpp` (need code sandboxing). |

Not worth adding: `paloma_*` (perplexity, custom harness), `multitask_*` /
`custom_loss_*` (aggregates / loss probes), `copycolors:mc` (niche).

<!-- BEGIN auto:results (analyze.py --pool custom_swissai_hf) -->
## Results

Cross-corpus agreement by pool (headline = the pure 3-seed pool `seeds_28_1797_1904`). Regenerate with `python analysis/rq03_allenai_comparison/analyze.py --pool custom_swissai_hf`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), each pool's best cross-corpus variant. The English overlap universe is 7 tasks; the above-random gate leaves the `n_shared` shown per pool. Where `n_shared` is small (≤5) the correlations are over a handful of points and should be read as indicative, not robust:

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_1904` (1 seed) | `projection` | 0.90 | 0.80 | 4 |
| `seeds_28_1797` (2 seeds) | `dispersion_shifted` | 1.00 | 1.00 | 4 |
| `seeds_28_1797_1904` (3 seeds) | `dispersion_shifted` | 0.98 | 1.00 | 4 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_dispersion_shifted.png)

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)
<!-- END auto:results -->

## External model-set tier (`all/external`)

The `external` tier (cross-model dispersion over the 270M…70B external ladder, no
mixture axis) compares its per-task SNR against the same AllenAI DataDecide SNR
table. Because the capable external models clear the above-random gate on **6** of
the 7 shared English tasks (vs the 4 the custom pool retains), the cross-corpus
correlation rests on a wider base — the strongest version of the agreement result.
Outputs in `all/external/`; regenerate with
`python analysis/rq03_allenai_comparison/analyze.py --pool external`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR)
(values) and Spearman ρ (rank), best cross-corpus variant per model set. The same
discrepancy/dispersion family wins on every set; relative-spread (incl. AllenAI's
own `rel_std`) does not transfer:

| model set | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_28_1797_1904` (pure 3-seed) | `dispersion_shifted` | 0.98 | 1.00 | 4 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |
| **`external`** (`all/external`) | **`star_discrepancy_shifted`** | **+0.892** | **+0.829** | **6** |

Both corpora rank `hellaswag` / `piqa` at the top and `arc_challenge` / `csqa` at
the bottom (top-5 Jaccard 0.67); the runner-up variants are again discrepancy /
dispersion members (`discrepancy` 0.80, `dispersion_shifted` 0.78), with
relative-spread weak (`rel_std` ≈ 0.28).

![Apertus vs AllenAI SNR — external tier, best variant](all/external/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

![Apertus vs AllenAI SNR across variants — external tier](all/external/snr_apertus_vs_snr_allenai_grid.png)

**Caveat (unchanged).** The shared universe is still only 7 English tasks, so even
at n_shared = 6 this is indicative, not robust; and 1 shared task is the aliased
`global_mmlu_full_en → mmlu` (different MMLU content; `commonsense_qa → csqa` is
the other alias). See `all/external/agreement.md`.

## TODO

- [ ] Add `mmlu_pro` / BBH to widen the 7-task shared universe.
- [ ] Bootstrap CIs on the cross-corpus Pearson r and Spearman ρ.
- [ ] Re-run the original `mmlu` lm-eval task on Apertus and drop the MMLU alias
      for a like-for-like comparison.

## Files

- `pretraining/<pool>/pearson_r_per_variant.csv` — cross-corpus Pearson r for
  every SNR variant (the headline table).
- `…/shared_task_agreement.csv` — best cross-corpus variant + Pearson r /
  Spearman ρ over the shared tasks (the per-pool result row).
- `…/pearson_r_size_sweep.csv` — r vs the size used for the comparison.
- `…/agreement.csv`, `agreement.md` — top-K reliable-benchmark overlap +
  the MMLU-aliasing caveat.
- `…/top_apertus.csv`, `top_allenai.csv`, `task_overlap.csv`.
- `…/snr_apertus_vs_snr_allenai_*.png` — per-variant + grid scatters.
