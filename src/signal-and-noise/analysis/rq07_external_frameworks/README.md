# External frameworks — Does our SNR ranking agree with AllenAI DataDecide?

## Research question

> Do the SNR variants and the "reliable benchmark" set we find on our ladder
> also hold on AllenAI's DataDecide / OLMo corpus, on the English benchmarks
> both share?

> ⚠️ **Only six English benchmarks overlap, and three survive the gate.** The ladder
> shares `arc_easy`, `arc_challenge`, `hellaswag`, `csqa` (via `commonsense_qa`),
> `openbookqa` and MMLU (via the Global-MMLU English split) with DataDecide, and at
> the 1B rung the above-random gate keeps only `arc_easy`, `arc_challenge` and `hellaswag`.
>
> With so small a universe, top-K set overlap is uninformative (any K ≥ N is
> trivially 1.0), so the evidence is the **correlation of SNR over the shared
> tasks** (values: Pearson r; ranking: Spearman ρ).

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **On the `predictivity` pool SNR values and rank order cannot be compared across corpora** — variant `aad` (rq04's global best, not selected here), Pearson r of log₁₀(SNR) **0.87**, Spearman ρ **1.00**, over only **3** shared English tasks after the above-random gate — too few for a correlation to mean anything.
- **The shared universe is the English tasks both corpora evaluate** (ARC, HellaSwag, MMLU via the Global-MMLU English split, PIQA/CSQA/OpenBookQA where run), so the evidence is the SNR *correlation* over that handful, not top-K Jaccard (trivially 1.0 on so small a universe).
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the ladder pools: `predictivity`
(seed 1904, every cell, every data build and ladder, SwiGLU included) and
`predictivity_seeds` (every seed); the 36-sweep tiers stay as history. For each
pool we correlate, over the shared English tasks that clear the above-random
gate, log₁₀ SNR on our ladder with log₁₀ SNR on DataDecide (Pearson r) and their
rank order (Spearman ρ), for each of the 22 SNR variants.

The comparison runs at every size pair (175M↔150M, 350M↔300M, 600M↔750M,
1B↔1B, and the unmatched 1.7B↔1B), and the headline is the matched 1B↔1B pair.
The study's reference is 1.7B, but DataDecide stops at 1B and SNR grows with
size, so an unmatched headline would mix a size effect into the agreement.

DataDecide's "signal" is the spread over its 25 data recipes at a fixed size;
ours is the spread over the language settings and the data scheme (A/B/C ×
temperature T) at a fixed size. Read a low correlation as a difference in what
the two model populations vary before reading it as a failure of the SNR definition.

> ⚠️ **Methodological caveat — MMLU aliasing.** Apertus's
> `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's
> `mmlu[_<subject>]` rows so the comparison can use the MMLU subjects, but
> **the two are not the same content**: Apertus runs the Cohere-Full
> translation/post-edit of MMLU (English split), AllenAI runs the original
> Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ.
>
> Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus
> checkpoints, then drop the alias and compare like-for-like. MMLU does not clear
> the gate at our 1B rung today, so the alias does not enter the headline r; see
> `pretraining/<pool>/agreement.md` for the full caveat.

## Key figure

![Ladder SNR against DataDecide SNR on the shared English tasks](pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png)

Population: pool `predictivity` (seed 1904, every cell and data build at the 1B rung, SwiGLU included) against DataDecide's 1B rung (25 data recipes); SNR variant `aad` (the one the [surrogates analysis](../rq04_surrogates/README.md) ranks first); the 3 shared English tasks that clear the above-random gate (ARC Easy, ARC Challenge, HellaSwag); log₁₀ SNR, each axis on its own scale.

**Key finding.** Both corpora rank the three gated shared tasks the same way (ARC Easy > HellaSwag > ARC Challenge; Spearman ρ 1.00, Pearson r of log₁₀ SNR 0.87, p = 0.33), but three points are too few for the agreement to be evidence.

**Key findings**

- **Same order, n = 3.** `aad` SNR is 1.03 / 0.82 / 0.50 on our ladder and 9.76 / 5.04 / 4.16 on DataDecide for ARC Easy / HellaSwag / ARC Challenge.
- **Our SNR is 6.1–9.4× smaller.** DataDecide's `aad` SNR exceeds ours by 9.4× on ARC Easy, 6.1× on HellaSwag and 8.2× on ARC Challenge; the log-scale correlation ignores this offset, so compare orders across corpora, never SNR levels.
- **The seed pool agrees.** On `predictivity_seeds` (every seed) the same three tasks keep the same order (ρ 1.00, Pearson r 0.84).
- **Half of DataDecide's shared tasks are gated out on our side.** DataDecide also scores MMLU (5.01), CSQA (3.45) and OpenBookQA (1.69), but none of them clears the above-random gate at our 1B rung, so the top-5 overlap is 3 of 5 (Jaccard 0.60).

**Follow-ups**

- The same scatter at 1.7B↔1B, to check whether the order holds at our reference size (`aad` r 0.97 there, see [pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv)).
- Bootstrap CIs over the three tasks, to put a number on how little n = 3 constrains r.

GitHub: [snr_apertus_vs_snr_allenai_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png) · [snr_apertus_vs_snr_allenai_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.csv). Every variant and the size sweep: [Results](#results).

## Methodology

- **Apertus side** (the predictivity ladder, 175M–1.7B in the size sweep): the
  per-task SNR table `pretraining/<pool>/snr_variants_per_task.csv` from the noise
  and SNR analysis ([`../rq03_noise_and_snr/`](../rq03_noise_and_snr/)). Each pool
  writes its own cross-corpus output (`pretraining/predictivity/`,
  `pretraining/predictivity_seeds/`).
- **AllenAI side** (DataDecide ladder, 25 mixes × 5 ckpts at sizes
  150M / 300M / 750M / 1B): pulled once at build time and run through
  [build_allenai_variants.py](build_allenai_variants.py), which reuses every
  primitive from
  [run_apertus_snr_variants.py](../rq03_noise_and_snr/run_apertus_snr_variants.py)
  (`per_model_inputs`, `variant_signal_noise_snr`) plus
  `compute_size_decision_accuracy` from
  [compute_da.py](../rq02_decision_accuracy/compute_da.py) and the 22-aggregator
  `AGGREGATION_FUNCTIONS` list. The shared driver groups by `model` for the
  signal pool and by `model_family` (model name minus the size token) for DA — so
  neither corpus needs a `seed` column to contribute.
- **Task-name reconciliation.** Apertus ran only the multilingual
  `global_mmlu_full_en_<subject>` view of MMLU on the full ckpt-series; AllenAI
  uses the vanilla `mmlu_<subject>` names. We alias
  `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]` and the parent-task filter
  collapses the subjects into `mmlu`, so the shared set on the ladder is **6
  standalone English tasks** (`arc_challenge`, `arc_easy`, `csqa`, `hellaswag`,
  `mmlu`, `openbookqa`; the 36-sweep also shared `piqa`).
- **Correlation axis.** `log10(snr_<V>_<size>)` on each corpus, Pearson r (values)
  and Spearman ρ (rank) over the shared tasks that clear the above-random gate on
  our side (3 at 600M, 1B and 1.7B; 2 at 175M and 350M, too few for a correlation).
- **Reference HF models skipped.** SmolLM3-3B / Olmo-3-7B / Apertus-8B each have a
  single training mix, so the data-mix-spread term in every SNR variant is
  undefined; including them would force the comparison onto raw `primary_score` (a
  capability number), conflating "task is reliable" with "task is easy".

**Which variant families transfer across corpora** — Pearson r of log₁₀ SNR at
1B↔1B over the 3 gated shared tasks, `predictivity` / `predictivity_seeds`
(from `pearson_r_per_variant.csv`; every r rests on n = 3):

| family | members (r, seed 1904 / all seeds) | reading |
|---|---|---|
| **dispersion** | `dispersion` 0.80/0.77, `range` 0.80/0.77, `mpd` 0.95/0.96, `aad` 0.87/0.84, `rms_deviation` 0.99/0.99, `quartile_deviation` 0.80/0.40, `dist_std` 1.00/0.99, `mpsd` 0.97/1.00 | positive in both pools (0.40–1.00) |
| **discrepancy** | `discrepancy` 0.84/0.84, `star_discrepancy` 0.92/0.93, `star_discrepancy_shifted` 0.84/0.85, `dispersion_shifted` 0.95/0.96, `rel_star_discrepancy` 0.81/0.71, `gini` −0.82/−0.85 | positive except `gini` |
| **relative-spread** | `rel_std` −0.65/−0.96, `rel_mpd` −0.92/−0.99, `rel_mpsd` 0.16/0.03, `rel_dispersion` 0.01/−0.07, `iqr` 0.88/−0.87 | does not transfer (incl. AllenAI's default `rel_std`); `iqr` flips sign between pools |
| **depth** | `tukey` 0.54/0.84, `projection` 0.59/0.79 | positive, pool-dependent |
| **robust** | `mad` 0.80/0.57 | positive |

**Enlarging the shared universe.** The 6-task overlap (3 after the gate; 7 on the
36-sweep) is the binding constraint: 235 of DataDecide's 241 task rows (subjects
and formats counted separately, `task_overlap.csv`) are not in the shared set, and
57 of them are MMLU subject rows that the ladder has only as the collapsed `mmlu` parent.
Highest-yield additions, by category:

| category | missing | how to add |
|---|---:|---|
| `mmlu_<subject>:mc` (multi-choice MMLU) | 57 | Apertus ran rank-classification (`global_mmlu_full_en_*`); add the `:mc` form. |
| `mmlu_pro` (+ categories) | 15 | In OLMES and lm-eval (`mmlu_pro` / `mmlu_pro_<category>`). |
| BBH (Big-Bench Hard) | 27 | `bbh_*` matches directly in lm-eval. |
| AGI Eval | 19 | `agi_eval_*` (OLMES) ↔ `agieval_*` (lm-eval). |
| Math | 14 | `gsm8k`, `gsm_plus`, `minerva_math_*` (lm-eval: `gsm8k`, `hendrycks_math_*`). |
| OLMES core knowledge / commonsense | 10 | `boolq`, `piqa`, `socialiqa`, `winogrande` and the `:mc` forms of these plus `openbookqa` and `csqa`. |
| Generative QA | 6 | `drop`, `squad`, `triviaqa`, `medmcqa` (+ `:mc`), `jeopardy`. |
| `arc_*:mc`, `hellaswag:mc`, Code | 7 | Multi-choice ARC/HellaSwag; `codex_humaneval`, `mbpp` (need code sandboxing). |

**The harness tasks to evaluate on the ladder** are registered as the `allenai`
group of `configs/tasks.json` (lm-eval names, all rank-classification, all
scoreable by a sub-2B model): `arc_easy`, `arc_challenge`, `hellaswag`, `mmlu`
(the English original, which removes the Global-MMLU aliasing caveat), `piqa`,
`openbookqa`, `commonsense_qa` (DataDecide `csqa`), `social_iqa` (`socialiqa`),
`winogrande`, `boolq`, `medmcqa`. That is 11 shared task families (plus the 57
MMLU subjects) against the 3 that clear the gate today.

DataDecide's generative and code tasks
(`gsm8k`, `minerva_*`, `drop`, `squad`, `triviaqa`, `jeopardy`, `mbpp`,
`codex_humaneval`), BBH, AGI Eval and the `:mc` formats sit at the floor below
2B parameters and would only add gated-out rows.

Not worth adding: `paloma_*` (perplexity, custom harness), `multitask_*` /
`custom_loss_*` (aggregates / loss probes), `copycolors:mc` (niche).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-07 15:51** (full refresh, commit `7966367c`).

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Cross-corpus agreement by pool (headline = `predictivity`). Regenerate with `python analysis/rq07_external_frameworks/analyze.py --pool predictivity`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), at the variant rq04 selected on our ladder (the per-variant grid below shows the other 21). The above-random gate leaves the `n_shared` shown per pool; where it is small (≤5) the correlations are over a handful of points and should be read as indicative, not robust:

| pool | variant (from rq04) | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `predictivity` (grid, seed 1904) | `aad` | 0.87 | 1.00 | 3 |
| `predictivity_seeds` (all seeds) | `dist_std` | 0.99 | 1.00 | 3 |

![Ladder vs AllenAI SNR — rq04's variant](pretraining/predictivity/snr_apertus_vs_snr_allenai_aad.png)

![Ladder vs AllenAI SNR across variants](pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png)
<!-- END auto:results -->

[snr_apertus_vs_snr_allenai_aad.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_aad.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png) ·
[pearson_r_per_variant.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_per_variant.csv) ·
[shared_task_agreement.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/shared_task_agreement.csv) ·
[pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv) ·
[agreement.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/agreement.md)

Population: pools `predictivity` (seed 1904) and `predictivity_seeds` (every seed), all 22 SNR variants, the shared English tasks that clear the above-random gate on our side (n per size pair in `pearson_r_size_sweep.csv`). The previous figure showed one variant at one size; these show whether the agreement depends on the variant and the size.

**Key findings**

- **One variant is significant at 1B↔1B on seed 1904, on three points.** Pearson r ranges from −0.92 (`rel_mpd`) to 1.00 (`dist_std`, p = 0.04, the only p < 0.05), and 15 of the 22 variants have r ≥ 0.6.
- **The seed pool moves individual variants a lot.** On `predictivity_seeds` only `mpsd` (r 1.00, p = 0.04) passes p < 0.05 (`dist_std` drops to p = 0.10), while `iqr` flips from 0.88 to −0.87.
- **Relative-spread variants do not transfer.** AllenAI's default `rel_std` has r −0.65 (seed 1904) and −0.96 (all seeds) at 1B↔1B, against 0.84–0.87 for `aad`.
- **The correlation exists only from 600M up.** At 175M↔150M and 350M↔300M only 2 shared tasks clear the gate, so r is undefined; `aad` has r 0.95 on seed 1904 (0.99 on every seed) at 600M↔750M, and 0.97 at the unmatched 1.7B↔1B.
- **Three points make r unstable across sizes.** `rel_std` goes from −0.65 at 1B↔1B to 1.00 at 1.7B↔1B, each over three tasks, so read any single variant's r as indicative only.

**Follow-ups**

- Bootstrap or leave-one-task-out r per variant, to separate stable variants from those three points happen to favour.
- The size sweep as a figure (r against size pair, one line per family), once more shared tasks clear the gate.

## TODO

- [ ] Add `mmlu_pro` / BBH to widen the 6-task shared universe.
- [ ] Bootstrap CIs on the cross-corpus Pearson r and Spearman ρ.
- [ ] Re-run the original `mmlu` lm-eval task on Apertus and drop the MMLU alias
      for a like-for-like comparison.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`; reference **1B**, the 86-task old
list) or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, cross-model dispersion with no mixture axis). Its harness,
task set and reference differ from the ladder's, and its shared universe with
DataDecide is the 7 standalone English tasks (the ladder's `auto` list shares
3 after the gate), so its correlations are a replication on a different
population, never rows of the ladder's table.

## External model-set tier (`all/external`, 36-sweep)

The `external` tier (cross-model dispersion over the 270M…70B external ladder, no
mixture axis) compares its per-task SNR against the same AllenAI DataDecide SNR
table. Because the capable external models clear the above-random gate on **6** of
the 7 shared English tasks (vs the 4 the custom pool retains), the cross-corpus
correlation rests on a wider base — the strongest version of the agreement result.
Outputs in `all/external/`; regenerate with
`python analysis/rq07_external_frameworks/analyze.py --pool external`.

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

[snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/snr_apertus_vs_snr_allenai_grid.png) ·
[agreement.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/agreement.md)

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **On the pure 3-seed pool (`seeds_28_1797_1904`) SNR values and rank order agree across corpora** — best variant `dispersion_shifted`, Pearson r of log₁₀(SNR) **0.98**, Spearman ρ **1.00**, but over only **4** shared English tasks after the above-random gate — near-saturated, so indicative rather than robust.
- **Seed-count trend is not robust** — Pearson r 0.90 → 1.00 → 0.98 (1 → 2 → 3 seeds) is over only ~4 shared tasks; with so few points the values saturate near 1.0 and don't form a reliable monotone trend.
- **Dispersion + discrepancy families transfer; relative-spread does not** — the cross-corpus winners are discrepancy/dispersion variants (`projection`, `dispersion_shifted`, `dispersion_shifted`), not the mean-normalised relative-spread family (incl. AllenAI's own `rel_std`).
- **Only 7 English tasks overlap the two corpora, and the above-random gate leaves just 4 of them** — so the evidence is the SNR *correlation* over that handful, not top-K Jaccard (trivially 1.0 on so small a universe). `custom_swissai_hf` keeps n_shared = **4**.

### Results

Cross-corpus agreement by pool (headline = the pure 3-seed pool `seeds_28_1797_1904`). Regenerate with `python analysis/rq07_external_frameworks/analyze.py --pool custom_swissai_hf`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), each pool's best cross-corpus variant. The English overlap universe is 7 tasks; the above-random gate leaves the `n_shared` shown per pool. Where `n_shared` is small (≤5) the correlations are over a handful of points and should be read as indicative, not robust:

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_1904` (1 seed) | `projection` | 0.90 | 0.80 | 4 |
| `seeds_28_1797` (2 seeds) | `dispersion_shifted` | 1.00 | 1.00 | 4 |
| `seeds_28_1797_1904` (3 seeds) | `dispersion_shifted` | 0.98 | 1.00 | 4 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |

![Apertus vs AllenAI SNR — 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_dispersion_shifted.png)

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

[snr_apertus_vs_snr_allenai_dispersion_shifted.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_dispersion_shifted.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

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
