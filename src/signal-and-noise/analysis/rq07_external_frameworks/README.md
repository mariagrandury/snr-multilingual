# External frameworks: does our SNR ranking agree with AllenAI DataDecide?

## Research question

> Do the SNR variants and the "reliable benchmark" set that we find on our
> ladder also hold on AllenAI's DataDecide / OLMo corpus, for the English
> benchmarks that both evaluate?

> **Only nine English benchmarks overlap, and all nine pass the gate at 1B.** The
> ladder shares `arc_easy`, `arc_challenge`, `hellaswag`, `csqa` (through the cloze twin
> `rf_commonsense_qa`), `openbookqa`, MMLU (through the cloze twin `rf_mmlu`), `piqa`,
> `winogrande` and `boolq` with DataDecide. At the 1B rung the above-random gate keeps all nine.
>
> With so few tasks, top-K set overlap tells us little (any K ≥ N gives 1.0
> trivially). The evidence is therefore the **correlation of SNR over the shared
> tasks**: Pearson r on the values and Spearman ρ on the ranking.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **On the `predictivity` pool, SNR values and rank order agree across corpora.** With variant `rel_std` (rq04's global best, not selected here), the Pearson r of log₁₀(SNR) is **0.75** and the Spearman ρ **0.78**, over the 9 shared English tasks.
- **The shared tasks are the English tasks both corpora evaluate** (`hellaswag`, `mmlu`, `arc_easy`, `winogrande`, `arc_challenge`, `csqa`, `piqa`, `openbookqa`, `boolq`, with MMLU and CSQA read through our cloze twins). The evidence is therefore the SNR *correlation* over this handful, not the top-K Jaccard, which is trivially 1.0 on so few tasks.
- **Like for like since 2026-10-08.** Both sides use the same 20 % noise window (5 points at 1B), the same detrended checkpoint noise (rule 4) and the same metric (`build_allenai_variants.py`). The signal populations still differ: 25 pretraining corpora against our design variants.
<!-- END auto:highlight -->

## Experimental setup

**Setup.**
- **Measured.** We test whether our ladder and DataDecide agree on which English tasks separate their models, using the SNR of each task on both.
- **Models.** We compare our 1B rung (seed 1904, every cell, ladder and data build) with the 1B rung of DataDecide (25 data recipes).
- **Tasks.** The tasks are the English tasks that both evaluate and that clear the above-chance gate on our side.
- **How.** SNR is the relative standard deviation of the final scores across models divided by the relative checkpoint noise (the `rel_std` variant). We compare it on a log scale.

Outputs live under `pretraining/<pool>/` for the two ladder pools: `predictivity`
(seed 1904, every cell, every data build and ladder, SwiGLU included) and
`predictivity_seeds` (every seed). The 36-sweep tiers stay as history. For each
pool and each of the 22 SNR variants, we correlate log₁₀ SNR on our ladder with
log₁₀ SNR on DataDecide (Pearson r), and their rank orders (Spearman ρ), over
the shared English tasks that clear the above-random gate.

The comparison runs at every size pair (175M↔150M, 350M↔300M, 600M↔750M,
1B↔1B, and the unmatched 1.7B↔1B). The headline is the matched 1B↔1B pair.
The study's reference is 1.7B, but DataDecide stops at 1B and SNR grows with
size, so an unmatched headline would mix a size effect into the agreement.

DataDecide's "signal" is the spread over its 25 data recipes at a fixed size.
Ours is the spread over the language settings and the data scheme (A/B/C ×
temperature T) at a fixed size. A low correlation may therefore reflect what
the two model populations vary, before it reflects a failure of the SNR definition.

> **Methodological caveat: MMLU aliasing (resolved 2026-10-08).** Until the
> 2026-10-08 refresh, Apertus's `global_mmlu_full_en[_<subject>]` rows were aliased
> to AllenAI's `mmlu[_<subject>]` rows, but **the two do not hold the same content**.
> Apertus ran the Cohere-Full translation/post-edit of MMLU (English split), and AllenAI
> the original Hendrycks et al. MMLU.
>
> The ladder now evaluates the original MMLU (`cais/mmlu`) and its cloze twin
> `rf_mmlu`, which matches DataDecide's RC `mmlu` and clears the gate at our
> 1B rung. The alias is gone and MMLU enters the headline r (see
> `pretraining/<pool>/agreement.md`).

## Key figure

![Ladder SNR against DataDecide SNR on the shared English tasks](pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png)

Population: pool `predictivity` (seed 1904, every cell and data build at the 1B rung, SwiGLU included) against DataDecide's 1B rung (25 data recipes). SNR variant: `rel_std`, the one the [surrogates analysis](../rq04_surrogates/README.md) ranks first. Tasks: the 9 shared English tasks that clear the above-random gate (ARC Easy, ARC Challenge, HellaSwag, MMLU, CommonsenseQA, OpenBookQA, PIQA, Winogrande, BoolQ). Axes: log₁₀ SNR, each on its own scale.

**Key finding.** **Both corpora rank the nine gated shared tasks in a similar order.** The Spearman ρ is 0.77 and the Pearson r of log₁₀ SNR 0.77 (p = 0.016). With nine points, this is indicative, not robust.

**Key findings**

- **Both corpora put BoolQ last and OpenBookQA near the bottom, but their top task differs (n = 9).** The `rel_std` SNR is 4.09 / 8.18 / 2.97 / 4.94 / 2.64 / 1.78 / 3.51 / 3.23 / 1.35 on our ladder and 15.67 / 15.34 / 12.61 / 13.30 / 9.29 / 4.09 / 6.53 / 4.00 / 3.56 on DataDecide, for ARC Easy / HellaSwag / ARC Challenge / MMLU / CSQA / OpenBookQA / PIQA / Winogrande / BoolQ. The top task is HellaSwag on ours and ARC Easy on DataDecide.
- **Our SNR is 1.2–4.2× smaller.** DataDecide's `rel_std` SNR exceeds ours by 4.2× on ARC Challenge, 3.8× on ARC Easy, 3.5× on CSQA, 2.7× on MMLU, 2.6× on BoolQ, 2.3× on OpenBookQA, 1.9× on HellaSwag, 1.9× on PIQA and 1.2× on Winogrande. The log-scale correlation ignores this offset, so compare orders across corpora, never SNR levels.
- **The seed pool agrees more.** On `predictivity_seeds` (every seed) the agreement rises to ρ 0.88 and Pearson r 0.85.
- **None of the shared tasks is gated out on our side.** MMLU and CSQA (through the cloze twins), OpenBookQA, PIQA, Winogrande and BoolQ all clear the above-random gate at our 1B rung. The two top-5 lists share three tasks (ARC Easy, HellaSwag and MMLU, Jaccard 0.43).

**Follow-ups**

- The same scatter at 1.7B↔1B, to check whether the order holds at our reference size (`rel_std` r 0.85 there, see [pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv)).
- Bootstrap CIs over the nine tasks, to measure how little n = 9 constrains r.

GitHub: [snr_apertus_vs_snr_allenai_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png) · [snr_apertus_vs_snr_allenai_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.csv). Every variant and the size sweep: [Results](#results).

## Methodology

- **Apertus side** (the predictivity ladder, 175M–1.7B in the size sweep): the
  per-task SNR table `pretraining/<pool>/snr_variants_per_task.csv` from the noise
  and SNR analysis ([`../rq03_noise_and_snr/`](../rq03_noise_and_snr/)). Each pool
  writes its own cross-corpus output (`pretraining/predictivity/`,
  `pretraining/predictivity_seeds/`).
- **AllenAI side** (DataDecide ladder, 25 mixes at seed 6198, sizes
  150M / 300M / 750M / 1B, every 2,500-step checkpoint of the
  `datadecide_intermediate` split): computed by
  [build_allenai_variants.py](build_allenai_variants.py), which reuses every
  primitive from
  [run_apertus_snr_variants.py](../rq03_noise_and_snr/run_apertus_snr_variants.py)
  (`per_model_inputs`, `ckpt_noise_columns`, `variant_signal_noise_snr`) plus
  `compute_size_decision_accuracy` from
  [compute_da.py](../rq02_decision_accuracy/compute_da.py) and the 22-aggregator
  `AGGREGATION_FUNCTIONS` list.
- **Like for like (2026-10-08).** Both sides read the same noise window (the
  k/20 points in the last 20 % of each run, 5 points at 1B) and the same checkpoint
  noise (the residual SD around a line through the window, RULES.md rule 4). The
  raw SD sits beside it as `ckpt_noise_raw_<size>`. Both sides also use the same
  metric: each DataDecide task is read on the column that matches our task's metric
  (`acc_norm` → `acc_per_char`, `acc` → `acc_raw`), not on DataDecide's own primary
  metric (`acc_uncond` for ARC-C, OBQA and CSQA). Before that date the table read the
  `core` split's last five checkpoints (86.5–100 % of the 1B run), with the raw SD on
  DataDecide's primary metric. The signal populations still differ: 25 pretraining
  corpora against our design variants, whose English spread is 4–7× smaller
  (`plan/signal-audit-2026-10-08.md`, F1).
- **Task-name reconciliation.** DataDecide's RC (cloze) MMLU and CSQA are matched
  to our cloze twins `rf_mmlu` and `rf_commonsense_qa`. Our letter-format `mmlu` and
  `commonsense_qa` are named `mmlu:mc` / `csqa:mc`, DataDecide's MC format, which
  its checkpoint series does not carry, so they are not shared. The
  `global_mmlu_full_en → mmlu` alias of the 36-sweep is gone. The shared set on the
  ladder is the nine English tasks both evaluate (`arc_challenge`, `arc_easy`, `boolq`,
  `csqa`, `hellaswag`, `mmlu`, `openbookqa`, `piqa`, `winogrande`).
- **Correlation axis.** `log10(snr_<V>_<size>)` on each corpus, with Pearson r (values)
  and Spearman ρ (ranks) over the shared tasks that clear the above-random gate on
  our side: 9 at 600M, 1B and 1.7B, and 7 at 175M and 350M, where ARC Challenge and
  Winogrande are at chance.
- **Reference HF models skipped.** SmolLM3-3B, Olmo-3-7B and Apertus-8B each have a
  single training mix, so the data-mix spread in every SNR variant is undefined.
  Including them would force the comparison onto the raw `primary_score`, a
  capability number, which would confuse "the task is reliable" with "the task is easy".

**Which variant families transfer across corpora.** Pearson r of log₁₀ SNR at
1B↔1B over the 9 gated shared tasks, `predictivity` / `predictivity_seeds`
(from `pearson_r_per_variant.csv`, every r rests on n = 9):

| family | members (r, seed 1904 / all seeds) | reading |
|---|---|---|
| **dispersion** | `dispersion` 0.87/0.91, `range` 0.87/0.91, `mpd` 0.63/0.73, `aad` 0.62/0.69, `rms_deviation` 0.75/0.81, `quartile_deviation` 0.35/0.38, `dist_std` 0.81/0.86, `mpsd` 0.59/0.68 | positive in both pools (0.35–0.91), `quartile_deviation` weakest |
| **discrepancy** | `discrepancy` 0.82/0.90, `star_discrepancy` 0.84/0.91, `star_discrepancy_shifted` 0.72/0.76, `dispersion_shifted` 0.79/0.89, `rel_star_discrepancy` 0.87/0.93, `gini` 0.21/0.45 | positive, `gini` weakest |
| **relative-spread** | `rel_std` 0.77/0.85, `rel_mpd` 0.65/0.78, `rel_mpsd` 0.58/0.70, `rel_dispersion` 0.90/0.94, `iqr` 0.65/0.55 | transfers (AllenAI's default `rel_std` included), `rel_dispersion` highest of all 22 |
| **depth** | `tukey` 0.58/0.85, `projection` 0.69/0.79 | positive, `tukey` pool-dependent |
| **robust** | `mad` 0.31/0.36 | weak in both pools |

**Enlarging the shared universe.** The 9-task overlap (all 9 kept by the gate at
1B, against 7 on the 36-sweep) is the binding constraint. The DataDecide side has
only 10 tasks with a checkpoint series (the `datadecide_intermediate` split,
`task_overlap.csv`): `arc_challenge`, `arc_easy`, `boolq`, `csqa`, `hellaswag`,
`mmlu`, `openbookqa`, `piqa`, `socialiqa` and `winogrande`. Only `socialiqa` is not
shared yet. It is registered as `social_iqa` in the `allenai` group of
`configs/tasks.json`, but the ladder report has no `social_iqa` row, so
evaluating it on the ladder is the one addition that can widen the overlap, to 10.

No other task can be shared, whatever its score on our side. DataDecide's
series carries none of the multi-choice `:mc` formats, `mmlu_pro`, BBH, AGI Eval,
the math, generative and code tasks, `medmcqa`, `paloma_*`, `multitask_*`,
`custom_loss_*` or `copycolors:mc`. For BBH the floor is also real on our ladder:
none of the 6 `bbh_cloze` and 17 `bbh_mcq` subtasks clears the above-random gate
at 1.7B (rq00's `first_size_above_random.csv` gives "never" for both), although
our cloze reformulation `rf_bbh_mcq` passes on 10 of 17 subtasks at 1.7B (the
family from 600M).

Hand-written numbers in this README come from the ladder-report snapshot
**2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended
checkpoint noise, the nine shared tasks).

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Cross-corpus agreement by pool (headline = `predictivity`). Regenerate with `python analysis/rq07_external_frameworks/analyze.py --pool predictivity`.

**Cross-corpus agreement over the shared English tasks.** Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), at the variant rq04 selected on our ladder (the per-variant grid below shows the other 21). The above-random gate leaves the `n_shared` shown per pool. Where it is small (≤ 5), the correlations rest on a handful of points and are indicative, not robust:

| pool | variant (from rq04) | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `predictivity` (grid, seed 1904) | `rel_std` | 0.75 | 0.78 | 9 |
| `predictivity_seeds` (all seeds) | `rel_std` | 0.84 | 0.85 | 9 |

![Ladder vs AllenAI SNR, rq04's variant](pretraining/predictivity/snr_apertus_vs_snr_allenai_rel_std.png)

![Ladder vs AllenAI SNR across variants](pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png)
<!-- END auto:results -->

[snr_apertus_vs_snr_allenai_rel_std.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_rel_std.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png) ·
[pearson_r_per_variant.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_per_variant.csv) ·
[shared_task_agreement.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/shared_task_agreement.csv) ·
[pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv) ·
[agreement.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/agreement.md)

Population: pools `predictivity` (seed 1904) and `predictivity_seeds` (every seed), all 22 SNR variants, the shared English tasks that clear the above-random gate on our side (n per size pair in `pearson_r_size_sweep.csv`). The key figure showed one variant at one size. These figures show whether the agreement depends on the variant and the size.

**Key findings**

- **On nine points, 12 of the 22 variants are significant at 1B↔1B on seed 1904.** Pearson r ranges from 0.21 (`gini`) to 0.90 (`rel_dispersion`, p = 0.001), and 16 of the 22 variants have r ≥ 0.6.
- **The seed pool raises the agreement.** On `predictivity_seeds`, 18 variants pass p < 0.05: `aad`, `mpd`, `mpsd`, `rel_mpd`, `rel_mpsd` and `tukey` join, and none drops out. The largest move is `tukey`'s, by 0.27 (0.58 → 0.85), and `iqr` keeps its sign (0.65 → 0.55).
- **Relative-spread variants transfer.** AllenAI's default `rel_std` has r 0.77 (seed 1904) and 0.85 (all seeds) at 1B↔1B, against 0.62–0.69 for `aad`, and `rel_dispersion` is the highest of all 22 (0.90 and 0.94).
- **The correlation is defined at every size pair.** At 175M↔150M and 350M↔300M, 7 shared tasks clear the gate (ARC Challenge and Winogrande are at chance). `rel_std` has r 0.36 and 0.57 there on seed 1904, 0.68 at 600M↔750M (on seed 1904 and on every seed), and 0.85 at the unmatched 1.7B↔1B.
- **Seven to nine points make r unstable across sizes.** `rel_std` spans 0.36–0.85 over the five size pairs and `rel_mpd` 0.31–0.73, while `rel_star_discrepancy` stays at 0.82–0.95. Read any single variant's r as indicative only.

**Follow-ups**

- Bootstrap or leave-one-task-out r per variant, to separate the stable variants from those that nine points happen to favour.
- The size sweep as a figure (r against size pair, one line per family), now that seven or nine shared tasks clear the gate at every size pair.

## TODO

- [ ] Evaluate `social_iqa` (DataDecide `socialiqa`) on the ladder, the only DataDecide task
      not yet shared, to widen the shared universe from 9 to 10 tasks.
- [ ] Bootstrap CIs on the cross-corpus Pearson r and Spearman ρ.
- [x] Re-run the original `mmlu` lm-eval task on Apertus and drop the MMLU alias
      for a like-for-like comparison (done 2026-10-08: `rf_mmlu` against DataDecide's RC `mmlu`).

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`, reference **1B**, the old list of 86
tasks) or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, cross-model dispersion with no mixture axis). Its harness,
task set and reference differ from the ladder's. Its shared universe with
DataDecide is the 7 standalone English tasks (the ladder shares 9 after the
gate). Its correlations are therefore a replication on a different population,
never rows of the ladder's table.

## External model-set tier (`all/external`, 36-sweep)

The `external` tier (cross-model dispersion over the 270M…70B external ladder, no
mixture axis) compares its per-task SNR against the same AllenAI DataDecide SNR
table. The capable external models clear the above-random gate on **6** of
the 7 shared English tasks (against the 4 the custom pool keeps), so the cross-corpus
correlation rests on a wider base. This is the strongest version of the agreement result.
Outputs are in `all/external/`. Regenerate them with
`python analysis/rq07_external_frameworks/analyze.py --pool external`.

**Cross-corpus agreement over the shared English tasks.** Pearson r of log₁₀(SNR)
(values) and Spearman ρ (rank), for the best cross-corpus variant of each model set. The same
discrepancy/dispersion family wins on every set, and relative-spread (AllenAI's
own `rel_std` included) does not transfer:

| model set | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_28_1797_1904` (pure 3-seed) | `dispersion_shifted` | 0.98 | 1.00 | 4 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |
| **`external`** (`all/external`) | **`star_discrepancy_shifted`** | **+0.892** | **+0.829** | **6** |

Both corpora rank `hellaswag` / `piqa` at the top and `arc_challenge` / `csqa` at
the bottom (top-5 Jaccard 0.67). The runner-up variants are again discrepancy /
dispersion members (`discrepancy` 0.80, `dispersion_shifted` 0.78), and
relative-spread is weak (`rel_std` ≈ 0.28). This frozen pool predates the acc_norm pin for PIQA (2026-10-09): it was built on the AllenAI table that scored `piqa` by raw accuracy, while the ladder figures above read DataDecide's `acc_per_char`.

![Apertus vs AllenAI SNR, external tier, best variant](all/external/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png)

![Apertus vs AllenAI SNR across variants, external tier](all/external/snr_apertus_vs_snr_allenai_grid.png)

**Caveat (unchanged).** The shared universe of this sweep is only 7 English tasks, so even
at n_shared = 6 the result is indicative, not robust. One shared task is the aliased
`global_mmlu_full_en → mmlu` (different MMLU content), and `commonsense_qa → csqa` is
the other alias. See `all/external/agreement.md`.

[snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/snr_apertus_vs_snr_allenai_grid.png) ·
[agreement.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/all/external/agreement.md)

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history. The predictivity ladder regenerates the blocks above.

### Highlighted result

- **On the pure 3-seed pool (`seeds_28_1797_1904`), SNR values and rank order agree across corpora.** The best variant is `dispersion_shifted`, with a Pearson r of log₁₀(SNR) of **0.98** and a Spearman ρ of **1.00**, but over only **4** shared English tasks after the above-random gate. The values are near saturation, so the result is indicative rather than robust.
- **The seed-count trend is not robust.** Pearson r goes 0.90 → 1.00 → 0.98 (1 → 2 → 3 seeds) over only ~4 shared tasks. With so few points the values saturate near 1.0 and do not form a reliable monotone trend.
- **The dispersion and discrepancy families transfer, and relative-spread does not.** The cross-corpus winners are discrepancy/dispersion variants (`projection`, `dispersion_shifted`, `dispersion_shifted`), not the mean-normalised relative-spread family (AllenAI's own `rel_std` included).
- **Only 7 English tasks overlap the two corpora, and the above-random gate leaves just 4 of them.** The evidence is therefore the SNR *correlation* over that handful, not the top-K Jaccard (trivially 1.0 on so few tasks). `custom_swissai_hf` keeps n_shared = **4**.

### Results

Cross-corpus agreement by pool (headline = the pure 3-seed pool `seeds_28_1797_1904`). Regenerate with `python analysis/rq07_external_frameworks/analyze.py --pool custom_swissai_hf`.

**Cross-corpus agreement over the shared English tasks.** Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), for each pool's best cross-corpus variant. The English overlap is 7 tasks, and the above-random gate leaves the `n_shared` shown per pool. Where `n_shared` is small (≤ 5), the correlations rest on a handful of points and are indicative, not robust:

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_1904` (1 seed) | `projection` | 0.90 | 0.80 | 4 |
| `seeds_28_1797` (2 seeds) | `dispersion_shifted` | 1.00 | 1.00 | 4 |
| `seeds_28_1797_1904` (3 seeds) | `dispersion_shifted` | 0.98 | 1.00 | 4 |
| `custom_swissai_hf` (+ externals) | `mpsd` | 1.00 | 1.00 | 4 |

![Apertus vs AllenAI SNR, 3-seed pool, best variant](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_dispersion_shifted.png)

![Apertus vs AllenAI SNR across variants](pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

[snr_apertus_vs_snr_allenai_dispersion_shifted.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_dispersion_shifted.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/seeds_28_1797_1904/snr_apertus_vs_snr_allenai_grid.png)

## Files

- `pretraining/<pool>/pearson_r_per_variant.csv`: cross-corpus Pearson r for
  every SNR variant (the headline table).
- `…/shared_task_agreement.csv`: best cross-corpus variant + Pearson r /
  Spearman ρ over the shared tasks (the per-pool result row).
- `…/pearson_r_size_sweep.csv`: r vs the size used for the comparison.
- `…/agreement.csv`, `agreement.md`: top-K reliable-benchmark overlap +
  the MMLU-aliasing caveat.
- `…/top_apertus.csv`, `top_allenai.csv`, `task_overlap.csv`.
- `…/snr_apertus_vs_snr_allenai_*.png`: per-variant + grid scatters.
