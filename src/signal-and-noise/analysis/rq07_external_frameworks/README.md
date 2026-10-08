# External frameworks — Does our SNR ranking agree with AllenAI DataDecide?

## Research question

> Do the SNR variants and the "reliable benchmark" set we find on our ladder
> also hold on AllenAI's DataDecide / OLMo corpus, on the English benchmarks
> both share?

> ⚠️ **Only six English benchmarks overlap, and all six survive the gate.** The ladder
> shares `arc_easy`, `arc_challenge`, `hellaswag`, `csqa` (via the cloze twin `rf_commonsense_qa`),
> `openbookqa` and MMLU (via the cloze twin `rf_mmlu`) with DataDecide, and at
> the 1B rung the above-random gate keeps all six.
>
> With so small a universe, top-K set overlap is uninformative (any K ≥ N is
> trivially 1.0), so the evidence is the **correlation of SNR over the shared
> tasks** (values: Pearson r; ranking: Spearman ρ).

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **On the `predictivity` pool SNR values and rank order agree across corpora** — variant `rel_std` (rq04's global best, not selected here), Pearson r of log₁₀(SNR) **0.81**, Spearman ρ **0.83** over the 6 shared English tasks.
- **The shared universe is the English tasks both corpora evaluate** (ARC, HellaSwag, OpenBookQA, and MMLU / CSQA through our cloze twins), so the evidence is the SNR *correlation* over that handful, not top-K Jaccard (trivially 1.0 on so small a universe).
- **Like for like since 2026-10-08:** the same 20 % noise window (5 points at 1B), the same detrended checkpoint noise (rule 4) and the same metric on both sides (`build_allenai_variants.py`); the signal populations still differ (25 pretraining corpora against our design variants).
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

> ⚠️ **Methodological caveat — MMLU aliasing (resolved 2026-10-08).** Until the
> 2026-10-08 refresh Apertus's `global_mmlu_full_en[_<subject>]` rows were aliased
> to AllenAI's `mmlu[_<subject>]` rows, and **the two are not the same content**:
> Apertus ran the Cohere-Full translation/post-edit of MMLU (English split), AllenAI
> the original Hendrycks et al. MMLU.
>
> The ladder now evaluates the original MMLU (`cais/mmlu`) and its cloze twin
> `rf_mmlu`, which is matched to DataDecide's RC `mmlu` and clears the gate at our
> 1B rung, so the alias is gone and MMLU enters the headline r; see
> `pretraining/<pool>/agreement.md`.

## Key figure

![Ladder SNR against DataDecide SNR on the shared English tasks](pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png)

Population: pool `predictivity` (seed 1904, every cell and data build at the 1B rung, SwiGLU included) against DataDecide's 1B rung (25 data recipes); SNR variant `rel_std` (the one the [surrogates analysis](../rq04_surrogates/README.md) ranks first); the 6 shared English tasks that clear the above-random gate (ARC Easy, ARC Challenge, HellaSwag, MMLU, CommonsenseQA, OpenBookQA); log₁₀ SNR, each axis on its own scale.

**Key finding.** Both corpora rank the six gated shared tasks alike (Spearman ρ 0.83, Pearson r of log₁₀ SNR 0.81, p = 0.049). Six points make this indicative, not robust.

**Key findings**

- **Similar order, n = 6.** `rel_std` SNR is 4.09 / 8.18 / 2.97 / 4.94 / 2.64 / 1.78 on our ladder and 15.67 / 15.34 / 12.61 / 13.30 / 9.29 / 4.09 on DataDecide for ARC Easy / HellaSwag / ARC Challenge / MMLU / CSQA / OpenBookQA; both put CSQA and OpenBookQA last, and the top task differs (HellaSwag on ours, ARC Easy on DataDecide).
- **Our SNR is 1.9–4.2× smaller.** DataDecide's `rel_std` SNR exceeds ours by 3.8× on ARC Easy, 1.9× on HellaSwag, 4.2× on ARC Challenge, 2.7× on MMLU, 3.5× on CSQA and 2.3× on OpenBookQA; the log-scale correlation ignores this offset, so compare orders across corpora, never SNR levels.
- **The seed pool agrees.** On `predictivity_seeds` (every seed) the six tasks keep the same order on our side (ρ 0.83, Pearson r 0.83).
- **None of DataDecide's shared tasks is gated out on our side.** MMLU and CSQA (through the cloze twins) and OpenBookQA clear the above-random gate at our 1B rung, so the top-5 overlap is 5 of 5 (Jaccard 1.00).

**Follow-ups**

- The same scatter at 1.7B↔1B, to check whether the order holds at our reference size (`rel_std` r 0.79 there, see [pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv)).
- Bootstrap CIs over the six tasks, to put a number on how little n = 6 constrains r.

GitHub: [snr_apertus_vs_snr_allenai_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png) · [snr_apertus_vs_snr_allenai_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.csv). Every variant and the size sweep: [Results](#results).

## Methodology

- **Apertus side** (the predictivity ladder, 175M–1.7B in the size sweep): the
  per-task SNR table `pretraining/<pool>/snr_variants_per_task.csv` from the noise
  and SNR analysis ([`../rq03_noise_and_snr/`](../rq03_noise_and_snr/)). Each pool
  writes its own cross-corpus output (`pretraining/predictivity/`,
  `pretraining/predictivity_seeds/`).
- **AllenAI side** (DataDecide ladder, 25 mixes at seed 6198, sizes
  150M / 300M / 750M / 1B, every 2,500-step checkpoint of the
  `datadecide_intermediate` split): run through
  [build_allenai_variants.py](build_allenai_variants.py), which reuses every
  primitive from
  [run_apertus_snr_variants.py](../rq03_noise_and_snr/run_apertus_snr_variants.py)
  (`per_model_inputs`, `ckpt_noise_columns`, `variant_signal_noise_snr`) plus
  `compute_size_decision_accuracy` from
  [compute_da.py](../rq02_decision_accuracy/compute_da.py) and the 22-aggregator
  `AGGREGATION_FUNCTIONS` list.
- **Like for like (2026-10-08).** Both sides read the same noise window (the
  k/20 points in the last 20 % of each run, 5 points at 1B), the same checkpoint
  noise (the residual SD around a line through the window, RULES.md rule 4; the
  raw SD sits beside it as `ckpt_noise_raw_<size>`) and the same metric: each
  DataDecide task is read on the column matching our task's metric, `acc_norm` →
  `acc_per_char`, `acc` → `acc_raw`, not on DataDecide's own primary metric
  (`acc_uncond` for ARC-C, OBQA and CSQA). The table until then read the `core`
  split's last five checkpoints (86.5–100 % of the 1B run) with the raw SD on
  DataDecide's primary metric. The signal populations still differ: 25
  pretraining corpora against our design variants, whose English spread is 4–7×
  smaller (`plan/signal-audit-2026-10-08.md`, F1).
- **Task-name reconciliation.** DataDecide's RC (cloze) MMLU and CSQA are matched
  to our cloze twins `rf_mmlu` and `rf_commonsense_qa`; our letter-format `mmlu` and
  `commonsense_qa` are named `mmlu:mc` / `csqa:mc`, DataDecide's MC format, which
  its checkpoint series does not carry, so they are not shared. The
  `global_mmlu_full_en → mmlu` alias of the 36-sweep is gone. The shared set on the
  ladder is the English tasks both evaluate (`arc_challenge`, `arc_easy`, `csqa`,
  `hellaswag`, `mmlu`, `openbookqa`).
- **Correlation axis.** `log10(snr_<V>_<size>)` on each corpus, Pearson r (values)
  and Spearman ρ (rank) over the shared tasks that clear the above-random gate on
  our side (6 at 600M, 1B and 1.7B; 5 at 175M and 350M, where ARC Challenge is at chance).
- **Reference HF models skipped.** SmolLM3-3B / Olmo-3-7B / Apertus-8B each have a
  single training mix, so the data-mix-spread term in every SNR variant is
  undefined; including them would force the comparison onto raw `primary_score` (a
  capability number), conflating "task is reliable" with "task is easy".

**Which variant families transfer across corpora** — Pearson r of log₁₀ SNR at
1B↔1B over the 6 gated shared tasks, `predictivity` / `predictivity_seeds`
(from `pearson_r_per_variant.csv`; every r rests on n = 6):

| family | members (r, seed 1904 / all seeds) | reading |
|---|---|---|
| **dispersion** | `dispersion` 0.87/0.89, `range` 0.87/0.89, `mpd` 0.83/0.84, `aad` 0.78/0.80, `rms_deviation` 0.85/0.86, `quartile_deviation` 0.62/0.55, `dist_std` 0.88/0.88, `mpsd` 0.75/0.77 | positive in both pools (0.55–0.89) |
| **discrepancy** | `discrepancy` 0.86/0.87, `star_discrepancy` 0.90/0.91, `star_discrepancy_shifted` 0.70/0.70, `dispersion_shifted` 0.91/0.92, `rel_star_discrepancy` 0.94/0.95, `gini` 0.47/0.47 | positive, `gini` weakest |
| **relative-spread** | `rel_std` 0.81/0.83, `rel_mpd` 0.76/0.78, `rel_mpsd` 0.65/0.68, `rel_dispersion` 0.86/0.89, `iqr` 0.51/0.41 | transfers (incl. AllenAI's default `rel_std`); `iqr` weakest, same sign in both pools |
| **depth** | `tukey` 0.68/0.81, `projection` 0.85/0.85 | positive, `tukey` pool-dependent |
| **robust** | `mad` 0.81/0.76 | positive |

**Enlarging the shared universe.** The 6-task overlap (all 6 after the gate; 7 on the
36-sweep) is the binding constraint: 4 of the 10 DataDecide task rows of the
`datadecide_intermediate` split (`task_overlap.csv`) are not in the shared set
(`boolq`, `piqa`, `socialiqa`, `winogrande`).
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
MMLU subjects) against the 6 that clear the gate today.

DataDecide's generative and code tasks
(`gsm8k`, `minerva_*`, `drop`, `squad`, `triviaqa`, `jeopardy`, `mbpp`,
`codex_humaneval`), BBH, AGI Eval and the `:mc` formats sit at the floor below
2B parameters and would only add gated-out rows.

Not worth adding: `paloma_*` (perplexity, custom harness), `multitask_*` /
`custom_loss_*` (aggregates / loss probes), `copycolors:mc` (niche).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-08 12:06** (regenerated locally: acc_norm on the cloze-format originals)
and the outputs of the 2026-10-08 refresh (detrended checkpoint noise).

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Cross-corpus agreement by pool (headline = `predictivity`). Regenerate with `python analysis/rq07_external_frameworks/analyze.py --pool predictivity`.

**Cross-corpus agreement over the shared English tasks** — Pearson r of log₁₀(SNR) (values) and Spearman ρ (rank), at the variant rq04 selected on our ladder (the per-variant grid below shows the other 21). The above-random gate leaves the `n_shared` shown per pool; where it is small (≤5) the correlations are over a handful of points and should be read as indicative, not robust:

| pool | variant (from rq04) | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `predictivity` (grid, seed 1904) | `rel_std` | 0.81 | 0.83 | 6 |
| `predictivity_seeds` (all seeds) | `rel_std` | 0.83 | 0.83 | 6 |

![Ladder vs AllenAI SNR — rq04's variant](pretraining/predictivity/snr_apertus_vs_snr_allenai_rel_std.png)

![Ladder vs AllenAI SNR across variants](pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png)
<!-- END auto:results -->

[snr_apertus_vs_snr_allenai_rel_std.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_rel_std.png) ·
[snr_apertus_vs_snr_allenai_grid.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png) ·
[pearson_r_per_variant.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_per_variant.csv) ·
[shared_task_agreement.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/shared_task_agreement.csv) ·
[pearson_r_size_sweep.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/pearson_r_size_sweep.csv) ·
[agreement.md](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq07_external_frameworks/pretraining/predictivity/agreement.md)

Population: pools `predictivity` (seed 1904) and `predictivity_seeds` (every seed), all 22 SNR variants, the shared English tasks that clear the above-random gate on our side (n per size pair in `pearson_r_size_sweep.csv`). The previous figure showed one variant at one size; these show whether the agreement depends on the variant and the size.

**Key findings**

- **13 of the 22 variants are significant at 1B↔1B on seed 1904, on six points.** Pearson r ranges from 0.47 (`gini`) to 0.94 (`rel_star_discrepancy`, p = 0.005), and 20 of the 22 variants have r ≥ 0.6.
- **The seed pool moves individual variants little.** On `predictivity_seeds` 13 variants again pass p < 0.05 (`tukey` joins at r 0.81, `mad` drops to p = 0.08), no r moves by more than 0.14, and `iqr` keeps its sign (0.51 → 0.41).
- **Relative-spread variants transfer.** AllenAI's default `rel_std` has r 0.81 (seed 1904) and 0.83 (all seeds) at 1B↔1B, against 0.78–0.80 for `aad`.
- **The correlation is defined at every size pair.** At 175M↔150M and 350M↔300M 5 shared tasks clear the gate (ARC Challenge is at chance); `rel_std` has r 0.04 and 0.48 there, 0.59 on seed 1904 (0.59 on every seed) at 600M↔750M, and 0.79 at the unmatched 1.7B↔1B.
- **Five or six points make r unstable across sizes.** `rel_std` spans 0.04–0.81 over the five size pairs and `rel_mpd` −0.02–0.76, while `rel_star_discrepancy` stays at 0.93–0.97, so read any single variant's r as indicative only.

**Follow-ups**

- Bootstrap or leave-one-task-out r per variant, to separate stable variants from those six points happen to favour.
- The size sweep as a figure (r against size pair, one line per family), now that five or six shared tasks clear the gate at every size pair.

## TODO

- [ ] Add `mmlu_pro` / BBH to widen the 6-task shared universe.
- [ ] Bootstrap CIs on the cross-corpus Pearson r and Spearman ρ.
- [x] Re-run the original `mmlu` lm-eval task on Apertus and drop the MMLU alias
      for a like-for-like comparison (done 2026-10-08: `rf_mmlu` against DataDecide's RC `mmlu`).

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`; reference **1B**, the 86-task old
list) or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, cross-model dispersion with no mixture axis). Its harness,
task set and reference differ from the ladder's, and its shared universe with
DataDecide is the 7 standalone English tasks (the ladder's `auto` list shares
6 after the gate), so its correlations are a replication on a different
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
