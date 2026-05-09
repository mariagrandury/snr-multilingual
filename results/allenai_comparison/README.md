# `allenai_comparison/` — do our best SNR definitions transfer to AllenAI?

> Step-by-step plan: [INSTRUCTIONS.md](INSTRUCTIONS.md). Use one
> Claude session per research question — see
> [../PARALLEL_SESSIONS.md](../PARALLEL_SESSIONS.md).

## Research question

Do our "best" SNR definitions (from
[../snr_definition/](../snr_definition/)) correlate with the AllenAI
DataDecide / OLMo results? In particular, do the SAME benchmarks emerge
as "reliable" on both corpora?

## Setup

- **Apertus side** (12 custom pretrains × 13 ckpts, 3 data mixes): per-
  task SNR table from
  [`../snr_definition/snr_variants_per_task.csv`](../snr_definition/snr_variants_per_task.csv).
- **AllenAI side** (DataDecide ladder, 25 mixes × 5 ckpts at sizes
  150M / 300M / 750M / 1B): pulled at runtime via
  `snr.download.hf.pull_predictions_from_hf("allenai/signal-and-noise",
  split_name="core")` and run through
  [`build_allenai_variants.py`](build_allenai_variants.py), which
  reuses every primitive from
  [`multilingual/run_apertus_snr_variants.py`](../../multilingual/run_apertus_snr_variants.py)
  (`per_mix_inputs`, `variant_signal_noise_snr`, the 22-aggregator
  `AGGREGATION_FUNCTIONS` list).
- **Task-name reconciliation.** Apertus only ran the multilingual
  `global_mmlu_full_en_<subject>` view of MMLU on the full ckpt-series;
  the vanilla `mmlu_<subject>` rows in the Apertus parquet are
  single-shot and have no SNR. AllenAI uses the vanilla `mmlu_<subject>`
  names. We canonicalise Apertus names via
  `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]`. Without this aliasing
  the comparison collapses to 3 shared tasks (arc_*, hellaswag); with it,
  61 tasks have valid SNR on both sides.
- **Headline correlation axis:** `log10(snr_<V>_1B)` on each corpus,
  Pearson r over the 61 shared tasks.
- **Third corpus (`reference_hf`) — skipped.** SmolLM3-3B / Olmo-3-7B /
  Apertus-8B each have a single training mix (`main`/`stage1`), so the
  data-mix-spread term in every SNR variant is undefined. The rank-
  agreement comparison would have to use raw `primary_score` (a
  capability number, not a noise-relative one), conflating "task is
  reliable" with "task is easy for this model" — so we don't include it.

## Main results

### Headline scatter (best variant: `discrepancy`, n = 61, r = 0.697)

![headline scatter](snr_apertus_vs_snr_allenai_discrepancy.png)

The cross-corpus correlation of per-task `log10(SNR)` is positive and
strong for the discrepancy-family variants. The y > x bias is expected:
AllenAI's DataDecide has 25 data mixes vs Apertus's 3, so its
mix-spread "signal" is on a different absolute scale. What transfers is
the **rank** of tasks, not the absolute SNR magnitude.

### Per-variant Pearson r at the headline (1B ↔ 1B)

| variant | r | variant | r |
|---|---:|---|---:|
| discrepancy | **0.697** | rel_std | 0.040 |
| star_discrepancy | 0.661 | iqr | 0.042 |
| rel_star_discrepancy | 0.633 | rel_dispersion | 0.048 |
| dispersion_shifted | 0.620 | rel_mpd | 0.054 |
| gini | 0.550 | rel_mpsd | 0.007 |

Full table: [`pearson_r_per_variant.csv`](pearson_r_per_variant.csv).
Per-variant scatter grid (sorted by r): see
[`snr_apertus_vs_snr_allenai_grid.png`](snr_apertus_vs_snr_allenai_grid.png).

**Take-away:** the *discrepancy-family* variants
(`discrepancy`, `star_discrepancy`, `rel_star_discrepancy`,
`dispersion_shifted`, `gini`) transfer cleanly across corpora.
The *relative-spread* variants (`rel_std`, `rel_mpd`, `rel_mpsd`,
`iqr`, `rel_dispersion`) — including the upstream AllenAI default
`rel_std` — are essentially uncorrelated between the two corpora.

### Size sweep — correlation strengthens at smaller sizes

Mean Pearson r across all 22 variants at every matched-size pair:

| Apertus → AllenAI | mean r |
|---|---:|
| 175M → 150M | 0.517 |
| 350M → 300M | 0.445 |
| 600M → 750M | 0.384 |
| 1B → 1B     | 0.257 |

Counter-intuitively, the cross-corpus agreement is **strongest at the
smallest sizes** and degrades as the models grow. Apertus has only 3
mixes per size (vs AllenAI's 25), and one of the three 1B mixes
(`fwEdu90`) is half-trained — so the Apertus 1B SNR estimates are the
least statistically powerful, and that's precisely where the correlation
loses ground. The lesson is that the *shape* of which-tasks-are-noisy
emerges already at 175M; if you want to vet a benchmark for reliability
on Apertus-style data, you don't need to wait for the 1B run.
Full table: [`pearson_r_size_sweep.csv`](pearson_r_size_sweep.csv).

### Cross-corpus-reliable benchmarks (`discrepancy` SNR, top-K)

[`agreement.md`](agreement.md) tabulates the intersection / Jaccard of
each corpus's top-K tasks (ranked over the 63-task shared universe).

| K | \|∩\| | \|∩\|/K | Jaccard |
|---|---:|---:|---:|
| 5 | 3 | 0.60 | 0.43 |
| 10 | 7 | 0.70 | 0.54 |
| 20 | 13 | 0.65 | 0.48 |

**Benchmarks reliable in BOTH top-10s** — the headline "transferable
reliability" list:

- `arc_challenge`
- `arc_easy`
- `hellaswag`
- `mmlu`
- `mmlu_moral_scenarios`
- `mmlu_professional_law`
- `mmlu_professional_psychology`

These are the tasks where pretraining-data-mix differences produce a
measurable, robust ranking on **both** the Apertus 1B run and the
AllenAI 1B DataDecide ladder. They are good candidates for
small-scale-decision benchmarks regardless of which pretraining
distribution you're working with.

## Adding more benchmarks to Apertus (expand the shared-task universe)

The headline correlation runs over **61 shared tasks** — almost entirely
MMLU subjects + ARC + HellaSwag. AllenAI's `core` split has **178 tasks
that Apertus does not currently evaluate**. Adding them to the Apertus
eval suite would directly enlarge the cross-corpus comparison surface.
Task IDs below are the **OLMES** names AllenAI uses (suffix `:mc` =
multi-choice form, suffix `:rc` = rank-classification form). The
[OLMES harness](https://github.com/allenai/olmes) reproduces these task
IDs verbatim; `lm-evaluation-harness` has equivalents for most (sometimes
under different names — e.g. `agieval_*` instead of `agi_eval_*`,
`gsm8k_cot` for chain-of-thought variants).

### `mmlu_<subject>:mc` — 53 multi-choice MMLU subjects

Apertus only ran the rank-classification form (under
`global_mmlu_full_en_<subject>`). Adding the `:mc` form on the same
subjects would roughly double the MMLU comparison coverage on its own.

`mmlu_abstract_algebra:mc, mmlu_anatomy:mc, mmlu_astronomy:mc,
mmlu_business_ethics:mc, mmlu_clinical_knowledge:mc, mmlu_college_biology:mc,
mmlu_college_chemistry:mc, mmlu_college_computer_science:mc,
mmlu_college_mathematics:mc, mmlu_college_medicine:mc, mmlu_college_physics:mc,
mmlu_computer_security:mc, mmlu_conceptual_physics:mc, mmlu_econometrics:mc,
mmlu_electrical_engineering:mc, mmlu_elementary_mathematics:mc,
mmlu_formal_logic:mc, mmlu_global_facts:mc, mmlu_high_school_biology:mc,
mmlu_high_school_chemistry:mc, mmlu_high_school_computer_science:mc,
mmlu_high_school_european_history:mc, mmlu_high_school_geography:mc,
mmlu_high_school_government_and_politics:mc, mmlu_high_school_macroeconomics:mc,
mmlu_high_school_mathematics:mc, mmlu_high_school_microeconomics:mc,
mmlu_high_school_physics:mc, mmlu_high_school_psychology:mc,
mmlu_high_school_statistics:mc, mmlu_high_school_us_history:mc,
mmlu_high_school_world_history:mc, mmlu_human_aging:mc, mmlu_human_sexuality:mc,
mmlu_international_law:mc, mmlu_jurisprudence:mc, mmlu_logical_fallacies:mc,
mmlu_machine_learning:mc, mmlu_management:mc, mmlu_marketing:mc,
mmlu_medical_genetics:mc, mmlu_miscellaneous:mc, mmlu_moral_disputes:mc,
mmlu_moral_scenarios:mc, mmlu_nutrition:mc, mmlu_philosophy:mc,
mmlu_prehistory:mc, mmlu_public_relations:mc, mmlu_security_studies:mc,
mmlu_sociology:mc, mmlu_us_foreign_policy:mc, mmlu_virology:mc,
mmlu_world_religions:mc`

### `mmlu_pro` — MMLU-Pro (19 tasks)

Standalone harder benchmark with category subsets. Available in both
OLMES and `lm-evaluation-harness` (as `mmlu_pro` / `mmlu_pro_<category>`).

`mmlu_pro, mmlu_pro_biology:rc, mmlu_pro_business:rc, mmlu_pro_chemistry:rc,
mmlu_pro_computer science:rc, mmlu_pro_economics:rc, mmlu_pro_engineering:rc,
mmlu_pro_health:rc, mmlu_pro_history:rc, mmlu_pro_law:rc, mmlu_pro_math:rc,
mmlu_pro_other:rc, mmlu_pro_philosophy:rc, mmlu_pro_physics:rc,
mmlu_pro_psychology:rc` plus the four `mmlu_professional_*` /
`mmlu_professional_*:mc` pairs from the `:mc` family above.

### `arc_*:mc`, `hellaswag:mc` — 3 multi-choice forms

`arc_challenge:mc, arc_easy:mc, hellaswag:mc` — Apertus only has the
rank-classification forms.

### OLMES core knowledge / commonsense — 10 tasks

`boolq, boolq:mc, openbookqa, openbookqa:mc, piqa:mc, commonsense_qa,
commonsense_qa:mc, socialiqa, socialiqa:mc, winogrande, winogrande:mc,
truthfulqa_mc1, csqa, csqa:mc`. Standard `lm-evaluation-harness` task
IDs are `boolq`, `openbookqa`, `piqa`, `commonsense_qa`,
`social_iqa`, `winogrande`, `truthfulqa_mc1`.

### Math — 14 tasks

`gsm8k, gsm_plus, gsm_symbolic_main, gsm_symbolic_p1, gsm_symbolic_p2,
minerva_math_500, minerva_math_algebra, minerva_math_counting_and_probability,
minerva_math_geometry, minerva_math_intermediate_algebra,
minerva_math_number_theory, minerva_math_prealgebra, minerva_math_precalculus,
minerva`. `lm-evaluation-harness` provides `gsm8k`, `gsm8k_cot`, the
`hendrycks_math_*` set (Minerva is the same MATH benchmark under a
different name).

### Code — 4 tasks

`codex_humaneval, codex_humanevalplus, mbpp, mbppplus`. In
`lm-evaluation-harness`: `humaneval`, `humaneval_plus`, `mbpp`,
`mbpp_plus`. Need code-execution sandboxing to score.

### BBH (Big-Bench Hard) — 27 subsets

`bbh_boolean_expressions, bbh_causal_judgement, bbh_date_understanding,
bbh_disambiguation_qa, bbh_dyck_languages, bbh_formal_fallacies,
bbh_geometric_shapes, bbh_hyperbaton, bbh_logical_deduction_five_objects,
bbh_logical_deduction_seven_objects, bbh_logical_deduction_three_objects,
bbh_movie_recommendation, bbh_multistep_arithmetic_two, bbh_navigate,
bbh_object_counting, bbh_penguins_in_a_table,
bbh_reasoning_about_colored_objects, bbh_ruin_names,
bbh_salient_translation_error_detection, bbh_snarks, bbh_sports_understanding,
bbh_temporal_sequences, bbh_tracking_shuffled_objects_five_objects,
bbh_tracking_shuffled_objects_seven_objects,
bbh_tracking_shuffled_objects_three_objects, bbh_web_of_lies, bbh_word_sorting`.
In `lm-evaluation-harness`: `bbh_*` matches.

### AGI Eval — 19 subsets

`agi_eval, agi_eval_aqua-rat:mc, agi_eval_aqua-rat:rc,
agi_eval_gaokao-english:mc, agi_eval_gaokao-english:rc,
agi_eval_logiqa-en:mc, agi_eval_logiqa-en:rc, agi_eval_lsat-ar:mc,
agi_eval_lsat-ar:rc, agi_eval_lsat-lr:mc, agi_eval_lsat-lr:rc,
agi_eval_lsat-rc:mc, agi_eval_lsat-rc:rc, agi_eval_sat-en-without-passage:mc,
agi_eval_sat-en-without-passage:rc, agi_eval_sat-en:mc, agi_eval_sat-en:rc,
agi_eval_sat-math:mc, agi_eval_sat-math:rc`. In
`lm-evaluation-harness` these are `agieval_<task>` (single underscore,
no `:mc`/`:rc` split — pick the metric you want).

### Generative QA — 8 tasks

`autobencher, autobencher:mc, drop, medmcqa, medmcqa:mc, squad, triviaqa,
jeopardy`. In `lm-evaluation-harness`: `drop`, `squad`, `triviaqa`,
`medmcqa` are direct matches.

### Not recommended (harness flow / aggregate / not benchmarks)

- **`paloma_*` (13 tasks)** — perplexity / bytes-per-byte; needs the
  Paloma-specific harness flow.
- **`multitask_*` (4 tasks)** — aggregates of other benchmarks already
  covered above.
- **`custom_loss_*` (3 tasks)** — direct training-loss probes, not
  evaluation benchmarks in the usual sense.
- **`copycolors:mc`** — niche AllenAI test, low value.

## Directory contents

| file | description |
|---|---|
| [`INSTRUCTIONS.md`](INSTRUCTIONS.md) | execution plan |
| [`build_allenai_variants.py`](build_allenai_variants.py) | pulls AllenAI `core` parquet, runs the 22-variant SNR loop, writes `allenai_snr_variants_per_task.csv` |
| [`analyze.py`](analyze.py) | loads both per-task tables, applies the `global_mmlu_full_en_*` ↔ `mmlu_*` alias, writes every CSV / PNG / agreement.md below |
| `allenai_snr_variants_per_task.csv` | 241 tasks × 267 cols (22 variants × 4 sizes × 3 stats + 3 size-DA) |
| `task_overlap.csv` | per-task `in_apertus`, `in_allenai`, `shared` flags (post-aliasing) |
| `pearson_r_per_variant.csv` | headline 1B↔1B Pearson r per variant (sorted by r desc) |
| `pearson_r_size_sweep.csv` | long-format Pearson r at every matched-size pair |
| `snr_apertus_vs_snr_allenai_discrepancy.png` | headline scatter, best variant |
| `snr_apertus_vs_snr_allenai_grid.png` | per-variant scatter grid, sorted by r |
| `top_apertus.csv` / `top_allenai.csv` | top-20 tasks by `snr_discrepancy_1B` per corpus (over the shared-task universe) |
| `agreement.md` | top-K intersection / Jaccard table + per-corpus top-20 lists |
