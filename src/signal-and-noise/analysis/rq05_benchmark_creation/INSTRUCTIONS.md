# INSTRUCTIONS — `benchmark_creation`

Owner: one Claude session. Don't write outside this directory.

## Research question

Do **high-SNR** benchmarks share characteristics? Start with the
simplest signal: **data source** and **curation process**. (Other
candidates — instance length, language family, multiple-choice vs
free-form, fewshot count — are out of scope for the v0 pass.)

## Inputs

- `results/snr_definition/snr_variants_per_task.csv` — provided by
  Question 1 (depends on it).
- `results/benchmark_creation/data_info.md` — manually curated by the
  user. Each row should describe one benchmark family with:
  - `family` (must match `multilingual.analyze_snr_variants.benchmark_family`)
  - `data_source` (free text, e.g. "MMLU translated by Cohere", "PIQA
    translated by Google", "human-authored").
  - `curation_process` (e.g. "machine translation", "human
    translation", "originally multilingual", "filtered by humans").
  - Optional extras: `task_format` (mc vs gen), `domain` (general,
    STEM, …), `creation_year`, `n_languages`.
- Optional: `reference_hf-*.parquet` — not needed for v0.

`data_info.md` MUST exist before this analysis runs. If absent, the
session should write a stub explaining the schema and stop there.

## Plan

### Step 1 — wait for `data_info.md`

If the file is missing or empty:
1. Write a minimal `data_info.md` template (table with one row per
   family found in the parquet, columns: family, data_source,
   curation_process, task_format, domain, n_languages — empty cells
   for the user to fill in).
2. Stop. Do not produce charts. The README should say "awaiting
   benchmark metadata in `data_info.md`."

### Step 2 — join + per-family SNR aggregate

When `data_info.md` is filled in:

- Parse it into a DataFrame `meta` indexed by family.
- Load `results/snr_definition/snr_variants_per_task.csv`.
- Compute per-family SNR statistics for the selected best variant
  (default: `rel_std` since it's the simplest; verify by reading
  Question 1's headline). Aggregate: median, mean, max of
  `snr_<variant>_1B` across the per-language tasks in the family.
- Merge `meta` with the per-family stats → `per_family_snr.csv`.

### Step 3 — group analysis (data_source / curation)

For each categorical metadata column (`data_source`, `curation_process`,
…):
- Compute mean / median SNR per group.
- Render a strip-plot `snr_by_<col>.png`: x = SNR, y = group, dots =
  per-family points (so the user can see the spread).
- Run a one-way ANOVA (or Kruskal-Wallis if non-normal) and write the
  p-value into the README. Note: with ~12 families this is purely
  descriptive.

### Step 4 — README

Write `results/benchmark_creation/README.md`:

- Research question (verbatim).
- Setup: paragraph describing where SNR comes from
  (`snr_variants_per_task.csv`, variant choice, size = 1B), and that
  metadata comes from `data_info.md`.
- **If data_info.md is empty**: README explains the contract and shows
  the schema. No charts.
- **If data_info.md is filled**: embed `snr_by_data_source.png` and
  `snr_by_curation_process.png`; one-paragraph summary of the
  group-level finding (which sources/curations score higher SNR);
  short table of top-3 families per group.
- Directory contents.

## Notes / gotchas

- The set of families depends on `benchmark_family` mapping. Use the
  same helper Question 1 uses, so the join key is consistent.
- "High SNR" here means high under the best-Q1 variant. Don't average
  across variants — they differ in scale.
- Apertus ARC has separate English `arc_challenge` / `arc_easy`
  entries — they collapse to family `arc` already. Note in the
  README so the data_info row counts align.
- For families with very small per-family n (e.g., `paws` has 5
  languages, `xwinograd` has 4), the median is noisy. Show counts.

## Definition of done

- If `data_info.md` is empty: stub README + filled-in schema template.
- If `data_info.md` is filled: `per_family_snr.csv`,
  `snr_by_data_source.png`, `snr_by_curation_process.png`, and a
  README embedding both plots.
