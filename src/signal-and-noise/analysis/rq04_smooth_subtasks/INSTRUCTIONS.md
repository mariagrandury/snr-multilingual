# INSTRUCTIONS — `smooth_subtasks`

Owner: one Claude session. Don't write outside this directory.

## Research questions

Per benchmark, can we find a subset that **elevates SNR** (and thus DA)
either per-checkpoint or per-size? Two flavors:

1. **Logical (existing) split** — e.g., languages within a multilingual
   family, or the MMLU subject taxonomy inside `global_mmlu_full`.
2. **Per-sample search** — find the doc-id subset of a task that
   maximises SNR (no preexistent split required).

## Inputs

- `analysis/data/multilingual_snr/data/pretraining_custom-*.parquet`.
  Loaded via `snr.download.apertus.load_apertus_eval_results`.
- For per-sample analysis: `samples_<task>_*.jsonl` files. **These are
  NOT in the parquet.** They live on the cluster only
  (`/iopsstor/scratch/cscs/.../eval_logs/.../samples_*.jsonl`). Skip
  per-sample if running locally; describe in the README that this
  experiment requires cluster access.
- Reference HF data (`reference_hf-*.parquet`) is not useful here —
  per-mix SNR is undefined for one-mix models.

## Existing code (REUSE)

- [multilingual/smooth_subtasks.py](../../multilingual/smooth_subtasks.py)
  — three cases:
  - **Case 1**: per-benchmark family, subtask = language. CSV →
    `per_benchmark.csv`; plots → `per_benchmark_plots/`.
  - **Case 2**: `global_mmlu_full`, subtask = subject (mean across 10
    langs). CSV → `global_mmlu_full.csv`; plot →
    `global_mmlu_full_subjects.png`.
  - **Case 3**: `global_mmlu_full_<lang>`, subtask = subject (per
    language). CSV → `global_mmlu_full_per_language.csv`; plots →
    `global_mmlu_full_per_language_plots/`.
- [multilingual/smooth_subtasks_per_sample.py](../../multilingual/smooth_subtasks_per_sample.py)
  — Option D from `PROPOSALS.md`: variance prefilter + per-sample SNR
  ranking. **Cluster-only** — needs `samples_*.jsonl`.

The loader patch already wired Cases 1–3 to the local parquet (no
disk-walk needed for global_mmlu_full subjects — they live as keys in
the parquet, e.g., `global_mmlu_full_ar_anatomy`). The
`collect_multilingual_families` helper now filters out subject facets
so it gives the per-language aggregates only.

## Plan

### Step 1 — run Cases 1–3 fresh

```bash
python multilingual/smooth_subtasks.py
```

Should regenerate `per_benchmark.csv`, `per_benchmark_plots/`,
`global_mmlu_full.csv`, `global_mmlu_full_subjects.png`,
`global_mmlu_full_per_language.csv` and
`global_mmlu_full_per_language_plots/`.

### Step 2 — synthesise the headline finding per case

Write `summary.csv` with:
- columns: `case, task, size, full_set_snr, best_n, best_snr,
  snr_gain, best_subset_short`.
- rows: every (case, family/subject_set, size) row from the three CSVs,
  with `snr_gain = best_snr - full_set_snr`.

Then sort by descending `snr_gain` to surface the families /
languages where a subset matters most. This single CSV is what the
README will headline.

### Step 3 — pick the headline plot per case

- Case 1 candidate: a family with the largest `snr_gain` (likely
  `xnli` or `belebele`, but verify) — embed
  `per_benchmark_plots/<family>.png`.
- Case 2: `global_mmlu_full_subjects.png` (always).
- Case 3: pick the language with the largest `snr_gain` from Case 3 —
  embed `global_mmlu_full_per_language_plots/<lang>.png`.

### Step 4 — per-sample (skip if no cluster)

If the cluster eval_logs tree is *not* mounted (it isn't, locally),
add a paragraph to the README that:
- Documents the method (cite
  `multilingual/smooth_subtasks_per_sample.py`).
- States it requires `eval_logs/.../samples_*.jsonl` — produce a
  one-line example command.
- Notes that the prior cluster run's outputs are committed under
  `per_sample/` (audit-only; do not regenerate locally).

If a cluster path *is* available, run:
```bash
python multilingual/smooth_subtasks_per_sample.py
```
and embed `per_sample/summary_all.csv` as a top-snr-gain table in the
README (sorted descending by `snr_gain`).

### Step 5 — README

Write `results/smooth_subtasks/README.md`:

- **Research questions** — verbatim.
- **Setup** — describe the SNR primitive (per-mix last-5-ckpt arrays;
  combined-subset score = mean across (mix, step) of the included
  subtasks/samples), the variance prefilter, and the random-order
  baseline.
- **Main results — Case 1**: 1–2 sentence finding ("subset > full set
  for X families…"); embed best-gain family plot.
- **Main results — Case 2**: same.
- **Main results — Case 3**: same.
- **Per-sample**: short paragraph (cluster-only).
- **Directory contents** — explicit list.

## Notes / gotchas

- `snr_for_subset` averages per-(mix, step) scores across whichever
  subtasks happen to be present at that ckpt. Strict inner-join leaves
  most cells empty for arc/global_mmlu (not every language at every
  ckpt), so the relaxed join is by design — keep it.
- For Case 2 (`global_mmlu_full` averaged across the 10 languages),
  some (mix, step) combos may have <10 languages; the
  `n_languages` column reports the average. Flag in the README that
  results below e.g. 8 langs are weaker.
- The 175M models are smaller than the SNR pipeline expected; for some
  tasks the mix-mean spread is below the noise floor → SNR ≈ 0.
  This is fine; the cumulative-SNR sweep already handles it.

## Definition of done

- All three case CSVs/plots regenerated.
- `summary.csv` ranks all (case, task, size) by `snr_gain`.
- `README.md` embeds at least three headline plots and lists every
  file in the directory.
