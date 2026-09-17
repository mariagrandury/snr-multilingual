# Running the four research questions in parallel

The four research questions live under `results/<question>/` with one
`INSTRUCTIONS.md` each. They are intentionally **decoupled**: spin up one
fresh Claude session per question and they will not collide because
each writes only inside its own subdirectory.

## Shared setup (already done)

- HF dataset `multilingual-snr/multilingual-snr-eval-results`
  downloaded to `analysis/data/multilingual_snr/data/`:
  - `pretraining_custom-00000-of-00001.parquet` — 12 Apertus pretrained
    models (175M/350M/600M/1B × fwEdu30/60/90; seed 1904; 13 ckpts each;
    819 tasks).
  - `reference_hf-00000-of-00001.parquet` — external HF references
    (`SmolLM3-3B-Base`, `SmolLM3-3B-checkpoints`, `Olmo-3-1025-7B`,
    `Apertus-8B-2509`).
- Loader API in [snr/download/apertus.py](../snr/download/apertus.py):
  ```python
  from snr.download.apertus import (
      load_apertus_eval_results,        # 12 custom Apertus pretrains
      load_reference_hf_eval_results,   # SmolLM3 / Olmo-3 / Apertus-8B
      load_all_eval_results,            # both, with a `source` column
  )
  ```
  Schema: `model, mix, size, step, task, primary_score, seed, tokens,
  compute` (mix is the short form `fwEdu30/60/90` for Apertus, `main` /
  `stage1` for HF refs).
- Multilingual code lives under [analysis/](../).
  Helpers exposed:
  - `analysis.rq03_noise_and_snr.analyze_snr_variants.assign_language` /
    `benchmark_family` — language code + family inference from task name.
  - `analysis.rq08_subset_selection.smooth_subtasks.collect_multilingual_families` —
    `{family: [per_language_aggregate_tasks]}` (filters out
    per-(lang, subject) facets).
  - `analysis.rq08_subset_selection.smooth_subtasks.load_gmf_subjects_df` /
    `load_gmf_per_language_df` — global_mmlu_full subject views.

## How to launch one session per question

Open four terminals (or `claude --resume` four times); in each, copy the
matching prompt below verbatim. Each session is self-contained and
should write only inside its own results subdir.

### Session 1 — snr_definition
> Read `analysis/rq03_noise_and_snr/INSTRUCTIONS.md` and execute the plan.
> Reuse `analysis/rq03_noise_and_snr/run_apertus_snr_variants.py` and
> `analysis/rq04_surrogates/analyze_snr_variants.py`. Do not write outside
> `analysis/rq03_noise_and_snr/`.

### Session 2 — smooth_subtasks
> Read `analysis/rq08_subset_selection/INSTRUCTIONS.md` and execute the plan.
> Reuse `analysis/rq08_subset_selection/smooth_subtasks.py` and
> `analysis/rq08_subset_selection/smooth_subtasks_per_sample.py`. Do not write outside
> `analysis/rq08_subset_selection/`.

### Session 3 — allenai_comparison
> Read `analysis/rq07_external_frameworks/INSTRUCTIONS.md` and execute the plan.
> Inputs: `analysis/rq03_noise_and_snr/snr_variants_per_task.csv` (must
> exist before this session starts — depends on Session 1 having run at
> least once) plus the upstream allenai parquet pulled from HF. Do not
> write outside `analysis/rq07_external_frameworks/`.

### Session 4 — benchmark_creation
> Read `analysis/rq09_benchmark_design/INSTRUCTIONS.md`. Wait for the user
> to paste benchmark metadata into `data_info.md` before computing
> correlations; in the meantime stub the README and analysis script.
> Do not write outside `analysis/rq09_benchmark_design/`.

## Coordination rules

- **One session per directory.** Concurrent runs of the same script in
  the same dir will overwrite each other's PNGs/CSVs.
- **Loader is shared but read-only** — none of the questions need to
  modify `snr/download/apertus.py`. If you find a bug, surface it; do
  not silently patch.
- **Reference HF data is opt-in.** Apertus data alone is enough to
  answer Questions 1–2; the reference HF table is most useful for
  Question 3 (cross-corpus DA) and as an extra-data sanity in Question 1.
- **Naming.** All sessions share `assign_language` and
  `benchmark_family`. Don't reimplement those — import them.

## Output contract per question

Each `results/<question>/` should contain:

- `README.md` — research questions, experiment setup, main results
  (with the most relevant plots inlined as `![](...)`), and an
  enumeration of the directory's contents.
- `INSTRUCTIONS.md` (this file's sibling per-question version) — the
  detailed plan you executed.
- All CSVs / PNGs the README references.

When done, do a final pass: every plot mentioned in the README must
exist on disk; every file in the directory should be referenced (or
mentioned in the directory listing) in the README.
