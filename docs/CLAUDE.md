# snr-multilingual — local docs

The user wants to preview the project site locally on their laptop.

## What the site is

Two-part static site, deployed by Netlify:
- **MkDocs Material** at `/` — the project showcase plus the repo docs
- **Slidev** at `/slides/` — renders `documents/slides.md`

Tabs of the MkDocs side:
- **Home** (`index.md`), **Model ladder** (`ladder.md`), **Findings**
  (`findings/*.md`, one page per RQ), **Benchmarks** (`benchmarks.md`) and
  **Recommender** (`recommend.md`) are
  hand-written showcase pages. Their charts are `<div class="viz"
  data-viz="NAME">` blocks drawn by `interactive/app.js` (one `VIEWS.NAME`
  function each, Observable Plot + d3 from jsdelivr) from
  `interactive/data/*.json`. Regenerate the JSON with
  `python3 scripts/build_site_data.py` (it filters committed rqNN_ tables,
  needs `git lfs pull` for the CSVs; the Model ladder's grid comes from
  `launch_trainings.py`'s registry and its data mixtures from the builder's
  plans via `src/pretrain/data/data_progress.py`, so run it on the cluster).
  The mixtures' hover cards read `hello`, `thank_you` and `speakers` from
  `configs/languages.json`. The Recommender's look-up table (`lookup.json`)
  is the one threshold the builder computes: per (benchmark, language, size)
  the fewest tokens of the language from which every checkpoint stays above
  chance, read from the tokens-seen cell table
  (`pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv`, in
  `rq00_chance_vs_train_tokens/` or, before that move, `rq01_scaling_predictability/`);
  cells without a token count (mixture plans unreachable) are left out with a
  printed warning. The Evaluation section's score curves read rq00's
  `score_curves.csv` and `predictivity_all/benchmark_curves.csv`; its
  benchmark cards read `configs/tasks.json`, whose per-task `example` and
  `source` come from `python3 src/evals/scripts/derive_task_options.py
  --examples` (one scored item per task from the eval samples; re-run it
  with `--only-missing` after adding benchmarks). A findings page quotes its RQ README's
  "Highlighted result" block with `<!-- highlight: rqNN_name -->`, expanded by
  `mkdocs_hooks.py` at build time — so its numbers follow the pipeline. The
  Benchmarks table reads `configs/multilingual_benchmarks.csv`, which the
  same hook publishes as `interactive/data/benchmarks.csv`.
- **Docs** — thin stubs that `--8<--` include a README from elsewhere in the
  repo (e.g. `docs/pretraining.md` includes `src/pretrain/README.md`,
  `docs/repo.md` the root README). Edit the original READMEs, not the stubs.

The Findings pages read one JSON per RQ. RQ0 (`gate.md`) reads `gate.json`
(the first-size map, the tokens-seen curves, the format examples from
`configs/tasks.json`, the reformulation table and the public-model floors)
and `gate_example.json` plus `gate_example/<benchmark>.json` (the gate
explainer, one file per benchmark, fetched on selection). The floors' list of
public models per task needs the external parquet: run the builder with
`SNR_MULTILINGUAL_DATA_DIR` pointing at a checkout that has it, otherwise the
list is empty (the builder says so); unreleased internal checkpoints are left
off it (`UNRELEASED_LINES`). RQ12 and RQ13 (`above_chance_items.json`,
`english_only.json`) are skipped until their analysis folders are in the
checkout, and their pages say so.

## Anonymity (double-blind review)

The site must not link or name our HF orgs, W&B, GitHub, personal pages,
authors or cluster paths — a leak gets the paper rejected. Where such a link
belongs, write `{{ anonymity_notice }}` (the sentence is `extra.anonymity_notice`
in `mkdocs.yml`; `app.js` reads it from `<meta name="anonymity-notice">`).
`mkdocs_hooks.py` also turns blocked links in included READMEs into that
sentence, redacts usernames and storage paths, and warns at the end of the
build on anything left (`anonymity: <file> contains [...]`): a clean build
prints no such line. Author names, the affiliation (swiss-ai, EPFL), the
cluster (CSCS, Clariden, Alps), `claude.ai/` links and the internal
checkpoints (`apertus3-*`, `*from8b*`) are redacted in prose and code too;
`custom_swissai_hf` shows as `custom_hf`. The Slidev deck is not covered:
its nav entry is commented out in `mkdocs.yml` and `build.sh` builds it only
with `BUILD_SLIDES=1`. PNG figures under `repo/` are not scanned (text in an
image cannot be redacted): check their titles and legends by eye.

After the figures are regenerated, re-check the hand-written numbers (no
data file feeds them yet): `index.md` cards (647 / 331 pairs, median
R² ≥ 0.97, DA 0.97 vs 0.56, 6.6×), `recommend.md` ("27 of 46 languages"),
and the "Key findings" admonition in `findings/gate.md`.

## Files

- `mkdocs.yml` — Material theme config, nav, snippets extension wired
  to include READMEs from repo root
- `docs/` — showcase pages, stub pages, `interactive/` (app.js, app.css, data/)
- `scripts/build_site_data.py` — analysis tables → `docs/interactive/data/`
- `requirements-docs.txt` — `mkdocs`, `mkdocs-material`, `pymdown-extensions`
- `documents/package.json` — Slidev (pnpm)
- `build.sh` — full build: mkdocs → `site/`, slidev → `site/slides/`
- `netlify.toml` — runs `bash build.sh`, publishes `site/`,
  ignores rebuilds when no `.md`, `docs/` or MkDocs config file changed

## Preview locally

Docs only (fast, just Python):

```bash
pip install -r requirements-docs.txt
mkdocs serve
# → http://127.0.0.1:8000
```

Full site incl. slides (needs Node + pnpm):

```bash
bash build.sh
# then serve site/ with any static server, e.g.
python -m http.server -d site 8000
```

Slides only (live-reload):

```bash
cd documents
pnpm install
pnpm dev
```

## Known build warnings

`mkdocs build` emits warnings about README links pointing to source
files (`.py`, `.sh`, `.sbatch`). These are not docs and won't render —
the warnings are expected. Don't enable `--strict`.

## What the user is likely to ask

- Style/theme tweaks in `mkdocs.yml` (palette, nav features)
- Adding plot/notebook rendering (suggest `mkdocs-jupyter` or static
  PNG embeds)
- Fixing a specific README link warning
- Adjusting the Netlify ignore rule
