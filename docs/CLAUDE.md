# snr-multilingual — local docs

The user wants to preview the project site locally on their laptop.

## What the site is

Two-part static site, deployed by Netlify:
- **MkDocs Material** at `/` — the project showcase plus the repo docs
- **Slidev** at `/slides/` — renders `documents/slides.md`

Tabs of the MkDocs side:
- **Home** (`index.md`: the paper's title, abstract, finding and
  contributions, verbatim), **Model ladder** (`ladder.md`), **Findings**
  (`findings/*.md`: Measurability, Agreement and Anticipation, the paper's
  three analysis sections with its wording and interactive versions of its
  figures), **Recommender** (`recommend.md`) and **Benchmarks**
  (`benchmarks.md`) are hand-written showcase pages. The earlier one-page-per-RQ
  findings are kept in `findings_all/`, excluded from the build
  (`exclude_docs` in `mkdocs.yml`) until they are added back. Their charts are `<div class="viz"
  data-viz="NAME">` blocks drawn by `interactive/app.js` (one `VIEWS.NAME`
  function each, Observable Plot + d3 from jsdelivr) from
  `interactive/data/*.json`. Regenerate the JSON with
  `python3 scripts/build_site_data.py` (snr env; it reads the rqNN_ outputs
  in the working tree, needs `git lfs pull` for the CSVs; most tables are
  only filtered and renamed, but the per-page modules in `scripts/site_data/`
  also count, pool and average, each module's docstring says what; a missing
  input file skips that JSON with a warning and keeps the old file; the Model ladder's grid comes from
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
  `score_curves.csv` and `predictivity_seeds/benchmark_curves.csv`; its
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
  Its nav entry is commented out in `mkdocs.yml` for the double-blind review;
  the pages are still built, searchable, in the sitemap and linked from Home
  and the Findings pages (mkdocs lists them as "not included in the nav").

The Findings pages draw the paper's four main-text figures with the views of
`interactive/views/paper.js` (`paperRq0`–`paperRq3`) from `paper_rq*.json`,
which `scripts/site_data/paper.py` builds from the very tables the paper
figures are drawn from; the Measurability page also shows the token-threshold
table of `gate.js`. The rest of this section describes the archived pages in
`findings_all/`: a page there reads its RQ's main JSON plus the JSON of its additional
views (`scripts/site_data/<page>.py`, often several files, e.g.
`noise_snr_*.json`). RQ0 (`gate.md`) reads `gate.json`
(the first-size map, the tokens-seen curves, the format examples from
`configs/tasks.json`, the reformulation table and the public-model floors),
`gate_extra.json` (the funnel per group, the item-count margins, the
majority-chance rerun, the early passes, the token exclusion and thresholds,
the floors per model line) and `gate_example.json` plus `gate_example/<benchmark>.json` (the gate
explainer, one file per benchmark, fetched on selection). The floors' list of
public models per task needs the external parquet: run the builder with
`SNR_MULTILINGUAL_DATA_DIR` pointing at a checkout that has it, otherwise the
list is empty (the builder says so); unreleased internal checkpoints are left
off it (`UNRELEASED_LINES`, and the same `UNRELEASED` pattern in
`scripts/site_data/gate.py` for the per-line view). RQ12 and RQ13
(`above_chance_items.json`, `english_only.json`) are built from their
analysis folders; the builder skips them, with a message, only in a checkout
without those folders.

## Anonymity (double-blind review)

The site must not link or name our HF orgs, W&B, GitHub, personal pages,
authors or cluster paths — a leak gets the paper rejected. Where such a link
belongs, write `{{ anonymity_notice }}` (the sentence is `REMOVED_LINK`
in `anonymity.py`; `app.js` reads it from `<meta name="anonymity-notice">`).
`mkdocs_hooks.py` also turns blocked links in included READMEs into that
sentence, redacts usernames and storage paths, and warns at the end of the
build on anything left (`anonymity: <file> contains [...]`): a clean build
prints no such line. Author names, the affiliation (swiss-ai, EPFL), the
cluster (CSCS, Clariden, Alps, bristen, daint, ETH, Lausanne), `claude.ai/` links and the internal
checkpoints (`apertus3-*`, `*from8b*`) are redacted in prose and code too;
`custom_swissai_hf` shows as `custom_hf`. The Slidev deck is not covered:
its nav entry is commented out in `mkdocs.yml` and `build.sh` builds it only
with `BUILD_SLIDES=1`. PNG figures under `repo/` are not scanned (text in an
image cannot be redacted): check their titles and legends by eye.

The text of `index.md` and of the three Findings pages is the paper's
(abstract, highlighted finding, contributions, sections 4.1–4.3), so it
follows the paper, not the analysis: when the paper's wording or numbers
change, copy them over. After the figures are regenerated, re-check the
other hand-written numbers (no data file feeds them yet): the ladder counts
in `ladder.md` (218 runs), `recommend.md`, and the archived pages in
`findings_all/`. The 2026-10-10 sweep logged every such number with its
source CSV and column; redo that log rather than re-deriving each number.
Captions in `interactive/views/*.js` read their numbers from the loaded JSON
where they can, so they follow a rebuild on their own.

Additional views of a page live in `interactive/views/<page>.js`, loaded
after `app.js` by `mkdocs.yml`: each registers on `window.SNRSITE.VIEWS`
with the helpers `app.js` exposes there. Their data comes from
`scripts/site_data/<page>.py` (`build() -> {json name: object}`), which
`build_site_data.py` runs after its own tables.

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

`mkdocs build --strict` builds cleanly (checked 2026-10-10): README links
to source files are rewritten to the anonymity notice by `mkdocs_hooks.py`,
so they no longer warn. Use `--strict`: the anonymity check in
`on_post_build` reports a leak as a warning, which `--strict` turns into a
failed build.

## What the user is likely to ask

- Style/theme tweaks in `mkdocs.yml` (palette, nav features)
- Adding plot/notebook rendering (suggest `mkdocs-jupyter` or static
  PNG embeds)
- Fixing a specific README link warning
- Adjusting the Netlify ignore rule
