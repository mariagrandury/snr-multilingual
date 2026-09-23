# snr-multilingual — local docs

The user wants to preview the project site locally on their laptop.

## What the site is

Two-part static site, deployed by Netlify:
- **MkDocs Material** at `/` — the project showcase plus the repo docs
- **Slidev** at `/slides/` — renders `documents/slides.md`

Tabs of the MkDocs side:
- **Home** (`index.md`), **Model ladder** (`ladder.md`), **Findings**
  (`findings/*.md`, one page per RQ) and **Recommender** (`recommend.md`) are
  hand-written showcase pages. Their charts are `<div class="viz"
  data-viz="NAME">` blocks drawn by `interactive/app.js` (one `VIEWS.NAME`
  function each, Observable Plot + d3 from jsdelivr) from
  `interactive/data/*.json`. Regenerate the JSON with
  `python3 scripts/build_site_data.py` (it only filters committed rqNN_ tables;
  needs `git lfs pull` for the CSVs). A findings page quotes its RQ README's
  "Highlighted result" block with `<!-- highlight: rqNN_name -->`, expanded by
  `mkdocs_hooks.py` at build time — so its numbers follow the pipeline.
- **Docs** — thin stubs that `--8<--` include a README from elsewhere in the
  repo (e.g. `docs/pretraining.md` includes `src/pretrain/README.md`,
  `docs/repo.md` the root README). Edit the original READMEs, not the stubs.

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
