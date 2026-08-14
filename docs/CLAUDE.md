# snr-multilingual — local docs

The user wants to preview the project site locally on their laptop.

## What the site is

Two-part static site, deployed by Netlify:
- **MkDocs Material** at `/` — renders the project READMEs
- **Slidev** at `/slides/` — renders `documents/slides.md`

The MkDocs side reads from `docs/` (8 thin stub pages, one per nav entry).
Each stub uses `--8<--` to include a README from elsewhere in the repo
(e.g. `docs/pretraining.md` includes `src/pretrain/README.md`). So edits
happen in the original READMEs, not in `docs/`.

## Files

- `mkdocs.yml` — Material theme config, nav, snippets extension wired
  to include READMEs from repo root
- `docs/` — stub pages
- `requirements-docs.txt` — `mkdocs`, `mkdocs-material`, `pymdown-extensions`
- `documents/package.json` — Slidev (pnpm)
- `build.sh` — full build: mkdocs → `site/`, slidev → `site/slides/`
- `netlify.toml` — runs `bash build.sh`, publishes `site/`,
  ignores rebuilds when no `.md` changed

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
