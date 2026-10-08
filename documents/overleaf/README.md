# Overleaf exports

The paper is edited on Overleaf and generated in the repo (`documents/paper/`).
Each Overleaf download goes here as `paper-overleaf-<MM-DD-HHh>/` (or the
`.zip` itself). The exports are not committed.

## Compare an export with the repo

```bash
python3 scripts/overleaf_diff.py                    # the newest export here
python3 scripts/overleaf_diff.py documents/overleaf/paper-overleaf-10-08-20h
python3 scripts/overleaf_diff.py EXPORT.zip         # unpacked next to the zip
```

It pairs files by name, so the two layouts can differ (Overleaf's `main.tex`
at the root, the repo's in `sections/`, the generated pages in
`sections/rqs/`, a renumbered `app_06_snr.tex` against `app_snr.tex`). It
prints one line per differing file and the files found on one side only. For
a text file it also says which side moved:

- `repo ahead (Overleaf = repo <commit>)`: the Overleaf copy is an older repo
  version. Copy the repo file to Overleaf.
- `diverged (X lines on Overleaf, Y in the repo, against repo <commit>)`: the
  Overleaf copy matches no repo version. X lines are Overleaf edits to port
  into the repo, Y are repo changes to bring to Overleaf, both counted against
  the closest repo version.

Figures and other binaries are compared by checksum. Then look at the text:

```bash
python3 scripts/overleaf_diff.py EXPORT --diff 04_analysis   # word diff in the terminal
python3 scripts/overleaf_diff.py EXPORT --diff               # every differing file
python3 scripts/overleaf_diff.py EXPORT --code 04_analysis   # VS Code diff tabs (Overleaf left, repo right)
python3 scripts/overleaf_diff.py EXPORT --html               # one report: <export>-diff.html here
```

Red is the Overleaf text, green the repo text.

The pages under `sections/rqs/` (and `tables/`, `figures/`) are regenerated
from the analysis, so an Overleaf edit to one of them is lost at the next
refresh. Port a layout change into its generator
(`documents/paper/figures/make_rq_appendix.py`) rather than into the page.
