# Documents

Static documents:

- snr_paper.pdf: Paper presenting the original Signal-to-Noise framework that inspired this project, it is only focused on English and we want to extend it to multilingual benchmarks.
- snr_preliminary.pdf: Thesis of one member of the team with preliminary results about this project, the code used for this report is under "preliminary_analysis/".
- research_proposal.pdf: Motivation and research plan for the project.

Slides (Work in progress):

- slides.md: Source to generate slides using slidev with the scholarly theme.

## Deck figures

`figures/` regenerates every PNG under `public/ladder/` from the published
ladder report. `data.py` resolves that report exactly as the analysis does
(`snr.download.ladder.ladder_dir`: an explicit path, then `$SNR_LADDER_DIR`,
then a download from the Hub), `style.py` holds the one palette all of them
share, and `predictivity.py` holds the RQ6 measurements.

```bash
cd documents/figures
for f in fig_setup fig_languages fig_benchmarks fig_predictivity fig_rq6_sketch; do
  python3 $f.py
done
python3 fig_from_analysis.py    # after src/signal-and-noise/analysis/report_figures/make_figures.py
```

- `fig_setup.py`: the planned grid and what happened in each cell, the 90M
  optimizer timescale, the seed holdout.
- `fig_languages.py`: bits per byte gained per language, English against the rest.
- `fig_benchmarks.py`: the smallest size at which each benchmark family clears
  chance in each language, and bits per byte against the surviving benchmarks.
- `fig_predictivity.py`: smallest predictive proxy size, overall and per language.
- `fig_rq6_sketch.py`: RQ6 in the shape the team sketched, plus the measurement
  behind it.
- `fig_from_analysis.py`: rasterises the report figures the deck reuses
  (`fig1_gate`, `fig3_reliability_map`, `fig4_subset_sweep`), which
  `make_figures.py` writes as PDFs.

Per-language figures cover `configs/languages.json` `groups.trained`, the 50
languages of the L50 mixture. `groups.main` is the older 12-language set the
earlier decks used.

inform eval design decisions, check if correlation with dadat source, preprocessing pipeline, etc

recommended taxonomy
